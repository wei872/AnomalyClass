"""FabricNCD 主体网络.

数据流 (以 512x512 输入为例):

```
RGB ──► 轻量主干 (MobileNetV3-Small, 1.9 GFLOPs 级别的 1/4~1/32 特征)
          │p2(1/4)      │p3(1/8)      │p4(1/16)      │p5(1/32)
          ▼             │             │              ▼
   Gabor 纹理编码 ──────┘             │      全局上下文 (空洞卷积+线性注意力)
   (16 方向/相位响应)                 │              │
          └──────────► 全局-局部融合 ◄───────────────┘
                          │ + FiLM(全局)
                          ▼
               [布面地理 embedding]  ◄── 布面/背景掩码头 (1/4)
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
     异常头 (1/8)     嵌入头 (64-d)     原型分类器 (19 类)
          │               │                │
          │               ▼                │
          │        正常特征记忆库 ───────────┘
          ▼
   异常分数 = (1-w)·sigmoid(异常头) + w·(1-记忆库相似度)
```

轻量化: 主干 2.5M 参数, 其余头部合计 < 1.5M, 512 输入总计算量约 2 GFLOPs,
单张 CPU (4 核) 推理约 60-120 ms, TensorRT INT8 可到 10 ms 级别.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .glocal import GeographyEmbedding, GlobalContext, GlocalFusion
from .heads import AnomalyHead, EmbeddingHead, FabricMaskHead, PrototypeClassifier
from .memory import MemoryBank
from .texture import TextureEncoder, descriptor_from_stats, estimate_weave_period


class FabricNCD(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        image_size = data_cfg.get("image_size", [512, 512])
        if isinstance(image_size, str):  # "auto": 按数据集原图比例推断 (H, W)
            try:
                from ..data.yolo import resolve_image_size

                image_size = list(resolve_image_size(cfg))
            except Exception:  # pragma: no cover - 数据集不可达时兜底
                image_size = [1024, 576]
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        self.image_size = image_size
        self.cfg = cfg

        # ---------------- 主干 ----------------
        bb_cfg = dict(model_cfg.get("backbone", {}))
        bb_cfg.setdefault("image_size", int(min(image_size)))
        self.backbone = build_backbone(bb_cfg)
        c_p2 = self.backbone.out_channels["p2"]
        c_p3 = self.backbone.out_channels["p3"]
        c_p4 = self.backbone.out_channels["p4"]
        c_p5 = self.backbone.out_channels["p5"]

        # ---------------- 纹理 ----------------
        tex_cfg = model_cfg.get("texture", {})
        tex_channels = int(tex_cfg.get("channels", 32))
        self.texture = TextureEncoder(
            in_channels=c_p2,
            out_channels=tex_channels,
            orientations=tex_cfg.get("orientations", (0, 45, 90, 135)),
            wavelengths=tex_cfg.get("wavelengths", (4.0, 8.0)),
            kernel_size=int(tex_cfg.get("kernel_size", 15)),
            use_imaginary=bool(tex_cfg.get("use_imaginary", True)),
            use_weave_norm=bool(tex_cfg.get("use_weave_norm", True)),
        )

        # ---------------- 全局 / 局部 ----------------
        neck_cfg = model_cfg.get("neck", {})
        c_neck = int(neck_cfg.get("channels", 96))
        self.global_branch = GlobalContext(
            c_p5,
            int(neck_cfg.get("global_channels", c_neck)),
            dilations=neck_cfg.get("dilations", (2, 4)),
            heads=int(neck_cfg.get("attention_heads", 4)),
            use_attention=bool(neck_cfg.get("use_attention", True)),
            weave_dim=6,
        )
        c_glob = int(neck_cfg.get("global_channels", c_neck))
        self.fusion = GlocalFusion(
            c_p3=c_p3,
            c_p4=c_p4,
            c_glob=c_glob,
            c_out=c_neck,
            c_local=c_p3 + tex_channels,
        )

        # ---------------- 布面 / 布边 ----------------
        self.fabric_enabled = bool(model_cfg.get("fabric_mask", {}).get("enabled", True))
        self.mask_stem = nn.Sequential(
            nn.Conv2d(tex_channels + c_neck, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(inplace=True),
        )
        self.fabric_head = FabricMaskHead(48, mid=24)
        self.geography = GeographyEmbedding(
            c_neck,
            pool_sizes=model_cfg.get("fabric_mask", {}).get("pool_sizes", (3, 7, 15, 31, 63)),
            use_xy=bool(model_cfg.get("fabric_mask", {}).get("use_xy", False)),
            border_prior=bool(model_cfg.get("fabric_mask", {}).get("border_prior", True)),
        )

        # ---------------- 检测 / 分类 ----------------
        head_cfg = model_cfg.get("heads", {})
        embed_dim = int(head_cfg.get("embed_dim", 64))
        self.anomaly_head = AnomalyHead(c_neck, mid=int(head_cfg.get("anomaly_mid", 32)))
        self.embed_head = EmbeddingHead(c_neck, dim=embed_dim)
        num_classes = int(head_cfg.get("num_classes", 19))
        self.classifier = PrototypeClassifier(
            dim=embed_dim,
            num_classes=num_classes,
            temperature=float(head_cfg.get("proto_temperature", 0.1)),
        )

        # ---------------- 记忆库 ----------------
        mem_cfg = model_cfg.get("memory", {})
        self.memory_enabled = bool(mem_cfg.get("enabled", True))
        self.memory_weight = float(mem_cfg.get("score_weight", 0.35))
        # 余弦距离在正常布面上量级很小 (0~0.05), 归一化后才能与学习式打分可比
        self.memory_distance_scale = float(mem_cfg.get("distance_scale", 0.08))
        self.memory = MemoryBank(
            capacity=int(mem_cfg.get("capacity", 2048)),
            dim=embed_dim,
            query_stride=int(mem_cfg.get("query_stride", 1)),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels != 1:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,
        compute_memory: bool = True,
        weave_desc: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """images: (B,3,H,W), 已做 ImageNet 归一化.

        Returns dict with:
            ``anomaly_logit``  (B,1,H/8,W/8)
            ``anomaly_score``  (B,1,H/8,W/8)  综合分数 (0~1)
            ``learned_score``  (B,1,H/8,W/8)
            ``memory_score``   (B,1,H/8,W/8) 或 None
            ``cloth_logit`` / ``cloth_prob`` / ``boundary_logit`` (B,1,H/4,W/4)
            ``embedding``      (B,d,H/8,W/8)
            ``class_logits``   (B,C,H/8,W/8)
            ``weave``          周期估计 dict
        """
        feats = self.backbone(images)
        p2, p3, p4, p5 = feats["p2"], feats["p3"], feats["p4"], feats["p5"]

        tex = self.texture(images, p2)
        if weave_desc is None:
            weave = estimate_weave_period(images)
            weave_desc = descriptor_from_stats(weave)
        else:
            # 部署时可在外部分别计算织造周期 (numpy 版), 避免把 FFT 导出到计算图
            weave = {}
        glob = self.global_branch(p5, weave_desc)
        cond = glob.mean(dim=(-2, -1))
        # 纹理编码在 1/4, 融合到 1/8 的局部特征上
        tex_local = F.interpolate(tex, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        local = torch.cat([p3, tex_local], dim=1)
        fused = self.fusion(local, p4, glob, cond)

        # 布面掩码 (1/4)
        fused_up = F.interpolate(fused, size=tex.shape[-2:], mode="bilinear", align_corners=False)
        mask_feat = self.mask_stem(torch.cat([tex, fused_up], dim=1))
        mask_out = self.fabric_head(mask_feat)
        cloth_prob = torch.sigmoid(mask_out["cloth_logit"])

        # 布面地理调制 (1/8)
        if self.fabric_enabled:
            fused = self.geography(cloth_prob, fused)

        anomaly_logit = self.anomaly_head(fused)
        emb = self.embed_head(fused)
        class_logits = self.classifier(emb)

        learned_score = torch.sigmoid(anomaly_logit)
        memory_score: Optional[torch.Tensor] = None
        memory_distance: Optional[torch.Tensor] = None
        if compute_memory and self.memory_enabled and self.memory.is_ready:
            sim = self.memory.similarity(emb)
            memory_distance = (1.0 - sim).clamp_min(0.0)
            scale = max(self.memory_distance_scale, 1e-3)
            # 平滑映射 d/(d+scale): 单调、落在 (0,1), 且**不会饱和**.
            # 原来的 clamp(d/scale) 会让距离超过 scale 的正常像素全部堆在 1.0,
            # 于是正常图的 999 分位也变成 1.0, 阈值退化成"不可达", 检出率恒为 0.
            memory_score = memory_distance / (memory_distance + scale)
            anomaly_score = (1.0 - self.memory_weight) * learned_score + self.memory_weight * memory_score
        else:
            anomaly_score = learned_score

        return {
            "anomaly_logit": anomaly_logit,
            "anomaly_score": anomaly_score,
            "learned_score": learned_score,
            "memory_score": memory_score,
            "memory_distance": memory_distance,
            "cloth_logit": mask_out["cloth_logit"],
            "boundary_logit": mask_out["boundary_logit"],
            "cloth_prob": cloth_prob,
            "embedding": emb,
            "class_logits": class_logits,
            "weave": weave,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_memory(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """用正常布面图像更新记忆库. mask: (B,1,H,W) 布面掩码, 只取布面区域入池."""
        was_training = self.training
        self.eval()
        out = self.forward(images, compute_memory=False)
        emb = out["embedding"]  # (B,d,h,w)
        b, d, h, w = emb.shape
        flat = emb.permute(0, 2, 3, 1).reshape(-1, d)
        if mask is not None:
            keep = F.interpolate(mask.float(), size=(h, w), mode="area").reshape(-1) > 0.5
            if keep.any():
                flat = flat[keep]
        self.memory.update(flat)
        self.train(was_training)

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    def trainable_parameters(self, backbone_lr: Optional[float] = None, base_lr: float = 1e-3) -> list:
        """分组学习率: 主干更小的 lr, 头部正常 lr."""
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("backbone.") else head_params).append(param)
        groups = [{"params": head_params, "lr": base_lr}]
        if backbone_params:
            groups.insert(0, {"params": backbone_params, "lr": backbone_lr or base_lr * 0.2})
        return groups

