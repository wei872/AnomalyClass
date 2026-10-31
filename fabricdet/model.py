# -*- coding: utf-8 -*-
"""FabricDefectNet: 端到端 检测-定位-分类-未知缺陷 一体化轻量模型。

结构:
  DINOv2 ViT-S/14 (stride 14, 21.9M)            <- 自监督预训练骨干, 泛化强
    -> 1x1 投影 + 2x 轻量上采样 (stride 7)       <- 小缺陷(蚊虫小黑点等)定位分辨率
    -> 共享轻量卷积 trunk (GroupNorm, bs=1 也稳定)
       |-- obj 分支:  目标置信度 (anchor-free, FCOS 式)
       |-- box 分支:  (l,t,r,b) 距离回归 -> xyxy
       |-- emb 分支:  L2 归一化 128-d 度量嵌入
    -> 可学习类原型 (19 x 128) + 余弦相似度      <- 度量分类头: 开集/未知缺陷的天然接口

开集原理: 分类不靠 softmax 全连接层, 而靠"嵌入与类原型的余弦相似度"。
训练用监督对比学习把 19 类各自聚成紧簇; 未见过的缺陷嵌入落在所有原型之外,
max-cos < tau_unknown 即判为 "未知缺陷", 从而做到 "没见过的缺陷也能报出来"。
tau_unknown 与 conf 阈值在训练结束时用 val 集自动标定, 写进权重文件, 一次训练直接可用。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2ViT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvGN(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=None):
        super().__init__()
        p = (k - 1) // 2 if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.gn = nn.GroupNorm(32 if c_out % 32 == 0 else c_out, c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Neck(nn.Module):
    """384-d tokens -> 128-d stride-7 特征图。"""

    def __init__(self, in_dim=384, mid=256, out=128):
        super().__init__()
        self.proj = ConvGN(in_dim, mid, 1)
        self.up = nn.ConvTranspose2d(mid, out, kernel_size=2, stride=2)
        self.gn = nn.GroupNorm(32, out)
        self.act = nn.SiLU(inplace=True)
        self.refine = ConvGN(out, out, 3)

    def forward(self, x):
        x = self.proj(x)
        x = self.act(self.gn(self.up(x)))
        return self.refine(x)


class Head(nn.Module):
    def __init__(self, c=128, emb_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(ConvGN(c, c, 3), ConvGN(c, c, 3))
        self.obj = nn.Conv2d(c, 1, 3, padding=1)
        self.box = nn.Conv2d(c, 4, 3, padding=1)
        self.emb = nn.Conv2d(c, emb_dim, 3, padding=1)
        nn.init.zeros_(self.obj.bias)

    def forward(self, x):
        t = self.trunk(x)
        return self.obj(t), self.box(t), self.emb(t)


class FabricDefectNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 128, input_size: int = 448,
                 backbone_variant: str = "vits14"):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.input_size = input_size
        self.stride = 7                      # backbone 14 / neck 上采样 2
        self.backbone = DINOv2ViT(img_size=518, patch_size=14, embed_dim=384, depth=12, num_heads=6)
        self.neck = Neck(384, 256, 128)
        self.head = Head(128, emb_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.1)))
        self._grid_cache: dict = {}

    # ---------------------------------------------------------------- grid
    def grid_centers(self, h: int, w: int, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._grid_cache:
            ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * self.stride
            xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * self.stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._grid_cache[key] = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
        return self._grid_cache[key]

    # ------------------------------------------------------------- forward
    def forward_raw(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        obj, box, emb = self.head(feat)
        return obj, box, emb

    def decode(self, obj, box, emb):
        """-> boxes (B,N,4) px, obj_logits (B,N), cosine sims (B,N,C)."""
        B, _, H, W = obj.shape
        centers = self.grid_centers(H, W, obj.device, obj.dtype)             # (N,2)
        ltrb = torch.exp(box.clamp(-8.0, 10.0).permute(0, 2, 3, 1).reshape(B, -1, 4)) * self.stride
        x1 = centers[:, 0] - ltrb[..., 0]
        y1 = centers[:, 1] - ltrb[..., 1]
        x2 = centers[:, 0] + ltrb[..., 2]
        y2 = centers[:, 1] + ltrb[..., 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        obj_logits = obj.reshape(B, -1)
        emb = F.normalize(emb.permute(0, 2, 3, 1).reshape(B, -1, self.emb_dim), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sims = emb @ proto.t()
        return boxes, obj_logits, sims

    def forward_deploy(self, x):
        obj, box, emb = self.forward_raw(x)
        return self.decode(obj, box, emb)

    def forward(self, x):
        return self.forward_deploy(x)

    def forward_train(self, x, targets):
        obj, box, emb = self.forward_raw(x)
        return obj, box, emb

    # ------------------------------------------------------------- helpers
    def flops_params(self, input_size: int | None = None):
        s = input_size or self.input_size
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                self.forward_deploy(torch.zeros(1, 3, s, s))
            flops = fc.get_total_flops()
        except Exception:
            flops = -1
        params = sum(p.numel() for p in self.parameters())
        return flops, params
