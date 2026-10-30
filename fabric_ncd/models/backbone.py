"""轻量级多尺度主干网络.

两种实现:

``TorchVisionBackbone``
    直接复用 ``torchvision.models`` 的 MobileNetV3 / EfficientNet, 支持 ImageNet
    预训练权重, 用于线上训练 (泛化性主要来源之一). 特征层通过一次前向探测自动
    定位, 不依赖版本相关的层索引.

``LiteBackbone``
    纯 PyTorch 实现 (MobileNetV3-Small 拓扑), 不依赖 torchvision, 用于无网络
    环境 / 单元测试 / 端侧部署. 参数量约 2.5M (width=1.0), 512 输入约 0.2 GFLOPs.

两者接口一致: ``forward(x) -> {"p2": 1/4, "p3": 1/8, "p4": 1/16, "p5": 1/32}``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger("fabric_ncd.backbone")

STRIDES = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: str = "hswish",
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if act == "hswish":
            layers.append(nn.Hardswish(inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU6(inplace=True))
        elif act in ("none", "identity", None):
            pass
        else:  # pragma: no cover
            raise ValueError(f"未知激活: {act}")
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class InvertedResidual(nn.Module):
    """MobileNetV3 的倒残差块 (扩展 -> 深度可分离 -> SE -> 投影)."""

    def __init__(
        self,
        c_in: int,
        c_exp: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
    ) -> None:
        super().__init__()
        self.use_res = stride == 1 and c_in == c_out
        layers: List[nn.Module] = []
        if c_in != c_exp:
            layers.append(ConvBNAct(c_in, c_exp, 1, 1, act="hswish"))
        layers.append(ConvBNAct(c_exp, c_exp, kernel_size, stride, groups=c_exp, act="hswish"))
        if use_se:
            layers.append(SqueezeExcite(c_exp))
        layers.append(ConvBNAct(c_exp, c_out, 1, 1, act="none"))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res else out


def _round_channels(ch: int, width: float, divisor: int = 8) -> int:
    ch = int(ch * width)
    return max(divisor, int((ch + divisor / 2) // divisor * divisor))


# --------------------------------------------------------------------------- #
# 纯 PyTorch 主干 (MobileNetV3-Small 拓扑)
# --------------------------------------------------------------------------- #
class LiteBackbone(nn.Module):
    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()
        c = lambda ch: _round_channels(ch, width_mult)  # noqa: E731
        self.features = nn.Sequential(
            ConvBNAct(3, c(16), 3, 2),  # 1/2
            ConvBNAct(c(16), c(16), 3, 2),  # 1/4  -> p2
            InvertedResidual(c(16), c(16), c(16), 3, 2, True),  # 1/8 -> p3
            InvertedResidual(c(16), c(72), c(24), 3, 1, False),
            InvertedResidual(c(24), c(88), c(24), 3, 2, False),  # 1/16 -> p4
            InvertedResidual(c(24), c(96), c(40), 5, 2, True),
            InvertedResidual(c(40), c(240), c(40), 5, 1, True),
            InvertedResidual(c(40), c(240), c(40), 5, 1, True),
            InvertedResidual(c(40), c(120), c(48), 5, 1, True),
            InvertedResidual(c(48), c(144), c(48), 5, 1, True),
            InvertedResidual(c(48), c(288), c(96), 5, 2, True),  # 1/32 -> p5
            InvertedResidual(c(96), c(576), c(96), 5, 1, True),
            InvertedResidual(c(96), c(576), c(96), 5, 1, True),
            ConvBNAct(c(96), c(576), 1, 1),
        )
        # 与 features 下标对齐的抽头 (1/4, 1/8, 1/16, 1/32)
        self.taps: Dict[int, str] = {1: "p2", 3: "p3", 4: "p4", 10: "p5"}
        self.out_channels: Dict[str, int] = {
            "p2": c(16),
            "p3": c(24),
            "p4": c(24),
            "p5": c(96),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.taps:
                outs[self.taps[idx]] = x
        return outs


# --------------------------------------------------------------------------- #
# torchvision 主干 (可选 ImageNet 预训练)
# --------------------------------------------------------------------------- #
class TorchVisionBackbone(nn.Module):
    """包装 torchvision 主干, 自动探测 p2..p5 抽头位置.

    支持的 arch: ``mobilenet_v3_small``, ``mobilenet_v3_large``,
    ``efficientnet_b0`` / ``b1`` / ``b2``, ``squeezenet1_1``.
    """

    _SUPPORTED = (
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "squeezenet1_1",
    )

    def __init__(
        self,
        arch: str = "mobilenet_v3_small",
        pretrained: bool = True,
        image_size: int = 512,
        weights_path: str | None = None,
        max_channels: int = 320,
    ) -> None:
        super().__init__()
        if arch not in self._SUPPORTED:
            raise ValueError(f"不支持的 arch={arch}, 可选: {self._SUPPORTED}")
        cfg_max_channels = max_channels
        self.arch = arch
        self.features = self._build(arch, pretrained)
        self.taps, self.out_channels = self._probe(image_size, int(cfg_max_channels))
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state.get("model", state))
            missing, unexpected = self.load_state_dict(state, strict=False)
            LOGGER.info("载入自定义主干权重 %s (missing=%d, unexpected=%d)", weights_path, len(missing), len(unexpected))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Sequential:
        try:
            import torchvision.models as tvm
        except Exception as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "未安装 torchvision, 请把 model.backbone.arch 设为 'lite' 或安装 torchvision"
            ) from exc

        weights = None
        if pretrained:
            weights = "DEFAULT"
        try:
            model = getattr(tvm, arch)(weights=weights)
        except Exception as exc:  # 无网络 / 权重下载失败时退化为随机初始化
            LOGGER.warning("ImageNet 预训练权重加载失败 (%s), 退化为随机初始化", exc)
            model = getattr(tvm, arch)(weights=None)
        return model.features

    # ------------------------------------------------------------------ #
    def _probe(self, image_size: int, max_channels: int = 320) -> Tuple[Dict[int, str], Dict[str, int]]:
        """跑一次假前向, 找出每个目标 stride 上的特征层.

        规则: 在目标 stride 上取 **最深的、通道数不超过 max_channels 的** 层.
        这样可以跳过 MobileNetV3 末尾的 576/960 维 1x1 投影, 避免无谓的
        计算量 (该投影对分割/度量任务帮助有限).
        """
        self.features.eval()
        taps: Dict[int, str] = {}
        channels: Dict[str, int] = {}
        was_training = self.training
        with torch.no_grad():
            x = torch.zeros(1, 3, image_size, image_size)
            records: List[Tuple[int, int, int]] = []  # (idx, stride, ch)
            for idx, layer in enumerate(self.features):
                x = layer(x)
                stride = max(1, int(round(image_size / x.shape[-1])))
                records.append((idx, stride, int(x.shape[1])))

            for key, stride in STRIDES.items():
                candidates = [(i, ch) for i, s, ch in records if s == stride]
                if not candidates:  # 该 stride 不存在 -> 取最接近的
                    nearest = min({s for _, s, _ in records}, key=lambda s: (abs(s - stride), -s))
                    candidates = [(i, ch) for i, s, ch in records if s == nearest]
                small = [c for c in candidates if c[1] <= max_channels]
                idx, ch = (small or candidates)[-1]
                if idx in taps:  # 极端情况下两个 stride 落到同一层
                    continue
                taps[idx] = key
                channels[key] = ch
            for key in STRIDES:
                if key not in channels:  # pragma: no cover
                    last = records[-1]
                    taps[last[0]] = key
                    channels[key] = last[2]
        self.features.train(was_training)
        return taps, channels

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.taps:
                outs[self.taps[idx]] = x
        return outs

    def freeze_stages(self, num_stages: int) -> None:
        """冻结前 num_stages 个子模块 (浅层纹理/边缘特征), 降低过拟合."""
        for idx, layer in enumerate(self.features):
            if idx < num_stages:
                for p in layer.parameters():
                    p.requires_grad = False
                for m in layer.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()


def build_backbone(cfg: Dict) -> nn.Module:
    arch = str(cfg.get("arch", "mobilenet_v3_small")).lower()
    if arch in ("lite", "lite_backbone", "custom"):
        return LiteBackbone(width_mult=float(cfg.get("width_mult", 1.0)))
    backbone = TorchVisionBackbone(
        arch=arch,
        pretrained=bool(cfg.get("pretrained", True)),
        image_size=int(cfg.get("image_size", 512)),
        weights_path=cfg.get("weights_path"),
        max_channels=int(cfg.get("max_channels", 320)),
    )
    if int(cfg.get("freeze_stages", 0)) > 0:
        backbone.freeze_stages(int(cfg["freeze_stages"]))
    return backbone
