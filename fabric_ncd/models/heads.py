"""预测头: 异常图 / 布面掩码 / 嵌入 / 原型分类器."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, groups: int = 1, act: bool = True) -> None:
        layers: List[nn.Module] = [
            nn.Conv2d(c_in, c_out, kernel_size, 1, kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if act:
            layers.append(nn.Hardswish(inplace=True))
        super().__init__(*layers)


class AnomalyHead(nn.Module):
    """1/8 特征 -> 1 通道异常 logit."""

    def __init__(self, channels: int, mid: int = 32) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, mid, 1),
            ConvBNAct(mid, mid, 3, groups=mid),
        )
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.out(self.block(feat))


class FabricMaskHead(nn.Module):
    """布面 / 背景二分类 + 布边 (边界带) 预测, 输出在 1/4 尺度."""

    def __init__(self, channels: int, mid: int = 24) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, mid, 1),
            ConvBNAct(mid, mid, 3, groups=mid),
            ConvBNAct(mid, mid, 1),
        )
        self.cloth = nn.Conv2d(mid, 1, 1)
        self.boundary = nn.Conv2d(mid, 1, 1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.block(feat)
        return {"cloth_logit": self.cloth(f), "boundary_logit": self.boundary(f)}


class EmbeddingHead(nn.Module):
    """用于分类 / 新类发现的度量嵌入 (L2 归一化)."""

    def __init__(self, channels: int, dim: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, dim, 1, act=False),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.block(feat), dim=1, eps=1e-6)


class PrototypeClassifier(nn.Module):
    """余弦原型分类器, 支持训练后扩展 (新类发现) 与置信度输出."""

    def __init__(self, dim: int = 64, num_classes: int = 19, temperature: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        prototypes = F.normalize(torch.randn(num_classes, dim), dim=1)
        self.prototypes = nn.Parameter(prototypes)
        self.num_classes = num_classes

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B,dim,C,H,W) 或 (B,dim) -> logits (B,C,*)"""
        proto = F.normalize(self.prototypes, dim=1)
        if emb.dim() == 4:
            b, d, h, w = emb.shape
            logits = torch.einsum("bdhw,cd->bchw", emb, proto) / self.temperature
            return logits
        return emb @ proto.t() / self.temperature

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        prototypes = F.normalize(prototypes.float(), dim=1)
        if prototypes.shape[1] != self.dim:
            raise ValueError(f"原型维度 {prototypes.shape[1]} 与分类器维度 {self.dim} 不一致")
        self.prototypes = nn.Parameter(prototypes.to(self.prototypes.device))
        self.num_classes = prototypes.shape[0]

    @torch.no_grad()
    def add_prototypes(self, prototypes: torch.Tensor) -> None:
        new = F.normalize(prototypes.float(), dim=1).to(self.prototypes.device)
        merged = torch.cat([F.normalize(self.prototypes, dim=1), new], dim=0)
        self.set_prototypes(merged)


class RegionClassifier(nn.Module):
    """对区域做掩码平均池化 + 分类, 供推理与 NCD 复用."""

    def __init__(self, classifier: PrototypeClassifier) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """emb: (B,dim,H,W); mask: (B,1,H,W) 0/1 或软掩码 -> (B,C) 与 (B,dim)."""
        if mask is None:
            pooled = emb.mean(dim=(-2, -1))
        else:
            weight = mask.clamp_min(0)
            denom = weight.sum(dim=(-2, -1)).clamp_min(1e-6)
            pooled = (emb * weight).sum(dim=(-2, -1)) / denom
        pooled = F.normalize(pooled, dim=1)
        return self.classifier(pooled), pooled
