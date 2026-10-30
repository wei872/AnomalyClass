"""正常样本特征记忆库.

无监督异常定位的核心: 只用 **正常布面** 的 patch 嵌入建库, 推理时计算每个
patch 到库的余弦距离, 作为 "没见过 = 异常" 的连续分数. 与学习式异常头互补:

* 记忆库不依赖任何疵点标签 -> 对新布种 / 新疵点类型有更好的开放集泛化;
* 学习式异常头对小/弱疵点更敏感, 且能利用少量标注.

库采用 reservoir + k-center 贪心抽样, 容量固定 (默认 2048 x 64), 显存/内存
开销 ~0.5 MB, 完全满足轻量化要求.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, capacity: int = 2048, dim: int = 64, eps: float = 1e-6, query_stride: int = 1) -> None:
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.eps = eps
        # >1 时先把 query 特征平均池化再算相似度: 4 倍减少乘加, 分数图上采样回去,
        # 对区域级检测几乎无损, 是实时推理里最划算的一档.
        self.query_stride = int(query_stride)
        self.register_buffer("features", torch.zeros(capacity, dim), persistent=True)
        self.register_buffer("size", torch.zeros(1, dtype=torch.long), persistent=True)

    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        return int(self.size.item()) > 0

    def reset(self) -> None:
        self.features.zero_()
        self.size.zero_()

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, use_kcenter: bool = True) -> None:
        """embeddings: (M, d) 或 (B, d, H, W), 已 L2 归一化."""
        if embeddings.dim() == 4:
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, embeddings.shape[1])
        embeddings = F.normalize(embeddings.float(), dim=1)
        if embeddings.numel() == 0:
            return

        current = int(self.size.item())
        combined = torch.cat([self.features[:current], embeddings], dim=0)
        if combined.shape[0] <= self.capacity:
            selected = combined
        elif use_kcenter:
            selected = self._greedy_kcenter(combined, self.capacity)
        else:  # 随机蓄水池
            idx = torch.randperm(combined.shape[0], device=combined.device)[: self.capacity]
            selected = combined[idx]
        selected = F.normalize(selected, dim=1)
        self.features[: selected.shape[0]] = selected.to(self.features.device)
        self.size[0] = selected.shape[0]

    @staticmethod
    def _greedy_kcenter(features: torch.Tensor, k: int) -> torch.Tensor:
        """贪心 k-center: 先随机种子, 再反复选离已选集合最远的点."""
        device = features.device
        n = features.shape[0]
        # 先做一次随机下采样, 控制复杂度 (n 很大时)
        if n > 20 * k:
            idx = torch.randperm(n, device=device)[: 20 * k]
            features = features[idx]
            n = features.shape[0]
        selected = [torch.randint(n, (1,), device=device).item()]
        min_dist = 1.0 - features @ features[selected[0]]
        for _ in range(1, k):
            idx = int(torch.argmax(min_dist).item())
            selected.append(idx)
            dist = 1.0 - features @ features[idx]
            min_dist = torch.minimum(min_dist, dist)
        return features[torch.tensor(selected, device=device)]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def similarity(self, embeddings: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """返回每个位置与库的最大余弦相似度 (B,1,H,W) 或 (B,P)."""
        stride = max(1, int(getattr(self, "query_stride", 1)))
        if stride > 1 and embeddings.dim() == 4 and min(embeddings.shape[-2:]) >= stride:
            full = embeddings.shape[-2:]
            small = F.avg_pool2d(embeddings, stride, stride, ceil_mode=True)
            small = F.normalize(small, dim=1)
            sims = self.similarity(small, chunk=chunk)
            return F.interpolate(sims, size=full, mode="bilinear", align_corners=False)
        squeeze = embeddings.dim() == 4
        if squeeze:
            b, d, h, w = embeddings.shape
            flat = embeddings.permute(0, 2, 3, 1).reshape(b, -1, d)
        else:
            flat = embeddings
            b = flat.shape[0]
        if not self.is_ready:
            sims = torch.zeros(b, flat.shape[1], device=embeddings.device)
        else:
            bank = F.normalize(self.features[: int(self.size.item())], dim=1)
            outputs = []
            for start in range(0, flat.shape[1], chunk):
                part = flat[:, start : start + chunk]
                sim = torch.einsum("bpd,nd->bpn", part, bank)
                outputs.append(sim.max(dim=-1).values)
            sims = torch.cat(outputs, dim=1)
        if squeeze:
            sims = sims.view(b, 1, h, w)
        return sims

    def distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """余弦距离 = 1 - 最大相似度."""
        return (1.0 - self.similarity(embeddings)).clamp(0.0, 2.0)

    @torch.no_grad()
    def compactness(self, embeddings: torch.Tensor) -> torch.Tensor:
        """正常特征的紧凑度损失 (可微版本见 ``losses.compact_loss``)."""
        return self.distance(embeddings).mean()


def compact_loss(bank: MemoryBank, embeddings: torch.Tensor) -> torch.Tensor:
    """可微的 "正常嵌入靠近记忆库" 正则项."""
    if not bank.is_ready:
        return embeddings.new_zeros(())
    bank_feat = F.normalize(bank.features[: int(bank.size.item())].detach(), dim=1)
    sim = torch.einsum("bdhw,nd->bnhw", F.normalize(embeddings, dim=1), bank_feat)
    return (1.0 - sim.max(dim=1).values).mean()
