"""Dinomaly + kNN 记忆库 融合检测器 (共用同一个冻结 encoder, 只多一次最近邻搜索).

动机: 重建式 (Dinomaly, 带缺陷监督) 与 记忆库式 (AnomalyDINO/PatchCore) 的误差模式不同 —— 前者对"能重建出来的
缺陷"会漏, 后者对"记忆库没覆盖到的正常变化"会误报. 两张异常图各自按"正常图典型分数"归一化后加权融合,
是工业异常检测里最稳的涨点手段之一.

  fused = w * dino_map / ref_d + (1 - w) * knn_map / ref_k

ref_* = 正常 val 图上该图 (平滑后) 最大值的均值, 使两张图的"正常水平"都≈1; w 由 scripts/fuse_detectors.py
在 val 上按 (I-AUROC, 间隙) 自动选, w=1 (纯 Dinomaly) 和 w=0 (纯 kNN) 也是候选, 所以融合在 val 上不会比单个差.
零额外 encoder 开销: kNN 特征直接取 Dinomaly 前向里已经算出的目标层 token (需要 knn_layers ⊆ Dinomaly 目标层 2~9).
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knn_detect import nearest_neighbor
from .util import cal_anomaly_maps


class FusionModel(nn.Module):
    def __init__(self, dino: nn.Module, bank: torch.Tensor, knn_layers: Sequence[int], w: float = 0.5,
                 ref_d: float = 1.0, ref_k: float = 1.0, chunk: int = 8192):
        super().__init__()
        self.dino = dino
        self.register_buffer('bank', bank)
        self.knn_layers = [int(i) for i in knn_layers]
        missing = [l for l in self.knn_layers if l not in dino.target_layers]
        if missing:
            raise ValueError(f'knn_layers={self.knn_layers} 必须包含在 Dinomaly 目标层 {dino.target_layers} 内 (缺 {missing}); '
                             f'请用 fit_knn.py --layer-sets 选 2~9 之间的层')
        self.idx = [dino.target_layers.index(l) for l in self.knn_layers]
        self.w, self.ref_d, self.ref_k, self.chunk = float(w), float(ref_d), float(ref_k), int(chunk)

    def forward(self, x):                                   # 与 Dinomaly 同接口 (训练/其他调用方)
        return self.dino(x)

    @torch.no_grad()
    def component_maps(self, x: torch.Tensor, out_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(dino_map, knn_map), 均为 (B,1,S,S) 未归一化未平滑."""
        if self.bank.shape[0] == 0:
            raise RuntimeError('记忆库为空')
        self.dino.keep_raw = True
        try:
            en, de = self.dino(x)
            raw = self.dino._raw_en
        finally:
            self.dino.keep_raw = False
            self.dino._raw_en = None
        d_map, _ = cal_anomaly_maps(en, de, out_size=out_size)
        nreg = self.dino.encoder.num_register_tokens
        feats = [F.normalize(raw[i][:, 1 + nreg:].float(), dim=-1) for i in self.idx]
        f = F.normalize(torch.stack(feats, 0).mean(0), dim=-1)          # (B,N,C)
        side = en[0].shape[-1]
        k = []
        for b in range(f.shape[0]):
            sim, _ = nearest_neighbor(f[b], self.bank, self.chunk, want_feat=False)
            k.append((1.0 - sim).reshape(1, side, side))
        k_map = F.interpolate(torch.stack(k), size=(out_size, out_size), mode='bilinear', align_corners=True)
        return d_map.float(), k_map.float()

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor, out_size: int) -> torch.Tensor:
        d_map, k_map = self.component_maps(x, out_size)
        return self.w * d_map / self.ref_d + (1.0 - self.w) * k_map / self.ref_k
