"""免训练异常检测器: 冻结 DINOv2 的中层 patch 特征 + 最近邻记忆库 (PatchCore / AnomalyDINO 思路).

为什么加它:
  * "训练" = 抽一遍正常图特征 + 贪心 coreset 压缩 (GPU 上几十秒~几分钟), 而 Dinomaly 需要 1 万+ iter (1660 上数小时);
  * 推理只跑 encoder 的前 N 层 (不跑 8 层 decoder) + 一次矩阵乘, 比 Dinomaly 更快;
  * 纹理类 (布匹) 数据上, 基于 DINOv2 patch 最近邻的方法 (AnomalyDINO, WACV'25) 在 MVTec-AD 上
    不训练就能达到 ~99% 图级 AUROC. 当"基线 / 快速迭代 / 数据变了先看一眼"用非常合适.

和现有流水线的衔接 (零改动): forward(x) 返回 (en, de) 两个特征图列表, 与 Dinomaly 同接口,
其中 en = 查询 patch 特征, de = 记忆库里最近邻的特征 -> `1 - cos(en, de)` 恰好就是 kNN 余弦距离.
cal_anomaly_maps / calibrate / evaluate_detector / AnomalyDetector / pipeline / infer 都直接可用.
"""
import math
import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinomaly import encoder as encoder_mod

DEFAULT_LAYERS = (5, 7, 9)       # 0-based block 序号; 与 Dinomaly 的目标层 (2~9) 同一段中层, 偏纹理而不是偏语义


@torch.no_grad()
def nearest_neighbor(f: torch.Tensor, bank: torch.Tensor, chunk: int = 8192, want_feat: bool = True):
    """f: (N,C) 已归一化; bank: (M,C) 已归一化 -> (最近邻余弦相似度 (N,), 最近邻特征 (N,C) 或 None). 分块, 显存 O(N*chunk)."""
    best = torch.full((f.shape[0],), -2.0, device=f.device)
    idx = torch.zeros(f.shape[0], dtype=torch.long, device=f.device)
    bank = bank.to(f.device)
    for s in range(0, bank.shape[0], chunk):
        sim = (f @ bank[s:s + chunk].t()).float()
        v, j = sim.max(dim=1)
        upd = v > best
        best = torch.where(upd, v, best)
        idx = torch.where(upd, j + s, idx)
    return best, (bank[idx] if want_feat else None)


class KNNModel(nn.Module):
    def __init__(self, encoder: nn.Module, layers: Sequence[int] = DEFAULT_LAYERS, bank: Optional[torch.Tensor] = None,
                 chunk: int = 8192):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.layers = sorted(int(i) for i in layers)
        depth = len(encoder.blocks)
        if self.layers[-1] >= depth or self.layers[0] < 0:
            raise ValueError(f'layers={self.layers} 超出 encoder 的 {depth} 层')
        self.chunk = int(chunk)
        dim = getattr(encoder, 'embed_dim', None) or encoder.cls_token.shape[-1]
        self.register_buffer('bank', bank if bank is not None else torch.zeros(0, dim))
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = getattr(self.encoder, 'n_storage_tokens', 0)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,N,C) L2 归一化的 patch 特征 (多层先各自归一化再平均, 再归一化)."""
        nreg = self.encoder.num_register_tokens
        h = self.encoder.prepare_tokens(x)
        outs = []
        for i, blk in enumerate(self.encoder.blocks):
            if i > self.layers[-1]:
                break
            h = blk(h)
            if i in self.layers:
                outs.append(F.normalize(h[:, 1 + nreg:].float(), dim=-1))
        return F.normalize(torch.stack(outs, 0).mean(0), dim=-1)

    @torch.no_grad()
    def nearest(self, f: torch.Tensor):
        """f: (N,C) -> (最近邻余弦相似度 (N,), 最近邻特征 (N,C)); 分块算, 显存 O(N*chunk)."""
        if self.bank.shape[0] == 0:
            raise RuntimeError('记忆库为空: 请先 fit (scripts/fit_knn.py) 再推理')
        return nearest_neighbor(f, self.bank, self.chunk)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        f = self.features(x)                                     # (B,N,C)
        side = int(math.isqrt(f.shape[1]))
        en, de = [], []
        for b in range(B):
            _, nn_f = self.nearest(f[b])
            de.append(nn_f.float())
            en.append(f[b].float())
        en = torch.stack(en).permute(0, 2, 1).reshape(B, -1, side, side).contiguous()
        de = torch.stack(de).permute(0, 2, 1).reshape(B, -1, side, side).contiguous()
        return [en], [de]


def build_knn(backbone: str = 'dinov2reg_vit_small_14', layers: Sequence[int] = DEFAULT_LAYERS,
              weights_dir: Optional[str] = None, allow_random_init: bool = False, logger=None,
              bank: Optional[torch.Tensor] = None) -> KNNModel:
    enc = encoder_mod.load_encoder(backbone, weights_dir=weights_dir, allow_random_init=allow_random_init,
                                   logger=logger)
    return KNNModel(enc, layers=layers, bank=bank)


@torch.no_grad()
def kcenter_greedy(x: torch.Tensor, m: int, seed: int = 1) -> torch.Tensor:
    """贪心 k-center (PatchCore 的 coreset): 每次挑离已选集合最远的点. x: (P,d) -> 选中的下标 (m,).
    全程在 x 所在设备上做, 没有逐步的 host 同步."""
    P = x.shape[0]
    m = min(int(m), P)
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(P, (1,), generator=g))
    sel = torch.empty(m, dtype=torch.long, device=x.device)
    sel[0] = first
    xx = (x * x).sum(1)
    mind = (xx - 2 * (x @ x[first]) + xx[first]).clamp_min(0)
    for i in range(1, m):
        nxt = mind.argmax()
        sel[i] = nxt
        d = (xx - 2 * (x @ x[nxt]) + xx[nxt]).clamp_min(0)
        mind = torch.minimum(mind, d)
    return sel


@torch.no_grad()
def fit_knn(model: KNNModel, images: Sequence[np.ndarray], tf: Callable, device: str = 'cuda',
            bank_size: int = 20000, batch_size: int = 8, proj_dim: int = 128, seed: int = 1,
            pool_max: int = 400000, logger=None) -> int:
    """用一批正常图 (RGB ndarray 序列 / 可迭代, 逐张读入) 建记忆库, 返回记忆库大小.

    tf: 与推理一致的预处理 (get_eval_transform). 没有任何梯度训练.
    pool_max: 参与 coreset 选择的候选 patch 上限 (超过则每张图随机抽样), 控制内存与耗时.
    """
    log = logger or (lambda *a, **k: print(*a))
    model.eval().to(device)
    rng = np.random.default_rng(seed)
    n_img = len(images)
    per_img = None
    feats, batch = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        f = model.features(x)                                   # (B,N,C)
        for b in range(f.shape[0]):
            fb = f[b]
            if per_img is not None and fb.shape[0] > per_img:
                sel = torch.from_numpy(rng.choice(fb.shape[0], per_img, replace=False)).to(device)
                fb = fb[sel]
            feats.append(fb.half().cpu())
        batch.clear()

    side_tokens = None
    for i in range(n_img):
        img = images[i]
        if isinstance(img, tuple):          # NormalImageDataset 返回 (img, path)
            img = img[0]
        batch.append(tf(img))
        if side_tokens is None:
            side_tokens = (batch[0].shape[-1] // 14) * (batch[0].shape[-2] // 14)
            per_img = max(int(pool_max // max(n_img, 1)), 1) if side_tokens * n_img > pool_max else None
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 100 == 0:
            log(f'[knn] 抽特征 {i + 1}/{n_img}')
    flush()
    pool = torch.cat(feats).to(device)                          # (P,C) fp16
    P = pool.shape[0]
    m = min(int(bank_size), P)
    log(f'[knn] 候选 patch {P}, coreset -> {m}')
    if m >= P:
        sel = torch.arange(P, device=device)
    else:
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(pool.shape[1], proj_dim, generator=g).to(device) / math.sqrt(proj_dim)
        sel = kcenter_greedy(pool.float() @ proj, m, seed=seed)
    model.bank = F.normalize(pool[sel].float(), dim=-1).contiguous()     # 候选池存的是 fp16, 取出后重新归一化 (点积才严格等于余弦)
    log(f'[knn] 记忆库 {tuple(model.bank.shape)} ({model.bank.numel() * 4 / 1e6:.1f} MB)')
    return int(model.bank.shape[0])


def fit_knn_best(normal_dir: str, out_path: str, backbone: str = 'dinov2reg_vit_small_14', image_size: int = 448,
                 layer_sets: str = '5,7,9', bank_size: int = 20000, max_images: int = 300, batch_size: int = 8,
                 smooth_sigma: float = 4.0, val_images=None, val_labels=None, normal_val_dir=None,
                 eval_max_images: int = 0, device: str = 'cuda', seed: int = 1, weights_dir=None,
                 allow_random_init: bool = False, logger=None) -> str:
    """对每组层建一个记忆库并在 val 上评估, 把 (I-AUROC, 相对间隙) 最优的存成 out_path. 返回 out_path.
    layer_sets: 逗号=多层平均, 分号=多组候选. scripts/fit_knn.py 与 train_detector.py (训练后自动建库) 共用."""
    import random
    from .data.yolo import NormalImageDataset
    from .detect import AnomalyDetector, evaluate_detector, get_eval_transform
    log = logger or (lambda *a, **k: print(*a))
    ds = NormalImageDataset(normal_dir, verify=True, logger=log)
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    ids = ids[:max_images] if max_images else ids
    log(f'用 {len(ids)}/{len(ds)} 张正常图建记忆库')

    class _Sub:
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            return ds[ids[i]]

    tf = get_eval_transform(image_size, image_size)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    best_key = None
    sets = [tuple(int(v) for v in x.split(',') if v.strip()) for x in layer_sets.split(';') if x.strip()]
    for layers in sets:
        log(f'===== kNN layers={layers} =====')
        model = build_knn(backbone, layers=layers, weights_dir=weights_dir, allow_random_init=allow_random_init,
                          logger=log)
        fit_knn(model, _Sub(), tf, device=device, bank_size=bank_size, batch_size=batch_size, seed=seed, logger=log)
        hp = {'arch': 'knn', 'backbone': backbone, 'image_size': image_size, 'crop_size': image_size,
              'layers': list(layers), 'bank_size': int(model.bank.shape[0]), 'smooth_sigma': smooth_sigma}
        det = AnomalyDetector(model, device, image_size, image_size, smooth_sigma=smooth_sigma)
        key = (0.0, 0.0)
        if val_images and os.path.isdir(val_images):
            st = evaluate_detector(model, val_images, val_labels, device, image_size, image_size, logger=log,
                                   normal_dir=normal_val_dir, smooth_sigma=smooth_sigma,
                                   max_images=eval_max_images, return_stats=True)
            a, g = st['i_auroc'], st['gap_rel']
            key = (a if a == a else 0.0, g if g == g else -1e9)
        if best_key is None or key > best_key:
            best_key = key
            det.save(out_path, hp)
            log(f'*** 当前最优 kNN layers={layers} key={key}, 已保存 {out_path}')
        del model, det
    return out_path
