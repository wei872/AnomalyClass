"""Dinomaly + kNN 记忆库 融合检测器 (共用同一个冻结 encoder, 只多一次最近邻搜索).

动机: 重建式 (Dinomaly, 带缺陷监督) 与 记忆库式 (AnomalyDINO/PatchCore) 的误差模式不同 —— 前者对"能重建出来的
缺陷"会漏, 后者对"记忆库没覆盖到的正常变化"会误报. 两张异常图各自按"正常图典型分数"归一化后加权融合,
是工业异常检测里最稳的涨点手段之一.

  fused = w * dino_map / ref_d + (1 - w) * knn_map / ref_k

ref_* = 正常 val 图上该图 (平滑后) 最大值的均值, 使两张图的"正常水平"都≈1; w 由 scripts/fuse_detectors.py
在 val 上按 (I-AUROC, 间隙) 自动选, w=1 (纯 Dinomaly) 和 w=0 (纯 kNN) 也是候选, 所以融合在 val 上不会比单个差.
零额外 encoder 开销: kNN 特征直接取 Dinomaly 前向里已经算出的目标层 token (需要 knn_layers ⊆ Dinomaly 目标层 2~9).
"""
import os
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


def fuse_checkpoints(dino_ckpt: str, knn_ckpt: str, out_path: str, normal_dir: str, val_images: str, val_labels: str,
                     weights: str = '0,0.25,0.5,0.75,1', max_images: int = 300, score_ratio: float = 0.0,
                     map_size: int = 224, half: bool = False, device: str = 'cuda', allow_random_init: bool = False,
                     logger=None) -> dict:
    """把 Dinomaly 权重与 kNN 记忆库融合, 在 val 上按 (I-AUROC, 间隙) 选 w, 保存融合权重到 out_path.
    返回 {'w','auroc','gap','path'}. scripts/fuse_detectors.py 与 train_detector.py 共用."""
    import numpy as np
    from .data.yolo import NormalImageDataset, YoloDefectDataset
    from .detect import AnomalyDetector
    from .util import get_logger, roc_auc, score_from_map, torch_load
    log = logger or get_logger('fuse')
    dino_det = AnomalyDetector.load(dino_ckpt, device=device, allow_random_init=allow_random_init, logger=log,
                                    half=half)
    dino_hp = torch_load(dino_ckpt, map_location='cpu')['hparams']
    kck = torch_load(knn_ckpt, map_location='cpu')
    khp = kck['hparams']
    if khp.get('arch') != 'knn':
        raise ValueError(f'knn_ckpt 不是 fit_knn 产出的 kNN 检测器: {knn_ckpt}')
    if khp['backbone'] != dino_hp['backbone'] or int(khp['image_size']) != int(dino_hp['image_size']):
        raise ValueError(f'两个检测器的 backbone / image_size 必须一致: dino=({dino_hp["backbone"]},{dino_hp["image_size"]}) '
                         f'knn=({khp["backbone"]},{khp["image_size"]})')
    bank = kck['model']['bank'].to(device)
    fm = FusionModel(dino_det.model, bank, khp['layers'], w=0.5)
    det = AnomalyDetector(fm, device, dino_det.image_size, dino_det.crop_size,
                          smooth_sigma=dino_det.smooth_sigma, half=half)
    log(f'Dinomaly: {dino_ckpt} | kNN: {knn_ckpt} (layers={khp["layers"]}, bank={tuple(bank.shape)})')
    if not normal_dir or not os.path.isdir(normal_dir) or not val_images or not os.path.isdir(val_images):
        raise FileNotFoundError('需要 normalval(正常) 和 val(带标注缺陷) 目录来选融合权重')
    ds_n = NormalImageDataset(normal_dir, verify=True, logger=log)
    ds_d = YoloDefectDataset(val_images, val_labels, max_side=2048, verify=True, logger=log)

    @torch.no_grad()
    def maps(img):
        x = det.tf(img).unsqueeze(0).to(device)
        if det.half:
            with torch.autocast(device_type=det.device_type, dtype=det.amp_dtype):
                d, k = fm.component_maps(x, x.shape[-1])
        else:
            d, k = fm.component_maps(x, x.shape[-1])
        out = []
        for m in (d, k):
            m = det.gaussian(m.float())
            out.append(F.interpolate(m, size=(map_size, map_size), mode='bilinear',
                                     align_corners=False)[0, 0].cpu().numpy().astype(np.float16))
        return out

    n_n = min(max_images, len(ds_n))
    n_d = min(max_images, len(ds_d))
    log(f'缓存异常图: 正常 {n_n} + 缺陷 {n_d} (各一次前向, 之后扫权重不再跑模型)')
    mn = [maps(ds_n[i][0]) for i in range(n_n)]
    md = []
    for i in range(n_d):
        img, boxes, _, _ = ds_d[i]
        if boxes:
            md.append(maps(img))
    if not mn or not md:
        raise ValueError('需要同时有正常图和带标注的缺陷图')
    sc = lambda m: score_from_map(m.astype(np.float32), score_ratio)
    ref_d = float(np.mean([sc(m[0]) for m in mn]))
    ref_k = float(np.mean([sc(m[1]) for m in mn]))
    log(f'归一化参考 (正常图平均图级分数): ref_d={ref_d:.4f} ref_k={ref_k:.4f}')
    rows = []
    for w in [float(v) for v in weights.split(',')]:
        fuse = lambda m: (w * m[0].astype(np.float32) / ref_d + (1 - w) * m[1].astype(np.float32) / ref_k)
        sn = np.array([score_from_map(fuse(m), score_ratio) for m in mn])
        sd = np.array([score_from_map(fuse(m), score_ratio) for m in md])
        auroc = roc_auc([0] * len(sn) + [1] * len(sd), list(sn) + list(sd))
        gap = float((sd.min() - sn.max()) / max(sn.max(), 1e-8))
        rows.append((w, auroc, gap))
        tag = {1.0: ' (纯 Dinomaly)', 0.0: ' (纯 kNN)'}.get(w, '')
        log(f'w_dinomaly={w:.2f}{tag}: I-AUROC={auroc:.4f} 相对间隙={gap:+.3f}  (正常 max={sn.max():.3f} 缺陷 min={sd.min():.3f})')
    best = max(rows, key=lambda r: (round(r[1], 6), r[2]))
    w = best[0]
    log(f'*** 选定 w_dinomaly={w:.2f}: I-AUROC={best[1]:.4f} 间隙={best[2]:+.3f}')
    fm.w, fm.ref_d, fm.ref_k = w, ref_d, ref_k
    hp = dict(dino_hp)
    hp.update({'arch': 'fusion', 'knn_layers': list(khp['layers']), 'fusion_w': w, 'ref_d': ref_d, 'ref_k': ref_k,
               'bank_size': int(bank.shape[0]), 'smooth_sigma': float(dino_det.smooth_sigma)})
    det.save(out_path, hp)
    log(f'已保存融合权重 {out_path}')
    return {'w': w, 'auroc': best[1], 'gap': best[2], 'path': out_path}


def build_final_detector(out_dir: str, normal_train_dir: str, normal_val_dir: str, val_images: str, val_labels: str,
                         backbone: str, image_size: int, layer_sets: str = '5,7,9;3,5,7', bank_size: int = 20000,
                         max_images: int = 300, device: str = 'cuda', seed: int = 1, weights_dir=None,
                         allow_random_init: bool = False, logger=None) -> dict:
    """Dinomaly 训练结束后: 建 kNN 记忆库 -> 与 Dinomaly 融合 (w 在 val 上选, w=1/w=0 也是候选) -> 写 model_final.pth.
    w=1 -> model_final = Dinomaly (没有额外推理开销); w=0 -> 纯 kNN; 其余 -> 融合权重.
    返回 {'w','path','detector'}. 融合在 val 上不会比单个检测器差."""
    import shutil
    from .knn_detect import fit_knn_best
    log = logger or print
    best = os.path.join(out_dir, 'model_best.pth')
    knn_p = os.path.join(out_dir, 'knn_best.pth')
    fused_p = os.path.join(out_dir, 'model_fused.pth')
    final = os.path.join(out_dir, 'model_final.pth')
    fit_knn_best(normal_train_dir, knn_p, backbone=backbone, image_size=image_size, layer_sets=layer_sets,
                 bank_size=bank_size, max_images=max_images, val_images=val_images, val_labels=val_labels,
                 normal_val_dir=normal_val_dir, device=device, seed=seed, weights_dir=weights_dir,
                 allow_random_init=allow_random_init, logger=log)
    r = fuse_checkpoints(best, knn_p, fused_p, normal_val_dir or normal_train_dir, val_images, val_labels,
                         device=device, allow_random_init=allow_random_init, logger=log)
    w = r['w']
    if w >= 1.0:
        shutil.copyfile(best, final)
        what = '纯 Dinomaly (kNN 融合没有带来提升)'
    elif w <= 0.0:
        shutil.copyfile(knn_p, final)
        what = '纯 kNN (kNN 单独最好)'
    else:
        shutil.copyfile(fused_p, final)
        what = f'Dinomaly + kNN 融合 (w_dinomaly={w:.2f})'
    log(f'最终检测器: {what} -> {final}')
    return {'w': w, 'path': final, 'detector': what}
