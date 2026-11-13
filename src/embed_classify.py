"""缺陷分类器 v2: 冻结 DINOv2 嵌入 + 余弦分类头 + Balanced Softmax (面向小样本 / 类别不均衡).

为什么这样做 (对比旧的 "MobileNetV3 端到端微调"):
  * 特征: DINOv2 的 patch/CLS 特征对纹理与细粒度外观远强于 ImageNet 小 CNN; 骨干完全冻结, 稀有类不会被多数类
    "冲掉"表示 (长尾学习里 decoupled 训练的结论: 表示用自然分布学, 分类头再做均衡 — Kang et al. ICLR'20);
  * 均衡: 分类头用 Balanced Softmax / logit adjustment (训练时给 logits 加 tau*log(先验), 推理不加), 理论上
    等价于在均衡测试集上最小化误差 (Ren et al. NeurIPS'20, Menon et al. ICLR'21), 不用损失重加权 + 过采样叠加;
  * 速度: 骨干只前向一次, 特征缓存到内存, 训练分类头只要几秒~几十秒 (旧方案每个 epoch 都重读图 + 反传整个 CNN);
  * 稀有类: 按每类样本数给稀有类多抽几个增强视角 (框抖动/翻转/颜色), 特征级过采样, 成本只是多前向几次.

推理: 与 RegionClassifier 完全同接口 (predict / predict_probs / model(x)->logits / unknown_thr / temperature),
pipeline / evaluate / benchmark 通过 classify.load_classifier 自动识别.
权重文件只存分类头 (几十 KB), 骨干按名字从 backbones/weights 加载 (与检测器共用同一份权重文件).
"""
import json
import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .classify import (CropDataset, RegionClassifier, _split_samples, fit_temperature)
from .data.yolo import build_crop_index
from .dinomaly import encoder as encoder_mod
from .util import get_logger, torch_load


class EmbedExtractor(nn.Module):
    """冻结 encoder -> (B, 3C) 嵌入: [CLS, 末层 patch 均值, 中层 patch 均值] 各自 L2 归一化后拼接 (中层偏纹理)."""

    def __init__(self, encoder: nn.Module, amp: bool = True):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        depth = len(encoder.blocks)
        self.mid = max(depth * 2 // 3 - 1, 0)                   # ViT-S/B 12 层 -> 第 8 块
        self.amp = amp
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = getattr(self.encoder, 'n_storage_tokens', 0)
        self.dim = 3 * (getattr(encoder, 'embed_dim', None) or encoder.cls_token.shape[-1])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, nreg = self.encoder, self.encoder.num_register_tokens
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(self.amp and x.is_cuda)):
            h = enc.prepare_tokens(x)
            mid = None
            for i, blk in enumerate(enc.blocks):
                h = blk(h)
                if i == self.mid:
                    mid = h
            h = enc.norm(h)
        parts = [h[:, 0], h[:, 1 + nreg:].mean(1), mid[:, 1 + nreg:].mean(1)]
        parts = [F.normalize(p.float(), dim=-1) for p in parts]
        return torch.cat(parts, dim=-1) / math.sqrt(len(parts))


class CosineHead(nn.Module):
    """余弦分类头: logits = scale * cos(feat, w_c). 权重按类均值初始化, scale 可学习."""

    def __init__(self, dim: int, nc: int, init_scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nc, dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * F.normalize(f, dim=-1) @ F.normalize(self.weight, dim=-1).t()


class EmbedNet(nn.Module):
    """extractor + head -> logits, 与旧 CNN 分类器的 model(x) 同接口."""

    def __init__(self, extractor: EmbedExtractor, head: CosineHead):
        super().__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x):
        return self.head(self.extractor(x))


class EmbedRegionClassifier(RegionClassifier):
    """推理封装: 继承 RegionClassifier 的 TTA / unknown_thr / 温度逻辑, 只重写权重的存取."""

    def __init__(self, *a, meta: Optional[dict] = None, **k):
        super().__init__(*a, **k)
        self.meta = meta or {}

    def save(self, path: str, meta: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ck = {'head': self.model.head.state_dict(), 'meta': dict(meta, kind='embed')}
        if meta.get('backbone') == 'tiny_test_14':               # 随机初始化的冒烟测试骨干无法按名字复现, 连同存下
            ck['encoder'] = self.model.extractor.encoder.state_dict()
        torch.save(ck, path)

    @classmethod
    def load(cls, path: str, nc: int, names: List[str], device: str = 'cuda', logger=None,
             allow_random_init: bool = False):
        log = logger or (lambda *a, **k: print(*a))
        ck = torch_load(path, map_location='cpu')
        meta = ck['meta']
        enc = encoder_mod.load_encoder(meta['backbone'], weights_dir=meta.get('weights_dir'),
                                       allow_random_init=allow_random_init or 'encoder' in ck, logger=log)
        if 'encoder' in ck:
            enc.load_state_dict(ck['encoder'])
        ext = EmbedExtractor(enc)
        head = CosineHead(ext.dim, nc)
        head.load_state_dict(ck['head'])
        log(f'[classifier] 加载 DINOv2 嵌入分类器: {path} (backbone={meta["backbone"]}, crop={meta["crop_size"]})')
        return cls(EmbedNet(ext, head), nc=nc, unknown_id=nc, unknown_thr=meta.get('unknown_thr', 0.6),
                   temperature=meta.get('temperature', 1.0), device=device, crop_size=meta.get('crop_size', 224),
                   meta=meta)


@torch.no_grad()
def extract_features(extractor: EmbedExtractor, index, crop_size: int, device: str, augment: bool,
                     box_jitter: float, batch_size: int = 64, num_workers: int = 0, seed: int = 0):
    """index 中每个裁剪抽一次嵌入 (augment=True 时带框抖动/翻转/颜色增强). 返回 (feats (n,D) cpu, labels (n,))."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = CropDataset(index, crop_size, augment=augment, box_jitter=box_jitter if augment else 0.0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    fs, ys = [], []
    for x, y in loader:
        fs.append(extractor(x.to(device)).cpu())
        ys.append(y)
    if not fs:
        return torch.zeros(0, extractor.dim), torch.zeros(0, dtype=torch.long)
    return torch.cat(fs), torch.cat(ys)


def _balanced_acc(y, p) -> float:
    rec = [float((p[y == c] == c).mean()) for c in np.unique(y)]
    return float(np.mean(rec)) if rec else 0.0


def train_head(feats: torch.Tensor, labels: torch.Tensor, val_feats: torch.Tensor, val_labels: torch.Tensor, nc: int,
               epochs: int = 200, lr: float = 5e-3, batch_size: int = 512, tau: float = 1.0, wd: float = 1e-4,
               seed: int = 1, device: str = 'cpu', logger=None):
    """在缓存特征上训练余弦头 (Balanced Softmax). 按 val balanced accuracy 选 best. 返回 (head, 指标 dict)."""
    log = logger or (lambda *a, **k: print(*a))
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    head = CosineHead(feats.shape[1], nc).to(device)
    cnt = torch.bincount(labels, minlength=nc).float()
    with torch.no_grad():                                         # 类均值初始化: 一开始就是最近原型分类器
        for c in range(nc):
            if cnt[c] > 0:
                head.weight[c] = F.normalize(feats[labels == c].mean(0), dim=0).to(device)
    # logit adjustment 的先验必须是"训练时实际看到的类别分布" (即特征级过采样之后的 labels 计数)
    prior = (cnt / cnt.sum().clamp_min(1)).clamp_min(1e-6)
    adj = (tau * prior.log()).to(device)
    X, Y = feats.to(device), labels.to(device)
    vX, vY = val_feats.to(device), val_labels.numpy()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    steps = max(10, epochs * max(1, math.ceil(len(X) / batch_size)))      # 步数过少时 OneCycle 的 warmup 段长度为 0 会除零
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2.0 / steps))
    best, best_state = (-1.0, 0.0), None
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        for s in range(0, len(X), batch_size):
            idx = perm[s:s + batch_size]
            loss = F.cross_entropy(head(X[idx]) + adj, Y[idx])    # Balanced Softmax: 训练加先验, 推理不加
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 0 or ep == epochs:
            head.eval()
            with torch.no_grad():
                pred = head(vX).argmax(1).cpu().numpy()
            bal, acc = _balanced_acc(vY, pred), float((pred == vY).mean())
            if (bal, acc) > best:
                best = (bal, acc)
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if ep % 50 == 0 or ep == epochs:
                log(f'[embed-head] epoch {ep}/{epochs} loss={loss.item():.4f} val_acc={acc:.4f} val_bal_acc={bal:.4f}')
    head.load_state_dict(best_state)
    head.eval()
    return head, {'val_bal_acc': best[0], 'val_acc': best[1]}


def train_embed_classifier(train_images: str, train_labels: Optional[str], val_images: Optional[str] = None,
                           val_labels: Optional[str] = None, out_dir: str = 'checkpoints/classifier', nc: int = 19,
                           names: Optional[List[str]] = None, backbone: str = 'dinov2reg_vit_small_14',
                           crop_size: int = 224, crop_pad: float = 0.3, views: int = 4, min_per_class: int = 200,
                           max_reps: int = 16, box_jitter: float = 0.3, epochs: int = 200, lr: float = 5e-3,
                           tau: float = 1.0, val_ratio: float = 0.1, unknown_quantile: float = 0.05,
                           unknown_min: float = 0.5, unknown_max: float = 0.8, device: str = 'cuda',
                           num_workers: int = 0, seed: int = 1, logger=None, weights_dir: Optional[str] = None,
                           allow_random_init: bool = False, amp: bool = True) -> str:
    """训练 DINOv2 嵌入分类器, 返回 checkpoint 路径 (out_dir/best.pt).

    views: 每个训练裁剪至少抽几个视角 (第 1 个是干净的, 其余带增强); 稀有类按 min_per_class 目标补到更多视角
           (最多 max_reps 个), 即特征级过采样. crop_size 须为 14 的倍数 (默认 224 = 16x16 个 patch).
    tau:   logit adjustment 强度 (1=Balanced Softmax; 0=普通交叉熵; 0.5~1 都常用).
    """
    from collections import Counter
    log = logger or get_logger('train_embed_classifier')
    if crop_size % 14:
        raise ValueError(f'crop_size={crop_size} 必须是 14 的倍数')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    log(f'收集训练裁剪索引: {train_images}')
    train_index, bad = build_crop_index(train_images, train_labels, pad=crop_pad)
    if bad:
        log(f'警告: 跳过 {len(bad)} 个无法解码的图片, 如: {bad[0]}')
    if val_images and os.path.isdir(val_images):
        val_index, _ = build_crop_index(val_images, val_labels, pad=crop_pad)
    else:
        train_index, val_index = _split_samples(train_index, val_ratio, seed)
        log(f'未提供 val, 从 train 划分: train={len(train_index)} val={len(val_index)}')
    if not val_index:
        val_index = train_index[:max(1, len(train_index) // 10)]
    cnt = Counter(item[1][0] for item in train_index)
    names_all = (names or []) + ['未知类型']
    log(f'训练裁剪 {len(train_index)}, val {len(val_index)}; 每类: ' +
        ', '.join(f'{names_all[c] if c < len(names_all) else c}:{cnt[c]}' for c in sorted(cnt)))

    enc = encoder_mod.load_encoder(backbone, weights_dir=weights_dir, allow_random_init=allow_random_init, logger=log)
    ext = EmbedExtractor(enc, amp=amp).to(device).eval()

    # 特征缓存: 干净视角 + 增强视角 (稀有类更多)
    import time
    t0 = time.perf_counter()
    f0, y0 = extract_features(ext, train_index, crop_size, device, False, 0.0, num_workers=num_workers)
    aug_index = []
    for it_ in train_index:
        reps = int(min(max(views, math.ceil(min_per_class / max(cnt[it_[1][0]], 1))), max_reps))
        aug_index.extend([it_] * max(reps - 1, 0))
    fa, ya = (extract_features(ext, aug_index, crop_size, device, True, box_jitter, num_workers=num_workers,
                               seed=seed + 1) if aug_index else (f0[:0], y0[:0]))
    vf, vy = extract_features(ext, val_index, crop_size, device, False, 0.0, num_workers=num_workers)
    feats, labels = torch.cat([f0, fa]), torch.cat([y0, ya])
    log(f'特征缓存完成: 训练 {len(feats)} (干净 {len(f0)} + 增强 {len(fa)}), val {len(vf)}, '
        f'维度 {feats.shape[1]}, 耗时 {time.perf_counter() - t0:.1f}s')

    t1 = time.perf_counter()
    head, st = train_head(feats, labels, vf, vy, nc, epochs=epochs, lr=lr, tau=tau, seed=seed, device=device, logger=log)
    log(f'分类头训练 {time.perf_counter() - t1:.1f}s: val_acc={st["val_acc"]:.4f} val_bal_acc={st["val_bal_acc"]:.4f}')

    with torch.no_grad():
        val_logits = head(vf.to(device)).cpu()
    temperature = fit_temperature(val_logits, vy, log)
    max_probs = torch.softmax(val_logits / temperature, 1).max(1).values.numpy()
    unknown_thr = float(np.quantile(max_probs, unknown_quantile))
    unknown_thr = float(min(max(unknown_thr, unknown_min), unknown_max))
    log(f'[classifier] val 最大概率 {unknown_quantile:.0%} 分位={np.quantile(max_probs, unknown_quantile):.4f} '
        f'-> unknown_thr={unknown_thr:.4f}')

    meta = {'kind': 'embed', 'backbone': backbone, 'nc': nc, 'names': names or [], 'crop_size': crop_size,
            'crop_pad': crop_pad, 'unknown_thr': unknown_thr, 'temperature': temperature, 'tau': tau,
            'val_acc': st['val_acc'], 'val_bal_acc': st['val_bal_acc'], 'views': views,
            'min_per_class': min_per_class, 'box_jitter': box_jitter, 'unknown_id': nc, 'unknown_name': '未知类型'}
    if weights_dir:
        meta['weights_dir'] = weights_dir
    clf = EmbedRegionClassifier(EmbedNet(ext, head.to(device)), nc=nc, unknown_id=nc, unknown_thr=unknown_thr,
                                temperature=temperature, device=device, crop_size=crop_size, meta=meta)
    best_path = os.path.join(out_dir, 'best.pt')
    clf.save(best_path, meta)
    with open(os.path.join(out_dir, 'classifier_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'分类器训练完成: {best_path} (val_acc={st["val_acc"]:.4f}, val_bal_acc={st["val_bal_acc"]:.4f}, '
        f'unknown_thr={unknown_thr:.3f}, T={temperature:.3f})')
    return best_path
