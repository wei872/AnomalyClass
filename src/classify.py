"""异常区域分类器: 轻量 CNN (MobileNetV3-Small / ResNet18) 分类缺陷裁剪.

- 类别: fabric19.yaml 的 nc(19) 类 + 追加 "未知类型" (索引 nc).
- 未知判定: 区域分类最大 softmax 概率 < unknown_thr 时判为 "未知类型"
  (unknown_thr 在 val 集上按分位数校准并夹到 [0.5, 0.8], 保证与 argmax 一致).
- 输出每个区域的 20 维概率分布: probs[:nc] 为 19 类, probs[nc] 为未知概率.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.yolo import build_crop_index, pil_to_tensor_batch
from .util import get_logger, torch_load, imread_unicode, image_diag

ARCHS = ['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34', 'efficientnet_b0']


def _replace_head(m: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(m, 'classifier'):      # mobilenet/efficientnet: classifier[-1]
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:                             # resnet: fc
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_classifier(arch: str = 'mobilenet_v3_small', num_classes: int = 19, pretrained: bool = True,
                     pretrained_path: Optional[str] = None, logger=None) -> nn.Module:
    """构建分类器并替换最后一层为 num_classes 输出.

    pretrained_path: 本地预训练权重文件 (如浏览器手动下载的 efficientnet_b0_rwightman-7f5810bc.pth),
                     优先于在线下载, 避免国内网络下载损坏导致 hash 校验失败退化成随机初始化.
    """
    import torchvision
    log = logger or (lambda *a, **k: print(*a))
    if arch not in ARCHS:
        raise ValueError(f'不支持的分类器: {arch}, 可选 {ARCHS}')
    if pretrained_path:
        m = getattr(torchvision.models, arch)(weights=None)
        sd = torch_load(pretrained_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # 只保留形状一致的键 (预训练分类头 1000 类与我们的 num_classes 不一致, 会被过滤, 之后由 _replace_head 重建)
        msd = m.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and getattr(v, 'shape', None) == msd[k].shape}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        log(f'[classifier] 本地预训练权重: {pretrained_path} (missing={len(missing)}, unexpected={len(unexpected)}, 均为分类头)')
        return _replace_head(m, num_classes)
    weights = 'DEFAULT' if pretrained else None
    for attempt in (1, 2):
        try:
            m = getattr(torchvision.models, arch)(weights=weights)
            return _replace_head(m, num_classes)
        except Exception as e:
            if attempt == 2 or not pretrained:
                log(f'[classifier] 预训练权重加载失败 ({e}), 使用随机初始化训练')
                return _replace_head(getattr(torchvision.models, arch)(weights=None), num_classes)
            # 删除可能损坏的缓存文件后重试一次
            try:
                from torchvision.models import get_model_weights
                import torch.hub, os as _os
                fname = _os.path.basename(get_model_weights(arch).DEFAULT.url)
                f = _os.path.join(torch.hub.get_dir(), 'checkpoints', fname)
                if _os.path.exists(f):
                    _os.remove(f)
                    log(f'[classifier] 已删除损坏的缓存权重 {f}, 重试下载...')
            except Exception:
                pass


class CropDataset(Dataset):
    """惰性缺陷裁剪数据集: 只保存轻量索引 [(img_path, (cls,cx,cy,w,h), pad)],
    在 __getitem__ 里才读图 + 裁剪. 这样整个数据集可安全 pickle (Windows spawn 的
    DataLoader worker 要 pickle 数据集; 若把所有裁剪像素存在内存里会 MemoryError).
    """

    def __init__(self, index, crop_size: int = 224, augment: bool = False, max_side: int = 1024,
                 box_jitter: float = 0.0, rot90: bool = False):
        self.box_jitter = float(box_jitter)
        self.rot90 = rot90
        self.index = index
        self.crop_size = crop_size
        self.augment = augment
        self.max_side = max_side
        self._last_path = None
        self._last_img = None

    def __len__(self):
        return len(self.index)

    def _load(self, path: str):
        if path != self._last_path:
            img = imread_unicode(path, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f'无法读取图片: {path}  诊断: {image_diag(path)}')
            if self.max_side and max(img.shape[:2]) > self.max_side:
                s = self.max_side / max(img.shape[:2])
                img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                                 interpolation=cv2.INTER_AREA)
            self._last_path = path
            self._last_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._last_img

    def __getitem__(self, i):
        path, (c, cx, cy, w, h), pad = self.index[i]
        img = self._load(path)
        H, W = img.shape[:2]
        if self.augment and self.box_jitter > 0:
            # 框抖动: 推理时的框来自检测器热力图连通域 (大小/位置与人工 GT 框并不一致), 训练时随机缩放+平移 GT 框来模拟
            j = self.box_jitter
            w = w * float(np.exp(np.random.uniform(-j, j)))
            h = h * float(np.exp(np.random.uniform(-j, j)))
            cx = float(np.clip(cx + np.random.uniform(-j, j) * 0.5 * w, 0.0, 1.0))
            cy = float(np.clip(cy + np.random.uniform(-j, j) * 0.5 * h, 0.0, 1.0))
        bw, bh = w * W, h * H
        x1 = max(0, int((cx - w / 2) * W - bw * pad))
        y1 = max(0, int((cy - h / 2) * H - bh * pad))
        x2 = min(W, int(np.ceil((cx + w / 2) * W + bw * pad)))
        y2 = min(H, int(np.ceil((cy + h / 2) * H + bh * pad)))
        if x2 - x1 < 8:
            x2 = min(W, x1 + 8)
        if y2 - y1 < 8:
            y2 = min(H, y1 + 8)
        crop = img[y1:y2, x1:x2]
        x = pil_to_tensor_batch([crop], self.crop_size, augment=self.augment, rot90=self.rot90)[0]
        return x, c


class RegionClassifier:
    """封装分类器推理: 输入一批 RGB 裁剪 -> 20 维概率 (含未知类)."""

    def __init__(self, model: nn.Module, nc: int, unknown_id: int, unknown_thr: float = 0.6,
                 temperature: float = 1.0, device: str = 'cuda', crop_size: int = 224, batch_size: int = 32):
        self.model = model.to(device).eval()
        self.nc = nc
        self.unknown_id = unknown_id
        self.unknown_thr = float(min(max(unknown_thr, 0.5), 0.8))
        self.temperature = float(max(temperature, 1e-3))
        self.device = device
        self.crop_size = crop_size
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_probs(self, crops: Sequence[np.ndarray], tta: bool = True) -> np.ndarray:
        """crops: list of RGB ndarray -> (N, nc+1) 概率, 最后一列为未知类概率.

        tta=True 时对每张裁剪再做一次水平翻转, 两次 softmax 平均 (提升细粒度纹理分类稳定性).
        """
        n = len(crops)
        out = np.zeros((n, self.nc + 1), dtype=np.float32)
        for s in range(0, n, self.batch_size):
            batch = crops[s:s + self.batch_size]
            if not batch:
                continue
            xs = [pil_to_tensor_batch(list(batch), self.crop_size, augment=False)]
            if tta:
                xs.append(pil_to_tensor_batch([c[:, ::-1].copy() for c in batch], self.crop_size, augment=False))
            x = torch.cat(xs).to(self.device)
            logits = self.model(x) / self.temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            if tta:
                probs = 0.5 * (probs[:len(batch)] + probs[len(batch):])
            m = probs.max(axis=1)
            u = (1.0 - m) * (m < self.unknown_thr)          # 未知概率
            probs = probs * (1.0 - u)[:, None]
            out[s:s + len(batch), :self.nc] = probs
            out[s:s + len(batch), self.unknown_id] = u
        return out

    def predict(self, crops: Sequence[np.ndarray]) -> List[Dict]:
        probs = self.predict_probs(crops)
        results = []
        for p in probs:
            cid = int(np.argmax(p))
            results.append({'class_id': cid, 'class_conf': float(p[cid]), 'probs': p.tolist()})
        return results

    def save(self, path: str, meta: Dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'meta': meta}, path)

    @classmethod
    def load(cls, path: str, nc: int, names: List[str], device: str = 'cuda', logger=None):
        log = logger or (lambda *a, **k: print(*a))
        ckpt = torch_load(path, map_location='cpu')
        meta = ckpt['meta']
        model = build_classifier(meta['arch'], num_classes=nc, pretrained=False, logger=log)
        model.load_state_dict(ckpt['model'])
        log(f'[classifier] 加载权重: {path}')
        return cls(model, nc=nc, unknown_id=nc, unknown_thr=meta.get('unknown_thr', 0.6),
                   temperature=meta.get('temperature', 1.0), device=device, crop_size=meta.get('crop_size', 224))


def load_classifier(path: str, nc: int, names: List[str], device: str = 'cuda', logger=None,
                    allow_random_init: bool = False) -> 'RegionClassifier':
    """按 checkpoint 里的 meta['kind'] 选择加载 CNN 分类器 (默认) 或 DINOv2 嵌入分类器 ('embed')."""
    ckpt = torch_load(path, map_location='cpu')
    if ckpt.get('meta', {}).get('kind') == 'embed':
        from .embed_classify import EmbedRegionClassifier
        return EmbedRegionClassifier.load(path, nc=nc, names=names, device=device, logger=logger,
                                          allow_random_init=allow_random_init)
    return RegionClassifier.load(path, nc=nc, names=names, device=device, logger=logger)


# ------------------------- 训练 -------------------------
def _split_samples(samples, val_ratio: float, seed: int = 1):
    import random
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio)) if len(idx) > 1 else 0
    val_idx = set(idx[:n_val])
    train = [samples[i] for i in idx if i not in val_idx]
    val = [samples[i] for i in idx if i in val_idx]
    return train, val


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: str, logger=None) -> float:
    """在 val 上用 1D 温度缩放校准 softmax 温度 (降低 NLL)."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    return fit_temperature(torch.cat(logits), torch.cat(labels), logger)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, logger=None) -> float:
    """已有 val logits / 标签时直接拟合温度 (LBFGS, 降低 NLL), 范围夹到 [0.05, 10]."""
    log = logger or (lambda *a, **k: print(*a))
    T_param = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([T_param], lr=0.5, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / T_param, labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        t = float(T_param.item())
    except Exception:
        t = 1.0
    t = float(min(max(t, 0.05), 10.0))
    log(f'[classifier] 温度缩放 T={t:.3f}')
    return t


@torch.no_grad()
def _max_probs(model: nn.Module, loader: DataLoader, device: str, temperature: float = 1.0) -> np.ndarray:
    model.eval()
    out = []
    for x, y in loader:
        logits = model(x.to(device)) / max(temperature, 1e-3)
        out.append(torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_classifier(train_images: str, train_labels: Optional[str], val_images: Optional[str] = None,
                     val_labels: Optional[str] = None, out_dir: str = 'checkpoints/classifier', nc: int = 19,
                     names: Optional[List[str]] = None, arch: str = 'mobilenet_v3_small', crop_size: int = 224,
                     crop_pad: float = 0.3, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                     val_ratio: float = 0.1, unknown_quantile: float = 0.05, unknown_min: float = 0.5,
                     unknown_max: float = 0.8, pretrained: bool = True, device: str = 'cuda',
                     num_workers: int = 4, seed: int = 1, logger=None, pretrained_path: Optional[str] = None,
                     box_jitter: float = 0.3, rot90: bool = False, balance_power: float = 1.0) -> str:
    """在 YOLO 标注的缺陷裁剪上训练分类器, 并校准 unknown_thr / temperature. 返回 best 路径.

    box_jitter: 训练裁剪时对 GT 框做随机缩放/平移的幅度 (0=关), 模拟推理时检测器给出的框.
    rot90: 是否加随机 90° 旋转增强 (默认关: 经线/纬线类别靠方向区分).
    balance_power: 类别均衡强度, 只通过加权采样实现 (采样权重 ∝ n_c^-power; 1=完全均衡, 0=自然分布).
                   旧版同时用了加权采样 + 逆频率损失权重, 少数类被补偿了两次 (有效权重 ∝ 1/n_c²), 现只保留采样.
    best 按 val 的 **balanced accuracy** (各类召回的平均) 选, 而不是总体准确率 (多数类会掩盖少数类).
    """
    log = logger or get_logger('train_classifier')
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    log(f'收集训练裁剪索引: {train_images}')
    train_index, bad = build_crop_index(train_images, train_labels, pad=crop_pad)
    if bad:
        log(f'警告: 跳过 {len(bad)} 个无法解码的图片, 如: {bad[0]}')
    log(f'训练样本(缺陷裁剪)数: {len(train_index)}')
    if val_images and os.path.isdir(val_images):
        val_index, _ = build_crop_index(val_images, val_labels, pad=crop_pad)
        log(f'val 样本数: {len(val_index)}')
    else:
        train_index, val_index = _split_samples(train_index, val_ratio, seed)
        log(f'未提供 val, 从 train 划分: train={len(train_index)} val={len(val_index)}')
    if not val_index:
        val_index = train_index[:max(1, len(train_index) // 10)]

    train_ds = CropDataset(train_index, crop_size, augment=True, box_jitter=box_jitter, rot90=rot90)
    val_ds = CropDataset(val_index, crop_size, augment=False)

    # 每类样本数统计 + 类别权重 (19 类样本不均时, 少数类权重更大) + 加权随机采样
    from collections import Counter
    cnt = Counter(item[1][0] for item in train_index)
    names_all = (names or []) + ['未知类型']
    log('每类训练样本数: ' + ', '.join(f'{names_all[c] if c < len(names_all) else c}:{cnt.get(c, 0)}'
                                        for c in sorted(cnt)))
    # 类别均衡只用加权采样 (损失不再加权, 避免重复补偿)
    cw = np.array([max(cnt.get(c, 0), 1) ** (-float(balance_power)) for c in range(nc)], dtype=np.float64)
    cw = cw / cw[[c for c in range(nc) if cnt.get(c, 0) > 0]].mean() if cnt else cw
    sample_weights = torch.tensor([float(cw[item[1][0]]) for item in train_index], dtype=torch.float32)
    class_weight = None
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_index),
                                                     replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_classifier(arch, num_classes=nc, pretrained=pretrained,
                             pretrained_path=pretrained_path, logger=log).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_path = os.path.join(out_dir, 'best.pt')
    best_acc = -1.0                      # 这里记录的是 best 的 balanced accuracy
    best_plain_acc = 0.0
    bad_epochs = 0
    patience = 15
    mixup_p = 0.5          # MixUp 概率
    for ep in range(1, epochs + 1):
        model.train()
        losses, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # MixUp 增强 (细粒度纹理分类常用, 提升泛化)
            if np.random.rand() < mixup_p:
                lam = float(np.random.beta(0.2, 0.2))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                logits = model(x)
                loss = lam * nn.functional.cross_entropy(logits, y, weight=class_weight, label_smoothing=0.05) + \
                       (1 - lam) * nn.functional.cross_entropy(logits, y[perm], weight=class_weight,
                                                              label_smoothing=0.05)
            else:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y, weight=class_weight, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        model.eval()
        vt, vp = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(model(x.to(device)).argmax(1).cpu())
                vt.append(y)
        vt, vp = torch.cat(vt).numpy(), torch.cat(vp).numpy()
        plain_acc = float((vt == vp).mean()) if len(vt) else 0.0
        recalls = [float((vp[vt == c] == c).mean()) for c in np.unique(vt)]
        acc = float(np.mean(recalls)) if recalls else 0.0           # balanced accuracy
        log(f'[classifier] epoch {ep}/{epochs} loss={losses / max(total, 1):.4f} '
            f'train_acc={correct / max(total, 1):.4f} val_acc={plain_acc:.4f} val_bal_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_plain_acc = plain_acc
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'meta': {'arch': arch, 'nc': nc, 'crop_size': crop_size,
                                                              'val_acc': plain_acc, 'val_bal_acc': acc}}, best_path)
            log(f'  *** 新最优 val_bal_acc={acc:.4f} (val_acc={plain_acc:.4f}), 保存 {best_path}')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f'早停: {patience} 个 epoch val_bal_acc 未提升')
                break

    # 校准 temperature 和 unknown_thr
    ckpt = torch_load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    temperature = calibrate_temperature(model, val_loader, device, log)
    max_probs = _max_probs(model, val_loader, device, temperature)
    unknown_thr = float(np.quantile(max_probs, unknown_quantile))
    unknown_thr = float(min(max(unknown_thr, unknown_min), unknown_max))
    log(f'[classifier] val 最大概率分位数({unknown_quantile})={np.quantile(max_probs, unknown_quantile):.4f} '
        f'-> unknown_thr={unknown_thr:.4f} (未知类阈值)')

    meta = {'arch': arch, 'nc': nc, 'names': names or [], 'crop_size': crop_size, 'crop_pad': crop_pad,
            'unknown_thr': unknown_thr, 'temperature': temperature, 'val_acc': best_plain_acc,
            'val_bal_acc': best_acc, 'box_jitter': box_jitter, 'rot90': bool(rot90),
            'unknown_id': nc, 'unknown_name': '未知类型'}
    torch.save({'model': model.state_dict(), 'meta': meta}, best_path)
    with open(os.path.join(out_dir, 'classifier_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'分类器训练完成: {best_path} (val_acc={best_plain_acc:.4f}, val_bal_acc={best_acc:.4f}, unknown_thr={unknown_thr:.3f}, T={temperature:.3f})')
    return best_path
