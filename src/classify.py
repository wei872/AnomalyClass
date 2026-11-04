"""异常区域分类器: 轻量 CNN (MobileNetV3-Small / ResNet18) 分类缺陷裁剪.

- 类别: fabric19.yaml 的 nc(19) 类 + 追加 "未知类型" (索引 nc).
- 未知判定: 区域分类最大 softmax 概率 < unknown_thr 时判为 "未知类型"
  (unknown_thr 在 val 集上按分位数校准并夹到 [0.5, 0.8], 保证与 argmax 一致).
- 输出每个区域的 20 维概率分布: probs[:nc] 为 19 类, probs[nc] 为未知概率.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.yolo import build_crop_samples, pil_to_tensor_batch
from .util import get_logger

ARCHS = ['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34']


def build_classifier(arch: str = 'mobilenet_v3_small', num_classes: int = 19, pretrained: bool = True,
                     logger=None) -> nn.Module:
    """构建分类器并替换最后一层为 num_classes 输出."""
    import torchvision
    log = logger or (lambda *a, **k: print(*a))
    if arch not in ARCHS:
        raise ValueError(f'不支持的分类器: {arch}, 可选 {ARCHS}')
    weights = 'DEFAULT' if pretrained else None
    try:
        m = getattr(torchvision.models, arch)(weights=weights)
        if hasattr(m, 'classifier'):      # mobilenet: classifier[-1] 的 in_features 才是真正的特征维度
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:                             # resnet: fc
            m.fc = nn.Linear(m.fc.in_features, num_classes)
    except Exception as e:
        log(f'[classifier] 预训练权重加载失败 ({e}), 使用随机初始化训练')
        return build_classifier(arch, num_classes, pretrained=False, logger=log)
    return m


class CropDataset(Dataset):
    """缺陷裁剪数据集: samples = [(crop_rgb, cls, img_path, box)]."""

    def __init__(self, samples, crop_size: int = 224, augment: bool = False):
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        crop, cls, _, _ = self.samples[i]
        x = pil_to_tensor_batch([crop], self.crop_size, augment=self.augment)[0]
        return x, cls


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
    def predict_probs(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """crops: list of RGB ndarray -> (N, nc+1) 概率, 最后一列为未知类概率."""
        n = len(crops)
        out = np.zeros((n, self.nc + 1), dtype=np.float32)
        for s in range(0, n, self.batch_size):
            batch = crops[s:s + self.batch_size]
            if not batch:
                continue
            x = pil_to_tensor_batch(list(batch), self.crop_size, augment=False).to(self.device)
            logits = self.model(x) / self.temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
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
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        meta = ckpt['meta']
        model = build_classifier(meta['arch'], num_classes=nc, pretrained=False, logger=log)
        model.load_state_dict(ckpt['model'])
        log(f'[classifier] 加载权重: {path}')
        return cls(model, nc=nc, unknown_id=nc, unknown_thr=meta.get('unknown_thr', 0.6),
                   temperature=meta.get('temperature', 1.0), device=device, crop_size=meta.get('crop_size', 224))


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
    log = logger or (lambda *a, **k: print(*a))
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
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
                     num_workers: int = 4, seed: int = 1, logger=None) -> str:
    """在 YOLO 标注的缺陷裁剪上训练分类器, 并校准 unknown_thr / temperature. 返回 best 路径."""
    log = logger or get_logger('train_classifier')
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    log(f'收集训练裁剪: {train_images}')
    samples = build_crop_samples(train_images, train_labels, pad=crop_pad)
    log(f'训练样本(缺陷裁剪)数: {len(samples)}')
    if val_images and os.path.isdir(val_images):
        val_samples = build_crop_samples(val_images, val_labels, pad=crop_pad)
        train_samples = samples
        log(f'val 样本数: {len(val_samples)}')
    else:
        train_samples, val_samples = _split_samples(samples, val_ratio, seed)
        log(f'未提供 val, 从 train 划分: train={len(train_samples)} val={len(val_samples)}')
    if not val_samples:
        val_samples = train_samples[:max(1, len(train_samples) // 10)]

    train_ds = CropDataset(train_samples, crop_size, augment=True)
    val_ds = CropDataset(val_samples, crop_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_classifier(arch, num_classes=nc, pretrained=pretrained, logger=log).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_path = os.path.join(out_dir, 'best.pt')
    best_acc = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        losses, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        model.eval()
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                vcorrect += (pred == y).sum().item()
                vtotal += x.size(0)
        acc = vcorrect / max(vtotal, 1)
        log(f'[classifier] epoch {ep}/{epochs} loss={losses / max(total, 1):.4f} '
            f'train_acc={correct / max(total, 1):.4f} val_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save({'model': model.state_dict(), 'meta': {'arch': arch, 'nc': nc, 'crop_size': crop_size,
                                                              'val_acc': acc}}, best_path)
            log(f'  *** 新最优 val_acc={acc:.4f}, 保存 {best_path}')

    # 校准 temperature 和 unknown_thr
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    temperature = calibrate_temperature(model, val_loader, device, log)
    max_probs = _max_probs(model, val_loader, device, temperature)
    unknown_thr = float(np.quantile(max_probs, unknown_quantile))
    unknown_thr = float(min(max(unknown_thr, unknown_min), unknown_max))
    log(f'[classifier] val 最大概率分位数({unknown_quantile})={np.quantile(max_probs, unknown_quantile):.4f} '
        f'-> unknown_thr={unknown_thr:.4f} (未知类阈值)')

    meta = {'arch': arch, 'nc': nc, 'names': names or [], 'crop_size': crop_size, 'crop_pad': crop_pad,
            'unknown_thr': unknown_thr, 'temperature': temperature, 'val_acc': best_acc,
            'unknown_id': nc, 'unknown_name': '未知类型'}
    torch.save({'model': model.state_dict(), 'meta': meta}, best_path)
    with open(os.path.join(out_dir, 'classifier_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'分类器训练完成: {best_path} (val_acc={best_acc:.4f}, unknown_thr={unknown_thr:.3f}, T={temperature:.3f})')
    return best_path
