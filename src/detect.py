"""异常检测器: 基于 Dinomaly2 (无监督, 只用正常样本训练).

- 训练: 只用 normaltrain 纯正常图, 学 bottleneck + decoder, 损失为 loose cosine.
- 推理: anomaly map = 1 - cos(en, de) 高斯平滑; 图级异常分数 = map 最大值;
        与正常样本偏离的置信度 = score / (score + img_thr), img_thr 由 normalval 校准.
- 定位: map > mask_thr 的连通域 -> 缺陷框.
"""
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dinomaly import build_dinomaly
from .dinomaly.losses import global_cosine, global_cosine_hm_percent
from .util import (AverageMeter, cal_anomaly_maps, get_gaussian_kernel, list_images, mask_to_boxes,
                   rasterize_boxes, roc_auc, average_precision, f1_max, get_logger, torch_load)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------- transforms -------------------------
def get_eval_transform(image_size: int, crop_size: int):
    """与 Dinomaly2 一致: Resize(image_size) + CenterCrop(crop_size) + Normalize."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _RandomRotate90:
    def __call__(self, img):
        return T.functional.rotate(img, random.choice([0, 90, 180, 270]))


def get_train_transform(image_size: int, crop_size: int):
    """Dinomaly2 的 strong transform (RandomResizedCrop + 翻转 + 旋转90)."""
    scale = (crop_size / image_size) ** 2
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.RandomResizedCrop((crop_size, crop_size), scale=(scale, scale), ratio=(0.95, 1.05)),
        _RandomRotate90(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ------------------------- 检测器 -------------------------
class AnomalyDetector:
    """封装 Dinomaly 模型的推理接口."""

    def __init__(self, model: nn.Module, device: str = 'cuda', image_size: int = 448, crop_size: int = 392,
                 smooth_kernel: int = 5, smooth_sigma: float = 4.0, half: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.offset = (self.image_size - self.crop_size) // 2
        self.tf = get_eval_transform(self.image_size, self.crop_size)
        self.half = half and device.startswith('cuda')
        self.gaussian = get_gaussian_kernel(smooth_kernel, smooth_sigma, device=device)

    @torch.no_grad()
    def anomaly_map_crop(self, img_rgb: np.ndarray) -> np.ndarray:
        """输入 RGB ndarray (任意尺寸) -> crop_size×crop_size 的异常图 (float32, 已高斯平滑)."""
        x = self.tf(img_rgb).unsqueeze(0).to(self.device)
        if self.half:
            x = x.half()
        en, de = self.model(x)
        amap, _ = cal_anomaly_maps(en, de, out_size=x.shape[-1])
        amap = self.gaussian(amap.float())
        return amap[0, 0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, max_map_side: int = 1024) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """返回 (异常图 float32, 图级异常分数, 坐标缩放(sx,sy)).

        异常图长边封顶 max_map_side (防止超大原图把全分辨率图撑爆内存);
        返回的缩放 (sx,sy) = 原图宽高 / 异常图宽高, 用于把异常图上的框映射回原图坐标.
        """
        h0, w0 = img_rgb.shape[:2]
        amap_c = self.anomaly_map_crop(img_rgb)                     # C×C
        score = float(amap_c.max())
        canvas = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        o = self.offset
        canvas[o:o + self.crop_size, o:o + self.crop_size] = amap_c
        long = max(h0, w0)
        if long > max_map_side:
            nh = max(1, round(h0 * max_map_side / long))
            nw = max(1, round(w0 * max_map_side / long))
        else:
            nh, nw = h0, w0
        full = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return full, score, (w0 / nw, h0 / nh)

    # ---- 区域分割 ----
    def segment(self, full_map: np.ndarray, mask_thr: float, min_area_ratio: float = 2e-4,
                dilate: int = 7, box_pad: float = 0.2) -> List[Dict]:
        """异常图 -> 缺陷区域列表 [{box:(x1,y1,x2,y2), area, mean_score}], 坐标为原图像素."""
        h, w = full_map.shape
        min_area = max(int(min_area_ratio * h * w), 4)
        raw = mask_to_boxes((full_map > mask_thr).astype(np.uint8), min_area=min_area, dilate=dilate)
        regions = []
        for x1, y1, x2, y2, area in raw:
            bw, bh = x2 - x1, y2 - y1
            bx1 = max(0, int(x1 - bw * box_pad))
            by1 = max(0, int(y1 - bh * box_pad))
            bx2 = min(w, int(np.ceil(x2 + bw * box_pad)))
            by2 = min(h, int(np.ceil(y2 + bh * box_pad)))
            mean_score = float(full_map[by1:by2, bx1:bx2].mean())
            regions.append({'box': (bx1, by1, bx2, by2), 'area': int(area), 'mean_score': mean_score})
        regions.sort(key=lambda r: -r['mean_score'])
        return regions

    # ---- 序列化 ----
    def save(self, path: str, hparams: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'hparams': hparams}, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda', allow_random_init: bool = False, logger=None,
             half: bool = False):
        log = logger or (lambda *a, **k: print(*a))
        ckpt = torch_load(path, map_location='cpu')
        hp = ckpt['hparams']
        model = build_dinomaly(hp['backbone'], dropout=hp.get('dropout', 0.4), la=bool(hp.get('la', 1)),
                               lc=hp.get('lc', 2), cr=bool(hp.get('cr', 1)), allow_random_init=allow_random_init,
                               logger=log)
        model.load_state_dict(ckpt['model'])
        log(f'[detector] 加载权重: {path}')
        return cls(model, device=device, image_size=hp['image_size'], crop_size=hp['crop_size'], half=half)


# ------------------------- 训练 -------------------------
class NormalTransformDataset(torch.utils.data.Dataset):
    """正常图 + 训练 transform. 必须放在模块级别: Windows 上 DataLoader(num_workers>0)
    用 spawn 方式启动子进程, 会 pickle 数据集类, 函数内的局部类无法 pickle."""

    def __init__(self, images_dir: str, transform, logger=None):
        from .data.yolo import NormalImageDataset
        self.base = NormalImageDataset(images_dir, verify=True, logger=logger)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        return self.transform(img)


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 100, final_ratio: float = 1.0):
        self.optimizer = optimizer
        self.total = max(total_epochs, 1)
        self.warmup = max(warmup_epochs, 1)
        self.final_ratio = final_ratio
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_num < self.warmup:
                lr = base * self.step_num / self.warmup
            else:
                progress = (self.step_num - self.warmup) / max(self.total - self.warmup, 1)
                progress = min(progress, 1.0)
                lr = base * self.final_ratio * 0.5 * (1 + math.cos(math.pi * progress))
            g['lr'] = lr


def train_detector(normal_train_dir: str, out_dir: str, backbone: str = 'dinov2reg_vit_small_14',
                   image_size: int = 448, crop_size: int = 392, dropout: float = 0.4, la: int = 1, lc: int = 2,
                   cr: int = 1, ll: int = 1, ll_ratio: float = 0.9, ll_factor: float = 0.1,
                   total_iters: int = 40000, batch_size: int = 16, lr: float = 2e-3, warmup: int = 100,
                   eval_interval: int = 5000, eval_dir: Optional[str] = None, eval_labels_dir: Optional[str] = None,
                   device: str = 'cuda', num_workers: int = 4, seed: int = 1, weights_dir: str = None,
                   allow_random_init: bool = False, resume: Optional[str] = None, logger=None,
                   save_interval: int = 1000, no_eval: bool = False):
    """在纯正常图上训练 Dinomaly2 检测器. 返回保存路径."""
    from .data.yolo import YoloDefectDataset
    log = logger or get_logger('train_detector')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tf = get_train_transform(image_size, crop_size)
    train_ds = NormalTransformDataset(normal_train_dir, tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    log(f'训练正常样本数: {len(train_ds)}, batch={batch_size}, iters={total_iters}')

    model = build_dinomaly(backbone, dropout=dropout, la=la, lc=lc, cr=cr, weights_dir=weights_dir,
                           allow_random_init=allow_random_init, logger=log).to(device)
    if resume:
        model.load_state_dict(torch_load(resume, map_location='cpu')['model'])
        log(f'从 {resume} 恢复')
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad_(False)   # encoder 冻结 (前向本就在 no_grad 下), 节省显存
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = torch.optim.AdamW([
        {'params': model.bottleneck[0].parameters(), 'lr': lr * 0.1},
        {'params': model.bottleneck[1].parameters()},
        {'params': model.decoder.parameters()},
    ], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-10)
    scheduler = WarmupCosineScheduler(optimizer, total_epochs=total_iters, warmup_epochs=warmup)

    os.makedirs(out_dir, exist_ok=True)
    hparams = {'backbone': backbone, 'image_size': image_size, 'crop_size': crop_size, 'dropout': dropout,
               'la': la, 'lc': lc, 'cr': cr}
    best_path = os.path.join(out_dir, 'model_best.pth')
    last_path = os.path.join(out_dir, 'model_last.pth')
    best_auroc = -1.0

    it = 0
    loss_meter = AverageMeter()
    while it < total_iters:
        for img in train_loader:
            img = img.to(device)
            en, de = model(img)
            p = min(ll_ratio * it / 1000, ll_ratio) if ll else 0
            loss = global_cosine_hm_percent(en, de, p=p, factor=ll_factor) if ll else global_cosine(en, de)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            loss_meter.update(loss.item())
            it += 1
            if it % 100 == 0:
                log(f'iter [{it}/{total_iters}] loss={loss_meter.avg:.4f} lr={optimizer.param_groups[-1]["lr"]:.2e}')
                loss_meter.reset()
            if it % save_interval == 0 or it == total_iters:
                # 先无条件保存 last —— 即使评估炸了, 也不能丢掉已训练的权重
                AnomalyDetector(model, device, image_size, crop_size).save(last_path, hparams)
                log(f'已保存 {last_path}')
                if eval_dir and not no_eval:
                    try:
                        auroc = evaluate_detector(model, eval_dir, eval_labels_dir, device, image_size,
                                                  crop_size, logger=log)
                        if auroc > best_auroc:
                            best_auroc = auroc
                            AnomalyDetector(model, device, image_size, crop_size).save(best_path, hparams)
                            log(f'*** 新最优 I-AUROC={auroc:.4f}, 保存 {best_path}')
                    except Exception as e:
                        log(f'警告: 评估失败 ({e}), 已跳过, 不影响 checkpoint 保存 (last 已保存)')
                model.train()
            if it >= total_iters:
                break
    # 兜底: 保证至少有一个 checkpoint
    AnomalyDetector(model, device, image_size, crop_size).save(last_path, hparams)
    if not os.path.exists(best_path):
        AnomalyDetector(model, device, image_size, crop_size).save(best_path, hparams)
    log(f'训练完成. best={best_path} (I-AUROC={best_auroc:.4f}), last={last_path}')
    return best_path


# ------------------------- 评估 / 校准 -------------------------
@torch.no_grad()
def evaluate_detector(model: nn.Module, images_dir: str, labels_dir: Optional[str], device: str = 'cuda',
                      image_size: int = 448, crop_size: int = 392, batch_size: int = 8, logger=None,
                      eval_size: int = 256, read_max_side: int = 1024) -> float:
    """在带标签的 val 集上评估: 返回图级 AUROC.

    内存有界: 读图长边封顶 read_max_side; 像素级指标统一在 eval_size×eval_size 上计算
    (YOLO 框是归一化坐标, 缩放无关), 不会把异常图放大到原图全分辨率导致 OOM.
    """
    from .data.yolo import YoloDefectDataset
    log = logger or get_logger('eval')
    det = AnomalyDetector(model, device, image_size, crop_size)
    ds = YoloDefectDataset(images_dir, labels_dir, max_side=read_max_side)
    tf = get_eval_transform(image_size, crop_size)
    gt_sp, pr_sp, gt_px, pr_px = [], [], [], []
    for i in range(len(ds)):
        img, boxes, _, path = ds[i]
        h, w = img.shape[:2]
        x = tf(img).unsqueeze(0).to(device)
        en, de = model(x)
        amap, _ = cal_anomaly_maps(en, de, x.shape[-1])
        amap = det.gaussian(amap.float())                       # (1,1,C,C)
        amap = F.interpolate(amap, size=(eval_size, eval_size), mode='bilinear', align_corners=False)
        amap = amap[0, 0].cpu().numpy()
        score = float(amap.max())
        # GT: 像素框 -> eval_size 分辨率掩码
        mask = rasterize_boxes(eval_size, eval_size,
                               [(c, x1 / w * eval_size, y1 / h * eval_size,
                                 x2 / w * eval_size, y2 / h * eval_size) for c, x1, y1, x2, y2 in boxes])
        gt_sp.append(1 if len(boxes) > 0 else 0)
        pr_sp.append(score)
        gt_px.append(mask.ravel())
        pr_px.append(amap.ravel())
    gt_px = np.concatenate(gt_px)
    pr_px = np.concatenate(pr_px)
    gt_sp = np.asarray(gt_sp)
    pr_sp = np.asarray(pr_sp)

    def _safe(fn, y, s, default=float('nan')):
        try:
            if len(np.unique(y)) < 2:
                return default
            return float(fn(y, s))
        except Exception:
            return default

    i_auroc = _safe(roc_auc, gt_sp, pr_sp)
    p_auroc = _safe(roc_auc, gt_px, pr_px)
    i_ap = _safe(average_precision, gt_sp, pr_sp)
    p_ap = _safe(average_precision, gt_px, pr_px)
    i_f1 = _safe(f1_max, gt_sp, pr_sp)
    p_f1 = _safe(f1_max, gt_px, pr_px)
    log(f'[eval] I-AUROC={i_auroc:.4f} I-AP={i_ap:.4f} I-F1={i_f1:.4f} | '
        f'P-AUROC={p_auroc:.4f} P-AP={p_ap:.4f} P-F1={p_f1:.4f} (n={len(gt_sp)})')
    return i_auroc if i_auroc == i_auroc else 0.0   # NaN -> 0.0, 保证 best 逻辑可用


@torch.no_grad()
def calibrate(detector: AnomalyDetector, normal_dir: str, img_quantile: float = 1.0, img_factor: float = 1.0,
              pix_quantile: float = 0.9999, pix_factor: float = 1.0, logger=None) -> Dict:
    """在纯正常样本 (normalval) 上校准阈值:
      img_thr = quantile(正常图分数, img_quantile) * img_factor   (图级判异常阈值)
      pix_thr = quantile(正常图异常图像素, pix_quantile) * pix_factor (像素/区域阈值)
    异常置信度(与正常偏离程度) = score / (score + img_thr).
    """
    log = logger or get_logger('calibrate')
    from .data.yolo import NormalImageDataset
    ds = NormalImageDataset(normal_dir)
    scores, pix_all = [], []
    for i in range(len(ds)):
        img, path = ds[i]
        full, score, _ = detector.predict(img)
        scores.append(score)
        pix_all.append(full.ravel())
        if (i + 1) % 50 == 0:
            log(f'  校准进度 {i + 1}/{len(ds)}')
    scores = np.array(scores)
    pix = np.concatenate(pix_all)
    img_thr = float(np.quantile(scores, img_quantile) * img_factor)
    pix_thr = float(np.quantile(pix, pix_quantile) * pix_factor)
    result = {
        'normal_dir': os.path.abspath(normal_dir),
        'n_normal': int(len(scores)),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'score_max': float(scores.max()),
        'img_thr': img_thr,
        'pix_thr': pix_thr,
        'img_quantile': img_quantile,
        'pix_quantile': pix_quantile,
    }
    log(f'[calibrate] n={len(scores)} score mean={result["score_mean"]:.4f} max={result["score_max"]:.4f} '
        f'-> img_thr={img_thr:.4f} pix_thr={pix_thr:.4f}')
    return result


def deviation_confidence(score: float, img_thr: float) -> float:
    """与正常样本偏离的置信度, 单调映射到 [0,1): 等于阈值时为 0.5, 远超阈值趋近 1."""
    if img_thr <= 0:
        return 1.0 if score > 0 else 0.0
    return float(score / (score + img_thr))


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
