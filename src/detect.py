"""异常检测器: 基于 Dinomaly2, 正常样本 + 带框缺陷样本联合训练 (同一 batch 混合, 不是分开训练).

- 训练: 正常图走重建损失 (loose cosine + 可选 local_cosine), 缺陷图走监督 margin 损失
        (图级 hinge + 框级 hinge), 一起学 bottleneck + decoder;
- 推理: anomaly map = 1 - cos(en, de) 高斯平滑; 图级异常分数 = map 最大值;
        与正常样本偏离的置信度 = score / (score + img_thr), img_thr 由 normalval/val 缺陷校准.
- 定位: map > mask_thr 的连通域 -> 缺陷框 (之后交给分类器, 即 先检测后分类).
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

from .dinomaly import build_dinomaly, StableAdamW
from .dinomaly.losses import (global_cosine, global_cosine_hm_percent, local_cosine,
                               token_dist_map, image_scores, per_image_mean_dist, margin_hinge, box_margin, masked_recon)
from .util import (AverageMeter, cal_anomaly_maps, make_smoother, list_images, mask_to_boxes,
                   rasterize_boxes, roc_auc, score_from_map, average_precision, f1_max, get_logger, torch_load)

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


def get_train_transform(image_size: int, crop_size: int, rot90: bool = True, flip: bool = True):
    """Dinomaly2 的 strong transform (翻转 + 旋转90; crop_size<image_size 时加 RandomResizedCrop).

    crop_size==image_size (默认) 时不裁剪: 整图(含四周暗角边缘)进网络.
    """
    tf = [T.ToPILImage(), T.Resize((image_size, image_size))]
    if crop_size < image_size:
        scale = (crop_size / image_size) ** 2
        tf.append(T.RandomResizedCrop((crop_size, crop_size), scale=(scale, scale), ratio=(0.95, 1.05)))
    tf += ([_RandomRotate90()] if rot90 else []) + ([T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if flip else []) \
        + [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(tf)


class TrainAug:
    """几何增强, 图像与缺陷框掩码 **同步** 变换 (随机 rot90 / 水平 / 垂直翻转; crop<image 时 RandomResizedCrop).

    旧版只对图像做 rot90/翻转而掩码不动, 约 3/4 的缺陷样本框级监督落在错误位置. 现在同一组随机参数同时作用于两者.
    模块级类, 可被 Windows spawn pickle.
    """

    def __init__(self, image_size: int, crop_size: int, rot90: bool = True, flip: bool = True):
        self.flip = flip
        self.image_size = image_size
        self.crop_size = crop_size
        self.rot90 = rot90
        self.resize = T.Resize((image_size, image_size))
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __call__(self, img_rgb: np.ndarray, mask: Optional[np.ndarray] = None):
        """img_rgb: HWC uint8; mask: HW uint8 (读图分辨率) 或 None. 返回 (tensor, mask crop×crop uint8 或 None)."""
        S, C = self.image_size, self.crop_size
        pil = self.resize(T.functional.to_pil_image(img_rgb))
        m = None if mask is None else cv2.resize(mask, (S, S), interpolation=cv2.INTER_NEAREST)
        if C < S:
            scale = (C / S) ** 2
            i, j, h, w = T.RandomResizedCrop.get_params(pil, scale=(scale, scale), ratio=(0.95, 1.05))
            pil = T.functional.resized_crop(pil, i, j, h, w, (C, C))
            if m is not None:
                m = cv2.resize(np.ascontiguousarray(m[i:i + h, j:j + w]), (C, C), interpolation=cv2.INTER_NEAREST)
        k = random.randint(0, 3) if self.rot90 else 0
        if k:
            pil = T.functional.rotate(pil, 90 * k)          # 逆时针, 与 np.rot90 同向
            if m is not None:
                m = np.rot90(m, k)
        if self.flip and random.random() < 0.5:
            pil = T.functional.hflip(pil)
            if m is not None:
                m = m[:, ::-1]
        if self.flip and random.random() < 0.5:
            pil = T.functional.vflip(pil)
            if m is not None:
                m = m[::-1, :]
        return self.norm(pil), (None if m is None else np.ascontiguousarray(m))


# ------------------------- 检测器 -------------------------
class AnomalyDetector:
    """封装 Dinomaly 模型的推理接口."""

    def __init__(self, model: nn.Module, device: str = 'cuda', image_size: int = 448, crop_size: int = 448,
                 smooth_kernel: int = 0, smooth_sigma: float = 4.0, half: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.offset = (self.image_size - self.crop_size) // 2
        self.tf = get_eval_transform(self.image_size, self.crop_size)
        self.half = half and device.startswith('cuda')
        # FP16 用 autocast 而不是 model.half(): 权重保持 fp32, LayerNorm/softmax 等敏感算子自动留在 fp32
        # (旧版只把输入转 half、模型仍是 fp32, CUDA 上会 dtype 不匹配直接报错)
        self.device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
        self.amp_dtype = torch.float16
        self.score_ratio = 0.0              # 图级分数: 0=最大值, >0=最高 ratio 比例像素均值 (由 calibrate 写入阈值文件)
        if self.device_type == 'cuda':
            torch.backends.cudnn.benchmark = True      # 输入尺寸固定, 让 cudnn 选最快的卷积算法
        self.smooth_kernel = int(smooth_kernel)
        self.smooth_sigma = float(smooth_sigma)
        self.gaussian = make_smoother(smooth_sigma, smooth_kernel, device=device)

    def set_smooth_sigma(self, sigma: float):
        """调整异常图高斯平滑: 越大越抑制正常图的单点噪声(其max分数下降), 成片/成线的缺陷响应保留.
        sigma<=4 为原版 5×5 核 (与旧阈值兼容); sigma>4 核宽随 sigma 增长. 改后须重新 calibrate."""
        self.smooth_sigma = float(sigma)
        self.gaussian = make_smoother(sigma, self.smooth_kernel, device=self.device)

    def set_score_ratio(self, ratio: float):
        self.score_ratio = float(ratio)

    @torch.no_grad()
    def raw_map(self, img_rgb: np.ndarray) -> torch.Tensor:
        """输入 RGB ndarray -> (1,1,C,C) 未平滑的异常图 (fp32 tensor, 在 device 上). 校准扫参复用它."""
        x = self.tf(img_rgb).unsqueeze(0).to(self.device)
        if self.half:
            with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
                en, de = self.model(x)
            en = [e.float() for e in en]
            de = [d.float() for d in de]
        else:
            en, de = self.model(x)
        amap, _ = cal_anomaly_maps(en, de, out_size=x.shape[-1])
        return amap.float()

    @torch.no_grad()
    def anomaly_map_crop(self, img_rgb: np.ndarray) -> np.ndarray:
        """输入 RGB ndarray (任意尺寸) -> crop_size×crop_size 的异常图 (float32, 已高斯平滑)."""
        amap = self.gaussian(self.raw_map(img_rgb))
        return amap[0, 0].cpu().numpy().astype(np.float32)

    def predict(self, img_rgb: np.ndarray, max_map_side: int = 1024) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """返回 (异常图 float32, 图级异常分数, 坐标缩放(sx,sy)).

        异常图长边封顶 max_map_side (防止超大原图把全分辨率图撑爆内存);
        返回的缩放 (sx,sy) = 原图宽高 / 异常图宽高, 用于把异常图上的框映射回原图坐标.
        """
        h0, w0 = img_rgb.shape[:2]
        amap_c = self.anomaly_map_crop(img_rgb)                     # C×C
        score = score_from_map(amap_c, self.score_ratio)
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
        return cls(model, device=device, image_size=hp['image_size'], crop_size=hp['crop_size'], half=half,
                   smooth_sigma=float(hp.get('smooth_sigma', 4.0)))


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


class CombinedTrainDataset(torch.utils.data.Dataset):
    """正常图(flag=0) + 带 YOLO 框的缺陷图(flag=1) 混合训练集, 供监督 margin 损失使用.

    返回 (img_tensor, flag, mask_token): mask_token 为 (side,side) float32, 缺陷框内=1, 正常图全 0.
    模块级别定义: Windows 上 DataLoader(num_workers>0) 用 spawn pickle, 局部类无法 pickle.
    """

    def __init__(self, normal_dir: str, defect_dir: str, defect_labels_dir: Optional[str],
                 aug: 'TrainAug', image_size: int, crop_size: int, logger=None):
        from .data.yolo import NormalImageDataset, YoloDefectDataset
        self.normal = NormalImageDataset(normal_dir, verify=True, logger=logger)
        self.defect = YoloDefectDataset(defect_dir, defect_labels_dir, max_side=2048, verify=True, logger=logger)
        self.aug = aug
        self.image_size = image_size
        self.crop_size = crop_size
        self.side = crop_size // 14

    def __len__(self):
        return len(self.normal) + len(self.defect)

    def _token_mask(self, m: np.ndarray) -> torch.Tensor:
        """crop×crop 像素掩码 -> side×side token 掩码: 只要框与 token 有重叠就记为缺陷 token (INTER_AREA 后 >0).
        旧版 INTER_NEAREST 每 14×14 只采 1 个像素, 细长疵点的框会被直接丢掉 (掩码全 0, 框级监督失效)."""
        t = cv2.resize(m.astype(np.float32), (self.side, self.side), interpolation=cv2.INTER_AREA)
        return torch.from_numpy((t > 0).astype(np.float32))

    def __getitem__(self, i):
        if i < len(self.normal):
            img, _ = self.normal[i]
            t, _ = self.aug(img)
            return t, 0, torch.zeros(self.side, self.side)
        img, _boxes, mask, _path = self.defect[i - len(self.normal)]
        t, m = self.aug(img, mask)                       # 图像与掩码同步增强
        return t, 1, self._token_mask(m)


class WarmupCosineScheduler:
    """线性 warmup + 余弦退火到 base_lr * final_ratio (语义与官方 Dinomaly2 一致:
    final_ratio=1.0 即 warmup 后学习率恒定 — 官方默认; 0.0 = 退火到 0 — 本项目此前的行为, 仍为默认)."""

    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 100, final_ratio: float = 0.0):
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
                lr = base * (self.final_ratio + (1.0 - self.final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
            g['lr'] = lr


def train_detector(normal_train_dir: str, out_dir: str, backbone: str = 'dinov2reg_vit_small_14',
                   image_size: int = 448, crop_size: int = 448, dropout: float = 0.4, la: int = 1, lc: int = 2,
                   cr: int = 1, ll: int = 1, ll_ratio: float = 0.9, ll_factor: float = 0.1,
                   total_iters: int = 40000, batch_size: int = 16, lr: float = 2e-3, warmup: int = 100,
                   eval_interval: int = 5000, eval_dir: Optional[str] = None, eval_labels_dir: Optional[str] = None,
                   device: str = 'cuda', num_workers: int = 4, seed: int = 1, weights_dir: str = None,
                   allow_random_init: bool = False, resume: Optional[str] = None, logger=None,
                   save_interval: int = 1000, no_eval: bool = False,
                   local_loss_w: float = 0.0, hard_pos_ratio: float = 0.0,
                   smooth_sigma: float = 4.0,
                   defect_dir: Optional[str] = None, defect_labels_dir: Optional[str] = None,
                   margin_w: float = 1.0, margin: float = 0.05, box_margin_w: float = 1.0,
                   hard_normal_w: float = 0.0, local_loss_warmup: int = 0,
                   rot90: bool = True, margin_score: str = 'smooth', defect_bg_w: float = 0.0,
                   eval_normal_dir: Optional[str] = None, eval_max_images: int = 0,
                   flip: bool = True, optimizer: str = 'stableadamw', lr_final_ratio: float = 0.0,
                   defect_frac: float = -1.0):
    """联合训练 Dinomaly2 检测器: 正常图 (重建损失) + 带框缺陷图 (监督 margin: 图级 hinge + 框内 hinge)
    同一 batch 混合, 直接推开正常/缺陷分数分布, 解决 calibrate 重叠. 返回保存路径.

    rot90: 训练增强是否含随机 90° 旋转. 布匹经纬方向固定、且疵点类别与方向相关 (经线/纬线), 旋转会把
           "经线方向的纹理" 也当成正常, 削弱对细长经/纬线缺陷的敏感度; 方向固定的产线建议关 (--no-rot90).
    margin_score: 'smooth' = 图级 hinge 用与推理/校准完全相同的分数 (上采样 + 高斯平滑后的最大值);
                  'token' = 旧做法 (token 网格上未平滑的最大值, 与推理分数有偏差).
    defect_bg_w: >0 时缺陷图框外 token (正常纹理) 也参与重建损失, 多榨一份正常数据 (默认关).
    flip: 训练增强是否含随机水平/垂直翻转. 官方 Dinomaly2 的 2D 配方完全不做增强 (rot90=flip=False).
    defect_frac: 每个 batch 中缺陷图的期望占比. 缺陷图太少时 (如 5%), bs=16 约 44% 的 batch 一张缺陷图都没有,
                 margin / box 损失那一步等于没开. -1=自动 (自然占比 <20% 时抬到 25%, 否则保持自然); 0=自然 (均匀打乱);
                 >0=指定占比 (按带放回加权采样实现).
    optimizer: 'stableadamw' (官方配方, 更新量 RMS 超阈值时缩步长 + 坏梯度整步跳过, 抑制 loss 尖峰) / 'adamw'.
    lr_final_ratio: 余弦退火终点 = lr * 该值; 0 = 退火到 0 (默认), 1 = 恒定学习率 (官方默认).
    eval_normal_dir: 训练中评估时的纯正常 val 图目录. 没有它 val 只有缺陷图, I-AUROC 恒为 NaN, model_best 选不出来.
    """
    from .data.yolo import YoloDefectDataset
    log = logger or get_logger('train_detector')
    torch.manual_seed(seed)
    np.random.seed(seed)
    # image_size/crop_size 必须是 14 的倍数 (ViT patch=14), 且 crop_size<=image_size.
    # image_size 是 Resize 目标; crop_size 是中心裁剪后的网络输入.
    # 本项目默认 crop_size=image_size=448: 不裁边缘 (训练图四周有暗角, 属于正常外观, 边缘疵点也不能丢)
    if image_size % 14 or crop_size % 14 or crop_size > image_size:
        log(f'警告: image_size={image_size}, crop_size={crop_size} 应为14的倍数且 crop_size<=image_size, 否则位置编码会异常')
    random.seed(seed)

    tf = get_train_transform(image_size, crop_size, rot90=rot90, flip=flip)
    aug = TrainAug(image_size, crop_size, rot90=rot90, flip=flip)
    use_defect = bool(defect_dir and os.path.isdir(defect_dir)) and (margin_w > 0 or box_margin_w > 0)
    if use_defect:
        train_ds = CombinedTrainDataset(normal_train_dir, defect_dir, defect_labels_dir,
                                        aug, image_size, crop_size, logger=log)
        log(f'训练集: 正常 {len(train_ds.normal)} + 缺陷(带框) {len(train_ds.defect)} 混合, '
            f'margin_w={margin_w} margin={margin} box_margin_w={box_margin_w} hard_normal_w={hard_normal_w} '
            f'rot90={rot90} flip={flip} margin_score={margin_score} defect_bg_w={defect_bg_w} '
            f'optimizer={optimizer} lr_final_ratio={lr_final_ratio}')
    else:
        train_ds = NormalTransformDataset(normal_train_dir, tf)
        if defect_dir:
            log('注意: margin_w=box_margin_w=0, 缺陷监督关闭 (仅消融对照, 正式训练保持联合)')
    sampler = None
    if use_defect:
        n_n, n_d = len(train_ds.normal), len(train_ds.defect)
        nat = n_d / max(n_n + n_d, 1)
        frac = defect_frac
        if frac < 0:
            frac = 0.25 if nat < 0.2 else 0.0
        p_none = (1 - (frac if frac > 0 else nat)) ** batch_size
        log(f'缺陷图自然占比 {nat:.1%}; 目标占比 {"自然" if frac <= 0 else f"{frac:.0%}"}; '
            f'每个 batch 完全没有缺陷图的概率 ≈ {p_none:.1%}')
        if frac > 0:
            w = [(1 - frac) / max(n_n, 1)] * n_n + [frac / max(n_d, 1)] * n_d
            sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    log(f'训练样本数: {len(train_ds)}, batch={batch_size}, iters={total_iters}')

    model = build_dinomaly(backbone, dropout=dropout, la=la, lc=lc, cr=cr, weights_dir=weights_dir,
                           allow_random_init=allow_random_init, logger=log).to(device)
    if resume:
        model.load_state_dict(torch_load(resume, map_location='cpu')['model'])
        log(f'从 {resume} 恢复')
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad_(False)   # encoder 冻结 (前向本就在 no_grad 下), 节省显存
    trainable = nn.ModuleList([model.bottleneck, model.decoder])
    groups = [
        {'params': model.bottleneck[0].parameters(), 'lr': lr * 0.1},     # bottleneck 第一层小学习率 (官方: 2e-4)
        {'params': model.bottleneck[1].parameters()},
        {'params': model.decoder.parameters()},
    ]
    if optimizer == 'stableadamw':
        optimizer = StableAdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-10)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-10)
    else:
        raise ValueError(f'未知 optimizer: {optimizer} (stableadamw / adamw)')
    scheduler = WarmupCosineScheduler(optimizer, total_epochs=total_iters, warmup_epochs=warmup,
                                      final_ratio=lr_final_ratio)

    os.makedirs(out_dir, exist_ok=True)
    hparams = {'backbone': backbone, 'image_size': image_size, 'crop_size': crop_size, 'dropout': dropout,
               'la': la, 'lc': lc, 'cr': cr,
               'local_loss_w': float(local_loss_w), 'hard_pos_ratio': float(hard_pos_ratio),
               'smooth_sigma': float(smooth_sigma),
               'margin_w': float(margin_w), 'margin': float(margin), 'box_margin_w': float(box_margin_w),
               'hard_normal_w': float(hard_normal_w), 'local_loss_warmup': int(local_loss_warmup),
               'defect_sup': bool(use_defect), 'rot90': bool(rot90), 'margin_score': margin_score,
               'defect_bg_w': float(defect_bg_w), 'flip': bool(flip),
               'optimizer': type(optimizer).__name__ if not isinstance(optimizer, str) else optimizer,
               'lr_final_ratio': float(lr_final_ratio), 'defect_frac': float(defect_frac)}
    smoother = make_smoother(smooth_sigma, device=device)     # 训练端 margin 分数用的平滑 (= 推理端同一个核)
    best_path = os.path.join(out_dir, 'model_best.pth')
    last_path = os.path.join(out_dir, 'model_last.pth')
    best_key = None                       # (I-AUROC, 相对间隙): 字典序比较, AUROC 饱和为 1.0 后看间隙大小

    it = 0
    loss_meter = AverageMeter()
    while it < total_iters:
        for batch in train_loader:
            if use_defect:
                img, flag, tmask = batch
                img = img.to(device)
                flag = flag.to(device)
                tmask = tmask.to(device)
            else:
                img = batch.to(device)
                flag = None
            en, de = model(img)
            nmask = (flag == 0) if flag is not None else torch.ones(img.shape[0], dtype=torch.bool, device=device)
            dmask = ~nmask
            loss = img.new_zeros(())
            if nmask.any():
                en_n = [e[nmask] for e in en]
                de_n = [d[nmask] for d in de]
                p = min(ll_ratio * it / 1000, ll_ratio) if ll else 0
                loss = loss + (global_cosine_hm_percent(en_n, de_n, p=p, factor=ll_factor) if ll
                               else global_cosine(en_n, de_n))
                lw = local_loss_w
                if lw > 0 and local_loss_warmup > 0:
                    lw = lw * min(1.0, it / local_loss_warmup)   # 局部损失线性预热, 避免初期不稳
                if lw > 0:
                    # 局部逐位置重建损失: 细长经线等局部缺陷在全局 cosine 里占比太小, 需要局部项才能学到
                    loss = loss + lw * local_cosine(en_n, de_n, hard_ratio=hard_pos_ratio)
                if hard_normal_w > 0:
                    # 困难正常重加权: 对当前分数偏高的正常图加大重建损失, 压住 max_normal
                    d_n = per_image_mean_dist(en_n, de_n)
                    w = 1.0 + hard_normal_w * F.relu(d_n.detach() / d_n.detach().mean().clamp(min=1e-6) - 1.0)
                    loss = loss + hard_normal_w * (w * d_n).mean()
            if dmask.any() and (margin_w > 0 or box_margin_w > 0):
                # 监督 margin: 缺陷图必须比正常图分数高 margin (图级 all-pairs hinge)
                if margin_score == 'smooth':
                    # 与推理/校准同一个分数: 上采样到输入尺寸 -> 高斯平滑 -> 取最大
                    amap, _ = cal_anomaly_maps(en, de, out_size=img.shape[-1])
                    s = smoother(amap).flatten(1).max(dim=1).values
                else:
                    s = image_scores(en, de)
                if margin_w > 0 and nmask.any():
                    loss = loss + margin_w * margin_hinge(s[dmask], s[nmask], margin)
                if box_margin_w > 0:
                    # 框级 hinge: 缺陷框内 token 平均距离须 >= margin, 直接抬高 min_defect
                    loss = loss + box_margin_w * box_margin(token_dist_map(en, de)[dmask], tmask[dmask], margin)
                if defect_bg_w > 0:
                    # 缺陷图框外 = 正常纹理, 也要重建好
                    loss = loss + defect_bg_w * masked_recon(token_dist_map([e[dmask] for e in en],
                                                                           [d[dmask] for d in de]), tmask[dmask])
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
                model.train()
            if eval_dir and not no_eval and (it % eval_interval == 0 or it == total_iters):
                try:
                    st = evaluate_detector(model, eval_dir, eval_labels_dir, device, image_size, crop_size,
                                           logger=log, normal_dir=eval_normal_dir, smooth_sigma=smooth_sigma,
                                           max_images=eval_max_images, return_stats=True)
                    auroc = st['i_auroc']
                    if auroc != auroc:
                        # 评估集里只有一类, AUROC 无法比较: 退而保存最新权重, 而不是永远停在第一次评估
                        if best_key is None or best_key[0] != 'nan':
                            log('警告: 评估集缺少正常图 (或缺陷图), 无法算 I-AUROC; model_best 将跟随最新评估. '
                                '请配置 normalval (纯正常 val 图)')
                        best_key = ('nan', it)
                        better = True
                    else:
                        key = (auroc, st['gap_rel'] if st['gap_rel'] == st['gap_rel'] else -1e9)
                        better = best_key is None or best_key[0] == 'nan' or key > best_key
                        if better:
                            best_key = key
                    if better:
                        AnomalyDetector(model, device, image_size, crop_size).save(best_path, hparams)
                        log(f'*** 新最优 I-AUROC={auroc:.4f} 间隙={st["gap_rel"]:.3f}, 保存 {best_path}')
                except Exception as e:
                    log(f'警告: 评估失败 ({e}), 已跳过, 不影响 checkpoint 保存 (last 已保存)')
                model.train()
            if it >= total_iters:
                break
    # 兜底: 保证至少有一个 checkpoint
    AnomalyDetector(model, device, image_size, crop_size).save(last_path, hparams)
    if not os.path.exists(best_path):
        AnomalyDetector(model, device, image_size, crop_size).save(best_path, hparams)
    log(f'训练完成. best={best_path} (key={best_key}), last={last_path}')
    return best_path


# ------------------------- 评估 / 校准 -------------------------
@torch.no_grad()
def evaluate_detector(model: nn.Module, images_dir: str, labels_dir: Optional[str], device: str = 'cuda',
                      image_size: int = 448, crop_size: int = 448, batch_size: int = 8, logger=None,
                      eval_size: int = 256, read_max_side: int = 1024, normal_dir: Optional[str] = None,
                      smooth_sigma: float = 4.0, max_images: int = 0, return_stats: bool = False,
                      score_ratio: float = 0.0):
    """在带标签的 val 集上评估: 默认返回图级 AUROC; return_stats=True 返回 dict (含 i_auroc / gap_rel 等).

    - images_dir: 带 YOLO 标注的缺陷 val 图 (无标注的图按正常处理);
    - normal_dir: 纯正常 val 图 (normalval). 工程里 images/val 通常全是缺陷图, 不给 normal_dir 时图级
      AUROC 恒为 NaN (只有一类), 训练就没法选 model_best;
    - gap_rel = (min_defect - max_normal) / max_normal, 即校准时能拿到的相对间隙 (>0 说明可分);
    - smooth_sigma 与训练/校准/推理保持一致; 内存有界: 读图长边封顶 read_max_side, 像素级指标在
      eval_size×eval_size 上计算 (YOLO 框是归一化坐标, 缩放无关).
    """
    from .data.yolo import NormalImageDataset, YoloDefectDataset
    log = logger or get_logger('eval')
    det = AnomalyDetector(model, device, image_size, crop_size, smooth_sigma=smooth_sigma)
    ds = YoloDefectDataset(images_dir, labels_dir, max_side=read_max_side)
    # 图级分数 = score_ratio 比例的最高像素均值 (0=最大值), 须与校准/推理一致
    n_def = len(ds) if not max_images else min(max_images, len(ds))
    items = [('d', i) for i in range(n_def)]
    nds = None
    if normal_dir and os.path.isdir(normal_dir):
        nds = NormalImageDataset(normal_dir, verify=True, logger=log)
        n_nor = len(nds) if not max_images else min(max_images, len(nds))
        items += [('n', i) for i in range(n_nor)]
    tf = get_eval_transform(image_size, crop_size)
    gt_sp, pr_sp, gt_px, pr_px = [], [], [], []
    for kind, i in items:
        if kind == 'd':
            img, boxes, _, path = ds[i]
        else:
            img, boxes = nds[i][0], []
        h, w = img.shape[:2]
        x = tf(img).unsqueeze(0).to(device)
        en, de = model(x)
        amap, _ = cal_anomaly_maps(en, de, x.shape[-1])
        amap = det.gaussian(amap.float())                       # (1,1,C,C)
        amap = F.interpolate(amap, size=(eval_size, eval_size), mode='bilinear', align_corners=False)
        amap = amap[0, 0].cpu().numpy()
        score = score_from_map(amap, score_ratio)
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
    max_n = float(pr_sp[gt_sp == 0].max()) if (gt_sp == 0).any() else float('nan')
    min_d = float(pr_sp[gt_sp == 1].min()) if (gt_sp == 1).any() else float('nan')
    gap_rel = (min_d - max_n) / max(max_n, 1e-8) if (max_n == max_n and min_d == min_d) else float('nan')
    log(f'[eval] I-AUROC={i_auroc:.4f} I-AP={i_ap:.4f} I-F1={i_f1:.4f} | '
        f'P-AUROC={p_auroc:.4f} P-AP={p_ap:.4f} P-F1={p_f1:.4f} | '
        f'max_normal={max_n:.4f} min_defect={min_d:.4f} gap_rel={gap_rel:.3f} (n={len(gt_sp)})')
    if return_stats:
        return {'i_auroc': i_auroc, 'p_auroc': p_auroc, 'i_ap': i_ap, 'p_ap': p_ap, 'i_f1': i_f1, 'p_f1': p_f1,
                'max_normal': max_n, 'min_defect': min_d, 'gap_rel': gap_rel, 'n': int(len(gt_sp))}
    return i_auroc if i_auroc == i_auroc else 0.0   # NaN -> 0.0


@torch.no_grad()
def sweep_postprocess(detector: AnomalyDetector, normal_dir: str, defect_dir: str,
                      defect_labels_dir: Optional[str] = None, sigmas=(4.0, 6.0, 8.0, 12.0),
                      ratios=(0.0, 0.001, 0.005, 0.01), max_images: int = 0, logger=None) -> Dict:
    """不重训, 在 val 上扫 推理后处理参数 (平滑 sigma × 图级分数取法), 找正常/缺陷分数间隙最大的组合.

    每张图只前向一次, 再对同一张原始异常图套各组 (sigma, ratio). 指标 gap_rel = (min_defect - max_normal) / max_normal:
    >0 表示正常/缺陷可分 (校准能取到间隙中点), 越大越好; 同时报告图级 AUROC.
    ratio=0 为最大值; ratio>0 为最高 ratio 比例像素的均值 (官方 Dinomaly2 评估用 0.01).
    注意: 组合是在 val 上挑的, 样本少时 min/max 本身有噪声, 结果请当作参考.
    """
    log = logger or get_logger('sweep')
    from .data.yolo import NormalImageDataset, YoloDefectDataset
    smoothers = {float(s): make_smoother(s, device=detector.device) for s in sigmas}
    combos = [(float(s), float(r)) for s in smoothers for r in ratios]
    kmax = 0
    side = detector.crop_size
    ks = {float(r): max(1, int(side * side * r)) for r in ratios if r > 0}
    if ks:
        kmax = max(ks.values())

    def scores_of(img):
        raw = detector.raw_map(img)
        out = {}
        for s, sm in smoothers.items():
            flat = sm(raw).flatten()
            csum = torch.cumsum(torch.topk(flat, kmax).values, 0) if kmax else None
            for r in ratios:
                out[(s, float(r))] = float(flat.max()) if r <= 0 else float(csum[ks[float(r)] - 1] / ks[float(r)])
        return out

    ds_n = NormalImageDataset(normal_dir)
    ds_d = YoloDefectDataset(defect_dir, defect_labels_dir, max_side=2048)
    sc_n = {c: [] for c in combos}
    sc_d = {c: [] for c in combos}
    n_n = len(ds_n) if not max_images else min(max_images, len(ds_n))
    for i in range(n_n):
        for c, v in scores_of(ds_n[i][0]).items():
            sc_n[c].append(v)
    n_d = len(ds_d) if not max_images else min(max_images, len(ds_d))
    for i in range(n_d):
        img, boxes, _, _ = ds_d[i]
        if not boxes:
            continue
        for c, v in scores_of(img).items():
            sc_d[c].append(v)
    if not sc_d[combos[0]] or not sc_n[combos[0]]:
        raise ValueError('扫参需要同时有正常图和带标注的缺陷图')
    table = []
    for c in combos:
        n, d = np.array(sc_n[c]), np.array(sc_d[c])
        gap = (d.min() - n.max()) / max(n.max(), 1e-8)
        auc = roc_auc(np.r_[np.zeros(len(n)), np.ones(len(d))], np.r_[n, d])
        table.append({'smooth_sigma': c[0], 'score_ratio': c[1], 'max_normal': float(n.max()),
                      'min_defect': float(d.min()), 'gap_rel': float(gap), 'i_auroc': float(auc)})
    table.sort(key=lambda t: (-t['gap_rel'], -t['i_auroc']))
    log(f'[sweep] 正常 {len(sc_n[combos[0]])} 张 / 缺陷 {len(sc_d[combos[0]])} 张, 按相对间隙 gap_rel 排序:')
    log('    sigma  ratio   max_normal  min_defect  gap_rel  I-AUROC')
    for t in table:
        log(f'    {t["smooth_sigma"]:5.1f}  {t["score_ratio"]:.3f}   {t["max_normal"]:.4f}      {t["min_defect"]:.4f}     '
            f'{t["gap_rel"]:+.3f}   {t["i_auroc"]:.4f}')
    best = table[0]
    log(f'[sweep] 推荐 smooth_sigma={best["smooth_sigma"]} score_ratio={best["score_ratio"]} '
        f'(gap_rel={best["gap_rel"]:+.3f}{" 可分" if best["gap_rel"] > 0 else " 仍重叠, 需重训"})')
    return {'table': table, 'best': best}


@torch.no_grad()
def calibrate(detector: AnomalyDetector, normal_dir: str, defect_dir: Optional[str] = None,
              defect_labels_dir: Optional[str] = None, pix_quantile: float = 0.9999,
              max_images: int = 0, class_names: Optional[List[str]] = None, logger=None) -> Dict:
    """校准阈值. 目标: 所有正常图判正常、所有带标注缺陷图判异常 (不靠调参跷跷板).

    - normal_dir (纯正常图) 跑一遍 -> 正常分数分布, 记 max_normal;
    - 若提供 defect_dir/defect_labels_dir (YOLO 标注的缺陷图), 跑一遍 -> 缺陷图最低分 min_defect;
    - 若 min_defect > max_normal (两类分数可分): img_thr = 间隙中点 (max_normal+min_defect)/2,
      两侧留最大 margin, 正常/异常各 100%;
    - 若重叠 (min_defect <= max_normal): 不存在能同时 100% 的阈值, 告警并取 max_normal
      (优先保证正常全对), 同时报告重叠程度, 提示需要重训检测器让分布分开;
    - pix_thr = 正常图异常像素的 pix_quantile 分位数 (用于区域分割).
    - 异常置信度(与正常偏离程度) = score / (score + img_thr).
    """
    log = logger or get_logger('calibrate')
    from .data.yolo import NormalImageDataset, YoloDefectDataset

    # pix_thr 用直方图累积求分位数: 旧版把所有正常图的全分辨率异常图拼成一个大数组 (千张图 ≈ 数 GB, 会 OOM)
    HB, HR = 200000, 2.0                 # 余弦距离 ∈ [0,2], 分辨率 1e-5

    def score_dir(ds, cap: int = 0):
        scores, hist = [], np.zeros(HB, dtype=np.int64)
        n = len(ds) if not cap else min(cap, len(ds))
        for i in range(n):
            item = ds[i]
            img = item[0]
            full, score, _ = detector.predict(img)
            scores.append(score)
            hist += np.histogram(np.clip(full, 0, HR - 1e-6), bins=HB, range=(0, HR))[0]
            if (i + 1) % 10 == 0:
                log(f'  校准进度 {i + 1}/{n}')
        return np.array(scores), hist

    def hist_quantile(hist, q):
        cdf = np.cumsum(hist) / max(int(hist.sum()), 1)
        return float((int(np.searchsorted(cdf, q)) + 1) * HR / HB)

    ds_n = NormalImageDataset(normal_dir)
    scores, pix_hist = score_dir(ds_n, cap=max_images)
    max_normal = float(scores.max())
    result = {
        'normal_dir': os.path.abspath(normal_dir),
        'n_normal': int(len(scores)),
        'image_size': int(detector.image_size),
        'crop_size': int(detector.crop_size),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'max_normal': max_normal,
        'pix_thr': hist_quantile(pix_hist, pix_quantile),
        'smooth_sigma': float(detector.smooth_sigma),
        'score_ratio': float(detector.score_ratio),
        'pix_quantile': pix_quantile,
    }
    min_defect = None
    if defect_dir and os.path.isdir(defect_dir):
        ds_d = YoloDefectDataset(defect_dir, defect_labels_dir, max_side=2048)
        d_scores = []
        per_class = {}                       # cls -> [scores of images containing this class]
        n = len(ds_d) if not max_images else min(max_images, len(ds_d))
        for i in range(n):
            img, boxes, _, _ = ds_d[i]
            if not boxes:
                continue                      # 无标注框的 val 图按正常处理, 不参与缺陷侧
            _, score, _ = detector.predict(img)
            d_scores.append(score)
            for c, *_ in boxes:
                per_class.setdefault(int(c), []).append(score)
            if (i + 1) % 10 == 0:
                log(f'  缺陷校准进度 {i + 1}/{n}')
        if d_scores:
            min_defect = float(min(d_scores))
            result['n_defect'] = int(len(d_scores))
            result['min_defect'] = min_defect
            result['defect_p10'] = float(np.quantile(d_scores, 0.1))
            # 按类别诊断: 哪些缺陷类的图分数最低 (重叠主因)
            cls_stat = sorted(((float(np.min(v)), float(np.mean(v)), c, len(v))
                               for c, v in per_class.items()))
            result['per_class_min_score'] = {str(c): round(m, 4) for m, _, c, _ in cls_stat}
            names = class_names or getattr(detector, 'class_names', None)
            log('[calibrate] 各缺陷类最低图分数 (最差5类):')
            for m, mean, c, cnt in cls_stat[:5]:
                nm = names[c] if names and c < len(names) else c
                log(f'    {nm}: min={m:.4f} mean={mean:.4f} (n={cnt})  [max_normal={max_normal:.4f}]')
    if min_defect is not None and min_defect > max_normal:
        img_thr = (max_normal + min_defect) / 2.0
        result['separable'] = True
        result['margin'] = min_defect - max_normal
        log(f'[calibrate] 分布可分: max_normal={max_normal:.4f} < min_defect={min_defect:.4f} '
            f'(间隙={min_defect - max_normal:.4f}) -> img_thr 取间隙中点 {img_thr:.4f}, 正常/缺陷各 100%')
    else:
        img_thr = max_normal
        result['separable'] = False
        if min_defect is not None:
            log(f'[calibrate] ⚠️ 正常/缺陷分数重叠 (max_normal={max_normal:.4f} >= min_defect={min_defect:.4f}), '
                f'不存在能同时 100% 的阈值! 取 img_thr=max_normal={img_thr:.4f} (正常全对, 部分缺陷会漏). '
                f'请重训检测器 (更多 iters / --local-loss-w 1.0) 让分布分开')
        else:
            log(f'[calibrate] 未提供缺陷图, img_thr=max_normal={img_thr:.4f}')
    result['img_thr'] = float(img_thr)
    log(f'[calibrate] n_normal={result["n_normal"]} score mean={result["score_mean"]:.4f} max={max_normal:.4f} '
        f'-> img_thr={img_thr:.4f} pix_thr={result["pix_thr"]:.4f}')
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
