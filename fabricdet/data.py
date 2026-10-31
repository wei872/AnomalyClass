# -*- coding: utf-8 -*-
"""数据加载: YOLO 格式 + 纯正常图; 训练用"随机窗口裁剪"(等价于部署时的滑窗 tile),
验证用整图 letterbox。所有路径问题 (Windows 路径/缺失/中文路径/坏图) 都在这里兜底。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .model import IMAGENET_MEAN, IMAGENET_STD

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------- io
def imread_unicode(path: str | Path) -> np.ndarray | None:
    """中文/特殊字符路径安全读图。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def letterbox(img: np.ndarray, size: int, pad: int = 114):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    if nw != w or nh != h:
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[:nh, :nw] = img
    return canvas, s, (0, 0)


# ------------------------------------------------------------------- yaml
def resolve_dataset_paths(cfg: dict, data_root: str | None = None) -> dict:
    """yaml 里可能是别人机器的绝对路径 (如 D:/...): 不存在时用 data_root / 环境变量重建。"""
    env_root = data_root or os.environ.get("FABRIC19_ROOT") or os.environ.get("FABRICDET_DATA_ROOT")
    keys = {"train": "images/train", "val": "images/val",
            "normaltrain": "normal/images/train", "normalval": "normal/images/val"}
    out = {}
    for k, sub in keys.items():
        p = Path(str(cfg.get(k, "")).strip())
        if p.exists():
            out[k] = p
            continue
        cand = None
        if env_root:
            cand = Path(env_root) / sub
        if cand is None or not cand.exists():
            base = Path(str(cfg.get("path", "")).strip())
            cand = base / sub if str(base) not in ("", ".", "None") else None
        if cand is not None and cand.exists():
            out[k] = cand
        else:
            out[k] = None
    return out


def load_dataset_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    cfg["names"] = [str(n) for n in names]
    cfg["nc"] = int(cfg.get("nc", len(names)))
    if len(cfg["names"]) != cfg["nc"]:
        raise ValueError(f"dataset yaml: nc={cfg['nc']} 与 names 数量 {len(cfg['names'])} 不一致")
    return cfg


def collect_samples(img_dir: Path | None, lbl_dir: Path | None, normal: bool):
    if img_dir is None or not Path(img_dir).exists():
        return []
    items = []
    for p in sorted(Path(img_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXTS or p.name.startswith("."):
            continue
        lp = None
        if not normal and lbl_dir is not None:
            lp = Path(lbl_dir) / (p.stem + ".txt")
        items.append((str(p), str(lp) if lp else None))
    return items


def read_labels(lp: str | None, warn: set):
    boxes = []
    if lp is None:
        return boxes
    p = Path(lp)
    if not p.exists():
        if lp not in warn:
            warn.add(lp)
            print(f"[data] 警告: 标签缺失, 按正常图处理: {lp}")
        return boxes
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return boxes
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        x1, y1 = cx - w / 2, cy - h / 2
        boxes.append((c, x1, y1, x1 + w, y1 + h))   # 归一化 xyxy
    return boxes


# ---------------------------------------------------------------- dataset
class FabricDataset(Dataset):
    def __init__(self, split: str, paths: dict, names: list, size: int, train: bool,
                 cfg: dict | None = None):
        assert split in ("train", "val")
        self.train = train
        self.size = size
        self.cfg = cfg or {}
        self.names = names
        self.nc = len(names)
        self._warn: set = set()
        img_dir = paths[split]
        lbl_dir = img_dir.parent.parent / "labels" / split if img_dir else None
        self.samples = collect_samples(img_dir, lbl_dir, normal=False)
        nimg = paths.get("normal" + split)
        self.samples += collect_samples(nimg, None, normal=True)
        if not self.samples:
            raise FileNotFoundError(
                f"[data] split='{split}' 没有任何图片。\n"
                f"       期望目录: <root>/images/{split} 与 <root>/normal/images/{split}\n"
                f"       请用 --data-root 指定数据集根目录, 或修改 dataset yaml 中的路径。")
        random.Random(0).shuffle(self.samples)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------- augment
    def _augment(self, img, boxes):
        h, w = img.shape[:2]
        s = self.size
        # 训练窗口 = 部署 tile 的同一物理尺度 x 随机 jitter: 保证训练时缺陷的像素
        # 尺度分布与推理滑窗一致 (高分辨率图上小缺陷检出的关键)
        lo, hi = self.cfg.get("crop_jitter", (0.55, 1.3))
        side = int(min(max(int(s * random.uniform(lo, hi)), s // 4), min(h, w)))
        if min(h, w) <= s * 1.05 or side >= min(h, w):
            img, sc, _ = letterbox(img, s)
            boxes = [(c, x1 * sc, y1 * sc, x2 * sc, y2 * sc) for c, x1, y1, x2, y2 in boxes]
        else:
            if boxes and random.random() < 0.7:
                # 70% 概率以某个缺陷为中心裁窗, 避免小缺陷被随机窗漏掉
                _, ax1, ay1, ax2, ay2 = random.choice(boxes)
                bcx, bcy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                x0 = int(np.clip(bcx - side / 2 + random.uniform(-0.3, 0.3) * side, 0, w - side))
                y0 = int(np.clip(bcy - side / 2 + random.uniform(-0.3, 0.3) * side, 0, h - side))
            else:
                x0 = random.randint(0, w - side)
                y0 = random.randint(0, h - side)
            new_boxes = []
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1 = max(x1, x0), max(y1, y0)
                ix2, iy2 = min(x2, x0 + side), min(y2, y0 + side)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                orig = max((x2 - x1) * (y2 - y1), 1e-6)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if inter / orig < 0.5 or not (x0 <= cx < x0 + side and y0 <= cy < y0 + side):
                    continue
                sc = s / side
                new_boxes.append((c, (ix1 - x0) * sc, (iy1 - y0) * sc, (ix2 - x0) * sc, (iy2 - y0) * sc))
            boxes = new_boxes
            img = cv2.resize(img[y0:y0 + side, x0:x0 + side], (s, s), interpolation=cv2.INTER_LINEAR)
        # flips
        if random.random() < 0.5:
            img = img[:, ::-1]
            boxes = [(c, s - x2, y1, s - x1, y2) for c, x1, y1, x2, y2 in boxes]
        if random.random() < 0.5:
            img = img[::-1]
            boxes = [(c, x1, s - y2, x2, s - y1) for c, x1, y1, x2, y2 in boxes]
        # 小角度旋转 (纬斜类鲁棒性; 不做大角度以免经纬语义互换)
        if random.random() < 0.5:
            ang = random.uniform(-4, 4)
            m = cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0)
            img = cv2.warpAffine(img, m, (s, s), borderMode=cv2.BORDER_REPLICATE)
            nb = []
            rad = np.deg2rad(ang)
            ca, sa = np.cos(rad), np.sin(rad)
            for c, x1, y1, x2, y2 in boxes:
                pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
                px = (pts[:, 0] - s / 2) * ca + (pts[:, 1] - s / 2) * sa + s / 2
                py = -(pts[:, 0] - s / 2) * sa + (pts[:, 1] - s / 2) * ca + s / 2
                nb.append((c, px.min(), py.min(), px.max(), py.max()))
            boxes = nb
        # 光度/退化增强 (相机差异、光照、噪声、离焦)
        img = img.astype(np.float32)
        img *= random.uniform(0.75, 1.25)
        img = (img - 127.5) * random.uniform(0.75, 1.25) + 127.5
        if random.random() < 0.3:
            img = img + np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:
            # 光照渐变/暗角 (产线相机常见, 参考真实布匹图的边缘衰减)
            gx = np.linspace(-1, 1, img.shape[1], dtype=np.float32)[None, :]
            gy = np.linspace(-1, 1, img.shape[0], dtype=np.float32)[:, None]
            img *= (1 + random.uniform(-0.2, 0.2) * gx + random.uniform(-0.2, 0.2) * gy)
        if random.random() < 0.2:
            f = random.choice([2, 3])
            img = cv2.resize(img, (s // f, s // f), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        boxes = [(c, *np.clip([x1, y1, x2, y2], 0, s).tolist()) for c, x1, y1, x2, y2 in boxes]
        boxes = [(c, x1, y1, x2, y2) for c, x1, y1, x2, y2 in boxes
                 if (x2 - x1) > 3 and (y2 - y1) > 3]
        return np.clip(img, 0, 255).astype(np.uint8), boxes

    def _val_prep(self, img, boxes):
        h, w = img.shape[:2]
        img, sc, _ = letterbox(img, self.size)
        boxes = [(c, x1 * sc, y1 * sc, x2 * sc, y2 * sc) for c, x1, y1, x2, y2 in boxes]
        return img, boxes, sc

    def __getitem__(self, i):
        ip, lp = self.samples[i]
        img = imread_unicode(ip)
        if img is None:
            # 坏图: 返回纯背景, 训练不中断
            print(f"[data] 警告: 图像读取失败, 跳过内容: {ip}")
            img = np.full((self.size, self.size, 3), 114, np.uint8)
            boxes = []
            meta_scale = 1.0
        else:
            h, w = img.shape[:2]
            boxes = read_labels(lp, self._warn)
            bad = [c for c, *_ in boxes if not (0 <= c < self.nc)]
            if bad:
                raise ValueError(f"[data] {ip} 含非法类别 id {bad} (nc={self.nc})")
            boxes = [(c, x1 * w, y1 * h, x2 * w, y2 * h) for c, x1, y1, x2, y2 in boxes]
            if self.train:
                img, boxes = self._augment(img, boxes)
                meta_scale = 1.0
            else:
                img, boxes, meta_scale = self._val_prep(img, boxes)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()
        t = torch.tensor([[c, x1, y1, x2, y2] for c, x1, y1, x2, y2 in boxes],
                         dtype=torch.float32).reshape(-1, 5)
        return x, t, {"path": ip, "scale": float(meta_scale)}


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ts = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return xs, ts, metas


def class_frequency(ds: FabricDataset, nc: int) -> np.ndarray:
    cnt = np.zeros(nc, dtype=np.float64)
    for _, lp in ds.samples:
        for c, *_ in read_labels(lp, set()):
            if 0 <= c < nc:
                cnt[c] += 1
    return cnt


def build_class_weight(cnt: np.ndarray) -> torch.Tensor:
    tot = cnt.sum()
    if tot <= 0:
        return torch.ones(len(cnt))
    freq = np.maximum(cnt, 1.0) / tot
    w = (1.0 / freq) ** 0.5
    w = w / w.mean()
    w = np.clip(w, 0.5, 4.0)          # 防止极端权重 destabilize 训练
    return torch.tensor(w, dtype=torch.float32)
