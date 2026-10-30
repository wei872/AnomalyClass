"""数据集读取 (YOLO 格式标注 + 纯正常图).

目录约定 (与 ``dataset/fabric19.yaml`` 一致)::

    DATASET_ROOT/
      images/train/  images/val/           疵点图
      labels/train/  labels/val/           YOLO 标注 (cls cx cy w h, 归一化)
      normal/images/train/                纯正常布面图 (无标注)
      normal/images/val/

同时兼容:

* 多边形标注 (cls x1 y1 x2 y2 ... , 归一化), 会生成更精确的伪疵点掩码;
* 空标注文件 = 正常图;
* 可选的 ``mask_dir`` (布面/背景掩码), 目录结构与 images 对应, 文件名相同;
* Windows 绝对路径 (原 yaml 里的 ``D:/tyhn/...``) 在 Linux 下可通过
  ``data.root`` 重新映射到本机数据集根目录.

标注框一律转换为 **归一化 xyxy**, 与输入分辨率解耦.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from ..utils import IMAGENET_MEAN, IMAGENET_STD, load_yaml
from .augment import EvalTransform, Sample, TrainAugment
from .fabric_mask import estimate_cloth_mask

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# 路径解析
# --------------------------------------------------------------------------- #
def _is_abs_like(p: str) -> bool:
    return p.startswith("/") or bool(re.match(r"^[A-Za-z]:", p)) or p.startswith("\\\\")


def _suffix_after(path: str, markers: Sequence[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    for marker in markers:
        if marker in parts:
            idx = parts.index(marker)
            return "/".join(parts[idx:])
    return parts[-1]


def resolve_path(raw: Optional[str], root: Optional[str], markers: Sequence[str]) ->Optional[Path]:
    if raw is None:
        return None
    raw = str(raw)
    if _is_abs_like(raw):
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        if root:
            mapped = Path(root) / _suffix_after(raw, markers)
            return mapped
        return candidate
    if root:
        return Path(root) / raw
    return Path(raw)


@dataclass
class DatasetSpec:
    root: Optional[str]
    splits: Dict[str, Optional[Path]]
    labels: Dict[str, Optional[Path]]
    normal_splits: Dict[str, Optional[Path]]
    names: List[str]
    nc: int
    mask_dir: Optional[Path] = None

    def summary(self) -> str:
        lines = [f"root={self.root}", f"nc={self.nc}"]
        for split, path in self.splits.items():
            lines.append(f"  {split}: {path}")
        for split, path in self.normal_splits.items():
            lines.append(f"  normal/{split}: {path}")
        return "\n".join(lines)


def load_dataset_spec(cfg: Dict[str, Any], split_override: Optional[Dict[str, str]] = None) -> DatasetSpec:
    """读取 dataset yaml (如 dataset/fabric19.yaml) 并解析为实际路径."""
    data_cfg = cfg.get("data", {})
    yaml_path = data_cfg.get("dataset_yaml")
    if not yaml_path:
        raise ValueError("data.dataset_yaml 未配置")
    raw = load_yaml(yaml_path)
    root = data_cfg.get("root") or raw.get("path")
    root = str(root) if root else None
    if root and not root.startswith("/") and not _is_abs_like(root):
        root = str((Path(yaml_path).parent / root).resolve())

    markers = ("images", "labels", "normal")
    splits = {
        "train": resolve_path(raw.get("train"), root, markers),
        "val": resolve_path(raw.get("val"), root, markers),
    }
    normal_splits = {
        "train": resolve_path(raw.get("normaltrain") or raw.get("normal_train"), root, markers),
        "val": resolve_path(raw.get("normalval") or raw.get("normal_val"), root, markers),
    }
    if split_override:
        for key, value in split_override.items():
            splits[key] = Path(value)

    labels = {}
    for key, img_dir in splits.items():
        if img_dir is None:
            labels[key] = None
            continue
        label_dir = Path(str(img_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if not label_dir.exists():
            alt = Path(str(img_dir).replace("/images/", "/labels/"))
            label_dir = alt if alt.exists() else label_dir
        labels[key] = label_dir

    mask_dir = data_cfg.get("mask_dir")
    if mask_dir:
        mask_dir = resolve_path(mask_dir, root, markers)
    names = list(raw.get("names", []))
    nc = int(raw.get("nc", len(names)))
    return DatasetSpec(root=root, splits=splits, labels=labels, normal_splits=normal_splits, names=names, nc=nc, mask_dir=mask_dir)


# --------------------------------------------------------------------------- #
# 标注 / 图像读写
# --------------------------------------------------------------------------- #
def read_yolo_labels(path: Optional[Path]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """返回 (boxes (N,5) 归一化 xyxy, polygons 列表)."""
    boxes: List[List[float]] = []
    polygons: List[np.ndarray] = []
    if path is None or not Path(path).exists():
        return np.zeros((0, 5), dtype=np.float32), polygons
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            if coords.size == 4:
                cx, cy, w, h = coords
                if w <= 0 or h <= 0:
                    continue
                boxes.append([cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            elif coords.size >= 6 and coords.size % 2 == 0:
                pts = coords.reshape(-1, 2)
                boxes.append([cls, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
                polygons.append(np.concatenate([[cls], coords.reshape(-1)]))
    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if boxes_arr.size:
        boxes_arr[:, 1:5] = np.clip(boxes_arr[:, 1:5], 0.0, 1.0)
    return boxes_arr, polygons


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def boxes_to_mask(boxes: np.ndarray, polygons: Sequence[np.ndarray], h: int, w: int) -> np.ndarray:
    """由标注生成弱疵点掩码 (多边形优先, 否则用框)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not len(boxes) and not polygons:
        return mask
    pil = Image.fromarray(mask)
    drawer = ImageDraw.Draw(pil)
    for poly in polygons:
        coords = poly[1:].reshape(-1, 2)
        pts = [(float(x) * w, float(y) * h) for x, y in coords]
        if len(pts) >= 3:
            drawer.polygon(pts, fill=1)
    covered = [p[0] for p in polygons]
    for box in boxes:
        if int(box[0]) in covered:
            continue
        x1, y1, x2, y2 = box[1:5]
        drawer.rectangle([float(x1) * w, float(y1) * h, max(float(x2) * w, float(x1) * w + 1), max(float(y2) * h, float(y1) * h + 1)], fill=1)
    return np.asarray(pil, dtype=np.uint8)


def list_images(directory: Optional[Path]) -> List[Path]:
    if directory is None or not Path(directory).exists():
        return []
    out = [p for p in sorted(Path(directory).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    image: Path
    label: Optional[Path]
    normal: bool
    mask: Optional[Path] = None


class FabricDataset(Dataset):
    def __init__(
        self,
        spec: DatasetSpec,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        train: bool = True,
        augment_cfg: Optional[Dict[str, Any]] = None,
        normal_ratio: float = 0.4,
        limit: Optional[int] = None,
        cache_masks: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.split = split
        self.image_size = tuple(image_size)
        self.train = train
        self.normal_ratio = normal_ratio
        self.cache_masks = cache_masks

        aug_cfg = dict(augment_cfg or {})
        aug_cfg.setdefault("image_size", self.image_size)
        self.transform = TrainAugment(aug_cfg) if train else EvalTransform(aug_cfg)

        cfg_data = (augment_cfg or {})
        self.mask_work_size = cfg_data.get("mask_work_size", "auto")
        self.defect_entries = self._collect(spec.splits.get(split), spec.labels.get(split), spec.mask_dir, normal=False, split=split)
        self.normal_entries = self._collect(spec.normal_splits.get(split), None, spec.mask_dir, normal=True, split=split)
        # 空标注的图也算正常图
        empty = [e for e in self.defect_entries if e.label is None]
        self.defect_entries = [e for e in self.defect_entries if e.label is not None]
        self.normal_entries.extend(empty)

        self.entries = self._balance(self.defect_entries, self.normal_entries, normal_ratio if train else 0.5)
        if limit and int(limit) < len(self.entries):
            self.entries = self._stratified_limit(self.entries, int(limit))
        if not self.entries:
            raise FileNotFoundError(
                f"split={split} 未找到任何图像.\n{spec.summary()}\n"
                "请检查 dataset/fabric19.yaml 的路径, 或用 data.root 指向本机数据集根目录."
            )
        self._cloth_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect(
        image_dir: Optional[Path],
        label_dir: Optional[Path],
        mask_dir: Optional[Path],
        normal: bool,
        split: Optional[str] = None,
    ) -> List[Entry]:
        entries: List[Entry] = []
        for image in list_images(image_dir):
            label = None
            if label_dir is not None and not normal:
                candidate = Path(label_dir) / f"{image.stem}.txt"
                if not candidate.exists():
                    # 标签目录可能与图片有不同层级
                    matches = list(Path(label_dir).rglob(f"{image.stem}.txt"))
                    candidate = matches[0] if matches else candidate
                if candidate.exists():
                    label = candidate
            mask = None
            if mask_dir is not None:
                candidates = []
                for ext in (".png", ".bmp", ".jpg"):
                    candidates.append(Path(mask_dir) / f"{image.stem}{ext}")
                    if split:
                        candidates.append(Path(mask_dir) / split / f"{image.stem}{ext}")
                for candidate in candidates:
                    if candidate.exists():
                        mask = candidate
                        break
                if mask is None:  # 兜底: 目录层级与图片不一致时递归查找
                    hits = list(Path(mask_dir).rglob(f"{image.stem}.png"))
                    if hits:
                        mask = hits[0]
            entries.append(Entry(image=image, label=label, normal=normal, mask=mask))
        return entries

    @staticmethod
    def _balance(defect: List[Entry], normal: List[Entry], normal_ratio: float) -> List[Entry]:
        if not normal:
            return list(defect)
        if not defect:
            return list(normal)
        normal_ratio = float(np.clip(normal_ratio, 0.0, 1.0))
        n_normal = int(round(len(defect) * normal_ratio / max(1e-6, 1 - normal_ratio)))
        n_normal = min(n_normal, len(normal))
        step = max(1, len(normal) // max(n_normal, 1))
        sampled = normal[::step][:n_normal]
        return list(defect) + sampled


    @staticmethod
    def _stratified_limit(entries: List[Entry], limit: int) -> List[Entry]:
        """按正常/疵点比例截断, 避免 limit 直接把正常图全部截掉."""
        defects = [e for e in entries if not e.normal]
        normals = [e for e in entries if e.normal]
        if not defects:
            return normals[:limit]
        if not normals:
            return defects[:limit]
        ratio = len(normals) / len(entries)
        keep_normal = max(1, int(round(limit * ratio)))
        keep_defect = max(1, limit - keep_normal)
        step_n = max(1, len(normals) // keep_normal)
        step_d = max(1, len(defects) // keep_defect)
        return defects[::step_d][:keep_defect] + normals[::step_n][:keep_normal]

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def _cloth_mask(self, entry: Entry, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if entry.mask is not None:
            m = np.asarray(Image.open(entry.mask).convert("L"), dtype=np.uint8)
            if m.shape[:2] != (h, w):
                m = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST), dtype=np.uint8)
            return (m > 127).astype(np.uint8)
        key = str(entry.image)
        if self.cache_masks and key in self._cloth_cache:
            return self._cloth_cache[key]
        mask, confidence = estimate_cloth_mask(image, work_size=self.mask_work_size)
        if confidence <= 0:
            mask = np.ones((h, w), dtype=np.uint8)
        if self.cache_masks:
            if len(self._cloth_cache) > 4096:
                self._cloth_cache.clear()
            self._cloth_cache[key] = mask
        return mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        image = load_rgb(entry.image)
        h, w = image.shape[:2]
        boxes, polygons = read_yolo_labels(entry.label)
        defect_mask = boxes_to_mask(boxes, polygons, h, w)
        cloth_mask = self._cloth_mask(entry, image)

        sample: Sample = {
            "image": image,
            "defect_mask": defect_mask,
            "cloth_mask": cloth_mask,
            "boxes": boxes,
        }
        rng = np.random.default_rng()
        sample = self.transform(sample, rng) if self.train else self.transform(sample)

        image_t = _to_tensor(sample["image"])
        defect = torch.from_numpy(sample["defect_mask"][None].astype(np.float32))
        cloth = torch.from_numpy(sample["cloth_mask"][None].astype(np.float32))
        synth = sample.get("synth_mask")
        anomaly_target = defect
        if synth is not None:
            synth_t = torch.from_numpy(synth[None].astype(np.float32))
            anomaly_target = torch.clamp(defect + synth_t, 0, 1)
        boxes_t = torch.from_numpy(np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5))
        labels = boxes_t[:, 0].long() if boxes_t.numel() else torch.zeros(0, dtype=torch.long)

        return {
            "image": image_t,
            "defect_mask": defect,
            "anomaly_mask": anomaly_target,
            "cloth_mask": cloth,
            "boxes": boxes_t,
            "labels": labels,
            "image_label": int(labels[0]) if labels.numel() else -1,
            "normal": bool(entry.normal or labels.numel() == 0),
            "path": str(entry.image),
        }


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "defect_mask": torch.stack([b["defect_mask"] for b in batch]),
        "anomaly_mask": torch.stack([b["anomaly_mask"] for b in batch]),
        "cloth_mask": torch.stack([b["cloth_mask"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "image_label": torch.tensor([b["image_label"] for b in batch], dtype=torch.long),
        "normal": torch.tensor([b["normal"] for b in batch], dtype=torch.bool),
        "path": [b["path"] for b in batch],
    }
    return out


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build_datasets(cfg: Dict[str, Any]) -> Dict[str, FabricDataset]:
    data_cfg = cfg.get("data", {})
    spec = load_dataset_spec(cfg)
    image_size = data_cfg.get("image_size", [512, 512])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    aug_cfg = dict(data_cfg.get("augment", {}))
    aug_cfg["image_size"] = tuple(image_size)
    aug_cfg["mask_work_size"] = data_cfg.get("mask_work_size", "auto")
    normal_ratio = float(data_cfg.get("normal_ratio", 0.4))
    datasets: Dict[str, FabricDataset] = {
        "train": FabricDataset(
            spec, "train", tuple(image_size), train=True, augment_cfg=aug_cfg,
            normal_ratio=normal_ratio, limit=data_cfg.get("limit_train"),
        ),
        "val": FabricDataset(
            spec, "val", tuple(image_size), train=False, augment_cfg=aug_cfg,
            normal_ratio=0.5, limit=data_cfg.get("limit_val"),
        ),
    }
    return datasets


def build_dataloaders(cfg: Dict[str, Any], datasets: Optional[Dict[str, FabricDataset]] = None) -> Dict[str, DataLoader]:
    data_cfg = cfg.get("data", {})
    if datasets is None:
        datasets = build_datasets(cfg)
    batch_size = int(data_cfg.get("batch_size", 8))
    eval_batch = int(data_cfg.get("eval_batch_size", 4))
    workers = int(data_cfg.get("num_workers", 4))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=len(datasets["train"]) > batch_size,
            collate_fn=collate_fn,
            persistent_workers=workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch,
            shuffle=False,
            num_workers=max(0, workers // 2),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=workers > 1,
        ),
    }
    return loaders


def build_normal_dataset(cfg: Dict[str, Any], split: str = "train") -> Optional[FabricDataset]:
    """构建只含正常布面图的数据集 (用于训练前初始化记忆库)."""
    data_cfg = cfg.get("data", {})
    spec = load_dataset_spec(cfg)
    image_size = data_cfg.get("image_size", [512, 512])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    aug_cfg = dict(data_cfg.get("augment", {}))
    aug_cfg["image_size"] = tuple(image_size)
    aug_cfg["mask_work_size"] = data_cfg.get("mask_work_size", "auto")
    # 原尺度裁块训练时, 记忆库也必须用原尺度块建立, 否则分数尺度与推理时不一致
    native = str(aug_cfg.get("crop_mode", "scale")).lower() == "native"
    if native:
        aug_cfg["synth_prob"] = 0.0
    try:
        dataset = FabricDataset(
            spec, split, tuple(image_size), train=native, augment_cfg=aug_cfg, normal_ratio=0.0
        )
    except FileNotFoundError:
        return None
    normal = [e for e in dataset.entries if e.normal] + list(dataset.normal_entries)
    seen, uniq = set(), []
    for entry in normal:
        key = str(entry.image)
        if key not in seen:
            seen.add(key)
            uniq.append(entry)
    if not uniq:
        return None
    dataset.entries = uniq
    dataset._cloth_cache.clear()
    return dataset


def build_normal_loader(cfg: Dict[str, Any], split: str = "train") -> Optional[DataLoader]:
    dataset = build_normal_dataset(cfg, split)
    if dataset is None:
        return None
    workers = int(cfg.get("data", {}).get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("data", {}).get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )
