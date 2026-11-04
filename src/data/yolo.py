"""YOLO 格式数据集读取: 缺陷图 (images + labels, 框标注) 与纯正常图 (normal, 无标签)."""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..util import list_images, read_yolo_label, yolo_to_xyxy, rasterize_boxes, imread_unicode

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def label_path_for(img_path: str, labels_dir: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    return os.path.join(labels_dir, base)


class YoloDefectDataset(Dataset):
    """带 YOLO 框标注的缺陷数据集 (train / val 共用).

    每项: (img_rgb np.uint8 HWC, boxes_xyxy [(cls,x1,y1,x2,y2)], mask np.uint8, img_path)
    """

    def __init__(self, images_dir: str, labels_dir: Optional[str] = None):
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        yolo = read_yolo_label(label_path_for(img_path, self.labels_dir)) if self.labels_dir else []
        boxes = yolo_to_xyxy(yolo, w, h)
        mask = rasterize_boxes(h, w, boxes)
        return img, boxes, mask, img_path


class NormalImageDataset(Dataset):
    """纯正常样本 (无标签), 用于无监督检测器训练 / 校准. 返回 (img_rgb, img_path)."""

    def __init__(self, images_dir: str):
        self.images = list_images(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {img_path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_path


def extract_crops(img: np.ndarray, boxes: List[Tuple], pad: float = 0.3, min_side: int = 8):
    """按框裁剪缺陷区域 (加 pad 上下文), 返回 [(crop_rgb, (cls,x1,y1,x2,y2))]."""
    h, w = img.shape[:2]
    crops = []
    for c, x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(w, int(np.ceil(x2 + bw * pad)))
        y2 = min(h, int(np.ceil(y2 + bh * pad)))
        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue
        crops.append((img[y1:y2, x1:x2].copy(), (int(c), x1, y1, x2, y2)))
    return crops


def build_crop_samples(images_dir: str, labels_dir: Optional[str], pad: float = 0.3,
                       max_per_image: Optional[int] = None):
    """遍历整个 YOLO 数据集, 产出全部缺陷裁剪样本: [(crop_rgb, cls, img_path, box)]."""
    ds = YoloDefectDataset(images_dir, labels_dir)
    samples = []
    for i in range(len(ds)):
        img, boxes, _, img_path = ds[i]
        crops = extract_crops(img, boxes, pad=pad)
        if max_per_image:
            crops = crops[:max_per_image]
        for crop, (c, x1, y1, x2, y2) in crops:
            samples.append((crop, c, img_path, (x1, y1, x2, y2)))
    return samples


def pil_to_tensor_batch(crops: List[np.ndarray], crop_size: int = 224, augment: bool = False):
    """把一批 RGB 裁剪转成 (N,3,crop_size,crop_size) tensor (ImageNet 归一化)."""
    from torchvision import transforms as T
    if augment:
        tf = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop((crop_size, crop_size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch.stack([tf(c) for c in crops])
