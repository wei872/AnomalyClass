"""YOLO 格式数据集读取: 缺陷图 (images + labels, 框标注) 与纯正常图 (normal, 无标签)."""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..util import (list_images, read_yolo_label, yolo_to_xyxy, rasterize_boxes, imread_unicode,
                    image_diag)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def label_path_for(img_path: str, labels_dir: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    return os.path.join(labels_dir, base)


def verify_images(paths: List[str], logger=None, tag: str = '') -> Tuple[List[str], List[str]]:
    """逐个探测图片是否可解码; 返回 (可读列表, 坏文件列表). 坏文件会被跳过并打日志."""
    ok, bad = [], []
    for i, p in enumerate(paths):
        if imread_unicode(p, cv2.IMREAD_COLOR) is None:
            bad.append(p)
            if logger:
                logger(f'[data] 跳过无法解码的图片: {p}  诊断: {image_diag(p)}')
        else:
            ok.append(p)
        if logger and (i + 1) % 500 == 0:
            logger(f'[data] {tag} 探测进度 {i + 1}/{len(paths)} (坏文件 {len(bad)})')
    return ok, bad


class YoloDefectDataset(Dataset):
    """带 YOLO 框标注的缺陷数据集 (train / val 共用).

    每项: (img_rgb np.uint8 HWC, boxes_xyxy [(cls,x1,y1,x2,y2)], mask np.uint8, img_path)
    verify=True 时在初始化探测并跳过坏文件 (避免训练中途崩溃).
    """

    def __init__(self, images_dir: str, labels_dir: Optional[str] = None, verify: bool = True, logger=None,
                 max_side: int = 2048):
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir
        self.max_side = max_side
        self.skipped: List[str] = []
        if verify:
            self.images, self.skipped = verify_images(self.images, logger=logger, tag=images_dir)
            if not self.images:
                raise FileNotFoundError(f'目录中没有可解码的图片: {images_dir}')

    def __len__(self):
        return len(self.images)

    def _read(self, img_path: str):
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {img_path}  诊断: {image_diag(img_path)}')
        if self.max_side and max(img.shape[:2]) > self.max_side:   # 超大图先缩小, 防内存爆炸
            s = self.max_side / max(img.shape[:2])
            img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = self._read(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        yolo = read_yolo_label(label_path_for(img_path, self.labels_dir)) if self.labels_dir else []
        boxes = yolo_to_xyxy(yolo, w, h)
        mask = rasterize_boxes(h, w, boxes)
        return img, boxes, mask, img_path


class NormalImageDataset(Dataset):
    """纯正常样本 (无标签), 与带框缺陷图混合参与检测器联合训练 / 阈值校准. 返回 (img_rgb, img_path).

    注意: 读图必须用 imread_unicode (Windows 上 cv2.imread 不认中文路径).
    verify=True 时在初始化探测并跳过坏文件.
    """

    def __init__(self, images_dir: str, verify: bool = True, logger=None):
        self.images = list_images(images_dir)
        self.skipped: List[str] = []
        if verify:
            self.images, self.skipped = verify_images(self.images, logger=logger, tag=images_dir)
            if not self.images:
                raise FileNotFoundError(f'目录中没有可解码的图片: {images_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {img_path}  诊断: {image_diag(img_path)}')
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


def build_crop_index(images_dir: str, labels_dir: Optional[str], pad: float = 0.3):
    """轻量裁剪索引: ([(img_path, (cls, cx, cy, w, h) 归一化, pad)], 坏文件列表).

    只存路径 + 归一化框, 不加载任何像素 -> 整个索引可以安全 pickle
    (Windows spawn 的 DataLoader worker 需要 pickle 数据集, 存像素会 MemoryError).
    坏文件用 PIL 头部校验快速跳过 (Unicode 路径安全).
    """
    files = list_images(images_dir)
    index, bad = [], []
    for p in files:
        try:
            from PIL import Image
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(p)
            continue
        yolo = read_yolo_label(label_path_for(p, labels_dir)) if labels_dir else []
        for c, cx, cy, w, h in yolo:
            if w > 0.002 and h > 0.002:
                index.append((p, (int(c), float(cx), float(cy), float(w), float(h)), float(pad)))
    return index, bad


def build_crop_samples(images_dir: str, labels_dir: Optional[str], pad: float = 0.3,
                       max_per_image: Optional[int] = None):
    """遍历整个 YOLO 数据集, 产出全部缺陷裁剪样本: [(crop_rgb, cls, img_path, box)].

    注意: 会把所有裁剪像素加载进内存, 大数据集请用 build_crop_index + 惰性数据集.
    """
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


def _letterbox_pil(img_rgb: np.ndarray, size: int):
    """保持宽高比缩放到 size×size 画布中央 (灰边填充), 不压扁图像.

    布匹缺陷多为长条形 (断纬/纬斜/油点线), 直接 Resize 成正方形会破坏方向信息, letterbox 保留全貌.
    """
    from PIL import Image
    im = Image.fromarray(img_rgb)
    w, h = im.size
    if max(w, h) == 0:
        return Image.new('RGB', (size, size), (114, 114, 114))
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def pil_to_tensor_batch(crops: List[np.ndarray], crop_size: int = 224, augment: bool = False):
    """把一批 RGB 裁剪转成 (N,3,crop_size,crop_size) tensor (letterbox 保宽高比 + ImageNet 归一化).

    训练(augment=True)额外: 翻转/旋转90/颜色抖动/RandomErasing; 推理保持确定性.
    """
    from torchvision import transforms as T
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        def tf(c):
            im = _letterbox_pil(c, crop_size)
            im = T.RandomHorizontalFlip()(im)
            im = T.RandomVerticalFlip()(im)
            im = T.functional.rotate(im, int(np.random.choice([0, 90, 180, 270])))
            if np.random.rand() < 0.6:
                im = T.ColorJitter(0.3, 0.3, 0.3, 0.1)(im)
            x = T.ToTensor()(im)
            x = norm(x)
            if np.random.rand() < 0.3:
                x = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(x)
            return x
    else:
        def tf(c):
            return norm(T.ToTensor()(_letterbox_pil(c, crop_size)))
    return torch.stack([tf(c) for c in crops])
