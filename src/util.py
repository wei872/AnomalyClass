"""通用工具: 日志 / 随机种子 / 高斯核 / 评估指标 / 计时 / 可视化辅助."""
import logging
import math
import os
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


class CallableLogger:
    """可直接调用的 logger: log('...') 等价于 log.info('...')."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __call__(self, *a, **k):
        self._logger.info(*a, **k)

    def __getattr__(self, item):
        return getattr(self._logger, item)


def get_logger(name: str = 'anomalyclass', level: str = 'INFO') -> CallableLogger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    logger.propagate = False
    return CallableLogger(logger)


def setup_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'图片目录不存在: {folder}')
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(IMG_EXTENSIONS)]
    if not files:
        raise FileNotFoundError(f'目录中没有图片: {folder}')
    return files


def imread_unicode(path: str, flags: int = None):
    """安全读图 (支持中文/非 ASCII 路径, bmp/png/jpg 等任意 cv2/PIL 可解格式).

    三级回退:
      1. 内置 open(path,'rb') 读字节 (Windows 上走宽字符 API, 任何路径都可靠) -> cv2.imdecode
      2. PIL.Image.open(path) (同样 Unicode 安全, 能解部分 cv2 解不了的格式) -> 转 BGR ndarray
      3. cv2.imread(path) 兜底 (ASCII 路径 / 非 Windows 平台)
    全部失败返回 None.
    """
    import cv2
    if flags is None:
        flags = cv2.IMREAD_COLOR
    # 1) open + cv2.imdecode
    try:
        with open(path, 'rb') as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
        if img is not None:
            return img
    except Exception:
        pass
    # 2) PIL
    try:
        from PIL import Image
        with Image.open(path) as im:
            if flags == cv2.IMREAD_UNCHANGED:
                arr = np.array(im)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                if arr.ndim == 3:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return arr
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.array(im.convert('L'))
            return cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    # 3) cv2.imread 兜底
    try:
        return cv2.imread(path, flags)
    except Exception:
        return None


def image_diag(path: str) -> dict:
    """读图失败时的诊断信息: 是否存在 / 大小 / 文件头魔数."""
    info = {'path': path, 'exists': os.path.exists(path)}
    try:
        info['size'] = os.path.getsize(path) if info['exists'] else None
        if info['exists']:
            with open(path, 'rb') as f:
                info['magic_hex'] = f.read(16).hex()
    except Exception as e:
        info['error'] = str(e)
    return info


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """安全写图 (支持中文路径): cv2.imencode 编码后用内置 open('wb') 写出."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or '.png'
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img)


def get_gaussian_kernel(kernel_size: int = 5, sigma: float = 4, channels: int = 1, device='cpu',
                        padding_mode: str = 'zeros'):
    """与 Dinomaly2 utils.get_gaussian_kernel 相同: depthwise 高斯卷积 (nn.Conv2d, 可直接调用)."""
    import torch.nn as nn
    x_coord = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2. * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                groups=channels, bias=False, padding=kernel_size // 2, padding_mode=padding_mode, device=device)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


# ------------------------- 评估指标 (CPU, 不依赖 adeval) -------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def roc_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y_true).ravel(), np.asarray(y_score).ravel()))


def average_precision(y_true, y_score):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y_true).ravel(), np.asarray(y_score).ravel()))


def f1_max(y_true, y_score):
    from sklearn.metrics import precision_recall_curve
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return float(np.nanmax(f1s[:-1])) if len(f1s) > 1 else float(np.nanmax(f1s))


def best_threshold(y_true, y_score):
    """按 F1 最大选阈值 (用于报告)."""
    from sklearn.metrics import precision_recall_curve
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    return float(thrs[np.argmax(f1s)]) if len(thrs) else float(np.max(y_score))


# ------------------------- 计时 -------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


class FpsMeter:
    """指数滑动平均 FPS."""

    def __init__(self, ema: float = 0.9):
        self.ema = ema
        self.fps = 0.0
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else self.ema * self.fps + (1 - self.ema) * inst
        self._last = now
        return self.fps


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def torch_load(path: str, map_location='cpu'):
    """安全的 torch.load: 优先 weights_only=True (只允许张量/基本类型, 防 pickle 注入).

    本项目所有 checkpoint (detector / classifier / DINOv2 官方权重) 都是纯 state_dict 或
    {state_dict, 基本类型字典}, 均兼容 weights_only=True; 老版本 torch 无该参数时回退.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    """与 Dinomaly2 utils.cal_anomaly_maps 一致: 1 - cos(en, de), 上采样后按组平均."""
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    a_map_list = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


# ------------------------- 框 / 掩码工具 -------------------------
def yolo_to_xyxy(cls_cxcywh, img_w: int, img_h: int):
    """归一化 (cls, cx, cy, w, h) -> 像素 (cls, x1, y1, x2, y2)."""
    out = []
    for c, cx, cy, w, h in cls_cxcywh:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append((int(c), float(x1), float(y1), float(x2), float(y2)))
    return out


def xyxy_to_yolo(boxes, img_w: int, img_h: int):
    out = []
    for c, x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        out.append((int(c), cx, cy, w, h))
    return out


def read_yolo_label(txt_path: str) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            boxes.append((c, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def rasterize_boxes(img_h: int, img_w: int, boxes_xyxy, fill: int = 1) -> np.ndarray:
    """把 YOLO 框栅格化为像素级 GT 掩码 (用于像素级 AUROC 评估)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for c, x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(math.ceil(x2)), img_w), min(int(math.ceil(y2)), img_h)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = fill
    return mask


def expand_box(box, img_w: int, img_h: int, pad: float):
    c, x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - w * pad)
    y1 = max(0.0, y1 - h * pad)
    x2 = min(float(img_w), x2 + w * pad)
    y2 = min(float(img_h), y2 + h * pad)
    return (c, x1, y1, x2, y2)


def mask_to_boxes(mask: np.ndarray, min_area: int = 0, dilate: int = 0):
    """二值掩码 -> 连通域外接框. 返回 [(x1,y1,x2,y2,area)] (整数坐标)."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    boxes.sort(key=lambda b: -b[4])
    return boxes


# ------------------------- 中文可视化字体 -------------------------
_CJK_FONT_CANDIDATES = [
    'assets/fonts/NotoSansSC-Regular.otf',
    'assets/fonts/NotoSansCJKsc-Regular.otf',
    'assets/fonts/wqy-microhei.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simsun.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
]


def get_cjk_font(size: int = 16, font_path: str = None):
    """找一个能渲染中文的字体; 找不到返回 None (调用方退回英文标签)."""
    from PIL import ImageFont
    cands = [font_path] if font_path else []
    cands += _CJK_FONT_CANDIDATES
    for p in cands:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


_CLASS_PALETTE = [
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0),
    (255, 0, 128), (0, 255, 128), (192, 192, 0), (0, 192, 192), (192, 0, 192),
    (64, 64, 255), (64, 255, 64), (255, 64, 64), (160, 160, 160),
]
UNKNOWN_COLOR = (160, 160, 160)


def class_color(idx: int):
    return _CLASS_PALETTE[idx % len(_CLASS_PALETTE)]
