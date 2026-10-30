"""面向布匹的几何 / 光照增广.

与通用检测增广最大的不同是 **布幅增广 (``fabric_width_augment``)**:

把布面图像贴到一块随机边距的 "背景画布" 上, 边距可以是深色辊筒/背景, 也可以
是布面的延伸 (布幅超出视野). 布面掩码随之精确变化, 于是模型在训练中见过大量
"布边位置不同、布幅宽窄不同" 的样本, 学到的是 "到布边的距离/纹理连续性",
而不是 "疵点在某个固定坐标". 这正好解决布幅改变导致边缘位置漂移的问题.

其他增广: 随机缩放裁剪、翻转/90° 旋转、小角度旋转、颜色抖动、模糊/噪声、合成疵点.
所有几何变换都同步更新布面掩码、疵点掩码与标注框 (归一化坐标).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .synth import SYNTH_TYPES, apply_random_anomaly
from .weave import weave_scale_factor

Sample = Dict[str, object]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image)


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def resize_sample(sample: Sample, size: Tuple[int, int]) -> Sample:
    """size = (H, W); 标注框为归一化坐标, 无需修改."""
    h, w = size
    image = sample["image"]
    if image.shape[0] == h and image.shape[1] == w:
        return sample
    out = dict(sample)
    out["image"] = _np(_pil(image).resize((w, h), Image.BILINEAR))
    out["defect_mask"] = _np(_pil(sample["defect_mask"]).resize((w, h), Image.NEAREST))
    out["cloth_mask"] = _np(_pil(sample["cloth_mask"]).resize((w, h), Image.NEAREST))
    if "synth_mask" in sample:
        out["synth_mask"] = _np(_pil(sample["synth_mask"]).resize((w, h), Image.NEAREST))
    return out


def _clip_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 5)
    boxes = boxes.copy()
    boxes[:, 1:5] = np.clip(boxes[:, 1:5], 0.0, 1.0)
    keep = (boxes[:, 3] - boxes[:, 1] > 1e-3) & (boxes[:, 4] - boxes[:, 2] > 1e-3)
    return boxes[keep]


def aspect_jitter(sample: Sample, scale: Tuple[float, float], rng: np.random.Generator) -> Sample:
    """各向异性拉伸 (x/y 独立缩放), 用来吃掉数据集里长宽比不一致的问题.

    画布整体 resize, 归一化坐标不变 (框/掩码自动跟随), 相当于把内容横向/纵向拉伸
    ``sx`` / ``sy`` 倍. 训练时见过各种拉伸比例, 推理时那 1.9~2.4 的长宽比差异就不再是分布外样本.
    """
    lo, hi = float(scale[0]), float(scale[1])
    h, w = sample["image"].shape[:2]
    sx = float(rng.uniform(lo, hi))
    sy = float(rng.uniform(lo, hi))
    nh = int(np.clip(round(h * sy), 64, max(64, h * 2)))
    nw = int(np.clip(round(w * sx), 64, max(64, w * 2)))
    if (nh, nw) == (h, w):
        return sample
    out = dict(sample)
    out["image"] = _np(_pil(sample["image"]).resize((nw, nh), Image.BILINEAR))
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = _np(_pil(sample[key]).resize((nw, nh), Image.NEAREST))
    return out

def random_scale_crop(sample: Sample, scale: Tuple[float, float], rng: np.random.Generator) -> Sample:
    h, w = sample["image"].shape[:2]
    s = float(rng.uniform(*scale))
    ch, cw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    out = dict(sample)
    out["image"] = sample["image"][y0 : y0 + ch, x0 : x0 + cw]
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = sample[key][y0 : y0 + ch, x0 : x0 + cw]
    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5)
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * w - x0) / cw
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * h - y0) / ch
        boxes = _clip_boxes(boxes)
    out["boxes"] = boxes
    return out



def native_scale_crop(
    sample: Sample,
    out_size: Tuple[int, int],
    rng: np.random.Generator,
    weave_scale: float = 1.0,
    jitter: Tuple[float, float] = (0.9, 1.1),
) -> Sample:
    """在原分辨率（而非整图缩放）上随机裁剪一块 ``out_size`` 大小的区域.

    大幅面布匹图整图缩放会把几像素宽的断经/飞花抹掉，
    更合理的做法是"按原像素尺度裁块训练"，推理时再用切块（tile）覆盖整图。

    ``weave_scale`` 由织造周期估计给出: 裁剪窗口 = out_size / weave_scale，
    再缩放回 out_size，等价于把经纬间距归一化到目标尺度，但只在一小块上做，
    避免整图放大带来的内存与耗时。
    """
    h, w = sample["image"].shape[:2]
    H, W = int(out_size[0]), int(out_size[1])
    s = float(np.clip(weave_scale * rng.uniform(*jitter), 0.25, 4.0))
    ch = int(round(H / max(s, 1e-6)))
    cw = int(round(W / max(s, 1e-6)))

    # 图像比裁剪窗口还小 -> 先整体放大到够裁
    if ch > h or cw > w:
        factor = max(ch / max(h, 1), cw / max(w, 1))
        sample = resize_sample(sample, (max(H, int(round(h * factor))), max(W, int(round(w * factor)))))
        h, w = sample["image"].shape[:2]
        ch, cw = min(ch, h), min(cw, w)
    ch, cw = max(16, min(ch, h)), max(16, min(cw, w))

    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    out = dict(sample)
    out["image"] = sample["image"][y0 : y0 + ch, x0 : x0 + cw]
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = sample[key][y0 : y0 + ch, x0 : x0 + cw]

    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5).copy()
    if boxes.size:
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * w - x0) / cw
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * h - y0) / ch
        boxes = _clip_boxes(boxes)
    out["boxes"] = boxes
    return resize_sample(out, (H, W))


def fabric_width_augment(
    sample: Sample,
    max_margin: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    extend_prob: float = 0.35,
) -> Sample:
    """把布面贴到随机边距的背景画布上 (模拟布幅/走偏变化)."""
    rng = rng or np.random.default_rng()
    if max_margin <= 0:
        return sample
    image = sample["image"]
    h, w = image.shape[:2]
    ml = int(rng.uniform(0, max_margin) * w)
    mr = int(rng.uniform(0, max_margin) * w)
    mt = int(rng.uniform(0, max_margin * 0.5) * h)
    mb = int(rng.uniform(0, max_margin * 0.5) * h)
    if ml + mr + mt + mb == 0:
        return sample
    H, W = h + mt + mb, w + ml + mr

    # 背景画布: 深色 + 轻微渐变 + 噪声 (模拟辊筒/导带/背景板)
    base = float(rng.uniform(20, 70))
    grad = np.linspace(0, float(rng.uniform(0, 25)), H, dtype=np.float32)[:, None, None]
    canvas = np.clip(base + grad + rng.normal(0, 3, size=(H, W, 3)), 0, 255).astype(np.uint8)
    canvas = canvas.astype(np.float32)
    image_f = image.astype(np.float32)

    # 逐边: 要么背景, 要么 "布幅延伸" (复制边缘列/行)
    cloth = np.zeros((H, W), dtype=np.uint8)
    cloth[mt : mt + h, ml : ml + w] = sample["cloth_mask"]
    defect = np.zeros((H, W), dtype=np.uint8)
    defect[mt : mt + h, ml : ml + w] = sample["defect_mask"]
    synth = np.zeros((H, W), dtype=np.uint8)
    if "synth_mask" in sample:
        synth[mt : mt + h, ml : ml + w] = sample["synth_mask"]

    if ml > 0:
        if rng.random() < extend_prob:
            canvas[mt : mt + h, :ml] = image_f[:, :1]
            cloth[mt : mt + h, :ml] = 1
        else:
            canvas[mt : mt + h, :ml] = base + grad[mt : mt + h]
    if mr > 0:
        if rng.random() < extend_prob:
            canvas[mt : mt + h, ml + w :] = image_f[:, -1:]
            cloth[mt : mt + h, ml + w :] = 1
        else:
            canvas[mt : mt + h, ml + w :] = base + grad[mt : mt + h]
    canvas[mt : mt + h, ml : ml + w] = image_f

    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5)
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * w + ml) / W
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * h + mt) / H

    out = dict(sample)
    out["image"] = np.clip(canvas, 0, 255).astype(np.uint8)
    out["cloth_mask"] = cloth
    out["defect_mask"] = defect
    if "synth_mask" in sample:
        out["synth_mask"] = synth
    out["boxes"] = boxes
    return out


def hflip(sample: Sample) -> Sample:
    out = dict(sample)
    out["image"] = np.ascontiguousarray(sample["image"][:, ::-1])
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = np.ascontiguousarray(sample[key][:, ::-1])
    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5).copy()
    if boxes.size:
        x1 = boxes[:, 1].copy()
        boxes[:, 1] = 1.0 - boxes[:, 3]
        boxes[:, 3] = 1.0 - x1
    out["boxes"] = boxes
    return out


def vflip(sample: Sample) -> Sample:
    out = dict(sample)
    out["image"] = np.ascontiguousarray(sample["image"][::-1])
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = np.ascontiguousarray(sample[key][::-1])
    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5).copy()
    if boxes.size:
        y1 = boxes[:, 2].copy()
        boxes[:, 2] = 1.0 - boxes[:, 4]
        boxes[:, 4] = 1.0 - y1
    out["boxes"] = boxes
    return out


def rot90(sample: Sample, k: int = 1) -> Sample:
    k = k % 4
    if k == 0:
        return sample
    out = dict(sample)
    out["image"] = np.ascontiguousarray(np.rot90(sample["image"], k))
    for key in ("defect_mask", "cloth_mask", "synth_mask"):
        if key in sample:
            out[key] = np.ascontiguousarray(np.rot90(sample[key], k))
    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5).copy()
    if boxes.size:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i, 1:5]
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            if k == 1:  # 逆时针 90°
                corners = np.stack([corners[:, 1], 1.0 - corners[:, 0]], axis=1)
            elif k == 2:
                corners = 1.0 - corners
            else:
                corners = np.stack([1.0 - corners[:, 1], corners[:, 0]], axis=1)
            boxes[i, 1] = corners[:, 0].min()
            boxes[i, 3] = corners[:, 0].max()
            boxes[i, 2] = corners[:, 1].min()
            boxes[i, 4] = corners[:, 1].max()
    out["boxes"] = _clip_boxes(boxes)
    return out


def rotate_sample(sample: Sample, angle: float, fill: int = 0) -> Sample:
    """小角度旋转 (expand=False); 标注框用四角旋转后的外接框近似."""
    if abs(angle) < 1e-3:
        return sample
    image = sample["image"]
    h, w = image.shape[:2]
    out = dict(sample)
    out["image"] = _np(_pil(image).rotate(angle, resample=Image.BILINEAR, fillcolor=(fill, fill, fill)))
    out["defect_mask"] = _np(_pil(sample["defect_mask"]).rotate(angle, resample=Image.NEAREST, fillcolor=0))
    out["cloth_mask"] = _np(_pil(sample["cloth_mask"]).rotate(angle, resample=Image.NEAREST, fillcolor=0))
    if "synth_mask" in sample:
        out["synth_mask"] = _np(_pil(sample["synth_mask"]).rotate(angle, resample=Image.NEAREST, fillcolor=0))

    boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 5).copy()
    if boxes.size:
        rad = math.radians(angle)
        cos, sin = math.cos(rad), math.sin(rad)
        cx, cy = 0.5, 0.5
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i, 1:5]
            # 转到像素空间再旋转, 避免宽高比造成的形变
            px = np.array([[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]])
            px[:, 0] -= cx * w
            px[:, 1] -= cy * h
            rx = px[:, 0] * cos + px[:, 1] * sin
            ry = -px[:, 0] * sin + px[:, 1] * cos
            px[:, 0] = rx + cx * w
            px[:, 1] = ry + cy * h
            boxes[i, 1] = px[:, 0].min() / w
            boxes[i, 3] = px[:, 0].max() / w
            boxes[i, 2] = px[:, 1].min() / h
            boxes[i, 4] = px[:, 1].max() / h
    out["boxes"] = _clip_boxes(boxes)
    return out


def color_jitter(sample: Sample, strength: float, rng: np.random.Generator) -> Sample:
    if strength <= 0:
        return sample
    img = _pil(sample["image"])
    factors = [
        1 + float(rng.uniform(-strength, strength)),  # brightness
        1 + float(rng.uniform(-strength, strength)),  # contrast
        1 + float(rng.uniform(-strength, strength)),  # saturation
    ]
    img = ImageEnhance.Brightness(img).enhance(factors[0])
    img = ImageEnhance.Contrast(img).enhance(factors[1])
    img = ImageEnhance.Color(img).enhance(factors[2])
    if rng.random() < 0.5:  # gamma
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gamma = float(rng.uniform(0.8, 1.25))
        arr = np.clip(arr**gamma * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    out = dict(sample)
    out["image"] = _np(img)
    return out


def blur_noise(sample: Sample, rng: np.random.Generator) -> Sample:
    img = sample["image"]
    out = dict(sample)
    choice = rng.random()
    if choice < 0.25:
        radius = float(rng.uniform(0.3, 1.2))
        out["image"] = _np(_pil(img).filter(ImageFilter.GaussianBlur(radius)))
    elif choice < 0.45:
        sigma = float(rng.uniform(2.0, 8.0))
        noisy = img.astype(np.float32) + rng.normal(0, sigma, size=img.shape)
        out["image"] = np.clip(noisy, 0, 255).astype(np.uint8)
    return out


def weave_scale_normalize(
    sample: Sample,
    target_period: float = 8.0,
    max_scale: float = 2.0,
    output_size: Optional[Tuple[int, int]] = None,
) -> Sample:
    """把不同布种的织造周期对齐到 target_period (跨布种尺度归一化)."""
    scale = weave_scale_factor(sample["image"], target_period, 1.0 / max_scale, max_scale)
    if abs(scale - 1.0) < 0.05:
        return sample
    h, w = sample["image"].shape[:2]
    out = dict(sample)
    out["image"] = _np(_pil(sample["image"]).resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR))
    out["defect_mask"] = _np(
        _pil(sample["defect_mask"]).resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.NEAREST)
    )
    out["cloth_mask"] = _np(
        _pil(sample["cloth_mask"]).resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.NEAREST)
    )
    if "synth_mask" in sample:
        out["synth_mask"] = _np(
            _pil(sample["synth_mask"]).resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.NEAREST)
        )
    return out


# --------------------------------------------------------------------------- #
# 组合增广
# --------------------------------------------------------------------------- #
class TrainAugment:
    def __init__(self, cfg: Dict) -> None:
        cfg = cfg or {}
        self.scale = tuple(cfg.get("scale", (0.5, 1.0)))
        self.hflip_p = float(cfg.get("hflip", 0.5))
        self.vflip_p = float(cfg.get("vflip", 0.5))
        self.rot90_p = float(cfg.get("rot90", 0.25))
        self.rotate = float(cfg.get("rotate", 3.0))
        self.jitter = float(cfg.get("color_jitter", 0.3))
        self.width_margin = float(cfg.get("width_margin", 0.3))
        self.weave_norm = bool(cfg.get("weave_norm", True))
        self.target_period = float(cfg.get("target_period", 8.0))
        self.synth_prob = float(cfg.get("synth_prob", 0.5))
        self.synth_types: Sequence[str] = cfg.get("synth_types", list(SYNTH_TYPES))
        self.synth_max_ops = int(cfg.get("synth_max_ops", 2))
        self.synth_scale = float(cfg.get("synth_scale", 1.0))  # 1.0 表示全部样本都可能叠加
        self.crop_mode = str(cfg.get("crop_mode", "scale")).lower()  # scale | native
        self.crop_scale = tuple(cfg.get("crop_scale", (0.85, 1.15)))
        # 各向异性拉伸: 数据集里长宽比不统一 (实测 1.88~2.42) 时, 让模型在训练中就见惯
        self.aspect = tuple(cfg.get("aspect_jitter", (1.0, 1.0)))
        self.aspect_p = float(cfg.get("aspect_prob", 0.0))
        self.out_size = tuple(cfg.get("image_size", (512, 512)))

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        sample = dict(sample)
        sample.setdefault("synth_mask", np.zeros_like(sample["defect_mask"]))
        native = self.crop_mode == "native"

        if native:
            # 大幅面: 先做布幅增广 (全图 + 掩码同步), 再按原像素尺度裁块, 不做整图缩放
            if self.width_margin > 0 and rng.random() < 0.85:
                sample = fabric_width_augment(sample, self.width_margin, rng)
            if rng.random() < self.hflip_p:
                sample = hflip(sample)
            if rng.random() < self.vflip_p:
                sample = vflip(sample)
            if rng.random() < self.rot90_p:
                sample = rot90(sample, int(rng.integers(1, 4)))
            weave_scale = 1.0
            if self.weave_norm:
                weave_scale = weave_scale_factor(sample["image"], self.target_period, 0.5, 2.0)
            sample = native_scale_crop(
                sample, self.out_size, rng, weave_scale=weave_scale, jitter=self.crop_scale
            )
        else:
            if self.weave_norm:
                sample = weave_scale_normalize(sample, self.target_period)
            if rng.random() < 0.8:
                sample = random_scale_crop(sample, self.scale, rng)
            if self.width_margin > 0 and rng.random() < 0.85:
                sample = fabric_width_augment(sample, self.width_margin, rng)
            if rng.random() < self.hflip_p:
                sample = hflip(sample)
            if rng.random() < self.vflip_p:
                sample = vflip(sample)
            if rng.random() < self.rot90_p:
                sample = rot90(sample, int(rng.integers(1, 4)))

        # native 裁块模式下画布固定为 out_size 且末尾不再 resize, 拉伸会让 batch 尺寸不一致,
        # 所以只在整图缩放模式里做 (该模式末尾会统一 resize 回 out_size)
        if (
            not native
            and self.aspect_p > 0
            and self.aspect[1] > self.aspect[0]
            and rng.random() < self.aspect_p
        ):
            sample = aspect_jitter(sample, self.aspect, rng)

        if self.rotate > 0 and rng.random() < 0.5:
            sample = rotate_sample(sample, float(rng.uniform(-self.rotate, self.rotate)))

        # 合成疵点: 只作用于布面区域, 与真实疵点叠加
        if self.synth_prob > 0 and rng.random() < self.synth_prob:
            img, synth = apply_random_anomaly(
                sample["image"],
                sample["defect_mask"],
                rng,
                types=self.synth_types,
                max_ops=self.synth_max_ops,
            )
            sample["image"] = img
            sample["synth_mask"] = np.maximum(sample["synth_mask"], synth)

        sample = color_jitter(sample, self.jitter, rng)
        sample = blur_noise(sample, rng)
        if not native:
            sample = resize_sample(sample, self.out_size)
        return sample


class EvalTransform:
    def __init__(self, cfg: Dict) -> None:
        cfg = cfg or {}
        self.weave_norm = bool(cfg.get("weave_norm", True))
        self.target_period = float(cfg.get("target_period", 8.0))
        self.out_size = tuple(cfg.get("image_size", (512, 512)))

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        if self.weave_norm:
            sample = weave_scale_normalize(sample, self.target_period)
        return resize_sample(sample, self.out_size)
