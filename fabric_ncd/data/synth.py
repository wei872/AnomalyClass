"""合成疵点 (自监督 / 增广).

现实产线上 "正常布" 远多于疵点, 而 YOLO 一类全监督检测器在缺陷类别不均衡、
新疵点出现时泛化很差. 这里用与真实疵点形态相近的合成异常做自监督:

====================  ==========================================
生成器                模拟的真实疵点
====================  ==========================================
``scratch``           划伤 / 断经 / 小尾巴 (细长、高对比)
``stain``             油点线 / 污渍 / 棉团 (块状、低频色差)
``cutpaste``          小布头 / 异物 / 多股线 (纹理不连续)
``weave_shift``       纬斜 / 稀缝 / 断纬 (经纬结构局部错位)
``missing_yarn``      缺纬 / 稀缝 (整条纱线缺失)
``foreign_fiber``     飘絮 / 飞花 / 单根包不进 (细长异色纤维)
====================  ==========================================

所有函数都是纯 numpy + PIL 实现, 可在 dataloader 多进程里安全运行.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ImageMask = np.ndarray


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def _from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8)


def _random_point(rng: np.random.Generator, h: int, w: int, margin: float = 0.05) -> Tuple[float, float]:
    return (
        float(rng.uniform(margin * w, (1 - margin) * w)),
        float(rng.uniform(margin * h, (1 - margin) * h)),
    )


def _paste_mask(mask: np.ndarray, draw_mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    out = mask.copy()
    out = np.maximum(out, (draw_mask * alpha).astype(np.uint8))
    return out


# --------------------------------------------------------------------------- #
# 生成器
# --------------------------------------------------------------------------- #
def add_scratch(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    draw = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(draw)
    x0, y0 = _random_point(rng, h, w, 0.1)
    angle = float(rng.uniform(0, 2 * math.pi))
    length = float(rng.uniform(0.08, 0.45) * min(h, w))
    width = int(rng.integers(1, 5))
    points = [(x0, y0)]
    n_seg = int(rng.integers(2, 6))
    for i in range(n_seg):
        angle += float(rng.uniform(-0.35, 0.35))
        step = length / n_seg
        x0, y0 = x0 + step * math.cos(angle), y0 + step * math.sin(angle)
        points.append((x0, y0))
    drawer.line(points, fill=255, width=width, joint="curve")
    scratch_mask = np.asarray(draw, dtype=np.float32) / 255.0
    if rng.random() < 0.6:
        scratch_mask = np.asarray(
            Image.fromarray((scratch_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)),
            dtype=np.float32,
        ) / 255.0
    delta = float(rng.choice([-1, 1]) * rng.uniform(35, 110))
    out = image.astype(np.float32) + delta * scratch_mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), _paste_mask(mask, (scratch_mask > 0.25).astype(np.uint8))


def add_stain(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    low = (max(4, h // 32), max(4, w // 32))
    noise = rng.normal(size=(low[0], low[1])).astype(np.float32)
    noise = np.asarray(
        Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-6) * 255).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    threshold = float(rng.uniform(0.55, 0.75))
    blob = (noise > threshold).astype(np.float32)
    if blob.sum() < 20:
        cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
        r = rng.integers(min(h, w) // 32 + 2, min(h, w) // 8 + 3)
        yy, xx = np.ogrid[:h, :w]
        blob = (((yy - cy) ** 2 + (xx - cx) ** 2) < r * r).astype(np.float32)
    blob = np.asarray(
        Image.fromarray((blob * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32,
    ) / 255.0
    color = np.array(rng.uniform(0.15, 0.9, size=3), dtype=np.float32) * 255.0
    alpha = float(rng.uniform(0.35, 0.75))
    out = image.astype(np.float32) * (1 - alpha * blob[..., None]) + color * (alpha * blob[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), _paste_mask(mask, (blob > 0.4).astype(np.uint8))


def add_cutpaste(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    ph = int(rng.uniform(0.06, 0.25) * h)
    pw = int(rng.uniform(0.06, 0.25) * w)
    ph, pw = min(h, max(ph, 6)), min(w, max(pw, 6))
    y0 = int(rng.integers(0, max(1, h - ph)))
    x0 = int(rng.integers(0, max(1, w - pw)))
    patch = image[y0 : y0 + ph, x0 : x0 + pw].copy()
    pil_patch = _to_pil(patch)
    angle = float(rng.uniform(-45, 45))
    pil_patch = pil_patch.rotate(angle, expand=True, resample=Image.BILINEAR)
    if rng.random() < 0.5:
        pil_patch = pil_patch.transpose(Image.FLIP_LEFT_RIGHT)
    ph2, pw2 = pil_patch.size[1], pil_patch.size[0]
    if ph2 > h or pw2 > w:
        pil_patch.thumbnail((w, h), Image.Resampling.BILINEAR)
        ph2, pw2 = pil_patch.size[1], pil_patch.size[0]
    dy = int(rng.integers(0, max(1, h - ph2)))
    dx = int(rng.integers(0, max(1, w - pw2)))
    out = image.copy()
    out[dy : dy + ph2, dx : dx + pw2] = _from_pil(pil_patch)
    m = np.zeros((h, w), dtype=np.uint8)
    m[dy : dy + ph2, dx : dx + pw2] = 1
    return out, _paste_mask(mask, m)


def add_weave_shift(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """局部纬斜 / 稀缝: 在一个水平带内按正弦曲线整体横向错位."""
    h, w = mask.shape
    band_h = int(rng.uniform(0.04, 0.18) * h)
    y0 = int(rng.integers(0, max(1, h - band_h)))
    amplitude = float(rng.uniform(2.0, max(3.0, 0.02 * w)))
    period = float(rng.uniform(0.2, 0.9) * w)
    out = image.copy()
    xs = np.arange(w)
    shift = (amplitude * np.sin(2 * math.pi * xs / period)).astype(np.int32)
    for y in range(y0, y0 + band_h):
        out[y] = np.roll(image[y], int(shift[y % w]), axis=0)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0 : y0 + band_h] = 1
    return out, _paste_mask(mask, m)


def add_missing_yarn(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """缺纬 / 稀缝: 整条横向纱线被邻近纹理替代."""
    h, w = mask.shape
    thickness = int(rng.integers(1, max(2, h // 80)))
    y0 = int(rng.integers(h // 10, max(h // 10 + 1, h - h // 10 - thickness)))
    seg_w = int(rng.uniform(0.15, 0.7) * w)
    x0 = int(rng.integers(0, max(1, w - seg_w)))
    out = image.copy()
    src = np.clip(y0 + int(rng.choice([-1, 1]) * rng.integers(2, 6)), 0, max(0, h - thickness))
    out[y0 : y0 + thickness, x0 : x0 + seg_w] = image[src : src + thickness, x0 : x0 + seg_w]
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0 : y0 + thickness, x0 : x0 + seg_w] = 1
    return out, _paste_mask(mask, m)


def add_foreign_fiber(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, region: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """飘絮 / 飞花: 细长、高饱和度的异色纤维."""
    h, w = mask.shape
    draw = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(draw)
    x0, y0 = _random_point(rng, h, w)
    points = [(x0, y0)]
    angle = float(rng.uniform(0, 2 * math.pi))
    for _ in range(int(rng.integers(3, 8))):
        angle += float(rng.uniform(-0.5, 0.5))
        step = float(rng.uniform(0.02, 0.08) * min(h, w))
        x0, y0 = x0 + step * math.cos(angle), y0 + step * math.sin(angle)
        points.append((x0, y0))
    drawer.line(points, fill=255, width=int(rng.integers(1, 3)), joint="curve")
    fiber = np.asarray(draw, dtype=np.float32) / 255.0
    fiber = np.asarray(
        Image.fromarray((fiber * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)),
        dtype=np.float32,
    ) / 255.0
    color = np.array(rng.choice([[235, 235, 235], [210, 40, 40], [40, 70, 200], [240, 210, 60]]), dtype=np.float32)
    out = image.astype(np.float32) * (1 - fiber[..., None]) + color * fiber[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), _paste_mask(mask, (fiber > 0.3).astype(np.uint8))


SYNTH_FUNCS: Dict[str, Callable] = {
    "scratch": add_scratch,
    "stain": add_stain,
    "cutpaste": add_cutpaste,
    "weave_shift": add_weave_shift,
    "missing_yarn": add_missing_yarn,
    "foreign_fiber": add_foreign_fiber,
}
SYNTH_TYPES: Tuple[str, ...] = tuple(SYNTH_FUNCS.keys())


def apply_random_anomaly(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    types: Sequence[str] = SYNTH_TYPES,
    max_ops: int = 2,
    prob: float = 1.0,
    region: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机叠加 1~max_ops 个合成疵点, 返回 (image, 合成掩码)."""
    synth_mask = np.zeros_like(mask, dtype=np.uint8)
    if rng.random() > prob or not types or max_ops < 1:
        return image, synth_mask
    if region is not None:
        region = np.asarray(region, dtype=bool)
        if region.shape != mask.shape:
            raise ValueError("region 与掩码的尺寸必须一致")
        if not region.any():
            return image, synth_mask
    n_ops = int(rng.integers(1, max_ops + 1))
    for _ in range(n_ops):
        name = str(rng.choice(list(types)))
        func = SYNTH_FUNCS.get(name)
        if func is None:
            continue
        modified, op_mask = func(image, np.zeros_like(mask, dtype=np.uint8), rng, region)
        if region is not None:
            modified = np.where(region[..., None], modified, image)
            op_mask = op_mask * region
        image = modified
        synth_mask = np.maximum(synth_mask, op_mask)
    return image, synth_mask
