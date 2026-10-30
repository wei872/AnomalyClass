"""数据侧的织造周期估计 (numpy 版).

与 ``models/texture.py`` 中的 torch 版本保持同样思路, 供 dataloader 做
"织造尺度归一化": 把不同布种的经纬间距归一化到同一像素尺度, 相当于把
"纹理尺度"这一变化因素从网络中移除, 是跨布种泛化的关键预处理.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _peak_period(energy: np.ndarray, work: int, min_period: float, max_period: float) -> Tuple[float, float]:
    f_lo = max(1, int(np.floor(work / max_period)))
    f_hi = min(len(energy) - 1, int(np.ceil(work / min_period)))
    band = energy[f_lo : f_hi + 1]
    if band.size == 0:
        return 8.0, 0.0
    idx = int(np.argmax(band)) + f_lo
    peak = float(band.max())
    mean = float(band.mean() + 1e-8)
    # 抛物线插值
    if 0 < idx < len(energy) - 1:
        left, center, right = np.log(energy[idx - 1] + 1e-8), np.log(energy[idx] + 1e-8), np.log(energy[idx + 1] + 1e-8)
        denom = left - 2 * center + right
        if abs(denom) > 1e-6:
            idx = idx + float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
    period = work / max(idx, 1e-3)
    return float(np.clip(period, min_period, max_period)), peak / mean


def estimate_period(
    image: np.ndarray,
    work_size: int = 256,
    min_period: float = 2.0,
    max_period: float = 64.0,
) -> Tuple[float, float, float]:
    """返回 (period_x, period_y, confidence), 单位为输入图像像素."""
    gray = to_gray(image)
    h, w = gray.shape
    pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    small = np.asarray(pil.resize((work_size, work_size), Image.BILINEAR), dtype=np.float32)
    small = small - small.mean()
    win = np.hanning(work_size)
    window = np.outer(win, win)
    spec = np.abs(np.fft.rfft2(small * window)) ** 2
    spec[:, 0] = 0
    energy_y = spec.mean(axis=1)
    energy_x = spec.mean(axis=0)
    period_y, conf_y = _peak_period(energy_y, work_size, min_period, max_period)
    period_x, conf_x = _peak_period(energy_x, work_size, min_period, max_period)
    period_y *= h / work_size
    period_x *= w / work_size
    conf = min(conf_x, conf_y)
    return period_x, period_y, conf


def weave_scale_factor(
    image: np.ndarray,
    target_period: float = 8.0,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    min_confidence: float = 2.0,
) -> float:
    """计算使织造周期对齐到 ``target_period`` 的缩放系数."""
    px, py, conf = estimate_period(image)
    if conf < min_confidence:
        return 1.0
    period = float(np.sqrt(max(px * py, 1e-6)))
    scale = target_period / period
    return float(np.clip(scale, min_scale, max_scale))
