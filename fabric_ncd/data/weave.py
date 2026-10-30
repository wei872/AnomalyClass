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


def _peak_bin(energy: np.ndarray, dim: int, min_period: float, max_period: float) -> Tuple[float, float]:
    """在谱上找主峰, 返回 (整幅内周期数 k, 置信度). 周期 = dim / k."""
    nbins = len(energy)
    k_lo = max(1, int(np.floor(dim / max_period)))
    k_hi = max(k_lo + 1, min(nbins - 2, int(np.ceil(dim / min_period))))
    band = energy[k_lo : k_hi + 1]
    if band.size == 0:
        return float(dim) / 8.0, 0.0
    idx = int(np.argmax(band)) + k_lo
    peak = float(band.max())
    mean = float(band.mean() + 1e-8)
    k = float(idx)
    if 0 < idx < nbins - 1:
        left = np.log(energy[idx - 1] + 1e-8)
        center = np.log(energy[idx] + 1e-8)
        right = np.log(energy[idx + 1] + 1e-8)
        denom = left - 2 * center + right
        if abs(denom) > 1e-6:
            k = k + float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
    return max(k, 1e-3), peak / mean


def estimate_period(
    image: np.ndarray,
    work_size: int = 512,
    min_period: float = 2.0,
    max_period: float = 64.0,
) -> Tuple[float, float, float]:
    """返回 (period_x, period_y, confidence), 单位为输入图像像素.

    保持长宽比、只做整数倍池化: 两个方向分别用各自的真实尺寸把谱峰换算成周期
    (``period = dim / k``), 因此 1600x846 这类非正方形图像不会被压扁失真,
    高分辨率下也不会因为过度抽取而丢失细密经纬.
    """
    gray = to_gray(image)
    h, w = gray.shape
    factor = max(1, int(min(h, w) // max(8, int(work_size))))
    ph, pw = max(8, h // factor), max(8, w // factor)
    pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    small = np.asarray(pil.resize((pw, ph), Image.BILINEAR), dtype=np.float32)
    small = small - small.mean()
    window = np.outer(np.hanning(ph), np.hanning(pw))
    spec = np.abs(np.fft.rfft2(small * window)) ** 2
    spec[:, 0] = 0
    energy_y = spec.mean(axis=1)[: max(2, ph // 2)]
    energy_x = spec.mean(axis=0)
    k_y, conf_y = _peak_bin(energy_y, h, min_period, max_period)
    k_x, conf_x = _peak_bin(energy_x, w, min_period, max_period)
    period_y = float(np.clip(h / k_y, min_period, max_period))
    period_x = float(np.clip(w / k_x, min_period, max_period))
    conf = min(conf_x, conf_y)
    if conf < 2.0:
        period_x = period_y = 8.0
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
