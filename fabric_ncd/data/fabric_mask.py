"""布面 (前景) / 背景 掩码估计.

真实产线图里, 布幅两侧常出现辊筒、导带、深色背景或反光金属, 它们的纹理与
布面完全不同. 如果让异常/分类分支在这些区域上学习, 模型会把 "背景" 误判
为疵点, 并且当布幅变窄 / 走偏时大面积误报. 因此需要布面前景掩码:

1. 优先使用用户提供的真实掩码 (``mask_dir``);
2. 否则用本文件的 **无监督** 估计器: 局部纹理能量 + 饱和度 + 与画面边缘
   背景色的差异, 再做 Otsu 分割与形态学清理;
3. 两者都没有时退出为 "整幅都是布", 不引入错误先验.

估计器是启发式的, 结果只作为弱监督/预训练目标; 真实掩码一旦提供即可直接使用.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from ..ops import binary_close, binary_dilate, binary_open, box_filter, connected_components, local_std


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    prob = hist / total
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega) + 1e-8
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b))
    return float(centers[idx])



def _otsu_separability(values: np.ndarray, threshold: float) -> float:
    """类间方差 / 总方差, 衡量双峰程度 (0~1, 越大越像"前景+背景"两段)."""
    total_var = float(values.var())
    if total_var <= 1e-12:
        return 0.0
    fg = values[values > threshold]
    bg = values[values <= threshold]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    w_fg = fg.size / values.size
    between = w_fg * (1 - w_fg) * (float(fg.mean()) - float(bg.mean())) ** 2
    return float(between / total_var)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """把不与图像边界连通的背景洞填成前景."""
    labels, num = connected_components(mask == 0, connectivity=4)
    if num == 0:
        return mask
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    border.discard(0)
    keep_bg = np.isin(labels, list(border))
    return (~keep_bg).astype(np.uint8)


def estimate_cloth_mask(
    image: np.ndarray,
    min_coverage: float = 0.05,
    max_coverage: float = 0.995,
    edge_prior: bool = True,
) -> Tuple[np.ndarray, float]:
    """估计布面掩码 (1=布面, 0=背景).

    判据: 布面 = 高局部纹理能量 / 有颜色 / 与画面边缘背景色不同.
    只有当低分区域 (疑似背景) 形成足够大的、与画面边框相连的连通块, 且其
    纹理能量显著低于布面时才认为 "存在背景", 否则整幅都当作布面, 避免给
    网络引入错误的背景先验.

    Returns:
        (mask, confidence): confidence 越高表示分割越可信.
    """
    if image.ndim == 2:
        gray = image.astype(np.float32)
        color = np.stack([gray] * 3, axis=-1)
    else:
        color = image[..., :3].astype(np.float32)
        gray = 0.299 * color[..., 0] + 0.587 * color[..., 1] + 0.114 * color[..., 2]
    h, w = gray.shape

    # 1) 纹理能量 (局部标准差) — 布面有经纬纹理, 背景/辊筒更平滑
    tex = local_std(gray, radius=max(3, min(h, w) // 128))
    tex = tex / (np.percentile(tex, 99) + 1e-6)

    # 2) 饱和度 (彩色布 vs 灰度背景)
    if color.shape[-1] == 3:
        mx = color.max(axis=-1)
        mn = color.min(axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
    else:
        sat = np.zeros_like(gray)

    # 3) 与画面边缘背景色的差异
    border = max(2, min(h, w) // 64)
    border_pixels = np.concatenate(
        [
            color[:border].reshape(-1, 3),
            color[-border:].reshape(-1, 3),
            color[:, :border].reshape(-1, 3),
            color[:, -border:].reshape(-1, 3),
        ]
    )
    bg_color = np.median(border_pixels, axis=0)
    color_dist = np.linalg.norm(color - bg_color[None, None, :], axis=-1)
    color_dist = color_dist / (np.percentile(color_dist, 99) + 1e-6)

    score = 0.65 * tex + 0.15 * sat + (0.35 * color_dist if edge_prior else 0.0)
    lo, hi = np.percentile(score, 5), np.percentile(score, 95)
    score = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    # 平滑: 织造纹理会造成周期性高低起伏, 必须先平滑再分割
    smooth = box_filter(score, max(2, min(h, w) // 32))

    thr = otsu_threshold(smooth)
    candidate = (smooth > thr).astype(np.uint8)
    # 背景 = 低分区域; 只有背景足够大且确实 "低纹理" 才成立
    bg_ratio = 1.0 - float(candidate.mean())
    fg_mean = float(smooth[candidate > 0].mean()) if candidate.any() else 0.0
    bg_mean = float(smooth[candidate == 0].mean()) if (candidate == 0).any() else 0.0
    confidence = _otsu_separability(smooth, thr)

    has_background = (
        bg_ratio > 0.10
        and bg_mean < 0.65 * max(fg_mean, 1e-6)
        and confidence > 0.35
    )
    if not has_background:
        return np.ones((h, w), dtype=np.uint8), 0.2

    mask = candidate
    mask = binary_open(mask, radius=max(1, min(h, w) // 200))
    mask = binary_close(mask, radius=max(2, min(h, w) // 100))
    mask = _fill_holes(mask)

    labels, num = connected_components(mask)
    if num > 0:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        keep = np.flatnonzero(areas > max(32, 0.01 * h * w))
        if keep.size:
            mask = np.isin(labels, keep).astype(np.uint8)
            mask = _fill_holes(mask)

    coverage = float(mask.mean())
    if coverage < min_coverage or coverage > max_coverage or coverage < 1.0 - bg_ratio - 0.3:
        return np.ones((h, w), dtype=np.uint8), min(confidence, 0.2)
    return mask, confidence


def cloth_mask_from_boxes(
    boxes: np.ndarray,
    size: Tuple[int, int],
    dilate: int = 8,
) -> np.ndarray:
    """无掩码时的兜底: 以标注框外扩作为 "布面" 弱先验 (仅用于有疵点的图)."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box[1:5]
        ix1, ix2 = int(x1 * w), int(np.ceil(x2 * w))
        iy1, iy2 = int(y1 * h), int(np.ceil(y2 * h))
        mask[max(0, iy1) : min(h, iy2), max(0, ix1) : min(w, ix2)] = 1
    if dilate > 0 and mask.any():
        mask = binary_dilate(mask, dilate)
    return mask
