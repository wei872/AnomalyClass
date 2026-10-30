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
from PIL import Image
from ..ops import binary_close, binary_dilate, binary_open, box_filter, connected_components, local_std
from .weave import estimate_period


def _period_pixels(gray: np.ndarray) -> Tuple[float, float]:
    """估计织造周期 (该图像自身的像素).

    在 **原始分辨率** 上取几个 512x512 的窗口做估计, 绝不先缩小: 细密布的经纬
    只有几像素, 一旦缩小就会混叠, 周期会被估成完全错误的值 (6 px 能估成 28 px).
    多个窗口取中位数, 避免局部疵点/反光干扰.
    """
    h, w = gray.shape
    size = int(min(512, h, w))
    if size < 64:
        return 0.0, 0.0
    ys = [max(0, (h - size) // 2)]
    xs = [max(0, (w - size) // 2), max(0, (w - size) // 4), max(0, 3 * (w - size) // 4)]
    periods = []
    for y in ys:
        for x in xs:
            patch = gray[y : y + size, x : x + size]
            if float(patch.std()) < 3.0:  # 平坦区 (背景/辊筒) 没有纹理, 不参与周期统计
                continue
            patch = patch.astype(np.uint8)
            try:
                px, py, conf = estimate_period(patch)
            except Exception:  # pragma: no cover
                continue
            if conf >= 2.0:
                periods.append((float(px), float(py)))
    if not periods:
        return 0.0, 0.0
    arr = np.asarray(periods)
    # 取 25 分位而不是中位数: 宁可按"最细的"那片纹理定尺度, 也不要因为某个窗口
    # 落在背景/反光上而把整图降得过分, 导致经纬纹理混叠
    return float(np.percentile(arr[:, 0], 25)), float(np.percentile(arr[:, 1], 25))


def _texture_window(gray: np.ndarray) -> Tuple[int, int]:
    """返回 (纹理能量窗口半径, 平滑窗口半径), 单位为该图像自身的像素."""
    pxs, pys = _period_pixels(gray)
    if pxs <= 0 or pys <= 0:
        base = max(3.0, min(gray.shape) / 64.0)
        return int(np.clip(round(base), 3, 16)), int(np.clip(round(base * 2), 2, 32))
    period = float(np.sqrt(pxs * pys))
    tex_r = int(np.clip(round(period * 0.75), 2, 16))
    smooth_r = int(np.clip(round(period * 2.5), 2, 48))
    return tex_r, smooth_r


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
    work_size: int | str | None = None,
) -> Tuple[np.ndarray, float]:
    """布面掩码估计 (可在大图下降分辨率计算后最近邻回放).

    ``work_size``: 长边缩到这个尺寸再估计 (掩码本身是大块区域, 降分辨率几乎不影响
    位置). 回放时用最近邻, 保持 0/1 不产生灰边.

    取 ``"auto"`` 时按该图实测的织造周期自动决定: 保证缩放后经纬周期仍 >= 3 px.
    周期 6 px 的细密布只能缩到 1/2 (2440 -> 1224), 周期 20 px 的粗布可以缩到 1/6,
    这样既不会把纹理平均掉, 又能在粗纹理布上真正省时间.
    """
    arr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    h, w = arr.shape[:2]
    if isinstance(work_size, str) and work_size.lower() == "auto":
        work_size = auto_work_size(arr)
    if work_size and max(h, w) > int(work_size):
        scale = float(work_size) / float(max(h, w))
        nw, nh = max(64, int(round(w * scale))), max(64, int(round(h * scale)))
        small = np.asarray(
            Image.fromarray(arr[..., :3].astype(np.uint8)).resize((nw, nh), Image.BILINEAR),
            dtype=np.uint8,
        )
        mask_s, conf = _estimate_cloth_mask_core(small, min_coverage, max_coverage, edge_prior)
        if mask_s.shape[:2] != (h, w):
            mask_s = (
                np.asarray(Image.fromarray(mask_s * 255).resize((w, h), Image.NEAREST), dtype=np.uint8) > 127
            ).astype(np.uint8)
        return mask_s, conf
    return _estimate_cloth_mask_core(arr, min_coverage, max_coverage, edge_prior)


def auto_work_size(image: np.ndarray, min_period_px: float = 3.0, floor: int = 512) -> int | None:
    """按织造周期决定掩码估计的降采样尺寸; 返回 None 表示不降采样.

    约束: 缩放后经纬周期不能小于 ``min_period_px`` (否则局部纹理能量被平均掉,
    掩码估计会退化成"全是布"). 因此细密布几乎不能降采样, 粗纹理布可以降很多.
    """
    arr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    pxs, pys = _period_pixels(gray)
    long_side = max(h, w)
    if pxs <= 0 or pys <= 0:
        return None
    period = float(np.sqrt(pxs * pys))
    target = min_period_px * long_side / max(period, 1e-6)
    target = int(round(min(float(long_side), max(float(floor), target))))
    return target if target < long_side * 0.95 else None


def _estimate_cloth_mask_core(
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
    tex_r, smooth_r = _texture_window(gray)
    tex = local_std(gray, radius=tex_r)
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
    # 布面中心区域的代表色: 背景与布面颜色应显著不同 (深色辊筒/台面 vs 浅色布)
    cy0, cy1 = int(0.3 * h), int(0.7 * h) + 1
    cx0, cx1 = int(0.3 * w), int(0.7 * w) + 1
    center_color = np.median(color[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
    color_gap = float(np.linalg.norm(center_color - bg_color)) / 255.0
    color_dist = np.linalg.norm(color - bg_color[None, None, :], axis=-1)
    color_dist = color_dist / (np.percentile(color_dist, 99) + 1e-6)

    score = 0.65 * tex + 0.15 * sat + (0.35 * color_dist if edge_prior else 0.0)
    lo, hi = np.percentile(score, 5), np.percentile(score, 95)
    score = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    # 平滑: 织造纹理会造成周期性高低起伏, 必须先平滑再分割
    smooth = box_filter(score, smooth_r)

    thr = otsu_threshold(smooth)
    candidate = (smooth > thr).astype(np.uint8)
    fg_mean = float(smooth[candidate > 0].mean()) if candidate.any() else 0.0
    bg_mean = float(smooth[candidate == 0].mean()) if (candidate == 0).any() else 0.0
    confidence = _otsu_separability(smooth, thr)

    bg_ratio = 1.0 - float(candidate.mean())
    # 三种线索任意一条足够强即可认为"画面里确实有背景":
    # 1) 边框颜色与布面色差明显; 2) 低分区域的分数显著低于布面; 3) 低分区域占比很大.
    strong = color_gap > 0.06 or bg_mean < 0.40 * max(fg_mean, 1e-6) or bg_ratio > 0.45
    has_background = (
        bg_ratio > 0.10
        and bg_mean < 0.75 * max(fg_mean, 1e-6)
        and confidence > 0.35
        and strong
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
