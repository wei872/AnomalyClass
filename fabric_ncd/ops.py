"""不依赖 OpenCV / SciPy 的基础图像算子.

布匹检测的后处理只需要连通域、形态学、区域属性这几样; 这里用 numpy + PIL
自己实现, 避免给产线环境引入额外依赖. 若有 OpenCV 会自动优先使用 (更快),
但结果与纯 numpy 实现保持一致.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - 可选加速
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# 连通域
# --------------------------------------------------------------------------- #
def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def connected_components(binary: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """返回 (labels, num), labels 中 0 表示背景, 1..num 为连通域 (按出现顺序).

    实现: 游程编码 + 并查集. 游程提取完全向量化 (一次 diff), 只对"有游程的行"
    做相邻行合并, 因此稀疏掩码下开销接近 O(游程数).
    """
    binary = np.ascontiguousarray(binary.astype(bool))
    if binary.ndim != 2:
        raise ValueError("connected_components 仅支持二维数组")
    labels = np.zeros(binary.shape, dtype=np.int32)
    if not binary.any():
        return labels, 0
    if _HAS_CV2:  # pragma: no cover
        num, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=connectivity)
        return lab.astype(np.int32), num - 1

    h, w = binary.shape
    # 逐行游程提取 (向量化): 在每行右侧补 0, 用 diff 找 1 的上升沿/下降沿.
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1 : w + 1] = binary
    diff = np.diff(padded, axis=1)
    starts_yx = np.argwhere(diff == 1)
    ends_yx = np.argwhere(diff == -1)
    n_runs = starts_yx.shape[0]
    if n_runs == 0:
        return labels, 0
    cols_s = starts_yx[:, 1].astype(np.int64)   # 行内起点 (含)
    cols_e = ends_yx[:, 1].astype(np.int64)     # 行内终点 (不含)
    rows = starts_yx[:, 0]
    starts = cols_s + rows.astype(np.int64) * w  # 展平下标 (写 labels 用)
    ends = cols_e + rows.astype(np.int64) * w

    counts = np.bincount(rows, minlength=h)
    offsets = np.zeros(h + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    row_ids = np.nonzero(counts)[0]

    uf = _UnionFind(n_runs)
    tol = 1 if connectivity == 8 else 0
    for y in row_ids:
        if y == 0:
            continue
        if counts[y - 1] == 0:
            continue
        i, j = int(offsets[y - 1]), int(offsets[y])
        i_end, j_end = int(offsets[y]), int(offsets[y + 1])
        while i < i_end and j < j_end:
            ps, pe = cols_s[i], cols_e[i]
            cs, ce = cols_s[j], cols_e[j]
            if ps - tol < ce and cs - tol < pe:
                uf.union(i, j)
            if pe < ce:
                i += 1
            else:
                j += 1

    roots = np.fromiter((uf.find(k) for k in range(n_runs)), dtype=np.int64, count=n_runs)
    _, run_labels = np.unique(roots, return_inverse=True)
    run_labels = run_labels.astype(np.int32) + 1

    flat_labels = labels.reshape(-1)
    for k in range(n_runs):
        flat_labels[starts[k] : ends[k]] = run_labels[k]
    return labels, int(run_labels.max()) if n_runs else 0


# --------------------------------------------------------------------------- #
# 形态学
# --------------------------------------------------------------------------- #
def _shift_reduce(mask: np.ndarray, radius: int, op: str) -> np.ndarray:
    """可分离的平方结构元形态学: 先沿 x 再做 y, 共 2*(2r+1) 次平移.

    对 610x360 的掩码比 PIL 的 MinFilter/MaxFilter 快 20 倍以上,
    是实时推理里区域分析的主要加速点.
    """
    h, w = mask.shape
    src = mask.astype(np.uint8)
    fill = 0 if op == "max" else 1
    reduce = np.maximum if op == "max" else np.minimum

    out = src
    for d in range(1, radius + 1):
        shifted = np.full_like(src, fill)
        shifted[:, d:] = src[:, :-d]
        out = reduce(out, shifted)
        shifted = np.full_like(src, fill)
        shifted[:, :-d] = src[:, d:]
        out = reduce(out, shifted)
    tmp = out
    out = tmp
    for d in range(1, radius + 1):
        shifted = np.full_like(tmp, fill)
        shifted[d:, :] = tmp[:-d, :]
        out = reduce(out, shifted)
        shifted = np.full_like(tmp, fill)
        shifted[:-d, :] = tmp[d:, :]
        out = reduce(out, shifted)
    return out


def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    return _shift_reduce(mask, int(radius), "max")


def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    return _shift_reduce(mask, int(radius), "min")


def binary_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius), radius)


def binary_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, radius), radius)


# --------------------------------------------------------------------------- #
# 区域属性
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (右开区间)
    score: float = 0.0
    cls: int = -1
    cls_score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> Tuple[int, int]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]

    def as_dict(self) -> dict:
        return {
            "bbox": [int(v) for v in self.bbox],
            "area": int(self.area),
            "score": float(self.score),
            "cls": int(self.cls),
            "cls_score": float(self.cls_score),
            **self.extra,
        }


def mask_to_region(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def extract_regions(
    score_map: np.ndarray,
    binary: np.ndarray,
    min_area: int = 16,
    max_regions: int = 64,
    connectivity: int = 8,
) -> List[Region]:
    """从二值异常图提取候选区域, 按峰值分数排序."""
    labels, num = connected_components(binary, connectivity=connectivity)
    regions: List[Region] = []
    if num == 0:
        return regions
    flat_labels = labels.ravel()
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(1, num + 2))
    for i in range(num):
        idx = order[boundaries[i] : boundaries[i + 1]]
        if idx.size < min_area:
            continue
        region_mask = np.zeros(binary.shape, dtype=bool)
        region_mask.ravel()[idx] = True
        bbox = mask_to_region(region_mask)
        score = float(score_map[region_mask].max())
        regions.append(Region(label=i + 1, area=int(idx.size), bbox=bbox, score=score))
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[:max_regions]


def topk_mean(
    values: np.ndarray, k_ratio: float = 0.01, min_k: int = 16, k_pixels: int | None = None
) -> float:
    """图像级异常分数: 取最高 k% 像素的均值 (抗噪, 优于 max).

    ``k_pixels`` 优先于 ``k_ratio``: 固定 1% 在整图上意味着"最高的一万多个像素",
    而小疵点往往只占几千像素, 均值会被大量正常像素稀释, 于是疵点图的图像级分数
    反而低于正常图 (imageAUROC < 0.5). 用绝对像素数 (如 256) 才和疵点尺度无关.
    """
    flat = values.ravel()
    if flat.size == 0:
        return 0.0
    if k_pixels:
        k = max(1, int(k_pixels))  # 显式指定像素数时不受 min_k 影响
    else:
        k = max(min_k, int(flat.size * k_ratio))
    k = min(k, flat.size)
    part = np.partition(flat, -k)[-k:]
    return float(part.mean())


def threshold_from_normal_pixels(
    values: np.ndarray, quantile: float = 0.999, sigma_k: float = 6.0
) -> Dict[str, float]:
    """用正常图素级分数分布确定绝对阈值.

    ``max(高分位, 均值 + kσ)`` 里的 σ 项没有上界: 异常分数是 0~1 的概率, 训练后
    正常像素分布变得很尖 (σ 很小) 但均值抬高时, ``均值 + 6σ`` 会超过 1.0, 阈值
    高到**任何像素都不可能超过** -> 验证集区域数恒为 0, mAP/分类准确率一起归零.
    因此用正常像素的最大值兜底, 保证阈值一定落在可达范围内.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"threshold": 0.5, "quantile_thr": 0.5, "sigma_thr": 0.5, "max": 0.5}
    if float(arr.max()) <= 1e-6:
        # 布面掩码还没学出来时分数会被整片置 0, 阈值 0 会让每个像素都变成区域
        return {"threshold": 0.5, "quantile_thr": 0.0, "sigma_thr": 0.0, "max": float(arr.max())}
    q_thr = float(np.quantile(arr, float(quantile)))
    sigma_thr = float(arr.mean() + float(sigma_k) * arr.std())
    vmax = float(arr.max())
    # σ 项越过正常分布的实际最大值时它已经失去意义 (通常是分布有尖峰/长尾),
    # 退回分位数; 直接把阈值顶到最大值等于"任何像素都不算异常", 检出率必然为 0.
    thr = q_thr if sigma_thr >= vmax else max(q_thr, sigma_thr)
    return {
        "threshold": float(min(thr, vmax)),
        "quantile_thr": q_thr,
        "sigma_thr": sigma_thr,
        "max": vmax,
    }


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(regions: Sequence[Region], iou_thr: float = 0.5) -> List[Region]:
    kept: List[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(bbox_iou(region.bbox, k.bbox) < iou_thr for k in kept):
            kept.append(region)
    return kept

# --------------------------------------------------------------------------- #
# 积分图均值滤波
# --------------------------------------------------------------------------- #
def _integral(x: np.ndarray) -> np.ndarray:
    return np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")


def box_filter(x: np.ndarray, radius: int) -> np.ndarray:
    """O(1) 均值滤波 (边界按 edge 复制).

    k <= 9 时用"平移相加": 对 610x360 的分数图比 cumsum 快 2~3 倍;
    更大的核改用可分离 cumsum, 与半径无关.
    """
    r = max(1, int(radius))
    arr = np.asarray(x, dtype=np.float32)
    k = 2 * r + 1
    if k <= 9:
        h, w = arr.shape
        ap = np.pad(arr, ((r, r), (r, r)), mode="edge")
        rows = np.zeros_like(arr)
        for d in range(k):
            rows += ap[d : d + h, r : r + w]
        rp = np.pad(rows, ((0, 0), (r, r)), mode="edge")
        out = np.zeros_like(arr)
        for d in range(k):
            out += rp[:, d : d + w]
        out /= float(k * k)
        return out
    xp = np.pad(arr, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(xp, axis=1)
    sr = c[:, k:] - c[:, :-k]
    c2 = np.cumsum(sr, axis=0)
    out = c2[k:] - c2[:-k]
    return (out / float(k * k)).astype(np.float32)


def local_std(gray: np.ndarray, radius: int = 4) -> np.ndarray:
    mean = box_filter(gray, radius)
    mean_sq = box_filter(gray * gray, radius)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
