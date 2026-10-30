"""不依赖 OpenCV / SciPy 的基础图像算子.

布匹检测的后处理只需要连通域、形态学、区域属性这几样; 这里用 numpy + PIL
自己实现, 避免给产线环境引入额外依赖. 若有 OpenCV 会自动优先使用 (更快),
但结果与纯 numpy 实现保持一致.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

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
    """返回 (labels, num), labels 中 0 表示背景, 1..num 为连通域 (按出现顺序)."""
    binary = np.ascontiguousarray(binary.astype(bool))
    if binary.ndim != 2:
        raise ValueError("connected_components 仅支持二维数组")
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32), 0
    if _HAS_CV2:  # pragma: no cover
        num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=connectivity)
        return labels.astype(np.int32), num - 1

    h, w = binary.shape
    run_rows: List[List[Tuple[int, int]]] = [_row_runs(binary[y]) for y in range(h)]
    total_runs = sum(len(r) for r in run_rows)
    uf = _UnionFind(total_runs)

    # 给每个 run 编号
    offsets = np.cumsum([0] + [len(r) for r in run_rows])
    for y in range(1, h):
        prev, cur = run_rows[y - 1], run_rows[y]
        i = j = 0
        while i < len(prev) and j < len(cur):
            ps, pe = prev[i]
            cs, ce = cur[j]
            # x 方向邻接: 8 邻接允许 1 像素错位, 4 邻接要求严格重叠
            tol = 1 if connectivity == 8 else 0
            if ps - tol < ce and cs - tol < pe:
                uf.union(offsets[y - 1] + i, offsets[y] + j)
            if pe < ce:
                i += 1
            else:
                j += 1

    labels = np.zeros((h, w), dtype=np.int32)
    root_to_label = {}
    for y in range(h):
        for idx, (s, e) in enumerate(run_rows[y]):
            root = uf.find(offsets[y] + idx)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label) + 1
            labels[y, s:e] = root_to_label[root]
    return labels, len(root_to_label)


# --------------------------------------------------------------------------- #
# 形态学
# --------------------------------------------------------------------------- #
def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    if _HAS_CV2:  # pragma: no cover
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    from PIL import Image, ImageFilter

    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return (np.asarray(img) > 127).astype(np.uint8)


def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    if _HAS_CV2:  # pragma: no cover
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    from PIL import Image, ImageFilter

    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MinFilter(radius * 2 + 1))
    return (np.asarray(img) > 127).astype(np.uint8)


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


def topk_mean(values: np.ndarray, k_ratio: float = 0.01, min_k: int = 16) -> float:
    """图像级异常分数: 取最高 k% 像素的均值 (抗噪, 优于 max)."""
    flat = values.ravel()
    if flat.size == 0:
        return 0.0
    k = max(min_k, int(flat.size * k_ratio))
    k = min(k, flat.size)
    part = np.partition(flat, -k)[-k:]
    return float(part.mean())


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
    """O(1) 均值滤波 (积分图)."""
    r = max(1, int(radius))
    integ = _integral(x.astype(np.float64))
    h, w = x.shape
    ys = np.clip(np.arange(h) + r + 1, 0, h)
    ys0 = np.clip(np.arange(h) - r, 0, h)
    xs = np.clip(np.arange(w) + r + 1, 0, w)
    xs0 = np.clip(np.arange(w) - r, 0, w)
    total = (
        integ[np.ix_(ys, xs)]
        - integ[np.ix_(ys0, xs)]
        - integ[np.ix_(ys, xs0)]
        + integ[np.ix_(ys0, xs0)]
    )
    count = (ys - ys0)[:, None] * (xs - xs0)[None, :]
    return (total / np.maximum(count, 1)).astype(np.float32)


def local_std(gray: np.ndarray, radius: int = 4) -> np.ndarray:
    mean = box_filter(gray, radius)
    mean_sq = box_filter(gray * gray, radius)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
