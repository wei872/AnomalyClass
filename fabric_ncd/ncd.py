"""区域分析、区域合并与新类发现 (AnomalyNCD 思路的工程化实现).

与 AnomalyNCD 的对应关系:

======================  ==================================================
AnomalyNCD 组件        本实现
======================  ==================================================
主元素二值化 (MEBin)    ``RegionAnalyzer`` 里的自适应阈值 + 最大连通域选择,
                        保证 "以异常为中心" 再送入分类/聚类
掩码引导表征学习        ``losses.pooled_region_embeddings`` (掩码平均池化)
                        + SupCon
区域合并 (region merge) ``merge_region_scores`` 支持面积加权 / top-1 / max
新类发现                ``discover_classes`` 余弦 k-means + 匈牙利匹配 +
                        相似度阈值判定新类
======================  ==================================================

所有功能都只用 numpy, 可在 CPU 上以毫秒级完成, 不增加模型体积.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ops import (
    Region,
    binary_close,
    binary_open,
    box_filter,
    extract_regions,
    topk_mean,
)


# --------------------------------------------------------------------------- #
# 区域分析
# --------------------------------------------------------------------------- #
@dataclass
class RegionInfo:
    bbox: Tuple[float, float, float, float]  # 归一化 xyxy
    score: float
    cls: int = -1
    cls_score: float = 0.0
    probs: Optional[np.ndarray] = None
    area: float = 0.0

    def as_dict(self, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        name = names[self.cls] if names is not None and 0 <= self.cls < len(names) else str(self.cls)
        return {
            "bbox": [round(float(v), 5) for v in self.bbox],
            "score": round(float(self.score), 5),
            "cls": int(self.cls),
            "name": name,
            "cls_score": round(float(self.cls_score), 5),
            "area_ratio": round(float(self.area), 6),
        }


def auto_threshold(score: np.ndarray, cloth: Optional[np.ndarray] = None, sigma: float = 3.0) -> float:
    values = score[cloth > 0.5] if cloth is not None and (cloth > 0.5).any() else score.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    mean, std = float(values.mean()), float(values.std())
    q = float(np.percentile(values, 99.5))
    thr = max(mean + sigma * std, q)
    return float(np.clip(thr, 0.15, 0.95))


class RegionAnalyzer:
    def __init__(
        self,
        names: Optional[Sequence[str]] = None,
        min_area_ratio: float = 4e-4,
        smooth: int = 3,
        threshold: Optional[float] = None,
        sigma: float = 3.0,
        merge_strategy: str = "area_weighted",
        topk_ratio: float = 0.01,
        peak_ratio: float = 0.35,
        open_radius: int = 1,
        bbox_refine_ratio: float = 0.7,
    ) -> None:
        self.names = list(names) if names is not None else None
        self.min_area_ratio = min_area_ratio
        self.smooth = smooth
        self.threshold = threshold
        self.sigma = sigma
        self.merge_strategy = merge_strategy
        self.topk_ratio = topk_ratio
        self.peak_ratio = peak_ratio
        self.open_radius = open_radius
        self.bbox_refine_ratio = bbox_refine_ratio

    def analyze(
        self,
        score: np.ndarray,
        cloth: np.ndarray,
        class_probs: Optional[np.ndarray] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[RegionInfo], Dict[str, Any]]:
        """score/cloth: (h,w); class_probs: (C,h,w). input_size: (H,W) 用于归一化 bbox."""
        score = np.asarray(score, dtype=np.float32)
        cloth = np.asarray(cloth, dtype=np.float32)
        if self.smooth > 1:
            score = box_filter(score, self.smooth // 2)
        masked = score * (cloth > 0.5)

        thr = self.threshold if self.threshold is not None else auto_threshold(masked, cloth, self.sigma)
        binary = ((masked > thr) & (cloth > 0.5)).astype(np.uint8)
        binary = binary_close(binary_open(binary, self.open_radius), self.open_radius)

        h, w = score.shape
        min_area = max(4, int(self.min_area_ratio * h * w))
        raw_regions: List[Region] = extract_regions(masked, binary, min_area=min_area)
        # 主元素二值化 (MEBin 思路): 只保留峰值分数接近图像最大峰值的连通域,
        # 抑制阈值附近大面积低置信区域 -> 区域更紧凑, 定位精度更高.
        if raw_regions and self.peak_ratio > 0:
            top = max(r.score for r in raw_regions)
            cutoff = thr + self.peak_ratio * max(top - thr, 0.0)
            raw_regions = [r for r in raw_regions if r.score >= cutoff]
        infos: List[RegionInfo] = []
        for region in raw_regions:
            x1, y1, x2, y2 = region.bbox
            # 用高分核心收缩 bbox: 连通域常常包含阈值附近的大片低置信区域,
            # 收缩后定位更准 (尤其对细长疵点).
            sub_mask = binary[y1:y2, x1:x2] > 0
            if self.bbox_refine_ratio > 0:
                sub_scores = masked[y1:y2, x1:x2]
                # 沿 "阈值 -> 峰值" 的方向取一个更紧的核心区域
                core_thr = thr + self.bbox_refine_ratio * max(float(region.score) - thr, 0.0)
                core = sub_scores >= core_thr
                if core.sum() >= max(4, 0.05 * sub_mask.sum()):
                    ys, xs = np.nonzero(core)
                    ry1, ry2 = y1 + int(ys.min()), y1 + int(ys.max()) + 1
                    rx1, rx2 = x1 + int(xs.min()), x1 + int(xs.max()) + 1
                    x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                    sub_mask = core[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
            probs = None
            cls, cls_score = -1, 0.0
            if class_probs is not None:
                patch = class_probs[:, y1:y2, x1:x2]
                if sub_mask.any():
                    probs = patch[:, sub_mask].mean(axis=1)
                else:
                    probs = patch.reshape(patch.shape[0], -1).mean(axis=1)
                cls = int(np.argmax(probs))
                cls_score = float(probs[cls])
            area = float(sub_mask.sum()) / (h * w) if sub_mask.size else float(region.area) / (h * w)
            infos.append(
                RegionInfo(
                    bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                    score=float(masked[y1:y2, x1:x2].max()) if (y2 > y1 and x2 > x1) else float(region.score),
                    cls=cls,
                    cls_score=cls_score,
                    probs=probs,
                    area=area,
                )
            )

        image_level = {
            "anomaly_score": topk_mean(masked, self.topk_ratio),
            "threshold": float(thr),
            "num_regions": len(infos),
            "probs": self.merge_region_scores(infos),
        }
        return infos, image_level

    def merge_region_scores(self, regions: Sequence[RegionInfo]) -> Optional[np.ndarray]:
        """区域合并: 由区域级分类结果得到图像级类别分布."""
        if not regions:
            return None
        probs = [r.probs for r in regions if r.probs is not None]
        if not probs:
            return None
        stack = np.stack(probs)
        if self.merge_strategy == "max":
            merged = stack.max(axis=0)
        elif self.merge_strategy == "top1":
            merged = stack[int(np.argmax([r.score for r in regions]))]
        else:  # area_weighted
            weights = np.asarray([max(r.area, 1e-6) for r in regions], dtype=np.float32)
            weights = weights / weights.sum()
            merged = (stack * weights[:, None]).sum(axis=0)
        total = merged.sum()
        return merged / total if total > 0 else merged


def postprocess_outputs(
    outputs: Dict[str, Any],
    analyzer: RegionAnalyzer,
    input_size: Optional[Tuple[int, int]] = None,
) -> List[Tuple[List[RegionInfo], Dict[str, Any]]]:
    """把模型 batch 输出转成每个样本的区域列表 + 图像级结果 (numpy)."""
    score = outputs["anomaly_score"].detach().cpu().numpy()
    cloth = outputs["cloth_prob"].detach().cpu().numpy()
    class_logits = outputs["class_logits"].detach().cpu().numpy() if outputs.get("class_logits") is not None else None
    results = []
    for i in range(score.shape[0]):
        s = score[i, 0]
        # 布面掩码上采样到异常图尺度
        c = _resize_np(cloth[i, 0], s.shape)
        p = None
        if class_logits is not None:
            logits = class_logits[i]
            logits = _resize_np(logits, s.shape, channels_first=True)
            p = _softmax(logits, axis=0)
        infos, image_level = analyzer.analyze(s, c, p, input_size)
        results.append((infos, image_level))
    return results


def _resize_np(x: np.ndarray, size: Tuple[int, int], channels_first: bool = False) -> np.ndarray:
    from PIL import Image

    h, w = size
    if channels_first:
        return np.stack(
            [
                np.asarray(Image.fromarray(ch.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
                for ch in x
            ]
        )
    return np.asarray(Image.fromarray(x.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- #
# 聚类 / 新类发现
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8
    return x / norm


def kmeans_cosine(
    features: np.ndarray,
    k: int,
    iters: int = 50,
    seed: int = 0,
    n_init: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """余弦 k-means (球面 k-means), 返回 (labels, centroids, inertia)."""
    features = l2_normalize(np.asarray(features, dtype=np.float32))
    n = features.shape[0]
    k = int(min(max(1, k), n))
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(max(1, n_init)):
        idx = rng.choice(n, size=k, replace=False)
        centroids = features[idx].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _it in range(iters):
            sim = features @ centroids.T
            new_labels = sim.argmax(axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                labels = new_labels
                break
            labels = new_labels
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                members = features[labels == c]
                if len(members) == 0:
                    new_centroids[c] = features[rng.integers(0, n)]
                else:
                    new_centroids[c] = members.mean(axis=0)
            centroids = l2_normalize(new_centroids)
        inertia = float((1.0 - (features * centroids[labels]).sum(axis=1)).mean())
        if inertia < best[2]:
            best = (labels, centroids, inertia)
    return best


def silhouette_cosine(features: np.ndarray, labels: np.ndarray) -> float:
    features = l2_normalize(np.asarray(features, dtype=np.float32))
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return -1.0
    sim = features @ features.T
    dist = 1.0 - sim
    scores = []
    for i in range(len(features)):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            scores.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        scores.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(scores))


def select_k(
    features: np.ndarray,
    k_min: int = 2,
    k_max: int = 12,
    seed: int = 0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """在 [k_min, k_max] 上按轮廓系数选 k."""
    n = len(features)
    k_max = int(min(k_max, max(2, n - 1)))
    best = (-2.0, None, None, 2)
    for k in range(k_min, k_max + 1):
        labels, centers, _ = kmeans_cosine(features, k, seed=seed)
        score = silhouette_cosine(features, labels)
        if score > best[0]:
            best = (score, labels, centers, k)
    if best[1] is None:
        labels, centers, _ = kmeans_cosine(features, 2, seed=seed)
        return 2, labels, centers
    return best[3], best[1], best[2]


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """匈牙利算法 (最小代价匹配), 返回匹配对 (row, col)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def discover_classes(
    features: np.ndarray,
    known_prototypes: Optional[np.ndarray],
    known_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 12,
    novelty_threshold: float = 0.55,
    seed: int = 0,
) -> Dict[str, Any]:
    """对无标注疵点区域做聚类, 并与已知原型匹配, 未匹配的簇视为新类.

    Returns:
        dict(cluster_labels, centroids, assignments, novel_indices, novel_prototypes)
    """
    features = l2_normalize(np.asarray(features, dtype=np.float32))
    if k is None:
        k, labels, centroids = select_k(features, k_min, k_max, seed=seed)
    else:
        labels, centroids, _ = kmeans_cosine(features, k, seed=seed)
    assignments: Dict[int, Dict[str, Any]] = {}
    novel: List[int] = []

    if known_prototypes is not None and len(known_prototypes) > 0:
        known = l2_normalize(np.asarray(known_prototypes, dtype=np.float32))
        sim = centroids @ known.T  # (k, C)
        pairs = hungarian(-sim)
        matched_clusters = {c: (r, float(sim[c, r])) for c, r in pairs}
        for c in range(len(centroids)):
            if c in matched_clusters:
                r, score = matched_clusters[c]
                is_novel = score < novelty_threshold
                assignments[c] = {
                    "known_index": int(r),
                    "name": known_names[r] if known_names is not None and r < len(known_names) else str(r),
                    "similarity": score,
                    "novel": bool(is_novel),
                }
                if is_novel:
                    novel.append(c)
            else:
                assignments[c] = {"known_index": -1, "name": f"novel_{c}", "similarity": 0.0, "novel": True}
                novel.append(c)
    else:
        for c in range(len(centroids)):
            assignments[c] = {"known_index": -1, "name": f"novel_{c}", "similarity": 0.0, "novel": True}
            novel.append(c)

    return {
        "k": int(len(centroids)),
        "cluster_labels": labels,
        "centroids": centroids,
        "assignments": assignments,
        "novel_indices": sorted(set(novel)),
        "novel_prototypes": centroids[sorted(set(novel))] if novel else np.zeros((0, centroids.shape[1])),
    }
