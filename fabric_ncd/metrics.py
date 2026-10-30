"""评估指标 (纯 numpy 实现, 无 sklearn 依赖).

* 像素级: AUROC / PRO / F1 (MVTec 协议)
* 图像级: AUROC / AP (正常 vs 异常)
* 检测级: 各类 AP / mAP@0.5 (由异常区域转框)
* 分类级: Acc / macro-F1 / 混淆矩阵
* 聚类 (新类发现): NMI / ARI / cluster accuracy (匈牙利匹配)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ops import bbox_iou, connected_components
from .ncd import hungarian


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # 处理并列: 取平均秩
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    auc = (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 秩相关 (并列取平均秩, 无 scipy 依赖). 任一轴为常数时返回 0."""
    ra = _rankdata(np.asarray(a, dtype=np.float64).ravel())
    rb = _rankdata(np.asarray(b, dtype=np.float64).ravel())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def grouped_auroc(
    y_true: Sequence[float], y_score: Sequence[float], groups: Sequence[str]
) -> Tuple[float, int]:
    """同组内 AUROC (按正负样本对数量加权平均), 只用同时含两类的组.

    跨组比较会被"组"本身的差异污染 (不同布料批次/机台/曝光): 分数可能完美区分
    "这是哪一批布", 而同组内一点都分不开. 这个指标把组间偏移去掉, 只看同组内
    是否分得开. 返回 (AUROC, 有效对数); 没有任何组同时含两类时返回 (nan, 0).
    """
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    total_w = 0.0
    acc = 0.0
    for name in np.unique(groups):
        sel = groups == name
        lab, val = y_true[sel], y_score[sel]
        if lab.min() == lab.max():
            continue
        auc = auroc(lab, val)
        if not np.isfinite(auc):
            continue
        w = int(lab.sum()) * int((~lab).sum())
        acc += auc * w
        total_w += w
    if total_w <= 0:
        return float("nan"), 0
    return acc / total_w, int(total_w)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y_true.sum()
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def _subsample(y_true: np.ndarray, y_score: np.ndarray, max_pixels: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y_true)
    if n <= max_pixels:
        return y_true, y_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pixels, replace=False)
    return y_true[idx], y_score[idx]


def pixel_auroc(
    scores: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    max_pixels: int = 1_000_000,
) -> float:
    y_true = np.concatenate([g.ravel() > 0.5 for g in gts])
    y_score = np.concatenate([s.ravel() for s in scores])
    y_true, y_score = _subsample(y_true, y_score, max_pixels)
    return auroc(y_true, y_score)


def pixel_f1(scores: Sequence[np.ndarray], gts: Sequence[np.ndarray], threshold: float = 0.5) -> Dict[str, float]:
    pred = np.concatenate([(s.ravel() > threshold) for s in scores])
    gt = np.concatenate([(g.ravel() > 0.5) for g in gts])
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    prec = tp / max(tp + fp, 1e-8)
    rec = tp / max(tp + fn, 1e-8)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1}


def per_region_overlap(
    anomaly_scores: Sequence[np.ndarray],
    anomaly_gts: Sequence[np.ndarray],
    normal_scores: Sequence[np.ndarray],
    num_thresholds: int = 100,
) -> float:
    """PRO (Per-Region Overlap) — MVTec 官方协议.

    对每个真实疵点连通域单独计算重合率 (大疵点不会淹没小疵点), 再对假正率
    积分得到曲线下面积.
    """
    if not anomaly_scores or not normal_scores:
        return float("nan")
    all_scores = np.concatenate([s.ravel() for s in anomaly_scores])
    thresholds = np.quantile(all_scores, np.linspace(0, 1, num_thresholds))
    thresholds = np.unique(thresholds)

    # 预先切出每个真实区域
    regions: List[np.ndarray] = []
    region_scores: List[np.ndarray] = []
    for score, gt in zip(anomaly_scores, anomaly_gts):
        binary = (gt > 0.5).astype(np.uint8)
        labels, num = connected_components(binary)
        for c in range(1, num + 1):
            m = labels == c
            if m.sum() < 4:
                continue
            regions.append(m)
            region_scores.append(score[m])
    if not regions:
        return float("nan")

    overlaps = np.zeros(len(thresholds))
    fprs = np.zeros(len(thresholds))
    normal_pixels = np.concatenate([s.ravel() for s in normal_scores])
    for i, thr in enumerate(thresholds):
        overlaps[i] = np.mean([(s > thr).mean() for s in region_scores])
        fprs[i] = float((normal_pixels > thr).mean())
    order = np.argsort(fprs)
    fprs, overlaps = fprs[order], overlaps[order]
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid is None:  # pragma: no cover
        return float(np.sum(np.diff(fprs) * (overlaps[1:] + overlaps[:-1]) / 2.0))
    return float(trapezoid(overlaps, fprs))


# --------------------------------------------------------------------------- #
# 检测 / 分类
# --------------------------------------------------------------------------- #
def box_ap(
    predictions: Sequence[Dict],
    ground_truth: Sequence[Dict],
    iou_thr: float = 0.5,
    num_classes: int = 19,
) -> Tuple[np.ndarray, float]:
    """VOC 风格 AP@IoU. 每个元素: {"boxes": (N,4), "scores": (N,), "labels": (N,)}."""
    aps = np.full(num_classes, np.nan)
    for cls in range(num_classes):
        gts = [g for g in ground_truth if cls in g["labels"].tolist()]
        npos = sum(int((g["labels"] == cls).sum()) for g in gts)
        if npos == 0:
            continue
        records = []
        for img_idx, pred in enumerate(predictions):
            mask = pred["labels"] == cls
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                records.append((float(score), img_idx, box))
        if not records:
            aps[cls] = 0.0
            continue
        records.sort(key=lambda r: -r[0])
        gt_boxes = [
            [b for b, l in zip(g["boxes"], g["labels"]) if int(l) == cls] for g in ground_truth
        ]
        used = [np.zeros(len(b), dtype=bool) for b in gt_boxes]
        tp = np.zeros(len(records))
        fp = np.zeros(len(records))
        for i, (_, img_idx, box) in enumerate(records):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes[img_idx]):
                if used[img_idx][j]:
                    continue
                iou = bbox_iou(box, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                tp[i] = 1
                used[img_idx][best_j] = True
            else:
                fp[i] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(npos, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precision, recall):
            ap += p * (r - prev_r)
            prev_r = r
        aps[cls] = float(ap)
    valid = ~np.isnan(aps)
    return aps, float(np.nanmean(aps[valid])) if valid.any() else float("nan")



def box_ap_agnostic(
    predictions: Sequence[Dict],
    ground_truth: Sequence[Dict],
    iou_thr: float = 0.5,
) -> float:
    """类无关 AP@IoU: 只评估 "定位" 能力, 不受分类正确与否影响."""
    gts = []
    for g in ground_truth:
        labels = g["labels"]
        gts.append({"boxes": g["boxes"], "labels": np.zeros(len(labels), dtype=np.int64)})
    preds = []
    for p in predictions:
        preds.append(
            {
                "boxes": p["boxes"],
                "scores": p["scores"],
                "labels": np.zeros(len(p["labels"]), dtype=np.int64),
            }
        )
    _, ap = box_ap(preds, gts, iou_thr, num_classes=1)
    return ap


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {"acc": float("nan"), "f1_macro": float("nan")}
    cm = confusion_matrix(y_true, y_pred, num_classes)
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return {"acc": acc, "f1_macro": float(np.mean(f1s)) if f1s else float("nan")}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


# --------------------------------------------------------------------------- #
# 聚类
# --------------------------------------------------------------------------- #
def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    u_true = np.unique(labels_true)
    u_pred = np.unique(labels_pred)
    table = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    true_idx = {v: i for i, v in enumerate(u_true)}
    pred_idx = {v: i for i, v in enumerate(u_pred)}
    for t, p in zip(labels_true, labels_pred):
        table[true_idx[t], pred_idx[p]] += 1
    return table


def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    n = table.sum()
    if n == 0:
        return float("nan")
    pij = table / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(pij * np.log(pij / (pi * pj)))
        hi = -np.nansum(pi * np.log(pi))
        hj = -np.nansum(pj * np.log(pj))
    denom = np.sqrt(hi * hj)
    return float(mi / denom) if denom > 0 else 0.0


def ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    n = table.sum()

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    expected = sum_i * sum_j / max(comb2(n), 1e-8)
    max_index = 0.5 * (sum_i + sum_j)
    denom = max_index - expected
    return float((sum_ij - expected) / denom) if abs(denom) > 1e-12 else 0.0


def cluster_acc(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    u_true, t_idx = np.unique(labels_true, return_inverse=True)
    u_pred, p_idx = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(u_true), len(u_pred)), dtype=np.float64)
    np.add.at(table, (t_idx, p_idx), 1)
    pairs = hungarian(-table)
    correct = sum(table[r, c] for r, c in pairs)
    return float(correct / len(labels_true)) if len(labels_true) else float("nan")


# --------------------------------------------------------------------------- #
# 汇总器
# --------------------------------------------------------------------------- #
class MetricAccumulator:
    """在验证集上累积像素级 / 图像级 / 区域级结果."""

    def __init__(self, num_classes: int = 19, names: Optional[Sequence[str]] = None) -> None:
        self.num_classes = num_classes
        self.names = list(names) if names else [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.pixel_scores: List[np.ndarray] = []
        self.pixel_gts: List[np.ndarray] = []
        self.normal_scores: List[np.ndarray] = []
        self.image_scores: List[float] = []
        self.image_labels: List[int] = []  # 0 = 正常, 1 = 异常
        self.multi_true: List[int] = []
        self.multi_pred: List[int] = []
        self.pred_boxes: List[Dict] = []
        self.gt_boxes: List[Dict] = []
        self.regions: List[Dict] = []
        self.pixel_threshold: float = 0.5

    def update(
        self,
        anomaly_score: np.ndarray,
        anomaly_gt: np.ndarray,
        normal: bool,
        image_score: float,
        pred: Optional[Dict] = None,
        gt: Optional[Dict] = None,
        multi_true: int = -1,
        multi_pred: int = -1,
        regions: Optional[List[Dict]] = None,
    ) -> None:
        score = anomaly_score.astype(np.float32)
        gt_mask = (anomaly_gt > 0.5).astype(np.float32)
        self.image_scores.append(float(image_score))
        self.image_labels.append(0 if normal else 1)
        if normal:
            self.normal_scores.append(score)
            self.pixel_scores.append(score)
            self.pixel_gts.append(gt_mask)
        else:
            self.pixel_scores.append(score)
            self.pixel_gts.append(gt_mask)
            if multi_true >= 0:
                self.multi_true.append(int(multi_true))
                self.multi_pred.append(int(multi_pred))
        if pred is not None and gt is not None:
            self.pred_boxes.append(pred)
            self.gt_boxes.append(gt)
        if regions:
            self.regions.extend(regions)

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.image_scores:
            out["image_auroc"] = auroc(np.asarray(self.image_labels), np.asarray(self.image_scores))
            out["image_ap"] = average_precision(np.asarray(self.image_labels), np.asarray(self.image_scores))
        if self.pixel_scores:
            out["pixel_auroc"] = pixel_auroc(self.pixel_scores, self.pixel_gts)
            out.update(
                {
                    f"pixel_{k}": v
                    for k, v in pixel_f1(self.pixel_scores, self.pixel_gts, self.pixel_threshold).items()
                }
            )
            if self.normal_scores:
                out["pro"] = per_region_overlap(self.pixel_scores, self.pixel_gts, self.normal_scores)
        if self.multi_true:
            out.update(classification_report(self.multi_true, self.multi_pred, self.num_classes))
        if self.pred_boxes and self.gt_boxes:
            _, map50 = box_ap(self.pred_boxes, self.gt_boxes, 0.5, self.num_classes)
            out["map50"] = map50
            out["map50_agnostic"] = box_ap_agnostic(self.pred_boxes, self.gt_boxes, 0.5)
        return {k: float(v) for k, v in out.items() if not (isinstance(v, float) and np.isnan(v))}
