# -*- coding: utf-8 -*-
"""VOC 风格 mAP 评估 (IoU 0.5 与 0.5:0.95)。未知缺陷预测不计入已知类 AP, 单独统计。"""
from __future__ import annotations

import numpy as np

from .postprocess import UNKNOWN_CLS


def _ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = mrec[1:] != mrec[:-1]
    return float((mrec[1:] - mrec[:-1])[idx] @ mpre[1:][idx])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def evaluate_ap(gt_by_image: dict, pred_by_image: dict, num_classes: int,
                iou_thr: float = 0.5) -> tuple[np.ndarray, float, dict]:
    """gt_by_image: {name: [(cls, xyxy)]}; pred_by_image: {name: [dict(box,score,cls)]}
    返回 (per_class_ap, mAP, extra{tp,fp,fn, unknown_pred})
    """
    tp_cnt = np.zeros(num_classes, dtype=np.int64)
    fp_cnt = np.zeros(num_classes, dtype=np.int64)
    fn_cnt = np.zeros(num_classes, dtype=np.int64)
    unknown_pred = 0
    all_pred: dict[int, list] = {c: [] for c in range(num_classes)}
    for name, gts in gt_by_image.items():
        preds = [p for p in pred_by_image.get(name, []) if p["cls"] != UNKNOWN_CLS]
        unknown_pred += sum(1 for p in pred_by_image.get(name, []) if p["cls"] == UNKNOWN_CLS)
        gt_arr = np.array([g[1] for g in gts], dtype=np.float64).reshape(-1, 4)
        gt_cls = [g[0] for g in gts]
        used = np.zeros(len(gts), dtype=bool)
        for c in range(num_classes):
            fn_cnt[c] += sum(1 for g in gt_cls if g == c)
        if not preds:
            continue
        preds = sorted(preds, key=lambda p: -p["score"])
        for p in preds:
            c = p["cls"]
            if not (0 <= c < num_classes):
                continue
            cand = [i for i in range(len(gts)) if gt_cls[i] == c and not used[i]]
            if not cand or gt_arr.size == 0:
                fp_cnt[c] += 1
                all_pred[c].append((p["score"], 0))
                continue
            ious = _iou_matrix(np.array(p["box"], dtype=np.float64)[None], gt_arr[cand])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                used[cand[j]] = True
                tp_cnt[c] += 1
                fn_cnt[c] -= 1
                all_pred[c].append((p["score"], 1))
            else:
                fp_cnt[c] += 1
                all_pred[c].append((p["score"], 0))
    aps = np.zeros(num_classes)
    for c in range(num_classes):
        if tp_cnt[c] + fn_cnt[c] == 0:
            aps[c] = np.nan
            continue
        d = sorted(all_pred[c], key=lambda t: -t[0])
        tp = np.cumsum([x[1] for x in d])
        fp = np.cumsum([1 - x[1] for x in d])
        rec = tp / max(tp_cnt[c] + fn_cnt[c], 1)
        prec = tp / np.maximum(tp + fp, 1)
        aps[c] = _ap(rec, prec)
    valid = ~np.isnan(aps)
    return aps, float(aps[valid].mean()) if valid.any() else 0.0, \
        {"tp": tp_cnt, "fp": fp_cnt, "fn": fn_cnt, "unknown_pred": unknown_pred}
