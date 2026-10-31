# -*- coding: utf-8 -*-
"""后处理 (numpy, torch/ONNX 两条推理路径共用, 保证行为一致)。

decode 输出: boxes (N,4) px, obj_logits (N,), sims (N,C) 余弦相似度
规则:
  1) score = sigmoid(obj_logits) >= conf
  2) cls = argmax sims; 若 max(sims) < tau_unknown -> cls = -1 (未知缺陷)
  3) 类无关 NMS 合并重叠/跨 tile 重复框
"""
from __future__ import annotations

import numpy as np

UNKNOWN_CLS = -1
UNKNOWN_NAME = "未知缺陷"
UNKNOWN_NAME_EN = "unknown_defect"

DEFAULT_THRESHOLDS = {
    "conf": 0.25,          # 目标置信度阈值 (训练结束自动标定覆盖)
    "tau_unknown": 0.35,   # 余弦相似度开集阈值 (训练结束自动标定覆盖)
    "nms_iou": 0.45,
    "pre_nms_topk": 1000,
    "max_det": 300,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def postprocess(boxes: np.ndarray, obj_logits: np.ndarray, sims: np.ndarray,
                thresholds: dict | None = None, conf: float | None = None) -> list[dict]:
    """单张图 (已 decode) -> [{'box': xyxy, 'score': s, 'cls': c, 'max_sim': m}]"""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    c_thr = conf if conf is not None else th["conf"]
    score = sigmoid(obj_logits.astype(np.float64))
    max_sim = sims.max(axis=-1) if sims.shape[-1] > 0 else np.zeros(sims.shape[0])
    cls = sims.argmax(axis=-1) if sims.shape[-1] > 0 else np.zeros(sims.shape[0], dtype=np.int64)
    keep = score >= c_thr
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []
    if idx.size > th["pre_nms_topk"]:
        idx = idx[np.argsort(score[idx])[::-1][: th["pre_nms_topk"]]]
    # 开集判定: 余弦相似度低于 tau -> 未知缺陷
    cls = cls[idx].copy()
    cls[max_sim[idx] < th["tau_unknown"]] = UNKNOWN_CLS
    s = score[idx]
    b = boxes[idx]
    # 已知类优先 + 分数排序, 再类无关 NMS (跨 tile 重复框合并)
    order = np.lexsort((-s, (cls == UNKNOWN_CLS).astype(np.int64)))
    b, s, cls, idx = b[order], s[order], cls[order], idx[order]
    k = nms(b, s, th["nms_iou"])
    k = k[: th["max_det"]]
    return [{"box": b[i].astype(np.float32), "score": float(s[i]),
             "cls": int(cls[i]), "max_sim": float(max_sim[idx[i]])} for i in k]


def merge_tile_preds(preds: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """多 tile 预测合并: 已知类优先, 分数次之, 贪心抑制。"""
    if not preds:
        return []
    s = np.array([p["score"] for p in preds])
    known = np.array([p["cls"] != UNKNOWN_CLS for p in preds])
    order = np.lexsort((-s, (~known).astype(np.int64)))
    boxes = np.stack([p["box"] for p in preds])
    kept: list[int] = []
    suppressed = np.zeros(len(preds), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        iou = inter / (a + ai - inter + 1e-9)
        suppressed |= iou > iou_thr
    return [preds[i] for i in kept]
