"""Region clustering and multi-label image metrics without test-label gating."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def classification_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(targets) == 0:
        return {"count": 0}
    predictions = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (predictions, targets), 1)
    rows, cols = linear_sum_assignment(-confusion)
    mapping = dict(zip(rows, cols, strict=True))
    matched = np.array([mapping.get(int(p), -1) for p in predictions])
    labels = list(range(n_classes))
    return {
        "count": int(len(targets)), "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)),
        "hungarian_accuracy": float((matched == targets).mean()),
        "nmi": float(normalized_mutual_info_score(targets, predictions)),
        "ari": float(adjusted_rand_score(targets, predictions)),
        "per_class": {
            str(c): {"support": int((targets == c).sum()),
                     "f1": float(f1_score(targets == c, predictions == c, zero_division=0)),
                     "precision": float(precision_score(targets == c, predictions == c, zero_division=0)),
                     "recall": float(recall_score(targets == c, predictions == c, zero_division=0))}
            for c in labels
        },
    }


def merge_image_probabilities(records: list[dict], probabilities: np.ndarray,
                              area_temperature: float = 50.0) -> tuple[list[np.ndarray], list[list[int]]]:
    """Paper-style sqrt(mask-area)/temperature weighting, grouped by image_id."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["image_id"]].append(index)
    merged, image_targets = [], []
    for indices in groups.values():
        areas = np.array([records[i]["mask_area"] for i in indices], dtype=np.float64)
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError("Invalid mask area in image merging")
        scaled = np.sqrt(areas) / area_temperature
        weights = np.exp(scaled-scaled.max())
        weights /= weights.sum()
        merged.append(weights @ probabilities[indices])
        image_targets.append(sorted({records[i]["class_id"] for i in indices if "class_id" in records[i]}))
    return merged, image_targets


def decide_labels(probabilities: np.ndarray, threshold: float | None) -> list[int]:
    top = int(np.asarray(probabilities).argmax())
    if threshold is None:
        return [top]
    return np.flatnonzero(np.asarray(probabilities) >= threshold).tolist() or [top]


def image_metrics(image_targets: list[list[int]], probabilities: np.ndarray,
                  threshold: float | None = None, n_classes: int | None = None) -> tuple[dict, list[list[int]]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not image_targets:
        return {"count": 0}, []
    n_classes = probabilities.shape[1] if n_classes is None else n_classes
    if n_classes < probabilities.shape[1] or any(c < 0 or c >= n_classes for labels in image_targets for c in labels):
        raise ValueError("Image class ID outside the probability label space")
    predicted_labels = []
    predictions = np.zeros((len(image_targets), n_classes), dtype=np.int64)
    for index, probs in enumerate(probabilities):
        chosen = decide_labels(probs, threshold)
        predictions[index, chosen] = 1
        predicted_labels.append(chosen)
    targets = np.zeros_like(predictions)
    for index, classes in enumerate(image_targets):
        targets[index, classes] = 1
    return {
        "count": len(image_targets), "threshold": threshold,
        "decision_rule": "top1" if threshold is None else "threshold_with_top1_fallback",
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(targets, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(targets, predictions, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(targets, predictions)),
        "multiclass_image_fraction": float(np.mean([len(x) > 1 for x in image_targets])),
    }, predicted_labels
