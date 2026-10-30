"""基础算子 / 指标 / 新类发现单元测试."""

import numpy as np

from fabric_ncd.metrics import (
    ari,
    auroc,
    box_ap,
    cluster_acc,
    per_region_overlap,
    pixel_f1,
    nmi,
)
from fabric_ncd.ncd import RegionAnalyzer, discover_classes, hungarian, kmeans_cosine
from fabric_ncd.ops import bbox_iou, connected_components, extract_regions, topk_mean


def test_connected_components_two_blobs():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[12:18, 10:18] = 1
    labels, num = connected_components(mask)
    assert num == 2
    assert labels[3, 3] != labels[14, 14]
    assert labels[3, 3] > 0 and labels[14, 14] > 0


def test_connected_components_diagonal():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1
    _, num8 = connected_components(mask, connectivity=8)
    _, num4 = connected_components(mask, connectivity=4)
    assert num8 == 1
    assert num4 == 2


def test_extract_regions_and_topk():
    score = np.zeros((32, 32), dtype=np.float32)
    score[5:10, 5:10] = 0.9
    score[20:24, 20:24] = 0.6
    binary = (score > 0.5).astype(np.uint8)
    regions = extract_regions(score, binary, min_area=4)
    assert len(regions) == 2
    assert regions[0].score >= regions[1].score
    assert topk_mean(score, k_ratio=0.01, min_k=4) > 0.5


def test_ious():
    assert abs(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert abs(auroc(y, s) - 1.0) < 1e-6
    assert abs(auroc(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-6


def test_pixel_f1_and_pro():
    gt = np.zeros((16, 16), dtype=np.float32)
    gt[4:8, 4:8] = 1
    score = gt * 0.9 + 0.05
    m = pixel_f1([score], [gt], threshold=0.5)
    assert m["f1"] > 0.9
    normal = np.full((16, 16), 0.1, dtype=np.float32)
    pro = per_region_overlap([score], [gt], [normal], num_thresholds=20)
    assert 0.0 <= pro <= 1.0


def test_box_ap_simple():
    preds = [{"boxes": np.array([[0, 0, 10, 10]], np.float32), "scores": np.array([0.9], np.float32), "labels": np.array([0])}]
    gts = [{"boxes": np.array([[0, 0, 10, 10]], np.float32), "labels": np.array([0])}]
    aps, map50 = box_ap(preds, gts, 0.5, num_classes=2)
    assert abs(aps[0] - 1.0) < 1e-6
    assert abs(map50 - 1.0) < 1e-6


def test_cluster_metrics():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([1, 1, 0, 0, 2, 2])
    assert abs(cluster_acc(a, b) - 1.0) < 1e-6
    assert abs(nmi(a, b) - 1.0) < 1e-6
    assert abs(ari(a, b) - 1.0) < 1e-6


def test_hungarian_identity():
    cost = np.array([[0.1, 0.9, 0.9], [0.9, 0.2, 0.9], [0.9, 0.9, 0.3]])
    pairs = hungarian(cost)
    assert dict(pairs) == {0: 0, 1: 1, 2: 2}


def test_kmeans_and_discover():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8))
    feats = np.concatenate([c + 0.05 * rng.normal(size=(20, 8)) for c in centers])
    labels, centroids, inertia = kmeans_cosine(feats, 3, seed=0)
    assert labels.shape == (60,)
    assert centroids.shape == (3, 8)
    assert inertia < 0.1

    known = centers[:2]
    report = discover_classes(feats, known, ["a", "b"], k=3, novelty_threshold=0.9)
    assert report["k"] == 3
    assert len(report["assignments"]) == 3
    assert len(report["novel_prototypes"]) >= 1


def test_region_analyzer_merges():
    analyzer = RegionAnalyzer(names=["a", "b", "c"], threshold=0.5, min_area_ratio=0.001)
    score = np.zeros((32, 32), dtype=np.float32)
    score[8:16, 8:16] = 0.8
    cloth = np.ones_like(score)
    probs = np.zeros((3, 32, 32), dtype=np.float32)
    probs[1, 8:16, 8:16] = 0.9
    probs[0, 8:16, 8:16] = 0.1
    infos, image_level = analyzer.analyze(score, cloth, probs)
    assert len(infos) == 1
    assert infos[0].cls == 1
    assert image_level["probs"][1] > 0.5
