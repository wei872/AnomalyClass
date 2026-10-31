"""Tests for the normal-only anomaly path: the source of open-set detection."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomalyncd.anomaly import AnomalyScorer, binarize, build_bank, calibrate_bank, load_bank, regions_from_map
from anomalyncd.config import deep_merge
from anomalyncd.data import make_crop_from_mask, mask_to_bbox
from anomalyncd.detector import prepare_detector_dataset
from anomalyncd.end_to_end import box_iou, fuse_regions, predict_fabricncd, save_bundle
from anomalyncd.engine import resolve_device, train


@pytest.fixture(scope="module")
def weave(tmp_path_factory):
    """Consistent fabric-like texture plus one image with an injected defect."""
    root = tmp_path_factory.mktemp("weave")
    (root / "train").mkdir()
    (root / "val").mkdir()

    def texture(seed):
        local = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:96, 0:96]
        base = 120 + 22*np.sin(xx*2*np.pi/6) + 18*np.sin(yy*2*np.pi/6)
        return np.clip(base + local.normal(0, 3, base.shape), 0, 255).astype(np.uint8)

    for index in range(12):
        Image.fromarray(texture(index)).convert("RGB").save(root / f"train/n{index:02d}.png")
    for index in range(4):
        Image.fromarray(texture(100+index)).convert("RGB").save(root / f"val/n{index:02d}.png")
    truth = np.zeros((96, 96), bool)
    truth[30:50, 40:62] = True
    defective = texture(999).astype(np.int16)
    defective[truth] += 95
    Image.fromarray(np.clip(defective, 0, 255).astype(np.uint8)).convert("RGB").save(root / "defect.png")
    Image.fromarray(texture(555)).convert("RGB").save(root / "clean.png")
    return {"root": root, "truth": truth}


@pytest.fixture(scope="module")
def ad_config():
    return {
        "model": {"backbone": "tiny_vit", "pretrained_path": None, "image_size": 32,
                  "grad_from_block": 3, "mask_layers": 2, "n_heads": 2,
                  "hidden_dim": 32, "bottleneck_dim": 16},
        "anomaly": {"enabled": True, "image_size": 64, "layers": [1, 2, 3], "neighborhood": 3,
                    "bank_size": 256, "max_candidates": 2000, "coreset": True, "knn": 3,
                    "smooth_sigma": 1.0, "topk_ratio": 0.05, "min_area": 4,
                    "max_regions": 5, "threshold_steps": 20},
    }


@pytest.fixture(scope="module")
def bank_dir(weave, ad_config, tmp_path_factory):
    output = tmp_path_factory.mktemp("bank") / "bank"
    build_bank(weave["root"] / "train", output, ad_config, device_name="cpu",
               allow_random_init=True, batch_size=4, progress=lambda _: None)
    calibrate_bank(output, weave["root"] / "val", ad_config, device_name="cpu",
                   allow_random_init=True, progress=lambda _: None)
    return output


def test_binarize_rejects_whole_image_component():
    """A component covering the image is a failed threshold, not a defect."""
    dense = np.full((64, 64), 2.0, np.float32)
    settings = {"min_area": 4, "max_area_ratio": 0.35, "threshold_steps": 12}
    binary, info = binarize(dense, settings, pixel_threshold=0.1)
    assert not binary.any() and info["components"] == 0


def test_binarize_returns_empty_below_calibrated_normal_level():
    dense = np.full((64, 64), 0.4, np.float32)
    settings = {"min_area": 4, "max_area_ratio": 0.35, "threshold_steps": 12}
    binary, info = binarize(dense, settings, pixel_threshold=1.0)
    assert not binary.any() and info["components"] == 0


def test_mask_crop_keeps_defect_shape_not_rectangle():
    """A ring mask must stay a ring; a box mask would fill the hole with weave."""
    image = Image.new("RGB", (80, 80), "gray")
    mask = np.zeros((80, 80), bool)
    mask[20:40, 20:40] = True
    mask[26:34, 26:34] = False          # hole in the middle
    crop, mask_crop, _, area, bbox = make_crop_from_mask(image, mask, 0.1, 0.1)
    assert crop.size == mask_crop.size
    assert area == int(mask.sum()) < 20*20
    assert 0 < bbox[2] < 1 and 0 < bbox[3] < 1
    cx, cy, w, h = mask_to_bbox(mask)
    assert (cx, cy) == pytest.approx(bbox[:2])
    assert (w, h) == pytest.approx(bbox[2:])
    with pytest.raises(ValueError):
        make_crop_from_mask(image, np.zeros((80, 80), bool), 0.1, 0.1)


def test_fuse_marks_unmatched_anomaly_as_novel_candidate():
    detections = [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "detector_confidence": 0.9}]
    anomaly_regions = [
        {"bbox_xyxy": [1.0, 1.0, 11.0, 11.0], "area": 80, "anomaly_score": 1.4,
         "anomaly_score_raw": 0.7, "mask": np.ones((20, 20), bool)},
        {"bbox_xyxy": [50.0, 50.0, 60.0, 60.0], "area": 90, "anomaly_score": 2.1,
         "anomaly_score_raw": 0.9, "mask": np.ones((20, 20), bool)},
    ]
    fused = fuse_regions(detections, anomaly_regions, iou_threshold=0.3)
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert [item["source"] for item in fused] == ["fused", "anomaly_map"]
    assert fused[0]["novel_candidate"] is False
    assert fused[1]["novel_candidate"] is True and fused[1]["detector_confidence"] is None


def test_fuse_keeps_detector_only_region_without_fake_anomaly_score():
    detections = [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "detector_confidence": 0.8}]
    fused = fuse_regions(detections, [], iou_threshold=0.3)
    assert len(fused) == 1 and fused[0]["source"] == "detector"
    assert fused[0]["anomaly_score"] is None and fused[0]["mask"] is None


def test_bank_rejects_mismatched_encoder_weights(bank_dir, ad_config):
    """Distances are meaningless if the query encoder differs from the bank's."""
    metadata = load_bank(bank_dir)
    metadata["encoder_fingerprint"] = "0"*32
    with pytest.raises(ValueError, match="权重不一致"):
        AnomalyScorer(metadata, ad_config, resolve_device("cpu"), allow_random_init=True)


def test_bank_rejects_mismatched_settings(bank_dir, ad_config):
    metadata = load_bank(bank_dir)
    changed = deep_merge(ad_config, {"anomaly": {"image_size": 32}})
    with pytest.raises(ValueError, match="image_size"):
        AnomalyScorer(metadata, changed, resolve_device("cpu"), allow_random_init=True)


def test_anomaly_map_localizes_defect_and_rejects_clean_image(bank_dir, ad_config, weave):
    """The whole point: localize a defect with no box, no label, no detector."""
    scorer = AnomalyScorer(load_bank(bank_dir), ad_config, resolve_device("cpu"),
                           allow_random_init=True)
    scores, ious, counts = {}, {}, {}
    for name in ("defect", "clean"):
        with Image.open(weave["root"] / f"{name}.png") as handle:
            image = handle.convert("RGB")
        dense, raw = scorer.anomaly_map(image)
        assert dense.shape == (96, 96)
        scores[name] = scorer.normalize_image_score(raw)
        threshold = scorer.calibration["pixel_high"]
        binary, _ = binarize(dense, scorer.settings, threshold)
        regions = regions_from_map(dense, binary, scorer.settings,
                                   score_of=lambda v: float(scorer.normalize_pixel(v)))
        counts[name] = len(regions)
        best = 0.0
        for region in regions:
            union = float((region["mask"] | weave["truth"]).sum())
            best = max(best, float((region["mask"] & weave["truth"]).sum())/union if union else 0.0)
        ious[name] = best
    assert scores["defect"] > scores["clean"]
    assert counts["defect"] >= 1 and counts["clean"] == 0
    assert ious["defect"] > 0.2, f"anomaly mask missed the defect: IoU={ious['defect']}"


def test_detector_training_really_hides_novel_class_images(synthetic, tmp_path):
    """Novel boxes must not train the detector, nor become 'background'."""
    manifest = prepare_detector_dataset(synthetic["yaml"], tmp_path / "detector_data",
                                        synthetic["raw"], class_agnostic=True, mode="discovery",
                                        known_class_ids=list(range(10)), progress=lambda _: None)
    stats = manifest["stats"]
    assert stats["novel_class_ids"] == list(range(10, 19))
    # Synthetic train has one image per class; classes 10..18 must be removed.
    assert stats["train"]["dropped_novel_images"] == 9
    assert stats["train"]["images"] == 10
    assert stats["val"]["images"] == 19          # val stays complete for evaluation
    assert stats["val"]["novel_boxes"] == 9
    assert manifest["known_class_ids"] == list(range(10))
    with pytest.raises(ValueError, match="known_class_ids"):
        prepare_detector_dataset(synthetic["yaml"], tmp_path / "bad", synthetic["raw"],
                                 class_agnostic=True, mode="discovery", progress=lambda _: None)


def test_anomaly_only_bundle_finds_defect_yolo_never_detected(smoke_config, synthetic, weave,
                                                             bank_dir, ad_config, tmp_path):
    """End-to-end with no detector at all: open-set localization must still work."""
    cfg = deep_merge(smoke_config, {"anomaly": ad_config["anomaly"]})
    cfg["experiment"]["output_dir"] = str(tmp_path / "clf")
    train(cfg, allow_random_init=True)
    classifier = Path(cfg["experiment"]["output_dir"]) / "last.pt"
    bundle = tmp_path / "bundle"
    save_bundle(bundle, None, classifier, cfg, anomaly_bank=bank_dir)

    result = predict_fabricncd(bundle, weave["root"] / "defect.png", tmp_path / "pred",
                               device_name="cpu")
    assert result["detections"] == []                      # YOLO contributed nothing
    assert result["localization_source"] == "anomaly_map"
    assert result["anomaly_score_source"] == "normal_only_feature_distance"
    assert result["is_anomaly"] is True
    assert result["regions"], "anomaly map failed to propose the defect"
    assert result["novel_candidate_count"] == len(result["regions"])
    assert all(r["source"] == "anomaly_map" and r["novel_candidate"] for r in result["regions"])
    assert all(r["mask_shape"] == "segmentation" for r in result["regions"])
    # A segmentation mask is strictly tighter than its bounding box.
    assert any(r["mask_area"] < (r["bbox_xyxy"][2]-r["bbox_xyxy"][0])
               * (r["bbox_xyxy"][3]-r["bbox_xyxy"][1]) for r in result["regions"])
    assert result["image_prediction_ids"]
    assert Path(result["localization"]["heatmap"]).is_file()


def test_bundle_requires_detector_or_bank(smoke_config, tmp_path):
    classifier = tmp_path / "clf.pt"
    torch.save({"dummy": True}, classifier)
    with pytest.raises(ValueError, match="至少包含"):
        save_bundle(tmp_path / "empty_bundle", None, classifier, smoke_config)


def test_coreset_matvec_matches_naive_and_keeps_centre_norm():
    """The fast distance form must select exactly the same points as the naive one."""
    from anomalyncd.anomaly import _greedy_coreset

    def naive(features, target):
        centre = features.mean(dim=0)
        index = int(torch.argmax(((features-centre)**2).sum(dim=1)))
        chosen = [index]
        mind = torch.full((features.shape[0],), float("inf"))
        picked = torch.zeros(features.shape[0], dtype=torch.bool)
        for _ in range(1, target):
            mind = torch.minimum(mind, ((features-features[index])**2).sum(dim=1))
            picked[index] = True
            mind = mind.masked_fill(picked, -1.0)
            index = int(torch.argmax(mind))
            chosen.append(index)
        return torch.tensor(chosen)

    torch.manual_seed(0)
    for _ in range(3):
        features = torch.randn(200, 32)
        # Unequal norms are what break the version that drops ||c||^2.
        features[:8] += 6.0
        assert torch.equal(_greedy_coreset(features, 25), naive(features, 25))


def test_bank_build_bounds_memory_and_covers_every_normal_image(weave, ad_config, tmp_path):
    """Candidate count must be capped before features reach the device."""
    from anomalyncd.anomaly import anomaly_settings

    cfg = deep_merge(ad_config, {"anomaly": {"max_candidates": 100, "bank_size": 32,
                                             "coreset": False}})
    output = tmp_path / "small_bank"
    metadata = build_bank(weave["root"] / "train", output, cfg, device_name="cpu",
                          allow_random_init=True, batch_size=4, progress=lambda _: None)
    settings = anomaly_settings(cfg)
    # 12 images x 64 patches = 768 total, but only <=100 candidates may be kept.
    assert metadata["normal_patches"] == 12*8*8
    assert metadata["bank_size"] <= settings["max_candidates"]
    assert load_bank(output)["features"].shape[0] <= settings["max_candidates"]
