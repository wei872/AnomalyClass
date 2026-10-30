import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anomalyncd.config import class_order, load_class_names, load_dataset_spec
from anomalyncd.data import FabricRegions, make_crop, prepare_dataset, read_yolo_labels
from anomalyncd.transforms import PairTransform, TwoViews


def test_fabric19_names_and_windows_rebase(synthetic):
    spec = load_dataset_spec(synthetic["yaml"], synthetic["raw"])
    assert spec.nc == 19 and spec.names[0] == "纬线杂质" and spec.names[-1] == "飘絮"
    assert spec.paths["train"] == synthetic["raw"] / "images/train"
    assert spec.paths["normalval"] == synthetic["raw"] / "normal/images/val"
    assert len(load_class_names(synthetic["yaml"])) == 19


def test_windows_yaml_without_root_has_actionable_error(monkeypatch):
    monkeypatch.delenv("ANOMALYNCD_DATA_ROOT", raising=False)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="--data-root"):
        load_dataset_spec(root / "dataset/fabric19.yaml")


def test_bad_class_splits():
    with pytest.raises(ValueError, match="显式"):
        class_order(19, "discovery", None)
    with pytest.raises(ValueError, match="重复"):
        class_order(19, "discovery", [0, 0])
    with pytest.raises(ValueError, match="新类"):
        class_order(19, "discovery", list(range(19)))
    order, known = class_order(19, "discovery", [7, 2])
    assert order[:2] == [7, 2] and sorted(order) == list(range(19)) and known == [0, 1]


def test_yolo_validation(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert read_yolo_labels(path, 19)[0].class_id == 0
    for text in ("19 0.5 0.5 0.2 0.2", "0.5 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2",
                 "0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2", "0 0.5 0.5 0.2 0.2 0.3"):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_yolo_labels(path, 19)


def test_crops_edge_small_and_mask_alignment():
    image = Image.new("RGB", (100, 30), "blue")
    for bbox in ((0.01, 0.02, 0.02, 0.04), (0.99, 0.98, 0.02, 0.04), (0.5, 0.5, 0.98, 0.1)):
        crop, mask, box, area = make_crop(image, bbox, 0.1, 0.1)
        assert crop.size == mask.size and crop.width == crop.height and area > 0
        assert area == np.count_nonzero(np.asarray(mask))
        assert box[2]-box[0] == crop.width
    transform = PairTransform(32, training=True)
    for _ in range(20):
        crop, mask, _, _ = make_crop(image, (0.01, 0.02, 0.02, 0.04), 0.1, 0.1)
        tensor, mask_tensor = transform(crop, mask)
        assert tensor.shape == (3, 32, 32) and mask_tensor.shape == (1, 32, 32)
        assert bool(mask_tensor.any()) and torch.isfinite(tensor).all()


def test_manifest_and_multi_region_images(synthetic):
    manifest = synthetic["manifest"]
    assert len(manifest["splits"]["train"]) == 20
    assert len(manifest["splits"]["val"]) == 20
    assert manifest["normal_counts"] == {"normaltrain": 1, "normalval": 1}
    records = manifest["splits"]["val"]
    first = [r for r in records if r["image_id"] == records[0]["image_id"]]
    assert {r["class_id"] for r in first} == {0, 7}
    dataset = FabricRegions(synthetic["prepared"], "val", list(range(19)), 19, "supervised", PairTransform(32), evaluation=True)
    assert len(dataset) == 20
    assert dataset[0]["images"][0].shape == (3, 32, 32)


def test_ncd_masks_novel_labels_but_not_evaluation(synthetic):
    order, _ = class_order(19, "discovery", list(range(10)))
    transform = TwoViews(PairTransform(32, training=True))
    train = FabricRegions(synthetic["prepared"], "train", order, 10, "discovery", transform)
    for index in range(len(train)):
        sample = train[index]
        if train.labels[index] >= 10:
            assert sample["target"] == -100 and not sample["labelled"]
        else:
            assert sample["target"] == train.labels[index] and sample["labelled"]
    val = FabricRegions(synthetic["prepared"], "val", order, 10, "discovery", PairTransform(32), evaluation=True)
    assert all(val[i]["target"] == val.labels[i] for i in range(len(val)))


def test_duplicate_split_rejected(synthetic, tmp_path):
    raw = tmp_path / "copy"
    shutil.copytree(synthetic["raw"], raw)
    train = sorted((raw / "images/train").rglob("*.png"))[0]
    val = sorted((raw / "images/val").rglob("*.png"))[0]
    val.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="数据泄漏"):
        prepare_dataset(synthetic["yaml"], tmp_path / "out", raw, check_only=True, progress=lambda _: None)


def test_preprocess_refuses_to_overwrite_inputs(synthetic):
    with pytest.raises(ValueError, match="输出目录"):
        prepare_dataset(synthetic["yaml"], synthetic["raw"], synthetic["raw"], progress=lambda _: None)
    with pytest.raises(ValueError, match="输出目录"):
        prepare_dataset(synthetic["yaml"], synthetic["raw"] / "images/train/crops", synthetic["raw"], progress=lambda _: None)


def test_default_configs_use_lightweight_deitsmall8():
    from anomalyncd.config import load_config
    from anomalyncd.model import backbone_spec

    root = Path(__file__).resolve().parents[1]
    for name in ("fabric19.yaml", "fabric19_ncd.yaml"):
        cfg = load_config(root / "configs" / name)
        assert cfg["model"]["backbone"] == "dino_vits8"
        assert cfg["model"]["pretrained_path"] is None
    spec = backbone_spec("dino_vits8")
    assert spec["embed_dim"] == 384 and spec["patch_size"] == 8
    assert spec["weights"] == "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"


def test_detector_config_validation(smoke_config):
    import copy

    from anomalyncd.config import validate_config

    cfg = copy.deepcopy(smoke_config)
    cfg["detector"]["anomaly_threshold"] = 1.5
    with pytest.raises(ValueError, match="anomaly_threshold"):
        validate_config(cfg)
    cfg = copy.deepcopy(smoke_config)
    cfg["mode"] = "discovery"
    cfg["discovery"]["known_class_ids"] = list(range(10))
    cfg["detector"]["class_agnostic"] = False
    with pytest.raises(ValueError, match="class_agnostic"):
        validate_config(cfg)
