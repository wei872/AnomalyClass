"""数据管线测试: 合成疵点 / 布幅增广 / 标注框同步 / YOLO 读取."""

import numpy as np
import torch

from fabric_ncd.data.augment import (
    EvalTransform,
    TrainAugment,
    fabric_width_augment,
    hflip,
    rot90,
)
from fabric_ncd.data.fabric_mask import estimate_cloth_mask
from fabric_ncd.data.synth import SYNTH_TYPES, apply_random_anomaly
from fabric_ncd.data.yolo import FabricDataset, build_datasets, load_dataset_spec
from fabric_ncd.data.demo import make_dataset


def _blank_sample(h=64, w=64):
    image = np.full((h, w, 3), 128, dtype=np.uint8)
    return {
        "image": image,
        "defect_mask": np.zeros((h, w), dtype=np.uint8),
        "cloth_mask": np.ones((h, w), dtype=np.uint8),
        "boxes": np.array([[0.0, 0.25, 0.25, 0.75, 0.75]], dtype=np.float32),
    }


def test_synth_generators_produce_mask():
    rng = np.random.default_rng(0)
    for name in SYNTH_TYPES:
        img, mask = apply_random_anomaly(_blank_sample()["image"], np.zeros((64, 64), np.uint8), rng, types=[name])
        assert img.shape == (64, 64, 3)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8


def test_width_augment_updates_boxes_and_cloth():
    rng = np.random.default_rng(1)
    sample = _blank_sample()
    out = fabric_width_augment(sample, max_margin=0.4, rng=rng)
    assert out["image"].shape[:2] == out["cloth_mask"].shape[:2]
    assert out["boxes"].shape == (1, 5)
    assert 0 <= out["boxes"][0, 1] <= out["boxes"][0, 3] <= 1
    assert out["cloth_mask"].mean() <= 1.0


def test_flip_rot90_boxes():
    sample = _blank_sample()
    flipped = hflip(sample)
    np.testing.assert_allclose(flipped["boxes"][0, 1], 0.25, atol=1e-6)
    np.testing.assert_allclose(flipped["boxes"][0, 3], 0.75, atol=1e-6)
    rotated = rot90(sample, 1)
    assert rotated["image"].shape == np.rot90(sample["image"], 1).shape
    assert rotated["boxes"].shape == (1, 5)


def test_train_augment_shapes():
    aug = TrainAugment({"image_size": (48, 48), "synth_prob": 1.0})
    sample = _blank_sample()
    out = aug(sample, np.random.default_rng(3))
    assert out["image"].shape == (48, 48, 3)
    assert out["defect_mask"].shape == (48, 48)
    assert out["cloth_mask"].shape == (48, 48)
    assert out["boxes"].ndim == 2 and out["boxes"].shape[1] == 5


def test_eval_transform():
    aug = EvalTransform({"image_size": (32, 32)})
    out = aug(_blank_sample())
    assert out["image"].shape == (32, 32, 3)


def test_cloth_mask_estimation():
    from fabric_ncd.data.demo import weave_image

    rng = np.random.default_rng(0)
    img = weave_image((80, 80), rng)
    mask, conf = estimate_cloth_mask(img)
    assert mask.shape == (80, 80)
    assert mask.mean() > 0.9  # 纯布面 -> 几乎全前景

    canvas = np.full((120, 140, 3), 35, dtype=np.uint8)
    canvas[10:90, 20:100] = img
    mask, conf = estimate_cloth_mask(canvas)
    assert mask[50, 60] == 1
    assert mask[0, 0] == 0


def test_yolo_dataset(tmp_path):
    root = make_dataset(tmp_path, n_train=4, n_val=2, n_normal=2)
    spec = load_dataset_spec({"data": {"dataset_yaml": str(root / "dataset.yaml")}})
    ds = FabricDataset(spec, "train", (48, 48), train=False)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (3, 48, 48)
    assert sample["image"].dtype == torch.float32
    assert sample["cloth_mask"].shape == (1, 48, 48)
    loader_cfg = {
        "data": {
            "dataset_yaml": str(root / "dataset.yaml"),
            "image_size": [48, 48],
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "normal_ratio": 0.5,
        }
    }
    datasets = build_datasets(loader_cfg)
    assert "train" in datasets and "val" in datasets
