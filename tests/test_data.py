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


def test_cloth_mask_work_size_matches_full():
    """降分辨率估计的掩码应与全分辨率基本一致 (auto 会按经纬周期定降采样倍数)."""
    from fabric_ncd.data.demo import weave_image
    from fabric_ncd.data.fabric_mask import auto_work_size, estimate_cloth_mask

    rng = np.random.default_rng(4)
    cloth = weave_image((320, 480), rng)
    canvas = np.full((400, 640, 3), 35, dtype=np.uint8)
    canvas[40:360, 80:560] = cloth

    full, _ = estimate_cloth_mask(canvas)
    ws = auto_work_size(canvas)
    assert ws is None or ws >= 64
    fast, _ = estimate_cloth_mask(canvas, work_size="auto")

    inter = np.logical_and(full, fast).sum()
    union = np.logical_or(full, fast).sum()
    assert union > 0
    assert inter / union > 0.90
    assert full[0, 0] == 0 and full[200, 320] == 1


def test_export_defect_mask_from_yolo(tmp_path):
    """YOLO 标注应能落成疵点掩码 (训练时是实时生成的, 这里验证同一套逻辑)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from prepare_dataset import _export_defect_one

    root = make_dataset(tmp_path / "d", n_train=2, n_val=1, n_normal=1, size=(64, 96))
    images = sorted((root / "images" / "train").iterdir())
    labels = sorted((root / "labels" / "train").iterdir())
    assert images and labels
    out = tmp_path / "defect_masks"
    out.mkdir()
    path = _export_defect_one(str(images[0]), str(labels[0]), str(out), "auto")
    assert Path(path).exists()
    mask = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(path))
    assert mask.shape[0] <= 64 and mask.shape[1] <= 96
    assert (mask > 127).sum() > 0


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

def _strip_canvas(h, w, left, right, period=6, bg=35, seed=0):
    """窄布条画布: 背景占大半画面, 用来复现"周期被台阶边缘带偏"的场景."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cloth = 128 + 30 * np.sin(2 * np.pi * xx / period) + 30 * np.sin(2 * np.pi * yy / period)
    cloth = np.clip(cloth + rng.normal(0, 4, (h, w)), 0, 255)
    canvas = np.full((h, w, 3), bg, dtype=np.float32)
    canvas[:, left : w - right] = np.stack([cloth] * 3, -1)[:, left : w - right]
    return canvas.astype(np.uint8)


def test_period_not_fooled_by_cloth_edge():
    """窄布条上多数窗口压在背景/交界, 周期不能被台阶边缘估成几十像素."""
    from fabric_ncd.data.fabric_mask import _period_pixels

    img = _strip_canvas(1220, 720, left=600, right=0, period=6)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    px, py = _period_pixels(gray)
    assert 3.0 <= px <= 10.0 and 3.0 <= py <= 10.0, (px, py)


def test_cloth_mask_narrow_strip_auto_matches_full():
    """窄布条 (布只占画面 16%) 上 auto 降采样掩码仍要与全分辨率一致."""
    from fabric_ncd.data.fabric_mask import auto_work_size, estimate_cloth_mask

    img = _strip_canvas(1220, 720, left=600, right=0, period=6)
    full, _ = estimate_cloth_mask(img)
    ws = auto_work_size(img)
    assert ws is None or ws >= 64
    fast, _ = estimate_cloth_mask(img, work_size="auto")
    inter = np.logical_and(full, fast).sum()
    union = np.logical_or(full, fast).sum()
    assert union > 0
    assert inter / union > 0.95
    # 覆盖率应贴近真实的布面占比 (120/720), 不能糊到背景里
    assert abs(float(fast.mean()) - 120.0 / 720.0) < 0.02


def test_cloth_mask_small_work_size_falls_back_to_color():
    """强行给过小的 work_size (纹理被平均掉) 时, 应改用颜色线索而不是退化."""
    from fabric_ncd.data.fabric_mask import estimate_cloth_mask

    img = _strip_canvas(610, 360, left=100, right=100, period=6)
    full, _ = estimate_cloth_mask(img)
    fast, _ = estimate_cloth_mask(img, work_size=128)  # 约等于 2440 -> 512 的压缩比
    inter = np.logical_and(full, fast).sum()
    union = np.logical_or(full, fast).sum()
    assert inter / union > 0.95
    assert abs(float(fast.mean()) - float(full.mean())) < 0.03


def test_aspect_jitter_keeps_boxes_and_masks_aligned():
    """各向异性拉伸: 归一化坐标不变, 图像/掩码同步变形."""
    from fabric_ncd.data.augment import aspect_jitter

    sample = _blank_sample(64, 96)
    sample["defect_mask"][20:40, 30:60] = 1
    rng = np.random.default_rng(0)
    out = aspect_jitter(sample, (1.15, 1.15), rng)
    assert out["image"].shape[:2] == out["defect_mask"].shape[:2]
    assert out["image"].shape[0] != 64  # 确实变形了
    np.testing.assert_allclose(out["boxes"], sample["boxes"])
    # 掩码内容仍在框附近 (相对位置不变)
    frac = out["defect_mask"].mean()
    assert abs(frac - sample["defect_mask"].mean()) < 0.02


def test_train_augment_aspect_jitter_runs():
    aug = TrainAugment({"image_size": (48, 48), "aspect_jitter": (0.85, 1.18), "aspect_prob": 1.0})
    out = aug(_blank_sample(64, 96), np.random.default_rng(1))
    assert out["image"].shape == (48, 48, 3)
    assert out["defect_mask"].shape == (48, 48)
    assert out["cloth_mask"].shape == (48, 48)


def test_cloth_mask_full_mode(tmp_path):
    """整幅都是布的数据集: cloth_mask: full 必须给出全 1 掩码.

    估计器在这种图上常给出 5%~60% 的覆盖率, 于是 anomaly 损失只在那一小块像素上有
    监督, 其余像素的异常头输出完全不受约束 (实测表现为正常图 learned_max 打满 1.0).
    """
    root = make_dataset(tmp_path, n_train=2, n_val=1, n_normal=1, size=(64, 96))
    spec = load_dataset_spec({"data": {"dataset_yaml": str(root / "dataset.yaml")}})

    full = FabricDataset(spec, "train", (48, 48), train=False,
                         augment_cfg={"image_size": (48, 48), "cloth_mask": "full"})
    assert float(full[0]["cloth_mask"].min()) == 1.0

    est = FabricDataset(spec, "train", (48, 48), train=False,
                        augment_cfg={"image_size": (48, 48), "cloth_mask": "estimate"})
    # 估计模式保持原行为 (不一定全 1), 但必须仍是 0/1 掩码
    m = est[0]["cloth_mask"]
    assert set(m.unique().tolist()) <= {0.0, 1.0}
