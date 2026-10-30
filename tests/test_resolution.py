"""高分辨率输入相关测试: 非正方形输入 / 整数倍池化的周期估计 / 切块评估."""

import numpy as np
import torch

from fabric_ncd.data.weave import estimate_period
from fabric_ncd.models.fabric_ncd import FabricNCD
from fabric_ncd.models.texture import estimate_weave_period
from fabric_ncd.utils import load_yaml
from fabric_ncd.data.demo import make_dataset


def _stripes(h: int, w: int, px: float, py: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 128.0 + 40.0 * np.sin(2 * np.pi * x / px) + 40.0 * np.sin(2 * np.pi * y / py)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_period_estimate_non_square():
    """非正方形图像不能被压成正方形: 经纬周期应各自按真实尺寸换算."""
    rgb = np.stack([_stripes(1024, 1728, 6, 10)] * 3, axis=-1)
    px, py, conf = estimate_period(rgb)
    assert conf > 2.0
    assert abs(px - 6.0) < 0.6, px
    assert abs(py - 10.0) < 1.0, py


def test_period_estimate_matches_torch():
    rgb = np.stack([_stripes(1440, 2440, 6, 6)] * 3, axis=-1)
    px, py, _ = estimate_period(rgb)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None] / 255.0
    stats = estimate_weave_period(tensor)
    assert abs(px - float(stats["period_x"])) < 0.8
    assert abs(py - float(stats["period_y"])) < 0.8


def _model(size, num_classes=3):
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"]["image_size"] = list(size)
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["heads"]["num_classes"] = num_classes
    model = FabricNCD(cfg).eval()
    return cfg, model


def test_forward_non_square_input():
    cfg, model = _model([64, 96])
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        out = model(x)
    assert out["anomaly_score"].shape[-2:] == (8, 12)
    assert out["cloth_prob"].shape[-2:] == (16, 24)
    assert out["class_logits"].shape == (2, 3, 8, 12)


def test_forward_maps_single_and_tiled():
    from fabric_ncd.inference import as_hw, forward_maps

    assert as_hw(512) == (512, 512)
    assert as_hw([1024, 1728]) == (1024, 1728)
    assert as_hw(None, (32, 48)) == (32, 48)

    _, model = _model([64, 96])
    rgb = np.random.randint(0, 255, (150, 210, 3), dtype=np.uint8)
    single = forward_maps(model, rgb, torch.device("cpu"), (64, 96), tile=False)
    assert single["score"].shape == (150, 210)
    assert single["cloth"].shape == (150, 210)

    tiled = forward_maps(
        model, rgb, torch.device("cpu"), (64, 96), tile=True, tile_size=(64, 96), tile_overlap=16
    )
    assert tiled["score"].shape == (150, 210)
    assert np.isfinite(tiled["score"]).all()
    # 切块后应覆盖整图 (没有被融合到的区域会保持 0)
    assert tiled["score"].any()


def test_evaluate_tiled_smoke(tmp_path):
    from fabric_ncd.engine import evaluate_tiled

    root = make_dataset(tmp_path / "data", n_train=4, n_val=3, n_normal=2, size=(64, 64))
    cfg, model = _model([64, 96])
    cfg["data"]["dataset_yaml"] = str(root / "dataset.yaml")
    cfg["eval"].update({"threshold": 0.5, "tile_size": [48, 64], "tile_overlap": 16})
    metrics = evaluate_tiled(model, cfg, torch.device("cpu"), split="val")
    assert metrics["num_samples"] == 5  # 3 张疵点 + 2 张正常
    assert np.isfinite(metrics.get("pixel_auroc", 0.5))
