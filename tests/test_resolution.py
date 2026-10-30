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


def _portrait_dataset(root):
    """2440x1440 的缩比版: 高 240, 宽 160 (竖图)."""
    return make_dataset(root, n_train=2, n_val=1, n_normal=1, size=(240, 160))


def _landscape_dataset(root):
    return make_dataset(root, n_train=2, n_val=1, n_normal=1, size=(160, 240))


def test_image_size_auto_keeps_portrait(tmp_path):
    """input: (H, W) 顺序不能搞反 —— 竖图必须还给出竖的 (H, W), 且长宽比一致."""
    from fabric_ncd.data.yolo import build_datasets, resolve_image_size

    root = _portrait_dataset(tmp_path / "p")
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"dataset_yaml": str(root / "dataset.yaml"), "image_size": "auto", "short_side": 64})
    h, w = resolve_image_size(cfg)
    assert h > w, (h, w)  # 竖图: 高 > 宽
    assert min(h, w) == 64
    assert h % 32 == 0 and w % 32 == 0
    assert abs((h / w) - (240 / 160)) < 0.06
    # 解析后要写回配置, 下游 (模型/eval/infer) 才能看到具体数字
    assert cfg["data"]["image_size"] == [h, w]
    assert cfg["data"]["native_size"] == [240, 160]
    build_datasets(cfg)  # 端到端构建不报错


def test_image_size_auto_keeps_landscape(tmp_path):
    from fabric_ncd.data.yolo import resolve_image_size

    root = _landscape_dataset(tmp_path / "l")
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"dataset_yaml": str(root / "dataset.yaml"), "image_size": "auto", "short_side": 64})
    h, w = resolve_image_size(cfg)
    assert w > h, (h, w)
    assert min(h, w) == 64


def test_explicit_image_size_is_not_transposed(tmp_path):
    """显式 [96, 64] 必须原样当成 (H=96, W=64), 不能被当成 (W, H)."""
    from fabric_ncd.data.yolo import build_datasets

    root = _portrait_dataset(tmp_path / "p")
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"dataset_yaml": str(root / "dataset.yaml"), "image_size": [96, 64], "batch_size": 1})
    ds = build_datasets(cfg)
    sample = ds["train"][0]
    assert tuple(sample["image"].shape[-2:]) == (96, 64)


def test_image_size_auto_never_upscales(tmp_path):
    """原图短边已经小于 short_side 时不要放大 (白烧算力且不增加细节)."""
    from fabric_ncd.data.yolo import resolve_image_size

    root = _landscape_dataset(tmp_path / "l")  # 160 高 x 240 宽
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"dataset_yaml": str(root / "dataset.yaml"), "image_size": "auto",
                        "short_side": 1024, "max_scale": 1.0})
    h, w = resolve_image_size(cfg)
    assert min(h, w) <= 160 + 31, (h, w)  # 最多因为取 32 倍数多一档, 不会放大到 1024
    assert w > h


def test_image_size_auto_can_upscale_when_allowed(tmp_path):
    from fabric_ncd.data.yolo import resolve_image_size

    root = _landscape_dataset(tmp_path / "l")
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"dataset_yaml": str(root / "dataset.yaml"), "image_size": "auto",
                        "short_side": 1024, "max_scale": 2.0})
    h, w = resolve_image_size(cfg)
    assert min(h, w) >= 320, (h, w)


def test_auto_on_real_1600x846(tmp_path):
    """真实尺寸 1600(宽) x 846(高): auto 必须给出 [832, 1600] 且不放大."""
    from PIL import Image

    from fabric_ncd.data.yolo import native_image_size, resolve_image_size

    d = tmp_path / "images" / "train"
    d.mkdir(parents=True)
    Image.new("RGB", (1600, 846), (128, 128, 128)).save(d / "a.jpg")

    class _Spec:
        splits = {"train": d}
        normal_splits: dict = {}

    assert native_image_size(_Spec) == (846, 1600)
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"image_size": "auto", "short_side": 1024, "max_scale": 1.0})
    assert resolve_image_size(cfg, spec=_Spec) == (832, 1600)


def test_size_census_prefers_dominant_size(tmp_path):
    """尺寸不统一时, auto 要按**最常见的**那一档算, 而不是第一张/中位数."""
    from PIL import Image

    from fabric_ncd.data.yolo import image_size_census, native_image_size, resolve_image_size

    d = tmp_path / "images" / "train"
    d.mkdir(parents=True)
    for i in range(6):  # 主流尺寸: 316x128 (1.89:1 的缩比)
        Image.new("RGB", (316, 128)).save(d / f"a{i}.jpg")
    for i in range(2):  # 少数派: 180x96
        Image.new("RGB", (180, 96)).save(d / f"b{i}.jpg")

    class _Spec:
        splits = {"train": d}
        normal_splits: dict = {}

    census = image_size_census(_Spec)
    assert list(census)[0] == (128, 316)  # 首项 = 最常见
    assert census[(128, 316)] == 6 and census[(96, 180)] == 2
    assert native_image_size(_Spec) == (128, 316)

    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update({"image_size": "auto", "short_side": 64, "max_scale": 2.0})
    h, w = resolve_image_size(cfg, spec=_Spec)
    assert w > h  # 跟随主流尺寸的横图长宽比
