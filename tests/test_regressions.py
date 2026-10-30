"""训练、数据、推理和指标的关键回归测试。"""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fabric_ncd.data.augment import random_scale_crop
from fabric_ncd.data.demo import make_dataset
from fabric_ncd.data.synth import apply_random_anomaly
from fabric_ncd.data.yolo import DatasetSpec, FabricDataset, build_reference_loaders, load_dataset_spec
from fabric_ncd.engine import ModelEMA, Trainer
from fabric_ncd.inference import FabricPredictor, forward_maps
from fabric_ncd.losses import FabricCriterion, SupConLoss, dice_loss, pooled_region_embeddings
from fabric_ncd.metrics import ari, average_precision, classification_report, nmi, per_region_overlap
from fabric_ncd.models.backbone import LiteBackbone, TorchVisionBackbone
from fabric_ncd.models.fabric_ncd import FabricNCD
from fabric_ncd.models.memory import MemoryBank
from fabric_ncd.ncd import RegionAnalyzer, RegionInfo
from fabric_ncd.utils import load_model_weights, load_yaml, set_seed


def config(size=(64, 64)):
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"].update(image_size=list(size), num_workers=0, cloth_mask="full")
    cfg["model"]["backbone"].update(pretrained=False, freeze_stages=0)
    cfg["model"]["heads"]["num_classes"] = 3
    cfg["train"]["scale_sweep"] = []
    return cfg


def test_memory_stride_pools_exactly_once():
    torch.manual_seed(7)
    bank = MemoryBank(capacity=5, dim=3, query_stride=2)
    bank.update(torch.randn(5, 3))
    query = F.normalize(torch.randn(2, 3, 9, 13), dim=1)
    actual = bank.similarity(query)
    bank.query_stride = 1
    pooled = F.normalize(F.avg_pool2d(query, 2, 2, ceil_mode=True), dim=1)
    expected = F.interpolate(bank.similarity(pooled), size=query.shape[-2:], mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected)


def test_pretrained_backbone_is_not_reinitialized(monkeypatch):
    import fabric_ncd.models.fabric_ncd as model_module

    class FakeBackbone(nn.Module):
        out_channels = {"p2": 4, "p3": 4, "p4": 4, "p5": 4}

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
            self.bn = nn.BatchNorm2d(4)
            with torch.no_grad():
                self.conv.weight.fill_(0.123)
                self.bn.weight.fill_(0.456)
                self.bn.bias.fill_(0.789)

    backbone = FakeBackbone()
    monkeypatch.setattr(model_module, "build_backbone", lambda cfg: backbone)
    FabricNCD(config())
    torch.testing.assert_close(backbone.conv.weight, torch.full_like(backbone.conv.weight, 0.123))
    torch.testing.assert_close(backbone.bn.weight, torch.full_like(backbone.bn.weight, 0.456))
    torch.testing.assert_close(backbone.bn.bias, torch.full_like(backbone.bn.bias, 0.789))


def test_frozen_batchnorm_stays_frozen_during_training():
    model = TorchVisionBackbone(pretrained=False, image_size=64)
    model.freeze_stages(1)
    bn = next(m for m in model.features[0].modules() if isinstance(m, nn.BatchNorm2d))
    old_mean = bn.running_mean.clone()
    model.train()
    assert not bn.training
    model(torch.randn(2, 3, 64, 64))
    torch.testing.assert_close(bn.running_mean, old_mean)


def test_lite_backbone_p5_is_stride_32():
    backbone = LiteBackbone().eval()
    with torch.no_grad():
        out = backbone(torch.randn(1, 3, 128, 160))
    assert [out[k].shape[-2:] for k in ("p2", "p3", "p4", "p5")] == [
        (32, 40), (16, 20), (8, 10), (4, 5)
    ]


def test_region_supcon_propagates_gradients():
    torch.manual_seed(11)
    raw = torch.randn(4, 8, 4, 4, requires_grad=True)
    emb = F.normalize(raw, dim=1)
    boxes = [torch.tensor([[float(c), 0.1, 0.1, 0.9, 0.9]]) for c in (0, 0, 1, 1)]
    regions, labels = pooled_region_embeddings(emb, boxes, (4, 4))
    loss = SupConLoss()(regions, labels)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_dice_and_pseudo_labels_ignore_masked_pixels():
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[..., :2, :] = 1
    targets = torch.ones_like(logits)
    value = dice_loss(logits, targets, mask=mask)
    value.backward()
    assert torch.count_nonzero(logits.grad[..., 2:, :]) == 0

    criterion = FabricCriterion({"loss": {
        "anomaly": {"focal_weight": 0, "dice_weight": 1, "pseudo_ratio": 0.5},
        "supcon": {"weight": 0}, "compact": {"weight": 0},
    }})
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    outputs = {"anomaly_logit": logits, "memory_distance": torch.arange(16.).reshape(1, 1, 4, 4),
               "class_logits": torch.zeros(1, 2, 4, 4), "embedding": torch.zeros(1, 2, 4, 4)}
    target = {"defect_mask": torch.ones_like(logits), "cloth_mask": torch.ones_like(logits),
              "boxes": [torch.empty(0, 5)]}
    loss, _ = criterion(outputs, target)
    loss.backward()
    assert torch.count_nonzero(logits.grad.flatten()[:8]) == 0
    assert torch.count_nonzero(logits.grad.flatten()[8:]) == 8


def test_fabric_dice_and_bce_weights_are_independent():
    target = {"defect_mask": torch.zeros(1, 1, 4, 4), "cloth_mask": torch.ones(1, 1, 4, 4),
              "boxes": [torch.empty(0, 5)]}
    cloth = torch.zeros(1, 1, 4, 4)
    outputs = {"anomaly_logit": torch.zeros_like(cloth), "cloth_logit": cloth,
               "class_logits": torch.zeros(1, 2, 4, 4), "embedding": torch.zeros(1, 2, 4, 4)}
    for bce_weight, dice_weight in [(1, 0), (0, 1)]:
        criterion = FabricCriterion({"loss": {"fabric": {"bce_weight": bce_weight, "dice_weight": dice_weight},
                                               "supcon": {"weight": 0}, "compact": {"weight": 0}}})
        _, logs = criterion(outputs, target)
        expected = (F.binary_cross_entropy_with_logits(cloth, target["cloth_mask"]) if bce_weight
                    else dice_loss(cloth, target["cloth_mask"]))
        assert logs["loss_fabric"] == pytest.approx(float(expected))


def test_compact_loss_excludes_defect_pixels_and_backpropagates():
    bank = MemoryBank(capacity=2, dim=2)
    bank.update(torch.tensor([[1., 0.]]))
    emb = torch.tensor([[[[0.3, 0.3], [0.3, 0.3]], [[0.7, 0.7], [0.7, 0.7]]]], requires_grad=True)
    mask = torch.zeros(1, 1, 2, 2)
    mask[..., 0, 0] = 1
    outputs = {"anomaly_logit": torch.zeros_like(mask), "class_logits": torch.zeros(1, 2, 2, 2),
               "embedding": emb}
    target = {"defect_mask": mask, "cloth_mask": torch.ones_like(mask), "boxes": [torch.empty(0, 5)]}
    criterion = FabricCriterion({"loss": {"anomaly": {"focal_weight": 0, "dice_weight": 0},
                                           "supcon": {"weight": 0}, "compact": {"weight": 1}}})
    loss, logs = criterion(outputs, target, bank)
    assert logs["compact"] > 0
    loss.backward()
    assert torch.count_nonzero(emb.grad[..., 0, 0]) == 0
    assert emb.grad[..., 1, 1].abs().sum() > 0


def test_metrics_count_missed_images_and_tied_scores():
    report = classification_report([0, 1], [0, -1], 2)
    assert report["acc"] == pytest.approx(0.5)
    assert report["f1_macro"] == pytest.approx(0.5)
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert nmi([0, 0], [1, 1]) == pytest.approx(1)
    assert ari([0, 0], [1, 1]) == pytest.approx(1)


def test_aupro_perfect_detection_is_one():
    gt = np.zeros((16, 16), np.float32)
    gt[4:8, 4:8] = 1
    scores = 0.1 + 0.8 * gt
    assert per_region_overlap([scores], [gt], [np.full_like(gt, 0.1)]) == pytest.approx(1)


def test_empty_labels_normal_and_all_validation_images_are_used(tmp_path):
    images, labels, normal = (tmp_path / p for p in ("images", "labels", "normal"))
    for directory in (images, labels, normal):
        directory.mkdir()
    for name in ("defect", "empty", "missing"):
        Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(images / f"{name}.png")
    (labels / "defect.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (labels / "empty.txt").write_text(" \n")
    for i in range(4):
        Image.fromarray(np.full((32, 32, 3), 120, np.uint8)).save(normal / f"n{i}.png")
    spec = DatasetSpec(None, {"val": images}, {"val": labels}, {"val": normal}, ["defect"], 1)
    ds = FabricDataset(spec, "val", (32, 32), train=False,
                       augment_cfg={"image_size": (32, 32), "cloth_mask": "full", "weave_norm": False})
    assert len(ds) == 7
    assert sum(ds[i]["normal"] for i in range(len(ds))) == 6
    assert all(len(ds[i]["boxes"]) == 0 for i, e in enumerate(ds.entries) if e.normal)


def test_nested_same_name_labels_do_not_get_mixed_up(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    for sub, cls in (("a", 0), ("b", 1)):
        (images / sub).mkdir(parents=True)
        (labels / sub).mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(images / sub / "same.png")
        (labels / sub / "same.txt").write_text(f"{cls} 0.5 0.5 0.3 0.3\n")
    spec = DatasetSpec(None, {"val": images}, {"val": labels}, {"val": None}, ["a", "b"], 2)
    ds = FabricDataset(spec, "val", (32, 32), train=False,
                       augment_cfg={"image_size": (32, 32), "cloth_mask": "full", "weave_norm": False})
    assert [int(ds[i]["labels"][0]) for i in range(2)] == [0, 1]


def test_root_remap_preserves_normal_prefix_and_overrides_existing_paths(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        (root / "images" / "train").mkdir(parents=True)
        (root / "normal" / "images" / "train").mkdir(parents=True)
    yml = tmp_path / "dataset.yaml"
    yml.write_text(f"path: {old}\ntrain: {old}/images/train\nnormaltrain: {old}/normal/images/train\nnc: 1\nnames: {{0: defect}}\n")
    spec = load_dataset_spec({"data": {"dataset_yaml": str(yml), "root": str(new)}})
    assert spec.splits["train"] == new / "images" / "train"
    assert spec.normal_splits["train"] == new / "normal" / "images" / "train"
    assert spec.names == ["defect"]


def test_both_reference_sources_are_used(tmp_path):
    root = make_dataset(tmp_path / "data", n_train=3, n_val=2, n_normal=3, size=(64, 64))
    cfg = config()
    cfg["data"].update(dataset_yaml=str(root / "dataset.yaml"), eval_batch_size=2)
    cfg["train"]["memory_source"] = "both"
    loaders = build_reference_loaders(cfg, "train")
    assert len(loaders) == 2
    cfg["experiment"].update(device="cpu", amp=False, output_dir=str(tmp_path / "run"))
    trainer = Trainer(cfg)
    assert len(trainer.memory_loaders) == 2


def test_augmentation_respects_seed_and_cloth_region(tmp_path):
    root = make_dataset(tmp_path / "data", n_train=2, n_val=1, n_normal=1, size=(64, 64))
    cfg = config()
    cfg["data"].update(dataset_yaml=str(root / "dataset.yaml"))
    from fabric_ncd.data.yolo import build_datasets
    dataset = build_datasets(cfg)["train"]
    set_seed(42)
    a = dataset[0]
    set_seed(42)
    b = dataset[0]
    torch.testing.assert_close(a["image"], b["image"])
    torch.testing.assert_close(a["anomaly_mask"], b["anomaly_mask"])

    image = np.full((64, 64, 3), 128, np.uint8)
    region = np.zeros((64, 64), np.uint8)
    region[16:48, 16:48] = 1
    aug, mask = apply_random_anomaly(image, np.zeros_like(region), np.random.default_rng(4),
                                     types=["stain"], region=region)
    assert np.array_equal(aug[region == 0], image[region == 0])
    assert not mask[region == 0].any()


def test_scale_crop_larger_than_image_preserves_box_coordinates():
    sample = {"image": np.zeros((32, 40, 3), np.uint8), "cloth_mask": np.ones((32, 40), np.uint8),
              "defect_mask": np.ones((32, 40), np.uint8),
              "boxes": np.array([[0, 0.2, 0.3, 0.7, 0.8]], np.float32)}
    out = random_scale_crop(sample, (1.2, 1.2), np.random.default_rng(0))
    np.testing.assert_allclose(out["boxes"], sample["boxes"])
    assert out["image"].shape[:2] == (32, 40)


def test_regions_do_not_mix_enclosed_components_or_misaligned_class_map():
    score = np.zeros((24, 24), np.float32)
    score[2:22, 2:22] = 0.9
    score[4:20, 4:20] = 0
    score[9:13, 9:13] = 0.9
    probs = np.zeros((2, 24, 24), np.float32)
    probs[0] = 1
    probs[:, 9:13, 9:13] = np.array([0., 1.])[:, None, None]
    analyzer = RegionAnalyzer(threshold=0.5, smooth=1, open_radius=0, bbox_refine_ratio=0,
                              min_area_ratio=0, peak_ratio=0)
    regions, _ = analyzer.analyze(score, np.ones_like(score), probs)
    assert len(regions) == 2
    assert [r.cls for r in regions] == [0, 1]
    assert regions[0].area == pytest.approx((20 * 20 - 16 * 16) / (24 * 24))

    score = np.zeros((15, 21), np.float32)
    score[3:10, 5:13] = 0.9
    rng = np.random.default_rng(5)
    coarse = rng.random((2, 8, 11), dtype=np.float32)
    coarse /= coarse.sum(axis=0, keepdims=True)
    regions, _ = analyzer.analyze(score, np.ones_like(score), coarse, probs_scale=2)
    ys = ((np.arange(3, 10) + .5) * 8 / 15).astype(int)
    xs = ((np.arange(5, 13) + .5) * 11 / 21).astype(int)
    np.testing.assert_allclose(regions[0].probs, coarse[:, ys[:, None], xs].mean(axis=(1, 2)), rtol=1e-6)


def test_merge_regions_with_missing_probabilities():
    analyzer = RegionAnalyzer(merge_strategy="area_weighted")
    regions = [RegionInfo((0, 0, 1, 1), 1, probs=None),
               RegionInfo((0, 0, 1, 1), .5, probs=np.array([.2, .8]), area=0.2)]
    np.testing.assert_allclose(analyzer.merge_region_scores(regions), [0.2, 0.8])
    analyzer.merge_strategy = "top1"
    np.testing.assert_allclose(analyzer.merge_region_scores(regions), [0.2, 0.8])


class TileModel:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, x, weave_desc=None):
        self.batch_sizes.append(len(x))
        if weave_desc is not None:
            assert len(weave_desc) == len(x)
        n = len(x)
        logits = torch.empty(n, 2, 2, 2)
        for i in range(n):
            logits[i, 0].fill_(4 if i % 2 == 0 else -4)
            logits[i, 1].fill_(-4 if i % 2 == 0 else 4)
        emb = torch.tensor([-0.6, 0.8]).view(1, 2, 1, 1).expand(n, 2, 2, 2)
        return {"anomaly_score": torch.full((n, 1, 2, 2), 0.7),
                "cloth_prob": torch.full((n, 1, 2, 2), 0.8),
                "class_logits": logits, "embedding": emb, "weave": {}}


def test_tiled_maps_cover_odd_image_and_keep_probabilities_and_negative_embeddings():
    model = TileModel()
    maps = forward_maps(model, np.zeros((105, 157, 3), np.uint8), torch.device("cpu"), (32, 48),
                        tile=True, tile_size=(40, 55), tile_overlap=16, tile_batch=2,
                        want_embedding=True, out_stride=4, weave_desc=torch.zeros(1, 6), weave_stats={})
    np.testing.assert_allclose(maps["score"], 0.7, rtol=1e-6)
    np.testing.assert_allclose(maps["probs"].sum(axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(maps["embedding"][0], -0.6, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(maps["embedding"], axis=0), 1, atol=1e-6)
    assert len(model.batch_sizes) > 1 and max(model.batch_sizes) <= 2
    with pytest.raises(ValueError, match="tile_overlap"):
        forward_maps(model, np.zeros((105, 157, 3), np.uint8), torch.device("cpu"), (32, 32),
                     tile=True, tile_size=(32, 32), tile_overlap=32, weave_desc=torch.zeros(1, 6))


def test_legacy_border_prior_weights_migrate_without_changing_outputs():
    old_cfg = config()
    old_cfg["model"]["fabric_mask"].update(border_prior=False, use_xy=True)
    new_cfg = copy.deepcopy(old_cfg)
    new_cfg["model"]["fabric_mask"]["border_prior"] = True
    old = FabricNCD(old_cfg).eval()
    new = FabricNCD(new_cfg).eval()
    load_model_weights(new, old.state_dict())
    mask = torch.rand(1, 1, 8, 8)
    feat = torch.rand(1, 96, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(old.geography(mask, feat), new.geography(mask, feat))
    incompatible = copy.deepcopy(new_cfg)
    incompatible["model"]["heads"]["num_classes"] = 4
    with pytest.raises(RuntimeError, match="checkpoint"):
        load_model_weights(FabricNCD(incompatible), old.state_dict())


def test_missing_checkpoint_fails_before_building_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        FabricPredictor(checkpoint=tmp_path / "missing.pt", device="cpu")


def test_resume_restores_ema_and_update_count(tmp_path):
    source = nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(2)
    ema = ModelEMA(source)
    with torch.no_grad():
        ema.ema.weight.fill_(7)
    opt = torch.optim.AdamW(source.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model": source.state_dict(), "ema": ema.ema.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": 3, "best_score": 0.4,
                "global_step": 19, "ema_updates": 17}, checkpoint)
    trainer = object.__new__(Trainer)
    trainer.model = nn.Linear(2, 2)
    trainer.ema = ModelEMA(trainer.model)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1)
    trainer.scaler = None
    trainer.device = torch.device("cpu")
    trainer.logger = logging.getLogger("resume-test")
    trainer.load_checkpoint(checkpoint, load_optimizer=True)
    torch.testing.assert_close(trainer.model.weight, torch.full_like(trainer.model.weight, 2))
    torch.testing.assert_close(trainer.ema.ema.weight, torch.full_like(trainer.model.weight, 7))
    assert (trainer.start_epoch, trainer.global_step, trainer.ema.num_updates) == (4, 19, 17)


def test_old_lite_checkpoint_preserves_its_stride(tmp_path):
    cfg = config()
    cfg["model"]["backbone"].update(arch="lite", legacy_p5_stride64=True)
    original = FabricNCD(cfg).eval()
    old_config = copy.deepcopy(cfg)
    del old_config["model"]["backbone"]["legacy_p5_stride64"]
    path = tmp_path / "legacy.pt"
    torch.save({"version": 1, "config": old_config, "model": original.state_dict()}, path)
    predictor = FabricPredictor(checkpoint=path, device="cpu")
    assert predictor.model.backbone.legacy_p5_stride64
    x, desc = torch.randn(1, 3, 64, 64), torch.zeros(1, 6)
    with torch.no_grad():
        old_out = original(x, compute_memory=False, weave_desc=desc)["anomaly_score"]
        new_out = predictor.model(x, compute_memory=False, weave_desc=desc)["anomaly_score"]
    torch.testing.assert_close(old_out, new_out)


def test_gabor_supports_nonstandard_filter_counts():
    from fabric_ncd.models.texture import TextureEncoder

    encoder = TextureEncoder(4, 8, orientations=[0, 30, 60, 90, 120],
                             wavelengths=[4], use_imaginary=True).eval()
    with torch.no_grad():
        out = encoder(torch.rand(1, 3, 32, 32), torch.rand(1, 4, 8, 8))
    assert out.shape == (1, 8, 8, 8)
    with pytest.raises(ValueError, match="奇数"):
        TextureEncoder(4, 8, kernel_size=8)


def test_weave_estimator_preserves_axis_aligned_spectral_components():
    from fabric_ncd.data.weave import estimate_period

    h, w = 320, 480
    y, x = np.mgrid[:h, :w]
    gray = 128 + 40 * np.sin(2 * np.pi * x / 6) + 40 * np.sin(2 * np.pi * y / 10)
    image = np.repeat(gray.clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)
    px, py, confidence = estimate_period(image)
    assert confidence >= 2
    assert px == pytest.approx(6, abs=0.2)
    assert py == pytest.approx(10, abs=0.2)


def test_novel_cluster_is_not_reported_as_a_known_match():
    from fabric_ncd.ncd import discover_classes

    features = np.array([[1, 0], [0.99, 0.01]], dtype=np.float32)
    report = discover_classes(features, np.array([[0, 1]], np.float32), ["known"], k=1,
                              novelty_threshold=0.8)
    assert report["assignments"][0]["novel"]
    assert report["assignments"][0]["known_index"] == -1
    assert report["assignments"][0]["name"] == "novel_0"
    empty = discover_classes(np.empty((0, 2), np.float32), None, k=2)
    assert empty["k"] == 0 and empty["novel_prototypes"].shape == (0, 2)


def test_ema_keeps_its_own_memory_bank_when_embedding_has_changed():
    cfg = config()
    cfg["model"]["memory"]["capacity"] = 128
    model = FabricNCD(cfg)
    ema = ModelEMA(model)
    ema.num_updates = 1
    with torch.no_grad():
        ema.ema.embed_head.block[-1][1].bias.add_(1.0)
    trainer = object.__new__(Trainer)
    trainer.model, trainer.ema = model, ema
    trainer.device = torch.device("cpu")
    trainer.cfg, trainer.train_cfg = cfg, {}
    trainer.logger = logging.getLogger("memory-ema-test")
    trainer.memory_loaders = [[{"image": torch.randn(2, 3, 64, 64),
                                "cloth_mask": torch.ones(2, 1, 64, 64), "path": ["a", "b"]}]]
    trainer.warmup_memory(calibrate=False)
    assert model.memory.is_ready and ema.ema.memory.is_ready
    assert not torch.allclose(model.memory.features, ema.ema.memory.features)
    before = ema.ema.memory.features.clone()
    ema.update(model)
    torch.testing.assert_close(ema.ema.memory.features, before)


def test_both_reference_sources_are_interleaved_even_with_batch_limit():
    from fabric_ncd.engine import _interleave

    assert list(_interleave([["defect1", "defect2", "defect3"], ["normal1", "normal2"]]))[:2] == [
        "defect1", "normal1"
    ]
    trainer = object.__new__(Trainer)
    trainer.memory_loaders = [
        [{"path": ["d1"]}, {"path": ["d2"]}],
        [{"path": ["n1"]}, {"path": ["n2"]}],
    ]
    trainer._bank_paths = set()
    assert [batch["path"][0] for batch in trainer._reference_batches()][:2] == ["d1", "n1"]


def test_aupro_works_without_normal_images_and_counts_tiny_regions():
    score = np.zeros((8, 8), np.float32)
    mask = np.zeros_like(score)
    mask[1, 1] = 1
    mask[4:6, 4:6] = 1
    score[mask > 0] = 1
    assert per_region_overlap([score], [mask], []) == pytest.approx(1)


def test_background_reference_excludes_synthetic_anomalies():
    from fabric_ncd.data.yolo import DefectBackgroundDataset

    sample = {"image": torch.zeros(3, 10, 10), "cloth_mask": torch.ones(1, 10, 10),
              "defect_mask": torch.zeros(1, 10, 10), "anomaly_mask": torch.zeros(1, 10, 10),
              "boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long),
              "path": "x", "normal": False}
    sample["anomaly_mask"][0, 3:5, 3:5] = 1
    background = object.__new__(DefectBackgroundDataset)
    background.base, background.box_pad = [sample], 0
    out = background[0]
    assert out["cloth_mask"][0, 3:5, 3:5].sum() == 0
    assert out["cloth_mask"][0, 0, 0] == 1


def test_compact_loss_excludes_cells_even_if_only_one_defect_pixel():
    memory = MemoryBank(capacity=4, dim=2)
    memory.update(torch.tensor([[1.0, 0.0]]))
    anomaly = torch.zeros(1, 1, 4, 4)
    anomaly[0, 0, 0, 0] = 1
    outputs = {"anomaly_logit": torch.zeros(1, 1, 2, 2, requires_grad=True),
               "class_logits": torch.zeros(1, 2, 2, 2),
               "embedding": torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)}
    targets = {"defect_mask": anomaly, "cloth_mask": torch.ones_like(anomaly),
               "boxes": [torch.zeros(0, 5)]}
    crit = FabricCriterion({"loss": {"cls": {"weight": 0}, "supcon": {"weight": 0},
                                      "compact": {"weight": 1}}})
    _, logs = crit(outputs, targets, memory)
    assert "compact" not in logs


def test_dataset_name_dictionary_retains_numeric_class_indices(tmp_path):
    import yaml
    from fabric_ncd.data.yolo import load_dataset_spec

    file = tmp_path / "dataset.yaml"
    file.write_text(yaml.safe_dump({"path": str(tmp_path), "nc": 3, "names": {0: "first", 2: "third"}}))
    spec = load_dataset_spec({"data": {"dataset_yaml": str(file)}})
    assert spec.names == ["first", "class_1", "third"]


def test_invalid_yolo_labels_fail_with_location(tmp_path):
    from fabric_ncd.data.yolo import read_yolo_labels

    label = tmp_path / "bad.txt"
    for text in ("0 nan 0.5 0.2 0.3", "1.5 0.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.1", "0 nonsense"):
        label.write_text(text)
        with pytest.raises(ValueError, match="bad.txt:1"):
            read_yolo_labels(label)


def test_discovery_features_preserve_region_alignment(monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from tools.discover import _region_features

    maps = {"score": np.ones((8, 8), np.float32), "embedding": np.ones((2, 8, 8), np.float32)}
    regions = [{"bbox": [0, 0, 0, 0]}, {"bbox": [0.25, 0.25, 0.75, 0.75]}]
    features = _region_features(maps, regions, 0.5)
    assert len(features) == 2 and features[0] is None
    np.testing.assert_allclose(features[1], [2 ** -0.5, 2 ** -0.5], atol=1e-6)
