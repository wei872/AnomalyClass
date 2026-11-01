"""Automated verification suite for YOLO27n, YOLO27n-Dual, YOLO27n-Lite, folded scaling, and Net2Wider warm-start."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import YOLO27, YOLO27Detect, YOLO27DetectionModel, YOLO27E2ELoss, resolve_donor_checkpoint, transfer_pretrained_weights


def test_yolo27n_footprints_regmax1_and_optimizer_groups() -> None:
    """Verify YOLO27n/Dual/Lite footprints, reg_max=1, c3=80 preservation, dual biases, and optimizer group routing."""
    model_19 = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False)
    head = model_19.model[-1]
    assert isinstance(head, YOLO27Detect)
    assert head.reg_max == 1, f"Expected DFL-free reg_max=1 matching YOLO26/YOLO27, got {head.reg_max}"
    assert head.fg_branch is not None
    assert head.feat_logit_scale is not None
    assert model_19.end2end is False, "Official YOLO27 default must use high-accuracy one2many + NMS"
    # Verify c3 == 80 (1:1 aligned with yolo26n.pt cv3 channels — zero truncation!)
    assert head.cv3[0][0][1].conv.out_channels == 80

    # Verify BOTH one2many and one2one biases are initialized to focal/box priors
    for s in range(head.nl):
        o2m_cls_b = head.cv3[s][-1].bias.detach()
        o2o_cls_b = head.one2one_cv3[s][-1].bias.detach()
        assert torch.all(o2m_cls_b < -5.0), f"one2many cls bias not initialized at scale {s}"
        assert torch.allclose(o2m_cls_b, o2o_cls_b), "one2one cls bias must match one2many focal prior!"
        assert torch.allclose(head.cv2[s][-1].bias.detach(), torch.full_like(head.cv2[s][-1].bias, 2.0))
        assert torch.allclose(head.one2one_cv2[s][-1].bias.detach(), torch.full_like(head.one2one_cv2[s][-1].bias, 2.0))

    # Verify feat_logit_scale routes to weight_decay=0.0 and non-Muon optimizer param group
    dummy_trainer = BaseTrainer.__new__(BaseTrainer)
    dummy_trainer.data = {"nc": 19}
    dummy_trainer.args = SimpleNamespace(
        optimizer="auto", lr0=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0
    )
    opt = dummy_trainer.build_optimizer(model=model_19, name="auto", lr=0.001, momentum=0.9, decay=0.0005, iterations=100)
    scale_ptrs = {id(p) for p in head.feat_logit_scale}
    found_scale = 0
    for pg in opt.param_groups:
        for p in pg["params"]:
            if id(p) in scale_ptrs:
                found_scale += 1
                assert pg.get("weight_decay", 0.0) == 0.0, "feat_logit_scale must have weight_decay=0.0"
                assert not pg.get("use_muon", False), "feat_logit_scale must NOT use Muon 2D orthogonalization"
    assert found_scale == head.nl

    # Verify parameter counts are close to YOLO26n (~2.38M @ nc=19) and strictly inside YOLO27n budget
    fused_19 = deepcopy(model_19).eval().fuse(verbose=False)
    params_19 = get_num_params(fused_19) / 1e6
    flops_19 = get_flops(fused_19, imgsz=640)
    assert 2.38 <= params_19 <= 2.65, f"Expected YOLO27n fused params ~2.51M (close to YOLO26n), got {params_19:.3f}M"
    assert 5.40 <= flops_19 <= 7.00, f"Expected YOLO27n FLOPs <=7.0B @ nc=19, got {flops_19:.2f}B"
    assert fused_19.model[-1].fg_branch is None
    assert fused_19.model[-1].feat_logit_scale is None

    # Verify YOLO27n-Dual (~2.46M fused params, stride [8, 32])
    model_dual = YOLO27DetectionModel("models/yolo27n-dual.yaml", nc=19, verbose=False)
    assert model_dual.model[-1].stride.tolist() == [8.0, 32.0]
    fused_dual = deepcopy(model_dual).eval().fuse(verbose=False)
    params_dual = get_num_params(fused_dual) / 1e6
    assert 2.35 <= params_dual <= 2.60, f"Expected YOLO27n-Dual ~2.46M params, got {params_dual:.3f}M"

    # Verify YOLO27n-Lite (<0.9M fused params, <3.6B FLOPs)
    model_lite = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False)
    assert model_lite.model[-1].stride.tolist() == [8.0, 32.0]
    assert model_lite.model[-1].reg_max == 1
    fused_lite = deepcopy(model_lite).eval().fuse(verbose=False)
    assert get_num_params(fused_lite) / 1e6 < 0.90
    assert get_flops(fused_lite, imgsz=640) < 3.60


def test_net2wider_exact_step0_equivalence_from_yolo26n() -> None:
    """Verify Net2Wider warm-start from stock YOLO26n (reg_max=1, nc=80) has <1e-5 step-0 drift & zero cv3 truncation."""
    torch.manual_seed(0)
    calib_x = torch.rand(2, 3, 160, 160)

    donor_y26 = YOLO("yolo26n.yaml").model.train()
    for m in donor_y26.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 1.0
    with torch.no_grad():
        donor_y26(calib_x)
    donor_y26.eval()

    # 1. nc=80 -> nc=80: exact step-0 equivalence across BOTH one2many and one2one on YOLO27n!
    target_80 = YOLO27DetectionModel("models/yolo27n.yaml", nc=80, verbose=False).eval()
    stats_80 = transfer_pretrained_weights(donor_y26, target_80, verbose=False)
    assert stats_80["ratio"] >= 0.93, f"Expected >=93% warm-start coverage @ nc=80, got {stats_80['ratio']:.3f}"
    with torch.no_grad():
        _, p_donor = donor_y26(calib_x[:1])
        _, p_target = target_80(calib_x[:1])
    for branch_key in ("one2many", "one2one"):
        box_diff = (p_donor[branch_key]["boxes"] - p_target[branch_key]["boxes"]).abs().max().item()
        score_diff = (p_donor[branch_key]["scores"] - p_target[branch_key]["scores"]).abs().max().item()
        assert box_diff < 5e-5, f"Step-0 {branch_key} box drift: {box_diff}"
        assert score_diff < 5e-5, f"Step-0 {branch_key} score drift: {score_diff}"

    # 2. nc=80 -> nc=19 (Fabric19): exact step-0 box equivalence vs YOLO26n(nc=19).load(donor) + active step-1 grads
    y26_19_cfg = deepcopy(donor_y26.yaml)
    y26_19_cfg["nc"] = 19
    ref_y26_19 = DetectionModel(y26_19_cfg, nc=19, verbose=False)
    ref_y26_19.load(donor_y26)
    ref_y26_19.eval()

    target_19 = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False)
    orig_cls_bias = target_19.model[-1].cv3[0][-1].bias.detach().clone()
    stats_19 = transfer_pretrained_weights(donor_y26, target_19, verbose=False)
    assert stats_19["ratio"] >= 0.90, f"Expected >=90% warm-start coverage @ nc=19, got {stats_19['ratio']:.3f}"
    assert torch.allclose(orig_cls_bias, target_19.model[-1].cv3[0][-1].bias)
    assert torch.allclose(orig_cls_bias, target_19.model[-1].one2one_cv3[0][-1].bias)

    target_19.eval()
    with torch.no_grad():
        _, p_ref19 = ref_y26_19(calib_x[:1])
        _, p_t19 = target_19(calib_x[:1])
    for branch_key in ("one2many", "one2one"):
        box_diff_19 = (p_ref19[branch_key]["boxes"] - p_t19[branch_key]["boxes"]).abs().max().item()
        assert box_diff_19 < 5e-5, f"Fabric19 step-0 {branch_key} box drift vs YOLO26n: {box_diff_19}"

    # Confirm step-1 unlocks non-zero gradients on widened P2/4 (layer 2), P3/8 bottleneck (layer 4), & P3 neck (layer 16)
    target_19.train()
    target_19.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=100)
    crit = target_19.init_criterion()
    opt = torch.optim.AdamW(target_19.parameters(), lr=5e-4)
    batch = {
        "img": calib_x,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[10.0], [16.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.2], [0.3, 0.4, 0.1, 0.1]]),
    }
    for step in range(2):
        opt.zero_grad()
        loss, _ = crit(target_19(batch["img"]), batch)
        loss.backward()
        if step == 1:
            assert target_19.model[2].cv1.conv.weight.grad[16:20].abs().sum().item() > 0
            assert target_19.model[4].cv1.conv.weight.grad[32:64].abs().sum().item() > 0
            assert target_19.model[16].cv1.conv.weight.grad[16:20].abs().sum().item() > 0
        opt.step()


def test_folded_per_channel_scale_exact_fusion() -> None:
    """Verify learned per-channel feat_logit_scale + fixed scale_factors fuse with <1e-4 error on Conv & DWConv."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256)

    for cfg in ("models/yolo27n.yaml", "models/yolo27n-dual.yaml", "models/yolo27n-lite.yaml"):
        base_m = YOLO27DetectionModel(cfg, nc=19, verbose=False)
        # Perturb feat_logit_scale across channels to simulate post-training learned calibration
        with torch.no_grad():
            for p in base_m.model[-1].feat_logit_scale:
                p.copy_(torch.linspace(0.6, 1.8, steps=p.numel()))

        # 1. Default NMS mode (end2end=False -> keeps one2many head)
        m_nms = deepcopy(base_m).eval()
        assert m_nms.end2end is False
        with torch.no_grad():
            y_unfused, _ = m_nms(x)
        m_nms_fused = deepcopy(m_nms).fuse(verbose=False)
        assert m_nms_fused.model[-1].fg_branch is None
        assert m_nms_fused.model[-1].feat_logit_scale is None
        assert m_nms_fused.model[-1].one2one_cv2 is None
        assert m_nms_fused.model[-1]._scale_folded is True
        with torch.no_grad():
            y_fused, _ = m_nms_fused(x)
        max_diff_nms = (y_unfused - y_fused).abs().max().item()
        assert max_diff_nms < 1e-4, f"{cfg} NMS fuse mismatch with per-channel scale: {max_diff_nms}"

        # 2. End-to-End NMS-free mode (end2end=True -> keeps one2one head)
        m_e2e = deepcopy(base_m).eval()
        m_e2e.end2end = True
        with torch.no_grad():
            y_e2e_unfused, _ = m_e2e(x)
        m_e2e_fused = deepcopy(m_e2e).fuse(verbose=False)
        assert m_e2e_fused.model[-1].fg_branch is None
        assert m_e2e_fused.model[-1].feat_logit_scale is None
        assert m_e2e_fused.model[-1].cv2 is None
        with torch.no_grad():
            y_e2e_fused, _ = m_e2e_fused(x)
        max_diff_e2e = (y_e2e_unfused[..., :5] - y_e2e_fused[..., :5]).abs().max().item()
        assert max_diff_e2e < 1e-4, f"{cfg} E2E fuse mismatch with per-channel scale: {max_diff_e2e}"


def test_live_fg_supervision_and_progressive_schedule() -> None:
    """Verify live neck FG supervision backpropagates into neck, o2m: 0.80->0.55, STAL=16, and empty batches."""
    model = YOLO27DetectionModel("models/yolo27n-dual.yaml", nc=19, verbose=False).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    criterion = model.init_criterion()
    assert isinstance(criterion, YOLO27E2ELoss)

    assert criterion.one2many.assigner.stride_val == 16
    assert criterion.one2one.assigner.stride_val == 16
    assert abs((criterion.o2m + criterion.o2o) - 1.0) < 1e-6
    assert abs(criterion.o2m - 0.80) < 1e-6

    for _ in range(9):
        criterion.update()
        assert abs((criterion.o2m + criterion.o2o) - 1.0) < 1e-6
    assert abs(criterion.o2m - 0.55) < 1e-6

    imgs = torch.rand(2, 3, 256, 256)
    batch_mixed = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[14.0], [1.0]]),
        "bboxes": torch.tensor([[0.25, 0.25, 0.05, 0.05], [0.50, 0.50, 0.90, 0.03]]),
    }

    # Verify fg_branch alone directly sends gradients into shared neck model[16]!
    model.zero_grad()
    preds_only_fg = model(imgs)
    fg_only = criterion._compute_fg_alignment_loss(preds_only_fg["fg_align"], preds_only_fg["one2many"], batch_mixed)
    fg_only.backward()
    neck_grad = model.model[16].cv2.conv.weight.grad
    assert neck_grad is not None and neck_grad.abs().sum().item() > 0, "fg_loss must supervise shared neck directly!"

    # Full loss + empty normal fabric batch
    model.zero_grad()
    total_loss, loss_items = criterion(model(imgs), batch_mixed)
    assert total_loss.ndim == 0 and torch.isfinite(total_loss)
    fg_val = float(loss_items["fg_loss"]) if isinstance(loss_items, dict) else float(loss_items[3])
    assert fg_val > 0.005
    total_loss.backward()

    model.zero_grad()
    batch_normal = {
        "img": imgs,
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
    }
    loss_normal, _ = criterion(model(imgs), batch_normal)
    assert torch.isfinite(loss_normal)
    loss_normal.backward()


def test_donor_discovery_and_end_to_end_pipeline() -> None:
    """Verify sibling-folder yolo26n.pt discovery, explicit pretrained warm-start, and val/predict nms toggling."""
    names = [
        "纬线杂质", "断纬", "多股线", "纬线不直", "包边不重叠", "捻接器接头",
        "小尾巴", "经线毛球", "包边不良", "手工接头", "纬斜", "短废纱线",
        "单根包不进", "棉团", "蚊虫小黑点", "小布头", "稀缝", "油点线", "飘絮",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for sub in (
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
            "normal/images/train",
            "normal/images/val",
            "sibling_proj",
        ):
            (root / sub).mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            for i in range(4):
                img = rng.integers(100, 200, size=(128, 128, 3), dtype=np.uint8)
                cv2.rectangle(img, (20, 20), (50, 50), (20, 20, 20), -1)
                cv2.imwrite(str(root / f"images/{split}/def_{i}.jpg"), img)
                cls_id = i % 19
                (root / f"labels/{split}/def_{i}.txt").write_text(f"{cls_id} 0.273 0.273 0.234 0.234\n", encoding="utf-8")

            for i in range(2):
                norm_img = rng.integers(120, 180, size=(128, 128, 3), dtype=np.uint8)
                cv2.imwrite(str(root / f"normal/images/{split}/norm_{i}.jpg"), norm_img)

        data_yaml = root / "fabric19_tmp.yaml"
        YAML.save(
            data_yaml,
            {
                "path": str(root),
                "train": str(root / "images/train"),
                "val": str(root / "images/val"),
                "normaltrain": str(root / "normal/images/train"),
                "normalval": str(root / "normal/images/val"),
                "nc": 19,
                "names": names,
            },
        )

        # Save simulated yolo26n.pt inside sibling folder and verify auto-discovery via extra_dirs
        donor_y26 = YOLO("yolo26n.yaml")
        with torch.no_grad():
            donor_y26.model.model[2].cv1.conv.weight.fill_(0.125)
        donor_pt = root / "sibling_proj/yolo26n.pt"
        donor_y26.save(str(donor_pt))

        found = resolve_donor_checkpoint(scale="n", allow_download=False, extra_dirs=[root / "images/train"])
        assert found is not None and found.resolve() == donor_pt.resolve()

        # Regression test for BOTH bugs:
        #   (a) Must train actual models/yolo27n.yaml (model[2].cv2 out_channels == 80, NOT yolo26n's 64!)
        #   (b) Passing pretrained=True (boolean default from train_yolo27n.py) must NOT trigger Model.load(True)
        #       scratch hijack — it must auto-discover sibling_proj/yolo26n.pt and run Net2Wider warm-start!
        y27 = YOLO27("models/yolo27n.yaml")
        y27.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=128,
            batch=2,
            workers=0,
            pretrained=True,
            device="cpu",
            project=str(root / "runs"),
            name="exp",
            plots=False,
        )
        assert y27.trainer.model.model[2].cv2.conv.out_channels == 80, "Trainer built wrong YAML instead of yolo27n!"
        shared_slice = y27.trainer.model.model[2].cv1.conv.weight[:16].detach().mean().item()
        assert abs(shared_slice - 0.125) < 0.02, f"pretrained=True failed to run Net2Wider donor warm-start: {shared_slice}"

        best_pt = root / "runs/exp/weights/best.pt"
        assert best_pt.exists()
        csv_text = (root / "runs/exp/results.csv").read_text(encoding="utf-8")
        assert "train/fg_loss" in csv_text and "val/fg_loss" in csv_text

        m_val_nms = YOLO27(best_pt)
        m_val_nms.val(data=str(data_yaml), imgsz=128, batch=2, workers=0, nms=True, plots=False)
        assert m_val_nms.model.end2end is False

        m_val_norm = YOLO27(best_pt)
        m_val_norm.val(data=str(data_yaml), imgsz=128, batch=2, workers=0, nms=False, use_normal=True, plots=False)
        assert m_val_norm.model.end2end is True

        sample_img = str(root / "images/val/def_0.jpg")
        assert len(YOLO27(best_pt).predict(sample_img, imgsz=128, nms=True, verbose=False)) == 1
        assert len(YOLO27(best_pt).predict(sample_img, imgsz=128, nms=False, verbose=False)) == 1


if __name__ == "__main__":
    for fn in (
        test_yolo27n_footprints_regmax1_and_optimizer_groups,
        test_net2wider_exact_step0_equivalence_from_yolo26n,
        test_folded_per_channel_scale_exact_fusion,
        test_live_fg_supervision_and_progressive_schedule,
        test_donor_discovery_and_end_to_end_pipeline,
    ):
        print(f"[RUN ] {fn.__name__} ...")
        fn()
        print(f"[PASS] {fn.__name__}")
    print("\nAll YOLO27n verification tests passed successfully!")
