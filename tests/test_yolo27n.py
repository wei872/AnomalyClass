"""Automated verification suite for YOLO27n/Dual/Lite high-IoU localization, warm-start, and v8.4.33+ compatibility."""

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
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import (
    YOLO27,
    YOLO27BboxLoss,
    YOLO27Detect,
    YOLO27DetectionModel,
    YOLO27E2ELoss,
    YOLO27Trainer,
    apply_defect_invariant_bg_augment,
    resolve_donor_checkpoint,
    transfer_pretrained_weights,
)


def test_yolo27n_footprints_box_tower_and_musgd_guard() -> None:
    """Verify footprints, 1:1 pretrained c2=16/c3=80 head alignment, fixed scaling, and >10k-iter MuSGD protection."""
    model_19 = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False)
    head = model_19.model[-1]
    assert isinstance(head, YOLO27Detect)
    assert head.reg_max == 1, f"Expected DFL-free reg_max=1, got {head.reg_max}"
    assert head.c2_width == 16 and head.cv2[0][0].conv.out_channels == 16
    assert head.cv3[0][0][1].conv.out_channels == 80
    assert head.fg_branch is not None
    assert not hasattr(head, "feat_logit_scale") or getattr(head, "feat_logit_scale") is None
    assert model_19.end2end is False

    # Verify BOTH one2many and one2one biases are initialized to focal/box priors
    for s in range(head.nl):
        o2m_cls_b = head.cv3[s][-1].bias.detach()
        o2o_cls_b = head.one2one_cv3[s][-1].bias.detach()
        assert torch.all(o2m_cls_b < -5.0)
        assert torch.allclose(o2m_cls_b, o2o_cls_b)
        assert torch.allclose(head.cv2[s][-1].bias.detach(), torch.full_like(head.cv2[s][-1].bias, 2.0))
        assert torch.allclose(head.one2one_cv2[s][-1].bias.detach(), torch.full_like(head.one2one_cv2[s][-1].bias, 2.0))

    # Verify YOLO27Trainer.build_optimizer prevents MuSGD shock at iterations=25000 when warm-started!
    dummy_trainer = YOLO27Trainer.__new__(YOLO27Trainer)
    dummy_trainer._allow_warmstart = True
    dummy_trainer.data = {"nc": 19}
    dummy_trainer.args = SimpleNamespace(
        optimizer="auto", lr0=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0
    )
    opt = dummy_trainer.build_optimizer(model=model_19, name="auto", lr=0.01, momentum=0.9, decay=0.0005, iterations=25000)
    assert isinstance(opt, torch.optim.AdamW), f"Expected AdamW instead of MuSGD on warm-start, got {type(opt)}"

    # Verify fused parameter counts & FLOPs stay compact and auxiliary FG branch is stripped
    fused_19 = deepcopy(model_19).eval().fuse(verbose=False)
    params_19 = get_num_params(fused_19) / 1e6
    flops_19 = get_flops(fused_19, imgsz=640)
    assert 2.38 <= params_19 <= 2.60, f"Expected YOLO27n fused params ~2.50M, got {params_19:.3f}M"
    assert 5.40 <= flops_19 <= 7.00, f"Expected YOLO27n FLOPs <=7.0B @ nc=19, got {flops_19:.2f}B"
    assert fused_19.model[-1].fg_branch is None
    assert fused_19.model[-1].cv2 is not None and fused_19.model[-1].one2one_cv2 is None

    # Verify YOLO27n-Dual (~2.45M fused params, stride [8, 32])
    model_dual = YOLO27DetectionModel("models/yolo27n-dual.yaml", nc=19, verbose=False)
    assert model_dual.model[-1].stride.tolist() == [8.0, 32.0]
    assert model_dual.model[-1].cv2[0][0].conv.out_channels == 16
    fused_dual = deepcopy(model_dual).eval().fuse(verbose=False)
    params_dual = get_num_params(fused_dual) / 1e6
    assert 2.35 <= params_dual <= 2.55, f"Expected YOLO27n-Dual ~2.45M params, got {params_dual:.3f}M"

    # Verify YOLO27n-Lite (<0.9M fused params, <3.6B FLOPs)
    model_lite = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False)
    assert model_lite.model[-1].stride.tolist() == [8.0, 32.0]
    assert model_lite.model[-1].cv2[0][0].conv.out_channels == 16
    fused_lite = deepcopy(model_lite).eval().fuse(verbose=False)
    assert get_num_params(fused_lite) / 1e6 < 0.90
    assert get_flops(fused_lite, imgsz=640) < 3.60


def test_net2wider_exact_step0_equivalence_with_widened_c2() -> None:
    """Verify Net2Wider warm-start from YOLO26n (c2=16 -> c2=32) has <5e-5 step-0 drift & step-1 active grads."""
    torch.manual_seed(0)
    calib_x = torch.rand(2, 3, 160, 160)

    donor_y26 = YOLO("yolo26n.yaml").model.train()
    for m in donor_y26.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 1.0
    with torch.no_grad():
        donor_y26(calib_x)
    donor_y26.eval()

    # 1. nc=80 -> nc=80: >=93% 1:1 pretrained coverage & exact step-0 equivalence across BOTH one2many and one2one!
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

    # 2. nc=80 -> nc=19 (Fabric19): >=90% pretrained coverage & exact step-0 box equivalence vs YOLO26n(nc=19)
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

    # Confirm step-1 unlocks non-zero gradients on widened backbone & neck stages
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


def test_version_safe_fusion_and_end2end_setter() -> None:
    """Verify fixed scale folding, idempotent head.fuse() (8.4.33 safety), and model.end2end setter."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256)

    for cfg in ("models/yolo27n.yaml", "models/yolo27n-dual.yaml", "models/yolo27n-lite.yaml"):
        base_m = YOLO27DetectionModel(cfg, nc=19, verbose=False)

        # 1. Default NMS mode (end2end=False -> keeps one2many head, never deletes cv2/cv3!)
        m_nms = deepcopy(base_m).eval()
        assert m_nms.end2end is False
        with torch.no_grad():
            y_unfused, _ = m_nms(x)
        m_nms_fused = deepcopy(m_nms).fuse(verbose=False)
        # Call head.fuse() a second time to verify idempotence across 8.4.33 & 8.4.197
        m_nms_fused.model[-1].fuse()
        assert m_nms_fused.model[-1].fg_branch is None
        assert m_nms_fused.model[-1].cv2 is not None
        assert m_nms_fused.model[-1].one2one_cv2 is None
        assert m_nms_fused.model[-1]._scale_folded is True
        with torch.no_grad():
            y_fused, _ = m_nms_fused(x)
        max_diff_nms = (y_unfused - y_fused).abs().max().item()
        assert max_diff_nms < 1e-4, f"{cfg} NMS fuse mismatch: {max_diff_nms}"

        # 2. End-to-End NMS-free mode (end2end=True -> keeps one2one head)
        m_e2e = deepcopy(base_m).eval()
        m_e2e.end2end = True
        assert m_e2e.model[-1].end2end is True
        with torch.no_grad():
            y_e2e_unfused, _ = m_e2e(x)
        m_e2e_fused = deepcopy(m_e2e).fuse(verbose=False)
        m_e2e_fused.model[-1].fuse()
        assert m_e2e_fused.model[-1].fg_branch is None
        assert m_e2e_fused.model[-1].cv2 is None
        assert m_e2e_fused.model[-1].one2one_cv2 is not None
        with torch.no_grad():
            y_e2e_fused, _ = m_e2e_fused(x)
        max_diff_e2e = (y_e2e_unfused[..., :5] - y_e2e_fused[..., :5]).abs().max().item()
        assert max_diff_e2e < 1e-4, f"{cfg} E2E fuse mismatch: {max_diff_e2e}"


def test_anisotropic_edge_loss_and_reused_fg_supervision() -> None:
    """Verify YOLO27BboxLoss sharpens thin-defect edges and fg_align reuses o2m_assign on 8.4.33 & >=8.4.197."""
    # 1. Verify YOLO27BboxLoss gives much stronger vertical edge supervision on a 560x8 px weft break than stock L1
    stock_bbox = YOLO27BboxLoss(reg_max=1, edge_gain=0.0)
    aniso_bbox = YOLO27BboxLoss(reg_max=1, edge_gain=0.25, min_edge_px=24.0)

    anchor_pts = torch.tensor([[40.0, 40.0]])  # stride-8 grid units -> (320, 320) px
    stride_t = torch.tensor([[8.0]])
    imgsz_t = torch.tensor([640.0, 640.0])
    # GT weft strip: width=560 px (x: 40..600 -> 5..75), thickness=8 px (y: 316..324 -> 39.5..40.5)
    target_boxes = torch.tensor([[[5.0, 39.5, 75.0, 40.5]]])
    # Prediction off by 2.4 px vertically (0.3 grid cells on top & bottom -> IoU drops to ~0.62!)
    pred_dist_stock = torch.tensor([[[35.0, 0.8, 35.0, 0.8]]], requires_grad=True)
    pred_dist_aniso = torch.tensor([[[35.0, 0.8, 35.0, 0.8]]], requires_grad=True)
    pred_boxes = torch.tensor([[[5.0, 39.2, 75.0, 40.8]]])
    target_scores = torch.ones((1, 1, 1))
    fg_mask = torch.tensor([[True]])

    _, l_dfl_stock = stock_bbox(
        pred_dist_stock, pred_boxes, anchor_pts, target_boxes, target_scores, torch.tensor(1.0), fg_mask, imgsz_t, stride_t
    )
    _, l_dfl_aniso = aniso_bbox(
        pred_dist_aniso, pred_boxes, anchor_pts, target_boxes, target_scores, torch.tensor(1.0), fg_mask, imgsz_t, stride_t
    )
    l_dfl_stock.backward()
    l_dfl_aniso.backward()
    assert pred_dist_aniso.grad[0, 0, 1].abs().item() > 4.0 * pred_dist_stock.grad[0, 0, 1].abs().item()

    # 2. Verify live FG supervision, tail-class positive-only BCE, & legacy 8.4.33 3-tuple compatibility
    model = YOLO27DetectionModel("models/yolo27n-dual.yaml", nc=19, verbose=False).train()
    model._pos_class_weights = torch.linspace(1.0, 1.40, steps=19)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    criterion = model.init_criterion()
    assert isinstance(criterion, YOLO27E2ELoss)
    assert isinstance(criterion.one2many.bbox_loss, YOLO27BboxLoss)
    assert criterion.one2many.assigner.stride_val == 16
    # Verify negative background BCE is strictly 1.0x while rare-class positive BCE is boosted 1.40x
    dummy_logits = torch.zeros((1, 2, 19))
    dummy_targets = torch.zeros((1, 2, 19))
    dummy_targets[0, 0, 18] = 1.0
    bce_out = criterion.one2many.bce(dummy_logits, dummy_targets)
    assert abs(bce_out[0, 1, 18].item() / bce_out[0, 1, 0].item() - 1.0) < 1e-6
    assert abs(bce_out[0, 0, 18].item() / bce_out[0, 1, 0].item() - 1.40) < 1e-5

    imgs = torch.rand(2, 3, 256, 256)
    batch_mixed = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[14.0], [1.0]]),
        "bboxes": torch.tensor([[0.25, 0.25, 0.05, 0.05], [0.50, 0.50, 0.90, 0.03]]),
    }

    model.zero_grad()
    preds = model(imgs)
    assert isinstance(preds["fg_align"], dict)
    assert set(preds["fg_align"].keys()) == {"logits", "pert_logits", "embed", "pert_embed"}
    fg_only = criterion._compute_fg_alignment_loss(preds["fg_align"], preds["one2many"], batch_mixed)
    fg_only.backward()
    assert model.model[16].cv2.conv.weight.grad.abs().sum().item() > 0

    # Simulate legacy Ultralytics <=8.4.33 3-tuple o2m_assign = (fg_mask, target_scores, target_scores_sum)
    n_anc = preds["fg_align"]["logits"].shape[1]
    fake_fg_mask = torch.zeros((2, n_anc), dtype=torch.bool)
    fake_fg_mask[0, :4] = True
    fake_t_scores = torch.zeros((2, n_anc, 19))
    fake_t_scores[0, :4, 3] = 0.85
    legacy_fg_dict = {k: v.detach().requires_grad_(True) for k, v in preds["fg_align"].items()}
    legacy_fg_loss = criterion._compute_fg_alignment_from_o2m(
        legacy_fg_dict,
        preds["one2many"],
        (fake_fg_mask, fake_t_scores, fake_t_scores.sum()),
        has_gt=True,
    )
    legacy_fg_loss.backward()
    assert torch.isfinite(legacy_fg_loss) and legacy_fg_loss.item() > 0.0
    assert legacy_fg_dict["pert_logits"].grad.abs().sum().item() > 0.0
    assert legacy_fg_dict["pert_embed"].grad.abs().sum().item() > 0.0


def test_foreground_alignment_and_defect_invariant_bg_augment() -> None:
    """Verify FG alignment spec (YOLO26n=None; YOLO27n/Dual/Lite=Train-Only, Stripped on Fuse) & BG-only augment."""
    torch.manual_seed(7)

    # 1. YOLO26n has NO fg_branch; YOLO27n / YOLO27n-Dual / YOLO27n-Lite ALL enable fg_branch in training & strip on fuse
    m_y26 = YOLO("yolo26n.yaml").model
    assert getattr(m_y26.model[-1], "fg_branch", None) is None

    for cfg in ("models/yolo27n.yaml", "models/yolo27n-dual.yaml", "models/yolo27n-lite.yaml"):
        m_train = YOLO27DetectionModel(cfg, nc=19, verbose=False).train()
        head = m_train.model[-1]
        assert head.fg_branch is not None, f"{cfg} must enable ForegroundAlignmentBranch during training"
        m_train.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
        crit = m_train.init_criterion()

        x = torch.rand(2, 3, 192, 192)
        out = m_train(x)
        assert isinstance(out["fg_align"], dict)
        assert out["fg_align"]["embed"].shape[-1] == 16
        # Verify counterfactual background perturbation changes features while anchoring fg_prob=1 centers
        f0 = torch.randn(2, 16, 12, 12)
        fg_ones = torch.ones(2, 1, 12, 12)
        fg_zeros = torch.zeros(2, 1, 12, 12)
        assert (head.fg_branch._perturb_background_features(f0, fg_ones) - f0).abs().max().item() == 0.0
        assert (head.fg_branch._perturb_background_features(f0, fg_zeros) - f0).abs().mean().item() > 0.05

        batch = {
            "img": x,
            "batch_idx": torch.tensor([0.0, 1.0]),
            "cls": torch.tensor([[14.0], [1.0]]),
            "bboxes": torch.tensor([[0.30, 0.30, 0.08, 0.08], [0.50, 0.60, 0.80, 0.05]]),
        }
        loss, items = crit(out, batch)
        loss.backward()
        assert torch.isfinite(loss)
        fg_val = items["fg_loss"].item() if isinstance(items, dict) else items[3].item()
        assert fg_val > 0.0

        m_fused = deepcopy(m_train).eval().fuse(verbose=False)
        assert m_fused.model[-1].fg_branch is None, f"{cfg} must strip ForegroundAlignmentBranch on fuse()"

    # 2. Defect-invariant cloth background perturbation: background changes, GT defect pixels stay 100% bit-exact!
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 128), torch.linspace(0, 1, 128), indexing="ij")
    weave_a = (0.5 + 0.16 * torch.sin(xx * 90.0) + 0.12 * torch.cos(yy * 70.0)).clamp(0.0, 1.0)
    weave_b = (0.5 + 0.18 * torch.sin((xx + yy) * 55.0) + 0.10 * torch.cos(xx * 130.0)).clamp(0.0, 1.0)
    weave_skew = (0.5 + 0.15 * torch.sin((xx + 0.2 * yy) * 80.0)).clamp(0.0, 1.0)
    imgs = torch.stack([weave_a.expand(3, -1, -1), weave_b.expand(3, -1, -1), weave_skew.expand(3, -1, -1)], dim=0)
    # Stamp synthetic dark defect spots into sample 0 (32..56, 40..64) and thin strip into sample 1 (60..68, 16..112)
    imgs[0, :, 32:56, 40:64] = 0.08
    imgs[1, :, 60:68, 16:112] = 0.92

    bboxes = torch.tensor(
        [
            [52.0 / 128.0, 44.0 / 128.0, 24.0 / 128.0, 24.0 / 128.0],  # sample 0 localized defect
            [64.0 / 128.0, 64.0 / 128.0, 96.0 / 128.0, 8.0 / 128.0],   # sample 1 thin weft defect
            [0.50, 0.50, 0.96, 0.96],                                  # sample 2 full-frame weft skew (纬斜)
        ]
    )
    batch_idx = torch.tensor([0.0, 1.0, 2.0])
    aug = apply_defect_invariant_bg_augment(imgs, bboxes=bboxes, batch_idx=batch_idx, p=1.0)

    # GT defect interiors in samples 0 & 1 are bit-for-bit untouched (疵点不变)
    assert (aug[0, :, 32:56, 40:64] - imgs[0, :, 32:56, 40:64]).abs().max().item() == 0.0
    assert (aug[1, :, 60:68, 16:112] - imgs[1, :, 60:68, 16:112]).abs().max().item() == 0.0
    # Background cloth weave outside defects genuinely changes (背景发生改变)
    assert (aug[0, :, :20, :28] - imgs[0, :, :20, :28]).abs().mean().item() > 0.01
    assert (aug[1, :, :44, :] - imgs[1, :, :44, :]).abs().mean().item() > 0.01
    # Full-frame global defect (sample 2, 纬斜 w*h >= 0.55) is automatically skipped without distorting cloth skew
    assert (aug[2] - imgs[2]).abs().max().item() == 0.0

    # Verify YOLO27Trainer.preprocess_batch gates BG augmentation off cleanly during final close_mosaic epochs
    dummy_t = YOLO27Trainer.__new__(YOLO27Trainer)
    dummy_t.device = torch.device("cpu")
    dummy_t._bg_augment_prob = 1.0
    dummy_t.model = SimpleNamespace(training=True)
    dummy_t.epochs = 150
    dummy_t.args = SimpleNamespace(multi_scale=0.0, close_mosaic=5)

    raw_uint8 = (imgs[:2] * 255.0).round().to(torch.uint8)
    dummy_t.epoch = 10
    b_active = dummy_t.preprocess_batch({"img": raw_uint8.clone(), "bboxes": bboxes[:2], "batch_idx": batch_idx[:2]})
    dummy_t.epoch = 147  # inside final close_mosaic=5 window
    b_tail = dummy_t.preprocess_batch({"img": raw_uint8.clone(), "bboxes": bboxes[:2], "batch_idx": batch_idx[:2]})
    assert (b_active["img"][0, :, 32:56, 40:64] - b_tail["img"][0, :, 32:56, 40:64]).abs().max().item() == 0.0
    assert (b_active["img"][0, :, :20, :28] - b_tail["img"][0, :, :20, :28]).abs().mean().item() > 0.01


def test_deep_donor_discovery_failfast_guard_and_pipeline() -> None:
    """Verify multi-level nested donor discovery, fail-fast missing-donor guard, --scratch, and train/val/predict."""
    names = [
        "纬线杂质", "断纬", "多股线", "纬线不直", "包边不重叠", "捻接器接头",
        "小尾巴", "经线毛球", "包边不良", "手工接头", "纬斜", "短废纱线",
        "单根包不进", "棉团", "蚊虫小黑点", "小布头", "稀缝", "油点线", "飘絮",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Simulate Windows layout: dataset at root/yolo/dataset/source5, donor nested 2 levels deep in sibling!
        ds_dir = root / "yolo/dataset/source5"
        nested_donor_dir = root / "yolo/other_yolo26_proj/sub_weights"
        for sub in (
            ds_dir / "images/train",
            ds_dir / "images/val",
            ds_dir / "labels/train",
            ds_dir / "labels/val",
            ds_dir / "normal/images/train",
            ds_dir / "normal/images/val",
            nested_donor_dir,
        ):
            sub.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            for i in range(4):
                img = rng.integers(100, 200, size=(128, 128, 3), dtype=np.uint8)
                cv2.rectangle(img, (20, 20), (50, 50), (20, 20, 20), -1)
                cv2.imwrite(str(ds_dir / f"images/{split}/def_{i}.jpg"), img)
                cls_id = i % 19
                (ds_dir / f"labels/{split}/def_{i}.txt").write_text(
                    f"{cls_id} 0.273 0.273 0.234 0.234\n", encoding="utf-8"
                )
            for i in range(2):
                norm_img = rng.integers(120, 180, size=(128, 128, 3), dtype=np.uint8)
                cv2.imwrite(str(ds_dir / f"normal/images/{split}/norm_{i}.jpg"), norm_img)

        data_yaml = ds_dir / "fabric19_tmp.yaml"
        YAML.save(
            data_yaml,
            {
                "path": str(ds_dir),
                "train": str(ds_dir / "images/train"),
                "val": str(ds_dir / "images/val"),
                "normaltrain": str(ds_dir / "normal/images/train"),
                "normalval": str(ds_dir / "normal/images/val"),
                "nc": 19,
                "names": names,
            },
        )

        # 1. Verify fail-fast guard raises FileNotFoundError instead of silent scratch training when donor is absent!
        isolated_dir = root / "empty_island/a/b/c/d"
        isolated_dir.mkdir(parents=True, exist_ok=True)
        isolated_yaml = isolated_dir / "iso.yaml"
        YAML.save(
            isolated_yaml,
            {
                "path": str(isolated_dir),
                "train": str(ds_dir / "images/train"),
                "val": str(ds_dir / "images/val"),
                "nc": 19,
                "names": names,
            },
        )
        # Before creating donor_pt anywhere under root, verify fail-fast raises FileNotFoundError:
        raised = False
        try:
            YOLO27("models/yolo27n.yaml").train(
                data=str(isolated_yaml),
                epochs=1,
                imgsz=128,
                batch=2,
                workers=0,
                pretrained=True,
                allow_download=False,
                device="cpu",
                project=str(root / "runs_fail"),
                name="exp_fail",
                plots=False,
            )
        except FileNotFoundError as exc:
            raised = "yolo26n.pt" in str(exc)
        assert raised, "Missing donor with pretrained=True must fail fast instead of training from scratch!"

        # 2. Now save simulated yolo26n.pt 2 levels deep under root/yolo/other_yolo26_proj/sub_weights/yolo26n.pt
        donor_y26 = YOLO("yolo26n.yaml")
        with torch.no_grad():
            donor_y26.model.model[2].cv1.conv.weight.fill_(0.125)
        donor_pt = nested_donor_dir / "yolo26n.pt"
        donor_y26.save(str(donor_pt))

        found = resolve_donor_checkpoint(scale="n", allow_download=False, extra_dirs=[ds_dir])
        assert found is not None and found.resolve() == donor_pt.resolve()

        # 3. Verify full train -> val(nms=True/False) -> predict(nms=True/False) pipeline with warm-start!
        y27 = YOLO27("models/yolo27n.yaml")
        y27.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=128,
            batch=2,
            workers=0,
            pretrained=True,
            allow_download=False,
            device="cpu",
            project=str(root / "runs"),
            name="exp",
            plots=False,
        )
        assert y27.trainer.model.model[2].cv2.conv.out_channels == 80
        assert y27.trainer.args.cos_lr is True
        assert y27.trainer.args.close_mosaic == 5
        # Verify small-dataset ModelEMA calibration (decay at update 100 >> stock tau=2000's 0.048!)
        assert y27.trainer.ema.decay(100) > 0.65
        shared_slice = y27.trainer.model.model[2].cv1.conv.weight[:16].detach().mean().item()
        assert abs(shared_slice - 0.125) < 0.02, f"Expected Net2Wider warm-start ~0.125, got {shared_slice}"

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

        sample_img = str(ds_dir / "images/val/def_0.jpg")
        assert len(YOLO27(best_pt).predict(sample_img, imgsz=128, nms=True, verbose=False)) == 1
        assert len(YOLO27(best_pt).predict(sample_img, imgsz=128, nms=False, verbose=False)) == 1


if __name__ == "__main__":
    for fn in (
        test_yolo27n_footprints_box_tower_and_musgd_guard,
        test_net2wider_exact_step0_equivalence_with_widened_c2,
        test_version_safe_fusion_and_end2end_setter,
        test_anisotropic_edge_loss_and_reused_fg_supervision,
        test_foreground_alignment_and_defect_invariant_bg_augment,
        test_deep_donor_discovery_failfast_guard_and_pipeline,
    ):
        print(f"[RUN ] {fn.__name__} ...")
        fn()
        print(f"[PASS] {fn.__name__}")
    print("\nAll YOLO27n verification tests passed successfully!")
