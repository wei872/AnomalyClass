"""Automated verification suite for YOLO27n, YOLO27n-Lite, and fast-convergence fixes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import YOLO27, YOLO27Detect, YOLO27DetectionModel, YOLO27E2ELoss, transfer_pretrained_weights


def test_yolo27n_official_specs_and_dual_scale() -> None:
    """Verify YOLO27n matches official Ultralytics YOLO27n specs (~3.0M params, ~7.2B FLOPs @ nc=80) and dual-scale."""
    model_coco = YOLO27DetectionModel("models/yolo27n.yaml", nc=80, verbose=False)
    head = model_coco.model[-1]
    assert isinstance(head, YOLO27Detect)
    assert head.nl == 2, f"Expected 2 detection scales (P3/8, P5/32), got {head.nl}"
    assert head.stride.tolist() == [8.0, 32.0], f"Expected strides [8.0, 32.0], got {head.stride.tolist()}"
    assert head.fg_branch is not None, "Training model must carry ForegroundAlignmentBranch"

    fused_coco = deepcopy(model_coco).eval().fuse(verbose=False)
    params_m = get_num_params(fused_coco) / 1e6
    flops_b = get_flops(fused_coco, imgsz=640)
    assert 2.85 <= params_m <= 3.15, f"Expected ~3.0M params @ nc=80, got {params_m:.3f}M"
    assert 6.9 <= flops_b <= 7.5, f"Expected ~7.2B FLOPs @ nc=80, got {flops_b:.2f}B"
    assert fused_coco.model[-1].fg_branch is None, "fg_branch must be stripped on fuse()"


def test_yolo27n_lite_specs() -> None:
    """Verify YOLO27n-Lite ultra-lightweight variant builds and fuses cleanly (< 1.0M params, ~3.0 GFLOPs @ nc=19)."""
    model_lite = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False)
    fused_lite = deepcopy(model_lite).eval().fuse(verbose=False)
    params_m = get_num_params(fused_lite) / 1e6
    flops_b = get_flops(fused_lite, imgsz=640)
    assert params_m < 1.0, f"Expected <1.0M params for YOLO27n-Lite, got {params_m:.3f}M"
    assert flops_b < 3.5, f"Expected <3.5B FLOPs for YOLO27n-Lite, got {flops_b:.2f}B"


def test_smart_pretrained_weight_transfer() -> None:
    """Verify exact + channel-sliced warm-start from YOLO26n transfers >=70% params while keeping 19-cls focal bias."""
    donor = YOLO("yolo26n.yaml").model
    target = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False)
    orig_cls_bias = target.model[-1].cv3[0][-1].bias.detach().clone()

    stats = transfer_pretrained_weights(donor, target)
    assert stats["ratio"] >= 0.70, f"Expected >=70% warm-start coverage, got {stats['ratio']:.3f}"
    # Stem Conv must match donor 100% exactly
    assert torch.allclose(donor.model[0].conv.weight, target.model[0].conv.weight)
    # 19-class focal prior bias on cv3 must NOT be overwritten by 80-class donor
    assert torch.allclose(orig_cls_bias, target.model[-1].cv3[0][-1].bias)


def test_fuse_exact_equivalence_both_modes() -> None:
    """Verify fixed dual-scale feature scaling folds into Conv2d weights with exact numerical equivalence."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 320, 320)

    # 1. Default NMS mode (end2end=False -> keeps one2many head)
    m_nms = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False).eval()
    m_nms.end2end = False
    with torch.no_grad():
        y_unfused, _ = m_nms(x)
    m_nms_fused = deepcopy(m_nms).fuse(verbose=False)
    assert m_nms_fused.model[-1].fg_branch is None
    assert m_nms_fused.model[-1]._scale_folded is True
    with torch.no_grad():
        y_fused, _ = m_nms_fused(x)
    max_diff_nms = (y_unfused - y_fused).abs().max().item()
    assert max_diff_nms < 1e-4, f"NMS mode fuse mismatch: max_diff={max_diff_nms}"

    # 2. End-to-End NMS-free mode (end2end=True -> keeps one2one head)
    m_e2e = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False).eval()
    m_e2e.end2end = True
    with torch.no_grad():
        y_e2e_unfused, _ = m_e2e(x)
    m_e2e_fused = deepcopy(m_e2e).fuse(verbose=False)
    assert m_e2e_fused.model[-1].fg_branch is None
    assert m_e2e_fused.model[-1].cv2 is None
    with torch.no_grad():
        y_e2e_fused, _ = m_e2e_fused(x)
    max_diff_e2e = (y_e2e_unfused[..., :5] - y_e2e_fused[..., :5]).abs().max().item()
    assert max_diff_e2e < 1e-4, f"E2E mode fuse mismatch: max_diff={max_diff_e2e}"


def test_loss_convergence_fixes_and_empty_normal_batch() -> None:
    """Verify STAL stride_val=16 fix, constant o2m=1.0 schedule, and gradients on mixed + normal batches."""
    from types import SimpleNamespace

    model = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=6.0, epochs=100)
    criterion = model.init_criterion()
    assert isinstance(criterion, YOLO27E2ELoss)

    # Verify dual-scale STAL stride_val fix (16 px instead of 32 px!)
    assert criterion.one2many.assigner.stride_val == 16
    assert criterion.one2one.assigner.stride_val == 16
    assert criterion.o2m == 1.0

    # Verify o2m never decays below 1.0 while o2o ramps smoothly to 1.0
    for _ in range(99):
        criterion.update()
    assert criterion.o2m == 1.0
    assert abs(criterion.o2o - 1.0) < 1e-6

    # Case A: Mixed batch (image 0 has 2 small defects, image 1 is pure normal defect-free fabric)
    imgs = torch.rand(2, 3, 320, 320)
    batch_mixed = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[14.0], [7.0]]),  # 蚊虫小黑点, 经线毛球
        "bboxes": torch.tensor([[0.25, 0.25, 0.03, 0.03], [0.70, 0.60, 0.20, 0.15]]),
    }
    preds = model(imgs)
    assert "fg_logits" in preds
    assert preds["fg_logits"].shape == (2, 1, 1700)

    total_loss, loss_items = criterion(preds, batch_mixed)
    assert total_loss.ndim == 0 and torch.isfinite(total_loss)
    if isinstance(loss_items, dict):
        assert "fg_loss" in loss_items
        fg_val = float(loss_items["fg_loss"])
    else:
        assert loss_items.numel() == 4
        fg_val = float(loss_items[3])
    # Ensure fg_loss is O(0.1~2.0), NOT attenuated 6800x to ~1e-4 by VarifocalLoss.mean(1)
    assert fg_val > 0.05, f"fg_loss unexpectedly tiny ({fg_val}), check anchor normalization"
    total_loss.backward()

    head = model.model[-1]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.fg_branch.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.cv2.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.one2one_cv2.parameters())

    # Case B: Pure normal batch (0 defects across entire batch)
    model.zero_grad()
    batch_normal = {
        "img": imgs,
        "batch_idx": torch.zeros((0,), dtype=torch.float32),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 4), dtype=torch.float32),
    }
    preds_normal = model(imgs)
    loss_normal, _ = criterion(preds_normal, batch_normal)
    assert torch.isfinite(loss_normal)
    loss_normal.backward()


def test_end_to_end_train_val_predict_with_normaltrain() -> None:
    """Run an actual train -> val -> predict(nms=True/False) cycle on a synthetic 19-class fabric dataset."""
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
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": str(root / "images/train"),
                    "val": str(root / "images/val"),
                    "normaltrain": str(root / "normal/images/train"),
                    "normalval": str(root / "normal/images/val"),
                    "nc": 19,
                    "names": names,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        y27 = YOLO27("models/yolo27n-lite.yaml")
        y27.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=128,
            batch=2,
            workers=0,
            pretrained=False,
            device="cpu",
            project=str(root / "runs"),
            name="exp",
            plots=False,
        )
        best_pt = root / "runs/exp/weights/best.pt"
        assert best_pt.exists(), "Expected best.pt checkpoint to be saved"
        csv_text = (root / "runs/exp/results.csv").read_text(encoding="utf-8")
        assert "train/fg_loss" in csv_text and "val/fg_loss" in csv_text, f"Missing fg_loss in CSV: {csv_text}"

        # Validate both NMS (one2many) and E2E (one2one) pathways from saved best.pt
        m_val_nms = YOLO27(best_pt)
        m_val_nms.val(data=str(data_yaml), imgsz=128, batch=2, workers=0, nms=True, plots=False)
        assert m_val_nms.model.end2end is False

        m_val_e2e = YOLO27(best_pt)
        m_val_e2e.val(data=str(data_yaml), imgsz=128, batch=2, workers=0, nms=False, plots=False)
        assert m_val_e2e.model.end2end is True

        sample_img = str(root / "images/val/def_0.jpg")
        res_nms = YOLO27(best_pt).predict(sample_img, imgsz=128, nms=True, verbose=False)
        res_e2e = YOLO27(best_pt).predict(sample_img, imgsz=128, nms=False, verbose=False)
        assert len(res_nms) == 1 and len(res_e2e) == 1


if __name__ == "__main__":
    for fn in (
        test_yolo27n_official_specs_and_dual_scale,
        test_yolo27n_lite_specs,
        test_smart_pretrained_weight_transfer,
        test_fuse_exact_equivalence_both_modes,
        test_loss_convergence_fixes_and_empty_normal_batch,
        test_end_to_end_train_val_predict_with_normaltrain,
    ):
        print(f"[RUN ] {fn.__name__} ...")
        fn()
        print(f"[PASS] {fn.__name__}")
    print("\nAll YOLO27n verification tests passed successfully!")
