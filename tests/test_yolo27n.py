"""Automated verification suite for YOLO27n, YOLO27n-Lite, and zero-error Net2Wider warm-start."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import YOLO27, YOLO27Detect, YOLO27DetectionModel, YOLO27E2ELoss, transfer_pretrained_weights


def test_yolo27n_specs_and_lightweight_budget() -> None:
    """Verify YOLO27n stays lighter than official 3.0M / 7.2B budget and strips fg_branch on fuse()."""
    model_coco = YOLO27DetectionModel("models/yolo27n.yaml", nc=80, verbose=False)
    head = model_coco.model[-1]
    assert isinstance(head, YOLO27Detect)
    assert head.fg_branch is not None
    assert head.strip_adapters is not None

    fused_coco = deepcopy(model_coco).eval().fuse(verbose=False)
    params_m = get_num_params(fused_coco) / 1e6
    flops_b = get_flops(fused_coco, imgsz=640)
    assert 2.40 <= params_m <= 3.10, f"Expected <=3.0M params @ nc=80, got {params_m:.3f}M"
    assert 5.5 <= flops_b <= 7.3, f"Expected <=7.2B FLOPs @ nc=80, got {flops_b:.2f}B"
    assert fused_coco.model[-1].fg_branch is None, "fg_branch must be stripped on fuse()"


def test_yolo27n_lite_specs() -> None:
    """Verify YOLO27n-Lite dual-scale ultra-lightweight variant (<1.0M params, <3.5B FLOPs @ nc=19)."""
    model_lite = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False)
    assert model_lite.model[-1].stride.tolist() == [8.0, 32.0]
    fused_lite = deepcopy(model_lite).eval().fuse(verbose=False)
    params_m = get_num_params(fused_lite) / 1e6
    flops_b = get_flops(fused_lite, imgsz=640)
    assert params_m < 1.0, f"Expected <1.0M params for YOLO27n-Lite, got {params_m:.3f}M"
    assert flops_b < 3.5, f"Expected <3.5B FLOPs for YOLO27n-Lite, got {flops_b:.2f}B"


def test_zero_error_net2wider_pretrained_warmstart() -> None:
    """Verify function-preserving Net2Wider warm-start matches yolo26n to <1e-4 at step 0 and unlocks step-1 grads."""
    torch.manual_seed(0)
    donor = YOLO("yolo26n.yaml").model.train()
    for m in donor.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 1.0
    calib_x = torch.rand(2, 3, 160, 160)
    with torch.no_grad():
        donor(calib_x)
    donor.eval()

    # 1. Step-0 exact functional equivalence @ same nc
    target_same = YOLO27DetectionModel("models/yolo27n.yaml", nc=80, verbose=False).eval()
    stats = transfer_pretrained_weights(donor, target_same)
    assert stats["ratio"] >= 0.88, f"Expected >=88% warm-start coverage, got {stats['ratio']:.3f}"
    with torch.no_grad():
        _, p26 = donor(calib_x[:1])
        _, p27 = target_same(calib_x[:1])
    box_diff = (p26["one2one"]["boxes"] - p27["one2one"]["boxes"]).abs().max().item()
    score_diff = (p26["one2one"]["scores"] - p27["one2one"]["scores"]).abs().max().item()
    assert box_diff < 1e-4, f"Step-0 box outputs drifted from donor: max_diff={box_diff}"
    assert score_diff < 1e-4, f"Step-0 score outputs drifted from donor: max_diff={score_diff}"

    # 2. Target nc=19 preserves 19-class focal bias & unlocks widened + strip gradients on step 1
    target_19 = YOLO27DetectionModel("models/yolo27n.yaml", nc=19, verbose=False).train()
    orig_cls_bias = target_19.model[-1].cv3[0][-1].bias.detach().clone()
    transfer_pretrained_weights(donor, target_19)
    assert torch.allclose(orig_cls_bias, target_19.model[-1].cv3[0][-1].bias)

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
            assert target_19.model[2].cv1.conv.weight.grad[16:24].abs().sum().item() > 0
            assert target_19.model[-1].strip_adapters[0].dw_h.weight.grad.abs().sum().item() > 0
            assert target_19.model[-1].strip_adapters[0].dw_v.weight.grad.abs().sum().item() > 0
        opt.step()


def test_fuse_exact_equivalence_both_modes() -> None:
    """Verify fixed feature scaling + strip enhancers fuse with exact numerical equivalence."""
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


def test_loss_progloss_schedule_and_empty_normal_batch() -> None:
    """Verify STAL stride_val=16, balanced ProgLoss (o2m+o2o=1.0), fg_loss scale, and empty batches."""
    model = YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=19, verbose=False).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=100)
    criterion = model.init_criterion()
    assert isinstance(criterion, YOLO27E2ELoss)

    assert criterion.one2many.assigner.stride_val == 16
    assert criterion.one2one.assigner.stride_val == 16
    assert abs((criterion.o2m + criterion.o2o) - 1.0) < 1e-6

    for _ in range(99):
        criterion.update()
    assert abs(criterion.o2m - 0.2) < 1e-5
    assert abs(criterion.o2o - 0.8) < 1e-5

    imgs = torch.rand(2, 3, 320, 320)
    batch_mixed = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[14.0], [7.0]]),
        "bboxes": torch.tensor([[0.25, 0.25, 0.03, 0.03], [0.70, 0.60, 0.20, 0.15]]),
    }
    preds = model(imgs)
    total_loss, loss_items = criterion(preds, batch_mixed)
    assert total_loss.ndim == 0 and torch.isfinite(total_loss)
    fg_val = float(loss_items["fg_loss"]) if isinstance(loss_items, dict) else float(loss_items[3])
    assert fg_val > 0.02, f"fg_loss unexpectedly tiny ({fg_val})"
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


def test_end_to_end_train_val_predict_both_protocols() -> None:
    """Verify train -> val(nms=True/False) under default yolo26n-matched protocol & use_normal=True."""
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

        y27 = YOLO27("models/yolo27n.yaml")
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
        test_yolo27n_specs_and_lightweight_budget,
        test_yolo27n_lite_specs,
        test_zero_error_net2wider_pretrained_warmstart,
        test_fuse_exact_equivalence_both_modes,
        test_loss_progloss_schedule_and_empty_normal_batch,
        test_end_to_end_train_val_predict_both_protocols,
    ):
        print(f"[RUN ] {fn.__name__} ...")
        fn()
        print(f"[PASS] {fn.__name__}")
    print("\nAll YOLO27n verification tests passed successfully!")
