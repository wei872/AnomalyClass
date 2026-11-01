#!/usr/bin/env python3
"""Unified CLI for training, validating, profiling, and exporting YOLO27n on Fabric19.

Usage examples:
    # 1. Architecture & Complexity Profile (YOLO26n vs YOLO27n vs YOLO27n-Dual vs YOLO27n-Lite)
    python train_yolo27n.py --mode profile

    # 2. Train YOLO27n (Recommended highest mAP50-95 variant; auto-discovers yolo26n.pt or specify --donor)
    python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml \
        --epochs 150 --imgsz 640 --batch 16

    # Explicit pretrained donor path if yolo26n.pt is stored on another drive/folder:
    python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml \
        --donor D:/tyhn/yolo/yolo26n.pt --epochs 150 --imgsz 640 --batch 16

    # 3. Train YOLO27n-Dual (Pure Official YOLO27 Dual-Scale [P3/8, P5/32])
    python train_yolo27n.py --mode train --model models/yolo27n-dual.yaml --data dataset/fabric19.yaml \
        --epochs 150 --imgsz 640 --batch 16

    # 4. Train YOLO27n-Lite (Extreme edge lightweight ~0.79M fused params)
    python train_yolo27n.py --mode train --model models/yolo27n-lite.yaml --data dataset/fabric19.yaml \
        --epochs 150 --imgsz 640 --batch 32

    # 5. Validate (default: One-to-Many + NMS for maximum mAP50-95; add --no-nms for E2E One-to-One)
    python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --nms
    python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --no-nms
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import time
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import YOLO27, YOLO27DetectionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO27n Fabric19 Training, Validation & Profiling Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=("train", "val", "predict", "export", "profile"),
        help="Execution mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo27n.yaml",
        help="Model YAML path (models/yolo27n.yaml | models/yolo27n-dual.yaml | models/yolo27n-lite.yaml)",
    )
    parser.add_argument("--weights", type=str, default="", help="Trained .pt checkpoint path for val/predict/export")
    parser.add_argument(
        "--donor",
        "--pretrained",
        dest="donor",
        type=str,
        default="",
        help="Explicit path to pretrained yolo26n.pt donor for Net2Wider warm-start",
    )
    parser.add_argument("--data", type=str, default="dataset/fabric19.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="CUDA device, e.g. '0' or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizer ('auto' selects calibrated AdamW on warm-start, avoiding MuSGD oscillation)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=5,
        help="Final un-mosaicked epochs (5 prevents small-dataset overfitting spike while refining full-frame boxes)",
    )
    parser.add_argument(
        "--bg-augment-prob",
        type=float,
        default=0.50,
        help="Per-sample probability of defect-invariant cloth background perturbation in training (0.0 disables)",
    )
    parser.add_argument("--project", type=str, default="runs/detect", help="Output root directory")
    parser.add_argument("--name", type=str, default="yolo27n_fabric19", help="Experiment run name")
    parser.add_argument("--source", type=str, default="", help="Image/directory source for predict mode")
    parser.add_argument("--format", type=str, default="onnx", help="Target export format (onnx, engine, torchscript)")
    parser.add_argument(
        "--use-normal",
        action="store_true",
        help="Include normaltrain/normalval defect-free fabric samples to suppress weave false alarms",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train strictly from scratch without Net2Wider pretrained warm-start",
    )

    nms_group = parser.add_mutually_exclusive_group()
    nms_group.add_argument(
        "--nms",
        dest="nms",
        action="store_true",
        default=None,
        help="Use One-to-Many head + NMS post-processing (default for val/predict)",
    )
    nms_group.add_argument(
        "--no-nms",
        dest="nms",
        action="store_false",
        help="Use End-to-End One-to-One NMS-free inference (equivalent to nms=False)",
    )
    return parser.parse_args()


def _measure_latency_ms(model: torch.nn.Module, imgsz: int = 640, runs: int = 20) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / runs * 1000.0
    return elapsed


def run_profile(imgsz: int = 640, nc: int = 19) -> None:
    """Compare parameter count, FLOPs, scales, and latency across YOLO26n and YOLO27n variants."""
    y26_cfg = deepcopy(YOLO("yolo26n.yaml").model.yaml)
    y26_cfg["nc"] = nc

    candidates: list[tuple[str, Any]] = [
        ("YOLO26n (基准)", lambda: DetectionModel(y26_cfg, nc=nc, verbose=False)),
        ("YOLO27n (精度增强版)", lambda: YOLO27DetectionModel("models/yolo27n.yaml", nc=nc, verbose=False)),
        ("YOLO27n-Dual (双尺度版)", lambda: YOLO27DetectionModel("models/yolo27n-dual.yaml", nc=nc, verbose=False)),
        ("YOLO27n-Lite (极轻版)", lambda: YOLO27DetectionModel("models/yolo27n-lite.yaml", nc=nc, verbose=False)),
    ]

    header = (
        f"{'Model Variant':<24} | {'Train Params':>12} | {'Fused Params':>12} | "
        f"{'FLOPs @' + str(imgsz):>11} | {'Scales':<10} | {'FG Align':<15} | {'Latency (ms)':>12}"
    )
    print("\n" + "=" * len(header))
    print(f"Fabric19 (nc={nc}) Architecture Complexity & Inference Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, builder in candidates:
        m_train = builder()
        has_fg_train = getattr(m_train.model[-1], "fg_branch", None) is not None
        train_params = get_num_params(m_train)
        m_infer = deepcopy(m_train).eval()
        m_infer.fuse(verbose=False)
        has_fg_infer = getattr(m_infer.model[-1], "fg_branch", None) is not None
        fg_status = "训练启用/推理剥离" if (has_fg_train and not has_fg_infer) else ("有" if has_fg_infer else "无")
        fused_params = get_num_params(m_infer)
        flops = get_flops(m_infer, imgsz=imgsz)
        strides = [int(s) for s in m_train.stride.tolist()]
        scale_str = ",".join(f"P{int(torch.log2(torch.tensor(float(s))).item())}" for s in strides)
        lat_ms = _measure_latency_ms(m_infer, imgsz=imgsz)
        print(
            f"{label:<24} | {train_params / 1e6:>10.3f} M | {fused_params / 1e6:>10.3f} M | "
            f"{flops:>9.2f} B | {scale_str:<10} | {fg_status:<11} | {lat_ms:>9.2f} ms"
        )
    print("=" * len(header) + "\n")


def main() -> None:
    args = parse_args()

    if args.mode == "profile":
        run_profile(imgsz=args.imgsz)
        return

    target_model = args.weights if args.weights else args.model
    model = YOLO27(target_model)

    if args.mode == "train":
        train_kwargs: dict[str, Any] = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "optimizer": args.optimizer,
            "cos_lr": True,
            "close_mosaic": args.close_mosaic,
            "bg_augment_prob": args.bg_augment_prob,
            "project": args.project,
            "name": args.name,
            "use_normal": args.use_normal,
            "pretrained": not args.scratch,
        }
        if args.donor:
            train_kwargs["donor_weights"] = args.donor
        if args.device:
            train_kwargs["device"] = args.device
        model.train(**train_kwargs)

    elif args.mode == "val":
        nms_flag = True if args.nms is None else args.nms
        val_kwargs: dict[str, Any] = {
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "nms": nms_flag,
            "use_normal": args.use_normal,
        }
        if args.device:
            val_kwargs["device"] = args.device
        model.val(**val_kwargs)

    elif args.mode == "predict":
        if not args.source:
            raise ValueError("--source is required when --mode predict is selected.")
        nms_flag = True if args.nms is None else args.nms
        pred_kwargs: dict[str, Any] = {
            "source": args.source,
            "imgsz": args.imgsz,
            "nms": nms_flag,
            "save": True,
        }
        if args.device:
            pred_kwargs["device"] = args.device
        model.predict(**pred_kwargs)

    elif args.mode == "export":
        exp_kwargs: dict[str, Any] = {
            "format": args.format,
            "imgsz": args.imgsz,
            "nms": args.nms,
        }
        if args.device:
            exp_kwargs["device"] = args.device
        path = model.export(**exp_kwargs)
        print(f"Exported model saved to: {path}")


if __name__ == "__main__":
    main()
