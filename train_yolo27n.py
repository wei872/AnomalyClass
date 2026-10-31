#!/usr/bin/env python3
"""YOLO27n / YOLO27n-Lite training, validation, inference, profiling & export script for fabric19.yaml.

Usage examples:
  1. Profile model parameters, FLOPs & structure:
       python train_yolo27n.py --mode profile

  2. Train YOLO27n (apples-to-apples with pure yolo26n + zero-error Net2Wider warm-start + StripEnhancer):
       python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --epochs 150 --batch 16

  3. Train with normaltrain/normalval defect-free background folders also included:
       python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --use-normal

  4. Run validation (default high-accuracy one2many+NMS vs. --no-nms end-to-end one2one):
       python train_yolo27n.py --mode val --model runs/detect/train/weights/best.pt --data dataset/fabric19.yaml
       python train_yolo27n.py --mode val --model runs/detect/train/weights/best.pt --data dataset/fabric19.yaml --no-nms

  5. Export fused ONNX model:
       python train_yolo27n.py --mode export --model runs/detect/train/weights/best.pt --format onnx --no-nms
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from yolo27 import YOLO27


def profile_models(imgsz: int = 640, nc: int = 19) -> None:
    """Compare YOLO26n baseline vs. YOLO27n and YOLO27n-Lite (unfused training vs. fused deployment)."""
    print("=" * 92)
    print(f"布匹缺陷检测 (nc={nc}, imgsz={imgsz}x{imgsz}) 模型轻量化与架构对比")
    print("=" * 92)
    header = (
        f"{'Model':<18} | {'Scales (Strides)':<18} | {'Anchors':>8} | "
        f"{'Train Params':>13} | {'Fused Params':>13} | {'Fused FLOPs':>12}"
    )
    print(header)
    print("-" * 92)

    m26 = YOLO("yolo26n.yaml")
    m26.model = m26.task_map["detect"]["model"]("yolo26n.yaml", nc=nc, verbose=False)
    p26_train = get_num_params(m26.model)
    m26_fused = deepcopy(m26.model).eval().fuse(verbose=False)
    p26_fused = get_num_params(m26_fused)
    f26_fused = get_flops(m26_fused, imgsz=imgsz)
    anchors_26 = sum((imgsz // int(s)) ** 2 for s in m26.model.stride.tolist())
    print(
        f"{'YOLO26n (基准)':<18} | {str([int(s) for s in m26.model.stride.tolist()]):<18} | "
        f"{anchors_26:>8} | {p26_train / 1e6:>11.3f} M | {p26_fused / 1e6:>11.3f} M | {f26_fused:>9.2f} B"
    )

    for label, cfg in [
        ("YOLO27n (增强版)", "models/yolo27n.yaml"),
        ("YOLO27n-Lite(极轻)", "models/yolo27n-lite.yaml"),
    ]:
        m27 = YOLO27(cfg)
        if m27.model.model[-1].nc != nc:
            m27.model = m27.task_map["detect"]["model"](cfg, nc=nc, verbose=False)
        p_train = get_num_params(m27.model)
        m27_fused = deepcopy(m27.model).eval().fuse(verbose=False)
        p_fused = get_num_params(m27_fused)
        f_fused = get_flops(m27_fused, imgsz=imgsz)
        strides = [int(s) for s in m27.model.stride.tolist()]
        anchors = sum((imgsz // s) ** 2 for s in strides)
        print(
            f"{label:<18} | {str(strides):<18} | "
            f"{anchors:>8} | {p_train / 1e6:>11.3f} M | {p_fused / 1e6:>11.3f} M | {f_fused:>9.2f} B"
        )
    print("=" * 92)
    print("说明: 训练态包含双检测头 (one2many + one2one) 与前景对齐监督分支 (ForegroundAlignmentBranch);")
    print("      推理/导出 fuse() 后自动剥离前景对齐分支与未启用检测头, 并折叠多尺度固定特征缩放系数.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO27n Fabric Defect Detection CLI")
    parser.add_argument(
        "--mode",
        choices=["profile", "train", "val", "predict", "export"],
        default="profile",
        help="Execution mode",
    )
    parser.add_argument("--model", default="models/yolo27n.yaml", help="Model YAML or .pt checkpoint path")
    parser.add_argument("--data", default="dataset/fabric19.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (默认 16, 对齐官方 YOLO26n 默认配置)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default="", help="Device, e.g. '0' or 'cpu'")
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer (默认 'auto': 自动匹配 Ultralytics 多类别最优 lr0=0.000435 + momentum=0.9)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate override (默认由 optimizer='auto' 自动计算)",
    )
    parser.add_argument(
        "--pretrained",
        default="auto",
        help="Pretrained donor weights: 'auto' (自动查找本地/相邻目录或下载 yolo26n.pt), .pt 路径, 或 'none'",
    )
    parser.add_argument(
        "--use-normal",
        action="store_true",
        help="额外挂载 normaltrain / normalval 无缺陷背景目录（默认关闭，与纯 yolo26n 保持完全一致的数据划分）",
    )
    parser.add_argument("--source", default="", help="Image/folder source for predict mode")
    parser.add_argument("--format", default="onnx", help="Export format (onnx, engine, torchscript, etc.)")
    parser.add_argument(
        "--no-nms",
        action="store_true",
        help="Use NMS-free one-to-one head (nms=False) for inference/validation/export",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader worker processes",
    )
    return parser.parse_args()


def _resolve_pretrained_arg(val: str) -> bool | str:
    low = val.strip().lower()
    if low in {"none", "false", "0", "no"}:
        return False
    if low in {"auto", "true", "1", "yes"}:
        return True
    if Path(val).is_file():
        return val
    return val


def main() -> None:
    args = parse_args()
    if args.mode == "profile":
        profile_models(imgsz=args.imgsz, nc=19)
        return

    model = YOLO27(args.model)
    nms_flag = False if args.no_nms else True

    if args.mode == "train":
        train_kwargs: dict[str, object] = dict(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer=args.optimizer,
            workers=args.workers,
            pretrained=_resolve_pretrained_arg(args.pretrained),
            use_normal=args.use_normal,
        )
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        if args.device:
            train_kwargs["device"] = args.device
        model.train(**train_kwargs)

    elif args.mode == "val":
        val_kwargs = dict(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            nms=nms_flag,
            use_normal=args.use_normal,
            workers=args.workers,
        )
        if args.device:
            val_kwargs["device"] = args.device
        model.val(**val_kwargs)

    elif args.mode == "predict":
        if not args.source:
            raise ValueError("--source is required when --mode predict")
        pred_kwargs = dict(source=args.source, imgsz=args.imgsz, nms=nms_flag, save=True, workers=args.workers)
        if args.device:
            pred_kwargs["device"] = args.device
        model.predict(**pred_kwargs)

    elif args.mode == "export":
        model.export(format=args.format, imgsz=args.imgsz, nms=nms_flag)


if __name__ == "__main__":
    main()
