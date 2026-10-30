"""Shared CLI helpers; scripts also work when launched outside the repo root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anomalyncd.config import load_config  # noqa: E402


def base_parser(description: str, config_option: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if config_option:
        parser.add_argument("--config", type=Path, default=ROOT / "configs/fabric19.yaml")
    return parser


def parse_known_classes(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("known-classes 必须是逗号分隔整数，例如 0,1,2") from exc
    if not result:
        raise argparse.ArgumentTypeError("known-classes 不能为空")
    return result


def training_overrides(args: argparse.Namespace) -> dict:
    data, model, train, experiment, inference = {}, {}, {}, {}, {}
    for key in ("prepared_root",):
        if getattr(args, key, None) is not None:
            data[key] = str(Path(getattr(args, key)).expanduser().resolve())
    if getattr(args, "dataset_config", None) is not None:
        data["dataset_config"] = str(args.dataset_config.expanduser().resolve())
    for key in ("backbone", "image_size"):
        if getattr(args, key, None) is not None:
            model[key] = getattr(args, key)
    if getattr(args, "backbone_weights", None) is not None:
        model["pretrained_path"] = str(args.backbone_weights.expanduser().resolve())
    for key in ("batch_size", "num_workers", "epochs", "eval_every", "lr"):
        if getattr(args, key, None) is not None:
            train[key] = getattr(args, key)
    if getattr(args, "amp", None) is not None:
        train["amp"] = args.amp
    if getattr(args, "class_balanced", False):
        train["class_balanced_sampler"] = True
    for key in ("seed", "device", "output_dir"):
        if getattr(args, key, None) is not None:
            experiment[key] = str(Path(getattr(args, key)).expanduser().resolve()) if key == "output_dir" else getattr(args, key)
    if getattr(args, "image_threshold", None) is not None:
        inference["image_threshold"] = args.image_threshold
    if getattr(args, "top1", False):
        inference["image_threshold"] = None
    overrides = {k: v for k, v in dict(data=data, model=model, training=train,
                                     experiment=experiment, inference=inference).items() if v}
    if getattr(args, "mode", None):
        overrides["mode"] = args.mode
    if getattr(args, "known_classes", None) is not None:
        overrides["discovery"] = {"known_class_ids": args.known_classes}
    return overrides


def add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("supervised", "discovery"))
    parser.add_argument("--known-classes", type=parse_known_classes, help="NCD 已知类原始 YOLO ID，例如 0,1,2")
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backbone", choices=("dino_vitb8", "dino_vitb16", "dino_vits8", "dino_vits16", "tiny_vit"),
                        help="默认从配置读取；仓库默认 dino_vits8（dino_deitsmall8_pretrain.pth）")
    parser.add_argument("--backbone-weights", type=Path, help="本地 DINO 原始预训练 state_dict，不是分类 checkpoint")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--class-balanced", action="store_true")
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--image-threshold", type=float, help="图像多标签阈值；默认沿用配置")
    decisions.add_argument("--top1", action="store_true", help="每图仅输出概率最高的缺陷类")


def get_training_config(args: argparse.Namespace) -> dict:
    return load_config(args.config, training_overrides(args))
