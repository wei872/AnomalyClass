#!/usr/bin/env python3
"""Validate Fabric19 YOLO data and generate aligned region crops/masks."""
from pathlib import Path

from _cli import base_parser, get_training_config

from anomalyncd.data import prepare_dataset


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--dataset-config", type=Path, help="默认读取训练配置中的 dataset/fabric19.yaml")
    parser.add_argument("--data-root", type=Path, help="覆盖 YAML 的 Windows/旧数据根目录")
    parser.add_argument("--output", type=Path, help="默认使用配置 data.prepared_root")
    parser.add_argument("--padding", type=float)
    parser.add_argument("--min-crop-size", type=float)
    parser.add_argument("--check-only", action="store_true", help="只扫描图片和标签，不写裁剪数据")
    parser.add_argument("--overwrite", action="store_true", help="替换本工具之前生成的预处理目录")
    parser.add_argument("--allow-missing-labels", action="store_true", help="将缺标签图片视为背景并跳过；默认报错")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--allow-overlap", action="store_true", help="显式允许 train/val 完全相同的图片（不推荐）")
    overlap.add_argument("--drop-leaked-val", action="store_true",
                         help="排除与 train 完全相同的 val 图片（推荐，原始数据不改动）")
    args = parser.parse_args()
    cfg = get_training_config(args)
    output = args.output or Path(cfg["data"]["prepared_root"])
    manifest = prepare_dataset(cfg["data"]["dataset_config"], output, args.data_root,
                               args.padding if args.padding is not None else cfg["data"]["padding"],
                               args.min_crop_size if args.min_crop_size is not None else cfg["data"]["min_crop_size"],
                               overwrite=args.overwrite, allow_missing_labels=args.allow_missing_labels,
                               allow_overlap=args.allow_overlap, drop_leaked_val=args.drop_leaked_val,
                               check_only=args.check_only)
    if args.check_only:
        print(f"OK: {manifest['nc']} classes; {manifest['stats']['train']['regions']} train / "
              f"{manifest['stats']['val']['regions']} val regions; no crops written")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"错误: {exc}") from None
