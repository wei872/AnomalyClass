#!/usr/bin/env python3
"""Train Fabric19 supervised classification or novel-class discovery."""
from pathlib import Path

from _cli import add_training_options, base_parser, get_training_config

from anomalyncd.config import class_order, load_class_names
from anomalyncd.data import FabricRegions
from anomalyncd.engine import train, validate_prepared
from anomalyncd.model import validate_model_config
from anomalyncd.transforms import PairTransform, TwoViews


def main() -> None:
    parser = base_parser(__doc__)
    add_training_options(parser)
    parser.add_argument("--resume", type=Path, help="同一实验目录中的 last.pt")
    parser.add_argument("--allow-random-init", action="store_true", help="仅测试：不加载 DINO，结果不代表论文模型")
    parser.add_argument("--dry-run", action="store_true", help="验证配置、类别、数据和输入张量，不下载权重或训练")
    args = parser.parse_args()
    cfg = get_training_config(args)
    if args.dry_run:
        if args.resume:
            parser.error("--dry-run 不能与 --resume 一起使用")
        validate_model_config(cfg)
        names = load_class_names(cfg["data"]["dataset_config"])
        manifest = validate_prepared(cfg, cfg["data"]["prepared_root"])
        order, _ = class_order(len(names), cfg["mode"], cfg["discovery"]["known_class_ids"])
        n_known = len(cfg["discovery"]["known_class_ids"]) if cfg["mode"] == "discovery" else len(names)
        dataset = FabricRegions(cfg["data"]["prepared_root"], "train", order, n_known, cfg["mode"],
                                TwoViews(PairTransform(cfg["model"]["image_size"], training=True)))
        sample = dataset[0]
        print(f"OK: {len(names)} classes; {len(manifest['splits']['train'])} train / {len(manifest['splits']['val'])} val regions")
        print(f"mode={cfg['mode']}, class_order={order}, views={len(sample['images'])}, image_shape={list(sample['images'][0].shape)}")
        return
    train(cfg, resume=args.resume, allow_random_init=args.allow_random_init)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
