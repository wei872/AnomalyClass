#!/usr/bin/env python3
"""Train end-to-end FabricNCD: defect detector + region classifier + bundle."""
from __future__ import annotations

import copy
from pathlib import Path

from _cli import add_training_options, base_parser, get_training_config

from anomalyncd.config import class_order, load_class_names
from anomalyncd.data import prepare_dataset
from anomalyncd.detector import prepare_detector_dataset, train_detector
from anomalyncd.end_to_end import save_bundle
from anomalyncd.engine import train as train_classifier


def main() -> None:
    parser = base_parser(__doc__)
    add_training_options(parser)
    parser.add_argument("--data-root", type=Path, help="覆盖 dataset YAML 的原始数据根目录")
    parser.add_argument("--allow-random-init", action="store_true", help="仅测试：检测器和 DINO 都随机初始化")
    parser.add_argument("--detector-epochs", type=int)
    parser.add_argument("--detector-batch-size", type=int)
    parser.add_argument("--detector-image-size", type=int)
    parser.add_argument("--detector-weights", type=Path, help="本地 YOLO 检测器预训练权重")
    parser.add_argument("--dry-run", action="store_true", help="只验证配置和数据，不训练")
    args = parser.parse_args()
    cfg = get_training_config(args)
    if args.output_dir is None:
        cfg["experiment"]["output_dir"] = str(Path("outputs") / f"fabricncd_{cfg['mode']}")
    detector_cfg = cfg.setdefault("detector", {})
    if args.detector_epochs is not None:
        detector_cfg["epochs"] = args.detector_epochs
    if args.detector_batch_size is not None:
        detector_cfg["batch_size"] = args.detector_batch_size
    if args.detector_image_size is not None:
        detector_cfg["image_size"] = args.detector_image_size
    if args.detector_weights is not None:
        detector_cfg["weights"] = str(args.detector_weights.expanduser().resolve())
    if cfg["mode"] == "discovery" and not detector_cfg.get("class_agnostic", True):
        parser.error("NCD 模式要求 detector.class_agnostic=true")

    names = load_class_names(cfg["data"]["dataset_config"])
    known_ids = cfg["discovery"]["known_class_ids"] if cfg["mode"] == "discovery" else None
    class_order(len(names), cfg["mode"], known_ids)
    if args.dry_run:
        prepare_detector_dataset(cfg["data"]["dataset_config"], Path(cfg["experiment"]["output_dir"]) / "detector_data",
                                 args.data_root, class_agnostic=detector_cfg.get("class_agnostic", True),
                                 mode=cfg["mode"], known_class_ids=known_ids, check_only=True)
        print(f"OK: end-to-end FabricNCD {cfg['mode']}; classes={len(names)}; detector=class-agnostic defect detector")
        return

    output = Path(cfg["experiment"]["output_dir"]).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"端到端实验目录非空: {output}；请换 --output-dir")
    output.mkdir(parents=True)
    detector_data = output / "detector_data"
    prepare_detector_dataset(cfg["data"]["dataset_config"], detector_data, args.data_root,
                             class_agnostic=detector_cfg.get("class_agnostic", True),
                             mode=cfg["mode"], known_class_ids=known_ids)
    detector_weights = train_detector(cfg, detector_data / "data.yaml", output / "detector_run",
                                      cfg["experiment"]["device"], args.allow_random_init)

    regions = output / "regions"
    prepare_dataset(cfg["data"]["dataset_config"], regions, args.data_root,
                    padding=cfg["data"]["padding"], min_crop_size=cfg["data"]["min_crop_size"])
    classifier_cfg = copy.deepcopy(cfg)
    classifier_cfg["data"]["prepared_root"] = str(regions)
    classifier_cfg["experiment"]["output_dir"] = str(output / "classifier")
    train_classifier(classifier_cfg, allow_random_init=args.allow_random_init)
    classifier_checkpoint = output / "classifier" / ("best.pt" if cfg["mode"] == "supervised" else "last.pt")
    if not classifier_checkpoint.is_file():
        classifier_checkpoint = output / "classifier" / "last.pt"
    bundle_path = save_bundle(output, detector_weights, classifier_checkpoint, cfg,
                              extra={"detector_data": "detector_data/data.yaml", "regions": "regions/manifest.json"})
    print(f"完成：端到端 FabricNCD bundle = {bundle_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
