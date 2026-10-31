#!/usr/bin/env python3
"""Train end-to-end FabricNCD: defect detector + region classifier + bundle."""
from __future__ import annotations

import copy
from pathlib import Path

from _cli import add_training_options, base_parser, get_training_config

from anomalyncd.anomaly import build_bank, calibrate_bank
from anomalyncd.config import class_order, load_class_names, load_dataset_spec
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
    parser.add_argument("--anomaly-bank", type=Path,
                        help="复用已建好的异常特征库目录（默认用 normaltrain 现场建库）")
    parser.add_argument("--skip-anomaly-bank", action="store_true",
                        help="不建异常特征库；退化为纯 YOLO 定位，将失去未知类别检出能力")
    parser.add_argument("--dry-run", action="store_true", help="只验证配置和数据，不训练")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--drop-leaked-val", action="store_true",
                         help="预处理时排除与 train 完全相同的 val 图片（推荐，原始数据不改动）")
    overlap.add_argument("--allow-overlap", action="store_true",
                         help="显式接受 train/val 重复带来的数据泄漏（指标会偏高，不建议）")
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
    anomaly_cfg = cfg.get("anomaly") or {}
    use_bank = bool(anomaly_cfg.get("enabled", False)) and not args.skip_anomaly_bank
    if use_bank:
        spec = load_dataset_spec(cfg["data"]["dataset_config"], args.data_root)
        for key in ("normaltrain", "normalval"):
            if not spec.paths[key].is_dir():
                raise FileNotFoundError(
                    f"normal-only 异常检测需要 {key} 目录: {spec.paths[key]}；"
                    "如确实没有正常图，请加 --skip-anomaly-bank（将失去未知类别检出能力）")
    if args.dry_run:
        manifest = prepare_detector_dataset(
            cfg["data"]["dataset_config"], Path(cfg["experiment"]["output_dir"]) / "detector_data",
            args.data_root, class_agnostic=detector_cfg.get("class_agnostic", True),
            mode=cfg["mode"], known_class_ids=known_ids,
            allow_overlap=args.allow_overlap, drop_leaked_val=args.drop_leaked_val,
            check_only=True)
        print(f"OK: end-to-end FabricNCD {cfg['mode']}; classes={len(names)}; detector=class-agnostic defect detector")
        if cfg["mode"] == "discovery":
            print(f"  检测器隐藏新类: {manifest['stats']['novel_class_ids']}；"
                  f"train 排除含新类图片 {manifest['stats']['train'].get('dropped_novel_images', 0)} 张")
        if use_bank:
            print(f"  开集定位: normal-only DINO 特征库 @ {anomaly_cfg.get('image_size', 448)}px, "
                  f"layers={anomaly_cfg.get('layers', [3, 5, 7])}")
        else:
            print("  警告: 未启用异常特征库，定位仅来自 YOLO 框，无法检出未知类别")
        return

    output = Path(cfg["experiment"]["output_dir"]).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"端到端实验目录非空: {output}；请换 --output-dir")
    output.mkdir(parents=True)
    detector_data = output / "detector_data"
    prepare_detector_dataset(cfg["data"]["dataset_config"], detector_data, args.data_root,
                             class_agnostic=detector_cfg.get("class_agnostic", True),
                             mode=cfg["mode"], known_class_ids=known_ids,
                             allow_overlap=args.allow_overlap, drop_leaked_val=args.drop_leaked_val)
    detector_weights = train_detector(cfg, detector_data / "data.yaml", output / "detector_run",
                                      cfg["experiment"]["device"], args.allow_random_init)

    regions = output / "regions"
    prepare_dataset(cfg["data"]["dataset_config"], regions, args.data_root,
                    padding=cfg["data"]["padding"], min_crop_size=cfg["data"]["min_crop_size"],
                    allow_overlap=args.allow_overlap, drop_leaked_val=args.drop_leaked_val)
    classifier_cfg = copy.deepcopy(cfg)
    classifier_cfg["data"]["prepared_root"] = str(regions)
    classifier_cfg["experiment"]["output_dir"] = str(output / "classifier")
    train_classifier(classifier_cfg, allow_random_init=args.allow_random_init)
    classifier_checkpoint = output / "classifier" / ("best.pt" if cfg["mode"] == "supervised" else "last.pt")
    if not classifier_checkpoint.is_file():
        classifier_checkpoint = output / "classifier" / "last.pt"
    bank_dir = None
    if use_bank:
        if args.anomaly_bank is not None:
            bank_dir = args.anomaly_bank.expanduser().resolve()
            if not (bank_dir / "bank.json").is_file():
                raise FileNotFoundError(f"异常特征库缺少 bank.json: {bank_dir}")
            print(f"复用异常特征库: {bank_dir}")
        else:
            spec = load_dataset_spec(cfg["data"]["dataset_config"], args.data_root)
            bank_dir = output / "anomaly_bank_build"
            build_bank(spec.paths["normaltrain"], bank_dir, cfg,
                       device_name=cfg["experiment"]["device"],
                       allow_random_init=args.allow_random_init, progress=print)
            calibrate_bank(bank_dir, spec.paths["normalval"], cfg,
                           device_name=cfg["experiment"]["device"],
                           allow_random_init=args.allow_random_init, progress=print)
    bundle_path = save_bundle(output, detector_weights, classifier_checkpoint, cfg,
                              extra={"detector_data": "detector_data/data.yaml",
                                     "regions": "regions/manifest.json"},
                              anomaly_bank=bank_dir)
    print(f"完成：端到端 FabricNCD bundle = {bundle_path}")
    if use_bank:
        print("  开集定位已启用：anomaly_bank 随 bundle 打包，推理无需额外输入")
    else:
        print("  注意：bundle 无异常特征库，定位仅来自 YOLO 框，未知类别检出能力缺失")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
