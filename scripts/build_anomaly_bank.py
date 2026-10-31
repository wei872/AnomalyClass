#!/usr/bin/env python3
"""Build the normal-only DINO feature bank used for open-set anomaly localization."""
from __future__ import annotations

from pathlib import Path

from _cli import base_parser, training_overrides

from anomalyncd.anomaly import build_bank, calibrate_bank
from anomalyncd.config import load_config, load_dataset_spec


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--data-root", type=Path, help="覆盖 dataset YAML 的原始数据根目录")
    parser.add_argument("--normal-train", type=Path, help="覆盖正常图建库目录（默认 YAML 的 normaltrain）")
    parser.add_argument("--normal-val", type=Path, help="覆盖正常图标定目录（默认 YAML 的 normalval）")
    parser.add_argument("--output-dir", type=Path, required=True, help="特征库输出目录")
    parser.add_argument("--batch-size", type=int, default=8, help="编码 batch（不影响分类器配置）")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--backbone", choices=("dino_vitb8", "dino_vitb16", "dino_vits8",
                                               "dino_vits16", "tiny_vit"),
                        help="默认 dino_vits8（dino_deitsmall8_pretrain.pth）")
    parser.add_argument("--backbone-weights", type=Path, help="本地 DINO 预训练 state_dict")
    # Deliberately NOT --image-size: that shared flag would change the classifier
    # resolution. Anomaly resolution is independent.
    parser.add_argument("--anomaly-image-size", type=int, help="覆盖 anomaly.image_size")
    parser.add_argument("--bank-size", type=int, help="覆盖 anomaly.bank_size")
    parser.add_argument("--no-coreset", action="store_true", help="随机采样代替贪心 coreset（更快）")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="只建库不标定；未标定时异常分数没有绝对含义")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true", help="仅测试：随机初始化骨干")
    args = parser.parse_args()

    overrides = training_overrides(args)
    anomaly = {"enabled": True}
    if args.anomaly_image_size is not None:
        anomaly["image_size"] = args.anomaly_image_size
    if args.bank_size is not None:
        anomaly["bank_size"] = args.bank_size
    if args.no_coreset:
        anomaly["coreset"] = False
    overrides.setdefault("anomaly", {}).update(anomaly)
    cfg = load_config(args.config, overrides)

    device = cfg["experiment"]["device"]
    if args.normal_train is not None:
        normal_train = args.normal_train.expanduser().resolve()
        normal_val = args.normal_val.expanduser().resolve() if args.normal_val else None
    else:
        spec = load_dataset_spec(cfg["data"]["dataset_config"], args.data_root)
        normal_train = spec.paths["normaltrain"]
        normal_val = args.normal_val.expanduser().resolve() if args.normal_val else spec.paths["normalval"]

    output = args.output_dir.expanduser().resolve()
    settings = cfg["anomaly"]
    print(f"正常图（建库）: {normal_train}")
    print(f"正常图（标定）: {normal_val}")
    print(f"骨干: {cfg['model']['backbone']}  异常图分辨率: {settings.get('image_size', 448)}")
    build_bank(normal_train, output, cfg, device_name=device,
               allow_random_init=args.allow_random_init, batch_size=args.batch_size,
               overwrite=args.overwrite, progress=print)
    if not args.skip_calibration:
        if normal_val is None:
            raise ValueError("未提供标定目录；请用 --normal-val 指定，或加 --skip-calibration")
        calibrate_bank(output, normal_val, cfg, device_name=device,
                       allow_random_init=args.allow_random_init, progress=print)
    print(f"完成：异常特征库 = {output}")
    if args.skip_calibration:
        print("注意：未标定，异常分数无绝对含义；部署前请运行标定。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
