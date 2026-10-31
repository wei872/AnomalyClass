#!/usr/bin/env python3
"""End-to-end FabricNCD inference: one raw image -> detection/localization/classification/score."""
from __future__ import annotations

import json
from pathlib import Path

from _cli import base_parser

from anomalyncd.end_to_end import predict_fabricncd


def main() -> None:
    parser = base_parser(__doc__, config_option=False)
    parser.add_argument("--bundle", type=Path, required=True, help="train_fabricncd.py 输出的端到端 bundle 目录")
    parser.add_argument("--image", type=Path, required=True, help="原始图片；不需要框、标签或异常图")
    parser.add_argument("--output-dir", type=Path, help="结果目录；默认 outputs/predictions/<图片名>")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--conf-threshold", type=float, help="检测器置信度阈值")
    parser.add_argument("--iou-threshold", type=float, help="检测器 NMS IoU 阈值")
    parser.add_argument("--anomaly-threshold", type=float,
                        help="YOLO 置信度门槛（仅无异常特征库时作为图像异常分数）")
    parser.add_argument("--anomaly-score-threshold", type=float,
                        help="标定后的异常分数阈值；1.0=正常图上界，调低更灵敏、调高更少误报")
    parser.add_argument("--novel-only", action="store_true", help="NCD 专用：只在新增类中分类")
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--image-threshold", type=float, help="图像多标签分类阈值")
    decisions.add_argument("--top1", action="store_true", help="每图只输出最高概率缺陷类")
    args = parser.parse_args()
    output_dir = args.output_dir or Path("outputs/predictions") / args.image.stem
    result = predict_fabricncd(args.bundle, args.image, output_dir, args.device,
                               args.conf_threshold, args.iou_threshold, args.anomaly_threshold,
                               args.image_threshold, args.top1, args.novel_only,
                               anomaly_score_threshold=args.anomaly_score_threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
