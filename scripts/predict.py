#!/usr/bin/env python3
"""Classify one fabric image using YOLO detector proposals or an explicit whole-image fallback."""
import json
from pathlib import Path

from _cli import base_parser

from anomalyncd.inference import predict_image


def main() -> None:
    parser = base_parser(__doc__, config_option=False)
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练得到的 best.pt 或 last.pt")
    parser.add_argument("--image", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--boxes", type=Path, help="YOLO 预测 txt 或像素 XYXY JSON；输入类别 ID 被忽略")
    source.add_argument("--whole-image", action="store_true", help="无候选框时整图分类；假设图中确有缺陷")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--novel-only", action="store_true", help="NCD 专用：明确假设输入是新类，仅在新增类中分类")
    parser.add_argument("--save-crops", type=Path)
    parser.add_argument("--output", type=Path, help="结果 JSON；默认打印到终端")
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--image-threshold", type=float)
    decisions.add_argument("--top1", action="store_true")
    args = parser.parse_args()
    result = predict_image(args.checkpoint, args.image, args.boxes, args.whole_image, args.device,
                           args.conf_threshold, args.image_threshold, args.top1, args.novel_only, args.save_crops)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"预测已保存: {args.output.resolve()}")
    else:
        print(text)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
