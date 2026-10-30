#!/usr/bin/env python3
"""Evaluate a saved checkpoint on prepared validation regions."""
import json
from pathlib import Path

from _cli import base_parser

from anomalyncd.config import validate_config
from anomalyncd.engine import evaluate, load_checkpoint


def main() -> None:
    parser = base_parser(__doc__, config_option=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--image-threshold", type=float)
    decisions.add_argument("--top1", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint["config"]
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    if args.image_threshold is not None:
        cfg["inference"]["image_threshold"] = args.image_threshold
    if args.top1:
        cfg["inference"]["image_threshold"] = None
    validate_config(cfg)
    root = args.prepared_root or cfg["data"]["prepared_root"]
    output = args.output or args.checkpoint.resolve().parent / "evaluation"
    # Keep the decision rule used for this evaluation with its metrics.
    result = evaluate(args.checkpoint, root, output, args.device, config=cfg)
    result["config"] = cfg
    (Path(output) / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"错误: {exc}") from None
