#!/usr/bin/env python3
"""Audit Fabric19 train/val leakage and optionally quarantine leaked images."""
from __future__ import annotations

import json
from pathlib import Path

from _cli import base_parser

from anomalyncd.split_audit import audit_split, format_report, quarantine_leaked_val, restore_quarantine


def main() -> None:
    parser = base_parser(__doc__, config_option=False)
    parser.add_argument("--dataset-config", type=Path, default=Path("dataset/fabric19.yaml"))
    parser.add_argument("--data-root", type=Path, help="覆盖 YAML 的数据根目录")
    parser.add_argument("--perceptual", action="store_true", help="额外用感知哈希检出同源增强的近似重复")
    parser.add_argument("--max-distance", type=int, default=5, help="感知哈希汉明距离阈值")
    parser.add_argument("--max-pairs", type=int, default=200, help="报告中每类最多列出的样例数")
    parser.add_argument("--report", type=Path, help="把完整审计结果写入该 JSON")
    parser.add_argument("--quarantine", type=Path, help="把泄漏图片移动到该隔离目录（默认只预演）")
    parser.add_argument("--side", choices=("val", "train"), default="val", help="从哪一侧移出重复图片")
    parser.add_argument("--include-near", action="store_true", help="隔离时同时移出疑似增强重复")
    parser.add_argument("--apply", action="store_true", help="真正执行移动；不加则只预演")
    parser.add_argument("--restore", type=Path, help="从 quarantine.json 恢复被移出的图片")
    args = parser.parse_args()

    if args.restore:
        restore_quarantine(args.restore)
        return

    if args.quarantine:
        manifest = quarantine_leaked_val(args.dataset_config, args.quarantine, args.data_root,
                                         side=args.side, include_near=args.include_near,
                                         max_distance=args.max_distance, apply=args.apply)
        print(f"重复图片数: {manifest['count']}")
        for record in manifest["files"][:10]:
            print(f"  {record['key']}")
        if not args.apply:
            print("这是预演，未移动任何文件。确认无误后加 --apply 执行。")
        else:
            print(f"已隔离，可用 --restore {args.quarantine / 'quarantine.json'} 还原")
        return

    report = audit_split(args.dataset_config, args.data_root, args.perceptual,
                         args.max_distance, args.max_pairs)
    print(format_report(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完整报告: {args.report.resolve()}")
    if report["exact_cross_split_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"错误: {exc}") from None
