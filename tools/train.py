#!/usr/bin/env python
"""训练入口.

示例::

    python tools/train.py --config configs/fabric19.yaml
    python tools/train.py --config configs/fabric19.yaml --set data.root=/data/fabric
    python tools/train.py --config configs/fabric19.yaml --set model.backbone.arch=mobilenet_v3_large
    python tools/train.py --config configs/fabric19.yaml --resume runs/xxx/last.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.engine import Trainer  # noqa: E402
from fabric_ncd.utils import checkpoint_compat_config, ensure_dir, prepare_logging  # noqa: E402


def main() -> int:
    parser = base_parser("FabricNCD 训练")
    parser.add_argument("--resume", default=None, help="从 checkpoint 继续训练")
    parser.add_argument("--device", default=None, help="覆盖设备: cpu / cuda / mps")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用 ImageNet 预训练权重")
    parser.add_argument("--dry-run", action="store_true", help="只构建模型与数据, 不训练")
    args = parser.parse_args()

    resume_config = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = state.get("config")
    cfg = load_config(args.config, args.overrides, args.root, checkpoint_config=resume_config)
    if args.resume:
        cfg = checkpoint_compat_config(cfg, state)
        # 完整 checkpoint 会覆盖主干参数，不需要下载或加载 ImageNet 权重。
        cfg.setdefault("model", {}).setdefault("backbone", {}).update(pretrained=False, weights_path=None)
    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device
    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.output_dir:
        cfg.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.no_pretrained:
        cfg.setdefault("model", {}).setdefault("backbone", {})["pretrained"] = False

    out_dir = ensure_dir(cfg["experiment"].get("output_dir", "runs/fabric19"))
    logger = prepare_logging(str(out_dir / "train.log"))

    trainer = Trainer(cfg, logger=logger)
    if args.dry_run:
        logger.info("dry-run: 数据/模型构建成功, 跳过训练")
        return 0
    if args.resume:
        trainer.load_checkpoint(args.resume, load_optimizer=True)
    summary = trainer.fit()
    logger.info("训练结束. 最优综合分: %.4f, 输出目录: %s", summary["best_score"], out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
