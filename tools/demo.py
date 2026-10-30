#!/usr/bin/env python
"""一键演示: 生成合成布匹数据 -> 训练 -> 评估 -> 推理可视化.

不需要真实数据集, 用来验证环境与整条流水线是否正常::

    python tools/demo.py                 # 默认 6 轮, 128x128
    python tools/demo.py --epochs 2      # 更快
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.data.demo import add_defect, make_dataset, weave_image  # noqa: E402
from fabric_ncd.engine import Trainer  # noqa: E402
from fabric_ncd.inference import FabricPredictor  # noqa: E402
from fabric_ncd.utils import ensure_dir, get_logger, json_dump, save_yaml  # noqa: E402


def main() -> int:
    parser = base_parser("FabricNCD 一键演示")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--data-dir", default="data/demo_dataset")
    parser.add_argument("--out", default="runs/demo")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    out_dir = ensure_dir(args.out)
    root = Path(args.data_dir)
    if not (root / "dataset.yaml").exists():
        logger.info("生成合成数据集 -> %s", root)
        make_dataset(root, n_train=64, n_val=20, n_normal=24, size=(args.size, args.size))

    cfg = load_config(args.config, args.overrides, args.root)
    cfg["experiment"].update({"output_dir": str(out_dir), "device": "cpu", "amp": False, "seed": 0})
    cfg["data"].update(
        {
            "dataset_yaml": str(root / "dataset.yaml"),
            "image_size": [args.size, args.size],
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_workers": 0,
            "normal_ratio": 0.5,
            "limit_train": 64,
            "limit_val": 24,
        }
    )
    cfg["data"]["augment"].update({"synth_prob": 0.5})
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["heads"]["num_classes"] = 3
    cfg["train"].update(
        {
            "epochs": args.epochs,
            "warmup_epochs": 0,
            "log_interval": 4,
            "memory_warmup_batches": 4,
            "memory_refresh_interval": 2,
        }
    )
    cfg["eval"].update({"min_area_ratio": 0.001})
    save_yaml(cfg, out_dir / "config.yaml")

    trainer = Trainer(cfg, logger=logger)
    summary = trainer.fit()
    logger.info("最优综合分: %.4f", summary["best_score"])

    predictor = FabricPredictor(checkpoint=out_dir / "best.pt", config=cfg, device="cpu")
    rng = np.random.default_rng(7)
    report = {}
    for i in range(3):
        image = weave_image((args.size, args.size), rng)
        if i > 0:  # 第 0 张是正常布, 其余带疵点
            image, _ = add_defect(image, rng)
        result = predictor.predict(image)
        overlay = out_dir / f"demo_pred_{i}.png"
        predictor.save_overlay(image, result, overlay)
        report[f"demo_{i}"] = {"image": result["image"], "regions": result["regions"]}
        logger.info(
            "推理 %d: score=%.3f cls=%s regions=%d -> %s",
            i,
            result["image"]["image_score"],
            result["image"]["name"],
            len(result["regions"]),
            overlay,
        )
    json_dump(report, out_dir / "demo_inference.json")
    logger.info("演示完成, 输出目录: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
