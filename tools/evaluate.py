#!/usr/bin/env python
"""评估入口: 像素级 / 图像级 / 检测 / 分类指标.

示例::

    python tools/evaluate.py --ckpt runs/fabric19_fabricncd_s/best.pt
    python tools/evaluate.py --ckpt best.pt --split val --save-preds preds.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.data.yolo import build_eval_dataset, collate_fn  # noqa: E402
from fabric_ncd.engine import evaluate, evaluate_tiled  # noqa: E402
from fabric_ncd.losses import FabricCriterion  # noqa: E402
from fabric_ncd.models.fabric_ncd import FabricNCD  # noqa: E402
from fabric_ncd.ncd import RegionAnalyzer  # noqa: E402
from fabric_ncd.utils import checkpoint_compat_config, get_logger, json_dump, load_model_weights, model_config_for_weights, resolve_device  # noqa: E402


def main() -> int:
    parser = base_parser("FabricNCD 评估")
    parser.add_argument("--ckpt", default="runs/fabric19_fabricncd_s/best.pt")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="固定异常阈值 (默认自适应)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-preds", default=None, help="保存逐图预测 JSON")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint_compat_config(
        load_config(args.config, args.overrides, args.root, checkpoint_config=state.get("config")), state
    )
    if args.device:
        cfg.setdefault("experiment", {})["device"] = args.device
    device = resolve_device(cfg.get("experiment", {}).get("device", "auto"))
    names = state.get("names") or cfg.get("data", {}).get("names") or []

    model = FabricNCD(model_config_for_weights(cfg))
    load_model_weights(model, state.get("ema") or state["model"])
    scale = float((state.get("calibration") or {}).get("memory_distance_scale", 0.0) or 0.0)
    if scale > 0:
        model.memory_distance_scale = scale
    model.to(device).eval()

    from fabric_ncd.engine import analyzer_kwargs

    kwargs = analyzer_kwargs(cfg)
    kwargs.update(names=names)
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    analyzer = RegionAnalyzer(**kwargs)
    if cfg.get("eval", {}).get("tile", False):
        metrics = evaluate_tiled(
            model, cfg, device, split=args.split, analyzer=analyzer,
            max_samples=args.max_samples, collect_predictions=bool(args.save_preds),
            calibration=state.get("calibration"),
        )
    else:
        from torch.utils.data import DataLoader

        dataset = build_eval_dataset(cfg, args.split)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size or cfg.get("data", {}).get("eval_batch_size", 4),
            shuffle=False,
            num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
            collate_fn=collate_fn,
        )
        metrics = evaluate(
            model, loader, cfg, device, criterion=FabricCriterion(cfg).to(device),
            analyzer=analyzer, max_samples=args.max_samples,
            collect_predictions=bool(args.save_preds), calibration=state.get("calibration"),
        )
    preds = metrics.pop("predictions", None)
    logger.info("评估完成 (%d 张):", metrics.get("num_samples", 0))
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            logger.info("  %-16s %.4f", key, float(value))
    if args.save_preds:
        json_dump({"metrics": metrics, "predictions": preds}, args.save_preds)
        logger.info("逐图预测已保存到 %s", args.save_preds)
        if preds:
            summary_path = Path(args.save_preds).with_suffix(".summary.json")
            json_dump(metrics, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
