#!/usr/bin/env python
"""推理入口: 单图 / 目录批量检测, 输出可视化与 JSON.

示例::

    python tools/infer.py --ckpt runs/fabric19_fabricncd_s/best.pt --source a.jpg
    python tools/infer.py --ckpt best.pt --source /data/images --out outputs/pred --tile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, load_predict_overrides  # noqa: E402

from fabric_ncd.inference import FabricPredictor, as_hw  # noqa: E402
from fabric_ncd.utils import ensure_dir, get_logger, json_dump  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _size_arg(value):
    try:
        return as_hw(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = base_parser("FabricNCD 推理")
    parser.add_argument("--ckpt", default="runs/fabric19_fabricncd_s/best.pt")
    parser.add_argument("--source", required=True, help="图片路径或目录")
    parser.add_argument("--out", default="outputs/predictions", help="输出目录")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tile", action="store_true", help="大幅面图像切块推理")
    parser.add_argument("--tile-size", type=_size_arg, default=None, help="瓦片尺寸: 边长或 HxW")
    parser.add_argument("--out-stride", type=int, default=None, help="分数图下采样倍数 (实时可设 4)")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=None, help="CUDA 上开启 FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--no-heatmap", action="store_true", help="不画热力图 (只画框, 更快)")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_predict_overrides(args.config, args.overrides, args.root)
    predictor = FabricPredictor(
        checkpoint=args.ckpt,
        config=cfg,
        device=args.device,
        threshold=args.threshold,
        tile=True if args.tile else None,
        tile_size=args.tile_size,
        out_stride=args.out_stride,
        fp16=args.fp16,
        tile_overlap=args.tile_overlap,
    )
    out_dir = ensure_dir(args.out)
    source = Path(args.source)
    if source.is_dir():
        files = [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    elif source.exists():
        files = [source]
    else:
        raise FileNotFoundError(source)
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {source}")

    from PIL import Image

    all_results = {}
    # 同一子目录里同名不同扩展名的图片也不能覆盖彼此的可视化。
    overlay_counts = {}
    for path in files:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        key = relative.parent / f"{relative.stem}_pred.png"
        overlay_counts[key] = overlay_counts.get(key, 0) + 1
    for path in files:
        with Image.open(path) as img:
            image = img.convert("RGB")
        result = predictor.predict(image)
        all_results[str(path)] = {"regions": result["regions"], "image": result["image"]}
        if not args.no_overlay:
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            key = relative.parent / f"{relative.stem}_pred.png"
            if overlay_counts[key] > 1:
                key = relative.parent / f"{relative.name}_pred.png"
            overlay_path = out_dir / key
            ensure_dir(overlay_path.parent)
            predictor.save_overlay(image, result, overlay_path, draw_heatmap=not args.no_heatmap)
        logger.info(
            "%s | score=%.3f | %s(%s %.2f) | regions=%d",
            path.name,
            result["image"]["image_score"],
            result["image"]["name"],
            "abnormal" if result["image"]["is_abnormal"] else "normal",
            result["image"]["cls_score"],
            len(result["regions"]),
        )
    json_dump({"checkpoint": args.ckpt, "results": all_results}, out_dir / "results.json")
    logger.info("完成 %d 张, 结果保存在 %s", len(files), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
