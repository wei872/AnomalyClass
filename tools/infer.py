#!/usr/bin/env python
"""推理入口: 单图 / 目录批量检测, 输出可视化与 JSON.

示例::

    python tools/infer.py --ckpt runs/fabric19_fabricncd_s/best.pt --source a.jpg
    python tools/infer.py --ckpt best.pt --source /data/images --out outputs/pred --tile
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.inference import FabricPredictor  # noqa: E402
from fabric_ncd.utils import ensure_dir, get_logger, json_dump  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> int:
    parser = base_parser("FabricNCD 推理")
    parser.add_argument("--ckpt", default="runs/fabric19_fabricncd_s/best.pt")
    parser.add_argument("--source", required=True, help="图片路径或目录")
    parser.add_argument("--out", default="outputs/predictions", help="输出目录")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tile", action="store_true", help="大幅面图像切块推理")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--out-stride", type=int, default=None, help="分数图下采样倍数 (实时可设 4)")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=None, help="CUDA 上开启 FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--no-heatmap", action="store_true", help="不画热力图 (只画框, 更快)")
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_config(args.config, args.overrides)
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
        files = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    elif source.exists():
        files = [source]
    else:
        raise FileNotFoundError(source)
    if not files:
        raise FileNotFoundError(f"目录中没有图片: {source}")

    from PIL import Image

    all_results = {}
    for path in files:
        image = Image.open(path).convert("RGB")
        result = predictor.predict(image)
        all_results[str(path)] = {"regions": result["regions"], "image": result["image"]}
        if not args.no_overlay:
            overlay_path = out_dir / f"{path.stem}_pred.png"
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
