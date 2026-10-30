#!/usr/bin/env python
"""新类发现 (Novel Class Discovery).

对 **没有标签** 的疵点图做区域聚类, 再把簇与已训练的 19 个原型做匈牙利匹配:
相似度高 -> 归为已知类; 相似度低 -> 判为 "新疵点类". 输出建议的新类原型,
可直接写回分类器用于在线增量更新, 也可以只用来做人工复核清单.

示例::

    python tools/discover.py --ckpt runs/fabric19_fabricncd_s/best.pt \
        --source /data/fabric/unlabeled --out outputs/ncd
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.inference import FabricPredictor  # noqa: E402
from fabric_ncd.ncd import discover_classes, l2_normalize  # noqa: E402
from fabric_ncd.utils import ensure_dir, get_logger, json_dump  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _region_features(maps: Dict[str, np.ndarray], regions: List[Dict[str, Any]], thr: float) -> List[np.ndarray]:
    """区域级嵌入: 在区域内的异常像素上做掩码平均池化."""
    score = maps["score"]
    emb = maps["embedding"]
    h, w = score.shape
    feats = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        ix1, iy1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        ix2, iy2 = min(w, int(np.ceil(x2 * w))), min(h, int(np.ceil(y2 * h)))
        sub = score[iy1:iy2, ix1:ix2]
        sub_emb = emb[:, iy1:iy2, ix1:ix2]
        mask = sub >= thr
        if mask.sum() < 4:
            mask = sub >= np.percentile(sub, 80)
        if mask.sum() == 0:
            continue
        feat = (sub_emb * mask[None]).sum(axis=(1, 2)) / mask.sum()
        feats.append(l2_normalize(feat[None])[0])
    return feats


def main() -> int:
    parser = base_parser("FabricNCD 新类发现")
    parser.add_argument("--ckpt", default="runs/fabric19_fabricncd_s/best.pt")
    parser.add_argument("--source", required=True, help="无标签图片目录")
    parser.add_argument("--out", default="outputs/ncd")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=None, help="指定簇数; 默认轮廓系数自动选择")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--novelty-threshold", type=float, default=0.55, help="与已知原型相似度低于该值判为新类")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-crops", type=int, default=8, help="每个簇保存的样例抠图数量")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_config(args.config, args.overrides, args.root)
    predictor = FabricPredictor(checkpoint=args.ckpt, config=cfg, device=args.device)
    out_dir = ensure_dir(args.out)
    crops_dir = ensure_dir(out_dir / "cluster_samples")

    files = [p for p in sorted(Path(args.source).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise FileNotFoundError(f"未找到图片: {args.source}")

    features, meta, crops = [], [], []
    for path in files:
        image = Image.open(path).convert("RGB")
        result = predictor.predict(image, with_embedding=True)
        maps = result["maps"]
        if "embedding" not in maps:
            raise RuntimeError(
                "新类发现需要区域嵌入, 请关闭切块推理 (FabricPredictor(tile=False)); "
                "切块模式目前只返回异常图与类别概率."
            )
        thr = float(result["image"].get("threshold", 0.5))
        feats = _region_features(maps, result["regions"], thr)
        img_arr = np.asarray(image)
        h, w = img_arr.shape[:2]
        for region, feat in zip(result["regions"], feats):
            x1, y1, x2, y2 = region["bbox"]
            box = (int(x1 * w), int(y1 * h), int(np.ceil(x2 * w)), int(np.ceil(y2 * h)))
            features.append(feat)
            meta.append({"path": str(path), "bbox": region["bbox"], "score": region["score"],
                         "pred_cls": region["cls"], "pred_name": region["name"]})
            crops.append((img_arr, box))
    if not features:
        logger.warning("没有提取到任何疵点区域, 结束")
        return 0

    features_arr = np.stack(features)
    known = predictor.model.classifier.prototypes.detach().cpu().numpy()
    names = predictor.names or [str(i) for i in range(len(known))]
    report = discover_classes(
        features_arr,
        known,
        names,
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        novelty_threshold=args.novelty_threshold,
    )

    labels = report["cluster_labels"]
    assignments = report["assignments"]
    summary: Dict[str, Any] = {}
    for c in range(report["k"]):
        members = np.flatnonzero(labels == c)
        info = assignments[c]
        summary[c] = {
            **info,
            "count": int(len(members)),
            "mean_score": float(np.mean([meta[i]["score"] for i in members])) if len(members) else 0.0,
            "examples": [meta[i] for i in members[:5]],
        }
        # 保存每个簇的样例抠图
        for rank, i in enumerate(members[: args.save_crops]):
            arr, box = crops[i]
            pad = 8
            x1, y1, x2, y2 = box
            patch = arr[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
            if patch.size == 0:
                continue
            label = assignments[c]["name"].replace("/", "_")
            Image.fromarray(patch).save(crops_dir / f"cluster{c:02d}_{label}_{rank:02d}.png")

    payload = {
        "num_regions": int(len(features_arr)),
        "num_images": len(files),
        "num_clusters": report["k"],
        "novel_clusters": report["novel_indices"],
        "known_classes": names,
        "clusters": summary,
        "matched_known": sorted({v["known_index"] for v in assignments.values() if v["known_index"] >= 0}),
        "meta": meta,
    }
    json_dump(payload, out_dir / "ncd_report.json")

    novel = report["novel_prototypes"]
    if len(novel):
        prototypes = torch.cat(
            [torch.as_tensor(known, dtype=torch.float32), torch.as_tensor(novel, dtype=torch.float32)], dim=0
        )
        novel_names = [f"novel_{i}" for i in range(len(novel))]
        torch.save(
            {
                "prototypes": prototypes,
                "names": list(names) + novel_names,
                "known_count": len(known),
                "novelty_threshold": args.novelty_threshold,
            },
            out_dir / "discovered_prototypes.pt",
        )
        logger.info("发现 %d 个新类簇, 原型已保存到 %s", len(novel), out_dir / "discovered_prototypes.pt")

    for c, info in summary.items():
        logger.info(
            "簇 %2d | %-12s | n=%3d | sim=%.3f | %s",
            c,
            info["name"],
            info["count"],
            info["similarity"],
            "新类" if info["novel"] else "已知类",
        )
    logger.info("报告: %s", out_dir / "ncd_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
