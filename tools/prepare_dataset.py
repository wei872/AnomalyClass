#!/usr/bin/env python
"""数据集检查 / 路径重映射 / 布面掩码导出.

常见用法::

    # 1) 检查数据 (原 yaml 是 Windows 路径时用 --root 重映射)
    python tools/prepare_dataset.py --root /data/fabric/source5_augmenteds

    # 2) 生成路径已解析的 yaml
    python tools/prepare_dataset.py --root /data/fabric --write-yaml dataset/fabric19.resolved.yaml

    # 3) 预计算布面掩码 (加速训练, 也让布面前景/背景监督更稳定)
    python tools/prepare_dataset.py --root /data/fabric --export-masks

导出后把 ``data.mask_dir`` 指向生成的目录即可 (目录需与 images 同名文件).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.data.fabric_mask import estimate_cloth_mask  # noqa: E402
from fabric_ncd.data.yolo import list_images, load_dataset_spec, read_yolo_labels  # noqa: E402
from fabric_ncd.utils import ensure_dir, get_logger  # noqa: E402


def main() -> int:
    parser = base_parser("数据集检查 / 预处理")
    parser.add_argument("--root", default=None, help="本机数据集根目录 (用于重映射 Windows 路径)")
    parser.add_argument("--write-yaml", default=None, help="输出路径已解析的 yaml")
    parser.add_argument("--export-masks", action="store_true", help="导出无监督布面掩码")
    parser.add_argument("--mask-out", default=None, help="掩码输出根目录 (默认 <root>/cloth_masks)")
    parser.add_argument("--check-images", action="store_true", help="逐张打开图片, 检测损坏文件 (较慢)")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_config(args.config, args.overrides)
    if args.root:
        cfg.setdefault("data", {})["root"] = args.root
    spec = load_dataset_spec(cfg)

    logger.info("数据集根目录: %s", spec.root)
    logger.info("类别数: %d", spec.nc)
    if spec.root and str(spec.root).lower().startswith("d:"):
        logger.warning("仍是 Windows 路径, 请用 --root 指向本机数据目录")

    total_defect = 0
    hist = Counter()
    for split in ("train", "val"):
        images = list_images(spec.splits.get(split))
        labels = 0
        empty = 0
        missing = 0
        for img in images:
            label_path = Path(spec.labels[split]) / f"{img.stem}.txt" if spec.labels.get(split) else None
            if label_path is None or not label_path.exists():
                missing += 1
                continue
            boxes, _ = read_yolo_labels(label_path)
            labels += len(boxes)
            if len(boxes) == 0:
                empty += 1
            for b in boxes:
                hist[int(b[0])] += 1
        normals = list_images(spec.normal_splits.get(split))
        total_defect += len(images)
        logger.info(
            "%-5s | 图像 %5d | 标注 %6d | 空标注 %4d | 无标注文件 %4d | 正常图 %5d",
            split, len(images), labels, empty, missing, len(normals),
        )
        if args.check_images:
            bad = []
            for img in images[: 5000]:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except Exception:
                    bad.append(str(img))
            if bad:
                logger.warning("损坏图片 %d 张: %s", len(bad), bad[:5])
            else:
                logger.info("图片完整性检查通过 (前 %d 张)", min(len(images), 5000))

    names = spec.names or [f"class_{i}" for i in range(spec.nc)]
    for idx in range(spec.nc):
        name = names[idx] if idx < len(names) else f"class_{idx}"
        logger.info("  %2d %-10s %6d", idx, name, hist.get(idx, 0))
    if hist:
        counts = np.array([hist.get(i, 0) for i in range(spec.nc)], dtype=np.float64)
        valid = counts > 0
        if valid.any():
            ratio = counts[valid].max() / max(counts[valid].min(), 1)
            logger.info("类别不均衡比 (max/min): %.1f", ratio)
            if ratio > 20:
                logger.warning("类别高度不均衡, 建议提高 data.normal_ratio 或对稀有类做重采样")

    if args.write_yaml:
        payload = {
            "path": str(spec.root) if spec.root else None,
            "train": str(spec.splits.get("train")),
            "val": str(spec.splits.get("val")),
            "normaltrain": str(spec.normal_splits.get("train")) if spec.normal_splits.get("train") else None,
            "normalval": str(spec.normal_splits.get("val")) if spec.normal_splits.get("val") else None,
            "nc": spec.nc,
            "names": spec.names,
        }
        ensure_dir(Path(args.write_yaml).parent)
        with open(args.write_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        logger.info("已写出 %s", args.write_yaml)

    if args.export_masks:
        if not spec.root:
            raise ValueError("导出掩码需要知道数据集根目录, 请指定 --root")
        out_root = Path(args.mask_out or Path(spec.root) / "cloth_masks")
        for split in ("train", "val"):
            images = list_images(spec.splits.get(split))
            target = ensure_dir(out_root / split)
            for i, img in enumerate(images):
                with Image.open(img) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                mask, _ = estimate_cloth_mask(rgb)
                Image.fromarray((mask * 255).astype(np.uint8)).save(target / f"{img.stem}.png")
                if (i + 1) % 500 == 0:
                    logger.info("  %s: %d/%d", split, i + 1, len(images))
            logger.info("掩码导出完成: %s (%d 张)", target, len(images))
        logger.info("把配置中的 data.mask_dir 设为: %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
