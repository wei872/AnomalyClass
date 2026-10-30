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

import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.data.fabric_mask import auto_work_size, estimate_cloth_mask  # noqa: E402
from fabric_ncd.data.yolo import (  # noqa: E402
    boxes_to_mask,
    list_images,
    load_dataset_spec,
    read_yolo_labels,
)
from fabric_ncd.utils import ensure_dir, get_logger  # noqa: E402


def _dataset_base(spec) -> Path:
    """由实际存在的 split 目录反推数据集根目录.

    只看 ``spec.root`` 是不够的: 在 Windows 上写 ``--root /data/fabric`` 时, 这个
    字符串是"当前盘符相对路径"(实际是 C:\\data\\fabric), 而数据可能在 D: 盘.
    真实图像目录反而是 ``D:/.../source5_augmenteds/images/train``. 若直接拿
    ``spec.root`` 拼输出目录, 掩码就会写到并不存在的地方.
    """
    dirs = [p for p in list(spec.splits.values()) + list(spec.normal_splits.values()) if p is not None]
    existing = [Path(os.path.abspath(str(p))) for p in dirs if Path(p).exists()]
    if not existing:
        raise FileNotFoundError(
            "没有任何 split 目录存在, 无法确定数据集根目录. "
            "请用 --root 指向真实的数据集目录 (Windows 上要写完整盘符, 如 D:/data/fabric)."
        )
    base = Path(os.path.commonpath([str(p) for p in existing]))
    # 逐级上卷: images/train -> images -> 数据集根 (也兼容 train/val 同名目录的情况)
    for _ in range(3):
        name = base.name.lower()
        if name in ("images", "labels", "normal", "train", "val", "valid", "test"):
            base = base.parent
        else:
            break
    return base

def _parse_mask_size(value):
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("auto", "0", "none"):
        return "auto"
    return int(text)


def _export_one(image_path: str, out_dir: str, mask_size) -> tuple:
    """子进程: 估计一张图的布面掩码并写出 (顶层函数, Windows 下可 pickle)."""
    t0 = time.perf_counter()
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    size = auto_work_size(rgb) if mask_size == "auto" else mask_size
    mask, _ = estimate_cloth_mask(rgb, work_size=size)
    # 掩码本身是大块区域, 按估计尺寸保存即可: dataloader 会用最近邻还原到原图
    out_path = Path(out_dir) / f"{Path(image_path).stem}.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    return str(out_path), time.perf_counter() - t0


def _export_defect_one(image_path: str, label_path: str, out_dir: str, mask_size) -> str:
    """子进程: 把 YOLO 标注 (框/多边形) 画成疵点掩码, 用于核对标注是否读对.

    训练时并不需要这些文件: dataloader 会实时用 ``boxes_to_mask`` 从 txt 生成.
    这里只是把它们落盘, 方便肉眼确认类别/坐标没读错.
    """
    with Image.open(image_path) as im:
        w, h = im.size
    boxes, polygons = read_yolo_labels(Path(label_path))
    mask = boxes_to_mask(boxes, polygons, h, w)

    target_long = 640 if mask_size == "auto" else int(mask_size)
    scale = min(1.0, float(target_long) / float(max(h, w)))
    if scale < 1.0:
        nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
        mask = (np.asarray(Image.fromarray(mask * 255).resize((nw, nh), Image.NEAREST)) > 127).astype(np.uint8)

    out_path = Path(out_dir) / f"{Path(image_path).stem}.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    return str(out_path)


def main() -> int:
    parser = base_parser("数据集检查 / 预处理")
    parser.add_argument("--write-yaml", default=None, help="输出路径已解析的 yaml")
    parser.add_argument("--export-masks", action="store_true", help="导出无监督布面(前景)掩码")
    parser.add_argument("--export-defect-masks", action="store_true",
                        help="导出由 YOLO 标注框生成的疵点掩码 (用于核对标注是否读对)")
    parser.add_argument("--mask-out", default=None, help="布面掩码输出根目录 (默认 <数据集根>/cloth_masks)")
    parser.add_argument("--defect-mask-out", default=None, help="疵点掩码输出根目录 (默认 <数据集根>/defect_masks)")
    parser.add_argument("--mask-size", default="auto",
                        help="掩码估计/保存的长边尺寸: auto (按实测经纬周期自动决定) 或整数, 如 1024")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数 (默认 min(8, CPU 核数))")
    parser.add_argument("--limit", type=int, default=None, help="每类最多处理多少张 (先小规模试跑)")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的掩码 (默认跳过, 支持断点续跑)")
    parser.add_argument("--check-images", action="store_true", help="逐张打开图片, 检测损坏文件 (较慢)")
    parser.add_argument("--size-scan", type=int, default=400, help="每个 split 统计多少张图的尺寸")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_config(args.config, args.overrides, args.root)
    spec = load_dataset_spec(cfg)

    if args.root and not Path(args.root).exists():
        logger.warning(
            "--root %s 不存在. Windows 下要写完整盘符 (如 D:/data/fabric); "
            "只写 /data/fabric 会被当成当前盘符下的相对路径, 很容易写错地方.",
            args.root,
        )
    logger.info("数据集根目录: %s", spec.root)
    for split, path in list(spec.splits.items()) + list(spec.normal_splits.items()):
        if path is not None:
            logger.info("  %-12s %s%s", split, path, "" if Path(path).exists() else "   <-- 不存在")
    logger.info("类别数: %d", spec.nc)
    if spec.root and not Path(spec.root).exists():
        # 只有 root 在本机根本不存在时才提示 (用 --root 指过来了就不用再唠叨)
        logger.warning(
            "数据集根目录不存在: %s\n"
            "  yaml 里是别的机器上的路径时, 用 --root 指到本机目录, 例如 "
            "--root D:/tyhn/yolo/dataset/source5_augmenteds",
            spec.root,
        )

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
        # 原图尺寸分布: 尺寸不统一会直接影响 image_size=auto 的推导与训练时的缩放变形
        sizes = Counter()
        for img in (images + normals)[: args.size_scan]:
            try:
                with Image.open(img) as im:
                    w, h = im.size
            except Exception:
                continue
            sizes[(h, w)] += 1
        if sizes:
            top = ", ".join(f"{h}x{w} x{n}" for (h, w), n in sizes.most_common(3))
            logger.info("      | 尺寸(高x宽): %s%s", top, "" if len(sizes) == 1 else f"  <-- 共 {len(sizes)} 种")
            if len(sizes) > 1:
                logger.warning("      | 该 split 内尺寸不统一, 不同长宽比的图会被拉伸变形")
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
        if args.mask_out:
            out_root = Path(os.path.abspath(str(args.mask_out)))
        else:
            out_root = _dataset_base(spec) / "cloth_masks"
            if spec.root and not Path(spec.root).exists():
                logger.warning("data.root (%s) 不存在, 掩码改为写到图像实际所在的数据集根: %s",
                               spec.root, out_root)
        logger.info("掩码输出目录: %s", out_root)

        # 先抽样判断"是不是整幅都是布": 这种情况下布面掩码恒为 255, 没有任何信息,
        # 白导出 550 张反而浪费时间 (包边/布边信息由模型的 border_prior 负责).
        probe = list_images(spec.splits.get("train"))[:8]
        all_cloth = False
        if probe and not args.force:
            coverages = []
            for img_path in probe:
                with Image.open(img_path) as im:
                    small = np.asarray(im.convert("RGB"), dtype=np.uint8)
                m, _ = estimate_cloth_mask(small, work_size=512)
                coverages.append(float(m.mean()))
            all_cloth = sum(c > 0.995 for c in coverages) >= max(1, int(0.8 * len(coverages)))
            if all_cloth:
                logger.warning(
                    "抽样 %d 张, 布面覆盖率平均 %.2f%% —— 你的图整幅都是布, 布面掩码没有信息量, 已跳过导出.",
                    len(coverages), 100.0 * float(np.mean(coverages)),
                )
                logger.warning("直接用默认配置训练即可 (data.mask_dir 保持 null); 确要导出可加 --force")
        if all_cloth and not args.export_defect_masks:
            return 0
        workers = args.workers or min(8, os.cpu_count() or 1)
        mask_size = _parse_mask_size(args.mask_size)
        for split in ("train", "val"):
            if all_cloth:
                break
            images = list_images(spec.splits.get(split))
            if args.limit:
                images = images[: args.limit]
            target = ensure_dir(out_root / split)
            todo = [img for img in images if args.force or not (target / f"{img.stem}.png").exists()]
            skipped = len(images) - len(todo)
            logger.info("%s: 待处理 %d 张 (已存在跳过 %d 张, workers=%d, mask_size=%s)",
                        split, len(todo), skipped, workers, mask_size)
            if not todo:
                continue
            t_start = time.perf_counter()
            done = 0
            failed = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_export_one, str(img), str(target), mask_size): img for img in todo}
                for future in as_completed(futures):
                    img = futures[future]
                    done += 1
                    try:
                        out_path, cost = future.result()
                        if out_path is None:
                            failed += 1
                    except Exception as exc:  # pragma: no cover
                        failed += 1
                        logger.warning("失败: %s (%s)", img.name, exc)
                        continue
                    if done % 50 == 0 or done == len(todo):
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed / max(done, 1) * (len(todo) - done)
                        logger.info("  %s: %d/%d | %.1f 张/s | 剩余约 %.0f s | 失败 %d",
                                    split, done, len(todo), done / max(elapsed, 1e-6), eta, failed)
            logger.info("掩码导出完成: %s (%d 张, 失败 %d)", target, len(todo) - failed, failed)
        if not all_cloth:
            logger.info("把配置中的 data.mask_dir 设为: %s", out_root)

    if args.export_defect_masks:
        base = _dataset_base(spec)
        out_root = Path(os.path.abspath(str(args.defect_mask_out or base / "defect_masks")))
        workers = args.workers or min(8, os.cpu_count() or 1)
        mask_size = _parse_mask_size(args.mask_size)
        logger.info("疵点掩码输出目录: %s", out_root)
        for split in ("train", "val"):
            images = list_images(spec.splits.get(split))
            if args.limit:
                images = images[: args.limit]
            label_dir = spec.labels.get(split)
            target = ensure_dir(out_root / split)
            todo = [
                img for img in images
                if label_dir is not None and (Path(label_dir) / f"{img.stem}.txt").exists()
                and (args.force or not (target / f"{img.stem}.png").exists())
            ]
            logger.info("%s: 待处理 %d 张 (workers=%d)", split, len(todo), workers)
            if not todo:
                continue
            done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_export_defect_one, str(img), str(Path(label_dir) / f"{img.stem}.txt"),
                                str(target), mask_size): img
                    for img in todo
                }
                for future in as_completed(futures):
                    done += 1
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover
                        logger.warning("失败: %s (%s)", futures[future].name, exc)
                    if done % 100 == 0 or done == len(todo):
                        logger.info("  %s: %d/%d", split, done, len(todo))
            logger.info("疵点掩码导出完成: %s (%d 张)", target, len(todo))
        logger.info("这些掩码只用于人工核对标注; 训练时疵点掩码由 dataloader 实时生成, 无需配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
