"""Audit train/val overlap and safely quarantine leaked validation images.

Exact duplicates (identical bytes) are hard leakage.  Augmented copies of one
source image are *not* byte-identical, so an optional perceptual hash pass flags
likely same-source pairs as well.  Nothing is ever deleted: the fix moves files
into a quarantine folder and records a restore manifest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DatasetSpec, label_root, load_dataset_spec
from .data import IMAGE_EXTENSIONS, file_sha256

AUDIT_FORMAT = "fabricncd-split-audit"
QUARANTINE_FORMAT = "fabricncd-split-quarantine"
SPLITS = ("train", "val")
KINDS = ("defect", "normal")


@dataclass
class SplitImage:
    split: str
    kind: str
    relative: str
    path: Path
    sha256: str
    label_path: Path | None
    phash: int | None = None

    @property
    def key(self) -> str:
        return f"{self.split}/{self.kind}/{self.relative}"


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit dHash; robust to small brightness/contrast changes."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), "big")


def collect_images(spec: DatasetSpec, perceptual: bool = False,
                   progress=lambda _: None) -> list[SplitImage]:
    images: list[SplitImage] = []
    for split in SPLITS:
        for kind in KINDS:
            key = split if kind == "defect" else f"normal{split}"
            root = spec.paths.get(key)
            if root is None or not root.is_dir():
                continue
            labels = label_root(root) if kind == "defect" else None
            found = 0
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = path.relative_to(root).as_posix()
                label_path = None
                if labels is not None:
                    candidate = labels / Path(relative).with_suffix(".txt")
                    label_path = candidate if candidate.is_file() else None
                phash = None
                if perceptual:
                    with Image.open(path) as handle:
                        phash = difference_hash(handle)
                images.append(SplitImage(split, kind, relative, path.resolve(),
                                         file_sha256(path), label_path, phash))
                found += 1
            progress(f"scanned {split}/{kind}: {found} images")
    if not images:
        raise ValueError("没有扫描到任何图片；请检查 --data-root 是否正确")
    return images


def _hash_bands(value: int, parts: int):
    """Contiguous bit bands. Two hashes within `parts-1` bits must share a band."""
    edges = [round(64 * index / parts) for index in range(parts + 1)]
    for index in range(parts):
        low, high = edges[index], edges[index + 1]
        yield index, (value >> low) & ((1 << (high - low)) - 1)


def near_duplicate_pairs(train: list[SplitImage], val: list[SplitImage],
                         max_distance: int, limit: int) -> tuple[list[dict], int]:
    """Cross-split perceptual matches; banding keeps this near-linear."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(train):
        if item.phash is None:
            continue
        for band in _hash_bands(item.phash, max_distance + 1):
            buckets[band].append(index)
    pairs, total, seen = [], 0, set()
    for val_item in val:
        if val_item.phash is None:
            continue
        candidates: set[int] = set()
        for band in _hash_bands(val_item.phash, max_distance + 1):
            candidates.update(buckets.get(band, ()))
        for index in sorted(candidates):
            train_item = train[index]
            distance = (val_item.phash ^ train_item.phash).bit_count()
            if distance > max_distance:
                continue
            key = (train_item.key, val_item.key)
            if key in seen:
                continue
            seen.add(key)
            total += 1
            if len(pairs) < limit:
                pairs.append({"train": train_item.key, "val": val_item.key, "distance": distance})
    return pairs, total


def audit_split(dataset_config: str | Path, data_root: str | Path | None = None,
                perceptual: bool = False, max_distance: int = 5, max_pairs: int = 200,
                progress=lambda _: None) -> dict:
    if not 0 <= max_distance <= 16:
        raise ValueError("max-distance 必须在 0..16")
    spec = load_dataset_spec(dataset_config, data_root)
    images = collect_images(spec, perceptual, progress)
    counts = {f"{split}_{kind}": sum(i.split == split and i.kind == kind for i in images)
              for split in SPLITS for kind in KINDS}

    by_hash: dict[str, list[SplitImage]] = defaultdict(list)
    for item in images:
        by_hash[item.sha256].append(item)

    exact_cross, exact_within = [], []
    leaked_val: set[str] = set()
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        train = [i for i in group if i.split == "train"]
        val = [i for i in group if i.split == "val"]
        if train and val:
            for item in val:
                leaked_val.add(item.key)
                if len(exact_cross) < max_pairs:
                    exact_cross.append({"sha256": digest, "val": item.key,
                                        "train": [t.key for t in train]})
        for split in SPLITS:
            same = [i.key for i in group if i.split == split]
            if len(same) > 1 and len(exact_within) < max_pairs:
                exact_within.append({"sha256": digest, "split": split, "images": same})

    exact_cross_total = sum(
        1 for group in by_hash.values()
        for item in group
        if item.split == "val" and any(other.split == "train" for other in group)
    )
    report = {
        "format": AUDIT_FORMAT, "schema_version": 1,
        "dataset_config": str(spec.config_path), "data_root": str(spec.root) if spec.root else None,
        "counts": counts,
        "exact_cross_split_count": exact_cross_total,
        "exact_cross_split": sorted(exact_cross, key=lambda pair: pair["val"]),
        "exact_within_split_count": len(exact_within),
        "exact_within_split": exact_within,
        "perceptual_checked": perceptual,
        "leaked_val_images": sorted(leaked_val),
        "truncated": exact_cross_total > len(exact_cross),
    }
    if perceptual:
        train = [i for i in images if i.split == "train"]
        val = [i for i in images if i.split == "val" and i.key not in leaked_val]
        pairs, total = near_duplicate_pairs(train, val, max_distance, max_pairs)
        report["near_cross_split_count"] = total
        report["near_cross_split"] = sorted(pairs, key=lambda pair: (pair["distance"], pair["val"]))
        report["near_max_distance"] = max_distance
    return report


def _resolve_key(spec: DatasetSpec, key: str) -> tuple[Path, Path | None]:
    parts = key.split("/", 2)
    if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in KINDS:
        raise ValueError(f"非法图片 key: {key}")
    split, kind, relative = parts
    root_key = split if kind == "defect" else f"normal{split}"
    root = spec.paths[root_key]
    image_path = (root / relative).resolve()
    if root.resolve() not in image_path.parents:
        raise ValueError(f"图片路径越过数据目录: {key}")
    label_path = None
    if kind == "defect":
        candidate = label_root(root) / Path(relative).with_suffix(".txt")
        label_path = candidate if candidate.is_file() else None
    return image_path, label_path


def quarantine_leaked_val(dataset_config: str | Path, destination: str | Path,
                          data_root: str | Path | None = None, side: str = "val",
                          include_near: bool = False, max_distance: int = 5,
                          apply: bool = False, progress=lambda _: None) -> dict:
    """Move leaked images (and their labels) out of one split into quarantine."""
    if side not in SPLITS:
        raise ValueError("side 必须是 train 或 val")
    spec = load_dataset_spec(dataset_config, data_root)
    report = audit_split(dataset_config, data_root, perceptual=include_near,
                         max_distance=max_distance, max_pairs=1_000_000, progress=progress)
    targets = {key for key in report["leaked_val_images"] if key.startswith(f"{side}/")}
    if side == "train":
        # Mirror the leaked set onto the train side of the same duplicate groups.
        digests = {pair["sha256"] for pair in report["exact_cross_split"]}
        images = collect_images(spec, False, lambda _: None)
        targets = {i.key for i in images if i.split == "train" and i.sha256 in digests}
    if include_near:
        targets.update(pair[side] for pair in report.get("near_cross_split", []))

    destination = Path(destination).expanduser().resolve()
    guarded = list(spec.paths.values()) + ([spec.root] if spec.root else [])
    for root in guarded:
        root = root.resolve()
        if destination == root or root in destination.parents:
            raise ValueError(f"隔离目录不能位于数据目录内: {destination}")
        if destination in root.parents:
            raise ValueError(f"隔离目录不能包含数据目录: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"隔离目录非空: {destination}；请换一个空目录")

    moved = []
    for key in sorted(targets):
        image_path, label_path = _resolve_key(spec, key)
        if not image_path.is_file():
            raise FileNotFoundError(f"待隔离图片不存在: {image_path}")
        record = {"key": key, "image": str(image_path), "label": str(label_path) if label_path else None}
        if apply:
            image_target = destination / "files" / key
            image_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(image_target))
            record["moved_image"] = str(image_target)
            if label_path is not None:
                label_target = destination / "files" / str(Path(key).with_suffix(".txt"))
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(label_path), str(label_target))
                record["moved_label"] = str(label_target)
        moved.append(record)

    manifest = {"format": QUARANTINE_FORMAT, "schema_version": 1, "applied": apply,
                "side": side, "include_near": include_near,
                "dataset_config": str(spec.config_path),
                "data_root": str(spec.root) if spec.root else None,
                "count": len(moved), "files": moved,
                "audit": {k: v for k, v in report.items() if k not in {"exact_cross_split", "near_cross_split"}}}
    if apply:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "quarantine.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"{'已隔离' if apply else '将隔离'} {len(moved)} 个 {side} 图片"
             f"{'（含疑似增强重复）' if include_near else ''}")
    return manifest


def restore_quarantine(manifest_path: str | Path, progress=lambda _: None) -> int:
    """Undo a quarantine by moving files back to their original locations."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != QUARANTINE_FORMAT or not manifest.get("applied"):
        raise ValueError("不是已执行的隔离清单")
    for record in manifest["files"]:
        for source_key, target_key in (("moved_image", "image"), ("moved_label", "label")):
            source = record.get(source_key)
            if not source:
                continue
            source, target = Path(source), Path(record[target_key])
            if not source.is_file():
                raise FileNotFoundError(f"隔离文件缺失: {source}")
            if target.exists():
                raise FileExistsError(f"原始位置已存在文件，拒绝覆盖: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    progress(f"已恢复 {manifest['count']} 个图片")
    return manifest["count"]


def format_report(report: dict, preview: int = 10) -> str:
    lines = [f"图片统计: {report['counts']}",
             f"完全相同的跨 split 图片（val 侧）: {report['exact_cross_split_count']}"]
    for pair in report["exact_cross_split"][:preview]:
        lines.append(f"  val: {pair['val']}")
        lines.append(f"    == train: {', '.join(pair['train'])}")
    if report["truncated"]:
        lines.append("  ...（完整列表见 JSON 报告）")
    if report.get("exact_within_split_count"):
        lines.append(f"同一 split 内部重复组（不影响泄漏，仅供参考）: {report['exact_within_split_count']}")
    if report["perceptual_checked"]:
        lines.append(f"疑似同源增强重复（感知哈希距离 <= {report['near_max_distance']}）: {report['near_cross_split_count']}")
        for pair in report.get("near_cross_split", [])[:preview]:
            lines.append(f"  val: {pair['val']}  ~= train: {pair['train']}  (distance={pair['distance']})")
    else:
        lines.append("提示: 加 --perceptual 可额外检出「同一原图增强后分到两边」的近似重复")
    return "\n".join(lines)


def report_fingerprint(report: dict) -> str:
    payload = json.dumps(report.get("leaked_val_images", []), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
