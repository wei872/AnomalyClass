"""YOLO bbox -> anomaly-centred RGB/mask crops, with an explicit manifest.

Only the target box is white in its mask. Different boxes/classes are never
silently merged into an ambiguous single-class training sample.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .config import DatasetSpec, label_root, load_dataset_spec

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCHEMA_VERSION = 1
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Annotation:
    class_id: int
    bbox: tuple[float, float, float, float]  # normalized YOLO cx,cy,w,h


@dataclass
class ImageEntry:
    path: Path
    relative: str
    label_path: Path
    size: tuple[int, int]
    annotations: list[Annotation]
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yolo_labels(path: Path, nc: int) -> list[Annotation]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 YOLO 标签: {path}；纯正常图片应放在 normal/images 下")
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: 需要 5 列 cls cx cy w h；不支持分割多边形")
        try:
            cls, cx, cy, width, height = map(float, fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: 标签包含非数值") from exc
        if not cls.is_integer() or not 0 <= int(cls) < nc:
            raise ValueError(f"{path}:{line_number}: 类别 {cls} 不在 0..{nc-1}")
        values = (cx, cy, width, height)
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"{path}:{line_number}: 标签包含 NaN/Inf")
        if not all(-1e-5 <= x <= 1 + 1e-5 for x in values) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: YOLO 坐标必须归一化，宽高必须 >0")
        cx, cy = min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))
        # Clip the box, not its centre, to the image boundary.
        x0, x1 = max(0.0, cx - width / 2), min(1.0, cx + width / 2)
        y0, y1 = max(0.0, cy - height / 2), min(1.0, cy + height / 2)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{path}:{line_number}: 框与图像没有有效交集")
        result.append(Annotation(int(cls), ((x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0)))
    return result


def scan_split(spec: DatasetSpec, split: str, allow_missing_labels: bool = False) -> tuple[list[ImageEntry], dict]:
    root = spec.paths[split]
    if not root.is_dir():
        raise FileNotFoundError(f"{split} 图片目录不存在: {root}；请使用 --data-root")
    labels = label_root(root)
    if not labels.is_dir():
        raise FileNotFoundError(f"{split} 标签目录不存在: {labels}")
    entries, missing = [], []
    seen_ids = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        label_path = labels / Path(relative).with_suffix(".txt")
        if not label_path.is_file():
            missing.append(str(label_path))
            if not allow_missing_labels:
                continue
            annotations = []
        else:
            annotations = read_yolo_labels(label_path, spec.nc)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if width < 2 or height < 2:
            raise ValueError(f"图片尺寸过小: {path}")
        image_id = f"{split}_{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]}"
        if image_id in seen_ids:
            raise ValueError(f"重复图片 ID: {image_id}")
        seen_ids.add(image_id)
        entries.append(ImageEntry(path.resolve(), relative, label_path.resolve(), (width, height), annotations, file_sha256(path)))
    if missing and not allow_missing_labels:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"{split}: 缺少 {len(missing)} 个标签文件，例如:\n{preview}\n"
                                "正常图应放 normal/images；确认是背景图时可用 --allow-missing-labels 跳过")
    if not any(e.annotations for e in entries):
        raise ValueError(f"{split}: 没有找到任何有效缺陷框，不能训练/评估区域分类器")
    return entries, {"images": len(entries), "annotated_images": sum(bool(e.annotations) for e in entries),
                     "regions": sum(len(e.annotations) for e in entries), "missing_labels": len(missing),
                     "empty_images": sum(not e.annotations for e in entries)}


def audit_normals(spec: DatasetSpec) -> dict[str, int | None]:
    result = {}
    for split in ("normaltrain", "normalval"):
        root = spec.paths[split]
        result[split] = sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in root.rglob("*")) if root.is_dir() else None
    return result


def crop_geometry(size: tuple[int, int], bbox: tuple[float, float, float, float],
                  padding: float = 0.1, min_crop_size: float = 0.1) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Square context crop, shifting at edges; pad only if a box exceeds the short side."""
    width, height = size
    cx, cy, bw, bh = bbox
    x0 = max(0, min(width - 1, math.floor((cx-bw/2) * width)))
    y0 = max(0, min(height - 1, math.floor((cy-bh/2) * height)))
    x1 = min(width, max(x0 + 1, math.ceil((cx+bw/2) * width)))
    y1 = min(height, max(y0 + 1, math.ceil((cy+bh/2) * height)))
    side = math.ceil(max((x1-x0)*(1+padding), (y1-y0)*(1+padding), max(width,height)*min_crop_size, 1))
    centre_x, centre_y = (x0+x1)//2, (y0+y1)//2
    left, top = centre_x-side//2, centre_y-side//2
    if side <= width:
        left = max(0, min(left, width-side))
    if side <= height:
        top = max(0, min(top, height-side))
    return (left, top, left+side, top+side), (x0, y0, x1, y1)


def make_crop(image: Image.Image, bbox: tuple[float, float, float, float], padding: float,
              min_crop_size: float) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], int]:
    image = image.convert("RGB")
    crop_box, pixel_box = crop_geometry(image.size, bbox, padding, min_crop_size)
    mask = Image.new("L", image.size, 0)
    x0, y0, x1, y1 = pixel_box
    mask.paste(255, (x0, y0, x1, y1))
    # PIL crop pads out-of-image coordinates with zero, identically for image/mask.
    crop, mask_crop = image.crop(crop_box), mask.crop(crop_box)
    area = sum(mask_crop.histogram()[1:])
    if area == 0:
        raise ValueError(f"空缺陷掩码: {bbox}, size={image.size}")
    return crop, mask_crop, crop_box, area


def _metadata_fingerprint(spec: DatasetSpec, entries: dict[str, list[ImageEntry]], padding: float, min_crop_size: float) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([spec.names, padding, min_crop_size, SCHEMA_VERSION], ensure_ascii=False).encode())
    for split in ("train", "val"):
        for entry in entries[split]:
            label = entry.label_path.read_bytes() if entry.label_path.is_file() else b"<missing>"
            boxes = [[a.class_id, list(a.bbox)] for a in entry.annotations]
            digest.update(json.dumps([split, entry.relative, entry.size, entry.sha256, boxes], sort_keys=True).encode())
            digest.update(hashlib.sha256(label).digest())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(dataset_config: str | Path, output: str | Path, data_root: str | Path | None = None,
                    padding: float = 0.1, min_crop_size: float = 0.1, overwrite: bool = False,
                    allow_missing_labels: bool = False, allow_overlap: bool = False,
                    drop_leaked_val: bool = False, check_only: bool = False,
                    progress: Callable[[str], None] = print) -> dict:
    if not math.isfinite(padding) or not 0 <= padding <= 10 or not math.isfinite(min_crop_size) or not 0 <= min_crop_size <= 1:
        raise ValueError("padding 必须在 0..10，min_crop_size 必须在 0..1")
    spec = load_dataset_spec(dataset_config, data_root)
    entries, stats = {}, {}
    for split in ("train", "val"):
        entries[split], stats[split] = scan_split(spec, split, allow_missing_labels)
        stats[split]["class_region_counts"] = dict(sorted(Counter(a.class_id for e in entries[split] for a in e.annotations).items()))
        progress(f"{split}: {stats[split]}")
    if allow_overlap and drop_leaked_val:
        raise ValueError("--allow-overlap 与 --drop-leaked-val 不能同时使用")
    train_hashes = {e.sha256 for e in entries["train"]}
    overlap = sorted({e.relative for e in entries["val"] if e.sha256 in train_hashes})
    stats["exact_train_val_overlap"] = len(overlap)
    if overlap and drop_leaked_val:
        before = len(entries["val"])
        entries["val"] = [e for e in entries["val"] if e.sha256 not in train_hashes]
        if not entries["val"]:
            raise ValueError("剔除与 train 重复的图片后 val 为空；请重新划分数据集")
        stats["val"] = {"images": len(entries["val"]),
                        "annotated_images": sum(bool(e.annotations) for e in entries["val"]),
                        "regions": sum(len(e.annotations) for e in entries["val"]),
                        "missing_labels": stats["val"]["missing_labels"],
                        "empty_images": sum(not e.annotations for e in entries["val"]),
                        "class_region_counts": dict(sorted(Counter(
                            a.class_id for e in entries["val"] for a in e.annotations).items()))}
        stats["dropped_leaked_val"] = before - len(entries["val"])
        progress(f"已从 val 剔除 {stats['dropped_leaked_val']} 个与 train 完全相同的图片（原始数据未改动）")
    elif overlap and not allow_overlap:
        raise ValueError(
            f"train/val 有 {len(overlap)} 个完全相同的图片（例如 {overlap[:3]}），会造成数据泄漏。\n"
            "推荐加 --drop-leaked-val 在预处理时排除这些 val 图片（原始数据不改动）；"
            "或先用 scripts/audit_split.py 审计并隔离；"
            "确认接受泄漏风险时才加 --allow-overlap。")
    elif overlap:
        progress(f"WARNING: 已显式允许 {len(overlap)} 个 train/val 重复图片；指标可能偏高")
    for split in ("train", "val"):
        missing = sorted(set(range(spec.nc)) - set(stats[split]["class_region_counts"]))
        if missing:
            progress(f"WARNING: {split} 缺少类别 {[spec.names[i] for i in missing]}")
    normals = audit_normals(spec)
    progress(f"normal 目录图片数: {normals}（19 类区域分类不将正常图当作第 20 类）")
    fingerprint = _metadata_fingerprint(spec, entries, padding, min_crop_size)
    manifest = {"format": "anomalyncd-fabric19", "schema_version": SCHEMA_VERSION,
                "names": spec.names, "nc": spec.nc, "fingerprint": fingerprint,
                "crop": {"padding": padding, "min_crop_size": min_crop_size, "mask_source": "yolo_bbox"},
                "source_config": spec.config_path.name, "stats": stats, "normal_counts": normals, "splits": {}}
    if check_only:
        return manifest
    output = Path(output).expanduser().resolve()
    # Never permit an output directory to contain/delete the input data.
    all_inputs = [spec.config_path, *spec.paths.values(), *[e.path for group in entries.values() for e in group],
                  *[e.label_path for group in entries.values() for e in group]]
    for source in all_inputs:
        if source == output or output in source.parents:
            raise ValueError(f"输出目录不能包含输入数据: {output}")
    for source in spec.paths.values():
        if source.is_dir() and source in output.parents:
            raise ValueError(f"输出目录不能位于输入图片/标签目录内: {output}")
    if output.exists() and any(output.iterdir()):
        old_manifest = output / "manifest.json"
        if not overwrite or not old_manifest.is_file():
            raise FileExistsError(f"输出目录非空: {output}；使用新目录，或确认旧预处理目录后加 --overwrite")
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        if old.get("format") != "anomalyncd-fabric19":
            raise ValueError(f"拒绝覆盖非本工具生成的目录: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex[:8]}")
    try:
        for split in ("train", "val"):
            records = []
            (stage / "images" / split).mkdir(parents=True)
            (stage / "masks" / split).mkdir(parents=True)
            for entry_index, entry in enumerate(entries[split]):
                if not entry.annotations:
                    continue
                image_id = f"{split}_{hashlib.sha256(entry.relative.encode()).hexdigest()[:20]}"
                with Image.open(entry.path) as source:
                    for box_index, annotation in enumerate(entry.annotations):
                        crop, mask, crop_box, area = make_crop(source, annotation.bbox, padding, min_crop_size)
                        sample_id = f"{image_id}_box{box_index:04d}"
                        image_rel, mask_rel = f"images/{split}/{sample_id}.png", f"masks/{split}/{sample_id}.png"
                        crop.save(stage / image_rel)
                        mask.save(stage / mask_rel)
                        records.append({"sample_id": sample_id, "image_id": image_id, "source": entry.relative,
                                        "image": image_rel, "mask": mask_rel, "class_id": annotation.class_id,
                                        "bbox": list(annotation.bbox), "crop_box": list(crop_box),
                                        "source_size": list(entry.size), "source_sha256": entry.sha256, "mask_area": area})
                if (entry_index+1) % 100 == 0:
                    progress(f"prepared {split}: {entry_index+1}/{len(entries[split])} images")
            manifest["splits"][split] = records
        _write_json(stage / "manifest.json", manifest)
        moved = False
        if output.exists():
            output.rename(backup)
            moved = True
        stage.rename(output)
        if moved:
            shutil.rmtree(backup)
        progress(f"预处理完成: {output}")
        return manifest
    except BaseException:
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_manifest(root: str | Path, expected_names: list[str] | None = None) -> dict:
    root = Path(root).expanduser().resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}；请先运行 python scripts/prepare_data.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "anomalyncd-fabric19" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"预处理格式/版本不兼容: {path}")
    if manifest.get("nc") != len(manifest.get("names", [])):
        raise ValueError("manifest 类别数量不一致")
    if expected_names is not None and manifest["names"] != expected_names:
        raise ValueError("预处理数据的类别名/顺序与 dataset YAML 不一致，请重新预处理")
    if set(manifest.get("splits", {})) != {"train", "val"}:
        raise ValueError("manifest 必须包含 train 和 val")
    return manifest


class FabricRegions(Dataset):
    """NCD training returns -100 for novel targets, never their class labels.

    Validation is a separate dataset. Region merging uses stable image_id rather
    than parsing filenames, so Chinese names, nested folders and dots are safe.
    """
    def __init__(self, root: str | Path, split: str, class_order: list[int], n_known: int,
                 mode: str, transform=None, evaluation: bool = False):
        if split not in {"train", "val"}:
            raise ValueError("split 必须是 train/val")
        if mode not in {"supervised", "discovery"}:
            raise ValueError(mode)
        self.root = Path(root).expanduser().resolve()
        self.manifest = load_manifest(self.root)
        if sorted(class_order) != list(range(self.manifest["nc"])):
            raise ValueError("class_order 必须是全部 YOLO 类别 ID 的一个排列")
        if not 0 <= n_known <= len(class_order):
            raise ValueError("n_known 越界")
        self.records = self.manifest["splits"][split]
        if not self.records:
            raise ValueError(f"{split} 没有区域样本")
        self.split, self.mode, self.evaluation = split, mode, evaluation
        self.class_order, self.n_known = list(class_order), n_known
        self.output_index = {class_id: index for index, class_id in enumerate(class_order)}
        self.transform = transform
        self.labels = [self.output_index[r["class_id"]] for r in self.records]
        self.image_ids = [r["image_id"] for r in self.records]
        seen_samples = set()
        for record in self.records:
            for key in ("sample_id", "image_id", "image", "mask", "mask_area", "class_id"):
                if key not in record:
                    raise ValueError(f"manifest 缺少字段 {key}")
            if record["sample_id"] in seen_samples:
                raise ValueError(f"manifest 重复 sample_id: {record['sample_id']}")
            seen_samples.add(record["sample_id"])
            if not isinstance(record["class_id"], int) or isinstance(record["class_id"], bool) or not 0 <= record["class_id"] < self.manifest["nc"]:
                raise ValueError(f"manifest 类别越界: {record['class_id']}")
            for key in ("image", "mask"):
                path = (self.root / record[key]).resolve()
                if self.root not in path.parents:
                    raise ValueError(f"manifest 路径越过数据目录: {record[key]}")
            if not isinstance(record["mask_area"], (int, float)) or not math.isfinite(record["mask_area"]) or record["mask_area"] <= 0:
                raise ValueError("mask_area 必须 >0")
        if mode == "discovery" and not evaluation:
            present_known = {self.labels[i] for i in range(len(self.labels)) if self.labels[i] < n_known}
            present_novel = {self.labels[i] for i in range(len(self.labels)) if self.labels[i] >= n_known}
            if len(present_known) != n_known or len(present_novel) != len(class_order)-n_known:
                raise ValueError("NCD 的 train 中每个已知类和新类都必须有样本；val 不参与补样本")

    @property
    def training_labels(self) -> list[int]:
        return [label if self.mode == "supervised" or label < self.n_known else IGNORE_INDEX for label in self.labels]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / record["mask"]) as handle:
            mask = handle.convert("L")
        if image.size != mask.size:
            raise ValueError(f"图片与掩码尺寸不一致: {record['sample_id']}")
        if self.transform is not None:
            images, masks = self.transform(image, mask)
        else:
            from .transforms import PairTransform
            images, masks = PairTransform(224, training=False)(image, mask)
            images, masks = [images], [masks]
        label = self.labels[index]
        labelled = self.mode == "supervised" or label < self.n_known
        target = label if self.evaluation or labelled else IGNORE_INDEX
        return {"images": images if isinstance(images, list) else [images],
                "masks": masks if isinstance(masks, list) else [masks], "target": target, "labelled": labelled,
                "index": index, "mask_area": float(record["mask_area"])}
