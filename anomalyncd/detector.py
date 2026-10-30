"""Class-agnostic defect detector for end-to-end FabricNCD.

The detector is trained from YOLO boxes plus normal images as background.  In
NCD mode it must remain class-agnostic so novel class IDs are never used.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

from .config import DatasetSpec, load_dataset_spec
from .data import IMAGE_EXTENSIONS, file_sha256, scan_split

DETECTOR_FORMAT = "fabricncd-detector-data"


def _copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_detector_dataset(dataset_config: str | Path, output: str | Path,
                             data_root: str | Path | None = None, class_agnostic: bool = True,
                             mode: str = "supervised", known_class_ids: list[int] | None = None,
                             overwrite: bool = False, allow_overlap: bool = False,
                             check_only: bool = False, progress=print) -> dict:
    """Build a YOLO detection dataset with defect boxes and normal backgrounds.

    Normal images receive empty label files.  With ``class_agnostic`` every defect
    box becomes class 0 (defect); this is required for discovery mode.
    """
    if mode == "discovery" and not class_agnostic:
        raise ValueError("NCD 检测器必须 class_agnostic=true，避免使用新类类别 ID")
    spec: DatasetSpec = load_dataset_spec(dataset_config, data_root)
    output = Path(output).expanduser().resolve()
    all_inputs = [spec.config_path, *spec.paths.values()]
    for source in all_inputs:
        if source == output or output in source.parents:
            raise ValueError(f"输出目录不能包含输入数据: {output}")
    for source in spec.paths.values():
        if source.is_dir() and source in output.parents:
            raise ValueError(f"输出目录不能位于输入目录内: {output}")
    if output.exists() and any(output.iterdir()):
        old = output / "detector_manifest.json"
        if not overwrite or not old.is_file():
            raise FileExistsError(f"检测器数据目录非空: {output}；使用新目录或 --overwrite")
        if json.loads(old.read_text(encoding="utf-8")).get("format") != DETECTOR_FORMAT:
            raise ValueError(f"拒绝覆盖非检测器数据目录: {output}")

    entries, stats = {}, {}
    for split in ("train", "val"):
        entries[split], stats[split] = scan_split(spec, split)
    train_hashes = {e.sha256 for e in entries["train"]}
    overlap = sorted({e.relative for e in entries["val"] if e.sha256 in train_hashes})
    if overlap and not allow_overlap:
        raise ValueError(f"train/val 有 {len(overlap)} 个完全相同图片，会造成检测器数据泄漏")

    nc = 1 if class_agnostic else spec.nc
    names = ["defect"] if class_agnostic else spec.names
    manifest = {"format": DETECTOR_FORMAT, "schema_version": 1, "nc": nc, "names": names,
                "class_agnostic": class_agnostic, "mode": mode, "stats": stats, "splits": {}}
    if check_only:
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    try:
        for split in ("train", "val"):
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for entry in entries[split]:
                rel = Path("defect") / entry.relative
                target_image = image_dir / rel
                target_label = label_dir / rel.with_suffix(".txt")
                _copy_image(entry.path, target_image)
                lines = []
                for annotation in entry.annotations:
                    cls = 0 if class_agnostic else annotation.class_id
                    cx, cy, bw, bh = annotation.bbox
                    lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                records.append({"image": str(rel), "label": str(rel.with_suffix(".txt")),
                                "kind": "defect", "regions": len(lines), "sha256": entry.sha256})
            normal_split = "normaltrain" if split == "train" else "normalval"
            normal_root = spec.paths[normal_split]
            normal_count = 0
            if normal_root.is_dir():
                for path in sorted(normal_root.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    with Image.open(path) as image:
                        image.verify()
                    rel = Path("normal") / path.relative_to(normal_root)
                    target_image = image_dir / rel
                    target_label = label_dir / rel.with_suffix(".txt")
                    _copy_image(path, target_image)
                    target_label.parent.mkdir(parents=True, exist_ok=True)
                    target_label.write_text("", encoding="utf-8")
                    records.append({"image": str(rel), "label": str(rel.with_suffix(".txt")),
                                    "kind": "normal", "regions": 0, "sha256": file_sha256(path)})
                    normal_count += 1
            stats[split]["normal_images"] = normal_count
            stats[split]["total_images"] = len(records)
            manifest["splits"][split] = records
            progress(f"detector {split}: {stats[split]}")
        data_yaml = {"path": str(output), "train": "images/train", "val": "images/val",
                     "nc": nc, "names": names}
        (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (output / "detector_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        # Only remove a directory created by this call, and only on failure.
        if output.exists() and not (output / "data.yaml").exists():
            shutil.rmtree(output)
        raise


def train_detector(cfg: dict, data_yaml: str | Path, output_dir: str | Path,
                   device_name: str = "auto", allow_random_init: bool = False,
                   progress=print) -> Path:
    """Train a lightweight YOLO defect detector and return best weights."""
    import torch
    from ultralytics import YOLO

    detector = cfg.get("detector", {})
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")
    device = "0" if device_name in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    if allow_random_init:
        model_source = detector.get("model_yaml", "yolov8n.yaml")
        pretrained = False
    else:
        model_source = detector.get("weights") or detector.get("model", "yolov8n.pt")
        pretrained = True
    progress(f"training detector: {model_source} device={device}")
    model = YOLO(str(model_source))
    model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=int(detector.get("epochs", 100)),
        imgsz=int(detector.get("image_size", 640)),
        batch=int(detector.get("batch_size", 16)),
        device=device,
        workers=int(cfg["training"].get("num_workers", 4)),
        seed=int(cfg["experiment"].get("seed", 3407)),
        project=str(output_dir),
        name="run",
        exist_ok=True,
        pretrained=pretrained,
        amp=bool(cfg["training"].get("amp", True)) and device != "cpu",
        plots=False,
        verbose=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    weights = best if best.is_file() else last
    if not weights.is_file():
        raise FileNotFoundError(f"检测器训练没有生成权重: {save_dir}")
    return weights


def load_detector(weights: str | Path):
    from ultralytics import YOLO

    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"检测器权重不存在: {weights}")
    return YOLO(str(weights))


def detect_defects(detector, image_path: str | Path, conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45, image_size: int = 640,
                   device_name: str = "auto") -> list[dict]:
    """Run detector on a raw image; return pixel XYXY boxes and defect scores."""
    import torch

    device = "0" if device_name in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    results = detector.predict(source=str(Path(image_path).resolve()), conf=conf_threshold,
                               iou=iou_threshold, imgsz=image_size, device=device, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        for box, score in zip(xyxy, conf, strict=True):
            detections.append({"bbox_xyxy": [float(x) for x in box], "detector_confidence": float(score)})
    detections.sort(key=lambda item: item["detector_confidence"], reverse=True)
    return detections
