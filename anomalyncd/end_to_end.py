"""End-to-end FabricNCD: raw image -> detection -> localization -> classification -> score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import make_crop
from .detector import detect_defects, load_detector
from .engine import load_checkpoint, load_model, resolve_device, selected_head
from .metrics import decide_labels, merge_image_probabilities
from .transforms import PairTransform

BUNDLE_FORMAT = "fabricncd-end-to-end"
BUNDLE_VERSION = 1


def save_bundle(directory: str | Path, detector_weights: str | Path, classifier_checkpoint: str | Path,
                cfg: dict, extra: dict | None = None) -> Path:
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    detector_target = directory / "detector.pt"
    classifier_target = directory / "classifier.pt"
    import shutil

    shutil.copy2(detector_weights, detector_target)
    shutil.copy2(classifier_checkpoint, classifier_target)
    bundle = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
              "detector": "detector.pt", "classifier": "classifier.pt",
              "mode": cfg["mode"], "config": cfg, **(extra or {})}
    path = directory / "fabricncd_bundle.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_bundle(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "fabricncd_bundle.json"
    if not path.is_file():
        raise FileNotFoundError(f"FabricNCD bundle 不存在: {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("不是兼容的 FabricNCD 端到端 bundle")
    root = path.parent
    bundle["root"] = str(root)
    bundle["detector_path"] = str((root / bundle["detector"]).resolve())
    bundle["classifier_path"] = str((root / bundle["classifier"]).resolve())
    return bundle


def _pixel_box_to_normalized(box: list[float], size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = size
    x0, y0, x1, y1 = box
    x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
    y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"无效检测框: {box}")
    return ((x0+x1)/2/width, (y0+y1)/2/height, (x1-x0)/width, (y1-y0)/height)


def make_localization_map(size: tuple[int, int], detections: list[dict]) -> np.ndarray:
    """Box-supervised anomaly localization map in [0, 1].

    This is a weakly-supervised heatmap from detector boxes/scores, not a
    pixel-accurate segmentation mask.  Overlapping boxes take the maximum score.
    """
    width, height = size
    heatmap = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        x0, y0, x1, y1 = detection["bbox_xyxy"]
        left, right = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
        top, bottom = max(0, int(np.floor(y0))), min(height, int(np.ceil(y1)))
        if right <= left or bottom <= top:
            continue
        score = float(detection.get("anomaly_score", detection.get("detector_confidence", 0.0)))
        yy, xx = np.mgrid[top:bottom, left:right]
        cx, cy = (left+right)/2, (top+bottom)/2
        sx, sy = max(1.0, (right-left)/2), max(1.0, (bottom-top)/2)
        gaussian = np.exp(-0.5*(((xx-cx)/sx)**2 + ((yy-cy)/sy)**2))
        region = score * (0.5 + 0.5*gaussian)
        heatmap[top:bottom, left:right] = np.maximum(heatmap[top:bottom, left:right], region)
    return heatmap


def save_localization(image: Image.Image, heatmap: np.ndarray, detections: list[dict],
                      output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    heat_uint8 = np.clip(heatmap*255, 0, 255).astype(np.uint8)
    heat_image = Image.fromarray(heat_uint8, mode="L")
    heat_path = output_dir / "anomaly_heatmap.png"
    heat_image.save(heat_path)
    base = image.convert("RGB")
    red = Image.new("RGB", base.size, (255, 0, 0))
    mask = heat_image.point(lambda value: int(value*0.55))
    overlay = Image.composite(red, base, mask)
    draw = ImageDraw.Draw(overlay)
    for index, detection in enumerate(detections):
        x0, y0, x1, y1 = [int(round(v)) for v in detection["bbox_xyxy"]]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)
        label = f"#{index} {detection.get('prediction_name', 'defect')} {detection.get('anomaly_score', 0):.2f}"
        draw.text((x0+2, max(0, y0-12)), label, fill=(0, 255, 0))
    overlay_path = output_dir / "anomaly_overlay.jpg"
    overlay.save(overlay_path, quality=95)
    return {"heatmap": str(heat_path), "overlay": str(overlay_path)}


@torch.inference_mode()
def predict_fabricncd(bundle_path: str | Path, image_path: str | Path,
                      output_dir: str | Path | None = None, device_name: str = "auto",
                      conf_threshold: float | None = None, iou_threshold: float | None = None,
                      anomaly_threshold: float | None = None, image_threshold: float | None = None,
                      top1: bool = False, novel_only: bool = False,
                      detector=None) -> dict:
    """Run all four stages on one raw image without external boxes or labels."""
    bundle = load_bundle(bundle_path)
    cfg = bundle["config"]
    detector_cfg = cfg.get("detector", {})
    conf_threshold = float(detector_cfg.get("conf_threshold", 0.25) if conf_threshold is None else conf_threshold)
    iou_threshold = float(detector_cfg.get("iou_threshold", 0.45) if iou_threshold is None else iou_threshold)
    anomaly_threshold = float(detector_cfg.get("anomaly_threshold", 0.25) if anomaly_threshold is None else anomaly_threshold)
    detector_image_size = int(detector_cfg.get("image_size", 640))
    if not 0 <= conf_threshold <= 1 or not 0 <= iou_threshold <= 1 or not 0 <= anomaly_threshold <= 1:
        raise ValueError("检测/异常阈值必须在 0..1")
    if novel_only and bundle["mode"] != "discovery":
        raise ValueError("--novel-only 只适用于 discovery bundle")

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    if detector is None:
        detector = load_detector(bundle["detector_path"])
    detections = detect_defects(detector, image_path, conf_threshold, iou_threshold,
                                detector_image_size, device_name)
    image_anomaly_score = max((d["detector_confidence"] for d in detections), default=0.0)
    is_anomaly = bool(detections) and image_anomaly_score >= anomaly_threshold
    result = {
        "image": str(Path(image_path).resolve()), "image_size": list(image.size),
        "bundle": str(Path(bundle_path).resolve()), "mode": bundle["mode"],
        "pipeline": ["detection", "anomaly_localization", "anomaly_classification", "anomaly_score"],
        "is_anomaly": is_anomaly, "anomaly_score": float(image_anomaly_score),
        "anomaly_threshold": anomaly_threshold, "detector_confidence_threshold": conf_threshold,
        "detections": detections, "regions": [], "image_prediction_ids": [], "image_prediction_names": [],
        "note": "End-to-end box-supervised FabricNCD. Localization heatmap is weakly supervised by detector boxes, not pixel segmentation.",
    }
    if not is_anomaly:
        result["prediction"] = "normal"
        result["classification_note"] = "未检测到超过阈值的缺陷；判为正常，不强行输出缺陷类别"
        heatmap = np.zeros((image.height, image.width), dtype=np.float32)
    else:
        checkpoint = load_checkpoint(bundle["classifier_path"])
        classifier_cfg = checkpoint["config"]
        device = resolve_device(device_name)
        model = load_model(classifier_cfg, len(checkpoint["class_order"]), load_pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        head = selected_head(checkpoint)
        transform = PairTransform(classifier_cfg["model"]["image_size"], training=False)
        order = checkpoint["class_order"]
        names = checkpoint["names"]
        records, tensors, masks_list = [], [], []
        for detection in detections:
            bbox = _pixel_box_to_normalized(detection["bbox_xyxy"], image.size)
            crop, mask, crop_box, area = make_crop(image, bbox, classifier_cfg["data"]["padding"],
                                                   classifier_cfg["data"]["min_crop_size"])
            tensor, mask_tensor = transform(crop, mask)
            tensors.append(tensor)
            masks_list.append(mask_tensor)
            records.append({"image_id": "input", "mask_area": area, "crop_box": list(crop_box),
                            "bbox": list(bbox), "detector_confidence": detection["detector_confidence"]})
        probabilities = []
        batch_size = int(classifier_cfg["training"]["batch_size"])
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start+batch_size]).to(device)
            masks = torch.stack(masks_list[start:start+batch_size]).to(device)
            with torch.autocast(device_type=device.type, enabled=classifier_cfg["training"]["amp"] and device.type == "cuda"):
                _, logits = model(batch, masks)
            probs = torch.softmax(logits[head].float()/classifier_cfg["loss"]["temperature"], dim=-1)
            if not bool(torch.isfinite(probs).all()):
                raise FloatingPointError("分类器输出 NaN/Inf")
            probabilities.append(probs.cpu())
        probabilities = torch.cat(probabilities).numpy()
        if novel_only:
            n_known = checkpoint["n_known"]
            order = order[n_known:]
            probabilities = probabilities[:, n_known:]
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        image_probs, _ = merge_image_probabilities(records, probabilities, classifier_cfg["inference"]["area_temperature"])
        threshold = classifier_cfg["inference"]["image_threshold"] if image_threshold is None else image_threshold
        if top1:
            threshold = None
        chosen = decide_labels(image_probs[0], threshold)
        result["image_prediction_ids"] = [order[i] for i in chosen]
        result["image_prediction_names"] = [names[order[i]] for i in chosen]
        result["image_probabilities"] = image_probs[0].tolist()
        result["class_order"] = order
        result["class_names"] = [names[i] for i in order]
        result["prediction"] = "anomaly"
        for index, (detection, record, probs) in enumerate(zip(detections, records, probabilities, strict=True)):
            pred = int(probs.argmax())
            region = {
                **detection,
                "region_index": index,
                "crop_box": record["crop_box"],
                "bbox_normalized_cxcywh": record["bbox"],
                "prediction_id": order[pred], "prediction_name": names[order[pred]],
                "classification_confidence": float(probs[pred]),
                "anomaly_score": float(detection["detector_confidence"]),
                "probabilities": probs.tolist(),
            }
            if bundle["mode"] == "discovery" and not novel_only:
                n_known = checkpoint["n_known"]
                region["novel_probability"] = float(probs[n_known:].sum())
            result["regions"].append(region)
        heatmap = make_localization_map(image.size, result["regions"])
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result["localization"] = save_localization(image, heatmap, result["regions"], output_dir)
        result["localization_map_max"] = float(heatmap.max())
        json_path = output_dir / "result.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["result_json"] = str(json_path)
    return result
