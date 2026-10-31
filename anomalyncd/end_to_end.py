"""End-to-end FabricNCD: raw image -> detection -> localization -> classification -> score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import make_crop, make_crop_from_mask
from .detector import detect_defects, load_detector
from .engine import load_checkpoint, load_model, resolve_device, selected_head
from .metrics import decide_labels, merge_image_probabilities
from .transforms import PairTransform

BUNDLE_FORMAT = "fabricncd-end-to-end"
BUNDLE_VERSION = 2
SUPPORTED_BUNDLE_VERSIONS = (1, 2)


def save_bundle(directory: str | Path, detector_weights: str | Path | None,
                classifier_checkpoint: str | Path, cfg: dict, extra: dict | None = None,
                anomaly_bank: str | Path | None = None) -> Path:
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    classifier_target = directory / "classifier.pt"
    import shutil

    shutil.copy2(classifier_checkpoint, classifier_target)
    bundle = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
              "detector": None, "classifier": "classifier.pt",
              "mode": cfg["mode"], "config": cfg, **(extra or {})}
    if detector_weights is not None:
        shutil.copy2(detector_weights, directory / "detector.pt")
        bundle["detector"] = "detector.pt"
    if anomaly_bank is not None:
        source = Path(anomaly_bank).expanduser().resolve()
        if not (source / "bank.json").is_file():
            raise FileNotFoundError(f"异常特征库缺少 bank.json: {source}")
        target = directory / "anomaly_bank"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        bundle["anomaly_bank"] = "anomaly_bank"
    if bundle["detector"] is None and "anomaly_bank" not in bundle:
        raise ValueError("bundle 必须至少包含 YOLO 检测器或异常特征库之一")
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
    if (bundle.get("format") != BUNDLE_FORMAT
            or bundle.get("version") not in SUPPORTED_BUNDLE_VERSIONS):
        raise ValueError("不是兼容的 FabricNCD 端到端 bundle")
    root = path.parent
    bundle["root"] = str(root)
    bundle["detector_path"] = str((root / bundle["detector"]).resolve()) if bundle.get("detector") else None
    bundle["classifier_path"] = str((root / bundle["classifier"]).resolve())
    bundle["anomaly_bank_path"] = (str((root / bundle["anomaly_bank"]).resolve())
                                   if bundle.get("anomaly_bank") else None)
    return bundle


def box_iou(first: list[float], second: list[float]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    width = max(0.0, min(ax1, bx1)-max(ax0, bx0))
    height = max(0.0, min(ay1, by1)-max(ay0, by0))
    intersection = width*height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax1-ax0)*max(0.0, ay1-ay0)
    area_b = max(0.0, bx1-bx0)*max(0.0, by1-by0)
    union = area_a+area_b-intersection
    return float(intersection/union) if union > 0 else 0.0


def fuse_regions(detections: list[dict], anomaly_regions: list[dict],
                 iou_threshold: float = 0.3) -> list[dict]:
    """Merge supervised YOLO boxes with normal-only anomaly regions.

    An anomaly region with no matching YOLO box is the interesting case: the
    supervised detector never learned that defect, so it is flagged as a novel
    candidate.  Where both agree we keep the anomaly *mask* (tighter than a box)
    and record both confidences.
    """
    used: set[int] = set()
    fused: list[dict] = []
    for region in anomaly_regions:
        best_index, best_iou = None, 0.0
        for index, detection in enumerate(detections):
            if index in used:
                continue
            value = box_iou(region["bbox_xyxy"], detection["bbox_xyxy"])
            if value > best_iou:
                best_index, best_iou = index, value
        item = dict(region)
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            item["source"] = "fused"
            item["detector_confidence"] = float(detections[best_index]["detector_confidence"])
            item["fusion_iou"] = best_iou
            item["novel_candidate"] = False
        else:
            item["source"] = "anomaly_map"
            item["detector_confidence"] = None
            item["novel_candidate"] = True
        fused.append(item)
    for index, detection in enumerate(detections):
        if index in used:
            continue
        x0, y0, x1, y1 = detection["bbox_xyxy"]
        fused.append({"bbox_xyxy": list(detection["bbox_xyxy"]),
                      "detector_confidence": float(detection["detector_confidence"]),
                      "anomaly_score": None, "anomaly_score_raw": None,
                      "area": int(max(0.0, x1-x0)*max(0.0, y1-y0)),
                      "mask": None, "source": "detector", "novel_candidate": False})
    return fused


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
        # Fused regions carry anomaly_score=None when only YOLO fired, so an
        # explicit None check is required (dict.get's default would not apply).
        score = detection.get("anomaly_score")
        if score is None:
            score = detection.get("detector_confidence")
        score = float(score) if score is not None else 0.0
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
        score = detection.get("anomaly_score")
        if score is None:
            score = detection.get("detector_confidence")
        score_text = f"{float(score):.2f}" if score is not None else "n/a"
        label = f"#{index} {detection.get('prediction_name', 'defect')} {score_text}"
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
                      anomaly_score_threshold: float | None = None,
                      detector=None, anomaly_scorer=None) -> dict:
    """Raw image -> open-set anomaly map + YOLO proposals -> masks -> class + score.

    With an anomaly bank the localization is a real pixel-level map derived only
    from normal images, so defect types absent from training can still be found.
    Without a bank it falls back to the original box-supervised behaviour.
    """
    bundle = load_bundle(bundle_path)
    cfg = bundle["config"]
    detector_cfg = cfg.get("detector") or {}
    anomaly_cfg = cfg.get("anomaly") or {}
    conf_threshold = float(detector_cfg.get("conf_threshold", 0.25) if conf_threshold is None else conf_threshold)
    iou_threshold = float(detector_cfg.get("iou_threshold", 0.45) if iou_threshold is None else iou_threshold)
    anomaly_threshold = float(detector_cfg.get("anomaly_threshold", 0.25) if anomaly_threshold is None else anomaly_threshold)
    score_threshold = float(anomaly_cfg.get("score_threshold", 1.0)
                            if anomaly_score_threshold is None else anomaly_score_threshold)
    fuse_iou = float(anomaly_cfg.get("fuse_iou", 0.3))
    detector_image_size = int(detector_cfg.get("image_size", 640))
    if not 0 <= conf_threshold <= 1 or not 0 <= iou_threshold <= 1 or not 0 <= anomaly_threshold <= 1:
        raise ValueError("检测/异常阈值必须在 0..1")
    if novel_only and bundle["mode"] != "discovery":
        raise ValueError("--novel-only 只适用于 discovery bundle")

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    device = resolve_device(device_name)

    scorer, dense, bin_info, map_image_score = None, None, {}, 0.0
    anomaly_regions: list[dict] = []
    if bundle.get("anomaly_bank_path"):
        from .anomaly import AnomalyScorer, binarize, load_bank, regions_from_map

        bank = load_bank(bundle["anomaly_bank_path"])
        scorer = anomaly_scorer or AnomalyScorer(bank, cfg, device,
                                                 allow_random_init=bool(bank.get("random_init")))
        dense, raw_score = scorer.anomaly_map(image)
        map_image_score = scorer.normalize_image_score(raw_score)
        pixel_threshold = (bank.get("calibration") or {}).get("pixel_high")
        binary, bin_info = binarize(dense, scorer.settings, pixel_threshold)
        anomaly_regions = regions_from_map(dense, binary, scorer.settings,
                                           score_of=lambda v: float(scorer.normalize_pixel(v)))

    detections: list[dict] = []
    if bundle.get("detector_path"):
        active = detector if detector is not None else load_detector(bundle["detector_path"])
        detections = detect_defects(active, image_path, conf_threshold, iou_threshold,
                                    detector_image_size, device_name)

    proposals = fuse_regions(detections, anomaly_regions, fuse_iou)
    detector_score = max((d["detector_confidence"] for d in detections), default=0.0)
    from_map = scorer is not None and map_image_score >= score_threshold
    from_detector = detector_score >= anomaly_threshold and bool(detections)
    if scorer is None:
        is_anomaly = from_detector
        image_score = detector_score
        score_source = "detector_confidence"
    else:
        is_anomaly = bool(proposals) and (from_map or from_detector)
        image_score = map_image_score
        score_source = "normal_only_feature_distance"

    pipeline = ["detection", "anomaly_localization", "anomaly_classification", "anomaly_score"]
    if scorer is not None:
        pipeline.insert(0, "open_set_anomaly_map")
    result = {
        "image": str(Path(image_path).resolve()), "image_size": list(image.size),
        "bundle": str(Path(bundle_path).resolve()), "mode": bundle["mode"],
        "pipeline": pipeline, "is_anomaly": is_anomaly,
        "anomaly_score": float(image_score), "anomaly_score_source": score_source,
        "anomaly_threshold": score_threshold if scorer is not None else anomaly_threshold,
        "detector_confidence_threshold": conf_threshold,
        "anomaly_from_map": bool(from_map), "anomaly_from_detector": bool(from_detector),
        "detector_confidence_max": float(detector_score),
        "binarization": bin_info, "detections": detections, "regions": [],
        "image_prediction_ids": [], "image_prediction_names": [],
        "novel_candidate_count": sum(1 for p in proposals if p.get("novel_candidate")),
        "localization_source": "anomaly_map" if scorer is not None else "detector_boxes",
        "note": ("Open-set localization from a normal-only DINO feature bank; regions with no "
                 "matching YOLO box are novel-defect candidates."
                 if scorer is not None else
                 "Box-supervised fallback: heatmap is a Gaussian inside detector boxes, not segmentation."),
    }

    if not is_anomaly:
        result["prediction"] = "normal"
        result["classification_note"] = "未发现超过阈值的异常；判为正常，不强行输出缺陷类别"
        heatmap = (np.clip(scorer.normalize_pixel(dense), 0, 1).astype(np.float32)
                   if dense is not None else np.zeros((image.height, image.width), np.float32))
    else:
        checkpoint = load_checkpoint(bundle["classifier_path"])
        classifier_cfg = checkpoint["config"]
        model = load_model(classifier_cfg, len(checkpoint["class_order"]), load_pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        head = selected_head(checkpoint)
        transform = PairTransform(classifier_cfg["model"]["image_size"], training=False)
        order, names = checkpoint["class_order"], checkpoint["names"]
        records, tensors, masks_list = [], [], []
        for proposal in proposals:
            if proposal.get("mask") is not None:
                crop, mask, crop_box, area, bbox = make_crop_from_mask(
                    image, proposal["mask"], classifier_cfg["data"]["padding"],
                    classifier_cfg["data"]["min_crop_size"])
            else:
                bbox = _pixel_box_to_normalized(proposal["bbox_xyxy"], image.size)
                crop, mask, crop_box, area = make_crop(image, bbox, classifier_cfg["data"]["padding"],
                                                       classifier_cfg["data"]["min_crop_size"])
            tensor, mask_tensor = transform(crop, mask)
            tensors.append(tensor)
            masks_list.append(mask_tensor)
            records.append({"image_id": "input", "mask_area": area, "crop_box": list(crop_box),
                            "bbox": list(bbox), "detector_confidence": proposal.get("detector_confidence")})
        probabilities = []
        batch_size = int(classifier_cfg["training"]["batch_size"])
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start+batch_size]).to(device)
            masks = torch.stack(masks_list[start:start+batch_size]).to(device)
            with torch.autocast(device_type=device.type,
                                enabled=classifier_cfg["training"]["amp"] and device.type == "cuda"):
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
        image_probs, _ = merge_image_probabilities(records, probabilities,
                                                   classifier_cfg["inference"]["area_temperature"])
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
        for index, (proposal, record, probs) in enumerate(zip(proposals, records, probabilities, strict=True)):
            pred = int(probs.argmax())
            region = {key: value for key, value in proposal.items() if key != "mask"}
            region.update({
                "region_index": index, "crop_box": record["crop_box"],
                "bbox_normalized_cxcywh": record["bbox"], "mask_area": record["mask_area"],
                "mask_shape": "segmentation" if proposal.get("mask") is not None else "box",
                "prediction_id": order[pred], "prediction_name": names[order[pred]],
                "classification_confidence": float(probs[pred]),
                "probabilities": probs.tolist(),
            })
            if bundle["mode"] == "discovery" and not novel_only:
                region["novel_probability"] = float(probs[checkpoint["n_known"]:].sum())
            result["regions"].append(region)
        heatmap = (np.clip(scorer.normalize_pixel(dense), 0, 1).astype(np.float32)
                   if dense is not None else make_localization_map(image.size, result["regions"]))
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result["localization"] = save_localization(image, heatmap, result["regions"], output_dir)
        result["localization_map_max"] = float(heatmap.max())
        json_path = output_dir / "result.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["result_json"] = str(json_path)
    return result
