"""Single-image region classification using detector proposals, without labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image

from .data import make_crop
from .engine import load_checkpoint, load_model, resolve_device, selected_head
from .metrics import decide_labels, merge_image_probabilities
from .transforms import PairTransform


def read_proposals(path: str | Path, size: tuple[int, int], confidence_threshold: float = 0.25) -> list[dict]:
    """Read YOLO prediction txt (cls cx cy w h [conf]) or pixel-XYXY JSON.

    The input class ID is deliberately ignored: classification is our model's job.
    JSON: {"boxes": [[x1,y1,x2,y2], ...], "scores": [0.9, ...]}.
    """
    path = Path(path)
    width, height = size
    proposals = []
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
            raise ValueError("boxes JSON 必须包含 boxes: [[x1,y1,x2,y2], ...]，坐标为原图像素")
        boxes, scores = value["boxes"], value.get("scores", [1.0]*len(value["boxes"]))
        if not isinstance(scores, list) or len(scores) != len(boxes):
            raise ValueError("scores 长度必须与 boxes 一致")
        for box, score in zip(boxes, scores, strict=True):
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
                raise ValueError("每个像素框必须包含 4 个有限数值")
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("score 必须在 0..1")
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                raise ValueError("像素框必须满足 x2>x1 且 y2>y1")
            x0, x1 = max(0.0, x0/width), min(1.0, x1/width)
            y0, y1 = max(0.0, y0/height), min(1.0, y1/height)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("像素框与图像没有有效交集")
            normalized = ((x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0)
            proposals.append((normalized, float(score)))
    else:
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) not in (5, 6):
                raise ValueError(f"{path}:{number}: 预测 txt 需要 5 或 6 列")
            try:
                values = list(map(float, fields))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: 非数值预测") from exc
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"{path}:{number}: NaN/Inf")
            cx, cy, bw, bh = values[1:5]
            score = values[5] if len(values) == 6 else 1.0
            if not all(0 <= x <= 1 for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0 or not 0 <= score <= 1:
                raise ValueError(f"{path}:{number}: 坐标必须归一化且宽高 >0，置信度必须在 0..1")
            proposals.append(((cx, cy, bw, bh), score))
    result = [{"bbox": bbox, "detector_confidence": score} for bbox, score in proposals if score >= confidence_threshold]
    if not result:
        raise ValueError("没有满足置信度阈值的缺陷候选框；不会把无检测框的图片自动判为正常")
    return result


@torch.inference_mode()
def predict_image(checkpoint_path: str | Path, image_path: str | Path, boxes: str | Path | None = None,
                  whole_image: bool = False, device_name: str = "auto", confidence_threshold: float = 0.25,
                  image_threshold: float | None = None, top1: bool = False, novel_only: bool = False,
                  save_crops: str | Path | None = None) -> dict:
    if whole_image == (boxes is not None):
        raise ValueError("请且仅请指定 --boxes 或 --whole-image")
    if image_threshold is not None and (not math.isfinite(image_threshold) or not 0 <= image_threshold <= 1):
        raise ValueError("image_threshold 必须在 0..1")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold 必须在 0..1")
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    if novel_only and checkpoint["mode"] != "discovery":
        raise ValueError("--novel-only 只适用于 discovery checkpoint")
    device = resolve_device(device_name)
    model = load_model(cfg, len(checkpoint["class_order"]), load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    head = selected_head(checkpoint)
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    proposals = [{"bbox": (0.5, 0.5, 1.0, 1.0), "detector_confidence": None}] if whole_image else read_proposals(boxes, image.size, confidence_threshold)
    transform = PairTransform(cfg["model"]["image_size"], training=False)
    records, tensors, masks_list, crops = [], [], [], []
    for proposal in proposals:
        if whole_image:
            crop = image
            mask = Image.new("L", image.size, 255)
            crop_box = (0, 0, image.width, image.height)
            area = image.width * image.height
        else:
            crop, mask, crop_box, area = make_crop(image, proposal["bbox"], cfg["data"]["padding"], cfg["data"]["min_crop_size"])
        tensor, mask_tensor = transform(crop, mask)
        tensors.append(tensor)
        masks_list.append(mask_tensor)
        crops.append((crop, mask))
        records.append({**proposal, "image_id": "input", "crop_box": list(crop_box), "mask_area": area})
    probabilities = []
    batch_size = cfg["training"]["batch_size"]
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start+batch_size]).to(device)
        masks = torch.stack(masks_list[start:start+batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=cfg["training"]["amp"] and device.type == "cuda"):
            _, logits = model(batch, masks)
        probs = torch.softmax(logits[head].float()/cfg["loss"]["temperature"], dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise FloatingPointError("推理输出 NaN/Inf")
        probabilities.append(probs.cpu())
    probabilities = torch.cat(probabilities).numpy()
    order = checkpoint["class_order"]
    if novel_only:
        n_known = checkpoint["n_known"]
        order = order[n_known:]
        probabilities = probabilities[:, n_known:]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    image_probs, _ = merge_image_probabilities(records, probabilities, cfg["inference"]["area_temperature"])
    threshold = cfg["inference"]["image_threshold"] if image_threshold is None else image_threshold
    if top1:
        threshold = None
    chosen = decide_labels(image_probs[0], threshold)
    names = checkpoint["names"]
    result = {
        "image": str(Path(image_path).resolve()), "image_size": list(image.size),
        "checkpoint": str(Path(checkpoint_path).resolve()), "mode": checkpoint["mode"],
        "selected_head": head, "class_order": order, "class_names": [names[i] for i in order],
        "probability_order": "class_order", "image_threshold": threshold,
        "image_prediction_ids": [order[i] for i in chosen],
        "image_prediction_names": [names[order[i]] for i in chosen],
        "image_probabilities": image_probs[0].tolist(),
        "regions": [{**record, "bbox_format": "normalized_cxcywh", "crop_box_format": "pixel_xyxy",
                     "prediction_id": order[int(probs.argmax())],
                     "prediction_name": names[order[int(probs.argmax())]],
                     "probabilities": probs.tolist()} for record, probs in zip(records, probabilities, strict=True)],
        "note": "Anomaly-region classification, not normal/abnormal detection. whole_image assumes the input contains a defect.",
        "random_backbone_smoke_test": bool(checkpoint.get("allow_random_init", False)),
    }
    if save_crops:
        directory = Path(save_crops).expanduser().resolve()
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"save-crops 目录非空: {directory}；请使用新目录，避免覆盖已有裁剪")
        directory.mkdir(parents=True, exist_ok=True)
        for index, (crop, mask) in enumerate(crops):
            crop.save(directory / f"region_{index:04d}.png")
            mask.save(directory / f"region_{index:04d}_mask.png")
        result["crops_directory"] = str(directory)
    return result
