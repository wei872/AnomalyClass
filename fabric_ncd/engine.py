"""训练 / 验证流程.

流程概览:

1. **记忆库预热**: 用训练集里的正常布面图跑一遍前向, 建立正常 patch 记忆库
   (不需要任何疵点标签, 因此新布种可以直接冷启动).
2. **联合训练**: 异常分割 + 布面/布边 + 疵点分类 + 对比学习 + 记忆库紧凑度.
   每个 batch 里同时有正常图与疵点图 (默认 4:6).
3. **周期刷新记忆库**: 每 ``memory_refresh_interval`` 轮用最新的正常特征
   重建记忆库, 避免嵌入漂移导致分数尺度变化.
4. **验证**: 像素 AUROC / PRO / F1 + 图像 AUROC / mAP@0.5 + 分类 Acc / F1.
   检测框由异常区域经区域分析 + 区域合并得到 (端到端只用一个模型).
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.yolo import (
    DefectBackgroundDataset,
    build_background_loader,
    build_dataloaders,
    build_normal_loader,
    build_reference_loaders,
    load_dataset_spec,
)
from .losses import FabricCriterion
from .metrics import MetricAccumulator, auroc, grouped_auroc
from .models.fabric_ncd import FabricNCD
from .ncd import RegionAnalyzer, postprocess_outputs
from .ops import bbox_iou, box_filter, threshold_from_normal_pixels, topk_mean
from .utils import (
    AverageMeter,
    Timer,
    describe_device,
    checkpoint_compat_config,
    ensure_dir,
    format_seconds,
    get_logger,
    json_dump,
    load_model_weights,
    move_to,
    resolve_device,
    save_yaml,
    set_seed,
)


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.num_updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # 训练初期用较小的 decay (EMA warmup), 否则前若干步几乎完全等于初始化权重
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        # BN 统计量直接同步，不做平滑；记忆库各自维护。
        for (ema_buf_name, ema_buf), (buf_name, buf) in zip(
            self.ema.named_buffers(), model.named_buffers()
        ):
            if "memory" in ema_buf_name:
                continue
            if ema_buf.shape == buf.shape and ema_buf.dtype == buf.dtype:
                ema_buf.copy_(buf.detach())
        # EMA 的嵌入与训练模型不同，记忆库由 warmup_memory 分别构建，不能逐步覆盖。


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _reshuffled(cfg: Dict[str, Any], loader) -> Optional[Any]:
    """按 loader 的种类重新建一个**洗牌版** (只取前 N 批时用, 避免每轮都抽同一批图)."""
    dataset = getattr(loader, "dataset", None)
    if isinstance(dataset, DefectBackgroundDataset):
        return build_background_loader(cfg, "train", shuffle=True)
    if dataset is not None:
        return build_normal_loader(cfg, "train", shuffle=True)
    return None


def _interleave(loaders):
    """轮询不同来源，限制批次数时也能覆盖每种来源。"""
    active = [iter(loader) for loader in loaders]
    while active:
        remaining = []
        for iterator in active:
            try:
                yield next(iterator)
                remaining.append(iterator)
            except StopIteration:
                pass
        active = remaining


def analyzer_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """``RegionAnalyzer`` 的参数统一从 ``eval`` 段取.

    以前 peak_ratio / bbox_refine_ratio / topk_ratio / open_radius 四项只写在配置里、
    代码从不读取 (恰好等于 RegionAnalyzer 的默认值, 所以改配置也没反应).
    """
    ev = cfg.get("eval", {}) or {}
    return {
        "use_cloth": bool(ev.get("cloth_gate", True)),
        "threshold": ev.get("threshold"),
        "sigma": float(ev.get("threshold_sigma", 3.0)),
        "min_area_ratio": float(ev.get("min_area_ratio", 1e-4)),
        "merge_strategy": str(ev.get("merge_strategy", "area_weighted")),
        "open_radius": int(ev.get("open_radius", 0)),
        "peak_ratio": float(ev.get("peak_ratio", 0.35)),
        "bbox_refine_ratio": float(ev.get("bbox_refine_ratio", 0.7)),
        "topk_ratio": float(ev.get("topk_ratio", 0.01)),
    }


def _source_of(path: str, depth: int = 1) -> str:
    """文件名前缀 = 布料批次号 (DA9670287_... -> DA9670287)."""
    parts = [p for p in Path(str(path)).stem.split("_") if p]
    return "_".join(parts[:depth]) if parts else Path(str(path)).stem


def gate_by_cloth(score, cloth, enabled: bool = True):
    """按布面掩码裁剪异常分数. 整幅都是布时应关闭, 否则会把疵点本身裁掉."""
    if not enabled:
        return score
    return score * (cloth > 0.5)


def image_score_of(score: np.ndarray, eval_cfg: dict) -> float:
    """图像级异常分数 (评估 / 校准 / 推理共用一套口径).

    ``eval.image_score_mode``:
      - ``topk``   (默认) 最高若干像素的均值;
      - ``contrast`` 上述峰值**减去本图中位数**: 抹掉"整图底色"级别的逐图偏移.
        异常图的亮度/纹理整体偏低时, 绝对分数会被整图偏移主导, 表现为
        imageAUROC < 0.5 (正常图反而更高), 而局部对比度仍然是对的.
    """
    k_pixels = int(eval_cfg.get("topk_pixels", 0) or 0)
    peak = topk_mean(score, float(eval_cfg.get("topk_ratio", 0.01)), k_pixels=k_pixels or None)
    mode = str(eval_cfg.get("image_score_mode", "topk")).lower()
    if mode == "contrast":
        return float(peak - float(np.median(score)))
    return float(peak)


def analysis_stride_of(cfg: Dict[str, Any]) -> int:
    """区域后处理/阈值校准共用的下采样倍数.

    以前 ``evaluate`` 在 1/8 特征上做区域分析, ``evaluate_tiled`` 用全分辨率,
    推理又用 ``infer.out_stride``; 三套分辨率对应三套阈值. 现在统一读
    ``eval.analysis_stride``, 未配置时回退到推理的 ``out_stride``.
    """
    ev = cfg.get("eval", {}) or {}
    raw = ev.get("analysis_stride")
    if raw is None:
        raw = (cfg.get("infer", {}) or {}).get("out_stride", 1)
    if raw is None:
        raw = 1
    return max(1, int(raw))


def _downsample_np(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr
    h, w = arr.shape[-2:]
    sh, sw = max(8, int(math.ceil(h / stride))), max(8, int(math.ceil(w / stride)))
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(sh, sw), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _downsample_mask_np(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr.astype(bool)
    h, w = arr.shape[-2:]
    sh, sw = max(8, int(math.ceil(h / stride))), max(8, int(math.ceil(w / stride)))
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(sh, sw), mode="nearest")
    return t[0, 0].numpy() > 0.5


@torch.no_grad()
def evaluate(
    model: FabricNCD,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    criterion: Optional[FabricCriterion] = None,
    analyzer: Optional[RegionAnalyzer] = None,
    max_samples: Optional[int] = None,
    collect_predictions: bool = False,
    calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    model.eval()
    names = _class_names(cfg)
    analyzer = analyzer or RegionAnalyzer(names=names, **analyzer_kwargs(cfg))
    # 未显式指定阈值时优先使用正常集校准出的绝对阈值
    if analyzer.threshold is None and calibration:
        analyzer.threshold = float(calibration.get("pixel_threshold", 0.0)) or None
    eval_cfg = cfg.get("eval", {})
    gate = bool(eval_cfg.get("cloth_gate", True))
    analysis_stride = analysis_stride_of(cfg)
    acc = MetricAccumulator(num_classes=len(names), names=names)
    if calibration and calibration.get("pixel_threshold_full") is not None:
        acc.pixel_threshold = float(calibration["pixel_threshold_full"])
    elif analyzer.threshold is not None:
        acc.pixel_threshold = float(analyzer.threshold)
    loss_meter = AverageMeter()
    predictions: List[Dict[str, Any]] = []
    seen = 0
    defect_images = 0
    detected_images = 0
    region_total = 0
    # 框外误报: 与任何 GT 框都不重叠的预测区域. 没有同批次良品图时, 这才是
    # 图上真正的"假阳"——每张疵点图的框外区域就是这块布的正常纹理.
    outside_fp = 0
    outside_fp_images = 0
    normal_region_total = 0
    normal_images = 0
    # 图像级聚合方式对比: 小疵点用 top-k 均值容易被稀释, 这里同时统计 peak 与小 top-k
    agg = {"topk": [], "top16": [], "max": []}
    agg_labels: List[int] = []
    agg_src: List[str] = []

    for batch in loader:
        if max_samples is not None:
            remaining = int(max_samples) - seen
            if remaining <= 0:
                break
            batch_size = batch["image"].shape[0]
            if remaining < batch_size:
                batch = {
                    key: (value[:remaining] if isinstance(value, (torch.Tensor, list, tuple))
                          and len(value) == batch_size else value)
                    for key, value in batch.items()
                }
        batch = move_to(batch, device)
        outputs = model(batch["image"])
        if criterion is not None:
            targets = {
                "defect_mask": batch["anomaly_mask"],
                "cloth_mask": batch["cloth_mask"],
                "boxes": batch["boxes"],
            }
            loss, _ = criterion(outputs, targets, memory=model.memory)
            loss_meter.update(float(loss), batch["image"].shape[0])

        h, w = batch["image"].shape[-2:]
        score = F.interpolate(outputs["anomaly_score"], size=(h, w), mode="bilinear", align_corners=False)
        cloth = F.interpolate(outputs["cloth_prob"], size=(h, w), mode="bilinear", align_corners=False)
        score = gate_by_cloth(score, cloth, gate)
        score_np = score.squeeze(1).detach().cpu().numpy()

        analysis_outputs = outputs
        if analysis_stride > 1:
            ah, aw = max(8, int(math.ceil(h / analysis_stride))), max(8, int(math.ceil(w / analysis_stride)))
            analysis_outputs = dict(outputs)
            analysis_outputs["anomaly_score"] = F.interpolate(
                outputs["anomaly_score"], size=(ah, aw), mode="bilinear", align_corners=False
            )
            analysis_outputs["cloth_prob"] = F.interpolate(
                outputs["cloth_prob"], size=(ah, aw), mode="bilinear", align_corners=False
            )
        region_results = postprocess_outputs(analysis_outputs, analyzer, input_size=(h, w))
        for i in range(score.shape[0]):
            normal = bool(batch["normal"][i].item())
            image_score = image_score_of(score_np[i], eval_cfg)
            agg["topk"].append(image_score)
            agg["top16"].append(topk_mean(score_np[i], k_pixels=16))
            agg["max"].append(float(score_np[i].max()))
            agg_labels.append(0 if normal else 1)
            agg_src.append(_source_of(str(batch["path"][i])))
            infos, image_level = region_results[i]
            region_total += len(infos)
            if normal:
                normal_images += 1
                normal_region_total += len(infos)
            if not normal:
                defect_images += 1
                if infos:
                    detected_images += 1
            gt_mask = batch["anomaly_mask"][i, 0].detach().cpu().numpy()

            gt_labels = batch["labels"][i].detach().cpu().numpy().astype(int)
            gt_boxes_np = batch["boxes"][i].detach().cpu().numpy()
            gt_dict = {
                "boxes": _xyxy_pixels(gt_boxes_np, w, h),
                "labels": gt_labels,
            }
            pred_boxes, pred_scores, pred_labels = _regions_to_boxes(infos, w, h, cfg)
            if not normal:  # 分母包含没有预测区域的疵点图
                gt_xyxy = gt_dict["boxes"]
                outside_fp_images += 1
                for box in pred_boxes:
                    if len(gt_xyxy) == 0:
                        outside_fp += 1
                        continue
                    if max(bbox_iou(list(map(float, box[:4])), list(map(float, g[:4]))) for g in gt_xyxy) < 0.10:
                        outside_fp += 1

            multi_true, multi_pred = -1, -1
            if len(gt_labels):
                counts = np.bincount(gt_labels)
                multi_true = int(counts.argmax())
                if image_level.get("probs") is not None:
                    multi_pred = int(np.argmax(image_level["probs"]))

            acc.update(
                anomaly_score=score_np[i],
                anomaly_gt=gt_mask,
                normal=normal,
                image_score=image_score,
                pred={"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels},
                gt=gt_dict,
                multi_true=multi_true,
                multi_pred=multi_pred,
            )
            if collect_predictions:
                predictions.append(
                    {
                        "path": batch["path"][i],
                        "image_score": image_score,
                        "normal": normal,
                        "regions": [r.as_dict(names) for r in infos],
                        "image_probs": None if image_level.get("probs") is None else image_level["probs"].tolist(),
                        "image_cls": multi_pred,
                        "gt_cls": multi_true,
                        "threshold": image_level.get("threshold"),
                        "weave": _weave_to_dict(outputs["weave"], i),
                    }
                )
        seen += score.shape[0]
        if max_samples and seen >= max_samples:
            break

    metrics = acc.compute()
    if criterion is not None:
        metrics["val_loss"] = loss_meter.avg
    metrics["num_samples"] = seen
    metrics["det_rate"] = detected_images / defect_images if defect_images else 0.0
    metrics["num_defect_images"] = defect_images
    metrics["regions_per_image"] = region_total / max(1, seen)
    # 框外误报密度 = 每张图的假阳区域数 (与任何 GT 框 IoU<0.1 的区域).
    # 没有同批次良品图时, 这是唯一能在"这块布"上量化假阳的指标.
    metrics["outside_fp_per_image"] = outside_fp / max(1, outside_fp_images)
    metrics["normal_regions_per_image"] = normal_region_total / max(1, normal_images)
    if agg_labels and 0 < sum(agg_labels) < len(agg_labels):
        for key, values in agg.items():
            lot_auc, lot_pairs = grouped_auroc(agg_labels, values, agg_src)
            if key == "topk":
                metrics["image_auroc_same_lot"] = lot_auc
                metrics["image_auroc_same_lot_pairs"] = lot_pairs
            metrics[f"image_auroc_{key}"] = auroc(np.asarray(agg_labels), np.asarray(values))
    if collect_predictions:
        metrics["predictions"] = predictions
    return metrics


@torch.no_grad()
def evaluate_tiled(
    model: "FabricNCD",
    cfg: Dict[str, Any],
    device: torch.device,
    split: str = "val",
    analyzer: Optional[RegionAnalyzer] = None,
    calibration: Optional[Dict[str, float]] = None,
    max_samples: Optional[int] = None,
    collect_predictions: bool = False,
    tile_size: Optional[int | Tuple[int, int]] = None,
    tile_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """大幅面切块评估 —— 使用与 ``FabricPredictor(tile=True)`` 相同的地图和区域分析.

    整图缩放会丢细节时 (小疵点只有几像素), 评估必须和推理走同一条路径:
    按原始像素尺度切块 -> 融合异常图 -> 区域分析 -> 计算指标.
    仅有整图缩放的 ``evaluate`` 会低估小疵点的检出能力.

    像素级指标按 ``eval.max_pixels`` 下采样后再统计 (AUROC/PRO 对下采样不敏感),
    检测框/分类指标仍使用原始坐标, 因此与部署时的输出一致.
    """
    from .inference import _resolve_image_size, _resize_map, as_hw, forward_maps
    from .data.yolo import FabricDataset, boxes_to_mask, load_rgb, read_yolo_labels

    model.eval()
    spec = load_dataset_spec(cfg)
    names = list(spec.names) if spec.names else _class_names(cfg)
    eval_cfg = cfg.get("eval", {})
    analyzer = analyzer or RegionAnalyzer(names=names, **analyzer_kwargs(cfg))
    if analyzer.threshold is None and calibration:
        analyzer.threshold = float(calibration.get("pixel_threshold", 0.0)) or None

    acc = MetricAccumulator(num_classes=len(names), names=names)
    if calibration and calibration.get("pixel_threshold_full") is not None:
        acc.pixel_threshold = float(calibration["pixel_threshold_full"])
    elif analyzer.threshold is not None:
        acc.pixel_threshold = float(analyzer.threshold)

    image_size = _resolve_image_size(cfg)
    infer_cfg = cfg.get("infer", {}) or {}
    tile_hw = as_hw(tile_size or eval_cfg.get("tile_size") or infer_cfg.get("tile_size") or image_size)
    overlap = int(tile_overlap if tile_overlap is not None else eval_cfg.get("tile_overlap", infer_cfg.get("tile_overlap", 128)))
    tile_batch = int(eval_cfg.get("tile_batch", infer_cfg.get("tile_batch", 4)))
    gate = bool(eval_cfg.get("cloth_gate", True))
    max_pixels = float(eval_cfg.get("max_pixels", 3e5))
    out_stride = max(1, int(infer_cfg.get("out_stride", 4)))
    probs_mul = max(1, int(infer_cfg.get("probs_mul", 2)))
    target_period = float(infer_cfg.get("weave_target_period", 8.0))
    memory_stride = infer_cfg.get("memory_stride")
    if memory_stride is not None and getattr(model, "memory", None) is not None:
        model.memory.query_stride = max(1, int(memory_stride))

    source = FabricDataset._collect(spec.splits.get(split), spec.labels.get(split), None, False, split)
    normal_items = FabricDataset._collect(spec.normal_splits.get(split), None, None, True, split)
    items = [item for item in source if not item.normal] + normal_items + [item for item in source if item.normal]
    limit = cfg.get("data", {}).get("limit_train" if split == "train" else "limit_val")
    if limit and int(limit) < len(items):
        items = FabricDataset._stratified_limit(items, int(limit))
    if max_samples is not None:
        items = items[:max(0, int(max_samples))]

    predictions: List[Dict[str, Any]] = []
    defect_images = detected_images = region_total = outside_fp = normal_images = normal_region_total = 0
    for entry in items:
        path, is_normal = entry.image, entry.normal
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        if is_normal:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            gt_labels = np.zeros((0,), dtype=np.int64)
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes, polygons = read_yolo_labels(entry.label)
            if len(boxes) and spec.nc > 0 and ((boxes[:, 0] < 0) | (boxes[:, 0] >= spec.nc)).any():
                raise ValueError(f"标签类别超出 0..{spec.nc - 1}: {entry.label}")
            gt_mask = boxes_to_mask(boxes, polygons, h, w)
            gt_labels = boxes[:, 0].astype(np.int64) if len(boxes) else np.zeros((0,), dtype=np.int64)
            gt_boxes = _xyxy_pixels(boxes, w, h)

        maps = forward_maps(
            model, rgb, device, image_size,
            tile=True, tile_size=tile_hw, tile_overlap=overlap, tile_batch=tile_batch,
            out_stride=out_stride, probs_mul=probs_mul,
            weave_norm=bool(infer_cfg.get("weave_norm", True)), target_period=target_period,
        )
        score = gate_by_cloth(maps["score"], maps["cloth"], gate)
        infos, image_level = analyzer.analyze(
            maps["score"], maps["cloth"], maps["probs"], probs_scale=probs_mul
        )
        score_for_image = _resize_map(score, (h, w)) if infer_cfg.get("image_score_full_res", True) else score
        image_score = image_score_of(score_for_image, eval_cfg)

        # 与部署共享低分辨率融合图，仅在像素指标需要时放大或限额下采样。
        factor = max(1, int(math.ceil(math.sqrt((h * w) / max(max_pixels, 1.0)))))
        sample_h, sample_w = math.ceil(h / factor), math.ceil(w / factor)
        sampled_gt = F.interpolate(torch.from_numpy(gt_mask.astype(np.float32))[None, None],
                                   size=(sample_h, sample_w), mode="nearest")[0, 0].numpy()
        sampled_score = _resize_map(score, (sample_h, sample_w))
        pred_boxes, pred_scores, pred_labels = _regions_to_boxes(infos, w, h, cfg)
        region_total += len(infos)
        if is_normal:
            normal_images += 1
            normal_region_total += len(infos)
        else:
            defect_images += 1
            detected_images += bool(infos)
            for box in pred_boxes:
                if not len(gt_boxes) or max(bbox_iou(box[:4], gt[:4]) for gt in gt_boxes) < 0.10:
                    outside_fp += 1

        multi_true, multi_pred = -1, -1
        if len(gt_labels):
            multi_true = int(np.bincount(gt_labels).argmax())
            if image_level.get("probs") is not None:
                multi_pred = int(np.argmax(image_level["probs"]))

        acc.update(
            anomaly_score=sampled_score,
            anomaly_gt=sampled_gt,
            normal=is_normal,
            image_score=image_score,
            pred={"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels},
            gt={"boxes": gt_boxes, "labels": gt_labels},
            multi_true=multi_true,
            multi_pred=multi_pred,
        )
        if collect_predictions:
            predictions.append(
                {
                    "path": str(path),
                    "image_score": image_score,
                    "normal": is_normal,
                    "regions": [r.as_dict(names) for r in infos],
                    "image_probs": None if image_level.get("probs") is None else image_level["probs"].tolist(),
                    "image_cls": multi_pred,
                    "gt_cls": multi_true,
                    "threshold": image_level.get("threshold"),
                    "tile": list(tile_hw),
                }
            )

    metrics = acc.compute()
    metrics["num_samples"] = len(items)
    metrics["det_rate"] = detected_images / defect_images if defect_images else 0.0
    metrics["num_defect_images"] = defect_images
    metrics["regions_per_image"] = region_total / max(1, len(items))
    metrics["outside_fp_per_image"] = outside_fp / max(1, defect_images)
    metrics["normal_regions_per_image"] = normal_region_total / max(1, normal_images)
    metrics["tile_size"] = list(tile_hw)
    metrics["tile_overlap"] = overlap
    if collect_predictions:
        metrics["predictions"] = predictions
    return metrics


def _class_names(cfg: Dict[str, Any]) -> List[str]:
    data_cfg = cfg.get("data", {})
    names = data_cfg.get("names")
    if names:
        return list(names)
    try:
        spec = load_dataset_spec(cfg)
        if spec.names:
            return list(spec.names)
        num = spec.nc
    except Exception:  # pragma: no cover
        num = 0
    num = int(num or cfg.get("model", {}).get("heads", {}).get("num_classes", 19))
    return [f"class_{i}" for i in range(num)]


def _xyxy_pixels(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.reshape(-1, 5)
    out = np.stack([boxes[:, 1] * w, boxes[:, 2] * h, boxes[:, 3] * w, boxes[:, 4] * h], axis=1)
    return out.astype(np.float32)


def _regions_to_boxes(
    infos, w: int, h: int, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_cls_score = float(cfg.get("eval", {}).get("min_cls_score", 0.0))
    boxes, scores, labels = [], [], []
    for region in infos:
        if region.cls < 0 or region.cls_score < min_cls_score:
            continue
        x1, y1, x2, y2 = region.bbox
        boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
        scores.append(region.score)
        labels.append(region.cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def _weave_to_dict(weave: Dict[str, torch.Tensor], index: int) -> Dict[str, float]:
    if not weave:
        return {}
    return {
        "period_x": float(weave["period_x"][index]),
        "period_y": float(weave["period_y"][index]),
        "anisotropy": float(weave["anisotropy"][index]),
        "confidence": float(weave["confidence"][index]),
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: Dict[str, Any], logger=None) -> None:
        self.cfg = cfg
        exp = cfg.get("experiment", {})
        self.output_dir = ensure_dir(exp.get("output_dir", "runs/fabric19"))
        self.logger = logger or get_logger("fabric_ncd", str(self.output_dir / "train.log"))
        self.device = resolve_device(exp.get("device", "auto"))
        set_seed(int(exp.get("seed", 42)))

        self.train_cfg = cfg.get("train", {})
        self.epochs = int(self.train_cfg.get("epochs", 60))
        self.amp = bool(exp.get("amp", True)) and self.device.type == "cuda"
        scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
        self.scaler = (scaler_cls(enabled=self.amp) if scaler_cls is not None
                       else torch.cuda.amp.GradScaler(enabled=self.amp))

        # 类别数以数据集 yaml 为准, 避免配置与标注不一致导致越界
        try:
            spec = load_dataset_spec(cfg)
            if spec.nc:
                configured = int(cfg.get("model", {}).get("heads", {}).get("num_classes", spec.nc))
                if configured != spec.nc:
                    self.logger.warning(
                        "配置中 num_classes=%d 与数据集 nc=%d 不一致, 已按数据集覆盖",
                        configured,
                        spec.nc,
                    )
                cfg.setdefault("model", {}).setdefault("heads", {})["num_classes"] = int(spec.nc)
                if spec.names:
                    cfg.setdefault("data", {})["names"] = list(spec.names)
        except Exception as exc:  # pragma: no cover - 数据缺失时保持配置
            self.logger.warning("读取数据集类别信息失败: %s", exc)

        self.datasets = build_dataloaders(cfg)
        data_cfg = cfg.get("data", {})
        self.logger.info(
            "训练设备: %s | AMP: %s | num_workers: %s",
            describe_device(self.device),
            "on" if self.amp else "off",
            data_cfg.get("num_workers"),
        )
        self.logger.info(
            "输入尺寸: %s | batch_size: %s | epochs: %s",
            data_cfg.get("image_size"),
            data_cfg.get("batch_size"),
            self.epochs,
        )
        self._warn_memory(data_cfg)
        self.train_loader = self.datasets["train"]
        self.val_loader = self.datasets["val"]
        # 参照集 (记忆库 + 阈值校准) 的来源:
        #   normal         = normal/images/train 里的良品图;
        #   defect_outside = 疵点图的**框外布面** (没有同批次良品图时唯一可信的参照);
        #   both           = 两者都用.
        self.memory_source = str(self.train_cfg.get("memory_source", "both")).lower()
        self.normal_loader = build_normal_loader(cfg, "train")
        self.background_loader = (
            build_background_loader(cfg, "train")
            if "defect" in self.memory_source or "both" in self.memory_source
            else None
        )
        self.memory_loaders = []
        if ("defect" in self.memory_source or "both" in self.memory_source) and self.background_loader is not None:
            self.memory_loaders.append(self.background_loader)
        if self.normal_loader is not None and (
            "normal" in self.memory_source or "both" in self.memory_source or not self.memory_loaders
        ):
            self.memory_loaders.append(self.normal_loader)
        if not self.memory_loaders:
            self.logger.warning(
                "参照集为空 (memory_source=%s): 记忆分支将退化为常数", self.memory_source
            )
        else:
            kinds = []
            if self.background_loader is not None and self.background_loader in self.memory_loaders:
                kinds.append(f"疵点图框外布面 {len(self.background_loader.dataset)} 张")
            if self.normal_loader is not None and self.normal_loader in self.memory_loaders:
                kinds.append(f"良品图 {len(self.normal_loader.dataset)} 张")
            self.logger.info("参照集 (记忆库/阈值校准): %s", " + ".join(kinds) or "无")

        self.model = FabricNCD(cfg).to(self.device)
        self.criterion = FabricCriterion(cfg).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.ema = ModelEMA(self.model, float(self.train_cfg.get("ema", 0.999))) if self.train_cfg.get("ema", 0.999) > 0 else None
        # 参数全部来自 eval 段 (含 cloth_gate: 漏传时配置会被静默忽略)
        self.analyzer = RegionAnalyzer(names=_class_names(cfg), **analyzer_kwargs(cfg))

        self.start_epoch = 0
        self.calibration: Dict[str, float] = {}
        self._memory_stale = False
        self._calib_calls = 0
        self.best_score = -float("inf")
        self.bad_epochs = 0
        self._bank_paths = set()
        self.history: List[Dict[str, Any]] = []
        self.global_step = 0
        self.logger.info(
            "模型参数量: %.2fM (主干 %.2fM)",
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            sum(p.numel() for p in self.model.backbone.parameters()) / 1e6,
        )

    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.train_cfg.get("lr", 5e-4))
        bb_lr = float(self.train_cfg.get("backbone_lr", lr * 0.2))
        wd = float(self.train_cfg.get("weight_decay", 0.05))
        groups = self.model.trainable_parameters(backbone_lr=bb_lr, base_lr=lr)
        return torch.optim.AdamW(groups, weight_decay=wd)

    def _build_scheduler(self):
        warmup = int(self.train_cfg.get("warmup_epochs", 3))
        steps_per_epoch = max(1, len(self.train_loader))
        total = self.epochs * steps_per_epoch
        warmup_steps = max(1, warmup * steps_per_epoch)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------ #
    def _reference_batches(self, skip_bank: bool = True):
        """参照集批次 (良品图 + / 或 疵点图框外布面). 记忆库与阈值校准共用同一套来源.

        ``skip_bank``: 跳过已经进了记忆库的图. 这一步必须做 —— 如果校准用的就是建库用的
        那几张, 正常像素到库的中位距离会被压到 ≈0 (实测 0.001), 于是
        ``memory_score = d/(d+scale)`` 对任何没入库的图都直接饱和到 1.0, 记忆分支退化
        成一个常数偏置, 图像级分数整体被它带平.
        """
        bank = getattr(self, "_bank_paths", set()) if skip_bank else set()
        for batch in _interleave(getattr(self, "memory_loaders", []) or []):
            paths = [str(p) for p in batch.get("path", [])]
            if not paths:
                yield batch
                continue
            keep = [i for i, p in enumerate(paths) if p not in bank]
            if not keep:
                continue
            if len(keep) == len(paths):
                yield batch
                continue
            filtered = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == len(paths):
                    filtered[key] = value[keep]
                elif isinstance(value, (list, tuple)) and len(value) == len(paths):
                    filtered[key] = [value[i] for i in keep]
                else:
                    filtered[key] = value
            yield filtered

    @torch.no_grad()
    def _bank_self_distance(self, chunk: int = 512) -> float:
        """库内每个特征到其他特征的最近余弦距离，用于检测特征塌缩。"""
        bank = self.model.memory
        n = int(bank.size.item())
        if n < 2:
            return 0.0
        features = F.normalize(bank.features[:n].float(), dim=1)
        distances = []
        for start in range(0, n, chunk):
            sim = features[start:start + chunk] @ features.T
            row = torch.arange(sim.shape[0], device=sim.device)
            sim[row, row + start] = -float("inf")
            distances.append((1.0 - sim.max(dim=1).values).clamp_min(0))
        return float(torch.cat(distances).mean())

    def warmup_memory(self, max_batches: Optional[int] = None, calibrate: bool = True) -> None:
        if not self.memory_loaders or not self.model.memory_enabled:
            self.logger.warning("没有可用的正常图, 跳过记忆库预热 (将仅使用学习式异常头)")
            return
        self.model.eval()
        loaders = list(self.memory_loaders)
        # 只取前 max_batches 批时, 不洗牌 = 每轮都只看最前面那几十张;
        # 若这些图与验证图的布料批次/曝光不同, 记忆库会系统性漏掉某些正常外观.
        # 洗牌后每个 epoch 抽到的子集不同, 覆盖率随训练累积.
        if max_batches and bool(self.train_cfg.get("memory_shuffle", True)):
            loaders = [_reshuffled(self.cfg, loader) or loader for loader in loaders]
        batches = 0
        self._bank_paths = set()
        with torch.no_grad():
            for batch in _interleave(loaders):
                for path in batch.get("path", []):
                    self._bank_paths.add(str(path))
                batch = move_to(batch, self.device)
                self.model.update_memory(batch["image"], mask=batch["cloth_mask"])
                if self.ema is not None and self.ema.num_updates > 0:
                    self.ema.ema.update_memory(batch["image"], mask=batch["cloth_mask"])
                batches += 1
                if max_batches and batches >= max_batches:
                    break
        self.model.train()
        self.logger.info(
            "记忆库预热完成: %d 个 patch 特征 (来自 %d 张参照图; 阈值校准会跳过这些图)",
            int(self.model.memory.size.item()), len(self._bank_paths),
        )
        # 启动时 EMA 与模型参数相同，可直接复制；训练开始后分别用各自嵌入建库。
        if self.ema is not None and self.ema.num_updates == 0:
            self.ema.ema.memory.features.copy_(self.model.memory.features)
            self.ema.ema.memory.size.copy_(self.model.memory.size)
        if calibrate:
            self.calibrate()

    @torch.no_grad()
    def scale_sweep(
        self,
        scales,
        memory_batches: int = 12,
        max_samples: int = 60,
    ) -> List[Dict[str, Any]]:
        """用**当前权重**扫不同输入尺度, 判断"看不清"是不是瓶颈.

        训练结束时自动跑一次: 不需要留 checkpoint, 也不用重训就能回答"要不要提分辨率".
        pixelAUROC 与阈值无关, 所以早期 checkpoint 也能看出趋势.
        记忆库在这其中会被临时重建, 结束后原样恢复.
        """
        model = self.ema.ema if self.ema is not None else self.model
        was_training = model.training
        model.eval()
        saved_features = model.memory.features.detach().clone()
        saved_size = model.memory.size.detach().clone()
        base_cfg = copy.deepcopy(self.cfg)
        results: List[Dict[str, Any]] = []
        base_time = 0.0
        try:
            for scale in scales:
                cfg_s = copy.deepcopy(base_cfg)
                data_cfg = cfg_s.setdefault("data", {})
                data_cfg["image_size"] = "auto"
                data_cfg["short_side"] = 8192  # 交给 max_scale 决定缩放, 允许 >1 放大
                data_cfg["max_scale"] = float(scale)
                datasets = build_dataloaders(cfg_s)
                val_loader = datasets["val"]
                resolved = tuple(cfg_s["data"]["image_size"])
                model.memory.reset()
                sweep_loaders = build_reference_loaders(cfg_s, "train", shuffle=False)
                built = 0
                for batch in _interleave(sweep_loaders):
                    model.update_memory(
                        batch["image"].to(self.device), mask=batch["cloth_mask"].to(self.device)
                    )
                    built += 1
                    if built >= memory_batches:
                        break
                started = time.time()
                metrics = evaluate(
                    model, val_loader, cfg_s, self.device, criterion=None,
                    analyzer=RegionAnalyzer(names=None, **analyzer_kwargs(cfg_s)),
                    max_samples=max_samples or None,
                )
                elapsed = time.time() - started
                base_time = base_time or elapsed
                row = {
                    "scale": float(scale),
                    "input_hw": resolved,
                    "pixel_auroc": float(metrics.get("pixel_auroc", float("nan"))),
                    "image_auroc": float(metrics.get("image_auroc", float("nan"))),
                    "seconds": elapsed,
                }
                results.append(row)
                self.logger.info(
                    "  尺度 %.2f | 输入 %dx%d | pixelAUROC %.4f | imageAUROC %.4f | %.0fs",
                    row["scale"], resolved[0], resolved[1],
                    row["pixel_auroc"], row["image_auroc"], elapsed,
                )
        finally:
            model.memory.features.copy_(saved_features)
            model.memory.size.copy_(saved_size)
            model.train(was_training)

        if len(results) >= 2:
            delta = results[-1]["pixel_auroc"] - results[0]["pixel_auroc"]
            if delta > 0.05:
                self.logger.info(
                    "  分辨率判读: 提高尺度 pixelAUROC +%.3f -> **分辨率是瓶颈**, 建议粗到细 (整图粗筛 + 候选区复检) 或切块推理",
                    delta,
                )
            else:
                self.logger.info(
                    "  分辨率判读: 提高尺度 pixelAUROC %+.3f -> 分辨率不是主要瓶颈, 别再吃显存, 去调损失/增广/标注",
                    delta,
                )
        return results

    def _set_memory_scale(self, scale: float) -> None:
        """记忆项尺度同步给训练模型与 EMA (两者都会被用于评估/推理)."""
        for m in (self.model, self.ema.ema if self.ema is not None else None):
            if m is not None and hasattr(m, "memory_distance_scale"):
                m.memory_distance_scale = float(scale)

    @torch.no_grad()
    def calibrate(
        self,
        quantile: Optional[float] = None,
        image_quantile: Optional[float] = None,
        max_batches: int = 40,
        max_pixels: int = 500_000,
    ) -> Dict[str, float]:
        """用正常布面图校准绝对阈值.

        没有校准的话, "每张图都取相对最高的 k%" 会让正常图也切出区域 (必然假阳).
        这里统计正常图的像素级/图像级异常分数分布, 取高分位作为固定阈值, 使
        异常分数具有跨图可比性. 校准结果随 checkpoint 一起保存.
        """
        if not getattr(self, "memory_loaders", None):
            return self.calibration
        eval_cfg = self.cfg.get("eval", {})
        analysis_stride = analysis_stride_of(self.cfg)
        q = float(quantile if quantile is not None else eval_cfg.get("normal_quantile", 0.999))
        qi = float(image_quantile if image_quantile is not None else eval_cfg.get("image_quantile", 0.995))
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        # 一次前向同时收集: 学习式分数 / 记忆库距离 / 布面掩码.
        # 拿到距离分布后先定记忆项尺度, 再据此**重算**综合分数与阈值, 不用多跑一遍.
        learned_parts: List[np.ndarray] = []
        dist_parts: List[np.ndarray] = []
        mask_parts: List[np.ndarray] = []
        # 像素阈值与区域分析统一到 analysis_stride; 图像级分数仍用全分辨率.
        pixel_learned: List[np.ndarray] = []
        pixel_dist: List[np.ndarray] = []
        pixel_masks: List[np.ndarray] = []
        seen = 0
        # 若参照图全部进了记忆库 (memory_warmup_batches = 0 / 图很少), 跳过就什么都不剩,
        # 这时退回全部图 —— 尺度会被压小, 但至少不会算不出阈值 (会由 scale floor 兜底).
        n_ref = sum(len(getattr(l, "dataset", [])) for l in getattr(self, "memory_loaders", []))
        skip_bank = len(getattr(self, "_bank_paths", ())) < n_ref
        if not skip_bank:
            self.logger.warning(
                "参照图全部在记忆库里 (共 %d 张), 阈值校准只能复用这些图; "
                "记忆项尺度会被压小, 请增大参照集或减小 memory_warmup_batches",
                n_ref,
            )
        for batch_index, batch in enumerate(self._reference_batches(skip_bank=skip_bank)):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_to(batch, self.device)
            out = model(batch["image"])
            h, w = batch["image"].shape[-2:]
            learned = F.interpolate(
                out["learned_score"], size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1).float().cpu().numpy()
            cloth = F.interpolate(out["cloth_prob"], size=(h, w), mode="bilinear", align_corners=False)
            keep = (
                (cloth > 0.5).squeeze(1).cpu().numpy()
                if bool(eval_cfg.get("cloth_gate", True))
                else np.ones_like(cloth.squeeze(1).cpu().numpy(), dtype=bool)
            )
            # 只用"真的是布"的像素: 框外布面样本的 cloth_mask 就是它的取样区域,
            # 良品图则是真实布面掩码. 不乘这一项, 校准会把背景/疵点像素也算进正常分布.
            if batch.get("cloth_mask") is not None:
                keep = keep & (batch["cloth_mask"].squeeze(1).cpu().numpy() > 0.5)
            dist = out.get("memory_distance")
            if dist is None:
                dist_np = np.zeros_like(learned)
            else:
                if tuple(dist.shape[-2:]) != (h, w):
                    dist = F.interpolate(dist.float(), size=(h, w), mode="bilinear", align_corners=False)
                dist_np = dist.squeeze(1).float().cpu().numpy()
            for i in range(learned.shape[0]):
                if not keep[i].any():  # 这张图没有可信的布面像素, 不能当"正常"用
                    continue
                step = max(1, learned[i].size // 20000)
                learned_parts.append(learned[i].ravel()[::step])
                dist_parts.append(dist_np[i].ravel()[::step])
                mask_parts.append(keep[i].ravel()[::step])
                if analysis_stride > 1:
                    pl = _downsample_np(learned[i], analysis_stride)
                    pd = _downsample_np(dist_np[i], analysis_stride)
                    pm = _downsample_mask_np(keep[i], analysis_stride)
                else:
                    pl, pd, pm = learned[i], dist_np[i], keep[i]
                pixel_learned.append(pl)
                pixel_dist.append(pd)
                pixel_masks.append(pm)
                seen += 1
        if not learned_parts:
            return self.calibration

        learned_all = np.concatenate(learned_parts)
        dist_all = np.concatenate(dist_parts)
        mask_all = np.concatenate(mask_parts) > 0.5
        if not mask_all.any():
            mask_all = np.ones_like(learned_all, dtype=bool)

        # 记忆项尺度 = 正常像素距离中位数 (中位数正好映射到 0.5), 让尺度跟着嵌入走
        scale = float(self.calibration.get("memory_distance_scale", self.model.memory_distance_scale))
        use_memory = bool(self.model.memory_enabled) and bool(np.any(dist_all[mask_all] > 0))
        if use_memory:
            median_dist = float(np.median(dist_all[mask_all]))
            # 尺度过小 = 参照像素本来就在库里 (余弦距离≈0), 记忆分支会整体饱和到 1.0.
            # 正常是 0.01~0.05 量级; 真的掉到 0 附近时先兜底, 并明确告警.
            floor = float(self.train_cfg.get("memory_scale_floor", 0.01))
            if median_dist < floor:
                self.logger.warning(
                    "记忆项尺度只有 %.4f (正常 0.01~0.05): 校准用的像素基本就在记忆库里, "
                    "记忆分支会整体饱和 -> 已抬到 %.3f. 检查参照集是否与建库图重叠",
                    median_dist, floor,
                )
                median_dist = floor
            scale = median_dist
            self._set_memory_scale(scale)
            # 中位距离 ~1.0 意味着"每个正常像素在记忆库里都找不到相似的特征"
            # (余弦相似度≈0), 此时记忆分支退化成常数, 只剩学习式分数在起作用.
            # 原因通常是记忆库建好之后嵌入又变了, 需要重建 (memory_refresh_interval).
            self._memory_stale = median_dist > float(
                self.cfg.get("train", {}).get("memory_stale_distance", 0.5)
            )
            if self._memory_stale:
                self.logger.warning(
                    "记忆库与当前嵌入脱节: 正常像素到库的中位距离 %.3f (余弦相似度≈%.2f), "
                    "记忆分支此刻基本无区分度 -> 建议把 train.memory_refresh_interval 设为 1",
                    median_dist,
                    1.0 - median_dist,
                )

        weight = float(self.model.memory_weight)
        pixel_scores: List[np.ndarray] = []
        for pl, pd, pm in zip(pixel_learned, pixel_dist, pixel_masks):
            if use_memory:
                mem = pd / (pd + scale)
                s = (1.0 - weight) * pl + weight * mem
            else:
                s = pl
            smooth = int(getattr(self.analyzer, "smooth", 1))
            if smooth > 1:
                s = box_filter(s, smooth // 2)
            pixel_scores.append(s[pm].astype(np.float32))
        pixel_values = np.concatenate([s.ravel() for s in pixel_scores]) if pixel_scores else np.zeros(0, np.float32)
        if pixel_values.size > max_pixels:
            idx = np.random.default_rng(0).choice(pixel_values.size, max_pixels, replace=False)
            pixel_values = pixel_values[idx]
        k = float(eval_cfg.get("calib_sigma", 6.0))
        thr_stats = threshold_from_normal_pixels(pixel_values, q, k)
        # 图像级分数和像素级指标都跑全分辨率, 阈值必须也在全分辨率分布上取.
        # 第一遍只收集了 analysis_stride 的像素阈值; 这里再跑一遍拿全分辨率
        # score, 避免把 1/4 分辨率的分布套到全分辨率指标上.
        full_parts: List[np.ndarray] = []
        image_scores: List[float] = []
        seen_full = 0
        for batch_index, batch in enumerate(self._reference_batches(skip_bank=skip_bank)):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_to(batch, self.device)
            out = model(batch["image"])
            h, w = batch["image"].shape[-2:]
            learned = F.interpolate(
                out["learned_score"], size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1).float().cpu().numpy()
            cloth = F.interpolate(out["cloth_prob"], size=(h, w), mode="bilinear", align_corners=False)
            keep = (
                (cloth > 0.5).squeeze(1).cpu().numpy()
                if bool(eval_cfg.get("cloth_gate", True))
                else np.ones_like(cloth.squeeze(1).cpu().numpy(), dtype=bool)
            )
            if batch.get("cloth_mask") is not None:
                keep = keep & (batch["cloth_mask"].squeeze(1).cpu().numpy() > 0.5)
            dist = out.get("memory_distance")
            if dist is None:
                dist_np = np.zeros_like(learned)
            else:
                if tuple(dist.shape[-2:]) != (h, w):
                    dist = F.interpolate(dist.float(), size=(h, w), mode="bilinear", align_corners=False)
                dist_np = dist.squeeze(1).float().cpu().numpy()
            for i in range(learned.shape[0]):
                if not keep[i].any():
                    continue
                if use_memory:
                    mem = dist_np[i] / (dist_np[i] + scale)
                    score = (1.0 - weight) * learned[i] + weight * mem
                else:
                    score = learned[i]
                valid_score = score[keep[i]].astype(np.float32)
                image_scores.append(image_score_of(valid_score, eval_cfg))
                full_parts.append(valid_score[:: max(1, valid_score.size // 20000)])
                seen_full += 1
        pixel_values_full = np.concatenate(full_parts) if full_parts else np.zeros(0, np.float32)
        if pixel_values_full.size > max_pixels:
            idx = np.random.default_rng(0).choice(pixel_values_full.size, max_pixels, replace=False)
            pixel_values_full = pixel_values_full[idx]
        thr_full_stats = threshold_from_normal_pixels(pixel_values_full, q, k)
        if thr_stats["max"] <= 1e-6:
            self.logger.warning(
                "校准失败: 正常图分数全为 0 (布面掩码可能还没学出来), 暂用 0.5 作为阈值"
            )
        q_thr, sigma_thr, vmax = thr_stats["quantile_thr"], thr_stats["sigma_thr"], thr_stats["max"]
        image_thr_new = float(
            threshold_from_normal_pixels(np.asarray(image_scores), qi, 3.0)["threshold"]
        )
        # 8~16 张正常图估 99.9 分位本身抖动很大 (实测阈值在 0.50~0.61 之间跳),
        # 会让"检出率"在 0 / 0.6 之间来回翻 —— 那是采样噪声, 不是模型变化.
        # 首次直接采用, 之后用指数滑动平均跟上, 读数才稳定可比.
        smooth = float(eval_cfg.get("calib_smooth", 0.5))
        pixel_thr_new = float(thr_stats["threshold"])
        pixel_thr_full_new = float(thr_full_stats["threshold"])
        if self._calib_calls > 0 and 0.0 < smooth < 1.0:
            prev = self.calibration.get("pixel_threshold")
            prev_full = self.calibration.get("pixel_threshold_full")
            prev_img = self.calibration.get("image_threshold")
            if prev is not None:
                pixel_thr_new = smooth * float(prev) + (1.0 - smooth) * pixel_thr_new
            if prev_full is not None:
                pixel_thr_full_new = smooth * float(prev_full) + (1.0 - smooth) * pixel_thr_full_new
            if prev_img is not None:
                image_thr_new = smooth * float(prev_img) + (1.0 - smooth) * image_thr_new
        self._calib_calls += 1
        self.calibration = {
            "pixel_threshold": pixel_thr_new,
            "pixel_thr_quantile": q_thr,
            "pixel_thr_sigma": sigma_thr,
            "pixel_max": vmax,
            "pixel_mean": float(pixel_values.mean()),
            "pixel_std": float(pixel_values.std()),
            "pixel_threshold_full": pixel_thr_full_new,
            "pixel_thr_full_quantile": thr_full_stats["quantile_thr"],
            "pixel_thr_full_sigma": thr_full_stats["sigma_thr"],
            "pixel_full_max": thr_full_stats["max"],
            "pixel_full_mean": float(pixel_values_full.mean()),
            "pixel_full_std": float(pixel_values_full.std()),
            "memory_distance_scale": float(scale if use_memory else 0.0),
            "image_threshold": image_thr_new,
            "image_mean": float(np.mean(image_scores)),
            "image_std": float(np.std(image_scores)),
            "num_normal_images": len(image_scores),
            "quantile": q,
            "analysis_stride": analysis_stride,
        }
        self.logger.info(
            "阈值校准: pixel_thr=%.4f (q%.3f=%.4f, mean+%.0fσ=%.4f, 正常最大值=%.4f) | "
            "pixel_thr_full=%.4f | image_thr=%.4f (正常图均值 %.4f) | 记忆项尺度 %.3f | "
            "%d 张正常图 | 后处理 stride=%d",
            self.calibration["pixel_threshold"],
            q,
            q_thr,
            k,
            sigma_thr,
            vmax,
            self.calibration["pixel_threshold_full"],
            self.calibration["image_threshold"],
            float(np.mean(image_scores)),
            float(self.calibration["memory_distance_scale"]),
            len(image_scores),
            analysis_stride,
        )
        self.model.train()
        return self.calibration

    # ------------------------------------------------------------------ #
    def audit_cloth_coverage(self, batches: int = 4) -> None:
        """开工前量一次布面掩码覆盖率 = anomaly 损失真正监督到的像素比例.

        ``region = cloth_mask > 0.5`` 是异常损失的取样区域: 覆盖率 5% 就意味着
        95% 的像素**没有任何负样本约束**, 异常头在那里会自由漂移 (实测正常图的
        learned_max 会打满 1.0, 图级分数整体反向). 所以这里在训练开始时就报出来.
        """
        if self.train_loader is None:
            return
        mode = str(self.cfg.get("data", {}).get("cloth_mask", "estimate")).lower()
        covers: List[float] = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                cloth = batch.get("cloth_mask")
                if cloth is not None:
                    covers.append(float((cloth > 0.5).float().mean()))
                if i + 1 >= batches:
                    break
        if not covers:
            return
        mean_cov = float(np.mean(covers))
        bg_negative = bool(getattr(getattr(self, "criterion", None), "ano_bg_negative", True))
        supervised = 1.0 if bg_negative else mean_cov
        self.logger.info(
            "布面掩码覆盖率 %.3f (data.cloth_mask=%s) | 异常损失监督比例 %.3f "
            "(background_as_negative=%s)",
            mean_cov,
            mode,
            supervised,
            bg_negative,
        )
        if not bg_negative and mean_cov < 0.90 and mode != "full":
            self.logger.warning(
                "异常损失只监督了 %.0f%% 像素 (其余像素无约束, 异常头会漂移到 1.0). "
                "整幅都是布的产线图应设 data.cloth_mask: full, 或打开 "
                "model.loss.anomaly.background_as_negative",
                mean_cov * 100.0,
            )

    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, Any]:
        if self.start_epoch >= self.epochs:
            raise ValueError(
                f"checkpoint 已训练 {self.start_epoch} 轮，而 train.epochs={self.epochs}；请增大总轮数再续训"
            )
        if self.start_epoch == 0:
            self.audit_cloth_coverage()
        if self.start_epoch == 0 or (self.model.memory_enabled and not self.model.memory.is_ready):
            self.warmup_memory(int(self.train_cfg.get("memory_warmup_batches", 50)))
        else:
            self.logger.info("续训: 保留 checkpoint 中的记忆库与校准结果")
        refresh = int(self.train_cfg.get("memory_refresh_interval", 5))
        best_path = self.output_dir / "best.pt"
        patience = int(self.train_cfg.get("early_stop_patience", 15))
        timer = Timer()

        for epoch in range(self.start_epoch, self.epochs):
            train_logs = self.train_one_epoch(epoch)
            if refresh > 0 and (epoch + 1) % refresh == 0:
                if getattr(self, "memory_loaders", None):
                    self.model.memory.reset()
                    if self.ema is not None:
                        self.ema.ema.memory.reset()
                    # 紧随其后还会按 eval.calib_interval 校准一次, 这里不再重复校准
                    self.warmup_memory(
                        int(self.train_cfg.get("memory_warmup_batches", 24)), calibrate=False
                    )

            metrics: Dict[str, Any] = {}
            if (epoch + 1) % int(self.train_cfg.get("val_interval", 1)) == 0 or epoch == self.epochs - 1:
                model_for_eval = self.ema.ema if self.ema is not None else self.model
                # 阈值必须跟着模型走: 固定阈值在训练中会逐渐失配, 表现为验证集区域数
                # 突然变 0 (map50/acc 一起塌), 所以默认每轮用正常图重新校准一次.
                if self.cfg.get("eval", {}).get("threshold") is None and getattr(self, "memory_loaders", None):
                    interval = int(self.cfg.get("eval", {}).get("calib_interval", 1))
                    if (epoch + 1) % max(1, interval) == 0:
                        self.calibrate(max_batches=int(self.cfg.get("eval", {}).get("calib_batches", 4)))
                    self.analyzer.threshold = self.calibration.get("pixel_threshold")
                metrics = self.eval_model(model_for_eval)
                score = self._composite_score(metrics)
                metrics = {k: v for k, v in metrics.items() if k != "predictions"}
                self.logger.info(
                    "Epoch %03d | loss %.4f | pixelAUROC %.4f (同图 %.4f) PRO %.4f imgAUROC %.4f "
                    "mAP50 %.4f acc %.4f | score %.4f | %s",
                    epoch + 1,
                    train_logs.get("loss_total", 0.0),
                    metrics.get("pixel_auroc", float("nan")),
                    metrics.get("pixel_auroc_in_image", float("nan")),
                    metrics.get("pro", float("nan")),
                    metrics.get("image_auroc", float("nan")),
                    metrics.get("map50", float("nan")),
                    metrics.get("acc", float("nan")),
                    score,
                    format_seconds(timer.elapsed),
                )
                self.logger.info(
                    "  检测 | pixelAUROC %.4f (同图 %.4f) | imgAUROC %.4f (top16 %.3f / max %.3f) | 同批次AUROC %.3f | "
                    "mAP50 %.4f | 检出率 %.2f (%d 张疵点图, 平均 %.1f 区域/图, 框外误报 %.2f/图) | "
                    "正常图 %.2f 区域/图 | thr %.4f",
                    metrics.get("pixel_auroc", float("nan")),
                    metrics.get("pixel_auroc_in_image", float("nan")),
                    metrics.get("image_auroc", float("nan")),
                    metrics.get("image_auroc_top16", float("nan")),
                    metrics.get("image_auroc_max", float("nan")),
                    metrics.get("image_auroc_same_lot", float("nan")),
                    metrics.get("map50", float("nan")),
                    metrics.get("det_rate", 0.0),
                    int(metrics.get("num_defect_images", 0)),
                    metrics.get("regions_per_image", 0.0),
                    metrics.get("outside_fp_per_image", 0.0),
                    metrics.get("normal_regions_per_image", 0.0),
                    float(self.analyzer.threshold or 0.0),
                )
                if (
                    metrics.get("det_rate", 1.0) <= 0.0
                    and int(metrics.get("num_defect_images", 0)) > 0
                    and epoch + 1 > int(self.train_cfg.get("warmup_epochs", 0))
                ):
                    self.logger.warning(
                        "验证集没有检出任何区域: 阈值 %.4f = 正常图 %.1f 分位 (正常最大值 %.4f). "
                        "如果 pixelAUROC 尚可, 说明正/负分数分不开, 优先查 %s; "
                        "否则调 eval.calib_sigma / eval.normal_quantile",
                        float(self.analyzer.threshold or 0.0),
                        float(self.calibration.get("quantile", 0.999)) * 100.0,
                        float(self.calibration.get("pixel_max", 0.0)),
                        "记忆库是否脱节" if self._memory_stale else "异常头是否学到东西",
                    )
                if score > self.best_score:
                    self.best_score = score
                    self.bad_epochs = 0
                    self.save_checkpoint(best_path, epoch, metrics, best=True)
                else:
                    self.bad_epochs += 1
            else:
                self.logger.info(
                    "Epoch %03d | loss %.4f | %s",
                    epoch + 1,
                    train_logs.get("loss_total", 0.0),
                    format_seconds(timer.elapsed),
                )

            self._log_cls_progress(epoch, train_logs)
            record = {"epoch": epoch + 1, **train_logs, **metrics, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(record)
            self._write_history(record)
            # 中断时仍能从最近完整轮次恢复，而不是只剩训练结束时的 last.pt。
            self.save_checkpoint(self.output_dir / "last.pt", epoch, metrics, best=False)

            if patience > 0 and self.bad_epochs >= patience:
                self.logger.info("早停: 连续 %d 轮无提升", self.bad_epochs)
                break

        sweep = self.train_cfg.get("scale_sweep") or []
        if sweep:
            self.logger.info("分辨率体检: 用当前权重扫 %s (不落盘, 不影响已保存的 checkpoint)", list(sweep))
            try:
                self.scale_sweep(
                    [float(x) for x in sweep],
                    memory_batches=int(self.train_cfg.get("scale_sweep_memory_batches", 12)),
                    max_samples=int(self.train_cfg.get("scale_sweep_samples", 60)),
                )
            except Exception as exc:  # pragma: no cover - 体检失败不该影响训练结果
                self.logger.warning("分辨率体检失败: %s", exc)
        summary = {
            "best_score": self.best_score,
            "history": self.history,
            "output_dir": str(self.output_dir),
        }
        json_dump(summary, self.output_dir / "history.json")
        save_yaml(self.cfg, self.output_dir / "config.yaml")
        if best_path.exists():
            self.load_checkpoint(best_path)
            final = self.eval_model(self.model)
            final.pop("predictions", None)
            json_dump(final, self.output_dir / "final_metrics.json")
            summary["final_metrics"] = final
        return summary

    # ------------------------------------------------------------------ #
    def eval_model(self, model: "FabricNCD") -> Dict[str, Any]:
        """验证入口: ``eval.tile=true`` 时走切块评估, 与部署时的推理路径一致."""
        if bool(self.cfg.get("eval", {}).get("tile", False)):
            return evaluate_tiled(
                model, self.cfg, self.device, split="val",
                analyzer=self.analyzer, calibration=self.calibration,
            )
        return evaluate(
            model, self.val_loader, self.cfg, self.device, self.criterion, self.analyzer,
            calibration=self.calibration,
        )

    # ------------------------------------------------------------------ #
    def _log_cls_progress(self, epoch: int, train_logs: Dict[str, float]) -> None:
        """把分类头单独说清楚: 交叉熵 vs 均匀基线 ln(C), 以及框内像素准确率.

        只有被框覆盖的样本才有 cls 梯度, 所以准确率只在有标注的 batch 上统计.
        """
        if "cls" not in train_logs:
            return
        proto = getattr(getattr(self.model, "classifier", None), "prototypes", None)
        if proto is None:
            return
        num_classes = int(proto.shape[0])
        baseline = math.log(max(2, num_classes))
        acc = train_logs.get("cls_acc")
        acc_text = "%.3f" % acc if acc is not None else "-"
        tail = ""
        if epoch + 1 <= int(self.train_cfg.get("warmup_epochs", 0)):
            tail = " | 仍在 warmup, lr 未到峰值"
        self.logger.info(
            "  分类头 | cls %.3f (均匀基线 ln%d=%.3f, %s) | 框内准确率 %s%s",
            float(train_logs["cls"]),
            num_classes,
            baseline,
            "已低于基线" if float(train_logs["cls"]) < baseline else "仍高于基线",
            acc_text,
            tail,
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss_total", "anomaly", "cls", "cls_acc", "supcon", "cloth", "edge")}
        timer = Timer()
        total = max(1, len(self.train_loader))

        for step, batch in enumerate(self.train_loader):
            batch = move_to(batch, self.device)
            images = batch["image"]
            try:
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    outputs = self.model(images)
                    targets = {
                        "defect_mask": batch["anomaly_mask"],
                        "cloth_mask": batch["cloth_mask"],
                        "boxes": batch["boxes"],
                    }
                    loss, logs = self.criterion(outputs, targets, memory=self.model.memory)

                self.optimizer.zero_grad(set_to_none=True)
                scale_before = self.scaler.get_scale() if (self.scaler is not None and self.amp) else None
                if self.scaler is not None and self.amp:
                    self.scaler.scale(loss).backward()
                    if float(self.train_cfg.get("grad_clip", 5.0)) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_cfg["grad_clip"]))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if float(self.train_cfg.get("grad_clip", 5.0)) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.train_cfg["grad_clip"]))
                    self.optimizer.step()
            except torch.cuda.OutOfMemoryError:
                raise self._oom_error() from None
            # AMP: scaler 遇到 inf/nan 会跳过这一步的 optimizer.step(), 此时不能推进 lr
            # (否则会报 "Detected call of lr_scheduler.step() before optimizer.step()")
            if scale_before is None or self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)

            for key in meters:
                if key in logs:
                    meters[key].update(logs[key])

            if (step + 1) % int(self.train_cfg.get("log_interval", 10)) == 0 or step == total - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                acc_meter = meters["cls_acc"]
                self.logger.info(
                    "  iter %3d/%3d | loss %.4f (ano %.4f cls %.4f acc %s supcon %.4f) | lr %.2e | %.1fs",
                    step + 1,
                    total,
                    meters["loss_total"].avg,
                    meters["anomaly"].avg,
                    meters["cls"].avg,
                    "%.3f" % acc_meter.avg if acc_meter.count else "  -  ",
                    meters["supcon"].avg,
                    lr,
                    timer.elapsed,
                )
        return {k: v.avg for k, v in meters.items() if v.count > 0}

    # 实测系数: 1024x1728 (1.77 Mpx) 训练含反向约 1.40 GB -> 0.79 GB/百万像素
    MEM_GB_PER_MPX = 0.79

    def _warn_memory(self, data_cfg: Dict[str, Any]) -> None:
        """按输入尺寸 × batch 粗估训练显存, 超卡就提前说, 别等 OOM."""
        try:
            size = data_cfg.get("image_size")
            if isinstance(size, str) or not size:
                return
            h, w = int(size[0]), int(size[1])
            batch = int(data_cfg.get("batch_size", 1))
        except Exception:  # pragma: no cover
            return
        est = self.MEM_GB_PER_MPX * (h * w) / 1e6 * max(batch, 1)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.logger.info("预计训练显存 (估算): %.2f GB (非 CUDA 设备, 仅供参考)", est)
            return
        total = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        self.logger.info("预计训练显存 (估算): %.2f GB / 显卡 %.1f GB", est, total)
        if est > 0.85 * total:
            self.logger.warning(
                "按 %.0fx%d x batch %d 估算要 %.1f GB, 超过 %.1f GB 显存的 85%%, 很可能 OOM. "
                "把 batch_size 减半, 或 --set data.short_side=%d",
                h, w, batch, est, total,
                max(256, int(min(h, w) * 0.75) // 32 * 32),
            )

    def _oom_error(self) -> RuntimeError:
        """把 CUDA OOM 变成可以照着做的提示 (6GB 卡上最常见)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        data = self.cfg.get("data", {})
        return RuntimeError(
            "显存不足 (CUDA OOM). 当前 data.image_size=%s, data.batch_size=%s.\n"
            "按这个顺序降 (image_size=auto 时改短边即可, 长宽比自动保持):\n"
            "  --set data.short_side=768  ->  --set data.short_side=640  ->  --set data.short_side=512\n"
            "  或 --set data.batch_size=2 (别低于 2)\n"
            "不要用加大 tile 的方式省显存 (tile=true 反而更慢), 也不要靠打乱长宽比硬凑."
            % (data.get("image_size"), data.get("batch_size"))
        )

    # ------------------------------------------------------------------ #
    def _composite_score(self, metrics: Dict[str, float]) -> float:
        weights = self.train_cfg.get(
            "score_weights",
            {"pixel_auroc": 0.35, "pro": 0.15, "image_auroc": 0.2, "map50": 0.15, "f1_macro": 0.15},
        )
        total, weight_sum = 0.0, 0.0
        for key, weight in weights.items():
            value = metrics.get(key)
            if value is None or not np.isfinite(value):
                continue
            total += float(weight) * float(value)
            weight_sum += float(weight)
        return total / weight_sum if weight_sum > 0 else 0.0

    # ------------------------------------------------------------------ #
    def save_checkpoint(
        self, path: Path, epoch: int, metrics: Dict[str, Any], best: bool = False
    ) -> None:
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_score": self.best_score,
            "global_step": self.global_step,
            "bad_epochs": self.bad_epochs,
            "bank_paths": sorted(self._bank_paths),
            "calib_calls": self._calib_calls,
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "ema_updates": self.ema.num_updates if self.ema is not None else 0,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "calibration": self.calibration,
            "config": self.cfg,
            "names": _class_names(self.cfg),
            "version": 2,
        }
        if self.ema is not None:
            payload["ema"] = self.ema.ema.state_dict()
        # 原子替换：中途异常不会把上一个完整 checkpoint 覆盖成半截文件。
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        if best:
            self.logger.info("保存最优模型 -> %s (score=%.4f)", path, self.best_score)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False) -> Dict[str, Any]:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") == 1 and hasattr(self, "cfg"):
            cfg = checkpoint_compat_config(self.cfg, state)
            legacy = cfg.get("model", {}).get("backbone", {}).get("legacy_p5_stride64")
            backbone = getattr(self.model, "backbone", None)
            if legacy is not None and hasattr(backbone, "set_legacy_p5_stride64"):
                backbone.set_legacy_p5_stride64(bool(legacy))
                if self.ema is not None:
                    self.ema.ema.backbone.set_legacy_p5_stride64(bool(legacy))
            self.cfg = cfg
        key = "ema" if "ema" in state and not load_optimizer else "model"
        load_model_weights(self.model, state[key])
        if self.ema is not None:
            load_model_weights(self.ema.ema, state.get("ema") or state["model"])
            self.ema.num_updates = int(state.get("ema_updates", state.get("global_step", state.get("scheduler", {}).get("last_epoch", 0))))
        if load_optimizer and "optimizer" not in state:
            raise ValueError("checkpoint 缺少优化器状态，不能断点续训；请作为预训练权重重新开始")
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            if self.scaler is not None and state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])
            self.global_step = int(state.get("global_step", state.get("scheduler", {}).get("last_epoch", 0)))
            self.start_epoch = int(state.get("epoch", 0)) + 1
            self.bad_epochs = int(state.get("bad_epochs", 0))
            self._bank_paths = set(state.get("bank_paths", []))
            self._calib_calls = int(state.get("calib_calls", 0))
            rng = state.get("rng_state") or {}
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([item.cpu() for item in rng["cuda"]])
            saved_data = (state.get("config") or {}).get("data", {})
            current_data = getattr(self, "cfg", {}).get("data", {})
            if saved_data and any(saved_data.get(key) != current_data.get(key) for key in ("root", "dataset_yaml")):
                if getattr(self.model, "memory_enabled", False):
                    self.model.memory.reset()
                    if self.ema is not None:
                        self.ema.ema.memory.reset()
                    self._bank_paths.clear()
                    self.logger.warning("数据集路径已更改，续训时将重新建立记忆库并校准")
            if (getattr(self.model, "memory_enabled", False)
                    and self.model.memory.is_ready and not self._bank_paths):
                self.logger.warning("旧 checkpoint 未记录建库图像，后续校准可能复用入库图；建议刷新记忆库")
        self.best_score = float(state.get("best_score", -1))
        if state.get("calibration"):
            self.calibration = dict(state["calibration"])
            scale = float(self.calibration.get("memory_distance_scale", 0.0) or 0.0)
            if scale > 0:
                self._set_memory_scale(scale)
        self.logger.info("载入权重 %s (%s)", path, key)
        return state

    def _write_history(self, record: Dict[str, Any]) -> None:
        path = self.output_dir / "history.csv"
        write_header = not path.exists()
        keys = [
            "epoch", "loss_total", "anomaly", "cls", "cls_acc", "supcon", "pixel_auroc", "pro",
            "image_auroc", "map50", "acc", "f1_macro", "lr",
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in keys})
