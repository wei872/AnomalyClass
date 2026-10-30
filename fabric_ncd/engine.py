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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.yolo import build_dataloaders, build_normal_loader, load_dataset_spec
from .losses import FabricCriterion
from .metrics import MetricAccumulator
from .models.fabric_ncd import FabricNCD
from .ncd import RegionAnalyzer, postprocess_outputs
from .ops import topk_mean
from .utils import (
    AverageMeter,
    Timer,
    describe_device,
    ensure_dir,
    format_seconds,
    get_logger,
    json_dump,
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
        # BN 统计量与记忆库直接同步, 不做平滑
        for (ema_buf_name, ema_buf), (buf_name, buf) in zip(
            self.ema.named_buffers(), model.named_buffers()
        ):
            if "memory" in ema_buf_name:
                continue
            if ema_buf.shape == buf.shape and ema_buf.dtype == buf.dtype:
                ema_buf.copy_(buf.detach())
        if getattr(model, "memory", None) is not None:
            self.ema.memory.features.copy_(model.memory.features)
            self.ema.memory.size.copy_(model.memory.size)


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
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
    analyzer = analyzer or RegionAnalyzer(
        names=names,
        threshold=cfg.get("eval", {}).get("threshold"),
        sigma=float(cfg.get("eval", {}).get("threshold_sigma", 3.0)),
        min_area_ratio=float(cfg.get("eval", {}).get("min_area_ratio", 4e-4)),
        merge_strategy=str(cfg.get("eval", {}).get("merge_strategy", "area_weighted")),
    )
    # 未显式指定阈值时优先使用正常集校准出的绝对阈值
    if analyzer.threshold is None and calibration:
        analyzer.threshold = float(calibration.get("pixel_threshold", 0.0)) or None
    acc = MetricAccumulator(num_classes=len(names), names=names)
    if analyzer.threshold is not None:
        acc.pixel_threshold = float(analyzer.threshold)
    loss_meter = AverageMeter()
    predictions: List[Dict[str, Any]] = []
    seen = 0

    for batch in loader:
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
        score = score * (cloth > 0.5).float()
        score_np = score.squeeze(1).detach().cpu().numpy()

        region_results = postprocess_outputs(outputs, analyzer, input_size=(h, w))
        for i in range(score.shape[0]):
            normal = bool(batch["normal"][i].item())
            image_score = topk_mean(score_np[i], float(cfg.get("eval", {}).get("topk_ratio", 0.01)))
            infos, image_level = region_results[i]
            gt_mask = batch["anomaly_mask"][i, 0].detach().cpu().numpy()

            gt_labels = batch["labels"][i].detach().cpu().numpy().astype(int)
            gt_boxes_np = batch["boxes"][i].detach().cpu().numpy()
            gt_dict = {
                "boxes": _xyxy_pixels(gt_boxes_np, w, h),
                "labels": gt_labels,
            }
            pred_boxes, pred_scores, pred_labels = _regions_to_boxes(infos, w, h, cfg)

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
    """大幅面切块评估 —— 与 ``FabricPredictor(tile=True)`` 完全一致的指标.

    整图缩放会丢细节时 (小疵点只有几像素), 评估必须和推理走同一条路径:
    按原始像素尺度切块 -> 融合异常图 -> 区域分析 -> 计算指标.
    仅有整图缩放的 ``evaluate`` 会低估小疵点的检出能力.

    像素级指标按 ``eval.max_pixels`` 下采样后再统计 (AUROC/PRO 对下采样不敏感),
    检测框/分类指标仍使用原始坐标, 因此与部署时的输出一致.
    """
    from .inference import _resolve_image_size, as_hw, forward_maps
    from .data.yolo import boxes_to_mask, list_images, load_rgb, read_yolo_labels

    model.eval()
    spec = load_dataset_spec(cfg)
    names = list(spec.names) if spec.names else _class_names(cfg)
    eval_cfg = cfg.get("eval", {})
    analyzer = analyzer or RegionAnalyzer(
        names=names,
        threshold=eval_cfg.get("threshold"),
        sigma=float(eval_cfg.get("threshold_sigma", 3.0)),
        min_area_ratio=float(eval_cfg.get("min_area_ratio", 4e-4)),
        merge_strategy=str(eval_cfg.get("merge_strategy", "area_weighted")),
    )
    if analyzer.threshold is None and calibration:
        analyzer.threshold = float(calibration.get("pixel_threshold", 0.0)) or None

    acc = MetricAccumulator(num_classes=len(names), names=names)
    if analyzer.threshold is not None:
        acc.pixel_threshold = float(analyzer.threshold)

    image_size = _resolve_image_size(cfg)
    tile_hw = as_hw(tile_size or eval_cfg.get("tile_size") or image_size)
    overlap = int(tile_overlap if tile_overlap is not None else eval_cfg.get("tile_overlap", 128))
    tile_batch = int(eval_cfg.get("tile_batch", 4))
    max_pixels = float(eval_cfg.get("max_pixels", 3e5))

    defect_images = list_images(spec.splits.get(split))
    labels_dir = spec.labels.get(split)
    normal_images = list_images(spec.normal_splits.get(split))
    items: List[Tuple[Path, bool]] = [(p, False) for p in defect_images] + [(p, True) for p in normal_images]
    if max_samples:
        items = items[:max_samples]

    predictions: List[Dict[str, Any]] = []
    for path, is_normal in items:
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        if is_normal:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            gt_labels = np.zeros((0,), dtype=np.int64)
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            label_path = None if labels_dir is None else Path(labels_dir) / f"{path.stem}.txt"
            boxes, polygons = read_yolo_labels(label_path)
            gt_mask = boxes_to_mask(boxes, polygons, h, w)
            gt_labels = boxes[:, 0].astype(np.int64) if len(boxes) else np.zeros((0,), dtype=np.int64)
            gt_boxes = _xyxy_pixels(boxes, w, h)

        maps = forward_maps(
            model, rgb, device, image_size,
            tile=True, tile_size=tile_hw, tile_overlap=overlap, tile_batch=tile_batch,
        )
        score = maps["score"] * (maps["cloth"] > 0.5)
        infos, image_level = analyzer.analyze(score, maps["cloth"], maps["probs"])
        image_score = topk_mean(score, float(eval_cfg.get("topk_ratio", 0.01)))

        # 像素级统计下采样, 大幅面图 (2000x3000) 全分辨率累计会吃掉数 GB 内存
        factor = max(1, int(math.ceil(math.sqrt((h * w) / max(max_pixels, 1.0)))))
        pred_boxes, pred_scores, pred_labels = _regions_to_boxes(infos, w, h, cfg)

        multi_true, multi_pred = -1, -1
        if len(gt_labels):
            multi_true = int(np.bincount(gt_labels).argmax())
            if image_level.get("probs") is not None:
                multi_pred = int(np.argmax(image_level["probs"]))

        acc.update(
            anomaly_score=score[::factor, ::factor],
            anomaly_gt=gt_mask[::factor, ::factor],
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
        return list(spec.names)
    except Exception:  # pragma: no cover
        num = int(cfg.get("model", {}).get("heads", {}).get("num_classes", 19))
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
        self.scaler = torch.amp.GradScaler(enabled=self.amp) if hasattr(torch, "amp") else None

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
        self.normal_loader = build_normal_loader(cfg, "train")

        self.model = FabricNCD(cfg).to(self.device)
        self.criterion = FabricCriterion(cfg).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.ema = ModelEMA(self.model, float(self.train_cfg.get("ema", 0.999))) if self.train_cfg.get("ema", 0.999) > 0 else None
        self.analyzer = RegionAnalyzer(
            names=_class_names(cfg),
            threshold=cfg.get("eval", {}).get("threshold"),
            sigma=float(cfg.get("eval", {}).get("threshold_sigma", 3.0)),
            min_area_ratio=float(cfg.get("eval", {}).get("min_area_ratio", 4e-4)),
            merge_strategy=str(cfg.get("eval", {}).get("merge_strategy", "area_weighted")),
        )

        self.start_epoch = 0
        self.calibration: Dict[str, float] = {}
        self.best_score = -float("inf")
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
    def warmup_memory(self, max_batches: Optional[int] = None) -> None:
        if self.normal_loader is None or not self.model.memory_enabled:
            self.logger.warning("没有可用的正常图, 跳过记忆库预热 (将仅使用学习式异常头)")
            return
        self.model.eval()
        batches = 0
        with torch.no_grad():
            for batch in self.normal_loader:
                batch = move_to(batch, self.device)
                self.model.update_memory(batch["image"], mask=batch["cloth_mask"])
                batches += 1
                if max_batches and batches >= max_batches:
                    break
        self.model.train()
        self.logger.info("记忆库预热完成: %d 个正常 patch 特征", int(self.model.memory.size.item()))
        # 先把新记忆库同步给 EMA, 再用 EMA 做阈值校准, 否则校准用的是旧记忆库,
        # 得到的阈值会系统性偏高.
        if self.ema is not None:
            self.ema.update(self.model)
        self.calibrate()

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
        if self.normal_loader is None:
            return self.calibration
        eval_cfg = self.cfg.get("eval", {})
        q = float(quantile if quantile is not None else eval_cfg.get("normal_quantile", 0.999))
        qi = float(image_quantile if image_quantile is not None else eval_cfg.get("image_quantile", 0.995))
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        pixels: List[np.ndarray] = []
        image_scores: List[float] = []
        seen = 0
        for batch in self.normal_loader:
            batch = move_to(batch, self.device)
            out = model(batch["image"])
            h, w = batch["image"].shape[-2:]
            score = F.interpolate(out["anomaly_score"], size=(h, w), mode="bilinear", align_corners=False)
            cloth = F.interpolate(out["cloth_prob"], size=(h, w), mode="bilinear", align_corners=False)
            score = (score * (cloth > 0.5).float()).squeeze(1).cpu().numpy()
            for i in range(score.shape[0]):
                image_scores.append(topk_mean(score[i], float(eval_cfg.get("topk_ratio", 0.01))))
                flat = score[i].ravel()
                if flat.size > 20000:
                    flat = flat[:: max(1, flat.size // 20000)]
                pixels.append(flat)
                seen += 1
            if seen >= max_batches * batch["image"].shape[0]:
                break
        if not pixels:
            return self.calibration
        pixel_values = np.concatenate(pixels)
        if pixel_values.size > max_pixels:
            idx = np.random.default_rng(0).choice(pixel_values.size, max_pixels, replace=False)
            pixel_values = pixel_values[idx]
        self.calibration = {
            "pixel_threshold": float(
                max(
                    np.quantile(pixel_values, q),
                    pixel_values.mean() + float(eval_cfg.get("calib_sigma", 6.0)) * pixel_values.std(),
                )
            ),
            "pixel_mean": float(pixel_values.mean()),
            "pixel_std": float(pixel_values.std()),
            "image_threshold": float(
                max(np.quantile(image_scores, qi), np.mean(image_scores) + 3.0 * np.std(image_scores))
            ),
            "image_mean": float(np.mean(image_scores)),
            "image_std": float(np.std(image_scores)),
            "num_normal_images": len(image_scores),
            "quantile": q,
        }
        self.logger.info(
            "阈值校准完成: pixel_thr=%.4f (q=%.4f) | image_thr=%.4f | %d 张正常图",
            self.calibration["pixel_threshold"],
            q,
            self.calibration["image_threshold"],
            len(image_scores),
        )
        self.model.train()
        return self.calibration

    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, Any]:
        self.warmup_memory(int(self.train_cfg.get("memory_warmup_batches", 50)))
        refresh = int(self.train_cfg.get("memory_refresh_interval", 5))
        best_path = self.output_dir / "best.pt"
        patience = int(self.train_cfg.get("early_stop_patience", 15))
        bad_epochs = 0
        timer = Timer()

        for epoch in range(self.start_epoch, self.epochs):
            train_logs = self.train_one_epoch(epoch)
            if refresh > 0 and (epoch + 1) % refresh == 0:
                if self.normal_loader is not None:
                    self.model.memory.reset()
                    if self.ema is not None:
                        self.ema.ema.memory.reset()
                    self.warmup_memory(int(self.train_cfg.get("memory_warmup_batches", 50)))

            metrics: Dict[str, Any] = {}
            if (epoch + 1) % int(self.train_cfg.get("val_interval", 1)) == 0 or epoch == self.epochs - 1:
                model_for_eval = self.ema.ema if self.ema is not None else self.model
                if self.cfg.get("eval", {}).get("threshold") is None and self.calibration:
                    self.analyzer.threshold = self.calibration.get("pixel_threshold")
                metrics = self.eval_model(model_for_eval)
                score = self._composite_score(metrics)
                metrics = {k: v for k, v in metrics.items() if k != "predictions"}
                self.logger.info(
                    "Epoch %03d | loss %.4f | pixelAUROC %.4f PRO %.4f imgAUROC %.4f mAP50 %.4f acc %.4f | score %.4f | %s",
                    epoch + 1,
                    train_logs.get("loss_total", 0.0),
                    metrics.get("pixel_auroc", float("nan")),
                    metrics.get("pro", float("nan")),
                    metrics.get("image_auroc", float("nan")),
                    metrics.get("map50", float("nan")),
                    metrics.get("acc", float("nan")),
                    score,
                    format_seconds(timer.elapsed),
                )
                if score > self.best_score:
                    self.best_score = score
                    bad_epochs = 0
                    self.save_checkpoint(best_path, epoch, metrics, best=True)
                else:
                    bad_epochs += 1
            else:
                self.logger.info(
                    "Epoch %03d | loss %.4f | %s",
                    epoch + 1,
                    train_logs.get("loss_total", 0.0),
                    format_seconds(timer.elapsed),
                )

            record = {"epoch": epoch + 1, **train_logs, **metrics, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(record)
            self._write_history(record)

            if bad_epochs >= patience:
                self.logger.info("早停: 连续 %d 轮无提升", bad_epochs)
                break

        self.save_checkpoint(self.output_dir / "last.pt", epoch, {}, best=False)
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
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss_total", "anomaly", "cls", "supcon", "cloth", "edge")}
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
            self.scheduler.step()
            self.global_step += 1
            if self.ema is not None:
                self.ema.update(self.model)

            for key in meters:
                if key in logs:
                    meters[key].update(logs[key])
            meters["loss_total"].update(logs.get("loss_total", float(loss)))

            if (step + 1) % int(self.train_cfg.get("log_interval", 10)) == 0 or step == total - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "  iter %3d/%3d | loss %.4f (ano %.4f cls %.4f supcon %.4f) | lr %.2e | %.1fs",
                    step + 1,
                    total,
                    meters["loss_total"].avg,
                    meters["anomaly"].avg,
                    meters["cls"].avg,
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
            self.logger.info("预计训练显存 (估算): %.1f GB (非 CUDA 设备, 仅供参考)", est)
            return
        total = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        self.logger.info("预计训练显存 (估算): %.1f GB / 显卡 %.1f GB", est, total)
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
            "calibration": self.calibration,
            "config": self.cfg,
            "names": _class_names(self.cfg),
            "version": 1,
        }
        if self.ema is not None:
            payload["ema"] = self.ema.ema.state_dict()
        torch.save(payload, path)
        if best:
            self.logger.info("保存最优模型 -> %s (score=%.4f)", path, self.best_score)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False) -> Dict[str, Any]:
        state = torch.load(path, map_location=self.device, weights_only=False)
        key = "ema" if "ema" in state and not load_optimizer else "model"
        try:
            self.model.load_state_dict(state[key])
        except RuntimeError as exc:
            # 结构升级后旧权重形状可能不一致 (例如 geography 新增 border_prior 输入),
            # 尽量加载其余权重并明确告警, 而不是直接崩掉
            missing, unexpected = self.model.load_state_dict(state[key], strict=False)
            self.logger.warning(
                "权重与当前结构不一致 (%s): 跳过 %d 个参数, 多余 %d 个; 建议重新训练",
                exc.__class__.__name__, len(missing), len(unexpected),
            )
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            self.start_epoch = int(state.get("epoch", 0)) + 1
        self.best_score = float(state.get("best_score", -1))
        if state.get("calibration"):
            self.calibration = dict(state["calibration"])
        self.logger.info("载入权重 %s (%s)", path, key)
        return state

    def _write_history(self, record: Dict[str, Any]) -> None:
        path = self.output_dir / "history.csv"
        write_header = not path.exists()
        keys = [
            "epoch", "loss_total", "anomaly", "cls", "supcon", "pixel_auroc", "pro",
            "image_auroc", "map50", "acc", "f1_macro", "lr",
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow({k: record.get(k, "") for k in keys})
