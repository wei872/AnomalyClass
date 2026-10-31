"""Configuration loading, portable YOLO paths and strict label-space validation."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

DEFAULT_SPLIT_PATHS = {
    "train": "images/train",
    "val": "images/val",
    "normaltrain": "normal/images/train",
    "normalval": "normal/images/val",
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML 映射: {path}")
    return value


def deep_merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    value = read_yaml(path)
    parent = value.pop("extends", None)
    if parent:
        result = load_config(path.parent / parent)
        result = deep_merge(result, value)
    else:
        result = value
    result = deep_merge(result, overrides or {})
    dataset_path = Path(result["data"]["dataset_config"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = path.parent / dataset_path
    result["data"]["dataset_config"] = str(dataset_path.resolve())
    validate_config(result)
    return result


def _number(value: Any, name: str, low: float, high: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} 超出范围 [{low}, {high}]")


def validate_config(cfg: dict) -> None:
    if cfg["mode"] not in {"supervised", "discovery"}:
        raise ValueError("mode 必须是 supervised 或 discovery")
    model, train = cfg["model"], cfg["training"]
    if model["backbone"] not in {"dino_vitb8", "dino_vitb16", "dino_vits8", "dino_vits16", "tiny_vit"}:
        raise ValueError(f"不支持的 backbone: {model['backbone']}")
    if model["n_views"] != 2 or not 1 <= model["mask_layers"] <= 12:
        raise ValueError("当前损失要求 n_views=2，mask_layers 必须在 1..12")
    for key in ("image_size", "n_heads", "hidden_dim", "bottleneck_dim"):
        if not isinstance(model[key], int) or model[key] <= 0:
            raise ValueError(f"model.{key} 必须是正整数")
    for key in ("batch_size", "accumulation_steps", "epochs"):
        if not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"training.{key} 必须是正整数")
    if not isinstance(train["num_workers"], int) or train["num_workers"] < 0:
        raise ValueError("num_workers 必须是非负整数")
    if not isinstance(train["eval_every"], int) or not isinstance(cfg["experiment"]["print_freq"], int):
        raise ValueError("eval_every/print_freq 必须是整数")
    if not isinstance(train["class_balanced_sampler"], bool) or not isinstance(train["amp"], bool):
        raise ValueError("amp/class_balanced_sampler 必须是布尔值")
    if not isinstance(model["grad_from_block"], int) or not 0 <= model["grad_from_block"] <= 12:
        raise ValueError("grad_from_block 必须是 0..12 的整数")
    if train["eval_every"] < 0 or cfg["experiment"]["print_freq"] <= 0:
        raise ValueError("eval_every 必须 >=0，print_freq 必须 >0")
    _number(train["lr"], "lr", 1e-12)
    _number(train["min_lr"], "min_lr", 0, train["lr"])
    _number(train["weight_decay"], "weight_decay", 0)
    _number(train["max_grad_norm"], "max_grad_norm", 0)
    _number(cfg["data"]["padding"], "padding", 0, 10)
    _number(cfg["data"]["min_crop_size"], "min_crop_size", 0, 1)
    _number(cfg["inference"]["area_temperature"], "area_temperature", 1e-6)
    threshold = cfg["inference"]["image_threshold"]
    if threshold is not None:
        _number(threshold, "image_threshold", 0, 1)
    for key in ("sup_weight", "supcon_weight", "memax_weight"):
        _number(cfg["loss"][key], key, 0)
    _number(cfg["loss"]["sup_weight"], "sup_weight", 0, 1)
    for key in ("temperature", "contrastive_temperature", "supcon_temperature", "teacher_temp", "warmup_teacher_temp"):
        _number(cfg["loss"][key], key, 1e-6)
    warmup = cfg["loss"]["warmup_teacher_temp_epochs"]
    if not isinstance(warmup, int) or not 0 <= warmup <= train["epochs"]:
        raise ValueError("warmup_teacher_temp_epochs 必须是 0..epochs 的整数")
    if cfg["experiment"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必须是 auto/cpu/cuda")
    if not isinstance(cfg["experiment"]["seed"], int) or not 0 <= cfg["experiment"]["seed"] < 2**32:
        raise ValueError("seed 必须是 0..2^32-1 的整数")
    detector = cfg.get("detector")
    if detector is not None:
        if not isinstance(detector, dict):
            raise ValueError("detector 必须是映射")
        for key in ("image_size", "epochs", "batch_size"):
            if not isinstance(detector.get(key), int) or detector[key] <= 0:
                raise ValueError(f"detector.{key} 必须是正整数")
        for key in ("conf_threshold", "iou_threshold", "anomaly_threshold"):
            _number(detector.get(key), f"detector.{key}", 0, 1)
        if not isinstance(detector.get("class_agnostic", True), bool):
            raise ValueError("detector.class_agnostic 必须是布尔值")
        if cfg["mode"] == "discovery" and not detector.get("class_agnostic", True):
            raise ValueError("NCD 模式要求 detector.class_agnostic=true，避免使用新类 ID")
    anomaly = cfg.get("anomaly")
    if anomaly is not None:
        if not isinstance(anomaly, dict):
            raise ValueError("anomaly 必须是映射")
        if not isinstance(anomaly.get("enabled", False), bool):
            raise ValueError("anomaly.enabled 必须是布尔值")
        for key in ("image_size", "bank_size", "max_candidates", "knn", "min_area",
                    "max_regions", "threshold_steps", "neighborhood"):
            if key in anomaly and (not isinstance(anomaly[key], int) or anomaly[key] <= 0):
                raise ValueError(f"anomaly.{key} 必须是正整数")
        if "image_size" in anomaly:
            patch = 16 if str(cfg["model"]["backbone"]).endswith("16") else 8
            if anomaly["image_size"] % patch:
                raise ValueError(f"anomaly.image_size 必须能被 patch_size={patch} 整除")
        if "neighborhood" in anomaly and anomaly["neighborhood"] % 2 == 0:
            raise ValueError("anomaly.neighborhood 必须是奇数")
        if "coreset" in anomaly and not isinstance(anomaly["coreset"], bool):
            raise ValueError("anomaly.coreset 必须是布尔值")
        for key in ("smooth_sigma",):
            if key in anomaly:
                _number(anomaly[key], f"anomaly.{key}", 0, 100)
        if "topk_ratio" in anomaly:
            value = anomaly["topk_ratio"]
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError("anomaly.topk_ratio 必须在 (0, 1]")
        layers = anomaly.get("layers")
        if layers is not None:
            if not isinstance(layers, (list, tuple)) or not layers:
                raise ValueError("anomaly.layers 必须是非空列表")
            depth = 4 if cfg["model"]["backbone"] == "tiny_vit" else 12
            if any(not isinstance(x, int) or not 0 <= x < depth for x in layers):
                raise ValueError(f"anomaly.layers 必须在 0..{depth-1}")
        if "max_area_ratio" in anomaly:
            value = anomaly["max_area_ratio"]
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError("anomaly.max_area_ratio 必须在 (0, 1]")
        if "fuse_iou" in anomaly:
            _number(anomaly["fuse_iou"], "anomaly.fuse_iou", 0, 1)
        if "score_threshold" in anomaly:
            _number(anomaly["score_threshold"], "anomaly.score_threshold", 0, 10)


def load_class_names(path: str | Path) -> list[str]:
    """Read only the label space; training on prepared crops needs no raw-data drive."""
    raw = read_yaml(path)
    names_raw, nc = raw.get("names"), raw.get("nc")
    if isinstance(names_raw, dict):
        if set(names_raw) != set(range(len(names_raw))):
            raise ValueError("names 字典的 ID 必须连续且从 0 开始")
        names_raw = [names_raw[i] for i in range(len(names_raw))]
    if not isinstance(names_raw, list) or not names_raw or not all(isinstance(x, str) and x.strip() for x in names_raw):
        raise ValueError("names 必须是非空类别名列表或连续 ID 字典")
    if not isinstance(nc, int) or nc != len(names_raw):
        raise ValueError(f"nc={nc} 与 names 数量 {len(names_raw)} 不一致")
    if len(set(names_raw)) != nc:
        raise ValueError("names 中存在重复类别名")
    return names_raw


@dataclass(frozen=True)
class DatasetSpec:
    config_path: Path
    root: Path | None
    paths: dict[str, Path]
    names: list[str]

    @property
    def nc(self) -> int:
        return len(self.names)


def _path_parts(value: str) -> list[str]:
    """Accept Windows YAML paths even when running on Linux."""
    return list(PureWindowsPath(value.replace("\\", "/")).parts)


def load_dataset_spec(path: str | Path, data_root: str | Path | None = None) -> DatasetSpec:
    path = Path(path).expanduser().resolve()
    raw = read_yaml(path)
    names_raw = load_class_names(path)

    old_root_value = raw.get("path")
    override = data_root or os.environ.get("ANOMALYNCD_DATA_ROOT")
    root = Path(override).expanduser().resolve() if override else None
    if root is None and old_root_value:
        old_root_text = str(old_root_value)
        if re.match(r"^[A-Za-z]:[\\/]", old_root_text) and os.name != "nt":
            raise ValueError(
                f"数据根目录是 Windows 路径: {old_root_text}。请传入 --data-root /实际数据根目录，"
                "或设置 ANOMALYNCD_DATA_ROOT；不会尝试访问不存在的 D: 盘。"
            )
        root = Path(old_root_text).expanduser()
        if not root.is_absolute():
            root = path.parent / root
        root = root.resolve()
    paths = {}
    old_parts = _path_parts(str(old_root_value)) if old_root_value else []
    for split, default in DEFAULT_SPLIT_PATHS.items():
        value = raw.get(split)
        if value is None:
            relative = Path(default)
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{split} 必须是单个目录路径")
            normalized = value.replace("\\", "/")
            windows_absolute = bool(re.match(r"^[A-Za-z]:/", normalized))
            absolute = windows_absolute or normalized.startswith("/")
            candidate = Path(value) if os.name == "nt" else Path(normalized)
            if not absolute:
                relative = candidate
                if root is None:
                    root = path.parent
            elif root is not None and override:
                # Rebase absolute entries below the original root onto --data-root.
                # Reject unrelated absolute paths instead of silently reading other data.
                parts = _path_parts(value)
                left = [x.casefold() for x in parts] if windows_absolute else parts
                right = [x.casefold() for x in old_parts] if windows_absolute else old_parts
                if right and left[:len(right)] == right:
                    suffix = parts[len(right):]
                    relative = Path(*suffix) if suffix else Path(default)
                else:
                    raise ValueError(f"{split}={value} 不在 YAML path={old_root_value} 下；"
                                     "请将 train/val 等路径改成相对数据根目录的路径")
            else:
                paths[split] = candidate.resolve()
                continue
        paths[split] = (root / relative).resolve() if root else (path.parent / relative).resolve()
    return DatasetSpec(path, root, paths, names_raw)


def label_root(image_root: Path) -> Path:
    """images/train -> labels/train, or custom/train/images -> custom/train/labels."""
    if image_root.parent.name.lower() == "images":
        return image_root.parent.with_name("labels") / image_root.name
    return image_root.parent / "labels" / image_root.name


def class_order(nc: int, mode: str, known_ids: list[int] | None) -> tuple[list[int], list[int]]:
    """Return output-index -> original YOLO ID, plus labelled output indices."""
    if mode == "supervised":
        return list(range(nc)), list(range(nc))
    if mode != "discovery":
        raise ValueError(f"未知模式: {mode}")
    if not known_ids or not all(isinstance(x, int) and not isinstance(x, bool) for x in known_ids):
        raise ValueError("NCD 必须显式提供 discovery.known_class_ids 或 --known-classes，例如 0,1,2")
    if len(set(known_ids)) != len(known_ids) or any(not 0 <= x < nc for x in known_ids):
        raise ValueError("known_class_ids 必须无重复且位于 0..nc-1")
    novel = [i for i in range(nc) if i not in known_ids]
    if not novel:
        raise ValueError("NCD 至少需要一个新类；全部有标签请使用 supervised")
    return list(known_ids) + novel, list(range(len(known_ids)))
