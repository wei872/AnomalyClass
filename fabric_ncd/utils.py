"""通用工具: 配置、日志、随机种子、设备、指标累积."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: str | os.PathLike) -> None:
    import yaml

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置, override 优先."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def get_by_path(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------- #
# 运行环境
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """设备的人话描述 (CUDA 时带显卡型号与显存), 便于确认真的用上了 GPU."""
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            props = torch.cuda.get_device_properties(index)
            return f"cuda:{index} {props.name} ({props.total_memory / 2**30:.1f} GB)"
        except Exception:  # pragma: no cover
            return f"cuda:{index}"
    return str(device)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "fabric_ncd", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复挂 handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class AverageMeter:
    """滑动平均, 用于 loss / 指标累积."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.last = 0.0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)
        self.sum += value * n
        self.count += n
        self.last = value

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def reset(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def json_dump(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"不可序列化类型: {type(obj)}")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_to(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(v, device) for v in obj)
    return obj


def describe_environment(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "python": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if logger:
        logger.info("环境: %s", info)
    return info


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def freeze_module(module: torch.nn.Module, train_bn: bool = False) -> None:
    for p in module.parameters():
        p.requires_grad = False
    if not train_bn:
        for m in module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False


def iter_chunks(total: int, chunk: int) -> Iterable[range]:
    for start in range(0, total, chunk):
        yield range(start, min(start + chunk, total))


def prepare_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """初始化项目日志 (同时接管子模块 logger)."""
    logger = get_logger("fabric_ncd", log_file)
    logger.setLevel(level)
    logging.getLogger("fabric_ncd").setLevel(level)
    return logger
