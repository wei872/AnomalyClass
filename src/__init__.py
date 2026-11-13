"""AnomalyClass 包.

懒加载 (PEP 562): `import src` / `from src.data import ...` 不会立刻 import torch 及重型子模块,
这样 Windows DataLoader spawn 子进程重新 import 主脚本时保持轻量快速.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import FabricAnomalyPipeline, ImageResult, RegionResult, build_pipeline
    from .detect import AnomalyDetector, train_detector, calibrate, evaluate_detector, deviation_confidence
    from .classify import RegionClassifier, train_classifier, build_classifier
    from .data import load_fabric_config, FabricConfig, UNKNOWN_CLASS_NAME

_LAZY = {
    'FabricAnomalyPipeline': '.pipeline', 'ImageResult': '.pipeline',
    'RegionResult': '.pipeline', 'build_pipeline': '.pipeline',
    'AnomalyDetector': '.detect', 'train_detector': '.detect', 'calibrate': '.detect',
    'evaluate_detector': '.detect', 'deviation_confidence': '.detect',
    'RegionClassifier': '.classify', 'train_classifier': '.classify', 'build_classifier': '.classify',
    'load_fabric_config': '.data', 'FabricConfig': '.data', 'UNKNOWN_CLASS_NAME': '.data',
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
