"""命令行公共工具: 配置加载与参数覆盖."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fabric_ncd.utils import deep_update, load_yaml


def base_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, help="配置文件 (默认使用内置配置；加载 checkpoint 时优先使用其训练配置)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖任意配置项, 例如 --set data.image_size=[384,384] --set train.epochs=30")
    parser.add_argument("--root", default=None,
                        help="本机数据集根目录, 等价于 --set data.root=... (Windows 必须写完整盘符, 如 D:/tyhn/yolo/dataset/source5_augmenteds)")
    return parser


def set_root(cfg: Dict[str, Any], root: str | None) -> Dict[str, Any]:
    """把 ``--root`` 写进 ``data.root``, 并对 Windows 盘符陷阱给出提示。

    ``/data/fabric`` 在 Windows 上是"当前盘符下的相对路径"(C:\\data\\fabric),
    而不是 D 盘, 这是最容易踩的坑, 所以在解析阶段就说清楚。
    """
    if not root:
        return cfg
    text = str(root).strip().strip('"').strip("'")
    if os.name == "nt" and text.startswith("/") and not text.startswith("//"):
        print(
            f"[warn] --root {text} 在 Windows 上会被解析成当前盘符下的相对路径, "
            f"不是 D 盘。请写完整盘符, 例如 D:{text}"
        )
    cfg.setdefault("data", {})["root"] = text
    return cfg


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fabric19.yaml"


def load_config(
    path: str | Path | None,
    overrides: List[str] | None = None,
    root: str | None = None,
    checkpoint_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # 默认值 < checkpoint 配置 < 显式 --config < --set / --root。
    cfg = load_yaml(DEFAULT_CONFIG)
    if checkpoint_config:
        cfg = deep_update(cfg, checkpoint_config)
    if path is not None:
        cfg = deep_update(cfg, load_yaml(path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if root:
        cfg = set_root(cfg, root)
    return cfg


def load_predict_overrides(
    path: str | Path | None, overrides: List[str] | None = None, root: str | None = None
) -> Dict[str, Any]:
    """只返回显式覆盖，避免把默认配置反向覆盖 checkpoint 的训练配置。"""
    cfg = load_yaml(path) if path is not None else {}
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if root:
        cfg = set_root(cfg, root)
    return cfg
