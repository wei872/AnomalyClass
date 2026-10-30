"""命令行公共工具: 配置加载与参数覆盖."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fabric_ncd.utils import load_yaml


def base_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/fabric19.yaml", help="训练/推理配置文件")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖任意配置项, 例如 --set data.image_size=[384,384] --set train.epochs=30")
    return parser


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


def load_config(path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
