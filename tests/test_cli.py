"""命令行公共参数: --root 必须被所有工具接受, 并写进 data.root."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from tools._common import base_parser, load_config, set_root

ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted(p.name for p in (ROOT / "tools").glob("*.py") if not p.name.startswith("_"))


def test_root_flag_is_shared_by_all_tools():
    """每个工具都走 base_parser, 所以都该有 --root (train/infer/evaluate/... 一致)."""
    parser = base_parser("t")
    ns = parser.parse_args(["--root", "D:/tyhn/yolo/dataset/source5_augmenteds"])
    assert ns.root == "D:/tyhn/yolo/dataset/source5_augmenteds"


def test_all_tools_pass_root_to_load_config():
    """防止再出现"某个工具忘了接 --root" (train.py 曾因此报 unrecognized arguments)."""
    offenders = []
    for name in TOOLS:
        tree = ast.parse((ROOT / "tools" / name).read_text(encoding="utf-8"))
        has_root_flag = False
        passes_root = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                fname = getattr(func, "id", getattr(func, "attr", ""))
                if fname == "load_config":
                    flags = [getattr(a, "attr", "") for a in node.args]
                    if "root" in flags:
                        passes_root = True
                if fname == "add_argument" and node.args:
                    arg = getattr(node.args[0], "value", "")
                    if arg == "--root":
                        has_root_flag = True
        if has_root_flag and not passes_root:  # 自己声明了 --root 却没传下去
            offenders.append(name)
    assert not offenders, offenders


def test_set_root_overrides_config(tmp_path):
    cfg = {"data": {"root": "D:/wrong"}}
    out = set_root(dict(cfg), str(tmp_path))
    assert out["data"]["root"] == str(tmp_path)


def test_load_config_applies_root(tmp_path):
    yml = tmp_path / "c.yaml"
    yml.write_text("data:\n  root: D:/original\n  image_size: [64, 64]\n", encoding="utf-8")
    cfg = load_config(yml, ["train.epochs=3"], root="D:/tyhn/yolo/dataset/source5_augmenteds")
    assert cfg["data"]["root"] == "D:/tyhn/yolo/dataset/source5_augmenteds"
    assert cfg["train"]["epochs"] == 3


def test_train_help_lists_root():
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "train.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0
    assert "--root" in out.stdout
