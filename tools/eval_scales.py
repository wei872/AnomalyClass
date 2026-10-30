#!/usr/bin/env python
"""分辨率体检: 用已训练的 checkpoint 直接扫一遍输入分辨率, 看"看得清"到底是不是瓶颈.

背景: 训练一轮很贵 (大图上 4 分钟), 但"要不要更高分辨率"这个问题其实**不用重训**就能
先估个方向 —— 同一个权重在不同输入尺度上的 pixelAUROC / imageAUROC 差异, 基本决定了
再去重训是否值得.

用法::

    python tools/eval_scales.py --checkpoint runs/fabric19_fabricncd_s/best.pt
python tools/eval_scales.py --checkpoint best.pt --scales 0.5,0.65,0.8,1.0 --samples 60

输出示例::

    尺度   输入尺寸       Mpx    pixelAUROC  imageAUROC  相对耗时
    0.80   800x1952      1.56   0.7012      0.8341      1.00x
    1.00   1012x2448     2.48   0.7623      0.8802      1.62x

判读: pixelAUROC 随尺度明显上升 -> 分辨率是瓶颈 (小疵点在 1/8 特征图上不足一格);
平坦 -> 分辨率不是瓶颈, 别再花显存, 回去调损失/增广/阈值.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from fabric_ncd.data.yolo import build_dataloaders, build_normal_loader  # noqa: E402
from fabric_ncd.engine import evaluate  # noqa: E402
from fabric_ncd.models.fabric_ncd import FabricNCD  # noqa: E402
from fabric_ncd.ncd import RegionAnalyzer  # noqa: E402
from fabric_ncd.utils import get_logger, load_yaml, resolve_device  # noqa: E402

LOGGER = get_logger("eval_scales")


def _parse_scales(text: str) -> List[float]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or [0.8]


def _build_model(state: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> FabricNCD:
    model = FabricNCD(cfg)
    key = "ema" if "ema" in state else "model"
    try:
        model.load_state_dict(state[key])
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state[key], strict=False)
        LOGGER.warning(
            "权重与当前结构不完全一致: 跳过 %d 个参数 / %d 个多余参数",
            len(missing),
            len(unexpected),
        )
    calibration = state.get("calibration") or {}
    scale = float(calibration.get("memory_distance_scale", 0.0) or 0.0)
    if scale > 0:
        model.memory_distance_scale = scale
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description="分辨率体检: 扫输入尺度看 pixelAUROC 曲线")
    ap.add_argument("--checkpoint", required=True, help="训练产出的 best.pt / last.pt")
    ap.add_argument("--scales", default="0.5,0.65,0.8,1.0", help="缩放比例, 逗号分隔 (<=1 表示不放大)")
    ap.add_argument("--samples", type=int, default=0, help="验证集最多用多少张 (0 = 全部)")
    ap.add_argument("--memory-batches", type=int, default=20, help="重建记忆库用多少个 batch 正常图")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(state.get("config") or load_yaml("configs/fabric19.yaml"))
    cfg.setdefault("experiment", {})["device"] = str(device)

    rows = []
    base_time = None
    for scale in _parse_scales(args.scales):
        cfg_s = copy.deepcopy(cfg)
        data_cfg = cfg_s.setdefault("data", {})
        data_cfg["image_size"] = "auto"
        # short_side 放开, 完全由 max_scale 决定缩放 (>1 即放大)
        data_cfg["short_side"] = 8192
        data_cfg["max_scale"] = float(scale)

        datasets = build_dataloaders(cfg_s)
        resolved_hw = tuple(cfg_s["data"]["image_size"])
        val_loader = datasets["val"]
        model = _build_model(state, cfg_s, device)

        normal_loader = build_normal_loader(cfg_s, "train")
        if normal_loader is not None:
            model.memory.reset()
            with torch.no_grad():
                for i, batch in enumerate(normal_loader):
                    images = batch["image"].to(device)
                    mask = batch["cloth_mask"].to(device)
                    model.update_memory(images, mask=mask)
                    if i + 1 >= args.memory_batches:
                        break

        hw = resolved_hw
        analyzer = RegionAnalyzer(
            names=None,
            use_cloth=bool(cfg_s.get("eval", {}).get("cloth_gate", True)),
            threshold=None,
        )
        started = time.time()
        metrics = evaluate(
            model, val_loader, cfg_s, device, criterion=None,
            analyzer=analyzer, max_samples=args.samples or None,
        )
        elapsed = time.time() - started
        base_time = base_time or elapsed
        rows.append(
            {
                "scale": scale,
                "hw": hw,
                "pixel_auroc": metrics.get("pixel_auroc", float("nan")),
                "image_auroc": metrics.get("image_auroc", float("nan")),
                "time": elapsed,
            }
        )
        LOGGER.info(
            "尺度 %.2f | pixelAUROC %.4f | imageAUROC %.4f | %.0fs",
            scale, rows[-1]["pixel_auroc"], rows[-1]["image_auroc"], elapsed,
        )

    LOGGER.info("%-6s %-16s %-10s %-12s %-8s", "尺度", "输入尺寸", "pixelAUROC", "imageAUROC", "相对耗时")
    for row in rows:
        hw = row["hw"]
        size_text = f"{hw[0]}x{hw[1]}" if hw else "-"
        LOGGER.info(
            "%-6.2f %-16s %-10.4f %-12.4f %.2fx",
            row["scale"], size_text, row["pixel_auroc"], row["image_auroc"], row["time"] / max(base_time, 1e-6),
        )
    if len(rows) >= 2:
        delta = rows[-1]["pixel_auroc"] - rows[0]["pixel_auroc"]
        if delta > 0.05:
            LOGGER.info("判读: 提高尺度能让 pixelAUROC 明显变好 (Δ=%.3f) -> 分辨率是瓶颈, 值得提分辨率/切块", delta)
        else:
            LOGGER.info("判读: 分辨率不是主要瓶颈 (Δ=%.3f) -> 去调损失/增广/标注质量, 别再吃显存", delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
