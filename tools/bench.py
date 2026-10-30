#!/usr/bin/env python
"""轻量化基准: 参数量 / 计算量 / 延迟 / 内存.

示例::

    python tools/bench.py --size 512 --device cpu
    python tools/bench.py --size 640 --device cuda --iters 50
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.models.fabric_ncd import FabricNCD  # noqa: E402
from fabric_ncd.models.memory import MemoryBank  # noqa: E402
from fabric_ncd.models.texture import GaborBank  # noqa: E402
from fabric_ncd.utils import json_dump, model_config_for_weights, resolve_device  # noqa: E402


def count_macs(model: nn.Module, input_size: Tuple[int, int], batch_size: int = 1) -> Tuple[float, Dict[str, float]]:
    """统计卷积/全连接/Gabor/记忆库乘加次数 (MACs), 单位 G."""
    total = {"conv": 0.0, "linear": 0.0, "gabor": 0.0, "memory": 0.0}

    def conv_hook(module: nn.Conv2d, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        macs = out.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        total["conv"] += macs

    def linear_hook(module: nn.Linear, inputs, output):
        total["linear"] += output.numel() * module.in_features

    def gabor_hook(module: GaborBank, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        total["gabor"] += out.numel() * module.kernel_size * module.kernel_size

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, GaborBank):
            handles.append(m.register_forward_hook(gabor_hook))
    was_training = model.training
    model.eval()
    try:
        parameter = next(model.parameters())
        with torch.no_grad():
            output = model(torch.zeros(batch_size, 3, *input_size, device=parameter.device, dtype=parameter.dtype))
        # 查询通过 memory.similarity() 直接调用而非 Module.forward，forward hook 不会触发。
        memory = getattr(model, "memory", None)
        if getattr(model, "memory_enabled", False) and isinstance(memory, MemoryBank):
            emb = output["embedding"]
            h, w = emb.shape[-2:]
            stride = max(1, int(memory.query_stride))
            if min(h, w) >= stride:
                h, w = (h + stride - 1) // stride, (w + stride - 1) // stride
            bank_size = int(memory.size.item()) or int(memory.capacity)
            total["memory"] = emb.shape[0] * h * w * bank_size * memory.dim
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    macs = sum(total.values()) / 1e9
    return macs, {k: v / 1e9 for k, v in total.items()}


@torch.no_grad()
def latency(model: nn.Module, input_size: Tuple[int, int], device: torch.device, iters: int = 20,
            batch_size: int = 1) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 3, *input_size, device=device)
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times_arr = np.asarray(times)
    return {"mean_ms": float(times_arr.mean()), "p50_ms": float(np.median(times_arr)),
            "p95_ms": float(np.percentile(times_arr, 95)), "fps": float(batch_size * 1000.0 / times_arr.mean())}


def _parse_hw(text) -> Tuple[int, int]:
    """'512' -> (512, 512); '832x1600' -> (832, 1600) 即 (H, W)."""
    text = str(text).lower()
    for sep in ("x", "*", ","):
        if sep in text:
            a, b = text.split(sep)[:2]
            return int(a), int(b)
    value = int(text)
    return value, value


def main() -> int:
    parser = base_parser("FabricNCD 轻量化基准")
    parser.add_argument("--size", default="512",
                        help="输入尺寸: 整数 (正方形) 或 HxW, 如 832x1600")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.iters < 1:
        parser.error("--batch-size 和 --iters 必须为正")
    cfg = load_config(args.config, args.overrides, args.root)
    size_hw = _parse_hw(args.size)
    if min(size_hw) < 1:
        parser.error("--size 必须为正")
    cfg.setdefault("data", {})["image_size"] = [size_hw[0], size_hw[1]]
    device = resolve_device(args.device)
    # 基准只关心结构，无需下载预训练权重；构造满容量记忆库以测实际查询开销。
    model = FabricNCD(model_config_for_weights(cfg)).to(device).eval()
    if model.memory_enabled:
        with torch.no_grad():
            model.memory.features.copy_(torch.nn.functional.normalize(torch.randn_like(model.memory.features), dim=1))
            model.memory.size.fill_(model.memory.capacity)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    macs, macs_parts = count_macs(model, size_hw, args.batch_size)
    lat = latency(model, size_hw, device, args.iters, args.batch_size)

    report = {
        "device": str(device),
        "input_size": [size_hw[0], size_hw[1]],
        "batch_size": args.batch_size,
        "memory_bank_size": int(model.memory.size.item()),
        "params_M": round(total_params / 1e6, 2),
        "backbone_params_M": round(backbone_params / 1e6, 2),
        "macs_G": round(macs, 3),
        "flops_G": round(macs * 2, 3),
        "macs_conv_G": round(macs_parts["conv"], 3),
        "macs_linear_G": round(macs_parts["linear"], 3),
        "macs_gabor_G": round(macs_parts["gabor"], 3),
        "macs_memory_G": round(macs_parts["memory"], 6),
        **{k: round(v, 2) for k, v in lat.items()},
    }
    for key, value in report.items():
        print(f"{key:>20}: {value}")
    if args.out:
        json_dump(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
