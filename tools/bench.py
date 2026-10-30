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
from fabric_ncd.utils import json_dump, resolve_device  # noqa: E402


def count_macs(model: nn.Module, input_size: Tuple[int, int]) -> float:
    """统计卷积/全连接乘加次数 (MACs), 单位 G."""
    total = [0.0]

    def conv_hook(module: nn.Conv2d, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        macs = out.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        total[0] += macs

    def linear_hook(module: nn.Linear, inputs, output):
        total[0] += output.numel() * module.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, *input_size))
    for h in handles:
        h.remove()
    model.train(was_training)
    return total[0] / 1e9


@torch.no_grad()
def latency(model: nn.Module, input_size: Tuple[int, int], device: torch.device, iters: int = 20) -> Dict[str, float]:
    model.eval()
    x = torch.randn(1, 3, *input_size, device=device)
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
            "p95_ms": float(np.percentile(times_arr, 95)), "fps": float(1000.0 / times_arr.mean())}


def main() -> int:
    parser = base_parser("FabricNCD 轻量化基准")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides, args.root)
    cfg.setdefault("data", {})["image_size"] = [args.size, args.size]
    device = resolve_device(args.device)
    model = FabricNCD(cfg).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    macs = count_macs(model, (args.size, args.size))
    lat = latency(model, (args.size, args.size), device, args.iters)

    report = {
        "device": str(device),
        "input_size": [args.size, args.size],
        "params_M": round(total_params / 1e6, 2),
        "backbone_params_M": round(backbone_params / 1e6, 2),
        "macs_G": round(macs, 3),
        "flops_G": round(macs * 2, 3),
        **{k: round(v, 2) for k, v in lat.items()},
    }
    for key, value in report.items():
        print(f"{key:>20}: {value}")
    if args.out:
        json_dump(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
