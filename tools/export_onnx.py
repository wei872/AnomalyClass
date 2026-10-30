#!/usr/bin/env python
"""导出 ONNX (供 TensorRT / OpenVINO / onnxruntime 部署).

注意: 区域分析 (阈值/连通域/区域合并) 是后处理, 不在计算图内, 需要部署端
用 ``fabric_ncd.ncd.RegionAnalyzer`` 的等价逻辑实现 (numpy 版可直接移植).

示例::

    python tools/export_onnx.py --ckpt runs/fabric19_fabricncd_s/best.pt --out model.onnx --size 512
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import base_parser, load_config  # noqa: E402

from fabric_ncd.models.fabric_ncd import FabricNCD  # noqa: E402
from fabric_ncd.utils import deep_update, get_logger  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """只输出部署需要的张量 (去掉 dict 与调试信息).

    织造周期描述子 ``weave_desc`` (B,6) 作为第二个输入, 由部署端用 numpy 版
    ``fabric_ncd.data.weave.estimate_period`` 计算 (或直接喂 0). 这样计算图里
    不含 FFT, 对 TensorRT / OpenVINO 更友好.
    """

    def __init__(self, model: FabricNCD) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, weave_desc: torch.Tensor):
        out = self.model(x, weave_desc=weave_desc)
        return out["anomaly_logit"], out["cloth_logit"], out["boundary_logit"], out["class_logits"], out["embedding"]


def main() -> int:
    parser = base_parser("导出 ONNX")
    parser.add_argument("--ckpt", default="runs/fabric19_fabricncd_s/best.pt")
    parser.add_argument("--out", default="model.onnx")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic", action="store_true", help="动态 H/W (TensorRT 建议固定尺寸)")
    parser.add_argument("--check", action="store_true", help="用 onnxruntime 校验数值一致性")
    args = parser.parse_args()

    logger = get_logger("fabric_ncd")
    cfg = load_config(args.config, args.overrides)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = deep_update(state.get("config", {}), cfg)
    size = args.size or int(cfg.get("data", {}).get("image_size", [512])[0])
    model = FabricNCD(cfg)
    model.load_state_dict(state.get("ema", state["model"]))
    model.eval()

    wrapper = ExportWrapper(model).eval()  # 必须在 eval 模式下导出, 否则 BatchNorm 会用当前 batch 统计量
    dummy = torch.randn(1, 3, size, size)
    dummy_weave = torch.zeros(1, 6)
    dynamic_axes = {"images": {0: "batch"}, "weave_desc": {0: "batch"}}
    if args.dynamic:
        dynamic_axes["images"].update({2: "height", 3: "width"})
        for name in ("anomaly_logit", "cloth_logit", "boundary_logit", "class_logits", "embedding"):
            dynamic_axes[name] = {0: "batch", 2: "h", 3: "w"}
    torch.onnx.export(
        wrapper,
        (dummy, dummy_weave),
        args.out,
        input_names=["images", "weave_desc"],
        output_names=["anomaly_logit", "cloth_logit", "boundary_logit", "class_logits", "embedding"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("ONNX 已导出: %s (输入 %dx%d)", args.out, size, size)

    if args.check:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = wrapper(dummy, dummy_weave)
        onnx_out = session.run(None, {"images": dummy.numpy(), "weave_desc": dummy_weave.numpy()})
        for name, a, b in zip(
            ["anomaly_logit", "cloth_logit", "boundary_logit", "class_logits", "embedding"], torch_out, onnx_out
        ):
            diff = float(np.max(np.abs(a.numpy() - b)))
            logger.info("  输出 %-16s 最大误差 %.2e", name, diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
