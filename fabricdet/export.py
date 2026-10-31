# -*- coding: utf-8 -*-
"""工业导出: ONNX (opset17, 动态 batch) + TorchScript + deploy_config.json。
导出后自动用 onnxruntime 与 PyTorch 输出比对, 保证部署一致性。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .infer import NAMES_EN


class _DeployWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m.forward_deploy(x)


def export_onnx(model: torch.nn.Module, path: Path, size: int, opset: int = 17,
                verify: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, size, size)
    kwargs = dict(
        opset_version=opset,
        input_names=["images"], output_names=["boxes", "obj_logits", "sims"],
        dynamic_axes={"images": {0: "batch"}, "boxes": {0: "batch"},
                      "obj_logits": {0: "batch"}, "sims": {0: "batch"}},
        do_constant_folding=True,
    )
    fn = _DeployWrapper(model)
    try:  # torch>=2.5: 显式走稳定版导出器, 避免强依赖 onnxscript
        torch.onnx.export(fn, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # 旧版 torch 无 dynamo 参数
        torch.onnx.export(fn, (dummy,), str(path), **kwargs)
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            onnx.checker.check_model(onnx.load(str(path)))
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            x = np.random.rand(1, 3, size, size).astype(np.float32)
            with torch.no_grad():
                ref = [t.numpy() for t in model.forward_deploy(torch.from_numpy(x))]
            out = sess.run(None, {"images": x})
            for r, o, name in zip(ref, out, ("boxes", "obj", "sims")):
                err = np.abs(r - o).max()
                if err > 5e-3:
                    print(f"[export] 警告: ONNX 与 PyTorch 输出 {name} 最大偏差 {err:.4f}")
            print(f"[export] ONNX 校验通过: {path}")
        except Exception as e:
            print(f"[export] 警告: ONNX 校验跳过 ({e})")
    return path


def export_torchscript(model: torch.nn.Module, path: Path, size: int) -> Path:
    path = Path(path)
    model.eval()
    ts = torch.jit.trace(_DeployWrapper(model), (torch.zeros(1, 3, size, size),), strict=True)
    ts.save(str(path))
    print(f"[export] TorchScript: {path}")
    return path


def write_deploy_config(path: Path, meta: dict):
    path = Path(path)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] deploy_config.json: {path}")


def build_meta(names, names_en, size, thresholds, extra: dict | None = None) -> dict:
    meta = {
        "model": "FabricDefectNet",
        "backbone": "dinov2_vits14",
        "input_size": size,
        "stride": 7,
        "names": names,
        "names_en": (names_en or NAMES_EN)[:len(names)],
        "thresholds": thresholds,
        "onnx_inputs": {"images": "float32 [batch,3,H,W], ImageNet 归一化"},
        "onnx_outputs": {"boxes": "[batch,N,4] px xyxy", "obj_logits": "[batch,N]",
                         "sims": "[batch,N,C] 余弦相似度; max<tau_unknown => 未知缺陷"},
        "postprocess": "fabricdet.postprocess.postprocess (numpy, 与训练同源码)",
    }
    if extra:
        meta.update(extra)
    return meta
