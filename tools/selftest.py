# -*- coding: utf-8 -*-
"""环境/流水线自检: 合成数据 -> 训练(1 epoch) -> 标定 -> 导出 -> pt/onnx 一致性 -> 开集规则。
    python tools/selftest.py            # 全部通过打印 ALL PASS
用于新机器验收: 任何一个环节坏掉都会在此暴露, 而不是在产线上。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK = True


def stage(name):
    print(f"\n=== [selftest] {name} ===", flush=True)


def check(cond, msg):
    global OK
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        OK = False


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fabricdet_selftest_"))
    py = sys.executable

    stage("1. 合成数据生成")
    r = subprocess.run([py, str(ROOT / "tools" / "make_synthetic_dataset.py"),
                        "--out", str(tmp / "syn"), "--train", "24", "--val", "6"],
                       capture_output=True, text=True)
    check(r.returncode == 0 and (tmp / "syn" / "images" / "train").exists(), "合成数据集生成")

    stage("2. DINOv2 checkpoint 兼容性 (官方 key 格式)")
    from fabricdet.dinov2 import DINOv2ViT, load_pretrained
    bb = DINOv2ViT()
    sd = bb.state_dict()
    sd["mask_token"] = torch.zeros(1, 1, 384)
    torch.save(sd, tmp / "fake_official.pth")
    bb2 = DINOv2ViT()
    load_pretrained(bb2, tmp / "fake_official.pth", verbose=False)
    a = bb(torch.zeros(1, 3, 112, 112))
    b = bb2(torch.zeros(1, 3, 112, 112))
    check(torch.allclose(a, b), "官方格式 state_dict 加载后输出一致")

    stage("3. 一次训练 (1 epoch) + 标定 + 导出")
    r = subprocess.run([py, str(ROOT / "train.py"), "--data", str(tmp / "syn" / "synthetic_fabric19.yaml"),
                        "--device", "cpu",
                        "--set", "epochs=1", "batch=2", "accum=1", "input_size=224", "tile=224",
                        "freeze_epochs=1", "eval_every=1", "num_workers=0",
                        "out_dir=" + str(tmp / "runs"), "backbone_ckpt=scratch"],
                       capture_output=True, text=True)
    check(r.returncode == 0, "train.py 全流程退出码 0")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    dep = tmp / "runs" / "deploy"
    check((dep / "best.onnx").exists(), "ONNX 导出")
    check((dep / "deploy_config.json").exists(), "deploy_config.json 导出")

    stage("4. pt / onnx 双后端一致性")
    from fabricdet.infer import Detector
    d1 = Detector(tmp / "runs" / "best.pt", device="cpu")
    d2 = Detector(dep / "best.onnx", device="cpu")
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        ref = [t.numpy() for t in d1.model.forward_deploy(torch.from_numpy(x))]
    out = d2.ort.run(None, {"images": x})
    err = max(float(np.abs(p - q).max()) for p, q in zip(ref, out))
    check(err < 5e-3, f"ONNX 与 PyTorch 最大偏差 {err:.2e} < 5e-3")

    stage("5. 滑窗 tile + 跨 tile 合并 + 开集规则")
    from fabricdet.postprocess import merge_tile_preds, postprocess, UNKNOWN_CLS
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]], np.float32)
    obj = np.log(np.array([0.9, 0.85, 0.7]) / (1 - np.array([0.9, 0.85, 0.7])))
    sims = np.zeros((3, 4), np.float32)
    sims[0, 1] = 0.8
    sims[1, 1] = 0.8
    sims[2, :] = 0.05          # 与所有原型都不像 -> 未知
    res = postprocess(boxes, obj, sims, {"conf": 0.3, "tau_unknown": 0.3})
    check(len(res) == 2, "重叠框被 NMS 合并")
    check(any(p["cls"] == UNKNOWN_CLS for p in res), "低余弦相似度判为未知缺陷")
    check(any(p["cls"] == 1 for p in res), "高余弦相似度判为对应已知类")
    merged = merge_tile_preds(res + [dict(res[0], box=res[0]["box"] + 3)], 0.5)
    check(len(merged) == 2, "跨 tile 重复合并")

    stage("6. mAP 指标自检 (完美预测应为 1.0)")
    from fabricdet.metrics import evaluate_ap
    gt = {"a": [(0, np.array([10, 10, 50, 50], np.float32))]}
    pr = {"a": [{"box": np.array([10, 10, 50, 50], np.float32), "score": 0.9, "cls": 0}]}
    _, m, _ = evaluate_ap(gt, pr, 2, 0.5)
    check(abs(m - 1.0) < 1e-9, f"完美预测 mAP50={m}")

    stage("7. 高分辨率滑窗推理不崩溃")
    big = np.full((500, 900, 3), 200, np.uint8)
    preds = d1.predict(big, tile=224, overlap=0.25)
    check(isinstance(preds, list), f"900x500 滑窗推理 OK ({len(preds)} 预测)")

    print("\n" + ("ALL PASS ✅" if OK else "HAS FAILURES ❌"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
