# -*- coding: utf-8 -*-
"""独立评估: 用训练好的权重在 val 集上做部署级 (滑窗) 评估。
    python eval.py --ckpt runs/fabric19/best.pt --data dataset/fabric19.yaml --data-root ...
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from fabricdet.data import imread_unicode, load_dataset_yaml, read_labels, resolve_dataset_paths
from fabricdet.engine import final_eval
from fabricdet.infer import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="dataset/fabric19.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=10000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    det = Detector(a.ckpt, device=a.device)
    dcfg = load_dataset_yaml(a.data)
    paths = resolve_dataset_paths(dcfg, a.data_root)
    if paths.get("val") is None:
        raise SystemExit("[eval] 缺少 val 目录, 请用 --data-root 指定")
    cfg = {"tile": a.tile or det.size, "overlap": 0.25}
    th = dict(det.thresholds)
    if a.conf is not None:
        th["conf"] = a.conf
    items = []
    for p in sorted(Path(paths["val"]).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        lp = paths["val"].parent.parent / "labels" / "val" / (p.stem + ".txt")
        gt = [(c, np.array([x1 * w, y1 * h, x2 * w, y2 * h], np.float32))
              for c, x1, y1, x2, y2 in read_labels(str(lp), set())]
        items.append((str(p), gt))
    if len(items) > a.max_images:
        items = random.Random(0).sample(items, a.max_images)
    model = det.model
    if model is None:
        raise SystemExit("[eval] ONNX 后端请用 python -m fabricdet.infer; eval.py 需要 .pt 权重")
    device = det.device
    aps50, map50, map95, extra, fppi = final_eval(model, items, device, cfg, th)
    print(f"[eval] {len(items)} 张 | mAP50={map50:.4f} mAP75={map95:.4f} FPPI={fppi:.2f} "
          f"未知预测={extra['unknown_pred']} | conf={th['conf']} tau={th['tau_unknown']}")
    for n, ap, tp, fn in zip(dcfg["names"], aps50, extra["tp"], extra["fn"]):
        print(f"  {n:<8s} AP50={0.0 if np.isnan(ap) else ap:.3f}  tp={tp} fn={fn}")


if __name__ == "__main__":
    main()
