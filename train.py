# -*- coding: utf-8 -*-
"""一次训练, 直接可用:
    python train.py --data dataset/fabric19.yaml --data-root /path/to/source5_augmenteds
流程: 构建数据 -> DINOv2 预训练骨干 -> 分阶段端到端训练 -> val 选 best ->
      阈值自动标定 (conf / 未知缺陷 tau) -> 部署级复评 -> 保存 best.pt + ONNX + TorchScript
训练结束后 runs/fabric19/deploy/ 内即可直接工业部署, 无需再改任何代码。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fabricdet.data import (FabricDataset, build_class_weight, class_frequency, collate,
                            imread_unicode, load_dataset_yaml, read_labels, resolve_dataset_paths)
from fabricdet.dinov2 import load_pretrained
from fabricdet.engine import (EMA, Scheduler, calibrate_thresholds, fast_eval, final_eval,
                              reset_sched_counter, train_one_epoch)
from fabricdet.export import build_meta, export_onnx, export_torchscript, write_deploy_config
from fabricdet.infer import NAMES_EN
from fabricdet.losses import FabricLoss
from fabricdet.model import FabricDefectNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/fabric19.yaml")
    ap.add_argument("--data-root", default=None, help="数据集根目录 (覆盖 yaml 中失效的绝对路径)")
    ap.add_argument("--cfg", default="configs/fabric19_train.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value", help="覆盖 cfg, 如 epochs=30")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8")) if Path(a.cfg).exists() else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cfg[k] = v
    dcfg = load_dataset_yaml(a.data)
    paths = resolve_dataset_paths(dcfg, a.data_root)
    missing = [k for k, v in paths.items() if v is None]
    if missing:
        raise SystemExit(f"[train] 数据集路径缺失: {missing}\n"
                         f"        请用 --data-root 指定根目录 (含 images/train, labels/train, normal/...)")
    set_seed(int(cfg.get("seed", 42)))
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(cfg.get("input_size", 448))
    assert size % 14 == 0, "input_size 必须是 14 的倍数"

    train_ds = FabricDataset("train", paths, dcfg["names"], size, True, cfg)
    val_ds = FabricDataset("val", paths, dcfg["names"], size, False, cfg)
    nc = dcfg["nc"]
    cnt = class_frequency(train_ds, nc)
    cw = build_class_weight(cnt)
    print(f"[train] 数据集: train={len(train_ds)} (缺陷图+正常图), val={len(val_ds)}, nc={nc}")
    print(f"[train] 类别样本数: " + ", ".join(f"{n}:{int(c)}" for n, c in zip(dcfg['names'], cnt)))

    nw = min(int(cfg.get("num_workers", 8)), max(os.cpu_count() or 1, 1))
    bs = int(cfg.get("batch", 8))
    accum = int(cfg.get("accum", 4))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          collate_fn=collate, drop_last=False, persistent_workers=nw > 0)
    model = FabricDefectNet(nc, emb_dim=int(cfg.get("emb_dim", 128)), input_size=size)
    src = load_pretrained(model.backbone, cfg.get("backbone_ckpt", "auto"))
    from_scratch = src == "scratch"
    model.to(device)
    flops, params = model.flops_params(size)
    print(f"[train] 模型参数 {params / 1e6:.1f}M, FLOPs@{size} = {flops / 1e9:.1f}G, 骨干来源: {src}")

    epochs = int(cfg.get("epochs", 60))
    if from_scratch:
        epochs = int(epochs * 1.5)
        print(f"[train] 无预训练权重: epochs 自动调整为 {epochs}")
    freeze_ep = max(1, int(cfg.get("freeze_epochs", 6))) // (2 if from_scratch else 1)

    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    lr = float(cfg.get("lr", 6e-4))
    opt = torch.optim.AdamW([{"params": hd, "lr": lr},
                             {"params": bb, "lr": lr * float(cfg.get("backbone_lr_ratio", 0.1))}],
                            weight_decay=float(cfg.get("weight_decay", 0.05)))
    for p in bb:
        p.requires_grad_(False)
    total_it = (len(train_dl) // accum + 1) * epochs
    sched = Scheduler(opt, int(cfg.get("warmup_epochs", 1)) * max(len(train_dl) // accum, 1),
                      total_it)
    reset_sched_counter()
    use_amp = cfg.get("amp", "auto") in ("auto", "cuda", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, float(cfg.get("ema_decay", 0.9995)))
    loss_fn = FabricLoss(nc, cfg, cw)

    out = Path(cfg.get("out_dir", "runs/fabric19"))
    (out / "deploy").mkdir(parents=True, exist_ok=True)

    def save_ckpt(path, state, thresholds=None, extra=None):
        meta = build_meta(dcfg["names"], NAMES_EN, size,
                          thresholds or {"conf": float(cfg.get("conf_default", 0.25)),
                                         "tau_unknown": 0.35, "nms_iou": float(cfg.get("nms_iou", 0.45))},
                          extra={"emb_dim": int(cfg.get("emb_dim", 128)),
                                 "dataset": str(a.data), "backbone_pretrain": src,
                                 "class_counts": {n: int(c) for n, c in zip(dcfg['names'], cnt)}})
        if extra:
            meta.update(extra)
        torch.save({"state": state, "meta": meta}, path)

    best_map, best_ep = -1.0, -1
    t_start = time.time()
    for ep in range(epochs):
        if ep == freeze_ep:
            for p in bb:
                p.requires_grad_(True)
            print(f"[train] epoch {ep}: 解冻骨干, 端到端微调")
        t0 = time.time()
        train_one_epoch(model, train_dl, loss_fn, opt, sched, scaler, device, ema, ep,
                        accum=accum, use_amp=use_amp)
        if (ep + 1) % int(cfg.get("eval_every", 5)) == 0 or ep == epochs - 1:
            train_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.load_into(model)
            m50, extra = fast_eval(model, val_ds, device)
            print(f"[train] epoch {ep}: val mAP50 = {m50:.4f} "
                  f"(tp={extra['tp'].sum()} fp={extra['fp'].sum()} fn={extra['fn'].sum()}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if (m50, ep) > (best_map, best_ep):   # 同分时取更晚 epoch (训练更充分)
                best_map, best_ep = m50, ep
                save_ckpt(out / "best.pt", {k: v.clone() for k, v in ema.shadow.items()},
                          extra={"val_map50": round(m50, 4), "best_epoch": ep})
            model.load_state_dict(train_state)   # 恢复训练权重, 继续训练
            torch.save({"state": model.state_dict(), "meta": {"note": "last"}}, out / "last.pt")
    print(f"[train] 训练完成: best mAP50={best_map:.4f} @ epoch {best_ep}, 用时 {(time.time() - t_start) / 60:.1f} min")

    # ---------------- 标定 + 部署级复评 + 导出 (一次训练, 全部自动) ----------------
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"])
    model.to(device).eval()

    val_items = []
    for ip, lp in val_ds.samples:
        img = imread_unicode(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = [(c, np.array([x1 * w, y1 * h, x2 * w, y2 * h], np.float32))
              for c, x1, y1, x2, y2 in read_labels(lp, set())]
        val_items.append((ip, gt))
    cap = 300
    if len(val_items) > cap:
        val_items = random.Random(0).sample(val_items, cap)
    print(f"[train] 阈值自动标定 (val {len(val_items)} 张, 部署同款滑窗推理)...")
    thresholds, cstats = calibrate_thresholds(model, val_items, device, cfg)
    print(f"[train] 标定结果: conf={thresholds['conf']}  tau_unknown={thresholds['tau_unknown']}  {cstats}")
    aps50, map50, map95, extra50, fppi = final_eval(model, val_items, device, cfg, thresholds)
    print(f"[train] 部署级 val: mAP50={map50:.4f}  mAP75={map95:.4f}  FPPI={fppi:.2f}  未知预测={extra50['unknown_pred']}")
    print("[train]  per-class AP50:")
    for n, ap, tp, fn in zip(dcfg["names"], aps50, extra50["tp"], extra50["fn"]):
        print(f"    {n:<8s} AP={0.0 if np.isnan(ap) else ap:.3f}  tp={tp} fn={fn}")

    save_ckpt(out / "best.pt", ck["state"], thresholds=thresholds,
              extra={"val_map50": round(map50, 4), "val_map75": round(map95, 4),
                     "fppi": round(float(fppi), 3), "calibration": cstats,
                     "best_epoch": best_ep, "emb_dim": int(cfg.get("emb_dim", 128)),
                     "dataset": str(a.data), "backbone_pretrain": src,
                     "class_counts": {n: int(c) for n, c in zip(dcfg['names'], cnt)}})
    deploy = out / "deploy"
    export_onnx(model, deploy / "best.onnx", size)
    export_torchscript(model, deploy / "best.torchscript.pt", size)
    meta = build_meta(dcfg["names"], NAMES_EN, size, thresholds,
                      extra={"val_map50": round(map50, 4), "val_map75": round(map95, 4)})
    write_deploy_config(deploy / "deploy_config.json", meta)
    summary = {"best_epoch": best_ep, "val_map50": map50, "val_map75": map95, "fppi": float(fppi),
               "thresholds": thresholds, "calibration": cstats,
               "per_class_ap50": {n: (None if np.isnan(v) else round(float(v), 4))
                                  for n, v in zip(dcfg["names"], aps50)},
               "params_M": round(params / 1e6, 2), "flops_G": round(flops / 1e9, 2),
               "input_size": size, "backbone_pretrain": src,
               "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] 完成。部署文件: {deploy}/ (best.pt / best.onnx / best.torchscript.pt / deploy_config.json)\n"
          f"        推理: python -m fabricdet.infer --ckpt {out / 'best.pt'} --source <图片/目录>\n"
          f"        summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
