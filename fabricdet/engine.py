# -*- coding: utf-8 -*-
"""训练引擎: AMP/EMA/余弦调度/骨干分阶段解冻 + 阈值自动标定 + 部署级评估。"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from .metrics import evaluate_ap
from .postprocess import DEFAULT_THRESHOLDS, UNKNOWN_CLS, postprocess


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_into(self, model: torch.nn.Module):
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()})


class Scheduler:
    def __init__(self, opt, warmup_iters, total_iters, lr_min_ratio=0.01):
        self.opt, self.w, self.t, self.r = opt, warmup_iters, total_iters, lr_min_ratio
        self.base = [g["lr"] for g in opt.param_groups]

    def step(self, it):
        if it < self.w:
            f = (it + 1) / max(self.w, 1)
        else:
            p = (it - self.w) / max(self.t - self.w, 1)
            f = self.r + (1 - self.r) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * f


def train_one_epoch(model, loader, loss_fn, opt, sched, scaler, device, ema, epoch,
                    accum=1, use_amp=False, log_every=20):
    model.train()
    losses = {}
    t0 = time.time()
    opt.zero_grad()
    for it, (xs, ts, _) in enumerate(loader):
        xs = xs.to(device, non_blocking=True)
        ts = [t.to(device) for t in ts]
        if use_amp:
            with torch.amp.autocast("cuda"):
                obj, box, emb = model.forward_train(xs, ts)
                ld = loss_fn(model, obj, box, emb, ts)
            scaler.scale(ld["total"] / accum).backward()
        else:
            obj, box, emb = model.forward_train(xs, ts)
            ld = loss_fn(model, obj, box, emb, ts)
            (ld["total"] / accum).backward()
        if (it + 1) % accum == 0:
            if use_amp:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step(sched_it[0])
            sched_it[0] += 1
            if ema is not None:
                ema.update(model)
        for k, v in ld.items():
            losses[k] = losses.get(k, 0.0) + float(v) / max(len(loader), 1)
        if log_every and it % log_every == 0:
            print(f"  [ep{epoch} {it}/{len(loader)}] loss={ld['total']:.4f} "
                  f"obj={ld.get('obj', 0):.3f} cls={ld.get('cls', 0):.3f} "
                  f"giou={ld.get('giou', 0):.3f} sc={ld.get('supcon', 0):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return losses


sched_it = [0]


def reset_sched_counter():
    sched_it[0] = 0


@torch.no_grad()
def fast_eval(model, val_ds, device, conf=0.1, tau=0.35, max_images=10000):
    """整图 letterbox 快速 mAP50 (仅用于训练中挑 best)。"""
    model.eval()
    from torch.utils.data import DataLoader, SequentialSampler
    from .data import collate
    loader = DataLoader(val_ds, batch_size=4, sampler=SequentialSampler(val_ds),
                        collate_fn=collate, num_workers=0)
    gt_by, pred_by = {}, {}
    th = dict(DEFAULT_THRESHOLDS, conf=conf, tau_unknown=tau)
    for xs, ts, metas in loader:
        xs = xs.to(device)
        boxes, obj, sims = model.forward_deploy(xs)
        boxes, obj, sims = boxes.cpu().numpy(), obj.cpu().numpy(), sims.cpu().numpy()
        for b in range(xs.shape[0]):
            sc = metas[b]["scale"]
            preds = postprocess(boxes[b], obj[b], sims[b], th)
            for p in preds:
                p["box"] = (p["box"] / sc).astype(np.float32)
            gt = [(int(row[0]), row[1:5].numpy() / sc) for row in ts[b]]
            gt_by[metas[b]["path"]] = gt
            pred_by[metas[b]["path"]] = preds
    _, map50, extra = evaluate_ap(gt_by, pred_by, model.num_classes, 0.5)
    return map50, extra


@torch.no_grad()
def collect_val_predictions(model, items, device, tile, overlap, thresholds, batch=4):
    """items: [(path, gt[(cls, xyxy)])] -> {path: [pred]} (部署同款滑窗推理)"""
    from .infer import predict_with_model
    out = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        for path, _ in chunk:
            out[path] = predict_with_model(model, path, tile=tile, overlap=overlap,
                                           thresholds=thresholds, device=device)
    return out


def calibrate_thresholds(model, items, device, cfg) -> tuple[dict, dict]:
    """在 val 集上自动标定 conf 与 tau_unknown (开集阈值), 写进权重文件。"""
    permissive = dict(DEFAULT_THRESHOLDS, conf=0.03, tau_unknown=-1.0)
    preds = collect_val_predictions(model, items, device, cfg["tile"], cfg["overlap"], permissive,
                                    batch=1)
    tp_s, tp_sim, fp_s = [], [], []
    from .metrics import _iou_matrix
    for path, gts in items:
        gt_arr = np.array([g[1] for g in gts], dtype=np.float64).reshape(-1, 4)
        gt_cls = [g[0] for g in gts]
        used = np.zeros(len(gts), bool)
        for p in sorted(preds.get(path, []), key=lambda q: -q["score"]):
            if gt_arr.size == 0:
                fp_s.append(p["score"])
                continue
            cand = [i for i in range(len(gts)) if not used[i]]
            if not cand:
                fp_s.append(p["score"])
                continue
            ious = _iou_matrix(np.array(p["box"], dtype=np.float64)[None], gt_arr[cand])[0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                used[cand[j]] = True
                if gt_cls[cand[j]] == p["cls"]:
                    tp_s.append(p["score"])
                    tp_sim.append(p["max_sim"])
                else:
                    fp_s.append(p["score"])     # 定位对但分类错: 按 FP 处理以保守标定 conf
            else:
                fp_s.append(p["score"])
    tp_s, fp_s = np.array(tp_s), np.array(fp_s)
    if tp_s.size == 0:
        # val 上一个 TP 都没有 (训练未收敛/数据问题): 保留安全默认阈值, 不用脏标定
        th = dict(DEFAULT_THRESHOLDS)
        return th, {"n_tp": 0, "n_fp": int(fp_s.size), "calib_f1": 0.0,
                    "tp_sim_mean": None, "tp_sim_p2": None,
                    "warning": "val 无 TP, 阈值保留默认值, 请检查数据/训练轮数"}
    best_conf, best_f1 = 0.25, -1.0
    for thr in np.arange(0.05, 0.81, 0.025):
        tp = int((tp_s >= thr).sum())
        fp = int((fp_s >= thr).sum())
        n_gt = len(tp_s) + 0  # 召回以可检出 TP 为上限 (标定目的: 分数门限)
        rec = tp / max(n_gt, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        if f1 > best_f1:
            best_f1, best_conf = f1, float(thr)
    tp_sim = np.array(tp_sim)
    if tp_sim.size >= 8:
        tau = float(np.clip(np.percentile(tp_sim, 2.0), 0.10, 0.80))
    else:
        tau = 0.35
    th = dict(DEFAULT_THRESHOLDS, conf=round(best_conf, 3), tau_unknown=round(tau, 3))
    stats = {"n_tp": int(tp_s.size), "n_fp": int(fp_s.size), "calib_f1": round(float(best_f1), 3),
             "tp_sim_mean": round(float(tp_sim.mean()), 3) if tp_sim.size else None,
             "tp_sim_p2": round(float(tau), 3) if tp_sim.size else None}
    return th, stats


@torch.no_grad()
def final_eval(model, items, device, cfg, thresholds):
    preds = collect_val_predictions(model, items, device, cfg["tile"], cfg["overlap"], thresholds,
                                    batch=1)
    gt_by = {p: g for p, g in items}
    pred_by = {p: preds.get(p, []) for p, _ in items}
    aps50, map50, extra50 = evaluate_ap(gt_by, pred_by, model.num_classes, 0.5)
    aps95, map95, _ = evaluate_ap(gt_by, pred_by, model.num_classes, 0.75)
    n_img = max(len(items), 1)
    fppi = extra50["fp"].sum() / n_img
    return aps50, map50, map95, extra50, fppi
