#!/usr/bin/env python
"""图级分数探针: 为什么 imageAUROC < 0.5 (正常图分数反而更高)?

不重训, 直接用已有 checkpoint 跑一遍验证集, 逐图打印各种图像级聚合分数, 并对比
"疵点图组" 与 "正常图组" 的分布, 定位到底是:
  1. 聚合口径不对 (top-k 像素数太大, 把小疵点的峰值稀释了);
  2. 布面掩码把疵点裁掉了 (score * cloth 之后疵点归零);
  3. 两组图本身分布不同 (验证集正常图与训练集不是同一种布).

用法::

    python tools/diagnose_scores.py --checkpoint runs/fabric19_fabricncd_s/best.pt
    python tools/diagnose_scores.py --checkpoint best.pt --csv scores.csv
"""

from __future__ import annotations

import argparse
import csv
import copy
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from fabric_ncd.data.yolo import build_dataloaders, build_normal_loader  # noqa: E402
from fabric_ncd.metrics import auroc  # noqa: E402
from fabric_ncd.models.fabric_ncd import FabricNCD  # noqa: E402
from fabric_ncd.ops import topk_mean  # noqa: E402
from fabric_ncd.utils import get_logger, load_yaml, resolve_device  # noqa: E402

LOGGER = get_logger("diagnose")


def _load(ckpt: str, device: torch.device, cfg_override: Dict) -> tuple:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(state.get("config") or load_yaml("configs/fabric19.yaml"))
    LOGGER.info("checkpoint: %s (epoch %s)", ckpt, state.get("epoch", "?"))
    cfg.update(cfg_override or {})
    model = FabricNCD(cfg)
    key = "ema" if "ema" in state else "model"
    model.load_state_dict(state[key])
    calibration = state.get("calibration") or {}
    scale = float(calibration.get("memory_distance_scale", 0.0) or 0.0)
    if scale > 0:
        model.memory_distance_scale = scale
    return model.to(device).eval(), cfg, calibration


def main() -> int:
    ap = argparse.ArgumentParser(description="图级分数诊断: 查 imageAUROC < 0.5 的原因")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples", type=int, default=0, help="每个子集最多多少张 (0 = 全部)")
    ap.add_argument("--memory-batches", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default="", help="把逐图明细写到 csv")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg, calibration = _load(args.checkpoint, device, {})
    datasets = build_dataloaders(cfg)
    loader = datasets["val"]

    normal_loader = build_normal_loader(cfg, "train")
    if normal_loader is not None:
        model.memory.reset()
        with torch.no_grad():
            for i, batch in enumerate(normal_loader):
                model.update_memory(
                    batch["image"].to(device), mask=batch["cloth_mask"].to(device)
                )
                if i + 1 >= args.memory_batches:
                    break

    rows: List[Dict] = []
    topk = int(cfg.get("eval", {}).get("topk_pixels", 256) or 256)
    gate = bool(cfg.get("eval", {}).get("cloth_gate", True))
    LOGGER.info("布面蒙版裁剪 (eval.cloth_gate): %s", "开" if gate else "关")
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["image"])
            h, w = batch["image"].shape[-2:]
            score = F.interpolate(out["anomaly_score"], size=(h, w), mode="bilinear", align_corners=False)
            cloth_p = F.interpolate(out["cloth_prob"], size=(h, w), mode="bilinear", align_corners=False)
            cloth_bin = (cloth_p > 0.5).float()
            masked = score * cloth_bin if gate else score
            learned = F.interpolate(out["learned_score"], size=(h, w), mode="bilinear", align_corners=False)
            for i in range(score.shape[0]):
                normal = bool(batch["normal"][i].item())
                raw = score[i, 0].detach().cpu().numpy()
                mkd = masked[i, 0].detach().cpu().numpy()
                cl = cloth_bin[i, 0].detach().cpu().numpy()
                lr = learned[i, 0].detach().cpu().numpy()
                gt_boxes = batch["boxes"][i].detach().cpu().numpy()
                rows.append(
                    {
                        "path": Path(batch["path"][i]).name,
                        "defect": 0 if normal else 1,
                        "n_gt": int(gt_boxes.shape[0]) if gt_boxes.ndim > 1 else 0,
                        "topk_gated": topk_mean(mkd, k_pixels=topk),
                        "topk_raw": topk_mean(raw, k_pixels=topk),
                        "top16_raw": topk_mean(raw, k_pixels=16),
                        "max_raw": float(raw.max()),
                        "mean1pct_raw": float(np.sort(raw.ravel())[-max(1, raw.size // 100):].mean()),
                        "topk_minus_med": float(topk_mean(raw, k_pixels=topk) - np.median(raw)),
                        "max_minus_med": float(raw.max() - np.median(raw)),
                        "median": float(np.median(raw)),
                        "cloth_cov": float(cl.mean()),
                        "learned_max": float(lr.max()),
                    }
                )
            if args.samples and len(rows) >= 2 * args.samples:
                break

    labels = np.array([r["defect"] for r in rows])
    LOGGER.info("验证集样本: 疵点 %d 张 / 正常 %d 张", int((labels == 1).sum()), int((labels == 0).sum()))

    keys = [
        "topk_gated", "topk_raw", "top16_raw", "max_raw", "mean1pct_raw",
        "topk_minus_med", "max_minus_med", "median", "learned_max", "cloth_cov",
    ]
    LOGGER.info("%-15s %-10s %-10s %-10s %-10s", "聚合方式", "疵点图均值", "正常图均值", "AUROC", "组内标准差")
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        d_mean = float(vals[labels == 1].mean()) if (labels == 1).any() else float("nan")
        n_mean = float(vals[labels == 0].mean()) if (labels == 0).any() else float("nan")
        score = auroc(labels, vals) if key != "cloth_cov" else float("nan")
        spread = float(vals.std()) if key != "cloth_cov" else float("nan")
        LOGGER.info("%-15s %-10.4f %-10.4f %-10.4f %-10.4f", key, d_mean, n_mean, score, spread)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("逐图明细已写入 %s", args.csv)

    # ------- 自动判读 -------
    scored = [k for k in keys if k not in ("cloth_cov", "median")]
    best = max(scored, key=lambda k: auroc(labels, np.array([r[k] for r in rows], dtype=np.float64)))
    best_auc = auroc(labels, np.array([r[best] for r in rows], dtype=np.float64))
    LOGGER.info("最佳聚合方式: %s (AUROC %.4f)", best, best_auc)
    if best_auc >= 0.90:
        LOGGER.info("判读: 该 checkpoint 上所有口径都分得开, 无需处理 (换 last.pt 再看当前轮次)")
        return 0
    auc_gated = auroc(labels, np.array([r["topk_gated"] for r in rows]))
    auc_raw = auroc(labels, np.array([r["topk_raw"] for r in rows]))
    auc_contrast = auroc(labels, np.array([r["topk_minus_med"] for r in rows]))
    if auc_contrast - auc_raw > 0.10:
        LOGGER.info(
            "判读: 绝对分数被逐图底色偏移带偏 (topk %.3f -> 去中位数 %.3f) -> 设 eval.image_score_mode: contrast",
            auc_raw, auc_contrast,
        )
    elif auc_raw - auc_gated > 0.10:
        LOGGER.info(
            "判读: 布面蒙版在裁掉疵点 (关闭 %s -> %s) -> 把 eval.cloth_gate 设为 false",
            f"{auc_gated:.3f}", f"{auc_raw:.3f}",
        )
    elif best in ("top16_raw", "max_raw") and best_auc > 0.75 and auc_raw < 0.7:
        LOGGER.info(
            "判读: top-k(%d 像素) 把小疵点峰值稀释了 -> 把 eval.topk_pixels 调小 (试试 16~64)", topk,
        )
    elif auroc(labels, np.array([r["learned_max"] for r in rows])) > 0.75 and auc_raw < 0.7:
        LOGGER.info("判读: 学习式异常头本身能分开, 融合后的分数反而被记忆项拖累 -> 调 model.memory.score_weight")
    else:
        LOGGER.info(
            "判读: 各口径都分不开 -> 先确认这是不是早期 checkpoint; 若确认是多轮之后的权重, "
            "则检查两组图是否同源 (正常验证图与训练正常图是否同种布)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
