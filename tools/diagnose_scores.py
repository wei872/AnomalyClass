#!/usr/bin/env python
"""图级分数探针: imageAUROC < 0.5 到底是哪里错了?

不重训, 直接用已有 checkpoint 跑一遍验证集, 逐图算各种图像级聚合分数, 然后回答
**分数在跟踪什么**:

  1. 聚合口径 (top-k 稀释 / 峰值 vs 均值);
  2. 布面掩码把疵点裁掉 (score * cloth);
  3. **采集域混淆**: 疵点图和正常图来自不同的布料批次/机台/曝光, 分数只是在区分"来源";
  4. 分数跟着**亮度/清晰度**走 (等于一个曝光检测器).

第 3、4 条是本工具的重点: 按文件名前缀 (批次号) 分组, 给出

* 每个批次内部的 AUROC (同源内还分得开吗);
* 跨批次的 AUROC (只比 "A 批的正常图" vs "B 批的疵点图");
* 训练正常图的批次覆盖 vs 验证集的批次覆盖 (三个集合的交集决定这个指标能不能用);
* 各聚合分数与图像外观统计 (亮度/对比度/饱和度/清晰度) 的 Spearman 秩相关.

用法::

    # 完整探针 (需要 GPU/CPU 跑一遍验证集, 记忆库默认用**全部**训练正常图)
    python tools/diagnose_scores.py --checkpoint runs/fabric19_fabricncd_s/last.pt --csv scores.csv

    # 只分析已经导出的 csv (秒级, 不用加载模型和数据集)
    python tools/diagnose_scores.py --from-csv scores_last.csv
"""

from __future__ import annotations

import copy
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fabric_ncd.metrics import auroc, spearman  # noqa: E402
from fabric_ncd.utils import get_logger  # noqa: E402

LOGGER = get_logger("diagnose")

# 逐图统计的所有口径. topk_gated 需要布面掩码; 其余全是 raw / 对比度口径.
KEYS = [
    "topk_gated", "topk_raw", "top16_raw", "max_raw", "mean1pct_raw",
    "topk_minus_med", "max_minus_med", "median", "learned_max", "cloth_cov",
]
# 外观统计 (越大越亮/越平/越花/越锐)
STATS = ["brightness", "contrast", "saturation", "sharpness"]


# --------------------------------------------------------------------------- #
# 分组 / 外观统计
# --------------------------------------------------------------------------- #
def source_of(name: str, depth: int = 1) -> str:
    """取文件名前缀作为"批次号": DA9670287_1777190506830_毛羽_0_orig.png -> DA9670287."""
    stem = Path(str(name)).stem
    parts = [p for p in stem.split("_") if p]
    if not parts:
        return Path(str(name)).stem
    return "_".join(parts[: max(1, depth)])


def appearance_stats(path: str) -> Dict[str, float]:
    """原图的亮度/对比度/饱和度/清晰度 (缩小到长边 256 后算, 快且稳定)."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL 一定在依赖里
        return {k: float("nan") for k in STATS}
    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return {k: float("nan") for k in STATS}
    gray = arr.mean(axis=2)
    colorfulness = float(np.abs(arr - gray[..., None]).mean())
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float(np.var(gx) + np.var(gy))
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": colorfulness,
        "sharpness": sharpness,
    }


# --------------------------------------------------------------------------- #
# 分析
# --------------------------------------------------------------------------- #
def _pair_auc(rows: List[Dict], key: str, same_src: bool) -> Tuple[float, int]:
    """同源内 / 跨源的 AUROC, 都按"正负样本对"直接统计 (与类别构成无关).

    ``same_src=True`` 只用**同一批次**里的 (疵点, 正常) 对: 去掉采集差异后还分得开吗?
    ``same_src=False`` 只用**不同批次**之间的对: 分数把批次差异本身也吃进去了多少?
    """
    pos = [(float(r[key]), r["src"]) for r in rows if r["defect"] == 1]
    neg = [(float(r[key]), r["src"]) for r in rows if r["defect"] == 0]
    num = 0.0
    den = 0
    for pv, ps in pos:
        for nv, ns in neg:
            if (ps == ns) != same_src:
                continue
            den += 1
            if pv > nv:
                num += 1.0
            elif pv == nv:
                num += 0.5
    return ((num / den) if den else float("nan")), den


def _bank_report(bank_src: Dict[str, int], bank_images: int, total_normals: int,
                 val_src: Dict[str, int]) -> None:
    """记忆库到底覆盖了哪些批次 —— 只取前若干批 + 不洗牌时, 这里会非常偏.

    ``list_images`` 是**按文件名排序**的, 所以"前 N 批"= 字典序最前面的那些文件
    (例如所有 ``DA...`` 排在所有 ``KBA...`` 之前). 一旦记忆库只装了一种批次,
    另一种批次的正常图在库里找不到近邻, 记忆分支分数就会整片虚高 ->
    imageAUROC 反向, 而 pixelAUROC 看起来还正常.
    """
    if not bank_src:
        return
    LOGGER.info(
        "记忆库构成: %d 张 / %d 个来源 (训练正常图共 %d 张)",
        bank_images, len(bank_src), total_normals,
    )
    top = sorted(bank_src.items(), key=lambda kv: -kv[1])[:5]
    LOGGER.info("  最多的几个来源: %s", " | ".join(f"{k} {v} 张" for k, v in top))
    if val_src:
        missing = sorted(set(val_src) - set(bank_src))
        need = [s for s in missing if val_src.get(s)]
        LOGGER.info(
            "  验证集的 %d 个来源里有 %d 个**完全不在记忆库里**%s",
            len(val_src), len(need),
            (": " + ", ".join(need[:8]) + (" ..." if len(need) > 8 else "")) if need else "",
        )
        if len(need) >= max(1, len(val_src) // 2):
            LOGGER.info(
                "  -> 记忆库严重偏科: 这些批次的正常图会被当作'没见过'而分数虚高. "
                "用 --memory-batches 0 (全部正常图) 复测, 训练侧用 train.memory_shuffle: true"
            )


def _group_table(rows: List[Dict], key: str) -> None:
    by_src: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)
    LOGGER.info(
        "%-16s %6s %6s %10s %10s %10s %10s",
        "来源(批次)", "疵点", "正常", "疵点均值", "正常均值", "组内AUROC", "布面覆盖",
    )
    for src in sorted(by_src):
        sub = by_src[src]
        lab = np.array([r["defect"] for r in sub])
        vals = np.array([r[key] for r in sub], dtype=np.float64)
        d = float(vals[lab == 1].mean()) if (lab == 1).any() else float("nan")
        n = float(vals[lab == 0].mean()) if (lab == 0).any() else float("nan")
        auc = auroc(lab, vals) if lab.min() != lab.max() else float("nan")
        cov = np.array([r.get("cloth_cov", np.nan) for r in sub], dtype=np.float64)
        cov_m = float(np.nanmean(cov)) if np.isfinite(cov).any() else float("nan")
        LOGGER.info("%-16s %6d %6d %10.4f %10.4f %10.4f %10.4f",
                    src, int((lab == 1).sum()), int((lab == 0).sum()), d, n, auc, cov_m)


def _coverage(train_src: Dict[str, int], val_src: Dict[str, int]) -> None:
    """训练正常图的批次 vs 验证集的批次, 直接决定 imageAUROC 能不能用."""
    t, v = set(train_src), set(val_src)
    LOGGER.info(
        "批次覆盖: 训练正常图 %d 个批次 / 验证集 %d 个批次 / 交集 %d",
        len(t), len(v), len(t & v),
    )
    only_val = sorted(v - t)
    if only_val:
        LOGGER.info(
            "  验证集有 %d 个批次在训练正常图里**从未出现**: %s",
            len(only_val), ", ".join(only_val[:8]) + (" ..." if len(only_val) > 8 else ""),
        )
    only_train = sorted(t - v)
    if only_train:
        LOGGER.info(
            "  训练正常图有 %d 个批次不在验证集里: %s",
            len(only_train), ", ".join(only_train[:8]) + (" ..." if len(only_train) > 8 else ""),
        )


def analyze(rows: List[Dict], topk: int = 256, gate: bool = True) -> int:
    labels = np.array([r["defect"] for r in rows])
    keys = [k for k in KEYS if k != "cloth_cov" and k in rows[0]]
    LOGGER.info("样本: 疵点 %d 张 / 正常 %d 张 / 来源 %d 个",
                int((labels == 1).sum()), int((labels == 0).sum()),
                len({r["src"] for r in rows}))

    LOGGER.info("%-15s %-10s %-10s %-10s %-10s %-10s", "聚合方式", "疵点均值", "正常均值", "AUROC", "同源内", "跨源")
    table: Dict[str, Tuple[float, float, float, float]] = {}
    within_pairs = 0
    cross_pairs = 0
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        d_mean = float(vals[labels == 1].mean()) if (labels == 1).any() else float("nan")
        n_mean = float(vals[labels == 0].mean()) if (labels == 0).any() else float("nan")
        auc = auroc(labels, vals)
        within, w_pairs = _pair_auc(rows, key, same_src=True)
        cross, c_pairs = _pair_auc(rows, key, same_src=False)
        within_pairs = max(within_pairs, w_pairs)
        cross_pairs = cross_pairs or c_pairs
        table[key] = (auc, within, cross)
        LOGGER.info("%-15s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f", key, d_mean, n_mean, auc, within, cross)

    if "cloth_cov" in rows[0]:
        low = sorted({r["src"] for r in rows if r.get("cloth_cov", 1.0) < 0.3})
        if low:
            LOGGER.info(
                "注意: %d 个批次的平均布面覆盖 < 0.3 (布面掩码认为这些图几乎没有布): %s "
                "-> 这些图要么不是布, 要么曝光/尺度与训练不同源",
                len(low), ", ".join(low[:6]) + (" ..." if len(low) > 6 else ""),
            )
    if within_pairs == 0:
        LOGGER.info(
            "提示: 没有任何一个批次同时含疵点图和正常图 -> 同源内验证做不了, "
            "imgAUROC 里混着'批次差异'和'疵点差异'两种信号, 无法区分"
        )
    if "brightness" in rows[0]:
        LOGGER.info("分数与图像外观的秩相关 (Spearman, |rho|>0.5 说明分数在跟踪外观):")
        for key in ("topk_raw", "learned_max", "topk_minus_med"):
            if key not in rows[0]:
                continue
            sv = np.array([r[key] for r in rows], dtype=np.float64)
            parts = []
            for stat in STATS:
                uv = np.array([r.get(stat, np.nan) for r in rows], dtype=np.float64)
                ok = np.isfinite(uv) & np.isfinite(sv)
                parts.append(f"{stat} {spearman(uv[ok], sv[ok]):+.2f}" if ok.sum() > 2 else f"{stat} n/a")
            LOGGER.info("  %-14s %s", key, " | ".join(parts))

    scored = [k for k in keys if k != "median"]
    if not scored:
        return 0
    best = max(scored, key=lambda k: table[k][0] if np.isfinite(table[k][0]) else -1)
    best_auc, best_within, best_cross = table[best]
    LOGGER.info("最佳聚合方式: %s (AUROC %.4f)", best, best_auc)

    # ---------------- 自动判读 ----------------
    if np.isfinite(best_auc) and best_auc >= 0.90:
        LOGGER.info("判读: 该 checkpoint 上已经分得开, 无需处理")
        return 0

    gate_on = bool(gate)
    auc_gated = table["topk_gated"][0] if "topk_gated" in table else float("nan")
    auc_raw = table["topk_raw"][0]
    auc_contrast = table["topk_minus_med"][0] if "topk_minus_med" in table else np.nan
    within_best = max((v[1] for v in table.values() if np.isfinite(v[1])), default=np.nan)
    cross_best = max((v[2] for v in table.values() if np.isfinite(v[2])), default=np.nan)
    bright_rho = 0.0
    if "brightness" in rows[0]:
        sv = np.array([r["topk_raw"] for r in rows], dtype=np.float64)
        bv = np.array([r["brightness"] for r in rows], dtype=np.float64)
        ok = np.isfinite(bv) & np.isfinite(sv)
        if ok.sum() > 2:
            bright_rho = abs(spearman(bv[ok], sv[ok]))

    if np.isfinite(within_best) and within_best >= 0.70 and (not np.isfinite(cross_best) or within_best > cross_best + 0.15):
        LOGGER.info(
            "判读: 同源内 AUROC %.3f 明显高于跨源 %.3f -> 分数是好的, 指标被跨批次的采集差异拖垮; "
            "报告指标时按批次分组, 或把验证集限制在同一批次", within_best, cross_best,
        )
    elif (np.isfinite(cross_best) and abs(cross_best - 0.5) >= 0.25
          and (within_pairs == 0 or (np.isfinite(within_best) and within_best < 0.70))):
        LOGGER.info(
            "判读: 同源内只分到 %.3f, 而跨源能分到 %.3f -> **分数在区分布料批次/采集条件, 不是在区分疵点**. "
            "验证集的正常图与疵点图来自不同批次, 这个 imgAUROC 不可用: 见上面的 '批次覆盖' 一行, "
            "缺的批次要进训练正常图 (normal/images/train), 或者按批次重新切分 train/val", within_best, cross_best,
        )
    elif bright_rho >= 0.5:
        LOGGER.info(
            "判读: 分数与整图亮度秩相关 %.2f -> 模型在测曝光, 不在测疵点; 先把正常图按批次补齐 (含亮/暗两种工况)", bright_rho,
        )
    elif np.isfinite(auc_contrast) and auc_contrast - auc_raw > 0.10:
        LOGGER.info(
            "判读: 绝对分数被逐图底色偏移带偏 (topk %.3f -> 去中位数 %.3f) -> 设 eval.image_score_mode: contrast",
            auc_raw, auc_contrast,
        )
    elif gate_on and np.isfinite(auc_raw) and np.isfinite(auc_gated) and auc_raw - auc_gated > 0.10:
        LOGGER.info(
            "判读: 布面蒙版在裁掉疵点 (关闭 %.3f -> %.3f) -> 把 eval.cloth_gate 设为 false", auc_gated, auc_raw,
        )
    elif best in ("top16_raw", "max_raw") and best_auc > 0.75 and auc_raw < 0.7:
        LOGGER.info("判读: top-k(%d 像素) 把小疵点峰值稀释了 -> 把 eval.topk_pixels 调小 (试试 16~64)", topk)
    elif np.isfinite(table["learned_max"][0]) and table["learned_max"][0] > 0.75 and auc_raw < 0.7:
        LOGGER.info("判读: 学习式异常头本身能分开, 融合后的分数被记忆项拖累 -> 调 model.memory.score_weight")
    else:
        LOGGER.info(
            "判读: 各口径都分不开 -> 先确认 checkpoint 的轮次 (早期权重没有判据价值); "
            "确认是多轮之后仍如此, 就按上面的批次覆盖表检查训练正常图是否覆盖了验证集的全部批次"
        )
    return 0


# --------------------------------------------------------------------------- #
# 两种入口
# --------------------------------------------------------------------------- #
def _write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("逐图明细已写入 %s", path)


def _read_csv(path: str, depth: int) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row: Dict = {"path": raw.get("path", ""), "src": source_of(raw.get("path", ""), depth)}
            for key, val in raw.items():
                if key in ("path", "src"):
                    continue
                try:
                    row[key] = float(val)
                except (TypeError, ValueError):
                    row[key] = val
            row["defect"] = int(float(raw.get("defect", 0) or 0))
            rows.append(row)
    return rows


def _dataset_report(args) -> int:
    """不加载模型, 只审数据集目录: 各 split 的批次构成 / 排序后的前 N 张是什么 / 覆盖率.

    用来回答两个问题:
      1. 验证集的正常图, 在 ``normal/images/train`` 里存在吗?
      2. ``memory_warmup_batches`` 不洗牌时取的"前 N 批", 里面都是哪些批次?
    """
    from fabric_ncd.data.yolo import list_images, load_dataset_spec
    from tools._common import load_config

    cfg = load_config(args.config, args.overrides, root=args.root)
    spec = load_dataset_spec(cfg)

    dirs = {
        "images/train (疵点)": spec.splits.get("train"),
        "images/val (疵点)": spec.splits.get("val"),
        "normal/train (正常)": spec.normal_splits.get("train"),
        "normal/val (正常)": spec.normal_splits.get("val"),
    }
    hist: Dict[str, Dict[str, int]] = {}
    for title, directory in dirs.items():
        files = list_images(directory)
        counts: Dict[str, int] = defaultdict(int)
        for f in files:
            counts[source_of(f.name, args.group_depth)] += 1
        hist[title] = dict(counts)
        LOGGER.info("%-22s %5d 张 / %3d 个来源", title, len(files), len(counts))
        if counts:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
            LOGGER.info("    最多: %s", " | ".join(f"{k} {v}" for k, v in top))

    normal_train = hist.get("normal/train (正常)", {})
    normal_val = hist.get("normal/val (正常)", {})
    if normal_val:
        need = sorted(set(normal_val) - set(normal_train))
        LOGGER.info(
            "验证集正常图: %d 个来源, 其中 %d 个不在 normal/images/train 里%s",
            len(normal_val), len(need),
            (": " + ", ".join(need) + "") if need else "",
        )
        if need:
            LOGGER.info(
                "  -> 这些批次对模型来说从没当过'正常', 在它们上面算 imgAUROC 没有意义; "
                "把它们补进 normal/images/train (它们本来就是正常图)"
            )
    # ---- 缺陷图与正常图是否来自**同一批**布料 (决定 imgAUROC 有没有意义) ----
    defect_src = set(hist.get("images/train (疵点)", {})) | set(hist.get("images/val (疵点)", {}))
    normal_src = set(hist.get("normal/train (正常)", {})) | set(hist.get("normal/val (正常)", {}))
    inter = defect_src & normal_src
    LOGGER.info(
        "缺陷来源 %d 个 / 正常来源 %d 个 / 交集 %d 个",
        len(defect_src), len(normal_src), len(inter),
    )
    if not inter:
        LOGGER.info(
            "  -> **致命**: 缺陷图和正常图完全来自不同的布料批次. 模型只要学会"
            "'这是 A 批还是 B 批'就能把训练集分对, 而这个能力与\"有没有疵点\"无关; "
            "任何跨类的 imageAUROC / 检出率都会被这个捷径污染."
        )
        LOGGER.info(
            "  -> 唯一可用的是**同批次内**的指标 (同一批里既有疵点图也有正常图). "
            "请把与缺陷图同批次的正常图补进 normal/images/train 与 normal/images/val"
        )
    elif len(inter) < len(defect_src) * 0.5:
        LOGGER.info("  -> 交集偏少: 多数缺陷批次没有同批正常图, 跨类指标仍然不可信")

    # ---- 布面掩码估计: 训练时 anomaly 损失的取样区域 ----
    if args.mask_check > 0:
        _mask_report(dirs, args.mask_check, args.group_depth)

    # 不洗牌时 "前 N 批" 的构成: 文件按名字排序, 这里按同样顺序取前 48 张
    n_first = 48
    for title, directory in dirs.items():
        if "normal" not in title or not directory:
            continue
        files = list_images(directory)[:n_first]
        counts: Dict[str, int] = defaultdict(int)
        for f in files:
            counts[source_of(f.name, args.group_depth)] += 1
        if counts:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
            LOGGER.info(
                "%s 排序后的前 %d 张 (bs2 时 = memory_warmup_batches 24 实际抽到的), 共 %d 个来源: %s%s",
                title, len(files), len(counts),
                " | ".join(f"{k} {v}" for k, v in top),
                " ..." if len(counts) > len(top) else "",
            )
    return 0


def _mask_report(dirs: Dict[str, object], per_src: int, depth: int) -> None:
    """无监督布面掩码估计的覆盖率 = anomaly 损失真正取样/布面头拿到的弱监督.

    如果某批正常图的估计覆盖率只有几个百分点, 那么这批图在异常损失里几乎不参与
    (``region = cloth_mask > 0.5``), 异常头在这批外观上等于没有负样本约束 ->
    分数会在这些图上失控. 这一行直接量这件事.
    """
    from fabric_ncd.data.fabric_mask import estimate_cloth_mask
    from fabric_ncd.data.yolo import list_images, load_rgb

    LOGGER.info("布面掩码估计覆盖率 (每批最多 %d 张):", per_src)
    for title, directory in dirs.items():
        if not directory:
            continue
        per_group: Dict[str, List[float]] = defaultdict(list)
        for f in list_images(directory):
            key = source_of(f.name, depth)
            if len(per_group[key]) >= per_src:
                continue
            try:
                image = load_rgb(f)
                mask, conf = estimate_cloth_mask(image, work_size="auto")
                per_group[key].append(float(mask.mean()))
            except Exception as exc:  # pragma: no cover
                LOGGER.info("  %s %s: 估计失败 (%s)", title, f.name, exc)
        if not per_group:
            continue
        flat = [v for vs in per_group.values() for v in vs]
        n_supervised = float(np.mean(flat))
        LOGGER.info("  %-22s 平均 %.3f | 最低 %.3f | 最低的批次: %s", title,
                    n_supervised, float(np.min(flat)),
                    ", ".join(f"{k} {np.mean(v):.2f}" for k, v in
                              sorted(per_group.items(), key=lambda kv: np.mean(kv[1]))[:4]))
    LOGGER.info(
        "  -> 覆盖率就是 anomaly 损失真正监督到的像素比例. 整幅都是布时它应该 ~1.0; "
        "明显偏低就设 data.cloth_mask: full (否则其余像素的异常头输出完全不受约束)"
    )


def _probe(args) -> int:
    import torch  # 只有完整探针才需要
    from torch.nn import functional as F

    from fabric_ncd.data.yolo import (
        build_background_loader,
        build_dataloaders,
        build_normal_dataset,
        build_normal_loader,
    )
    from fabric_ncd.models.fabric_ncd import FabricNCD
    from fabric_ncd.ops import topk_mean
    from fabric_ncd.utils import load_yaml, resolve_device

    from tools._common import apply_overrides

    def _load(ckpt: str, device):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(state.get("config") or load_yaml("configs/fabric19.yaml"))
        LOGGER.info("配置来自 checkpoint 里保存的那一份 (训练时的配置), 可用 --set 覆盖")
        if args.overrides:
            cfg = apply_overrides(cfg, args.overrides)
        if args.root:
            cfg.setdefault("data", {})["root"] = args.root
        LOGGER.info("checkpoint: %s (epoch %s)", ckpt, state.get("epoch", "?"))
        model = FabricNCD(cfg)
        model.load_state_dict(state["ema" if "ema" in state else "model"])
        calibration = state.get("calibration") or {}
        scale = float(calibration.get("memory_distance_scale", 0.0) or 0.0)
        if scale > 0:
            model.memory_distance_scale = scale
        return model.to(device).eval(), cfg

    device = resolve_device(args.device)
    model, cfg = _load(args.checkpoint, device)
    loader = build_dataloaders(cfg)["val"]

    # ---- 批次覆盖: 训练正常图 vs 验证集 ----
    train_src: Dict[str, int] = {}
    train_ds = build_normal_dataset(cfg, "train")
    if train_ds is not None:
        for entry in train_ds.entries:
            key = source_of(entry.image.name, args.group_depth)
            train_src[key] = train_src.get(key, 0) + 1

    # 参照集必须和训练时一致 (train.memory_source): 否则探针里的记忆项和引擎里的不是
    # 同一个东西, 分数就不可比 —— 这是最容易把人带偏的地方.
    memory_source = str(cfg.get("train", {}).get("memory_source", "both")).lower()
    bank_loaders = []
    if "defect" in memory_source:
        bg = build_background_loader(cfg, "train", shuffle=False)
        if bg is not None:
            bank_loaders.append(("疵点图框外布面", bg))
    nl = build_normal_loader(cfg, "train")
    if nl is not None and ("normal" in memory_source or not bank_loaders):
        bank_loaders.append(("良品图", nl))
    bank_images = 0
    bank_src: Dict[str, int] = defaultdict(int)
    if bank_loaders:
        model.memory.reset()
        with torch.no_grad():
            for kind, bank_loader in bank_loaders:
                LOGGER.info("记忆库来源: %s (%d 张)", kind, len(bank_loader.dataset))
                for i, batch in enumerate(bank_loader):
                    model.update_memory(batch["image"].to(device), mask=batch["cloth_mask"].to(device))
                    bank_images += int(batch["image"].shape[0])
                    for path in batch["path"]:
                        key = source_of(path, args.group_depth)
                        bank_src[key] += 1
                    if args.memory_batches and i + 1 >= args.memory_batches:
                        break
                if args.memory_batches and bank_images >= args.memory_batches * int(
                    cfg.get("data", {}).get("eval_batch_size", 2)
                ):
                    break
        LOGGER.info("记忆库: 用 %d 张参照图建立", bank_images)

    topk = int(cfg.get("eval", {}).get("topk_pixels", 256) or 256)
    gate = bool(cfg.get("eval", {}).get("cloth_gate", True))
    LOGGER.info("布面蒙版裁剪 (eval.cloth_gate): %s", "开" if gate else "关")

    rows: List[Dict] = []
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
                path = str(batch["path"][i])
                raw = score[i, 0].detach().cpu().numpy()
                mkd = masked[i, 0].detach().cpu().numpy()
                cl = cloth_bin[i, 0].detach().cpu().numpy()
                lr = learned[i, 0].detach().cpu().numpy()
                gt_boxes = batch["boxes"][i].detach().cpu().numpy()
                row = {
                    "path": Path(path).name,
                    "src": source_of(path, args.group_depth),
                    "defect": 0 if bool(batch["normal"][i].item()) else 1,
                    "n_gt": int(gt_boxes.shape[0]) if gt_boxes.ndim > 1 else 0,
                    "topk_gated": topk_mean(mkd, k_pixels=topk),
                    "topk_raw": topk_mean(raw, k_pixels=topk),
                    "top16_raw": topk_mean(raw, k_pixels=16),
                    "max_raw": float(raw.max()),
                    "mean1pct_raw": float(np.sort(raw.ravel())[-max(1, raw.size // 100):].mean()),
                    "topk_minus_med": float(topk_mean(raw, k_pixels=topk) - np.median(raw)),
                    "max_minus_med": float(raw.max() - np.median(raw)),
                    "median": float(np.median(raw)),
                    "learned_max": float(lr.max()),
                    "cloth_cov": float(cl.mean()),
                }
                row.update(appearance_stats(path))
                rows.append(row)
            if args.samples and len(rows) >= 2 * args.samples:
                break

    val_src: Dict[str, int] = defaultdict(int)
    for r in rows:
        val_src[r["src"]] += 1
    _coverage(train_src, val_src)
    _bank_report(bank_src, bank_images, len(train_ds) if train_ds else 0, val_src)
    if args.csv:
        _write_csv(rows, args.csv)
    _group_table(rows, "topk_raw")
    return analyze(rows, topk=topk, gate=gate)


def main() -> int:
    from tools._common import base_parser

    ap = base_parser("图级分数诊断: 分数到底在跟踪什么")
    ap.add_argument("--checkpoint", default="", help="完整探针需要; 用 --from-csv 时可省")
    ap.add_argument("--from-csv", default="", help="只分析已导出的逐图 csv, 不加载模型")
    ap.add_argument("--samples", type=int, default=0, help="最多多少张 (0 = 全部)")
    ap.add_argument("--memory-batches", type=int, default=0,
                    help="建立记忆库用多少批正常图 (0 = 全部; 小于全部会让部分批次变成'没见过'而虚高)")
    ap.add_argument("--group-depth", type=int, default=1, help="按文件名前几段分组 (默认 1 = 批次号)")
    ap.add_argument("--dataset-only", action="store_true",
                    help="不加载模型, 只审数据集目录 (批次构成 / 前 N 张 / 覆盖率), 秒级")
    ap.add_argument("--mask-check", type=int, default=0,
                    help="配合 --dataset-only: 每个批次抽 N 张估计布面掩码覆盖率 (训练时 anomaly 损失的取样区域)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default="", help="把逐图明细写到 csv")
    args = ap.parse_args()

    if args.dataset_only:
        return _dataset_report(args)
    if args.from_csv:
        rows = _read_csv(args.from_csv, args.group_depth)
        if not rows:
            LOGGER.info("csv 是空的: %s", args.from_csv)
            return 1
        _group_table(rows, "topk_raw")
        return analyze(rows)
    if not args.checkpoint:
        ap.error("需要 --checkpoint 或 --from-csv")
    return _probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
