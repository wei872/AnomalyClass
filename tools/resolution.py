#!/usr/bin/env python
"""按你的原始图像尺寸推荐网络输入分辨率.

背景: 布匹原图常见 1600(宽) x 846(高), 长宽比 1.89:1. 网络输入必须保持长宽比,
否则经纬方向被拉伸成正方形, 织造周期估计失效. 另一个约束是"缩放后织造周期
还剩几个像素": 周期 < 3 px 时 Gabor/FFT 分支基本看不到经纬结构.

用法::

    python tools/resolution.py --source D:/tyhn/yolo/dataset/source5_augmenteds --native 846x1600
    python tools/resolution.py --source a.jpg --native 846x1600 --samples 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fabric_ncd.data.weave import estimate_period  # noqa: E402
from fabric_ncd.data.yolo import list_images, load_rgb  # noqa: E402

# 候选输入尺寸 (H, W): 保持 1600:846 ≈ 1.89 比例, 宽高均为 32 的倍数
CANDIDATES: List[Tuple[int, int, str]] = [
    (416, 800, "极速 / 纯 CPU"),
    (512, 960, "低配"),
    (640, 1216, "轻量"),
    (704, 1344, "平衡"),
    (832, 1600, "默认 = 原图 (不缩放)"),
    (1024, 1952, "超采样 (小疵点)"),
]

BASELINE_MACS = 0.618  # 512x512 实测
BASELINE_PIXELS = 512 * 512
MEM_PER_MPX = 0.79  # GB / 百万像素 (训练含反向, 实测 1.40 GB @ 1.77 Mpx)


def parse_size(text: str) -> Tuple[int, int]:
    for sep in ("x", "*", ",", "X"):
        if sep in text:
            a, b = text.split(sep)[:2]
            return int(a), int(b)
    value = int(text)
    return value, value


def main() -> int:
    parser = argparse.ArgumentParser(description="推荐输入分辨率")
    parser.add_argument("--source", required=True, help="图片或目录 (最好是正常布面图)")
    parser.add_argument("--native", default="846x1600", help="原始尺寸, 形如 846x1600 (高x宽)")
    parser.add_argument("--samples", type=int, default=20, help="最多采样多少张")
    parser.add_argument("--target-period", type=float, default=4.0, help="缩放后期望保留的周期像素")
    parser.add_argument("--defect-px", type=float, default=None,
                        help="最小疵点在原图上的边长 (像素), 例如 6; 要求缩放后 >= 3 px")
    args = parser.parse_args()

    native = parse_size(args.native)
    if min(native) <= 0:
        parser.error("--native 尺寸必须为正")
    src = Path(args.source)
    files = list_images(src) if src.is_dir() else ([src] if src.exists() else [])
    if not files:
        print(f"未找到图片: {args.source}", file=sys.stderr)
        return 1

    rows = []
    for path in files[: max(1, args.samples)]:
        rgb = load_rgb(path)
        px, py, conf = estimate_period(rgb)
        if conf < 2.0:  # 无纹理/纯色, 结果不可信
            continue
        rows.append((px, py, conf))
    if not rows:
        print("所有样本的织造周期置信度都 < 2.0, 无法给出建议 (检查是否拍到布面)。")
        return 1

    arr = np.asarray(rows, dtype=np.float64)
    px_med = float(np.median(arr[:, 0]))
    py_med = float(np.median(arr[:, 1]))
    conf_med = float(np.median(arr[:, 2]))
    period_native = float(np.sqrt(max(px_med * py_med, 1e-6)))

    print(f"样本数 {len(rows)} (跳过低置信度 {len(files[: args.samples]) - len(rows)})")
    print(f"原始分辨率下织造周期: 经向(水平) {px_med:.2f} px, 纬向(垂直) {py_med:.2f} px, "
          f"等效 {period_native:.2f} px, 置信度 {conf_med:.2f}")
    print()
    print(f"{'输入 HxW':>12} {'缩放':>6} {'缩放后周期':>10} {'GMACs':>7} {'显存/张':>8}  说明")
    best = None
    for (reference_h, _, note) in CANDIDATES:
        # 以原 846 高参考图的候选缩放档位为准，按输入原图长宽比重建候选尺寸。
        nominal = reference_h / 846.0
        h = max(64, int(round(native[0] * nominal / 32.0)) * 32)
        w = max(64, int(round(native[1] * nominal / 32.0)) * 32)
        scale_y, scale_x = h / native[0], w / native[1]
        scale = (scale_x + scale_y) / 2.0
        period = float(np.sqrt(max(px_med * scale_x * py_med * scale_y, 1e-6)))
        macs = BASELINE_MACS * (h * w) / BASELINE_PIXELS
        mem = MEM_PER_MPX * (h * w) / 1e6
        flag = ""
        ok_defect = True if args.defect_px is None else (args.defect_px * scale >= 3.0)
        if period >= args.target_period and ok_defect:
            if best is None:
                best = (h, w, scale, period)
        elif period < 3.0:
            flag = "  <-- 周期不足 3 px, 经纬结构会糊掉"
        print(f"{h:>5}x{w:<6} {scale:>6.2f} {period:>9.2f}  {macs:>6.2f} {mem:>7.2f}GB  {note}{flag}")

    print()
    if best is not None:
        h, w, scale, period = best
        print(f"建议: data.image_size: [{h}, {w}]  (缩放 {scale:.2f}, 周期保留 {period:.2f} px)")
        print(f"      训练显存约 {MEM_PER_MPX * (h * w) / 1e6:.1f} GB/张, batch_size 按显存折算")
        print("      这是'够用'的最小尺寸; 若仍漏检小疵点, 往上调一档 (显存与耗时近似按面积增长)")
    else:
        print(f"警告: 所有候选尺寸都未满足周期或最小疵点约束 (原始等效周期 {period_native:.2f} px)。")
        print("      可尝试更高输入分辨率、切块，或用更高倍率镜头重新采集。")
    if min(px_med, py_med) > 0 and abs(px_med - py_med) / min(px_med, py_med) > 0.6:
        print(f"提示: 经纬周期差异较大 ({px_med:.1f} vs {py_med:.1f} px), 属于正常织物各向异性, "
              "模型已用 texture 分支显式建模, 无需额外处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
