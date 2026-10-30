#!/usr/bin/env python
"""实时性基准: 逐段耗时 + 流水线吞吐, 直接回答 "能不能 40 FPS".

示例::

    # 用一帧真实图 (推荐, 包含解码与预处理开销)
    python tools/rt_bench.py --ckpt runs/fabric19/best.pt --source line.png --fp16

    # 用若干张图循环跑 300 帧
    python tools/rt_bench.py --ckpt best.pt --source /data/fabric/val --frames 300 --fp16

    # 关闭流水线, 只看单帧串行延迟
    python tools/rt_bench.py --ckpt best.pt --source line.png --mode sequential
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import base_parser, load_predict_overrides  # noqa: E402

from fabric_ncd.inference import FabricPredictor  # noqa: E402
from fabric_ncd.realtime import RealtimePipeline, load_rgb_fast, summarize  # noqa: E402
from fabric_ncd.utils import json_dump  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect(source: str) -> List[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _parser():
    parser = base_parser("实时性基准 (逐段耗时 + 流水线 FPS)")
    parser.add_argument("--ckpt", default=None, help="checkpoint (best.pt)")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--source", required=True, help="图片或目录")
    parser.add_argument("--frames", type=int, default=100, help="总帧数")
    parser.add_argument("--warmup", type=int, default=10, help="预热帧数 (不计入统计)")
    parser.add_argument("--mode", default="pipeline", choices=["pipeline", "sequential"])
    parser.add_argument("--size", default=None, help="覆盖输入尺寸, 形如 832x1600 或 768 (HxW)")
    parser.add_argument("--out-stride", type=int, default=None, help="分数图下采样倍数 (默认取配置)")
    parser.add_argument("--memory-stride", type=int, default=None, help="记忆库查询池化倍数")
    parser.add_argument("--weave-interval", type=int, default=None, help="每 N 帧重估一次织造周期")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=None, help="CUDA 上开启 FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--no-prep-gpu", dest="prep_gpu", action="store_false", default=None, help="预处理放到 CPU")
    parser.add_argument("--target-fps", type=float, default=40.0)
    parser.add_argument("--json", default=None, help="结果写入 json")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.target_fps <= 0:
        parser.error("--frames 和 --target-fps 必须为正，--warmup 必须非负")
    cfg = load_predict_overrides(args.config, args.overrides, args.root)
    if args.size:
        from fabric_ncd.inference import as_hw
        cfg.setdefault("data", {})["image_size"] = list(as_hw(args.size))
    if args.ckpt is None:
        print("未提供 checkpoint：使用随机权重，仅测速度。", file=sys.stderr)
        cfg.setdefault("model", {}).setdefault("backbone", {})["pretrained"] = False

    predictor = FabricPredictor(
        checkpoint=args.ckpt,
        config=cfg,
        device=args.device,
        out_stride=args.out_stride,
        fp16=args.fp16,
        prep_gpu=args.prep_gpu,
        weave_interval=args.weave_interval,
        memory_stride=args.memory_stride,
    )
    device = predictor.device
    images = _collect(args.source)
    if not images:
        print(f"未找到图片: {args.source}", file=sys.stderr)
        return 1

    budget = 1000.0 / max(args.target_fps, 1e-6)
    print(f"设备: {device} | 输入: {predictor.image_size} | out_stride={predictor.out_stride} "
          f"| fp16={predictor.fp16} | prep_gpu={predictor.prep_gpu} | weave_interval={predictor.weave_interval} "
          f"| memory_stride={predictor.memory_stride}")
    print(f"目标: {args.target_fps:.0f} FPS -> 单帧预算 {budget:.1f} ms | 图片 {len(images)} 张, 共 {args.frames} 帧\n")

    if args.mode == "sequential":
        stats = _run_sequential(predictor, images, args.frames, args.warmup)
    else:
        stats = _run_pipeline(predictor, images, args.frames, args.warmup)

    print(f"{'阶段':<10}{'mean':>9}{'p50':>9}{'p95':>9}   (ms)")
    for stage in ("read", "infer", "analyze", "latency"):
        if f"{stage}_mean" not in stats:
            continue
        print(f"{stage:<10}{stats[f'{stage}_mean']:>9.2f}{stats[f'{stage}_p50']:>9.2f}{stats[f'{stage}_p95']:>9.2f}")
    fps = stats.get("fps_wall", 0.0)
    print(f"\n吞吐: {fps:.1f} FPS  (目标 {args.target_fps:.0f})  ->  {'达标 ✔' if fps >= args.target_fps else '未达标 ✘'}")
    _advice(stats, budget, args.target_fps)
    if args.json:
        json_dump(stats, args.json)
        print(f"\n结果已写入 {args.json}")
    return 0


def _run_sequential(predictor: FabricPredictor, images: List[Path], frames: int, warmup: int) -> Dict[str, Any]:
    from fabric_ncd.realtime import FrameResult

    # 预热不计入统计 (cudnn autotune / 显存池 / 织造周期缓存)
    for i in range(warmup):
        result = predictor.analyze_maps(predictor.predict_maps(load_rgb_fast(images[i % len(images)])))

    results: List[FrameResult] = []
    t_start = time.perf_counter()
    for i in range(frames):
        path = images[i % len(images)]
        t0 = time.perf_counter()
        rgb = load_rgb_fast(path)
        t1 = time.perf_counter()
        maps = predictor.predict_maps(rgb)
        t2 = time.perf_counter()
        result = predictor.analyze_maps(maps)
        t3 = time.perf_counter()
        results.append(
            FrameResult(
                index=i,
                result=result,
                read_ms=(t1 - t0) * 1000,
                infer_ms=(t2 - t1) * 1000,
                analyze_ms=(t3 - t2) * 1000,
                latency_ms=(t3 - t0) * 1000,
            )
        )
    wall = time.perf_counter() - t_start
    return summarize(results, wall)


def _run_pipeline(predictor: FabricPredictor, images: List[Path], frames: int, warmup: int) -> Dict[str, Any]:
    """三线程流水线吞吐测量.

    采用"最多领先 queue_size 帧"的提交方式 (block=True): 既不让流水线饿着,
    也不丢帧, 这样测到的是不丢帧前提下的最大吞吐.
    """
    pipe = RealtimePipeline(predictor, queue_size=2).start()
    lead = 3

    def run(count: int, start_index: int) -> List:
        submitted, results = 0, []
        while len(results) < count:
            while submitted < count and submitted - len(results) < lead:
                pipe.submit(images[(start_index + submitted) % len(images)], block=True)
                submitted += 1
            item = pipe.poll(timeout=30.0)
            if item is None:
                raise TimeoutError(f"实时流水线超时；错误: {pipe.errors}")
            if item.error:
                raise RuntimeError(item.error)
            results.append(item)
        return results

    try:
        run(warmup, 0)  # 预热必须全部完成后再开始计时
        t_start = time.perf_counter()
        results = run(frames, warmup)
        wall = time.perf_counter() - t_start
        return summarize(results, wall)
    finally:
        pipe.stop()


def _advice(stats: Dict[str, Any], budget: float, target: float) -> None:
    read = stats.get("read_p50", 0.0)
    analyze = stats.get("analyze_p50", 0.0)
    tips: List[str] = []
    if stats.get("infer_p50", 0.0) > budget:
        tips.append("前向超预算: 先降 data.image_size (例如 [768,1280]), 再开 --fp16, 并确认 out_stride>=4")
    if analyze > budget * 0.5:
        tips.append("区域分析偏重: 调大 --out-stride (4 -> 6/8), 或降低 eval.open_radius")
    if read > budget * 0.4:
        tips.append("解码偏重: 用 cv2 解码 (已自动优先), 或让采集端只送 ROI/灰度图")
    if not tips:
        tips.append("各段都在预算内: 如需更低延迟, 可缩小 queue_size, 或把 image_size 再降一档")
    print("\n建议:")
    for tip in tips:
        print(f"  - {tip}")


if __name__ == "__main__":
    raise SystemExit(main())
