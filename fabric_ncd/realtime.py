"""实时推理流水线: 读图 / 前向 (GPU) / 区域分析 (CPU) 三级重叠.

40 FPS 的瓶颈通常不是网络本身, 而是三件事串行叠加:

1. 解码 1440x2440 JPEG/PNG (5~15 ms, 纯 CPU);
2. 网络前向 (GPU, 视分辨率 3~10 ms);
3. 区域分析 / 连通域 / 类别池化 (CPU, 数 ms).

串行执行时三者相加很容易超过 25 ms. 本模块把三段放到三个线程, 用容量很小的
队列串联 (容量 1~2, 控制端到端延迟不堆积), 此时单帧延迟 ≈ 最慢的一段,
吞吐 ≈ 1 / 平均每段耗时. 这是产线上常见的 "三工位" 结构.

用法::

    pipe = RealtimePipeline(predictor).start()
    for item in stream:
        pipe.submit(item)
    for res in pipe.results(timeout=5):
        print(res.index, res.latency_ms, len(res.result["regions"]))
    pipe.stop()
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image

try:  # 有 cv2 时解码快 2~3 倍
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False
    cv2 = None


def load_rgb_fast(source: Any) -> np.ndarray:
    """路径 / PIL / ndarray -> uint8 RGB (H, W, 3)."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1).astype(np.uint8)
        return arr[..., :3].astype(np.uint8, copy=False)
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"), dtype=np.uint8)
    path = str(source)
    if _HAS_CV2:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with Image.open(Path(path)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class FrameResult:
    index: int
    result: Dict[str, Any]
    read_ms: float = 0.0
    infer_ms: float = 0.0
    analyze_ms: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def regions(self) -> List[Dict[str, Any]]:
        return self.result.get("regions", []) if self.result else []


class RealtimePipeline:
    """三级流水线. ``predictor`` 需提供 ``predict_maps`` / ``analyze_maps``."""

    def __init__(self, predictor, queue_size: int = 2, overlay: Optional[Callable] = None) -> None:
        self.predictor = predictor
        self.overlay = overlay
        self.queue_size = max(1, int(queue_size))
        self._q_raw: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        self._q_mid: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        self._q_ana: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        # 结果队列不设上限也绝不丢弃: 丢掉一条就是漏检一条, 宁可占一点内存
        self._q_out: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._count = 0
        self._errors: List[str] = []

    # ------------------------------------------------------------------ #
    def start(self) -> "RealtimePipeline":
        for name, fn in (
            ("reader", self._reader_loop),
            ("infer", self._infer_loop),
            ("analyze", self._analyze_loop),
        ):
            t = threading.Thread(target=fn, name=f"fabric-{name}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for q in (self._q_raw, self._q_mid, self._q_ana, self._q_out):
            try:
                q.put_nowait(None)  # 唤醒阻塞的线程
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def __enter__(self) -> "RealtimePipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    def submit(self, source: Any, meta: Optional[Dict[str, Any]] = None, block: bool = False) -> bool:
        """投递一帧 (路径 / PIL / ndarray). 队列满时默认丢弃最旧帧, 保证实时性."""
        item = (self._count, source, meta or {}, time.perf_counter())
        self._count += 1
        while True:
            try:
                self._q_raw.put_nowait(item)
                return True
            except queue.Full:
                if block:
                    time.sleep(0.001)
                    continue
                try:  # 丢最旧的一帧
                    self._q_raw.get_nowait()
                except queue.Empty:
                    pass
                continue

    def poll(self, timeout: float = 0.0) -> Optional[FrameResult]:
        try:
            item = self._q_out.get(timeout=timeout) if timeout > 0 else self._q_out.get_nowait()
        except queue.Empty:
            return None
        return item

    def results(self, total: Optional[int] = None, timeout: float = 10.0) -> Iterator[FrameResult]:
        """按完成顺序产出结果. ``total`` 为 None 时直到 :meth:`stop`."""
        produced = 0
        while total is None or produced < total:
            item = self.poll(timeout=timeout)
            if item is None:
                if self._stop.is_set():
                    return
                continue
            produced += 1
            yield item

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    # ------------------------------------------------------------------ #
    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            item = self._q_raw.get()
            if item is None:
                return
            index, source, meta, t_in = item
            try:
                t_pick = time.perf_counter()  # 只统计解码本身, 不含排队等待
                rgb = load_rgb_fast(source)
                read_ms = (time.perf_counter() - t_pick) * 1000.0
                self._put(self._q_mid, (index, rgb, meta, t_in, read_ms))
            except Exception as exc:  # pragma: no cover
                self._errors.append(f"reader[{index}]: {exc}")

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            item = self._q_mid.get()
            if item is None:
                return
            index, rgb, meta, t_in, read_ms = item
            try:
                t0 = time.perf_counter()
                maps = self.predictor.predict_maps(rgb)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                self._put(self._q_ana, (index, maps, meta, t_in, read_ms, infer_ms))
            except Exception as exc:  # pragma: no cover
                self._errors.append(f"infer[{index}]: {exc}")

    def _analyze_loop(self) -> None:
        while not self._stop.is_set():
            item = self._q_ana.get()
            if item is None:
                return
            index, maps, meta, t_in, read_ms, infer_ms = item
            try:
                t0 = time.perf_counter()
                result = self.predictor.analyze_maps(maps)
                analyze_ms = (time.perf_counter() - t0) * 1000.0
                if self.overlay is not None:
                    result["overlay"] = self.overlay(result, meta)
                self._q_out.put(
                    FrameResult(
                        index=index,
                        result=result,
                        read_ms=read_ms,
                        infer_ms=infer_ms,
                        analyze_ms=analyze_ms,
                        latency_ms=(time.perf_counter() - t_in) * 1000.0,
                        meta=meta,
                    ),
                )
            except Exception as exc:  # pragma: no cover
                self._errors.append(f"analyze[{index}]: {exc}")

    @staticmethod
    def _put(q: "queue.Queue", item: Any) -> None:
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


def summarize(results: List[FrameResult], wall_seconds: Optional[float] = None) -> Dict[str, float]:
    """汇总各段耗时 (p50/p95) 与吞吐. ``wall_seconds`` 为整段实测墙钟时间."""
    if not results:
        return {}
    import statistics

    def stat(values: List[float]) -> Dict[str, float]:
        return {
            "mean": float(statistics.fmean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    out: Dict[str, float] = {"frames": len(results)}
    for key, attr in (("read", "read_ms"), ("infer", "infer_ms"), ("analyze", "analyze_ms"), ("latency", "latency_ms")):
        for name, value in stat([getattr(r, attr) for r in results]).items():
            out[f"{key}_{name}"] = value
    if wall_seconds and wall_seconds > 0:
        out["fps_wall"] = float(len(results) / wall_seconds)
    else:
        lat = out.get("latency_p50", 0.0)
        out["fps_wall"] = float(1000.0 / lat) if lat > 0 else 0.0
    return out
