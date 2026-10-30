"""实时推理流水线: 读图 / 前向 (GPU) / 区域分析 (CPU) 三级重叠.

40 FPS 的瓶颈通常不是网络本身, 而是三件事串行叠加:

1. 解码 1600x846 JPEG/PNG (6~12 ms, 纯 CPU);
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
    from .inference import to_rgb_uint8

    # 相机可直接提供 HWC ndarray 或 CHW/HWC torch tensor。
    import torch
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu()
        if source.ndim == 3 and source.shape[0] in (1, 3, 4) and source.shape[-1] not in (1, 3, 4):
            source = source.permute(1, 2, 0)
        source = source.numpy()
    if isinstance(source, np.ndarray):
        return to_rgb_uint8(source)
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
    """三级流水线. 中间队列使用背压；仅非阻塞 submit 可丢弃最旧待处理帧。

    每个已开始处理的帧均产出结果（失败时 ``FrameResult.error`` 非空）。
    ``stop`` 是立即停止，不保证处理完尚在队列中的帧；实例不能再次启动。
    """

    def __init__(self, predictor, queue_size: int = 2, overlay: Optional[Callable] = None) -> None:
        self.predictor = predictor
        self.overlay = overlay
        self.queue_size = max(1, int(queue_size))
        self._q_raw: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        self._q_mid: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        self._q_ana: "queue.Queue" = queue.Queue(maxsize=self.queue_size)
        self._q_out: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._submit_lock = threading.Lock()
        self._started = False
        self._count = 0
        self._dropped = 0
        self._errors: List[str] = []

    def start(self) -> "RealtimePipeline":
        if self._stop.is_set():
            raise RuntimeError("流水线已经停止，请创建新实例")
        if self._started:
            return self
        self._started = True
        for name, fn in (
            ("reader", self._reader_loop),
            ("infer", self._infer_loop),
            ("analyze", self._analyze_loop),
        ):
            t = threading.Thread(target=fn, name=f"fabric-{name}", daemon=True)
            self._threads.append(t)
            t.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        # 每个队列的 get/put 有超时，满队列时也不会永久阻塞。
        for q in (self._q_raw, self._q_mid, self._q_ana):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        self._q_out.put_nowait(None)  # 唤醒 results()
        for t in self._threads:
            t.join(timeout=timeout)

    def __enter__(self) -> "RealtimePipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def submit(self, source: Any, meta: Optional[Dict[str, Any]] = None, block: bool = False) -> bool:
        """投递帧。block=True 全程背压、不主动丢帧；否则输入队列满时丢最旧帧。"""
        if not self._started or self._stop.is_set():
            raise RuntimeError("请先启动流水线；已停止的流水线不能提交")
        with self._submit_lock:
            item = (self._count, source, dict(meta or {}), time.perf_counter())
            while not self._stop.is_set():
                try:
                    if block:
                        self._q_raw.put(item, timeout=0.1)
                    else:
                        self._q_raw.put_nowait(item)
                    self._count += 1
                    return True
                except queue.Full:
                    if not block:
                        try:
                            self._q_raw.get_nowait()
                            self._dropped += 1
                        except queue.Empty:
                            pass
            raise RuntimeError("流水线已停止")

    def poll(self, timeout: float = 0.0) -> Optional[FrameResult]:
        try:
            return self._q_out.get(timeout=timeout) if timeout > 0 else self._q_out.get_nowait()
        except queue.Empty:
            return None

    def results(self, total: Optional[int] = None, timeout: float = 10.0) -> Iterator[FrameResult]:
        """按完成顺序产出结果（包括带 error 的失败帧）。"""
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

    @property
    def dropped(self) -> int:
        """非阻塞提交时从输入队列丢弃的帧数。"""
        return self._dropped

    def _get(self, q: "queue.Queue") -> Any:
        while not self._stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _put(self, q: "queue.Queue", item: Any) -> bool:
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _error(self, stage: str, index: int, exc: Exception, meta: Dict[str, Any],
               t_in: float, read_ms: float = 0.0, infer_ms: float = 0.0) -> None:
        message = f"{stage}[{index}]: {type(exc).__name__}: {exc}"
        self._errors.append(message)
        self._q_out.put(FrameResult(
            index=index, result={}, read_ms=read_ms, infer_ms=infer_ms,
            latency_ms=(time.perf_counter() - t_in) * 1000.0, error=message, meta=meta,
        ))

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            item = self._get(self._q_raw)
            if item is None:
                return
            index, source, meta, t_in = item
            try:
                t_pick = time.perf_counter()
                rgb = load_rgb_fast(source)
                read_ms = (time.perf_counter() - t_pick) * 1000.0
                self._put(self._q_mid, (index, rgb, meta, t_in, read_ms))
            except Exception as exc:
                self._error("reader", index, exc, meta, t_in)

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            item = self._get(self._q_mid)
            if item is None:
                return
            index, rgb, meta, t_in, read_ms = item
            try:
                t0 = time.perf_counter()
                maps = self.predictor.predict_maps(rgb)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                self._put(self._q_ana, (index, maps, meta, t_in, read_ms, infer_ms))
            except Exception as exc:
                self._error("infer", index, exc, meta, t_in, read_ms)

    def _analyze_loop(self) -> None:
        while not self._stop.is_set():
            item = self._get(self._q_ana)
            if item is None:
                return
            index, maps, meta, t_in, read_ms, infer_ms = item
            try:
                t0 = time.perf_counter()
                result = self.predictor.analyze_maps(maps)
                analyze_ms = (time.perf_counter() - t0) * 1000.0
                if self.overlay is not None:
                    result["overlay"] = self.overlay(result, meta)
                self._q_out.put(FrameResult(
                    index=index, result=result, read_ms=read_ms, infer_ms=infer_ms,
                    analyze_ms=analyze_ms, latency_ms=(time.perf_counter() - t_in) * 1000.0, meta=meta,
                ))
            except Exception as exc:
                self._error("analyze", index, exc, meta, t_in, read_ms, infer_ms)


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
