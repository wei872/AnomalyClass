"""实时流水线：背压、失败帧以及停止时的行为。"""

import threading
import time

import numpy as np
import pytest
import torch

from fabric_ncd.realtime import RealtimePipeline, load_rgb_fast


class SlowPredictor:
    def predict_maps(self, rgb):
        return {"index": int(rgb[0, 0, 0])}

    def analyze_maps(self, maps):
        time.sleep(0.004)
        return {"image": {"index": maps["index"]}, "regions": []}


def test_blocking_submit_applies_backpressure_at_every_stage():
    pipe = RealtimePipeline(SlowPredictor(), queue_size=1).start()
    try:
        for i in range(20):
            pipe.submit(np.full((8, 8, 3), i, np.uint8), block=True)
        results = list(pipe.results(total=20, timeout=2))
        assert [r.index for r in results] == list(range(20))
        assert [r.result["image"]["index"] for r in results] == list(range(20))
        assert pipe.dropped == 0 and not pipe.errors
    finally:
        pipe.stop()


def test_failures_yield_results_instead_of_hanging(tmp_path):
    class BrokenPredictor:
        def predict_maps(self, rgb):
            value = int(rgb[0, 0, 0])
            if value == 1:
                raise ValueError("inference failed")
            return {"index": value}

        def analyze_maps(self, maps):
            raise ValueError("analysis failed")

    pipe = RealtimePipeline(BrokenPredictor(), queue_size=1).start()
    try:
        pipe.submit(tmp_path / "missing.png", meta={"kind": "read"}, block=True)
        pipe.submit(np.ones((8, 8, 3), np.uint8), meta={"kind": "infer"}, block=True)
        pipe.submit(np.full((8, 8, 3), 2, np.uint8), meta={"kind": "analyze"}, block=True)
        results = list(pipe.results(total=3, timeout=2))
        assert {r.index for r in results} == {0, 1, 2}
        assert {r.meta["kind"] for r in results} == {"read", "infer", "analyze"}
        assert all(r.error for r in results)
        assert len(pipe.errors) == 3
    finally:
        pipe.stop()


def test_stop_unblocks_blocking_submit_and_rejects_reuse():
    ready = threading.Event()
    release = threading.Event()

    class WaitingPredictor(SlowPredictor):
        def predict_maps(self, rgb):
            ready.set()
            release.wait(3)
            return super().predict_maps(rgb)

    pipe = RealtimePipeline(WaitingPredictor(), queue_size=1).start()
    image = np.zeros((8, 8, 3), np.uint8)
    pipe.submit(image, block=True)
    assert ready.wait(2)
    finished = threading.Event()
    errors = []

    def producer():
        try:
            for _ in range(20):
                pipe.submit(image, block=True)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            finished.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        pipe.stop(timeout=0.01)
        assert finished.wait(2)
        assert errors
        with pytest.raises(RuntimeError):
            pipe.submit(image)
        with pytest.raises(RuntimeError):
            pipe.start()
    finally:
        release.set()
        pipe.stop(timeout=1)
        t.join(timeout=1)
    assert all(not t.is_alive() for t in pipe._threads)


def test_load_rgb_fast_handles_float_tensor():
    tensor = torch.zeros(3, 12, 16)
    tensor[0] = 1.0
    rgb = load_rgb_fast(tensor)
    assert rgb.shape == (12, 16, 3) and rgb.dtype == np.uint8
    assert (rgb[..., 0] == 255).all() and not rgb[..., 1:].any()
