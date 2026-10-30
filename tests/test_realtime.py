"""实时路径测试: 输出下采样一致性 / 流水线 / 形态学与连通域加速的正确性."""

import time

import numpy as np
import torch

from fabric_ncd.data.demo import weave_image
from fabric_ncd.inference import FabricPredictor
from fabric_ncd.ncd import RegionAnalyzer
from fabric_ncd.ops import binary_dilate, binary_erode, box_filter, connected_components
from fabric_ncd.realtime import RealtimePipeline
from fabric_ncd.utils import load_yaml


def _cfg(size=(64, 96)):
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"]["image_size"] = list(size)
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["heads"]["num_classes"] = 3
    cfg["infer"].update({"out_stride": 4, "probs_mul": 2, "fp16": False, "prep_gpu": True})
    return cfg


def _predictor():
    return FabricPredictor(config=_cfg(), device="cpu")


def test_out_stride_and_probs_scale():
    predictor = _predictor()
    rgb = np.random.randint(0, 255, (150, 210, 3), dtype=np.uint8)
    maps = predictor.predict_maps(rgb)
    assert maps["score"].shape == (38, 53)          # ceil(150/4), ceil(210/4)
    assert maps["probs"].shape[-2:] == (19, 27)     # 再降一档
    assert maps["probs_scale"] == 2
    assert "embedding" not in maps                  # 默认不算嵌入


def test_analyze_with_coarse_probs_matches_full():
    """类别图降采样后, 区域分类结果应与全分辨率一致."""
    rng = np.random.default_rng(0)
    h, w = 120, 180
    score = np.full((h, w), 0.05, np.float32)
    score[30:60, 40:80] = 0.9
    cloth = np.ones((h, w), np.float32)
    probs = rng.random((19, h // 2, w // 2)).astype(np.float32) * 0.05
    probs[3] += 0.9            # 类别 3 在每处都占主导
    probs /= probs.sum(0, keepdims=True)

    analyzer = RegionAnalyzer(names=[f"c{i}" for i in range(19)], threshold=0.5, min_area_ratio=1e-3)
    infos_coarse, _ = analyzer.analyze(score, cloth, probs, probs_scale=2)
    assert infos_coarse
    upsampled = np.repeat(np.repeat(probs, 2, axis=1), 2, axis=2)[:, :h, :w]
    infos_full, _ = analyzer.analyze(score, cloth, upsampled, probs_scale=1)
    assert [r.cls for r in infos_coarse] == [r.cls for r in infos_full]
    assert infos_coarse[0].cls == 3


def test_morphology_and_cc_correctness():
    rng = np.random.default_rng(1)
    mask = (rng.random((40, 55)) > 0.75).astype(np.uint8)
    dil = binary_dilate(mask, 1)
    assert (dil >= mask).all()
    assert binary_erode(dil, 1).max() <= dil.max()
    labels, num = connected_components(dil)
    assert num >= 1 and labels.max() == num

    x = rng.random((30, 40)).astype(np.float32)
    smooth = box_filter(x, 1)
    assert smooth.shape == x.shape
    assert abs(float(smooth.mean()) - float(x.mean())) < 1e-4


def test_pipeline_runs_and_reports_stages():
    predictor = _predictor()
    image = weave_image((120, 180), np.random.default_rng(2))
    pipe = RealtimePipeline(predictor, queue_size=8).start()
    total = 6
    for _ in range(total):
        pipe.submit(image, block=True)
    results = []
    t0 = time.perf_counter()
    while len(results) < total:
        item = pipe.poll(timeout=10.0)
        if item is None:
            break
        results.append(item)
    wall = time.perf_counter() - t0
    pipe.stop()

    assert len(results) == total
    assert all(r.result["image"]["image_score"] >= 0 for r in results)
    assert all(r.latency_ms > 0 for r in results)
    assert wall < 30.0
    # 结果按完成顺序递增
    assert [r.index for r in results] == sorted(r.index for r in results)


def test_predict_maps_then_analyze_equals_predict():
    predictor = _predictor()
    image = weave_image((120, 180), np.random.default_rng(3))
    direct = predictor.predict(image)
    split = predictor.analyze_maps(predictor.predict_maps(image))
    assert [r["bbox"] for r in direct["regions"]] == [r["bbox"] for r in split["regions"]]
    assert direct["image"]["image_score"] == split["image"]["image_score"]
    assert torch.isfinite(torch.tensor(float(direct["image"]["image_score"])))
