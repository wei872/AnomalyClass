"""阈值校准 / 图像级分数的回归测试."""

import numpy as np

from fabric_ncd.ops import threshold_from_normal_pixels, topk_mean


def test_threshold_never_exceeds_normal_max():
    # 正常像素分数分布很尖且接近 1: 均值+6σ 会超过 1.0, 阈值必须被最大值兜底,
    # 否则任何像素都超过不了阈值, 验证集区域数恒为 0.
    rng = np.random.default_rng(0)
    values = np.clip(rng.normal(0.90, 0.05, 20000), 0.0, 1.0).astype(np.float32)
    stats = threshold_from_normal_pixels(values, quantile=0.999, sigma_k=6.0)
    assert stats["sigma_thr"] > 1.0
    assert stats["threshold"] <= stats["max"] + 1e-6
    assert stats["threshold"] >= stats["quantile_thr"] - 1e-6


def test_threshold_prefers_larger_of_quantile_and_sigma():
    rng = np.random.default_rng(0)
    values = rng.random(20000).astype(np.float32) * 0.5
    stats = threshold_from_normal_pixels(values, quantile=0.999, sigma_k=1.0)
    if stats["sigma_thr"] >= stats["max"]:
        assert stats["threshold"] == stats["quantile_thr"]
    else:
        assert stats["threshold"] == max(stats["quantile_thr"], stats["sigma_thr"])
    assert stats["threshold"] <= stats["max"]


def test_sigma_term_falls_back_to_quantile_when_unreachable():
    # 分布有尖峰时 mean+kσ 会超过正常最大值, 此时阈值必须退回分位数,
    # 否则阈值等于"任何像素都达不到", 检出率恒为 0.
    rng = np.random.default_rng(1)
    values = np.clip(rng.normal(0.60, 0.10, 50000), 0.0, 1.0).astype(np.float32)
    stats = threshold_from_normal_pixels(values, quantile=0.999, sigma_k=6.0)
    assert stats["sigma_thr"] >= stats["max"]
    assert stats["threshold"] == stats["quantile_thr"]


def test_topk_pixels_not_diluted_by_image_size():
    # 10 个高亮像素: 固定 256 像素取均值 -> 全是高亮; 反而 1% 比例在 1e6 像素上
    # 会取一万个像素, 均值被稀释到接近 0 (小疵点图像级分数被拉低的原因).
    score = np.zeros(1_000_000, dtype=np.float32)
    score[:10] = 1.0
    assert topk_mean(score, k_pixels=10) == 1.0
    assert topk_mean(score, k_ratio=0.01) < 0.01
