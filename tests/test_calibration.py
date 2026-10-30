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


def test_region_analyzer_cloth_gate_switch():
    """整幅都是布时必须能关掉布面蒙版, 否则疵点会被蒙版裁掉."""
    import numpy as np

    from fabric_ncd.ncd import RegionAnalyzer

    score = np.zeros((60, 60), dtype=np.float32)
    score[20:30, 20:30] = 0.9                      # 疵点
    cloth = np.ones((60, 60), dtype=np.float32)
    cloth[20:30, 20:30] = 0.2                      # 布面头把疵点判成"非布"

    gated = RegionAnalyzer(threshold=0.5, use_cloth=True)
    infos_gated, _ = gated.analyze(score, cloth)
    ungated = RegionAnalyzer(threshold=0.5, use_cloth=False)
    infos_ungated, _ = ungated.analyze(score, cloth)

    assert len(infos_gated) == 0                   # 被蒙版裁掉 -> 检不出
    assert len(infos_ungated) == 1                 # 关闭蒙版 -> 正常检出


def test_image_score_mode_contrast_removes_global_offset():
    """整图底色偏移不该影响图像级判据: contrast 口径要能把叠加的常数消掉."""
    import numpy as np

    from fabric_ncd.engine import image_score_of

    base = np.full((64, 64), 0.2, dtype=np.float32)
    defect = base.copy()
    defect[10:20, 10:20] = 0.6
    topk_cfg = {"topk_pixels": 16}
    contrast_cfg = {"topk_pixels": 16, "image_score_mode": "contrast"}

    # 给"异常图"整体叠一个负偏移 (模拟底色更暗)
    defect_shifted = defect - 0.1
    normal_shifted = base.copy()

    assert image_score_of(defect, topk_cfg) > image_score_of(base, topk_cfg)
    # 叠加偏移后 topk 失效, contrast 仍然分得开
    assert image_score_of(defect_shifted, contrast_cfg) > image_score_of(normal_shifted, contrast_cfg)


def test_spearman_and_pair_auc():
    """探针的两个基础统计量: 秩相关 / 同源内与跨源的成对 AUROC."""
    import numpy as np

    from fabric_ncd.metrics import spearman

    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0  # 常数列

    from tools.diagnose_scores import _pair_auc, source_of

    assert source_of("DA9670287_1777190506830_毛羽_0_orig.png") == "DA9670287"
    assert source_of("KBA26050009_20261015_104918_712034.bmp") == "KBA26050009"

    # 每批内部疵点都略高于正常 (0.9>0.8 / 0.3>0.2), 但 A、B 两批之间差 0.6:
    # 同源内 AUROC = 1.0, 跨源只剩 0.5 -> 说明"批次偏移"把判据吃掉了.
    rows = [
        {"src": "A", "defect": 1, "s": 0.9},
        {"src": "A", "defect": 0, "s": 0.8},
        {"src": "B", "defect": 1, "s": 0.3},
        {"src": "B", "defect": 0, "s": 0.2},
    ]
    assert _pair_auc(rows, "s", same_src=True) == (1.0, 2)
    assert _pair_auc(rows, "s", same_src=False) == (0.5, 2)


def test_grouped_auroc_ignores_between_group_shift():
    """跨批次偏移不该影响"同批次内是否分得开"这个指标."""
    import numpy as np

    from fabric_ncd.metrics import grouped_auroc

    # A 批: 疵点 0.9 / 正常 0.8;  B 批: 疵点 0.3 / 正常 0.2
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.3, 0.2]
    groups = ["A", "A", "B", "B"]
    auc, pairs = grouped_auroc(labels, scores, groups)
    assert pairs == 2 and auc == 1.0
    # 全局 AUROC 只有 0.75: 批次偏移把 B 批的两张图整体压到 A 批下面, 顺序被搅乱
    from fabric_ncd.metrics import auroc

    assert abs(auroc(np.asarray(labels), np.asarray(scores)) - 0.75) < 1e-9
    # 没有任何组同时含两类 -> nan
    auc2, pairs2 = grouped_auroc([1, 1, 0, 0], scores, groups)
    assert np.isnan(auc2) and pairs2 == 0
