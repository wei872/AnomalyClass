"""检测器关键修复的回归测试 (CPU 秒级, 不需要权重/数据集).

运行: python tests/test_detector_fixes.py
"""
import os
import random
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import CropDataset
import src.classify as classify_mod
from src.data.yolo import pil_to_tensor_batch
from src.detect import (CombinedTrainDataset, train_detector, TrainAug, calibrate, evaluate_detector, AnomalyDetector,
                        WarmupCosineScheduler, sweep_postprocess)
from src.dinomaly import build_dinomaly
from src.dinomaly.losses import masked_recon
from src.dinomaly import StableAdamW
from src.util import score_from_map, GaussianSmoother, get_gaussian_kernel, imwrite_unicode, make_smoother, xyxy_to_yolo


def _make_ds(d, size, box, crop):
    for sub in ['n', 'imgs', 'labels']:
        os.makedirs(f'{d}/{sub}')
    imwrite_unicode(f'{d}/n/a.png', np.full((size, size, 3), 100, np.uint8))
    img = np.full((size, size, 3), 100, np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = 255                                  # 缺陷 = 亮块
    imwrite_unicode(f'{d}/imgs/b.png', img)
    open(f'{d}/labels/b.txt', 'w').write(' '.join(map(str, xyxy_to_yolo([(0, x1, y1, x2, y2)], size, size)[0])))
    return CombinedTrainDataset(f'{d}/n', f'{d}/imgs', f'{d}/labels', TrainAug(size, crop), size, crop)


def test_mask_follows_image_aug():
    """随机 rot90/翻转后, 缺陷 token 掩码必须仍然盖住图里的亮块."""
    random.seed(0)
    for size, crop, box in [(112, 112, (0, 0, 28, 28)), (112, 112, (10, 70, 60, 84)), (112, 98, (30, 30, 70, 70))]:
        with tempfile.TemporaryDirectory() as d:
            ds = _make_ds(d, size, box, crop)
            bad = 0
            for _ in range(100):
                t, flag, m = ds[1]
                bright = (t[0] > t[0].median() + 0.5).float().numpy()
                b = cv2.resize(bright, (m.shape[0], m.shape[0]), interpolation=cv2.INTER_AREA) > 0.25
                if (b * m.numpy()).sum() == 0:
                    bad += 1
            assert bad == 0, f'size={size} crop={crop} box={box}: {bad}/100 个样本掩码与缺陷不重合'
    print('ok  掩码与图像增强同步')


def test_thin_box_not_lost():
    """细长框 (2px 宽) 下采样到 token 网格后不能变成全 0 (旧版 INTER_NEAREST 会丢)."""
    random.seed(1)
    with tempfile.TemporaryDirectory() as d:
        ds = _make_ds(d, 448, (200, 0, 202, 400), 448)
        ds.aug = TrainAug(448, 448, rot90=False)
        for _ in range(20):
            _, _, m = ds[1]
            assert m.sum() >= 20, f'细长框 token 数 {m.sum()}'
    print('ok  细长框 token 掩码保留')


def test_no_rot90():
    aug = TrainAug(112, 112, rot90=False)
    img = np.zeros((112, 112, 3), np.uint8)
    img[:, :56] = 255                                        # 左半亮: 不旋转时, 亮边只会在左右翻转, 不会跑到上/下
    for _ in range(30):
        t, _ = aug(img)
        g = t[0]
        assert abs(float(g[:56].mean() - g[56:].mean())) < 1e-3, '关闭 rot90 后出现了 90° 旋转'
    print('ok  --no-rot90 不旋转')


def test_smoother_equivalent():
    x = torch.rand(2, 1, 64, 64)
    for sg, k, pm in [(4, 5, 'zeros'), (8, 49, 'replicate')]:
        a = make_smoother(sg)(x)
        b = get_gaussian_kernel(k, sg, padding_mode=pm)(x)
        assert (a - b).abs().max() < 1e-5
    # sigma=8 现在确实比 sigma=4 平滑得更狠
    spike = torch.zeros(1, 1, 64, 64)
    spike[..., 32, 32] = 1
    assert make_smoother(8)(spike).max() < make_smoother(4)(spike).max() * 0.2
    print('ok  可分离高斯平滑 == 原 2D 高斯, 且 sigma 越大越平')


def test_masked_recon():
    dmap = torch.ones(2, 1, 8, 8)
    fg = torch.zeros(2, 8, 8)
    fg[:, 2:4, 2:4] = 1
    v = masked_recon(dmap, fg, dilate=1)
    assert abs(float(v) - 1.0) < 1e-6                        # 只统计框外 -> 平均仍是 1
    dmap2 = dmap.clone()
    dmap2[:, :, 2:4, 2:4] = 100.0                            # 框内(缺陷)再大也不应计入
    assert abs(float(masked_recon(dmap2, fg)) - 1.0) < 1e-6
    print('ok  masked_recon 只算框外')


def test_eval_and_calibrate():
    """val 只有缺陷图时, 给 normal_dir 才有 I-AUROC / 间隙; calibrate 记录 smooth_sigma 且直方图分位数近似正确."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        for sub in ['dimg', 'dlab', 'nimg']:
            os.makedirs(f'{d}/{sub}')
        for i in range(4):
            img = rng.integers(100, 140, (112, 112, 3), dtype=np.uint8)
            img[20:60, 20:60] = 250
            imwrite_unicode(f'{d}/dimg/{i}.png', img)
            open(f'{d}/dlab/{i}.txt', 'w').write(' '.join(map(str, xyxy_to_yolo([(0, 20, 20, 60, 60)], 112, 112)[0])))
            imwrite_unicode(f'{d}/nimg/{i}.png', rng.integers(100, 140, (112, 112, 3), dtype=np.uint8))
        model = build_dinomaly('tiny_test_14', allow_random_init=True)
        st0 = evaluate_detector(model, f'{d}/dimg', f'{d}/dlab', 'cpu', 112, 112, return_stats=True)
        assert st0['i_auroc'] != st0['i_auroc'], '只有缺陷图时 AUROC 应为 NaN (旧版据此永远选不出 best)'
        st = evaluate_detector(model, f'{d}/dimg', f'{d}/dlab', 'cpu', 112, 112, normal_dir=f'{d}/nimg',
                               return_stats=True)
        assert st['i_auroc'] == st['i_auroc'] and st['n'] == 8 and st['gap_rel'] == st['gap_rel']
        det = AnomalyDetector(model, 'cpu', 112, 112, smooth_sigma=8.0)
        res = calibrate(det, f'{d}/nimg', f'{d}/dimg', f'{d}/dlab')
        assert res['smooth_sigma'] == 8.0 and res['pix_thr'] > 0
        pix = np.concatenate([det.predict(cv2.imread(f'{d}/nimg/{i}.png')[..., ::-1])[0].ravel() for i in range(4)])
        assert abs(res['pix_thr'] - np.quantile(pix, 0.9999)) < 2e-5 + 1e-3 * abs(res['pix_thr'])
    print('ok  evaluate(normal_dir) / calibrate(smooth_sigma, 直方图分位数)')


def test_stable_adamw():
    """StableAdamW 能收敛; 梯度含 NaN 时整步跳过 (参数不变); 更新量 RMS 被限幅."""
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 1)
    opt = StableAdamW(net.parameters(), lr=1e-2, weight_decay=0.0)
    x, y = torch.randn(64, 4), torch.randn(64, 1)
    l0 = float(((net(x) - y) ** 2).mean())
    for _ in range(200):
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        opt.step()
    assert float(((net(x) - y) ** 2).mean()) < l0
    before = [p.clone() for p in net.parameters()]
    opt.zero_grad()
    ((net(x) - y) ** 2).mean().backward()
    list(net.parameters())[1].grad.fill_(float('nan'))
    opt.step()
    assert opt.skipped_steps == 1 and all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    print('ok  StableAdamW 收敛 / NaN 整步跳过')


def test_scheduler_semantics():
    p = torch.nn.Parameter(torch.zeros(1))
    for fr, end in [(0.0, 0.0), (1.0, 1.0), (0.1, 0.1)]:
        opt = torch.optim.SGD([p], lr=1.0)
        sch = WarmupCosineScheduler(opt, total_epochs=100, warmup_epochs=10, final_ratio=fr)
        for _ in range(100):
            sch.step()
        assert abs(opt.param_groups[0]['lr'] - end) < 1e-6, (fr, opt.param_groups[0]['lr'])
    print('ok  lr 调度: final_ratio 0=退火到0 / 1=恒定(官方) / 0.1')


def test_score_from_map():
    a = np.zeros((100, 100), np.float32)
    a[0, 0] = 1.0                                            # 单点噪声峰值
    a[50:52, :] = 0.5                                        # 一条细线 (200 px = 2%)
    assert score_from_map(a, 0) == 1.0
    assert abs(score_from_map(a, 0.01) - (1.0 + 99 * 0.5) / 100) < 1e-6   # 单点被 top-1% 均值稀释
    print('ok  score_from_map: max / top-ratio 均值')


def test_half_path_and_sweep():
    """autocast 路径 (CPU/bfloat16 代跑) 能前向且与 fp32 接近; sweep_postprocess 输出排序后的扫参表."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        for sub in ['dimg', 'dlab', 'nimg']:
            os.makedirs(f'{d}/{sub}')
        for i in range(4):
            img = rng.integers(100, 140, (112, 112, 3), dtype=np.uint8)
            img[20:60, 20:60] = 250
            imwrite_unicode(f'{d}/dimg/{i}.png', img)
            open(f'{d}/dlab/{i}.txt', 'w').write(' '.join(map(str, xyxy_to_yolo([(0, 20, 20, 60, 60)], 112, 112)[0])))
            imwrite_unicode(f'{d}/nimg/{i}.png', rng.integers(100, 140, (112, 112, 3), dtype=np.uint8))
        model = build_dinomaly('tiny_test_14', allow_random_init=True)
        det = AnomalyDetector(model, 'cpu', 112, 112)
        img = cv2.imread(f'{d}/nimg/0.png')[..., ::-1]
        ref = det.raw_map(img)
        det.half, det.device_type, det.amp_dtype = True, 'cpu', torch.bfloat16
        amp = det.raw_map(img)
        assert amp.dtype == torch.float32 and (amp - ref).abs().max() < 0.1
        det.half = False
        sw = sweep_postprocess(det, f'{d}/nimg', f'{d}/dimg', f'{d}/dlab', sigmas=(4.0, 8.0), ratios=(0.0, 0.01))
        gaps = [t['gap_rel'] for t in sw['table']]
        assert len(gaps) == 4 and gaps == sorted(gaps, reverse=True) and sw['best'] is sw['table'][0]
        # 扫参里 (sigma=4, ratio=0) 的分数必须与 detector 的正常推理路径一致
        t44 = [t for t in sw['table'] if t['smooth_sigma'] == 4.0 and t['score_ratio'] == 0.0][0]
        mx = max(det.predict(cv2.imread(f'{d}/nimg/{i}.png')[..., ::-1])[1] for i in range(4))
        assert abs(t44['max_normal'] - mx) < 1e-5
        det.set_score_ratio(0.01)
        res = calibrate(det, f'{d}/nimg')
        assert res['score_ratio'] == 0.01
    print('ok  autocast 路径 / sweep_postprocess / thresholds 记录 score_ratio')


def test_classifier_aug():
    """分类器: 默认不做 90° 旋转 (经/纬方向是类别信息); 训练裁剪对 GT 框做抖动, 评估不抖."""
    stripes = np.zeros((64, 64, 3), np.uint8)
    stripes[::4] = 255                                       # 水平线 (纬向)

    def orient(rot90):
        vert = 0
        for _ in range(40):
            t = pil_to_tensor_batch([stripes], 64, augment=True, rot90=rot90)[0, 0]
            if (t[1:] - t[:-1]).abs().mean() < (t[:, 1:] - t[:, :-1]).abs().mean():
                vert += 1
        return vert

    assert orient(False) == 0, '默认不应出现 90° 旋转'
    assert orient(True) > 0
    with tempfile.TemporaryDirectory() as d:
        imwrite_unicode(f'{d}/a.png', np.full((200, 200, 3), 90, np.uint8))
        idx = [(f'{d}/a.png', (0, 0.5, 0.5, 0.3, 0.3), 0.3)]
        shapes = {True: set(), False: set()}
        orig = classify_mod.pil_to_tensor_batch
        try:
            classify_mod.pil_to_tensor_batch = lambda crops, *a, **k: (shapes[k['augment']].add(crops[0].shape[:2]),
                                                                       orig(crops, *a, **k))[1]
            for aug in (True, False):
                ds = CropDataset(idx, 64, augment=aug, box_jitter=0.3)
                for _ in range(30):
                    ds[0]
        finally:
            classify_mod.pil_to_tensor_batch = orig
        assert len(shapes[False]) == 1 and len(shapes[True]) > 5, (shapes[False], len(shapes[True]))
    print('ok  分类器: 默认不旋转 / 训练框抖动 / 评估不抖')


def test_defect_frac_log():
    """缺陷图极少时, auto 模式把每个 batch 的缺陷占比抬到 25%."""
    rng = np.random.default_rng(0)
    msgs = []
    with tempfile.TemporaryDirectory() as d:
        for sub in ['n', 'd', 'l']:
            os.makedirs(f'{d}/{sub}')
        for i in range(30):
            imwrite_unicode(f'{d}/n/{i}.png', rng.integers(100, 140, (112, 112, 3), dtype=np.uint8))
        for i in range(2):
            imwrite_unicode(f'{d}/d/{i}.png', rng.integers(100, 140, (112, 112, 3), dtype=np.uint8))
            open(f'{d}/l/{i}.txt', 'w').write(' '.join(map(str, xyxy_to_yolo([(0, 20, 20, 60, 60)], 112, 112)[0])))
        train_detector(f'{d}/n', f'{d}/out', backbone='tiny_test_14', image_size=112, crop_size=112, total_iters=2,
                       batch_size=4, device='cpu', num_workers=0, allow_random_init=True, no_eval=True,
                       defect_dir=f'{d}/d', defect_labels_dir=f'{d}/l', logger=lambda *a, **k: msgs.append(str(a[0])))
    line = [m for m in msgs if '缺陷图自然占比' in m]
    assert line and '目标占比 25%' in line[0], line
    print('ok  defect_frac auto: 缺陷图极少时抬到 25%')


if __name__ == '__main__':
    test_mask_follows_image_aug()
    test_thin_box_not_lost()
    test_no_rot90()
    test_smoother_equivalent()
    test_masked_recon()
    test_eval_and_calibrate()
    test_stable_adamw()
    test_scheduler_semantics()
    test_score_from_map()
    test_half_path_and_sweep()
    test_classifier_aug()
    test_defect_frac_log()
    print('ALL PASSED ✅')
