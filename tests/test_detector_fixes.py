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
from src.util import score_from_map, get_gaussian_kernel, imwrite_unicode, make_smoother, xyxy_to_yolo


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


def test_sdpa_and_enc_amp():
    """encoder 注意力改用融合 SDPA: 与手写 softmax 数值等价; enc_fp16 在 CPU 上自动关闭, 不改变结果."""
    from src.dinomaly.vit_dinov2 import Attention
    torch.manual_seed(0)
    a = Attention(48, 3, True).eval()
    x = torch.randn(2, 37, 48)
    q, k, v = a.qkv(x).reshape(2, 37, 3, 3, 16).permute(2, 0, 3, 1, 4).unbind(0)
    ref = ((q @ k.transpose(-2, -1) * a.scale).softmax(-1) @ v).transpose(1, 2).reshape(2, 37, 48)
    ref = a.proj(ref)
    assert (a(x) - ref).abs().max() < 1e-5
    from src.dinomaly import build_dinomaly
    m = build_dinomaly('tiny_test_14', allow_random_init=True, logger=lambda *a, **k: None).eval()
    img = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        e0, d0 = m(img)
        m.enc_amp = True
        e1, d1 = m(img)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(e0 + d0, e1 + d1))
    print('ok  encoder SDPA 数值等价 / enc_fp16 在 CPU 上为空操作')


def test_early_stop():
    """连续 N 次评估 model_best 没提升 -> 提前结束训练."""
    import src.detect as D
    rng = np.random.default_rng(0)
    calls = []

    def fake_eval(*a, **k):
        calls.append(1)
        return {'i_auroc': 0.9, 'gap_rel': 0.1}            # 第一次是新最优, 之后都不提升

    orig = D.evaluate_detector
    D.evaluate_detector = fake_eval
    msgs = []
    try:
        with tempfile.TemporaryDirectory() as d:
            for sub in ['n', 'v']:
                os.makedirs(f'{d}/{sub}')
            for i in range(6):
                imwrite_unicode(f'{d}/n/{i}.png', rng.integers(100, 140, (56, 56, 3), dtype=np.uint8))
            D.train_detector(f'{d}/n', f'{d}/out', backbone='tiny_test_14', image_size=56, crop_size=56,
                             total_iters=12, batch_size=2, device='cpu', num_workers=0, allow_random_init=True,
                             eval_dir=f'{d}/n', eval_interval=2, early_stop=2, save_interval=100,
                             logger=lambda *a, **k: msgs.append(str(a[0])))
    finally:
        D.evaluate_detector = orig
    assert len(calls) == 3, calls                            # iter2 最优, iter4/6 未提升 -> 在 iter6 停
    assert any('早停' in m for m in msgs)
    print('ok  early_stop: 3 次评估后在 iter 6 提前结束')


def _toy_texture(rng, defect=False, size=112):
    base = np.tile(np.sin(np.arange(size)[None, :] / 2.0) * 40 + 128, (size, 1))
    img = base + rng.normal(0, 6, (size, size))
    if defect:
        y, x = rng.integers(10, 80, 2)
        img[y:y + 24, x:x + 24] = img[y:y + 24, x:x + 24].T * 0.5 + 60
    return np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8)


def test_knn_detector():
    """免训练 kNN 记忆库检测器: 能分开正常/缺陷, 与 Dinomaly 同接口, 存取一致, 能直接 calibrate / 接入 pipeline."""
    from src.detect import get_eval_transform
    from src.knn_detect import build_knn, fit_knn, kcenter_greedy
    from src.util import roc_auc
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    quiet = lambda *a, **k: None
    m = build_knn('tiny_test_14', layers=(0, 1), allow_random_init=True, logger=quiet)
    n = fit_knn(m, [_toy_texture(rng) for _ in range(30)], get_eval_transform(112, 112), device='cpu',
                bank_size=200, logger=quiet)
    assert n == 200 and m.bank.shape == (200, 64)
    det = AnomalyDetector(m, 'cpu', 112, 112)
    ns = [det.predict(_toy_texture(rng))[1] for _ in range(12)]
    ds = [det.predict(_toy_texture(rng, True))[1] for _ in range(12)]
    assert roc_auc([0] * 12 + [1] * 12, ns + ds) >= 0.9, (ns, ds)
    # coreset 覆盖半径应不差于随机抽样
    x = torch.randn(1500, 16)
    cov = lambda idx: torch.cdist(x, x[idx]).min(1).values.max().item()
    assert cov(kcenter_greedy(x, 80)) <= cov(torch.randint(0, 1500, (80,)))
    with tempfile.TemporaryDirectory() as d:
        hp = {'arch': 'knn', 'backbone': 'tiny_test_14', 'image_size': 112, 'crop_size': 112, 'layers': [0, 1],
              'smooth_sigma': 4.0}
        det.save(f'{d}/k.pth', hp)
        det2 = AnomalyDetector.load(f'{d}/k.pth', device='cpu', allow_random_init=True, logger=quiet)
        img = _toy_texture(rng, True)
        assert abs(det.predict(img)[1] - det2.predict(img)[1]) < 1e-5
        # 下游: calibrate 直接可用
        for sub in ['nimg', 'dimg', 'dlab']:
            os.makedirs(f'{d}/{sub}')
        for i in range(4):
            imwrite_unicode(f'{d}/nimg/{i}.png', _toy_texture(rng))
            imwrite_unicode(f'{d}/dimg/{i}.png', _toy_texture(rng, True))
            open(f'{d}/dlab/{i}.txt', 'w').write('0 0.5 0.5 0.3 0.3')
        res = calibrate(det2, f'{d}/nimg', f'{d}/dimg', f'{d}/dlab')
        assert res['img_thr'] > 0 and res['pix_thr'] > 0
    print('ok  kNN 记忆库检测器: AUROC>=0.9 / coreset / 存取一致 / calibrate 可用')


def test_balanced_softmax_head():
    """类别极不均衡 + 两类重叠时, Balanced Softmax (tau=1) 的 balanced accuracy 应高于普通 CE (tau=0)."""
    import torch.nn.functional as F
    from src.embed_classify import train_head
    g = torch.Generator().manual_seed(0)
    D = 32                                                       # 像真实嵌入: 有一个大的公共分量, 类别只是其上的小偏移, 再 L2 归一化
    common = torch.randn(D, generator=g)
    common = 3 * common / common.norm()
    d0 = torch.randn(D, generator=g)
    d0 = d0 / d0.norm()
    d1 = d0 + 0.55 * 3 / D ** 0.5 * torch.randn(D, generator=g)
    d1 = d1 / d1.norm()

    def make(n0, n1, noise=0.3):
        x0 = common + d0 + noise * torch.randn(n0, D, generator=g)
        x1 = common + d1 + noise * torch.randn(n1, D, generator=g)
        return F.normalize(torch.cat([x0, x1]), dim=1), torch.cat([torch.zeros(n0, dtype=torch.long),
                                                                    torch.ones(n1, dtype=torch.long)])
    X, Y = make(1000, 20)
    vX, vY = make(500, 500)
    res = {}
    for tau in (0.0, 1.0):
        head, st = train_head(X, Y, vX, vY, 2, epochs=60, tau=tau, device='cpu', logger=lambda *a, **k: None)
        res[tau] = st['val_bal_acc']
    assert res[1.0] > res[0.0] + 0.03, res
    print(f'ok  Balanced Softmax: balanced acc {res[0.0]:.3f} (CE) -> {res[1.0]:.3f}')


def test_embed_classifier_pipeline():
    """DINOv2 嵌入分类器: 小样本不均衡合成数据上训练 -> load_classifier 自动识别 -> 存取一致 -> 接入 pipeline."""
    from src.classify import load_classifier
    from src.embed_classify import EmbedRegionClassifier, train_embed_classifier
    from src.pipeline import FabricAnomalyPipeline
    rng = np.random.default_rng(0)

    def mk(cls):
        img = rng.normal(128, 10, (160, 160)).astype(np.float32)
        y, x = rng.integers(30, 100, 2)
        if cls == 0:
            img[y:y + 8, x:x + 40] += 90; b = (x + 20, y + 4, 40, 8)
        elif cls == 1:
            img[y:y + 40, x:x + 8] += 90; b = (x + 4, y + 20, 8, 40)
        else:
            for _ in range(8):
                yy, xx = rng.integers(y, y + 30), rng.integers(x, x + 30)
                img[yy:yy + 4, xx:xx + 4] += 120
            b = (x + 15, y + 15, 34, 34)
        return np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8), [v / 160 for v in b]

    with tempfile.TemporaryDirectory() as d:
        k = 0
        for split in ['tr', 'va']:
            os.makedirs(f'{d}/{split}/img'); os.makedirs(f'{d}/{split}/lab')
            for c, n in {0: 60, 1: 20, 2: 6}.items():
                for _ in range(n if split == 'tr' else max(n // 2, 4)):
                    img, (cx, cy, w, h) = mk(c)
                    imwrite_unicode(f'{d}/{split}/img/{k}.png', img)
                    open(f'{d}/{split}/lab/{k}.txt', 'w').write(f'{c} {cx} {cy} {w} {h}')
                    k += 1
        quiet = lambda *a, **kw: None
        path = train_embed_classifier(f'{d}/tr/img', f'{d}/tr/lab', f'{d}/va/img', f'{d}/va/lab', out_dir=f'{d}/out',
                                      nc=3, names=['a', 'b', 'c'], backbone='tiny_test_14', crop_size=56, epochs=60,
                                      device='cpu', allow_random_init=True, logger=quiet)
        cls = load_classifier(path, nc=3, names=['a', 'b', 'c', '未知类型'], device='cpu', logger=quiet)
        assert isinstance(cls, EmbedRegionClassifier)
        assert os.path.getsize(path) < 5e6                       # 只存分类头 (+ 冒烟用的 tiny 骨干)
        import json
        meta = json.load(open(f'{d}/out/classifier_meta.json', encoding='utf-8'))
        assert meta['kind'] == 'embed' and meta['val_bal_acc'] >= 0.9, meta
        crops = [mk(c)[0][20:120, 20:120] for c in (0, 1, 2)]
        pr = cls.predict_probs(crops)
        assert pr.shape == (3, 4) and np.allclose(pr.sum(1), 1, atol=1e-4)
        cls2 = load_classifier(path, nc=3, names=['a', 'b', 'c', '未知类型'], device='cpu', logger=quiet)
        assert np.allclose(pr, cls2.predict_probs(crops), atol=1e-5)
        det = AnomalyDetector(build_dinomaly('tiny_test_14', allow_random_init=True, logger=quiet), 'cpu', 112, 112)
        pipe = FabricAnomalyPipeline(det, cls, ['a', 'b', 'c', '未知类型'], img_thr=-1.0, pix_thr=0.0, device='cpu',
                                     logger=quiet)
        res = pipe.process(cv2.cvtColor(mk(0)[0], cv2.COLOR_RGB2BGR), 'x.png')
        assert res.is_anomaly and all(r.class_name for r in res.regions)
    print('ok  DINOv2 嵌入分类器: 训练 / 自动识别 / 存取一致 / 接入 pipeline')


def test_fusion_detector():
    """融合检测器: w=1 退化为 Dinomaly, w=0 退化为 kNN (与独立实现逐点一致); 存取一致; 层不在 Dinomaly 目标层内时报错."""
    from src.detect import get_eval_transform, model_anomaly_map
    from src.fusion import FusionModel
    from src.knn_detect import KNNModel, fit_knn
    from src.util import cal_anomaly_maps
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    quiet = lambda *a, **k: None
    dino = build_dinomaly('tiny_test_14', allow_random_init=True, logger=quiet).eval()
    knn = KNNModel(dino.encoder, layers=(0, 1))                       # 与 Dinomaly 共用同一个 encoder
    fit_knn(knn, [_toy_texture(rng) for _ in range(20)], get_eval_transform(112, 112), device='cpu', bank_size=150,
            logger=quiet)
    x = get_eval_transform(112, 112)(_toy_texture(rng, True)).unsqueeze(0)
    with torch.no_grad():
        en, de = dino(x)
        d_ref, _ = cal_anomaly_maps(en, de, 112)
        k_ref = model_anomaly_map(knn, x, 112)
        for w, ref in ((1.0, d_ref / 0.7), (0.0, k_ref / 0.03)):
            fm = FusionModel(dino, knn.bank.clone(), (0, 1), w=w, ref_d=0.7, ref_k=0.03).eval()
            assert (fm.anomaly_map(x, 112) - ref).abs().max() < 1e-4, w
        fm = FusionModel(dino, knn.bank.clone(), (0, 1), w=0.4, ref_d=0.7, ref_k=0.03).eval()
        mix = fm.anomaly_map(x, 112)
        assert (mix - (0.4 * d_ref / 0.7 + 0.6 * k_ref / 0.03)).abs().max() < 1e-4
        assert dino.keep_raw is False and dino._raw_en is None
    try:
        FusionModel(dino, knn.bank.clone(), (5,))
        raise AssertionError('应当拒绝不在目标层内的 knn_layers')
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        hp = {'arch': 'fusion', 'backbone': 'tiny_test_14', 'image_size': 112, 'crop_size': 112, 'dropout': 0.4,
              'la': 1, 'lc': 2, 'cr': 1, 'knn_layers': [0, 1], 'fusion_w': 0.4, 'ref_d': 0.7, 'ref_k': 0.03,
              'smooth_sigma': 4.0}
        det = AnomalyDetector(fm, 'cpu', 112, 112)
        det.save(f'{d}/f.pth', hp)
        det2 = AnomalyDetector.load(f'{d}/f.pth', device='cpu', allow_random_init=True, logger=quiet)
        img = _toy_texture(rng, True)
        assert abs(det.predict(img)[1] - det2.predict(img)[1]) < 1e-5
    print('ok  融合检测器: w=1 / w=0 退化一致, 混合正确, 存取一致')


def test_tiny_head_and_calibrate_checkpoint():
    """极少步数的分类头训练不再除零; calibrate_checkpoint 能从权重文件一步写出 thresholds.json."""
    import json
    import torch.nn.functional as F
    from src.embed_classify import train_head
    from src.detect import calibrate_checkpoint
    g = torch.Generator().manual_seed(0)
    X = F.normalize(torch.randn(8, 16, generator=g), dim=1)
    Y = torch.tensor([0, 1] * 4)
    train_head(X, Y, X, Y, nc=2, epochs=3, logger=lambda *a, **k: None)
    rng = np.random.default_rng(0)
    quiet = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as d:
        for sub in ['dimg', 'dlab', 'nimg']:
            os.makedirs(f'{d}/{sub}')
        for i in range(3):
            img = rng.integers(100, 140, (112, 112, 3), dtype=np.uint8)
            img[20:60, 20:60] = 250
            imwrite_unicode(f'{d}/dimg/{i}.png', img)
            open(f'{d}/dlab/{i}.txt', 'w').write(' '.join(map(str, xyxy_to_yolo([(0, 20, 20, 60, 60)], 112, 112)[0])))
            imwrite_unicode(f'{d}/nimg/{i}.png', rng.integers(100, 140, (112, 112, 3), dtype=np.uint8))
        model = build_dinomaly('tiny_test_14', allow_random_init=True)
        det = AnomalyDetector(model, 'cpu', 112, 112, smooth_sigma=4.0)
        det.save(f'{d}/m.pth', {'backbone': 'tiny_test_14', 'image_size': 112, 'crop_size': 112, 'smooth_sigma': 4.0})
        res = calibrate_checkpoint(f'{d}/m.pth', f'{d}/thresholds.json', f'{d}/nimg', f'{d}/dimg', f'{d}/dlab',
                                   device='cpu', sweep=True, allow_random_init=True, logger=quiet)
        assert os.path.exists(f'{d}/thresholds.json') and json.load(open(f'{d}/thresholds.json'))['img_thr'] == res['img_thr']
        # 训练后自动 kNN + 融合 -> model_final.pth, 且 resolve_det_ckpt 优先选它
        import shutil
        from src.detect import resolve_det_ckpt
        from src.fusion import build_final_detector
        os.makedirs(f'{d}/ck')
        shutil.copyfile(f'{d}/m.pth', f'{d}/ck/model_best.pth')
        assert resolve_det_ckpt(None, f'{d}/ck').endswith('model_best.pth')
        r = build_final_detector(f'{d}/ck', f'{d}/nimg', f'{d}/nimg', f'{d}/dimg', f'{d}/dlab', backbone='tiny_test_14',
                                 image_size=112, layer_sets='0,1', bank_size=500, max_images=3, device='cpu',
                                 allow_random_init=True, logger=quiet)
        assert os.path.exists(f'{d}/ck/model_final.pth') and resolve_det_ckpt(None, f'{d}/ck').endswith('model_final.pth')
        det_f = AnomalyDetector.load(f'{d}/ck/model_final.pth', device='cpu', allow_random_init=True, logger=quiet)
        assert det_f.predict(cv2.imread(f'{d}/nimg/0.png')[..., ::-1])[1] == det_f.predict(cv2.imread(f'{d}/nimg/0.png')[..., ::-1])[1]
        assert 0.0 <= r['w'] <= 1.0
    print('ok  少步数分类头不除零; calibrate_checkpoint 一步写出 thresholds.json; 训练后 kNN+融合产出 model_final.pth')


def test_detect_only_cli():
    """纯检测入口的参数存在 (--help 可解析)."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for script, flag in [('infer.py', '--input'), ('evaluate.py', '--no-classify'), ('infer.py', '--no-classify')]:
        r = subprocess.run([sys.executable, os.path.join(root, 'scripts', script), '--help'], capture_output=True,
                           text=True, cwd=root)
        assert r.returncode == 0 and flag in r.stdout, (script, r.stderr[-300:])
    print('ok  infer --input/--no-classify, evaluate --no-classify 参数存在')


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
    test_sdpa_and_enc_amp()
    test_early_stop()
    test_knn_detector()
    test_balanced_softmax_head()
    test_embed_classifier_pipeline()
    test_fusion_detector()
    test_tiny_head_and_calibrate_checkpoint()
    test_detect_only_cli()
    print('ALL PASSED ✅')
