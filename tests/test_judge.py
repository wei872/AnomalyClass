"""二次判断分类器 (src/judge.py) 的合成数据测试: CPU + tiny 骨干, 无需下载权重.

覆盖: 训练 (含留一类校准) -> 保存/加载 -> analyse/decide -> 流水线 (推翻误报 / 未知异常 / 严重度门槛).
运行: python tests/test_judge.py  或  pytest tests/test_judge.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import JudgeClassifier, token_mask, train_judge
from src.pipeline import FabricAnomalyPipeline
from src.util import imwrite_unicode, xyxy_to_yolo

NC = 4
NAMES = ['a', 'b', 'c', 'd']
COLORS = [(30, 30, 30), (245, 245, 245), (40, 40, 220), (40, 200, 40)]


def _tex(rng, s=128):
    img = rng.integers(120, 170, size=(s, s, 3), dtype=np.uint8)
    for _ in range(30):
        y, x = rng.integers(0, s), rng.integers(0, s)
        cv2.line(img, (x, y), (min(s - 1, x + 15), y), (190, 190, 190), 1)
    return img


def _make(root, rng):
    for split, n in (('train', 60), ('val', 24)):
        os.makedirs(f'{root}/images/{split}', exist_ok=True)
        os.makedirs(f'{root}/labels/{split}', exist_ok=True)
        for i in range(n):
            img, c = _tex(rng), i % NC
            x1, y1 = int(rng.integers(10, 70)), int(rng.integers(10, 70))
            x2, y2 = x1 + 36, y1 + 36
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], -1)
            imwrite_unicode(f'{root}/images/{split}/{i:03d}.png', img)
            with open(f'{root}/labels/{split}/{i:03d}.txt', 'w') as f:
                f.write(' '.join(map(str, xyxy_to_yolo([(c, x1, y1, x2, y2)], 128, 128)[0])) + '\n')
    for split, n in (('train', 30), ('val', 10)):
        os.makedirs(f'{root}/normal/{split}', exist_ok=True)
        for i in range(n):
            imwrite_unicode(f'{root}/normal/{split}/{i:03d}.png', _tex(rng))


def test_token_mask():
    m = token_mask((0, 0, 112, 56), 112, 112, 112)       # 上半幅 8x4 token 全覆盖
    assert m.shape == (8, 8) and abs(m[:4].sum() - 32) < 1e-4 and m[4:].sum() < 1e-4
    m = token_mask((0, 0, 100, 50), 100, 50, 112)        # letterbox: 100x50 缩放到 112x56, 上下各留灰边
    assert m[0].sum() < 1e-4 and abs(m.sum() - 8 * 4) < 1e-3


def test_judge_train_and_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(3)
        _make(tmp, rng)
        out = os.path.join(tmp, 'cls')
        best = train_judge(f'{tmp}/images/train', f'{tmp}/labels/train', f'{tmp}/images/val', f'{tmp}/labels/val',
                           f'{tmp}/normal/train', f'{tmp}/normal/val', out_dir=out, nc=NC, names=NAMES,
                           backbone='tiny_test_14', crop_size=56, ctx_size=112, views=2, min_per_class=8,
                           epochs=20, loco_epochs=10, device='cpu', allow_random_init=True, logger=lambda *a, **k: None)
        cls = JudgeClassifier.load(best, nc=NC, names=NAMES + ['未知类型'], device='cpu', logger=lambda *a, **k: None)
        assert cls.fuser is not None and cls.scorer is not None
        img = _tex(rng)
        ana = cls.analyse(img, [(10, 10, 50, 50), (60, 60, 100, 100)])
        assert ana['probs'].shape == (2, NC + 1) and ana['p_nov'].shape == (2,)
        assert abs(float(ana['probs'][0].sum()) - 1) < 1e-4
        # decide: 新颖且严重 -> 未知; 新颖但不严重 -> 最像的已知类
        p = np.array([0.7, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        assert cls.decide(p, 0.99, True)[2] and cls.decide(p, 0.99, True)[0] == NC
        assert cls.decide(p, 0.99, False) [0] == 0 and not cls.decide(p, 0.99, False)[2]
        assert cls.decide(p, 0.0, True)[0] == 0

        class FakeDet:
            def predict(self, img, max_map_side=1024):
                return np.zeros(img.shape[:2], np.float32), 0.9, (1.0, 1.0)

            def segment(self, *a, **k):
                return [{'box': (10, 10, 60, 60), 'area': 900, 'mean_score': 0.8},
                        {'box': (70, 70, 100, 100), 'area': 400, 'mean_score': 0.5}]

        pipe = FabricAnomalyPipeline(FakeDet(), cls, NAMES + ['未知类型'], img_thr=0.5, pix_thr=0.5, device='cpu',
                                     logger=lambda *a, **k: None, severe_score=0.6, severe_area=0.08)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res = pipe.process(bgr, 't.png')
        d = res.to_dict()
        assert res.is_anomaly or res.judge_override
        if res.is_anomaly:
            assert len(res.regions) == 2 and all(r.classified for r in res.regions)
            assert res.verdict in NAMES + ['未知类型'] and 0 <= res.final_class_id <= NC
            assert res.regions[0].severe and not res.regions[1].severe      # 面积占比 0.15 vs 0.055, 门槛 0.08
        assert 'verdict' in d and 'judge_override' in d

        # 强制: 所有区域都像"正常" -> 推翻 Dinomaly
        cls.analyse = lambda im, boxes, tta=True: {'probs': np.tile(np.array([0, 0, 0, 0, 1.0], np.float32), (len(boxes), 1)),
                                                   'p_nov': np.zeros(len(boxes), np.float32)}
        r2 = pipe.process(bgr, 't2.png')
        assert r2.judge_override and not r2.is_anomaly and r2.verdict == '正常' and r2.heat is None
        # 强制: 新颖 + 严重 -> 未知异常
        cls.analyse = lambda im, boxes, tta=True: {'probs': np.tile(np.array([.4, .3, .2, .1, 0], np.float32), (len(boxes), 1)),
                                                   'p_nov': np.ones(len(boxes), np.float32)}
        r3 = pipe.process(bgr, 't3.png')
        assert r3.regions[0].unknown and r3.verdict == '未知类型' and r3.final_class_id == NC
        assert not r3.regions[1].unknown                                      # 第二个区域不严重 -> 不判未知


if __name__ == '__main__':
    test_token_mask()
    test_judge_train_and_pipeline()
    print('PASSED')
