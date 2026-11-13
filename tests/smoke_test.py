"""端到端冒烟测试 (CPU, 合成数据, tiny 模型, 无需下载权重).

流程: 造合成布匹数据集(正常纹理 + 缺陷块, YOLO 标注) -> 训练检测器几步 -> 校准阈值
      -> 训练分类器几轮 -> 跑完整流水线 -> 校验输出结构.

运行: python tests/smoke_test.py
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import train_classifier, RegionClassifier
from src.data import load_fabric_config
from src.detect import AnomalyDetector, calibrate, train_detector, deviation_confidence
from src.pipeline import FabricAnomalyPipeline
from src.util import get_logger, setup_seed, xyxy_to_yolo, imwrite_unicode, imread_unicode

NC = 3
NAMES = ['污点', '破洞', '色差']


def make_texture(rng, h=160, w=160):
    base = rng.integers(120, 180, size=(h, w, 3), dtype=np.uint8)
    for _ in range(40):
        y, x = rng.integers(0, h), rng.integers(0, w)
        cv2.line(base, (x, y), (min(w - 1, x + rng.integers(5, 30)), min(h - 1, y + rng.integers(1, 4))),
                 (200, 200, 200), 1)
    return base


def make_sample(rng, defect=None):
    img = make_texture(rng)
    boxes = []
    if defect is not None:
        h, w = img.shape[:2]
        bw = int(w * 0.25)
        bh = int(h * 0.25)
        x1 = int(rng.integers(10, w - bw - 10))
        y1 = int(rng.integers(10, h - bh - 10))
        x2, y2 = x1 + bw, y1 + bh
        if defect == 0:      # 污点: 深色块
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
        elif defect == 1:    # 破洞: 白色块
            cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1)
        else:                # 色差: 红色块
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 220), -1)
        boxes.append((defect, x1, y1, x2, y2))
    return img, boxes


def write_yolo(txt, boxes, w, h):
    with open(txt, 'w') as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write(' '.join(map(str, xyxy_to_yolo([(c, x1, y1, x2, y2)], w, h)[0])) + '\n')


def build_dataset(root):
    rng = np.random.default_rng(7)
    for split in ['train', 'val']:
        os.makedirs(f'{root}/images/{split}', exist_ok=True)
        os.makedirs(f'{root}/labels/{split}', exist_ok=True)
        n = 24 if split == 'train' else 10
        for i in range(n):
            defect = None if i % 3 == 0 else i % NC
            img, boxes = make_sample(rng, defect)
            # 故意用中文文件名, 验证 Unicode 路径读写
            name = f'布匹样本_{i:03d}.png' if i % 2 == 0 else f'{i:03d}.png'
            imwrite_unicode(f'{root}/images/{split}/{name}', img)
            stem = os.path.splitext(name)[0]
            write_yolo(f'{root}/labels/{split}/{stem}.txt', boxes, img.shape[1], img.shape[0])
    for split in ['train', 'val']:
        os.makedirs(f'{root}/normal/images/{split}', exist_ok=True)
        for i in range(20 if split == 'train' else 8):
            imwrite_unicode(f'{root}/normal/images/{split}/正常_{i:03d}.png', make_texture(rng))
        # 故意放一个 0 字节的坏文件 (中文名 bmp), 应被探测并跳过而不是炸训练
        with open(f'{root}/normal/images/{split}/坏文件_000 - 副本.bmp', 'wb') as f:
            f.write(b'')
    yaml_path = f'{root}/fabric_test.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {root}\ntrain: {root}/images/train\nval: {root}/images/val\n'
                f'normaltrain: {root}/normal/images/train\nnormalval: {root}/normal/images/val\n'
                f'nc: {NC}\nnames: {NAMES}\n')
    return yaml_path


def main():
    log = get_logger('smoke')
    setup_seed(1)
    tmp = tempfile.mkdtemp(prefix='anomalyclass_smoke_')
    log(f'临时目录: {tmp}')
    try:
        yaml_path = build_dataset(tmp)
        cfg = load_fabric_config(yaml_path)
        log(cfg.summary())
        assert cfg.nc == NC and cfg.class_names[-1] == '未知类型'

        # 坏文件应被跳过
        from src.data.yolo import NormalImageDataset
        nds = NormalImageDataset(cfg.normaltrain_images, verify=True, logger=log)
        assert len(nds.skipped) == 1 and '坏文件' in nds.skipped[0]
        assert len(nds.images) == 20
        log(f'坏文件跳过正常: skipped={nds.skipped}')

        device = 'cpu'
        # 1) 训练检测器 (tiny, 少量迭代)
        det_dir = os.path.join(tmp, 'det')
        best = train_detector(cfg.normaltrain_images, det_dir, backbone='tiny_test_14',
                              image_size=112, crop_size=98, total_iters=6, batch_size=4,
                              eval_interval=6, eval_dir=cfg.val_images, eval_labels_dir=cfg.val_labels,
                              device=device, num_workers=0, allow_random_init=True, logger=log,
                              defect_dir=cfg.train_images, defect_labels_dir=cfg.train_labels,
                              margin_w=1.0, margin=0.05, box_margin_w=1.0, hard_normal_w=0.5,
                              local_loss_w=1.0, hard_pos_ratio=0.2, local_loss_warmup=3,
                              rot90=False, margin_score='smooth', defect_bg_w=0.5,
                              eval_normal_dir=cfg.normalval_images, smooth_sigma=6.0)
        assert os.path.exists(best)

        # 2) 校准阈值
        det = AnomalyDetector.load(best, device=device, allow_random_init=True, logger=log)
        thr = calibrate(det, cfg.normalval_images, logger=log)
        assert thr['img_thr'] > 0
        thr_path = os.path.join(tmp, 'thresholds.json')
        with open(thr_path, 'w') as f:
            json.dump(thr, f)

        # 3) 训练分类器
        cls_dir = os.path.join(tmp, 'cls')
        cls_best = train_classifier(cfg.train_images, cfg.train_labels, cfg.val_images, cfg.val_labels,
                                    out_dir=cls_dir, nc=NC, names=NAMES, arch='mobilenet_v3_small',
                                    crop_size=64, epochs=2, batch_size=8, num_workers=0,
                                    pretrained=False, device=device, logger=log)
        assert os.path.exists(cls_best)
        cls = RegionClassifier.load(cls_best, nc=NC, names=NAMES + ['未知类型'], device=device, logger=log)

        # 4) 完整流水线
        pipe = FabricAnomalyPipeline(det, cls, cfg.class_names, img_thr=thr['img_thr'], pix_thr=thr['pix_thr'],
                                     device=device, logger=log)
        rng = np.random.default_rng(99)
        img_def, boxes = make_sample(rng, defect=2)
        img_norm, _ = make_sample(rng, defect=None)
        res_def = pipe.process(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), '缺陷图.png')
        res_norm = pipe.process(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB), '正常图.png')
        # 中文路径的文件级读写也要通
        p_cn = os.path.join(tmp, '中文目录_测试图.png')
        assert imwrite_unicode(p_cn, cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB))
        assert imread_unicode(p_cn) is not None
        res_file = pipe.process_file(p_cn)
        assert res_file.is_anomaly == res_def.is_anomaly
        d = res_def.to_dict()
        json.dumps(d, ensure_ascii=False)  # 必须可 JSON 序列化
        log(f'defect: is_anomaly={d["is_anomaly"]} conf={d["anomaly_conf"]:.3f} regions={len(d["regions"])}')
        log(f'normal: is_anomaly={res_norm.is_anomaly} conf={res_norm.anomaly_conf:.3f}')
        for r in d['regions']:
            log(f'  region box={r["box"]} class={r["class_name"]}({r["class_id"]}) '
                f'class_conf={r["class_conf"]:.3f} anomaly_conf={r["anomaly_conf"]:.3f} probs_len={len(r["probs"])}')
            assert 0 <= r['class_id'] <= NC
            assert len(r['probs']) == NC + 1
            assert 0.0 <= r['anomaly_conf'] < 1.0
        assert 0.0 <= d['anomaly_conf'] < 1.0
        # 可视化也要能跑通
        vis = pipe.visualize(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), res_def)
        assert vis.shape[:2] == img_def.shape[:2]
        log(f'deviation_confidence(0)={deviation_confidence(0.0, thr["img_thr"]):.3f}, '
            f'({thr["img_thr"]})={deviation_confidence(thr["img_thr"], thr["img_thr"]):.3f} (应为0.5)')
        assert abs(deviation_confidence(thr['img_thr'], thr['img_thr']) - 0.5) < 1e-6

        # top_k 行为: 伪造 2 个区域, top_k=1 时只应有 1 个真正分类, 另 1 个为未知类型
        import numpy as _np
        pipe.top_k = 1
        fake_map = _np.zeros((160, 160), dtype=_np.float32)
        pipe.det.predict = lambda img, max_map_side=1024: (fake_map, 0.9, (1.0, 1.0))
        pipe.det.segment = lambda *a, **k: [
            {'box': (10, 10, 40, 40), 'area': 900, 'mean_score': 0.9},
            {'box': (100, 100, 130, 130), 'area': 800, 'mean_score': 0.5},
        ]
        res_top1 = pipe.process(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), 'topk.png')
        assert len(res_top1.regions) == 2
        n_classified = sum(1 for r in res_top1.regions if r.classified)
        assert n_classified == 1, f'top_k=1 应只分类 1 个区域, 实际 {n_classified}'
        assert res_top1.regions[0].classified and not res_top1.regions[1].classified
        assert res_top1.regions[1].class_id == pipe.nc and res_top1.regions[1].class_name == '未知类型'
        pipe.top_k = 0   # 全部分类
        res_all = pipe.process(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), 'all.png')
        assert all(r.classified for r in res_all.regions)
        log(f'top_k 行为正常: top_k=1 分类 {n_classified}/2 个区域, top_k=0 全部分类')
        # 阈值语义: score > img_thr -> 异常; <= img_thr -> 正常 (calibrate 取间隙中点, 两侧都分开, 无跷跷板调参)
        t = thr['img_thr']
        pipe.det.predict = lambda img, max_map_side=1024: (fake_map, t * 1.5, (1.0, 1.0))
        r_hi = pipe.process(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), 'hi.png')
        assert r_hi.is_anomaly, r_hi.anomaly_score
        pipe.det.predict = lambda img, max_map_side=1024: (fake_map, t * 0.5, (1.0, 1.0))
        r_lo = pipe.process(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB), 'lo.png')
        assert not r_lo.is_anomaly and len(r_lo.regions) == 0
        log('阈值语义正常: score>img_thr 判异常, 否则判正常')

        # crop_size==image_size 时训练transform不裁边缘 (整图进网络, 暗角边缘保留)
        from src.detect import get_train_transform
        x_full = get_train_transform(112, 112)(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB))
        assert x_full.shape == (3, 112, 112), x_full.shape
        log('crop_size==image_size 时训练不裁边缘: 输出整图 (3,112,112)')
        log('SMOKE TEST PASSED ✅')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
