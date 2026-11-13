"""把训练好的 Dinomaly 与 kNN 记忆库融合成一个检测器 (共用 encoder, 只多一次最近邻搜索), 权重在 val 上自动选.

  fused = w * dino_map / ref_d + (1 - w) * knn_map / ref_k      (w=1 纯 Dinomaly, w=0 纯 kNN, 都是候选)

步骤:
  python scripts/train_detector.py ...                       # -> checkpoints/detector/model_best.pth
  python scripts/fit_knn.py --layer-sets "5,7,9" ...         # -> checkpoints/knn/model_best.pth  (层必须在 2~9 之间)
  python scripts/fuse_detectors.py --dino-ckpt checkpoints/detector/model_best.pth \
      --knn-ckpt checkpoints/knn/model_best.pth --out checkpoints/fused/model_best.pth
  python scripts/calibrate.py --det-ckpt checkpoints/fused/model_best.pth --out checkpoints/fused/thresholds.json --sweep
注意: 融合分数的尺度不同于单个检测器, 必须对融合 ckpt 重新 calibrate (不要沿用旧 thresholds.json).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.data import load_fabric_config
    from src.data.yolo import NormalImageDataset, YoloDefectDataset
    from src.detect import AnomalyDetector
    from src.fusion import FusionModel
    from src.util import get_logger, roc_auc, score_from_map, torch_load
    p = argparse.ArgumentParser(description='融合 Dinomaly + kNN 记忆库检测器')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--dino-ckpt', default='checkpoints/detector/model_best.pth')
    p.add_argument('--knn-ckpt', default='checkpoints/knn/model_best.pth')
    p.add_argument('--out', default='checkpoints/fused/model_best.pth')
    p.add_argument('--weights', default='0,0.25,0.5,0.75,1', help='候选的 Dinomaly 权重 w (逗号分隔)')
    p.add_argument('--max-images', type=int, default=300, help='正常/缺陷 val 图各最多用多少张')
    p.add_argument('--score-ratio', type=float, default=0.0, help='选 w 时图级分数取法 (0=最大值)')
    p.add_argument('--map-size', type=int, default=224, help='选 w 时缓存的异常图分辨率 (省内存)')
    p.add_argument('--half', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()

    log = get_logger('fuse')
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    dino_det = AnomalyDetector.load(args.dino_ckpt, device=args.device, allow_random_init=args.allow_random_init,
                                    logger=log, half=args.half)
    dino_hp = torch_load(args.dino_ckpt, map_location='cpu')['hparams']
    kck = torch_load(args.knn_ckpt, map_location='cpu')
    khp = kck['hparams']
    if khp.get('arch') != 'knn':
        raise ValueError(f'--knn-ckpt 不是 fit_knn.py 产出的 kNN 检测器: {args.knn_ckpt}')
    if khp['backbone'] != dino_hp['backbone'] or int(khp['image_size']) != int(dino_hp['image_size']):
        raise ValueError(f'两个检测器的 backbone / image_size 必须一致: dino=({dino_hp["backbone"]},{dino_hp["image_size"]}) '
                         f'knn=({khp["backbone"]},{khp["image_size"]})')
    bank = kck['model']['bank'].to(args.device)
    fm = FusionModel(dino_det.model, bank, khp['layers'], w=0.5)
    det = AnomalyDetector(fm, args.device, dino_det.image_size, dino_det.crop_size,
                          smooth_sigma=dino_det.smooth_sigma, half=args.half)
    log(f'Dinomaly: {args.dino_ckpt} | kNN: {args.knn_ckpt} (layers={khp["layers"]}, bank={tuple(bank.shape)})')

    normal_dir = cfg.normalval_images or cfg.normaltrain_images
    if not normal_dir or not os.path.isdir(normal_dir) or not cfg.val_images or not os.path.isdir(cfg.val_images):
        raise FileNotFoundError('需要 normalval(正常) 和 val(带标注缺陷) 目录来选融合权重')
    ds_n = NormalImageDataset(normal_dir, verify=True, logger=log)
    ds_d = YoloDefectDataset(cfg.val_images, cfg.val_labels, max_side=2048, verify=True, logger=log)

    @torch.no_grad()
    def maps(img):
        x = det.tf(img).unsqueeze(0).to(args.device)
        if det.half:
            with torch.autocast(device_type=det.device_type, dtype=det.amp_dtype):
                d, k = fm.component_maps(x, x.shape[-1])
        else:
            d, k = fm.component_maps(x, x.shape[-1])
        out = []
        for m in (d, k):
            m = det.gaussian(m.float())
            out.append(F.interpolate(m, size=(args.map_size, args.map_size), mode='bilinear',
                                     align_corners=False)[0, 0].cpu().numpy().astype(np.float16))
        return out

    n_n = min(args.max_images, len(ds_n))
    n_d = min(args.max_images, len(ds_d))
    log(f'缓存异常图: 正常 {n_n} + 缺陷 {n_d} (各一次前向, 之后扫权重不再跑模型)')
    mn = [maps(ds_n[i][0]) for i in range(n_n)]
    md = []
    for i in range(n_d):
        img, boxes, _, _ = ds_d[i]
        if boxes:
            md.append(maps(img))
    if not mn or not md:
        raise ValueError('需要同时有正常图和带标注的缺陷图')
    sc = lambda m: score_from_map(m.astype(np.float32), args.score_ratio)
    ref_d = float(np.mean([sc(m[0]) for m in mn]))
    ref_k = float(np.mean([sc(m[1]) for m in mn]))
    log(f'归一化参考 (正常图平均图级分数): ref_d={ref_d:.4f} ref_k={ref_k:.4f}')

    rows = []
    for w in [float(v) for v in args.weights.split(',')]:
        fuse = lambda m: (w * m[0].astype(np.float32) / ref_d + (1 - w) * m[1].astype(np.float32) / ref_k)
        sn = np.array([score_from_map(fuse(m), args.score_ratio) for m in mn])
        sd = np.array([score_from_map(fuse(m), args.score_ratio) for m in md])
        auroc = roc_auc([0] * len(sn) + [1] * len(sd), list(sn) + list(sd))
        gap = float((sd.min() - sn.max()) / max(sn.max(), 1e-8))
        rows.append((w, auroc, gap))
        tag = {1.0: ' (纯 Dinomaly)', 0.0: ' (纯 kNN)'}.get(w, '')
        log(f'w_dinomaly={w:.2f}{tag}: I-AUROC={auroc:.4f} 相对间隙={gap:+.3f}  (正常 max={sn.max():.3f} 缺陷 min={sd.min():.3f})')
    best = max(rows, key=lambda r: (round(r[1], 6), r[2]))
    w = best[0]
    log(f'*** 选定 w_dinomaly={w:.2f}: I-AUROC={best[1]:.4f} 间隙={best[2]:+.3f}')
    if w >= 1.0:
        log('融合没有带来提升: 直接用 Dinomaly 即可 (仍会保存融合 ckpt, 但推理会多算一次 kNN, 不建议用)')
    elif w <= 0.0:
        log('kNN 单独最好: 建议直接用 fit_knn.py 的 ckpt (更省)')
    fm.w, fm.ref_d, fm.ref_k = w, ref_d, ref_k
    hp = dict(dino_hp)
    hp.update({'arch': 'fusion', 'knn_layers': list(khp['layers']), 'fusion_w': w, 'ref_d': ref_d, 'ref_k': ref_k,
               'bank_size': int(bank.shape[0]), 'smooth_sigma': float(dino_det.smooth_sigma)})
    det.save(args.out, hp)
    log(f'已保存 {args.out}. 下一步 (必须重新校准):\n  python scripts/calibrate.py --det-ckpt {args.out} '
        f'--out {os.path.join(os.path.dirname(args.out), "thresholds.json")} --sweep')


if __name__ == '__main__':
    main()
