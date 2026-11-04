"""校准阈值: 在纯正常样本 (normalval) 上跑检测器, 得到 img_thr / pix_thr, 并在 val 集上评估.

示例:
  python scripts/calibrate.py --data-yaml dataset/fabric19.yaml \
      --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fabric_config
from src.detect import AnomalyDetector, calibrate, evaluate_detector, save_json
from src.util import get_logger


def main():
    p = argparse.ArgumentParser(description='在正常样本上校准异常阈值, 并评估 val 集')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--det-ckpt', default='checkpoints/detector/model_best.pth')
    p.add_argument('--out', default='checkpoints/detector/thresholds.json')
    p.add_argument('--img-quantile', type=float, default=1.0, help='正常图分数分位数 (1.0=最大值)')
    p.add_argument('--img-factor', type=float, default=1.0, help='img_thr 安全系数')
    p.add_argument('--pix-quantile', type=float, default=0.9999)
    p.add_argument('--pix-factor', type=float, default=1.0)
    p.add_argument('--half', action='store_true', help='FP16 推理 (1660 上更快)')
    p.add_argument('--no-eval', action='store_true', help='跳过 val 评估')
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()

    log = get_logger('calibrate')
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    det = AnomalyDetector.load(args.det_ckpt, device=args.device, allow_random_init=args.allow_random_init,
                               logger=log, half=args.half)
    normal_val = cfg.normalval_images or cfg.normaltrain_images
    if not normal_val or not os.path.isdir(normal_val):
        raise FileNotFoundError(f'正常校准集不存在: {normal_val}')
    result = calibrate(det, normal_val, img_quantile=args.img_quantile, img_factor=args.img_factor,
                       pix_quantile=args.pix_quantile, pix_factor=args.pix_factor, logger=log)
    if not args.no_eval and cfg.val_images and os.path.isdir(cfg.val_images):
        evaluate_detector(det.model, cfg.val_images, cfg.val_labels, device=args.device,
                          image_size=det.image_size, crop_size=det.crop_size, logger=log)
    save_json(result, args.out)
    log(f'阈值已保存: {args.out}  (img_thr={result["img_thr"]:.4f}, pix_thr={result["pix_thr"]:.4f})')


if __name__ == '__main__':
    main()
