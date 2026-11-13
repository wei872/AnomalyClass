"""校准阈值: 在 normalval 纯正常图 + val 缺陷图(带 YOLO 标注) 上跑检测器.

若正常/缺陷分数分布可分 (min_defect > max_normal), img_thr 取间隙中点 ——
所有正常图判正常、所有缺陷图判异常, 不靠调参跷跷板; 若重叠则告警并报告.

示例:
  python scripts/calibrate.py --data-yaml dataset/fabric19.yaml \
      --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.data import load_fabric_config
    from src.detect import AnomalyDetector, calibrate, evaluate_detector, save_json, sweep_postprocess
    from src.util import get_logger
    p = argparse.ArgumentParser(description='校准异常阈值 (normalval 正常分数 + val 缺陷分数间隙中点) 并评估 val 集')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--det-ckpt', default='checkpoints/detector/model_best.pth')
    p.add_argument('--out', default='checkpoints/detector/thresholds.json')
    p.add_argument('--pix-quantile', type=float, default=0.9999, help='正常图异常像素分位数 (区域分割阈值)')
    p.add_argument('--max-images', type=int, default=0, help='0=全部; 调试/加速可限制每侧校准图数量')
    p.add_argument('--smooth-sigma', type=float, default=0, help='覆盖异常图平滑核(须与训练一致, 0=用权重自带的); 调大可缓解正常/缺陷分数重叠')
    p.add_argument('--sweep', action='store_true',
                   help='不重训, 在 val 上扫 平滑sigma × 图级分数取法(最大值/最高N%%像素均值), 自动采用相对间隙最大的组合并写入 thresholds.json')
    p.add_argument('--score-ratio', type=float, default=-1,
                   help='手工指定图级分数: 0=最大值, 0.01=最高1%%像素均值 (官方评估用); -1=不改 (默认最大值)')
    p.add_argument('--half', action='store_true', help='FP16 推理 (autocast, 1660 上更快)')
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
    if args.smooth_sigma and args.smooth_sigma != det.smooth_sigma:
        det.set_smooth_sigma(args.smooth_sigma)
        log(f'平滑核 sigma={args.smooth_sigma} (已写入 thresholds.json, 推理自动沿用)')
    else:
        log(f'平滑核 sigma={det.smooth_sigma} (取自权重 hparams, 已写入 thresholds.json)')
    if args.sweep:
        sw = sweep_postprocess(det, normal_val, cfg.val_images, cfg.val_labels, max_images=args.max_images, logger=log)
        det.set_smooth_sigma(sw['best']['smooth_sigma'])
        det.set_score_ratio(sw['best']['score_ratio'])
        log(f'已采用扫参最优: smooth_sigma={det.smooth_sigma} score_ratio={det.score_ratio}')
    elif args.score_ratio >= 0:
        det.set_score_ratio(args.score_ratio)
    log(f'图级分数取法: ' + ('最大值' if det.score_ratio <= 0 else f'最高 {det.score_ratio:.1%} 像素均值'))
    result = calibrate(det, normal_val,
                       defect_dir=cfg.val_images, defect_labels_dir=cfg.val_labels,
                       pix_quantile=args.pix_quantile, max_images=args.max_images,
                       class_names=cfg.class_names, logger=log)
    if not args.no_eval and cfg.val_images and os.path.isdir(cfg.val_images):
        evaluate_detector(det.model, cfg.val_images, cfg.val_labels, device=args.device,
                          image_size=det.image_size, crop_size=det.crop_size, logger=log,
                          normal_dir=normal_val, smooth_sigma=det.smooth_sigma, max_images=args.max_images,
                          score_ratio=det.score_ratio)
    save_json(result, args.out)
    log(f'阈值已保存: {args.out}  (img_thr={result["img_thr"]:.4f}, pix_thr={result["pix_thr"]:.4f}, '
        f'separable={result.get("separable")})')


if __name__ == '__main__':
    main()
