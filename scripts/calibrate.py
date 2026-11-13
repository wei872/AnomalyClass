"""校准阈值: 在 normalval 纯正常图 + val 缺陷图(带 YOLO 标注) 上跑检测器.

若正常/缺陷分数分布可分 (min_defect > max_normal), img_thr 取间隙中点 ——
所有正常图判正常、所有缺陷图判异常, 不靠调参跷跷板; 若重叠则告警并报告.

最短用法 (默认读 checkpoints/detector/model_best.pth, 写 checkpoints/detector/thresholds.json, 默认已开 --sweep):
  python scripts/calibrate.py --data-yaml dataset/fabric19.yaml

指定权重/输出:
  python scripts/calibrate.py --data-yaml dataset/fabric19.yaml \
      --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.data import load_fabric_config
    from src.detect import calibrate_checkpoint
    from src.util import get_logger
    p = argparse.ArgumentParser(description='校准异常阈值 (normalval 正常分数 + val 缺陷分数间隙中点) 并评估 val 集')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--det-ckpt', default=None, help='默认 checkpoints/detector/model_final.pth (没有则 model_best.pth)')
    p.add_argument('--out', default=None, help='默认写在检测器权重同目录: <det-ckpt 目录>/thresholds.json')
    p.add_argument('--pix-quantile', type=float, default=0.9999, help='正常图异常像素分位数 (区域分割阈值)')
    p.add_argument('--max-images', type=int, default=0, help='0=全部; 调试/加速可限制每侧校准图数量')
    p.add_argument('--smooth-sigma', type=float, default=0, help='覆盖异常图平滑核(须与训练一致, 0=用权重自带的); 调大可缓解正常/缺陷分数重叠')
    p.add_argument('--sweep', dest='sweep', action='store_true', default=True,
                   help='(默认开) 不重训, 在 val 上扫 平滑sigma × 图级分数取法(最大值/最高N%%像素均值), 自动采用相对间隙最大的组合并写入 thresholds.json')
    p.add_argument('--no-sweep', dest='sweep', action='store_false', help='不扫参, 直接用权重自带的 smooth_sigma 与最大值')
    p.add_argument('--score-ratio', type=float, default=-1,
                   help='手工指定图级分数: 0=最大值, 0.01=最高1%%像素均值 (官方评估用); -1=不改 (默认最大值)')
    p.add_argument('--half', action='store_true', help='FP16 推理 (autocast, 1660 上更快)')
    p.add_argument('--no-eval', action='store_true', help='跳过 val 评估')
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()
    from src.detect import resolve_det_ckpt
    args.det_ckpt = resolve_det_ckpt(args.det_ckpt)

    log = get_logger('calibrate')
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.det_ckpt)), 'thresholds.json')
    if not os.path.exists(args.det_ckpt):
        raise SystemExit(f'找不到检测器权重: {args.det_ckpt}  (先运行 python scripts/train_detector.py)')
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    normal_val = cfg.normalval_images or cfg.normaltrain_images
    if not normal_val or not os.path.isdir(normal_val):
        raise FileNotFoundError(f'正常校准集不存在: {normal_val}')
    calibrate_checkpoint(args.det_ckpt, args.out, normal_val, cfg.val_images, cfg.val_labels,
                         class_names=cfg.class_names, device=args.device, sweep=args.sweep,
                         smooth_sigma=args.smooth_sigma, score_ratio=args.score_ratio,
                         pix_quantile=args.pix_quantile, max_images=args.max_images, half=args.half,
                         run_eval=not args.no_eval, allow_random_init=args.allow_random_init, logger=log)


if __name__ == '__main__':
    main()
