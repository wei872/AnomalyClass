"""把训练好的 Dinomaly 与 kNN 记忆库融合成一个检测器 (共用 encoder, 只多一次最近邻搜索), 权重在 val 上自动选.

  fused = w * dino_map / ref_d + (1 - w) * knn_map / ref_k      (w=1 纯 Dinomaly, w=0 纯 kNN, 都是候选)

注意: 这一步已并入 train_detector.py (训练结束自动建 kNN 库 + 融合 + 选 w), 本脚本只用于单独重做融合.
  python scripts/fit_knn.py --out checkpoints/detector/knn_best.pth --layer-sets "5,7,9"   # 层必须在 2~9 之间
  python scripts/fuse_detectors.py            # 默认读 detector/model_best.pth + knn_best.pth, 写 model_fused.pth
注意: 融合分数的尺度不同于单个检测器, 必须对融合 ckpt 重新 calibrate (不要沿用旧 thresholds.json).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    from src.data import load_fabric_config
    from src.fusion import fuse_checkpoints
    from src.util import get_logger
    p = argparse.ArgumentParser(description='融合 Dinomaly + kNN 记忆库检测器')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--dino-ckpt', default='checkpoints/detector/model_best.pth')
    p.add_argument('--knn-ckpt', default='checkpoints/detector/knn_best.pth')
    p.add_argument('--out', default='checkpoints/detector/model_fused.pth')
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
    fuse_checkpoints(args.dino_ckpt, args.knn_ckpt, args.out, cfg.normalval_images or cfg.normaltrain_images,
                         cfg.val_images, cfg.val_labels, weights=args.weights, max_images=args.max_images,
                         score_ratio=args.score_ratio, map_size=args.map_size, half=args.half, device=args.device,
                         allow_random_init=args.allow_random_init, logger=log)
    log(f'下一步 (必须重新校准): python scripts/calibrate.py --det-ckpt {args.out}')


if __name__ == '__main__':
    main()
