"""训练异常区域分类器 (YOLO 框裁剪 -> 19 类缺陷; 低置信度自动归为 "未知类型").

示例:
  python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml \
      --arch mobilenet_v3_small --epochs 30 --batch-size 64 --out checkpoints/classifier
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classify import train_classifier
from src.data import load_fabric_config
from src.util import get_logger, setup_seed


def main():
    p = argparse.ArgumentParser(description='训练缺陷区域分类器 (mobilenet_v3_small 推荐, 1660 实时)')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--arch', default='mobilenet_v3_small',
                   choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34',
                            'efficientnet_b0'],
                   help='精度优先可选 resnet18/efficientnet_b0 (1660 上仍实时); 速度优先 mobilenet_v3_small')
    p.add_argument('--crop-size', type=int, default=224, help='224 速度快; 288/320 更准(letterbox 保宽高比)')
    p.add_argument('--crop-pad', type=float, default=0.3, help='裁剪时四周扩展比例')
    p.add_argument('--epochs', type=int, default=50, help='默认 50, 配合早停(patience=15)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-ratio', type=float, default=0.1, help='未提供 val 时从 train 划分比例')
    p.add_argument('--unknown-quantile', type=float, default=0.05, help='unknown_thr 取 val 最大概率的该分位数')
    p.add_argument('--unknown-min', type=float, default=0.5)
    p.add_argument('--unknown-max', type=float, default=0.8)
    p.add_argument('--out', default='checkpoints/classifier')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    args = p.parse_args()

    log = get_logger('train_classifier')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    if not cfg.train_images or not os.path.isdir(cfg.train_images):
        raise FileNotFoundError(f'训练图片目录不存在: {cfg.train_images}')
    train_classifier(train_images=cfg.train_images, train_labels=cfg.train_labels,
                     val_images=cfg.val_images if cfg.val_images and os.path.isdir(cfg.val_images) else None,
                     val_labels=cfg.val_labels, out_dir=args.out, nc=cfg.nc, names=cfg.names, arch=args.arch,
                     crop_size=args.crop_size, crop_pad=args.crop_pad, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, val_ratio=args.val_ratio,
                     unknown_quantile=args.unknown_quantile, unknown_min=args.unknown_min,
                     unknown_max=args.unknown_max, pretrained=not args.no_pretrained, device=args.device,
                     num_workers=args.num_workers, seed=args.seed, logger=log)


if __name__ == '__main__':
    main()
