"""训练异常检测器 (Dinomaly2, 无监督, 只用纯正常样本).

示例 (1660 机器):
  python scripts/train_detector.py --data-yaml dataset/fabric19.yaml \
      --backbone dinov2reg_vit_small_14 --image-size 448 --crop-size 392 \
      --total-iters 40000 --batch-size 16 --out checkpoints/detector
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_fabric_config
from src.detect import train_detector
from src.util import get_logger, setup_seed


def main():
    p = argparse.ArgumentParser(description='训练 Dinomaly2 异常检测器 (仅用正常样本)')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None, help='数据集根目录覆盖 (也可用环境变量 FABRIC19_ROOT)')
    p.add_argument('--backbone', default='dinov2reg_vit_small_14',
                   help='dinov2reg_vit_small_14(推荐,1660实时) / dinov2reg_vit_base_14 / tiny_test_14')
    p.add_argument('--image-size', type=int, default=448)
    p.add_argument('--crop-size', type=int, default=392)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--la', type=int, default=1, help='线性注意力 1/0')
    p.add_argument('--lc', type=int, default=2, help='loose constraint 分组 0/1/2/3/4/11/12/14')
    p.add_argument('--cr', type=int, default=1, help='context-aware recenter 1/0')
    p.add_argument('--ll', type=int, default=1, help='loose loss 1/0')
    p.add_argument('--ll-ratio', type=float, default=0.9)
    p.add_argument('--ll-factor', type=float, default=0.1)
    p.add_argument('--total-iters', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--eval-interval', type=int, default=5000)
    p.add_argument('--out', default='checkpoints/detector')
    p.add_argument('--weights-dir', default=None, help='backbone 预训练权重目录')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true', help='权重下载失败时随机初始化 (仅冒烟测试)')
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    log = get_logger('train_detector')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    normal_train = cfg.normaltrain_images
    if not normal_train or not os.path.isdir(normal_train):
        raise FileNotFoundError(f'纯正常训练集不存在: {normal_train} (请检查 {args.data_yaml} 中的 normaltrain 路径)')
    train_detector(normal_train_dir=normal_train, out_dir=args.out, backbone=args.backbone,
                   image_size=args.image_size, crop_size=args.crop_size, dropout=args.dropout, la=args.la,
                   lc=args.lc, cr=args.cr, ll=args.ll, ll_ratio=args.ll_ratio, ll_factor=args.ll_factor,
                   total_iters=args.total_iters, batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                   eval_interval=args.eval_interval, eval_dir=cfg.val_images, eval_labels_dir=cfg.val_labels,
                   device=args.device, num_workers=args.num_workers, seed=args.seed,
                   weights_dir=args.weights_dir, allow_random_init=args.allow_random_init,
                   resume=args.resume, logger=log)


if __name__ == '__main__':
    main()
