"""训练异常区域分类器 (YOLO 框裁剪 -> 19 类缺陷; 低置信度自动归为 "未知类型").

默认 --kind embed: 冻结 DINOv2 嵌入 + 余弦头 + Balanced Softmax (小样本/类别不均衡友好, 训练只要几分钟);
--kind cnn: 旧方案, 端到端微调 MobileNetV3 等 (逐 epoch 读图, 慢). 两者可用 evaluate.py --mode gt 做 A/B 对比.

示例:
  python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml --out checkpoints/classifier
  python scripts/train_classifier.py --kind cnn --arch mobilenet_v3_small --epochs 50 --out checkpoints/classifier_cnn
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def main():
    from src.classify import train_classifier
    from src.data import load_fabric_config
    from src.util import get_logger, setup_seed
    p = argparse.ArgumentParser(description='训练缺陷区域分类器 (mobilenet_v3_small 推荐, 1660 实时)')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--kind', default='embed', choices=['embed', 'cnn'],
                   help='embed=DINOv2 冻结嵌入+余弦头 (默认); cnn=端到端微调小 CNN (旧方案)')
    p.add_argument('--backbone', default='dinov2reg_vit_small_14', help='[embed] 嵌入骨干 (与检测器共用同一份权重文件)')
    p.add_argument('--views', type=int, default=4, help='[embed] 每个训练裁剪至少抽几个视角 (1 干净 + 其余增强)')
    p.add_argument('--min-per-class', type=int, default=200,
                   help='[embed] 稀有类按此目标多抽增强视角 (特征级过采样, 最多 16 个视角)')
    p.add_argument('--tau', type=float, default=1.0, help='[embed] logit adjustment 强度: 1=Balanced Softmax, 0=普通 CE')
    p.add_argument('--head-epochs', type=int, default=200, help='[embed] 分类头训练轮数 (特征已缓存, 很快)')
    p.add_argument('--weights-dir', default=None, help='[embed] backbone 权重目录 (默认 backbones/weights)')
    p.add_argument('--allow-random-init', action='store_true', help='[embed] 仅冒烟测试')
    p.add_argument('--arch', default='mobilenet_v3_small',
                   choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet34',
                            'efficientnet_b0'],
                   help='精度优先可选 resnet18/efficientnet_b0 (1660 上仍实时); 速度优先 mobilenet_v3_small')
    p.add_argument('--crop-size', type=int, default=224, help='分类器 letterbox 目标尺寸, 训练与推理用同一个(不用调)')
    p.add_argument('--crop-pad', type=float, default=0.3, help='裁剪时四周扩展比例')
    p.add_argument('--epochs', type=int, default=50, help='默认 50, 配合早停(patience=15)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-ratio', type=float, default=0.1, help='未提供 val 时从 train 划分比例')
    p.add_argument('--unknown-quantile', type=float, default=0.05, help='unknown_thr 取 val 最大概率的该分位数')
    p.add_argument('--unknown-min', type=float, default=0.5)
    p.add_argument('--unknown-max', type=float, default=0.8)
    p.add_argument('--box-jitter', type=float, default=0.3,
                   help='训练裁剪时对 GT 框随机缩放/平移的幅度 (0=关), 模拟推理时检测器给出的框 (默认 0.3)')
    p.add_argument('--rot90', action='store_true',
                   help='加随机 90° 旋转增强. 默认关: 经线/纬线类别靠方向区分, 旋转会让纬向缺陷变成经向 (标签噪声)')
    p.add_argument('--balance-power', type=float, default=1.0,
                   help='类别均衡强度 (仅加权采样, 权重∝n_c^-power): 1=完全均衡, 0.5=折中, 0=自然分布')
    p.add_argument('--out', default='checkpoints/classifier')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--pretrained-path', default=None,
                   help='本地预训练权重文件 (如手动下载的 efficientnet_b0_rwightman-7f5810bc.pth), 优先于在线下载')
    p.add_argument('--num-workers', type=int,
                   default=(0 if sys.platform == 'win32' else 4),
                   help='Windows 下默认 0 (主进程加载, 避免 spawn 子进程重 import 卡顿), 可手动指定')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    args = p.parse_args()

    log = get_logger('train_classifier')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    if not cfg.train_images or not os.path.isdir(cfg.train_images):
        raise FileNotFoundError(f'训练图片目录不存在: {cfg.train_images}')
    if args.kind == 'embed':
        from src.embed_classify import train_embed_classifier
        train_embed_classifier(train_images=cfg.train_images, train_labels=cfg.train_labels,
                               val_images=cfg.val_images if cfg.val_images and os.path.isdir(cfg.val_images) else None,
                               val_labels=cfg.val_labels, out_dir=args.out, nc=cfg.nc, names=cfg.names,
                               backbone=args.backbone, crop_size=args.crop_size, crop_pad=args.crop_pad,
                               views=args.views, min_per_class=args.min_per_class, box_jitter=args.box_jitter,
                               epochs=args.head_epochs, tau=args.tau, val_ratio=args.val_ratio,
                               unknown_quantile=args.unknown_quantile, unknown_min=args.unknown_min,
                               unknown_max=args.unknown_max, device=args.device, num_workers=args.num_workers,
                               seed=args.seed, logger=log, weights_dir=args.weights_dir,
                               allow_random_init=args.allow_random_init)
        return
    train_classifier(train_images=cfg.train_images, train_labels=cfg.train_labels,
                     val_images=cfg.val_images if cfg.val_images and os.path.isdir(cfg.val_images) else None,
                     val_labels=cfg.val_labels, out_dir=args.out, nc=cfg.nc, names=cfg.names, arch=args.arch,
                     crop_size=args.crop_size, crop_pad=args.crop_pad, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, val_ratio=args.val_ratio,
                     unknown_quantile=args.unknown_quantile, unknown_min=args.unknown_min,
                     unknown_max=args.unknown_max, pretrained=not args.no_pretrained,
                     pretrained_path=args.pretrained_path, device=args.device,
                     num_workers=args.num_workers, seed=args.seed, logger=log,
                     box_jitter=args.box_jitter, rot90=args.rot90, balance_power=args.balance_power)


if __name__ == '__main__':
    main()
