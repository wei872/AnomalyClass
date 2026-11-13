"""免训练 kNN 记忆库检测器 (DINOv2 中层 patch + coreset). 已并入 train_detector.py (训练结束自动建库并与 Dinomaly 融合),
本脚本用于单独重建记忆库 (数据/布样变了, 几分钟重建, 不用重训):

  python scripts/fit_knn.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector/knn_best.pth
  python scripts/fit_knn.py --layer-sets "5,7,9;3,5,7"      # 多组层各建库并在 val 评估, 保留最好的
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.data import load_fabric_config
    from src.knn_detect import fit_knn_best
    from src.util import get_logger, setup_seed
    p = argparse.ArgumentParser(description='免训练 kNN 记忆库检测器')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--out', default='checkpoints/detector/knn_best.pth')
    p.add_argument('--backbone', default='dinov2reg_vit_small_14')
    p.add_argument('--image-size', type=int, default=448, help='须为 14 的倍数 (默认 448, 不裁边)')
    p.add_argument('--layer-sets', default='5,7,9;3,5,7',
                   help='encoder 层 (0-based, 逗号=多层平均, 分号=多组候选并择优); 要与 Dinomaly 融合须在 2~9')
    p.add_argument('--bank-size', type=int, default=20000, help='coreset 记忆库大小 (越大越准, 推理最近邻耗时与它成正比)')
    p.add_argument('--max-images', type=int, default=300, help='最多用多少张正常图建库 (布匹纹理重复, 300 张通常够)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--smooth-sigma', type=float, default=4.0)
    p.add_argument('--eval-max-images', type=int, default=0, help='评估每类最多用多少张 (0=全部)')
    p.add_argument('--weights-dir', default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true', help='仅冒烟测试')
    args = p.parse_args()
    log = get_logger('fit_knn')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    if not cfg.normaltrain_images or not os.path.isdir(cfg.normaltrain_images):
        raise FileNotFoundError(f'纯正常训练集不存在: {cfg.normaltrain_images}')
    fit_knn_best(cfg.normaltrain_images, args.out, backbone=args.backbone, image_size=args.image_size,
                 layer_sets=args.layer_sets, bank_size=args.bank_size, max_images=args.max_images,
                 batch_size=args.batch_size, smooth_sigma=args.smooth_sigma, val_images=cfg.val_images,
                 val_labels=cfg.val_labels, normal_val_dir=cfg.normalval_images,
                 eval_max_images=args.eval_max_images, device=args.device, seed=args.seed,
                 weights_dir=args.weights_dir, allow_random_init=args.allow_random_init, logger=log)
    log(f'完成: {args.out}')


if __name__ == '__main__':
    main()
