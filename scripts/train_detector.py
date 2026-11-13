"""训练异常检测器 (Dinomaly2): 正常样本 + 带框缺陷样本联合训练 (同一 batch 混合, 不是分开训练).
正常图走重建损失, 缺陷图走监督 margin 损失 (图级/框级 hinge), 直接推开正常/缺陷分数分布. 分辨率默认 448.

示例 (1660 机器, 直接重训):
  python scripts/train_detector.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector
  # 断点续训: 加 --resume checkpoints/detector/model_last.pth
  # 消融对照 (关闭缺陷监督): 加 --no-defect-sup
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.data import load_fabric_config
    from src.detect import train_detector
    from src.util import get_logger, setup_seed
    p = argparse.ArgumentParser(description='训练 Dinomaly2 异常检测器 (仅用正常样本, 默认 448)')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None, help='数据集根目录覆盖 (也可用环境变量 FABRIC19_ROOT)')
    p.add_argument('--backbone', default='dinov2reg_vit_small_14',
                   help='dinov2reg_vit_small_14(推荐,1660@448实时) / dinov2reg_vit_base_14 / tiny_test_14(仅冒烟测试)')
    p.add_argument('--image-size', type=int, default=448)
    p.add_argument('--crop-size', type=int, default=448, help='网络输入尺寸, 默认等于image_size(不裁边缘, 暗角等正常边缘要学到); 须为14的倍数')
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--la', type=int, default=1, help='线性注意力 1/0')
    p.add_argument('--lc', type=int, default=2, help='loose constraint 分组 0/1/2/3/4/11/12/14')
    p.add_argument('--cr', type=int, default=1, help='context-aware recenter 1/0')
    p.add_argument('--ll', type=int, default=1, help='loose loss 1/0')
    p.add_argument('--ll-ratio', type=float, default=0.9)
    p.add_argument('--ll-factor', type=float, default=0.1)
    p.add_argument('--total-iters', type=int, default=20000, help='1660 上 bs=16 约 2s/iter, 20000 约 11h; 可 --resume 续训')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--eval-interval', type=int, default=5000,
                   help='每 N 个 iter 在 val(缺陷图 + normalval 正常图) 上评估并按 (I-AUROC, 相对间隙) 更新 model_best')
    p.add_argument('--save-interval', type=int, default=1000,
                   help='每 N 个 iter 无条件保存 model_last.pth (即使评估失败也不丢权重)')
    p.add_argument('--no-eval', action='store_true', help='训练期间跳过 val 评估 (只保存权重)')
    p.add_argument('--local-loss-w', type=float, default=0.0,
                   help='局部逐位置cosine损失权重 (细长经线缺陷建议 1.0): 原全局cosine会把细线信号稀释')
    p.add_argument('--smooth-sigma', type=float, default=4.0,
                   help='异常图高斯平滑核sigma: 调大(如8)可压正常图单点噪声、保留成片/成线缺陷, 缓解分数重叠')
    p.add_argument('--hard-pos-ratio', type=float, default=0.0,
                   help='配合--local-loss-w: 只对距离最大的top-k比例位置做损失 (如 0.2), hard mining')
    p.add_argument('--margin-w', type=float, default=1.0,
                   help='图级 margin 损失权重: 缺陷图分数须比同batch正常图高 --margin (hinge), 直接拉开分布')
    p.add_argument('--margin', type=float, default=0.05, help='margin 目标间隙 (cos 距离量级, 你的分数约0.02~0.05)')
    p.add_argument('--box-margin-w', type=float, default=1.0,
                   help='框级 margin: 缺陷框内 token 平均距离须 >= margin, 直接抬高 min_defect (细线类关键)')
    p.add_argument('--hard-normal-w', type=float, default=0.0,
                   help='困难正常重加权: 分数偏高的正常图重建损失加大, 压 max_normal (如 0.5)')
    p.add_argument('--local-loss-warmup', type=int, default=0,
                   help='局部损失线性预热 iters (0=不预热), 如 2000 避免初期不稳')
    p.add_argument('--no-defect-sup', action='store_true',
                   help='关闭缺陷监督 (margin/box 全关), 仅消融对照用; 正式训练保持正常+缺陷联合')
    p.add_argument('--no-rot90', action='store_true',
                   help='关闭训练增强里的随机 90° 旋转. 布匹经纬方向固定时建议关: 否则经线方向纹理也被当作正常, 细长经/纬线缺陷变钝')
    p.add_argument('--no-flip', action='store_true', help='关闭训练增强里的随机翻转 (官方 2D 配方 = 无任何增强: --no-rot90 --no-flip)')
    p.add_argument('--optimizer', choices=['stableadamw', 'adamw'], default='stableadamw',
                   help='stableadamw=官方 Dinomaly2 配方 (默认, 抑制 loss 尖峰/坏梯度整步跳过); adamw=旧行为')
    p.add_argument('--lr-final-ratio', type=float, default=0.0,
                   help='余弦退火终点 = lr*该值. 0=退火到0 (默认); 1=warmup 后恒定 (官方默认)')
    p.add_argument('--defect-frac', type=float, default=-1,
                   help='每个 batch 的缺陷图期望占比: -1=自动(自然占比<20%%时抬到25%%, 否则自然) / 0=自然 / 0.25=指定. '
                        '缺陷图太少时 margin/box 损失经常整步没有正样本')
    p.add_argument('--margin-score', choices=['smooth', 'token'], default='smooth',
                   help="图级 hinge 用的分数: smooth=与推理/校准同款 (上采样+高斯平滑后取最大, 默认); token=旧做法(token网格未平滑)")
    p.add_argument('--defect-bg-w', type=float, default=0.0,
                   help='>0: 缺陷图框外 token(正常纹理) 也参与重建损失, 多用一份正常数据 (如 0.5); 默认关')
    p.add_argument('--eval-max-images', type=int, default=0, help='训练中评估每类最多用多少张 (0=全部)')
    p.add_argument('--out', default='checkpoints/detector')
    p.add_argument('--weights-dir', default=None, help='backbone 预训练权重目录 (默认 backbones/weights)')
    p.add_argument('--num-workers', type=int,
                   default=(0 if sys.platform == 'win32' else 4),
                   help='Windows 下默认 0 (主进程加载, 避免 spawn 子进程重 import 卡顿), 可手动指定')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true', help='权重下载失败时随机初始化 (仅冒烟测试)')
    p.add_argument('--resume', default=None, help='从 checkpoint 续训, 如 checkpoints/detector/model_last.pth')
    args = p.parse_args()

    log = get_logger('train_detector')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    normal_train = cfg.normaltrain_images
    if not normal_train or not os.path.isdir(normal_train):
        raise FileNotFoundError(f'纯正常训练集不存在: {normal_train} (请检查 {args.data_yaml} 中的 normaltrain 路径)')
    defect_dir = None if args.no_defect_sup else cfg.train_images
    defect_labels = None if args.no_defect_sup else cfg.train_labels
    train_detector(normal_train_dir=normal_train, out_dir=args.out, backbone=args.backbone,
                   image_size=args.image_size, crop_size=args.crop_size, dropout=args.dropout, la=args.la,
                   lc=args.lc, cr=args.cr, ll=args.ll, ll_ratio=args.ll_ratio, ll_factor=args.ll_factor,
                   total_iters=args.total_iters, batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                   eval_interval=args.eval_interval, eval_dir=cfg.val_images, eval_labels_dir=cfg.val_labels,
                   device=args.device, num_workers=args.num_workers, seed=args.seed,
                   local_loss_w=args.local_loss_w, hard_pos_ratio=args.hard_pos_ratio,
                   smooth_sigma=args.smooth_sigma,
                   defect_dir=defect_dir, defect_labels_dir=defect_labels,
                   margin_w=args.margin_w, margin=args.margin, box_margin_w=args.box_margin_w,
                   hard_normal_w=args.hard_normal_w, local_loss_warmup=args.local_loss_warmup,
                   rot90=not args.no_rot90, flip=not args.no_flip, optimizer=args.optimizer,
                   lr_final_ratio=args.lr_final_ratio, defect_frac=args.defect_frac, margin_score=args.margin_score, defect_bg_w=args.defect_bg_w,
                   eval_normal_dir=cfg.normalval_images, eval_max_images=args.eval_max_images,
                   weights_dir=args.weights_dir, allow_random_init=args.allow_random_init,
                   resume=args.resume, logger=log, save_interval=args.save_interval, no_eval=args.no_eval)


if __name__ == '__main__':
    main()
