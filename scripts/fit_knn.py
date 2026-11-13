"""免训练 kNN 记忆库检测器 (DINOv2 中层 patch + coreset): 几分钟出一个能用的检测器, 并立刻在 val 上评估.

用途: ① 当 Dinomaly 的快速基线 (看数据本身是否可分); ② 实时性要求高时直接上线 (不跑 decoder);
      ③ 数据/布样变了, 几分钟重建记忆库, 不用重训.

示例:
  python scripts/fit_knn.py --data-yaml dataset/fabric19.yaml --out checkpoints/knn
  # 比较几组层 (每组都建库+评估, 保留最好的存成 model_best.pth):
  python scripts/fit_knn.py --layer-sets "5,7,9;8,10;11"
  # 然后与 Dinomaly 完全相同的下游流程:
  python scripts/calibrate.py --det-ckpt checkpoints/knn/model_best.pth --out checkpoints/knn/thresholds.json
  python scripts/infer.py --det-ckpt checkpoints/knn/model_best.pth --thresholds checkpoints/knn/thresholds.json ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import random
    import numpy as np
    from src.data import load_fabric_config
    from src.data.yolo import NormalImageDataset
    from src.detect import AnomalyDetector, evaluate_detector, get_eval_transform
    from src.knn_detect import build_knn, fit_knn
    from src.util import get_logger, setup_seed
    p = argparse.ArgumentParser(description='免训练 kNN 记忆库检测器')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--out', default='checkpoints/knn')
    p.add_argument('--backbone', default='dinov2reg_vit_small_14')
    p.add_argument('--image-size', type=int, default=448, help='须为 14 的倍数; 校准/推理沿用 (默认 448, 不裁边)')
    p.add_argument('--layer-sets', default='5,7,9',
                   help='用哪几层 encoder 特征 (0-based block 序号, 逗号=多层平均, 分号=多组候选并择优). 默认 5,7,9')
    p.add_argument('--bank-size', type=int, default=20000, help='coreset 记忆库大小 (越大越准, 但最近邻搜索耗时与它成正比; 448 下每张图约 1024xbank_size 次点积)')
    p.add_argument('--max-images', type=int, default=300, help='最多用多少张正常图建库 (随机抽, 布匹纹理重复, 300 张通常够)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--smooth-sigma', type=float, default=4.0)
    p.add_argument('--eval-max-images', type=int, default=0, help='评估每类最多用多少张 (0=全部)')
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--weights-dir', default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true', help='仅冒烟测试')
    args = p.parse_args()

    log = get_logger('fit_knn')
    setup_seed(args.seed)
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    log(cfg.summary())
    normal_dir = cfg.normaltrain_images
    if not normal_dir or not os.path.isdir(normal_dir):
        raise FileNotFoundError(f'纯正常训练集不存在: {normal_dir}')
    ds = NormalImageDataset(normal_dir, verify=True, logger=log)
    ids = list(range(len(ds)))
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.max_images] if args.max_images else ids
    log(f'用 {len(ids)}/{len(ds)} 张正常图建记忆库')

    class _Sub:
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            return ds[ids[i]]

    tf = get_eval_transform(args.image_size, args.image_size)
    os.makedirs(args.out, exist_ok=True)
    best, best_key = None, None
    sets = [tuple(int(v) for v in s.split(',') if v.strip()) for s in args.layer_sets.split(';') if s.strip()]
    for layers in sets:
        log(f'===== layers={layers} =====')
        model = build_knn(args.backbone, layers=layers, weights_dir=args.weights_dir,
                          allow_random_init=args.allow_random_init, logger=log)
        fit_knn(model, _Sub(), tf, device=args.device, bank_size=args.bank_size, batch_size=args.batch_size,
                seed=args.seed, logger=log)
        hp = {'arch': 'knn', 'backbone': args.backbone, 'image_size': args.image_size, 'crop_size': args.image_size,
              'layers': list(layers), 'bank_size': int(model.bank.shape[0]), 'smooth_sigma': args.smooth_sigma}
        det = AnomalyDetector(model, args.device, args.image_size, args.image_size, smooth_sigma=args.smooth_sigma)
        key = (0.0, 0.0)
        if not args.no_eval and cfg.val_images and os.path.isdir(cfg.val_images):
            st = evaluate_detector(model, cfg.val_images, cfg.val_labels, args.device, args.image_size,
                                   args.image_size, logger=log, normal_dir=cfg.normalval_images,
                                   smooth_sigma=args.smooth_sigma, max_images=args.eval_max_images, return_stats=True)
            a, g = st['i_auroc'], st['gap_rel']
            key = (a if a == a else 0.0, g if g == g else -1e9)
        if best_key is None or key > best_key:
            best_key = key
            det.save(os.path.join(args.out, 'model_best.pth'), hp)
            log(f'*** 当前最优 layers={layers} key={key}, 已保存 {os.path.join(args.out, "model_best.pth")}')
    log(f'完成. 下一步: python scripts/calibrate.py --det-ckpt {os.path.join(args.out, "model_best.pth")} '
        f'--out {os.path.join(args.out, "thresholds.json")} --sweep')


if __name__ == '__main__':
    main()
