"""实时性 benchmark: 测量检测器 / 分类器 / 整体流水线的延迟与 FPS.

在 1660 上运行:
  python scripts/benchmark.py --data-yaml dataset/fabric19.yaml --device cuda:0
  python scripts/benchmark.py --device cuda:0 --half --image-size 336   # 快速档位

会打印每个环节的 ms/图 与 FPS, 以及模型参数量 / FLOPs (用 fvcore 可选, 未装则跳过).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def _flops(model, input_shape, device):
    try:
        from thop import profile
        x = torch.randn(*input_shape, device=device)
        macs, params = profile(model, inputs=(x,), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            x = torch.randn(*input_shape, device=device)
            return FlopCountAnalysis(model, x).total() / 1e9
        except Exception:
            return None


def main():
    import cv2
    import numpy as np
    import torch
    from src.data import load_fabric_config
    from src.detect import AnomalyDetector, load_json
    from src.classify import RegionClassifier
    from src.pipeline import FabricAnomalyPipeline
    from src.util import AverageMeter, get_logger
    p = argparse.ArgumentParser(description='推理速度 benchmark')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--det-ckpt', default='checkpoints/detector/model_best.pth')
    p.add_argument('--cls-ckpt', default='checkpoints/classifier/best.pt')
    p.add_argument('--thresholds', default='checkpoints/detector/thresholds.json')
    p.add_argument('--image-size', type=int, default=None)
    p.add_argument('--half', action='store_true')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--batch-crops', type=int, default=8, help='模拟每图的缺陷区域数')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()

    log = get_logger('benchmark')
    if args.device.startswith('cuda') and torch.cuda.is_available():
        log(f'GPU: {torch.cuda.get_device_name(0)}  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    class_names = cfg.class_names

    det = AnomalyDetector.load(args.det_ckpt, device=args.device, allow_random_init=args.allow_random_init,
                               logger=log, half=args.half)
    if args.image_size and args.image_size != det.image_size:
        det.image_size = args.image_size
        det.crop_size = args.image_size
        det.offset = (det.image_size - det.crop_size) // 2
        from src.detect import get_eval_transform
        det.tf = get_eval_transform(det.image_size, det.crop_size)
    thr = load_json(args.thresholds) if os.path.exists(args.thresholds) else {'img_thr': 0.5, 'pix_thr': 0.25}
    cls = RegionClassifier.load(args.cls_ckpt, nc=len(class_names) - 1, names=class_names,
                                device=args.device, logger=log) if os.path.exists(args.cls_ckpt) else None
    pipe = FabricAnomalyPipeline(det, cls, class_names, img_thr=thr['img_thr'], pix_thr=thr.get('pix_thr'),
                                 device=args.device, logger=log)

    # 随机图像
    rng = np.random.default_rng(0)
    img = (rng.random((640, 640, 3)) * 255).astype(np.uint8)

    # detector
    for _ in range(args.warmup):
        det.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        det.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t_det = (time.perf_counter() - t0) / args.iters * 1000

    # classifier: 模拟 batch-crops 个区域
    crop = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    crops = [crop] * args.batch_crops
    for _ in range(args.warmup):
        cls.predict_probs(crops) if cls else None
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        if cls:
            cls.predict_probs(crops)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t_cls = (time.perf_counter() - t0) / args.iters * 1000 if cls else 0.0

    # 整体流水线
    for _ in range(args.warmup):
        pipe.process(img, 'bench')
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        pipe.process(img, 'bench')
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    t_pipe = (time.perf_counter() - t0) / args.iters * 1000

    gflops = _flops(det.model, (1, 3, det.crop_size, det.crop_size), args.device)
    n_params = sum(q.numel() for q in det.model.parameters())
    n_train = sum(q.numel() for q in det.model.parameters() if q.requires_grad)

    log('==================== benchmark ====================')
    log(f'device={args.device} half={args.half} image_size={det.image_size} crop_size={det.crop_size}')
    log(f'detector: {t_det:.1f} ms/img -> {1000 / t_det:.1f} FPS'
        f'   (总参数 {n_params / 1e6:.1f}M, 可训练 {n_train / 1e6:.2f}M'
        + (f', 约 {gflops:.1f} GFLOPs/img' if gflops else '') + ')')
    if cls:
        log(f'classifier: {t_cls:.1f} ms/{args.batch_crops} crops -> {1000 / t_cls:.1f} FPS '
            f'({t_cls / args.batch_crops:.2f} ms/crop)')
    log(f'pipeline: {t_pipe:.1f} ms/img -> {1000 / t_pipe:.1f} FPS')
    log('===================================================')
    log('GTX 1660 (6GB, ~5 TFLOPS FP32 / ~10 TFLOPS FP16) 参考: '
        'dinov2reg_vit_small_14 @448 约 30-60 FPS, @336 约 80-120 FPS, 整体流水线 >15 FPS 即满足实时; '
        '加 --half 可再翻倍.')


if __name__ == '__main__':
    main()
