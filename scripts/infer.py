"""推理入口: 单图 / 图片目录 / 视频文件 / 摄像头实时检测 + 分类.

最短用法 (训练完直接用, 检测器/阈值/分类器路径都有默认值):
  python scripts/infer.py --input 图片或目录或视频 [--out runs/infer]
  --input 自动识别: 文件夹 / 图片文件 / 视频文件 / 数字(摄像头编号). 也可以分别用 --image/--dir/--video/--camera.
  默认: 检测器 checkpoints/detector/model_best.pth, 阈值取检测器同目录 thresholds.json, 分类器 checkpoints/classifier/best.pt
  (分类器不存在时只检测不分类; --no-classify 也可强制只检测)

示例:
  # 单图
  python scripts/infer.py --image test.jpg --out runs/infer
  # 目录
  python scripts/infer.py --dir images/ --out runs/infer
  # 视频 / 摄像头 (实时, 显示 FPS)
  python scripts/infer.py --video fabric.mp4 --out runs/demo.mp4
  python scripts/infer.py --camera 0 --show

输出: runs/infer 下每张图的可视化 PNG + results.json
  results.json 每图: {is_anomaly, anomaly_conf(与正常偏离置信度), regions:[{box, class_name,
                      class_conf, anomaly_conf, ...}]}, 其中 class_id==19 为 "未知类型".
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def main():
    import cv2
    from src.data import load_fabric_config
    from src.pipeline import build_pipeline
    from src.util import FpsMeter, get_logger, list_images, imread_unicode, imwrite_unicode
    p = argparse.ArgumentParser(description='布匹异常检测 + 缺陷分类推理')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--input', default=None, help='图片 / 图片目录 / 视频文件 / 摄像头编号 (自动识别, 等价于 --image/--dir/--video/--camera)')
    p.add_argument('--det-ckpt', default=None, help='默认 checkpoints/detector/model_final.pth (没有则 model_best.pth)')
    p.add_argument('--cls-ckpt', default='checkpoints/classifier/best.pt')
    p.add_argument('--no-classify', action='store_true',
                   help='纯正异常检测: 不加载分类器, 区域只定位(标未知类型/classified=false), 更快')
    p.add_argument('--thresholds', default=None, help='默认取检测器权重同目录的 thresholds.json')
    p.add_argument('--image', default=None)
    p.add_argument('--dir', default=None)
    p.add_argument('--video', default=None)
    p.add_argument('--camera', type=int, default=None)
    p.add_argument('--out', default='runs/infer')
    p.add_argument('--image-size', type=int, default=None, help='覆盖检测输入尺寸 (必须与训练一致, 默认 448, 慎用)')
    p.add_argument('--top-k', type=int, default=1,
                   help='只对最异常的 top_k 个热力区域做分类 (0=全部分类), 默认 1')
    p.add_argument('--smooth-sigma', type=float, default=0, help='覆盖异常图平滑核 (默认 0 = 自动沿用 thresholds.json 里校准时的值, 一般不用设)')
    p.add_argument('--region-score-ratio', type=float, default=0.5,
                   help='区域掩码阈值 = max(全局阈值, 图内最高分*该比例), 只圈最热区域')
    p.add_argument('--half', action='store_true', help='FP16 推理 (GTX 1660 上约 2x 加速)')
    p.add_argument('--show', action='store_true', help='弹出窗口显示 (视频/摄像头)')
    p.add_argument('--font', default=None, help='中文字体文件路径 (可选)')
    p.add_argument('--no-heatmap', action='store_true')
    p.add_argument('--heat-floor-ratio', type=float, default=1.0,
                   help='热力图显示下限 = pix_thr * 该值, 低于它的像素不上色; 噪声还多就调大 (如 1.3~1.6), 淡了就调小')
    p.add_argument('--heat-region-only', action='store_true', help='热力图只显示在检出的区域框内')
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()
    from src.detect import resolve_det_ckpt
    args.det_ckpt = resolve_det_ckpt(args.det_ckpt)

    log = get_logger('infer')
    if args.input is not None:
        inp = args.input
        if os.path.isdir(inp):
            args.dir = inp
        elif inp.isdigit():
            args.camera = int(inp)
        elif inp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            args.video = inp
        else:
            args.image = inp
    if not (args.image or args.dir or args.video or args.camera is not None):
        p.error('请用 --input 指定图片 / 目录 / 视频 / 摄像头编号')
    if not os.path.exists(args.det_ckpt):
        raise SystemExit(f'找不到检测器权重: {args.det_ckpt}\n请先训练: python scripts/train_detector.py (或用 --det-ckpt 指定)')
    if args.thresholds is None:
        args.thresholds = os.path.join(os.path.dirname(os.path.abspath(args.det_ckpt)), 'thresholds.json')
    if not os.path.exists(args.thresholds):
        raise SystemExit(f'找不到阈值文件: {args.thresholds}\n请先校准阈值: '
                         f'python scripts/calibrate.py --det-ckpt {args.det_ckpt}')
    if not args.no_classify and not os.path.exists(args.cls_ckpt):
        log(f'⚠️ 找不到分类器权重 {args.cls_ckpt}, 本次只检测不分类 (先运行 python scripts/train_classifier.py)')
        args.no_classify = True
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    class_names = cfg.class_names
    log(f'类别({len(class_names)}): {class_names}')
    cls_ckpt = None if args.no_classify else args.cls_ckpt
    pipe = build_pipeline(args.det_ckpt, cls_ckpt, args.thresholds, class_names,
                          device=args.device, half=args.half, image_size=args.image_size, logger=log,
                          top_k=args.top_k, region_score_ratio=args.region_score_ratio,
                          heat_floor_ratio=args.heat_floor_ratio, heat_region_only=args.heat_region_only)
    if args.smooth_sigma and abs(args.smooth_sigma - pipe.det.smooth_sigma) > 1e-6:
        pipe.det.set_smooth_sigma(args.smooth_sigma)
        log(f'[infer] ⚠️ 手工覆盖平滑核 sigma={args.smooth_sigma} (阈值是在别的 sigma 下校准的, 须重新 calibrate)')
    log(f'img_thr={pipe.img_thr:.4f} mask_thr={pipe.mask_thr:.4f} device={args.device} half={args.half}')
    os.makedirs(args.out, exist_ok=True)

    def run_one(img_bgr, name):
        res = pipe.process(img_bgr, name)
        vis = pipe.visualize(img_bgr, res, font_path=args.font, show_heatmap=not args.no_heatmap)
        out_img = os.path.join(args.out, os.path.splitext(os.path.basename(name))[0] + '_vis.png')
        imwrite_unicode(out_img, vis)   # 支持中文文件名
        d = res.to_dict()
        d['vis'] = out_img
        log(f'{name}: is_anomaly={d["is_anomaly"]} anomaly_conf={d["anomaly_conf"]:.3f} '
            f'latency={d["latency_ms"]:.1f}ms regions={len(d["regions"])}')
        for r in d['regions']:
            log(f'   - box={r["box"]} class={r["class_name"]}({r["class_id"]}) '
                f'class_conf={r["class_conf"]:.3f} anomaly_conf={r["anomaly_conf"]:.3f}')
        return d

    if args.image:
        img = imread_unicode(args.image, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {args.image}')
        d = run_one(img, os.path.basename(args.image))
        _append_json(os.path.join(args.out, 'results.json'), d)
    elif args.dir:
        results = []
        for f in list_images(args.dir):
            img = imread_unicode(f, cv2.IMREAD_COLOR)
            if img is None:
                log(f'跳过无法读取的图片: {f}')
                continue
            results.append(run_one(img, os.path.basename(f)))
        with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as fo:
            json.dump(results, fo, ensure_ascii=False, indent=2)
        log(f'共处理 {len(results)} 张, 结果: {os.path.join(args.out, "results.json")}')
    elif args.video or args.camera is not None:
        src = args.camera if args.camera is not None else args.video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频源: {src}')
        writer = None
        if args.video and args.out:
            fps0 = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(args.out, exist_ok=True)
            out_video = os.path.join(args.out, 'demo.mp4')
            writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps0, (w, h))
        meter = FpsMeter()
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pipe.process(frame, f'frame{n}')
            meter.tick()
            vis = pipe.visualize(frame, res, font_path=args.font, show_heatmap=not args.no_heatmap)
            cv2.putText(vis, f'FPS {meter.fps:.1f}  {res.latency_ms:.1f}ms', (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if writer:
                writer.write(vis)
            if args.show:
                cv2.imshow('AnomalyClass', vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if n % 30 == 0:
                log(f'frame {n}: anomaly={res.is_anomaly} conf={res.anomaly_conf:.3f} '
                    f'regions={len(res.regions)} FPS={meter.fps:.1f}')
            n += 1
        cap.release()
        if writer:
            writer.release()
            log(f'视频已保存: {out_video}')
        if args.show:
            cv2.destroyAllWindows()
        log(f'共 {n} 帧, 平均 FPS={meter.fps:.1f}')
    else:
        p.error('请指定 --image / --dir / --video / --camera 之一')


def _append_json(path, obj):
    data = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    data.append(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
