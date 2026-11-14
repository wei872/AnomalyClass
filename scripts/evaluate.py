"""用 YOLO 标注评估分类效果 (疵点框就是 GT).

两种模式:
  --mode gt        直接把 val 里每个 YOLO 疵点框裁出来喂分类器 -> 分类准确率/每类准确率/混淆矩阵/未知率
  --mode pipeline  跑完整流水线 (top_k=0 全部分类), 检测区域与 GT 框按 IoU 匹配 ->
                   图级判对率 + 检测框召回率 + 匹配区域上的分类准确率

示例:
  python scripts/evaluate.py --mode gt                          # 分类器验证: GT 框直接喂分类器 (不需要检测器)
  python scripts/evaluate.py --mode pipeline --no-classify      # 纯检测: 只评正/异常 + 检测框召回 (不需要分类器)
  python scripts/evaluate.py --mode pipeline --top-k 0          # 检测 + 分类端到端
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def iou(b1, b2):
    """(x1,y1,x2,y2) IoU."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def confusion_report(names, y_true, y_pred):
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print('混淆矩阵 (行=真实, 列=预测):')
    header = 'true\\pred ' + ' '.join(f'{i:>6d}' for i in range(n))
    print(header)
    for i in range(n):
        print(f'{i:>6d} {names[i][:6]:<6} ' + ' '.join(f'{v:>6d}' for v in cm[i]))
    per_cls = []
    for i in range(n):
        tot = cm[i].sum()
        per_cls.append((names[i], cm[i, i] / tot if tot else float('nan'), int(tot)))
    print('每类准确率:')
    for name, acc, tot in per_cls:
        print(f'  {name}: {acc:.3f} (n={tot})' if tot else f'  {name}: n=0')
    accs = [a for _, a, t in per_cls if t]
    if accs:
        print(f'macro 平均(各类召回均值, 不被多数类掩盖)={np.mean(accs):.4f}')
    pairs = sorted(((int(cm[i, j]), i, j) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0), reverse=True)
    if pairs:
        print('最常见的混淆 (真实 -> 预测, 次数, 占该真实类比例) — 先看这里决定是合并近义类、补数据还是改标注:')
        for c, i, j in pairs[:8]:
            print(f'  {names[i]} -> {names[j]}: {c} ({c / max(cm[i].sum(), 1):.0%})')


def gt_judge_eval(cls, pipe, images_dir, labels_dir, class_names, max_images, log):
    """二次判断分类器的 GT 验证: 每张图把全部 GT 框交给 judge (整图+区域+检测器热力), severe 视为 True (GT 框都当"严重").
    报告: 缺陷类准确率 (含被判未知/正常的算错)、被判未知比例、被判正常比例、混淆矩阵."""
    from src.data.yolo import YoloDefectDataset
    from src.judge import heat_to_grid
    nc = cls.nc
    ds = YoloDefectDataset(images_dir, labels_dir, max_side=1024)
    n = len(ds) if not max_images else min(max_images, len(ds))
    y_true, y_pred, n_norm = [], [], 0
    for i in range(n):
        img, boxes, _, _ = ds[i]
        if not boxes:
            continue
        heat = None
        if cls.meta.get('use_heat'):
            full_map, _, _ = pipe.det.predict(img, max_map_side=1024)
            heat = heat_to_grid(full_map, pipe.img_thr, cls.ctx_size)
        ana = cls.analyse(img, [b[1:] for b in boxes], heat=heat)
        for j, b in enumerate(boxes):
            cid, _, unk = cls.decide(ana['probs'][j], float(ana['p_nov'][j]), True)
            if ana['probs'][j][nc] >= cls.normal_thr:
                n_norm += 1
            y_true.append(int(b[0]))
            y_pred.append(cid)
        if (i + 1) % 20 == 0:
            log(f'  进度 {i + 1}/{n}')
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if not len(y_true):
        log('[gt分类] val 里没有缺陷框')
        return
    log(f'[gt分类/二次判断] 总准确率={float((y_true == y_pred).mean()):.4f}  被判未知类型={float((y_pred == nc).mean()):.4f}  '
        f'p(正常)>=normal_thr 的缺陷框={n_norm}/{len(y_true)}  (n={len(y_true)})')
    confusion_report(class_names, y_true, y_pred)


def main():
    import numpy as np
    from src.data import load_fabric_config
    from src.data.yolo import YoloDefectDataset
    from src.util import get_logger
    p = argparse.ArgumentParser(description='用 YOLO 标注评估分类/检测效果')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--mode', choices=['gt', 'pipeline'], default='gt',
                   help='gt=直接用YOLO疵点框评分类器; pipeline=端到端(检测框IoU匹配)')
    p.add_argument('--cls-ckpt', default='checkpoints/classifier/best.pt')
    p.add_argument('--no-classify', action='store_true', help='pipeline 模式: 纯检测评估, 不加载分类器')
    p.add_argument('--det-ckpt', default=None, help='默认 checkpoints/detector/model_final.pth (没有则 model_best.pth)')
    p.add_argument('--judge-override', action='store_true', help='pipeline 模式: 允许二次判断推翻异常 (默认关, 与不开的结果对比即为推翻的净影响)')
    p.add_argument('--thresholds', default=None, help='默认取检测器权重同目录的 thresholds.json')
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--crop-pad', type=float, default=0.3)
    p.add_argument('--iou-thr', type=float, default=0.3, help='pipeline 模式匹配 GT 框的 IoU 阈值')
    p.add_argument('--top-k', type=int, default=0, help='pipeline 模式 top_k (0=全部分类)')
    p.add_argument('--region-score-ratio', type=float, default=0.5)
    p.add_argument('--image-size', type=int, default=None)
    p.add_argument('--half', action='store_true')
    p.add_argument('--max-images', type=int, default=0, help='0=全部')
    p.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    p.add_argument('--allow-random-init', action='store_true')
    args = p.parse_args()
    from src.detect import resolve_det_ckpt
    args.det_ckpt = resolve_det_ckpt(args.det_ckpt)
    if args.thresholds is None:
        args.thresholds = os.path.join(os.path.dirname(os.path.abspath(args.det_ckpt)), 'thresholds.json')

    log = get_logger('evaluate')
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    class_names = cfg.class_names
    nc = cfg.nc
    images_dir = cfg.train_images if args.split == 'train' else cfg.val_images
    labels_dir = cfg.train_labels if args.split == 'train' else cfg.val_labels
    log(f'评估集: {images_dir}  标签: {labels_dir}  类别数(含未知): {len(class_names)}')

    if args.mode == 'gt':
        from src.classify import load_classifier
        from src.data.yolo import build_crop_index
        cls = load_classifier(args.cls_ckpt, nc=nc, names=class_names, device=args.device, logger=log)
        if getattr(cls, 'is_judge', False):
            from src.pipeline import build_pipeline
            pipe = build_pipeline(args.det_ckpt, None, args.thresholds, class_names, device=args.device,
                                  half=args.half, image_size=args.image_size, logger=log)
            gt_judge_eval(cls, pipe, images_dir, labels_dir, class_names, args.max_images, log)
            return
        index, bad = build_crop_index(images_dir, labels_dir, pad=args.crop_pad)
        if bad:
            log(f'跳过 {len(bad)} 个坏图')
        if args.max_images:
            index = index[:args.max_images * 50]
        log(f'GT 疵点框样本数: {len(index)}')
        # 惰性读图裁剪
        import cv2
        from src.classify import CropDataset
        from torch.utils.data import DataLoader
        ds = CropDataset(index, crop_size=cls.crop_size, augment=False, max_side=1024)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        y_true, y_pred = [], []
        import torch
        for x, y in loader:
            with torch.no_grad():
                logits = cls.model(x.to(args.device)) / cls.temperature
                pr = torch.softmax(logits, dim=1).cpu().numpy()
            m = pr.max(axis=1)
            u = (1.0 - m) * (m < cls.unknown_thr)
            pr = pr * (1 - u)[:, None]
            pred = np.where(u > 0, nc, pr.argmax(axis=1))
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        acc = float((y_true == y_pred).mean())
        unknown_rate = float((y_pred == nc).mean())
        log(f'[gt分类] 总准确率={acc:.4f}  未知类型比例={unknown_rate:.4f}  (n={len(y_true)})')
        confusion_report(class_names, y_true, y_pred)
    else:
        from src.pipeline import build_pipeline
        pipe = build_pipeline(args.det_ckpt, None if args.no_classify else args.cls_ckpt, args.thresholds, class_names,
                              device=args.device, half=args.half, image_size=args.image_size, logger=log,
                              top_k=args.top_k, region_score_ratio=args.region_score_ratio,
                              judge_override=args.judge_override)
        ds = YoloDefectDataset(images_dir, labels_dir, max_side=2048)
        n = len(ds) if not args.max_images else min(args.max_images, len(ds))
        img_ok = 0
        det_matched = 0
        gt_total = 0
        cls_ok = 0
        cls_total = 0
        y_true, y_pred = [], []
        fn_list, fp_list, pos_scores = [], [], []
        tp = fp = fn = tn = 0
        for i in range(n):
            img, boxes, _, path = ds[i]
            import cv2
            res = pipe.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), os.path.basename(path))
            gt_anom = len(boxes) > 0
            img_ok += int(res.is_anomaly == gt_anom)
            tp += int(gt_anom and res.is_anomaly)
            fn += int(gt_anom and not res.is_anomaly)
            fp += int((not gt_anom) and res.is_anomaly)
            tn += int((not gt_anom) and not res.is_anomaly)
            if gt_anom:
                pos_scores.append(res.anomaly_score)
                if not res.is_anomaly:
                    fn_list.append((res.anomaly_score, res.img_thr, len(res.regions), os.path.basename(path)))
            elif res.is_anomaly:
                fp_list.append((res.anomaly_score, res.img_thr, len(res.regions), os.path.basename(path)))
            for c, x1, y1, x2, y2 in boxes:
                gt_total += 1
                best, best_iou = None, 0.0
                for r in res.regions:
                    v = iou((x1, y1, x2, y2), r.box)
                    if v > best_iou:
                        best, best_iou = r, v
                if best is not None and best_iou >= args.iou_thr:
                    det_matched += 1
                    if best.classified:
                        cls_total += 1
                        y_true.append(int(c))
                        y_pred.append(best.class_id)
                        cls_ok += int(best.class_id == int(c))
            if (i + 1) % 20 == 0:
                log(f'  进度 {i + 1}/{n}')
        # 正常图 (normalval) 也要算进图级指标, 否则误报永远统计不到
        n_norm = 0
        if args.split == 'val' and cfg.normalval_images and os.path.isdir(cfg.normalval_images):
            from src.data.yolo import NormalImageDataset
            dsn = NormalImageDataset(cfg.normalval_images, verify=True)
            n_norm = len(dsn) if not args.max_images else min(args.max_images, len(dsn))
            import cv2
            for i in range(n_norm):
                img, path = dsn[i]
                res = pipe.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), os.path.basename(path))
                img_ok += int(not res.is_anomaly)
                tn += int(not res.is_anomaly)
                fp += int(res.is_anomaly)
                if res.is_anomaly:
                    fp_list.append((res.anomaly_score, res.img_thr, len(res.regions), os.path.basename(path)))
            log(f'[pipeline] 另评 normalval 正常图 {n_norm} 张')
        n = n + n_norm
        log(f'[pipeline] 图级准确率={img_ok / n:.4f}  检测框召回(IoU>={args.iou_thr})='
            f'{det_matched}/{gt_total}={det_matched / max(gt_total, 1):.4f}  '
            f'匹配区域分类准确率={cls_ok}/{cls_total}={cls_ok / max(cls_total, 1):.4f}')
        log(f'[检测] 缺陷图判异常 {tp}/{tp + fn} (漏检 {fn})  正常图判正常 {tn}/{tn + fp} (误报 {fp})  '
            f'召回={tp / max(tp + fn, 1):.4f} 精确率={tp / max(tp + fp, 1):.4f}')
        if cls_total:
            confusion_report(class_names, np.array(y_true), np.array(y_pred))
        if pos_scores:
            ps = np.array(pos_scores)
            log(f'[pipeline] 缺陷图 score 分位数: p10={np.quantile(ps, 0.1):.4f} p50={np.quantile(ps, 0.5):.4f} '
                f'p90={np.quantile(ps, 0.9):.4f} max={ps.max():.4f}  (img_thr={pipe.img_thr:.4f})')
        if fn_list:
            fn_list.sort()
            log(f'[pipeline] 漏检图(有缺陷但判正常) 共 {len(fn_list)} 张, score 最低 10 张 (margin=score-img_thr):')
            for s, t, nr, pth in fn_list[:10]:
                log(f'    {pth}: score={s:.4f} img_thr={t:.4f} regions={nr} margin={s - t:+.4f}')
        if fp_list:
            fp_list.sort(key=lambda x: -x[0])
            log(f'[pipeline] 误报图(无缺陷但判异常) 共 {len(fp_list)} 张, score 最高 10 张:')
            for s, t, nr, pth in fp_list[:10]:
                log(f'    {pth}: score={s:.4f} img_thr={t:.4f} regions={nr} margin={s - t:+.4f}')


if __name__ == '__main__':
    main()
