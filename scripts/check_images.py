"""诊断工具: 扫描目录下所有图片, 报告无法解码的文件 (含文件大小/魔数), 帮助定位中文名/坏文件问题.

示例:
  python scripts/check_images.py --dir "D:/tyhn/yolo/dataset/source5_augmented/normal/images/train"
  python scripts/check_images.py --data-yaml dataset/fabric19.yaml   # 扫描 yaml 里全部目录
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util import get_logger, list_images, imread_unicode, image_diag
import cv2


def scan(dir_path, log):
    files = list_images(dir_path)
    bad = []
    for i, p in enumerate(files):
        if imread_unicode(p, cv2.IMREAD_COLOR) is None:
            bad.append(p)
        if (i + 1) % 200 == 0:
            log(f'  {dir_path}: {i + 1}/{len(files)} (坏文件 {len(bad)})')
    log(f'{dir_path}: 共 {len(files)} 张, 可解码 {len(files) - len(bad)}, 坏文件 {len(bad)}')
    for p in bad:
        log(f'  坏文件: {p}  诊断: {image_diag(p)}')
    return bad


def main():
    p = argparse.ArgumentParser(description='扫描图片目录, 找出无法解码的文件')
    p.add_argument('--dir', default=None)
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    args = p.parse_args()
    log = get_logger('check_images')
    dirs = []
    if args.dir:
        dirs.append(args.dir)
    else:
        from src.data import load_fabric_config
        cfg = load_fabric_config(args.data_yaml, root_override=args.root)
        for k in ['train_images', 'val_images', 'normaltrain_images', 'normalval_images']:
            v = getattr(cfg, k)
            if v and os.path.isdir(v):
                dirs.append(v)
    total_bad = 0
    for d in dirs:
        total_bad += len(scan(d, log))
    log(f'全部完成, 坏文件总数: {total_bad}')


if __name__ == '__main__':
    main()
