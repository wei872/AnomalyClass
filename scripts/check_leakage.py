"""检查 train / val 是否有近重复图 (数据泄漏).

数据集目录名叫 source5_augmented: 如果是 **先离线增强、再划分 train/val**, 同一张原图的翻转/旋转/调亮副本会同时
出现在 train 和 val 里. 后果: val 指标虚高, 阈值 / unknown_thr / model_best 都在"见过的图"上校准, 上线后表现明显更差.

做法: 每张图缩成 9x8 灰度的 dHash (对亮度/对比度缩放不敏感); val 图取 8 种旋转/翻转版本的哈希, 只要有一个
与 train 某图的汉明距离 <= --thr 就算近重复. 同时给出 "随机无关图对" 的距离中位数作参照 (布匹纹理很均匀,
dHash 的区分度有限, 如果参照距离本身就很小, 结果只能当线索, 需要人工抽看).

示例:
  python scripts/check_leakage.py --data-yaml dataset/fabric19.yaml
  python scripts/check_leakage.py --train-dir A --val-dir B --show 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.util import get_logger, list_images, imread_unicode

_POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def _bits(g: np.ndarray) -> np.ndarray:
    """9x8 灰度 -> 64 bit (相邻列比较) -> uint8[8]."""
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def dihedral_hashes(img_bgr: np.ndarray, all8: bool) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 200.0)
    g = cv2.resize(g, (9, 9), interpolation=cv2.INTER_AREA).astype(np.float32)
    outs = []
    variants = [g] if not all8 else [np.rot90(g[:, :], k) for k in range(4)] + \
        [np.rot90(g[:, ::-1], k) for k in range(4)]
    for v in variants:
        v = np.ascontiguousarray(v)[:9, :9]
        outs.append(_bits(v[:8, :9]))
    return np.stack(outs)                                    # (1 or 8, 8)


def hash_dir(d: str, all8: bool, log, max_n: int = 0):
    files = list_images(d)
    if max_n and len(files) > max_n:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_n).astype(int)]
    hs, names = [], []
    for i, p in enumerate(files):
        img = imread_unicode(p, cv2.IMREAD_REDUCED_COLOR_4)       # 缩 1/4 解码, 比完整解码快很多
        if img is None:
            continue
        hs.append(dihedral_hashes(img, all8))
        names.append(p)
        if (i + 1) % 500 == 0:
            log(f'  {d}: {i + 1}/{len(files)}')
    return np.stack(hs) if hs else np.zeros((0, 1, 8), np.uint8), names


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,V,8) b: (m,8) -> (n,m) 最小汉明距离 (对 V 个变体取最小)."""
    out = np.full((a.shape[0], b.shape[0]), 64, dtype=np.int32)
    for v in range(a.shape[1]):
        x = np.bitwise_xor(a[:, v, None, :], b[None, :, :])
        out = np.minimum(out, _POP[x].sum(-1).astype(np.int32))
    return out


def leakage(train_dirs, val_dirs, thr: int, log, show: int = 0, max_n: int = 0):
    tr, tn = [], []
    for d in train_dirs:
        h, n = hash_dir(d, False, log, max_n)
        if len(n):
            tr.append(h[:, 0, :])
            tn += n
    tr = np.concatenate(tr) if tr else np.zeros((0, 8), np.uint8)
    res = {}
    for d in val_dirs:
        h, n = hash_dir(d, True, log, max_n)
        if not len(n) or not len(tr):
            continue
        dist = hamming(h, tr)
        nn = dist.min(1)
        # 参照: 把 train 顺序错位后的"无关对"距离 (只用 v=0 的变体, 偏保守)
        rnd = np.sort(hamming(h[:, :1], np.roll(tr, len(tr) // 2, axis=0))[np.arange(len(n)),
                                                                        np.arange(len(n)) % len(tr)])
        dup = int((nn <= thr).sum())
        res[d] = (dup, len(n))
        log(f'{d}: {dup}/{len(n)} ({dup / len(n):.1%}) 张在 train 里有近重复 (汉明距离<={thr}); '
            f'最近邻距离中位数 {np.median(nn):.0f}, 无关对距离中位数 {np.median(rnd):.0f}')
        for j in np.argsort(nn)[:show]:
            log(f'    d={nn[j]:2d}  {n[j]}  ~  {tn[int(dist[j].argmin())]}')
    return res


def main():
    p = argparse.ArgumentParser(description='检查 train/val 近重复 (数据泄漏)')
    p.add_argument('--data-yaml', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None)
    p.add_argument('--train-dir', action='append', default=None, help='可重复; 不给则从 yaml 读 train + normaltrain')
    p.add_argument('--val-dir', action='append', default=None, help='可重复; 不给则从 yaml 读 val + normalval')
    p.add_argument('--thr', type=int, default=6, help='汉明距离阈值 (64 bit), 默认 6')
    p.add_argument('--show', type=int, default=5, help='每个 val 目录打印最相似的前 N 对, 供人工抽看')
    p.add_argument('--max-images', type=int, default=0, help='每个目录最多抽样多少张 (0=全部)')
    args = p.parse_args()
    log = get_logger('check_leakage')
    tr_dirs, va_dirs = args.train_dir, args.val_dir
    if not tr_dirs or not va_dirs:
        from src.data import load_fabric_config
        cfg = load_fabric_config(args.data_yaml, root_override=args.root)
        tr_dirs = tr_dirs or [d for d in (cfg.train_images, cfg.normaltrain_images) if d and os.path.isdir(d)]
        va_dirs = va_dirs or [d for d in (cfg.val_images, cfg.normalval_images) if d and os.path.isdir(d)]
    leakage(tr_dirs, va_dirs, args.thr, log, args.show, args.max_images)
    log('提示: 比例很高 (>10%) 说明 val 与 train 来自同一批原图的增强副本; 请改成按原图分组划分, 再重新校准阈值.')


if __name__ == '__main__':
    main()
