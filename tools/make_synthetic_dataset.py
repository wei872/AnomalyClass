# -*- coding: utf-8 -*-
"""生成布匹纹理合成数据集 (与 fabric19 同目录结构/同类别), 用途:
  1) 无真实数据时验证整条 训练->标定->导出->推理 流水线;
  2) 新机器环境自检。
    python tools/make_synthetic_dataset.py --out data/synthetic_fabric19 --train 60 --val 12
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

NRNG = np.random.default_rng(0)
NAMES = ['纬线杂质', '断纬', '多股线', '纬线不直', '包边不重叠', '捻接器接头',
         '小尾巴', '经线毛球', '包边不良', '手工接头', '纬斜', '短废纱线',
         '单根包不进', '棉团', '蚊虫小黑点', '小布头', '稀缝', '油点线', '飘絮']


def base_fabric(h, w, rng):
    img = np.full((h, w), 205.0, np.float32)
    x = np.arange(w)[None, :].astype(np.float32)
    y = np.arange(h)[:, None].astype(np.float32)
    p_warp = rng.uniform(5, 7)
    img += 6 * np.sin(2 * np.pi * x / p_warp)                       # 经线竖纹
    p_weft = rng.uniform(40, 56)
    phase = 2 * np.pi * (y + 1.5 * np.sin(x / rng.uniform(150, 300))) / p_weft
    img -= 14 * (np.abs(((phase / (2 * np.pi)) % 1.0) - 0.5) < 0.06)   # 纬线横纹
    img += NRNG.normal(0, 2.2, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1 - 0.18 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 4
    img *= vig
    return np.clip(img, 0, 255)


def _box(h, w, cx, cy, bw, bh):
    x1, y1 = max(cx - bw / 2, 0), max(cy - bh / 2, 0)
    x2, y2 = min(cx + bw / 2, w), min(cy + bh / 2, h)
    return x1, y1, x2, y2


def draw_defect(img, cls, rng):
    h, w = img.shape
    cx, cy = rng.uniform(0.12, 0.88) * w, rng.uniform(0.12, 0.88) * h
    dark, light = -70.0, 55.0
    if cls == 0:      # 纬线杂质: 纬线上深色粗短段
        bw, bh = rng.uniform(40, 90), rng.uniform(6, 12)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), dark * 0.6, -1)
    elif cls == 1:    # 断纬: 纬线缺口(亮)
        bw, bh = rng.uniform(30, 70), rng.uniform(4, 8)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), light, -1)
    elif cls == 2:    # 多股线: 竖向粗条
        bw, bh = rng.uniform(8, 14), rng.uniform(60, 140)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), dark * 0.5, -1)
    elif cls == 3:    # 纬线不直: 阶梯横线
        bw, bh = rng.uniform(80, 160), rng.uniform(10, 20)
        pts = np.array([[cx - bw / 2, cy], [cx - bw / 6, cy], [cx - bw / 6, cy + bh * 0.6],
                        [cx + bw / 2, cy + bh * 0.6]], np.int32)
        cv2.polylines(img, [pts], False, dark * 0.7, 3)
    elif cls in (4, 8, 12):   # 包边类: 边缘亮带
        cx = rng.choice([rng.uniform(0.02, 0.06), rng.uniform(0.94, 0.98)]) * w
        bw, bh = rng.uniform(10, 24), rng.uniform(120, 240)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), light * 0.8, -1)
    elif cls in (5, 9):       # 接头: 结点团
        r = int(rng.uniform(6, 12))
        cv2.circle(img, (int(cx), int(cy)), r, dark * 0.65, -1)
        cv2.line(img, (int(cx) - r * 3, int(cy)), (int(cx) + r * 3, int(cy)), dark * 0.4, 2)
        bw = bh = r * 2.5
    elif cls == 6:    # 小尾巴: 细弯线
        bw, bh = rng.uniform(20, 45), rng.uniform(20, 45)
        pts = np.array([[cx, cy], [cx + bw * 0.3, cy + bh * 0.4], [cx + bw * 0.15, cy + bh]], np.int32)
        cv2.polylines(img, [pts], False, dark * 0.6, 2)
    elif cls == 7:    # 经线毛球: 毛茸团
        r = int(rng.uniform(8, 15))
        for _ in range(40):
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0, r)
            cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), 1, dark * 0.5, -1)
        bw = bh = r * 2.2
    elif cls == 10:   # 纬斜: 斜线
        bw, bh = rng.uniform(120, 220), rng.uniform(30, 60)
        cv2.line(img, (int(cx - bw / 2), int(cy + bh / 2)), (int(cx + bw / 2), int(cy - bh / 2)), dark * 0.5, 3)
    elif cls == 11:   # 短废纱线: 随机短线
        bw, bh = rng.uniform(25, 60), rng.uniform(6, 20)
        a = rng.uniform(0, np.pi)
        dx, dy = np.cos(a) * bw / 2, np.sin(a) * bw / 2
        cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), dark * 0.55, 2)
    elif cls == 13:   # 棉团: 亮絮团
        r = int(rng.uniform(10, 20))
        overlay = img.copy()
        cv2.circle(overlay, (int(cx), int(cy)), r, light, -1)
        cv2.GaussianBlur(overlay, (15, 15), 0, overlay)
        img[:] = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        bw = bh = r * 2.2
    elif cls == 14:   # 蚊虫小黑点: 小黑点
        r = int(rng.uniform(2, 5))
        cv2.circle(img, (int(cx), int(cy)), r, -95, -1)
        bw = bh = r * 2 + 4
    elif cls == 15:   # 小布头: 矩形块
        bw, bh = rng.uniform(30, 70), rng.uniform(20, 50)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), 150, -1)
    elif cls == 16:   # 稀缝: 亮横带
        bw, bh = rng.uniform(60, 120), rng.uniform(10, 18)
        cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), light * 0.6, -1)
    elif cls == 17:   # 油点线: 模糊深色条纹
        bw, bh = rng.uniform(60, 140), rng.uniform(8, 16)
        overlay = img.copy()
        cv2.rectangle(overlay, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), dark * 0.45, -1)
        cv2.GaussianBlur(overlay, (21, 21), 0, overlay)
        img[:] = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    else:             # 18 飘絮: 浅细弯线
        bw, bh = rng.uniform(30, 70), rng.uniform(10, 30)
        pts = np.array([[cx - bw / 2, cy], [cx, cy - bh / 2], [cx + bw / 2, cy + bh / 3]], np.int32)
        cv2.polylines(img, [pts], False, light * 0.5, 1)
    x1, y1, x2, y2 = _box(h, w, cx, cy, bw, bh)
    return (x1, y1, x2, y2)


def make_split(root: Path, split: str, n: int, n_normal: int, rng, wh=(1024, 512)):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "normal" / "images" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w, h = wh
        img = base_fabric(h, w, rng)
        lines = []
        for _ in range(rng.randint(1, 3)):
            cls = rng.randrange(19)
            x1, y1, x2, y2 = draw_defect(img, cls, rng)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw > 0.005 and bh > 0.005:
                lines.append(f"{cls} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
        g = np.clip(img, 0, 255).astype(np.uint8)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.imencode(".jpg", g)[1].tofile(str(root / "images" / split / f"{split}_{i:04d}.jpg"))
        (root / "labels" / split / f"{split}_{i:04d}.txt").write_text("\n".join(lines), encoding="utf-8")
    for i in range(n_normal):
        w, h = wh
        g = np.clip(base_fabric(h, w, rng), 0, 255).astype(np.uint8)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.imencode(".jpg", g)[1].tofile(str(root / "normal" / "images" / split / f"n_{split}_{i:04d}.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_fabric19")
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.out)
    make_split(root, "train", a.train, max(a.train // 3, 4), rng)
    make_split(root, "val", a.val, max(a.val // 2, 3), rng)
    yaml_txt = f"""path: {root.resolve()}/
train: {root.resolve()}/images/train
val: {root.resolve()}/images/val
normaltrain: {root.resolve()}/normal/images/train
normalval: {root.resolve()}/normal/images/val
nc: 19
names: {NAMES}
"""
    (root / "synthetic_fabric19.yaml").write_text(yaml_txt, encoding="utf-8")
    print(f"[syn] 合成数据集 -> {root}  (train={a.train}, val={a.val})")
    print(f"[syn] yaml: {root / 'synthetic_fabric19.yaml'}")


if __name__ == "__main__":
    main()
