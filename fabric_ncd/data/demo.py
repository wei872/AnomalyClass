"""合成布匹数据生成器: 用于快速自检 / 演示 (不依赖真实数据).

生成 "经纬纹理 + 随机疵点" 的迷你 YOLO 数据集, ``tools/demo.py`` 与单元测试
都用它来验证整条流水线. 真实训练请使用 dataset/fabric19.yaml 指向的数据.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def weave_image(size: Tuple[int, int], rng: np.random.Generator, period: float = 6.0) -> np.ndarray:
    h, w = size
    xs = np.arange(w)[None, :]
    ys = np.arange(h)[:, None]
    base = 140 + 25 * np.sin(2 * np.pi * xs / period) * np.sin(2 * np.pi * ys / (period * 1.3))
    color = np.array([0.9, 0.85, 0.8]) * rng.uniform(0.8, 1.1)
    img = base[..., None] * color[None, None, :]
    img += rng.normal(0, 4, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_defect(img: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
    """在布面上加一条暗线或一块污渍, 返回 (图像, 像素框)."""
    h, w = img.shape[:2]
    out = img.copy()
    if rng.integers(0, 2) == 0:
        x1, y1 = int(rng.integers(4, w // 2)), int(rng.integers(4, h // 2))
        x2 = x1 + int(rng.integers(6, max(8, w // 3)))
        y2 = y1 + int(rng.integers(4, max(6, h // 4)))
        pil = Image.fromarray(out)
        ImageDraw.Draw(pil).line(
            [x1, y1, x2, y2], fill=(30, 30, 35), width=int(rng.integers(1, 3))
        )
        out = np.asarray(pil).copy()
    else:
        cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
        r = int(rng.integers(4, max(5, min(h, w) // 6)))
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        sub = out[max(0, y1) : y2, max(0, x1) : x2].astype(np.int16)
        sub = np.clip(sub + rng.integers(-60, 20), 0, 255).astype(np.uint8)
        out[max(0, y1) : y2, max(0, x1) : x2] = sub
    box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
    return out, box


def make_dataset(root: Path, n_train: int = 8, n_val: int = 4, n_normal: int = 6, size=(96, 96)) -> Path:
    """生成临时 YOLO 数据集并返回数据根目录."""
    root = Path(root).resolve()
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = weave_image(size, rng)
            img, box = add_defect(img, rng)
            h, w = img.shape[:2]
            cls = int(rng.integers(0, 3))
            Image.fromarray(img).save(img_dir / f"{split}_{i}.png")
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            (lbl_dir / f"{split}_{i}.txt").write_text(f"{cls} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
        normal_dir = root / "normal" / "images" / split
        normal_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_normal):
            Image.fromarray(weave_image(size, rng)).save(normal_dir / f"normal_{split}_{i}.png")
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\n"
        f"train: {root}/images/train\n"
        f"val: {root}/images/val\n"
        f"normaltrain: {root}/normal/images/train\n"
        f"normalval: {root}/normal/images/val\n"
        "nc: 3\n"
        "names: ['scratch', 'stain', 'cut']\n",
        encoding="utf-8",
    )
    return root
