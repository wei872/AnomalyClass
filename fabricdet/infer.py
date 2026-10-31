# -*- coding: utf-8 -*-
"""推理 SDK + CLI。支持 PyTorch 权重 (.pt) 与 ONNX 两种后端, 行为完全一致。
工业用法:
    from fabricdet.infer import Detector
    det = Detector("runs/fabric19/deploy/best.onnx")   # 或 best.pt
    results = det.predict("img.jpg")                    # [{'box','score','cls','name','unknown'}]
    det.save_vis("img.jpg", "out.jpg")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import imread_unicode, letterbox
from .model import IMAGENET_MEAN, IMAGENET_STD, FabricDefectNet
from .postprocess import (DEFAULT_THRESHOLDS, UNKNOWN_CLS, UNKNOWN_NAME,
                          UNKNOWN_NAME_EN, merge_tile_preds, postprocess)

NAMES_EN = ["weft_inclusion", "broken_weft", "multi_strand_yarn", "weft_crooked",
            "edge_no_overlap", "splicer_joint", "small_tail", "warp_fuzz_ball",
            "edge_defect", "manual_joint", "weft_skew", "short_waste_yarn",
            "single_end_miss", "cotton_clump", "insect_dark_spot", "fabric_scrap",
            "sparse_stitch", "oil_stain_line", "floating_fiber"]

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wenquanyi/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]


def _find_cjk_font():
    for p in _CJK_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


# ------------------------------------------------------------------ tiles
def tile_windows(h: int, w: int, tile: int, overlap: float):
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not xs or xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if not ys or ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    return [(x, y) for y in ys for x in xs]


def _preprocess_crop(crop: np.ndarray, size: int):
    mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, np.float32).reshape(1, 1, 3)
    x = (crop.astype(np.float32) / 255.0 - mean) / std
    return x.transpose(2, 0, 1)


@torch.no_grad()
def predict_with_model(model, image, tile: int = 448, overlap: float = 0.25,
                       thresholds: dict | None = None, device="cpu",
                       conf: float | None = None, batch: int = 4) -> list[dict]:
    """image: 路径或 BGR ndarray。返回原图坐标预测列表。"""
    if isinstance(image, (str, Path)):
        img = imread_unicode(image)
        if img is None:
            return []
    else:
        img = image
    h, w = img.shape[:2]
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    use_tiles = max(h, w) > tile * 1.15
    windows = tile_windows(h, w, tile, overlap) if use_tiles else [(0, 0)]
    crops, offsets = [], []
    for (x0, y0) in windows:
        if use_tiles:
            crop = img[y0:y0 + tile, x0:x0 + tile]
            offsets.append((x0, y0, 1.0))
        else:
            crop, sc, _ = letterbox(img, tile)
            offsets.append((0, 0, sc))
        crops.append(crop)
    preds: list[dict] = []
    model.eval()
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        xs = torch.from_numpy(np.stack([_preprocess_crop(c, tile) for c in chunk])).to(device)
        if device != "cpu" and torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                boxes, obj, sims = model.forward_deploy(xs)
        else:
            boxes, obj, sims = model.forward_deploy(xs)
        boxes, obj, sims = boxes.float().cpu().numpy(), obj.float().cpu().numpy(), sims.float().cpu().numpy()
        for j in range(len(chunk)):
            x0, y0, sc = offsets[i + j]
            for p in postprocess(boxes[j], obj[j], sims[j], th, conf=conf):
                b = p["box"] / sc
                p["box"] = np.array([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], np.float32)
                preds.append(p)
    if use_tiles:
        preds = merge_tile_preds(preds, th["nms_iou"])
    return preds


# ---------------------------------------------------------------- detector
class Detector:
    def __init__(self, ckpt: str | Path, device: str | None = None, names_en: list | None = None):
        self.ckpt = Path(ckpt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.meta: dict = {}
        self.names: list[str] = []
        self.names_en = names_en or NAMES_EN
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.model = None
        self.ort = None
        if self.ckpt.suffix in (".onnx",):
            self._load_onnx()
        elif self.ckpt.is_dir():
            self._load_onnx(self.ckpt / "best.onnx")
        else:
            self._load_pt()

    # ------------------------------------------------------------ loaders
    def _load_pt(self):
        ck = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        self.meta = ck.get("meta", {})
        self.names = self.meta.get("names", [])
        self.names_en = self.meta.get("names_en", self.names_en)[:len(self.names)] or self.names_en
        self.thresholds.update(self.meta.get("thresholds", {}))
        size = int(self.meta.get("input_size", 448))
        self.model = FabricDefectNet(len(self.names), emb_dim=int(self.meta.get("emb_dim", 128)),
                                     input_size=size)
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.size = size

    def _load_onnx(self, path: Path | None = None):
        import onnxruntime as ort
        path = Path(path or self.ckpt)
        cfg = path.parent / "deploy_config.json"
        if cfg.exists():
            self.meta = json.loads(cfg.read_text(encoding="utf-8"))
            self.names = self.meta.get("names", [])
            self.names_en = self.meta.get("names_en", self.names_en)
            self.thresholds.update(self.meta.get("thresholds", {}))
            self.size = int(self.meta.get("input_size", 448))
        else:
            self.size = 448
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" \
            else ["CPUExecutionProvider"]
        self.ort = ort.InferenceSession(str(path), providers=providers)
        self.onnx_path = path

    # ------------------------------------------------------------ predict
    def predict(self, image, conf: float | None = None, tile: int | None = None,
                overlap: float = 0.25) -> list[dict]:
        tile = tile or self.size
        if self.model is not None:
            raw = predict_with_model(self.model, image, tile=tile, overlap=overlap,
                                     thresholds=self.thresholds, device=self.device, conf=conf)
        else:
            raw = self._predict_onnx(image, tile, overlap, conf)
        for p in raw:
            p["unknown"] = p["cls"] == UNKNOWN_CLS
            p["name"] = UNKNOWN_NAME if p["unknown"] else self.names[p["cls"]]
            p["name_en"] = UNKNOWN_NAME_EN if p["unknown"] else \
                (self.names_en[p["cls"]] if p["cls"] < len(self.names_en) else self.names[p["cls"]])
        return raw

    def _predict_onnx(self, image, tile, overlap, conf):
        if isinstance(image, (str, Path)):
            img = imread_unicode(image)
            if img is None:
                return []
        else:
            img = image
        h, w = img.shape[:2]
        th = dict(self.thresholds)
        use_tiles = max(h, w) > tile * 1.15
        windows = tile_windows(h, w, tile, overlap) if use_tiles else [(0, 0)]
        preds = []
        for (x0, y0) in windows:
            if use_tiles:
                crop, off, sc = img[y0:y0 + tile, x0:x0 + tile], (x0, y0), 1.0
            else:
                crop, sc, _ = letterbox(img, tile)
                off = (0, 0)
            x = _preprocess_crop(crop, tile)[None].astype(np.float32)
            boxes, obj, sims = self.ort.run(None, {"images": x})
            for p in postprocess(boxes[0], obj[0], sims[0], th, conf=conf):
                b = p["box"] / sc
                p["box"] = np.array([b[0] + off[0], b[1] + off[1], b[2] + off[0], b[3] + off[1]], np.float32)
                preds.append(p)
        if use_tiles:
            preds = merge_tile_preds(preds, th["nms_iou"])
        return preds

    # ---------------------------------------------------------------- vis
    def draw(self, img: np.ndarray, preds: list[dict], font_path: str | None = "auto"):
        out = img.copy()
        fp = _find_cjk_font() if font_path == "auto" else font_path
        pil = None
        if fp:
            try:
                from PIL import Image, ImageDraw, ImageFont
                pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                font = ImageFont.truetype(fp, max(14, img.shape[1] // 60))
            except Exception:
                pil = None
        for p in preds:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            color = (0, 0, 255) if p["cls"] == UNKNOWN_CLS else (0, 200, 0)
            nm = p.get("name") or (UNKNOWN_NAME if p["cls"] == UNKNOWN_CLS
                                   else self.names[p["cls"]] if 0 <= p["cls"] < len(self.names) else "?")
            nm_en = p.get("name_en") or (UNKNOWN_NAME_EN if p["cls"] == UNKNOWN_CLS
                                         else self.names_en[p["cls"]] if p["cls"] < len(self.names_en) else nm)
            label = f"{'?' if p['cls'] == UNKNOWN_CLS else p['cls']} " \
                    f"{nm_en if pil is None else nm} {p['score']:.2f}"
            if pil is not None:
                draw.rectangle([x1, y1, x2, y2], outline=color[::-1], width=2)
                draw.text((x1 + 2, max(y1 - 18, 0)), label, fill=color[::-1], font=font)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1 + 2, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if pil is not None:
            out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return out

    def save_vis(self, src, dst, conf=None, tile=None):
        img = src if isinstance(src, np.ndarray) else imread_unicode(src)
        if img is None:
            return []
        preds = self.predict(img, conf=conf, tile=tile)
        cv2.imencode(".jpg", self.draw(img, preds))[1].tofile(str(dst))
        return preds

    def benchmark(self, tile: int | None = None, n: int = 50) -> float:
        tile = tile or self.size
        x = np.zeros((1, 3, tile, tile), np.float32)
        for _ in range(3):
            self._bench_once(x)
        t0 = time.time()
        for _ in range(n):
            self._bench_once(x)
        return (time.time() - t0) / n * 1000

    def _bench_once(self, x):
        if self.model is not None:
            with torch.no_grad():
                self.model.forward_deploy(torch.from_numpy(x).to(self.device))
        else:
            self.ort.run(None, {"images": x})


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def process_video(det: "Detector", src: str, dst: str, conf=None, tile=None, skip: int = 1):
    """视频/相机流: 逐帧检测+标注+写回 mp4。"""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[infer] 无法打开视频/流: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % skip == 0:
            preds = det.predict(frame, conf=conf, tile=tile)
            frame = det.draw(frame, preds)
        vw.write(frame)
        n += 1
    cap.release()
    vw.release()
    print(f"[infer] 视频 {n} 帧 -> {dst}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="FabricDefectNet 推理 CLI")
    ap.add_argument("--ckpt", required=True, help="best.pt 或 best.onnx / deploy 目录")
    ap.add_argument("--source", required=True, help="图片/目录/视频/rtsp")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--save-dir", default="runs/infer")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--fps", action="store_true", help="仅测速")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    det = Detector(a.ckpt, device=a.device)
    print(f"[infer] 后端: {'ONNX' if det.ort else 'PyTorch'} | 设备: {det.device} | "
          f"类别 {len(det.names)} | 阈值 conf={det.thresholds['conf']} tau_unknown={det.thresholds['tau_unknown']}")
    if a.fps:
        ms = det.benchmark(tile=a.tile)
        print(f"[infer] 单 tile {a.tile or det.size}px: {ms:.2f} ms ({1000 / ms:.1f} FPS)")
        return
    src = Path(a.source)
    save = Path(a.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    if a.source.lower().startswith(("rtsp://", "rtmp://", "http://")) or (
            src.is_file() and src.suffix.lower() in VIDEO_EXTS):
        process_video(det, a.source, str(save / "out.mp4"), conf=a.conf, tile=a.tile)
        return
    files = [src] if src.is_file() else sorted(
        p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    all_json = {}
    for f in files:
        preds = det.save_vis(f, save / f.name, conf=a.conf, tile=a.tile)
        all_json[str(f)] = [{"box": [round(float(v), 1) for v in p["box"]], "score": round(p["score"], 3),
                             "cls": p["cls"], "name": p["name"], "unknown": p["unknown"]} for p in preds]
        n_unk = sum(1 for p in preds if p["unknown"])
        print(f"[infer] {f.name}: {len(preds)} 缺陷 (未知 {n_unk}) -> {save / f.name}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(all_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[infer] JSON -> {a.json_out}")


if __name__ == "__main__":
    main()
