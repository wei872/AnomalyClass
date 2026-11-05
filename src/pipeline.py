"""端到端流水线: 异常检测 (Dinomaly2) -> 区域定位 (连通域) -> 缺陷分类 (轻量 CNN + 未知类型).

每张图输出:
  is_anomaly      : 图级是否异常 (score > img_thr)
  anomaly_conf    : 与正常样本偏离的置信度 = score/(score+img_thr) ∈ [0,1)
  regions         : 每个异常区域 {box, anomaly_conf, class_id, class_name, class_conf, probs}
                    class_id == nc 时为 "未知类型" (分类置信度低于阈值)
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .classify import RegionClassifier
from .detect import AnomalyDetector, deviation_confidence
from .util import (class_color, get_cjk_font, get_logger, imread_unicode, UNKNOWN_COLOR)


@dataclass
class RegionResult:
    box: Tuple[int, int, int, int]
    area: int
    anomaly_score: float          # 区域内平均异常分数
    anomaly_conf: float           # 区域与正常偏离的置信度
    class_id: int
    class_name: str
    class_conf: float             # 分类置信度
    classified: bool = True       # 是否真正过了分类器 (top_k 之外的区域为 False, 标未知类型)
    probs: List[float] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d['box'] = list(self.box)
        return d


@dataclass
class ImageResult:
    image: str
    width: int
    height: int
    is_anomaly: bool
    anomaly_score: float
    anomaly_conf: float           # 与正常样本偏离的置信度
    img_thr: float
    regions: List[RegionResult] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self):
        return {
            'image': self.image,
            'width': self.width,
            'height': self.height,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': round(self.anomaly_score, 6),
            'anomaly_conf': round(self.anomaly_conf, 6),
            'img_thr': round(self.img_thr, 6),
            'latency_ms': round(self.latency_ms, 2),
            'regions': [r.to_dict() for r in self.regions],
        }


class FabricAnomalyPipeline:
    """检测 + 分类一体化推理流水线 (GTX 1660 实时目标)."""

    def __init__(self, detector: AnomalyDetector, classifier: Optional[RegionClassifier],
                 class_names: List[str], img_thr: float, pix_thr: Optional[float] = None,
                 mask_ratio: float = 0.5, min_area_ratio: float = 2e-4, dilate: int = 7, box_pad: float = 0.2,
                 device: str = 'cuda', logger=None, max_map_side: int = 1024, max_vis_side: int = 1600,
                 top_k: int = 1, region_score_ratio: float = 0.5):
        self.det = detector
        self.cls = classifier
        self.class_names = class_names          # 长度 nc+1, 末尾为 '未知类型'
        self.nc = len(class_names) - 1
        self.img_thr = float(img_thr)
        self.pix_thr = float(pix_thr) if pix_thr is not None else self.img_thr
        self.mask_thr = self.pix_thr * mask_ratio if pix_thr is not None else self.img_thr * mask_ratio
        self.mask_ratio = mask_ratio
        self.min_area_ratio = min_area_ratio
        self.dilate = dilate
        self.box_pad = box_pad
        self.device = device
        self.max_map_side = max_map_side
        self.max_vis_side = max_vis_side
        # 只对最异常的 top_k 个热力区域做分类 (top_k<=0 表示全部分类);
        # 掩码阈值同时跟随图内最高分, 只圈最热的区域
        self.top_k = int(top_k)
        self.region_score_ratio = float(region_score_ratio)
        self.log = logger or get_logger('pipeline')

    # ---------------- 核心推理 ----------------
    @torch.no_grad()
    def process(self, img_bgr: np.ndarray, image_name: str = '') -> ImageResult:
        import time
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        # 异常图长边封顶 max_map_side (防超大图内存爆炸), 框再映射回原图坐标
        full_map, score, (sx, sy) = self.det.predict(img_rgb, max_map_side=self.max_map_side)
        is_anom = score > self.img_thr
        conf = deviation_confidence(score, self.img_thr)
        result = ImageResult(image=image_name, width=w0, height=h0, is_anomaly=bool(is_anom),
                             anomaly_score=score, anomaly_conf=conf, img_thr=self.img_thr)
        if is_anom:
            # 掩码阈值 = max(全局阈值, 图内最高分 * region_score_ratio): 只圈最热的异常区域
            mask_thr_eff = max(self.mask_thr, score * self.region_score_ratio)
            regions = self.det.segment(full_map, mask_thr_eff, self.min_area_ratio, self.dilate, self.box_pad)
            # segment 已按 mean_score 降序; 只对最异常的 top_k 个区域送分类器
            k = len(regions) if self.top_k <= 0 else min(self.top_k, len(regions))
            items = []
            for r in regions:
                x1, y1, x2, y2 = r['box']
                box = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
                items.append((r, box))
            cls_map = {}
            if self.cls is not None and k > 0:
                crops, idxs = [], []
                for i in range(k):
                    r, box = items[i]
                    bx1, by1 = max(0, box[0]), max(0, box[1])
                    bx2, by2 = min(w0, box[2]), min(h0, box[3])
                    if bx2 > bx1 and by2 > by1:
                        crops.append(img_rgb[by1:by2, bx1:bx2])
                        idxs.append(i)
                if crops:
                    for j, cr in enumerate(self.cls.predict(crops)):   # 一次 batch 推理
                        cls_map[idxs[j]] = cr
            for i, (r, box) in enumerate(items):
                r_conf = deviation_confidence(r['mean_score'], self.img_thr)
                cr = cls_map.get(i)
                if cr is None:
                    # 未送分类器 (top_k 之外) 或分类器为空 -> 未知类型
                    result.regions.append(RegionResult(
                        box=box, area=r['area'], anomaly_score=r['mean_score'], anomaly_conf=r_conf,
                        class_id=self.nc, class_name='未知类型', class_conf=0.0, classified=False, probs=[]))
                else:
                    cid = cr['class_id']
                    result.regions.append(RegionResult(
                        box=box, area=r['area'], anomaly_score=r['mean_score'], anomaly_conf=r_conf,
                        class_id=cid,
                        class_name=self.class_names[cid] if cid < len(self.class_names) else '未知类型',
                        class_conf=cr['class_conf'], classified=True, probs=cr['probs']))
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def process_file(self, path: str) -> ImageResult:
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'无法读取图片: {path}')
        return self.process(img, image_name=os.path.basename(path))

    # ---------------- 可视化 ----------------
    def visualize(self, img_bgr: np.ndarray, result: ImageResult, font_path: Optional[str] = None,
                  show_heatmap: bool = True) -> np.ndarray:
        """画热力图 + 缺陷框 + 标签 (中文名 + 分类置信度 + 异常置信度).

        显示图长边封顶 max_vis_side (超大原图不至于把可视化撑爆); result.regions 仍是原图坐标.
        """
        from PIL import Image, ImageDraw
        h0, w0 = img_bgr.shape[:2]
        vscale = min(1.0, self.max_vis_side / max(h0, w0))
        if vscale < 1.0:
            vis = cv2.resize(img_bgr, (max(1, round(w0 * vscale)), max(1, round(h0 * vscale))),
                             interpolation=cv2.INTER_AREA)
        else:
            vis = img_bgr.copy()
        if show_heatmap and result.is_anomaly:
            full_map, _, (sx, sy) = self.det.predict(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                                     max_map_side=self.max_map_side)
            m = cv2.resize(full_map, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
            lo, hi = np.percentile(m, 50), max(np.percentile(m, 99), 1e-6)
            mn = np.clip((m - lo) / (hi - lo), 0, 1)
            heat = cv2.applyColorMap((mn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.65, heat, 0.35, 0)
        for r in result.regions:
            x1, y1, x2, y2 = [int(round(v * vscale)) for v in r.box]
            color = UNKNOWN_COLOR if r.class_id == self.nc else class_color(r.class_id)
            # cv2 画框用 BGR
            bgr = (color[2], color[1], color[0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2)
            label = f'{r.class_name} {r.class_conf:.2f} | 异常 {r.anomaly_conf:.2f}'
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = get_cjk_font(16, font_path)
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = 8 * len(label), 18
            ty = max(0, y1 - th - 6)
            draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=bgr)
            draw.text((x1 + 4, ty + 3), label, font=font, fill=(255, 255, 255))
            vis = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        status = f'ANOMALY conf={result.anomaly_conf:.2f}' if result.is_anomaly else f'normal conf={result.anomaly_conf:.2f}'
        cv2.putText(vis, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if result.is_anomaly else (0, 180, 0), 2)
        return vis


def build_pipeline(det_ckpt: str, cls_ckpt: Optional[str], thresholds_json: str,
                   class_names: List[str], device: str = 'cuda', half: bool = False,
                   image_size: Optional[int] = None, logger=None, top_k: int = 1,
                   region_score_ratio: float = 0.5) -> FabricAnomalyPipeline:
    """从 checkpoint 路径构建完整流水线."""
    from .detect import load_json
    log = logger or get_logger('pipeline')
    det = AnomalyDetector.load(det_ckpt, device=device, logger=log, half=half)
    if image_size and image_size != det.image_size:
        det.image_size = image_size
        det.crop_size = int(image_size * 392 / 448)
        det.offset = (det.image_size - det.crop_size) // 2
        from .detect import get_eval_transform
        det.tf = get_eval_transform(det.image_size, det.crop_size)
    thr = load_json(thresholds_json)
    cls = None
    if cls_ckpt and os.path.exists(cls_ckpt):
        cls = RegionClassifier.load(cls_ckpt, nc=len(class_names) - 1, names=class_names, device=device, logger=log)
    else:
        log('[pipeline] 未提供分类器权重, 只做异常检测 (区域标为未知类型)')
    return FabricAnomalyPipeline(det, cls, class_names, img_thr=thr['img_thr'], pix_thr=thr.get('pix_thr'),
                                  device=device, logger=log, top_k=top_k,
                                  region_score_ratio=region_score_ratio)
