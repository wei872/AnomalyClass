"""端到端流水线: 异常检测 (Dinomaly2) -> 区域定位 (连通域) -> 二次判断 (整图+区域综合: 19 类 / 未知异常 / 推翻为正常).

每张图输出:
  is_anomaly      : 图级是否异常 (score > img_thr, 阈值由 calibrate 在间隙中点取, 正常/异常各自分开)
  anomaly_conf    : 与正常样本偏离的置信度 = score/(score+img_thr) ∈ [0,1)
  regions         : 每个异常区域 {box, anomaly_score, anomaly_conf, class_id, class_name, class_conf, probs}
                    只对最异常的 top_k 个区域做分类, 其余标 "未知类型" (classified=false)
                    class_id == nc 且分类置信度低于 unknown_thr 时也为 "未知类型"
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .classify import RegionClassifier, load_classifier
from .detect import AnomalyDetector, deviation_confidence
from .judge import heat_to_grid
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
    unknown: bool = False         # 二次判断: 新颖且严重 -> 没见过的异常类型 (class_id=nc)
    novelty: float = 0.0          # 二次判断的新颖度 p(未知) ∈ [0,1]
    severe: bool = False          # Dinomaly 视角: 区域够大且图级分数够高

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
    verdict: str = ''             # 最终结论: 正常 / 缺陷类名 / 未知类型 (二次判断给出; 纯检测时只有 正常/异常)
    final_class_id: int = -1      # 整图最终类别 (-1=正常, nc=未知类型)
    final_conf: float = 0.0
    judge_override: bool = False  # Dinomaly 判异常, 但二次判断推翻为正常 (需 --judge-override, 默认关)
    suspect_normal: bool = False  # 二次判断认为所有区域都像正常纹理 (未开推翻时只标记, 不改 is_anomaly)
    heat: Optional[np.ndarray] = field(default=None, repr=False)   # 异常图 (仅供可视化复用, 不进 json)

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
            'verdict': self.verdict,
            'final_class_id': self.final_class_id,
            'final_conf': round(self.final_conf, 6),
            'judge_override': self.judge_override,
            'suspect_normal': self.suspect_normal,
            'regions': [r.to_dict() for r in self.regions],
        }


class FabricAnomalyPipeline:
    """检测 + 分类一体化推理流水线 (GTX 1660 @448 实时目标)."""

    def __init__(self, detector: AnomalyDetector, classifier: Optional[RegionClassifier],
                 class_names: List[str], img_thr: float, pix_thr: Optional[float] = None,
                 mask_ratio: float = 0.5, min_area_ratio: float = 2e-4, dilate: int = 7, box_pad: float = 0.3,
                 device: str = 'cuda', logger=None, max_map_side: int = 1024, max_vis_side: int = 1600,
                 top_k: int = 1, region_score_ratio: float = 0.5,
                 heat_floor_ratio: float = 1.0, heat_region_only: bool = False,
                 severe_score: Optional[float] = None, severe_area: Optional[float] = None,
                 judge_override: bool = False):
        self.det = detector
        self.cls = classifier
        self.class_names = class_names          # 长度 nc+1, 末尾为 '未知类型'
        self.nc = len(class_names) - 1
        self.img_thr = float(img_thr)
        self.pix_thr = float(pix_thr) if pix_thr is not None else self.img_thr
        self.mask_thr = self.pix_thr * mask_ratio
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
        # 热力图显示: 低于 heat_floor = pix_thr*heat_floor_ratio 的像素完全透明 (正常图噪声上限就是 pix_thr);
        # heat_region_only=True 时只在检出的区域框内显示热力
        self.heat_floor_ratio = float(heat_floor_ratio)
        self.heat_region_only = bool(heat_region_only)
        # 二次判断的"严重度"门槛 (calibrate 写在 thresholds.json 里); 缺省: 图级分数 >= 1.3*img_thr 且框面积 >= 0.5%
        self.severe_score = float(severe_score) if severe_score is not None else 1.3 * self.img_thr
        self.severe_area = float(severe_area) if severe_area is not None else 0.005
        self.judge_max_regions = 8
        self.judge_override = bool(judge_override)   # 二次判断是否有权把 Dinomaly2 的异常改回正常
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
                             anomaly_score=score, anomaly_conf=conf, img_thr=self.img_thr,
                             heat=full_map if is_anom else None)
        if is_anom:
            items = self.propose(full_map, score, sx, sy)
            k = len(items) if self.top_k <= 0 else min(self.top_k, len(items))
            if getattr(self.cls, 'is_judge', False):
                k = min(len(items), self.judge_max_regions)      # 二次判断要看全部区域 (不受 top_k 限制, 上限防极端图)
                self._judge(img_rgb, result, items, k, score, w0, h0, full_map)
            else:
                self._classify_crops(img_rgb, result, items, k, w0, h0)
        else:
            result.verdict, result.final_class_id = '正常', -1
            result.final_conf = 1.0 - conf
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def propose(self, full_map: np.ndarray, score: float, sx: float, sy: float):
        """异常图 -> 异常区域 [(区域 dict, 原图像素框)], 按区域均分降序. 推理与二次判断训练共用这一条路径."""
        # 掩码阈值 = max(全局阈值, 图内最高分 * region_score_ratio): 只圈最热的异常区域
        mask_thr_eff = max(self.mask_thr, score * self.region_score_ratio)
        regions = self.det.segment(full_map, mask_thr_eff, self.min_area_ratio, self.dilate, self.box_pad)
        items = []
        for r in regions:
            x1, y1, x2, y2 = r['box']
            items.append((r, (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))))
        return items

    def _region(self, r, box, **kw) -> RegionResult:
        return RegionResult(box=box, area=r['area'], anomaly_score=r['mean_score'],
                            anomaly_conf=deviation_confidence(r['mean_score'], self.img_thr), **kw)

    def _classify_crops(self, img_rgb, result, items, k, w0, h0):
        """旧式分类 (cnn / embed): 每个区域裁剪后单独分类, top_k 之外标未知类型 (classified=false)."""
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
            cr = cls_map.get(i)
            if cr is None:
                result.regions.append(self._region(r, box, class_id=self.nc, class_name='未知类型',
                                                   class_conf=0.0, classified=False, probs=[]))
            else:
                cid = cr['class_id']
                result.regions.append(self._region(
                    r, box, class_id=cid, class_name=self.class_names[cid] if cid < len(self.class_names) else '未知类型',
                    class_conf=cr['class_conf'], classified=True, probs=cr['probs'], unknown=cid == self.nc))
        if result.regions:
            top = result.regions[0]
            result.final_class_id, result.final_conf, result.verdict = top.class_id, top.class_conf, top.class_name
        else:
            result.verdict, result.final_class_id = '异常', self.nc

    def _judge(self, img_rgb, result, items, k, score, w0, h0, full_map):
        """二次判断: 整图 + 区域一起看 (src/judge.py). 先推翻误报, 再给每个区域 类别 / 未知异常."""
        cls = self.cls
        sel = list(range(k))
        heat = heat_to_grid(full_map, self.img_thr, cls.ctx_size) if cls.meta.get('use_heat') else None
        ana = cls.analyse(img_rgb, [items[i][1] for i in sel], heat=heat) if sel else None
        severe_img = score >= self.severe_score
        judged = {}
        if ana is not None:
            for j, i in enumerate(sel):
                r, box = items[i]
                probs, p_nov = ana['probs'][j], float(ana['p_nov'][j])
                sev = bool(severe_img and (box[2] - box[0]) * (box[3] - box[1]) >= self.severe_area * w0 * h0)
                cid, cconf, unk = cls.decide(probs, p_nov, sev)
                judged[i] = (cid, cconf, unk, p_nov, sev, probs)
            # 所有区域都"像正常纹理": 默认只标记 suspect_normal, 不改 Dinomaly2 的结论; 开 --judge-override 才推翻
            if len(judged) == len(items) and all(ana['probs'][j][self.nc] >= cls.normal_thr for j in range(len(sel))):
                result.suspect_normal = True
                if self.judge_override:
                    result.is_anomaly, result.judge_override, result.heat = False, True, None
                    result.verdict, result.final_class_id, result.final_conf = '正常', -1, float(
                        np.mean([ana['probs'][j][self.nc] for j in range(len(sel))]))
                    return
        for i, (r, box) in enumerate(items):
            if i in judged:
                cid, cconf, unk, p_nov, sev, probs = judged[i]
                result.regions.append(self._region(
                    r, box, class_id=cid, class_name=self.class_names[cid], class_conf=cconf, classified=True,
                    probs=[round(float(v), 5) for v in probs[:self.nc]], unknown=unk, novelty=p_nov, severe=sev))
            else:
                result.regions.append(self._region(r, box, class_id=self.nc, class_name='未知类型', class_conf=0.0,
                                                   classified=False, probs=[]))
        if judged:
            top = result.regions[min(judged)]
            result.final_class_id, result.final_conf, result.verdict = top.class_id, top.class_conf, top.class_name
        else:
            result.verdict, result.final_class_id = '异常', self.nc

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
            full_map = result.heat
            if full_map is None:                       # 兼容: result 不带异常图时才重算
                full_map, _, _ = self.det.predict(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                                  max_map_side=self.max_map_side)
            m = cv2.resize(full_map, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
            region_boxes = None
            if self.heat_region_only:
                region_boxes = [tuple(int(round(v * vscale)) for v in r.box) for r in result.regions]
            vis = render_heatmap(vis, m, floor=self.pix_thr * self.heat_floor_ratio,
                                 peak=float(m.max()), region_boxes=region_boxes)   # 图级分数可能是 top-N% 均值, 峰值取图中真实最大值
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


def render_heatmap(vis_bgr: np.ndarray, amap: np.ndarray, floor: float, peak: float, max_alpha: float = 0.6,
                   region_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
    """把异常图以 **绝对阈值** 叠到图上.

    旧做法按每张图自己的 50%~99% 分位数拉伸, 会把正常背景里本来就很低的起伏放大成满屏红蓝噪声.
    现在: 分数 <= floor (正常图噪声上限) 的像素 alpha=0, 完全不上色; floor~peak 线性映射到 JET 颜色,
    alpha 也随强度增大 -> 只有真正偏离正常的地方才亮起来.
    region_boxes 非 None 时, 框外的热力一律清零 (框坐标与 vis_bgr 同尺度).
    """
    peak = max(float(peak), floor + 1e-6)
    t = np.clip((amap - floor) / (peak - floor), 0.0, 1.0)
    if region_boxes is not None:
        keep = np.zeros(t.shape, dtype=bool)
        for x1, y1, x2, y2 in region_boxes:
            keep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        t = np.where(keep, t, 0.0)
    heat = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (t * max_alpha)[..., None].astype(np.float32)
    out = vis_bgr.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def warn_threshold_mismatch(thr: dict, det, log) -> None:
    """核对 thresholds.json 记录的训练分辨率 vs 当前推理分辨率, 不一致则告警 (分数尺度不同会全漏/全报)."""
    if 'image_size' in thr and int(thr['image_size']) != int(det.image_size):
        log(f'⚠️ 警告: 阈值文件是在 image_size={thr["image_size"]} 下校准的, 当前推理为 {det.image_size} —— '
            f'分数尺度不同会导致全漏/全报! 请用训练时的分辨率重新 calibrate')


def build_pipeline(det_ckpt: str, cls_ckpt: Optional[str], thresholds_json: str,
                   class_names: List[str], device: str = 'cuda', half: bool = False,
                   image_size: Optional[int] = None, logger=None, top_k: int = 1,
                   region_score_ratio: float = 0.5, heat_floor_ratio: float = 1.0,
                   heat_region_only: bool = False, judge_override: bool = False) -> FabricAnomalyPipeline:
    """从 checkpoint 路径构建完整流水线 (cls_ckpt=None 时为纯正异常检测, 不加载分类器)."""
    from .detect import load_json
    log = logger or get_logger('pipeline')
    det = AnomalyDetector.load(det_ckpt, device=device, logger=log, half=half)
    if image_size and image_size != det.image_size:
        # 慎用: 推理分辨率必须与训练一致, 否则分数尺度变化、阈值失效
        det.image_size = image_size
        det.crop_size = image_size   # 不裁边缘 (与默认训练配置一致)
        det.offset = (det.image_size - det.crop_size) // 2
        from .detect import get_eval_transform
        det.tf = get_eval_transform(det.image_size, det.crop_size)
    thr = load_json(thresholds_json)
    warn_threshold_mismatch(thr, det, log)
    # 阈值是在某个平滑 sigma 下校准的: 推理自动沿用, 不再依赖手工保持 训练/校准/推理 三处一致
    if 'smooth_sigma' in thr and abs(float(thr['smooth_sigma']) - det.smooth_sigma) > 1e-6:
        log(f'[pipeline] 阈值校准时 smooth_sigma={thr["smooth_sigma"]} (权重自带 {det.smooth_sigma}), 推理沿用校准值')
        det.set_smooth_sigma(float(thr['smooth_sigma']))
    if 'score_ratio' in thr:
        det.set_score_ratio(float(thr['score_ratio']))      # 图级分数取法同样沿用校准时的设置
    cls = None
    if cls_ckpt and os.path.exists(cls_ckpt):
        cls = load_classifier(cls_ckpt, nc=len(class_names) - 1, names=class_names, device=device, logger=log)
        trained_thr = getattr(cls, 'meta', {}).get('img_thr')
        if trained_thr and abs(trained_thr - thr['img_thr']) > 0.05 * trained_thr:
            log(f'⚠️ 二次判断分类器训练时 img_thr={trained_thr:.4f}, 现在的 thresholds.json 是 {thr["img_thr"]:.4f}: '
                f'热力特征按 img_thr 归一化, 差异较大时请重新运行 train_classifier.py')
    else:
        log('[pipeline] 未提供分类器权重, 只做异常检测 (区域标为未知类型)')
    return FabricAnomalyPipeline(det, cls, class_names, img_thr=thr['img_thr'], pix_thr=thr.get('pix_thr'),
                                  device=device, logger=log, top_k=top_k,
                                  region_score_ratio=region_score_ratio, heat_floor_ratio=heat_floor_ratio,
                                  heat_region_only=heat_region_only, severe_score=thr.get('severe_score'),
                                  severe_area=thr.get('severe_area'), judge_override=judge_override)
