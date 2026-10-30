"""推理封装: 预处理 / 大图切块 / 区域分析 / 可视化.

单张图片的典型用法::

    predictor = FabricPredictor("runs/fabric19_fabricncd_s/best.pt")
    result = predictor.predict(pil_image)
    predictor.save_overlay(pil_image, result, "out.png")

对产线大幅面 (例如 4096x1024) 图像, ``predict(..., tile=True)`` 会按
``tile_size`` 切块并带重叠区融合异常图, 避免整图缩放导致小疵点消失.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .models.fabric_ncd import FabricNCD
from .ncd import RegionAnalyzer
from .ops import topk_mean
from .utils import IMAGENET_MEAN, IMAGENET_STD, deep_update, load_yaml, resolve_device

_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _resolve_image_size(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """解析 ``data.image_size``; 值为 ``"auto"`` 时按数据集原图比例推断."""
    raw = cfg.get("data", {}).get("image_size", [512, 512])
    if isinstance(raw, str):
        from .data.yolo import resolve_image_size

        try:
            return resolve_image_size(cfg)
        except Exception as exc:  # pragma: no cover - 数据集不可达时的兜底
            print(f"[warn] image_size=auto 解析失败 ({exc}), 回退 1024x576")
            return 1024, 576
    return as_hw(raw)


def as_hw(value: Any, default: Tuple[int, int] = (512, 512)) -> Tuple[int, int]:
    """把 int / [h, w] / (h, w) 统一成 (h, w)."""
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, (int, float)):
        return int(value), int(value)
    if len(value) == 1:
        return int(value[0]), int(value[0])
    return int(value[0]), int(value[1])


def to_rgb_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    return arr[..., :3].astype(np.uint8)


def _prep_tensor(rgb: np.ndarray, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray(rgb).resize((int(size[1]), int(size[0])), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    arr = (resized - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None].to(device)


def _prep_tensor_gpu(
    rgb: np.ndarray,
    size: Tuple[int, int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
) -> torch.Tensor:
    """在 GPU 上做 resize + 归一化.

    1600x846 的图在 CPU 上用 PIL 双线性缩放要 2~3 ms, 放到 GPU 上
    只要 0.2 ms, 而且省掉一次 CPU 反量化/量化. 代价是一次 10 MB 的 H2D 传输.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(device, non_blocking=True)
    x = tensor.permute(2, 0, 1)[None].float().div_(255.0)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=(int(size[0]), int(size[1])), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    if dtype is not None:
        x = x.to(dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _out_hw(size: Tuple[int, int], stride: int) -> Tuple[int, int]:
    stride = max(1, int(stride))
    return max(8, int(math.ceil(size[0] / stride))), max(8, int(math.ceil(size[1] / stride)))


@torch.no_grad()
def _maps_single(
    model,
    rgb,
    device,
    input_size,
    want_embedding,
    out_stride: int = 1,
    prep: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    weave_desc: Optional[torch.Tensor] = None,
    probs_mul: int = 2,
):
    h, w = rgb.shape[:2]
    if prep == "gpu" and device.type == "cuda":
        tensor = _prep_tensor_gpu(rgb, input_size, device, dtype, channels_last)
    else:
        tensor = _prep_tensor(rgb, input_size, device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
    out = model(tensor) if weave_desc is None else model(tensor, weave_desc=weave_desc)
    sh, sw = _out_hw((h, w), out_stride)
    score = F.interpolate(out["anomaly_score"], size=(sh, sw), mode="bilinear", align_corners=False)
    cloth = F.interpolate(out["cloth_prob"], size=(sh, sw), mode="bilinear", align_corners=False)
    probs = None
    pmul = max(1, int(probs_mul))
    if out.get("class_logits") is not None:
        ph_, pw_ = _out_hw((h, w), out_stride * pmul)
        probs = torch.softmax(
            F.interpolate(out["class_logits"], size=(ph_, pw_), mode="bilinear", align_corners=False), dim=1
        )
    maps = {
        "score": score[0, 0].float().cpu().numpy(),
        "cloth": cloth[0, 0].float().cpu().numpy(),
        "probs": None if probs is None else probs[0].float().cpu().numpy(),
        "weave": {k: float(v[0]) for k, v in (out.get("weave") or {}).items()},
        "stride": max(1, int(out_stride)),
        "probs_scale": pmul,
        "shape": (h, w),
    }
    if want_embedding:
        emb = F.interpolate(out["embedding"], size=(sh, sw), mode="bilinear", align_corners=False)
        maps["embedding"] = F.normalize(emb.float(), dim=1)[0].cpu().numpy()
    return maps


@torch.no_grad()
def _maps_tiled(
    model,
    rgb: np.ndarray,
    device: torch.device,
    input_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    tile_overlap: int,
    tile_batch: int = 4,
    want_embedding: bool = False,
    out_stride: int = 1,
    prep: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    probs_mul: int = 2,
):
    h, w = rgb.shape[:2]
    sh, sw = _out_hw((h, w), out_stride)
    pmul = max(1, int(probs_mul))
    psh, psw = _out_hw((h, w), out_stride * pmul)
    score_map = np.zeros((sh, sw), dtype=np.float32)
    cloth_map = np.zeros((sh, sw), dtype=np.float32)
    probs_map = np.zeros((model.num_classes, psh, psw), dtype=np.float32)
    emb_map = None
    tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
    step_y = max(8, tile_h - tile_overlap)
    step_x = max(8, tile_w - tile_overlap)

    tiles, positions = [], []
    for y in range(0, max(1, h - tile_overlap), step_y):
        for x in range(0, max(1, w - tile_overlap), step_x):
            y2, x2 = min(y + tile_h, h), min(x + tile_w, w)
            y1, x1 = max(0, y2 - tile_h), max(0, x2 - tile_w)
            if prep == "gpu" and device.type == "cuda":
                tiles.append(_prep_tensor_gpu(rgb[y1:y2, x1:x2], input_size, device, dtype, channels_last))
            else:
                tiles.append(_prep_tensor(rgb[y1:y2, x1:x2], input_size, device))
            positions.append((y1, x1, y2, x2))
    if not tiles:
        return _maps_single(
            model, rgb, device, input_size, want_embedding,
            out_stride=out_stride, prep=prep, dtype=dtype, channels_last=channels_last,
        )

    if want_embedding:
        emb_dim = int(model.classifier.dim)
        emb_map = np.zeros((emb_dim, sh, sw), dtype=np.float32)

    s = max(1, int(out_stride))
    for start in range(0, len(tiles), max(1, tile_batch)):
        chunk = torch.cat(tiles[start : start + tile_batch], dim=0)
        if dtype is not None:
            chunk = chunk.to(dtype)
        out = model(chunk)
        for j, idx in enumerate(range(start, min(start + tile_batch, len(tiles)))):
            y1, x1, y2, x2 = positions[idx]
            dy1, dx1 = y1 // s, x1 // s
            dy2 = min(sh, int(math.ceil(y2 / s)))
            dx2 = min(sw, int(math.ceil(x2 / s)))
            th, tw = max(1, dy2 - dy1), max(1, dx2 - dx1)
            score = F.interpolate(out["anomaly_score"][j : j + 1], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
            cloth = F.interpolate(out["cloth_prob"][j : j + 1], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
            score_map[dy1:dy2, dx1:dx2] = np.maximum(score_map[dy1:dy2, dx1:dx2], score.float().cpu().numpy())
            cloth_map[dy1:dy2, dx1:dx2] = np.maximum(cloth_map[dy1:dy2, dx1:dx2], cloth.float().cpu().numpy())
            if out.get("class_logits") is not None:
                s2 = s * pmul
                py1, px1 = y1 // s2, x1 // s2
                py2 = min(psh, int(math.ceil(y2 / s2)))
                px2 = min(psw, int(math.ceil(x2 / s2)))
                pth, ptw = max(1, py2 - py1), max(1, px2 - px1)
                p = F.interpolate(out["class_logits"][j : j + 1], size=(pth, ptw), mode="bilinear", align_corners=False)
                p = torch.softmax(p.float(), dim=1)[0].cpu().numpy()
                probs_map[:, py1:py2, px1:px2] = np.maximum(probs_map[:, py1:py2, px1:px2], p)
            if emb_map is not None:
                e = F.interpolate(out["embedding"][j : j + 1], size=(th, tw), mode="bilinear", align_corners=False)
                e = F.normalize(e.float(), dim=1)[0].cpu().numpy()
                emb_map[:, dy1:dy2, dx1:dx2] = np.maximum(emb_map[:, dy1:dy2, dx1:dx2], e)
    maps = {
        "score": score_map, "cloth": cloth_map, "probs": probs_map, "weave": {},
        "stride": s, "probs_scale": pmul, "shape": (h, w),
    }
    if emb_map is not None:
        maps["embedding"] = emb_map
    return maps


@torch.no_grad()
def forward_maps(
    model,
    image: Image.Image | np.ndarray,
    device: torch.device,
    input_size: Tuple[int, int],
    tile: bool = False,
    tile_size: int | Tuple[int, int] = 512,
    tile_overlap: int = 128,
    tile_batch: int = 4,
    want_embedding: bool = False,
    out_stride: int = 1,
    prep: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    weave_desc: Optional[torch.Tensor] = None,
    probs_mul: int = 2,
) -> Dict[str, np.ndarray]:
    """统一的前向入口: 整图缩放或切块, 输出原图分辨率下的 score/cloth/probs.

    训练/评估/推理共用同一个函数, 保证三者的一致性.
    """
    rgb = to_rgb_uint8(image)
    h, w = rgb.shape[:2]
    tile_hw = as_hw(tile_size, tuple(input_size))
    if tile and (h > tile_hw[0] or w > tile_hw[1]):
        return _maps_tiled(
            model, rgb, device, input_size, tile_hw, tile_overlap, tile_batch, want_embedding,
            out_stride=out_stride, prep=prep, dtype=dtype, channels_last=channels_last, probs_mul=probs_mul,
        )
    return _maps_single(
        model, rgb, device, input_size, want_embedding,
        out_stride=out_stride, prep=prep, dtype=dtype, channels_last=channels_last,
        weave_desc=weave_desc, probs_mul=probs_mul,
    )


class FabricPredictor:
    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        config: Optional[str | Path | Dict[str, Any]] = None,
        device: str = "auto",
        threshold: Optional[float] = None,
        tile: Optional[bool] = None,
        tile_size: Optional[int | Tuple[int, int]] = None,
        tile_overlap: Optional[int] = None,
        out_stride: Optional[int] = None,
        fp16: Optional[bool] = None,
        prep_gpu: Optional[bool] = None,
        weave_interval: Optional[int] = None,
        memory_stride: Optional[int] = None,
        channels_last: Optional[bool] = None,
    ) -> None:
        self.device = resolve_device(device)
        state: Dict[str, Any] = {}
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # 配置优先级: 用户 config > checkpoint 内配置 > 默认 configs/fabric19.yaml
        cfg: Dict[str, Any] = {}
        if state.get("config"):
            cfg = deep_update(cfg, state["config"])
        default_cfg = Path("configs/fabric19.yaml")
        if default_cfg.exists():
            cfg = deep_update(load_yaml(default_cfg), cfg)
        if config is not None:
            user_cfg = load_yaml(config) if isinstance(config, (str, Path)) else config
            cfg = deep_update(cfg, user_cfg)

        # image_size=auto 时优先用 checkpoint 里记录的 (训练时已解析成具体数值),
        # 免得推理端因为读不到数据集而回退到猜的尺寸
        if isinstance(cfg.get("data", {}).get("image_size"), str):
            saved = (state.get("config") or {}).get("data", {}).get("image_size")
            if saved and not isinstance(saved, str):
                cfg.setdefault("data", {})["image_size"] = list(saved)

        self.cfg = cfg
        self.calibration: Dict[str, float] = dict(state.get("calibration") or {})
        self.names: List[str] = list(state.get("names") or cfg.get("data", {}).get("names") or [])
        self.model = FabricNCD(cfg)
        if checkpoint:
            key = "ema" if "ema" in state else "model"
            try:
                self.model.load_state_dict(state[key])
            except RuntimeError as exc:
                # 结构升级 (例如新增 border_prior 两路输入) 后旧权重形状不一致: 尽量加载
                missing, unexpected = self.model.load_state_dict(state[key], strict=False)
                print(
                    f"[warn] 权重与当前结构不完全一致 ({exc.__class__.__name__}), "
                    f"已跳过 {len(missing)} 个参数 / {len(unexpected)} 个多余参数; "
                    "建议用新结构重新训练或微调"
                )
        # 记忆项尺度必须和训练时校准出来的一致, 否则记忆库分数尺度漂移
        scale = float(self.calibration.get("memory_distance_scale", 0.0) or 0.0)
        if scale > 0 and hasattr(self.model, "memory_distance_scale"):
            self.model.memory_distance_scale = scale
        self.model.to(self.device).eval()
        if not self.names:
            self.names = [f"class_{i}" for i in range(self.model.num_classes)]

        infer_cfg = cfg.get("infer", {})
        self.image_size = _resolve_image_size(cfg)
        self.tile = bool(infer_cfg.get("tile", False) if tile is None else tile)
        # 默认切块尺寸 == 网络输入尺寸: 该尺寸下切块是 1:1 原像素, 不做二次缩放
        self.tile_size = as_hw(tile_size or infer_cfg.get("tile_size") or self.image_size)
        self.tile_overlap = int(tile_overlap if tile_overlap is not None else infer_cfg.get("tile_overlap", 128))
        self.tile_batch = int(infer_cfg.get("tile_batch", 4))

        # ---------------- 实时推理相关开关 ---------------- #
        # out_stride: 输出分数图的下采样倍数 (加速的主要手段, 不改变网络计算量)
        self.out_stride = max(1, int(out_stride if out_stride is not None else infer_cfg.get("out_stride", 1)))
        # 类别概率图再降一档 (只影响区域平均, 不影响定位) —— 省掉 19 通道上采样
        self.probs_mul = max(1, int(infer_cfg.get("probs_mul", 2)))
        # fp16: 1660/20 系等无 Tensor Core 的卡也能吃到 ~1.5-2x (FP16 双发射)
        want_fp16 = bool(infer_cfg.get("fp16", False) if fp16 is None else fp16)
        self.fp16 = bool(want_fp16 and self.device.type == "cuda")
        self.prep_gpu = bool(
            (infer_cfg.get("prep_gpu", True) if prep_gpu is None else prep_gpu) and self.device.type == "cuda"
        )
        self.channels_last = bool(
            (infer_cfg.get("channels_last", True) if channels_last is None else channels_last)
            and self.device.type == "cuda"
        )
        self.weave_interval = max(1, int(weave_interval if weave_interval is not None else infer_cfg.get("weave_interval", 1)))
        self._weave_desc: Optional[torch.Tensor] = None
        self._frame_idx = 0
        stride = memory_stride if memory_stride is not None else infer_cfg.get("memory_stride")
        if stride is not None and getattr(self.model, "memory", None) is not None:
            self.model.memory.query_stride = max(1, int(stride))
        self.memory_stride = int(getattr(self.model.memory, "query_stride", 1))

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.dtype = torch.float16 if self.fp16 else None
        if self.dtype is not None:
            self.model = self.model.half()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        eval_cfg = cfg.get("eval", {})
        # 阈值优先级: 命令行 > 配置 > 正常集校准值 (checkpoint 内) > 自适应
        calib_thr = self.calibration.get("pixel_threshold")
        resolved_thr = threshold if threshold is not None else eval_cfg.get("threshold")
        if resolved_thr is None and calib_thr is not None:
            resolved_thr = float(calib_thr)
        self.analyzer = RegionAnalyzer(
            names=self.names,
            threshold=resolved_thr,
            sigma=float(eval_cfg.get("threshold_sigma", 3.0)),
            min_area_ratio=float(eval_cfg.get("min_area_ratio", 4e-4)),
            merge_strategy=str(eval_cfg.get("merge_strategy", "area_weighted")),
        )
        self.threshold = self.analyzer.threshold

    # ------------------------------------------------------------------ #
    def preprocess(self, image: Image.Image | np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        rgb = to_rgb_uint8(image)
        return _prep_tensor(rgb, self.image_size, self.device), rgb.shape[:2]

    def _forward_single(self, image: Image.Image | np.ndarray, want_embedding: bool = False) -> Dict[str, np.ndarray]:
        return forward_maps(
            self.model, image, self.device, self.image_size,
            tile=False, want_embedding=want_embedding,
            out_stride=self.out_stride, prep="gpu" if self.prep_gpu else "cpu",
            dtype=self.dtype, channels_last=self.channels_last,
            weave_desc=self._weave_desc, probs_mul=self.probs_mul,
        )

    def _forward_tiled(self, image: Image.Image | np.ndarray, want_embedding: bool = False) -> Dict[str, np.ndarray]:
        return forward_maps(
            self.model, image, self.device, self.image_size,
            tile=True, tile_size=self.tile_size, tile_overlap=self.tile_overlap,
            tile_batch=self.tile_batch, want_embedding=want_embedding,
            out_stride=self.out_stride, prep="gpu" if self.prep_gpu else "cpu",
            dtype=self.dtype, channels_last=self.channels_last, probs_mul=self.probs_mul,
        )

    # ------------------------------------------------------------------ #
    def _should_refresh_weave(self) -> bool:
        if self.weave_interval <= 1:
            return True
        return self._weave_desc is None or (self._frame_idx % self.weave_interval == 0)

    def _cache_weave(self, maps: Dict[str, np.ndarray]) -> None:
        stats = maps.get("weave") or {}
        if not stats:
            return
        from .models.texture import descriptor_from_stats

        desc = descriptor_from_stats(
            {k: torch.tensor([float(v)], device=self.device) for k, v in stats.items()}
        )
        # 半精度模型要求描述子也是 half, 否则第一层 Linear 会 dtype 不匹配
        self._weave_desc = desc.to(dtype=self.dtype) if self.dtype is not None else desc

    def predict_maps(self, image: Image.Image | np.ndarray, with_embedding: bool = False) -> Dict[str, np.ndarray]:
        """第一阶段: 网络前向, 返回原图坐标系下的分数图/布面图/类别概率 (GPU 段)."""
        refresh = self._should_refresh_weave()
        if refresh:
            self._weave_desc = None
        forward = self._forward_tiled if self.tile else self._forward_single
        maps = forward(image, want_embedding=with_embedding)
        if refresh:
            self._cache_weave(maps)
        self._frame_idx += 1
        return maps

    def analyze_maps(self, maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """第二阶段: 区域分析 + 图像级判定 (纯 CPU, 可与下一帧的 GPU 计算重叠)."""
        score = maps["score"] * (maps["cloth"] > 0.5)
        infos, image_level = self.analyzer.analyze(
            score, maps["cloth"], maps["probs"], probs_scale=int(maps.get("probs_scale", 1))
        )
        image_score = topk_mean(score)
        image_level["image_score"] = image_score
        if self.calibration:
            mean = self.calibration.get("image_mean", 0.0)
            std = max(self.calibration.get("image_std", 0.0), 1e-6)
            image_level["image_z"] = float((image_score - mean) / std)
        image_level["cloth_ratio"] = float((maps["cloth"] > 0.5).mean())
        image_level["weave"] = maps.get("weave", {})
        pred_cls, pred_score = -1, 0.0
        if image_level.get("probs") is not None:
            pred_cls = int(np.argmax(image_level["probs"]))
            pred_score = float(image_level["probs"][pred_cls])
        return {
            "regions": [r.as_dict(self.names) for r in infos],
            "image": {
                **{k: v for k, v in image_level.items() if k != "probs"},
                "cls": pred_cls,
                "name": self.names[pred_cls] if 0 <= pred_cls < len(self.names) else "unknown",
                "cls_score": pred_score,
                "is_abnormal": bool(
                    image_level["image_score"]
                    > float(
                        self.calibration.get(
                            "image_threshold",
                            self.cfg.get("infer", {}).get("image_score_threshold", 0.5),
                        )
                    )
                ),
            },
            "maps": maps,
        }

    def predict(self, image: Image.Image | np.ndarray, with_embedding: bool = False) -> Dict[str, Any]:
        return self.analyze_maps(self.predict_maps(image, with_embedding=with_embedding))

    # ------------------------------------------------------------------ #
    def save_overlay(
        self,
        image: Image.Image | np.ndarray,
        result: Dict[str, Any],
        path: str | Path,
        draw_heatmap: bool = True,
    ) -> None:
        rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
        h, w = rgb.shape[:2]
        score = result["maps"]["score"]
        if score.shape != (h, w):
            score = np.asarray(Image.fromarray(score.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
        if draw_heatmap:
            # 以校准阈值为参考做可视化归一化: 阈值处约为 0.5, 背景接近 0
            thr = float(self.analyzer.threshold or 0.5)
            norm = np.clip(score / max(2.0 * thr, 1e-6), 0, 1)
            heat = _colorize(norm)
            alpha = (0.7 * np.clip((norm - 0.15) / 0.45, 0, 1))[..., None]
            rgb = (rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
        pil = Image.fromarray(rgb)
        drawer = ImageDraw.Draw(pil)
        font = _load_font(max(10, int(min(h, w) / 44)))
        for region in result["regions"]:
            x1, y1, x2, y2 = region["bbox"]
            box = [x1 * w, y1 * h, x2 * w, y2 * h]
            drawer.rectangle(box, outline=(255, 40, 40), width=2)
            label = f"{region['name']} {region['score']:.2f}"
            tw, th = drawer.textbbox((0, 0), label, font=font)[2:]
            tx, ty = box[0], max(0, box[1] - th - 3)
            drawer.rectangle([tx, ty, tx + tw + 4, ty + th + 3], fill=(255, 40, 40))
            drawer.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=font)
        header = (
            f"{result['image']['name']} {result['image']['cls_score']:.2f} | "
            f"s={result['image']['image_score']:.2f} | n={len(result['regions'])}"
        )
        tw, th = drawer.textbbox((0, 0), header, font=font)[2:]
        drawer.rectangle([0, 0, min(w, tw + 6), th + 4], fill=(0, 0, 0))
        drawer.text((3, 1), header, fill=(0, 255, 128), font=font)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pil.save(path)


def _colorize(score: np.ndarray) -> np.ndarray:
    """简易 jet 伪彩色."""
    s = np.clip(score, 0, 1)
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return rgb.astype(np.uint8)
