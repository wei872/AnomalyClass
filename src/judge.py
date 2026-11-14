"""二次判断分类器 (Judge): 在 Dinomaly 判「异常」之后, 看整张图再下结论.

和旧的"裁剪区域 -> 单独分类"不同, 这里每个异常区域的特征由三部分拼成 (冻结 DINOv2, 与检测器共用权重文件):

  * 区域特征  : 区域裁剪 (外扩 30%) 的 CLS / patch 均值 / 中层 patch 均值        -> 看清缺陷本身的纹理细节
  * 整图上下文: 整张图的 CLS; 区域内 token 均值 (重点看); 区域外 token 均值 ("其他区域"也参与判断)
  * 几何      : 区域面积 / 宽高比 (经/纬线类缺陷的形状本身就是类别信息)

分类头在这个拼接特征上学习 (余弦头 + Balanced Softmax), 输出 nc+1 类: 前 nc 类是缺陷类型, 最后一类是 "正常"
(= Dinomaly 误报的纹理). 这样第二阶段可以推翻 Dinomaly 的"异常"结论, 但推翻要非常保守 (阈值取 val 缺陷样本
上 p(正常) 的最大值之上).

未知异常 (没见过的缺陷类型) 的判断 = 新颖度 AND 严重度:
  * 新颖度: 5 个互补的开集分数 (Mahalanobis 到最近类 [Mahalanobis++ 式 L2 归一化特征 + PCA + 共享协方差]、
    到最近类原型的余弦距离、最大概率、top1-top2 间隔、能量) 由一个 Logistic 回归融合. 回归的训练数据来自
    "留一类" (leave-one-class-out): 轮流把某个已知类当作"没见过的类" (头里去掉它), 它的 val 样本为正例,
    其余已知类 val 样本为负例 —— 不需要真实的未知样本.
  * 严重度 (来自 Dinomaly): 图级分数和区域面积都不在 val 缺陷里最弱的 25% -> "区域较大且异常严重".
  只有 新颖 且 严重 才判「未知异常」; 严重度低的区域哪怕不像任何一类, 也给最接近的类 (微弱/很小的区域本来就难分).

参考: Mahalanobis++ (Mueller & Hein 2025, L2 归一化让 Mahalanobis 在 ViT 上稳定), kNN-OOD (Sun et al. ICML'22),
OpenOOD v1.5 基准 (ViM / kNN / Mahalanobis 家族是后验开集检测里最强的一批), Balanced Softmax (Ren et al. NeurIPS'20).
"""
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .classify import _split_samples, fit_temperature
from .data.yolo import build_crop_index, pil_to_tensor_batch, verify_images
from .dinomaly import encoder as encoder_mod
from .util import get_logger, imread_unicode, list_images

GEOM_DIM = 4
GEOM_SCALE = 0.35
N_BLOCKS = 7


# ------------------------------------------------------------------ 几何 / 视图
def geom_feature(w: float, h: float, W: int, H: int) -> np.ndarray:
    """box 的归一化宽高 (w,h ∈ (0,1]) -> 4 维几何特征: [log 面积, log 宽高比, w, h] (大致缩放到 [-1,1])."""
    a = max(w * h, 1e-6)
    asp = max(w * W, 1e-3) / max(h * H, 1e-3)
    return np.array([(math.log10(a) + 2.5) / 1.5, float(np.clip(math.log(asp) / 1.5, -2, 2)), w, h],
                    dtype=np.float32)


def token_mask(box_xyxy: Sequence[float], W: int, H: int, ctx_size: int) -> np.ndarray:
    """box (原图像素坐标) -> 整图 letterbox 后 token 网格 (g,g) 上的覆盖率 ∈ [0,1] (token 被框覆盖的面积比例)."""
    g = ctx_size // 14
    s = ctx_size / max(W, H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    ox, oy = (ctx_size - nw) // 2, (ctx_size - nh) // 2
    x1, y1, x2, y2 = box_xyxy
    x1, x2 = x1 * s + ox, x2 * s + ox
    y1, y2 = y1 * s + oy, y2 * s + oy
    edges = np.arange(g, dtype=np.float32) * 14
    ox_ = np.clip(np.minimum(x2, edges + 14) - np.maximum(x1, edges), 0, 14) / 14
    oy_ = np.clip(np.minimum(y2, edges + 14) - np.maximum(y1, edges), 0, 14) / 14
    return np.outer(oy_, ox_).astype(np.float32)


def crop_box(img: np.ndarray, box_xyxy: Sequence[float], pad: float, min_side: int = 8) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad))
    cx2, cy2 = min(W, int(math.ceil(x2 + bw * pad))), min(H, int(math.ceil(y2 + bh * pad)))
    if cx2 - cx1 < min_side:
        cx2 = min(W, cx1 + min_side)
    if cy2 - cy1 < min_side:
        cy2 = min(H, cy1 + min_side)
    return img[cy1:cy2, cx1:cx2]


def make_sample(img_rgb: np.ndarray, box_xyxy: Sequence[float], crop_size: int, ctx_size: int, pad: float,
                ctx_tensor: Optional[torch.Tensor] = None):
    """一个区域 -> (crop tensor, ctx tensor, token mask, geom). ctx_tensor 给定时复用 (同一张图的多个区域)."""
    H, W = img_rgb.shape[:2]
    crop = pil_to_tensor_batch([crop_box(img_rgb, box_xyxy, pad)], crop_size, augment=False)[0]
    if ctx_tensor is None:
        ctx_tensor = pil_to_tensor_batch([img_rgb], ctx_size, augment=False)[0]
    x1, y1, x2, y2 = box_xyxy
    m = torch.from_numpy(token_mask(box_xyxy, W, H, ctx_size))
    g = torch.from_numpy(geom_feature((x2 - x1) / W, (y2 - y1) / H, W, H))
    return crop, ctx_tensor, m, g


# ------------------------------------------------------------------ 特征抽取
class JudgeExtractor(nn.Module):
    """冻结 DINOv2: (区域裁剪, 整图, token 掩码, 几何) -> 拼接特征 (7C + 4)."""

    def __init__(self, encoder: nn.Module, amp: bool = True):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        depth = len(encoder.blocks)
        self.mid = max(depth * 2 // 3 - 1, 0)
        self.amp = amp
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = getattr(self.encoder, 'n_storage_tokens', 0)
        self.C = getattr(encoder, 'embed_dim', None) or encoder.cls_token.shape[-1]
        self.dim = N_BLOCKS * self.C + GEOM_DIM

    @torch.no_grad()
    def tokens(self, x: torch.Tensor):
        enc, nreg = self.encoder, self.encoder.num_register_tokens
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(self.amp and x.is_cuda)):
            h = enc.prepare_tokens(x)
            mid = None
            for i, blk in enumerate(enc.blocks):
                h = blk(h)
                if i == self.mid:
                    mid = h
            h = enc.norm(h)
        return h[:, 0].float(), h[:, 1 + nreg:].float(), mid[:, 1 + nreg:].float()

    @torch.no_grad()
    def forward(self, crop: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        c_cls, c_last, c_mid = self.tokens(crop)
        i_cls, i_last, i_mid = self.tokens(ctx)
        m = mask.flatten(1).unsqueeze(-1).to(i_last.dtype)                    # (B,N,1)
        inn = m.sum(1).clamp_min(1e-3)
        out = (1 - m)
        foc_last = (i_last * m).sum(1) / inn                                  # 区域内 token 均值: 重点看
        foc_mid = (i_mid * m).sum(1) / inn
        rest_last = (i_last * out).sum(1) / out.sum(1).clamp_min(1e-3)        # 区域外 token 均值: 其他区域
        blocks = [c_cls, c_last.mean(1), c_mid.mean(1), i_cls, foc_last, foc_mid, rest_last]
        f = torch.cat([F.normalize(b, dim=-1) for b in blocks], -1) / math.sqrt(len(blocks))
        return torch.cat([f, geom.to(f.dtype) * GEOM_SCALE], -1)


class CosineHead(nn.Module):
    def __init__(self, dim: int, n_out: int, init_scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * F.normalize(f, dim=-1) @ F.normalize(self.weight, dim=-1).t()


# ------------------------------------------------------------------ 训练数据集
class JudgeDataset(Dataset):
    """items: [(img_path, label, (cx,cy,w,h) 归一化 | None)]. box=None 的是正常图, 随机取一个与缺陷框同分布大小的框.
    augment=True: 整图翻转 + 颜色抖动 (整图/区域一致) + 框抖动 (模拟检测器框与人工框的差异)."""

    def __init__(self, items, crop_size: int = 224, ctx_size: int = 224, pad: float = 0.3, augment: bool = False,
                 box_jitter: float = 0.3, geom_pool: Optional[List[Tuple[float, float]]] = None,
                 max_side: int = 1024, seed: int = 0):
        self.items, self.crop_size, self.ctx_size, self.pad = items, crop_size, ctx_size, pad
        self.augment, self.box_jitter, self.max_side, self.seed = augment, box_jitter, max_side, seed
        self.geom_pool = geom_pool or [(0.1, 0.1)]
        self._last_path, self._last_img = None, None

    def __len__(self):
        return len(self.items)

    def _load(self, path: str):
        if path != self._last_path:
            img = imread_unicode(path, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f'无法读取图片: {path}')
            if self.max_side and max(img.shape[:2]) > self.max_side:
                s = self.max_side / max(img.shape[:2])
                img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                                 interpolation=cv2.INTER_AREA)
            self._last_path, self._last_img = path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._last_img

    def __getitem__(self, i):
        path, label, box = self.items[i]
        img = self._load(path)
        H, W = img.shape[:2]
        rng = np.random if self.augment else np.random.RandomState(self.seed * 100003 + i)
        if box is None:                                        # 正常图: 随机框
            w, h = self.geom_pool[int(rng.randint(len(self.geom_pool)))]
            cx, cy = float(rng.uniform(w / 2, 1 - w / 2)) if w < 1 else 0.5, float(rng.uniform(h / 2, 1 - h / 2)) if h < 1 else 0.5
        else:
            cx, cy, w, h = box
        if self.augment:
            if rng.rand() < 0.5:
                img, cx = img[:, ::-1], 1.0 - cx
            if rng.rand() < 0.5:
                img, cy = img[::-1], 1.0 - cy
            img = np.ascontiguousarray(img)
            if rng.rand() < 0.6:
                from torchvision import transforms as T
                from PIL import Image
                img = np.array(T.ColorJitter(0.3, 0.3, 0.3, 0.1)(Image.fromarray(img)))
            if box is not None and self.box_jitter > 0:
                j = self.box_jitter
                w = float(min(1.0, w * np.exp(rng.uniform(-j, j))))
                h = float(min(1.0, h * np.exp(rng.uniform(-j, j))))
                cx = float(np.clip(cx + rng.uniform(-j, j) * 0.5 * w, 0.0, 1.0))
                cy = float(np.clip(cy + rng.uniform(-j, j) * 0.5 * h, 0.0, 1.0))
        bxy = (max(0.0, (cx - w / 2) * W), max(0.0, (cy - h / 2) * H), min(float(W), (cx + w / 2) * W),
               min(float(H), (cy + h / 2) * H))
        crop, ctx, m, g = make_sample(img, bxy, self.crop_size, self.ctx_size, self.pad)
        return crop, ctx, m, g, int(label)


@torch.no_grad()
def extract_features(ext: JudgeExtractor, ds: JudgeDataset, device: str, batch_size: int = 32, num_workers: int = 0):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    fs, ys = [], []
    for crop, ctx, m, g, y in loader:
        fs.append(ext(crop.to(device), ctx.to(device), m.to(device), g.to(device)).cpu())
        ys.append(y)
    if not fs:
        return torch.zeros(0, ext.dim), torch.zeros(0, dtype=torch.long)
    return torch.cat(fs), torch.cat(ys)


# ------------------------------------------------------------------ 开集 (新颖度) 打分
class NoveltyScorer:
    """Mahalanobis++ 式: L2 归一化特征 -> PCA(k) -> 类均值 + 共享协方差 (收缩). 同时给出到最近类原型的余弦距离."""

    def __init__(self, k: int = 128, shrink: float = 0.1):
        self.k, self.shrink = k, shrink
        self.mean = self.comps = self.cls_means_x = self.cls_means_z = self.prec = None
        self.classes: List[int] = []

    @staticmethod
    def _norm(x):
        return F.normalize(x.float(), dim=-1)

    def fit_basis(self, feats: torch.Tensor, seed: int = 1, max_n: int = 6000):
        X = self._norm(feats)
        if len(X) > max_n:
            idx = torch.from_numpy(np.random.RandomState(seed).choice(len(X), max_n, replace=False))
            Xs = X[idx]
        else:
            Xs = X
        self.mean = Xs.mean(0)
        k = int(min(self.k, Xs.shape[0] - 1, Xs.shape[1]))
        _, _, Vh = torch.linalg.svd(Xs - self.mean, full_matrices=False)
        self.comps = Vh[:k].contiguous()
        self.k = k

    def fit_classes(self, feats: torch.Tensor, labels: torch.Tensor, classes: Sequence[int]):
        X = self._norm(feats)
        Z = (X - self.mean) @ self.comps.t()
        cls, mx, mz, resid = [], [], [], []
        for c in classes:
            sel = labels == c
            if int(sel.sum()) < 2:
                continue
            cls.append(int(c))
            mx.append(F.normalize(X[sel].mean(0), dim=-1))
            mz.append(Z[sel].mean(0))
            resid.append(Z[sel] - mz[-1])
        self.classes = cls
        self.cls_means_x = torch.stack(mx)
        self.cls_means_z = torch.stack(mz)
        R = torch.cat(resid)
        S = R.t() @ R / max(len(R) - 1, 1)
        S = (1 - self.shrink) * S + self.shrink * (torch.trace(S) / self.k) * torch.eye(self.k)
        self.prec = torch.linalg.inv(S + 1e-6 * torch.eye(self.k))

    @torch.no_grad()
    def score(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (最近类的 Mahalanobis 距离 / k, 到最近类原型的余弦距离)."""
        X = self._norm(feats)
        Z = (X - self.mean) @ self.comps.t()
        d = Z.unsqueeze(1) - self.cls_means_z.unsqueeze(0)                       # (N,K,k)
        maha = torch.einsum('nki,ij,nkj->nk', d, self.prec, d).min(1).values / self.k
        cos = (X @ self.cls_means_x.t()).max(1).values
        return maha, 1.0 - cos

    def state(self) -> dict:
        return {'mean': self.mean, 'comps': self.comps, 'cls_means_x': self.cls_means_x,
                'cls_means_z': self.cls_means_z, 'prec': self.prec, 'classes': self.classes, 'k': self.k,
                'shrink': self.shrink}

    @classmethod
    def from_state(cls, st: dict) -> 'NoveltyScorer':
        o = cls(st['k'], st['shrink'])
        o.mean, o.comps, o.cls_means_x = st['mean'], st['comps'], st['cls_means_x']
        o.cls_means_z, o.prec, o.classes = st['cls_means_z'], st['prec'], list(st['classes'])
        return o


def novelty_features(maha: torch.Tensor, cosd: torch.Tensor, logits: torch.Tensor, temperature: float,
                     nc: int, valid: Optional[Sequence[int]] = None) -> np.ndarray:
    """5 个开集分数 (只在缺陷类上算, 不含"正常"类; valid 给出有效类, 留一类时屏蔽被留出的类)."""
    lg = logits[:, :nc].float() / temperature
    if valid is not None:
        mask = torch.full((nc,), float('-inf'))
        mask[list(valid)] = 0.0
        lg = lg + mask
    p = torch.softmax(lg, dim=1)
    top2 = p.topk(2, dim=1).values
    energy = torch.logsumexp(lg, dim=1)
    f = torch.stack([torch.log1p(maha), cosd, top2[:, 0], top2[:, 0] - top2[:, 1], energy], 1)
    return f.numpy()


class NoveltyFuser:
    """5 个开集分数 -> p(未知). 标准化 + 逻辑回归, 参数可 json 存取."""

    def __init__(self, mean=None, std=None, coef=None, intercept=0.0):
        self.mean, self.std, self.coef, self.intercept = mean, std, coef, intercept

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=500)
        lr.fit((X - self.mean) / self.std, y)
        self.coef, self.intercept = lr.coef_[0], float(lr.intercept_[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.std) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self):
        return {'mean': list(map(float, self.mean)), 'std': list(map(float, self.std)),
                'coef': list(map(float, self.coef)), 'intercept': float(self.intercept)}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d['mean']), np.array(d['std']), np.array(d['coef']), d['intercept'])


# ------------------------------------------------------------------ 推理
class JudgeClassifier:
    """二次判断分类器. analyse() 一次给出每个区域的 (nc+1 类概率, 新颖度 p(未知))."""
    is_judge = True

    def __init__(self, extractor: JudgeExtractor, head: CosineHead, nc: int, meta: dict,
                 scorer: Optional[NoveltyScorer], fuser: Optional[NoveltyFuser], device: str = 'cuda'):
        self.extractor = extractor.to(device).eval()
        self.head = head.to(device).eval()
        self.nc, self.meta, self.scorer, self.fuser, self.device = nc, meta, scorer, fuser, device
        self.crop_size = int(meta.get('crop_size', 224))
        self.ctx_size = int(meta.get('ctx_size', 224))
        self.pad = float(meta.get('crop_pad', 0.3))
        self.temperature = float(meta.get('temperature', 1.0))
        self.unknown_thr = float(meta.get('unknown_thr', 0.5))
        self.nov_thr = float(meta.get('nov_thr', 0.5))
        self.normal_thr = float(meta.get('normal_thr', 0.95))
        if self.scorer is not None:
            for a in ('mean', 'comps', 'cls_means_x', 'cls_means_z', 'prec'):
                setattr(self.scorer, a, getattr(self.scorer, a).to(device))

    @torch.no_grad()
    def _views(self, img_rgb: np.ndarray, boxes: Sequence[Sequence[float]]):
        H, W = img_rgb.shape[:2]
        ctx = pil_to_tensor_batch([img_rgb], self.ctx_size, augment=False)[0]
        crops, ctxs, ms, gs = [], [], [], []
        for b in boxes:
            c, x, m, g = make_sample(img_rgb, b, self.crop_size, self.ctx_size, self.pad, ctx_tensor=ctx)
            crops.append(c); ctxs.append(x); ms.append(m); gs.append(g)
        return [torch.stack(t).to(self.device) for t in (crops, ctxs, ms, gs)]

    @torch.no_grad()
    def analyse(self, img_rgb: np.ndarray, boxes: Sequence[Sequence[float]], tta: bool = True) -> Dict:
        """boxes: 原图像素坐标 (x1,y1,x2,y2) 列表. 返回 {'probs': (R,nc+1), 'p_nov': (R,), 'logits'}.
        probs 已温度缩放; 第 nc 列是"正常". tta: 再对整图+框做一次水平翻转取平均."""
        if len(boxes) == 0:
            return {'probs': np.zeros((0, self.nc + 1), np.float32), 'p_nov': np.zeros(0, np.float32)}
        H, W = img_rgb.shape[:2]
        variants = [(img_rgb, list(boxes))]
        if tta:
            variants.append((np.ascontiguousarray(img_rgb[:, ::-1]), [(W - b[2], b[1], W - b[0], b[3]) for b in boxes]))
        probs_all, nov_all = [], []
        for im, bx in variants:
            f = self.extractor(*self._views(im, bx))
            logits = self.head(f).cpu()
            probs_all.append(torch.softmax(logits / self.temperature, 1).numpy())
            if self.scorer is not None and self.fuser is not None:
                maha, cosd = self.scorer.score(f.cpu() if self.scorer.mean.device.type == 'cpu' else f)
                nf = novelty_features(maha.cpu(), cosd.cpu(), logits, self.temperature, self.nc)
                nov_all.append(self.fuser.predict(nf))
            else:                                                  # 没做留一类校准: 退回最大概率阈值
                pd = torch.softmax(logits[:, :self.nc] / self.temperature, 1).max(1).values.numpy()
                nov_all.append((pd < self.unknown_thr).astype(np.float32))
        return {'probs': np.mean(probs_all, 0).astype(np.float32), 'p_nov': np.mean(nov_all, 0).astype(np.float32)}

    def decide(self, probs: np.ndarray, p_nov: float, severe: bool) -> Tuple[int, float, bool]:
        """单个区域: -> (class_id, conf, is_unknown). 新颖 且 严重 -> 未知异常 (class_id=nc); 否则取最像的缺陷类."""
        pd = probs[:self.nc]
        pd = pd / max(float(pd.sum()), 1e-8)
        cid = int(np.argmax(pd))
        if p_nov >= self.nov_thr and severe:
            return self.nc, float(p_nov), True
        return cid, float(pd[cid]), False

    # ---- 存取 ----
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ck = {'head': self.head.state_dict(), 'meta': dict(self.meta, kind='judge')}
        if self.scorer is not None:
            ck['scorer'] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.scorer.state().items()}
        if self.fuser is not None:
            ck['fuser'] = self.fuser.to_dict()
        if self.meta.get('backbone') == 'tiny_test_14':
            ck['encoder'] = self.extractor.encoder.state_dict()
        torch.save(ck, path)

    @classmethod
    def load(cls, path: str, nc: int, names: List[str], device: str = 'cuda', logger=None,
             allow_random_init: bool = False) -> 'JudgeClassifier':
        from .util import torch_load
        log = logger or (lambda *a, **k: print(*a))
        ck = torch_load(path, map_location='cpu')
        meta = ck['meta']
        enc = encoder_mod.load_encoder(meta['backbone'], weights_dir=meta.get('weights_dir'),
                                       allow_random_init=allow_random_init or 'encoder' in ck, logger=log)
        if 'encoder' in ck:
            enc.load_state_dict(ck['encoder'])
        ext = JudgeExtractor(enc)
        head = CosineHead(ext.dim, nc + 1)
        head.load_state_dict(ck['head'])
        scorer = NoveltyScorer.from_state(ck['scorer']) if 'scorer' in ck else None
        fuser = NoveltyFuser.from_dict(ck['fuser']) if 'fuser' in ck else None
        log(f'[judge] 加载二次判断分类器: {path} (backbone={meta["backbone"]}, 新颖度={"LR+Mahalanobis" if fuser else "最大概率阈值"}, '
            f'nov_thr={meta.get("nov_thr", 0.5):.2f}, normal_thr={meta.get("normal_thr", 0.95):.3f})')
        return cls(ext, head, nc, meta, scorer, fuser, device)


# ------------------------------------------------------------------ 训练
def _bal_acc(y: np.ndarray, p: np.ndarray, classes: Sequence[int]) -> float:
    rec = [float((p[y == c] == c).mean()) for c in classes if (y == c).any()]
    return float(np.mean(rec)) if rec else 0.0


def train_judge(train_images: str, train_labels: Optional[str], val_images: Optional[str], val_labels: Optional[str],
                normal_train: Optional[str], normal_val: Optional[str], out_dir: str = 'checkpoints/classifier',
                nc: int = 19, names: Optional[List[str]] = None, backbone: str = 'dinov2reg_vit_small_14',
                crop_size: int = 224, ctx_size: int = 224, crop_pad: float = 0.3, views: int = 4,
                min_per_class: int = 200, max_reps: int = 16, box_jitter: float = 0.3, epochs: int = 200,
                tau: float = 1.0, max_normal: int = 1500, loco: bool = True, loco_epochs: int = 40,
                severe_q: float = 0.25, device: str = 'cuda', num_workers: int = 0, seed: int = 1, logger=None,
                weights_dir: Optional[str] = None, allow_random_init: bool = False, amp: bool = True) -> str:
    """训练二次判断分类器, 返回 out_dir/best.pt."""
    from collections import Counter
    from .embed_classify import train_head
    log = logger or get_logger('train_judge')
    if crop_size % 14 or ctx_size % 14:
        raise ValueError('crop_size / ctx_size 必须是 14 的倍数')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    def defect_items(images, labels):
        idx, bad = build_crop_index(images, labels, pad=crop_pad)
        if bad:
            log(f'警告: 跳过 {len(bad)} 个无法解码的图片')
        return [(p, int(b[0]), (b[1], b[2], b[3], b[4])) for p, b, _ in idx]

    tr = defect_items(train_images, train_labels)
    if val_images and os.path.isdir(val_images):
        va = defect_items(val_images, val_labels)
    else:
        sp_tr, sp_va = _split_samples([(p, (c,) + b, 0.0) for p, c, b in tr], 0.1, seed)
        tr = [(p, b[0], b[1:]) for p, b, _ in sp_tr]
        va = [(p, b[0], b[1:]) for p, b, _ in sp_va]
        log(f'未提供 val, 从 train 划分: train={len(tr)} val={len(va)}')
    if not va:
        va = tr[:max(1, len(tr) // 10)]
    cnt = Counter(c for _, c, _ in tr)
    nm = (names or []) + ['未知异常']
    log(f'训练框 {len(tr)}, val 框 {len(va)}; 每类: ' + ', '.join(f'{nm[c] if c < len(nm) else c}:{cnt[c]}' for c in sorted(cnt)))
    geom_pool = [(b[2], b[3]) for _, _, b in tr]

    # "正常"类: 正常图 + 随机框 (框大小取自缺陷框分布), 用来学"正常纹理长什么样", 第二阶段据此推翻 Dinomaly 的误报
    NORMAL = nc
    def normal_items(folder, cap):
        fs = list_images(folder) if folder and os.path.isdir(folder) else []
        rng.shuffle(fs)
        ok, bad = [], 0
        for p in fs:                                         # 逐张探测, 够 cap 就停, 坏图 (0 字节/损坏) 跳过
            if len(ok) >= cap:
                break
            if verify_images([p])[0]:
                ok.append(p)
            else:
                bad += 1
        if bad:
            log(f'警告: 跳过 {bad} 个无法解码的正常图')
        return [(p, NORMAL, None) for p in ok]
    nt = normal_items(normal_train, max_normal)
    nv = normal_items(normal_val, 300)
    if not nv and nt:
        nv, nt = nt[:max(1, len(nt) // 10)], nt[max(1, len(nt) // 10):]
    log(f'正常样本: train {len(nt)} 张, val {len(nv)} 张 (每张随机取一个缺陷大小的框)')

    enc = encoder_mod.load_encoder(backbone, weights_dir=weights_dir, allow_random_init=allow_random_init, logger=log)
    ext = JudgeExtractor(enc, amp=amp).to(device).eval()
    mk = lambda items, aug, s=0: JudgeDataset(items, crop_size, ctx_size, crop_pad, augment=aug, box_jitter=box_jitter,
                                                geom_pool=geom_pool, seed=s)
    t0 = time.perf_counter()
    f0, y0 = extract_features(ext, mk(tr + nt, False, seed), device, num_workers=num_workers)
    aug_items = []
    for it in tr:
        reps = int(min(max(views, math.ceil(min_per_class / max(cnt[it[1]], 1))), max_reps))
        aug_items.extend([it] * max(reps - 1, 0))
    aug_items += nt * max(views - 1, 0) if nt else []
    fa, ya = extract_features(ext, mk(aug_items, True, seed + 1), device, num_workers=num_workers) if aug_items \
        else (f0[:0], y0[:0])
    vf, vy = extract_features(ext, mk(va + nv, False, seed + 7), device, num_workers=num_workers)
    feats, labels = torch.cat([f0, fa]), torch.cat([y0, ya])
    log(f'特征缓存: 训练 {len(feats)} (干净 {len(f0)} + 增强 {len(fa)}), val {len(vf)}, 维度 {feats.shape[1]}, '
        f'耗时 {time.perf_counter() - t0:.1f}s')

    head, st = train_head(feats, labels, vf, vy, nc + 1, epochs=epochs, tau=tau, seed=seed, device=device, logger=log)
    with torch.no_grad():
        val_logits = head(vf.to(device)).cpu()
    T = fit_temperature(val_logits, vy, log)
    vy_np = vy.numpy()
    pred = val_logits.argmax(1).numpy()
    defect_mask = vy_np < nc
    log(f'val: 含"正常"类 acc={st["val_acc"]:.4f} bal_acc={st["val_bal_acc"]:.4f} | 仅缺陷框 acc='
        f'{float((pred[defect_mask] == vy_np[defect_mask]).mean()):.4f} (含被判成正常的)')
    probs_v = torch.softmax(val_logits / T, 1).numpy()

    # 推翻"异常"的阈值: 必须高于 val 缺陷样本里 p(正常) 的最大值, 宁可少推翻也不能把真缺陷判成正常
    p_norm_def = probs_v[defect_mask, NORMAL]
    p_norm_nor = probs_v[~defect_mask, NORMAL]
    normal_thr = float(min(max(0.9, (p_norm_def.max() if len(p_norm_def) else 0.0) + 0.02), 0.995))
    log(f'[judge] 推翻阈值 normal_thr={normal_thr:.3f}: val 缺陷框被推翻 {int((p_norm_def >= normal_thr).sum())}/{len(p_norm_def)}, '
        f'val 正常样本被推翻(=误报被纠正) {int((p_norm_nor >= normal_thr).sum())}/{len(p_norm_nor)}')

    # 新颖度 (未知异常): 留一类校准 Mahalanobis++/原型/MSP/间隙/能量 的融合
    d_sel = labels < nc
    scorer = NoveltyScorer()
    scorer.fit_basis(feats[d_sel], seed=seed)
    scorer.fit_classes(feats[d_sel], labels[d_sel], range(nc))
    meta_extra = {}
    fuser = None
    nov_thr = 0.5
    va_def = vf[torch.from_numpy(defect_mask)]
    va_lab = vy[torch.from_numpy(defect_mask)]
    known_classes = [c for c in range(nc) if cnt[c] >= 10 and int((va_lab == c).sum()) >= 3]
    if loco and len(known_classes) >= 3:
        t1 = time.perf_counter()
        Xs, Ys = [], []
        for c in known_classes:
            keep = labels != c
            h_c, _ = train_head(feats[keep], labels[keep], vf[vy != c], vy[vy != c], nc + 1, epochs=loco_epochs,
                                tau=tau, seed=seed, device=device, logger=lambda *a, **k: None)
            with torch.no_grad():
                lg = h_c(va_def.to(device)).cpu()
            sc = NoveltyScorer(scorer.k, scorer.shrink)
            sc.mean, sc.comps = scorer.mean, scorer.comps
            sel = (labels < nc) & (labels != c)
            sc.fit_classes(feats[sel], labels[sel], [k for k in range(nc) if k != c])
            maha, cosd = sc.score(va_def)
            valid = [k for k in range(nc) if k != c]
            Xs.append(novelty_features(maha, cosd, lg, T, nc, valid))
            Ys.append((va_lab == c).numpy().astype(np.int64))
        Xl, Yl = np.concatenate(Xs), np.concatenate(Ys)
        fuser = NoveltyFuser().fit(Xl, Yl)
        p = fuser.predict(Xl)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yl, p))
        known_p = p[Yl == 0]
        nov_thr = float(min(max(np.quantile(known_p, 0.95), 0.5), 0.95))
        log(f'[judge] 留一类校准 ({len(known_classes)} 类, {time.perf_counter() - t1:.0f}s): 把"没见过的类"当未知的 AUROC={auc:.3f}; '
            f'nov_thr={nov_thr:.2f} (已知类误判为未知 ≈{float((known_p >= nov_thr).mean()):.1%}, '
            f'未知类被抓住 ≈{float((p[Yl == 1] >= nov_thr).mean()):.1%})')
        meta_extra['loco_auroc'] = auc
    else:
        log('[judge] 样本太少, 跳过留一类校准, 未知异常退回"最大概率 < unknown_thr"规则')
    max_prob_val = probs_v[defect_mask, :nc].max(1)
    unknown_thr = float(min(max(np.quantile(max_prob_val, 0.05), 0.4), 0.8))

    meta = {'kind': 'judge', 'backbone': backbone, 'nc': nc, 'names': names or [], 'crop_size': crop_size,
            'ctx_size': ctx_size, 'crop_pad': crop_pad, 'temperature': T, 'tau': tau, 'nov_thr': nov_thr,
            'normal_thr': normal_thr, 'unknown_thr': unknown_thr, 'val_acc': st['val_acc'],
            'val_bal_acc': st['val_bal_acc'], 'severe_q': severe_q, 'views': views, 'min_per_class': min_per_class,
            'unknown_id': nc, 'unknown_name': '未知异常', 'normal_id': NORMAL}
    meta.update(meta_extra)
    if weights_dir:
        meta['weights_dir'] = weights_dir
    clf = JudgeClassifier(ext, head.to(device), nc, meta, scorer, fuser, device)
    best = os.path.join(out_dir, 'best.pt')
    clf.save(best)
    with open(os.path.join(out_dir, 'classifier_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in meta.items()}, f, ensure_ascii=False, indent=2)
    log(f'二次判断分类器训练完成: {best} (val_bal_acc={st["val_bal_acc"]:.4f}, nov_thr={nov_thr:.2f}, '
        f'normal_thr={normal_thr:.3f}, T={T:.3f})')
    return best
