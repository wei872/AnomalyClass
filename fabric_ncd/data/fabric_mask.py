"""布面 (前景) / 背景 掩码估计.

真实产线图里, 布幅两侧常出现辊筒、导带、深色背景或反光金属, 它们的纹理与
布面完全不同. 如果让异常/分类分支在这些区域上学习, 模型会把 "背景" 误判
为疵点, 并且当布幅变窄 / 走偏时大面积误报. 因此需要布面前景掩码:

1. 优先使用用户提供的真实掩码 (``mask_dir``);
2. 否则用本文件的 **无监督** 估计器: 局部纹理能量 + 饱和度 + 与画面边缘
   背景色的差异, 再做 Otsu 分割与形态学清理;
3. 两者都没有时退出为 "整幅都是布", 不引入错误先验.

估计器是启发式的, 结果只作为弱监督/预训练目标; 真实掩码一旦提供即可直接使用.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
from ..ops import binary_close, binary_dilate, binary_open, box_filter, connected_components, local_std
from .weave import estimate_period


def _period_pixels(gray: np.ndarray) -> Tuple[float, float]:
    """估计织造周期 (该图像自身的像素).

    在 **原始分辨率** 上取几个 512x512 的窗口做估计, 绝不先缩小: 细密布的经纬
    只有几像素, 一旦缩小就会混叠, 周期会被估成完全错误的值 (6 px 能估成 28 px).
    多个窗口取中位数, 避免局部疵点/反光干扰.
    """
    h, w = gray.shape
    # 窗口位置不再均匀撒点: 布幅只占画面 16% 时, 均匀采样几乎全落在背景上, 侥幸
    # 压到布边的窗口量出来的是台阶边缘的伪周期 (6px 能估成 46px), auto 于是把图
    # 缩得过狠, 经纬纹理被平均掉, 掩码大面积偏移. 改成"先找纹理最旺的地方, 再在那里量".
    field = _texture_field(gray)
    if field is None:
        return 0.0, 0.0
    # 补丁尺度自适应: 布幅很窄时 512 的窗口一定压到背景上 (找不到一块纯布),
    # 这时退到 256 / 128 / 64, 用仍然"整块是纹理"的最大尺度.
    sizes = [s for s in (int(min(512, h, w)), 256, 128, 64) if 64 <= s <= min(h, w)]
    periods: list = []
    for size in sizes:
        anchors = _anchors_for_size(field, size)
        if not anchors:
            continue
        for cy, cx in anchors:
            y = int(np.clip(cy - size // 2, 0, h - size))
            x = int(np.clip(cx - size // 2, 0, w - size))
            patch = gray[y : y + size, x : x + size]
            if not _is_texture_patch(patch):
                continue
            try:
                px, py, conf = estimate_period(patch.astype(np.uint8))
            except Exception:  # pragma: no cover
                continue
            if conf >= 2.0:
                periods.append((float(px), float(py)))
                if len(periods) >= 3:  # 够做分位数了, 不必把所有锚点都量完
                    break
        if periods:  # 用最大的可行尺度就够了, 不再继续往小退
            break
    if not periods:
        return 0.0, 0.0
    arr = np.asarray(periods)
    # 取 25 分位而不是中位数: 宁可按"最细的"那片纹理定尺度, 也不要因为某个窗口
    # 落在背景/反光上而把整图降得过分, 导致经纬纹理混叠
    return float(np.percentile(arr[:, 0], 25)), float(np.percentile(arr[:, 1], 25))



def _block_mean(a: np.ndarray, step: int) -> np.ndarray:
    h, w = a.shape
    hh, ww = (h // step) * step, (w // step) * step
    if hh == 0 or ww == 0:
        return a.astype(np.float32)
    a = a[:hh, :ww].astype(np.float32)
    return a.reshape(hh // step, step, ww // step, step).mean(axis=(1, 3))


def _texture_field(gray: np.ndarray, long_side: int = 256) -> dict | None:
    """"哪里是布"的低分辨率场: 梯度幅值的块均值 + 阈值掩码.

    刻意先在**原始分辨率**上求梯度再分块平均, 而不是先抽样再算方差:
    抽样遇到 10px 的经纬周期可能恰好每次都采到同一相位, 纹理直接被采样成常数 (混叠),
    布面就会被误判成平坦背景.
    """
    h, w = gray.shape
    if min(h, w) < 64:
        return None
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad = gx[:-1, :] + gy[:, :-1]  # 对齐到 (h-1, w-1)
    step = max(1, int(round(max(h, w) / float(long_side))))
    energy = _block_mean(grad, step)
    if energy.size == 0:
        return None
    energy = box_filter(energy, 1)
    peak = float(energy.max())
    if peak < 1.5:  # 整幅都很平坦: 没有可量的纹理
        return None
    return {
        "step": step,
        "energy": energy,
        "mask": (energy > 0.3 * peak).astype(np.float32),
    }


def _anchors_for_size(field: dict, size: int, count: int = 3) -> list:
    """找出能完整容纳 ``size`` 方窗的纹理区中心 (原图坐标).

    做法: 把粗糙纹理掩码按窗口半径"内缩", 剩下的格子为中心时窗口整块落在纹理里;
    再按能量取峰并做非极大值抑制. 这样锚点落在布面**内部**, 而不是布/背景台阶边缘
    (台阶边缘的能量最高, 直接取 argmax 一定取到边缘).
    """
    step = field["step"]
    r = max(1, int(np.ceil(size / 2.0 / step)))
    inner = box_filter(field["mask"], r) > 0.999
    if not inner.any():
        return []
    score = np.where(inner, field["energy"], -1.0)
    min_dist = max(2, r)
    picked: list = []
    for flat_idx in np.argsort(score.ravel())[::-1]:
        if score.ravel()[flat_idx] <= 0:
            break
        ys, xs = np.unravel_index(int(flat_idx), score.shape)
        if any(abs(int(ys) - py) < min_dist and abs(int(xs) - px) < min_dist for py, px in picked):
            continue
        picked.append((int(ys), int(xs)))
        if len(picked) >= count:
            break
    return [(y * step, x * step) for y, x in picked]


def _is_texture_patch(patch: np.ndarray, min_std: float = 3.0) -> bool:
    """窗口是否"整块都是纹理" (而非平坦背景, 也不是布/背景交界).

    交界窗口的四象限亮度差异极大, 任何周期估计都会被台阶边缘带偏, 必须排除.
    """
    if float(patch.std()) < min_std:
        return False
    h, w = patch.shape
    quads = [
        patch[: h // 2, : w // 2],
        patch[: h // 2, w - w // 2 :],
        patch[h - h // 2 :, : w // 2],
        patch[h - h // 2 :, w - w // 2 :],
    ]
    stds = [float(q.std()) for q in quads]
    if min(stds) < min_std:  # 有平坦象限 -> 交界或背景
        return False
    if max(stds) > 2.5 * min(stds):  # 纹理强度要均匀
        return False
    # 四象限亮度还要接近: 布/背景交界处"每块都有纹理"但明暗差一大截,
    # 这种窗口估出的周期完全被台阶边缘带偏 (6px 能估成 46px)
    means = [float(q.mean()) for q in quads]
    return (max(means) - min(means)) <= max(0.5 * min(stds), 6.0)

def _texture_window(gray: np.ndarray) -> Tuple[int, int]:
    """返回 (纹理能量窗口半径, 平滑窗口半径), 单位为该图像自身的像素."""
    pxs, pys = _period_pixels(gray)
    if pxs <= 0 or pys <= 0:
        base = max(3.0, min(gray.shape) / 64.0)
        return int(np.clip(round(base), 3, 16)), int(np.clip(round(base * 2), 2, 32))
    return _period_radii(float(np.sqrt(pxs * pys)))


def _period_radii(period: float, scale: float = 1.0) -> Tuple[int, int]:
    """由**原始分辨率**的经纬周期换算出 (纹理窗口半径, 平滑半径).

    ``scale`` 用于降分辨率估计: 半径必须同步缩小, 否则在小图上用 40px 的窗口平滑,
    回到原图就是 80px 的边界模糊, 窄布条会直接糊到背景里 (覆盖率从 0.169 涨到 0.197).
    也正因为不能就地量小图的周期 (混叠后偏大), 这里一律以原图周期为准.
    """
    tex_r = int(np.clip(round(period * 0.75 * scale), 2, max(2, int(round(16 * scale)))))
    smooth_r = int(np.clip(round(period * 2.5 * scale), 2, max(2, int(round(48 * scale)))))
    return tex_r, smooth_r


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(values.mean())
    prob = hist / total
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega) + 1e-8
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b))
    return float(centers[idx])



def _otsu_separability(values: np.ndarray, threshold: float) -> float:
    """类间方差 / 总方差, 衡量双峰程度 (0~1, 越大越像"前景+背景"两段)."""
    total_var = float(values.var())
    if total_var <= 1e-12:
        return 0.0
    fg = values[values > threshold]
    bg = values[values <= threshold]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    w_fg = fg.size / values.size
    between = w_fg * (1 - w_fg) * (float(fg.mean()) - float(bg.mean())) ** 2
    return float(between / total_var)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """把不与图像边界连通的背景洞填成前景."""
    labels, num = connected_components(mask == 0, connectivity=4)
    if num == 0:
        return mask
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    border.discard(0)
    keep_bg = np.isin(labels, list(border))
    return (~keep_bg).astype(np.uint8)


def estimate_cloth_mask(
    image: np.ndarray,
    min_coverage: float = 0.05,
    max_coverage: float = 0.995,
    edge_prior: bool = True,
    work_size: int | str | None = None,
) -> Tuple[np.ndarray, float]:
    """布面掩码估计 (可在大图下降分辨率计算后最近邻回放).

    ``work_size``: 长边缩到这个尺寸再估计 (掩码本身是大块区域, 降分辨率几乎不影响
    位置). 回放时用最近邻, 保持 0/1 不产生灰边.

    取 ``"auto"`` 时按该图实测的织造周期自动决定: 保证缩放后经纬周期仍 >= 3 px.
    周期 6 px 的细密布只能缩到 1/2 (2440 -> 1224), 周期 20 px 的粗布可以缩到 1/6,
    这样既不会把纹理平均掉, 又能在粗纹理布上真正省时间.

    降采样后还会**回头量一次**小图上的周期: 若纹理已被平均掉 (周期测不到或 < 3 px),
    说明自动尺度不准 (布幅很窄时尤其容易, 周期会被台阶边缘带偏), 于是逐次放大
    ``work_size`` 重试; 实在测不出就退回全分辨率 —— 宁可贵一点, 也不要给出偏移的掩码.
    几何清理 (开/闭/填洞/连通块) 一律在**输出分辨率**上做.
    """
    arr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    h, w = arr.shape[:2]
    long_side = max(h, w)
    period = 0.0
    if isinstance(work_size, str) and work_size.lower() == "auto":
        work_size, period = _plan_work_size(arr)
        for _ in range(3):
            if not work_size or long_side <= int(work_size):
                break
            small = _downscale(arr, int(work_size))
            if _weave_survives(small):
                break
            work_size = min(long_side, int(round(work_size * 2)))
        if work_size and int(work_size) >= long_side:
            work_size = None  # 退到全分辨率

    if work_size and long_side > int(work_size):
        small = _downscale(arr, int(work_size))
        scale = float(small.shape[1]) / float(w)
        radii = _period_radii(period, scale) if period > 0 else None
        # 用户强行给了过小的 work_size 时, 经纬纹理已被平均掉, 纹理线索不可信,
        # 这时把权重让给"颜色/亮度与边缘背景不同"这条不受缩放影响的线索.
        # (实测 2440 宽缩到 512: 纹理主导 IoU 0.906 → 颜色主导 0.99+)
        if _weave_survives(small):
            weights = {"tex_weight": 0.65, "color_weight": 0.35}
        else:
            weights = {"tex_weight": 0.20, "color_weight": 0.80}
        candidate, conf, bg_ratio = _score_mask(
            small, edge_prior=edge_prior, radii=radii, **weights
        )
        if candidate is None:
            return np.ones((h, w), dtype=np.uint8), 0.2
        if w / max(float(small.shape[1]), 1.0) <= 2.5:
            # 倍数不大时在小图上清理再回放: 边界误差 <= 1 个回放像素, 但省 200ms+
            mask = _upsample_mask(_clean_mask(candidate, *small.shape[:2]), h, w)
        else:
            # 用户强行给了很小的 work_size: 回放后阶梯明显, 必须回到原图再清理
            mask = _clean_mask(_upsample_mask(candidate, h, w), h, w)
        return _gate(mask, conf, bg_ratio, h, w, min_coverage, max_coverage)
    return _estimate_cloth_mask_core(arr, min_coverage, max_coverage, edge_prior)


def _downscale(arr: np.ndarray, long_target: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = float(long_target) / float(max(h, w))
    nw, nh = max(64, int(round(w * scale))), max(64, int(round(h * scale)))
    if (nh, nw) == (h, w):
        return arr
    return np.asarray(
        Image.fromarray(arr[..., :3].astype(np.uint8)).resize((nw, nh), Image.BILINEAR),
        dtype=np.uint8,
    )


def _upsample_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """最近邻回放, 保持 0/1 不产生灰边."""
    if mask.shape[:2] == (h, w):
        return mask.astype(np.uint8)
    return (
        np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8
        )
        > 127
    ).astype(np.uint8)


def _weave_survives(small: np.ndarray, min_period_px: float = 2.5) -> bool:
    """缩小后的图还能不能量到经纬周期 (量不到 = 纹理已被平均掉)."""
    if small.ndim == 2:
        gray = small.astype(np.float32)
    else:
        color = small[..., :3].astype(np.float32)
        gray = 0.299 * color[..., 0] + 0.587 * color[..., 1] + 0.114 * color[..., 2]
    pxs, pys = _period_pixels(gray)
    if pxs <= 0 or pys <= 0:
        return False
    return float(min(pxs, pys)) >= min_period_px


def auto_work_size(image: np.ndarray, min_period_px: float = 3.0, floor: int = 512) -> int | None:
    """按织造周期决定掩码估计的降采样尺寸; 返回 None 表示不降采样.

    约束: 缩放后经纬周期不能小于 ``min_period_px`` (否则局部纹理能量被平均掉,
    掩码估计会退化成"全是布"). 因此细密布几乎不能降采样, 粗纹理布可以降很多.
    """
    return _plan_work_size(image, min_period_px, floor)[0]


def _plan_work_size(
    image: np.ndarray, min_period_px: float = 3.0, floor: int = 512
) -> Tuple[int | None, float]:
    """返回 (降采样长边 或 None, 原图实测经纬周期)."""
    arr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    pxs, pys = _period_pixels(gray)
    long_side = max(h, w)
    if pxs <= 0 or pys <= 0:
        return None, 0.0
    period = float(np.sqrt(pxs * pys))
    target = min_period_px * long_side / max(period, 1e-6)
    target = int(round(min(float(long_side), max(float(floor), target))))
    return (target if target < long_side * 0.95 else None), period


def _estimate_cloth_mask_core(
    image: np.ndarray,
    min_coverage: float = 0.05,
    max_coverage: float = 0.995,
    edge_prior: bool = True,
) -> Tuple[np.ndarray, float]:
    """在给定分辨率上直接估计布面掩码 (1=布面, 0=背景)."""
    h, w = image.shape[:2]
    candidate, confidence, bg_ratio = _score_mask(image, edge_prior=edge_prior)
    if candidate is None:
        return np.ones((h, w), dtype=np.uint8), 0.2
    mask = _clean_mask(candidate, h, w)
    return _gate(mask, confidence, bg_ratio, h, w, min_coverage, max_coverage)


def _score_mask(
    image: np.ndarray,
    edge_prior: bool = True,
    radii: Tuple[int, int] | None = None,
    tex_weight: float = 0.65,
    color_weight: float = 0.35,
) -> Tuple[np.ndarray | None, float, float]:
    """打分 + Otsu 阈值, 返回 (候选掩码 或 None=整幅都是布, 置信度, 背景占比).

    这一层是**分辨率相关**的 (纹理窗口按该分辨率下的经纬周期定), 几何清理不在这里做.

    判据: 布面 = 高局部纹理能量 / 有颜色 / 与画面边缘背景色不同.
    只有当低分区域 (疑似背景) 形成足够大的、与画面边框相连的连通块, 且其
    纹理能量显著低于布面时才认为 "存在背景", 否则整幅都当作布面, 避免给
    网络引入错误的背景先验.

    Returns:
        (mask, confidence): confidence 越高表示分割越可信.
    """
    if image.ndim == 2:
        gray = image.astype(np.float32)
        color = np.stack([gray] * 3, axis=-1)
    else:
        color = image[..., :3].astype(np.float32)
        gray = 0.299 * color[..., 0] + 0.587 * color[..., 1] + 0.114 * color[..., 2]
    h, w = gray.shape

    # 1) 纹理能量 (局部标准差) — 布面有经纬纹理, 背景/辊筒更平滑
    tex_r, smooth_r = radii if radii is not None else _texture_window(gray)
    tex = local_std(gray, radius=tex_r)
    tex = tex / (np.percentile(tex, 99) + 1e-6)

    # 2) 饱和度 (彩色布 vs 灰度背景)
    if color.shape[-1] == 3:
        mx = color.max(axis=-1)
        mn = color.min(axis=-1)
        sat = (mx - mn) / (mx + 1e-6)
    else:
        sat = np.zeros_like(gray)

    # 3) 与画面边缘背景色的差异
    border = max(2, min(h, w) // 64)
    border_pixels = np.concatenate(
        [
            color[:border].reshape(-1, 3),
            color[-border:].reshape(-1, 3),
            color[:, :border].reshape(-1, 3),
            color[:, -border:].reshape(-1, 3),
        ]
    )
    bg_color = np.median(border_pixels, axis=0)
    # 布面中心区域的代表色: 背景与布面颜色应显著不同 (深色辊筒/台面 vs 浅色布)
    cy0, cy1 = int(0.3 * h), int(0.7 * h) + 1
    cx0, cx1 = int(0.3 * w), int(0.7 * w) + 1
    center_color = np.median(color[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
    color_gap = float(np.linalg.norm(center_color - bg_color)) / 255.0
    color_dist = np.linalg.norm(color - bg_color[None, None, :], axis=-1)
    color_dist = color_dist / (np.percentile(color_dist, 99) + 1e-6)

    if not edge_prior:
        color_weight = 0.0
    score = tex_weight * tex + 0.15 * sat + color_weight * color_dist
    lo, hi = np.percentile(score, 5), np.percentile(score, 95)
    score = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    # 平滑: 织造纹理会造成周期性高低起伏, 必须先平滑再分割
    smooth = box_filter(score, smooth_r)

    thr = otsu_threshold(smooth)
    candidate = (smooth > thr).astype(np.uint8)
    fg_mean = float(smooth[candidate > 0].mean()) if candidate.any() else 0.0
    bg_mean = float(smooth[candidate == 0].mean()) if (candidate == 0).any() else 0.0
    confidence = _otsu_separability(smooth, thr)

    bg_ratio = 1.0 - float(candidate.mean())
    # 三种线索任意一条足够强即可认为"画面里确实有背景":
    # 1) 边框颜色与布面色差明显; 2) 低分区域的分数显著低于布面; 3) 低分区域占比很大.
    strong = color_gap > 0.06 or bg_mean < 0.40 * max(fg_mean, 1e-6) or bg_ratio > 0.45
    has_background = (
        bg_ratio > 0.10
        and bg_mean < 0.75 * max(fg_mean, 1e-6)
        and confidence > 0.35
        and strong
    )
    if not has_background:
        return None, 0.2, 1.0
    return candidate, confidence, bg_ratio


def _clean_mask(candidate: np.ndarray, h: int, w: int) -> np.ndarray:
    """在**目标分辨率**上做形态学清理与连通块过滤.

    低分辨率估计只在低分辨率上做"打分 + 阈值", 几何清理必须回到输出分辨率:
    否则 512 宽上半径为 1 的开运算, 换算到 1440 宽就是 3px 的边界偏差,
    窄布条上会直接吃掉 IoU.
    """
    mask = np.ascontiguousarray(candidate, dtype=np.uint8)
    mask = binary_open(mask, radius=max(1, min(h, w) // 200))
    mask = binary_close(mask, radius=max(2, min(h, w) // 100))
    mask = _fill_holes(mask)

    labels, num = connected_components(mask)
    if num > 0:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        keep = np.flatnonzero(areas > max(32, 0.01 * h * w))
        if keep.size:
            mask = np.isin(labels, keep).astype(np.uint8)
            mask = _fill_holes(mask)
    return mask


def _gate(
    mask: np.ndarray,
    confidence: float,
    bg_ratio: float,
    h: int,
    w: int,
    min_coverage: float = 0.05,
    max_coverage: float = 0.995,
) -> Tuple[np.ndarray, float]:
    """覆盖率不合理时宁可退回"整幅都是布", 也不引入错误背景先验."""
    coverage = float(mask.mean())
    if coverage < min_coverage or coverage > max_coverage or coverage < 1.0 - bg_ratio - 0.3:
        return np.ones((h, w), dtype=np.uint8), min(confidence, 0.2)
    return mask, confidence


def cloth_mask_from_boxes(
    boxes: np.ndarray,
    size: Tuple[int, int],
    dilate: int = 8,
) -> np.ndarray:
    """无掩码时的兜底: 以标注框外扩作为 "布面" 弱先验 (仅用于有疵点的图)."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box[1:5]
        ix1, ix2 = int(x1 * w), int(np.ceil(x2 * w))
        iy1, iy2 = int(y1 * h), int(np.ceil(y2 * h))
        mask[max(0, iy1) : min(h, iy2), max(0, ix1) : min(w, ix2)] = 1
    if dilate > 0 and mask.any():
        mask = binary_dilate(mask, dilate)
    return mask
