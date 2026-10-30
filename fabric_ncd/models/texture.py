"""经纬纹理先验模块.

布匹的共性来自 "经纬交织" 形成的准周期结构, 而不是颜色或语义. 因此这里显式
注入两类与布料种类无关的先验, 这是本方案泛化能力强于通用检测器 (YOLO) 的
关键之一:

1. **Gabor 滤波器组** (``GaborBank``): 固定 (不学习) 的多方向 / 多波长二维
   Gabor 核, 方向 0°/90° 直接对应纬纱/经纱, 45°/135° 对应斜纹; 实部捕捉
   "线" 的强弱, 虚部捕捉相位突变 (纬斜 / 稀缝 / 错位).
2. **织造周期估计** (``estimate_weave_period``): 对灰度图做加窗 FFT, 在
   允许的周期区间内找峰值, 得到经向 / 纬向周期 (像素) 与各向异性. 它既用于
   数据侧的 "织造尺度归一化", 也作为全局 token 注入网络, 使模型对 "这种布
   的纹理尺度" 显式建模, 换布料种类时不需要重新学习尺度.

所有模块都很轻: 8 个 15x15 深度可分离卷积核 + 一次 256x256 FFT.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Gabor 核
# --------------------------------------------------------------------------- #
def gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
    imaginary: bool = False,
) -> torch.Tensor:
    half = kernel_size // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    x_theta = xs * math.cos(theta) + ys * math.sin(theta)
    y_theta = -xs * math.sin(theta) + ys * math.cos(theta)
    envelope = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
    phase = 2 * math.pi * x_theta / lambd + psi
    kernel = envelope * (torch.sin(phase) if imaginary else torch.cos(phase))
    kernel = kernel - kernel.mean()  # 去直流, 对亮度不敏感
    return kernel / (kernel.norm() + 1e-8)


class GaborBank(nn.Module):
    """固定 Gabor 卷积 (深度可分离, 单通道输入, 输出 K 个响应图)."""

    def __init__(
        self,
        orientations: List[float] | Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
        wavelengths: List[float] | Tuple[float, ...] = (4.0, 8.0),
        kernel_size: int = 15,
        gamma: float = 0.5,
        use_imaginary: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Gabor kernel_size 必须是不小于 3 的奇数")
        if not orientations or not wavelengths or any(float(wl) <= 0 for wl in wavelengths):
            raise ValueError("Gabor 方向和正波长不能为空")
        kernels = []
        for wl in wavelengths:
            sigma = 0.56 * float(wl)
            for deg in orientations:
                theta = math.radians(float(deg))
                kernels.append(
                    gabor_kernel(kernel_size, sigma, theta, float(wl), gamma, psi=0.0, imaginary=False)
                )
                if use_imaginary:
                    kernels.append(
                        gabor_kernel(kernel_size, sigma, theta, float(wl), gamma, psi=0.0, imaginary=True)
                    )
        weight = torch.stack(kernels).unsqueeze(1)  # (K,1,k,k)
        self.register_buffer("weight", weight, persistent=False)
        self.kernel_size = kernel_size
        self.out_channels = len(kernels)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(F.pad(gray, (pad,) * 4, mode="reflect"), self.weight)


# --------------------------------------------------------------------------- #
# 织造周期 / 频谱统计
# --------------------------------------------------------------------------- #
def _hann2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, device=device, dtype=dtype)
    return w[:, None] * w[None, :]


def _parabolic_peak(spec: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """对数谱上的抛物线插值, 返回亚像素峰位 (B,)."""
    idx = index.clamp(1, spec.shape[-1] - 2)
    left = torch.log(spec.gather(-1, (idx - 1).unsqueeze(-1)).squeeze(-1) + 1e-8)
    center = torch.log(spec.gather(-1, idx.unsqueeze(-1)).squeeze(-1) + 1e-8)
    right = torch.log(spec.gather(-1, (idx + 1).unsqueeze(-1)).squeeze(-1) + 1e-8)
    denom = left - 2 * center + right
    delta = torch.where(denom.abs() > 1e-6, 0.5 * (left - right) / (denom + 1e-8), torch.zeros_like(denom))
    return idx.to(spec.dtype) + delta.clamp(-0.5, 0.5)


@torch.no_grad()
def estimate_weave_period(
    images: torch.Tensor,
    work_size: int = 512,
    min_period: float = 2.0,
    max_period: float = 64.0,
    default_period: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """估计经向 / 纬向织造周期 (输入像素单位).

    高分辨率输入 (如 1600x832) 下如果直接把整图压成 256x256 的正方形再做 FFT,
    非正方形方向的频率会被严重抽取: 1600x832 -> 256x256 后, 水平方向相当于
    14 倍抽取, 周期 < 14 px 的经纬纹理在谱上完全消失, 周期估计退化成噪声.

    因此这里改为 **保持长宽比、只做整数倍池化**, 两个方向的频率分别用各自的
    真实尺寸归一化 ``period = dim / k`` (k 为整幅内的周期数), 这样:
    * 非正方形输入不再引入系统性的比例误差;
    * 分析尺寸控制在 ``work_size`` 附近, 长边不会因为保持比例而爆炸.
    """
    if images.dim() != 4:
        raise ValueError("images 需为 (B,3,H,W)")
    if images.dtype in (torch.float16, torch.bfloat16):  # FFT 在 fp32 上做，避免半精度兼容问题
        images = images.float()
    b, _, h, w = images.shape
    gray = (
        0.299 * images[:, :1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )
    factor = max(1, int(min(h, w) // max(8, int(work_size))))
    ph, pw = max(8, h // factor), max(8, w // factor)
    small = F.adaptive_avg_pool2d(gray, (ph, pw))
    small = small - small.mean(dim=(-2, -1), keepdim=True)

    wy = torch.hann_window(ph, periodic=False, device=small.device, dtype=small.dtype)
    wx = torch.hann_window(pw, periodic=False, device=small.device, dtype=small.dtype)
    spec = torch.fft.rfft2(small * wy[None, None, :, None] * wx[None, None, None, :]).abs().pow(2)

    energy_y = spec.mean(dim=3).squeeze(1)[:, : max(2, ph // 2)]  # (B, ph/2)
    energy_x = spec.mean(dim=2).squeeze(1)                        # (B, pw/2+1)
    energy_y = energy_y.clone()
    energy_x = energy_x.clone()
    energy_y[:, 0] = 0.0
    energy_x[:, 0] = 0.0

    def _pick(energy: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """k: 整幅内的周期数 (浮点) -> 周期 = dim / k."""
        nbins = energy.shape[1]
        k_lo = max(1, int(math.floor(dim / max_period)))
        k_hi = min(nbins - 2, int(math.ceil(dim / min_period)))
        if k_lo > k_hi:
            return (energy.new_full((energy.shape[0],), dim / default_period),
                    energy.new_zeros((energy.shape[0],)))
        band = energy[:, k_lo : k_hi + 1]
        idx = band.argmax(dim=1) + k_lo
        k = _parabolic_peak(energy, idx.clamp(1, nbins - 2)).clamp(min=1e-3)
        conf = band.max(dim=1).values / (band.mean(dim=1) + 1e-8)
        return k, conf

    k_y, conf_y = _pick(energy_y, h)
    k_x, conf_x = _pick(energy_x, w)

    period_y = (h / k_y).clamp(min_period, max_period)
    period_x = (w / k_x).clamp(min_period, max_period)

    conf = torch.minimum(conf_x, conf_y)
    # 置信度太低 (无纹理/纯色) 时回退到默认周期
    low = conf < 2.0
    period_x = torch.where(low, torch.full_like(period_x, default_period), period_x)
    period_y = torch.where(low, torch.full_like(period_y, default_period), period_y)

    e_x = energy_x.sum(dim=1)
    e_y = energy_y.sum(dim=1)
    anisotropy = (e_x - e_y) / (e_x + e_y + 1e-8)
    return {
        "period_x": period_x,
        "period_y": period_y,
        "anisotropy": anisotropy,
        "confidence": conf,
    }


def descriptor_from_stats(stats: Dict[str, torch.Tensor], ref_period: float = 8.0) -> torch.Tensor:
    """把 ``estimate_weave_period`` 的结果编码成 (B,6) 的尺度不变描述子."""
    px = torch.log2(stats["period_x"] / ref_period)
    py = torch.log2(stats["period_y"] / ref_period)
    ratio = torch.log2(stats["period_x"] / (stats["period_y"] + 1e-6))
    conf = torch.log1p(stats["confidence"])
    return torch.stack([px, py, ratio, stats["anisotropy"], conf, torch.zeros_like(conf)], dim=1)


def weave_descriptor(images: torch.Tensor, ref_period: float = 8.0) -> torch.Tensor:
    """将周期统计编码成 (B, 6) 的尺度不变描述子, 供全局分支使用."""
    return descriptor_from_stats(estimate_weave_period(images), ref_period)


# --------------------------------------------------------------------------- #
# 纹理归一化 / 编码
# --------------------------------------------------------------------------- #
class WeaveNorm(nn.Module):
    """高频归一化: 去掉光照、色差与低频背景, 只保留织造纹理.

    ``x_norm = (x - lowpass(x)) / (std(x - lowpass(x)) + eps)``
    """

    def __init__(self, kernel_size: int = 9, eps: float = 1e-4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        low = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), self.kernel_size, stride=1)
        high = x - low
        std = high.flatten(2).std(dim=2).clamp_min(self.eps)[:, :, None, None]
        return (high / std).clamp(-6.0, 6.0)


class TextureEncoder(nn.Module):
    """Gabor 响应 + 1/4 特征融合, 输出纹理增强的局部特征."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        orientations: List[float] | Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
        wavelengths: List[float] | Tuple[float, ...] = (4.0, 8.0),
        kernel_size: int = 15,
        use_imaginary: bool = True,
        use_weave_norm: bool = True,
    ) -> None:
        super().__init__()
        self.gabor = GaborBank(orientations, wavelengths, kernel_size, use_imaginary=use_imaginary)
        self.weave_norm = WeaveNorm() if use_weave_norm else None
        gabor_norm = nn.GroupNorm(math.gcd(8, self.gabor.out_channels), self.gabor.out_channels)
        self.gabor_norm = gabor_norm
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + self.gabor.out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, images: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * images[:, :1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        if self.weave_norm is not None:
            gray = self.weave_norm(gray)
        g = self.gabor_norm(self.gabor(gray))
        if g.shape[-2:] != feat.shape[-2:]:
            g = F.interpolate(g, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([feat, g], dim=1))
