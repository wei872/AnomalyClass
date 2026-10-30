"""全局 / 局部特征融合, 布面地理 (布边 vs 布中) 感知.

设计动机:

* 疵点在局部 (1/8) 尺度上才可分 (断经、棉团只有几像素宽), 但 "是不是疵点"
  往往需要全局上下文 — 例如整幅布的纬斜、整幅偏色、织造周期是否一致.
  因此用 1/32 上的线性注意力 (O(N), 512 输入只有 256 个 token) 建模全局,
  再自顶向下与 1/8 局部特征融合.
* 布边与布中纹理不一致, 且布幅变化会让布边位置移动. 我们不把 "位置" 硬编码
  进网络, 而是从 ``fabric mask`` 概率图推导 "到布边的软距离", 用 4 级池化
  金字塔近似距离场 (可微、几乎零开销), 再以 embedding 的形式加到局部特征上.
  布幅变 -> 预测的布边变 -> embedding 自动跟随, 模型无需见过固定宽度.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        layers: List[nn.Module] = [
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
        ]
        if act:
            layers.append(nn.Hardswish(inplace=True))
        super().__init__(*layers)


class LinearAttention2d(nn.Module):
    """线性注意力 (ELU+1 核), 复杂度 O(N) 而非 O(N^2)."""

    def __init__(self, dim: int, heads: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dim_head = dim // heads
        self.eps = eps
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        n = h * w
        shape = (b, self.heads, self.dim_head, n)
        q = F.elu(q.reshape(shape)) + 1.0
        k = F.elu(k.reshape(shape)) + 1.0
        v = v.reshape(shape)
        kv = torch.einsum("bhdn,bhen->bhde", k, v) / n
        k_sum = k.mean(dim=-1)
        out = torch.einsum("bhdn,bhde->bhen", q, kv)
        denom = torch.einsum("bhdn,bhd->bhn", q, k_sum).unsqueeze(-2) + self.eps
        out = (out / denom).reshape(b, c, h, w)
        return x + self.proj(out)


class GlobalContext(nn.Module):
    """1/32 特征上的全局建模 (空洞卷积 + 线性注意力 + 织造尺度 token)."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dilations: Sequence[int] = (2, 4),
        heads: int = 4,
        use_attention: bool = True,
        weave_dim: int = 6,
    ) -> None:
        super().__init__()
        self.stem = ConvBNAct(c_in, c_out, 1)
        # 每个膨胀率一个 depthwise 空洞卷积
        self.dilated = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_out, c_out, 3, padding=d, dilation=d, groups=c_out, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.Hardswish(inplace=True),
                )
                for d in dilations
            ]
        )
        self.attention = LinearAttention2d(c_out, heads=heads) if use_attention else nn.Identity()
        self.weave_proj = nn.Sequential(nn.Linear(weave_dim, c_out), nn.Hardswish(inplace=True), nn.Linear(c_out, c_out))
        self.out = ConvBNAct(c_out, c_out, 1)

    def forward(self, p5: torch.Tensor, weave_desc: torch.Tensor | None = None) -> torch.Tensor:
        g = self.stem(p5)
        for layer in self.dilated:
            g = g + layer(g)
        g = self.attention(g)
        if weave_desc is not None:
            token = self.weave_proj(weave_desc)  # (B, C)
            g = g + token[:, :, None, None]
        return self.out(g)


class FiLM(nn.Module):
    """用全局描述子对局部特征做通道级仿射调制 (scale/shift)."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(cond).chunk(2, dim=1)
        return feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class GlocalFusion(nn.Module):
    """自顶向下融合 p4/p5 全局上下文与 p3 局部细节, 输出 1/8 特征."""

    def __init__(
        self,
        c_p3: int,
        c_p4: int,
        c_glob: int,
        c_out: int,
        c_local: int | None = None,
    ) -> None:
        super().__init__()
        c_local = c_local or c_p3
        self.lat_p3 = ConvBNAct(c_local, c_out, 1)
        self.lat_p4 = nn.Conv2d(c_p4, c_out, 1, bias=False)
        self.lat_glob = nn.Conv2d(c_glob, c_out, 1, bias=False)
        # 全程 1x1 / depthwise, 控制 1/8 分辨率上的计算量
        self.fuse = nn.Sequential(
            ConvBNAct(c_out, c_out, 1),
            ConvBNAct(c_out, c_out, 3, groups=c_out),
            ConvBNAct(c_out, c_out, 1),
        )
        self.film = FiLM(c_glob, c_out)
        self.out_channels = c_out

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        glob: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        size = p3.shape[-2:]
        x = self.lat_p3(p3)
        x = x + F.interpolate(self.lat_p4(p4), size=size, mode="bilinear", align_corners=False)
        x = x + F.interpolate(self.lat_glob(glob), size=size, mode="bilinear", align_corners=False)
        x = self.fuse(x)
        return self.film(x, cond)


class GeographyEmbedding(nn.Module):
    """布面地理编码: 由布面概率图得到 "到布边距离" 的软表示.

    * 多尺度平均池化金字塔近似距离场 (由内向外扩散), 可微且几乎零成本;
    * 边界带响应 ``4p(1-p)`` 显式标记布边过渡区, 用于加强布边注意力;
    * 归一化坐标只作为辅助弱信号 (可选), 使模型在布幅变化时依赖距离而非绝对位置.
    """

    def __init__(
        self,
        channels: int,
        pool_sizes: Sequence[int] = (3, 7, 15, 31, 63),
        use_xy: bool = False,
    ) -> None:
        super().__init__()
        self.pool_sizes = list(pool_sizes)
        self.use_xy = use_xy
        in_dim = len(self.pool_sizes) + 2  # 金字塔 + 边界带 + 原概率
        if use_xy:
            in_dim += 2
        self.embed = nn.Sequential(
            nn.Conv2d(in_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, mask_prob: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """mask_prob: (B,1,h,w) 与 feat 空间尺寸一致或在 1/4 尺度."""
        if mask_prob.shape[-2:] != feat.shape[-2:]:
            mask_prob = F.interpolate(mask_prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        responses = [mask_prob]
        for k in self.pool_sizes:
            pad = k // 2
            pooled = F.avg_pool2d(F.pad(mask_prob, (pad,) * 4, mode="replicate"), k, stride=1)
            responses.append(pooled)
        band = 4.0 * mask_prob * (1.0 - mask_prob)  # 布边带
        responses.append(band)
        if self.use_xy:
            b, _, h, w = mask_prob.shape
            ys = torch.linspace(-1, 1, h, device=mask_prob.device, dtype=mask_prob.dtype)
            xs = torch.linspace(-1, 1, w, device=mask_prob.device, dtype=mask_prob.dtype)
            yy = ys[None, None, :, None].expand(b, 1, h, w)
            xx = xs[None, None, None, :].expand(b, 1, h, w)
            responses.extend([yy, xx])
        geo = torch.cat(responses, dim=1)
        return feat * (1.0 + self.embed(geo))
