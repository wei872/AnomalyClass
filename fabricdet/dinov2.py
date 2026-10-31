# -*- coding: utf-8 -*-
"""DINOv2 ViT-S/14 骨干网络 (与官方 checkpoint 完全兼容的自包含实现).

为什么自包含: 工业部署机器通常无法安装 facebookresearch/dinov2 仓库 (依赖 timm/xformers 等),
这里仅用纯 PyTorch 复现官方结构,  state_dict 的 key 与官方 `dinov2_vits14_pretrain.pth`
一一对应, 因此可以直接 load, 不需要任何第三方仓库。

官方最轻量模型: dinov2_vits14  (21.9M params, embed=384, depth=12, heads=6, patch=14)
"""
from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 官方权重 (ViT-S/14, LVD-142M 自监督预训练)
OFFICIAL_URLS = {
    "vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
}
CACHE_DIR = Path(os.environ.get("FABRICDET_CACHE", Path.home() / ".cache" / "fabricdet"))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # SDPA: 自动融合 kernel, 导出 ONNX 时退化为标准 matmul, 两条路都正确
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """与官方 key 对齐: patch_embed.proj.{weight,bias}"""

    def __init__(self, in_chans=3, embed_dim=384, patch_size=14):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class DINOv2ViT(nn.Module):
    """ViT-S/14。输入 H,W 必须为 14 的倍数; 位置编码双三次插值, 支持任意分辨率。"""

    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 out_indices: tuple = (11,)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        n = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=1e-6)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

    def interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        N = self.pos_embed.shape[1] - 1
        h0, w0 = h // self.patch_size, w // self.patch_size
        if h0 * w0 == N and h0 == w0:
            return self.pos_embed
        cls_tok = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]
        M = int(math.sqrt(N))
        patch_pos = F.interpolate(
            patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0), mode="bicubic", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, h0 * w0, dim)
        return torch.cat([cls_tok, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"输入尺寸 ({H},{W}) 必须是 patch_size={self.patch_size} 的倍数")
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, hw, D)
        tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        tok = tok + self.interpolate_pos_encoding(H, W)
        outs = []
        for i, blk in enumerate(self.blocks):
            tok = blk(tok)
            if i in self.out_indices:
                outs.append(tok)
        outs = [self.norm(o) for o in outs]
        return [o[:, 1:] for o in outs], (H // self.patch_size, W // self.patch_size)

    def forward(self, x):
        (tok,), grid = self.forward_features(x)
        B, _, D = tok.shape
        return tok.transpose(1, 2).reshape(B, D, grid[0], grid[1])   # (B, D, h, w)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def download_pretrained(variant: str = "vits14", verbose: bool = True) -> Path | None:
    """下载官方预训练权重到本地缓存; 离线环境返回 None (调用方退化为从头训练)。"""
    url = OFFICIAL_URLS.get(variant)
    if url is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / f"dinov2_{variant}_pretrain.pth"
    if dst.exists() and dst.stat().st_size > 10_000_000:
        return dst
    try:
        if verbose:
            print(f"[dinov2] 下载官方预训练权重: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "fabricdet/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
            f.write(r.read())
        return dst
    except Exception as e:  # 离线 / 防火墙: 不致命
        if verbose:
            print(f"[dinov2] 警告: 预训练权重下载失败 ({e})。\n"
                  f"         将使用随机初始化骨干 (效果下降)。可手动下载后设置环境变量\n"
                  f"         FABRICDET_DINOV2_CKPT=/path/dinov2_vits14_pretrain.pth 再训练。")
        dst.unlink(missing_ok=True)
        return None


def load_pretrained(backbone: nn.Module, ckpt: str | os.PathLike | None = "auto",
                    variant: str = "vits14", verbose: bool = True) -> str:
    """加载官方 DINOv2 权重。返回来源描述: 'official' / 'local' / 'scratch'。

    ckpt: 'auto' = 缓存->下载; 或本地 .pth 路径; None/'scratch' = 随机初始化。
    """
    path: Path | None = None
    env = os.environ.get("FABRICDET_DINOV2_CKPT")
    if ckpt in (None, "", "scratch"):
        path = Path(env) if env else None
        if path is None:
            return "scratch"
    elif ckpt == "auto":
        path = Path(env) if env else None
        if path is None or not path.exists():
            path = CACHE_DIR / f"dinov2_{variant}_pretrain.pth"
            if not (path.exists() and path.stat().st_size > 10_000_000):
                path = download_pretrained(variant, verbose)
    else:
        path = Path(ckpt)
        if not path.exists() and env:
            path = Path(env)
    if path is None or not path.exists():
        if verbose:
            print("[dinov2] 警告: 未找到预训练权重, 骨干随机初始化。")
        return "scratch"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
    if all(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items()}
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    # 允许缺少的 key: 无 (结构一致); unexpected 通常是 head/mask_token
    real_missing = [k for k in missing if not k.startswith(("head", "dino_head", "ibot"))]
    if verbose:
        print(f"[dinov2] 加载预训练权重: {path}  (忽略无关 key {len(unexpected)} 个, "
              f"缺失 {len(real_missing)} 个)")
    if real_missing:
        print(f"[dinov2] 警告: 以下骨干 key 未加载: {real_missing[:8]}")
    return "official" if str(path).endswith("pretrain.pth") else "local"
