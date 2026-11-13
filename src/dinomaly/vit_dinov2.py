"""自包含的 DINOv2 ViT 实现 (checkpoint key 与官方 dinov2 完全兼容, 不依赖 timm/xformers).

支持架构: dinov2reg_vit_small_14 / dinov2reg_vit_base_14 / dinov2reg_vit_large_14 /
dinov2_vit_small_14 / dinov2_vit_base_14, 以及本项目测试用的 tiny_test_14 (随机初始化).

官方权重下载地址 (用户机器需可访问 dl.fbaipublicfiles.com):
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth 等
"""
import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def make_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten_embedding=True):
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten_embedding:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    """标准多头注意力 (key: attn.qkv / attn.proj, 与 dinov2 ckpt 一致)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """dinov2 block (key: norm1 / attn / ls1 / mlp / ls2 / norm2)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, init_values=1e-5,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = nn.Identity()  # 简化: 不做 stochastic depth
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4., qkv_bias=True, init_values=1e-5, num_register_tokens=0,
                 block_fn=Block, norm_layer=partial(nn.LayerNorm, eps=1e-6), interpolate_antialias=False,
                 interpolate_offset=0.1):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.n_storage_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     init_values=init_values, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'register_tokens'):
            trunc_normal_(self.register_tokens, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, w, h):
        # 与官方 dinov2 完全一致 (此时 x 尚未插入 register tokens)
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy), mode='bicubic', antialias=self.interpolate_antialias)
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if hasattr(self, 'register_tokens'):
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False, norm=True):
        x = self.prepare_tokens(x)
        out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in n:
                out.append(x)
        if norm:
            out = [self.norm(o) for o in out]
        if reshape:
            res = []
            for o in out:
                cls_tok = o[:, :1]
                patch = o[:, 1 + self.num_register_tokens:]
                if return_class_token:
                    res.append(torch.cat((patch, cls_tok), dim=1))
                else:
                    res.append(patch)
            return res
        return out


def vit_small(patch_size=14, num_register_tokens=0, img_size=518, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
                                 num_register_tokens=num_register_tokens, img_size=img_size, **kwargs)


def vit_base(patch_size=14, num_register_tokens=0, img_size=518, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
                                 num_register_tokens=num_register_tokens, img_size=img_size, **kwargs)


def vit_large(patch_size=14, num_register_tokens=0, img_size=518, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
                                 num_register_tokens=num_register_tokens, img_size=img_size, **kwargs)


def vit_tiny(patch_size=14, num_register_tokens=0, img_size=224, **kwargs):
    """测试用超小 ViT (随机初始化, 仅用于冒烟测试代码通路)."""
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=64, depth=2, num_heads=2,
                                 num_register_tokens=num_register_tokens, img_size=img_size, **kwargs)


_ARCHS = {
    'small': vit_small,
    'base': vit_base,
    'large': vit_large,
    'tiny': vit_tiny,
}


def build_vit(arch: str, patch_size: int = 14, num_register_tokens: int = 0, img_size: int = 518, **kwargs):
    if arch not in _ARCHS:
        raise ValueError(f'未知架构: {arch}, 可选: {list(_ARCHS)}')
    return _ARCHS[arch](patch_size=patch_size, num_register_tokens=num_register_tokens, img_size=img_size,
                        **kwargs)
