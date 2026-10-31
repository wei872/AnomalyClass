"""DINO-initialized MGViT with SimGCD-style projection/classification heads."""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .mgvit import MaskedGuidedVisionTransformer


class MultiHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.last_layer = nn.ModuleList([weight_norm(nn.Linear(in_dim, out_dim, bias=False)) for _ in range(n_heads)])
        self.apply(self._init_weights)
        for head in self.last_layer:
            with torch.no_grad():
                head.parametrizations.weight.original0.fill_(1)
            head.parametrizations.weight.original0.requires_grad_(False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # With functional weight_norm, module.weight is a computed tensor.
            # Initialize the underlying direction parameter, not that temporary.
            weight = module.parametrizations.weight.original1 if hasattr(module, "parametrizations") else module.weight
            nn.init.trunc_normal_(weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor):
        projection = self.mlp(features)
        normalized = nn.functional.normalize(features, dim=-1)
        return projection, [head(normalized) for head in self.last_layer]


class AnomalyNCDModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, n_classes: int, cfg: dict):
        super().__init__()
        self.backbone = backbone
        self.projector = MultiHead(feature_dim, n_classes, cfg["hidden_dim"], cfg["bottleneck_dim"], cfg["n_heads"])

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        return self.projector(self.backbone(images, masks))


def backbone_spec(name: str) -> dict:
    specs = {
        "dino_vitb8": dict(patch_size=8, embed_dim=768, num_heads=12, depth=12,
                           weights="dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"),
        "dino_vitb16": dict(patch_size=16, embed_dim=768, num_heads=12, depth=12,
                            weights="dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"),
        "dino_vits8": dict(patch_size=8, embed_dim=384, num_heads=6, depth=12,
                           weights="dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"),
        "dino_vits16": dict(patch_size=16, embed_dim=384, num_heads=6, depth=12,
                            weights="dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"),
        # Explicit opt-in CPU smoke-test architecture, NOT a paper model.
        "tiny_vit": dict(patch_size=8, embed_dim=64, num_heads=4, depth=4, weights=None),
    }
    if name not in specs:
        raise ValueError(f"Unsupported backbone: {name}")
    return specs[name]


def validate_model_config(cfg: dict) -> dict:
    settings = cfg["model"]
    spec = backbone_spec(settings["backbone"])
    if settings["image_size"] % spec["patch_size"]:
        raise ValueError(f"image_size 必须能被 patch_size={spec['patch_size']} 整除")
    if spec["embed_dim"] % spec["num_heads"]:
        raise ValueError("embed_dim 必须能被 num_heads 整除")
    if not 1 <= settings["mask_layers"] <= spec["depth"]:
        raise ValueError(f"{settings['backbone']}: mask_layers 必须在 1..{spec['depth']}")
    if not 0 <= settings["grad_from_block"] <= spec["depth"]:
        raise ValueError(f"{settings['backbone']}: grad_from_block 必须在 0..{spec['depth']}")
    return spec


def resize_checkpoint_pos_embed(state: dict, target: torch.Tensor) -> dict:
    """Resize DINO's square patch-position grid while preserving its CLS position."""
    pos = state.get("pos_embed")
    if pos is None or pos.shape == target.shape:
        return state
    if pos.ndim != 3 or target.ndim != 3 or pos.shape[0] != 1 or target.shape[0] != 1 or pos.shape[2] != target.shape[2]:
        raise ValueError("DINO pos_embed 维度不兼容")
    old_tokens, new_tokens = pos.shape[1] - 1, target.shape[1] - 1
    old_side, new_side = math.isqrt(old_tokens), math.isqrt(new_tokens)
    if old_side * old_side != old_tokens or new_side * new_side != new_tokens:
        raise ValueError("DINO pos_embed 必须是方形 patch 网格")
    patches = pos[:, 1:].reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2).float()
    patches = nn.functional.interpolate(patches, size=(new_side, new_side), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, new_tokens, -1)
    state = dict(state)
    state["pos_embed"] = torch.cat((pos[:, :1], patches.to(pos.dtype)), dim=1)
    return state


def build_backbone(cfg: dict, load_pretrained: bool = True, allow_random_init: bool = False,
                   image_size: int | None = None) -> MaskedGuidedVisionTransformer:
    """Frozen DINO backbone alone, optionally at a different input resolution.

    Anomaly detection needs patch tokens from the *pretrained* backbone, not from
    the class-finetuned one, so it builds its own copy through this helper.
    """
    settings = dict(cfg["model"])
    if image_size is not None:
        settings["image_size"] = image_size
    cfg = {**cfg, "model": settings}
    name = settings["backbone"]
    spec = validate_model_config(cfg)
    backbone = MaskedGuidedVisionTransformer(
        img_size=[settings["image_size"]], patch_size=spec["patch_size"], embed_dim=spec["embed_dim"],
        depth=spec["depth"], active_depth=settings["mask_layers"], num_heads=spec["num_heads"],
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if load_pretrained:
        if name == "tiny_vit" and not allow_random_init:
            raise ValueError("tiny_vit 只用于显式 --allow-random-init 的测试，不是论文预训练模型")
        path = settings.get("pretrained_path")
        if path:
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"DINO 权重不存在: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
        elif spec["weights"] and not allow_random_init:
            url = "https://dl.fbaipublicfiles.com/dino/" + spec["weights"]
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        else:
            state = None
        if state is not None:
            for key in ("state_dict", "model", "teacher"):
                if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
            state = resize_checkpoint_pos_embed(state, backbone.pos_embed.data)
            # No silent partial loading or silently random DINO fallback.
            backbone.load_state_dict(state, strict=True)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone


def load_model(cfg: dict, n_classes: int, load_pretrained: bool = True,
               allow_random_init: bool = False) -> AnomalyNCDModel:
    settings = cfg["model"]
    spec = validate_model_config(cfg)
    backbone = build_backbone(cfg, load_pretrained, allow_random_init)
    for parameter_name, parameter in backbone.named_parameters():
        if parameter_name.startswith("blocks."):
            block = int(parameter_name.split(".")[1])
            if block >= settings["grad_from_block"]:
                parameter.requires_grad_(True)
    return AnomalyNCDModel(backbone, spec["embed_dim"], n_classes, settings)


def get_params_groups(model: nn.Module) -> list[dict]:
    regularized, not_regularized = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (not_regularized if name.endswith("bias") or parameter.ndim == 1 else regularized).append(parameter)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]
