"""按 Dinomaly2 dinomaly_2D.py 的默认配置组装 Dinomaly 模型."""
from functools import partial

import torch.nn as nn

from . import encoder as encoder_mod
from .blocks import Attention, Block, LinearAttention2
from .uad import Dinomaly

_FUSE_LAYERS = {
    0: ([[0], [1], [2], [3], [4], [5], [6], [7]], [[0], [1], [2], [3], [4], [5], [6], [7]]),
    1: ([[0, 1, 2, 3, 4, 5, 6, 7]], [[0, 1, 2, 3, 4, 5, 6, 7]]),
    2: ([[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]),
    3: ([[0, 1, 2], [3, 4, 5], [6, 7]], [[0, 1, 2], [3, 4, 5], [6, 7]]),
    4: ([[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]]),
    11: ([[7]], [[7]]),
    12: ([[3], [7]], [[3], [7]]),
    14: ([[1], [3], [5], [7]], [[1], [3], [5], [7]]),
}

_TARGET_LAYERS = {
    'small': [2, 3, 4, 5, 6, 7, 8, 9],
    'base': [2, 3, 4, 5, 6, 7, 8, 9],
    'large': [4, 6, 8, 10, 12, 14, 16, 18],
    'tiny': [0, 1],
}


def build_dinomaly(backbone: str = 'dinov2reg_vit_small_14', dropout: float = 0.4, la: bool = True,
                   lc: int = 2, cr: bool = True, weights_dir: str = None, allow_random_init: bool = False,
                   logger=None) -> Dinomaly:
    """与 Dinomaly2 dinomaly_2D.py 默认超参一致的模型构建."""
    encoder = encoder_mod.load_encoder(backbone, weights_dir=weights_dir,
                                       allow_random_init=allow_random_init, logger=logger)
    _, _, arch, _ = encoder_mod.parse_backbone_name(backbone)
    embed_dim = encoder_mod.encoder_embed_dim(backbone)
    num_heads = encoder_mod.encoder_num_heads(backbone)
    target_layers = _TARGET_LAYERS[arch]
    fuse_enc, fuse_dec = _FUSE_LAYERS[lc]
    fuse_bn = list(range(len(target_layers)))
    if arch == 'tiny':  # 测试用小模型只有 2 层, 分组需匹配
        fuse_enc = fuse_dec = [[i] for i in range(len(target_layers))]

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                      nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)),
    ])

    attn_cls = LinearAttention2 if la else Attention
    decoder = nn.ModuleList([
        Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
              norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=partial(attn_cls, eps=1e-8) if la else attn_cls)
        for _ in range(8)
    ])

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                     remove_class_token=False, fuse_layer_encoder=fuse_enc, fuse_layer_decoder=fuse_dec,
                     fuse_layer_bottleneck=fuse_bn, context_aware_recenter=cr)
    model.init_weights()
    return model


def detector_hparams(backbone: str, image_size: int, crop_size: int, dropout: float, la: int, lc: int, cr: int):
    return {
        'backbone': backbone,
        'image_size': int(image_size),
        'crop_size': int(crop_size),
        'dropout': float(dropout),
        'la': int(la),
        'lc': int(lc),
        'cr': int(cr),
    }
