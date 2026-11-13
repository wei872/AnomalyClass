"""Dinomaly (Dinomaly2 的 2D UAD 模型), 从 Dinomaly2 models/uad.py 裁剪.

结构: 冻结 encoder (DINOv2) + 可训练 bottleneck (噪声瓶颈) + 可训练 decoder,
训练目标为 encoder 特征与 decoder 重建特征的 (loose) 余弦损失.
推理时 anomaly map = 1 - cos(en, de).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class Dinomaly(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=(2, 3, 4, 5, 6, 7, 8, 9),
            fuse_layer_encoder=((0, 1, 2, 3), (4, 5, 6, 7)),
            fuse_layer_decoder=((0, 1, 2, 3), (4, 5, 6, 7)),
            fuse_layer_bottleneck=(0, 1, 2, 3, 4, 5, 6, 7),
            mask_neighbor_size=0,
            remove_class_token=False,
            context_aware_recenter=True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = list(target_layers)
        self.fuse_layer_encoder = [list(g) for g in fuse_layer_encoder]
        self.fuse_layer_decoder = [list(g) for g in fuse_layer_decoder]
        self.fuse_layer_bottleneck = list(fuse_layer_bottleneck)
        self.remove_class_token = remove_class_token
        self.context_aware_recenter = context_aware_recenter
        self.mask_neighbor_size = mask_neighbor_size
        self.keep_raw = False       # True: forward 后把 encoder 各目标层的原始 token 留在 self._raw_en (融合检测器复用, 不重复跑 encoder)
        self._raw_en = None
        self.enc_amp = False        # True: 冻结的 encoder 用 fp16 autocast 前向 (无梯度, 不需要 GradScaler); 特征转回 fp32 后再进 decoder/损失
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = getattr(self.encoder, 'n_storage_tokens', 0)

    def init_weights(self):
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        en_list = []
        amp = bool(self.enc_amp) and x.is_cuda
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            x = self.encoder.prepare_tokens(x)
            for i, blk in enumerate(self.encoder.blocks):
                if i > self.target_layers[-1]:
                    break
                x = blk(x)
                if i in self.target_layers:
                    en_list.append(x)
        if amp:
            en_list = [e.float() for e in en_list]

        if self.keep_raw:
            self._raw_en = en_list
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list_bn = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = self.fuse_feature([en_list_bn[idx] for idx in self.fuse_layer_bottleneck]).detach()
        else:
            x = self.fuse_feature([en_list[idx] for idx in self.fuse_layer_bottleneck]).detach()

        for blk in self.bottleneck:
            x = blk(x)

        attn_mask = self.generate_mask(side, x.device) if self.mask_neighbor_size > 0 else None

        de_list = []
        for blk in self.decoder:
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        if self.context_aware_recenter:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] - e[:, :1, :] for e in en]
            en = [F.layer_norm(e, normalized_shape=(e.shape[-1],), eps=1e-8) for e in en]
        else:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cpu'):
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                h2s = max(idx_h1 - hm // 2, 0)
                h2e = min(idx_h1 + hm // 2 + 1, h)
                w2s = max(idx_w1 - wm // 2, 0)
                w2e = min(idx_w1 + wm // 2 + 1, w)
                mask[idx_h1, idx_w1, h2s:h2e, w2s:w2e] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        n_reg = self.encoder.num_register_tokens
        mask_all = torch.ones(h * w + 1 + n_reg, h * w + 1 + n_reg, device=device)
        mask_all[1 + n_reg:, 1 + n_reg:] = mask
        return mask_all
