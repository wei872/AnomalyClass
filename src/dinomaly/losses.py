"""Dinomaly2 训练损失 (从 Dinomaly2 utils.py 裁剪): global cosine + loose (hard-mining percentile) 版本."""
from functools import partial

import torch
import torch.nn.functional as F


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    return loss / len(a)


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    """Loose loss: 丢弃距离最大的 (1-p) 比例区域 (默认 p=0.9 丢弃 10% 最远点), 其梯度乘 factor."""
    cos_loss = F.cosine_similarity
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - F.cosine_similarity(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    return loss / len(a)


def _as_tokens(t):
    """(B,N,C) 或 (B,C,H,W) -> (B,N,C)."""
    if t.dim() == 4:
        b, c, h, w = t.shape
        return t.flatten(2).transpose(1, 2)
    return t


def local_cosine(a, b, hard_ratio: float = 0.0, stop_grad: bool = True):
    """逐位置(局部)cosine 损失: 每个 token 位置独立重建, 细线/局部纹理不再被全局池化稀释.

    a, b: 与 global_cosine 相同的特征列表 [(B,N,C) 或 (B,C,H,W)].
    hard_ratio>0 时只对距离最大的 top-k 比例位置取 mean (hard mining), 集中训练火力在最难重建的局部.
    """
    loss = 0.0
    n = len(a)
    for item in range(n):
        ai = _as_tokens(a[item].detach() if stop_grad else a[item])
        bi = _as_tokens(b[item])
        b_, n_, c_ = ai.shape
        d = 1 - F.cosine_similarity(ai.reshape(b_, n_, c_), bi.reshape(b_, n_, c_), dim=-1)  # (B,N)
        if hard_ratio > 0:
            k = max(1, int(d.numel() * hard_ratio))
            d = torch.topk(d.reshape(-1), k=k)[0]
        loss = loss + d.mean()
    return loss / n


# ---------- 监督 margin 损失 (用带框缺陷图作正样本, 直接推开正常/缺陷分数分布) ----------

def token_dist_map(en, de):
    """token 网格上的 1-cos(en,de), 按组平均 -> (B,1,H,W). 与推理端 cal_anomaly_maps(out_size=H) 同构 (不插值)."""
    maps = []
    for a, b in zip(en, de):
        m = 1 - F.cosine_similarity(a, b)      # (B,H,W)
        maps.append(m.unsqueeze(1))            # (B,1,H,W)
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)     # (B,1,H,W), 与 cal_anomaly_maps 同构


def image_scores(en, de):
    """图级分数: 每图异常图最大值 (与 calibrate 的 score 一致, 训练时不做高斯平滑) -> (B,)"""
    return token_dist_map(en, de).flatten(1).max(dim=1).values


def per_image_mean_dist(en, de):
    """每图异常图均值 -> (B,), 用于困难正常样本重加权."""
    return token_dist_map(en, de).flatten(1).mean(dim=1)


def margin_hinge(s_pos, s_neg, margin):
    """all-pairs hinge: 缺陷图分数 s_pos 须比正常图 s_neg 高出 margin.
    任一侧为空时返回 0 (保持计算图连通)."""
    if s_pos.numel() == 0 or s_neg.numel() == 0:
        return s_pos.sum() * 0.0 + s_neg.sum() * 0.0
    return F.relu(margin + s_neg.unsqueeze(1) - s_pos.unsqueeze(0)).mean()


def box_margin(dmap, mask, margin):
    """缺陷框内 token 平均距离须 >= margin (直接抬高 min_defect).
    dmap: (B,1,H,W); mask: (B,H,W) 0/1 -> 标量."""
    d = dmap.squeeze(1)
    m = mask.to(d.dtype)
    denom = m.sum(dim=(1, 2)).clamp(min=1.0)
    box_mean = (d * m).sum(dim=(1, 2)) / denom
    return F.relu(margin - box_mean).mean()


def masked_recon(dmap, fg_mask, dilate: int = 1):
    """缺陷图框外 token 的重建损失: 这些 token 是正常纹理, 应当像正常图一样被重建好 (距离越小越好).

    dmap: (B,1,H,W) token 距离图; fg_mask: (B,H,W) 缺陷框 token 掩码. 框向外膨胀 dilate 个 token 避开边界标注误差.
    没有框外 token 时返回 0 (保持计算图连通).
    """
    if dmap.numel() == 0:
        return dmap.sum() * 0.0
    fg = fg_mask.to(dmap.dtype).unsqueeze(1)
    if dilate > 0:
        fg = F.max_pool2d(fg, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
    bg = 1.0 - fg
    return (dmap * bg).sum() / bg.sum().clamp(min=1.0)
