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
