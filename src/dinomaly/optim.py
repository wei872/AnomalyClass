"""StableAdamW (与官方 Dinomaly2 训练配方一致).

移植自 Dinomaly2 optimizers/StableAdamW.py (Apache-2.0), 思路来自 "Stable and low-precision training for
large-scale vision-language models": 在 AdamW 上, 当更新量的 RMS 超过 clip_threshold 时按比例缩小步长,
抑制 loss 尖峰. 与官方的差别: 梯度出现 NaN/Inf 时 **整步跳过** (官方只在遇到第一个坏参数时 return,
此前的参数可能已被更新过一半).
"""
import math

import torch
from torch.optim.optimizer import Optimizer


class StableAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 clip_threshold: float = 1.0):
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError('invalid lr / eps / weight_decay')
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError('invalid betas')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      amsgrad=amsgrad, clip_threshold=clip_threshold))
        self.skipped_steps = 0

    @staticmethod
    def _rms(t: torch.Tensor) -> float:
        return float(t.norm(2) / (t.numel() ** 0.5))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grads = [p.grad for g in self.param_groups for p in g['params'] if p.grad is not None]
        if any(not torch.isfinite(gr).all() for gr in grads):
            self.skipped_steps += 1
            return loss
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    torch.max(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                    denom = (state['max_exp_avg_sq'].sqrt() / math.sqrt(bc2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group['eps'])
                lr_scale = max(1.0, self._rms(grad / denom) / group['clip_threshold'])
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / bc1 / lr_scale)
        return loss
