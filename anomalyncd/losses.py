"""AnomalyNCD/SimGCD objectives, with disjoint known/novel logits and safe edges.

Unlike the original AeBAD setting, known and novel classes here come from the
same 19-class fabric label space. There is no extra normal output or anomaly-map
confidence correction: the supplied supervision is a YOLO box, not a heatmap.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss; anchors without a positive contribute zero."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, views, D], normalized by caller.
        batch, views = features.shape[:2]
        if batch == 0:
            return features.sum() * 0
        # reshape([B, V, D]) is SAMPLE-major: [s0v0, s0v1, s1v0, ...].
        # The label mask must use the same order; tiling the [B, B] mask with
        # .repeat(views, views) would be view-major and pair the wrong samples.
        flat = F.normalize(features.reshape(batch*views, -1), dim=-1)
        similarity = flat @ flat.T / self.temperature
        eye = torch.eye(batch*views, dtype=torch.bool, device=features.device)
        similarity = similarity.masked_fill(eye, torch.finfo(similarity.dtype).min)
        flat_labels = labels.repeat_interleave(views)
        positives = (flat_labels[:, None] == flat_labels[None, :]) & ~eye
        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        counts = positives.sum(dim=1)
        valid = counts > 0
        if not bool(valid.any()):
            return features.sum() * 0
        loss = -(log_prob[valid] * positives[valid]).sum(dim=1) / counts[valid]
        return loss.mean()


def info_nce_loss(features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """View-major [all view0, all view1], not sample-major ordering."""
    features = F.normalize(features, dim=-1)
    batch = features.shape[0] // 2
    similarity = features @ features.T / temperature
    eye = torch.eye(2*batch, dtype=torch.bool, device=features.device)
    logits = similarity.masked_fill(eye, torch.finfo(similarity.dtype).min)
    targets = torch.cat((torch.arange(batch, 2*batch), torch.arange(batch))).to(features.device)
    return F.cross_entropy(logits, targets)


class DistillLoss(nn.Module):
    def __init__(self, n_known: int, n_classes: int, epochs: int, warmup_epochs: int = 40,
                 warmup_temp: float = 0.07, teacher_temp: float = 0.04,
                 student_temp: float = 0.1, memax_weight: float = 4.0):
        super().__init__()
        self.n_known, self.n_classes = n_known, n_classes
        schedule = [teacher_temp] * epochs
        if warmup_epochs:
            schedule[:warmup_epochs] = torch.linspace(warmup_temp, teacher_temp, warmup_epochs).tolist()
        self.schedule = schedule
        self.student_temp, self.teacher_temp, self.memax_weight = student_temp, teacher_temp, memax_weight

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, epoch: int) -> torch.Tensor:
        if student.shape[0] == 0:
            return student.sum() * 0
        # All teacher ops are detached; no validation labels enter this function.
        probs = F.softmax(teacher[:, self.n_known:].detach() / self.schedule[epoch], dim=-1)
        q0, q1 = probs.chunk(2, dim=0)
        s0, s1 = (student / self.student_temp).chunk(2, dim=0)
        # Keep known logits in the denominator, but never use known GT as novel targets.
        cross_view = -(q1 * F.log_softmax(s0, dim=-1)[:, self.n_known:]).sum(-1).mean()
        cross_view = cross_view - (q0 * F.log_softmax(s1, dim=-1)[:, self.n_known:]).sum(-1).mean()
        cross_view = cross_view / 2
        mean = F.softmax(student / self.student_temp, dim=-1).mean(0).clamp_min(1e-8)
        # KL(mean prediction || uniform), equivalent to the original ME-Max term.
        memax = (mean * mean.log()).sum() + torch.log(torch.tensor(float(self.n_classes), device=student.device))
        return cross_view + self.memax_weight * memax


class Objectives(nn.Module):
    def __init__(self, cfg: dict, n_classes: int, n_known: int):
        super().__init__()
        self.cfg, self.mode = cfg, cfg["mode"]
        self.n_classes, self.n_known = n_classes, n_known
        loss = cfg["loss"]
        self.supcon = SupConLoss(loss["supcon_temperature"])
        self.distill = DistillLoss(n_known, n_classes, cfg["training"]["epochs"],
                                   loss["warmup_teacher_temp_epochs"], loss["warmup_teacher_temp"],
                                   loss["teacher_temp"], loss["temperature"], loss["memax_weight"])

    def forward(self, projection: torch.Tensor, logits: list[torch.Tensor], targets: torch.Tensor,
                labelled: torch.Tensor, epoch: int) -> tuple[torch.Tensor, dict, list[float]]:
        loss_cfg = self.cfg["loss"]
        # Dataset yields view-major lists; concatenation preserves that order.
        labels = targets.repeat(2)
        known = labelled.repeat(2)
        novel = ~known
        features = F.normalize(projection, dim=-1)
        zero = projection.sum() * 0
        supcon = zero
        if known.any():
            supcon = self.supcon(torch.stack(features[known].chunk(2), dim=1), labels[known].chunk(2)[0])
        ce_heads, cluster_heads = [], []
        for head in logits:
            if known.any():
                supervised = F.cross_entropy(head[known] / loss_cfg["temperature"], labels[known])
            else:
                supervised = zero
            ce_heads.append(supervised)
            if self.mode == "discovery":
                cluster_heads.append(self.distill(head[novel], head[novel].detach(), epoch) if novel.any() else zero)
            else:
                cluster_heads.append(zero)
        ce = torch.stack(ce_heads).mean()
        cluster = torch.stack(cluster_heads).mean()
        if self.mode == "supervised":
            total = ce + loss_cfg["supcon_weight"] * supcon
        else:
            contrastive = info_nce_loss(projection, loss_cfg["contrastive_temperature"])
            weight = loss_cfg["sup_weight"]
            total = (1-weight)*cluster + weight*ce + (1-weight)*contrastive + weight*supcon
        components = {"total": float(total.detach()), "ce": float(ce.detach()),
                      "cluster": float(cluster.detach()), "supcon": float(supcon.detach())}
        head_losses = [float(x.detach()) for x in (ce_heads if self.mode == "supervised" else cluster_heads)]
        return total, components, head_losses
