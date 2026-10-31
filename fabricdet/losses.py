# -*- coding: utf-8 -*-
"""损失函数: focal(obj) + GIoU/L1(box, centerness 加权) + 余弦CE(分类) + SupCon(开集度量)。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.5, gamma: float = 2.0,
                       pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    if pos_weight is not None:
        loss = loss * (targets * pos_weight + (1 - targets))
    return loss.mean()


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    inter_x1, inter_y1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
    inter_x2, inter_y2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    enc_x1, enc_y1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    enc_x2, enc_y2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    enc = (enc_x2 - enc_x1).clamp(min=1e-7) * (enc_y2 - enc_y1).clamp(min=1e-7)
    return 1 - iou + (enc - union) / enc


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """监督对比: 同类嵌入拉近, 异类推远 -> 类内紧簇/类间分离, 是开集未知检测的基础。"""
    if emb.shape[0] < 2:
        return emb.sum() * 0.0
    sim = emb @ emb.t() / temperature
    eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()
    self_mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    exp = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-8)
    pos_cnt = eq & ~self_mask
    denom = pos_cnt.sum(dim=1)
    valid = denom > 0
    if valid.sum() == 0:
        return emb.sum() * 0.0
    mean_log_prob_pos = (log_prob * pos_cnt).sum(dim=1)[valid] / denom[valid]
    return -mean_log_prob_pos.mean()


def build_targets(gt_boxes: torch.Tensor, gt_cls: torch.Tensor, h: int, w: int, stride: int,
                  ring_value: float = 0.4):
    """FCOS 式 Assign + CenterNet 式软目标:
      - 中心点落在 gt 内的 cell 为正样本 (box/cls/obj=1); 过小 gt 保底最近中心 cell;
      - gt 外扩 1 cell 的环带为软正样本 (obj=ring_value): 缓解 stride 网格上正样本过少
        导致的 conf 分支正梯度消失 (小缺陷只有 1~4 个正 cell 的关键问题)。
    返回 assign (h*w,) long (-1 为负), centerness (h*w,), obj_soft (h*w,)
    """
    device = gt_boxes.device
    n = h * w
    ys = (torch.arange(h, device=device, dtype=gt_boxes.dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=gt_boxes.dtype) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = gx.reshape(-1), gy.reshape(-1)
    assign = torch.full((n,), -1, dtype=torch.long, device=device)
    centerness = torch.zeros(n, device=device, dtype=gt_boxes.dtype)
    obj_soft = torch.zeros(n, device=device, dtype=gt_boxes.dtype)
    if gt_boxes.numel() == 0:
        return assign, centerness, obj_soft
    x1, y1, x2, y2 = gt_boxes.unbind(-1)
    l, t = cx[:, None] - x1[None], cy[:, None] - y1[None]
    r, b = x2[None] - cx[:, None], y2[None] - cy[:, None]
    dist = torch.stack([l, t, r, b], dim=-1)
    dmin = dist.min(dim=-1).values
    inside = dmin > 0                                                      # (n, G)
    ring = (dmin > -stride) & ~inside
    area = (x2 - x1) * (y2 - y1)
    cost = torch.where(inside, area[None].expand(n, -1), torch.full_like(inside, 1e18, dtype=gt_boxes.dtype))
    best = cost.argmin(dim=1)
    has = cost.min(dim=1).values < 1e17
    assign = torch.where(has, best, torch.full_like(best, -1))
    # 保底: 每个 gt 至少一个正样本 (中心最近 cell)
    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    ci = ((bcy / stride).floor().clamp(0, h - 1).long() * w
          + (bcx / stride).floor().clamp(0, w - 1).long())
    for gi in range(gt_boxes.shape[0]):
        cur = assign[ci[gi]]
        if cur.item() < 0 or area[cur].item() > area[gi].item():
            assign[ci[gi]] = gi
    pos = assign >= 0
    obj_soft[pos] = 1.0
    ring_cells = ring.any(dim=-1) & ~pos
    obj_soft[ring_cells] = ring_value
    g = assign[pos]
    d = dist[pos][torch.arange(g.shape[0], device=device), g]
    lr, tb = d[:, 0] * d[:, 2], d[:, 1] * d[:, 3]
    centerness[pos] = torch.sqrt(torch.clamp(lr / (tb + 1e-7), min=1e-4)).clamp(0.05, 1.0)
    return assign, centerness, obj_soft


class FabricLoss:
    def __init__(self, num_classes: int, cfg: dict, class_weight: torch.Tensor | None = None):
        self.s = 7
        self.w_obj = cfg.get("obj_w", 1.0)
        self.w_giou = cfg.get("giou_w", 1.0)
        self.w_l1 = cfg.get("l1_w", 0.5)
        self.w_cls = cfg.get("cls_w", 1.0)
        self.w_sc = cfg.get("supcon_w", 0.3)
        self.gamma = cfg.get("focal_gamma", 2.0)
        self.alpha = cfg.get("focal_alpha", 0.5)
        self.scale = cfg.get("logit_scale_init", 10.0)
        cw = torch.ones(num_classes) if class_weight is None else class_weight
        self.register = cw  # placeholder, moved to device lazily
        self._cw_dev = None

    def _class_weight(self, device):
        if self._cw_dev is None or self._cw_dev.device != device:
            self._cw_dev = self.register.to(device)
        return self._cw_dev

    def __call__(self, model, obj, box, emb, targets, scale=None):
        B, _, H, W = obj.shape
        device = obj.device
        obj_l = obj.reshape(B, H * W)
        box_p = box.permute(0, 2, 3, 1).reshape(B, H * W, 4)
        emb_p = emb.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        centers = model.grid_centers(H, W, device, obj.dtype)

        obj_t = torch.zeros_like(obj_l)
        cw = self._class_weight(device)
        ce_losses, giou_losses, l1_losses, pos_ws = [], [], [], []
        all_emb, all_lbl = [], []
        for b in range(B):
            gt = targets[b]
            if gt.numel() == 0:
                continue
            gcls = gt[:, 0].long()
            gbox = gt[:, 1:5]
            assign, cn, obj_soft = build_targets(gbox, gcls, H, W, self.s)
            pos = assign >= 0
            obj_t[b] = obj_soft
            if pos.sum() == 0:
                continue
            g = assign[pos]
            cnp = cn[pos]
            # box
            l = centers[pos, 0] - gbox[g, 0]
            t = centers[pos, 1] - gbox[g, 1]
            r = gbox[g, 2] - centers[pos, 0]
            bb = gbox[g, 3] - centers[pos, 1]
            tgt_ltrb = torch.stack([l, t, r, bb], dim=-1).clamp(min=0)
            pred_ltrb = torch.exp(box_p[b][pos].clamp(-8.0, 10.0)) * self.s
            pred_xyxy = torch.stack([centers[pos, 0] - pred_ltrb[:, 0], centers[pos, 1] - pred_ltrb[:, 1],
                                     centers[pos, 0] + pred_ltrb[:, 2], centers[pos, 1] + pred_ltrb[:, 3]], -1)
            gl = giou_loss(pred_xyxy, gbox[g])
            giou_losses.append((gl * cnp).sum() / cnp.sum())
            l1_losses.append((F.l1_loss(pred_ltrb, tgt_ltrb, reduction="none").mean(-1) * cnp).sum() / cnp.sum())
            # cls (余弦原型 + 可学习温度)
            e = torch.nn.functional.normalize(emb_p[b][pos], dim=-1)
            proto = torch.nn.functional.normalize(model.prototypes, dim=-1)
            s = scale if scale is not None else model.logit_scale.exp()
            logits = e @ proto.t() * s
            ce = torch.nn.functional.cross_entropy(logits, gcls[g], weight=cw, reduction="none")
            ce_losses.append((ce * cnp).sum() / cnp.sum())
            pos_ws.append(cw[gcls[g]])
            all_emb.append(e)
            all_lbl.append(gcls[g])

        loss_obj = sigmoid_focal_loss(obj_l, obj_t, self.alpha, self.gamma)
        out = {"obj": loss_obj * self.w_obj}
        total = loss_obj * self.w_obj
        if ce_losses:
            l_cls = torch.stack(ce_losses).mean()
            l_giou = torch.stack(giou_losses).mean()
            l_l1 = torch.stack(l1_losses).mean()
            e_cat = torch.cat(all_emb)
            l_cat = torch.cat(all_lbl)
            l_sc = supcon_loss(e_cat, l_cat)
            out.update(cls=l_cls * self.w_cls, giou=l_giou * self.w_giou,
                       l1=l_l1 * self.w_l1, supcon=l_sc * self.w_sc)
            total = total + l_cls * self.w_cls + l_giou * self.w_giou + l_l1 * self.w_l1 + l_sc * self.w_sc
        out["total"] = total
        return out
