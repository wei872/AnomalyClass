"""YOLO27 loss functions: Anisotropic High-IoU Box Loss + Live Neck FG Alignment + Balanced Dual-Head Criterion.

Reference: https://docs.ultralytics.com/zh/models/yolo27

Key training-dynamics and ``mAP50-95`` localization improvements over stock ``E2ELoss``:
    1. Anisotropic Box-Scale Smooth-L1 Edge Refinement (``YOLO27BboxLoss``):
       In ``reg_max=1`` models (``YOLO26n``), Ultralytics normalizes ``ltrb`` L1 error by
       ``imgsz=640``, shrinking a ``2 px`` thickness error on an ``8 px`` weft break
       (``断纬``) or ``12 px`` pinhole (``蚊虫小黑点``) down to ``2 / 640 = 0.003`` while
       pure ``F.l1_loss`` has a discontinuous ``±1`` sign gradient at ``0 px``.
       ``YOLO27BboxLoss`` adds a bounded anisotropic box-scale Smooth-L1 (``beta=0.10``,
       ``min_edge_px=24.0``) term that sharpens ``1..4 px`` edge alignment for ``AP75..AP95``
       (``mAP50-95``) and settles quadratically near ``0 px`` without late-epoch chatter.
    2. Normalized Dual-Head Progressive Schedule (``o2m: 0.80 -> 0.55``, ``o2o = 1 - o2m``):
       Preserves strong multi-positive gradient signal on the shared neck and default
       ``one2many + NMS`` head throughout ``close_mosaic`` fine-tuning while keeping
       ``o2m + o2o == 1.0``.
    3. Zero-Jitter Native TAL + Reused Foreground Alignment Supervision (``fg_gain = 0.05``):
       Runs native Ultralytics ``v8DetectionLoss`` for both heads (zero ``fg_prior``
       perturbation of ``one2one`` positive anchors) and reuses ``one2many``'s foreground
       assignment to supervise ``ForegroundAlignmentBranch`` on live neck features without
       invoking ``TaskAlignedAssigner`` a 3rd time per batch.
    4. Dual-Scale STAL Calibration:
       Adjusts ``stride_val = 16`` when ``[P3/8, P5/32]`` is used so Small-Target-Aware
       Label Assignment activates accurately on tiny ``< 16 px`` fabric defects.
"""

from __future__ import annotations

import inspect
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors

_HAS_TAL_TOPK2 = "tal_topk2" in inspect.signature(v8DetectionLoss.__init__).parameters


def _varifocal_sum(pred_logits: torch.Tensor, gt_quality: torch.Tensor, fg_label: torch.Tensor) -> torch.Tensor:
    """FP32 Varifocal sum matching v8DetectionLoss target_scores_sum normalization scale."""
    pred_f = pred_logits.float()
    gt_f = gt_quality.float()
    lbl_f = fg_label.float()
    weight = 0.75 * pred_f.sigmoid().pow(2.0) * (1.0 - lbl_f) + gt_f * lbl_f
    return (F.binary_cross_entropy_with_logits(pred_f, gt_f, reduction="none") * weight).sum()


class _ForegroundBalancedBCE(nn.Module):
    """BCE loss boosting ONLY foreground positive targets on rare tail classes (keeping background at 1.0)."""

    def __init__(self, pos_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weights is not None:
            self.register_buffer("pos_weights", pos_weights.detach().float().view(1, 1, -1), persistent=False)
        else:
            self.pos_weights = None

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
        if self.pos_weights is not None:
            w = self.pos_weights.to(device=loss.device, dtype=loss.dtype)
            pos_mask = (target_scores > 0.0).to(dtype=loss.dtype)
            loss = loss * (1.0 + (w - 1.0) * pos_mask)
        return loss


class YOLO27BboxLoss(BboxLoss):
    """High-IoU ``BboxLoss`` adding anisotropic box-scale Smooth-L1 edge refinement for ``reg_max=1``."""

    def __init__(self, reg_max: int = 1, edge_gain: float = 0.25, min_edge_px: float = 24.0) -> None:
        super().__init__(reg_max=reg_max)
        self.edge_gain = float(edge_gain)
        self.min_edge_px = float(min_edge_px)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CIoU + image-normalized L1 + anisotropic box-scale Smooth-L1 edge refinement."""
        loss_iou, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )

        if not self.dfl_loss and self.edge_gain > 0.0:
            fg_idx = fg_mask.nonzero(as_tuple=True)
            if fg_idx[0].numel() > 0:
                weight = target_scores[fg_idx].sum(-1, keepdim=True)
                stride_fg = stride[fg_idx[1]]
                anchors_fg = anchor_points[fg_idx[1]]
                gt_fg = target_bboxes[fg_idx]

                x1y1, x2y2 = gt_fg.chunk(2, dim=-1)
                target_ltrb_px = torch.cat((anchors_fg - x1y1, x2y2 - anchors_fg), dim=-1) * stride_fg
                pred_ltrb_px = pred_dist[fg_idx] * stride_fg

                gt_px = gt_fg * stride_fg
                gt_w = (gt_px[:, 2] - gt_px[:, 0]).clamp(min=self.min_edge_px)
                gt_h = (gt_px[:, 3] - gt_px[:, 1]).clamp(min=self.min_edge_px)
                denom = torch.stack((gt_w, gt_h, gt_w, gt_h), dim=-1)

                edge_rel = F.smooth_l1_loss(
                    pred_ltrb_px / denom,
                    target_ltrb_px / denom,
                    beta=0.10,
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
                loss_dfl = loss_dfl + self.edge_gain * ((edge_rel * weight).sum() / target_scores_sum)

        return loss_iou, loss_dfl


class YOLO27DetectionLoss(v8DetectionLoss):
    """Per-branch detection criterion with anisotropic edge refinement and dual-scale STAL calibration."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        if _HAS_TAL_TOPK2:
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        else:
            super().__init__(model, tal_topk=tal_topk2 or tal_topk)

        self.bbox_loss = YOLO27BboxLoss(self.reg_max, edge_gain=0.20).to(self.device)
        pos_class_weights = getattr(model, "_pos_class_weights", None)
        if pos_class_weights is not None:
            self.bce = _ForegroundBalancedBCE(pos_class_weights).to(self.device)

        # >=8.4.194 defines self.loss_names & returns dict; <=8.4.33 returns 1D Tensor
        self._returns_dict: bool = hasattr(self, "loss_names")
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")

        # Dual-scale [P3/8, P5/32]: set STAL stride threshold to 16 instead of 32
        if (
            self.stride.numel() == 2
            and round(float(self.stride[0].item())) == 8
            and round(float(self.stride[1].item())) == 32
        ):
            self.assigner.stride_val = 16

        if class_weights is not None:
            w = class_weights.to(device=self.device, dtype=torch.float32)
            if hasattr(self, "class_weights"):
                self.class_weights = w
                self._legacy_class_weights: torch.Tensor | None = None
            else:
                self._legacy_class_weights = w
        else:
            self._legacy_class_weights = None

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Compute assigned targets and losses natively across Ultralytics 8.4.33 and >=8.4.197."""
        if self._legacy_class_weights is None:
            return super().get_assigned_targets_and_loss(preds, batch)

        # Fallback path only when custom class_weights is supplied on legacy Ultralytics <=8.4.33
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_raw = self.bce(pred_scores, target_scores.to(dtype))
        cls_raw = cls_raw * self._legacy_class_weights.to(dtype=cls_raw.dtype, device=cls_raw.device).view(1, 1, -1)
        loss[1] = cls_raw.sum() / target_scores_sum

        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        detached = loss.detach()
        if self._returns_dict:
            return (
                (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
                loss,
                dict(zip(self.loss_names, detached)),
            )
        return (fg_mask, target_scores, target_scores_sum), loss

    def __call__(
        self,
        preds: Any,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate batch-scaled branch loss scalar and formatted loss items."""
        if self._legacy_class_weights is None:
            raw_loss, items = super().__call__(preds, batch)
            return raw_loss.sum(), items

        preds = preds[1] if isinstance(preds, tuple) else preds
        res = self.get_assigned_targets_and_loss(preds, batch)
        loss = res[1]
        total = loss.sum() * preds["boxes"].shape[0]
        detached = loss.detach()
        items: torch.Tensor | dict[str, torch.Tensor] = (
            dict(zip(self.loss_names, detached)) if self._returns_dict else detached
        )
        return total, items


class YOLO27E2ELoss:
    """Dual-head + Foreground Alignment criterion for YOLO27n / YOLO27n-Dual / YOLO27n-Lite."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_weights: torch.Tensor | None = None,
        fg_gain: float = 0.05,
        init_o2m: float = 0.80,
        final_o2m: float = 0.55,
    ) -> None:
        self.one2many = YOLO27DetectionLoss(model, tal_topk=10, class_weights=class_weights)
        self.one2one = YOLO27DetectionLoss(model, tal_topk=7, tal_topk2=1, class_weights=class_weights)
        self._returns_dict: bool = self.one2many._returns_dict
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "fg_loss")

        self.fg_gain = float(fg_gain)

        self.init_o2m = float(init_o2m)
        self.final_o2m = float(final_o2m)
        self.o2m = self.init_o2m
        self.o2o = 1.0 - self.o2m

        total_epochs = max(getattr(self.one2many.hyp, "epochs", 100), 1)
        self.decay_steps = max(total_epochs - 1, 1)
        self.step = 0

    def update(self) -> None:
        """Advance dual-head schedule once per epoch while preserving ``o2m + o2o == 1.0``."""
        self.step = min(self.step + 1, self.decay_steps)
        progress = self.step / self.decay_steps
        self.o2m = self.init_o2m + (self.final_o2m - self.init_o2m) * progress
        self.o2o = 1.0 - self.o2m

    @staticmethod
    def _loss_dict_or_tensor_to_vec(loss_items: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize 3-component loss_items across Ultralytics 8.4.33 (Tensor) and >=8.4.194 (dict)."""
        if isinstance(loss_items, dict):
            return torch.stack(list(loss_items.values()))
        return loss_items

    def _compute_fg_alignment_from_o2m(
        self,
        fg_logits: torch.Tensor,
        o2m_preds: dict[str, torch.Tensor],
        o2m_assign: tuple[Any, ...],
        has_gt: bool,
    ) -> torch.Tensor:
        """Supervise live neck foreground logits by reusing one2many's TAL assignment (zero extra TAL pass)."""
        dtype = fg_logits.dtype
        batch_size = fg_logits.shape[0]

        if not has_gt:
            fg_zero = torch.zeros_like(fg_logits)
            return (_varifocal_sum(fg_logits, fg_zero, fg_zero) / max(batch_size, 1)) * self.fg_gain

        with torch.no_grad():
            if len(o2m_assign) == 3:
                # Ultralytics <=8.4.33 returns (fg_mask, target_scores, target_scores_sum)
                fg_mask, target_scores, _ = o2m_assign
                fg_quality = target_scores.amax(dim=-1, keepdim=True).clamp(0.0, 1.0).to(dtype)
                fg_label = fg_mask.unsqueeze(-1).to(dtype)
            else:
                # Ultralytics >=8.4.194 returns (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor)
                fg_mask, _, target_bboxes, anchor_points, stride_tensor = o2m_assign
                fg_quality = torch.zeros_like(fg_logits)
                fg_idx = fg_mask.nonzero(as_tuple=True)
                if fg_idx[0].numel() > 0:
                    pred_distri = o2m_preds["boxes"].detach().permute(0, 2, 1).contiguous()
                    pred_boxes_fg = (
                        self.one2many.bbox_decode(anchor_points, pred_distri)[fg_idx] * stride_tensor[fg_idx[1]]
                    )
                    gt_boxes_fg = target_bboxes[fg_idx]
                    iou_fg = bbox_iou(pred_boxes_fg, gt_boxes_fg, xywh=False, CIoU=False).squeeze(-1).clamp(0.0, 1.0)
                    fg_quality[fg_idx[0], fg_idx[1], 0] = iou_fg.to(dtype)
                fg_label = fg_mask.unsqueeze(-1).to(dtype)

            norm = fg_quality.sum().clamp(min=1.0)

        return (_varifocal_sum(fg_logits, fg_quality, fg_label) / norm) * self.fg_gain

    def _compute_fg_alignment_loss(
        self,
        fg_logits: torch.Tensor,
        o2m_preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Standalone helper for computing foreground alignment loss from raw ``o2m_preds`` and ``batch``."""
        o2m_res = self.one2many.get_assigned_targets_and_loss(o2m_preds, batch)
        return self._compute_fg_alignment_from_o2m(
            fg_logits,
            o2m_preds,
            o2m_res[0],
            has_gt=batch["bboxes"].numel() > 0,
        )

    def __call__(
        self, preds: Any, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate weighted ``one2many`` + conflict-free ``one2one`` + live neck FG alignment loss."""
        preds = preds[1] if isinstance(preds, tuple) else preds

        if not isinstance(preds, dict) or "one2many" not in preds:
            total_single, single_items = self.one2many(preds, batch)
            vec_single = self._loss_dict_or_tensor_to_vec(single_items)
            zero_fg = torch.zeros(1, device=vec_single.device, dtype=vec_single.dtype)
            vec4 = torch.cat([vec_single, zero_fg], dim=0)
            return total_single.sum(), (dict(zip(self.loss_names, vec4)) if self._returns_dict else vec4)

        one2many = preds["one2many"]
        one2one = preds["one2one"]
        fg_logits: torch.Tensor | None = preds.get("fg_align")
        batch_size = one2many["boxes"].shape[0]

        o2m_res = self.one2many.get_assigned_targets_and_loss(one2many, batch)
        o2m_assign, o2m_loss_vec = o2m_res[0], o2m_res[1]
        loss_o2m = o2m_loss_vec.sum() * batch_size
        vec_o2m = o2m_loss_vec.detach()

        loss_o2o, items_o2o = self.one2one(one2one, batch)
        vec_o2o = self._loss_dict_or_tensor_to_vec(items_o2o)

        det_items = vec_o2m * self.o2m + vec_o2o * self.o2o
        total_loss = loss_o2m.sum() * self.o2m + loss_o2o.sum() * self.o2o

        if fg_logits is not None:
            fg_loss_unscaled = self._compute_fg_alignment_from_o2m(
                fg_logits,
                one2many,
                o2m_assign,
                has_gt=batch["bboxes"].numel() > 0,
            )
            total_loss = total_loss + fg_loss_unscaled * batch_size
            fg_item = fg_loss_unscaled.detach().reshape(1)
        else:
            fg_item = torch.zeros(1, device=det_items.device, dtype=det_items.dtype)

        all_items = torch.cat([det_items, fg_item], dim=0)
        if self._returns_dict:
            return total_loss, dict(zip(self.loss_names, all_items))
        return total_loss, all_items
