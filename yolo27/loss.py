"""YOLO27n training criterion with Foreground Alignment Supervision and calibrated ProgLoss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


def _varifocal_sum(
    pred_logits: torch.Tensor,
    gt_score: torch.Tensor,
    pos_mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Compute elementwise-summed Varifocal Loss without ``.mean(1)`` anchor-count attenuation."""
    pred_f = pred_logits.float()
    gt_f = gt_score.float()
    mask_f = pos_mask.float()
    weight = alpha * pred_f.sigmoid().pow(gamma) * (1.0 - mask_f) + gt_f * mask_f
    with torch.amp.autocast(device_type=pred_logits.device.type, enabled=False):
        bce = F.binary_cross_entropy_with_logits(pred_f, gt_f, reduction="none")
    return (bce * weight).sum()


class YOLO27DetectionLoss(v8DetectionLoss):
    """``v8DetectionLoss`` subclass adding ``model.class_weights`` support on Ultralytics <= 8.4.33."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self._base_handles_class_weights = hasattr(self, "class_weights")
        if not self._base_handles_class_weights:
            cw = getattr(model, "class_weights", None)
            self._custom_class_weights = cw.to(self.device).view(1, 1, -1) if cw is not None else None
        else:
            self._custom_class_weights = None

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], **assign_kwargs
    ) -> tuple:
        if self._base_handles_class_weights or self._custom_class_weights is None:
            return super().get_assigned_targets_and_loss(preds, batch, **assign_kwargs)

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
            **assign_kwargs,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype)) * self._custom_class_weights.to(dtype)
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
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
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class YOLO27E2ELoss(E2ELoss):
    """End-to-end dual-head loss with Foreground Alignment Supervision and balanced ProgLoss.

    Key design:
      1. Balanced ProgLoss (``o2m + o2o = 1.0``, ``o2m: 0.8 -> 0.2``, ``o2o: 0.2 -> 0.8``):
         matches Ultralytics ``optimizer='auto'`` gradient norm scale (never doubling step size),
         keeps a ``2x`` healthier late-epoch gradient floor (``0.2`` vs ``0.1``) for widened P2/P3
         and strip-enhancer layers, and stabilizes neck features in late epochs so detached
         ``one2one`` converges cleanly.
      2. STAL ``stride_val = 2 * stride[0] = 16`` & ``tal_topk = 13`` for small/elongated defects.
      3. Properly normalized Foreground Alignment Supervision (``fg_gain = 0.15``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn=YOLO27DetectionLoss,
        fg_gain: float = 0.15,
        o2o_align_gain: float = 0.25,
    ) -> None:
        super().__init__(model, loss_fn=loss_fn)
        self.one2many = loss_fn(model, tal_topk=13)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)

        fine_stride_floor = int(self.one2many.stride[0].item() * 2)
        self.one2many.assigner.stride_val = fine_stride_floor
        self.one2one.assigner.stride_val = fine_stride_floor

        # ProgLoss with total = 1.0: o2m decays 0.8 -> 0.2 while o2o ramps 0.2 -> 0.8
        self.total = 1.0
        self.o2m = 0.8
        self.o2m_copy = 0.8
        self.final_o2m = 0.2
        self.o2o = self.total - self.o2m

        self.fg_gain = fg_gain
        self.o2o_align_gain = o2o_align_gain

        base_names = list(getattr(self.one2many, "loss_names", ["box_loss", "cls_loss", "dfl_loss"]))
        if "fg_loss" not in base_names:
            self.loss_names = tuple([*base_names, "fg_loss"])
        else:
            self.loss_names = tuple(base_names)

    def _compute_iou_quality(
        self,
        branch_preds: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached IoU alignment quality in [0, 1] at positive anchor locations."""
        quality = torch.zeros_like(fg_mask, dtype=branch_preds["scores"].dtype)
        if not fg_mask.any():
            return quality
        pred_distri = branch_preds["boxes"].permute(0, 2, 1).contiguous()
        pred_bboxes = self.one2many.bbox_decode(anchor_points, pred_distri.detach())
        scaled_targets = target_bboxes / stride_tensor
        iou = bbox_iou(pred_bboxes[fg_mask], scaled_targets[fg_mask], xywh=False, CIoU=False).squeeze(-1)
        quality[fg_mask] = iou.clamp(0.0, 1.0).to(quality.dtype)
        return quality

    def _foreground_alignment_loss(
        self,
        fg_logits: torch.Tensor | None,
        one2many: dict[str, torch.Tensor],
        one2one: dict[str, torch.Tensor],
        o2m_assign: tuple[torch.Tensor, ...],
        o2o_assign: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute Foreground Alignment Supervision loss across all detection anchors."""
        o2o_scores = one2one["scores"]  # (B, nc, A)
        if fg_logits is None:
            return o2o_scores.sum() * 0.0

        fg_mask_o2m, _, target_bboxes_o2m, anchor_points, stride_tensor = o2m_assign
        fg_mask_o2o, _, target_bboxes_o2o, _, _ = o2o_assign

        q_o2m = self._compute_iou_quality(
            one2many,
            fg_mask_o2m,
            target_bboxes_o2m,
            anchor_points,
            stride_tensor,
        )
        q_o2o = self._compute_iou_quality(
            one2one,
            fg_mask_o2o,
            target_bboxes_o2o,
            anchor_points,
            stride_tensor,
        )

        # 1. Auxiliary Foreground Branch supervision
        fg_pred = fg_logits.squeeze(1)  # (B, A)
        fg_pos_mask = (fg_mask_o2m.bool() | fg_mask_o2o.bool()).float()
        fg_target = torch.maximum(q_o2o, 0.75 * q_o2m).detach()
        fg_norm = fg_target.sum().clamp(min=1.0)
        loss_fg_branch = _varifocal_sum(fg_pred, fg_target, fg_pos_mask) / fg_norm

        # 2. One-to-One Foreground Consistency Alignment
        o2o_fg_logit = o2o_scores.amax(dim=1)  # (B, A)
        fg_prob_detached = fg_pred.detach().sigmoid()
        o2o_pos_mask = fg_mask_o2o.bool().float()
        o2o_fg_target = torch.where(
            fg_mask_o2o.bool(),
            0.5 * q_o2o + 0.5 * fg_prob_detached,
            torch.zeros_like(q_o2o),
        ).detach()
        o2o_norm = o2o_fg_target.sum().clamp(min=1.0)
        loss_o2o_align = _varifocal_sum(o2o_fg_logit, o2o_fg_target, o2o_pos_mask) / o2o_norm

        return loss_fg_branch + self.o2o_align_gain * loss_o2o_align

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
        """Compute scalar total loss and version-compatible loss_items (dict or Tensor)."""
        parsed = self.one2many.parse_output(preds)
        one2many = parsed["one2many"]
        one2one = parsed["one2one"]
        fg_logits = parsed.get("fg_logits")

        bs = one2many["boxes"].shape[0]
        o2m_assign, loss_o2m, _ = self.one2many.get_assigned_targets_and_loss(one2many, batch)
        o2o_assign, loss_o2o, loss_items_o2o = self.one2one.get_assigned_targets_and_loss(one2one, batch)

        det_loss = (loss_o2m * self.o2m + loss_o2o * self.o2o) * bs
        fg_loss = self._foreground_alignment_loss(fg_logits, one2many, one2one, o2m_assign, o2o_assign)
        fg_loss_scaled = fg_loss * self.fg_gain * bs

        total_loss = det_loss.sum() + fg_loss_scaled.sum()

        fg_scalar = fg_loss.detach() * self.fg_gain
        if isinstance(loss_items_o2o, dict):
            loss_items: dict[str, torch.Tensor] | torch.Tensor = {
                **loss_items_o2o,
                "fg_loss": fg_scalar,
            }
        else:
            fg_item = fg_scalar.flatten().to(loss_items_o2o.dtype)
            loss_items = torch.cat([loss_items_o2o.detach().flatten(), fg_item])

        return total_loss, loss_items
