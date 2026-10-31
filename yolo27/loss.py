"""YOLO27n training criterion with Foreground Alignment Supervision and fast convergence fixes."""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.loss import E2ELoss, VarifocalLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou


class YOLO27E2ELoss(E2ELoss):
    """End-to-end dual-head loss with Foreground Alignment Supervision for YOLO27n/s.

    Key convergence improvements over stock ``E2ELoss`` on dual-scale custom datasets:
      1. Full unattenuated ``o2m = 1.0`` gradient throughout training: because ``one2one``
         detaches neck features (``x.detach()``), ``one2many`` is the sole detection gradient
         source for the backbone & neck as well as the default NMS validation head. Keeping
         ``o2m = 1.0`` prevents mid/late-epoch backbone gradient starvation while ``o2o``
         smoothly ramps from ``0.5 -> 1.0``.
      2. Dual-scale STAL ``stride_val`` fix: Ultralytics ``TaskAlignedAssigner`` defaults
         ``stride_val = stride[1]`` (16 on 3-scale ``[8, 16, 32]``, which jumps to **32** on
         dual-scale ``[8, 32]`` and dilutes tiny-defect candidates with background anchors).
         Restores ``stride_val = 2 * stride[0] = 16`` with ``tal_topk = 13`` for high recall.
      3. Accurate ``one2many`` IoU quality targets in Foreground Alignment Supervision and
         cross-version compatibility for both Tensor and dict ``loss_items``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn=v8DetectionLoss,
        fg_gain: float = 0.5,
        o2o_align_gain: float = 0.25,
    ) -> None:
        super().__init__(model, loss_fn=loss_fn)
        # Denser positive candidate pool for small/elongated fabric defects on fine P3/8 map
        self.one2many = loss_fn(model, tal_topk=13)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)

        # Fix dual-scale [8, 32] STAL minimum candidate box size: 2 * P3_stride = 16 px (not P5_stride = 32 px!)
        fine_stride_floor = int(self.one2many.stride[0].item() * 2)
        self.one2many.assigner.stride_val = fine_stride_floor
        self.one2one.assigner.stride_val = fine_stride_floor

        # Keep both o2m = 1.0 (sole detection gradient into backbone/neck & NMS head) and
        # o2o = 1.0 (detached end-to-end validation head, zero interference with backbone) at full strength
        self.o2m = 1.0
        self.init_o2o = 1.0
        self.final_o2o = 1.0
        self.o2o = self.init_o2o

        self.fg_gain = fg_gain
        self.o2o_align_gain = o2o_align_gain
        self.vfl = VarifocalLoss(gamma=2.0, alpha=0.75)

        base_names = list(getattr(self.one2many, "loss_names", ["box_loss", "cls_loss", "dfl_loss"]))
        if "fg_loss" not in base_names:
            self.loss_names = tuple([*base_names, "fg_loss"])
        else:
            self.loss_names = tuple(base_names)

    def update(self) -> None:
        """Keep ``o2m = 1.0`` and ``o2o = 1.0`` at full gradient throughout training."""
        self.updates += 1
        max_ep = max(getattr(self.one2one.hyp, "epochs", 100) - 1, 1)
        prog = min(max(self.updates / max_ep, 0.0), 1.0)
        self.o2m = 1.0
        self.o2o = self.init_o2o + prog * (self.final_o2o - self.init_o2o)

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
        """Compute Foreground Alignment Supervision loss across all dual-scale anchors."""
        o2o_scores = one2one["scores"]  # (B, nc, A)
        if fg_logits is None:
            return o2o_scores.sum() * 0.0

        fg_mask_o2m, _, target_bboxes_o2m, anchor_points, stride_tensor = o2m_assign
        fg_mask_o2o, _, target_bboxes_o2o, _, _ = o2o_assign

        # Use one2many's own regressed boxes on fg_mask_o2m so all dense positive anchors get accurate IoU quality
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
        loss_fg_branch = (
            self.vfl(
                fg_pred.unsqueeze(-1),
                fg_target.unsqueeze(-1),
                fg_pos_mask.unsqueeze(-1),
            )
            / fg_norm
        )

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
        loss_o2o_align = (
            self.vfl(
                o2o_fg_logit.unsqueeze(-1),
                o2o_fg_target.unsqueeze(-1),
                o2o_pos_mask.unsqueeze(-1),
            )
            / o2o_norm
        )

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

        # Scalar total_loss works across all Ultralytics versions (both scaler.scale(loss).backward() and loss.sum())
        total_loss = det_loss.sum() + fg_loss_scaled.sum()

        # Log one2one loss items (matches Ultralytics E2ELoss & end2end validator mAP head) + fg_loss
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
