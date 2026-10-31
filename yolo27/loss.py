"""YOLO27n training criterion with Foreground Alignment Supervision."""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.loss import E2ELoss, VarifocalLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou


class YOLO27E2ELoss(E2ELoss):
    """End-to-end dual-head loss with Foreground Alignment Supervision for YOLO27n/s."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn=v8DetectionLoss,
        fg_gain: float = 1.0,
        o2o_align_gain: float = 0.5,
    ) -> None:
        super().__init__(model, loss_fn=loss_fn)
        self.fg_gain = fg_gain
        self.o2o_align_gain = o2o_align_gain
        self.vfl = VarifocalLoss(gamma=2.0, alpha=0.75)

        # 让进度条 / logger 知道多了 fg_loss 这一项
        base_names = list(getattr(self, "loss_names", ["box_loss", "cls_loss", "dfl_loss"]))
        if "fg_loss" not in base_names:
            self.loss_names = base_names + ["fg_loss"]

    def _compute_iou_quality(
        self,
        branch_preds: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached CIoU/IoU alignment quality in [0, 1] at positive anchor locations."""
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

        q_o2m = self._compute_iou_quality(
            {"boxes": one2one["boxes"], "scores": o2o_scores},
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

        # 1. 辅助前景分支监督
        fg_pred = fg_logits.squeeze(1)  # (B, A)
        fg_pos_mask = (fg_mask_o2m.bool() | fg_mask_o2o.bool()).float()
        fg_target = torch.maximum(q_o2o, 0.5 * q_o2m).detach()
        fg_norm = fg_target.sum().clamp(min=1.0)
        loss_fg_branch = (
            self.vfl(
                fg_pred.unsqueeze(-1),
                fg_target.unsqueeze(-1),
                fg_pos_mask.unsqueeze(-1),
            )
            / fg_norm
        )

        # 2. one-to-one 前景一致性对齐
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

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total progressive dual-head loss + foreground alignment supervision loss."""
        parsed = self.one2many.parse_output(preds)
        one2many = parsed["one2many"]
        one2one = parsed["one2one"]
        fg_logits = parsed.get("fg_logits")

        bs = one2many["boxes"].shape[0]
        o2m_assign, loss_o2m, _ = self.one2many.get_assigned_targets_and_loss(one2many, batch)
        o2o_assign, loss_o2o, loss_items_o2o = self.one2one.get_assigned_targets_and_loss(one2one, batch)

        det_loss = (loss_o2m * self.o2m + loss_o2o * self.o2o) * bs
        fg_loss = self._foreground_alignment_loss(fg_logits, one2one, o2m_assign, o2o_assign)
        fg_loss_scaled = fg_loss * self.fg_gain * bs

        # total_loss 必须是标量，trainer 里会做 loss.backward()
        total_loss = det_loss.sum() + fg_loss_scaled.sum()

        # loss_items_o2o 是 Tensor（[box_loss, cls_loss, dfl_loss]），不能 ** 展开成 dict
        fg_item = (fg_loss.detach() * self.fg_gain).flatten().to(loss_items_o2o.dtype)
        loss_items = torch.cat([loss_items_o2o.detach().flatten(), fg_item])
        return total_loss, loss_items