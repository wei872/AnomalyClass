"""YOLO27n dual-head training criterion with non-interfering Foreground Alignment Supervision."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
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
    """``v8DetectionLoss`` subclass adding optional ``model.class_weights`` support."""

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


class YOLO27E2ELoss:
    """Official YOLO27 dual-head loss with full primary ``one2many`` supervision and clean ``fg_branch``.

    Why this outperforms YOLO26's ``E2ELoss``:
      1. Full 1.0x ``one2many`` supervision across all epochs: in YOLO26, ``E2ELoss`` decays ``o2m``
         down to ``0.1`` (starving backbone/neck gradients in the final 50 epochs because ``one2one``
         inputs are detached via ``x.detach()``). Keeping ``o2m = 1.0`` gives uninterrupted dense
         multi-positive supervision to the widened P2/P3 neck and default NMS ``one2many`` head,
         while ``o2o = 1.0`` independently trains the detached ``one2one`` head with zero gradient
         conflict in the shared backbone.
      2. Non-interfering Foreground Alignment Supervision (``fg_gain = 0.05``): supervises only the
         training-only ``fg_branch`` on shared neck features without distorting ``one2many`` or
         ``one2one`` classification logits.
      3. Dual-scale STAL safety: sets ``stride_val = 16`` when ``len(stride) == 2`` (``[8, 32]`` in
         ``yolo27n-lite``) so small defects are never forced onto stride 32.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn=YOLO27DetectionLoss,
        fg_gain: float = 0.05,
    ) -> None:
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)

        if len(self.one2many.stride) == 2:
            fine_stride_floor = int(self.one2many.stride[0].item() * 2)
            self.one2many.assigner.stride_val = fine_stride_floor
            self.one2one.assigner.stride_val = fine_stride_floor

        self.o2m = 1.0
        self.o2o = 1.0
        self.fg_gain = fg_gain

        base_names = list(getattr(self.one2many, "loss_names", ["box_loss", "cls_loss", "dfl_loss"]))
        if "fg_loss" not in base_names:
            self.loss_names = tuple([*base_names, "fg_loss"])
        else:
            self.loss_names = tuple(base_names)

    def _foreground_branch_loss(
        self,
        fg_logits: torch.Tensor | None,
        one2many: dict[str, torch.Tensor],
        o2m_assign: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Supervise the training-only neck foreground saliency branch using detached IoU quality."""
        if fg_logits is None or self.fg_gain <= 0.0:
            return one2many["scores"].sum() * 0.0

        fg_mask_o2m, _, target_bboxes_o2m, anchor_points, stride_tensor = o2m_assign
        quality = torch.zeros_like(fg_mask_o2m, dtype=one2many["scores"].dtype)
        if fg_mask_o2m.any():
            pred_distri = one2many["boxes"].permute(0, 2, 1).contiguous()
            pred_bboxes = self.one2many.bbox_decode(anchor_points, pred_distri.detach())
            scaled_targets = target_bboxes_o2m / stride_tensor
            iou = bbox_iou(pred_bboxes[fg_mask_o2m], scaled_targets[fg_mask_o2m], xywh=False, CIoU=False).squeeze(-1)
            quality[fg_mask_o2m] = iou.clamp(0.0, 1.0).to(quality.dtype)

        fg_pred = fg_logits.squeeze(1)  # (B, A)
        fg_target = quality.detach()
        fg_norm = fg_target.sum().clamp(min=1.0)
        return _varifocal_sum(fg_pred, fg_target, fg_mask_o2m.float()) / fg_norm

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
        """Compute scalar total loss and version-compatible loss_items (dict or Tensor)."""
        parsed = self.one2many.parse_output(preds)
        one2many = parsed["one2many"]
        one2one = parsed["one2one"]
        fg_logits = parsed.get("fg_logits")

        bs = one2many["boxes"].shape[0]
        o2m_assign, loss_o2m, loss_items_o2m = self.one2many.get_assigned_targets_and_loss(one2many, batch)
        _, loss_o2o, _ = self.one2one.get_assigned_targets_and_loss(one2one, batch)

        det_loss = (loss_o2m * self.o2m + loss_o2o * self.o2o) * bs
        fg_loss = self._foreground_branch_loss(fg_logits, one2many, o2m_assign)
        fg_loss_scaled = fg_loss * self.fg_gain * bs

        total_loss = det_loss.sum() + fg_loss_scaled.sum()

        fg_scalar = fg_loss.detach() * self.fg_gain
        if isinstance(loss_items_o2o := loss_items_o2m, dict):
            loss_items: dict[str, torch.Tensor] | torch.Tensor = {
                **loss_items_o2o,
                "fg_loss": fg_scalar,
            }
        else:
            fg_item = fg_scalar.flatten().to(loss_items_o2m.dtype)
            loss_items = torch.cat([loss_items_o2m.detach().flatten(), fg_item])

        return total_loss, loss_items
