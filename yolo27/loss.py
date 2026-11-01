"""YOLO27 loss functions: Foreground Alignment Supervision + Balanced Dual-Head Criterion.

Reference: https://docs.ultralytics.com/zh/models/yolo27

Key training-dynamics improvements over stock Ultralytics ``E2ELoss``:
    1. Normalized Dual-Head Progressive Schedule (``o2m: 0.80 -> 0.55``, ``o2o = 1 - o2m``):
       Stock ``E2ELoss`` decays ``one2many`` from ``0.80 -> 0.10``, starving the shared
       backbone/neck and the default ``one2many + NMS`` head of multi-positive gradients
       during the decisive ``close_mosaic`` fine-tuning phase. Keeping ``o2m >= 0.55``
       with ``o2m + o2o == 1.0`` preserves rich multi-positive feature learning while
       giving ``one2one`` ``0.20 -> 0.45`` weight without gradient-norm inflation.
    2. Live Neck Foreground Alignment Supervision (``fg_gain = 0.15``):
       Supervises ``ForegroundAlignmentBranch`` on **live** neck features using continuous
       IoU-quality targets from ``one2many`` via ``VarifocalLoss``, suppressing repetitive
       woven-fabric background false positives (including defect-free normal samples) at
       zero inference cost (stripped on ``fuse()``).
    3. Conflict-Free FG-Guided One-to-One Anchor Selection (``tal_topk=7, tal_topk2=1``):
       Combines two-stage conflict-free candidate filtering with detached ``fg_prior``
       saliency modulation so adjacent yarn/strip defects never drop anchors on collision
       and ``one2one`` locks onto true defect centers without any inference coupling.
    4. Dual-Scale STAL Calibration:
       Adjusts ``stride_val = 16`` when ``[P3/8, P5/32]`` is used so Small-Target-Aware
       Label Assignment activates accurately on tiny ``< 16 px`` fabric defects.
"""

from __future__ import annotations

import inspect
from typing import Any
import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

_HAS_TAL_TOPK2 = "tal_topk2" in inspect.signature(v8DetectionLoss.__init__).parameters


def _varifocal_sum(pred_logits: torch.Tensor, gt_quality: torch.Tensor, fg_label: torch.Tensor) -> torch.Tensor:
    """FP32 Varifocal sum matching v8DetectionLoss target_scores_sum normalization scale."""
    pred_f = pred_logits.float()
    gt_f = gt_quality.float()
    lbl_f = fg_label.float()
    weight = 0.75 * pred_f.sigmoid().pow(2.0) * (1.0 - lbl_f) + gt_f * lbl_f
    return (F.binary_cross_entropy_with_logits(pred_f, gt_f, reduction="none") * weight).sum()


class YOLO27DetectionLoss(v8DetectionLoss):
    """Per-branch detection criterion with dual-scale STAL fix and FG-guided TAL assignment."""

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
            self._custom_class_weights: torch.Tensor | None = class_weights.to(
                device=self.device, dtype=torch.float32
            )
        else:
            self._custom_class_weights = None

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        fg_prior: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        """Compute assigned targets and per-component losses with optional FG-guided TAL."""
        if self._custom_class_weights is None and fg_prior is None:
            return super().get_assigned_targets_and_loss(preds, batch)

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
        assign_scores = pred_scores.detach().sigmoid()
        if fg_prior is not None:
            assign_scores = assign_scores * (0.75 + 0.25 * fg_prior.detach().to(dtype=assign_scores.dtype))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            assign_scores,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_raw = self.bce(pred_scores, target_scores.to(dtype))
        if self._custom_class_weights is not None:
            cls_raw = cls_raw * self._custom_class_weights.to(dtype=cls_raw.dtype, device=cls_raw.device).view(1, 1, -1)
        elif getattr(self, "class_weights", None) is not None:
            cls_raw = cls_raw * self.class_weights
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
        fg_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate batch-scaled branch loss scalar and formatted loss items."""
        if self._custom_class_weights is None and fg_prior is None:
            raw_loss, items = super().__call__(preds, batch)
            return raw_loss.sum(), items

        preds = preds[1] if isinstance(preds, tuple) else preds
        res = self.get_assigned_targets_and_loss(preds, batch, fg_prior=fg_prior)
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
        fg_gain: float = 0.08,
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

    def _compute_fg_alignment_loss(
        self,
        fg_logits: torch.Tensor,
        o2m_preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Supervise live neck foreground saliency logits using one2many IoU-quality targets."""
        loss_fn = self.one2many
        device = loss_fn.device
        dtype = fg_logits.dtype
        batch_size = fg_logits.shape[0]

        if batch["bboxes"].numel() == 0:
            fg_target = torch.zeros_like(fg_logits)
            fg_label = torch.zeros_like(fg_logits)
            return (_varifocal_sum(fg_logits, fg_target, fg_label) / max(batch_size, 1)) * self.fg_gain

        with torch.no_grad():
            pred_distri = o2m_preds["boxes"].detach().permute(0, 2, 1).contiguous()
            pred_scores = o2m_preds["scores"].detach().permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(o2m_preds["feats"], loss_fn.stride, 0.5)
            imgsz = torch.tensor(o2m_preds["feats"][0].shape[2:], device=device, dtype=dtype) * loss_fn.stride[0]

            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = loss_fn.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            pred_bboxes = loss_fn.bbox_decode(anchor_points, pred_distri)
            _, _, target_scores, fg_mask, _ = loss_fn.assigner(
                pred_scores.sigmoid(),
                (pred_bboxes * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            fg_quality = target_scores.amax(dim=-1, keepdim=True).clamp(0.0, 1.0).to(dtype)
            fg_label = fg_mask.unsqueeze(-1).to(dtype)
            norm = fg_quality.sum().clamp(min=1.0)

        return (_varifocal_sum(fg_logits, fg_quality, fg_label) / norm) * self.fg_gain

    def __call__(
        self, preds: Any, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate weighted ``one2many`` + FG-guided ``one2one`` + live neck FG alignment loss."""
        preds = preds[1] if isinstance(preds, tuple) else preds

        if not isinstance(preds, dict) or "one2many" not in preds:
            total_single, single_items = self.one2many(preds, batch)
            vec_single = self._loss_dict_or_tensor_to_vec(single_items)
            zero_fg = torch.zeros(1, device=vec_single.device, dtype=vec_single.dtype)
            vec4 = torch.cat([vec_single, zero_fg], dim=0)
            return total_single.sum(), (dict(zip(self.loss_names, vec4)) if self._returns_dict else vec4)

        one2many = preds["one2many"]
        one2one = preds["one2one"]
        fg_logits: torch.Tensor | None = preds.get("fg_align") if isinstance(preds, dict) else None
        fg_prior: torch.Tensor | None = fg_logits.detach().sigmoid() if fg_logits is not None else None

        loss_o2m, items_o2m = self.one2many(one2many, batch)
        loss_o2o, items_o2o = self.one2one(one2one, batch, fg_prior=fg_prior)

        vec_o2m = self._loss_dict_or_tensor_to_vec(items_o2m)
        vec_o2o = self._loss_dict_or_tensor_to_vec(items_o2o)
        det_items = vec_o2m * self.o2m + vec_o2o * self.o2o
        total_loss = loss_o2m.sum() * self.o2m + loss_o2o.sum() * self.o2o

        if fg_logits is not None:
            batch_size = fg_logits.shape[0]
            fg_loss_unscaled = self._compute_fg_alignment_loss(fg_logits, one2many, batch)
            total_loss = total_loss + fg_loss_unscaled * batch_size
            fg_item = fg_loss_unscaled.detach().reshape(1)
        else:
            fg_item = torch.zeros(1, device=det_items.device, dtype=det_items.dtype)

        all_items = torch.cat([det_items, fg_item], dim=0)
        if self._returns_dict:
            return total_loss, dict(zip(self.loss_names, all_items))
        return total_loss, all_items
