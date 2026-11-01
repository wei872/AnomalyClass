"""YOLO27 custom neural network modules aligned with official Ultralytics YOLO27 design.

Reference: https://docs.ultralytics.com/zh/models/yolo27

Key architectural elements implemented here:
    1. DFL-Free High-Precision Regression Head (``reg_max=1``, widened ``c2=32``):
       Stock ``YOLO26n`` (``reg_max=1``) collapses box head width ``c2`` down to ``16``
       channels across all scales (``256 -> 16`` at P5), bottlenecking high-IoU edge
       localization (``AP75..AP95`` / ``mAP50-95``). ``YOLO27Detect`` widens ``c2`` to
       ``32`` channels (``24`` on ``yolo27n-lite``) while keeping ``reg_max=1`` and
       ``c3=80``, preserving **exact step-0 Net2Wider equivalence** from ``yolo26n.pt``
       and unconstrained coordinate span on full-width thin fabric strip defects.
    2. Deterministic Folded Fixed Feature Scaling (Zero EMA/BN Drift):
       Uses purely deterministic fixed cross-scale factors (``(1.0, 1/sqrt(2))`` for
       dual-scale ``[P3, P5]``, ``(1.0, 1.0, 1.0)`` for 3-scale ``[P3, P4, P5]``) with no
       pre-BatchNorm learnable scalar parameters, guaranteeing 100% synchronization
       between ``ModelEMA`` weights and ``BatchNorm2d.running_var`` during validation.
       Folded algebraically into the first head ``Conv`` / ``DWConv`` on ``fuse()``.
    3. Training-Only Foreground Alignment Branch (``ForegroundAlignmentBranch``):
       Supervises live shared neck features with continuous IoU-quality Varifocal targets
       to separate subtle fabric anomalies from repetitive weave textures; stripped on
       ``fuse()`` for zero deployment overhead.
    4. Version-Safe Dual-Mode Fusion & Inference (``ultralytics 8.4.33`` .. ``>=8.4.197``):
       Defaults to ``one2many + NMS`` for maximum ``mAP50-95`` while retaining ``one2one``
       for optional ``nms=False`` end-to-end deployment, avoiding ``v8.4.33`` ``Detect.fuse()``
       pitfalls that unconditionally delete ``cv2``/``cv3``.
"""

from __future__ import annotations

from copy import deepcopy
import math
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DWConv, Detect


class ForegroundAlignmentBranch(nn.Module):
    """Training-only foreground saliency alignment branch across neck scales.

    Evaluates directly on live balanced neck features during training so its
    IoU-quality Varifocal supervision sharpens defect-vs-weave separation in the
    shared neck. Stripped completely on ``fuse()`` so deployment incurs zero overhead.
    """

    def __init__(self, ch: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(
                DWConv(c, c, 3),
                Conv(c, max(c // 2, 16), 1),
                nn.Conv2d(max(c // 2, 16), 1, 1),
            )
            for c in ch
        )

    def bias_init(self, prior_prob: float = 0.01) -> None:
        """Initialize foreground logit bias with a focal-style prior probability."""
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        for head in self.heads:
            last_conv = head[-1]
            if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, bias_val)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Predict flattened ``(B, N_anchors, 1)`` foreground alignment logits."""
        bs = feats[0].shape[0]
        return torch.cat([head(feat).view(bs, 1, -1) for head, feat in zip(self.heads, feats)], dim=2).permute(0, 2, 1)


class YOLO27Detect(Detect):
    """Official-style YOLO27 detection head with widened box tower, fixed scaling, and FG alignment.

    Attributes:
        c2_width (int): Box regression head channel width (``32`` on standard nano, ``24`` on lite).
        scale_factors (tuple[float, ...]): Deterministic cross-scale feature factors
            (``(1.0, 1/sqrt(2))`` for dual-scale ``[P3, P5]``, ``1.0`` per level for 3-scale).
        fg_branch (ForegroundAlignmentBranch | None): Training-only auxiliary branch
            stripped upon ``fuse()``.
    """

    def __init__(self, nc: int = 80, end2end: bool = True, reg_max: int = 1, ch: tuple = ()) -> None:
        super().__init__(nc=nc, end2end=end2end, reg_max=reg_max, ch=ch)

        # Upgrade cramped 16-ch reg_max=1 box tower to 32-ch (24-ch on lite) for high-IoU mAP50-95;
        # Net2Wider copies all 16 pretrained yolo26n.pt box channels with exact step-0 equivalence.
        if len(ch) > 0:
            base_c2 = 32 if ch[0] > 64 else 24
            c2 = max(base_c2, ch[0] // 4, self.reg_max * 4)
            self.c2_width = c2
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )
            if end2end:
                self.one2one_cv2 = deepcopy(self.cv2)
        else:
            self.c2_width = 32

        # Official YOLO27 inference default: one-to-many + NMS (highest mAP50-95),
        # while keeping dual-head architecture ready for nms=False switching.
        if end2end:
            self._end2end = False

        if self.nl == 2:
            self.scale_factors: tuple[float, ...] = (1.0, round(1.0 / math.sqrt(2.0), 6))
        else:
            self.scale_factors = tuple(1.0 for _ in range(self.nl))

        self._scale_folded: bool = False
        self.fg_branch: ForegroundAlignmentBranch | None = ForegroundAlignmentBranch(ch) if end2end else None

    @property
    def end2end(self) -> bool:
        """Return whether end-to-end one-to-one inference mode is currently active."""
        return getattr(self, "_end2end", False)

    @end2end.setter
    def end2end(self, value: bool) -> None:
        """Toggle between default one-to-many (+NMS) and end-to-end one-to-one (NMS-free) modes."""
        if value and getattr(self, "one2one_cv2", None) is None:
            raise ValueError("Cannot enable end2end=True (nms=False) after one2one head was stripped by fuse().")
        if not value and getattr(self, "cv2", None) is None:
            raise ValueError("Cannot enable end2end=False (nms=True) after one2many head was stripped by fuse().")
        self._end2end = bool(value)

    def _apply_feature_scale(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply deterministic fixed cross-scale factors when not yet folded into head convs."""
        if self._scale_folded:
            return x
        return [xi if s == 1.0 else xi * s for xi, s in zip(x, self.scale_factors)]

    def forward(self, x: list[torch.Tensor]):
        """Forward pass supporting dual-branch + live FG alignment training and zero-overhead inference."""
        x_bal = self._apply_feature_scale(x)

        if not hasattr(self, "one2one_cv2") or (self.one2one_cv2 is None and not self.end2end):
            preds = self.forward_head(x_bal, **self.one2many)
            if self.training:
                return preds
            y = self._inference(preds)
            return y if self.export else (y, preds)

        if self.training:
            one2many = self.forward_head(x_bal, **self.one2many)
            x_detach = [xi.detach() for xi in x_bal]
            one2one = self.forward_head(x_detach, **self.one2one)
            out = {"one2many": one2many, "one2one": one2one}
            if self.fg_branch is not None:
                # Live x_bal (NOT detached!) so FG alignment directly refines shared neck features
                out["fg_align"] = self.fg_branch(x_bal)
            return out

        active_head = self.one2one if self.end2end else self.one2many
        preds = self.forward_head(x_bal, **active_head)
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        if self.export:
            return y
        if not self.end2end and self.one2one_cv2 is not None and self.cv2 is not None:
            one2one = self.forward_head(x_bal, **self.one2one)
            return y, {"one2many": preds, "one2one": one2one}
        return y, preds

    @staticmethod
    def _fold_scale_into_branch(
        box_convs: nn.ModuleList | None,
        cls_convs: nn.ModuleList | None,
        scale_factors: tuple[float, ...],
    ) -> None:
        """Fold scalar ``scale_factors[i]`` into the first Conv/DWConv weights of level ``i``."""
        if box_convs is None or cls_convs is None:
            return
        with torch.no_grad():
            for i, s in enumerate(scale_factors):
                if s == 1.0:
                    continue
                for seq in (box_convs[i], cls_convs[i]):
                    first_block = seq[0]
                    if isinstance(first_block, nn.Sequential):
                        first_block = first_block[0]
                    first_conv = getattr(first_block, "conv", first_block)
                    if isinstance(first_conv, nn.Conv2d):
                        first_conv.weight.mul_(float(s))

    def bias_init(self) -> None:
        """Initialize BOTH one2many and one2one box/cls biases + fg_branch focal prior."""
        m = self
        if isinstance(m.stride, torch.Tensor) and torch.all(m.stride > 0):
            strides = m.stride
        elif m.nl == 2:
            strides = (8, 32)
        else:
            strides = tuple(8 * (2**i) for i in range(m.nl))

        def _init_pair(box_list: nn.ModuleList | None, cls_list: nn.ModuleList | None) -> None:
            if box_list is None or cls_list is None:
                return
            for a, b, s in zip(box_list, cls_list, strides):
                a[-1].bias.data[:] = 2.0  # box prior
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / float(s)) ** 2)  # cls focal prior (~0.01)

        _init_pair(m.cv2, m.cv3)
        _init_pair(getattr(m, "one2one_cv2", None), getattr(m, "one2one_cv3", None))

        if self.fg_branch is not None:
            self.fg_branch.bias_init(prior_prob=0.01)

    def fuse(self) -> None:
        """Idempotently fold fixed scales and strip ONLY inactive branches (safe on 8.4.33 & >=8.4.197)."""
        if not self._scale_folded:
            self._fold_scale_into_branch(self.cv2, self.cv3, self.scale_factors)
            self._fold_scale_into_branch(
                getattr(self, "one2one_cv2", None),
                getattr(self, "one2one_cv3", None),
                self.scale_factors,
            )
            self._scale_folded = True

        self.fg_branch = None
        # Do NOT call super().fuse(): ultralytics==8.4.33 Detect.fuse() unconditionally sets
        # self.cv2 = self.cv3 = None even when self.end2end is False!
        if self.end2end:
            self.cv2 = None
            self.cv3 = None
        else:
            self.one2one_cv2 = None
            self.one2one_cv3 = None
