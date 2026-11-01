"""YOLO27 custom neural network modules aligned with official Ultralytics YOLO27 design.

Reference: https://docs.ultralytics.com/zh/models/yolo27

Key architectural elements implemented here:
    1. DFL-Free Unconstrained Regression Head (``reg_max=1`` default):
       Preserves 100% of ``yolo26n.pt``'s pretrained box regression head at step 0
       and removes DFL-16's ``15 * stride = 240 px`` coordinate ceiling on ``P3/8``,
       enabling full-width thin fabric strip defects to be regressed cleanly.
    2. Folded Multi-Scale & Per-Channel Feature Calibration:
       Combines fixed balanced cross-scale factors (``(1.0, 1/sqrt(2))`` for dual-scale
       ``[P3, P5]``, ``(1.0, 1.0, 1.0)`` for 3-scale ``[P3, P4, P5]``) with learnable
       per-channel calibration gates (``feat_logit_scale``, placed in the no-decay
       non-Muon optimizer group). On ``fuse()``, the effective per-channel scale is
       folded algebraically into the first ``Conv`` / ``DWConv`` of ``cv2`` and ``cv3``
       with zero extra inference parameters or FLOPs.
    3. Training-Only Foreground Alignment Branch (``ForegroundAlignmentBranch``):
       Supervises shared neck features directly with IoU-aware Varifocal foreground
       quality targets to separate subtle fabric anomalies from repetitive weave
       textures and guide one-to-one positive anchor assignment during training.
       Stripped completely on ``fuse()`` / deployment.
    4. Dual-Mode Inference (One-to-Many + NMS Default, Optional End-to-End ``nms=False``):
       Follows official YOLO27 semantics: defaults to the high-recall ``one2many`` head
       with NMS, while initializing and training both branches cleanly so ``nms=False``
       switches to end-to-end ``one2one`` inference seamlessly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DWConv, Detect


class ForegroundAlignmentBranch(nn.Module):
    """Training-only foreground saliency alignment branch across neck scales.

    Evaluates directly on live balanced neck features during training so its
    IoU-quality Varifocal supervision sharpens defect-vs-weave separation in the
    shared neck AND provides a clean foreground prior for one-to-one anchor
    assignment. Stripped on ``fuse()`` so deployment incurs zero overhead.
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
    """Official-style YOLO27 detection head with folded feature calibration and FG alignment.

    Attributes:
        scale_factors (tuple[float, ...]): Fixed cross-scale feature factors
            (``(1.0, 1/sqrt(2))`` for dual-scale ``[P3, P5]``, ``1.0`` per level for 3-scale).
        feat_logit_scale (nn.ParameterList | None): Learnable per-channel feature
            calibration gates (named ``*logit_scale*`` and 1D so Ultralytics' optimizer
            builder disables weight decay and Muon orthogonalization). Folded into first
            head convolutions and removed on ``fuse()``.
        fg_branch (ForegroundAlignmentBranch | None): Training-only auxiliary branch
            stripped upon ``fuse()``.
    """

    def __init__(self, nc: int = 80, end2end: bool = True, reg_max: int = 1, ch: tuple = ()) -> None:
        super().__init__(nc=nc, end2end=end2end, reg_max=reg_max, ch=ch)
        # Official YOLO27 inference default: one-to-many + NMS (highest mAP),
        # while keeping dual-head architecture ready for nms=False switching.
        if end2end:
            self._end2end = False

        if self.nl == 2:
            self.scale_factors: tuple[float, ...] = (1.0, round(1.0 / math.sqrt(2.0), 6))
        else:
            self.scale_factors = tuple(1.0 for _ in range(self.nl))

        self.feat_logit_scale: nn.ParameterList | None = nn.ParameterList(
            nn.Parameter(torch.ones(c, dtype=torch.float32)) for c in ch
        )
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

    def _effective_channel_scale(self, idx: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Compute clamped per-channel scale ``(C,)`` for level ``idx``."""
        base = float(self.scale_factors[idx])
        if self.feat_logit_scale is None:
            return torch.tensor(base, dtype=dtype, device=device)
        return (self.feat_logit_scale[idx].clamp(0.25, 4.0) * base).to(dtype=dtype, device=device)

    def _apply_feature_scale(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply fixed + learnable per-channel calibration when not yet folded into head convs."""
        if self._scale_folded:
            return x
        out: list[torch.Tensor] = []
        for i, xi in enumerate(x):
            scale_vec = self._effective_channel_scale(i, dtype=xi.dtype, device=xi.device)
            if scale_vec.ndim == 0:
                out.append(xi if scale_vec.item() == 1.0 else xi * scale_vec)
            else:
                out.append(xi * scale_vec.view(1, -1, 1, 1))
        return out

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
        scale_vecs: list[torch.Tensor],
    ) -> None:
        """Fold per-channel ``scale_vecs[i]`` into the first Conv/DWConv weights of level ``i``."""
        if box_convs is None or cls_convs is None:
            return
        with torch.no_grad():
            for i, s_vec in enumerate(scale_vecs):
                for seq in (box_convs[i], cls_convs[i]):
                    first_block = seq[0]
                    # In Detect.cv3, seq[0] is Sequential(DWConv, Conv); unwrap to the first Conv/DWConv block
                    if isinstance(first_block, nn.Sequential):
                        first_block = first_block[0]
                    first_conv = getattr(first_block, "conv", first_block)
                    if isinstance(first_conv, nn.Conv2d):
                        s = s_vec.to(dtype=first_conv.weight.dtype, device=first_conv.weight.device)
                        if s.ndim == 0:
                            if s.item() != 1.0:
                                first_conv.weight.mul_(s)
                        elif first_conv.groups == first_conv.in_channels:
                            # Depthwise conv: weight shape is (C_in, 1, kH, kW)
                            first_conv.weight.mul_(s.view(-1, 1, 1, 1))
                        else:
                            # Standard conv: weight shape is (C_out, C_in, kH, kW)
                            first_conv.weight.mul_(s.view(1, -1, 1, 1))

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
        """Fold per-channel feature calibration into head convs and strip auxiliary branches."""
        if not self._scale_folded:
            ref_param = next(self.parameters())
            scale_vecs = [
                self._effective_channel_scale(i, dtype=ref_param.dtype, device=ref_param.device).detach()
                for i in range(self.nl)
            ]
            self._fold_scale_into_branch(self.cv2, self.cv3, scale_vecs)
            self._fold_scale_into_branch(
                getattr(self, "one2one_cv2", None),
                getattr(self, "one2one_cv3", None),
                scale_vecs,
            )
            self.feat_logit_scale = None
            self._scale_folded = True

        self.fg_branch = None
        super().fuse()
