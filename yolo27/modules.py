"""YOLO27 custom neural network modules aligned with official Ultralytics YOLO27 design.

Reference: https://docs.ultralytics.com/zh/models/yolo27

Key architectural elements implemented here:
    1. DFL-Free 1:1 Pretrained Regression Head (``reg_max=1``, ``c2=16``, ``c3=80``):
       Aligns box head width ``c2=16`` and cls head width ``c3=80`` 1:1 with
       ``yolo26n.pt`` so P4/P5 box & cls convs transfer 100% exact without injecting
       random Kaiming filters on few-shot (10~50 instances/class) fabric datasets,
       while preserving unconstrained coordinate span on full-frame/thin strip defects.
    2. Deterministic Folded Fixed Feature Scaling (Zero EMA/BN Drift):
       Uses purely deterministic fixed cross-scale factors (``(1.0, 1/sqrt(2))`` for
       dual-scale ``[P3, P5]``, ``(1.0, 1.0, 1.0)`` for 3-scale ``[P3, P4, P5]``) with no
       pre-BatchNorm learnable scalar parameters, guaranteeing 100% synchronization
       between ``ModelEMA`` weights and ``BatchNorm2d.running_var`` during validation.
       Folded algebraically into the first head ``Conv`` / ``DWConv`` on ``fuse()``.
    3. Background-Varied Foreground Alignment Auxiliary Branch (训练期启用，推理剥离):
       Enabled across ``YOLO27n``, ``YOLO27n-Dual``, and ``YOLO27n-Lite`` during training
       and completely stripped (``self.fg_branch = None``) on ``fuse()``. Synthesizes a
       counterfactual background-style/context-perturbed feature view while keeping
       defect centers anchored, supervising both clean/perturbed IoU-Varifocal saliency
       consistency and L2-normalized foreground defect embedding invariance
       (``背景发生改变，疵点不变，泛化能力增强``).
    4. Version-Safe Dual-Mode Fusion & Inference (``ultralytics 8.4.33`` .. ``>=8.4.197``):
       Defaults to ``one2many + NMS`` for maximum ``mAP50-95`` while retaining ``one2one``
       for optional ``nms=False`` end-to-end deployment.
"""

from __future__ import annotations

from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Detect


class ForegroundAlignmentBranch(nn.Module):
    """Training-only background-invariant foreground alignment auxiliary branch.

    Evaluates shared live neck features under both the original cloth background and a
    counterfactual cross-sample MixStyle + spatial-context perturbed background while
    anchoring defect centers. Uses buffer-free ``GroupNorm`` so dual-pass forward never
    perturbs BatchNorm running statistics or DDP buffer sync, and is completely stripped
    upon ``fuse()`` for zero inference parameters, FLOPs, or latency.
    """

    def __init__(self, ch: tuple[int, ...] | list[int], embed_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoders = nn.ModuleList()
        self.saliency_convs = nn.ModuleList()
        self.embed_projs = nn.ModuleList()

        for c in ch:
            mid = max(c // 2, 16)
            g_in = math.gcd(c, 8) or 1
            g_mid = math.gcd(mid, 8) or 1
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                    nn.GroupNorm(g_in, c),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(c, mid, kernel_size=1, bias=False),
                    nn.GroupNorm(g_mid, mid),
                    nn.SiLU(inplace=True),
                )
            )
            self.saliency_convs.append(nn.Conv2d(mid, 1, kernel_size=1))
            self.embed_projs.append(nn.Conv2d(mid, self.embed_dim, kernel_size=1, bias=False))

    def bias_init(self, prior_prob: float = 0.01) -> None:
        """Initialize foreground saliency logit bias with a focal-style prior probability."""
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        for conv in self.saliency_convs:
            if conv.bias is not None:
                nn.init.constant_(conv.bias, bias_val)

    @staticmethod
    def _perturb_background_features(feat: torch.Tensor, fg_prob_detached: torch.Tensor) -> torch.Tensor:
        """Perturb background channel style & spatial context while keeping foreground defect centers anchored."""
        bsz = feat.shape[0]
        fg_gate = fg_prob_detached.clamp(0.0, 1.0)
        bg_gate = 1.0 - fg_gate

        mu = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-5)
        feat_norm = (feat - mu) / std

        if bsz >= 2:
            lam = torch.empty((bsz, 1, 1, 1), device=feat.device, dtype=feat.dtype).uniform_(0.35, 0.75)
            mu_alt = (1.0 - lam) * mu + lam * torch.roll(mu, shifts=1, dims=0)
            std_alt = (1.0 - lam) * std + lam * torch.roll(std, shifts=1, dims=0)
        else:
            scale_j = torch.empty((1, 1, 1, 1), device=feat.device, dtype=feat.dtype).uniform_(0.80, 1.20)
            shift_j = torch.empty((1, 1, 1, 1), device=feat.device, dtype=feat.dtype).uniform_(-0.15, 0.15)
            mu_alt = mu + shift_j * std
            std_alt = std * scale_j

        ctx_shift = 0.20 * torch.roll((feat - mu) * bg_gate, shifts=(1, 1), dims=(2, 3))
        feat_bg_varied = feat_norm * std_alt + mu_alt + ctx_shift
        return fg_gate * feat + bg_gate * feat_bg_varied

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute clean & background-perturbed foreground logits and L2-normalized embeddings."""
        bsz = feats[0].shape[0]
        logits_list: list[torch.Tensor] = []
        pert_logits_list: list[torch.Tensor] = []
        embed_list: list[torch.Tensor] = []
        pert_embed_list: list[torch.Tensor] = []

        for enc, sal_conv, emb_proj, feat in zip(self.encoders, self.saliency_convs, self.embed_projs, feats):
            h_clean = enc(feat)
            s_clean = sal_conv(h_clean)
            z_clean = F.normalize(emb_proj(h_clean), dim=1, eps=1e-6)

            if self.training:
                feat_pert = self._perturb_background_features(feat, s_clean.detach().sigmoid())
                h_pert = enc(feat_pert)
                s_pert = sal_conv(h_pert)
                z_pert = F.normalize(emb_proj(h_pert), dim=1, eps=1e-6)
            else:
                s_pert = s_clean
                z_pert = z_clean

            logits_list.append(s_clean.view(bsz, 1, -1))
            pert_logits_list.append(s_pert.view(bsz, 1, -1))
            embed_list.append(z_clean.view(bsz, self.embed_dim, -1))
            pert_embed_list.append(z_pert.view(bsz, self.embed_dim, -1))

        return {
            "logits": torch.cat(logits_list, dim=2).permute(0, 2, 1),
            "pert_logits": torch.cat(pert_logits_list, dim=2).permute(0, 2, 1),
            "embed": torch.cat(embed_list, dim=2).permute(0, 2, 1),
            "pert_embed": torch.cat(pert_embed_list, dim=2).permute(0, 2, 1),
        }


class YOLO27Detect(Detect):
    """Official-style YOLO27 detection head with fixed scaling and background-invariant FG alignment.

    Attributes:
        c2_width (int): Box regression head channel width (``16``, 1:1 aligned with ``yolo26n.pt``).
        scale_factors (tuple[float, ...]): Deterministic cross-scale feature factors
            (``(1.0, 1/sqrt(2))`` for dual-scale ``[P3, P5]``, ``1.0`` per level for 3-scale).
        fg_branch (ForegroundAlignmentBranch | None): Training-only foreground alignment
            auxiliary branch active across all YOLO27 variants and stripped upon ``fuse()``.
    """

    def __init__(self, nc: int = 80, end2end: bool = True, reg_max: int = 1, ch: tuple = ()) -> None:
        super().__init__(nc=nc, end2end=end2end, reg_max=reg_max, ch=ch)

        # Lock c2=16 (when reg_max=1) 1:1 with yolo26n.pt even when ch[0]=80
        if len(ch) > 0:
            c2 = max(16, self.reg_max * 4)
            self.c2_width = c2
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )
            if end2end:
                self.one2one_cv2 = deepcopy(self.cv2)
        else:
            self.c2_width = 16

        # Official YOLO27 inference default: one-to-many + NMS (highest mAP50-95),
        # while keeping dual-head architecture ready for nms=False switching.
        if end2end:
            self._end2end = False

        if self.nl == 2:
            self.scale_factors: tuple[float, ...] = (1.0, round(1.0 / math.sqrt(2.0), 6))
        else:
            self.scale_factors = tuple(1.0 for _ in range(self.nl))

        self._scale_folded: bool = False
        # Enabled across ALL YOLO27 variants (YOLO27n / YOLO27n-Dual / YOLO27n-Lite) in training;
        # stripped to None on fuse() for zero inference cost.
        self.fg_branch: ForegroundAlignmentBranch | None = ForegroundAlignmentBranch(ch) if len(ch) > 0 else None

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
                if self.fg_branch is not None:
                    return {"one2many": preds, "fg_align": self.fg_branch(x_bal)}
                return preds
            y = self._inference(preds)
            return y if self.export else (y, preds)

        if self.training:
            one2many = self.forward_head(x_bal, **self.one2many)
            x_detach = [xi.detach() for xi in x_bal]
            one2one = self.forward_head(x_detach, **self.one2one)
            out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"one2many": one2many, "one2one": one2one}
            if self.fg_branch is not None:
                # Live x_bal (NOT detached!) so background-invariant FG alignment refines shared neck features
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
        """Idempotently fold fixed scales and strip auxiliary fg_branch + inactive head."""
        if not self._scale_folded:
            self._fold_scale_into_branch(self.cv2, self.cv3, self.scale_factors)
            self._fold_scale_into_branch(
                getattr(self, "one2one_cv2", None),
                getattr(self, "one2one_cv3", None),
                self.scale_factors,
            )
            self._scale_folded = True

        # Training-only Foreground Alignment Auxiliary Supervision is 100% stripped at inference!
        self.fg_branch = None
        # Do NOT call super().fuse(): ultralytics==8.4.33 Detect.fuse() unconditionally sets
        # self.cv2 = self.cv3 = None even when self.end2end is False!
        if self.end2end:
            self.cv2 = None
            self.cv3 = None
        else:
            self.one2one_cv2 = None
            self.one2one_cv3 = None
