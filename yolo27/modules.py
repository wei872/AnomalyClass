"""YOLO27n custom detection head, warp/weft strip enhancer, and foreground alignment branch."""

from __future__ import annotations

import math

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class FabricStripEnhancer(nn.Module):
    """Zero-initialized orthogonal strip-convolution residual block (1xK + Kx1) for woven fabric defects.

    Captures directional horizontal weft (纬线: 纬斜/断纬/纬线不直) and vertical warp (经线: 稀缝/油点线/多股线)
    context with wide directional receptive fields while adding negligible parameters (~35K total).
    Because the final pointwise BatchNorm scale ``gamma`` is initialized to ``0.0``, at step 0 this block
    is an exact identity ``y = x`` (preserving 100% of ``yolo26n.pt`` pretrained activations), while step-1
    gradients immediately unlock directional strip features.
    """

    def __init__(self, c: int, k: int = 9) -> None:
        super().__init__()
        p = k // 2
        self.dw_h = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, p), groups=c, bias=False)
        self.dw_v = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(p, 0), groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.pw = Conv(c, c, 1, act=False)
        self.reset_zero_residual()

    def reset_zero_residual(self) -> None:
        """Initialize residual projection BN weight & bias to zero so step-0 output equals input ``x``."""
        if hasattr(self.pw, "bn"):
            nn.init.zeros_(self.pw.bn.weight)
            nn.init.zeros_(self.pw.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.dw_h(x) + self.dw_v(x)))
        return x + self.pw(h)


class ForegroundAlignmentBranch(nn.Module):
    """Training-only lightweight per-location foreground/background discrimination branch.

    Predicts a 1-channel foreground objectness logit at every spatial anchor across the
    detection feature maps. Stripped completely during model fusion/export so it adds
    zero parameters and zero FLOPs at deployment.
    """

    def __init__(self, ch: list[int] | tuple[int, ...]) -> None:
        super().__init__()
        c_fg = max(16, ch[0] // 4)
        self.heads = nn.ModuleList(
            nn.Sequential(
                DWConv(c, c, 3),
                Conv(c, c_fg, 1),
                nn.Conv2d(c_fg, 1, 1),
            )
            for c in ch
        )

    def bias_init(self, prior_prob: float = 0.01) -> None:
        """Initialize final 1x1 conv bias with focal foreground prior probability."""
        bias_val = math.log(prior_prob / (1.0 - prior_prob))
        for head in self.heads:
            head[-1].bias.data.fill_(bias_val)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Return concatenated foreground logits of shape (B, 1, total_anchors)."""
        bs = feats[0].shape[0]
        return torch.cat([head(x).view(bs, 1, -1) for head, x in zip(self.heads, feats)], dim=-1)


class YOLO27Detect(Detect):
    """YOLO27 detection head with fixed feature scaling, strip-context residual, and foreground branch."""

    def __init__(
        self,
        nc: int = 19,
        reg_max: int = 1,
        end2end: bool = True,
        ch: list[int] | tuple[int, ...] = (),
        scale_factors: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if scale_factors is None:
            if self.nl == 2:
                scale_factors = (1.0, 1.0 / math.sqrt(2.0))
            else:
                scale_factors = tuple(1.0 / math.sqrt(1.0 + 0.5 * i) for i in range(self.nl))
        if len(scale_factors) != self.nl:
            raise ValueError(f"Expected {self.nl} scale_factors for {self.nl} detection scales, got {scale_factors}")
        self.scale_factors: tuple[float, ...] = tuple(float(s) for s in scale_factors)
        self._scale_folded: bool = False
        self.strip_adapters: nn.ModuleList | None = nn.ModuleList(FabricStripEnhancer(c, k=9) for c in ch)
        self.fg_branch: ForegroundAlignmentBranch | None = ForegroundAlignmentBranch(ch) if end2end else None

    @classmethod
    def from_detect(
        cls,
        det: Detect,
        scale_factors: tuple[float, ...] | None = None,
    ) -> YOLO27Detect:
        """Upgrade a parsed Ultralytics ``Detect`` head into ``YOLO27Detect`` while preserving metadata."""
        ch = tuple(branch[0].conv.in_channels for branch in det.cv2)
        end2end = getattr(det, "one2one_cv2", None) is not None
        cls.legacy = det.legacy
        new_head = cls(
            nc=det.nc,
            reg_max=det.reg_max,
            end2end=end2end,
            ch=ch,
            scale_factors=scale_factors,
        )
        for attr in ("i", "f", "type", "inplace"):
            if hasattr(det, attr):
                setattr(new_head, attr, getattr(det, attr))
        new_head.np = sum(p.numel() for p in new_head.parameters())
        return new_head

    def bias_init(self) -> None:
        """Initialize box, class, foreground alignment biases, and zero-residual strip adapters."""
        super().bias_init()
        if getattr(self, "fg_branch", None) is not None:
            self.fg_branch.bias_init(prior_prob=0.01)
        if getattr(self, "strip_adapters", None) is not None:
            for adapter in self.strip_adapters:
                adapter.reset_zero_residual()

    def _apply_fixed_scale(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply fixed multi-scale feature balancing when not yet folded into conv weights."""
        if self._scale_folded:
            return x
        return [xi * s if s != 1.0 else xi for xi, s in zip(x, self.scale_factors)]

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run orthogonal strip enhancement, fixed feature scaling, and dual-pathway detection head."""
        strip_adapters = getattr(self, "strip_adapters", None)
        if strip_adapters is not None:
            x = [adapter(xi) for adapter, xi in zip(strip_adapters, x)]

        x_bal = self._apply_fixed_scale(x)

        preds = self.forward_head(x_bal, **self.one2many)
        if getattr(self, "one2one_cv2", None) is not None:
            x_detach = [xi.detach() for xi in x_bal] if self.training else x_bal
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        if getattr(self, "fg_branch", None) is not None and not self.export and isinstance(preds, dict):
            preds["fg_logits"] = self.fg_branch(x_bal)

        if self.training:
            return preds

        infer_src = preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds
        y = self._inference(infer_src)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    @staticmethod
    def _fold_scale_into_first_conv(branch_list: nn.ModuleList | None, scale_factors: tuple[float, ...]) -> None:
        """Fold fixed input feature scaling ``s_i`` into the first ``nn.Conv2d`` weight of each scale branch."""
        if branch_list is None:
            return
        for branch, s in zip(branch_list, scale_factors):
            if s == 1.0:
                continue
            first_conv = next(m for m in branch.modules() if isinstance(m, nn.Conv2d))
            first_conv.weight.data.mul_(s)

    def fuse(self) -> None:
        """Strip training-only foreground branch, remove unused detection head, and fold fixed feature scales."""
        self.fg_branch = None

        if self.end2end:
            self.cv2 = self.cv3 = None
        else:
            self.one2one_cv2 = self.one2one_cv3 = None

        if not self._scale_folded:
            self._fold_scale_into_first_conv(self.cv2, self.scale_factors)
            self._fold_scale_into_first_conv(self.cv3, self.scale_factors)
            self._fold_scale_into_first_conv(getattr(self, "one2one_cv2", None), self.scale_factors)
            self._fold_scale_into_first_conv(getattr(self, "one2one_cv3", None), self.scale_factors)
            self._scale_folded = True
