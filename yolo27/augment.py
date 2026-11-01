"""Training-only Defect-Invariant Background Perturbation (背景发生改变，疵点不变，泛化能力增强).

Constructs paired counterfactual cloth images ``img_pert`` from ``(imgs, batch_idx, bboxes)``
for YOLO27's Foreground Alignment Auxiliary Supervision:
  - **疵点不变 (Defect Invariant)**: Every pixel inside every Ground-Truth defect bounding box
    (``fg_exact == 1``) is preserved bit-for-bit (``max_abs_diff == 0.0``).
  - **背景发生改变 (Background Changed)**:
      1. Donor defect regions are first hole-filled with spatially shifted clean cloth weave
         from the same roll so cross-sample weave swapping never introduces flat/blurry
         ghost-box artifacts.
      2. Multi-band zero-mean cloth yarn residuals (``h_fine``, ``h_mid``) are swapped across
         batch samples with random flip/transpose grain orientations, modulated in weave
         contrast, and paired with a smooth ambient illumination gradient strictly outside
         the ``17 px`` feathered defect boundary.
  - **全局纹理保护**: Full-frame global anomalies (e.g. ``纬斜`` with ``w * h >= 0.55``) are
    automatically excluded so whole-cloth warp/weft skew cues are never distorted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def apply_defect_invariant_bg_augment(
    imgs: torch.Tensor,
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    prob: float = 1.0,
    full_frame_area_thr: float = 0.55,
    *,
    p: float | None = None,
) -> torch.Tensor:
    """Perturb fabric background weave & lighting while keeping GT defect pixels 100% invariant.

    Args:
        imgs: Normalized ``(B, C, H, W)`` float tensor in ``[0, 1]``.
        batch_idx: ``(N_gt,)`` tensor mapping each GT box to sample index ``0 .. B-1``.
        bboxes: ``(N_gt, 4)`` normalized ``[cx, cy, w, h]`` boxes in ``[0, 1]``.
        prob: Per-image probability of applying background perturbation.
        full_frame_area_thr: Skip samples containing a GT box whose normalized area
            ``w * h >= full_frame_area_thr`` (e.g. full-image ``纬斜`` weft skew).
        p: Optional alias for ``prob``.

    Returns:
        Counterfactual ``(B, C, H, W)`` float tensor in ``[0, 1]`` with altered cloth
        background and bit-identical pixels inside all GT defect boxes.
    """
    if p is not None:
        prob = float(p)
    if prob <= 0.0 or imgs.ndim != 4 or imgs.shape[0] == 0:
        return imgs

    bsz, _, height, width = imgs.shape
    device = imgs.device
    dtype = imgs.dtype

    fg_exact = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
    fg_dilated = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
    skip_sample = torch.zeros(bsz, device=device, dtype=torch.bool)

    margin_y = max(2, round(height * 0.008))
    margin_x = max(2, round(width * 0.008))

    if bboxes is not None and bboxes.numel() > 0:
        b_indices = batch_idx.view(-1).long()
        boxes = bboxes.view(-1, 4)
        for bi, box in zip(b_indices.tolist(), boxes.tolist()):
            if bi < 0 or bi >= bsz:
                continue
            cx, cy, bw, bh = box
            if bw * bh >= full_frame_area_thr:
                skip_sample[bi] = True
                continue
            x1 = max(0, int(round((cx - 0.5 * bw) * width)))
            y1 = max(0, int(round((cy - 0.5 * bh) * height)))
            x2 = min(width, max(x1 + 1, int(round((cx + 0.5 * bw) * width))))
            y2 = min(height, max(y1 + 1, int(round((cy + 0.5 * bh) * height))))
            fg_exact[bi, 0, y1:y2, x1:x2] = 1.0

            dx1 = max(0, x1 - margin_x)
            dy1 = max(0, y1 - margin_y)
            dx2 = min(width, x2 + margin_x)
            dy2 = min(height, y2 + margin_y)
            fg_dilated[bi, 0, dy1:dy2, dx1:dx2] = 1.0

    if prob >= 1.0:
        apply_mask = ~skip_sample
    else:
        apply_mask = (torch.rand(bsz, device=device) < prob) & (~skip_sample)
    if not bool(apply_mask.any()):
        return imgs

    # Two-stage 9x9 feather ramp outside dilated box while clamping exact GT box interior to 1.0
    fg_smooth = F.avg_pool2d(fg_dilated, kernel_size=9, stride=1, padding=4)
    fg_smooth = F.avg_pool2d(fg_smooth, kernel_size=9, stride=1, padding=4)
    fg_mask = torch.maximum(fg_exact, fg_smooth)
    bg_mask = (1.0 - fg_mask) * apply_mask.to(dtype=dtype).view(bsz, 1, 1, 1)

    # Multi-band zero-mean weave decomposition (preserves local luminance continuity across borders)
    low_fine = F.avg_pool2d(imgs, kernel_size=7, stride=1, padding=3)
    low_mid = F.avg_pool2d(imgs, kernel_size=23, stride=1, padding=11)
    h_fine = imgs - low_fine
    h_mid = low_fine - low_mid

    # Inpaint donor defect regions with shifted clean cloth weave BEFORE cross-sample swapping
    # so swapped weave never carries flat zero-texture holes shaped like donor bounding boxes.
    clean_gate = 1.0 - fg_dilated
    shift_y = max(1, height // 3)
    shift_x = max(1, width // 3)
    alt_fine = torch.roll(h_fine, shifts=(shift_y, shift_x), dims=(2, 3))
    alt_mid = torch.roll(h_mid, shifts=(shift_y, shift_x), dims=(2, 3))
    alt_gate = torch.roll(clean_gate, shifts=(shift_y, shift_x), dims=(2, 3))
    pure_fine = torch.where(clean_gate > 0.5, h_fine, torch.where(alt_gate > 0.5, alt_fine, torch.zeros_like(h_fine)))
    pure_mid = torch.where(clean_gate > 0.5, h_mid, torch.where(alt_gate > 0.5, alt_mid, torch.zeros_like(h_mid)))

    if bsz >= 2:
        donor_fine = torch.roll(pure_fine, shifts=1, dims=0)
        donor_mid = torch.roll(pure_mid, shifts=1, dims=0)
    else:
        donor_fine = torch.roll(pure_fine, shifts=(shift_y, shift_x), dims=(2, 3))
        donor_mid = torch.roll(pure_mid, shifts=(shift_y, shift_x), dims=(2, 3))

    # Randomize donor weave grain orientation so background yarn patterns genuinely diversify
    if bool(torch.rand((), device=device) < 0.5):
        donor_fine = donor_fine.flip(dims=[3])
        donor_mid = donor_mid.flip(dims=[3])
    if bool(torch.rand((), device=device) < 0.5):
        donor_fine = donor_fine.flip(dims=[2])
        donor_mid = donor_mid.flip(dims=[2])
    if height == width and bool(torch.rand((), device=device) < 0.5):
        donor_fine = donor_fine.transpose(2, 3)
        donor_mid = donor_mid.transpose(2, 3)

    alpha = torch.empty((bsz, 1, 1, 1), device=device, dtype=dtype).uniform_(0.40, 0.75)
    beta = torch.empty((bsz, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.30, 0.35)

    # Smooth ambient LED-bar illumination ramp strictly outside feathered defect boundaries
    gx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
    gy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
    sx = torch.empty((bsz, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.05, 0.05)
    sy = torch.empty((bsz, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.05, 0.05)
    illum_shift = sx * gx + sy * gy

    delta_bg = (
        alpha * (donor_fine - h_fine)
        + (0.5 * alpha) * (donor_mid - h_mid)
        + beta * (h_fine + 0.5 * h_mid)
        + illum_shift
    )

    out = (imgs + bg_mask * delta_bg).clamp(0.0, 1.0)
    # Enforce exact floating-point bit identity inside fg_exact (疵点不变)
    return torch.where(fg_exact > 0.5, imgs, out)
