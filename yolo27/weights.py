"""Smart exact + partial-slice pretrained weight warm-start for YOLO27n / YOLO27n-Lite.

Transfers pretrained weights from ``yolo26n.pt`` (or ``yolo26s.pt`` / any compatible
Ultralytics checkpoint) into ``YOLO27DetectionModel``:
  1. Exact-shape tensors (stem Conv, P3/P5 C3k2, SPPF, C2PSA, etc.) are copied 1:1.
  2. Widened high-resolution stages (e.g. P2 C3k2 256->384, P4 C3k2 512->768,
     P3 fine head 256->320) receive subspace-sliced weights from the donor model,
     with C2f/C3k2 ``cv1`` chunk boundaries preserved and extra widened input
     channels scaled down (0.1x) so step-0 activations stay clean while gradients
     flow into all widened channels immediately.
  3. Dual-scale box regression branches (P3 idx 0 <- donor P3 idx 0;
     P5 idx 1 <- donor P5 idx 2) are warm-started while preserving the target
     dataset's 19-class focal prior biases on ``cv3`` / ``one2one_cv3``.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

import torch
from torch import nn

from ultralytics.utils import LOGGER, SETTINGS


def _copy_sliced_tensor(src: torch.Tensor, dst: torch.Tensor, key: str) -> int:
    """Copy overlapping channel subspace from ``src`` into ``dst`` and return copied element count."""
    if src.shape == dst.shape:
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        return dst.numel()

    if src.ndim != dst.ndim or src.ndim not in (1, 2, 4):
        return 0
    if src.ndim == 4 and src.shape[2:] != dst.shape[2:]:
        return 0

    src_cast = src.to(device=dst.device, dtype=dst.dtype)

    # Dampen newly widened weight channels so random noise does not swamp pretrained activations at epoch 1,
    # while keeping non-zero variance so symmetry is broken and all widened channels learn immediately.
    if dst.ndim >= 2:
        dst.mul_(0.1)

    # Special alignment for C2f / C3k2 cv1 which splits output channels into 2 equal halves via chunk(2, dim=1)
    if (
        ".cv1.conv.weight" in key
        and dst.ndim == 4
        and src.shape[0] % 2 == 0
        and dst.shape[0] % 2 == 0
        and src.shape[0] != dst.shape[0]
    ):
        s_half, d_half = src.shape[0] // 2, dst.shape[0] // 2
        c_half = min(s_half, d_half)
        c_in = min(src.shape[1], dst.shape[1])
        dst[:c_half, :c_in].copy_(src_cast[:c_half, :c_in])
        dst[d_half : d_half + c_half, :c_in].copy_(src_cast[s_half : s_half + c_half, :c_in])
        return 2 * c_half * c_in * dst.shape[2] * dst.shape[3]

    if (
        any(tag in key for tag in (".cv1.bn.weight", ".cv1.bn.bias", ".cv1.bn.running_mean", ".cv1.bn.running_var"))
        and dst.ndim == 1
        and src.shape[0] % 2 == 0
        and dst.shape[0] % 2 == 0
        and src.shape[0] != dst.shape[0]
    ):
        # Reset BN defaults on uncopied channels before copying the two aligned halves
        if key.endswith((".weight", ".running_var")):
            dst.fill_(1.0)
        else:
            dst.zero_()
        s_half, d_half = src.shape[0] // 2, dst.shape[0] // 2
        c_half = min(s_half, d_half)
        dst[:c_half].copy_(src_cast[:c_half])
        dst[d_half : d_half + c_half].copy_(src_cast[s_half : s_half + c_half])
        return 2 * c_half

    if dst.ndim == 1:
        if key.endswith((".weight", ".running_var")):
            dst.fill_(1.0)
        else:
            dst.zero_()

    slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape))
    dst[slices].copy_(src_cast[slices])
    copied = 1
    for sl in slices:
        copied *= sl.stop
    return copied


@torch.no_grad()
def transfer_pretrained_weights(donor: nn.Module | dict[str, torch.Tensor], target: nn.Module) -> dict[str, float]:
    """Transfer exact and channel-sliced pretrained weights from ``donor`` into ``target``."""
    if isinstance(donor, dict) and "model" in donor and isinstance(donor["model"], nn.Module):
        donor = donor["model"]
    donor_sd = donor if isinstance(donor, dict) else donor.float().state_dict()
    target_sd = target.state_dict()

    # Detect whether donor & target have identical class count on classification head
    head_idx = len(getattr(target, "model", [])) - 1
    donor_cls_key = f"model.{head_idx}.cv3.0.2.bias"
    target_cls_key = f"model.{head_idx}.cv3.0.2.bias"
    same_nc = (
        donor_cls_key in donor_sd
        and target_cls_key in target_sd
        and donor_sd[donor_cls_key].shape == target_sd[target_cls_key].shape
    )

    # Detect whether donor head has 3 scales (P3/P4/P5) while target has 2 scales (P3/P5)
    donor_has_3_scales = f"model.{head_idx}.cv2.2.0.conv.weight" in donor_sd
    target_has_2_scales = (
        f"model.{head_idx}.cv2.1.0.conv.weight" in target_sd
        and f"model.{head_idx}.cv2.2.0.conv.weight" not in target_sd
    )

    exact_tensors, sliced_tensors = 0, 0
    copied_numel = 0
    total_numel = sum(v.numel() for v in target_sd.values() if v.ndim >= 1)

    for key, dst in target_sd.items():
        if dst.ndim == 0:
            # Skip scalar num_batches_tracked buffers
            continue

        # Preserve target dataset focal bias initialization on final nc-class 1x1 projection layers,
        # while warm-starting the preceding DWConv + Conv spatial/channel feature towers (blocks 0 & 1)!
        if not same_nc and any(tag in key for tag in (".cv3.", ".one2one_cv3.")) and ".2." in key:
            continue
        if ".fg_branch." in key and key not in donor_sd:
            continue

        src_key = key
        if donor_has_3_scales and target_has_2_scales:
            p5_prefixes = (
                (f"model.{head_idx}.cv2.1.", f"model.{head_idx}.cv2.2."),
                (f"model.{head_idx}.one2one_cv2.1.", f"model.{head_idx}.one2one_cv2.2."),
                (f"model.{head_idx}.cv3.1.", f"model.{head_idx}.cv3.2."),
                (f"model.{head_idx}.one2one_cv3.1.", f"model.{head_idx}.one2one_cv3.2."),
            )
            for tgt_pref, src_pref in p5_prefixes:
                if key.startswith(tgt_pref):
                    src_key = key.replace(tgt_pref, src_pref, 1)
                    break

        src = donor_sd.get(src_key)
        if src is None:
            continue

        if src.shape == dst.shape:
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            exact_tensors += 1
            copied_numel += dst.numel()
        else:
            n_copied = _copy_sliced_tensor(src, dst, key)
            if n_copied > 0:
                sliced_tensors += 1
                copied_numel += n_copied

    ratio = copied_numel / max(total_numel, 1)
    LOGGER.info(
        f"YOLO27 Warm-Start: transferred {exact_tensors} exact + {sliced_tensors} sliced tensors "
        f"({copied_numel / 1e6:.2f}M / {total_numel / 1e6:.2f}M elements, {ratio * 100:.1f}%)"
    )
    return {
        "exact_tensors": exact_tensors,
        "sliced_tensors": sliced_tensors,
        "copied_numel": copied_numel,
        "total_numel": total_numel,
        "ratio": ratio,
    }


def resolve_donor_checkpoint(donor_name: str = "yolo26n.pt", allow_download: bool = True) -> Path | None:
    """Locate ``donor_name`` on disk or download quickly when network is reachable (never hangs offline)."""
    cand_paths = [
        Path(donor_name),
        Path.cwd() / donor_name,
        Path(SETTINGS.get("weights_dir", "")) / donor_name,
    ]
    for p in cand_paths:
        if p.is_file() and p.stat().st_size > 100_000:
            return p

    if not allow_download:
        return None

    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{Path(donor_name).name}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=3.0):
            pass
    except Exception:
        LOGGER.warning(
            f"YOLO27 Warm-Start: 未找到本地 '{donor_name}' 且当前无法直连 GitHub Releases，跳过预训练热启动。"
            f" 强烈建议将 {donor_name} 放入项目根目录以大幅加速前 20 轮 mAP 收敛！"
        )
        return None

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(donor_name))
        if downloaded.is_file() and downloaded.stat().st_size > 100_000:
            return downloaded
    except Exception as exc:
        LOGGER.warning(f"YOLO27 Warm-Start: 自动下载 {donor_name} 失败 ({exc})，回退至随机初始化。")
    return None
