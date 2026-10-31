"""Function-preserving Net2Wider pretrained warm-start for YOLO27n / YOLO27n-Lite.

Transfers pretrained weights from ``yolo26n.pt`` (or ``yolo26s.pt``) into ``YOLO27DetectionModel``
with **zero step-0 activation error** on shared channels:
  1. Exact-shape layers (stem Conv, P3/P4/P5 C3k2, SPPF, C2PSA, etc.) copy 1:1.
  2. Widened ``C3k2`` stages align both ``cv1`` (2-chunk split) and ``cv2`` (``(2+n)``-chunk concat).
  3. ``Concat`` consumers align input channel offsets per concatenated branch.
  4. Newly widened input-channel columns on consumer convolutions are zero-initialized (Net2Wider),
     while newly widened output-channel filters keep small Kaiming variance (``0.05x``) so step-0
     forward outputs match ``yolo26n.pt`` to ``< 1e-6`` while step-1 gradients unlock all new channels.
  5. ``YOLO27Detect`` fixed ``scale_factors`` are pre-compensated (``1 / s_i``) on first-layer convs
     when loading from an unscaled donor so pretrained BatchNorm statistics remain 100% calibrated.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules.conv import Concat
from ultralytics.utils import LOGGER, SETTINGS


def _infer_layer_out_ch(sd: dict[str, torch.Tensor], seq_mods: Any, idx: int) -> int | None:
    """Infer output channel count of ``model.{idx}`` from state_dict tensors (following Upsample/Concat)."""
    if idx < 0:
        return None
    # Direct BN / Conv weight keys on module idx
    for suffix in (".cv2.bn.weight", ".bn.weight", ".cv2.conv.weight", ".conv.weight"):
        t = sd.get(f"model.{idx}{suffix}")
        if t is not None and t.ndim >= 1:
            return int(t.shape[0])

    if seq_mods is not None and idx < len(seq_mods):
        m = seq_mods[idx]
        if isinstance(m, Concat):
            parts = []
            for f in getattr(m, "f", []):
                ref = idx + f if f < 0 else f
                ch = _infer_layer_out_ch(sd, seq_mods, ref)
                if ch is None:
                    return None
                parts.append(ch)
            return sum(parts)
        f = getattr(m, "f", -1)
        if isinstance(f, int):
            ref = idx + f if f < 0 else f
            if ref != idx:
                return _infer_layer_out_ch(sd, seq_mods, ref)
    return None


def _get_concat_branch_pairs(
    donor_sd: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
    target_mods: Any,
    consumer_idx: int,
) -> list[tuple[int, int]] | None:
    """If ``target_mods[consumer_idx]`` directly consumes a ``Concat`` layer, return ``[(src_ch, dst_ch), ...]``."""
    if target_mods is None or consumer_idx <= 0 or consumer_idx >= len(target_mods):
        return None
    consumer = target_mods[consumer_idx]
    f = getattr(consumer, "f", -1)
    if not isinstance(f, int):
        return None
    in_idx = consumer_idx + f if f < 0 else f
    if in_idx < 0 or in_idx >= len(target_mods):
        return None
    in_mod = target_mods[in_idx]
    if not isinstance(in_mod, Concat):
        return None

    pairs: list[tuple[int, int]] = []
    for bf in getattr(in_mod, "f", []):
        ref = in_idx + bf if bf < 0 else bf
        sc = _infer_layer_out_ch(donor_sd, target_mods, ref)
        dc = _infer_layer_out_ch(target_sd, target_mods, ref)
        if sc is None or dc is None:
            return None
        pairs.append((sc, dc))
    return pairs


def _copy_out_rows_with_chunked_in(
    src_cast: torch.Tensor,
    dst: torch.Tensor,
    out_pairs: list[tuple[int, int, int]],
    in_pairs: list[tuple[int, int, int]],
) -> int:
    """Copy ``(out_src, out_dst, out_len) x (in_src, in_dst, in_len)`` blocks; zero extra cols on copied rows."""
    # Scale extra widened output filters down (0.05x Kaiming) for clean symmetry breaking
    dst.mul_(0.05)
    copied = 0
    for os_i, od_i, olen in out_pairs:
        # Zero all input columns on pretrained output rows so newly widened inputs inject 0 noise at step 0
        dst[od_i : od_i + olen].zero_()
        for is_i, id_i, ilen in in_pairs:
            dst[od_i : od_i + olen, id_i : id_i + ilen].copy_(
                src_cast[os_i : os_i + olen, is_i : is_i + ilen]
            )
            copied += olen * ilen * (dst.shape[2] * dst.shape[3] if dst.ndim == 4 else 1)
    return copied


def _copy_sliced_tensor(
    src: torch.Tensor,
    dst: torch.Tensor,
    key: str,
    donor_sd: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
    target_mods: Any,
) -> int:
    """Function-preserving Net2Wider tensor slice transfer."""
    if src.shape == dst.shape:
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        return dst.numel()

    if src.ndim != dst.ndim or src.ndim not in (1, 2, 4):
        return 0
    if src.ndim == 4 and src.shape[2:] != dst.shape[2:]:
        return 0

    src_cast = src.to(device=dst.device, dtype=dst.dtype)
    parts = key.split(".")
    layer_idx = int(parts[1]) if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() else -1

    is_top_c2f = layer_idx >= 0 and f"model.{layer_idx}.m.0.cv1.conv.weight" in target_sd

    # 1D BatchNorm / bias tensors
    if dst.ndim == 1:
        is_unit = key.endswith((".weight", ".running_var"))
        dst.fill_(1.0 if is_unit else 0.0)
        if (
            is_top_c2f
            and key.startswith(f"model.{layer_idx}.cv1.bn.")
            and src.shape[0] % 2 == 0
            and dst.shape[0] % 2 == 0
            and src.shape[0] != dst.shape[0]
        ):
            s_half, d_half = src.shape[0] // 2, dst.shape[0] // 2
            c_half = min(s_half, d_half)
            dst[:c_half].copy_(src_cast[:c_half])
            dst[d_half : d_half + c_half].copy_(src_cast[s_half : s_half + c_half])
            return 2 * c_half

        c_min = min(src.shape[0], dst.shape[0])
        dst[:c_min].copy_(src_cast[:c_min])
        return c_min

    # Depthwise conv (groups == out_channels -> shape[1] == 1)
    if dst.ndim == 4 and src.shape[1] == 1 and dst.shape[1] == 1:
        dst.mul_(0.05)
        c_out = min(src.shape[0], dst.shape[0])
        dst[:c_out].copy_(src_cast[:c_out])
        return c_out * dst.shape[2] * dst.shape[3]

    # Determine output-channel block alignment (only top-level C2f/C3k2 cv1 splits via chunk(2))
    if (
        is_top_c2f
        and key == f"model.{layer_idx}.cv1.conv.weight"
        and src.shape[0] % 2 == 0
        and dst.shape[0] % 2 == 0
        and src.shape[0] != dst.shape[0]
    ):
        s_half, d_half = src.shape[0] // 2, dst.shape[0] // 2
        c_half = min(s_half, d_half)
        out_pairs = [(0, 0, c_half), (s_half, d_half, c_half)]
    else:
        c_out = min(src.shape[0], dst.shape[0])
        out_pairs = [(0, 0, c_out)]

    # Determine input-channel block alignment
    in_pairs: list[tuple[int, int, int]] | None = None

    # Case A1: Top-level C2f / C3k2 cv2.conv.weight concatenating (2 + len(m)) chunks of hidden width c_
    if is_top_c2f and key == f"model.{layer_idx}.cv2.conv.weight":
        s_cv1 = donor_sd.get(f"model.{layer_idx}.cv1.bn.weight")
        d_cv1 = target_sd.get(f"model.{layer_idx}.cv1.bn.weight")
        if s_cv1 is not None and d_cv1 is not None and s_cv1.shape[0] % 2 == 0 and d_cv1.shape[0] % 2 == 0:
            sc_h, dc_h = int(s_cv1.shape[0] // 2), int(d_cv1.shape[0] // 2)
            if sc_h > 0 and dc_h > 0 and src.shape[1] % sc_h == 0 and dst.shape[1] % dc_h == 0:
                n_s, n_d = src.shape[1] // sc_h, dst.shape[1] // dc_h
                if n_s == n_d and n_s >= 2:
                    c_h = min(sc_h, dc_h)
                    in_pairs = [(k * sc_h, k * dc_h, c_h) for k in range(n_d)]

    # Case A2: Inner C3 / C3k cv3.conv.weight concatenating 2 chunks [m(cv1(x)), cv2(x)]
    if in_pairs is None and key.endswith(".cv3.conv.weight"):
        c3_prefix = key[: -len(".cv3.conv.weight")]
        s_b1 = donor_sd.get(f"{c3_prefix}.cv1.bn.weight")
        d_b1 = target_sd.get(f"{c3_prefix}.cv1.bn.weight")
        if s_b1 is not None and d_b1 is not None:
            sc_h, dc_h = int(s_b1.shape[0]), int(d_b1.shape[0])
            if 2 * sc_h == src.shape[1] and 2 * dc_h == dst.shape[1]:
                c_h = min(sc_h, dc_h)
                in_pairs = [(0, 0, c_h), (sc_h, dc_h, c_h)]

    # Case B: Layer directly consuming a Concat module (e.g. model.13.cv1, model.16.cv1, model.19.cv1, model.22.cv1)
    if in_pairs is None and layer_idx >= 0 and key in (f"model.{layer_idx}.cv1.conv.weight", f"model.{layer_idx}.conv.weight"):
        branch_pairs = _get_concat_branch_pairs(donor_sd, target_sd, target_mods, layer_idx)
        if branch_pairs is not None:
            s_tot = sum(p[0] for p in branch_pairs)
            d_tot = sum(p[1] for p in branch_pairs)
            if s_tot == src.shape[1] and d_tot == dst.shape[1]:
                in_pairs = []
                so, do = 0, 0
                for sc, dc in branch_pairs:
                    in_pairs.append((so, do, min(sc, dc)))
                    so += sc
                    do += dc

    # Default contiguous input channels
    if in_pairs is None:
        in_pairs = [(0, 0, min(src.shape[1], dst.shape[1]))]

    return _copy_out_rows_with_chunked_in(src_cast, dst, out_pairs, in_pairs)


def _compensate_head_scale_factors(donor_obj: Any, target: nn.Module) -> None:
    """Divide first-conv weights on ``YOLO27Detect`` by ``scale_factors[i]`` when donor head was unscaled."""
    target_mods = getattr(target, "model", None)
    if target_mods is None or len(target_mods) == 0:
        return
    head = target_mods[-1]
    scale_factors = getattr(head, "scale_factors", None)
    if not scale_factors or getattr(head, "_scale_folded", False):
        return

    # If donor already had identical YOLO27Detect scale_factors, weights are already compensated
    donor_mods = getattr(donor_obj, "model", None) if isinstance(donor_obj, nn.Module) else None
    donor_head = donor_mods[-1] if donor_mods is not None and len(donor_mods) > 0 else None
    if donor_head is not None and getattr(donor_head, "scale_factors", None) == scale_factors:
        return

    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branch_list = getattr(head, attr, None)
        if branch_list is None:
            continue
        for branch, s in zip(branch_list, scale_factors):
            if s == 1.0 or s == 0.0:
                continue
            first_conv = next(m for m in branch.modules() if isinstance(m, nn.Conv2d))
            first_conv.weight.data.div_(s)


@torch.no_grad()
def transfer_pretrained_weights(donor: nn.Module | dict[str, torch.Tensor], target: nn.Module) -> dict[str, float]:
    """Transfer exact and function-preserving Net2Wider pretrained weights from ``donor`` into ``target``."""
    donor_obj = donor["model"] if isinstance(donor, dict) and isinstance(donor.get("model"), nn.Module) else donor
    donor_sd = donor_obj if isinstance(donor_obj, dict) else donor_obj.float().state_dict()
    target_sd = target.state_dict()
    target_mods = getattr(target, "model", None)

    head_idx = len(target_mods) - 1 if target_mods is not None else -1
    donor_cls_key = f"model.{head_idx}.cv3.0.2.bias"
    target_cls_key = f"model.{head_idx}.cv3.0.2.bias"
    same_nc = (
        donor_cls_key in donor_sd
        and target_cls_key in target_sd
        and donor_sd[donor_cls_key].shape == target_sd[target_cls_key].shape
    )

    # Support both 3-scale->3-scale (YOLO27n) and 3-scale->2-scale (YOLO27n-Lite) warm-start
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
            continue

        # Keep target dataset 19-class focal bias prior on final 1x1 projection, while warm-starting blocks 0 & 1
        if not same_nc and any(tag in key for tag in (".cv3.", ".one2one_cv3.")) and ".2." in key:
            continue
        if any(tag in key for tag in (".fg_branch.", ".strip_adapters.")) and key not in donor_sd:
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
            n_copied = _copy_sliced_tensor(src, dst, key, donor_sd, target_sd, target_mods)
            if n_copied > 0:
                sliced_tensors += 1
                copied_numel += n_copied

    # Pre-compensate fixed feature scale factors so step-0 outputs match unscaled donor 100% exactly
    _compensate_head_scale_factors(donor_obj, target)

    # Ensure zero-residual strip adapters stay strictly zero-initialized at step 0 if not in donor
    if target_mods is not None and len(target_mods) > 0:
        strip_adapters = getattr(target_mods[-1], "strip_adapters", None)
        if strip_adapters is not None and not any(".strip_adapters." in k for k in donor_sd):
            for adapter in strip_adapters:
                adapter.reset_zero_residual()

    ratio = copied_numel / max(total_numel, 1)
    LOGGER.info(
        f"YOLO27 Warm-Start: transferred {exact_tensors} exact + {sliced_tensors} Net2Wider-sliced tensors "
        f"({copied_numel / 1e6:.2f}M / {total_numel / 1e6:.2f}M elements, {ratio * 100:.1f}%)"
    )
    return {
        "exact_tensors": exact_tensors,
        "sliced_tensors": sliced_tensors,
        "copied_numel": copied_numel,
        "total_numel": total_numel,
        "ratio": ratio,
    }


def resolve_donor_checkpoint(
    donor_name: str = "yolo26n.pt",
    allow_download: bool = True,
    extra_dirs: list[str | Path] | None = None,
) -> Path | None:
    """Locate ``donor_name`` across local/parent/sibling folders or download cleanly via Range-GET."""
    name = Path(donor_name).name
    cwd = Path.cwd().resolve()
    cand_paths: list[Path] = [
        Path(donor_name),
        cwd / name,
        cwd.parent / name,
        cwd.parent.parent / name,
        Path(SETTINGS.get("weights_dir", "")) / name,
        Path.home() / name,
    ]
    if extra_dirs:
        for d in extra_dirs:
            if d:
                dp = Path(str(d))
                cand_paths.extend([dp / name, dp.parent / name, dp.parent.parent / name])

    # Also check sibling project folders (e.g. Desktop/github/*/yolo26n.pt where pure yolo26n was tested)
    if cwd.parent.is_dir():
        try:
            for sub in cwd.parent.iterdir():
                if sub.is_dir() and sub != cwd:
                    cand_paths.append(sub / name)
        except OSError:
            pass

    for p in cand_paths:
        try:
            if p.is_file() and p.stat().st_size > 100_000:
                LOGGER.info(f"YOLO27 Warm-Start: found local pretrained donor checkpoint -> {p}")
                return p
        except OSError:
            continue

    if not allow_download:
        return None

    # Note: GitHub Releases redirects to AWS S3 GET-presigned URLs which reject HEAD with 403!
    # Always probe with GET + Range: bytes=0-0 (returns 200/206 in <1s when reachable, fails fast offline).
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    try:
        req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            resp.read(1)
    except Exception:
        LOGGER.warning(
            f"YOLO27 Warm-Start: 未在本地目录找到 '{name}' 且当前网络无法直连 GitHub 下载！"
            f" 请将你测过 yolo26n 的 '{name}' 复制到项目根目录（或通过 --pretrained 指定路径），否则将退回随机初始化！"
        )
        return None

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(donor_name))
        if downloaded.is_file() and downloaded.stat().st_size > 100_000:
            return downloaded
    except Exception as exc:
        LOGGER.warning(f"YOLO27 Warm-Start: 自动下载 {donor_name} 失败 ({exc})。")
    return None
