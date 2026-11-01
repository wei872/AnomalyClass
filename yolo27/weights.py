"""Function-preserving Net2Wider pretrained warm-start for YOLO27n / YOLO27n-Dual / YOLO27n-Lite.

Transfers pretrained weights from ``yolo26n.pt`` (or ``yolo26s.pt`` / ``yolo11n.pt``) into
``YOLO27DetectionModel`` with **exact step-0 functional equivalence** (``max_diff < 1e-5``):
  1. Exact-shape layers (stem Conv, P4/P5 C3k2, SPPF, C2PSA, and ``c3=80`` classification
     conv towers) copy 1:1 with zero truncation.
  2. Widened ``C3k2`` stages (``P2/4`` width, ``P3/8`` 2x inner bottleneck, ``P3/8`` fine neck)
     and widened ``c2=32`` box regression towers align ``cv1`` (2-chunk split on both
     ``conv.weight`` and ``bn.*``), inner ``C3k.cv3`` (2-chunk concat), ``cv2``
     (``(2+n)``-chunk concat), and ``reg_max=1`` final ``1x1`` box projections.
  3. ``Concat`` consumers align input channel offsets per concatenated branch.
  4. Newly widened input-channel columns are zeroed ONLY on copied donor output rows,
     preserving Kaiming initialization across ``[new_out_rows, all_in_cols]`` so step-0
     outputs match the donor to ``< 1e-5`` while step-1 gradients immediately unlock all
     widened filters.
  5. ``YOLO27Detect`` deterministic ``scale_factors`` are pre-compensated (``1 / s_i``) on
     first-layer head convs when loading from an unscaled donor.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request

import torch
from torch import nn

from ultralytics.nn.modules.conv import Concat
from ultralytics.utils import LOGGER, SETTINGS, YAML

_SKIP_SCAN_DIRS = frozenset(
    {
        "images",
        "labels",
        "normal",
        "normaltrain",
        "normalval",
        "runs",
        "__pycache__",
        "node_modules",
        "site-packages",
        ".git",
        ".venv",
        "venv",
        "env",
        "conda",
        ".idea",
        ".vscode",
        ".pytest_cache",
        "Unselected files",
    }
)


def _extract_donor_object(donor: str | Path | dict[str, Any] | nn.Module) -> nn.Module | dict[str, torch.Tensor]:
    """Load or unwrap an ``nn.Module`` (preferring EMA weights) or state_dict from path/dict/model."""
    if isinstance(donor, (str, Path)):
        from ultralytics.nn.tasks import load_checkpoint

        weights, ckpt = load_checkpoint(str(donor))
        if isinstance(ckpt, dict):
            ema_or_model = ckpt.get("ema") or ckpt.get("model")
            if isinstance(ema_or_model, nn.Module):
                return ema_or_model
        if isinstance(weights, nn.Module):
            return weights
        donor = ckpt
    if isinstance(donor, dict) and (isinstance(donor.get("ema"), nn.Module) or isinstance(donor.get("model"), nn.Module)):
        return donor.get("ema") or donor.get("model")
    if isinstance(donor, nn.Module) and hasattr(donor, "ckpt") and hasattr(donor, "model"):
        return donor.model
    return donor


def _infer_layer_out_ch(sd: dict[str, torch.Tensor], seq_mods: Any, idx: int) -> int | None:
    """Infer output channel count of ``model.{idx}`` from state_dict tensors (following Upsample/Concat)."""
    if idx < 0:
        return None
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
    """Copy ``(out_src, out_dst, out_len) x (in_src, in_dst, in_len)`` blocks cleanly.

    Zeros uncovered input columns ONLY on copied donor output rows, leaving newly widened
    output rows ``[new_out, :]`` with their full Kaiming initialization so step-0 outputs
    remain 100% identical to the donor while step-1 gradients unlock all widened channels.
    """
    copied = 0
    for os_i, od_i, olen in out_pairs:
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
        c_out = min(src.shape[0], dst.shape[0])
        dst[:c_out].copy_(src_cast[:c_out])
        return c_out * dst.shape[2] * dst.shape[3]

    # Determine output-channel block alignment (top-level C2f/C3k2 cv1 splits via chunk(2, 1))
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

    if in_pairs is None:
        in_pairs = [(0, 0, min(src.shape[1], dst.shape[1]))]

    return _copy_out_rows_with_chunked_in(src_cast, dst, out_pairs, in_pairs)


def _compensate_head_scale_factors(donor_obj: Any, target: nn.Module) -> None:
    """Pre-divide first head conv weights by deterministic ``scale_factors[i]`` when donor is unscaled."""
    target_mods = getattr(target, "model", None)
    if target_mods is None or len(target_mods) == 0:
        return
    head = target_mods[-1]
    scale_factors = getattr(head, "scale_factors", None)
    if not scale_factors or getattr(head, "_scale_folded", False):
        return

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
def transfer_pretrained_weights(
    donor: str | Path | nn.Module | dict[str, Any],
    target: nn.Module,
    verbose: bool = True,
) -> dict[str, float]:
    """Transfer exact and function-preserving Net2Wider pretrained weights from ``donor`` into ``target``."""
    donor_obj = _extract_donor_object(donor)
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
    donor_box_key = f"model.{head_idx}.cv2.0.2.bias"
    target_box_key = f"model.{head_idx}.cv2.0.2.bias"
    same_reg_max = (
        donor_box_key in donor_sd
        and target_box_key in target_sd
        and donor_sd[donor_box_key].shape == target_sd[target_box_key].shape
    )

    donor_has_3_scales = f"model.{head_idx}.cv2.2.0.conv.weight" in donor_sd
    target_has_2_scales = (
        f"model.{head_idx}.cv2.1.0.conv.weight" in target_sd
        and f"model.{head_idx}.cv2.2.0.conv.weight" not in target_sd
    )

    exact_tensors, sliced_tensors = 0, 0
    copied_numel = 0
    total_numel = sum(v.numel() for v in target_sd.values() if v.is_floating_point() and v.ndim >= 1)

    for key, dst in target_sd.items():
        if not dst.is_floating_point() or dst.ndim == 0:
            continue

        # Keep target dataset 19-class focal bias prior on final 1x1 projection when nc differs
        if not same_nc and any(tag in key for tag in (".cv3.", ".one2one_cv3.")) and ".2." in key:
            continue
        # Guard final 1x1 box projection only when reg_max differs
        if not same_reg_max and any(tag in key for tag in (".cv2.", ".one2one_cv2.")) and ".2." in key:
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
            n_copied = _copy_sliced_tensor(src, dst, key, donor_sd, target_sd, target_mods)
            if n_copied > 0:
                sliced_tensors += 1
                copied_numel += n_copied

    # Mirror cv2/cv3 feature convs into one2one_cv2/one2one_cv3 when donor is NMS-only (e.g. yolo11n.pt)
    if target_mods is not None and len(target_mods) > 0:
        head = target_mods[-1]
        donor_has_o2o = any(".one2one_cv2." in k for k in donor_sd)
        if not donor_has_o2o and getattr(head, "one2one_cv2", None) is not None:
            for s in range(head.nl):
                head.one2one_cv2[s].load_state_dict(head.cv2[s].state_dict())
                head.one2one_cv3[s].load_state_dict(head.cv3[s].state_dict())

    _compensate_head_scale_factors(donor_obj, target)

    ratio = copied_numel / max(total_numel, 1)
    if verbose:
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


def _is_valid_ckpt(path: Path) -> bool:
    """Return True if ``path`` exists as a regular file > 100 KB."""
    try:
        return path.is_file() and path.stat().st_size > 100_000
    except OSError:
        return False


def _is_filesystem_root(path: Path) -> bool:
    """Return True if ``path`` is a bare filesystem or Windows drive root (e.g. ``/`` or ``D:\\``)."""
    try:
        return path.parent == path
    except OSError:
        return True


def _add_ancestors(seed: Path, out: list[Path], max_up: int = 3) -> None:
    """Append ``seed`` and up to ``max_up`` non-root parent directories."""
    cur = seed
    for _ in range(max_up + 1):
        if _is_filesystem_root(cur):
            break
        out.append(cur)
        cur = cur.parent


def _scan_runs_args_for_donor(runs_roots: Sequence[Path], donor_name: str) -> Path | None:
    """Check recent Ultralytics ``runs/**/args.yaml`` files for an existing path to ``donor_name``."""
    checked = 0
    for rdir in runs_roots:
        try:
            if not rdir.is_dir():
                continue
            for args_yaml in rdir.rglob("args.yaml"):
                checked += 1
                if checked > 40:
                    return None
                cfg = YAML.load(args_yaml)
                if not isinstance(cfg, dict):
                    continue
                for key in ("model", "pretrained"):
                    val = cfg.get(key)
                    if isinstance(val, str) and Path(val).name.lower() == donor_name.lower():
                        cand = Path(val)
                        if not cand.is_absolute():
                            cand = (args_yaml.parent.parent.parent / cand).resolve()
                        if _is_valid_ckpt(cand):
                            return cand
        except Exception:
            continue
    return None


def _bfs_find_donor(roots: Sequence[Path], donor_name: str, max_depth: int = 3, max_dirs: int = 400) -> Path | None:
    """Fast breadth-first directory search skipping dataset/image/venv folders."""
    visited: set[str] = set()
    queue: deque[tuple[Path, int]] = deque()

    for r in roots:
        if _is_filesystem_root(r):
            continue
        key = os.path.normcase(str(r))
        if key not in visited:
            visited.add(key)
            queue.append((r, 0))

    dirs_scanned = 0
    while queue and dirs_scanned < max_dirs:
        cur_dir, depth = queue.popleft()
        dirs_scanned += 1

        for direct in (cur_dir / donor_name, cur_dir / "weights" / donor_name):
            if _is_valid_ckpt(direct):
                return direct

        if depth >= max_depth:
            continue

        try:
            with os.scandir(cur_dir) as it:
                for entry in it:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name = entry.name
                    if name.startswith(".") or name in _SKIP_SCAN_DIRS:
                        continue
                    sub_path = Path(entry.path)
                    sub_key = os.path.normcase(str(sub_path))
                    if sub_key not in visited:
                        visited.add(sub_key)
                        queue.append((sub_path, depth + 1))
        except OSError:
            continue
    return None


def _cache_donor_locally(found: Path, repo_root: Path, donor_name: str) -> None:
    """Cache discovered donor into gitignored ``repo_root/weights/`` so future runs & AMP checks are instant."""
    try:
        tmp_root = Path(tempfile.gettempdir()).resolve()
        resolved_found = found.resolve()
        if str(resolved_found).startswith(str(tmp_root)):
            return  # Do not cache ephemeral unit-test checkpoints into the repository
        cache_dst = repo_root / "weights" / donor_name
        if resolved_found != cache_dst.resolve() and not _is_valid_ckpt(cache_dst):
            cache_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_found, cache_dst)
            LOGGER.info(f"YOLO27 Warm-Start: cached '{donor_name}' -> {cache_dst}")
    except Exception:
        pass


def _download_with_mirrors(donor_name: str, dst_path: Path) -> Path | None:
    """Attempt download via domestic GitHub release mirrors first, then official Ultralytics downloader."""
    base_gh = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{donor_name}"
    mirror_urls = (
        f"https://gh-proxy.com/{base_gh}",
        f"https://ghfast.top/{base_gh}",
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_part = dst_path.with_suffix(".pt.part")

    for url in mirror_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp, open(tmp_part, "wb") as f:
                shutil.copyfileobj(resp, f)
            if _is_valid_ckpt(tmp_part):
                tmp_part.replace(dst_path)
                LOGGER.info(f"YOLO27 Warm-Start: downloaded '{donor_name}' via mirror -> {dst_path}")
                return dst_path
        except Exception:
            try:
                tmp_part.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(dst_path))
        if _is_valid_ckpt(downloaded):
            return downloaded
    except Exception as exc:
        LOGGER.warning(f"YOLO27 Warm-Start: automatic download of '{donor_name}' failed ({exc}).")
    return None


def resolve_donor_checkpoint(
    scale: str = "n",
    explicit_donor: str | Path | None = None,
    allow_download: bool = True,
    extra_dirs: Sequence[str | Path] | None = None,
) -> Path | None:
    """Locate ``yolo26{scale}.pt`` across repo, Ultralytics SETTINGS, sibling/dataset roots, or download."""
    repo_root = Path(__file__).resolve().parents[1]

    # 1. Explicit CLI path (--donor) or YOLO27_DONOR environment variable
    for raw_exp in (explicit_donor, os.environ.get("YOLO27_DONOR")):
        if raw_exp:
            p = Path(str(raw_exp)).expanduser()
            if _is_valid_ckpt(p):
                LOGGER.info(f"YOLO27 Warm-Start: using explicit pretrained donor -> {p}")
                _cache_donor_locally(p, repo_root, p.name)
                return p

    scale_key = (scale or "n").lower()
    if scale_key.endswith(".pt"):
        donor_name = Path(scale_key).name
    else:
        if scale_key not in {"n", "s", "m", "l", "x"}:
            scale_key = "n"
        donor_name = f"yolo26{scale_key}.pt"

    cwd = Path.cwd().resolve()

    # 2. Fast direct checks in repo/cwd/SETTINGS/user home/Windows AppData
    direct_candidates: list[Path] = [
        repo_root / "weights" / donor_name,
        repo_root / donor_name,
        cwd / donor_name,
        cwd / "weights" / donor_name,
    ]
    for s_key in ("weights_dir", "runs_dir", "datasets_dir"):
        s_val = SETTINGS.get(s_key)
        if s_val:
            sp = Path(str(s_val))
            direct_candidates.extend([sp / donor_name, sp.parent / donor_name, sp.parent / "weights" / donor_name])

    home = Path.home()
    direct_candidates.extend(
        [
            home / donor_name,
            home / "Downloads" / donor_name,
            home / "Desktop" / donor_name,
            home / ".cache" / "ultralytics" / donor_name,
        ]
    )
    for env_key in ("APPDATA", "LOCALAPPDATA"):
        env_val = os.environ.get(env_key)
        if env_val:
            ap = Path(env_val) / "Ultralytics"
            direct_candidates.extend([ap / donor_name, ap / "weights" / donor_name])

    for cand in direct_candidates:
        if _is_valid_ckpt(cand):
            LOGGER.info(f"YOLO27 Warm-Start: found local pretrained donor -> {cand}")
            _cache_donor_locally(cand, repo_root, donor_name)
            return cand

    # 3. Inspect recent Ultralytics runs/**/args.yaml (finds exact path used in user's yolo26 baseline!)
    runs_roots: list[Path] = [cwd / "runs", repo_root / "runs"]
    if SETTINGS.get("runs_dir"):
        runs_roots.append(Path(str(SETTINGS["runs_dir"])))
    from_runs = _scan_runs_args_for_donor(runs_roots, donor_name)
    if from_runs is not None:
        LOGGER.info(f"YOLO27 Warm-Start: found donor referenced in Ultralytics runs -> {from_runs}")
        _cache_donor_locally(from_runs, repo_root, donor_name)
        return from_runs

    # 4. Bounded multi-level search across project & dataset ancestors (e.g. D:/tyhn/yolo/**)
    bfs_roots: list[Path] = []
    _add_ancestors(cwd, bfs_roots, max_up=3)
    _add_ancestors(repo_root, bfs_roots, max_up=3)
    if SETTINGS.get("runs_dir"):
        _add_ancestors(Path(str(SETTINGS["runs_dir"])).parent, bfs_roots, max_up=2)
    if SETTINGS.get("datasets_dir"):
        _add_ancestors(Path(str(SETTINGS["datasets_dir"])), bfs_roots, max_up=2)
    if extra_dirs:
        for d in extra_dirs:
            if d:
                _add_ancestors(Path(str(d)), bfs_roots, max_up=3)

    bfs_found = _bfs_find_donor(bfs_roots, donor_name, max_depth=3, max_dirs=400)
    if bfs_found is not None:
        LOGGER.info(f"YOLO27 Warm-Start: auto-discovered local pretrained donor -> {bfs_found}")
        _cache_donor_locally(bfs_found, repo_root, donor_name)
        return bfs_found

    # 5. Multi-mirror + official asset download fallback
    if allow_download:
        downloaded = _download_with_mirrors(donor_name, repo_root / "weights" / donor_name)
        if downloaded is not None:
            return downloaded

    return None
