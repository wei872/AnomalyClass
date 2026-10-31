"""Normal-only DINO patch-feature anomaly detection (open-set localization).

A memory bank of *normal* patch features is built once.  A test image is scored by
the distance from each of its patches to that bank, which yields a pixel-level
anomaly map.  Because the bank never contains defects, this flags defect types
that were never labelled or even seen during training -- the capability a
supervised YOLO detector cannot provide.

The anomaly map is then binarised with a stability search over thresholds (in the
spirit of AnomalyNCD's MEBin) to produce region masks that follow the defect
instead of a rectangle, so mask-guided classification sees mostly defect pixels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .model import build_backbone
from .transforms import MEAN, STD

BANK_FORMAT = "fabricncd-anomaly-bank"
BANK_VERSION = 1
DEFAULT_LAYERS = (3, 5, 7)


def anomaly_settings(cfg: dict) -> dict:
    settings = cfg.get("anomaly") or {}
    return {
        "image_size": int(settings.get("image_size", 448)),
        "layers": list(settings.get("layers", DEFAULT_LAYERS)),
        "neighborhood": int(settings.get("neighborhood", 3)),
        "bank_size": int(settings.get("bank_size", 8192)),
        "max_candidates": int(settings.get("max_candidates", 40000)),
        "coreset": bool(settings.get("coreset", True)),
        "knn": int(settings.get("knn", 3)),
        "smooth_sigma": float(settings.get("smooth_sigma", 4.0)),
        "topk_ratio": float(settings.get("topk_ratio", 0.01)),
        "min_area": int(settings.get("min_area", 24)),
        "max_area_ratio": float(settings.get("max_area_ratio", 0.35)),
        "max_regions": int(settings.get("max_regions", 20)),
        "threshold_steps": int(settings.get("threshold_steps", 24)),
    }


def _image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = TF.resize(image.convert("RGB"), (size, size), InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def _aggregate_neighborhood(tokens: torch.Tensor, side: int, kernel: int) -> torch.Tensor:
    """Local average pooling on the patch grid, as in PatchCore.

    A single patch is noise-sensitive; averaging its neighbourhood makes the
    normal descriptor stable while keeping the spatial grid.
    """
    if kernel <= 1:
        return tokens
    batch, count, dim = tokens.shape
    grid = tokens.transpose(1, 2).reshape(batch, dim, side, side)
    pooled = F.avg_pool2d(grid, kernel, stride=1, padding=kernel // 2)
    return pooled.flatten(2).transpose(1, 2)


class PatchEncoder:
    """Frozen pretrained DINO used purely as a texture descriptor."""

    def __init__(self, cfg: dict, device: torch.device, allow_random_init: bool = False):
        self.settings = anomaly_settings(cfg)
        self.device = device
        if allow_random_init:
            # Test-only: the bank and every later query MUST use identical weights.
            # Without a fixed seed each encoder would be a different random network
            # and all distances would be meaningless. Global RNG is left untouched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20240517)
                self.backbone = build_backbone(cfg, load_pretrained=True, allow_random_init=True,
                                               image_size=self.settings["image_size"])
        else:
            self.backbone = build_backbone(cfg, load_pretrained=True, allow_random_init=False,
                                           image_size=self.settings["image_size"])
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        self.patch_size = self.backbone.patch_size
        self.side = self.settings["image_size"] // self.patch_size

    def fingerprint(self) -> str:
        """Identify the exact descriptor weights a bank was built with."""
        import hashlib

        digest = hashlib.sha256()
        for name, tensor in sorted(self.backbone.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()[:32]

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        """Return [len(images), side*side, D*len(layers)] L2-normalised features."""
        out = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([_image_tensor(im, self.settings["image_size"])
                                 for im in images[start:start+batch_size]]).to(self.device)
            tokens, side = self.backbone.get_patch_tokens(batch, self.settings["layers"])
            layers = [_aggregate_neighborhood(tokens[:, index], side, self.settings["neighborhood"])
                      for index in range(tokens.shape[1])]
            features = F.normalize(torch.cat(layers, dim=-1), dim=-1)
            out.append(features.reshape(features.shape[0], -1, features.shape[-1]).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.side**2, 1), np.float32)


def _greedy_coreset(features: torch.Tensor, target: int, progress=lambda _: None) -> torch.Tensor:
    """k-centre greedy selection for coverage of the normal feature cloud."""
    count = features.shape[0]
    if target >= count:
        return torch.arange(count, device=features.device)
    # ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2 via a matvec.  The naive
    # (features-c)**2 form allocates a [N, D] temporary on every one of the
    # `target` iterations (176 MiB x 8192 at production scale).
    # The ||c||^2 term MUST be kept: min_distance compares across iterations,
    # so dropping a per-centre constant would corrupt the selection.
    norms = (features*features).sum(dim=1)
    centre = features.mean(dim=0)
    index = int(torch.argmax(norms - 2*(features @ centre) + (centre*centre).sum()))
    chosen = [index]
    min_distance = torch.full((count,), float("inf"), device=features.device)
    picked = torch.zeros(count, dtype=torch.bool, device=features.device)
    for step in range(1, target):
        distance = (norms - 2*(features @ features[index]) + norms[index]).clamp_min_(0)
        min_distance = torch.minimum(min_distance, distance)
        picked[index] = True
        min_distance = min_distance.masked_fill(picked, -1.0)
        index = int(torch.argmax(min_distance))
        chosen.append(index)
        if step % 1024 == 0:
            progress(f"coreset {step}/{target}")
    return torch.tensor(chosen, device=features.device)


def build_bank(normal_root: str | Path, output: str | Path, cfg: dict, device_name: str = "auto",
               allow_random_init: bool = False, batch_size: int = 8, overwrite: bool = False,
               progress=lambda _: None) -> dict:
    from .data import IMAGE_EXTENSIONS
    from .engine import resolve_device

    normal_root = Path(normal_root).expanduser().resolve()
    if not normal_root.is_dir():
        raise FileNotFoundError(f"正常图目录不存在: {normal_root}")
    paths = sorted(p for p in normal_root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"{normal_root} 下没有正常图；normal-only 异常检测必须有正常样本")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"特征库目录非空: {output}；使用新目录或 --overwrite")
    device = resolve_device(device_name)
    encoder = PatchEncoder(cfg, device, allow_random_init)
    settings = encoder.settings
    progress(f"编码 {len(paths)} 张正常图 @ {settings['image_size']}px, layers={settings['layers']}")

    # Subsample per image WHILE encoding. Materialising every patch first would
    # need 320 x 3136 x 1152 x 4B = 4.3 GiB of host RAM (twice that during
    # concatenate) and then the same 4.3 GiB on the GPU before downsampling --
    # an OOM on 8 GB cards. A per-image quota also guarantees every normal image
    # contributes, which global random sampling does not.
    per_image = max(1, -(-settings["max_candidates"] // len(paths)))
    patches_per_image = encoder.side**2
    total = len(paths)*patches_per_image
    rng = np.random.default_rng(0)
    collected = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start+batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        features = encoder.encode(images, batch_size)
        keep = min(per_image, features.shape[1])
        for index in range(features.shape[0]):
            pick = rng.choice(features.shape[1], size=keep, replace=False)
            collected.append(features[index, pick])
        if (start // batch_size) % 10 == 0:
            progress(f"encoded {min(start+batch_size, len(paths))}/{len(paths)}")
    candidates = np.concatenate(collected, axis=0)
    if candidates.shape[0] > settings["max_candidates"]:
        pick = rng.choice(candidates.shape[0], size=settings["max_candidates"], replace=False)
        candidates = candidates[pick]
    progress(f"正常 patch 总数 {total:,}；分层采样候选 {candidates.shape[0]} x {candidates.shape[1]}")
    pool = torch.from_numpy(candidates).to(device)
    if settings["coreset"] and pool.shape[0] > settings["bank_size"]:
        indices = _greedy_coreset(pool, settings["bank_size"], progress)
        pool = pool[indices]
    elif pool.shape[0] > settings["bank_size"]:
        pool = pool[:settings["bank_size"]]
    bank = pool.cpu().numpy().astype(np.float32)
    progress(f"特征库规模: {bank.shape[0]} x {bank.shape[1]}")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "bank.npz", features=bank)
    metadata = {"format": BANK_FORMAT, "version": BANK_VERSION,
                "normal_images": len(paths), "normal_patches": total,
                "bank_size": int(bank.shape[0]), "feature_dim": int(bank.shape[1]),
                "settings": settings, "backbone": cfg["model"]["backbone"],
                "encoder_fingerprint": encoder.fingerprint(),
                "random_init": bool(allow_random_init), "calibration": None}
    (output / "bank.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n",
                                     encoding="utf-8")
    progress(f"特征库已保存: {output}")
    return metadata


def load_bank(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "bank.json"
    if not path.is_file():
        raise FileNotFoundError(f"异常特征库不存在: {path}；请先运行 scripts/build_anomaly_bank.py")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != BANK_FORMAT or metadata.get("version") != BANK_VERSION:
        raise ValueError(f"不兼容的异常特征库: {path}")
    with np.load(path.parent / "bank.npz") as data:
        metadata["features"] = data["features"].astype(np.float32)
    metadata["root"] = str(path.parent)
    return metadata


class AnomalyScorer:
    """Nearest-neighbour distance to the normal bank -> pixel anomaly map."""

    def __init__(self, bank: dict, cfg: dict, device: torch.device, allow_random_init: bool = False):
        self.bank = bank
        self.settings = bank["settings"]
        self.device = device
        self.encoder = PatchEncoder(cfg, device, allow_random_init)
        if self.encoder.settings["image_size"] != self.settings["image_size"]:
            raise ValueError("异常特征库与当前 anomaly.image_size 不一致，请重建特征库")
        if self.encoder.settings["layers"] != self.settings["layers"]:
            raise ValueError("异常特征库与当前 anomaly.layers 不一致，请重建特征库")
        if self.encoder.settings["neighborhood"] != self.settings["neighborhood"]:
            raise ValueError("异常特征库与当前 anomaly.neighborhood 不一致，请重建特征库")
        expected = bank.get("encoder_fingerprint")
        if expected and expected != self.encoder.fingerprint():
            raise ValueError(
                "异常特征库与当前 DINO 权重不一致（bank 由另一份权重建立），"
                "距离分数无意义；请用当前权重重建特征库")
        self.features = torch.from_numpy(bank["features"]).to(device)
        self.calibration = bank.get("calibration") or {}

    @torch.inference_mode()
    def patch_scores(self, image: Image.Image) -> np.ndarray:
        features = self.encoder.encode([image], batch_size=1)[0]
        query = torch.from_numpy(features).to(self.device)
        k = max(1, min(int(self.settings["knn"]), self.features.shape[0]))
        scores = []
        for start in range(0, query.shape[0], 4096):
            distance = torch.cdist(query[start:start+4096], self.features)
            scores.append(distance.topk(k, largest=False).values.mean(dim=1))
        return torch.cat(scores).cpu().numpy()

    def anomaly_map(self, image: Image.Image) -> tuple[np.ndarray, float]:
        side = self.encoder.side
        raw = self.patch_scores(image)
        grid = torch.from_numpy(raw).reshape(1, 1, side, side)
        dense = F.interpolate(grid, size=(image.height, image.width),
                              mode="bilinear", align_corners=False)[0, 0].numpy()
        sigma = float(self.settings["smooth_sigma"])
        if sigma > 0:
            dense = ndimage.gaussian_filter(dense, sigma)
        top = max(1, int(round(raw.size*float(self.settings["topk_ratio"]))))
        image_score = float(np.mean(np.sort(raw)[-top:]))
        return dense.astype(np.float32), image_score

    def normalize_image_score(self, raw: float) -> float:
        """Map a raw distance onto the normal-image scale (1.0 == normal upper bound)."""
        low = self.calibration.get("image_low")
        high = self.calibration.get("image_high")
        if low is None or high is None or high <= low:
            return float(raw)
        return float(max(0.0, (raw-low)/(high-low)))

    def normalize_pixel(self, values: np.ndarray | float):
        low = self.calibration.get("pixel_low")
        high = self.calibration.get("pixel_high")
        if low is None or high is None or high <= low:
            return values
        return np.maximum(0.0, (np.asarray(values)-low)/(high-low))


def calibrate_bank(bank_dir: str | Path, normal_val_root: str | Path, cfg: dict,
                   device_name: str = "auto", allow_random_init: bool = False,
                   progress=lambda _: None) -> dict:
    """Score held-out normal images so anomaly scores become interpretable.

    Without this, a raw feature distance has no absolute meaning and any
    threshold would be arbitrary.
    """
    from .data import IMAGE_EXTENSIONS
    from .engine import resolve_device

    bank_dir = Path(bank_dir).expanduser().resolve()
    metadata = load_bank(bank_dir)
    root = Path(normal_val_root).expanduser().resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"{root} 下没有正常图，无法标定异常分数")
    scorer = AnomalyScorer(metadata, cfg, resolve_device(device_name), allow_random_init)
    image_scores, pixel_samples = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as handle:
            dense, score = scorer.anomaly_map(handle.convert("RGB"))
        image_scores.append(score)
        pixel_samples.append(np.quantile(dense, [0.5, 0.99, 0.999]))
        if (index+1) % 20 == 0:
            progress(f"calibrated {index+1}/{len(paths)}")
    pixels = np.concatenate(pixel_samples)
    calibration = {
        "normal_val_images": len(paths),
        "image_low": float(np.median(image_scores)),
        "image_high": float(np.quantile(image_scores, 0.995)) if len(paths) > 1 else float(max(image_scores)*1.05),
        "pixel_low": float(np.median(pixels)),
        "pixel_high": float(np.quantile(pixels, 0.999)),
    }
    if calibration["image_high"] <= calibration["image_low"]:
        calibration["image_high"] = calibration["image_low"]*1.05 + 1e-6
    if calibration["pixel_high"] <= calibration["pixel_low"]:
        calibration["pixel_high"] = calibration["pixel_low"]*1.05 + 1e-6
    path = bank_dir / "bank.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["calibration"] = calibration
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    progress(f"标定完成: {calibration}")
    return calibration


def _valid_components(labelled: np.ndarray, min_area: int, max_area: int) -> list[int]:
    """Component ids that are defect-sized.

    A component covering most of the image is a failed threshold, not a defect;
    keeping it would make every image look like one giant anomaly.
    """
    sizes = np.bincount(labelled.ravel())
    return [index for index in range(1, len(sizes)) if min_area <= sizes[index] <= max_area]


def binarize(dense: np.ndarray, settings: dict, pixel_threshold: float | None = None) -> tuple[np.ndarray, dict]:
    """MEBin-style stability search over thresholds, rejecting degenerate components.

    A single hand-set threshold either fragments a defect or floods the whole
    image.  Sweeping thresholds and taking the widest plateau of constant
    *defect-sized* component count is far more robust.
    """
    empty = np.zeros(dense.shape, bool)
    highest = float(dense.max())
    min_area = int(settings["min_area"])
    max_area = int(float(settings.get("max_area_ratio", 0.35))*dense.size)
    info = {"threshold": 0.0, "components": 0, "stable_width": 0, "swept": 0}
    if highest <= 0:
        return empty, info
    # Nothing above the calibrated normal level means no anomaly at all.
    if pixel_threshold is not None and highest <= float(pixel_threshold):
        info["threshold"] = float(pixel_threshold)
        return empty, info

    lowest = float(pixel_threshold) if pixel_threshold is not None else float(np.quantile(dense, 0.95))
    lowest = max(lowest, float(np.median(dense)))
    if lowest >= highest:
        lowest = highest*0.5
    steps = max(2, int(settings["threshold_steps"]))
    candidates = np.linspace(lowest, highest, steps)
    counts, binaries = [], []
    for value in candidates:
        binary = dense >= value
        labelled, _ = ndimage.label(binary)
        keep = _valid_components(labelled, min_area, max_area)
        counts.append(len(keep))
        binaries.append((binary, labelled, keep))

    best_start, best_length, current_start = -1, 0, 0
    for index in range(1, len(counts)):
        if counts[index] != counts[current_start]:
            current_start = index
        length = index-current_start+1
        if counts[current_start] > 0 and length > best_length:
            best_start, best_length = current_start, length
    if best_start < 0:
        info.update(threshold=float(lowest), swept=len(candidates))
        return empty, info
    middle = best_start + best_length // 2
    binary, labelled, keep = binaries[middle]
    selected = np.isin(labelled, keep) if keep else empty
    return selected, {"threshold": float(candidates[middle]), "components": len(keep),
                      "stable_width": int(best_length), "swept": len(candidates)}


def regions_from_map(dense: np.ndarray, binary: np.ndarray, settings: dict,
                     score_of=None) -> list[dict]:
    """Connected components -> pixel boxes plus true defect masks."""
    labelled, count = ndimage.label(binary)
    if count == 0:
        return []
    sizes = np.bincount(labelled.ravel())
    objects = ndimage.find_objects(labelled)
    max_area = int(float(settings.get("max_area_ratio", 0.35))*dense.size)
    regions = []
    for index, span in enumerate(objects, start=1):
        if span is None or not int(settings["min_area"]) <= sizes[index] <= max_area:
            continue
        ys, xs = span
        mask = labelled == index
        raw_max = float(dense[mask].max())
        region = {
            "bbox_xyxy": [float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)],
            "area": int(sizes[index]),
            "anomaly_score_raw": raw_max,
            "anomaly_score_mean_raw": float(dense[mask].mean()),
            "mask": mask,
            "source": "anomaly_map",
        }
        if score_of is not None:
            region["anomaly_score"] = float(score_of(raw_max))
        regions.append(region)
    regions.sort(key=lambda item: item["anomaly_score_raw"], reverse=True)
    return regions[:int(settings["max_regions"])]
