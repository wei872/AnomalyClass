"""Synchronized image/mask transforms adapted from AnomalyNCD's transform.py."""
from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur, InterpolationMode
from torchvision.transforms import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PairTransform:
    def __init__(self, image_size: int = 224, training: bool = False, crop_pct: float = 0.875,
                 rotate90: bool = False):
        if not 0 < crop_pct <= 1:
            raise ValueError("crop_pct must be in (0, 1]")
        self.image_size, self.training = image_size, training
        self.resize_size = int(image_size / crop_pct)
        self.rotate90 = rotate90
        self.jitter = ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.9, 1.1))

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.size != mask.size:
            raise ValueError("image/mask size mismatch")
        image, mask = image.convert("RGB"), mask.convert("L")
        size = (self.resize_size, self.resize_size)
        image = F.resize(image, size, InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, size, InterpolationMode.NEAREST)
        margin = self.resize_size - self.image_size
        left = top = margin // 2
        if self.training and margin:
            bounds = mask.getbbox()
            if bounds is None:
                raise ValueError("AnomalyNCD training requires a nonempty region mask")
            cx, cy = (bounds[0]+bounds[2])//2, (bounds[1]+bounds[3])//2
            # Unlike unrestricted random cropping, retain the defect centre.
            lx, ux = max(0, cx-self.image_size+1), min(margin, cx)
            ly, uy = max(0, cy-self.image_size+1), min(margin, cy)
            left, top = random.randint(lx, ux), random.randint(ly, uy)
        # torchvision.functional.crop takes (left, top, width, height), NOT xyxy.
        box = (left, top, self.image_size, self.image_size)
        image, mask = F.crop(image, *box), F.crop(mask, *box)
        if self.training:
            if random.random() < 0.5:
                image, mask = F.hflip(image), F.hflip(mask)
            if random.random() < 0.5:
                image, mask = F.vflip(image), F.vflip(mask)
            # Disabled by default: a quarter-turn can exchange warp/weft semantics.
            if self.rotate90:
                angle = random.choice((0, 90, 180, 270))
                image, mask = F.rotate(image, angle), F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            image = self.jitter(image)
            image = F.adjust_sharpness(image, random.uniform(0.5, 1.5))
            if random.random() < 0.2:
                image = GaussianBlur(3, sigma=(0.1, 1.0))(image)
        image = F.normalize(F.to_tensor(image), MEAN, STD)
        mask = F.to_tensor(mask)
        if self.training and not bool(mask.any()):
            raise ValueError("Augmentation removed the anomaly mask")
        return image, mask


class TwoViews:
    def __init__(self, transform: PairTransform):
        self.transform = transform

    def __call__(self, image: Image.Image, mask: Image.Image):
        pairs = [self.transform(image, mask) for _ in range(2)]
        return [p[0] for p in pairs], [p[1] for p in pairs]


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
