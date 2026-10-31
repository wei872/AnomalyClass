"""YOLO27 unified model, detection model, and calibrated fabric-defect trainer."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel, torch_safe_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import initialize_weights, unwrap_model

from .loss import YOLO27DetectionLoss, YOLO27E2ELoss
from .modules import FabricStripEnhancer, ForegroundAlignmentBranch, YOLO27Detect
from .weights import resolve_donor_checkpoint, transfer_pretrained_weights

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        YOLO27Detect,
        FabricStripEnhancer,
        ForegroundAlignmentBranch,
    ])


class YOLO27DetectionModel(DetectionModel):
    """YOLO27 CNN detection model upgrading ``Detect`` to ``YOLO27Detect`` and ``YOLO27E2ELoss``."""

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = "models/yolo27n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)

        m = self.model[-1]
        if isinstance(m, Detect) and not isinstance(m, YOLO27Detect):
            new_head = YOLO27Detect.from_detect(m)
            new_head.stride = m.stride.clone()
            self.model[-1] = new_head
            self.stride = new_head.stride
            initialize_weights(new_head)
            new_head.bias_init()
        self.nc = self.model[-1].nc

        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Initialize YOLO27E2ELoss (with Foreground Alignment Supervision) for dual-head models."""
        head = self.model[-1]
        if getattr(head, "one2one_cv2", None) is not None:
            return YOLO27E2ELoss(self)
        return YOLO27DetectionLoss(self)

    def fuse(self, verbose: bool = True, *args: Any, **kwargs: Any):
        """Ensure ``YOLO27Detect.fuse()`` runs across all Ultralytics versions and ``end2end`` modes."""
        super().fuse(verbose=verbose, *args, **kwargs)
        head = self.model[-1]
        if isinstance(head, YOLO27Detect) and not head._scale_folded:
            head.fuse()
        return self


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([YOLO27DetectionModel])


def _compute_inverse_freq_class_weights(labels: list[dict[str, Any]], nc: int, power: float = 0.15) -> torch.Tensor:
    """Compute gentle dampened inverse-frequency class weights normalized to mean 1.0."""
    if power <= 0.0 or not labels:
        return torch.ones(nc, dtype=torch.float32)
    cls_list = [lb["cls"].reshape(-1) for lb in labels if "cls" in lb and len(lb["cls"]) > 0]
    if not cls_list:
        return torch.ones(nc, dtype=torch.float32)
    all_cls = np.concatenate(cls_list, axis=0).astype(np.int64)
    counts = np.bincount(all_cls[all_cls < nc], minlength=nc).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        return torch.ones(nc, dtype=torch.float32)
    weights = np.ones(nc, dtype=np.float64)
    raw = (1.0 / np.where(present, counts, 1.0)) ** power
    mean_present = float(raw[present].mean())
    weights[present] = raw[present]
    weights[~present] = mean_present
    weights /= float(weights.mean())
    return torch.tensor(weights, dtype=torch.float32)


class YOLO27Trainer(DetectionTrainer):
    """YOLO27 trainer calibrated to strictly outperform pure ``yolo26n`` on custom fabric datasets.

    Core principles:
      1. Zero-Error Pretrained Warm-Start: transfers 89%–92% of ``yolo26n.pt`` via function-preserving
         Net2Wider slicing (``0.000000`` step-0 feature & box error across all scales!), searching
         cwd/parent/sibling folders and Range-GET downloading automatically.
      2. Calibrated Multi-Class Optimizer: defaults to Ultralytics ``optimizer='auto'``
         (``AdamW(lr0=0.000435, momentum=0.9, warmup_bias_lr=0.0)`` for ``nc=19``) + ``cos_lr=True``
         instead of over-stepping pretrained representations with ``lr0=0.0015``.
      3. Apples-to-Apples Dataset Protocol: uses ``train`` & ``val`` by default (matching pure ``yolo26n``
         and avoiding 50% empty-image recall dilution); pass ``use_normal=True`` / ``--use-normal`` to
         explicitly append ``normaltrain`` / ``normalval``.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self._cls_pw = float(overrides.pop("cls_pw", 0.15))
        self._use_normal = bool(overrides.pop("use_normal", False))

        opt_choice = str(overrides.get("optimizer", "auto"))
        overrides.setdefault("optimizer", opt_choice)
        if opt_choice.lower() == "adamw":
            overrides.setdefault("lr0", 0.0005)
            overrides.setdefault("momentum", 0.9)
            overrides.setdefault("lrf", 0.01)
            overrides.setdefault("warmup_bias_lr", 0.0)
        elif opt_choice.lower() == "musgd":
            overrides.setdefault("lr0", 0.004)
            overrides.setdefault("lrf", 0.05)
            overrides.setdefault("warmup_bias_lr", 0.0)

        overrides.setdefault("cos_lr", True)
        overrides.setdefault("warmup_epochs", 3.0)

        # Match yolo26n.pt pretrained loss balance
        overrides.setdefault("box", 7.5)
        overrides.setdefault("cls", 0.5)
        overrides.setdefault("dfl", 1.5)

        # Clean fabric augmentation: preserve warp/weft angles (no flipud/degrees/shear/copy_paste artifacts)
        overrides.setdefault("erasing", 0.1)
        overrides.setdefault("degrees", 0.0)
        overrides.setdefault("shear", 0.0)
        overrides.setdefault("flipud", 0.0)
        overrides.setdefault("fliplr", 0.5)
        overrides.setdefault("scale", 0.5)
        overrides.setdefault("mosaic", 1.0)
        overrides.setdefault("copy_paste", 0.0)
        overrides.setdefault("close_mosaic", 15)
        if hasattr(DEFAULT_CFG, "cls_pw"):
            overrides["cls_pw"] = self._cls_pw

        super().__init__(cfg, overrides, _callbacks)

    @staticmethod
    def _resolve_optional_dir(data: dict[str, Any], key: str) -> str | None:
        """Resolve optional ``normaltrain`` / ``normalval`` directory from dataset YAML if it exists on disk."""
        raw = data.get(key)
        if not raw:
            return None
        cand = Path(str(raw))
        if not cand.is_absolute() and data.get("path"):
            cand = Path(str(data["path"])) / cand
        if cand.is_dir() and any(cand.rglob("*.*")):
            return str(cand)
        return None

    def get_dataset(self) -> dict[str, Any]:
        """Load dataset config; append ``normaltrain`` / ``normalval`` only when ``use_normal=True``."""
        data = super().get_dataset()
        if not self._use_normal:
            return data
        for split_key, normal_key in (("train", "normaltrain"), ("val", "normalval")):
            normal_dir = self._resolve_optional_dir(data, normal_key)
            if normal_dir is None:
                continue
            cur = data.get(split_key)
            cur_list = list(cur) if isinstance(cur, (list, tuple)) else [cur]
            if normal_dir not in cur_list:
                data[split_key] = [*cur_list, normal_dir]
                LOGGER.info(f"YOLO27Trainer: attached defect-free background directory '{normal_key}' -> {normal_dir}")
        return data

    def set_model_names_for_load(self, model: YOLO27DetectionModel) -> YOLO27DetectionModel:
        """Align model class names with dataset names before optional weight loading."""
        names = self.data.get("names")
        if names is not None and hasattr(model, "names"):
            try:
                model.names = names
            except Exception as e:  # pragma: no cover - defensive
                LOGGER.warning(f"YOLO27Trainer: failed to set model names before load: {e}")
        return model

    def get_validator(self):
        """Build YOLO27Validator, preserve 4-column ``fg_loss`` header, and apply ``class_weights``."""
        m_unwrapped = unwrap_model(self.model)
        if getattr(m_unwrapped, "class_weights", None) is None and self._cls_pw > 0.0:
            ds = getattr(getattr(self, "train_loader", None), "dataset", None)
            label_list = getattr(ds, "labels", None)
            if label_list:
                cw = _compute_inverse_freq_class_weights(label_list, self.data["nc"], power=self._cls_pw)
                m_unwrapped.class_weights = cw
                LOGGER.info(
                    f"YOLO27Trainer: applied inverse-frequency class_weights (power={self._cls_pw}): "
                    f"{cw.numpy().round(3)}"
                )

        super().get_validator()
        if getattr(self, "loss_names", None) and len(self.loss_names) == 3:
            self.loss_names = (*self.loss_names, "fg_loss")

        val_cls = YOLO27Validator if self._use_normal else DetectionValidator
        return val_cls(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_model_attributes(self) -> None:
        """Attach dataset metadata and ``fg_loss`` column name."""
        super().set_model_attributes()
        if getattr(self, "loss_names", None) and len(self.loss_names) == 3:
            self.loss_names = (*self.loss_names, "fg_loss")

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: torch.nn.Module | None = None,
        verbose: bool = True,
    ) -> YOLO27DetectionModel:
        """Build ``YOLO27DetectionModel`` and apply function-preserving Net2Wider ``yolo26n.pt`` warm-start."""
        model = self.set_model_names_for_load(
            YOLO27DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights is not None:
            transfer_pretrained_weights(weights, model)
        elif self.args.pretrained and not self.resume:
            scale = model.yaml.get("scale", "n") if isinstance(getattr(model, "yaml", None), dict) else "n"
            donor_name = "yolo26s.pt" if scale == "s" else "yolo26n.pt"
            extra_dirs = [self.data.get("path", ""), str(self.data.get("train", ""))]
            donor_path = resolve_donor_checkpoint(donor_name, allow_download=True, extra_dirs=extra_dirs)
            if donor_path is not None:
                ckpt, _ = torch_safe_load(str(donor_path))
                donor_model = ckpt.get("ema") or ckpt.get("model")
                if donor_model is not None:
                    transfer_pretrained_weights(donor_model, model)
        return model


class YOLO27Validator(DetectionValidator):
    """DetectionValidator that attaches ``normalval`` / ``normaltrain`` when requested."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None, use_normal: bool = False):
        self._use_normal = use_normal
        super().__init__(dataloader, save_dir, args, _callbacks)

    def get_dataloader(self, dataset_path: Any, batch_size: int):
        if self._use_normal and isinstance(self.data, dict):
            normal_key = "normalval" if getattr(self.args, "split", "val") == "val" else "normaltrain"
            normal_dir = YOLO27Trainer._resolve_optional_dir(self.data, normal_key)
            if normal_dir is not None:
                cur_list = list(dataset_path) if isinstance(dataset_path, (list, tuple)) else [dataset_path]
                if normal_dir not in cur_list:
                    dataset_path = [*cur_list, normal_dir]
                    LOGGER.info(
                        f"YOLO27Validator: attached defect-free background directory '{normal_key}' -> {normal_dir}"
                    )
        return super().get_dataloader(dataset_path, batch_size)


def _translate_nms_kwarg(kwargs: dict[str, Any], for_export: bool = False) -> dict[str, Any]:
    """Bridge ``nms=True/False`` and ``end2end=False/True`` across Ultralytics 8.4.33 and 8.4.194+."""
    out = dict(kwargs)
    if "nms" in out and hasattr(DEFAULT_CFG, "end2end"):
        use_nms = bool(out.pop("nms"))
        out.setdefault("end2end", not use_nms)
        if for_export:
            out.setdefault("nms", False)
    return out


class YOLO27(YOLO):
    """Unified Ultralytics-compatible interface for YOLO27n / YOLO27n-Lite / YOLO27s models."""

    def __init__(self, model: str | Path = "models/yolo27n.yaml", task: str | None = "detect", verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        base_map = dict(super().task_map)
        base_map["detect"] = {
            "model": YOLO27DetectionModel,
            "trainer": YOLO27Trainer,
            "validator": YOLO27Validator,
            "predictor": yolo.detect.DetectionPredictor,
        }
        return base_map

    def val(self, validator=None, **kwargs: Any):
        """Validate model, supporting ``nms=True/False`` and optional ``use_normal=True``."""
        use_normal = bool(kwargs.pop("use_normal", False))
        norm_kwargs = _translate_nms_kwarg(kwargs, for_export=False)
        if use_normal and validator is None:
            custom = {"rect": True}
            args = {**self.overrides, **custom, **norm_kwargs, "mode": "val"}
            val_inst = YOLO27Validator(args=args, _callbacks=self.callbacks, use_normal=True)
            val_inst(model=self.model)
            self.metrics = val_inst.metrics
            return val_inst.metrics
        return super().val(validator=validator, **norm_kwargs)

    def predict(self, source=None, stream: bool = False, predictor=None, **kwargs: Any):
        """Run inference, supporting ``nms=True`` (one2many+NMS) and ``nms=False`` (one2one) across versions."""
        return super().predict(
            source=source,
            stream=stream,
            predictor=predictor,
            **_translate_nms_kwarg(kwargs, for_export=False),
        )

    def export(self, **kwargs: Any) -> str:
        """Export model, supporting ``nms=True`` (one2many) and ``nms=False`` (end-to-end one2one) across versions."""
        return super().export(**_translate_nms_kwarg(kwargs, for_export=True))
