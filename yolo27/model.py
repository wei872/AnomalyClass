"""YOLO27 unified model, detection model, and fabric-defect-aware trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights

from .loss import YOLO27E2ELoss
from .modules import ForegroundAlignmentBranch, YOLO27Detect

# Register custom classes for PyTorch safe checkpoint serialization when supported
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        YOLO27Detect,
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

        # Upgrade final Detect head to YOLO27Detect (dual-scale fixed balance + training-only foreground branch)
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
        return v8DetectionLoss(self)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([YOLO27DetectionModel])


class YOLO27Trainer(DetectionTrainer):
    """YOLO27 trainer supporting YOLO26/27 MuSGD recipe and fabric19 ``normaltrain`` / ``normalval``."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        # Default to MuSGD (Muon + SGD hybrid optimizer introduced in YOLO26/27) if left at 'auto'
        overrides.setdefault("optimizer", "MuSGD")
        overrides.setdefault("close_mosaic", 10)
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
        """Load dataset config and automatically append ``normaltrain`` / ``normalval`` background folders."""
        data = super().get_dataset()
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
        """Align model class names with dataset names before optional weight loading.

        This hook exists so that when ``model.load(weights)`` is called, the model's
        detection head ``names`` already matches the dataset YAML, which avoids
        class-name mismatch warnings / head re-initialization surprises.
        """
        names = self.data.get("names")
        if names is not None and hasattr(model, "names"):
            try:
                model.names = names
            except Exception as e:  # pragma: no cover - defensive
                LOGGER.warning(f"YOLO27Trainer: failed to set model names before load: {e}")
        return model

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: torch.nn.Module | None = None,
        verbose: bool = True,
    ) -> YOLO27DetectionModel:
        """Build ``YOLO27DetectionModel`` for training."""
        model = self.set_model_names_for_load(
            YOLO27DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model


class YOLO27(YOLO):
    """Unified Ultralytics-compatible interface for YOLO27n / YOLO27n-Lite / YOLO27s models.

    Examples:
        >>> from yolo27 import YOLO27
        >>> model = YOLO27("models/yolo27n.yaml")          # standard YOLO27n (~2.94M params, ~7.2 GFLOPs)
        >>> model_lite = YOLO27("models/yolo27n-lite.yaml") # ultra-light YOLO27n-Lite (~0.75M params, ~3.0 GFLOPs)
        >>> model.train(data="dataset/fabric19.yaml", epochs=150, imgsz=640)
        >>> results_nms = model.predict("sample.jpg")                 # default one-to-many + NMS
        >>> results_e2e = model.predict("sample.jpg", nms=False)      # NMS-free one-to-one end-to-end
    """

    def __init__(self, model: str | Path = "models/yolo27n.yaml", task: str | None = "detect", verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        base_map = dict(super().task_map)
        base_map["detect"] = {
            "model": YOLO27DetectionModel,
            "trainer": YOLO27Trainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        }
        return base_map