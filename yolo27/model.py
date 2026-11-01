"""YOLO27 high-level model, trainer, and validator interfaces.

Integrates ``YOLO27DetectionModel`` seamlessly into Ultralytics' training, validation,
prediction, and export pipelines across both ``ultralytics==8.4.33`` and ``>=8.4.197``:
    - Upgraded ``YOLO27Detect`` head (``reg_max=1`` DFL-free regression + folded
      per-channel/cross-scale calibration + training-only ``ForegroundAlignmentBranch``).
    - Zero-shock Net2Wider warm-start from ``yolo26n.pt`` / ``yolo26s.pt`` with
      automatic local, sibling, dataset-root, cache, and asset-download resolution
      (protecting against Ultralytics ``Model.load(True)`` scratch hijack!).
    - Official YOLO27 inference convention: defaults to ``one2many + NMS`` for maximum
      mAP, with ``nms=False`` switching cleanly to end-to-end ``one2one`` inference.
    - Built-in support for ``normaltrain`` / ``normalval`` defect-free fabric samples in
      ``dataset/fabric19.yaml`` via ``use_normal=True``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from .loss import YOLO27DetectionLoss, YOLO27E2ELoss
from .modules import ForegroundAlignmentBranch, YOLO27Detect
from .weights import resolve_donor_checkpoint, transfer_pretrained_weights


def _merge_split_paths(primary: Any, extra: Any) -> list[str]:
    """Combine primary defect split path(s) with defect-free ``normal*`` split path(s)."""
    out: list[str] = []
    for item in (primary, extra):
        if not item:
            continue
        if isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item if x)
        else:
            out.append(str(item))
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _upgrade_detect_head(model: DetectionModel, reg_max_override: int | None = None) -> None:
    """Replace terminal ``Detect`` head with ``YOLO27Detect`` while preserving graph metadata."""
    head = model.model[-1]
    if isinstance(head, Detect) and not isinstance(head, YOLO27Detect):
        ch = tuple(int(m[0].conv.in_channels) for m in head.cv2)
        end2end = hasattr(head, "one2one_cv2") and head.one2one_cv2 is not None
        reg_max = int(reg_max_override if reg_max_override is not None else getattr(head, "reg_max", 1))
        new_head = YOLO27Detect(nc=head.nc, end2end=end2end, reg_max=reg_max, ch=ch)
        new_head.i = head.i
        new_head.f = head.f
        new_head.type = f"{YOLO27Detect.__module__}.{YOLO27Detect.__qualname__}"
        new_head.np = sum(x.numel() for x in new_head.parameters())
        model.model[-1] = new_head


class YOLO27DetectionModel(DetectionModel):
    """Ultralytics ``DetectionModel`` subclass powered by ``YOLO27Detect`` and ``YOLO27E2ELoss``."""

    def __init__(
        self,
        cfg: str | Path | dict[str, Any] = "models/yolo27n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        if isinstance(cfg, (str, Path)):
            from ultralytics.nn.tasks import yaml_model_load

            cfg_dict = yaml_model_load(str(cfg))
        else:
            cfg_dict = deepcopy(cfg)

        yaml_reg_max = int(cfg_dict.get("reg_max", 1))

        super().__init__(cfg=cfg_dict, ch=ch, nc=nc, verbose=False)
        _upgrade_detect_head(self, reg_max_override=yaml_reg_max)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            self.model.eval()
            m.training = True
            preds = self.forward(torch.zeros(1, ch, s, s))
            feats = preds["one2many"]["feats"] if isinstance(preds, dict) else preds["feats"]
            m.stride = torch.tensor([s / x.shape[-2] for x in feats])
            self.stride = m.stride
            self.model.train()
            m.bias_init()

        from ultralytics.utils.torch_utils import initialize_weights

        initialize_weights(self)
        if isinstance(m, YOLO27Detect):
            m.bias_init()

        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Initialize balanced ``YOLO27E2ELoss`` (with live FG alignment) or ``YOLO27DetectionLoss``."""
        class_weights = getattr(self, "_class_weights", None)
        if getattr(self, "end2end", False) or hasattr(self.model[-1], "one2one_cv2"):
            return YOLO27E2ELoss(self, class_weights=class_weights)
        return YOLO27DetectionLoss(self, class_weights=class_weights)


class YOLO27Validator(DetectionValidator):
    """Validator supporting both ``nms=True`` (one2many+NMS) and ``nms=False`` (one2one E2E)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        self._use_normal: bool = False
        if isinstance(args, dict) and "use_normal" in args:
            args = dict(args)
            self._use_normal = bool(args.pop("use_normal"))
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "fg_loss"

    def check_det_dataset(self, dataset: str) -> dict[str, Any]:
        data = super().check_det_dataset(dataset)
        if self._use_normal:
            if data.get("normaltrain"):
                data["train"] = _merge_split_paths(data.get("train"), data.get("normaltrain"))
            if data.get("normalval"):
                data["val"] = _merge_split_paths(data.get("val"), data.get("normalval"))
        return data


class YOLO27Trainer(DetectionTrainer):
    """Trainer building ``YOLO27DetectionModel`` with automatic Net2Wider warm-start from ``yolo26*.pt``."""

    _preset_use_normal: bool = False
    _preset_donor_weights: str | None = None
    _preset_allow_warmstart: bool | None = None

    def __init__(self, cfg=DEFAULT_CFG_DICT, overrides: dict[str, Any] | None = None, _callbacks=None) -> None:
        overrides = dict(overrides or {})
        self._use_normal: bool = bool(overrides.pop("use_normal", self._preset_use_normal))
        self._donor_weights: str | None = overrides.pop("donor_weights", self._preset_donor_weights)

        raw_pretrained = overrides.get("pretrained", True)
        if isinstance(raw_pretrained, (str, Path)) and self._donor_weights is None:
            self._donor_weights = str(raw_pretrained)
            overrides["pretrained"] = False

        if self._preset_allow_warmstart is not None:
            self._allow_warmstart: bool = bool(self._preset_allow_warmstart)
        else:
            self._allow_warmstart = raw_pretrained is not False

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "fg_loss"

    def get_dataset(self) -> dict[str, Any]:
        """Load dataset dict and optionally merge ``normaltrain`` / ``normalval`` splits."""
        data = super().get_dataset()
        if self._use_normal:
            if data.get("normaltrain"):
                data["train"] = _merge_split_paths(data.get("train"), data.get("normaltrain"))
                LOGGER.info(f"YOLO27Trainer: merged normaltrain -> {data['train']}")
            if data.get("normalval"):
                data["val"] = _merge_split_paths(data.get("val"), data.get("normalval"))
                LOGGER.info(f"YOLO27Trainer: merged normalval -> {data['val']}")
        return data

    def _collect_dataset_search_dirs(self) -> list[str]:
        """Gather dataset directories so ``yolo26n.pt`` stored beside datasets is found automatically."""
        dirs: list[str] = []
        data = getattr(self, "data", None)
        if isinstance(data, dict):
            for k in ("path", "train", "val"):
                v = data.get(k)
                if isinstance(v, (str, Path)):
                    dirs.append(str(v))
                elif isinstance(v, (list, tuple)):
                    dirs.extend(str(item) for item in v if isinstance(item, (str, Path)))
        return dirs

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> YOLO27DetectionModel:
        """Instantiate ``YOLO27DetectionModel`` and apply Net2Wider warm-start when starting from YAML."""
        model = YOLO27DetectionModel(
            cfg or "models/yolo27n.yaml",
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
        )
        if weights is not None:
            if isinstance(weights, nn.Module) and isinstance(getattr(weights, "model", [None])[-1], YOLO27Detect):
                model.load(weights)
            else:
                transfer_pretrained_weights(weights, model, verbose=verbose and RANK in {-1, 0})
        elif self._allow_warmstart and not getattr(self.args, "resume", False):
            scale = str(model.yaml.get("scale", "n") or "n")
            donor = resolve_donor_checkpoint(
                scale=scale,
                explicit_donor=self._donor_weights,
                allow_download=True,
                extra_dirs=self._collect_dataset_search_dirs(),
            )
            if donor is not None:
                transfer_pretrained_weights(donor, model, verbose=verbose and RANK in {-1, 0})
            elif RANK in {-1, 0}:
                LOGGER.warning(
                    f"WARNING: Could not find or download yolo26{scale}.pt for Net2Wider warm-start! "
                    f"Training from scratch will converge slower and lower mAP than YOLO26n. "
                    f"Place yolo26{scale}.pt in the project root or pass --donor path/to/yolo26{scale}.pt."
                )
        return model

    def get_validator(self) -> YOLO27Validator:
        """Return ``YOLO27Validator`` matching the 4-component loss layout."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "fg_loss"
        return YOLO27Validator(
            self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args), _callbacks=self.callbacks
        )


def _translate_nms_kwarg(
    kwargs: dict[str, Any],
    model: nn.Module | None,
    default_nms: bool | None = True,
) -> dict[str, Any]:
    """Translate ``nms=True/False/None`` cleanly across Ultralytics 8.4.33 and >=8.4.197."""
    out = dict(kwargs)
    nms_flag = out.pop("nms", default_nms)
    use_end2end = nms_flag is False

    if model is not None:
        head = model.model[-1] if hasattr(model, "model") else None
        if isinstance(head, YOLO27Detect):
            head.end2end = use_end2end

    if "end2end" in DEFAULT_CFG_DICT:
        out["end2end"] = use_end2end
    elif "nms" in DEFAULT_CFG_DICT:
        if nms_flag is not None:
            out["nms"] = bool(nms_flag)
    return out


class YOLO27(YOLO):
    """Drop-in ``YOLO`` user interface for ``YOLO27n`` / ``YOLO27n-Dual`` / ``YOLO27n-Lite``."""

    def __init__(self, model: str | Path = "models/yolo27n.yaml", task: str | None = None, verbose: bool = False):
        super().__init__(model=str(model), task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detect task to ``YOLO27DetectionModel``, ``YOLO27Trainer``, and ``YOLO27Validator``."""
        base_map = dict(super().task_map)
        base_map["detect"] = {
            "model": YOLO27DetectionModel,
            "trainer": YOLO27Trainer,
            "validator": YOLO27Validator,
            "predictor": DetectionPredictor,
        }
        return base_map

    def train(self, trainer=None, **kwargs: Any):
        """Train model while preventing ``Model.load(True)`` from hijacking Net2Wider warm-start on YAML configs."""
        train_kwargs = dict(kwargs)
        use_normal = bool(train_kwargs.pop("use_normal", False))
        donor_weights = train_kwargs.pop("donor_weights", None)

        is_yaml_start = str(self.overrides.get("model", "")).lower().endswith((".yaml", ".yml")) and not self.ckpt
        raw_pretrained = train_kwargs.get("pretrained", True)
        if is_yaml_start:
            if isinstance(raw_pretrained, (str, Path)) and donor_weights is None:
                donor_weights = str(raw_pretrained)
            allow_warmstart = raw_pretrained is not False
            # Critical: pass pretrained=False into Model.train() so Model.load(True) does NOT
            # overwrite self.ckpt with random scratch self.model and bypass YOLO27Trainer.get_model()!
            train_kwargs["pretrained"] = False
        else:
            allow_warmstart = None

        base_trainer_cls = trainer or YOLO27Trainer

        class _BoundYOLO27Trainer(base_trainer_cls):
            _preset_use_normal = use_normal
            _preset_donor_weights = str(donor_weights) if donor_weights is not None else None
            _preset_allow_warmstart = allow_warmstart

        return super().train(trainer=_BoundYOLO27Trainer, **train_kwargs)

    def val(self, validator=None, **kwargs: Any):
        """Validate model; defaults to ``nms=True`` (one-to-many + NMS) per YOLO27 spec."""
        use_normal = bool(kwargs.pop("use_normal", False))
        kwargs = _translate_nms_kwarg(kwargs, self.model, default_nms=True)
        if use_normal:
            custom_val = validator or YOLO27Validator

            class _NormalEnabledValidator(custom_val):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self._use_normal = True

            return super().val(validator=_NormalEnabledValidator, **kwargs)
        return super().val(validator=validator, **kwargs)

    def predict(self, source=None, stream: bool = False, predictor=None, **kwargs: Any):
        """Run inference; defaults to ``nms=True`` (pass ``nms=False`` for E2E one-to-one)."""
        kwargs = _translate_nms_kwarg(kwargs, self.model, default_nms=True)
        return super().predict(source=source, stream=stream, predictor=predictor, **kwargs)

    def export(self, **kwargs: Any):
        """Export model; defaults to raw one-to-many head (pass ``nms=False`` for E2E one-to-one)."""
        kwargs = _translate_nms_kwarg(kwargs, self.model, default_nms=None)
        return super().export(**kwargs)
