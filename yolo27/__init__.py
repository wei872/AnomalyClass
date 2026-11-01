"""YOLO27n / YOLO27n-Dual / YOLO27n-Lite implementation for fabric defect detection.

Reference:
    https://docs.ultralytics.com/zh/models/yolo27
"""

import ultralytics.nn.modules as _ult_modules
import ultralytics.nn.tasks as _ult_tasks

from .loss import YOLO27DetectionLoss, YOLO27E2ELoss
from .model import YOLO27, YOLO27DetectionModel, YOLO27Trainer, YOLO27Validator
from .modules import ForegroundAlignmentBranch, YOLO27Detect
from .weights import resolve_donor_checkpoint, transfer_pretrained_weights

# Register custom modules in Ultralytics namespaces so torch.load / parse_model resolve cleanly
setattr(_ult_modules, "YOLO27Detect", YOLO27Detect)
setattr(_ult_modules, "ForegroundAlignmentBranch", ForegroundAlignmentBranch)
setattr(_ult_tasks, "YOLO27Detect", YOLO27Detect)
setattr(_ult_tasks, "ForegroundAlignmentBranch", ForegroundAlignmentBranch)

__all__ = (
    "YOLO27",
    "YOLO27Detect",
    "YOLO27DetectionLoss",
    "YOLO27DetectionModel",
    "YOLO27E2ELoss",
    "YOLO27Trainer",
    "YOLO27Validator",
    "ForegroundAlignmentBranch",
    "transfer_pretrained_weights",
    "resolve_donor_checkpoint",
)
