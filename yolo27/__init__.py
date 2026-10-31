"""YOLO27n implementation based on official Ultralytics YOLO27 architecture specifications.

Reference:
    https://docs.ultralytics.com/zh/models/yolo27
"""

from .loss import YOLO27DetectionLoss, YOLO27E2ELoss
from .model import YOLO27, YOLO27DetectionModel, YOLO27Trainer, YOLO27Validator
from .modules import FabricStripEnhancer, ForegroundAlignmentBranch, YOLO27Detect
from .weights import resolve_donor_checkpoint, transfer_pretrained_weights

__all__ = (
    "YOLO27",
    "YOLO27Detect",
    "YOLO27DetectionLoss",
    "YOLO27DetectionModel",
    "YOLO27E2ELoss",
    "YOLO27Trainer",
    "YOLO27Validator",
    "FabricStripEnhancer",
    "ForegroundAlignmentBranch",
    "transfer_pretrained_weights",
    "resolve_donor_checkpoint",
)
