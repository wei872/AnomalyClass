"""YOLO27n implementation based on official Ultralytics YOLO27 architecture specifications.

Reference:
    https://docs.ultralytics.com/zh/models/yolo27
"""

from .loss import YOLO27E2ELoss
from .model import YOLO27, YOLO27DetectionModel, YOLO27Trainer
from .modules import ForegroundAlignmentBranch, YOLO27Detect
from .weights import resolve_donor_checkpoint, transfer_pretrained_weights

__all__ = (
    "YOLO27",
    "YOLO27Detect",
    "YOLO27DetectionModel",
    "YOLO27E2ELoss",
    "YOLO27Trainer",
    "ForegroundAlignmentBranch",
    "transfer_pretrained_weights",
    "resolve_donor_checkpoint",
)
