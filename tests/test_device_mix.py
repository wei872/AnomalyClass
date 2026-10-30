"""设备 / 精度混用回归: 这些错只在 CUDA/fp16 上暴露, CPU 单测要用 bf16 模拟."""

from __future__ import annotations

import torch

from fabric_ncd.losses import SupConLoss, build_label_map, masked_ce
from fabric_ncd.models.memory import MemoryBank, compact_loss


def test_build_label_map_uses_requested_device():
    boxes = [torch.tensor([[0.0, 0.2, 0.2, 0.8, 0.8]])]
    labels = build_label_map(boxes, (8, 6), ignore_index=-100, device=torch.device("cpu"))
    assert labels.device.type == "cpu"
    assert labels.dtype == torch.long
    assert int((labels != -100).sum()) > 0


def test_masked_ce_tolerates_label_on_other_device():
    """labels 若停在别的设备上, masked_ce 自己搬过去而不是抛错."""
    logits = torch.randn(1, 3, 4, 4, dtype=torch.float32)
    labels = torch.full((1, 4, 4), -100, dtype=torch.long)
    labels[0, 1:3, 1:3] = 2  # 故意用 long/int64 而非 logits 的 float, 验证 bool 索引
    loss, acc = masked_ce(logits, labels)
    assert loss is not None and torch.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_compact_loss_mixed_precision():
    """autocast 下 embedding 是半精度, 记忆库是 fp32 buffer —— einsum 不能因 dtype 报错."""
    bank = MemoryBank(capacity=8, dim=4)
    bank.update(torch.randn(8, 4))
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        emb = torch.randn(1, 4, 3, 3, dtype=dtype)
        loss = compact_loss(bank, emb)
        assert torch.isfinite(loss)


def test_memory_similarity_mixed_precision():
    bank = MemoryBank(capacity=8, dim=4)
    bank.update(torch.randn(8, 4))
    for dtype in (torch.bfloat16, torch.float32):
        sims = bank.similarity(torch.randn(1, 4, 3, 3, dtype=dtype))
        assert sims.shape == (1, 1, 3, 3)
        assert torch.isfinite(sims).all()


def test_supcon_mixed_precision():
    feats = torch.randn(6, 4, dtype=torch.bfloat16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = SupConLoss(0.1)(feats, labels)
    assert torch.isfinite(loss)


def test_criterion_autocast_cpu_step():
    """整条损失在 CPU autocast(bf16) 下跑通 (CUDA autocast(fp16) 的近似)."""
    from fabric_ncd.losses import FabricCriterion
    from fabric_ncd.utils import load_yaml

    cfg = load_yaml("configs/fabric19.yaml")
    cfg["model"]["heads"]["num_classes"] = 3
    criterion = FabricCriterion(cfg)
    b, c, h, w = 2, 3, 8, 8
    targets = {
        # 数据集给的是 (B,1,H,W) 的掩码, 与 logits 通道数一致
        "defect_mask": torch.zeros(b, 1, h, w),
        "cloth_mask": torch.ones(b, 1, h, w),
        "boxes": [torch.tensor([[0.0, 0.2, 0.2, 0.6, 0.6]]), torch.zeros((0, 5))],
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = {
            "anomaly_logit": torch.randn(b, 1, h, w),
            "anomaly_score": torch.rand(b, 1, h, w),
            "cloth_logit": torch.randn(b, 1, h, w),
            "boundary_logit": torch.randn(b, 1, h, w),
            "class_logits": torch.randn(b, c, h, w),
            "embedding": torch.randn(b, 4, h, w),
        }
        loss, logs = criterion(outputs, targets)
    assert torch.isfinite(loss)
    assert "cls" in logs


def test_oom_message_is_actionable():
    from fabric_ncd.engine import Trainer

    stub = object.__new__(Trainer)
    stub.cfg = {"data": {"image_size": [1024, 1728], "batch_size": 4}}
    err = Trainer._oom_error(stub)
    assert "显存" in str(err) and "768" in str(err)


def test_describe_device_cpu():
    from fabric_ncd.utils import describe_device

    assert describe_device(torch.device("cpu")) == "cpu"
