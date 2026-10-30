"""模型结构 / 损失 / 记忆库测试 (小分辨率, CPU 可跑)."""

import pytest
import torch

from fabric_ncd.losses import FabricCriterion, build_label_map, sigmoid_focal_loss
from fabric_ncd.models.fabric_ncd import FabricNCD
from fabric_ncd.utils import load_yaml


def tiny_config(size=64, pretrained=False, num_classes=5):
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["data"]["image_size"] = [size, size]
    cfg["model"]["backbone"]["pretrained"] = pretrained
    cfg["model"]["heads"]["num_classes"] = num_classes
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = tiny_config()
    m = FabricNCD(cfg)
    m.eval()
    return m


def test_forward_shapes(model):
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["anomaly_logit"].shape == (2, 1, 8, 8)
    assert out["cloth_logit"].shape == (2, 1, 16, 16)
    assert out["class_logits"].shape == (2, 5, 8, 8)
    assert out["anomaly_score"].min() >= 0
    assert out["anomaly_score"].max() <= 1


def test_weave_stats_range(model):
    x = torch.rand(3, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["weave"]["period_x"].shape == (3,)
    assert (out["weave"]["period_x"] >= 2).all()


def test_geography_uses_border_prior_when_no_background():
    """整幅都是布 (布面概率恒为 1) 时, 地理编码仍应给出空间变化 (来自到边框距离)."""
    from fabric_ncd.models.glocal import GeographyEmbedding

    geo = GeographyEmbedding(8, pool_sizes=(3, 7), use_xy=False, border_prior=True)
    feat = torch.ones(1, 8, 16, 24)
    prob = torch.ones(1, 1, 16, 24)  # 全是布
    with torch.no_grad():
        out = geo(prob, feat)
    assert out.shape == feat.shape
    assert torch.isfinite(out).all()
    # 左侧边框与画面中央的响应必须不同, 否则等于没有边缘信息
    assert float((out[:, :, :, 0] - out[:, :, :, feat.shape[-1] // 2]).abs().max()) > 1e-4


def test_backward_and_loss(model):
    cfg = tiny_config()
    criterion = FabricCriterion(cfg)
    model.train()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    targets = {
        "defect_mask": torch.zeros(2, 1, 64, 64),
        "cloth_mask": torch.ones(2, 1, 64, 64),
        "boxes": [torch.tensor([[1.0, 0.2, 0.2, 0.4, 0.4]]), torch.zeros(0, 5)],
    }
    targets["defect_mask"][0, 0, 12:28, 12:28] = 1
    loss, logs = criterion(out, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_memory_bank(model):
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    model.update_memory(x)
    assert model.memory.is_ready
    with torch.no_grad():
        out = model(x)
    assert out["memory_score"] is not None
    assert out["memory_score"].shape == (2, 1, 8, 8)
    model.memory.reset()
    assert not model.memory.is_ready


def test_build_label_map():
    boxes = [torch.tensor([[2.0, 0.25, 0.25, 0.75, 0.75]])]
    labels = build_label_map(boxes, (10, 10), ignore_index=-100, shrink=1.0)
    assert labels[0, 5, 5] == 2
    assert labels[0, 0, 0] == -100


def test_focal_loss_extremes():
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = sigmoid_focal_loss(logits, targets, reduction="mean")
    assert loss.item() < 1e-4


def test_lite_backbone_path():
    cfg = tiny_config()
    cfg["model"]["backbone"]["arch"] = "lite"
    m = FabricNCD(cfg).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 64))
    assert out["anomaly_logit"].shape[-2:] == (8, 8)


def test_memory_score_does_not_saturate():
    """记忆项必须平滑映射: 距离超过尺度时不能全堆到 1.0.

    否则正常图的 999 分位也是 1.0, 阈值退化为不可达, 验证集检出率恒为 0.
    """
    torch.manual_seed(0)
    cfg = tiny_config()
    m = FabricNCD(cfg)
    m.eval()
    # 造一个和当前嵌入相差很远的记忆库, 让距离远大于尺度
    with torch.no_grad():
        feats = torch.randn(64, m.memory.dim)
        m.memory.update(feats, use_kcenter=False)
    m.memory_distance_scale = 0.01
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out["memory_distance"] is not None
    assert float(out["memory_distance"].max()) > m.memory_distance_scale
    assert float(out["memory_score"].max()) < 1.0
    assert float(out["memory_score"].min()) >= 0.0
