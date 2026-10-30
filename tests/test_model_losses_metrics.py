import copy

import numpy as np
import pytest
import torch

from anomalyncd.losses import Objectives, SupConLoss, info_nce_loss
from anomalyncd.metrics import classification_metrics, image_metrics, merge_image_probabilities
from anomalyncd.model import load_model


def test_mgvit_batch_one_and_small_mask_gradients(smoke_config):
    model = load_model(smoke_config, 19, load_pretrained=False)
    for batch in (1, 3):
        images = torch.randn(batch, 3, 32, 32)
        masks = torch.zeros(batch, 1, 32, 32)
        masks[:, :, 2:3, 2:3] = 1
        projection, logits = model(images, masks)
        assert projection.shape == (batch, 16)
        assert all(x.shape == (batch, 19) for x in logits)
        model.zero_grad()
        sum(x.square().mean() for x in logits).backward()
        assert all(torch.isfinite(x).all() for x in logits)
    masks.zero_()
    assert torch.isfinite(model(images, masks)[0]).all()


def test_contrastive_singleton_and_no_positive_are_finite():
    features = torch.randn(2, 4, requires_grad=True)
    loss = info_nce_loss(features)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    loss = SupConLoss()(torch.randn(1, 2, 4, requires_grad=True), torch.tensor([0]))
    assert float(loss) == 0 and torch.isfinite(loss)


@pytest.mark.parametrize("known_mask", [[True, False], [False, False], [True, True]])
def test_objectives_handle_mixed_and_single_source_batches(smoke_config, known_mask):
    torch.manual_seed(4)
    criterion = Objectives(smoke_config, 19, 10)
    projection = torch.randn(4, 16, requires_grad=True)
    logits = [torch.randn(4, 19, requires_grad=True) for _ in range(2)]
    labelled = torch.tensor(known_mask)
    targets = torch.tensor([2 if x else -100 for x in known_mask])
    loss, parts, heads = criterion(projection, logits, targets, labelled, 0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(projection.grad).all()
    assert len(heads) == 2 and all(np.isfinite(heads))


def test_ncd_loss_invariant_to_novel_label_id_and_permutation(smoke_config):
    cfg = copy.deepcopy(smoke_config)
    cfg["mode"] = "discovery"
    torch.manual_seed(9)
    projection = torch.randn(6, 16)
    logits = [torch.randn(6, 19) for _ in range(2)]
    labelled = torch.tensor([True, False, False])
    a, _, _ = Objectives(cfg, 19, 10)(projection, logits, torch.tensor([3, -100, -100]), labelled, 0)
    # Changing hidden novel IDs must not affect any training loss.
    b, _, _ = Objectives(cfg, 19, 10)(projection, logits, torch.tensor([3, 17, 11]), labelled, 0)
    torch.testing.assert_close(a, b)


def test_area_merge_uses_image_id_not_filename_or_first_label():
    records = [{"image_id": "same", "class_id": 0, "mask_area": 1},
               {"image_id": "same", "class_id": 1, "mask_area": 10000},
               {"image_id": "other", "class_id": 0, "mask_area": 4}]
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    merged, targets = merge_image_probabilities(records, probs, area_temperature=50)
    assert targets == [[0, 1], [0]] and len(merged) == 2
    assert merged[0][1] > 0.8
    metrics, predictions = image_metrics(targets, np.array(merged), threshold=0.15, n_classes=19)
    assert predictions == [[0, 1], [0, 1]]
    assert metrics["micro_f1"] == pytest.approx(6 / 7)


def test_region_metrics_perfect_and_chance_shapes():
    probs = np.eye(19)
    result = classification_metrics(np.arange(19), probs)
    assert result["accuracy"] == result["macro_f1"] == result["nmi"] == 1
    assert len(result["per_class"]) == 19


def test_dino_position_grid_resize_preserves_cls_and_rejects_bad_shapes():
    from anomalyncd.model import resize_checkpoint_pos_embed

    pos = torch.arange(26 * 8, dtype=torch.float32).reshape(1, 26, 8)
    target = torch.zeros(1, 10, 8)
    resized = resize_checkpoint_pos_embed({"pos_embed": pos}, target)["pos_embed"]
    assert resized.shape == target.shape
    torch.testing.assert_close(resized[:, 0], pos[:, 0])
    assert pos.shape == (1, 26, 8)  # input checkpoint is not mutated
    with pytest.raises(ValueError, match="方形"):
        resize_checkpoint_pos_embed({"pos_embed": torch.zeros(1, 25, 8)}, target)
    with pytest.raises(ValueError, match="维度"):
        resize_checkpoint_pos_embed({"pos_embed": torch.zeros(1, 26, 7)}, target)


def test_local_checkpoint_strict_loading_and_model_geometry_validation(tmp_path, smoke_config):
    import copy

    from anomalyncd.model import validate_model_config

    cfg = copy.deepcopy(smoke_config)
    model = load_model(cfg, 19, load_pretrained=False)
    path = tmp_path / "local_backbone.pth"
    torch.save(model.backbone.state_dict(), path)
    cfg["model"]["pretrained_path"] = str(path)
    loaded = load_model(cfg, 19, allow_random_init=True)
    for key, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, loaded.backbone.state_dict()[key])
    bad = torch.load(path, weights_only=True)
    bad.pop("pos_embed")
    torch.save(bad, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(cfg, 19, allow_random_init=True)
    cfg["model"]["image_size"] = 33
    with pytest.raises(ValueError, match="整除"):
        validate_model_config(cfg)
    cfg["model"]["image_size"] = 32
    cfg["model"]["mask_layers"] = 5
    with pytest.raises(ValueError, match="mask_layers"):
        validate_model_config(cfg)
