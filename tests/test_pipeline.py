import copy
import json
from pathlib import Path

import pytest
import torch

import anomalyncd.engine as engine
from anomalyncd.engine import evaluate, load_checkpoint, train
from anomalyncd.inference import predict_image, read_proposals


def test_supervised_train_evaluate_checkpoint_and_predict(smoke_config, synthetic, tmp_path):
    metrics = train(smoke_config, allow_random_init=True)
    output = Path(smoke_config["experiment"]["output_dir"])
    assert metrics["region_all"]["count"] == 20
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()
    assert (output / "final_evaluation/region_predictions.csv").is_file()
    checkpoint = load_checkpoint(output / "best.pt")
    assert checkpoint["epoch"] in (1, 2) and checkpoint["names"] == synthetic["manifest"]["names"]
    history = json.loads((output / "history.json").read_text())
    best_epoch = max(history, key=lambda row: row["validation_macro_f1"])["epoch"]
    assert checkpoint["epoch"] == best_epoch
    assert checkpoint["allow_random_init"] is True
    evaluated = evaluate(output / "best.pt", synthetic["prepared"], tmp_path / "eval", "cpu")
    assert evaluated["region_all"]["count"] == 20
    image = sorted((synthetic["raw"] / "images/val").rglob("*.png"))[0]
    boxes = tmp_path / "detector.txt"
    boxes.write_text("-1 0.2 0.25 0.15 0.2 0.9\n0 0.75 0.72 0.16 0.21 0.8\n")
    prediction = predict_image(output / "best.pt", image, boxes=boxes, device_name="cpu", top1=True)
    assert len(prediction["regions"]) == 2 and len(prediction["image_prediction_ids"]) == 1
    assert prediction["class_order"] == list(range(19))
    assert prediction["random_backbone_smoke_test"] is True
    whole = predict_image(output / "best.pt", image, whole_image=True, device_name="cpu")
    assert len(whole["regions"]) == 1


def test_discovery_training_and_novel_only_inference(smoke_config, tmp_path):
    cfg = copy.deepcopy(smoke_config)
    cfg["mode"] = "discovery"
    cfg["discovery"]["known_class_ids"] = list(range(10))
    cfg["training"]["eval_every"] = 0
    cfg["experiment"]["output_dir"] = str(tmp_path / "ncd")
    metrics = train(cfg, allow_random_init=True)
    output = Path(cfg["experiment"]["output_dir"])
    assert not (output / "best.pt").exists()
    # The first image also has a known-class (7) second box.
    assert metrics["region_novel_subset"]["count"] == 9
    assert metrics["region_known_subset"]["count"] == 11
    assert metrics["region_novel_closed_set_diagnostic"]["local_class_order"] == list(range(10, 19))
    image = sorted(Path(cfg["data"]["prepared_root"]).rglob("images/val/*.png"))[0]
    result = predict_image(output / "last.pt", image, whole_image=True, device_name="cpu", novel_only=True)
    assert result["class_order"] == list(range(10, 19))
    assert all(10 <= x < 19 for x in result["image_prediction_ids"])


def test_resume_continues_original_schedule(smoke_config, tmp_path):
    cfg = copy.deepcopy(smoke_config)
    cfg["training"]["eval_every"] = 0
    cfg["experiment"]["output_dir"] = str(tmp_path / "interrupted")

    class StopAfterEpoch(Exception):
        pass

    real_train_epoch = engine.train_one_epoch
    calls = 0
    def stop_after_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StopAfterEpoch()
        return real_train_epoch(*args, **kwargs)

    engine.train_one_epoch = stop_after_one
    try:
        with pytest.raises(StopAfterEpoch):
            train(cfg, allow_random_init=True)
    finally:
        engine.train_one_epoch = real_train_epoch
    output = Path(cfg["experiment"]["output_dir"])
    assert load_checkpoint(output / "last.pt")["epoch"] == 1
    train(cfg, resume=output / "last.pt")
    resumed = load_checkpoint(output / "last.pt")
    assert resumed["epoch"] == 2 and resumed["scheduler"]["last_epoch"] == 2
    assert resumed["optimizer"]["state"]

    cfg["experiment"]["output_dir"] = str(tmp_path / "uninterrupted")
    train(cfg, allow_random_init=True)
    baseline = load_checkpoint(Path(cfg["experiment"]["output_dir"]) / "last.pt")
    for key in baseline["model"]:
        torch.testing.assert_close(baseline["model"][key], resumed["model"][key], rtol=0, atol=0)


def test_json_proposals_ignore_input_classes(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"boxes": [[10, 5, 30, 20]], "scores": [0.9]}))
    proposals = read_proposals(path, (100, 50))
    assert proposals[0]["bbox"] == pytest.approx((0.2, 0.25, 0.2, 0.3))
    with pytest.raises(ValueError, match="没有满足"):
        read_proposals(path, (100, 50), confidence_threshold=0.95)


def test_inference_rejects_invalid_boxes_and_nonempty_crop_directory(smoke_config, synthetic, tmp_path):
    from anomalyncd.engine import train

    train(smoke_config, allow_random_init=True)
    checkpoint = Path(smoke_config["experiment"]["output_dir"]) / "last.pt"
    image = sorted((synthetic["raw"] / "images/val").rglob("*.png"))[0]
    boxes = tmp_path / "bad.json"
    boxes.write_text(json.dumps({"boxes": [[80, 5, 90, 20]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="有效交集"):
        predict_image(checkpoint, image, boxes=boxes, device_name="cpu")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="save-crops"):
        predict_image(checkpoint, image, whole_image=True, device_name="cpu", save_crops=occupied)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"
