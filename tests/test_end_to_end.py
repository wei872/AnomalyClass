import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from anomalyncd.detector import prepare_detector_dataset
from anomalyncd.end_to_end import predict_fabricncd, save_bundle
from anomalyncd.engine import train


def test_detector_dataset_uses_normals_and_hides_ncd_class_ids(synthetic, tmp_path):
    output = tmp_path / "detector_data"
    manifest = prepare_detector_dataset(synthetic["yaml"], output, synthetic["raw"],
                                        class_agnostic=True, mode="discovery",
                                        known_class_ids=list(range(10)), progress=lambda _: None)
    assert manifest["nc"] == 1 and manifest["names"] == ["defect"]
    assert manifest["stats"]["train"]["normal_images"] == 1
    labels = list((output / "labels/train").rglob("*.txt"))
    assert labels
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert line.split()[0] == "0"
    normal_labels = list((output / "labels/train/normal").rglob("*.txt"))
    assert normal_labels and all(not p.read_text(encoding="utf-8").strip() for p in normal_labels)
    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == 1
    with pytest.raises(ValueError, match="class_agnostic"):
        prepare_detector_dataset(synthetic["yaml"], tmp_path / "bad", synthetic["raw"],
                                 class_agnostic=False, mode="discovery", progress=lambda _: None)


class FakeBoxes:
    def __init__(self, boxes):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        self.conf = torch.tensor([0.9] * len(boxes), dtype=torch.float32)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, **kwargs):
        return [SimpleNamespace(boxes=FakeBoxes(self.boxes))]


def _bundle(smoke_config, tmp_path):
    train(smoke_config, allow_random_init=True)
    classifier = Path(smoke_config["experiment"]["output_dir"]) / "last.pt"
    dummy_detector = tmp_path / "dummy_detector.pt"
    dummy_detector.write_bytes(b"fake")
    bundle = tmp_path / "bundle"
    save_bundle(bundle, dummy_detector, classifier, smoke_config)
    return bundle


def test_end_to_end_raw_image_outputs_all_four_stages(smoke_config, synthetic, tmp_path):
    bundle = _bundle(smoke_config, tmp_path)
    image = sorted((synthetic["raw"] / "images/val").rglob("*.png"))[0]
    output = tmp_path / "pred"
    result = predict_fabricncd(bundle, image, output, device_name="cpu",
                               detector=FakeDetector([[5, 8, 20, 20], [40, 28, 55, 42]]))
    assert result["pipeline"] == ["detection", "anomaly_localization", "anomaly_classification", "anomaly_score"]
    assert result["is_anomaly"] is True and result["prediction"] == "anomaly"
    assert result["anomaly_score"] == pytest.approx(0.9)
    assert len(result["detections"]) == 2 and len(result["regions"]) == 2
    assert all("prediction_id" in r and "anomaly_score" in r for r in result["regions"])
    assert result["image_prediction_ids"]
    assert Path(result["localization"]["heatmap"]).is_file()
    assert Path(result["localization"]["overlay"]).is_file()
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert saved["is_anomaly"] is True


def test_end_to_end_normal_image_is_not_forced_into_defect_class(smoke_config, synthetic, tmp_path):
    bundle = _bundle(smoke_config, tmp_path)
    image = sorted((synthetic["raw"] / "normal/images/val").rglob("*.png"))[0]
    result = predict_fabricncd(bundle, image, tmp_path / "normal_pred", device_name="cpu",
                               detector=FakeDetector([]))
    assert result["is_anomaly"] is False and result["prediction"] == "normal"
    assert result["anomaly_score"] == 0.0
    assert result["regions"] == [] and result["image_prediction_ids"] == []
    assert result["localization_map_max"] == 0.0
