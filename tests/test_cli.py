import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_cli(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_prepare_train_evaluate_predict_cli(smoke_config, synthetic, tmp_path):
    config_path = tmp_path / "config.yaml"
    smoke_config["experiment"]["output_dir"] = str(tmp_path / "cli_run")
    config_path.write_text(yaml.safe_dump(smoke_config, allow_unicode=True), encoding="utf-8")
    checked = run_cli("prepare_data.py", "--dataset-config", synthetic["base"] / "fabric19.yaml",
                      "--data-root", synthetic["raw"], "--check-only")
    assert "OK" in checked
    dry = run_cli("train.py", "--config", config_path, "--dry-run")
    assert "mode=supervised" in dry
    run_cli("train.py", "--config", config_path, "--epochs", "1", "--eval-every", "0",
            "--device", "cpu", "--allow-random-init")
    checkpoint = tmp_path / "cli_run" / "last.pt"
    eval_dir = tmp_path / "cli_eval"
    run_cli("evaluate.py", "--checkpoint", checkpoint, "--prepared-root", synthetic["prepared"],
            "--output", eval_dir, "--device", "cpu", "--batch-size", "3", "--top1")
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["region_all"]["count"] == 20
    assert metrics["config"]["training"]["batch_size"] == 3
    assert metrics["config"]["inference"]["image_threshold"] is None
    image = sorted((synthetic["raw"] / "images/val").rglob("*.png"))[0]
    output = tmp_path / "prediction.json"
    run_cli("predict.py", "--checkpoint", checkpoint, "--image", image, "--whole-image",
            "--device", "cpu", "--top1", "--output", output)
    prediction = json.loads(output.read_text(encoding="utf-8"))
    assert prediction["class_order"] == list(range(19))
    assert len(prediction["regions"]) == 1
