import json
from pathlib import Path
import pytest

import deploy_to_coralboard


def test_derive_symlink_name():
    p1 = Path("models/torq/261030-0800/torq_yolov8n_261030-0800.vmfb")
    assert deploy_to_coralboard.derive_symlink_name(p1) == "torq_yolov8n.vmfb"

    p2 = Path("models/torq/261030-0800/torq_yolov8n_261030-0800.json")
    assert deploy_to_coralboard.derive_symlink_name(p2) == "torq_yolov8n.json"

    p3 = Path("models/torq/torq_yolov8n.vmfb")
    assert deploy_to_coralboard.derive_symlink_name(p3) == "torq_yolov8n.vmfb"


def test_detect_latest_model_and_labels(tmp_path):
    models_dir = tmp_path / "models" / "torq"
    run_dir = models_dir / "261030-0800"
    run_dir.mkdir(parents=True)

    vmfb_file = run_dir / "torq_yolov8n_261030-0800.vmfb"
    vmfb_file.write_text("dummy-binary")

    labels_file = run_dir / "torq_yolov8n_261030-0800.json"
    labels_file.write_text(json.dumps({"0": "person"}))

    detected_model = deploy_to_coralboard.detect_latest_model(models_dir)
    assert detected_model == vmfb_file

    detected_labels = deploy_to_coralboard.detect_matching_labels(detected_model, models_dir)
    assert detected_labels == labels_file


def test_deploy_artifacts_local(tmp_path):
    model_file = tmp_path / "torq_yolov8n_261030-0800.vmfb"
    model_file.write_text("model-content")

    labels_file = tmp_path / "torq_yolov8n_261030-0800.json"
    labels_file.write_text(json.dumps({"0": "person"}))

    labels_txt = tmp_path / "torq_yolov8n_261030-0800.txt"
    labels_txt.write_text("person\n")

    dest_dir = tmp_path / "deployed"

    success = deploy_to_coralboard.deploy_artifacts(
        model_path=model_file,
        labels_path=labels_file,
        deploy_dir=str(dest_dir),
    )
    assert success is True
    assert (dest_dir / "torq_yolov8n_261030-0800.vmfb").is_file()
    assert (dest_dir / "torq_yolov8n.vmfb").is_file()
    assert (dest_dir / "torq_yolov8n_261030-0800.json").is_file()
    assert (dest_dir / "torq_yolov8n.json").is_file()
    assert (dest_dir / "labels.json").is_file()
    assert (dest_dir / "labels.txt").is_file()
