import os
import tempfile
from pathlib import Path
import pytest
import yaml

import config_utils


def test_load_config_default():
    cfg = config_utils.load_config()
    assert isinstance(cfg, dict)
    assert "frigate" in cfg
    assert "cvat" in cfg
    assert "profiles" in cfg


def test_load_config_custom(tmp_path):
    custom_yaml = tmp_path / "custom_config.yaml"
    custom_yaml.write_text(yaml.safe_dump({"frigate": {"limit": 123}}))

    cfg = config_utils.load_config(str(custom_yaml))
    assert cfg.get("frigate", {}).get("limit") == 123


def test_load_config_missing():
    cfg = config_utils.load_config("/non/existent/path.yaml")
    assert cfg == {}


@pytest.mark.parametrize(
    "mode_input, expected",
    [
        ("cvat-deficit", "cvat-deficit"),
        ("cvat_deficit", "cvat-deficit"),
        ("accuracy-deficit", "accuracy-deficit"),
        ("accuracy_deficit", "accuracy-deficit"),
        ("accuracy", "accuracy-deficit"),
        ("equal-split", "equal-split"),
        ("equal_split", "equal-split"),
        ("strict-equal", "strict-equal"),
        ("strict_equal", "strict-equal"),
        ("natural", "natural"),
        (None, "cvat-deficit"),
        ("", "cvat-deficit"),
    ],
)
def test_normalize_balance_mode_valid(mode_input, expected):
    assert config_utils.normalize_balance_mode(mode_input) == expected


def test_normalize_balance_mode_invalid():
    with pytest.raises(ValueError, match="Invalid balance_mode"):
        config_utils.normalize_balance_mode("random-unknown-mode")


def test_get_profile_resolution():
    p_local = config_utils.get_profile("local")
    assert p_local["name"] == "local"
    assert p_local["remote"] is False
    assert p_local["batch"] == 8
    assert p_local["patience"] == 50

    p_spark = config_utils.get_profile("spark")
    assert p_spark["name"] == "spark"
    assert p_spark["remote"] is True
    assert p_spark["patience"] == 50

    # Test hyphen and underscore normalization for spark-yolo26n
    p_26n_hyphen = config_utils.get_profile("spark-yolo26n")
    p_26n_under = config_utils.get_profile("spark_yolo26n")
    assert p_26n_hyphen["base_model"] == "yolo26n.pt"
    assert p_26n_under["base_model"] == "yolo26n.pt"
    assert p_26n_hyphen["remote"] is True
    assert p_26n_hyphen["patience"] == 75
    assert p_26n_under["patience"] == 75


def test_get_profile_default_patience_fallback(tmp_path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(yaml.safe_dump({"profiles": {"minimal": {"device": "cpu"}}}))
    p = config_utils.get_profile("minimal", config_path=str(custom_yaml))
    assert p["patience"] == 50


def test_coco_80_canonical_classes():
    classes = config_utils.COCO_80_CLASSES
    assert len(classes) == 80
    assert classes[0] == "person"
    assert classes[2] == "car"
    assert classes[16] == "dog"

    assert config_utils.get_coco_id("person") == 0
    assert config_utils.get_coco_id("PERSON") == 0
    assert config_utils.get_coco_id("dog") == 16
    assert config_utils.get_coco_id("traffic light") == 9
    assert config_utils.get_coco_id("traffic_light") == 9
    assert config_utils.get_coco_id("traffic-light") == 9
    assert config_utils.get_coco_id("non_existent_label_xyz") is None


def test_dump_profile_env():
    env_str = config_utils.dump_profile_env("local")
    assert "PROFILE_NAME='local'" in env_str
    assert "PROFILE_REMOTE='false'" in env_str
    assert "PROFILE_BATCH=" in env_str
    assert "PROFILE_BASE_MODEL=" in env_str
    assert "PROFILE_PATIENCE='50'" in env_str


def test_prepare_training_data_yaml(tmp_path):
    # Create fake workspace directory structure
    yolo_dir = tmp_path / "yolo_dataset"
    (yolo_dir / "images" / "train").mkdir(parents=True)
    (yolo_dir / "images" / "val").mkdir(parents=True)

    data_yaml = config_utils.prepare_training_data_yaml(
        coco_dataset="coco128",
        workspace_dir=str(tmp_path),
    )
    assert os.path.isfile(data_yaml)

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "names" in data
    assert len(data["names"]) == 80
    assert data["names"][0] == "person"
