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
    p_torq = config_utils.get_profile("torq")
    assert p_torq["name"] == "torq"
    assert p_torq["target"] == "torq"
    assert p_torq["remote"] is False
    assert p_torq["batch"] == 8
    assert p_torq["patience"] == 50
    assert p_torq["optimizer"] == "AdamW"
    assert p_torq["lr0"] == 0.0005
    assert p_torq["lrf"] == 0.01
    assert p_torq["cos_lr"] is True
    assert p_torq["warmup_epochs"] == 5.0
    assert p_torq["warmup_momentum"] == 0.5
    assert p_torq["warmup_bias_lr"] == 0.0

    p_spark = config_utils.get_profile("spark-torq")
    assert p_spark["name"] == "spark-torq"
    assert p_spark["target"] == "torq"
    assert p_spark["remote"] is True
    assert p_spark["batch"] == 32
    assert p_spark["patience"] == 30
    assert p_spark["optimizer"] == "AdamW"
    assert p_spark["lr0"] == 0.0005
    assert p_spark["warmup_epochs"] == 5.0

    # Test hyphen and underscore normalization for spark-torq-yolo26n
    p_26n_hyphen = config_utils.get_profile("spark-torq-yolo26n")
    p_26n_under = config_utils.get_profile("spark_torq_yolo26n")
    assert p_26n_hyphen["base_model"] == "yolo26n.pt"
    assert p_26n_under["base_model"] == "yolo26n.pt"
    assert p_26n_hyphen["remote"] is True
    assert p_26n_hyphen["patience"] == 35
    assert p_26n_under["patience"] == 35
    assert p_26n_hyphen["optimizer"] == "AdamW"
    assert p_26n_hyphen["lr0"] == 0.0005
    assert p_26n_hyphen["target"] == "torq"

    p_cpu = config_utils.get_profile("cpu-torq")
    assert p_cpu["name"] == "cpu-torq"
    assert p_cpu["target"] == "torq"
    assert p_cpu["device"] == "cpu"

    # Test nvidia and spark-nvidia profiles
    p_nvidia = config_utils.get_profile("nvidia")
    assert p_nvidia["name"] == "nvidia"
    assert p_nvidia["remote"] is False
    assert p_nvidia["target"] == "nvidia"
    assert p_nvidia["base_model"] == "yolov9s.pt"
    assert p_nvidia["deploy_host"] == ""
    assert p_nvidia["batch"] == 16
    assert p_nvidia["epochs"] == 80
    assert p_nvidia["patience"] == 20
    assert p_nvidia["optimizer"] == "AdamW"
    assert p_nvidia["lr0"] == 0.0005

    p_spark_nvidia = config_utils.get_profile("spark-nvidia")
    assert p_spark_nvidia["name"] == "spark-nvidia"
    assert p_spark_nvidia["remote"] is True
    assert p_spark_nvidia["target"] == "nvidia"
    assert p_spark_nvidia["base_model"] == "yolov9s.pt"
    assert p_spark_nvidia["deploy_host"] == ""
    assert p_spark_nvidia["batch"] == 16
    assert p_spark_nvidia["epochs"] == 100
    assert p_spark_nvidia["patience"] == 30

    # Test hailo and spark-hailo profiles
    p_hailo = config_utils.get_profile("hailo")
    assert p_hailo["name"] == "hailo"
    assert p_hailo["remote"] is False
    assert p_hailo["target"] == "hailo"
    assert p_hailo["hw_arch"] == "hailo10h"
    assert p_hailo["base_model"] == "yolov8m.pt"
    assert p_hailo["imgsz"] == 640
    assert p_hailo["batch"] == 16

    p_spark_hailo = config_utils.get_profile("spark-hailo")
    assert p_spark_hailo["name"] == "spark-hailo"
    assert p_spark_hailo["remote"] is True
    assert p_spark_hailo["target"] == "hailo"
    assert p_spark_hailo["hw_arch"] == "hailo10h"
    assert p_spark_hailo["base_model"] == "yolov8m.pt"
    assert p_spark_hailo["imgsz"] == 640
    assert p_spark_hailo["batch"] == 16
    assert p_spark_hailo["epochs"] == 100

    # Unknown profile (including 'nvidia-spark') raises ValueError
    with pytest.raises(ValueError, match="Unknown profile"):
        config_utils.get_profile("nvidia-spark")
    with pytest.raises(ValueError, match="Unknown profile"):
        config_utils.get_profile("nonexistent-profile")


def test_get_profile_default_patience_fallback(tmp_path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        yaml.safe_dump(
            {
                "deployment": {"host": "root@coral:/models"},
                "profiles": {
                    "minimal": {"device": "cpu"},
                    "gpu_onnx": {"target": "onnx"},
                },
            }
        )
    )
    p = config_utils.get_profile("minimal", config_path=str(custom_yaml))
    assert p["patience"] == 50
    assert p["target"] == "torq"
    assert p["base_model"] == "yolov8n.pt"
    assert p["deploy_host"] == "root@coral:/models"
    assert p["optimizer"] == "AdamW"
    assert p["lr0"] == 0.0005
    assert p["warmup_epochs"] == 5.0
    assert p["warmup_bias_lr"] == 0.0

    p_gpu = config_utils.get_profile("gpu_onnx", config_path=str(custom_yaml))
    assert p_gpu["target"] == "nvidia"
    assert p_gpu["base_model"] == "yolov9s.pt"
    assert p_gpu["deploy_host"] == ""


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
    env_str = config_utils.dump_profile_env("torq")
    assert "PROFILE_NAME='torq'" in env_str
    assert "PROFILE_REMOTE='false'" in env_str
    assert "PROFILE_TARGET='torq'" in env_str
    assert "PROFILE_BATCH=" in env_str
    assert "PROFILE_BASE_MODEL=" in env_str
    assert "PROFILE_COCO_DATASET=" in env_str
    assert "PROFILE_FRIGATE_PLUS=" in env_str
    assert "PROFILE_FRIGATE_PLUS_DIR=" in env_str
    assert "PROFILE_PATIENCE='50'" in env_str
    assert "PROFILE_OPTIMIZER='AdamW'" in env_str
    assert "PROFILE_LR0='0.0005'" in env_str
    assert "PROFILE_LRF='0.01'" in env_str
    assert "PROFILE_COS_LR='True'" in env_str
    assert "PROFILE_WARMUP_EPOCHS='5.0'" in env_str
    assert "PROFILE_WARMUP_MOMENTUM='0.5'" in env_str
    assert "PROFILE_WARMUP_BIAS_LR='0.0'" in env_str

    nvidia_env_str = config_utils.dump_profile_env("nvidia")
    assert "PROFILE_NAME='nvidia'" in nvidia_env_str
    assert "PROFILE_TARGET='nvidia'" in nvidia_env_str
    assert "PROFILE_BASE_MODEL='yolov9s.pt'" in nvidia_env_str


def test_prepare_training_data_yaml(tmp_path, monkeypatch):
    # Create fake workspace directory structure
    yolo_dir = tmp_path / "datasets" / "yolo"
    (yolo_dir / "images" / "train").mkdir(parents=True)
    (yolo_dir / "images" / "val").mkdir(parents=True)

    fp_dir = tmp_path / "datasets" / "frigate-plus"
    (fp_dir / "images" / "train").mkdir(parents=True)
    (fp_dir / "images" / "val").mkdir(parents=True)

    coco_train = tmp_path / "datasets" / "coco128" / "images" / "train2017"
    coco_train.mkdir(parents=True)

    def mock_download(dataset_name, target_dir="datasets"):
        return {
            "train_dir": str(coco_train),
            "val_dir": str(coco_train),
        }

    monkeypatch.setattr(config_utils, "download_coco_dataset", mock_download)

    data_yaml = config_utils.prepare_training_data_yaml(
        coco_dataset="coco128",
        workspace_dir=str(tmp_path),
        include_frigate_plus=True,
        frigate_plus_dir=str(fp_dir),
    )
    assert os.path.isfile(data_yaml)
    assert str(data_yaml).endswith("datasets/data.yaml")
    assert (tmp_path / "datasets" / "yolo" / "data.yaml").is_file()

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert len(data["train"]) == 3
    assert len(data["val"]) == 2
    assert str(yolo_dir / "images" / "train") in data["train"]
    assert str(fp_dir / "images" / "train") in data["train"]
    assert str(coco_train) in data["train"]
    assert str(yolo_dir / "images" / "val") in data["val"]
    assert str(fp_dir / "images" / "val") in data["val"]
    assert str(coco_train) not in data["val"]
    assert "names" in data
    assert data["names"][0] == "person"


