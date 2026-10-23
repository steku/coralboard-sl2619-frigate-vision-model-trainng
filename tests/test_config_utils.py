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

    p_spark = config_utils.get_profile("torq-spark")
    assert p_spark["name"] == "torq-spark"
    assert p_spark["target"] == "torq"
    assert p_spark["remote"] is True
    assert p_spark["batch"] == 32
    assert p_spark["patience"] == 30
    assert p_spark["optimizer"] == "AdamW"
    assert p_spark["lr0"] == 0.0005
    assert p_spark["warmup_epochs"] == 5.0

    # Test hyphen and underscore normalization for torq-spark-yolo26n
    p_26n_hyphen = config_utils.get_profile("torq-spark-yolo26n")
    p_26n_under = config_utils.get_profile("torq_spark_yolo26n")
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

    # Test nvidia and nvidia-spark profiles
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

    p_spark_nvidia = config_utils.get_profile("nvidia-spark")
    assert p_spark_nvidia["name"] == "nvidia-spark"
    assert p_spark_nvidia["remote"] is True
    assert p_spark_nvidia["target"] == "nvidia"
    assert p_spark_nvidia["base_model"] == "yolov9s.pt"
    assert p_spark_nvidia["deploy_host"] == ""
    assert p_spark_nvidia["batch"] == 16
    assert p_spark_nvidia["epochs"] == 100
    assert p_spark_nvidia["patience"] == 30

    # Test hailo and hailo-spark profiles
    p_hailo = config_utils.get_profile("hailo")
    assert p_hailo["name"] == "hailo"
    assert p_hailo["remote"] is False
    assert p_hailo["target"] == "hailo"
    assert p_hailo["hw_arch"] == "hailo10h"
    assert p_hailo["base_model"] == "yolov8m.pt"
    assert p_hailo["imgsz"] == 640
    assert p_hailo["batch"] == 16

    p_spark_hailo = config_utils.get_profile("hailo-spark")
    assert p_spark_hailo["name"] == "hailo-spark"
    assert p_spark_hailo["remote"] is True
    assert p_spark_hailo["target"] == "hailo"
    assert p_spark_hailo["hw_arch"] == "hailo10h"
    assert p_spark_hailo["base_model"] == "yolov8m.pt"
    assert p_spark_hailo["imgsz"] == 640
    assert p_spark_hailo["batch"] == 16
    assert p_spark_hailo["epochs"] == 100

    # Unknown profile raises ValueError
    with pytest.raises(ValueError, match="Unknown profile"):
        config_utils.get_profile("spark-nvidia")
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


def test_get_cvat_credentials_precedence(monkeypatch):
    cfg = {"cvat": {"url": "http://cfg:8080", "username": "cfg_user", "password": "cfg_pass"}}
    monkeypatch.delenv("CVAT_URL", raising=False)
    monkeypatch.delenv("CVAT_USER", raising=False)
    monkeypatch.delenv("CVAT_PASS", raising=False)

    # Config fallback
    assert config_utils.get_cvat_credentials(cfg) == ("http://cfg:8080", "cfg_user", "cfg_pass")

    # Env override
    monkeypatch.setenv("CVAT_URL", "http://env:8080")
    monkeypatch.setenv("CVAT_USER", "env_user")
    monkeypatch.setenv("CVAT_PASS", "env_pass")
    assert config_utils.get_cvat_credentials(cfg) == ("http://env:8080", "env_user", "env_pass")

    # Explicit argument override
    assert config_utils.get_cvat_credentials(cfg, url="http://cli:8080", user="cli_u", password="cli_p") == (
        "http://cli:8080",
        "cli_u",
        "cli_p",
    )


def test_get_frigate_deployment_and_remote_settings():
    cfg = {
        "coco": {"dataset": "coco-val2017", "classes": ["person", "dog"]},
        "frigate_plus": {"enabled": True, "path": "/custom/fp"},
        "deployment": {
            "host": "root@coral:/models",
            "dir": "/tmp/models",
            "server_url": "http://coral:5000/v1/vision/detection",
            "restart_service": True,
            "service_name": "custom-service",
        },
        "remote_training": {
            "enabled": True,
            "host": "spark@10.0.0.1",
            "remote_dir": "/work/train",
            "device": "0",
            "batch": 64,
        },
    }

    coco_settings = config_utils.get_coco_config(cfg)
    assert coco_settings["dataset"] == "coco-val2017"

    fp_settings = config_utils.get_frigate_plus_config(cfg)
    assert fp_settings["enabled"] is True
    assert fp_settings["path"] == "/custom/fp"

    d_settings = config_utils.get_deployment_settings(cfg)
    assert d_settings["host"] == "root@coral:/models"
    assert d_settings["restart_service"] is True
    assert d_settings["service_name"] == "custom-service"

    r_settings = config_utils.get_remote_training_settings(cfg)
    assert r_settings["host"] == "spark@10.0.0.1"
    assert r_settings["batch"] == 64


def test_load_dataset_classes_formats(tmp_path):
    # List format
    y_list = tmp_path / "list_data.yaml"
    y_list.write_text(yaml.safe_dump({"names": ["person", "car"]}))
    assert config_utils.load_dataset_classes(str(y_list)) == {0: "person", 1: "car"}

    # Dict format
    y_dict = tmp_path / "dict_data.yaml"
    y_dict.write_text(yaml.safe_dump({"names": {0: "person", 80: "chicken"}}))
    assert config_utils.load_dataset_classes(str(y_dict)) == {0: "person", 80: "chicken"}

    # Invalid format raises ValueError
    y_bad = tmp_path / "bad_data.yaml"
    y_bad.write_text(yaml.safe_dump({"names": "invalid_string"}))
    with pytest.raises(ValueError, match="Invalid 'names' format"):
        config_utils.load_dataset_classes(str(y_bad))

    # Nonexistent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        config_utils.load_dataset_classes(str(tmp_path / "does_not_exist.yaml"))


def test_shared_evaluation_helpers(tmp_path):
    # 1. count_cvat_annotations
    ann_payload = {
        "shapes": [{"label_id": 1}, {"label_id": 2}],
        "tracks": [{"label_id": 1}],
        "tags": [{"label_id": 99}],
    }
    counts = config_utils.count_cvat_annotations(ann_payload, {1: "person", 2: "dog"})
    assert counts["person"] == 2
    assert counts["dog"] == 1
    assert counts["label_99"] == 1

    # 2. match_predictions_to_gt & summarize_class_metrics
    stats_per_class = {
        "person": {"gt": 2, "pred": 2, "tp": 0, "fp": 0, "fn": 0, "ious": []},
        "car": {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []},
    }
    gt_boxes = [
        {"class_id": 0, "label": "person", "bbox": [10.0, 10.0, 50.0, 50.0], "matched": False},
        {"class_id": 0, "label": "person", "bbox": [100.0, 100.0, 150.0, 150.0], "matched": False},
    ]
    parsed_preds = [
        {"label": "person", "confidence": 0.9, "bbox": [10.0, 10.0, 50.0, 50.0], "matched": False},
        {"label": "person", "confidence": 0.6, "bbox": [200.0, 200.0, 240.0, 240.0], "matched": False},
    ]
    config_utils.match_predictions_to_gt(parsed_preds, gt_boxes, 0.45, stats_per_class)
    assert stats_per_class["person"]["tp"] == 1
    assert stats_per_class["person"]["fp"] == 1
    assert stats_per_class["person"]["fn"] == 1

    overall, per_class, sorted_cls = config_utils.summarize_class_metrics(stats_per_class)
    assert overall["tp"] == 1
    assert overall["fp"] == 1
    assert overall["fn"] == 1
    assert overall["precision"] == 50.0
    assert overall["recall"] == 50.0
    assert overall["f1_score"] == 50.0
    assert overall["mean_iou"] == 1.0
    assert per_class["person"]["tested"] is True
    assert per_class["car"]["tested"] is False
    assert sorted_cls[0] == "person"



