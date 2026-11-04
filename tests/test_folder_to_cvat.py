#!/usr/bin/env python3
"""
Unit tests for simplified folder-to-cvat.py
"""

import json
from pathlib import Path
import pytest
import yaml


def test_load_labels_from_yaml_list(folder_cvat_module, tmp_path):
    yaml_file = tmp_path / "labels_list.yaml"
    yaml_file.write_text(yaml.dump(["person", "dog", "car"]), encoding="utf-8")
    labels = folder_cvat_module.load_labels_from_yaml(yaml_file)
    assert labels == ["person", "dog", "car"]


def test_load_labels_from_yaml_dict_labels_key(folder_cvat_module, tmp_path):
    yaml_file = tmp_path / "config_labels.yaml"
    yaml_file.write_text(yaml.dump({"labels": ["cat", "bird", "horse"]}), encoding="utf-8")
    labels = folder_cvat_module.load_labels_from_yaml(yaml_file)
    assert labels == ["cat", "bird", "horse"]

    yaml_file2 = tmp_path / "config_labels_dict.yaml"
    yaml_file2.write_text(yaml.dump({"labels": {"cat": 1, "bird": 2}}), encoding="utf-8")
    labels2 = folder_cvat_module.load_labels_from_yaml(yaml_file2)
    assert labels2 == ["cat", "bird"]


def test_load_labels_from_yaml_categories(folder_cvat_module, tmp_path):
    yaml_file = tmp_path / "categories.yaml"
    yaml_file.write_text(yaml.dump({"categories": {"person": 1, "dog": 2, "bicycle": 3}}), encoding="utf-8")
    labels = folder_cvat_module.load_labels_from_yaml(yaml_file)
    assert labels == ["person", "dog", "bicycle"]


def test_load_labels_from_yaml_yolo_names(folder_cvat_module, tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(yaml.dump({"names": {0: "person", 1: "bicycle", 2: "car"}}), encoding="utf-8")
    labels = folder_cvat_module.load_labels_from_yaml(yaml_file)
    assert labels == ["person", "bicycle", "car"]

    yaml_file_list = tmp_path / "data_list.yaml"
    yaml_file_list.write_text(yaml.dump({"names": ["person", "bicycle", "car"]}), encoding="utf-8")
    labels_list = folder_cvat_module.load_labels_from_yaml(yaml_file_list)
    assert labels_list == ["person", "bicycle", "car"]


def test_load_labels_from_yaml_errors(folder_cvat_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        folder_cvat_module.load_labels_from_yaml(tmp_path / "nonexistent.yaml")

    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(":\n  - [invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        folder_cvat_module.load_labels_from_yaml(bad_file)


def test_collect_images(folder_cvat_module, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    (img_dir / "frame_001.jpg").touch()
    (img_dir / "frame_002.png").touch()
    (img_dir / "frame_003.webp").touch()
    (img_dir / "frame_004.bmp").touch()
    (img_dir / "notes.txt").touch()
    (img_dir / "data.json").touch()

    images = folder_cvat_module.collect_images(img_dir)
    assert len(images) == 4
    filenames = [p.name for p in images]
    assert "frame_001.jpg" in filenames
    assert "frame_002.png" in filenames
    assert "frame_003.webp" in filenames
    assert "frame_004.bmp" in filenames
    assert "notes.txt" not in filenames
    assert "data.json" not in filenames


def test_collect_images_errors(folder_cvat_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        folder_cvat_module.collect_images(tmp_path / "nonexistent_dir")

    regular_file = tmp_path / "file.txt"
    regular_file.touch()
    with pytest.raises(NotADirectoryError):
        folder_cvat_module.collect_images(regular_file)


def test_save_cvat_task_state(folder_cvat_module, tmp_path):
    state_file = folder_cvat_module.save_cvat_task_state(
        task_id=99,
        task_name="Test Task",
        cvat_url="http://cvat.local:8080",
        task_url="http://cvat.local:8080/tasks/99",
        base_dir=tmp_path,
    )
    assert state_file.is_file()
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == 99
    assert data["task_name"] == "Test Task"
    assert data["cvat_url"] == "http://cvat.local:8080"
    assert data["task_url"] == "http://cvat.local:8080/tasks/99"
    assert "created_at" in data


def test_upload_folder_to_cvat_mock(folder_cvat_module, tmp_path, monkeypatch):
    class MockTask:
        id = 123
        name = "Mocked Task"

    class MockTasksAPI:
        def create_from_data(self, spec, resources):
            assert spec["name"] == "Mocked Task"
            assert spec["labels"] == [{"name": "person"}, {"name": "car"}]
            assert len(resources) == 2
            return MockTask()

    class MockClient:
        tasks = MockTasksAPI()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import cvat_sdk
    monkeypatch.setattr(cvat_sdk, "make_client", lambda url, credentials: MockClient())

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.touch()
    img2.touch()

    task_id = folder_cvat_module.upload_folder_to_cvat(
        images=[img1, img2],
        labels=["person", "car"],
        task_name="Mocked Task",
        cvat_url="http://cvat.local:8080",
        cvat_user="testuser",
        cvat_pass="testpass",
        base_dir=tmp_path,
    )

    assert task_id == 123
    assert (tmp_path / ".last_cvat_task").is_file()


def test_main_dry_run(folder_cvat_module, tmp_path, capsys):
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    (img_dir / "sample1.jpg").touch()
    (img_dir / "sample2.png").touch()

    yaml_file = tmp_path / "my_labels.yaml"
    yaml_file.write_text(yaml.dump(["cat", "dog"]), encoding="utf-8")

    exit_code = folder_cvat_module.main([
        str(img_dir),
        "--labels-file", str(yaml_file),
        "--task-name", "DryRunTest",
        "--no-detect",
        "--dry-run",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN] Validation Successful" in captured.out
    assert "Total Images : 2" in captured.out
    assert "Labels (2)  : ['cat', 'dog']" in captured.out
    assert "Task Name    : DryRunTest" in captured.out


def test_load_model_classes(folder_cvat_module, tmp_path):
    model_json = tmp_path / "custom_model.json"
    model_json.write_text(json.dumps({
        "names": {
            "0": "cat",
            "1": "dog",
            "2": "fox"
        }
    }), encoding="utf-8")

    classes = folder_cvat_module.load_model_classes(tmp_path / "custom_model.onnx")
    assert classes == {0: "cat", 1: "dog", 2: "fox"}


def test_resolve_execution_device(folder_cvat_module):
    # Test CPU resolution
    dev = folder_cvat_module.resolve_execution_device("some_model.onnx", requested_device="cpu")
    assert dev == "cpu"


def test_generate_coco_annotations_mock(folder_cvat_module, tmp_path, monkeypatch):
    from PIL import Image

    img1 = tmp_path / "test1.jpg"
    Image.new("RGB", (100, 100), color="red").save(img1)

    class MockBox:
        cls = [0]
        conf = [0.85]
        xyxy = [[10.0, 15.0, 60.0, 75.0]]

        def __len__(self):
            return 1

    class MockResults:
        boxes = MockBox()

    class MockYOLO:
        def __init__(self, model_path):
            pass

        def __call__(self, *args, **kwargs):
            import torch
            box_mock = MockBox()
            box_mock.cls = torch.tensor([0])
            box_mock.conf = torch.tensor([0.85])
            box_mock.xyxy = torch.tensor([[10.0, 15.0, 60.0, 75.0]])
            res = MockResults()
            res.boxes = box_mock
            return [res]

    import ultralytics
    monkeypatch.setattr(ultralytics, "YOLO", MockYOLO)

    model_file = tmp_path / "mock.onnx"
    model_file.touch()

    coco_data, labels = folder_cvat_module.generate_coco_annotations(
        images=[img1],
        model_path=model_file,
        device="cpu",
        conf_thresh=0.5,
        imgsz=320,
        model_classes={0: "person", 1: "car"},
    )

    assert labels == ["person", "car"]
    assert len(coco_data["images"]) == 1
    assert coco_data["images"][0]["file_name"] == "test1.jpg"
    assert coco_data["images"][0]["width"] == 100
    assert coco_data["images"][0]["height"] == 100

    assert len(coco_data["annotations"]) == 1
    ann = coco_data["annotations"][0]
    assert ann["category_id"] == 1
    assert ann["bbox"] == [10.0, 15.0, 50.0, 60.0]  # [x, y, w, h]


def test_upload_folder_to_cvat_with_annotations_mock(folder_cvat_module, tmp_path, monkeypatch):
    class MockTask:
        id = 456
        name = "Detection Task"
        imported_file = None

        def import_annotations(self, format_name, filename):
            assert format_name == "COCO 1.0"
            self.imported_file = filename

    mock_task_instance = MockTask()

    class MockTasksAPI:
        def create_from_data(self, spec, resources):
            assert spec["name"] == "Detection Task"
            return mock_task_instance

    class MockClient:
        tasks = MockTasksAPI()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import cvat_sdk
    monkeypatch.setattr(cvat_sdk, "make_client", lambda url, credentials: MockClient())

    img1 = tmp_path / "img1.jpg"
    img1.touch()
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text("{}", encoding="utf-8")

    task_id = folder_cvat_module.upload_folder_to_cvat(
        images=[img1],
        labels=["person"],
        task_name="Detection Task",
        cvat_url="http://cvat.local:8080",
        cvat_user="testuser",
        cvat_pass="testpass",
        annotations_path=ann_file,
        base_dir=tmp_path,
    )

    assert task_id == 456
    assert mock_task_instance.imported_file == str(ann_file)

