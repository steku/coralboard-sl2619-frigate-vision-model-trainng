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
        "--dry-run",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN] Validation Successful" in captured.out
    assert "Total Images : 2" in captured.out
    assert "Labels (2)  : ['cat', 'dog']" in captured.out
    assert "Task Name    : DryRunTest" in captured.out
