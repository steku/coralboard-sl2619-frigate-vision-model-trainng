import tempfile
from pathlib import Path
import pytest

import test_nvidia_accuracy


def test_compute_iou():
    # Identical boxes
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [0.0, 0.0, 10.0, 10.0]
    assert test_nvidia_accuracy.compute_iou(boxA, boxB) == 1.0

    # Disjoint boxes
    boxC = [20.0, 20.0, 30.0, 30.0]
    assert test_nvidia_accuracy.compute_iou(boxA, boxC) == 0.0

    # Half overlap: 5x10 overlap = 50 area. Box A=100, Box D=100, union = 150. IoU = 50/150 = 1/3
    boxD = [5.0, 0.0, 15.0, 10.0]
    assert pytest.approx(test_nvidia_accuracy.compute_iou(boxA, boxD), 0.01) == (1.0 / 3.0)


def test_parse_ground_truth(tmp_path):
    lbl_file = tmp_path / "sample.txt"
    # Format: class_id cx cy w h (normalized)
    lbl_file.write_text("0 0.5 0.5 0.2 0.4\n16 0.1 0.2 0.05 0.1\n")

    class_names = {0: "person", 16: "dog"}
    boxes = test_nvidia_accuracy.parse_ground_truth(lbl_file, img_w=1000, img_h=500, class_names=class_names)
    assert len(boxes) == 2

    b0 = boxes[0]
    assert b0["class_id"] == 0
    assert b0["label"] == "person"
    # cx=500, cy=250, w=200, h=200 -> x1=400, y1=150, x2=600, y2=350
    assert b0["bbox"] == [400.0, 150.0, 600.0, 350.0]

    b1 = boxes[1]
    assert b1["class_id"] == 16
    assert b1["label"] == "dog"


def test_resolve_execution_device():
    # If pt and cuda requested
    dev_pt = test_nvidia_accuracy.resolve_execution_device("model.pt", "cpu")
    assert dev_pt == "cpu"

    dev_onnx = test_nvidia_accuracy.resolve_execution_device("model.onnx", "cpu")
    assert dev_onnx == "cpu"


def test_nvidia_evaluate_pipeline_mock(tmp_path, monkeypatch):
    import argparse
    import json
    from PIL import Image
    import torch

    dataset_dir = tmp_path / "dataset"
    img_dir = dataset_dir / "images" / "val"
    lbl_dir = dataset_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    Image.new("RGB", (100, 100), color="red").save(img_dir / "img01.jpg")
    (lbl_dir / "img01.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (dataset_dir / "data.yaml").write_text("names:\n  0: person\n")

    fake_model = tmp_path / "fake.onnx"
    fake_model.write_text("onnx")

    class DummyBoxes:
        def __init__(self):
            self.cls = torch.tensor([0.0])
            self.conf = torch.tensor([0.92])
            self.xyxy = torch.tensor([[40.0, 40.0, 60.0, 60.0]])

        def __len__(self):
            return 1

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()
            self.speed = {"inference": 4.5}

    class DummyYOLO:
        def __init__(self, path):
            self.path = path

        def __call__(self, *args, **kwargs):
            return [DummyResult()]

    monkeypatch.setattr(test_nvidia_accuracy, "YOLO", DummyYOLO)

    out_json = tmp_path / "nvidia_report.json"
    args = argparse.Namespace(
        model=str(fake_model),
        dataset_dir=str(dataset_dir),
        labels_path=None,
        num_images=1,
        coco_images=0,
        coco_dataset="coco128",
        frigate_plus_images=0,
        frigate_plus_dir=str(tmp_path / "fp"),
        datasets_dir=str(tmp_path),
        split="val",
        all=False,
        device="cpu",
        imgsz=320,
        threshold=0.20,
        iou_threshold=0.45,
        seed=42,
        output_json=str(out_json),
        verbose=True,
    )

    test_nvidia_accuracy.evaluate(args)
    assert out_json.is_file()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["summary"]["true_positives"] == 1
    assert report["summary"]["precision"] == 100.0
    assert report["summary"]["recall"] == 100.0

