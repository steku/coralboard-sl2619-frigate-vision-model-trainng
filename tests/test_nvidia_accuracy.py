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
