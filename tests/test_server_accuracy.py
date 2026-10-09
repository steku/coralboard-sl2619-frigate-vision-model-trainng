import tempfile
from pathlib import Path
import pytest

import test_server_accuracy


def test_compute_iou_identical():
    box = [10.0, 10.0, 50.0, 50.0]
    assert test_server_accuracy.compute_iou(box, box) == 1.0


def test_compute_iou_disjoint():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert test_server_accuracy.compute_iou(box1, box2) == 0.0


def test_compute_iou_partial():
    # Box 1: [0, 0, 10, 10] -> Area = 100
    # Box 2: [5, 0, 15, 10] -> Area = 100
    # Intersection: [5, 0, 10, 10] -> Area = 5 * 10 = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 0.333333
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    iou = test_server_accuracy.compute_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-3) == (50.0 / 150.0)


def test_parse_ground_truth(tmp_path):
    lbl_file = tmp_path / "sample.txt"
    # Format: <class_id> <cx> <cy> <w> <h>
    # class 0 (person), center (0.5, 0.5), width 0.2, height 0.4
    lbl_file.write_text("0 0.5 0.5 0.2 0.4\n16 0.2 0.2 0.1 0.1\n")

    img_w, img_h = 1000, 1000
    class_names = {0: "person", 16: "dog"}
    boxes = test_server_accuracy.parse_ground_truth(lbl_file, img_w, img_h, class_names)

    assert len(boxes) == 2
    # First box: cx=500, cy=500, w=200, h=400 -> [400, 300, 600, 700]
    b1 = boxes[0]
    assert b1["class_id"] == 0
    assert b1["label"] == "person"
    assert b1["bbox"] == [400.0, 300.0, 600.0, 700.0]

    # Second box with COCO fallback
    b2 = boxes[1]
    assert b2["class_id"] == 16
    assert b2["label"] == "dog"


def test_parse_ground_truth_coco_fallback(tmp_path):
    lbl_file = tmp_path / "coco_sample.txt"
    # class 2 is car in COCO-80
    lbl_file.write_text("2 0.5 0.5 0.2 0.2\n")

    # Empty class_names dict -> should fallback to canonical COCO-80
    boxes = test_server_accuracy.parse_ground_truth(lbl_file, 100, 100, {})
    assert len(boxes) == 1
    assert boxes[0]["class_id"] == 2
    assert boxes[0]["label"] == "car"


def test_collect_dataset_pairs(tmp_path):
    dataset_dir = tmp_path / "dataset"
    img_dir = dataset_dir / "images" / "val"
    lbl_dir = dataset_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Valid matching pair
    (img_dir / "image1.jpg").write_text("fake_img")
    (lbl_dir / "image1.txt").write_text("0 0.5 0.5 0.1 0.1")

    # Image without label
    (img_dir / "image2.jpg").write_text("fake_img")

    # Label without image
    (lbl_dir / "image3.txt").write_text("0 0.5 0.5 0.1 0.1")

    pairs = test_server_accuracy.collect_dataset_pairs(dataset_dir, "val")
    assert len(pairs) == 1
    assert pairs[0][0].name == "image1.jpg"
    assert pairs[0][1].name == "image1.txt"


def test_collect_coco_pairs_coco128():
    # If datasets/coco128 exists in workspace
    pairs = test_server_accuracy.collect_coco_pairs("coco128", datasets_dir="datasets")
    assert isinstance(pairs, list)
    if pairs:
        assert pairs[0][0].suffix in (".jpg", ".jpeg", ".png", ".webp")
        assert pairs[0][1].suffix == ".txt"
        assert pairs[0][0].stem == pairs[0][1].stem


def test_evaluate_pipeline_mock(tmp_path, monkeypatch):
    # Setup mock dataset
    dataset_dir = tmp_path / "dataset"
    img_dir = dataset_dir / "images" / "val"
    lbl_dir = dataset_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    from PIL import Image
    test_img = Image.new("RGB", (100, 100), color="blue")
    test_img.save(img_dir / "test01.jpg")
    (lbl_dir / "test01.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Mock detection request
    mock_resp = {
        "predictions": [
            {"label": "person", "confidence": 0.95, "x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6}
        ],
        "timing": {"inference_ms": 12.5},
    }
    monkeypatch.setattr(
        test_server_accuracy,
        "send_detection_request",
        lambda server_url, img_path, threshold, timeout: (mock_resp, 20.0),
    )
    monkeypatch.setattr(test_server_accuracy, "check_server_health", lambda url: None)

    out_data = test_server_accuracy.run_accuracy_benchmark(
        server_url="http://mock-server:5000/v1/vision/detection",
        dataset_dir=str(dataset_dir),
        num_images=1,
        coco_images=0,
    )

    assert out_data["images_tested"] == 1
    assert out_data["overall"]["tp"] == 1
    assert out_data["overall"]["fp"] == 0
    assert out_data["overall"]["fn"] == 0
    assert out_data["overall"]["precision"] == 100.0
    assert out_data["overall"]["recall"] == 100.0
    assert out_data["overall"]["f1_score"] == 100.0


def test_default_coco_images_is_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test_server_accuracy.py"])
    args = test_server_accuracy.parse_args()
    assert args.coco_images == 0


