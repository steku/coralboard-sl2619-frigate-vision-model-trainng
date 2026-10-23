#!/usr/bin/env python3
"""
Unit tests for prepare_for_onnx2tf.py
"""

import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
import yaml
import cv2

import prepare_for_onnx2tf


def test_load_dataset_metadata(tmp_path):
    y_file = tmp_path / "data.yaml"
    y_file.write_text(yaml.safe_dump({"names": {0: "person", 16: "dog", 80: "chicken"}}))
    meta = prepare_for_onnx2tf.load_dataset_metadata(str(y_file))
    assert meta == {0: "person", 16: "dog", 80: "chicken"}


def test_find_label_file(tmp_path):
    ds_dir = tmp_path / "yolo"
    img_dir = ds_dir / "images" / "val"
    lbl_dir = ds_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img_path = img_dir / "sample01.jpg"
    lbl_path = lbl_dir / "sample01.txt"
    img_path.write_text("img")
    lbl_path.write_text("0 0.5 0.5 0.2 0.2\n")

    found = prepare_for_onnx2tf.find_label_file(img_path)
    assert found == lbl_path

    unlabeled_img = img_dir / "unlabeled.jpg"
    unlabeled_img.write_text("img")
    assert prepare_for_onnx2tf.find_label_file(unlabeled_img) is None


def test_select_balanced_calibration_images(tmp_path):
    ds_dir = tmp_path / "yolo"
    img_dir = ds_dir / "images" / "val"
    lbl_dir = ds_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    candidates = []
    # Create 10 images with class 0 (dominant) and 2 images with class 80 (rare)
    for i in range(10):
        ip = img_dir / f"dom_{i}.jpg"
        lp = lbl_dir / f"dom_{i}.txt"
        ip.write_text("img")
        lp.write_text("0 0.5 0.5 0.2 0.2\n")
        candidates.append(ip)

    for i in range(2):
        ip = img_dir / f"rare_{i}.jpg"
        lp = lbl_dir / f"rare_{i}.txt"
        ip.write_text("img")
        lp.write_text("80 0.5 0.5 0.2 0.2\n")
        candidates.append(ip)

    selected = prepare_for_onnx2tf.select_balanced_calibration_images(candidates, max_images=5)
    assert len(selected) == 5
    selected_names = {p.name for p in selected}
    # Both rare class images must be prioritized first
    assert "rare_0.jpg" in selected_names
    assert "rare_1.jpg" in selected_names


def test_step2_normalize_output(tmp_path):
    # Create a tiny valid YOLOv8-like ONNX model where output0 is Concat(boxes, scores)
    # and boxes is Mul(raw_boxes, stride_init)
    X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])
    Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 2100])

    raw_boxes_data = np.ones((1, 4, 2100), dtype=np.float32)
    scores_data = np.ones((1, 80, 2100), dtype=np.float32) * 0.5
    stride_data = np.ones((1, 1, 2100), dtype=np.float32) * 8.0

    raw_boxes_const = helper.make_tensor("raw_boxes_const", TensorProto.FLOAT, [1, 4, 2100], raw_boxes_data.flatten().tolist())
    scores_init = helper.make_tensor("scores", TensorProto.FLOAT, [1, 80, 2100], scores_data.flatten().tolist())
    stride_init = helper.make_tensor("stride_tensor", TensorProto.FLOAT, [1, 1, 2100], stride_data.flatten().tolist())

    const_node = helper.make_node("Constant", inputs=[], outputs=["raw_boxes"], value=raw_boxes_const, name="box_const")
    mul_node = helper.make_node("Mul", inputs=["raw_boxes", "stride_tensor"], outputs=["boxes_scaled"], name="box_mul")
    concat_node = helper.make_node("Concat", inputs=["boxes_scaled", "scores"], outputs=["output0"], axis=1, name="out_concat")

    graph = helper.make_graph(
        [const_node, mul_node, concat_node],
        "test_yolo_graph",
        [X],
        [Y],
        initializer=[scores_init, stride_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    raw_onnx = tmp_path / "raw.onnx"
    norm_onnx = tmp_path / "norm.onnx"
    onnx.save(model, str(raw_onnx))

    prepare_for_onnx2tf.step2_normalize_output(str(raw_onnx), str(norm_onnx), size=320)
    assert norm_onnx.is_file()

    loaded = onnx.load(str(norm_onnx))
    scaled_stride = [i for i in loaded.graph.initializer if i.name == "stride_tensor"][0]
    arr = onnx.numpy_helper.to_array(scaled_stride)
    assert np.allclose(arr, 8.0 / 320.0)


def test_step3_prepare_calibration_npy_and_dir(tmp_path):
    ds_dir = tmp_path / "yolo"
    img_dir = ds_dir / "images" / "val"
    lbl_dir = ds_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    dummy_img = np.full((100, 150, 3), 128, dtype=np.uint8)
    for i in range(3):
        cv2.imwrite(str(img_dir / f"cal_{i}.jpg"), dummy_img)
        (lbl_dir / f"cal_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    data_yaml = ds_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({
            "train": [str(img_dir)],
            "val": [str(img_dir)],
            "names": {0: "person"},
        })
    )

    # 1. Test .npy output for Torq
    npy_out = tmp_path / "calib.npy"
    prepare_for_onnx2tf.step3_prepare_calibration(
        data_yaml_path=str(data_yaml),
        calib_out=str(npy_out),
        size=320,
        max_images=2,
    )
    assert npy_out.is_file()
    arr = np.load(str(npy_out))
    assert arr.shape == (2, 320, 320, 3)
    assert arr.dtype == np.float32

    # 2. Test directory output for Hailo
    hailo_calib_dir = tmp_path / "hailo_calib"
    prepare_for_onnx2tf.step3_prepare_calibration(
        data_yaml_path=str(data_yaml),
        calib_out=str(hailo_calib_dir),
        size=640,
        max_images=2,
    )
    assert hailo_calib_dir.is_dir()
    jpgs = list(hailo_calib_dir.glob("*.jpg"))
    assert len(jpgs) == 2


def test_step4_generate_labels_json_and_symlinks(tmp_path):
    y_file = tmp_path / "data.yaml"
    y_file.write_text(yaml.safe_dump({"names": {0: "person", 80: "chicken"}}))

    labels_dir = tmp_path / "labels"
    labels_out = labels_dir / "hailo_yolov8m_261023-0800.json"

    prepare_for_onnx2tf.step4_generate_labels_json(
        data_yaml_path=str(y_file),
        labels_out=str(labels_out),
        tflite_path=str(tmp_path / "nonexistent.tflite"),
        base_model_name="yolov8m",
        imgsz=640,
        target="hailo",
    )

    assert labels_out.is_file()
    txt_out = labels_dir / "hailo_yolov8m_261023-0800.txt"
    assert txt_out.is_file()

    # Check soft links without timestamp
    assert (labels_dir / "hailo_yolov8m.json").is_symlink()
    assert (labels_dir / "hailo_yolov8m.txt").is_symlink()
    assert (labels_dir / "labels.json").is_symlink()
    assert (labels_dir / "labels.txt").is_symlink()

    with open(labels_out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["base_model_name"] == "yolov8m"
    assert payload["imgsz"] == [640, 640]
    assert payload["names"]["0"] == "person"
    assert payload["names"]["80"] == "chicken"


def test_find_latest_checkpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    assert prepare_for_onnx2tf.find_latest_checkpoint(str(runs_dir)) == "best.pt"

    ckpt1 = runs_dir / "exp1" / "weights" / "best.pt"
    ckpt2 = runs_dir / "exp2" / "weights" / "best.pt"
    ckpt1.parent.mkdir(parents=True)
    ckpt2.parent.mkdir(parents=True)
    ckpt1.write_text("old")
    ckpt2.write_text("new")

    found = prepare_for_onnx2tf.find_latest_checkpoint(str(runs_dir))
    assert Path(found) in (ckpt1, ckpt2)
