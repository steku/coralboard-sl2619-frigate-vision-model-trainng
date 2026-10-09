#!/usr/bin/env python3
"""
Custom YOLOv8 to Torq NPU Pre-Conversion & Export Helper
Following official Synaptics documentation:
https://developer.synaptics.com/blog/custom-yolo-to-torq
"""

import argparse
from collections import defaultdict, Counter
import json
import os
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import yaml


def load_dataset_metadata(data_yaml_path: str = "yolo_dataset/data.yaml"):
    path = Path(data_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found at {data_yaml_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError(f"Invalid format for 'names' in {data_yaml_path}")
    return names


def step1_export_onnx(model_pt: str, onnx_out: str, size: int = 320):
    from ultralytics import YOLO

    print(f"--> Exporting PyTorch model '{model_pt}' to ONNX (imgsz={size})...")
    model = YOLO(model_pt)
    exported_path = model.export(
        format="onnx",
        imgsz=size,
        dynamic=False,
        simplify=True,
        opset=18,
    )
    if os.path.exists(exported_path) and exported_path != onnx_out:
        shutil.copyfile(exported_path, onnx_out)
    print(f"    Exported successfully to: {onnx_out}")


def step2_normalize_output(onnx_in: str, onnx_out: str, size: int = 320):
    print(f"--> Applying clean coordinate normalization to ONNX graph...")
    m = onnx.load(onnx_in)

    out_info = m.graph.output[0]
    out_name = out_info.name
    # Expected shape: [1, num_channels, 2100]
    shape = [d.dim_value for d in out_info.type.tensor_type.shape.dim]
    num_channels = shape[1]
    print(f"    Output tensor: '{out_name}', shape: {shape} ({num_channels} channels)")

    # In YOLOv8 ONNX, the output node is Concat(boxes, scores).
    # Normalizing strides directly BEFORE concatenation ensures the box regression
    # and confidence branches are BOTH natively in [0, 1] before concatenation.
    # This prevents TFLite per-tensor activation quantization from clamping coordinates.
    out_nodes = [n for n in m.graph.node if out_name in n.output]
    if not out_nodes:
        raise RuntimeError(f"Could not locate producer node for output tensor '{out_name}'")

    out_node = out_nodes[0]
    if out_node.op_type == "Concat":
        box_input_name = out_node.input[0]
        box_nodes = [n for n in m.graph.node if box_input_name in n.output]
        if not box_nodes:
            raise RuntimeError(f"Could not locate producer node for box tensor '{box_input_name}'")

        box_node = box_nodes[0]
        stride_init = None
        for inp in box_node.input:
            for init in m.graph.initializer:
                if init.name == inp:
                    stride_init = init
                    break
            if stride_init is not None:
                break

        if stride_init is not None:
            arr = numpy_helper.to_array(stride_init)
            print(f"    Found stride initializer '{stride_init.name}' with values: {np.unique(arr)}")
            arr_norm = arr / float(size)
            new_init = numpy_helper.from_array(arr_norm.astype(np.float32), name=stride_init.name)
            stride_init.CopyFrom(new_init)
            print(f"    Scaled strides by 1/{size}: {np.unique(arr_norm)}")
        else:
            raise RuntimeError(f"Could not find stride initializer for box node '{box_node.name}'")
    else:
        raise RuntimeError(f"Expected final output node to be 'Concat', found '{out_node.op_type}'")

    onnx.checker.check_model(m)
    Path(onnx_out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, onnx_out)
    print(f"    Saved cleanly normalized model: {onnx_out} [1, {num_channels}, 2100]")


def find_label_file(image_path: Path) -> Path:
    """Resolve corresponding YOLO label .txt file for an image path."""
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        candidate = Path(*parts).with_suffix(".txt")
        if candidate.exists():
            return candidate
    candidate = image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    if candidate.exists():
        return candidate
    candidate = image_path.parent / f"{image_path.stem}.txt"
    if candidate.exists():
        return candidate
    return None


def select_balanced_calibration_images(
    unique_paths: list,
    names: dict = None,
    max_images: int = 300,
    seed: int = 42,
) -> list:
    """
    Selects up to max_images from unique_paths using stratified round-robin sampling across classes.
    Guarantees maximum representation for rare/minority classes to ensure accurate INT8 PTQ dynamic ranges.
    """
    if len(unique_paths) <= max_images:
        return unique_paths

    rng = random.Random(seed)

    img_classes = {}
    class_to_imgs = defaultdict(list)
    background_imgs = []

    for img in unique_paths:
        lbl = find_label_file(img)
        cls_ids = set()
        if lbl and lbl.exists():
            try:
                with open(lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_ids.add(int(parts[0]))
                            except ValueError:
                                pass
            except Exception:
                pass
        img_classes[img] = cls_ids
        if cls_ids:
            for c in cls_ids:
                class_to_imgs[c].append(img)
        else:
            background_imgs.append(img)

    for c in class_to_imgs:
        rng.shuffle(class_to_imgs[c])
    rng.shuffle(background_imgs)

    selected = set()
    # Sort classes by frequency ascending so rarest classes get prioritized
    sorted_classes = sorted(class_to_imgs.keys(), key=lambda c: len(class_to_imgs[c]))

    ptrs = {c: 0 for c in sorted_classes}
    still_has_images = True

    while len(selected) < max_images and still_has_images:
        progress_made = False
        for c in sorted_classes:
            if len(selected) >= max_images:
                break
            while ptrs[c] < len(class_to_imgs[c]):
                cand = class_to_imgs[c][ptrs[c]]
                ptrs[c] += 1
                if cand not in selected:
                    selected.add(cand)
                    progress_made = True
                    break
        if not progress_made:
            still_has_images = False

    # If quota remains, fill from background images, then remaining pool
    if len(selected) < max_images:
        for bg in background_imgs:
            if len(selected) >= max_images:
                break
            selected.add(bg)

    if len(selected) < max_images:
        remaining = [p for p in unique_paths if p not in selected]
        rng.shuffle(remaining)
        for r in remaining:
            if len(selected) >= max_images:
                break
            selected.add(r)

    selected_list = list(selected)
    rng.shuffle(selected_list)
    return selected_list


def step3_prepare_calibration(
    data_yaml_path: str = "yolo_dataset/data.yaml",
    calib_out: str = "calibration_video_300_nhwc_float32.npy",
    size: int = 320,
    max_images: int = 300,
    min_images: int = 50,
    calib_percent: float = None,
    strategy: str = "balanced",
    seed: int = 42,
):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    else:
        names = {}

    dataset_root = Path(data.get("path", "yolo_dataset"))
    val_rel = data.get("val", "images/val")
    train_rel = data.get("train", "images/train")

    val_dirs = val_rel if isinstance(val_rel, list) else [val_rel]
    train_dirs = train_rel if isinstance(train_rel, list) else [train_rel]

    search_dirs = []
    for vd in val_dirs:
        p_vd = Path(vd)
        search_dirs.append(p_vd if p_vd.is_absolute() else dataset_root / p_vd)
        search_dirs.append(p_vd)
    for td in train_dirs:
        p_td = Path(td)
        search_dirs.append(p_td if p_td.is_absolute() else dataset_root / p_td)
        search_dirs.append(p_td)

    paths = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                paths.extend(d.glob(ext))

    # De-duplicate
    unique_paths = list({p.resolve(): p for p in paths}.values())
    print(f"    Found {len(unique_paths)} candidate images across dataset.")
    if not unique_paths:
        raise RuntimeError("No calibration images found in dataset directories!")

    if calib_percent is not None:
        pct = (calib_percent / 100.0) if calib_percent > 1.0 else calib_percent
        raw_count = int(round(len(unique_paths) * pct))
        effective_count = max(min(len(unique_paths), min_images), min(raw_count, max_images))
        print(f"    Sample percentage: {calib_percent:.1f}% ({raw_count} images) -> clamped to {effective_count} (min={min_images}, max={max_images})")
    else:
        effective_count = min(len(unique_paths), max_images)

    print(f"--> Generating representative INT8 calibration dataset (strategy='{strategy}', count={effective_count})...")

    if strategy == "balanced":
        selected_paths = select_balanced_calibration_images(unique_paths, names, max_images=effective_count, seed=seed)
    else:
        rng = random.Random(seed)
        selected_paths = list(unique_paths)
        rng.shuffle(selected_paths)
        selected_paths = selected_paths[:effective_count]

    # Report class breakdown of selected calibration images
    class_counts = Counter()
    for p in selected_paths:
        lbl = find_label_file(p)
        if lbl and lbl.exists():
            try:
                with open(lbl, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_counts[int(parts[0])] += 1
            except Exception:
                pass

    summary = ", ".join(f"{names.get(c, str(c))}:{cnt}" for c, cnt in sorted(class_counts.items()))
    print(f"    Selected {len(selected_paths)} calibration images. Class instances: {summary if summary else 'None'}")

    samples = []
    for p in selected_paths:
        image = cv2.imread(str(p))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(size / width, size / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized = cv2.resize(image, (new_width, new_height))
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        top = (size - new_height) // 2
        left = (size - new_width) // 2
        padded[top : top + new_height, left : left + new_width] = resized
        samples.append(padded.astype(np.float32) / 255.0)

    calibration = np.stack(samples).astype(np.float32)
    Path(calib_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(calib_out, calibration)
    print(f"    Calibration array saved to: {calib_out}")
    print(f"    Shape: {calibration.shape}, Dtype: {calibration.dtype}, Range: [{calibration.min():.2f}, {calibration.max():.2f}]")


def step4_generate_labels_json(
    data_yaml_path: str = "yolo_dataset/data.yaml",
    labels_out: str = None,
    tflite_path: str = "onnx2tf_normalized_full_int8/best_normalized_output_full_integer_quant.tflite",
    model_name: str = "YOLO",
    base_model_name: str = None,
    base_name: str = None,
):
    names = load_dataset_metadata(data_yaml_path)
    resolved_base = base_model_name or base_name or model_name
    if labels_out is None:
        labels_dir = Path("models/labels")
        if labels_dir.exists():
            labels_out = str(labels_dir / f"labels_{resolved_base}.json")
        else:
            labels_out = f"labels_{resolved_base}.json"

    label_dict = {
        "description": f"Frigate custom trained {model_name} object detection model",
        "author": "Frigate Pipeline",
        "base_model_name": resolved_base,
        "base_name": resolved_base,
        "stride": 32,
        "task": "detect",
        "batch": 1,
        "imgsz": [320, 320],
        "names": {str(k): v for k, v in names.items()},
    }

    # If quantized TFLite model is available, include exact quantization parameters
    tflite_file = Path(tflite_path)
    if not tflite_file.exists() and Path(f"output/{tflite_path}").exists():
        tflite_file = Path(f"output/{tflite_path}")

    if tflite_file.exists():
        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(tflite_file))
            interp.allocate_tensors()
            in_d = interp.get_input_details()[0]
            out_d = interp.get_output_details()[0]
            label_dict["quantization"] = {
                "in_scale": float(in_d["quantization_parameters"]["scales"][0]),
                "in_zp": int(in_d["quantization_parameters"]["zero_points"][0]),
                "out_scale": float(out_d["quantization_parameters"]["scales"][0]),
                "out_zp": int(out_d["quantization_parameters"]["zero_points"][0]),
            }
        except Exception as e:
            print(f"    (Could not inspect TFLite quantization parameters: {e})")

    Path(labels_out).parent.mkdir(parents=True, exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2)
    print(f"--> Updated '{labels_out}' with {len(names)} classes and quantization config: {names}")


def find_latest_checkpoint(runs_dir: str = "runs/detect") -> str:
    path = Path(runs_dir)
    if path.is_dir():
        ckpts = sorted(path.glob("**/best.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1])
    return "best.pt"


def main():
    default_ckpt = find_latest_checkpoint()
    parser = argparse.ArgumentParser(description="Prepare ONNX and Calibration data for Torq compilation")
    parser.add_argument("--model-pt", default=default_ckpt, help=f"Path to best.pt checkpoint (default: {default_ckpt})")
    default_raw_onnx = "output/best.onnx" if Path("output/best.onnx").exists() else "best.onnx"
    default_norm_onnx = "output/best_normalized_output.onnx" if Path("output").exists() else "best_normalized_output.onnx"
    default_calib = "output/calibration_video_300_nhwc_float32.npy" if Path("output").exists() else "calibration_video_300_nhwc_float32.npy"
    parser.add_argument("--input-onnx", default=default_raw_onnx, help="Raw ONNX export input/target")
    parser.add_argument("--output-onnx", default=default_norm_onnx, help="Normalized ONNX output")
    parser.add_argument("--output-calib", default=default_calib, help="Calibration .npy output")
    parser.add_argument("--data-yaml", default="yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--output-labels", default=None, help="Path to labels JSON output (default: models/labels/labels_<base_model_name>.json)")
    parser.add_argument("--imgsz", type=int, default=320, help="Model image resolution (default: 320)")
    parser.add_argument("--max-calib", type=int, default=300, help="Max calibration images (default: 300)")
    parser.add_argument("--min-calib", type=int, default=50, help="Min calibration images when using percentage (default: 50)")
    parser.add_argument("--calib-percent", type=float, default=None, help="Calibration sample percentage (e.g. 20 for 20%%, clamped between --min-calib and --max-calib)")
    parser.add_argument("--calib-strategy", choices=["balanced", "random"], default="balanced", help="Calibration selection strategy (default: balanced)")
    parser.add_argument("--calib-seed", type=int, default=42, help="Random seed for calibration selection (default: 42)")
    parser.add_argument("--base-model-name", default=None, help="Base model name (default: derived from model-pt or YOLO)")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export (use existing input-onnx)")

    args = parser.parse_args()

    if not args.skip_export:
        step1_export_onnx(args.model_pt, args.input_onnx, args.imgsz)
    step2_normalize_output(args.input_onnx, args.output_onnx, args.imgsz)
    step3_prepare_calibration(
        args.data_yaml,
        args.output_calib,
        args.imgsz,
        max_images=args.max_calib,
        min_images=args.min_calib,
        calib_percent=args.calib_percent,
        strategy=args.calib_strategy,
        seed=args.calib_seed,
    )
    detected_name = args.base_model_name or (Path(args.model_pt).stem if args.model_pt else "YOLO")
    if args.output_labels:
        output_labels = args.output_labels
    else:
        labels_dir = Path("models/labels")
        if labels_dir.exists():
            output_labels = str(labels_dir / f"labels_{detected_name}.json")
        else:
            output_labels = f"labels_{detected_name}.json"

    step4_generate_labels_json(args.data_yaml, output_labels, model_name=detected_name, base_model_name=detected_name)
    print(f"Pre-conversion preparation finished successfully! Labels written to: {output_labels}")


if __name__ == "__main__":
    main()
