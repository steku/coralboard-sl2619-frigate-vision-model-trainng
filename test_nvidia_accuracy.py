#!/usr/bin/env python3
"""
Test NVIDIA GPU & ONNX Model Accuracy and Latency.

Evaluates an exported ONNX model or PyTorch checkpoint (.pt) on NVIDIA GPU (CUDA)
or CPU against local surveillance datasets (datasets/yolo, datasets/frigate-plus)
and external COCO reference images.

Produces a comprehensive detection accuracy report (Precision, Recall, F1-Score,
mean IoU per class and overall) and hardware execution latency statistics,
with optional machine-readable JSON export.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

import config_utils


def find_default_model() -> str:
    """Auto-detect the most appropriate default NVIDIA model (.onnx or .pt)."""
    candidates = [
        Path("models/compiled/custom_yolov9s.onnx"),
        Path("output/best_normalized_output.onnx"),
        Path("output/best.onnx"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    compiled_dir = Path("models/compiled")
    if compiled_dir.is_dir():
        onnx_files = sorted(compiled_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if onnx_files:
            return str(onnx_files[-1])

    runs_dir = Path("runs/detect")
    if runs_dir.is_dir():
        ckpts = sorted(runs_dir.glob("**/best.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1])

    if Path("models/base/yolov9s.pt").is_file():
        return "models/base/yolov9s.pt"
    if Path("yolov9s.pt").is_file():
        return "yolov9s.pt"

    return "models/compiled/custom_yolov9s.onnx"


def parse_args():
    default_model = find_default_model()
    default_ds = "./datasets/yolo" if Path("datasets/yolo").is_dir() else "./yolo_dataset"
    default_coco_ds = config_utils.load_config().get("coco", {}).get("dataset", "coco-val2017")
    default_fp_dir = config_utils.load_config().get("frigate_plus", {}).get("path", "./datasets/frigate-plus")
    default_device = "cuda"

    parser = argparse.ArgumentParser(
        description="Benchmark detection accuracy and inference latency for NVIDIA ONNX and PyTorch models.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model",
        "--model-path",
        "--onnx",
        "--weights",
        dest="model",
        default=default_model,
        help=f"Path to ONNX model (.onnx) or PyTorch checkpoint (.pt) (default: {default_model})",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device: 'cuda', '0', 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image resolution (default: 320)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=default_ds,
        help=f"Path to YOLO dataset directory containing images/ and labels/ (default: {default_ds})",
    )
    parser.add_argument(
        "--split",
        choices=["all", "val", "train"],
        default="all",
        help="Dataset split to evaluate (default: all - draws from entire dataset)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=60,
        help="Number of random dataset images to test (default: 60, set 0 or use --all for all images)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all images in the selected dataset split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling (default: 42)",
    )
    parser.add_argument(
        "--threshold",
        "--conf",
        dest="threshold",
        type=float,
        default=0.20,
        help="Confidence score threshold for detections (default: 0.20)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Intersection-over-Union (IoU) threshold for matching boxes (default: 0.45)",
    )
    parser.add_argument(
        "--output-json",
        "--json",
        dest="output_json",
        default=None,
        help="Optional path to export evaluation report as JSON",
    )
    parser.add_argument(
        "--coco-images",
        "--num-coco-images",
        dest="coco_images",
        type=int,
        default=0,
        help="Number of random COCO dataset images to include in benchmark (default: 0)",
    )
    parser.add_argument(
        "--coco-dataset",
        default=default_coco_ds,
        choices=["coco-val2017", "coco128"],
        help=f"COCO dataset subset for benchmark images (default: {default_coco_ds})",
    )
    parser.add_argument(
        "--datasets-dir",
        default="./datasets",
        help="Base directory containing downloaded COCO datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--frigate-plus-images",
        "--num-frigate-plus-images",
        dest="frigate_plus_images",
        type=int,
        default=0,
        help="Number of random Frigate+ dataset images to include in benchmark (default: 0)",
    )
    parser.add_argument(
        "--frigate-plus-dir",
        default=default_fp_dir,
        help=f"Directory containing Frigate+ dataset (default: {default_fp_dir})",
    )
    parser.add_argument(
        "--labels",
        "--labels-json",
        dest="labels_path",
        default=None,
        help="Optional path to custom labels.json metadata file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-image detection results",
    )
    return parser.parse_args()


def load_class_names(dataset_dir: Path, labels_path: Optional[str] = None) -> Dict[int, str]:
    """Load class index to name mapping from labels.json, data.yaml, or canonical COCO-80."""
    if labels_path and Path(labels_path).is_file():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            names_meta = meta.get("names", {})
            if isinstance(names_meta, dict):
                return {int(k): str(v) for k, v in names_meta.items()}
            elif isinstance(names_meta, list):
                return {i: str(v) for i, v in enumerate(names_meta)}
        except Exception as e:
            print(f"[WARN] Could not parse labels file '{labels_path}': {e}")

    for candidate in [
        Path("models/labels/labels_yolov9s.json"),
        Path("models/labels/labels.json"),
        Path("labels.json"),
        dataset_dir / "data.yaml",
        Path("datasets/data.yaml"),
    ]:
        if candidate.is_file():
            try:
                if candidate.suffix == ".json":
                    with open(candidate, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    names_meta = meta.get("names", {})
                    if names_meta:
                        if isinstance(names_meta, dict):
                            return {int(k): str(v) for k, v in names_meta.items()}
                        elif isinstance(names_meta, list):
                            return {i: str(v) for i, v in enumerate(names_meta)}
                else:
                    return config_utils.load_dataset_classes(str(candidate))
            except Exception:
                pass

    return config_utils.get_coco_80_dict()


def collect_dataset_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    """Locate all matching (image_path, label_path) pairs in dataset."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    splits_to_check = ["val", "train"] if split == "all" else [split]
    pairs = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for s in splits_to_check:
        s_img_dir = images_dir / s
        s_lbl_dir = labels_dir / s
        if not s_img_dir.is_dir():
            continue
        for img_path in sorted(s_img_dir.iterdir()):
            if img_path.suffix.lower() in valid_extensions:
                lbl_path = s_lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.is_file():
                    pairs.append((img_path, lbl_path))

    return pairs


def collect_coco_pairs(coco_dataset_name: str = "coco-val2017", datasets_dir: str = "datasets") -> List[Tuple[Path, Path]]:
    """Locate all matching (image_path, label_path) pairs for the specified COCO dataset."""
    try:
        coco_info = config_utils.download_coco_dataset(coco_dataset_name, target_dir=datasets_dir)
        val_img_dir = Path(coco_info.get("val_dir") or coco_info.get("train_dir"))
    except Exception as e:
        print(f"[WARN] Unable to prepare COCO dataset '{coco_dataset_name}': {e}")
        return []

    if not val_img_dir.is_dir():
        print(f"[WARN] COCO image directory not found at: {val_img_dir}")
        return []

    lbl_dir = Path(str(val_img_dir).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
    if not lbl_dir.is_dir():
        lbl_dir = val_img_dir.parent.parent / "labels" / val_img_dir.name

    if not lbl_dir.is_dir():
        print(f"[WARN] COCO label directory not found at: {lbl_dir}")
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    pairs = []
    for img_path in sorted(val_img_dir.iterdir()):
        if img_path.suffix.lower() in valid_extensions:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.is_file():
                pairs.append((img_path, lbl_path))

    return pairs


def parse_ground_truth(label_path: Path, img_w: int, img_h: int, class_names: Dict[int, str]) -> List[dict]:
    """Parse normalized YOLO label file into absolute pixel bounding boxes."""
    boxes = []
    if not label_path.exists():
        return boxes

    coco_names = config_utils.get_coco_80_dict()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h

            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(float(img_w), cx + w / 2.0)
            y2 = min(float(img_h), cy + h / 2.0)

            label_name = class_names.get(class_id) or coco_names.get(class_id, str(class_id))
            boxes.append({
                "class_id": class_id,
                "label": str(label_name).lower(),
                "bbox": [x1, y1, x2, y2],
                "matched": False,
            })
    return boxes


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def resolve_execution_device(model_path: str, requested_device: str) -> str:
    """Resolve compatible execution device for ONNX Runtime vs PyTorch."""
    is_onnx = str(model_path).endswith(".onnx")
    if is_onnx:
        if requested_device in ("cuda", "0"):
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return "cuda"
                else:
                    print("[INFO] onnxruntime-gpu not installed; running ONNX inference via CPUExecutionProvider.")
                    return "cpu"
            except Exception:
                return "cpu"
        return "cpu"

    if requested_device in ("cuda", "0") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested_device


def evaluate(args):
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        print(f"[ERROR] Model file not found at: {model_path}")
        print("        Train or export an ONNX/PyTorch model first using: ./train.sh -p nvidia")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir).resolve()
    coco_images_req = getattr(args, "coco_images", 0)
    coco_dataset_name = getattr(args, "coco_dataset", "coco-val2017")
    datasets_dir = getattr(args, "datasets_dir", "./datasets")
    frigate_plus_images_req = getattr(args, "frigate_plus_images", 0)
    frigate_plus_dir = Path(getattr(args, "frigate_plus_dir", "./datasets/frigate-plus")).resolve()

    class_names = load_class_names(dataset_dir, args.labels_path)
    target_split = args.split

    pairs = []
    if dataset_dir.is_dir():
        pairs = collect_dataset_pairs(dataset_dir, target_split)

    if not pairs and coco_images_req <= 0 and frigate_plus_images_req <= 0:
        print(f"[ERROR] No image/label pairs found in {dataset_dir} for split '{target_split}'.")
        sys.exit(1)

    # 1. Sample local YOLO dataset
    random.seed(args.seed)
    yolo_selected = []
    if pairs:
        if not args.all and args.num_images > len(pairs):
            all_pairs = collect_dataset_pairs(dataset_dir, "all")
            if target_split == "val" and len(all_pairs) > len(pairs):
                print(f"[WARN] Split '{target_split}' has {len(pairs)} images, but {args.num_images} requested.")
                print(f"[INFO] Expanding to '--split all' ({len(all_pairs)} images) to satisfy request.")
                pairs = all_pairs
                target_split = "all"

        if args.all:
            yolo_selected = pairs
        elif args.num_images > 0:
            yolo_selected = pairs if args.num_images >= len(pairs) else random.sample(pairs, args.num_images)
        elif args.num_images == 0 and coco_images_req <= 0 and frigate_plus_images_req <= 0:
            yolo_selected = pairs

    # 2. Sample external COCO images
    coco_selected = []
    if coco_images_req > 0:
        coco_pairs = collect_coco_pairs(coco_dataset_name, datasets_dir=datasets_dir)
        if not coco_pairs:
            print(f"[WARN] No images available in COCO dataset '{coco_dataset_name}'.")
        else:
            coco_rng = random.Random(args.seed + 1000)
            coco_selected = coco_pairs if coco_images_req >= len(coco_pairs) else coco_rng.sample(coco_pairs, coco_images_req)

    # 3. Sample Frigate+ images
    fp_selected = []
    if frigate_plus_images_req > 0:
        if not frigate_plus_dir.is_dir():
            print(f"[WARN] Frigate+ dataset directory not found at: {frigate_plus_dir}")
        else:
            fp_pairs = collect_dataset_pairs(frigate_plus_dir, target_split)
            if not fp_pairs:
                print(f"[WARN] No images found in Frigate+ directory: {frigate_plus_dir}")
            else:
                fp_rng = random.Random(args.seed + 2000)
                fp_selected = fp_pairs if frigate_plus_images_req >= len(fp_pairs) else fp_rng.sample(fp_pairs, frigate_plus_images_req)

    selected_pairs = yolo_selected + coco_selected + fp_selected
    if not selected_pairs:
        print("[ERROR] No images selected for evaluation.")
        sys.exit(1)

    resolved_device = resolve_execution_device(str(model_path), args.device)
    print("=" * 90)
    print("                 NVIDIA GPU / ONNX MODEL ACCURACY BENCHMARK")
    print("=" * 90)
    print(f"Model Path      : {model_path} ({'ONNX' if str(model_path).endswith('.onnx') else 'PyTorch'})")
    print(f"Execution Device: {resolved_device.upper()} (Requested: {args.device})")
    print(f"Resolution      : {args.imgsz}x{args.imgsz}")
    print(f"Evaluation Pool : {len(selected_pairs)} total ({len(yolo_selected)} dataset, {len(coco_selected)} COCO, {len(fp_selected)} Frigate+)")
    print(f"Confidence Conf : {args.threshold:.2f} | Matching IoU Thresh: {args.iou_threshold:.2f}")
    print("-" * 90)

    # Load model
    print(f"[INFO] Loading model with Ultralytics YOLO runtime on '{resolved_device}'...")
    model = YOLO(str(model_path))

    # Initialize tracking structures
    all_known_classes = set(class_names.values())
    stats_per_class = {
        cls: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
        for cls in all_known_classes
    }

    inference_latencies = []
    total_pipeline_latencies = []

    # Warmup inference
    dummy_img = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
    try:
        model(dummy_img, imgsz=args.imgsz, conf=args.threshold, device=resolved_device, verbose=False)
    except Exception as e:
        print(f"[WARN] Warmup inference error: {e}")

    print(f"\n[RUNNING] Executing inference across {len(selected_pairs)} images...\n")

    for idx, (img_path, lbl_path) in enumerate(selected_pairs, 1):
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        gt_boxes = parse_ground_truth(lbl_path, img_w, img_h, class_names)
        for gt in gt_boxes:
            lbl = gt["label"]
            if lbl not in stats_per_class:
                stats_per_class[lbl] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
            stats_per_class[lbl]["gt"] += 1

        t0 = time.perf_counter()
        try:
            results = model(
                str(img_path),
                imgsz=args.imgsz,
                conf=args.threshold,
                iou=args.iou_threshold,
                device=resolved_device,
                verbose=False,
            )
            t_total = (time.perf_counter() - t0) * 1000.0
            total_pipeline_latencies.append(t_total)
            speed_dict = getattr(results[0], "speed", {})
            inf_time = speed_dict.get("inference", t_total)
            inference_latencies.append(inf_time)
        except Exception as e:
            print(f"[WARN] Failed prediction on {img_path.name}: {e}")
            for gt in gt_boxes:
                stats_per_class[gt["label"]]["fn"] += 1
            continue

        parsed_preds = []
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                pred_lbl = str(class_names.get(cls_id, config_utils.get_coco_80_dict().get(cls_id, str(cls_id)))).lower()

                parsed_preds.append({
                    "label": pred_lbl,
                    "confidence": conf,
                    "bbox": xyxy,
                    "matched": False,
                })
                if pred_lbl not in stats_per_class:
                    stats_per_class[pred_lbl] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
                stats_per_class[pred_lbl]["pred"] += 1

        # Sort predictions by confidence descending
        parsed_preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Match predictions against ground truth
        for pred in parsed_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt["matched"] or gt["label"] != pred["label"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= args.iou_threshold and best_gt_idx >= 0:
                gt_boxes[best_gt_idx]["matched"] = True
                pred["matched"] = True
                stats_per_class[pred["label"]]["tp"] += 1
                stats_per_class[pred["label"]]["ious"].append(best_iou)
            else:
                stats_per_class[pred["label"]]["fp"] += 1

        for gt in gt_boxes:
            if not gt["matched"]:
                stats_per_class[gt["label"]]["fn"] += 1

        if args.verbose:
            print(f"[{idx:03d}/{len(selected_pairs):03d}] {img_path.name:<32} | GT: {len(gt_boxes):>2} | Preds: {len(parsed_preds):>2} | Infer: {inf_time:.1f}ms")

    # Generate Output Report
    total_gt = sum(v["gt"] for v in stats_per_class.values())
    total_pred = sum(v["pred"] for v in stats_per_class.values())
    total_tp = sum(v["tp"] for v in stats_per_class.values())
    total_fp = sum(v["fp"] for v in stats_per_class.values())
    total_fn = sum(v["fn"] for v in stats_per_class.values())
    all_ious = [iou for v in stats_per_class.values() for iou in v["ious"]]

    overall_prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
    overall_miou = (sum(all_ious) / len(all_ious)) if all_ious else 0.0

    print("=" * 90)
    print("                    NVIDIA MODEL ACCURACY REPORT")
    print("=" * 90)
    print(f"Model File      : {model_path.name}")
    print(f"Device / Runtime: {resolved_device.upper()} | imgsz: {args.imgsz}")
    print(f"Dataset Split   : {target_split} ({len(selected_pairs)} images: {len(yolo_selected)} dataset, {len(coco_selected)} COCO, {len(fp_selected)} Frigate+)")
    print(f"Thresholds      : Confidence >= {args.threshold:.2f} | Matching IoU >= {args.iou_threshold:.2f}")
    print("-" * 90)
    print(f"{'Class':<14} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'mIoU':>8}")
    print("-" * 90)

    sorted_classes = sorted(
        stats_per_class.keys(),
        key=lambda c: (stats_per_class[c]["gt"] == 0 and stats_per_class[c]["pred"] == 0, c),
    )

    class_report_data = {}
    for c in sorted_classes:
        st = stats_per_class[c]
        gt_c, pred_c, tp_c, fp_c, fn_c = st["gt"], st["pred"], st["tp"], st["fp"], st["fn"]
        prec_c = (tp_c / (tp_c + fp_c) * 100.0) if (tp_c + fp_c) > 0 else 0.0
        rec_c = (tp_c / (tp_c + fn_c) * 100.0) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        miou_c = (sum(st["ious"]) / len(st["ious"])) if st["ious"] else 0.0

        class_report_data[c] = {
            "gt": gt_c,
            "pred": pred_c,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": round(prec_c, 2),
            "recall": round(rec_c, 2),
            "f1_score": round(f1_c, 2),
            "mean_iou": round(miou_c, 3),
        }

        # Print active classes
        if gt_c > 0 or pred_c > 0:
            prec_str = f"{prec_c:.1f}%" if (tp_c + fp_c) > 0 else "N/A"
            rec_str = f"{rec_c:.1f}%" if (tp_c + fn_c) > 0 else "N/A"
            f1_str = f"{f1_c:.1f}%" if (prec_c + rec_c) > 0 else "N/A"
            miou_str = f"{miou_c:.2f}" if st["ious"] else "N/A"
            print(f"{c:<14} {gt_c:>6} {pred_c:>6} {tp_c:>6} {fp_c:>6} {fn_c:>6} {prec_str:>11} {rec_str:>10} {f1_str:>10} {miou_str:>8}")

    print("-" * 90)
    print(f"{'OVERALL':<14} {total_gt:>6} {total_pred:>6} {total_tp:>6} {total_fp:>6} {total_fn:>6} {overall_prec:>10.1f}% {overall_rec:>9.1f}% {overall_f1:>9.1f}% {overall_miou:>8.2f}")
    print("=" * 90)

    avg_inf = np.mean(inference_latencies) if inference_latencies else 0.0
    min_inf = np.min(inference_latencies) if inference_latencies else 0.0
    max_inf = np.max(inference_latencies) if inference_latencies else 0.0
    fps = (1000.0 / avg_inf) if avg_inf > 0 else 0.0

    print("\nSummary Statistics:")
    print(f"  Images Tested        : {len(selected_pairs)} ({len(yolo_selected)} dataset, {len(coco_selected)} COCO, {len(fp_selected)} Frigate+)")
    print(f"  Ground Truth Items   : {total_gt}")
    print(f"  Detections Produced  : {total_pred} (True Positives: {total_tp}, False Positives: {total_fp})")
    print(f"  Overall Precision    : {overall_prec:.1f}%")
    print(f"  Overall Recall       : {overall_rec:.1f}%")
    print(f"  Overall F1-Score     : {overall_f1:.1f}%")
    print(f"  Mean Bounding-Box IoU: {overall_miou:.3f}")
    print(f"  Inference Latency    : Avg: {avg_inf:.1f}ms (~{fps:.0f} FPS) | Min: {min_inf:.1f}ms | Max: {max_inf:.1f}ms\n")

    if args.output_json:
        report_payload = {
            "model_path": str(model_path),
            "device": resolved_device,
            "imgsz": args.imgsz,
            "dataset_split": target_split,
            "images_tested": len(selected_pairs),
            "dataset_images": len(yolo_selected),
            "coco_images": len(coco_selected),
            "frigate_plus_images": len(fp_selected),
            "confidence_threshold": args.threshold,
            "iou_threshold": args.iou_threshold,
            "summary": {
                "total_ground_truth": total_gt,
                "total_predictions": total_pred,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "precision": round(overall_prec, 2),
                "recall": round(overall_rec, 2),
                "f1_score": round(overall_f1, 2),
                "mean_iou": round(overall_miou, 3),
            },
            "timing_ms": {
                "avg_inference_ms": round(avg_inf, 2),
                "min_inference_ms": round(min_inf, 2),
                "max_inference_ms": round(max_inf, 2),
                "fps": round(fps, 1),
            },
            "class_metrics": class_report_data,
        }
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"[INFO] Detailed evaluation report saved to: {args.output_json}")


if __name__ == "__main__":
    evaluate(parse_args())
