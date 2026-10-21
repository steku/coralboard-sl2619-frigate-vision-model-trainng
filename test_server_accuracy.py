#!/usr/bin/env python3
"""
Test Coralboard Torq NPU Vision Proxy Server Accuracy.

Samples a configurable number of random images and their matching ground truth labels
from datasets/yolo/, sends HTTP requests mimicking Frigate NVR (multipart/form-data),
and generates a comprehensive detection accuracy and latency report.

Metrics reported:
  - Ground Truth (GT), Predictions (Pred), True Positives (TP), False Positives (FP), False Negatives (FN)
  - Precision, Recall, F1-Score, and Mean Bounding-Box IoU (mIoU) per class and overall
  - End-to-end HTTP roundtrip latency and hardware NPU inference execution times
  - Optional machine-readable JSON evaluation export (--output-json)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from PIL import Image

import config_utils

_dep = config_utils.get_deployment_settings()
DEFAULT_SERVER_URL = _dep["server_url"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark detection and label accuracy against the Coralboard Torq NPU proxy server.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Vision server endpoint URL (default: {DEFAULT_SERVER_URL})",
    )
    default_ds = "./datasets/yolo"
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
        help="Number of random images to test (default: 60, set 0 or use --all for all images)",
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
        type=float,
        default=0.20,
        help="Confidence score threshold sent to detector (default: 0.20)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Intersection-over-Union (IoU) threshold for matching boxes (default: 0.45)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="HTTP request timeout in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        help="Optional path to export evaluation report as JSON",
    )
    parser.add_argument(
        "--coco-images",
        dest="coco_images",
        type=int,
        default=0,
        help="Number of random COCO dataset images to include in benchmark (default: 0)",
    )
    default_coco_ds = config_utils.load_config().get("coco", {}).get("dataset", "coco-val2017")
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
        dest="frigate_plus_images",
        type=int,
        default=0,
        help="Number of random Frigate+ dataset images to include in benchmark (default: 0)",
    )
    default_fp_dir = config_utils.load_config().get("frigate_plus", {}).get("path", "./datasets/frigate-plus")
    parser.add_argument(
        "--frigate-plus-dir",
        default=default_fp_dir,
        help=f"Directory containing Frigate+ dataset (default: {default_fp_dir})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-image detection results",
    )
    return parser.parse_args()


def load_class_names(dataset_dir: Path) -> Dict[int, str]:
    """Load class index to name mapping from data.yaml or models/labels/*.json."""
    data_yaml = dataset_dir / "data.yaml"
    try:
        return config_utils.load_dataset_classes(str(data_yaml))
    except Exception as e:
        print(f"[WARN] Failed reading dataset classes ({e}), falling back to default mapping")
        return {0: "person", 1: "dog", 2: "cat", 3: "car"}


def collect_dataset_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    """Locate all matching (image_path, label_path) pairs."""
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

    # Map image directory to label directory (e.g. .../images/val2017 -> .../labels/val2017)
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
            label_name = str(label_name).lower()
            boxes.append({
                "class_id": class_id,
                "label": label_name,
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


def check_server_health(server_url: str) -> None:
    """Check if the server is reachable and report its status."""
    health_url = server_url.split("/v1/")[0] + "/health"
    try:
        resp = requests.get(health_url, timeout=2.0)
        if resp.status_code == 200:
            info = resp.json()
            backend = info.get("backend", "unknown")
            model = info.get("model_path", "unknown")
            ready = info.get("npu_ready", False)
            print(f"[INFO] Server connection verified: {health_url}")
            print(f"       Backend: {backend} | Model: {model} | NPU Ready: {ready}")
        else:
            print(f"[INFO] Server returned HTTP {resp.status_code} at {health_url}")
    except Exception:
        # Proceed even if health endpoint is not available
        pass


def send_detection_request(server_url: str, img_path: Path, threshold: float, timeout: float) -> Tuple[Optional[dict], float]:
    """Send detection request formatted as Frigate DeepStack multipart payload."""
    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        files = {"image": (img_path.name, f, "image/jpeg")}
        data = {"min_confidence": threshold}
        resp = requests.post(server_url, files=files, data=data, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    return resp.json(), latency_ms


def evaluate(args):
    dataset_dir = Path(args.dataset_dir).resolve()
    coco_images_req = getattr(args, "coco_images", 0)
    coco_dataset_name = getattr(args, "coco_dataset", "coco-val2017")
    datasets_dir = getattr(args, "datasets_dir", "./datasets")
    frigate_plus_images_req = getattr(args, "frigate_plus_images", 0)
    frigate_plus_dir_str = getattr(args, "frigate_plus_dir", "./datasets/frigate-plus")
    frigate_plus_dir = Path(frigate_plus_dir_str).resolve()

    has_dataset_dir = dataset_dir.is_dir()
    pairs = []
    if has_dataset_dir:
        class_names = load_class_names(dataset_dir)
        target_split = args.split
        pairs = collect_dataset_pairs(dataset_dir, target_split)
    else:
        class_names = config_utils.get_coco_80_dict()
        target_split = args.split

    if not pairs and coco_images_req <= 0 and frigate_plus_images_req <= 0:
        print(f"[ERROR] No image/label pairs found in {dataset_dir} for split '{target_split}'.")
        sys.exit(1)

    # Sample dataset pairs
    random.seed(args.seed)
    yolo_selected = []
    if pairs:
        # Handle case where requested num_images exceeds available images in the split
        if not args.all and args.num_images > len(pairs):
            all_pairs = collect_dataset_pairs(dataset_dir, "all")
            if target_split == "val" and len(all_pairs) > len(pairs):
                print(f"[WARN] Split '{target_split}' only contains {len(pairs)} images, but {args.num_images} were requested.")
                print(f"[INFO] Automatically expanding to '--split all' ({len(all_pairs)} images total) to evaluate the requested {args.num_images} images.")
                pairs = all_pairs
                target_split = "all"
            else:
                print(f"[WARN] Requested {args.num_images} images, but split '{target_split}' only contains {len(pairs)} images.")
                print(f"[INFO] Evaluating all {len(pairs)} available images in '{target_split}'.")

        print(f"[INFO] Discovered {len(pairs)} candidate images in split '{target_split}'.")

        if args.all:
            yolo_selected = pairs
        elif args.num_images > 0:
            if args.num_images >= len(pairs):
                yolo_selected = pairs
            else:
                yolo_selected = random.sample(pairs, args.num_images)
        elif args.num_images == 0 and coco_images_req <= 0 and frigate_plus_images_req <= 0:
            yolo_selected = pairs

    # Sample COCO pairs if requested
    coco_selected = []
    if coco_images_req > 0:
        coco_pairs = collect_coco_pairs(coco_dataset_name, datasets_dir)
        if not coco_pairs:
            print(f"[WARN] No valid image/label pairs found in COCO dataset '{coco_dataset_name}'.")
        else:
            print(f"[INFO] Discovered {len(coco_pairs)} candidate images in COCO dataset '{coco_dataset_name}'.")
            if coco_images_req >= len(coco_pairs):
                coco_selected = coco_pairs
            else:
                coco_rng = random.Random(args.seed + 1000)
                coco_selected = coco_rng.sample(coco_pairs, coco_images_req)

    # Sample Frigate+ pairs if requested
    fp_selected = []
    if frigate_plus_images_req > 0:
        if not frigate_plus_dir.is_dir():
            print(f"[WARN] Frigate+ dataset directory not found at: {frigate_plus_dir}")
        else:
            fp_pairs = collect_dataset_pairs(frigate_plus_dir, target_split)
            if not fp_pairs and target_split != "all":
                all_fp_pairs = collect_dataset_pairs(frigate_plus_dir, "all")
                if all_fp_pairs:
                    print(f"[INFO] Split '{target_split}' empty in Frigate+, using 'all' ({len(all_fp_pairs)} images).")
                    fp_pairs = all_fp_pairs
            if not fp_pairs:
                print(f"[WARN] No valid image/label pairs found in Frigate+ dataset at '{frigate_plus_dir}'.")
            else:
                print(f"[INFO] Discovered {len(fp_pairs)} candidate images in Frigate+ dataset '{frigate_plus_dir.name}'.")
                if frigate_plus_images_req >= len(fp_pairs):
                    fp_selected = fp_pairs
                else:
                    fp_rng = random.Random(args.seed + 2000)
                    fp_selected = fp_rng.sample(fp_pairs, frigate_plus_images_req)

    selected_pairs = yolo_selected + coco_selected + fp_selected
    if not selected_pairs:
        print("[ERROR] No images selected for evaluation. Please adjust --num-images, --coco-images, or --frigate-plus-images.")
        sys.exit(1)

    print(f"[INFO] Evaluating {len(selected_pairs)} total images ({len(yolo_selected)} dataset, {len(coco_selected)} COCO, {len(fp_selected)} Frigate+) against: {args.server_url}")
    check_server_health(args.server_url)

    # Initialize tracking structures
    all_known_classes = set(class_names.values())

    stats_per_class = {
        cls: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
        for cls in all_known_classes
    }

    roundtrip_latencies = []
    server_inference_times = []
    failed_requests = 0

    print(f"\n[RUNNING] Starting inference requests (score_threshold={args.threshold}, iou_threshold={args.iou_threshold})...\n")

    for idx, (img_path, lbl_path) in enumerate(selected_pairs, 1):
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        gt_boxes = parse_ground_truth(lbl_path, img_w, img_h, class_names)

        # Count ground truth instances
        for gt in gt_boxes:
            lbl = gt["label"]
            if lbl not in stats_per_class:
                stats_per_class[lbl] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []}
            stats_per_class[lbl]["gt"] += 1

        # Send request to server
        try:
            resp_json, latency_ms = send_detection_request(
                server_url=args.server_url,
                img_path=img_path,
                threshold=args.threshold,
                timeout=args.timeout,
            )
            roundtrip_latencies.append(latency_ms)
            if "timing" in resp_json and "inference_ms" in resp_json["timing"]:
                server_inference_times.append(float(resp_json["timing"]["inference_ms"]))
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Could not connect to vision server at {args.server_url}!")
            print("        Please ensure the proxy server is running and accessible.")
            sys.exit(1)
        except Exception as e:
            print(f"[WARN] Failed request for {img_path.name}: {e}")
            failed_requests += 1
            # Mark all GT as False Negatives
            for gt in gt_boxes:
                stats_per_class[gt["label"]]["fn"] += 1
            continue

        raw_predictions = resp_json.get("predictions", [])

        # Parse predictions
        parsed_preds = []
        for p in raw_predictions:
            pred_lbl = str(p.get("label", "unknown")).lower()
            conf = float(p.get("confidence", 0.0))

            # Handle both normalized [0..1] and absolute pixel coordinates
            if float(p.get("x_max", 0)) <= 1.05 and float(p.get("y_max", 0)) <= 1.05:
                px1 = float(p.get("x_min", 0)) * img_w
                py1 = float(p.get("y_min", 0)) * img_h
                px2 = float(p.get("x_max", 0)) * img_w
                py2 = float(p.get("y_max", 0)) * img_h
            else:
                px1 = float(p.get("x_min", 0))
                py1 = float(p.get("y_min", 0))
                px2 = float(p.get("x_max", 0))
                py2 = float(p.get("y_max", 0))

            parsed_preds.append({
                "label": pred_lbl,
                "confidence": conf,
                "bbox": [px1, py1, px2, py2],
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
                if gt["matched"]:
                    continue
                if gt["label"] != pred["label"]:
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

        # Any unmatched GT boxes are false negatives
        for gt in gt_boxes:
            if not gt["matched"]:
                stats_per_class[gt["label"]]["fn"] += 1

        if args.verbose:
            print(f"[{idx:02d}/{len(selected_pairs):02d}] {img_path.name} | GT: {len(gt_boxes)} | Preds: {len(parsed_preds)} | {latency_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # Generate Output Report
    # -------------------------------------------------------------------------
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
    print("                    CORALBOARD TORQ NPU ACCURACY REPORT")
    print("=" * 90)
    print(f"Endpoint URL    : {args.server_url}")
    print(f"Dataset Split   : {target_split} ({len(selected_pairs)} images: {len(yolo_selected)} dataset, {len(coco_selected)} COCO)")
    print(f"Confidence Thresh: {args.threshold:.2f} | Matching IoU Thresh: {args.iou_threshold:.2f}")
    print("-" * 90)
    print(f"{'Class':<12} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'mIoU':>8}")
    print("-" * 90)

    # Sort classes: active classes first, then alphabetical
    sorted_classes = sorted(
        stats_per_class.keys(),
        key=lambda c: (stats_per_class[c]["gt"] == 0 and stats_per_class[c]["pred"] == 0, c),
    )

    class_report_data = {}

    for c in sorted_classes:
        st = stats_per_class[c]
        gt_c = st["gt"]
        pred_c = st["pred"]
        tp_c = st["tp"]
        fp_c = st["fp"]
        fn_c = st["fn"]

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
            "tested": bool(gt_c > 0),
        }

        # Format output
        p_str = f"{prec_c:6.1f}%" if (tp_c + fp_c) > 0 else "   N/A"
        r_str = f"{rec_c:6.1f}%" if (tp_c + fn_c) > 0 else "   N/A"
        f_str = f"{f1_c:6.1f}%" if (prec_c + rec_c) > 0 else "   N/A"
        miou_str = f"{miou_c:6.2f}" if st["ious"] else "   N/A"

        print(f"{c:<12} {gt_c:6d} {pred_c:6d} {tp_c:6d} {fp_c:6d} {fn_c:6d} {p_str:>11} {r_str:>10} {f_str:>10} {miou_str:>8}")

    print("-" * 90)
    print(f"{'OVERALL':<12} {total_gt:6d} {total_pred:6d} {total_tp:6d} {total_fp:6d} {total_fn:6d} {overall_prec:10.1f}% {overall_rec:9.1f}% {overall_f1:9.1f}% {overall_miou:7.2f}")
    print("=" * 90)

    # Latency summary
    avg_latency = (sum(roundtrip_latencies) / len(roundtrip_latencies)) if roundtrip_latencies else 0.0
    min_latency = min(roundtrip_latencies) if roundtrip_latencies else 0.0
    max_latency = max(roundtrip_latencies) if roundtrip_latencies else 0.0
    avg_npu_ms = (sum(server_inference_times) / len(server_inference_times)) if server_inference_times else None

    print("\nSummary Statistics:")
    print(f"  Images Tested       : {len(selected_pairs)} ({len(yolo_selected)} dataset, {len(coco_selected)} COCO, {len(fp_selected)} Frigate+ | Failed requests: {failed_requests})")
    print(f"  Ground Truth Items  : {total_gt}")
    print(f"  Detections Produced : {total_pred} (True Positives: {total_tp}, False Positives: {total_fp})")
    print(f"  Overall Precision   : {overall_prec:.1f}%")
    print(f"  Overall Recall      : {overall_rec:.1f}%")
    print(f"  Overall F1-Score    : {overall_f1:.1f}%")
    print(f"  Mean Bounding-Box IoU: {overall_miou:.3f}")
    print(f"  HTTP Roundtrip Time : Avg: {avg_latency:.1f}ms | Min: {min_latency:.1f}ms | Max: {max_latency:.1f}ms")
    if avg_npu_ms is not None:
        print(f"  Server NPU Infer Time: Avg: {avg_npu_ms:.1f}ms (~{1000.0 / avg_npu_ms:.0f} FPS raw NPU capability)")
    print("")

    out_data = {
        "server_url": args.server_url,
        "split": target_split,
        "images_tested": len(selected_pairs),
        "dataset_images_tested": len(yolo_selected),
        "coco_images_tested": len(coco_selected),
        "frigate_plus_images_tested": len(fp_selected),
        "score_threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "overall": {
            "gt": total_gt,
            "pred": total_pred,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_prec, 2),
            "recall": round(overall_rec, 2),
            "f1_score": round(overall_f1, 2),
            "mean_iou": round(overall_miou, 3),
        },
        "latency": {
            "avg_roundtrip_ms": round(avg_latency, 2),
            "min_roundtrip_ms": round(min_latency, 2),
            "max_roundtrip_ms": round(max_latency, 2),
            "avg_server_npu_ms": round(avg_npu_ms, 2) if avg_npu_ms else None,
        },
        "per_class": class_report_data,
    }

    if getattr(args, "output_json", None):
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"[INFO] Detailed evaluation report saved to: {args.output_json}")

    return out_data


def run_accuracy_benchmark(
    server_url: str = DEFAULT_SERVER_URL,
    dataset_dir: str = "./datasets/yolo" if Path("datasets/yolo").is_dir() else "./yolo_dataset",
    num_images: int = 60,
    coco_images: int = 0,
    coco_dataset: Optional[str] = None,
    frigate_plus_images: int = 0,
    frigate_plus_dir: Optional[str] = None,
    datasets_dir: str = "./datasets",
    split: str = "all",
    all_images: bool = False,
    threshold: float = 0.20,
    iou_threshold: float = 0.45,
    timeout: float = 2.0,
    seed: int = 42,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Programmatic entrypoint to benchmark the vision proxy server and return the evaluation report.
    Prints the evaluation table to stdout and returns the out_data dict.
    """
    default_coco_ds = config_utils.load_config().get("coco", {}).get("dataset", "coco-val2017")
    default_fp_dir = config_utils.load_config().get("frigate_plus", {}).get("path", "./datasets/frigate-plus")
    args = argparse.Namespace(
        server_url=server_url,
        dataset_dir=dataset_dir,
        num_images=num_images,
        coco_images=coco_images,
        coco_dataset=coco_dataset or default_coco_ds,
        frigate_plus_images=frigate_plus_images,
        frigate_plus_dir=frigate_plus_dir or default_fp_dir,
        datasets_dir=datasets_dir,
        split=split,
        all=all_images,
        threshold=threshold,
        iou_threshold=iou_threshold,
        timeout=timeout,
        seed=seed,
        output_json=output_json,
        verbose=verbose,
    )
    return evaluate(args)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
