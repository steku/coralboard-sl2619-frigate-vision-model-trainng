#!/usr/bin/env python3
"""
Folder to CVAT: Upload images from a local folder to CVAT with GPU model pre-detections
and labels from an ONNX model or a YAML file.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
import yaml

import config_utils

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_default_model() -> str:
    """Find the best default NVIDIA model (.pt for GPU or .onnx)."""
    candidates = [
        Path("models/nvidia/nvidia_yolov9s.onnx"),
        Path("staging/yolov9s/weights/best.pt"),
        Path("models/nvidia/nvidia_yolo11s.onnx"),
        Path("staging/yolo11s/weights/best.pt"),
        Path("models/base/yolov9s.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return "models/nvidia/nvidia_yolov9s.onnx"


DEFAULT_MODEL = find_default_model()


def resolve_model_path(model_path: str | Path, device: str = "cuda") -> Path:
    """
    Resolve model path. When CUDA execution is requested on a GPU where ONNX Runtime
    cannot run CUDA (e.g. Pascal arch without CUDA 13 onnxruntime-gpu support),
    prefer the matching PyTorch .pt weights if available.
    """
    p = Path(model_path).resolve()
    if p.suffix.lower() == ".onnx" and device in ("cuda", "0"):
        pt_candidates: List[Path] = [p.with_suffix(".pt")]
        name_lower = p.name.lower()
        if "yolov9" in name_lower or "v9" in name_lower:
            pt_candidates.append(Path("staging/yolov9s/weights/best.pt"))
            pt_candidates.append(Path("staging/yolov9s/best.pt"))
            pt_candidates.append(Path("models/base/yolov9s.pt"))
        elif "yolo11" in name_lower or "11" in name_lower:
            pt_candidates.append(Path("staging/yolo11s/weights/best.pt"))
            pt_candidates.append(Path("staging/yolo11s/best.pt"))

        pt_candidates.append(Path("staging") / p.stem.replace("nvidia_", "") / "weights" / "best.pt")

        for cand in pt_candidates:
            if cand.is_file():
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        print(f"[INFO] Using PyTorch CUDA model '{cand}' for GPU acceleration on {gpu_name}.")
                        return cand.resolve()
                except Exception:
                    pass
    return p


def load_labels_from_yaml(yaml_path: str | Path) -> List[str]:
    """
    Extract a list of class labels from a YAML file.
    Supports:
      - labels: [person, dog, car]
      - categories: {person: 1, dog: 2, car: 3}
      - names: {0: person, 1: dog} or names: [person, dog] (YOLO style)
      - Top-level list: [person, dog, car]
      - Top-level dict: {person: 1, dog: 2}
    """
    path = Path(yaml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Labels YAML file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML file '{path}': {e}") from e

    if data is None:
        return []

    # 1. Top-level list of labels
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML content format in '{path}'; expected dict or list, got {type(data).__name__}")

    # 2. Check 'labels' key
    if "labels" in data:
        raw_labels = data["labels"]
        if isinstance(raw_labels, list):
            return [str(lbl).strip() for lbl in raw_labels if str(lbl).strip()]
        elif isinstance(raw_labels, dict):
            try:
                sorted_keys = sorted(raw_labels.keys(), key=lambda k: int(raw_labels[k]))
            except (ValueError, TypeError):
                sorted_keys = list(raw_labels.keys())
            return [str(k).strip() for k in sorted_keys if str(k).strip()]

    # 3. Check 'categories' key (standard config.yaml format)
    if "categories" in data and isinstance(data["categories"], dict):
        raw_cats = data["categories"]
        try:
            sorted_keys = sorted(raw_cats.keys(), key=lambda k: int(raw_cats[k]))
        except (ValueError, TypeError):
            sorted_keys = list(raw_cats.keys())
        return [str(k).strip() for k in sorted_keys if str(k).strip()]

    # 4. Check 'names' key (YOLO data.yaml format)
    if "names" in data:
        raw_names = data["names"]
        if isinstance(raw_names, list):
            return [str(name).strip() for name in raw_names if str(name).strip()]
        elif isinstance(raw_names, dict):
            try:
                sorted_keys = sorted(raw_names.keys(), key=lambda k: int(k))
            except (ValueError, TypeError):
                sorted_keys = list(raw_names.keys())
            return [str(raw_names[k]).strip() for k in sorted_keys if str(raw_names[k]).strip()]

    # 5. Top-level dictionary
    candidate_keys = [str(k).strip() for k in data.keys() if str(k).strip()]
    excluded_sections = {"frigate", "cvat", "storage", "modes", "deployment", "coco", "frigate_plus", "profiles", "remote_training"}
    filtered_keys = [k for k in candidate_keys if k.lower() not in excluded_sections]
    if filtered_keys:
        return filtered_keys

    return []


def load_model_classes(model_path: str | Path, labels_path: Optional[str | Path] = None) -> Dict[int, str]:
    """Load class index to name mapping from model JSON metadata, custom labels, or fallback."""
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
            print(f"[WARN] Failed to parse custom labels from '{labels_path}': {e}")

    m_path = Path(model_path).resolve()
    candidates = [
        m_path.with_suffix(".json"),
        m_path.parent / f"{m_path.stem}.json",
    ]
    name_lower = str(m_path).lower()
    if "v9" in name_lower or "yolov9" in name_lower:
        candidates.append(Path("models/nvidia/nvidia_yolov9s.json"))
        candidates.append(Path("models/nvidia/261029-1056/nvidia_yolov9s_261029-1056.json"))
        candidates.append(Path("models/nvidia/nvidia_yolo11s.json"))
    else:
        candidates.append(Path("models/nvidia/nvidia_yolo11s.json"))
        candidates.append(Path("models/nvidia/nvidia_yolov9s.json"))

    candidates.append(Path("models/nvidia/labels.json"))
    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                names_meta = meta.get("names", {})
                if isinstance(names_meta, dict):
                    return {int(k): str(v) for k, v in names_meta.items()}
                elif isinstance(names_meta, list):
                    return {i: str(v) for i, v in enumerate(names_meta)}
            except Exception as e:
                print(f"[WARN] Failed to parse model classes from '{cand}': {e}")

    return config_utils.get_coco_80_dict()


def resolve_execution_device(model_path: str | Path, requested_device: str = "cuda") -> str:
    """Resolve compatible execution device for ONNX Runtime or PyTorch."""
    is_onnx = str(model_path).endswith(".onnx")
    if is_onnx:
        if requested_device in ("cuda", "0"):
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return "cuda"
                else:
                    print("[INFO] onnxruntime-gpu CUDAExecutionProvider not available; running ONNX inference via CPU.")
                    return "cpu"
            except Exception:
                return "cpu"
        return "cpu"

    if requested_device in ("cuda", "0"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] CUDA requested but not available; falling back to CPU.")
                return "cpu"
        except Exception:
            return "cpu"
    return requested_device


def collect_images(folder_path: str | Path) -> List[Path]:
    """Discover all images in the folder."""
    path = Path(folder_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Specified image path is not a directory: {path}")

    images = [
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: (p.name.lower(), str(p)))
    return images


def generate_coco_annotations(
    images: List[Path],
    model_path: str | Path,
    device: str = "cuda",
    conf_thresh: float = 0.25,
    imgsz: int = 640,
    model_classes: Optional[Dict[int, str]] = None,
    allowed_labels: Optional[List[str]] = None,
) -> Tuple[dict, List[str]]:
    """
    Run detection inference on images and generate a COCO 1.0 dataset dictionary.
    Returns (coco_dict, list_of_label_names).
    """
    from PIL import Image
    from ultralytics import YOLO

    if model_classes is None:
        model_classes = load_model_classes(model_path)

    # Determine active classes
    if allowed_labels:
        allowed_set = {lbl.lower() for lbl in allowed_labels}
        filtered_classes = {k: v for k, v in model_classes.items() if v.lower() in allowed_set}
        # If any allowed_labels were not in model_classes, include them as well
        existing_names = {v.lower() for v in filtered_classes.values()}
        next_idx = max(model_classes.keys(), default=-1) + 1
        for lbl in allowed_labels:
            if lbl.lower() not in existing_names:
                filtered_classes[next_idx] = lbl
                next_idx += 1
        class_mapping = filtered_classes
    else:
        class_mapping = model_classes

    # Sort class IDs for deterministic ordering
    sorted_cls_ids = sorted(class_mapping.keys())
    labels = [class_mapping[i] for i in sorted_cls_ids]
    category_mapping = {class_mapping[cid]: (idx + 1) for idx, cid in enumerate(sorted_cls_ids)}

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for name, cid in category_mapping.items()
        ],
    }

    resolved_device = resolve_execution_device(model_path, device)
    print(f"[INFO] Running YOLO detection model '{model_path}' on device '{resolved_device.upper()}' (conf={conf_thresh:.2f}, imgsz={imgsz})...")
    model = YOLO(str(model_path))

    annotation_id = 1
    total_detections = 0
    detections_per_class: Dict[str, int] = {}

    for idx, img_path in enumerate(images, 1):
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"[WARN] Cannot read image file '{img_path.name}': {e}")
            continue

        image_id = idx
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        try:
            results = model(
                str(img_path),
                imgsz=imgsz,
                conf=conf_thresh,
                device=resolved_device,
                verbose=False,
            )
        except Exception as e:
            print(f"[WARN] Inference failed on '{img_path.name}': {e}")
            continue

        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            for b_idx in range(len(boxes)):
                cls_id = int(boxes.cls[b_idx].item())
                label_name = class_mapping.get(cls_id)
                if not label_name or label_name not in category_mapping:
                    continue

                xyxy = boxes.xyxy[b_idx].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy
                x = max(0.0, min(float(x1), float(img_w)))
                y = max(0.0, min(float(y1), float(img_h)))
                width = max(0.0, min(float(x2 - x1), float(img_w - x)))
                height = max(0.0, min(float(y2 - y1), float(img_h - y)))
                area = width * height

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_mapping[label_name],
                    "bbox": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                })
                annotation_id += 1
                total_detections += 1
                detections_per_class[label_name] = detections_per_class.get(label_name, 0) + 1

    print(f"[INFO] Generated {total_detections} pre-annotations across {len(images)} images.")
    if detections_per_class:
        print("[INFO] Detections breakdown by class:")
        for name, count in sorted(detections_per_class.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {count}")

    return coco_data, labels


save_cvat_task_state = config_utils.save_cvat_task_state


def upload_folder_to_cvat(
    images: List[Path],
    labels: List[str],
    task_name: str,
    cvat_url: str,
    cvat_user: str,
    cvat_pass: str,
    annotations_path: Optional[str | Path] = None,
    base_dir: Optional[Path] = None,
) -> int:
    """Connect to CVAT, create a task with labels, and upload all images and optional pre-annotations."""
    from cvat_sdk import make_client

    if not images:
        raise ValueError("Cannot create CVAT task with zero images.")
    if not labels:
        raise ValueError("Cannot create CVAT task with zero labels.")
    if not cvat_user or not cvat_pass:
        raise ValueError("CVAT username and password are required. Set in config.yaml or CVAT_USER / CVAT_PASS.")

    cvat_labels = [{"name": str(lbl)} for lbl in labels]
    image_paths = [str(p) for p in images]
    clean_url = cvat_url.rstrip("/")

    print(f"\n[INFO] Connecting to CVAT at {clean_url} as '{cvat_user}'...")
    with make_client(clean_url, credentials=(cvat_user, cvat_pass)) as client:
        print(f"[INFO] Creating CVAT task: '{task_name}' with {len(image_paths)} images and {len(cvat_labels)} labels...")
        task = client.tasks.create_from_data(
            spec={
                "name": task_name,
                "labels": cvat_labels,
            },
            resources=image_paths,
        )

        if annotations_path and Path(annotations_path).is_file():
            print(f"[INFO] Uploading COCO 1.0 pre-annotations from '{annotations_path}'...")
            task.import_annotations(
                format_name="COCO 1.0",
                filename=str(annotations_path),
            )

        task_url = f"{clean_url}/tasks/{task.id}"
        save_cvat_task_state(task.id, task.name, clean_url, task_url, base_dir=base_dir)

        print("\n" + "=" * 70)
        print(" [SUCCESS] CVAT Task Successfully Created!")
        print(f" Task ID  : {task.id}")
        print(f" Task Name: {task.name}")
        print(f" URL      : {task_url}")
        print(f" Images   : {len(image_paths)} uploaded")
        print(f" Labels   : {len(labels)} classes configured")
        if annotations_path:
            print(" Annotations: COCO 1.0 pre-annotations imported")
        print("=" * 70)
        return task.id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a folder of images into CVAT with GPU model pre-detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing images to import",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="Path to ONNX or PyTorch detection model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device for model detection: 'cuda', '0', 'cpu'",
    )
    parser.add_argument(
        "--conf", "--threshold",
        dest="conf",
        type=float,
        default=0.25,
        help="Confidence threshold for model detections (0.0 to 1.0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable model detection; upload images and YAML labels only without pre-annotations",
    )
    parser.add_argument(
        "-l", "--labels-file",
        dest="labels_file",
        default=None,
        help="Path to YAML file containing labels (defaults to config.yaml if present; filters detections when provided)",
    )
    parser.add_argument(
        "-t", "--task-name",
        default=None,
        help="Custom CVAT task name (defaults to '<folder_name> (YYYY-MM-DD HH:MM)')",
    )
    parser.add_argument(
        "--cvat-url",
        default=None,
        help="CVAT server base URL (overrides CVAT_URL / config.yaml)",
    )
    parser.add_argument(
        "--cvat-user",
        default=None,
        help="CVAT username (overrides CVAT_USER / config.yaml)",
    )
    parser.add_argument(
        "--cvat-pass",
        default=None,
        help="CVAT password (overrides CVAT_PASS / config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan images and run detections without uploading to CVAT",
    )
    config_utils.add_logging_args(parser)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    config_utils.setup_logging_from_args(args, "folder-to-cvat")

    # 1. Folder validation
    folder_path = Path(args.folder).resolve()
    if not folder_path.is_dir():
        print(f"[ERROR] Directory does not exist: {folder_path}")
        sys.exit(1)

    # 2. Ingest all images
    try:
        images = collect_images(folder_path)
    except Exception as e:
        print(f"[ERROR] Failed to collect images: {e}")
        sys.exit(1)

    if not images:
        print(f"[ERROR] No supported images found in '{folder_path}' ({', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}).")
        sys.exit(1)

    print(f"[INFO] Found {len(images)} images in '{folder_path}'.")

    # 3. Handle model detections vs YAML-only labels
    coco_data: Optional[dict] = None
    labels: List[str] = []

    if not args.no_detect:
        model_path = resolve_model_path(args.model, device=args.device)
        if not model_path.is_file():
            print(f"[ERROR] Model file not found at: {args.model}")
            print("        Specify a valid model with --model <path> or use --no-detect to upload without pre-annotations.")
            sys.exit(1)

        allowed_labels = None
        if args.labels_file:
            try:
                allowed_labels = load_labels_from_yaml(args.labels_file)
                print(f"[INFO] Loaded {len(allowed_labels)} target filter labels from '{args.labels_file}'.")
            except Exception as e:
                print(f"[ERROR] Failed to load labels from '{args.labels_file}': {e}")
                sys.exit(1)

        try:
            coco_data, labels = generate_coco_annotations(
                images=images,
                model_path=model_path,
                device=args.device,
                conf_thresh=args.conf,
                imgsz=args.imgsz,
                allowed_labels=allowed_labels,
            )
        except Exception as e:
            print(f"[ERROR] Detection inference failed: {e}")
            sys.exit(1)
    else:
        # Load labels from YAML
        yaml_target = args.labels_file
        if not yaml_target:
            detected_config = config_utils.find_config_file()
            if detected_config:
                yaml_target = str(detected_config)
            else:
                for candidate in ("labels.yaml", "labels.yml", "data.yaml"):
                    cand_path = folder_path / candidate
                    if cand_path.is_file():
                        yaml_target = str(cand_path)
                        break

        if not yaml_target:
            print("[ERROR] No labels YAML file specified and no default config.yaml found.")
            print("[ERROR] Specify labels YAML via -l / --labels-file <path_to_yaml>")
            sys.exit(1)

        try:
            labels = load_labels_from_yaml(yaml_target)
            print(f"[INFO] Loaded {len(labels)} labels from '{yaml_target}': {labels}")
        except Exception as e:
            print(f"[ERROR] Failed to load labels: {e}")
            sys.exit(1)

    if not labels:
        print("[ERROR] No valid labels available for CVAT task.")
        sys.exit(1)

    # 4. Resolve task name
    task_name = args.task_name
    if not task_name:
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        task_name = f"{folder_path.name} ({timestamp_str})"

    # 5. Dry-run
    if args.dry_run:
        print("\n" + "=" * 70)
        print(" [DRY-RUN] Validation Successful (No task created / No upload)")
        print(f" Target Folder: {folder_path}")
        print(f" Total Images : {len(images)}")
        print(f" First 5 Files: {[p.name for p in images[:5]]}")
        print(f" Labels ({len(labels)})  : {labels[:10]}{'...' if len(labels) > 10 else ''}")
        print(f" Task Name    : {task_name}")
        if coco_data:
            print(f" Detections   : {len(coco_data.get('annotations', []))} pre-annotations produced")
        print("=" * 70 + "\n")
        return 0

    # 6. CVAT credentials from config.yaml or environment
    cfg = config_utils.load_config(args.labels_file) if args.labels_file else config_utils.load_config()
    cvat_url, cvat_user, cvat_pass = config_utils.get_cvat_credentials(
        cfg=cfg,
        url=args.cvat_url,
        user=args.cvat_user,
        password=args.cvat_pass,
    )

    if not cvat_user or not cvat_pass:
        print("[ERROR] CVAT username and password are required.")
        print("[ERROR] Set them in 'config.yaml', via $CVAT_USER / $CVAT_PASS, or CLI flags.")
        sys.exit(1)

    # 7. Write annotations to temporary file if detections were generated
    temp_ann_path = None
    if coco_data and coco_data.get("annotations"):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(coco_data, tf, indent=2)
            temp_ann_path = tf.name

    # 8. Upload to CVAT
    try:
        upload_folder_to_cvat(
            images=images,
            labels=labels,
            task_name=task_name,
            cvat_url=cvat_url,
            cvat_user=cvat_user,
            cvat_pass=cvat_pass,
            annotations_path=temp_ann_path,
        )
    except Exception as e:
        print(f"[ERROR] Failed to upload folder to CVAT: {e}")
        sys.exit(1)
    finally:
        if temp_ann_path and os.path.exists(temp_ann_path):
            try:
                os.remove(temp_ann_path)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
