#!/usr/bin/env python3
"""
Folder to CVAT: Upload all images in a local folder to CVAT with labels from a YAML file.
"""

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import List, Optional
import yaml

import config_utils

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def save_cvat_task_state(
    task_id: int,
    task_name: str,
    cvat_url: str,
    task_url: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Save task info to .last_cvat_task."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    state_data = {
        "task_id": task_id,
        "task_name": task_name,
        "cvat_url": cvat_url,
        "task_url": task_url,
        "created_at": datetime.datetime.now().isoformat(),
    }
    state_file = base_dir / ".last_cvat_task"
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write state to {state_file}: {e}")
    return state_file


def upload_folder_to_cvat(
    images: List[Path],
    labels: List[str],
    task_name: str,
    cvat_url: str,
    cvat_user: str,
    cvat_pass: str,
    base_dir: Optional[Path] = None,
) -> int:
    """Connect to CVAT, create a task with labels, and upload all images."""
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

        task_url = f"{clean_url}/tasks/{task.id}"
        save_cvat_task_state(task.id, task.name, clean_url, task_url, base_dir=base_dir)

        print("\n" + "=" * 70)
        print(" [SUCCESS] CVAT Task Successfully Created!")
        print(f" Task ID  : {task.id}")
        print(f" Task Name: {task.name}")
        print(f" URL      : {task_url}")
        print(f" Images   : {len(image_paths)} uploaded")
        print(f" Labels   : {labels}")
        print("=" * 70)
        return task.id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a folder of images into CVAT with labels from a YAML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing images to import",
    )
    parser.add_argument(
        "-l", "--labels-file",
        dest="labels_file",
        default=None,
        help="Path to YAML file containing labels (defaults to config.yaml if present)",
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
        help="Scan images and parse labels without uploading to CVAT",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

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

    # 3. Load labels from YAML
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
        print(f"[ERROR] No valid labels found in '{yaml_target}'.")
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
        print(f" Labels ({len(labels)})  : {labels}")
        print(f" Task Name    : {task_name}")
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

    # 7. Upload to CVAT
    try:
        upload_folder_to_cvat(
            images=images,
            labels=labels,
            task_name=task_name,
            cvat_url=cvat_url,
            cvat_user=cvat_user,
            cvat_pass=cvat_pass,
        )
    except Exception as e:
        print(f"[ERROR] Failed to upload folder to CVAT: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
