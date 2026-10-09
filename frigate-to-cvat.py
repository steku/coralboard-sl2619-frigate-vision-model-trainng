#!/usr/bin/env python3
"""
Frigate to CVAT: Harvest snapshots & pre-annotations from Frigate NVR
and automatically create & upload review tasks in CVAT.

Supports:
- Configuration via YAML file (config.yaml) with CLI overrides.
- End-to-end harvesting and CVAT task creation in a single command.
- Offline/local-only harvest (--no-upload).
- Upload-only mode for re-trying or uploading an existing batch (--upload-only).
- Customizable Frigate and CVAT endpoints, credentials, filters, labels, and batch limits.
"""

import argparse
from collections import Counter
import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import requests
import yaml
from PIL import Image

import config_utils
import test_server_accuracy

DEFAULT_CONFIG_FILES = config_utils.DEFAULT_CONFIG_FILES

DEFAULT_CATEGORY_MAPPING = {
    "person": 1,
    "dog": 2,
    "cat": 3,
    "car": 4,
}

DEFAULT_CVAT_LABELS = [
    {"name": "person"},
    {"name": "dog"},
    {"name": "cat"},
    {"name": "car"},
]

PROCESSED_EVENTS_FILE = ".processed_events.json"


def load_processed_events(base_dir=None, yolo_dir="yolo_dataset", output_dir=None):
    """
    Collect all event IDs that have already been processed in previous trainings.
    Aggregates from:
    1. Persistent tracking file (.processed_events.json)
    2. Any existing files in yolo_dataset (train/val images and labels)
    3. Any existing files in output_dir (e.g. cvat_dataset)
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    tracker_file = base_dir / PROCESSED_EVENTS_FILE
    processed = set()

    # 1. Read persistent history
    if tracker_file.is_file():
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    processed.update(data.get("processed_event_ids", []))
                elif isinstance(data, list):
                    processed.update(data)
        except Exception as e:
            print(f"[WARN] Failed to read {tracker_file}: {e}")

    # 2. Scan yolo_dataset (images & labels)
    yolo_path = base_dir / yolo_dir
    if yolo_path.is_dir():
        for p in yolo_path.glob("**/*.*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".txt"):
                stem = re.sub(r"^t\d+_", "", p.stem)
                if stem and "." in stem:
                    processed.add(stem)

    # 3. Scan output_dir (e.g. cvat_dataset)
    if output_dir:
        out_path = Path(output_dir)
        if out_path.is_dir():
            ann = out_path / "annotations.json"
            if ann.is_file():
                try:
                    with open(ann, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        for img in meta.get("images", []):
                            fn = Path(img.get("file_name", "")).stem
                            if fn and "." in fn:
                                processed.add(fn)
                except Exception:
                    pass
            img_dir = out_path / "images"
            if img_dir.is_dir():
                for p in img_dir.glob("*.*"):
                    if p.stem and "." in p.stem:
                        processed.add(p.stem)

    return processed


def save_processed_events(new_event_ids, base_dir=None):
    """Save new event IDs into persistent tracking file."""
    if not new_event_ids:
        return

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    tracker_file = base_dir / PROCESSED_EVENTS_FILE
    existing = set()

    if tracker_file.is_file():
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    existing.update(data.get("processed_event_ids", []))
                elif isinstance(data, list):
                    existing.update(data)
        except Exception:
            pass

    existing.update(new_event_ids)
    data = {
        "processed_event_ids": sorted(list(existing)),
        "count": len(existing),
        "last_updated": datetime.datetime.now().isoformat(),
    }
    try:
        with open(tracker_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Recorded {len(new_event_ids)} new events to {tracker_file} (total tracked: {len(existing)}).")
    except Exception as e:
        print(f"[WARN] Failed to write {tracker_file}: {e}")


def fetch_events_from_frigate(
    frigate_url,
    target_labels,
    target_count,
    camera=None,
    processed_ids=None,
    timeout=10,
    max_pages=30,
):
    """
    Fetch events from Frigate API, skipping previously processed IDs and paginating backwards
    via 'before' timestamp until target_count events are found or no more events exist.
    """
    if processed_ids is None:
        processed_ids = set()

    events_url = f"{frigate_url.rstrip('/')}/api/events"
    collected = []
    seen_ids = set()
    before_ts = None
    batch_size = max(50, target_count * 2)

    for _ in range(max_pages):
        params = {
            "has_snapshot": 1,
            "limit": batch_size,
        }
        if camera:
            params["camera"] = camera
        if target_labels:
            params["labels"] = ",".join(target_labels)
        if before_ts is not None:
            params["before"] = before_ts

        try:
            resp = requests.get(events_url, params=params, timeout=timeout)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"  [WARN] Failed fetching events from Frigate: {e}")
            break

        if not batch:
            break

        for ev in batch:
            ev_id = ev.get("id")
            if not ev_id or ev_id in seen_ids:
                continue
            seen_ids.add(ev_id)

            # Skip events already processed in previous trainings
            if ev_id in processed_ids:
                continue

            # Ensure event has bounding box
            if not ev.get("data", {}).get("box"):
                continue

            collected.append(ev)
            if len(collected) >= target_count:
                return collected

        # Setup next page timestamp
        last_event_time = batch[-1].get("start_time")
        if not last_event_time or last_event_time == before_ts:
            break
        before_ts = last_event_time

    return collected


def fetch_cvat_object_counts(cvat_url, cvat_user, cvat_pass, timeout=10):
    """
    Fetch existing annotation counts per label across all tasks in CVAT via REST API.
    Returns a dict mapping label name to integer count, or None if CVAT query fails.
    """
    if not cvat_url or not cvat_user or not cvat_pass:
        return None

    base_url = cvat_url.rstrip("/")
    session = requests.Session()
    session.auth = (cvat_user, cvat_pass)

    try:
        r = session.get(f"{base_url}/api/tasks?page_size=100", timeout=timeout)
        r.raise_for_status()
        tasks = r.json().get("results", [])
    except Exception as e:
        print(f"[WARN] Failed connecting to CVAT at {base_url}: {e}")
        return None

    counts = Counter()
    for task in tasks:
        tid = task.get("id")
        if not tid:
            continue
        try:
            r_lbl = session.get(f"{base_url}/api/labels?task_id={tid}&page_size=100", timeout=timeout)
            if r_lbl.status_code == 200:
                id_to_name = {lbl["id"]: lbl["name"] for lbl in r_lbl.json().get("results", [])}
            else:
                id_to_name = {}

            r_ann = session.get(f"{base_url}/api/tasks/{tid}/annotations", timeout=timeout)
            if r_ann.status_code == 200:
                ann = r_ann.json()
                for item in ann.get("shapes", []) + ann.get("tracks", []) + ann.get("tags", []):
                    lid = item.get("label_id")
                    name = id_to_name.get(lid, f"label_{lid}")
                    counts[name] += 1
        except Exception as e:
            print(f"[WARN] Failed fetching annotations for CVAT task {tid}: {e}")

    return dict(counts)


def compute_cvat_balanced_quotas(cvat_counts, target_labels, limit):
    """
    Compute deficit-weighted quotas for target labels based on current CVAT annotations.
    Classes with fewer existing annotations receive larger quotas.
    Returns:
        sorted_labels: List[str] sorted by ascending CVAT count (fewest first)
        quotas: Dict[str, int] mapping label -> initial target quota
        deficits: Dict[str, int] mapping label -> deficit score
        health_status: Dict[str, str] mapping label -> 'CRITICAL', 'LOW', or 'GOOD'
    """
    counts = {lbl: int(cvat_counts.get(lbl, 0)) for lbl in target_labels}
    max_c = max(counts.values()) if counts and max(counts.values()) > 0 else 1

    # Health status based on count and imbalance ratio vs max
    health_status = {}
    for lbl, cnt in counts.items():
        ratio = (max_c / cnt) if cnt > 0 else 999.0
        if cnt < 20 or ratio > 5.0:
            health_status[lbl] = "CRITICAL"
        elif cnt < 100 or ratio > 2.5:
            health_status[lbl] = "LOW"
        else:
            health_status[lbl] = "GOOD"

    # Deficit floor: baseline weight for the top class (5% of max_c or at least 1)
    floor = max(1, round(max_c * 0.05))
    deficits = {lbl: max(1, (max_c - counts[lbl]) + floor) for lbl in target_labels}
    total_deficit = sum(deficits.values())

    # Raw quotas proportional to deficit
    quotas = {}
    for lbl in target_labels:
        quotas[lbl] = max(1, round(deficits[lbl] / total_deficit * limit))

    # Adjust rounding differences to match exact limit
    diff = limit - sum(quotas.values())
    sorted_by_deficit = sorted(target_labels, key=lambda l: deficits[l], reverse=True)
    for i in range(abs(diff)):
        lbl = sorted_by_deficit[i % len(sorted_by_deficit)]
        if diff > 0:
            quotas[lbl] += 1
        elif quotas[lbl] > 1:
            quotas[lbl] -= 1

    # Sort labels ascending by CVAT count so most starved classes query Frigate first
    sorted_labels = sorted(target_labels, key=lambda l: (counts[l], l))

    return sorted_labels, quotas, deficits, health_status


def extract_accuracy_scores(report_data, metric="f1_score"):
    """
    Extract per-class accuracy scores from test_server_accuracy evaluation data.
    Classes with score <= 0.0 or gt <= 0 (or tested == False) are considered untested and disregarded.
    Returns:
        tested_scores: Dict[str, float] mapping class name to numeric score
        disregarded: List[str] class names that were disregarded
    """
    if not isinstance(report_data, dict):
        return {}, []

    per_class = report_data.get("per_class", {})
    tested_scores = {}
    disregarded = []

    for cls_name, info in per_class.items():
        cls_name_clean = str(cls_name).lower().strip()
        if isinstance(info, (int, float)):
            score = float(info)
            gt = 1 if score > 0 else 0
            tested = (score > 0)
        elif isinstance(info, dict):
            score = float(info.get(metric, info.get("f1_score", 0.0)))
            gt = int(info.get("gt", 0))
            tested = bool(info.get("tested", gt > 0))
        else:
            continue

        if not tested or gt <= 0 or score <= 0.0:
            disregarded.append(cls_name_clean)
        else:
            tested_scores[cls_name_clean] = score

    return tested_scores, sorted(disregarded)


def compute_accuracy_balanced_quotas(scores, target_labels, limit, metric="f1_score"):
    """
    Compute deficit-weighted quotas based on model accuracy scores from test_server_accuracy.
    Classes with lower scores receive larger harvest quotas.
    Returns:
        sorted_labels: List[str] sorted by ascending accuracy (lowest score first)
        quotas: Dict[str, int] mapping label -> initial target quota
        deficits: Dict[str, float] mapping label -> deficit score
        health_status: Dict[str, str] mapping label -> 'CRITICAL', 'WEAK', or 'GOOD'
    """
    active_labels = [lbl for lbl in target_labels if lbl in scores and scores[lbl] > 0]
    if not active_labels:
        return [], {}, {}, {}

    max_score = max(scores[lbl] for lbl in active_labels)
    scale = 1.0 if max_score <= 1.0 else 100.0

    health_status = {}
    for lbl in active_labels:
        sc = scores[lbl]
        pct = sc if scale == 100.0 else (sc * 100.0)
        if pct < 40.0:
            health_status[lbl] = "CRITICAL"
        elif pct < 70.0:
            health_status[lbl] = "WEAK"
        else:
            health_status[lbl] = "GOOD"

    floor = 1.0
    deficits = {lbl: max(floor, (scale - scores[lbl]) + floor) for lbl in active_labels}
    total_deficit = sum(deficits.values())

    quotas = {}
    for lbl in active_labels:
        quotas[lbl] = max(1, round(deficits[lbl] / total_deficit * limit))

    diff = limit - sum(quotas.values())
    sorted_by_deficit = sorted(active_labels, key=lambda l: deficits[l], reverse=True)
    for i in range(abs(diff)):
        lbl = sorted_by_deficit[i % len(sorted_by_deficit)]
        if diff > 0:
            quotas[lbl] += 1
        elif quotas[lbl] > 1:
            quotas[lbl] -= 1

    # Sort labels ascending by score so the weakest classes query Frigate first
    sorted_labels = sorted(active_labels, key=lambda l: (scores[l], l))

    return sorted_labels, quotas, deficits, health_status


def load_accuracy_report(report_path):
    """Load JSON accuracy report from file path."""
    if not report_path or not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read accuracy report from '{report_path}': {e}")
        return None


def find_config_file(custom_path=None):
    """Locate configuration YAML file from explicit path or default candidates."""
    res = config_utils.find_config_file(custom_path)
    if custom_path and not res:
        print(f"[ERROR] Specified configuration file not found: {custom_path}")
        sys.exit(1)
    return res


def load_config(config_path=None):
    """
    Load settings from YAML file and merge with base defaults and environment variables.
    Precedence: CLI args > Environment variables > YAML config file > Hardcoded defaults.
    """
    cfg_file = find_config_file(config_path)
    cfg = {}

    if cfg_file:
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"[INFO] Loaded configuration from {cfg_file}")
        except Exception as e:
            print(f"[ERROR] Failed to parse config file '{cfg_file}': {e}")
            sys.exit(1)
    else:
        print("[INFO] No configuration file found; using default settings.")
        example_file = Path(__file__).resolve().parent / "config.example.yaml"
        if example_file.is_file():
            print("[TIP] You can copy 'config.example.yaml' to 'config.yaml' to configure your credentials.")

    # Base defaults
    defaults = {
        "config_file": str(cfg_file) if cfg_file else "config.yaml",
        "frigate_url": "http://localhost:5000",
        "camera": None,
        "limit": 200,
        "balance_mode": "cvat-deficit",
        "skip_processed": True,
        "clean": False,
        "snapshot_endpoint": "snapshot-clean.webp",
        "timeout": 10,
        "output_dir": "./cvat_dataset",
        "cvat_url": "http://192.168.2.1:8085",
        "cvat_user": "admin",
        "cvat_pass": "",
        "task_name": None,
        "task_name_format": "Frigate Batch Review ({timestamp})",
        "category_mapping": dict(DEFAULT_CATEGORY_MAPPING),
        "no_upload": False,
        "upload_only": False,
        "accuracy_report": None,
        "accuracy_test_images": 60,
        "accuracy_coco_images": 0,
        "accuracy_metric": "f1_score",
        "server_url": config_utils.get_deployment_settings(cfg).get("server_url", "http://192.168.100.2:5000/v1/vision/detection"),
    }

    # Overlay YAML config values
    if cfg:
        frigate_cfg = cfg.get("frigate", {})
        if isinstance(frigate_cfg, dict):
            if "url" in frigate_cfg and frigate_cfg["url"] is not None:
                defaults["frigate_url"] = str(frigate_cfg["url"]).strip()
            if "camera" in frigate_cfg:
                defaults["camera"] = frigate_cfg["camera"]
            if "limit" in frigate_cfg and frigate_cfg["limit"] is not None:
                defaults["limit"] = int(frigate_cfg["limit"])
            if "balance_mode" in frigate_cfg and frigate_cfg["balance_mode"]:
                defaults["balance_mode"] = config_utils.normalize_balance_mode(str(frigate_cfg["balance_mode"]))
            else:
                defaults["balance_mode"] = "cvat-deficit"
            if "skip_processed" in frigate_cfg and frigate_cfg["skip_processed"] is not None:
                defaults["skip_processed"] = bool(frigate_cfg["skip_processed"])
            if "clean" in frigate_cfg and frigate_cfg["clean"] is not None:
                defaults["clean"] = bool(frigate_cfg["clean"])
            if "snapshot_endpoint" in frigate_cfg and frigate_cfg["snapshot_endpoint"]:
                defaults["snapshot_endpoint"] = str(frigate_cfg["snapshot_endpoint"]).strip()
            if "timeout" in frigate_cfg and frigate_cfg["timeout"] is not None:
                defaults["timeout"] = int(frigate_cfg["timeout"])
            if "accuracy_report" in frigate_cfg:
                defaults["accuracy_report"] = frigate_cfg["accuracy_report"]
            if "accuracy_test_images" in frigate_cfg and frigate_cfg["accuracy_test_images"] is not None:
                defaults["accuracy_test_images"] = int(frigate_cfg["accuracy_test_images"])
            if "accuracy_coco_images" in frigate_cfg and frigate_cfg["accuracy_coco_images"] is not None:
                defaults["accuracy_coco_images"] = int(frigate_cfg["accuracy_coco_images"])
            if "accuracy_metric" in frigate_cfg and frigate_cfg["accuracy_metric"]:
                defaults["accuracy_metric"] = str(frigate_cfg["accuracy_metric"]).strip()
            if "server_url" in frigate_cfg and frigate_cfg["server_url"]:
                defaults["server_url"] = str(frigate_cfg["server_url"]).strip()

        cvat_cfg = cfg.get("cvat", {})
        if isinstance(cvat_cfg, dict):
            if "url" in cvat_cfg and cvat_cfg["url"] is not None:
                defaults["cvat_url"] = str(cvat_cfg["url"]).strip()
            username = cvat_cfg.get("username")
            if username is not None:
                defaults["cvat_user"] = str(username).strip()
            password = cvat_cfg.get("password")
            if password is not None:
                defaults["cvat_pass"] = str(password).strip()
            if "task_name" in cvat_cfg:
                defaults["task_name"] = cvat_cfg["task_name"]
            if "task_name_format" in cvat_cfg and cvat_cfg["task_name_format"]:
                defaults["task_name_format"] = str(cvat_cfg["task_name_format"])

        storage_cfg = cfg.get("storage", {})
        if isinstance(storage_cfg, dict) and "output_dir" in storage_cfg:
            defaults["output_dir"] = str(storage_cfg["output_dir"]).strip()
        elif "output_dir" in cfg:
            defaults["output_dir"] = str(cfg["output_dir"]).strip()

        # Parse categories / labels
        if "categories" in cfg and isinstance(cfg["categories"], dict):
            defaults["category_mapping"] = {
                str(k).strip(): int(v) for k, v in cfg["categories"].items()
            }
        elif "labels" in cfg and isinstance(cfg["labels"], list):
            defaults["category_mapping"] = {
                str(lbl).strip(): idx + 1 for idx, lbl in enumerate(cfg["labels"])
            }

        modes_cfg = cfg.get("modes", {})
        if isinstance(modes_cfg, dict):
            if "no_upload" in modes_cfg and modes_cfg["no_upload"] is not None:
                defaults["no_upload"] = bool(modes_cfg["no_upload"])
            if "upload_only" in modes_cfg and modes_cfg["upload_only"] is not None:
                defaults["upload_only"] = bool(modes_cfg["upload_only"])

    # Overlay environment variables
    if os.getenv("FRIGATE_URL"):
        defaults["frigate_url"] = os.getenv("FRIGATE_URL").strip()
    if os.getenv("CVAT_URL"):
        defaults["cvat_url"] = os.getenv("CVAT_URL").strip()
    if os.getenv("CVAT_USER"):
        defaults["cvat_user"] = os.getenv("CVAT_USER").strip()
    if os.getenv("CVAT_PASS"):
        defaults["cvat_pass"] = os.getenv("CVAT_PASS").strip()

    return defaults


def parse_args():
    # Pass 1: Extract optional --config / -c argument without printing help
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", default=None, help="Path to YAML configuration file")
    pre_args, _ = pre_parser.parse_known_args()

    # Load defaults from configuration file, env vars, and fallback defaults
    defaults = load_config(pre_args.config)

    # Pass 2: Define full argument parser using loaded defaults
    parser = argparse.ArgumentParser(
        description="Harvest detection events from Frigate NVR and upload to CVAT for review.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "-c", "--config",
        default=defaults["config_file"],
        help=f"Path to YAML configuration file (default: {defaults['config_file']})",
    )

    # Frigate settings
    parser.add_argument(
        "--frigate-url",
        default=defaults["frigate_url"],
        help=f"Frigate NVR URL (default: {defaults['frigate_url']})",
    )
    parser.add_argument(
        "--camera",
        default=defaults["camera"],
        help=f"Filter by specific Frigate camera name (default: {defaults['camera'] or 'all cameras'})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=defaults["limit"],
        help=f"Maximum number of Frigate events to retrieve (default: {defaults['limit']})",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=defaults["clean"],
        help="Clean output images and annotations before downloading to ensure a fresh batch.",
    )
    parser.add_argument(
        "--snapshot-endpoint",
        default=defaults["snapshot_endpoint"],
        help=f"Frigate snapshot endpoint filename (default: {defaults['snapshot_endpoint']})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=defaults["timeout"],
        help=f"HTTP request timeout in seconds (default: {defaults['timeout']})",
    )

    # Local storage settings
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help=f"Local directory to store downloaded images and annotations.json (default: {defaults['output_dir']})",
    )

    # CVAT settings
    parser.add_argument(
        "--cvat-url",
        default=defaults["cvat_url"],
        help=f"CVAT server URL (default: {defaults['cvat_url']})",
    )
    parser.add_argument(
        "--cvat-user",
        default=defaults["cvat_user"],
        help=f"CVAT username (default: {defaults['cvat_user']})",
    )
    parser.add_argument(
        "--cvat-pass",
        default=defaults["cvat_pass"],
        help="CVAT password (default: set via config or $CVAT_PASS)",
    )
    parser.add_argument(
        "--task-name",
        default=defaults["task_name"],
        help="Name for the newly created CVAT task (default: formatted from config task_name_format)",
    )

    # Categories / Labels override
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Override detection labels to harvest (e.g. --labels person dog car or --labels cat)",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["cvat-deficit", "accuracy-deficit", "equal-split", "strict-equal", "natural"],
        default=defaults["balance_mode"],
        help="Class balancing strategy (default: cvat-deficit): "
             "'cvat-deficit' (adaptive deficit quotas based on CVAT annotations), "
             "'accuracy-deficit' (deficit quotas based on test_server_accuracy.py scores, ignores 0), "
             "'equal-split' (even quota per class), "
             "'strict-equal' (cap all classes to lowest class count), "
             "or 'natural' (raw unweighted Frigate events up to limit).",
    )
    parser.add_argument(
        "--accuracy-report",
        default=defaults["accuracy_report"],
        help="Optional path to JSON accuracy report (if provided and exists, skips running live test_server_accuracy benchmark).",
    )
    parser.add_argument(
        "--accuracy-test-images",
        type=int,
        default=defaults["accuracy_test_images"],
        help="Number of random test images to evaluate during test_server_accuracy benchmark (default: 60).",
    )
    parser.add_argument(
        "--accuracy-coco-images",
        type=int,
        default=defaults["accuracy_coco_images"],
        help="Number of random COCO dataset images to include during test_server_accuracy benchmark (default: 0).",
    )
    parser.add_argument(
        "--accuracy-metric",
        choices=["f1_score", "recall", "precision", "mean_iou"],
        default=defaults["accuracy_metric"],
        help="Accuracy metric used to compute deficit quotas (default: f1_score).",
    )
    parser.add_argument(
        "--server-url",
        default=defaults["server_url"],
        help=f"Vision server endpoint URL for accuracy benchmarking (default: {defaults['server_url']})",
    )
    parser.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_processed"],
        help="Skip images/events that were already processed in previous training runs (default: True).",
    )

    # Execution modes
    parser.add_argument(
        "--no-upload",
        dest="no_upload",
        action="store_true",
        default=defaults["no_upload"],
        help="Download Frigate events and generate COCO dataset locally without uploading to CVAT.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        default=defaults["upload_only"],
        help="Skip downloading from Frigate; upload existing local dataset directly to CVAT.",
    )

    args = parser.parse_args()

    # Automatically set accuracy-deficit mode when accuracy report is provided
    if args.accuracy_report:
        args.balance_mode = "accuracy-deficit"

    # Resolve active category mapping and target labels to harvest
    if args.labels:
        specified = [lbl.strip() for lbl in args.labels]
        # If specified labels are all within configured defaults, preserve project IDs and CVAT labels
        if all(lbl in defaults["category_mapping"] for lbl in specified):
            category_mapping = defaults["category_mapping"]
            target_labels = specified
        else:
            category_mapping = {lbl: idx + 1 for idx, lbl in enumerate(specified)}
            target_labels = specified
    else:
        category_mapping = defaults["category_mapping"]
        target_labels = list(category_mapping.keys())

    cvat_labels = [{"name": name} for name in category_mapping.keys()]

    # Resolve task name
    task_name = args.task_name
    if not task_name:
        fmt = defaults.get("task_name_format", "Frigate Batch Review ({timestamp})")
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if "{timestamp}" in fmt:
            task_name = fmt.replace("{timestamp}", timestamp_str)
        elif "%" in fmt:
            task_name = datetime.datetime.now().strftime(fmt)
        else:
            task_name = f"{fmt} ({timestamp_str})"

    # Resolve balance strategy
    args.balance_mode = config_utils.normalize_balance_mode(args.balance_mode)

    # Attach computed settings to args namespace
    args.category_mapping = category_mapping
    args.target_labels = target_labels
    args.skip_processed = args.skip_processed
    args.cvat_labels = cvat_labels
    args.task_name = task_name

    return args


def harvest_from_frigate(
    frigate_url,
    output_dir,
    camera=None,
    limit=200,
    clean=False,
    category_mapping=None,
    target_labels=None,
    balance_mode="cvat-deficit",
    skip_processed=True,
    snapshot_endpoint="snapshot-clean.webp",
    timeout=10,
    cvat_url=None,
    cvat_user=None,
    cvat_pass=None,
    accuracy_report=None,
    accuracy_test_images=60,
    accuracy_coco_images=0,
    accuracy_metric="f1_score",
    server_url=None,
):
    if category_mapping is None:
        category_mapping = DEFAULT_CATEGORY_MAPPING

    if target_labels is None:
        target_labels = list(category_mapping.keys())

    images_dir = os.path.join(output_dir, "images")
    annotations_path = os.path.join(output_dir, "annotations.json")

    if clean:
        if os.path.exists(images_dir):
            print(f"[INFO] Cleaning existing images in {images_dir}...")
            shutil.rmtree(images_dir)
        if os.path.exists(annotations_path):
            print(f"[INFO] Removing previous annotations at {annotations_path}...")
            os.remove(annotations_path)

    os.makedirs(images_dir, exist_ok=True)

    # 1. Load previously processed event IDs to avoid duplicate images from prior trainings
    processed_ids = set()
    if skip_processed:
        processed_ids = load_processed_events(output_dir=output_dir)
        if processed_ids:
            print(f"[INFO] Skipping {len(processed_ids)} previously processed events from prior trainings.")
        else:
            print("[INFO] No previously processed event IDs found; all Frigate events eligible.")

    print(f"[INFO] Connecting to Frigate at {frigate_url}...")
    events = []
    seen_event_ids = set()

    is_balanced = (balance_mode != "natural")
    if is_balanced:
        # Determine if accuracy-deficit or CVAT-balanced mode should be used
        accuracy_data = None
        if balance_mode == "accuracy-deficit":
            if accuracy_report and os.path.isfile(accuracy_report):
                accuracy_data = load_accuracy_report(accuracy_report)
                if accuracy_data:
                    print(f"[INFO] Loaded accuracy evaluation report from: {accuracy_report}")

            if accuracy_data is None:
                resolved_url = server_url or config_utils.get_deployment_settings().get("server_url", "http://192.168.100.2:5000/v1/vision/detection")
                coco_info_str = f" + {accuracy_coco_images} COCO" if accuracy_coco_images > 0 else ""
                print(f"[INFO] Accuracy-deficit mode active. Running benchmark against vision server: {resolved_url} ({accuracy_test_images} dataset{coco_info_str} images)...")
                try:
                    accuracy_data = test_server_accuracy.run_accuracy_benchmark(
                        server_url=resolved_url,
                        num_images=accuracy_test_images,
                        coco_images=accuracy_coco_images,
                        output_json=accuracy_report if (accuracy_report and not os.path.isfile(accuracy_report)) else None,
                    )
                except Exception as e:
                    print(f"[ERROR] Failed running accuracy benchmark against server: {e}")
                    accuracy_data = None

        cvat_counts = None
        if balance_mode == "cvat-deficit":
            if cvat_url and cvat_user and cvat_pass:
                print(f"[INFO] CVAT-balanced harvest active. Querying CVAT at {cvat_url} for class distribution...")
                cvat_counts = fetch_cvat_object_counts(
                    cvat_url=cvat_url,
                    cvat_user=cvat_user,
                    cvat_pass=cvat_pass,
                    timeout=timeout,
                )
            else:
                print("[WARN] CVAT-balanced harvest active but CVAT credentials missing. Falling back to equal distribution.")

        label_events = {}

        if accuracy_data is not None:
            tested_scores, disregarded = extract_accuracy_scores(accuracy_data, metric=accuracy_metric)
            if disregarded:
                print(f"[INFO] Disregarding untested/0-value classes: {', '.join(disregarded)}")

            sorted_labels, quotas, deficits, health_status = compute_accuracy_balanced_quotas(
                scores=tested_scores,
                target_labels=target_labels,
                limit=limit,
                metric=accuracy_metric,
            )

            if sorted_labels:
                print(f"[INFO] Accuracy-deficit harvest quotas (metric: {accuracy_metric}, total limit: {limit}):")
                for lbl in sorted_labels:
                    sc = tested_scores[lbl]
                    st = health_status.get(lbl, "GOOD")
                    print(f"  - [{lbl:<8}]: quota = {quotas[lbl]:>3}  (accuracy: {sc:>5.1f}%, health: {st})")

                # Harvest from Frigate prioritized by lowest score first
                surplus = 0
                for idx, lbl in enumerate(sorted_labels):
                    rem_labels = sorted_labels[idx:]
                    if surplus > 0 and len(rem_labels) > 0:
                        rem_def_sum = sum(deficits[rl] for rl in rem_labels)
                        if rem_def_sum > 0:
                            share = round(surplus * (deficits[lbl] / rem_def_sum))
                            quotas[lbl] += share
                            surplus -= share

                    req_count = quotas[lbl]
                    lbl_evs = fetch_events_from_frigate(
                        frigate_url=frigate_url,
                        target_labels=[lbl],
                        target_count=req_count,
                        camera=camera,
                        processed_ids=processed_ids.union(seen_event_ids),
                        timeout=timeout,
                    )
                    label_events[lbl] = lbl_evs
                    for ev in lbl_evs:
                        eid = ev.get("id")
                        if eid:
                            seen_event_ids.add(eid)

                    got = len(lbl_evs)
                    if got < req_count:
                        unused = req_count - got
                        surplus += unused
                        print(f"  - [{lbl}]: retrieved {got}/{req_count} new/unprocessed events ({unused} unused quota rolled over to remaining deficit classes).")
                    else:
                        print(f"  - [{lbl}]: retrieved {got}/{req_count} new/unprocessed events from Frigate.")
            else:
                print("[WARN] No valid tested classes with score > 0 found in accuracy report. Falling back to equal distribution.")

        elif cvat_counts is not None:
            sorted_labels, quotas, deficits, health_status = compute_cvat_balanced_quotas(
                cvat_counts=cvat_counts,
                target_labels=target_labels,
                limit=limit,
            )
            total_cvat_ann = sum(cvat_counts.get(l, 0) for l in target_labels)
            print(f"[INFO] Total CVAT annotations analyzed: {total_cvat_ann:,} across {len(target_labels)} target classes.")
            print(f"[INFO] Deficit-weighted harvest quotas (total limit: {limit}):")
            for lbl in sorted_labels:
                cnt = cvat_counts.get(lbl, 0)
                status = health_status.get(lbl, "GOOD")
                print(f"  - [{lbl:<8}]: quota = {quotas[lbl]:>3}  (CVAT existing: {cnt:>4}, health: {status})")

            # Harvest from Frigate prioritized by lowest CVAT count first
            surplus = 0
            for idx, lbl in enumerate(sorted_labels):
                rem_labels = sorted_labels[idx:]
                # Dynamically distribute accumulated surplus among remaining labels based on deficit
                if surplus > 0 and len(rem_labels) > 0:
                    rem_def_sum = sum(deficits[rl] for rl in rem_labels)
                    if rem_def_sum > 0:
                        share = round(surplus * (deficits[lbl] / rem_def_sum))
                        quotas[lbl] += share
                        surplus -= share

                req_count = quotas[lbl]
                lbl_evs = fetch_events_from_frigate(
                    frigate_url=frigate_url,
                    target_labels=[lbl],
                    target_count=req_count,
                    camera=camera,
                    processed_ids=processed_ids.union(seen_event_ids),
                    timeout=timeout,
                )
                label_events[lbl] = lbl_evs
                for ev in lbl_evs:
                    eid = ev.get("id")
                    if eid:
                        seen_event_ids.add(eid)

                got = len(lbl_evs)
                if got < req_count:
                    unused = req_count - got
                    surplus += unused
                    print(f"  - [{lbl}]: retrieved {got}/{req_count} new/unprocessed events ({unused} unused quota rolled over to remaining deficit classes).")
                else:
                    print(f"  - [{lbl}]: retrieved {got}/{req_count} new/unprocessed events from Frigate.")

        else:
            quota = max(1, limit // max(1, len(target_labels)))
            print(f"[INFO] Balanced harvest active ({balance_mode}): requesting up to {quota} events per label across {len(target_labels)} classes: {target_labels}")

            for lbl in target_labels:
                lbl_evs = fetch_events_from_frigate(
                    frigate_url=frigate_url,
                    target_labels=[lbl],
                    target_count=quota,
                    camera=camera,
                    processed_ids=processed_ids.union(seen_event_ids),
                    timeout=timeout,
                )
                label_events[lbl] = lbl_evs
                for ev in lbl_evs:
                    eid = ev.get("id")
                    if eid:
                        seen_event_ids.add(eid)
                print(f"  - [{lbl}]: retrieved {len(lbl_evs)} new/unprocessed events from Frigate.")

        active_labels = [lbl for lbl, evs in label_events.items() if len(evs) > 0]
        empty_labels = [lbl for lbl, evs in label_events.items() if len(evs) == 0]

        if empty_labels:
            print(f"[WARN] No unprocessed events found for label(s): {empty_labels}")

        if not active_labels:
            print("[WARN] No unprocessed events found for any target category. Aborting harvest.")
            return 0

        # Strict equal mode clamps classes to lowest count
        if balance_mode == "strict-equal" and len(active_labels) > 1:
            min_count = min(len(label_events[lbl]) for lbl in active_labels)
            print(f"[INFO] Strict balance: Limiting all active classes to {min_count} images (lowest available count) to achieve equal distribution.")
            for lbl in active_labels:
                if len(label_events[lbl]) > min_count:
                    label_events[lbl] = label_events[lbl][:min_count]

        for lbl in label_events:
            for ev in label_events[lbl]:
                events.append(ev)
    else:
        events = fetch_events_from_frigate(
            frigate_url=frigate_url,
            target_labels=target_labels,
            target_count=limit,
            camera=camera,
            processed_ids=processed_ids,
            timeout=timeout,
        )

    print(f"[INFO] Total candidate events to process: {len(events)}.")
    if not events:
        print("[WARN] No new events found matching criteria.")
        return 0

    print(f"[INFO] Filtering for target categories: {list(category_mapping.keys())}")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for name, cid in category_mapping.items()
        ],
    }

    annotation_id = 1
    image_id = 1
    endpoint_name = snapshot_endpoint.lstrip("/")

    for event in events:
        event_id = event.get("id")
        label = event.get("label")
        box = event.get("data", {}).get("box")

        if label not in category_mapping or not box:
            continue

        file_name = f"{event_id}.jpg"
        img_path = os.path.join(images_dir, file_name)

        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        else:
            snapshot_url = f"{frigate_url.rstrip('/')}/api/events/{event_id}/{endpoint_name}"
            try:
                img_resp = requests.get(snapshot_url, timeout=timeout)
                if img_resp.status_code != 200:
                    continue
                img = Image.open(io.BytesIO(img_resp.content))
                img_w, img_h = img.size
                img.convert("RGB").save(img_path, "JPEG")
            except Exception as e:
                print(f"[WARN] Failed to download snapshot for event {event_id}: {e}")
                continue

        # Frigate data['box'] format: [x_norm, y_norm, width_norm, height_norm]
        x_norm, y_norm, w_norm, h_norm = box
        if all(0.0 <= v <= 1.0 for v in box):
            x = x_norm * img_w
            y = y_norm * img_h
            width = w_norm * img_w
            height = h_norm * img_h
        else:
            x, y, width, height = x_norm, y_norm, w_norm, h_norm

        x = max(0.0, min(float(x), float(img_w)))
        y = max(0.0, min(float(y), float(img_h)))
        width = max(0.0, min(float(width), float(img_w - x)))
        height = max(0.0, min(float(height), float(img_h - y)))
        area = width * height

        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
            "_label": label,
        })

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_mapping[label],
            "bbox": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
            "area": round(area, 2),
            "iscrowd": 0,
        })

        annotation_id += 1
        image_id += 1

    # Post-download strict balance enforcement across successfully downloaded images
    if balance_mode == "strict-equal":
        images_by_label = {lbl: [] for lbl in category_mapping.keys()}
        for img in coco_data["images"]:
            lbl = img.get("_label")
            if lbl in images_by_label:
                images_by_label[lbl].append(img)

        active_downloaded = [lbl for lbl, imgs in images_by_label.items() if len(imgs) > 0]
        if len(active_downloaded) > 1:
            actual_min = min(len(images_by_label[lbl]) for lbl in active_downloaded)
            print(f"[INFO] Strict balance verification: ensuring exactly {actual_min} images per active class.")
            keep_img_ids = set()
            for lbl in active_downloaded:
                for img in images_by_label[lbl][:actual_min]:
                    keep_img_ids.add(img["id"])
                # Clean trimmed image files from disk
                for img in images_by_label[lbl][actual_min:]:
                    trimmed_file = os.path.join(images_dir, img["file_name"])
                    if os.path.exists(trimmed_file):
                        try:
                            os.remove(trimmed_file)
                        except Exception:
                            pass

            coco_data["images"] = [img for img in coco_data["images"] if img["id"] in keep_img_ids]
            coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] in keep_img_ids]

    # Clean temporary _label property from images
    for img in coco_data["images"]:
        img.pop("_label", None)

    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    # 2. Record newly harvested event IDs so they are not repeated in future runs
    if skip_processed and coco_data["images"]:
        newly_harvested_ids = [Path(img["file_name"]).stem for img in coco_data["images"]]
        save_processed_events(newly_harvested_ids)

    category_counts = {}
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        cname = next((name for name, id_ in category_mapping.items() if id_ == cid), str(cid))
        category_counts[cname] = category_counts.get(cname, 0) + 1

    print(f"[SUCCESS] Harvested {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")
    if category_counts:
        print("[INFO] Harvested category breakdown:")
        for cat_name, cnt in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat_name}: {cnt}")
    print(f"[INFO] Saved dataset to {output_dir}")
    return len(coco_data["images"])


def upload_to_cvat(cvat_url, cvat_user, cvat_pass, output_dir, task_name=None, cvat_labels=None):
    from cvat_sdk import make_client

    if cvat_labels is None:
        cvat_labels = DEFAULT_CVAT_LABELS

    images_dir = os.path.join(output_dir, "images")
    annotations_path = os.path.join(output_dir, "annotations.json")

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    if not os.path.exists(annotations_path):
        print(f"[ERROR] Annotations file not found: {annotations_path}")
        sys.exit(1)

    # Load annotations to ensure only images associated with this batch are uploaded
    with open(annotations_path, "r", encoding="utf-8") as f:
        coco_meta = json.load(f)

    annotated_filenames = [img["file_name"] for img in coco_meta.get("images", [])]
    if annotated_filenames:
        image_files = [
            os.path.join(images_dir, fn)
            for fn in annotated_filenames
            if os.path.isfile(os.path.join(images_dir, fn))
        ]
    else:
        image_files = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

    if not image_files:
        print(f"[ERROR] No valid images found in {images_dir} to upload.")
        sys.exit(1)

    if not task_name:
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        task_name = f"Frigate Batch Review ({timestamp_str})"

    if not cvat_user or not cvat_pass:
        print("[ERROR] CVAT username and password are required.")
        print("[ERROR] Please set them in 'config.yaml', via --cvat-user / --cvat-pass, or $CVAT_USER / $CVAT_PASS.")
        sys.exit(1)

    print(f"\n[INFO] Connecting to CVAT at {cvat_url} as '{cvat_user}'...")
    try:
        with make_client(cvat_url, credentials=(cvat_user, cvat_pass)) as client:
            print(f"[INFO] Creating CVAT task: '{task_name}' with {len(image_files)} images...")
            task = client.tasks.create_from_data(
                spec={
                    "name": task_name,
                    "labels": cvat_labels,
                },
                resources=image_files,
            )

            print(f"[INFO] Uploading COCO 1.0 pre-annotations from {annotations_path}...")
            task.import_annotations(
                format_name="COCO 1.0",
                filename=annotations_path,
            )

            cvat_base = cvat_url.rstrip("/")
            task_url = f"{cvat_base}/tasks/{task.id}"

            # Save state to .last_cvat_task for automated pipeline pickup
            state_data = {
                "task_id": task.id,
                "task_name": task.name,
                "cvat_url": cvat_url,
                "task_url": task_url,
                "created_at": datetime.datetime.now().isoformat(),
            }
            state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_cvat_task")
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)
            except Exception:
                pass

            print("\n" + "=" * 70)
            print(" [SUCCESS] CVAT Task Successfully Created!")
            print(f" Task ID  : {task.id}")
            print(f" Task Name: {task.name}")
            print(f" URL      : {task_url}")
            print("=" * 70)
            print("\nNext Steps:")
            print("  1. Open the URL above to review and adjust annotations in CVAT.")
            print("  2. When done, build & compile directly with one command:")
            print("     ./train.sh --export\n")
            return task.id
    except Exception as e:
        print(f"[ERROR] Failed to create or upload task in CVAT: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    if args.no_upload and args.upload_only:
        print("[ERROR] Cannot specify both --no-upload and --upload-only.")
        sys.exit(1)

    # 1. Harvest from Frigate (unless --upload-only)
    if not args.upload_only:
        image_count = harvest_from_frigate(
            frigate_url=args.frigate_url,
            output_dir=args.output_dir,
            camera=args.camera,
            limit=args.limit,
            clean=args.clean,
            category_mapping=args.category_mapping,
            target_labels=args.target_labels,
            balance_mode=args.balance_mode,
            skip_processed=args.skip_processed,
            snapshot_endpoint=args.snapshot_endpoint,
            timeout=args.timeout,
            cvat_url=args.cvat_url,
            cvat_user=args.cvat_user,
            cvat_pass=args.cvat_pass,
            accuracy_report=args.accuracy_report,
            accuracy_test_images=args.accuracy_test_images,
            accuracy_coco_images=args.accuracy_coco_images,
            accuracy_metric=args.accuracy_metric,
            server_url=args.server_url,
        )
        if image_count == 0:
            print("[WARN] No matching events with images found. Aborting CVAT upload.")
            return

    # 2. Upload to CVAT (unless --no-upload)
    if not args.no_upload:
        upload_to_cvat(
            cvat_url=args.cvat_url,
            cvat_user=args.cvat_user,
            cvat_pass=args.cvat_pass,
            output_dir=args.output_dir,
            task_name=args.task_name,
            cvat_labels=args.cvat_labels,
        )


if __name__ == "__main__":
    main()
