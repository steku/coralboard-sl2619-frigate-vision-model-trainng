#!/usr/bin/env python3
"""
Shared configuration, CVAT client, and model metadata utilities for Frigate training.
"""

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import yaml

DEFAULT_CONFIG_FILES = [
    "config.yaml",
    "config.yml",
]


def find_config_file(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locate configuration YAML file from explicit path or default candidates."""
    if custom_path:
        path = Path(custom_path).resolve()
        if path.is_file():
            return path
        return None

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    search_dirs = [cwd]
    if script_dir != cwd:
        search_dirs.append(script_dir)

    for directory in search_dirs:
        for candidate in DEFAULT_CONFIG_FILES:
            target = directory / candidate
            if target.is_file():
                return target

    return None


def load_config(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and return parsed dictionary.
    Returns empty dict if no config file is found.
    """
    cfg_file = find_config_file(custom_path)
    if not cfg_file:
        return {}

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Failed to load config from {cfg_file}: {e}")
        return {}


def get_cvat_credentials(
    cfg: Optional[Dict[str, Any]] = None,
    url: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Extract and resolve CVAT (url, username, password) from CLI args, env vars, or config.
    """
    if cfg is None:
        cfg = load_config()

    cvat_cfg = cfg.get("cvat", {}) if isinstance(cfg.get("cvat"), dict) else {}

    resolved_url = (
        url
        or os.getenv("CVAT_URL")
        or cvat_cfg.get("url")
        or "http://192.168.2.1:8085"
    )
    resolved_user = (
        user
        or os.getenv("CVAT_USER")
        or cvat_cfg.get("username")
        or "admin"
    )
    resolved_pass = (
        password
        or os.getenv("CVAT_PASS")
        or cvat_cfg.get("password")
        or ""
    )

    return resolved_url.rstrip("/"), resolved_user, resolved_pass


def get_cvat_client(
    cfg: Optional[Dict[str, Any]] = None,
    url: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Create and return an authenticated CVAT SDK client.
    """
    from cvat_sdk import make_client

    resolved_url, resolved_user, resolved_pass = get_cvat_credentials(cfg, url, user, password)
    if not resolved_user or not resolved_pass:
        raise ValueError("CVAT username and password are required. Set in config.yaml or CVAT_USER/CVAT_PASS.")

    return make_client(resolved_url, credentials=(resolved_user, resolved_pass))


def load_dataset_classes(data_yaml_path: str = "datasets/data.yaml") -> Dict[int, str]:
    """
    Load index-to-class-name mapping from a YOLO data.yaml or models/labels/*.json.
    """
    p = Path(data_yaml_path)
    if not p.is_file() and data_yaml_path in ("datasets/data.yaml", "datasets/yolo/data.yaml"):
        for candidate in ("datasets/data.yaml", "datasets/yolo/data.yaml", "yolo_dataset/data.yaml"):
            if Path(candidate).is_file():
                p = Path(candidate)
                break

    if not p.is_file():
        # Check fallback to models/labels/*.json
        labels_dir = Path("models/labels")
        if labels_dir.is_dir():
            for f in labels_dir.glob("labels_*.json"):
                try:
                    import json
                    with open(f, "r", encoding="utf-8") as jf:
                        d = json.load(jf)
                        if "names" in d:
                            return {int(k): str(v) for k, v in d["names"].items()}
                except Exception:
                    pass
        raise FileNotFoundError(f"Dataset config not found at {data_yaml_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def get_deployment_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract deployment configuration (host, dir, service, inferred coral server url).
    """
    if cfg is None:
        cfg = load_config()

    dep = cfg.get("deployment", {}) if isinstance(cfg.get("deployment"), dict) else {}
    host = os.getenv("DEPLOY_HOST") or dep.get("host") or ""
    deploy_dir = os.getenv("DEPLOY_DIR") or dep.get("dir") or ""
    restart = bool(dep.get("restart_service", False))
    service = dep.get("service_name") or "coralboard-sl2619-vision"

    # Infer server URL from host if available (e.g. root@192.168.100.2:/path -> http://192.168.100.2:5000/v1/vision/detection)
    server_url = os.getenv("CORAL_SERVER_URL")
    if not server_url and host:
        # Extract IP or hostname between '@' and ':'
        match = re.search(r"@([^:/]+)", host)
        if match:
            ip = match.group(1)
            server_url = f"http://{ip}:5000/v1/vision/detection"

    if not server_url:
        server_url = "http://192.168.100.2:5000/v1/vision/detection"

    return {
        "host": host,
        "dir": deploy_dir,
        "restart_service": restart,
        "service_name": service,
        "server_url": server_url,
    }


def get_remote_training_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract remote training settings (host, remote_dir, device, batch).
    """
    if cfg is None:
        cfg = load_config()

    rem = cfg.get("remote_training", {}) if isinstance(cfg.get("remote_training"), dict) else {}
    host = os.getenv("SPARK_HOST") or os.getenv("DGX_HOST") or rem.get("host") or ""
    remote_dir = rem.get("remote_dir") or "~/frigate-model-training"
    device = str(rem.get("device", "0"))
    try:
        batch = int(rem.get("batch", 32))
    except (TypeError, ValueError):
        batch = 32
    try:
        workers = int(rem.get("workers", 8))
    except (TypeError, ValueError):
        workers = 8
    cache = rem.get("cache", "ram")
    if cache is True:
        cache = "ram"
    elif cache is False or cache is None:
        cache = "false"
    clean_images = bool(rem.get("clean_images", False))

    return {
        "host": host,
        "remote_dir": remote_dir,
        "device": device,
        "batch": batch,
        "workers": workers,
        "cache": str(cache),
        "clean_images": clean_images,
    }


def normalize_balance_mode(mode_str: Optional[str]) -> str:
    """
    Validate and return canonical balance mode:
      - 'cvat-deficit'     : Adaptive deficit-weighted quotas based on CVAT label annotations (default)
      - 'accuracy-deficit' : Deficit-weighted quotas based on test_server_accuracy.py scores
      - 'equal-split'      : Flat equal division across all classes
      - 'strict-equal'     : Clamps all classes to the lowest available class count
      - 'natural'          : No balancing / natural order up to limit
    """
    if not mode_str:
        return "cvat-deficit"

    clean = mode_str.lower().strip().replace("_", "-")
    if clean in ("accuracy", "accuracy-deficit"):
        return "accuracy-deficit"
    valid_modes = {"cvat-deficit", "accuracy-deficit", "equal-split", "strict-equal", "natural"}
    if clean in valid_modes:
        return clean
    raise ValueError(f"Invalid balance_mode '{mode_str}'. Valid modes: {', '.join(sorted(valid_modes))}")


# ==============================================================================
# COCO-80 Canonical Classes & Dataset Downloader
# ==============================================================================
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COCO_80_NAME_TO_ID = {name: idx for idx, name in enumerate(COCO_80_CLASSES)}


def get_coco_80_dict() -> Dict[int, str]:
    """Return dictionary of canonical COCO-80 index to class name."""
    return {idx: name for idx, name in enumerate(COCO_80_CLASSES)}


def get_coco_id(label_name: str) -> Optional[int]:
    """
    Lookup official COCO-80 integer class ID for a label name.
    Performs case-insensitive matching and handles hyphens/underscores/spaces.
    """
    if not label_name:
        return None
    normalized = str(label_name).strip().lower().replace("_", " ").replace("-", " ")
    if normalized in COCO_80_NAME_TO_ID:
        return COCO_80_NAME_TO_ID[normalized]
    # Check with hyphens (e.g., 'traffic-light', 'hair-drier')
    normalized_dash = normalized.replace(" ", "-")
    for name, idx in COCO_80_NAME_TO_ID.items():
        if name.replace(" ", "-") == normalized_dash:
            return idx
    return None


def get_coco_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract COCO settings from configuration file.
    Options:
      - dataset: 'coco128' (default) or 'coco-val2017'
    """
    if cfg is None:
        cfg = load_config()
    coco_cfg = cfg.get("coco", {}) if isinstance(cfg.get("coco"), dict) else {}
    ds = str(coco_cfg.get("dataset", "coco128")).strip().lower()
    if ds in ("val2017", "coco-val2017"):
        ds = "coco-val2017"
    else:
        ds = "coco128"

    return {
        "dataset": ds,
    }


def get_frigate_plus_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract Frigate+ dataset settings from configuration file.
    Options:
      - enabled: bool (defaults to True if directory exists, else False)
      - path: str (default 'datasets/frigate-plus')
    """
    if cfg is None:
        cfg = load_config()
    fp_cfg = cfg.get("frigate_plus", {}) if isinstance(cfg.get("frigate_plus"), dict) else {}
    enabled_val = fp_cfg.get("enabled", None)
    fp_path = str(fp_cfg.get("path", "datasets/frigate-plus")).strip()
    if enabled_val is None:
        enabled = Path(fp_path).is_dir()
    else:
        enabled = bool(enabled_val)
    return {
        "enabled": enabled,
        "path": fp_path,
    }


def download_coco_dataset(dataset_name: str, target_dir: str = "datasets") -> Dict[str, str]:
    """
    Download and prepare a COCO dataset subset for co-training.
    Supported options:
      - 'coco128': Ultra-lightweight anchor (128 images, ~7 MB, 80 classes)
      - 'coco-val2017': Robust validation split (5,000 images, ~850 MB, 80 classes)

    Returns dictionary with 'train_dir', 'val_dir', and 'name'.
    """
    from ultralytics.utils.downloads import download

    out_base = Path(target_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    ds_clean = dataset_name.lower().strip()
    if ds_clean == "coco128":
        ds_dir = out_base / "coco128"
        train_img_dir = ds_dir / "images" / "train2017"
        zip_candidate = out_base / "coco128.zip"
        if train_img_dir.is_dir() and any(train_img_dir.iterdir()):
            print(f"[INFO] COCO128 dataset already present at {ds_dir}")
        elif zip_candidate.is_file():
            print(f"[INFO] Extracting existing {zip_candidate}...")
            import zipfile
            with zipfile.ZipFile(zip_candidate, "r") as zf:
                zf.extractall(out_base)
        else:
            print(f"[INFO] Downloading COCO128 dataset (~7 MB) into {out_base}...")
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
            download(url, dir=out_base)

        return {
            "name": "coco128",
            "train_dir": str(train_img_dir),
            "val_dir": str(train_img_dir),
        }

    elif ds_clean in ("coco-val2017", "val2017"):
        ds_dir = out_base / "coco"
        val_img_dir = ds_dir / "images" / "val2017"
        val_lbl_dir = ds_dir / "labels" / "val2017"

        # 1. Download/extract labels if not present
        lbl_zip_candidate = out_base / "coco2017labels.zip"
        if val_lbl_dir.is_dir() and any(val_lbl_dir.iterdir()):
            print(f"[INFO] COCO2017 labels already present at {val_lbl_dir}")
        elif lbl_zip_candidate.is_file():
            print(f"[INFO] Extracting existing {lbl_zip_candidate}...")
            import zipfile
            with zipfile.ZipFile(lbl_zip_candidate, "r") as zf:
                zf.extractall(out_base)
        else:
            print(f"[INFO] Downloading COCO2017 YOLO labels (~46 MB) into {out_base}...")
            lbl_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
            download(lbl_url, dir=out_base)

        # 2. Download/extract val images if not present
        val_zip_candidate = ds_dir / "images" / "val2017.zip"
        if not val_zip_candidate.is_file():
            val_zip_candidate = out_base / "val2017.zip"

        if val_img_dir.is_dir() and any(val_img_dir.iterdir()):
            print(f"[INFO] COCO val2017 images already present at {val_img_dir}")
        elif val_zip_candidate.is_file():
            print(f"[INFO] Extracting existing {val_zip_candidate}...")
            import zipfile
            with zipfile.ZipFile(val_zip_candidate, "r") as zf:
                zf.extractall(ds_dir / "images")
        else:
            print(f"[INFO] Downloading COCO val2017 images (~800 MB, 5,000 images) into {ds_dir / 'images'}...")
            img_url = "http://images.cocodataset.org/zips/val2017.zip"
            download(img_url, dir=ds_dir / "images")

        return {
            "name": "coco-val2017",
            "train_dir": str(val_img_dir),
            "val_dir": str(val_img_dir),
        }

    else:
        raise ValueError(f"Unsupported COCO dataset: '{dataset_name}'. Choose 'coco128' or 'coco-val2017'.")


def prepare_training_data_yaml(
    coco_dataset: Optional[str] = "coco128",
    workspace_dir: str = ".",
    include_frigate_plus: Optional[bool] = None,
    frigate_plus_dir: Optional[str] = None,
) -> str:
    """
    Prepare external datasets and generate canonical 80-class data.yaml referencing
    local CVAT images, optional Frigate+ images, and external COCO images with absolute paths.
    Target output: datasets/data.yaml (and datasets/yolo/data.yaml for backward compatibility).
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    target_datasets = workspace / "datasets"
    yolo_dir = target_datasets / "yolo" if (target_datasets / "yolo").is_dir() or not (workspace / "yolo_dataset").is_dir() else (workspace / "yolo_dataset")
    out_yaml = target_datasets / "data.yaml"
    legacy_out_yaml = yolo_dir / "data.yaml"

    cfg = load_config()
    fp_cfg = get_frigate_plus_config(cfg)
    if include_frigate_plus is None:
        include_frigate_plus = fp_cfg["enabled"]
    if frigate_plus_dir is None:
        frigate_plus_dir = fp_cfg["path"]

    fp_path = (workspace / frigate_plus_dir).resolve() if not Path(frigate_plus_dir).is_absolute() else Path(frigate_plus_dir)

    names = get_coco_80_dict()
    for cand in (out_yaml, legacy_out_yaml):
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
                existing_names = existing.get("names", {})
                if isinstance(existing_names, dict):
                    for k, v in existing_names.items():
                        if int(k) >= 80:
                            names[int(k)] = str(v)
                break
            except Exception:
                pass

    train_paths = []
    val_paths = []

    # 1. Local CVAT dataset splits
    yolo_train = yolo_dir / "images" / "train"
    yolo_val = yolo_dir / "images" / "val"
    if yolo_train.is_dir():
        train_paths.append(str(yolo_train.resolve()))
    if yolo_val.is_dir():
        val_paths.append(str(yolo_val.resolve()))

    # 2. Frigate+ dataset splits
    if include_frigate_plus and fp_path.is_dir():
        fp_train = fp_path / "images" / "train"
        fp_val = fp_path / "images" / "val"
        if fp_train.is_dir():
            print(f"[DATASET] Including Frigate+ Train Directory: {fp_train.resolve()}")
            train_paths.append(str(fp_train.resolve()))
        if fp_val.is_dir():
            print(f"[DATASET] Including Frigate+ Val Directory  : {fp_val.resolve()}")
            val_paths.append(str(fp_val.resolve()))

    # 3. External COCO dataset splits
    if coco_dataset and str(coco_dataset).lower() != "none":
        print(f"[DATASET] Checking external COCO dataset: {coco_dataset}...")
        coco_info = download_coco_dataset(coco_dataset, target_dir=str(target_datasets))
        print(f"[DATASET] COCO Train Directory: {coco_info['train_dir']}")
        print(f"[DATASET] COCO Val Directory  : {coco_info['val_dir']}")
        train_paths.append(str(Path(coco_info["train_dir"]).resolve()))
        val_paths.append(str(Path(coco_info["val_dir"]).resolve()))

    data_dict = {
        "train": train_paths,
        "val": val_paths,
        "names": names,
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

    if yolo_dir.is_dir() and yolo_dir != target_datasets:
        try:
            with open(legacy_out_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(data_dict, f, sort_keys=False)
        except Exception:
            pass

    print(f"[DATASET] Unified data.yaml written to {out_yaml}")
    return str(out_yaml)


# ==============================================================================
# Training Execution Profile Resolution
# ==============================================================================
def get_profile(
    profile_name: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve execution profile from config.yaml.
    Supports named profiles (e.g. 'torq', 'spark-torq', 'nvidia', 'spark-nvidia', 'cpu-torq').
    """
    if cfg is None:
        cfg = load_config(config_path)

    default_profile = str(cfg.get("default_profile") or "torq").strip().lower()
    target_name = (profile_name or default_profile).strip().lower()

    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles"), dict) else {}
    p_raw = (
        profiles.get(target_name)
        or profiles.get(target_name.replace("_", "-"))
        or profiles.get(target_name.replace("-", "_"))
    )

    if not isinstance(p_raw, dict):
        if profile_name and profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(
                f"Unknown profile '{profile_name}' in config.yaml. Available profiles: {available}"
            )
        p_data: Dict[str, Any] = {}
    else:
        p_data = p_raw

    is_remote = bool(p_data.get("remote", False))
    host = str(p_data.get("host") or "")
    remote_dir = str(p_data.get("remote_dir") or "~/frigate-model-training")
    device = str(p_data.get("device") if p_data.get("device") is not None else "0")

    try:
        default_batch = 32 if is_remote else 8
        batch = int(p_data.get("batch", default_batch))
    except (TypeError, ValueError):
        batch = default_batch

    try:
        workers = int(p_data.get("workers", 8))
    except (TypeError, ValueError):
        workers = 8

    cache = p_data.get("cache", "ram")
    if cache is True:
        cache = "ram"
    elif cache is False or cache is None:
        cache = "false"

    try:
        epochs = int(p_data.get("epochs", 80))
    except (TypeError, ValueError):
        epochs = 80

    try:
        patience = int(p_data.get("patience", 50))
    except (TypeError, ValueError):
        patience = 50

    raw_target = str(
        p_data.get("target")
        or p_data.get("output_format")
        or ("nvidia" if "nvidia" in target_name or target_name == "onnx" else "torq")
    ).strip().lower()
    if raw_target in ("nvidia", "onnx", "gpu", "tensorrt"):
        target = "nvidia"
    else:
        target = "torq"

    default_base_model = "yolov9s.pt" if target == "nvidia" else "yolov8n.pt"
    base_model = str(p_data.get("base_model") or default_base_model)
    clean_images = bool(p_data.get("clean_images", False))

    coco_cfg = cfg.get("coco", {}) or {}
    coco_dataset = str(p_data.get("coco_dataset") or coco_cfg.get("dataset") or "coco128")

    optimizer = str(p_data.get("optimizer") or "AdamW")
    try:
        lr0 = float(p_data.get("lr0", 0.0005))
    except (TypeError, ValueError):
        lr0 = 0.0005

    try:
        lrf = float(p_data.get("lrf", 0.01))
    except (TypeError, ValueError):
        lrf = 0.01

    cos_lr_raw = p_data.get("cos_lr", True)
    if isinstance(cos_lr_raw, str):
        cos_lr = cos_lr_raw.strip().lower() not in ("false", "0", "no")
    else:
        cos_lr = bool(cos_lr_raw)

    try:
        warmup_epochs = float(p_data.get("warmup_epochs", 5.0))
    except (TypeError, ValueError):
        warmup_epochs = 5.0

    try:
        warmup_momentum = float(p_data.get("warmup_momentum", 0.5))
    except (TypeError, ValueError):
        warmup_momentum = 0.5

    try:
        warmup_bias_lr = float(p_data.get("warmup_bias_lr", 0.0))
    except (TypeError, ValueError):
        warmup_bias_lr = 0.0

    dep_cfg = cfg.get("deployment", {}) or {}
    if target == "nvidia":
        deploy_host = str(p_data.get("deploy_host") or "")
        deploy_dir = str(p_data.get("deploy_dir") or "")
    else:
        deploy_host = str(p_data.get("deploy_host") or dep_cfg.get("host") or "")
        deploy_dir = str(p_data.get("deploy_dir") or dep_cfg.get("dir") or "")

    fp_cfg = get_frigate_plus_config(cfg)
    frigate_plus_enabled = p_data.get("frigate_plus", fp_cfg["enabled"])
    frigate_plus_dir = str(p_data.get("frigate_plus_dir", fp_cfg["path"]))

    return {
        "name": target_name,
        "target": target,
        "remote": is_remote,
        "host": host,
        "remote_dir": remote_dir,
        "device": device,
        "batch": batch,
        "workers": workers,
        "cache": str(cache),
        "epochs": epochs,
        "patience": patience,
        "base_model": base_model,
        "coco_dataset": coco_dataset,
        "clean_images": clean_images,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "deploy_host": deploy_host,
        "deploy_dir": deploy_dir,
        "frigate_plus": bool(frigate_plus_enabled),
        "frigate_plus_dir": frigate_plus_dir,
    }


def dump_profile_env(
    profile_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    """Generate shell assignment statements for train.sh eval."""
    p = get_profile(profile_name, config_path=config_path)
    lines = [
        f"PROFILE_NAME='{p['name']}'",
        f"PROFILE_TARGET='{p['target']}'",
        f"PROFILE_REMOTE='{'true' if p['remote'] else 'false'}'",
        f"PROFILE_HOST='{p['host']}'",
        f"PROFILE_REMOTE_DIR='{p['remote_dir']}'",
        f"PROFILE_DEVICE='{p['device']}'",
        f"PROFILE_BATCH='{p['batch']}'",
        f"PROFILE_WORKERS='{p['workers']}'",
        f"PROFILE_CACHE='{p['cache']}'",
        f"PROFILE_EPOCHS='{p['epochs']}'",
        f"PROFILE_PATIENCE='{p['patience']}'",
        f"PROFILE_BASE_MODEL='{p['base_model']}'",
        f"PROFILE_COCO_DATASET='{p['coco_dataset']}'",
        f"PROFILE_CLEAN_IMAGES='{'true' if p['clean_images'] else 'false'}'",
        f"PROFILE_OPTIMIZER='{p['optimizer']}'",
        f"PROFILE_LR0='{p['lr0']}'",
        f"PROFILE_LRF='{p['lrf']}'",
        f"PROFILE_COS_LR='{'True' if p['cos_lr'] else 'False'}'",
        f"PROFILE_WARMUP_EPOCHS='{p['warmup_epochs']}'",
        f"PROFILE_WARMUP_MOMENTUM='{p['warmup_momentum']}'",
        f"PROFILE_WARMUP_BIAS_LR='{p['warmup_bias_lr']}'",
        f"PROFILE_DEPLOY_HOST='{p['deploy_host']}'",
        f"PROFILE_DEPLOY_DIR='{p['deploy_dir']}'",
        f"PROFILE_FRIGATE_PLUS='{'true' if p['frigate_plus'] else 'false'}'",
        f"PROFILE_FRIGATE_PLUS_DIR='{p['frigate_plus_dir']}'",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Configuration profile CLI utility")
    parser.add_argument("--profile-env", nargs="?", const="", default=None, help="Output shell env assignments for profile")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if args.profile_env is not None:
        import sys
        prof_arg = args.profile_env if args.profile_env else None
        try:
            print(dump_profile_env(prof_arg, args.config))
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)


