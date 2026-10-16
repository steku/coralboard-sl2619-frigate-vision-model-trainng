#!/usr/bin/env python3
"""
Export CVAT annotations and images directly into YOLO dataset format.

Features:
- Reads CVAT credentials and fallback classes from config.yaml (or CLI/env vars).
- Auto-detects task ID from .last_cvat_task or CVAT API ('latest').
- Supports single or multiple task IDs (e.g., --task-id 3 4).
- Appends new tasks to existing dataset by default without overwriting (use --replace to wipe).
- Automatically generates/updates data.yaml for Ultralytics YOLO training.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import zipfile
import yaml
from cvat_sdk import make_client

import config_utils

STATE_FILE = Path(__file__).resolve().parent / ".last_cvat_task"
load_config = config_utils.load_config


def get_latest_task_id(client) -> int:
    """Find the latest CVAT task ID from .last_cvat_task or CVAT API."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if "task_id" in data:
                    print(f"[INFO] Using task ID {data['task_id']} from .last_cvat_task ({data.get('task_name', '')})")
                    return int(data["task_id"])
        except Exception:
            pass

    # Fallback to querying CVAT API for the highest task ID
    tasks = client.tasks.list()
    if not tasks:
        raise RuntimeError("No tasks found on CVAT server.")
    sorted_tasks = sorted(tasks, key=lambda t: t.id, reverse=True)
    latest = sorted_tasks[0]
    print(f"[INFO] Found latest CVAT task {latest.id}: '{latest.name}'")
    return latest.id


def export_and_prepare_yolo(
    task_ids: list,
    output_dir: str = "./datasets/yolo",
    val_split: float = 0.2,
    seed: int = 42,
    append: bool = True,
    cvat_url: str = None,
    cvat_user: str = None,
    cvat_pass: str = None,
    fallback_classes: list = None,
    coco_dataset: str = "coco128",
    datasets_dir: str = "datasets",
    include_frigate_plus: bool = None,
    frigate_plus_dir: str = None,
):
    random.seed(seed)
    out_path = Path(output_dir)
    temp_zip = Path("cvat_export_temp.zip")
    temp_extract = Path("cvat_export_temp_dir")

    if not cvat_user or not cvat_pass:
        print("[ERROR] CVAT username and password are required.")
        print("[ERROR] Please set them in 'config.yaml', via --cvat-user / --cvat-pass, or $CVAT_USER / $CVAT_PASS.")
        sys.exit(1)

    print(f"[INFO] Connecting to CVAT at {cvat_url} as '{cvat_user}'...")
    with make_client(cvat_url, credentials=(cvat_user, cvat_pass)) as client:
        # Resolve 'latest' or empty task_ids
        resolved_ids = []
        raw_ids = []
        for tid in task_ids:
            for piece in str(tid).replace(",", " ").split():
                if piece:
                    raw_ids.append(piece)

        for tid in raw_ids:
            if str(tid).lower() == "all":
                all_tasks = sorted(client.tasks.list(), key=lambda t: t.id)
                resolved_ids.extend([t.id for t in all_tasks])
            elif str(tid).lower() in ("latest", "auto", ""):
                resolved_ids.append(get_latest_task_id(client))
            else:
                try:
                    resolved_ids.append(int(tid))
                except ValueError:
                    print(f"[ERROR] Invalid task ID: {tid}")
                    sys.exit(1)

        # Remove duplicate IDs preserving order
        resolved_ids = list(dict.fromkeys(resolved_ids))
        print(f"[INFO] Target CVAT Task IDs to export: {resolved_ids}")

        # Prepare output directories
        if not append:
            for split in ["train", "val"]:
                shutil.rmtree(out_path / "images" / split, ignore_errors=True)
                shutil.rmtree(out_path / "labels" / split, ignore_errors=True)

        for split in ["train", "val"]:
            (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        all_classes = []
        extra_custom_classes = []

        for task_id in resolved_ids:
            task = client.tasks.retrieve(task_id)
            print(f"\n[INFO] Exporting task {task.id} ('{task.name}') in YOLO 1.1 format...")
            if temp_zip.exists():
                temp_zip.unlink()
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

            task.export_dataset(
                format_name="YOLO 1.1",
                filename=str(temp_zip),
                include_images=True,
            )

            print(f"[INFO] Extracting exported archive for task {task.id}...")
            with zipfile.ZipFile(temp_zip, "r") as zf:
                zf.extractall(temp_extract)

            # Discover classes
            names_file = temp_extract / "obj.names"
            classes = []
            if names_file.exists():
                with open(names_file, "r") as f:
                    classes = [line.strip() for line in f if line.strip()]
            if not classes:
                classes = [lbl.name for lbl in task.labels]

            for c in classes:
                if c not in all_classes:
                    all_classes.append(c)

            # Build class index remapping for this task to canonical COCO-80
            task_remapping = {}
            for local_idx, cname in enumerate(classes):
                coco_id = config_utils.get_coco_id(cname)
                if coco_id is not None:
                    task_remapping[local_idx] = coco_id
                else:
                    if cname not in extra_custom_classes:
                        extra_custom_classes.append(cname)
                    task_remapping[local_idx] = 80 + extra_custom_classes.index(cname)
            print(f"[INFO] Task {task.id} class remapping to COCO-80:")
            for local_idx, cname in enumerate(classes):
                target_id = task_remapping[local_idx]
                print(f"  - [{local_idx}] '{cname}' -> COCO ID {target_id}")

            # Find exported images
            source_dir = temp_extract / "obj_train_data"
            if not source_dir.exists():
                source_dir = temp_extract / "images"
            if not source_dir.exists():
                source_dir = temp_extract

            image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            images = [
                f for f in source_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            images.sort()

            print(f"[INFO] Task {task.id}: Found {len(images)} images.")
            random.shuffle(images)

            val_count = int(len(images) * val_split)
            val_images = set(images[:val_count])
            train_images = set(images[val_count:])

            def copy_sample(img_file: Path, split: str):
                # Ensure unique destination filename across tasks
                dest_stem = f"t{task_id}_{img_file.stem}" if append or len(resolved_ids) > 1 else img_file.stem
                dest_img = out_path / "images" / split / f"{dest_stem}{img_file.suffix}"
                shutil.copy2(img_file, dest_img)

                label_file = img_file.with_suffix(".txt")
                dest_label = out_path / "labels" / split / f"{dest_stem}.txt"
                if label_file.exists():
                    remapped_lines = []
                    with open(label_file, "r", encoding="utf-8") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            try:
                                local_id = int(parts[0])
                                new_id = task_remapping.get(local_id, local_id)
                                parts[0] = str(new_id)
                                remapped_lines.append(" ".join(parts) + "\n")
                            except ValueError:
                                remapped_lines.append(line)
                    with open(dest_label, "w", encoding="utf-8") as lf:
                        lf.writelines(remapped_lines)
                else:
                    # Empty label file for background images
                    dest_label.touch()

            for img in train_images:
                copy_sample(img, "train")
            for img in val_images:
                copy_sample(img, "val")

            # Clean temporary files for this task
            if temp_zip.exists():
                temp_zip.unlink()
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

        # Count total accumulated dataset samples
        total_train = len(list((out_path / "images" / "train").glob("*.*")))
        total_val = len(list((out_path / "images" / "val").glob("*.*")))

        # Handle external COCO dataset for co-training
        if not coco_dataset or coco_dataset not in ("coco128", "coco-val2017"):
            coco_dataset = "coco128"
        print(f"\n[INFO] Integrating external COCO dataset '{coco_dataset}' for co-training...")
        coco_info = config_utils.download_coco_dataset(coco_dataset, target_dir=datasets_dir)
        print(f"[INFO] External COCO train images: {coco_info['train_dir']}")
        print(f"[INFO] External COCO val images  : {coco_info['val_dir']}")

        # Build names dictionary for data.yaml (canonical COCO-80 classes + any extra custom classes)
        final_names = config_utils.get_coco_80_dict()
        for idx, cname in enumerate(extra_custom_classes):
            final_names[80 + idx] = cname

        # Build train / val paths in data.yaml combining local CVAT dataset, Frigate+, and COCO
        train_list = [str((out_path / "images" / "train").resolve())]
        val_list = [str((out_path / "images" / "val").resolve())]

        cfg = load_config()
        fp_cfg = config_utils.get_frigate_plus_config(cfg)
        if include_frigate_plus is None:
            include_frigate_plus = fp_cfg["enabled"]
        if frigate_plus_dir is None:
            frigate_plus_dir = fp_cfg["path"]

        fp_path = Path(frigate_plus_dir).resolve()
        if include_frigate_plus and fp_path.is_dir():
            fp_train = fp_path / "images" / "train"
            fp_val = fp_path / "images" / "val"
            if fp_train.is_dir():
                print(f"[INFO] External Frigate+ train images: {fp_train.resolve()}")
                train_list.append(str(fp_train.resolve()))
            if fp_val.is_dir():
                print(f"[INFO] External Frigate+ val images  : {fp_val.resolve()}")
                val_list.append(str(fp_val.resolve()))

        if coco_dataset and str(coco_dataset).lower() != "none":
            train_list.append(str(Path(coco_info["train_dir"]).resolve()))
            val_list.append(str(Path(coco_info["val_dir"]).resolve()))

        yaml_content = "train:\n"
        for p in train_list:
            yaml_content += f"  - {p}\n"
        yaml_content += "val:\n"
        for p in val_list:
            yaml_content += f"  - {p}\n"
        yaml_content += "\nnames:\n"

        for idx in sorted(final_names.keys()):
            yaml_content += f"  {idx}: {final_names[idx]}\n"

        yaml_path = out_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        # Write composite datasets/data.yaml if out_path is a subdirectory under datasets
        composite_yaml_path = out_path.parent / "data.yaml" if out_path.parent.name == "datasets" else yaml_path
        if composite_yaml_path != yaml_path:
            with open(composite_yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

        print("\n" + "=" * 70)
        print(" [SUCCESS] YOLO Dataset Successfully Prepared!")
        print(f" Output Directory : {out_path.resolve()}")
        print(f" Total Train Images: {total_train}")
        print(f" Total Val Images  : {total_val}")
        print(f" Config File       : {composite_yaml_path}")
        print(f" Classes ({len(all_classes)})     : {all_classes}")
        print("=" * 70 + "\n")


def main():
    cfg = load_config()
    default_cvat_url, default_cvat_user, default_cvat_pass = config_utils.get_cvat_credentials(cfg)
    coco_cfg = config_utils.get_coco_config(cfg)
    default_coco_dataset = coco_cfg["dataset"]

    fallback_classes = None
    if "categories" in cfg and isinstance(cfg["categories"], dict):
        fallback_classes = list(cfg["categories"].keys())
    elif "labels" in cfg and isinstance(cfg["labels"], list):
        fallback_classes = list(cfg["labels"])

    parser = argparse.ArgumentParser(
        description="Export CVAT dataset directly to YOLO training format with canonical COCO-80 label alignment."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--export",
        dest="export",
        nargs="*",
        default=["latest"],
        help="CVAT Task ID(s) to export (default: 'latest' or read from .last_cvat_task)",
    )
    default_out = "./datasets/yolo" if Path("datasets/yolo").is_dir() or not Path("yolo_dataset").is_dir() else "./yolo_dataset"
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_out,
        help=f"Target YOLO dataset directory (default: {default_out})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--replace",
        "--overwrite",
        dest="append",
        action="store_false",
        default=True,
        help="Wipe and replace existing dataset instead of appending to it (default: append).",
    )
    parser.add_argument(
        "--coco-dataset",
        choices=["coco128", "coco-val2017"],
        default=default_coco_dataset,
        help=f"External COCO dataset to include for co-training: 'coco128' (~7 MB) or 'coco-val2017' (~850 MB) (default: {default_coco_dataset})",
    )
    default_fp_cfg = config_utils.get_frigate_plus_config(cfg)
    parser.add_argument(
        "--frigate-plus",
        dest="include_frigate_plus",
        action="store_true",
        default=default_fp_cfg["enabled"],
        help="Include Frigate+ dataset in data.yaml splits (default: %(default)s)",
    )
    parser.add_argument(
        "--no-frigate-plus",
        dest="include_frigate_plus",
        action="store_false",
        help="Exclude Frigate+ dataset from data.yaml splits",
    )
    parser.add_argument(
        "--frigate-plus-dir",
        type=str,
        default=default_fp_cfg["path"],
        help=f"Path to Frigate+ dataset (default: {default_fp_cfg['path']})",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="./datasets",
        help="Directory to store downloaded external datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--cvat-url",
        type=str,
        default=default_cvat_url,
        help=f"CVAT server URL (default: {default_cvat_url})",
    )
    parser.add_argument(
        "--cvat-user",
        type=str,
        default=default_cvat_user,
        help=f"CVAT username (default: {default_cvat_user})",
    )
    parser.add_argument(
        "--cvat-pass",
        type=str,
        default=default_cvat_pass,
        help="CVAT password (default: set via config or $CVAT_PASS)",
    )

    args = parser.parse_args()

    # Re-read config if custom path passed
    if args.config:
        custom_cfg = load_config(args.config)
        if custom_cfg:
            c_url, c_user, c_pass = config_utils.get_cvat_credentials(custom_cfg)
            if args.cvat_url == default_cvat_url:
                args.cvat_url = c_url
            if args.cvat_user == default_cvat_user:
                args.cvat_user = c_user
            if args.cvat_pass == default_cvat_pass:
                args.cvat_pass = c_pass
            c_coco = config_utils.get_coco_config(custom_cfg)
            if args.coco_dataset == default_coco_dataset and c_coco.get("dataset"):
                args.coco_dataset = c_coco["dataset"]
            c_fp = config_utils.get_frigate_plus_config(custom_cfg)
            if args.include_frigate_plus == default_fp_cfg["enabled"]:
                args.include_frigate_plus = c_fp["enabled"]
            if args.frigate_plus_dir == default_fp_cfg["path"]:
                args.frigate_plus_dir = c_fp["path"]
            if "categories" in custom_cfg and isinstance(custom_cfg["categories"], dict):
                fallback_classes = list(custom_cfg["categories"].keys())
            elif "labels" in custom_cfg and isinstance(custom_cfg["labels"], list):
                fallback_classes = list(custom_cfg["labels"])

    task_ids = args.export if args.export else ["latest"]
    export_and_prepare_yolo(
        task_ids=task_ids,
        output_dir=args.output_dir,
        val_split=args.val_split,
        append=args.append,
        cvat_url=args.cvat_url,
        cvat_user=args.cvat_user,
        cvat_pass=args.cvat_pass,
        fallback_classes=fallback_classes,
        coco_dataset=args.coco_dataset,
        datasets_dir=args.datasets_dir,
        include_frigate_plus=args.include_frigate_plus,
        frigate_plus_dir=args.frigate_plus_dir,
    )


if __name__ == "__main__":
    main()
