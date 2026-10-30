#!/usr/bin/env python3
"""
Unified Frigate Model Training & Compilation Pipeline.

Consolidates local GPU/CPU training and remote DGX Spark training into a
single unified runner with local GPU training as the default.

Workflows:
  1. Export: Optional CVAT annotations export (--export)
  2. Train: Local GPU (Default) or Remote DGX Spark (-p spark / -p torq-spark)
  3. Compile: Local INT8 quantization (onnx2tf), TOSA MLIR lowering,
              and Torq NPU compilation (torq-compile for Astra SL2610),
              Hailo-10H HEF compilation, or NVIDIA ONNX export
  4. Deploy: Automated deployment to Coralboard SL2619 vision service
"""

import argparse
import atexit
import datetime
import importlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import config_utils
import prepare_for_onnx2tf
from deploy_to_coralboard import deploy_artifacts, derive_symlink_name

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def ensure_terminal_onlcr() -> None:
    """Ensure terminal newline translation (onlcr) is restored."""
    if sys.stdin.isatty():
        try:
            subprocess.run(["stty", "onlcr"], check=False, stderr=subprocess.DEVNULL)
        except Exception:
            pass


atexit.register(ensure_terminal_onlcr)


def maybe_reexec_in_venv() -> None:
    """Seamlessly re-execute using the project's Python virtual environment if not active."""
    project_root = Path(__file__).resolve().parent
    venv_candidates = [
        project_root / "venv-torq-312" / "bin" / "python",
        project_root / "venv" / "bin" / "python",
    ]
    for candidate in venv_candidates:
        if candidate.exists():
            target_python = candidate.resolve()
            current_python = Path(sys.executable).resolve()
            if current_python != target_python:
                os.environ["VIRTUAL_ENV"] = str(candidate.parent.parent)
                os.environ["PATH"] = f"{candidate.parent}:{os.environ.get('PATH', '')}"
                os.execv(str(target_python), [str(target_python)] + sys.argv)
            break

    bin_dir = Path(sys.prefix) / "bin"
    if bin_dir.is_dir():
        current_path = os.environ.get("PATH", "")
        if str(bin_dir) not in current_path.split(os.pathsep):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}"


def format_duration(total_sec: int) -> str:
    """Format total seconds into human-readable duration."""
    total_sec = max(0, int(total_sec))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s ({total_sec}s)"
    elif mins > 0:
        return f"{mins}m {secs:02d}s ({total_sec}s)"
    else:
        return f"{secs}s"


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
    check: bool = True,
) -> int:
    """Run an external command, streaming output to console and/or stripping ANSI for log file."""
    if not log_file and console:
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
        if check and res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, cmd)
        return res.returncode

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    f = open(log_file, "a", encoding="utf-8") if log_file else None
    try:
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if console:
                sys.stdout.write(text)
                sys.stdout.flush()
            if f:
                clean_text = ANSI_ESCAPE_RE.sub("", text)
                f.write(clean_text)
                f.flush()
    finally:
        if f:
            f.close()
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def relocate_root_checkpoints(project_root: Path, base_dir: Path) -> None:
    """Move any stray *.pt checkpoint files from project root into models/base/."""
    base_dir.mkdir(parents=True, exist_ok=True)
    for pt in project_root.glob("*.pt"):
        if pt.is_file():
            dest = base_dir / pt.name
            try:
                shutil.move(str(pt), str(dest))
            except Exception:
                pass


def verify_required_tools(target_format: str, skip_compile: bool) -> Optional[Path]:
    """Verify tool binaries exist in active environment or return Hailo venv path."""
    required = ["yolo"]
    hailo_venv_path: Optional[Path] = None

    if not skip_compile:
        if target_format == "torq":
            required.extend(["onnx2tf", "tosa-converter-for-tflite", "torq-compile"])
        elif target_format == "hailo":
            hailo_candidate = Path("./venv-hailo-310")
            if (hailo_candidate / "bin" / "hailomz").is_file():
                hailo_venv_path = hailo_candidate
            else:
                required.append("hailomz")

    for tool in required:
        if not shutil.which(tool):
            print(f"[ERROR] Required tool '{tool}' not found in active environment. Run 'pip install -r requirements.txt'.")
            sys.exit(1)

    return hailo_venv_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frigate Model Training & Compiler Pipeline (Torq NPU & NVIDIA GPU ONNX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--profile",
        dest="profile",
        help="Select profile from config.yaml (e.g. 'torq', 'torq-spark', 'torq-spark-yolo26n', 'nvidia', 'nvidia-spark', 'hailo', 'hailo-spark', 'cpu-torq')",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        nargs="?",
        const="",
        default=None,
        help="Skip training; compile checkpoint directly (or latest best.pt)",
    )
    parser.add_argument(
        "--export",
        dest="export_tasks",
        nargs="*",
        default=None,
        help="Export CVAT task(s) to datasets/yolo/ before training (default: latest)",
    )
    parser.add_argument(
        "--deploy",
        dest="deploy_target",
        nargs="?",
        const="",
        default=None,
        help="Deploy compiled model & labels to local dir or remote host (user@ip:/path)",
    )
    parser.add_argument(
        "--skip-compile",
        dest="skip_compile",
        action="store_true",
        help="Train only; skip compilation/export",
    )
    parser.add_argument(
        "--skip-deploy",
        dest="skip_deploy",
        action="store_true",
        help="Skip deployment step even if configured in config.yaml",
    )
    parser.add_argument(
        "--pull",
        dest="pull_only",
        action="store_true",
        help="Pull latest weights from Spark and compile locally (skip training)",
    )
    parser.add_argument(
        "-m", "--base-model",
        dest="base_model",
        help="Base checkpoint / architecture (default: yolov8n.pt, or yolov9s.pt for nvidia)",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        dest="imgsz",
        type=int,
        help="Input image resolution (e.g. 320 or 640)",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        help="Number of dataloader worker threads",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        help="Dataset caching mode ('ram', 'disk', or 'false')",
    )
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        help="Training optimizer ('AdamW', 'SGD', etc.)",
    )
    parser.add_argument(
        "--lr0",
        dest="lr0",
        type=float,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lrf",
        dest="lrf",
        type=float,
        help="Final learning rate factor (lr0 * lrf)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        help="Training device ('0' local GPU, 'cpu', or '0,1' Spark)",
    )
    parser.add_argument(
        "--host",
        dest="spark_host",
        help="SSH destination for remote training (e.g. user@spark-host)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume/fine-tune from previous checkpoint (default: train from scratch)",
    )
    parser.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        help="Wipe and replace existing dataset instead of appending",
    )
    parser.add_argument(
        "--coco-dataset",
        dest="coco_dataset",
        help="COCO subset for co-training ('coco128' or 'coco-val2017')",
    )

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument(
        "--frigate-plus",
        dest="frigate_plus",
        action="store_true",
        default=None,
        help="Include Frigate+ dataset in training & validation splits",
    )
    fp_group.add_argument(
        "--no-frigate-plus",
        dest="frigate_plus",
        action="store_false",
        help="Exclude Frigate+ dataset from training splits",
    )

    clean_group = parser.add_mutually_exclusive_group()
    clean_group.add_argument(
        "--clean-images",
        dest="clean_images",
        action="store_true",
        default=None,
        help="Remove synced YOLO and Frigate+ dataset images from remote Spark host after training",
    )
    clean_group.add_argument(
        "--no-clean-images",
        dest="clean_images",
        action="store_false",
        help="Leave synced dataset images on remote Spark host after training",
    )

    parser.add_argument(
        "--extra-args",
        dest="extra_args",
        default="",
        help="Forward extra arguments directly to 'yolo detect train'",
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_file",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    config_utils.add_logging_args(parser)
    return parser.parse_args()


def print_timing_report(
    weights_path: Optional[Path],
    preprocess_sec: int,
    train_sec: int,
    postprocess_sec: int,
) -> None:
    """Print checkpoint metrics and pipeline stage timing breakdown."""
    total_elapsed = preprocess_sec + train_sec + postprocess_sec
    if weights_path and weights_path.is_file():
        try:
            config_utils.print_checkpoint_metrics(str(weights_path))
            print("--------------------------------------------------------------------------")
        except Exception:
            pass
    print("Timing Summary:")
    print(f"  Preprocessing  : {format_duration(preprocess_sec)}")
    print(f"  Training       : {format_duration(train_sec)}")
    print(f"  Postprocessing : {format_duration(postprocess_sec)}")
    print(f"  Total Time     : {format_duration(total_elapsed)}")


def main() -> int:
    ensure_terminal_onlcr()
    maybe_reexec_in_venv()

    args = parse_arguments()
    pipeline_start_ts = time.time()
    preprocess_start_ts = pipeline_start_ts
    preprocess_elapsed = 0
    training_elapsed = 0
    postprocess_start_ts = pipeline_start_ts
    postprocess_elapsed = 0

    project_root = Path.cwd()
    models_dir = project_root / "models"
    models_base_dir = models_dir / "base"
    staging_root = project_root / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    models_base_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration & profile
    cfg = config_utils.load_config(args.config_file)
    try:
        profile = config_utils.get_profile(args.profile, cfg=cfg, config_path=args.config_file)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    profile_name = profile["name"]
    target_format = profile["target"]
    os.environ["FRIGATE_TARGET"] = target_format

    timestamp = os.environ.get("FRIGATE_RUN_TIMESTAMP") or datetime.datetime.now().strftime("%y%m%d-%H%M")
    os.environ["FRIGATE_RUN_TIMESTAMP"] = timestamp

    models_target_dir = models_dir / target_format
    models_run_dir = models_target_dir / timestamp
    models_run_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    console_enabled = args.log_console if args.log_console is not None else profile.get("log_console", True)
    file_enabled = args.log_file if args.log_file is not None else profile.get("log_file", True)
    target_log_dir = Path(args.log_dir) if args.log_dir else (models_run_dir / "logs")
    log_file_path = (target_log_dir / "train.log") if file_enabled else None
    if file_enabled:
        target_log_dir.mkdir(parents=True, exist_ok=True)

    logger = config_utils.setup_logging_from_args(
        args,
        script_name="train",
        target=target_format,
    )

    # Apply Profile Overrides
    runner = "spark" if (profile["remote"] or args.pull_only or args.spark_host) else "local"
    spark_host = args.spark_host or profile["host"] or os.environ.get("SPARK_HOST") or os.environ.get("DGX_HOST") or ""
    remote_dir = profile["remote_dir"]

    batch = args.batch if args.batch is not None else profile["batch"]
    device = args.device if args.device is not None else profile["device"]
    epochs = args.epochs if args.epochs is not None else profile["epochs"]
    patience = args.patience if args.patience is not None else profile["patience"]
    workers = args.workers if args.workers is not None else profile["workers"]
    cache = args.cache if args.cache is not None else profile["cache"]
    base_model = args.base_model if args.base_model else profile["base_model"]
    imgsz = args.imgsz if args.imgsz is not None else profile["imgsz"]
    hw_arch = profile["hw_arch"]
    coco_dataset = args.coco_dataset if args.coco_dataset else profile["coco_dataset"]
    clean_images = args.clean_images if args.clean_images is not None else profile["clean_images"]

    frigate_plus = args.frigate_plus if args.frigate_plus is not None else profile["frigate_plus"]
    frigate_plus_dir = profile["frigate_plus_dir"]

    optimizer = args.optimizer if args.optimizer is not None else profile["optimizer"]
    lr0 = args.lr0 if args.lr0 is not None else profile["lr0"]
    lrf = args.lrf if args.lrf is not None else profile["lrf"]
    cos_lr = profile["cos_lr"]
    warmup_epochs = profile["warmup_epochs"]
    warmup_momentum = profile["warmup_momentum"]
    warmup_bias_lr = profile["warmup_bias_lr"]

    deploy_host = profile["deploy_host"]
    deploy_dir = profile["deploy_dir"]
    if args.deploy_target is not None:
        if ":" in args.deploy_target:
            deploy_host = args.deploy_target
        elif args.deploy_target != "":
            deploy_dir = args.deploy_target
    if args.skip_deploy:
        deploy_host = ""
        deploy_dir = ""

    from_scratch = not args.resume

    # Normalize model names
    base_model_name = Path(base_model).stem
    run_name = base_model_name
    run_name_stem = Path(run_name).stem
    staging_dir = staging_root / run_name
    staging_dir.mkdir(parents=True, exist_ok=True)

    labels_file = models_run_dir / f"{target_format}_{base_model_name}_{timestamp}.json"
    output_vmfb = models_run_dir / f"{target_format}_{run_name_stem}_{timestamp}.vmfb"
    output_onnx = models_run_dir / f"{target_format}_{run_name_stem}_{timestamp}.onnx"
    output_hef = models_run_dir / f"{target_format}_{run_name_stem}_{timestamp}.hef"
    data_yaml = Path("datasets/data.yaml")

    # Banner Output
    print("==========================================================================")
    print("  Frigate Training & Compilation Pipeline")
    print("==========================================================================")
    if runner == "spark":
        print(f"  Profile:       {profile_name.upper()} (Remote Host: {spark_host or '[Not Set]'})")
    else:
        print(f"  Profile:       {profile_name.upper()} (Local GPU, Device: {device})")

    if target_format == "nvidia":
        print(f"  Target:        NVIDIA Video Card (ONNX -> {output_onnx})")
    elif target_format == "hailo":
        print(f"  Target:        Hailo-10H NPU ({hw_arch} HEF -> {output_hef})")
    else:
        print(f"  Target:        Coralboard SL2619 Torq NPU (VMFB -> {output_vmfb})")

    print(f"  Hyperparams:   Batch: {batch}, Workers: {workers}, Epochs: {epochs}, Patience: {patience}, Cache: {cache}")
    print(f"  Fine-Tuning:   Optimizer: {optimizer}, lr0: {lr0}, lrf: {lrf}, cos_lr: {cos_lr}")
    print(f"  Warm-Up:       Epochs: {warmup_epochs}, Momentum: {warmup_momentum}, Bias LR: {warmup_bias_lr}")
    print(f"  Base Model:    {base_model} (Run Name: {run_name}, Co-Training: {coco_dataset})")
    print(f"  Resolution:    {imgsz}x{imgsz}")
    print("==========================================================================")

    hailo_venv_path = verify_required_tools(target_format, args.skip_compile)

    # --------------------------------------------------------------------------
    # Step 1: Optional CVAT Dataset Export
    # --------------------------------------------------------------------------
    if args.export_tasks is not None:
        print("\n==========================================================================")
        print("  Step 1: Exporting CVAT Dataset to YOLO Format")
        print("==========================================================================")
        raw_export_tasks = args.export_tasks if args.export_tasks else ["latest"]
        cvat_exporter = importlib.import_module("export-cvat-to-yolo")
        cvat_url, cvat_user, cvat_pass = config_utils.get_cvat_credentials(cfg)
        categories = cfg.get("categories", {}) if isinstance(cfg.get("categories"), dict) else {}
        fallback_classes = list(categories.keys()) if categories else None

        cvat_exporter.export_and_prepare_yolo(
            task_ids=raw_export_tasks,
            output_dir="./datasets/yolo",
            append=not args.replace,
            cvat_url=cvat_url,
            cvat_user=cvat_user,
            cvat_pass=cvat_pass,
            fallback_classes=fallback_classes,
            coco_dataset=coco_dataset,
            datasets_dir="datasets",
            include_frigate_plus=frigate_plus,
            frigate_plus_dir=frigate_plus_dir,
        )

    # --------------------------------------------------------------------------
    # Step 2: Training / Checkpoint Selection
    # --------------------------------------------------------------------------
    weights_path: Optional[Path] = None
    local_weights_dir = staging_dir / "weights"

    if args.compile_model is not None and args.compile_model != "":
        custom_pt = Path(args.compile_model)
        if not custom_pt.is_file():
            logger.error("Specified checkpoint '%s' does not exist!", custom_pt)
            return 1
        weights_path = custom_pt
        logger.info("Using specified checkpoint: %s", weights_path)
        preprocess_elapsed = int(time.time() - preprocess_start_ts)
        training_elapsed = 0
        postprocess_start_ts = time.time()

    elif args.compile_model is not None and args.compile_model == "":
        # Look for existing best.pt in staging/
        candidates = list(staging_root.glob(f"*{run_name}*/best.pt")) + list(staging_root.glob(f"*{run_name}*/weights/best.pt"))
        if not candidates:
            candidates = list(staging_root.glob(f"*{base_model_name}*/best.pt")) + list(staging_root.glob(f"*{base_model_name}*/weights/best.pt"))
        if not candidates:
            logger.error("No existing 'best.pt' found for '%s'. Run without --compile to train first.", run_name)
            return 1
        sorted_candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        weights_path = sorted_candidates[0]
        logger.info("Using existing checkpoint: %s", weights_path)
        preprocess_elapsed = int(time.time() - preprocess_start_ts)
        training_elapsed = 0
        postprocess_start_ts = time.time()

    elif runner == "spark":
        if not spark_host:
            logger.error("No Spark host specified! Set --host USER@HOST or 'remote_training.host' in config.yaml.")
            return 1

        print("\n==========================================================================")
        print(f"  Step 2: Training Remotely on NVIDIA DGX Spark ({spark_host})")
        print("==========================================================================")

        logger.info("Testing SSH connectivity and resolving remote workspace on %s...", spark_host)
        res = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", spark_host, f"eval \"TARGET_DIR={remote_dir}\" && mkdir -p \"${{TARGET_DIR}}/models/base\" && cd \"${{TARGET_DIR}}\" && pwd"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            logger.error("Could not connect to %s via SSH or resolve remote directory.", spark_host)
            return 1
        resolved_remote_dir = res.stdout.strip().splitlines()[-1]
        logger.info("Remote workspace verified: %s:%s", spark_host, resolved_remote_dir)

        # Cleanup stray ~ folder if present
        subprocess.run(["ssh", spark_host, "rm -rf \"$HOME/~\" 2>/dev/null || true"], check=False)

        if args.pull_only:
            logger.info("--pull specified: skipping remote execution, pulling existing weights from Spark...")
            preprocess_elapsed = int(time.time() - preprocess_start_ts)
            training_elapsed = 0
            postprocess_start_ts = time.time()
        else:
            # Setup remote venv if missing
            logger.info("Verifying remote Python environment on %s...", spark_host)
            setup_remote_venv_cmd = (
                f"cd \"{resolved_remote_dir}\" && "
                "if [ ! -d \"venv\" ]; then "
                "echo '[SPARK] Creating Python 3 virtual environment...' && "
                "python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && "
                "pip install ultralytics onnx onnxruntime pyyaml requests cvat-sdk; fi"
            )
            run_command(["ssh", spark_host, setup_remote_venv_cmd], log_file=log_file_path, console=console_enabled)

            # Sync dataset and code
            logger.info("Synchronizing dataset and code to %s:%s...", spark_host, resolved_remote_dir)
            scripts_to_sync = ["config_utils.py"]
            if Path(args.config_file).is_file():
                scripts_to_sync.append(args.config_file)

            if shutil.which("rsync"):
                fast_rsh = ["-e", "ssh -T -c aes128-gcm@openssh.com -o Compression=no"]
                run_command(["rsync", "-avW", "--inplace", "--copy-links"] + fast_rsh + scripts_to_sync + [f"{spark_host}:{resolved_remote_dir}/"], log_file=log_file_path, console=console_enabled)
                subprocess.run(["ssh", spark_host, f"mkdir -p '{resolved_remote_dir}/datasets'"], check=True)
                if data_yaml.is_file():
                    run_command(["rsync", "-avW", "--inplace"] + fast_rsh + [str(data_yaml), f"{spark_host}:{resolved_remote_dir}/datasets/data.yaml"], log_file=log_file_path, console=console_enabled)
                run_command([
                    "rsync", "-avW", "--inplace", "--delete",
                    "-e", "ssh -T -c aes128-gcm@openssh.com -o Compression=no",
                    "--exclude=coco*", "--exclude=*.zip", "--exclude=*.tar*", "--exclude=__pycache__",
                    "datasets/yolo/", f"{spark_host}:{resolved_remote_dir}/datasets/yolo/"
                ], log_file=log_file_path, console=console_enabled)
                if frigate_plus and Path(frigate_plus_dir).is_dir():
                    logger.info("Synchronizing Frigate+ dataset to %s:%s/datasets/frigate-plus/...", spark_host, resolved_remote_dir)
                    subprocess.run(["ssh", spark_host, f"mkdir -p '{resolved_remote_dir}/datasets/frigate-plus'"], check=True)
                    run_command([
                        "rsync", "-avW", "--inplace", "--delete",
                        "-e", "ssh -T -c aes128-gcm@openssh.com -o Compression=no",
                        "--exclude=__pycache__",
                        f"{frigate_plus_dir}/", f"{spark_host}:{resolved_remote_dir}/datasets/frigate-plus/"
                    ], log_file=log_file_path, console=console_enabled)
            else:
                for sc in scripts_to_sync:
                    run_command(["scp", sc, f"{spark_host}:{resolved_remote_dir}/"], log_file=log_file_path, console=console_enabled)
                subprocess.run(["ssh", spark_host, f"mkdir -p '{resolved_remote_dir}/datasets/yolo'/'{{images,labels}}'"], check=True)
                run_command(["scp", "-r", "datasets/yolo/images", "datasets/yolo/labels", f"{spark_host}:{resolved_remote_dir}/datasets/yolo/"], log_file=log_file_path, console=console_enabled)
                if frigate_plus and Path(frigate_plus_dir).is_dir():
                    subprocess.run(["ssh", spark_host, f"mkdir -p '{resolved_remote_dir}/datasets/frigate-plus'/'{{images,labels}}'"], check=True)
                    run_command(["scp", "-r", f"{frigate_plus_dir}/images", f"{frigate_plus_dir}/labels", f"{spark_host}:{resolved_remote_dir}/datasets/frigate-plus/"], log_file=log_file_path, console=console_enabled)

            # Sync previous checkpoint if resuming or base model
            if not from_scratch and (local_weights_dir / "best.pt").is_file():
                subprocess.run(["ssh", spark_host, f"mkdir -p '{resolved_remote_dir}/staging/{run_name}/weights'"], check=True)
                run_command(["scp", str(local_weights_dir / "best.pt"), f"{spark_host}:{resolved_remote_dir}/staging/{run_name}/weights/best.pt"], log_file=log_file_path, console=console_enabled)

            if (models_base_dir / base_model).is_file():
                run_command(["scp", str(models_base_dir / base_model), f"{spark_host}:{resolved_remote_dir}/models/base/"], log_file=log_file_path, console=console_enabled)
            elif Path(base_model).is_file():
                run_command(["scp", base_model, f"{spark_host}:{resolved_remote_dir}/models/base/"], log_file=log_file_path, console=console_enabled)

            # Prepare remote dataset manifest
            logger.info("Preparing remote dataset manifest on Spark...")
            prepare_remote_yaml_cmd = (
                f"cd \"{resolved_remote_dir}\" && "
                f"source \"{resolved_remote_dir}/venv/bin/activate\" && "
                f"export PYTHONPATH=\"{resolved_remote_dir}:${{PYTHONPATH:-}}\" && "
                f"python3 -c 'import sys, config_utils; config_utils.prepare_training_data_yaml(sys.argv[1], sys.argv[2], include_frigate_plus=(sys.argv[3] == \"True\"), frigate_plus_dir=sys.argv[4])' "
                f"\"{coco_dataset}\" \"{resolved_remote_dir}\" \"{'True' if frigate_plus else 'False'}\" \"{frigate_plus_dir}\""
            )
            run_command(["ssh", spark_host, prepare_remote_yaml_cmd], log_file=log_file_path, console=console_enabled)

            # Launch training on Spark
            preprocess_elapsed = int(time.time() - preprocess_start_ts)
            train_start_ts = time.time()
            logger.info("Launching YOLO fine-tuning on %s (device=%s, batch=%s, optimizer=%s, lr0=%s, warmup=%s)...", spark_host, device, batch, optimizer, lr0, warmup_epochs)

            remote_train_cmd = (
                f"cd \"{resolved_remote_dir}\" && "
                f"source \"{resolved_remote_dir}/venv/bin/activate\" && "
                f"export PYTHONPATH=\"{resolved_remote_dir}:${{PYTHONPATH:-}}\" && "
                "export PYTHONUNBUFFERED=1 && "
                f"START_MODEL='{base_model}' && "
                f"if [ -f \"models/base/{base_model}\" ]; then START_MODEL=\"models/base/{base_model}\"; fi && "
                f"if [ \"{'false' if from_scratch else 'true'}\" = \"true\" ]; then "
                f"LATEST_EXISTING=$(find \"{resolved_remote_dir}/staging\" -path \"*{run_name}*\" -name \"best.pt\" -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '); "
                "if [ -n \"${LATEST_EXISTING}\" ] && [ -f \"${LATEST_EXISTING}\" ]; then START_MODEL=\"${LATEST_EXISTING}\"; fi; fi && "
                "echo '==========================================================================' && "
                "echo '  Executing: yolo detect train on NVIDIA DGX Spark' && "
                "echo '==========================================================================' && "
                f"yolo detect train data='datasets/data.yaml' model=\"${{START_MODEL}}\" epochs={epochs} imgsz={imgsz} "
                f"batch={batch} workers={workers} cache={cache} patience={patience} optimizer='{optimizer}' "
                f"lr0={lr0} lrf={lrf} cos_lr={cos_lr} warmup_epochs={warmup_epochs} warmup_momentum={warmup_momentum} "
                f"warmup_bias_lr={warmup_bias_lr} device='{device}' project=\"{resolved_remote_dir}/staging\" "
                f"name='{run_name}' exist_ok=True {args.extra_args}"
            )
            run_command(["ssh", spark_host, remote_train_cmd], log_file=log_file_path, console=console_enabled)
            training_elapsed = int(time.time() - train_start_ts)
            postprocess_start_ts = time.time()

        # Retrieve weights from Spark
        logger.info("Retrieving trained best.pt from %s to local system...", spark_host)
        local_weights_dir.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["ssh", spark_host, f"find \"{resolved_remote_dir}/staging\" -path \"*{run_name}*\" -name \"best.pt\" -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '"],
            capture_output=True,
            text=True,
        )
        remote_best_pt = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        if not remote_best_pt:
            logger.error("Could not locate best.pt on %s:%s/staging/", spark_host, resolved_remote_dir)
            return 1

        run_command(["scp", f"{spark_host}:{remote_best_pt}", str(local_weights_dir / "best.pt")], log_file=log_file_path, console=console_enabled)
        remote_results_csv = str(Path(remote_best_pt).parent.parent / "results.csv")
        subprocess.run(["scp", f"{spark_host}:{remote_results_csv}", str(staging_dir / "results.csv")], check=False, stderr=subprocess.DEVNULL)
        weights_path = local_weights_dir / "best.pt"
        logger.info("Trained weights retrieved successfully to %s", weights_path)

        if clean_images:
            logger.info("Cleaning up remote YOLO and Frigate+ dataset images on %s...", spark_host)
            cleanup_cmd = f"rm -rf \"{resolved_remote_dir}/datasets/yolo\" \"{resolved_remote_dir}/datasets/frigate-plus\" \"{resolved_remote_dir}/datasets/data.yaml\" 2>/dev/null || true"
            subprocess.run(["ssh", spark_host, cleanup_cmd], check=False)

    else:
        # Local GPU/CPU Training
        yolo_train_img = Path("datasets/yolo/images/train")
        fp_train_img = Path(frigate_plus_dir) / "images/train"
        if data_yaml.stem == "data" and (yolo_train_img.is_dir() or fp_train_img.is_dir()):
            config_utils.prepare_training_data_yaml(coco_dataset, ".", include_frigate_plus=frigate_plus, frigate_plus_dir=frigate_plus_dir)

        if not data_yaml.is_file():
            logger.error("Dataset configuration file '%s' not found. Run with --export first.", data_yaml)
            return 1

        print("\n==========================================================================")
        print(f"  Step 2: Training Locally (Device: {device}, Batch: {batch})")
        print("==========================================================================")

        relocate_root_checkpoints(project_root, models_base_dir)

        start_model = base_model
        if (models_base_dir / base_model).is_file():
            start_model = str(models_base_dir / base_model)
        elif (models_base_dir / f"{base_model_name}.pt").is_file():
            start_model = str(models_base_dir / f"{base_model_name}.pt")

        if not from_scratch:
            candidates = list(staging_root.glob(f"*{run_name}*/best.pt")) + list(staging_root.glob(f"*{run_name}*/weights/best.pt"))
            if candidates:
                sorted_c = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                start_model = str(sorted_c[0])
                logger.info("Found existing trained checkpoint: %s. Fine-tuning from it...", start_model)

        preprocess_elapsed = int(time.time() - preprocess_start_ts)
        train_start_ts = time.time()
        logger.info("Executing: yolo detect train on device='%s' (optimizer=%s, lr0=%s, warmup=%s)...", device, optimizer, lr0, warmup_epochs)

        train_cmd = [
            "yolo", "detect", "train",
            f"data={data_yaml}",
            f"model={start_model}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"workers={workers}",
            f"cache={cache}",
            f"patience={patience}",
            f"optimizer={optimizer}",
            f"lr0={lr0}",
            f"lrf={lrf}",
            f"cos_lr={cos_lr}",
            f"warmup_epochs={warmup_epochs}",
            f"warmup_momentum={warmup_momentum}",
            f"warmup_bias_lr={warmup_bias_lr}",
            f"device={device}",
            f"project={staging_root}",
            f"name={run_name}",
            "exist_ok=True",
        ]
        if args.extra_args:
            train_cmd.extend(args.extra_args.split())

        run_command(train_cmd, log_file=log_file_path, console=console_enabled)
        training_elapsed = int(time.time() - train_start_ts)
        postprocess_start_ts = time.time()

        relocate_root_checkpoints(project_root, models_base_dir)

        trained_best = local_weights_dir / "best.pt"
        if not trained_best.is_file():
            candidates = list(staging_root.glob(f"*{run_name}*/best.pt")) + list(staging_root.glob(f"*{run_name}*/weights/best.pt"))
            if candidates:
                sorted_c = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                trained_best = sorted_c[0]

        if not trained_best.is_file():
            logger.error("Training completed but could not locate best.pt!")
            return 1

        weights_path = trained_best
        logger.info("Training complete! Checkpoint: %s", weights_path)

    # --------------------------------------------------------------------------
    # Step 3: Model Export & Target Compilation
    # --------------------------------------------------------------------------
    if args.skip_compile:
        postprocess_elapsed = int(time.time() - postprocess_start_ts)
        logger.info("--skip-compile specified: exiting pipeline before compilation.")
        print("\n==========================================================================")
        print_timing_report(weights_path, preprocess_elapsed, training_elapsed, postprocess_elapsed)
        print("==========================================================================")
        return 0

    if not weights_path or not weights_path.is_file():
        logger.error("Cannot compile: weights checkpoint '%s' not found.", weights_path)
        return 1

    # Ensure local datasets/data.yaml has valid paths for the current environment
    yolo_train_img = Path("datasets/yolo/images/train")
    fp_train_img = Path(frigate_plus_dir) / "images/train"
    if yolo_train_img.is_dir() or fp_train_img.is_dir():
        needs_refresh = False
        if data_yaml.is_file():
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f) or {}
                check_paths = (ydata.get("val", []) if isinstance(ydata.get("val"), list) else [ydata.get("val")]) + \
                              (ydata.get("train", []) if isinstance(ydata.get("train"), list) else [ydata.get("train")])
                if any(p and Path(str(p)).is_absolute() and not Path(str(p)).exists() for p in check_paths if p):
                    needs_refresh = True
            except Exception:
                needs_refresh = True
        else:
            needs_refresh = True

        if needs_refresh:
            logger.info("Refreshing %s with local workspace paths...", data_yaml)
            config_utils.prepare_training_data_yaml(coco_dataset, ".", include_frigate_plus=frigate_plus, frigate_plus_dir=frigate_plus_dir)

    print("\n==========================================================================")
    if target_format == "nvidia":
        print(f"  Step 3: NVIDIA GPU ONNX Export ({imgsz}x{imgsz})")
    elif target_format == "hailo":
        print(f"  Step 3: Hailo-10H NPU Export, Calibration & HEF Compilation ({imgsz}x{imgsz})")
    else:
        print(f"  Step 3: Torq NPU Export, Quantization & Compilation ({imgsz}x{imgsz})")
    print("==========================================================================")

    onnx_opset = 11 if target_format == "hailo" else 17
    exported_onnx = weights_path.with_suffix(".onnx")
    if not exported_onnx.is_file() or weights_path.stat().st_mtime > exported_onnx.stat().st_mtime:
        logger.info("Exporting PyTorch model to ONNX (%dx%d, opset=%d)...", imgsz, imgsz, onnx_opset)
        export_cmd = [
            "yolo", "export",
            f"model={weights_path}",
            f"imgsz={imgsz}",
            "format=onnx",
            f"opset={onnx_opset}",
        ]
        if target_format != "hailo":
            export_cmd.extend(["dynamic=False", "simplify=True"])
        run_command(export_cmd, log_file=log_file_path, console=console_enabled)

    if not exported_onnx.is_file():
        logger.error("Failed to locate exported ONNX at %s", exported_onnx)
        return 1

    staging_best_onnx = staging_dir / "best.onnx"
    shutil.copy2(exported_onnx, staging_best_onnx)

    primary_artifact: Path
    if target_format == "nvidia":
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_onnx, output_onnx)
        prepare_for_onnx2tf.step4_generate_labels_json(
            str(data_yaml),
            str(labels_file),
            "",
            base_model_name,
            base_model_name,
            target=target_format,
            imgsz=imgsz,
        )
        logger.info("NVIDIA GPU ONNX model generated successfully: %s", output_onnx)
        logger.info("Target is NVIDIA GPU ('%s'): skipping NPU quantization and compilation.", target_format)
        primary_artifact = output_onnx

    elif target_format == "hailo":
        logger.info("Preparing calibration image directory and labels for Hailo Model Zoo (%s)...", hw_arch)
        hailo_calib_dir = staging_dir / "hailo_calib_imgs"
        prepare_for_onnx2tf.step3_prepare_calibration(
            data_yaml_path=str(data_yaml),
            calib_out=str(hailo_calib_dir),
            size=imgsz,
            max_images=300,
            strategy="balanced",
        )
        prepare_for_onnx2tf.step4_generate_labels_json(
            data_yaml_path=str(data_yaml),
            labels_out=str(labels_file),
            tflite_path="",
            base_model_name=base_model_name,
            model_name=base_model_name,
            target=target_format,
            imgsz=imgsz,
        )

        dataset_classes = config_utils.load_dataset_classes(str(data_yaml))
        num_classes = len(dataset_classes)
        logger.info("Compiling ONNX to Hailo HEF via hailomz (arch=%s, classes=%d)...", hw_arch, num_classes)

        hailo_env = os.environ.copy()
        hmz_yaml_args = [run_name_stem]
        hailomz_bin = "hailomz"

        if hailo_venv_path and (hailo_venv_path / "bin" / "hailomz").is_file():
            hailo_venv_abs = hailo_venv_path.resolve()
            hailomz_bin = str(hailo_venv_abs / "bin" / "hailomz")
            (hailo_venv_abs / "etc/hailo").mkdir(parents=True, exist_ok=True)
            (hailo_venv_abs / "etc/hailo/check_system_requirements_was_called").touch()

            libdevice_matches = list(hailo_venv_abs.glob("lib/**/libdevice.10.bc"))
            if libdevice_matches:
                nvvm_dir = hailo_venv_abs / "nvvm/libdevice"
                nvvm_dir.mkdir(parents=True, exist_ok=True)
                sym = nvvm_dir / "libdevice.10.bc"
                if not sym.exists():
                    sym.symlink_to(libdevice_matches[0])
                hailo_env["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={hailo_venv_abs}"

            hailo_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            try:
                gpu_res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                if gpu_res.returncode == 0:
                    gpu_mb = int(gpu_res.stdout.strip().splitlines()[0])
                    if gpu_mb < 6000:
                        logger.warning("GPU has %d MiB VRAM (< 6000 MiB); running Hailo on CPU...", gpu_mb)
                        hailo_env["CUDA_VISIBLE_DEVICES"] = ""
            except Exception:
                pass

            hailo_env["VIRTUAL_ENV"] = str(hailo_venv_abs)
            hailo_env["PATH"] = f"{hailo_venv_abs / 'bin'}:{hailo_env.get('PATH', '')}"

            hmz_cfgs = list(hailo_venv_abs.glob("lib/**/hailo_model_zoo/cfg"))
            if hmz_cfgs:
                hmz_cfg = hmz_cfgs[0]
                post_cfg = hmz_cfg / "postprocess_config"
                alls_dir = hmz_cfg / "alls"
                if post_cfg.is_dir() and alls_dir.is_dir():
                    post_sym = alls_dir / "postprocess_config"
                    if not post_sym.exists() and not post_sym.is_symlink():
                        try:
                            post_sym.symlink_to("../postprocess_config")
                        except Exception:
                            pass
                if (hmz_cfg / f"networks/{run_name_stem}.yaml").is_file():
                    hmz_yaml_args = ["--yaml", str(hmz_cfg / f"networks/{run_name_stem}.yaml")]

        compile_cmd = [
            hailomz_bin, "compile",
            "--ckpt", str(staging_best_onnx.resolve()),
            "--calib-path", str(hailo_calib_dir.resolve()),
        ] + hmz_yaml_args + [
            "--end-node-names",
            "/model.22/cv2.0/cv2.0.2/Conv",
            "/model.22/cv3.0/cv3.0.2/Conv",
            "/model.22/cv2.1/cv2.1.2/Conv",
            "/model.22/cv3.1/cv3.1.2/Conv",
            "/model.22/cv2.2/cv2.2.2/Conv",
            "/model.22/cv3.2/cv3.2.2/Conv",
            "--classes", str(num_classes),
            "--hw-arch", hw_arch,
        ]
        run_command(compile_cmd, cwd=staging_dir, env=hailo_env, log_file=log_file_path, console=console_enabled)

        generated_hef = staging_dir / f"{run_name_stem}.hef"
        if not generated_hef.is_file() and Path(f"{run_name_stem}.hef").is_file():
            generated_hef = Path(f"{run_name_stem}.hef")
        if not generated_hef.is_file():
            logger.error("hailomz compile failed to produce HEF binary '%s.hef'", run_name_stem)
            return 1

        output_hef.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated_hef), str(output_hef))
        logger.info("Hailo HEF generated successfully: %s", output_hef)
        primary_artifact = output_hef

    else:
        # Torq NPU compilation
        logger.info("Normalizing ONNX coordinates and generating calibration dataset...")
        normalized_onnx = staging_dir / "best_normalized_output.onnx"
        calib_npy = staging_dir / "calibration_video_300_nhwc_float32.npy"

        prepare_for_onnx2tf.step2_normalize_output(
            onnx_in=str(staging_best_onnx),
            onnx_out=str(normalized_onnx),
            size=imgsz,
        )
        prepare_for_onnx2tf.step3_prepare_calibration(
            data_yaml_path=str(data_yaml),
            calib_out=str(calib_npy),
            size=imgsz,
            max_images=300,
            strategy="balanced",
        )
        prepare_for_onnx2tf.step4_generate_labels_json(
            data_yaml_path=str(data_yaml),
            labels_out=str(labels_file),
            tflite_path="",
            base_model_name=base_model_name,
            model_name=base_model_name,
            target=target_format,
            imgsz=imgsz,
        )

        logger.info("Executing Full INT8 Quantization via onnx2tf...")
        onnx2tf_dir = staging_dir / "onnx2tf_normalized_full_int8"
        if onnx2tf_dir.exists():
            shutil.rmtree(onnx2tf_dir)

        onnx2tf_cmd = [
            "onnx2tf",
            "-i", str(normalized_onnx),
            "-o", str(onnx2tf_dir),
            "-oiqt", "-iqd", "int8", "-oqd", "int8",
            "--tflite_backend", "tf_converter",
            "-cind", "images", str(calib_npy),
            "[[[[0.0,0.0,0.0]]]]", "[[[[1.0,1.0,1.0]]]]",
        ]
        run_command(onnx2tf_cmd, log_file=log_file_path, console=console_enabled)

        full_int8_tflite = onnx2tf_dir / "best_normalized_output_full_integer_quant.tflite"
        if not full_int8_tflite.is_file():
            logger.error("onnx2tf failed to generate full integer quantized TFLite at %s", full_int8_tflite)
            return 1

        # Update labels with exact TFLite quantization scales and zero points
        prepare_for_onnx2tf.step4_generate_labels_json(
            data_yaml_path=str(data_yaml),
            labels_out=str(labels_file),
            tflite_path=str(full_int8_tflite),
            base_model_name=base_model_name,
            model_name=base_model_name,
            target=target_format,
            imgsz=imgsz,
        )

        logger.info("Converting TFLite to TOSA MLIR...")
        tosa_mlir = staging_dir / f"{run_name_stem}_normalized_full_integer.tosa.mlir"
        tosa_cmd = [
            "tosa-converter-for-tflite",
            str(full_int8_tflite),
            "--text",
            "-o", str(tosa_mlir),
        ]
        run_command(tosa_cmd, log_file=log_file_path, console=console_enabled)

        logger.info("Compiling TOSA MLIR for Synaptics Astra SL2610 Torq NPU...")
        torq_cmd = [
            "torq-compile",
            "--torq-hw=SL2610",
            str(tosa_mlir),
            "-o", str(output_vmfb),
        ]
        run_command(torq_cmd, log_file=log_file_path, console=console_enabled)

        if not output_vmfb.is_file():
            logger.error("torq-compile failed to produce VMFB binary at %s", output_vmfb)
            return 1

        logger.info("Torq VMFB generated successfully: %s", output_vmfb)
        primary_artifact = output_vmfb

    # --------------------------------------------------------------------------
    # Create Non-Timestamped Soft Links in models/<target>/
    # --------------------------------------------------------------------------
    model_ext = primary_artifact.suffix
    model_symlink = models_target_dir / f"{target_format}_{run_name_stem}{model_ext}"
    if model_symlink.exists() or model_symlink.is_symlink():
        model_symlink.unlink()
    model_symlink.symlink_to(Path(timestamp) / primary_artifact.name)
    logger.info("Created soft link: %s -> %s/%s", model_symlink, timestamp, primary_artifact.name)

    labels_json_symlink = models_target_dir / f"{target_format}_{base_model_name}.json"
    if labels_json_symlink.exists() or labels_json_symlink.is_symlink():
        labels_json_symlink.unlink()
    labels_json_symlink.symlink_to(Path(timestamp) / labels_file.name)
    logger.info("Created soft link: %s -> %s/%s", labels_json_symlink, timestamp, labels_file.name)

    labels_txt_file = labels_file.with_suffix(".txt")
    labels_txt_symlink = models_target_dir / f"{target_format}_{base_model_name}.txt"
    if labels_txt_file.is_file():
        if labels_txt_symlink.exists() or labels_txt_symlink.is_symlink():
            labels_txt_symlink.unlink()
        labels_txt_symlink.symlink_to(Path(timestamp) / labels_txt_file.name)
        logger.info("Created soft link: %s -> %s/%s", labels_txt_symlink, timestamp, labels_txt_file.name)

    generic_labels_json = models_target_dir / "labels.json"
    if generic_labels_json.exists() or generic_labels_json.is_symlink():
        generic_labels_json.unlink()
    generic_labels_json.symlink_to(Path(timestamp) / labels_file.name)

    if labels_txt_file.is_file():
        generic_labels_txt = models_target_dir / "labels.txt"
        if generic_labels_txt.exists() or generic_labels_txt.is_symlink():
            generic_labels_txt.unlink()
        generic_labels_txt.symlink_to(Path(timestamp) / labels_txt_file.name)

    # --------------------------------------------------------------------------
    # Step 4: Optional Deployment
    # --------------------------------------------------------------------------
    if not args.skip_deploy and (deploy_host or deploy_dir):
        dep_cfg = cfg.get("deployment", {}) if isinstance(cfg.get("deployment"), dict) else {}
        restart_srv = bool(dep_cfg.get("restart_service", False))
        srv_name = dep_cfg.get("service_name", "coralboard-sl2619-vision")

        deploy_artifacts(
            model_path=primary_artifact,
            labels_path=labels_file,
            deploy_host=deploy_host,
            deploy_dir=deploy_dir,
            restart_service=restart_srv,
            service_name=srv_name,
            do_health_check=False,
            logger=logger,
        )

    # --------------------------------------------------------------------------
    # Summary Report
    # --------------------------------------------------------------------------
    print("\n==========================================================================")
    if target_format == "nvidia":
        print("  PIPELINE COMPLETE - ONNX MODEL READY FOR NVIDIA GPU INFERENCE!")
        print("==========================================================================")
        print("Artifacts Generated:")
        print(f"  1. ONNX Model      : {output_onnx} ({output_onnx.stat().st_size / (1024 * 1024):.1f}M)")
        print(f"     Soft Link       : {model_symlink}")
    elif target_format == "hailo":
        print("  PIPELINE COMPLETE - HEF MODEL READY FOR HAILO-10H NPU INFERENCE!")
        print("==========================================================================")
        print("Artifacts Generated:")
        print(f"  1. Hailo HEF Binary: {output_hef} ({output_hef.stat().st_size / (1024 * 1024):.1f}M)")
        print(f"     Soft Link       : {model_symlink}")
    else:
        print("  PIPELINE COMPLETE - MODEL READY FOR NPU INFERENCE!")
        print("==========================================================================")
        print("Artifacts Generated:")
        print(f"  1. Hardware Binary : {output_vmfb} ({output_vmfb.stat().st_size / (1024 * 1024):.1f}M)")
        print(f"     Soft Link       : {model_symlink}")

    print(f"  2. Label Metadata  : {labels_file}")
    print(f"     Soft Link       : {labels_json_symlink}")
    if labels_txt_file.is_file():
        print(f"  3. Frigate Labelmap: {labels_txt_file}")
        print(f"     Soft Link       : {labels_txt_symlink}")
        print(f"  4. PyTorch Weights : {weights_path}")
        if file_enabled and log_file_path:
            print(f"  5. Execution Logs  : {log_file_path.parent}")
    else:
        print(f"  3. PyTorch Weights : {weights_path}")
        if file_enabled and log_file_path:
            print(f"  4. Execution Logs  : {log_file_path.parent}")

    postprocess_elapsed = int(time.time() - postprocess_start_ts)
    print("--------------------------------------------------------------------------")
    print_timing_report(weights_path, preprocess_elapsed, training_elapsed, postprocess_elapsed)
    print("==========================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
