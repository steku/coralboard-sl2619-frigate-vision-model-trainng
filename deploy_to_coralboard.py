#!/usr/bin/env python3
"""
Deploy Compiled YOLO Model & Labels to Coralboard SL2619 Torq Vision Service.

Ensures BOTH the compiled .vmfb hardware binary and labels.json are distributed
together to avoid class index mismatches (e.g. COCO-80 fallback misclassifications).
"""

import argparse
import atexit
import datetime
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import config_utils


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


def detect_latest_model(models_dir: Path = Path("models/torq")) -> Optional[Path]:
    """Find the most recently modified .vmfb model."""
    if not models_dir.exists():
        return None

    candidates: List[Path] = []
    for pattern in ("torq_*.vmfb", "*.vmfb"):
        matched = list(models_dir.glob(pattern)) + list(models_dir.glob(f"*/{pattern}"))
        if matched:
            candidates.extend(matched)

    if not candidates:
        return None

    unique = {p.resolve(): p for p in candidates}
    sorted_candidates = sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return sorted_candidates[0]


def detect_matching_labels(model_path: Path, models_dir: Path = Path("models/torq")) -> Optional[Path]:
    """Find matching labels.json or labels file for a given model path."""
    model_stem = model_path.stem
    candidate_name = re.sub(r"^torq_", "", model_stem)
    candidate_base = re.sub(r"_\d{6}-\d{4}$", "", candidate_name)
    model_dir = model_path.parent

    check_paths = [
        model_dir / f"torq_{candidate_name}.json",
        models_dir / f"torq_{candidate_base}.json",
        models_dir / f"torq_{candidate_name}.json",
        models_dir / "labels.json",
    ]
    for p in check_paths:
        if p.is_file():
            return p

    # Fallback to latest json in models/torq
    all_json = list(models_dir.glob("*.json")) + list(models_dir.glob("*/*.json"))
    if all_json:
        unique = {p.resolve(): p for p in all_json}
        sorted_json = sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)
        return sorted_json[0]

    return None


def derive_symlink_name(artifact_path: Path, target_format: str = "torq") -> str:
    """Generate canonical non-timestamped filename (e.g., torq_yolov8n.vmfb)."""
    stem = artifact_path.stem
    stem_without_ts = re.sub(r"_\d{6}-\d{4}$", "", stem)
    ext = artifact_path.suffix
    return f"{stem_without_ts}{ext}"


def deploy_artifacts(
    model_path: Path,
    labels_path: Path,
    deploy_host: Optional[str] = None,
    deploy_dir: Optional[str] = None,
    restart_service: bool = False,
    service_name: str = "coralboard-sl2619-vision",
    do_health_check: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Deploy model and labels locally and/or via SCP to Coralboard."""
    log = logger or logging.getLogger("frigate")

    if not model_path.is_file():
        log.error("Compiled model (.vmfb) not found at: %s", model_path)
        return False

    if not labels_path.is_file():
        log.error("Matching labels file not found at: %s", labels_path)
        return False

    labels_txt = labels_path.with_suffix(".txt")
    has_labels_txt = labels_txt.is_file()

    model_size = f"{model_path.stat().st_size / (1024 * 1024):.1f}M"
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_lines = sum(1 for line in f if '"' in line)
    except Exception:
        labels_lines = 0

    log.info("Selected Model  : %s (%s)", model_path, model_size)
    log.info("Selected Labels : %s (%d lines)", labels_path, labels_lines)

    if not deploy_host and not deploy_dir:
        log.warning("No deployment target specified via flags or config.yaml!")
        print("\nTo deploy manually:")
        print(f"  Remote: scp {model_path} root@<CORALBOARD_IP>:/path/to/coralboard-sl2619-vision/models/{model_path.name}")
        print(f"          scp {labels_path} root@<CORALBOARD_IP>:/path/to/coralboard-sl2619-vision/models/{labels_path.name}")
        print("\nOr set 'deployment.host' in config.yaml and re-run.")
        return True

    artifact_basename = model_path.name
    model_symlink_basename = derive_symlink_name(model_path)
    labels_basename = labels_path.name
    labels_symlink_basename = derive_symlink_name(labels_path)
    labels_txt_basename = labels_txt.name if has_labels_txt else ""
    labels_txt_symlink_basename = derive_symlink_name(labels_txt) if has_labels_txt else ""

    # Local Directory Deployment
    if deploy_dir:
        dest_dir = Path(deploy_dir)
        log.info("Deploying to local directory: %s...", dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest_dir / artifact_basename)
        shutil.copy2(model_path, dest_dir / model_symlink_basename)
        shutil.copy2(labels_path, dest_dir / labels_basename)
        shutil.copy2(labels_path, dest_dir / labels_symlink_basename)
        shutil.copy2(labels_path, dest_dir / "labels.json")
        if has_labels_txt:
            shutil.copy2(labels_txt, dest_dir / labels_txt_basename)
            shutil.copy2(labels_txt, dest_dir / labels_txt_symlink_basename)
            shutil.copy2(labels_txt, dest_dir / "labels.txt")
        log.info(
            "Deployed %s and %s to %s/",
            artifact_basename,
            labels_basename,
            dest_dir,
        )

    # Remote SCP Deployment
    if deploy_host:
        clean_host = deploy_host.rstrip("/")
        log.info("Deploying via SCP to remote destination: %s...", clean_host)

        remote_targets = [
            (model_path, f"{clean_host}/{artifact_basename}"),
            (model_path, f"{clean_host}/{model_symlink_basename}"),
            (labels_path, f"{clean_host}/{labels_basename}"),
            (labels_path, f"{clean_host}/{labels_symlink_basename}"),
            (labels_path, f"{clean_host}/labels.json"),
        ]
        if has_labels_txt:
            remote_targets.extend([
                (labels_txt, f"{clean_host}/{labels_txt_basename}"),
                (labels_txt, f"{clean_host}/{labels_txt_symlink_basename}"),
                (labels_txt, f"{clean_host}/labels.txt"),
            ])

        for src, dst in remote_targets:
            res = subprocess.run(["scp", str(src), dst], check=False)
            if res.returncode != 0:
                log.error("Failed to SCP %s to %s", src, dst)
                return False

        log.info("Successfully uploaded %s and %s to %s!", artifact_basename, labels_basename, clean_host)

        remote_ssh = clean_host.split(":", 1)[0]
        remote_ip = remote_ssh.split("@")[-1]

        if restart_service:
            log.info("Restarting remote service '%s' on %s...", service_name, remote_ssh)
            res = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", remote_ssh, f"systemctl restart {service_name}"],
                check=False,
            )
            if res.returncode == 0:
                log.info("Service '%s' restarted successfully!", service_name)
            else:
                log.warning("Could not restart '%s' automatically via SSH. Please restart manually.", service_name)

        if do_health_check:
            log.info("Performing health check at http://%s:5000/health...", remote_ip)
            import time
            time.sleep(2)
            res = subprocess.run(
                ["curl", "-s", "--connect-timeout", "5", f"http://{remote_ip}:5000/health"],
                check=False,
            )
            if res.returncode != 0:
                log.warning("Health check endpoint did not respond.")
            print("")

    log.info("  DEPLOYMENT COMPLETE!")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy compiled YOLO model & labels to Coralboard Vision Service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_target = parser.add_argument_group("Deployment Targets")
    group_target.add_argument(
        "--host",
        dest="deploy_host",
        help="Remote destination for SCP (e.g. root@192.168.100.2:/root/coralboard-sl2619-vision/models)",
    )
    group_target.add_argument(
        "--dir",
        dest="deploy_dir",
        help="Local directory destination (e.g. ../coralboard-sl2619-vision/models)",
    )

    group_art = parser.add_argument_group("Artifact Options")
    group_art.add_argument(
        "--model",
        dest="model_path",
        help="Path to compiled .vmfb model (default: auto-detected)",
    )
    group_art.add_argument(
        "--labels",
        dest="labels_path",
        help="Path to labels JSON mapping (default: auto-detected)",
    )

    group_srv = parser.add_argument_group("Service Management")
    group_srv.add_argument(
        "--restart",
        dest="restart_service",
        action="store_true",
        help="Restart the remote vision service via SSH after deployment",
    )
    group_srv.add_argument(
        "--service-name",
        dest="service_name",
        default="coralboard-sl2619-vision",
        help="Name of systemd service on Coralboard (default: coralboard-sl2619-vision)",
    )
    group_srv.add_argument(
        "--check",
        dest="do_health_check",
        action="store_true",
        help="Perform HTTP health check on Coralboard after deployment",
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_file",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    config_utils.add_logging_args(parser)
    return parser


def main() -> int:
    ensure_terminal_onlcr()
    maybe_reexec_in_venv()

    parser = build_parser()
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config_file)
    dep_cfg = cfg.get("deployment", {}) if isinstance(cfg.get("deployment"), dict) else {}

    deploy_host = args.deploy_host or dep_cfg.get("host")
    deploy_dir = args.deploy_dir or dep_cfg.get("dir")
    restart_service = args.restart_service or bool(dep_cfg.get("restart_service", False))
    service_name = args.service_name or dep_cfg.get("service_name", "coralboard-sl2619-vision")

    log_settings = config_utils.get_logging_settings(cfg)
    console_enabled = args.log_console if args.log_console is not None else log_settings["console"]
    file_enabled = args.log_file if args.log_file is not None else log_settings["file"]
    timestamp = os.environ.get("FRIGATE_RUN_TIMESTAMP") or datetime.datetime.now().strftime("%y%m%d-%H%M")
    default_log_dir = Path("models/torq") / timestamp / "logs"
    log_dir = Path(args.log_dir) if args.log_dir else default_log_dir

    logger = config_utils.setup_logging_from_args(
        args,
        script_name="deploy_to_coralboard",
        target="torq",
    )

    model_path = Path(args.model_path) if args.model_path else detect_latest_model()
    if not model_path or not model_path.is_file():
        logger.error("Compiled model (.vmfb) not found! Please compile first or specify --model.")
        return 1

    labels_path = Path(args.labels_path) if args.labels_path else detect_matching_labels(model_path)
    if not labels_path or not labels_path.is_file():
        logger.error("Matching labels file not found! Specify --labels.")
        return 1

    success = deploy_artifacts(
        model_path=model_path,
        labels_path=labels_path,
        deploy_host=deploy_host,
        deploy_dir=deploy_dir,
        restart_service=restart_service,
        service_name=service_name,
        do_health_check=args.do_health_check,
        logger=logger,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
