#!/usr/bin/env python3
"""
Secure Web Interface for Frigate Model Training & Compilation Pipeline.

Features:
- Configurable port & host (--port, --host, or config.yaml web_ui settings).
- Zero external package dependencies (built with http.server.ThreadingHTTPServer).
- Security hardened: default bind to 127.0.0.1, strict parameter whitelisting,
  shell-free subprocess execution, path traversal guards, and security headers.
- Profile loading & full CLI parameter parity.
- Dynamic CVAT integration: connects to CVAT server, lists tasks & jobs,
  and allows task selection via interactive checkboxes.
- Live real-time log streaming via Server-Sent Events (SSE) in a scrollable console.
- Visual pipeline stage stepper tracking progress & completion status.
- Persistent run history record of past runs, statuses, and generated artifacts.
- Leaves CLI (train.py and train.sh) 100% operational and independent.
"""

import argparse
import atexit
from collections import deque
import datetime
import html
import http.server
import io
import json
import logging
import os
from pathlib import Path
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse

import config_utils

PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_FILE = PROJECT_ROOT / ".web_ui_history.json"
MAX_LOG_LINES = 10000
MAX_HISTORY_RUNS = 25
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

PIPELINE_STAGES = [
    {"id": "export", "name": "Step 1: CVAT Export"},
    {"id": "pretrain", "name": "Step 2: Pretraining"},
    {"id": "train", "name": "Step 3: Model Training"},
    {"id": "compile", "name": "Step 4: Compilation & Quantization"},
    {"id": "deploy", "name": "Step 5: Target Deployment"},
]


class RunManager:
    """Thread-safe manager for tracking training runs, streaming logs, and maintaining run history."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.active_run: Optional[Dict[str, Any]] = None
        self.log_buffer: deque = deque(maxlen=MAX_LOG_LINES)
        self.subscribers: Set[queue.Queue] = set()
        self.history: List[Dict[str, Any]] = self._load_history()
        self.logs_dir = self.project_root / ".web_ui_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = None
        self._pending_log_file_line: Optional[str] = None
        self.queue: List[Dict[str, Any]] = []
        self._queue_counter: int = 1

    def _load_history(self) -> List[Dict[str, Any]]:
        if HISTORY_FILE.is_file():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-MAX_HISTORY_RUNS:]
            except Exception:
                pass
        return []

    def _save_history(self) -> None:
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[-MAX_HISTORY_RUNS:], f, indent=2)
        except Exception:
            pass

    def finalize_pending_log(self) -> None:
        with self.lock:
            if self._pending_log_file_line and self.current_log_file:
                try:
                    self.current_log_file.write(self._pending_log_file_line + "\n")
                    self.current_log_file.flush()
                except Exception:
                    pass
                self._pending_log_file_line = None

    def add_log_line(self, line: str, replace: bool = False, pending: bool = False) -> None:
        clean = ANSI_ESCAPE_RE.sub("", line).rstrip("\r\n")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with self.lock:
            if replace and self.log_buffer:
                m = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", self.log_buffer[-1])
                if m:
                    timestamp = m.group(1)
                formatted = f"[{timestamp}] {clean}" if clean else ""
                self.log_buffer[-1] = formatted
            else:
                formatted = f"[{timestamp}] {clean}" if clean else ""
                self.log_buffer.append(formatted)

            if pending:
                self._pending_log_file_line = formatted
            else:
                if not replace and self._pending_log_file_line and self.current_log_file:
                    try:
                        self.current_log_file.write(self._pending_log_file_line + "\n")
                        self.current_log_file.flush()
                    except Exception:
                        pass
                self._pending_log_file_line = None

                if self.current_log_file:
                    try:
                        self.current_log_file.write(formatted + "\n")
                        self.current_log_file.flush()
                    except Exception:
                        pass

            self._update_stages_from_log(clean)
            subscribers_snapshot = list(self.subscribers)

        for q in subscribers_snapshot:
            try:
                q.put_nowait((formatted, replace))
            except queue.Full:
                pass

    def _update_stages_from_log(self, log_line: str) -> None:
        if not self.active_run:
            return
        stages = self.active_run.get("stages", {})
        low = log_line.lower()

        if "step 1: exporting cvat dataset" in low:
            stages["export"] = "running"
        elif "step 2: training" in low:
            if stages.get("export") == "running":
                stages["export"] = "completed"
            stages["pretrain"] = "running"
        elif "launching yolo fine-tuning" in low or "executing: yolo detect train" in low:
            if stages.get("export") == "running":
                stages["export"] = "completed"
            if stages.get("pretrain") == "running" or stages.get("pretrain") == "pending":
                stages["pretrain"] = "completed"
            stages["train"] = "running"
        elif "step 3:" in low and ("export" in low or "compilation" in low):
            if stages.get("pretrain") == "running":
                stages["pretrain"] = "completed"
            if stages.get("train") == "running":
                stages["train"] = "completed"
            stages["compile"] = "running"
        elif "step 4:" in low or "deploying artifacts" in low:
            if stages.get("compile") == "running":
                stages["compile"] = "completed"
            stages["deploy"] = "running"
        elif "pipeline complete" in low or "model ready for" in low:
            for s in ("export", "pretrain", "train", "compile"):
                if stages.get(s) == "running":
                    stages[s] = "completed"
            if stages.get("deploy") == "running":
                stages["deploy"] = "completed"

    def subscribe(self) -> Tuple[queue.Queue, List[str]]:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self.lock:
            history_lines = list(self.log_buffer)
            self.subscribers.add(q)
        return q, history_lines

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subscribers.discard(q)

    def is_running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            queue_snapshot = [
                {
                    "queue_id": item["queue_id"],
                    "profile": item["profile"],
                    "options": item["options"],
                    "queued_at_iso": item.get("queued_at_iso", ""),
                }
                for item in self.queue
            ]
            if not self.active_run:
                return {
                    "running": False,
                    "status": "idle",
                    "history": self.history,
                    "stages": {s["id"]: "pending" for s in PIPELINE_STAGES},
                    "queue": queue_snapshot,
                }

            is_alive = self.process is not None and self.process.poll() is None
            elapsed = int(time.time() - self.active_run["start_time"])
            return {
                "running": is_alive,
                "status": self.active_run.get("status", "running" if is_alive else "idle"),
                "run_id": self.active_run.get("run_id"),
                "profile": self.active_run.get("profile"),
                "start_time_iso": self.active_run.get("start_time_iso"),
                "elapsed_sec": elapsed,
                "stages": self.active_run.get("stages", {}),
                "artifacts": self.active_run.get("artifacts", []),
                "options": self.active_run.get("options", {}),
                "history": self.history,
                "queue": queue_snapshot,
            }

    def get_queued_runs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {
                    "queue_id": item["queue_id"],
                    "profile": item["profile"],
                    "options": item["options"],
                    "cmd_args": item.get("cmd_args", []),
                    "queued_at_iso": item.get("queued_at_iso", ""),
                }
                for item in self.queue
            ]

    def queue_run(
        self,
        cmd_args: List[str],
        profile_name: str,
        options_summary: Dict[str, Any],
        queue_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        with self.lock:
            is_active = self.process is not None and self.process.poll() is None

            # If editing an existing queue item:
            if queue_id:
                for item in self.queue:
                    if item.get("queue_id") == queue_id:
                        item["profile"] = profile_name
                        item["cmd_args"] = cmd_args
                        item["options"] = options_summary
                        item["updated_at_iso"] = datetime.datetime.now().isoformat()
                        return True, "updated", item
                item = {
                    "queue_id": queue_id,
                    "profile": profile_name,
                    "cmd_args": cmd_args,
                    "options": options_summary,
                    "queued_at_iso": datetime.datetime.now().isoformat(),
                }
                self.queue.append(item)
                return True, "queued", item

            # If idle and queue empty, start immediately below
            if not is_active and not self.queue:
                pass
            else:
                qid = f"q-{self._queue_counter}"
                self._queue_counter += 1
                item = {
                    "queue_id": qid,
                    "profile": profile_name,
                    "cmd_args": cmd_args,
                    "options": options_summary,
                    "queued_at_iso": datetime.datetime.now().isoformat(),
                }
                self.queue.append(item)
                return True, "queued", item

        # Start immediately if idle
        started, run_id = self.start_run(cmd_args, profile_name, options_summary)
        if started:
            return True, "started", {"run_id": run_id}
        else:
            return False, run_id, {}

    def cancel_queued_run(self, queue_id: Optional[str] = None) -> Tuple[bool, str]:
        with self.lock:
            if not self.queue:
                return False, "No queued runs found."
            if queue_id:
                initial_len = len(self.queue)
                self.queue = [item for item in self.queue if item.get("queue_id") != queue_id]
                if len(self.queue) < initial_len:
                    return True, f"Queued run '{queue_id}' cancelled."
                return False, f"Queued run '{queue_id}' not found."
            else:
                removed = self.queue.pop(0)
                return True, f"Queued run '{removed.get('queue_id')}' cancelled."

    def start_next_queued_run(self) -> Tuple[bool, str]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "A training run is currently active. The queued run will start automatically when it finishes."
            if not self.queue:
                return False, "No queued runs available."
            next_run = self.queue.pop(0)

        return self.start_run(next_run["cmd_args"], next_run["profile"], next_run["options"])

    def start_run(self, cmd_args: List[str], profile_name: str, options_summary: Dict[str, Any]) -> Tuple[bool, str]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "A training run is already in progress. Please wait for it to complete or abort it."

            run_id = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            self.log_buffer.clear()
            self._pending_log_file_line = None
            subscribers_snapshot = list(self.subscribers)

        for q in subscribers_snapshot:
            try:
                q.put_nowait(("", "clear"))
            except queue.Full:
                pass

        with self.lock:
            stages = {}
            for s in PIPELINE_STAGES:
                stages[s["id"]] = "pending"
            if not options_summary.get("do_export"):
                stages["export"] = "skipped"
            if options_summary.get("skip_compile"):
                stages["compile"] = "skipped"
            if options_summary.get("skip_deploy"):
                stages["deploy"] = "skipped"
            if options_summary.get("compile_only") or options_summary.get("pull_only"):
                stages["export"] = "skipped"
                stages["pretrain"] = "skipped"
                stages["train"] = "skipped"
                stages["compile"] = "pending"

            self.active_run = {
                "run_id": run_id,
                "profile": profile_name,
                "start_time": time.time(),
                "start_time_iso": datetime.datetime.now().isoformat(),
                "status": "running",
                "options": options_summary,
                "stages": stages,
                "artifacts": [],
                "cmd": cmd_args,
            }

            train_py = self.project_root / "train.py"
            python_bin = sys.executable
            full_cmd = [python_bin, str(train_py)] + cmd_args

            try:
                self.current_log_file = open(self.logs_dir / f"{run_id}.log", "w", encoding="utf-8")
            except Exception:
                self.current_log_file = None

            proc_env = os.environ.copy()
            proc_env["PYTHONUNBUFFERED"] = "1"
            current_user = proc_env.get("USER") or proc_env.get("USERNAME") or proc_env.get("LOGNAME") or "root"
            proc_env["USER"] = current_user
            proc_env["USERNAME"] = current_user
            proc_env["LOGNAME"] = current_user
            proc_env.setdefault("HOME", "/root")

            try:
                self.process = subprocess.Popen(
                    full_cmd,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=proc_env,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
            except Exception as e:
                if self.current_log_file:
                    try:
                        self.current_log_file.close()
                    except Exception:
                        pass
                    self.current_log_file = None
                self.active_run["status"] = "failed"
                return False, f"Failed to start training process: {e}"

            threading.Thread(target=self._monitor_process, args=(self.process, self.active_run), daemon=True).start()
            return True, run_id

    def _monitor_process(self, proc: subprocess.Popen, run_info: Dict[str, Any]) -> None:
        in_place_active = False
        try:
            if proc.stdout:
                reader = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
                for raw_line in iter(reader.readline, ""):
                    if not raw_line:
                        break
                    is_cr = raw_line.endswith("\r") and not raw_line.endswith("\r\n")
                    is_lf = raw_line.endswith("\n")
                    clean = ANSI_ESCAPE_RE.sub("", raw_line).rstrip("\r\n")

                    if not clean:
                        if is_lf:
                            if in_place_active:
                                self.finalize_pending_log()
                                in_place_active = False
                            else:
                                self.add_log_line("", replace=False)
                        # Bare carriage returns (\r, \r\x1b[K) only reset cursor position; do not replace or commit
                        continue

                    # Determine if this line should replace an existing in-progress line or start anew
                    should_replace = in_place_active
                    if should_replace:
                        # Headers, banners, explicit step logs never replace an existing line
                        if clean.startswith(("[", "==", "Profile:", "Target:", "Base Model:")) or "GPU_mem" in clean:
                            should_replace = False
                            self.finalize_pending_log()
                            in_place_active = False
                        elif self.log_buffer:
                            # If epoch numbers differ, commit the old epoch and start on a new line
                            prev_clean = re.sub(r"^\[\d\d:\d\d:\d\d\]\s*", "", self.log_buffer[-1]).strip()
                            clean_strip = clean.strip()
                            ep_prev = None
                            ep_curr = None
                            m_p = re.search(r"\bEpoch\s+(\d+)\b", prev_clean, re.I) or re.match(r"^(\d+)/\d+\b", prev_clean)
                            if m_p:
                                ep_prev = m_p.group(1)
                            m_c = re.search(r"\bEpoch\s+(\d+)\b", clean_strip, re.I) or re.match(r"^(\d+)/\d+\b", clean_strip)
                            if m_c:
                                ep_curr = m_c.group(1)
                            if ep_prev and ep_curr and ep_prev != ep_curr:
                                should_replace = False
                                self.finalize_pending_log()
                                in_place_active = False

                    if should_replace:
                        self.add_log_line(clean, replace=True, pending=is_cr)
                    else:
                        self.add_log_line(clean, replace=False, pending=is_cr)

                    if is_cr:
                        in_place_active = True
                    else:
                        self.finalize_pending_log()
                        in_place_active = False
        except Exception:
            pass
        finally:
            self.finalize_pending_log()
            with self.lock:
                if self.current_log_file:
                    try:
                        self.current_log_file.close()
                    except Exception:
                        pass
                    self.current_log_file = None

        return_code = proc.wait()
        end_time = time.time()
        elapsed = int(end_time - run_info["start_time"])

        next_run_to_start = None
        with self.lock:
            is_aborted = (run_info.get("status") == "aborting") or (
                return_code in (-signal.SIGINT, -signal.SIGTERM, getattr(signal, "SIGKILL", 9), -getattr(signal, "SIGKILL", 9), 130, 143, 137)
            )
            final_status = "aborted" if is_aborted else ("completed" if return_code == 0 else "failed")
            run_info["status"] = final_status
            run_info["return_code"] = return_code
            run_info["elapsed_sec"] = elapsed
            run_info["end_time_iso"] = datetime.datetime.now().isoformat()

            stages = run_info.get("stages", {})
            for sid, sval in stages.items():
                if sval == "running":
                    stages[sid] = "completed" if final_status == "completed" else "failed"

            artifacts = self._discover_artifacts()
            run_info["artifacts"] = artifacts

            self.history.append({
                "run_id": run_info["run_id"],
                "profile": run_info.get("profile", "unknown"),
                "status": final_status,
                "start_time_iso": run_info.get("start_time_iso", ""),
                "elapsed_sec": elapsed,
                "stages": dict(stages),
                "artifacts": artifacts,
                "return_code": return_code,
            })
            self._save_history()

            if final_status != "aborted" and self.queue:
                next_run_to_start = self.queue.pop(0)

        if next_run_to_start:
            self.add_log_line(
                f"[QUEUE] Active run finished ({final_status}). Auto-starting next queued run: profile '{next_run_to_start['profile']}' (Queue ID: {next_run_to_start['queue_id']})..."
            )
            self.start_run(next_run_to_start["cmd_args"], next_run_to_start["profile"], next_run_to_start["options"])
        elif final_status == "aborted" and self.queue:
            self.add_log_line("[QUEUE] Active run was aborted. Queued runs held in queue.")

    def _discover_artifacts(self) -> List[Dict[str, str]]:
        found = []
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            return found

        for ext in ("*.vmfb", "*.onnx", "*.hef", "*.json"):
            for p in models_dir.glob(f"*/*/{ext}"):
                if p.is_file() and (time.time() - p.stat().st_mtime) < 600:
                    found.append({
                        "name": p.name,
                        "path": str(p.relative_to(self.project_root)),
                        "size": f"{p.stat().st_size / (1024 * 1024):.2f} MB" if p.stat().st_size > 1024 * 1024 else f"{p.stat().st_size / 1024:.1f} KB",
                    })
        return found

    def abort_run(self) -> Tuple[bool, str]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return False, "No training run is currently active."

            proc = self.process
            if self.active_run:
                self.active_run["status"] = "aborting"

        try:
            pgid = None
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    p_pgid = os.getpgid(proc.pid)
                    if p_pgid == proc.pid:
                        pgid = p_pgid
                except Exception:
                    pass

            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            time.sleep(0.5)
            if proc.poll() is None:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            self.add_log_line("[WEB_UI] Training run was aborted by user. Queued runs held in queue.")
            return True, "Training run stopped successfully."
        except Exception as e:
            return False, f"Error stopping process: {e}"

    def get_run_logs(self, run_id: str) -> List[str]:
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
        if not clean_id:
            return []
        with self.lock:
            if self.active_run and self.active_run.get("run_id") == clean_id:
                return list(self.log_buffer)

        log_path = self.logs_dir / f"{clean_id}.log"
        if log_path.is_file():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    return [line.rstrip("\r\n") for line in f]
            except Exception:
                pass
        return [f"[INFO] Detailed logs not available on disk for run {clean_id}."]


def sanitize_input(value: Any) -> str:
    """Strip dangerous characters to prevent injection."""
    if value is None:
        return ""
    val_str = str(value).strip()
    return re.sub(r"[;&|`$<>]", "", val_str)


def fetch_cvat_tasks(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Connect to CVAT server and retrieve list of tasks and jobs."""
    cfg = config_utils.load_config(config_path)
    url, user, password = config_utils.get_cvat_credentials(cfg)
    if not user or not password:
        return {"error": "CVAT credentials not found in config.yaml or environment variables.", "tasks": [], "url": url}

    try:
        from cvat_sdk import make_client
        with make_client(url, credentials=(user, password)) as client:
            raw_tasks = client.tasks.list()
            tasks = []
            for t in sorted(raw_tasks, key=lambda x: x.id, reverse=True):
                job_info = []
                try:
                    jobs = client.jobs.list(task_id=t.id)
                    for j in jobs:
                        job_info.append({
                            "id": j.id,
                            "status": getattr(j, "status", ""),
                            "stage": getattr(j, "stage", ""),
                        })
                except Exception:
                    pass

                tasks.append({
                    "id": t.id,
                    "name": getattr(t, "name", f"Task {t.id}"),
                    "status": getattr(t, "status", "unknown"),
                    "size": getattr(t, "size", 0),
                    "jobs": job_info,
                })
            return {"tasks": tasks, "url": url}
    except Exception as e:
        return {"error": f"Failed to connect to CVAT at {url}: {e}", "tasks": [], "url": url}


def validate_profile_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize fields for creating or updating a profile in config.yaml."""
    if not isinstance(data, dict):
        raise ValueError("Profile payload must be a JSON object.")

    target = sanitize_input(data.get("target", "torq")).lower()
    if target not in ("torq", "hailo", "nvidia"):
        raise ValueError("Target must be one of 'torq', 'hailo', or 'nvidia'.")

    base_model = sanitize_input(data.get("base_model", ""))
    if not base_model:
        raise ValueError("base_model is required.")

    remote = bool(data.get("remote", False))
    host = sanitize_input(data.get("host", ""))
    remote_dir = sanitize_input(data.get("remote_dir", "~/frigate-model-training"))
    device = sanitize_input(data.get("device", "0")) or "0"

    try:
        batch = int(data.get("batch", 16))
        if batch <= 0:
            raise ValueError()
    except Exception:
        raise ValueError("batch must be a positive integer.")

    try:
        epochs = int(data.get("epochs", 80))
        if epochs <= 0:
            raise ValueError()
    except Exception:
        raise ValueError("epochs must be a positive integer.")

    try:
        patience = int(data.get("patience", 30))
        if patience <= 0:
            raise ValueError()
    except Exception:
        raise ValueError("patience must be a positive integer.")

    try:
        workers = int(data.get("workers", 8))
        if workers < 0:
            raise ValueError()
    except Exception:
        raise ValueError("workers must be a non-negative integer.")

    cache = sanitize_input(data.get("cache", "ram")).lower()
    if cache not in ("ram", "disk", "false"):
        cache = "ram"

    optimizer = sanitize_input(data.get("optimizer", "AdamW"))
    if optimizer not in ("AdamW", "SGD", "Adam", "auto"):
        optimizer = "AdamW"

    try:
        lr0 = float(data.get("lr0", 0.0005))
        if lr0 <= 0:
            raise ValueError()
    except Exception:
        raise ValueError("lr0 must be a positive float.")

    try:
        lrf = float(data.get("lrf", 0.01))
        if lrf <= 0:
            raise ValueError()
    except Exception:
        raise ValueError("lrf must be a positive float.")

    cos_lr = bool(data.get("cos_lr", True))

    try:
        warmup_epochs = float(data.get("warmup_epochs", 5.0))
        if warmup_epochs < 0:
            raise ValueError()
    except Exception:
        warmup_epochs = 5.0

    try:
        warmup_momentum = float(data.get("warmup_momentum", 0.5))
        if warmup_momentum < 0:
            raise ValueError()
    except Exception:
        warmup_momentum = 0.5

    try:
        warmup_bias_lr = float(data.get("warmup_bias_lr", 0.0))
        if warmup_bias_lr < 0:
            raise ValueError()
    except Exception:
        warmup_bias_lr = 0.0

    profile_dict: Dict[str, Any] = {
        "target": target,
        "remote": remote,
        "device": device,
        "batch": batch,
        "workers": workers,
        "cache": cache,
        "epochs": epochs,
        "patience": patience,
        "base_model": base_model,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
    }

    if remote:
        profile_dict["host"] = host
        profile_dict["remote_dir"] = remote_dir
        profile_dict["clean_images"] = bool(data.get("clean_images", True))

    if target == "hailo":
        hw_arch = sanitize_input(data.get("hw_arch", "hailo10h"))
        profile_dict["hw_arch"] = hw_arch or "hailo10h"
        try:
            imgsz = int(data.get("imgsz", 640))
            if imgsz > 0:
                profile_dict["imgsz"] = imgsz
        except Exception:
            profile_dict["imgsz"] = 640
    elif data.get("imgsz") is not None and str(data.get("imgsz")).strip() != "":
        try:
            imgsz = int(data.get("imgsz"))
            if imgsz > 0:
                profile_dict["imgsz"] = imgsz
        except Exception:
            pass

    # Deployment
    raw_dep = data.get("deployment")
    if isinstance(raw_dep, dict):
        d_host = sanitize_input(raw_dep.get("host")) or None
        d_dir = sanitize_input(raw_dep.get("dir")) or None
        profile_dict["deployment"] = {"host": d_host, "dir": d_dir}

    # Logging
    raw_log = data.get("logging")
    if isinstance(raw_log, dict):
        profile_dict["logging"] = {
            "console": bool(raw_log.get("console", True)),
            "file": bool(raw_log.get("file", True)),
        }

    return profile_dict


def build_validated_train_args(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Validate JSON payload against a strict whitelist of CLI options."""
    args: List[str] = []
    summary: Dict[str, Any] = {}

    profile = sanitize_input(payload.get("profile", ""))
    if profile:
        args.extend(["-p", profile])
        summary["profile"] = profile

    workflow = payload.get("workflow", "full")
    if workflow == "compile_only":
        custom_ckpt = sanitize_input(payload.get("custom_checkpoint", ""))
        if custom_ckpt:
            if not (PROJECT_ROOT / custom_ckpt).is_file() and not Path(custom_ckpt).is_file():
                raise ValueError(f"Specified checkpoint '{custom_ckpt}' does not exist.")
            args.extend(["--compile", custom_ckpt])
        else:
            args.append("--compile")
        summary["compile_only"] = True
    elif workflow == "skip_compile":
        args.append("--skip-compile")
        summary["skip_compile"] = True
    elif workflow == "pull_only":
        args.append("--pull")
        summary["pull_only"] = True

    if payload.get("skip_deploy"):
        args.append("--skip-deploy")
        summary["skip_deploy"] = True

    # CVAT Export & Task Selection
    if payload.get("do_export"):
        raw_export_tasks = payload.get("export_tasks")
        args.append("--export")
        tasks_list = []
        if isinstance(raw_export_tasks, list):
            for tid in raw_export_tasks:
                clean_tid = re.sub(r"[^a-zA-Z0-9_-]", "", str(tid).strip())
                if clean_tid:
                    args.append(clean_tid)
                    tasks_list.append(clean_tid)
        elif isinstance(raw_export_tasks, str) and raw_export_tasks.strip():
            for tid in raw_export_tasks.replace(",", " ").split():
                clean_tid = re.sub(r"[^a-zA-Z0-9_-]", "", tid.strip())
                if clean_tid:
                    args.append(clean_tid)
                    tasks_list.append(clean_tid)
        summary["do_export"] = True
        summary["export_tasks"] = tasks_list

    if payload.get("replace_dataset"):
        args.append("--replace")
        summary["replace"] = True

    if payload.get("resume"):
        args.append("--resume")
        summary["resume"] = True

    base_model = sanitize_input(payload.get("base_model", ""))
    if base_model:
        args.extend(["-m", base_model])
        summary["base_model"] = base_model

    # Integer parameters
    for int_field, flag in (
        ("epochs", "--epochs"),
        ("batch", "--batch"),
        ("imgsz", "--imgsz"),
        ("patience", "--patience"),
        ("workers", "--workers"),
    ):
        if payload.get(int_field) is not None and str(payload.get(int_field)).strip() != "":
            try:
                val = int(payload[int_field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid integer value for {int_field}.")
            if val <= 0:
                raise ValueError(f"{int_field} must be a positive integer.")
            args.extend([flag, str(val)])
            summary[int_field] = val

    # Float parameters
    for float_field, flag in (("lr0", "--lr0"), ("lrf", "--lrf")):
        if payload.get(float_field) is not None and str(payload.get(float_field)).strip() != "":
            try:
                fval = float(payload[float_field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid float value for {float_field}.")
            if fval <= 0:
                raise ValueError(f"{float_field} must be a positive number.")
            args.extend([flag, str(fval)])
            summary[float_field] = fval

    device = sanitize_input(payload.get("device", ""))
    if device:
        args.extend(["--device", device])
        summary["device"] = device

    cache = sanitize_input(payload.get("cache", ""))
    if cache in ("ram", "disk", "false"):
        args.extend(["--cache", cache])
        summary["cache"] = cache

    optimizer = sanitize_input(payload.get("optimizer", ""))
    if optimizer in ("AdamW", "SGD", "Adam", "auto"):
        args.extend(["--optimizer", optimizer])
        summary["optimizer"] = optimizer

    spark_host = sanitize_input(payload.get("spark_host", ""))
    if spark_host:
        args.extend(["--host", spark_host])
        summary["spark_host"] = spark_host

    coco = sanitize_input(payload.get("coco_dataset", ""))
    if coco in ("coco128", "coco-val2017"):
        args.extend(["--coco-dataset", coco])
        summary["coco_dataset"] = coco

    if payload.get("frigate_plus") is True:
        args.append("--frigate-plus")
        summary["frigate_plus"] = True
    elif payload.get("frigate_plus") is False:
        args.append("--no-frigate-plus")
        summary["frigate_plus"] = False

    if payload.get("clean_images") is True:
        args.append("--clean-images")
        summary["clean_images"] = True
    elif payload.get("clean_images") is False:
        args.append("--no-clean-images")
        summary["clean_images"] = False

    deploy_target = sanitize_input(payload.get("deploy_target", ""))
    if deploy_target:
        args.extend(["--deploy", deploy_target])
        summary["deploy_target"] = deploy_target

    extra_args = sanitize_input(payload.get("extra_args", ""))
    if extra_args:
        args.extend(["--extra-args", extra_args])
        summary["extra_args"] = extra_args

    # Logging options
    if payload.get("log_console") is True:
        args.append("--log-console")
    elif payload.get("log_console") is False:
        args.append("--no-log-console")

    if payload.get("log_file") is True:
        args.append("--log-file")
    elif payload.get("log_file") is False:
        args.append("--no-log-file")

    log_dir = sanitize_input(payload.get("log_dir", ""))
    if log_dir:
        args.extend(["--log-dir", log_dir])

    return args, summary


def get_html_content() -> str:
    """Return self-contained HTML/CSS/JS frontend application."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Frigate Model Training Studio</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --success: #22c55e;
      --danger: #ef4444;
      --warning: #f59e0b;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 1.5rem;
      overflow-x: hidden;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--card-border);
    }
    h1 { font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; color: var(--primary); }
    .nav-tabs {
      display: flex;
      gap: 0.4rem;
      background: #090d16;
      padding: 0.25rem;
      border-radius: 0.5rem;
      border: 1px solid var(--card-border);
    }
    .nav-tab {
      background: transparent;
      color: var(--text-muted);
      border: none;
      padding: 0.4rem 0.85rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .nav-tab:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.05);
    }
    .nav-tab.active {
      background: var(--card-bg);
      color: var(--primary);
      font-weight: 600;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    .badge {
      display: inline-block;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-idle { background: #334155; color: #cbd5e1; }
    .badge-running { background: rgba(56, 189, 248, 0.2); color: var(--primary); border: 1px solid var(--primary); }
    .badge-completed { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
    .badge-failed { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .badge-aborted { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
    .badge-queued { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }

    .grid { display: grid; grid-template-columns: 460px minmax(0, 1fr); gap: 1.5rem; }
    @media (max-width: 1080px) { .grid { grid-template-columns: minmax(0, 1fr); } }
    .grid > div { min-width: 0; max-width: 100%; }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
      min-width: 0;
      max-width: 100%;
    }
    .card h2 { font-size: 1.05rem; margin-bottom: 0.85rem; color: #e2e8f0; display: flex; align-items: center; justify-content: space-between; }

    .form-group { margin-bottom: 0.85rem; }
    label { display: block; font-size: 0.825rem; font-weight: 500; color: var(--text-muted); margin-bottom: 0.3rem; }
    input[type="text"], input[type="number"], select {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 0.5rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
    }
    input:focus, select:focus { outline: 2px solid var(--primary); border-color: transparent; }
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
    .form-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; }
    .form-row-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; }
    @media (max-width: 768px) {
      .form-row, .form-row-3, .form-row-4 { grid-template-columns: 1fr; }
    }
    .checkbox-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.4rem; }
    .checkbox-row input { width: auto; }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.6rem 1rem;
      font-weight: 600;
      font-size: 0.875rem;
      border-radius: 0.375rem;
      border: none;
      cursor: pointer;
      transition: background-color 0.15s ease;
      width: 100%;
    }
    .btn-sm {
      padding: 0.3rem 0.6rem;
      font-size: 0.75rem;
      width: auto;
      background: #334155;
      color: #e2e8f0;
    }
    .btn-sm:hover { background: #475569; }
    .btn-primary { background: var(--primary); color: #0f172a; }
    .btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
    .btn-danger { background: var(--danger); color: white; }
    .btn-danger:hover:not(:disabled) { background: #dc2626; }
    .btn-success { background: var(--success); color: #022c22; font-weight: 700; }
    .btn-success:hover:not(:disabled) { background: #16a34a; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* CVAT task checklist box */
    .cvat-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 0.375rem;
      max-height: 190px;
      overflow-y: auto;
      padding: 0.5rem;
      margin: 0.5rem 0;
      font-size: 0.825rem;
    }
    .cvat-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.4rem;
      border-bottom: 1px solid rgba(51, 65, 85, 0.4);
    }
    .cvat-item:last-child { border-bottom: none; }
    .cvat-item label { margin: 0; color: #e2e8f0; cursor: pointer; flex-grow: 1; }
    .cvat-meta { font-size: 0.75rem; color: var(--text-muted); }

    /* Stepper */
    .stepper { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; margin-bottom: 1rem; }
    .step-item {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      padding: 0.75rem;
      font-size: 0.8rem;
      text-align: center;
      font-weight: 500;
    }
    .step-pending { color: var(--text-muted); opacity: 0.6; }
    .step-running { border-color: var(--primary); color: var(--primary); animation: pulse 2s infinite; }
    .step-completed { border-color: var(--success); color: var(--success); }
    .step-failed { border-color: var(--danger); color: var(--danger); }
    .step-skipped { color: #64748b; text-decoration: line-through; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* Terminal Console */
    .console-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .console-box {
      background: #090d16;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      min-height: 160px;
      height: 440px;
      max-height: 85vh;
      overflow-y: auto;
      overflow-x: auto;
      resize: vertical;
      padding: 0.75rem 1rem;
      font-family: var(--font-mono);
      font-size: 0.825rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      max-width: 100%;
      box-sizing: border-box;
      color: #e2e8f0;
    }
    .console-box.nowrap {
      white-space: pre !important;
      word-break: normal !important;
      overflow-wrap: normal !important;
      overflow-x: auto !important;
    }
    .console-box p { margin: 0.15rem 0; line-height: 1.35; }

    /* History table & scroll container */
    .history-scroll-container {
      max-height: 250px;
      overflow-y: auto;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
    }
    .history-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .history-table th, .history-table td { text-align: left; padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--card-border); }
    .history-table thead th {
      position: sticky;
      top: 0;
      background: var(--card-bg);
      z-index: 2;
      border-bottom: 2px solid var(--card-border);
      color: var(--text-muted);
      font-weight: 600;
    }
    .history-row {
      cursor: pointer;
      transition: background-color 0.15s ease;
    }
    .history-row:hover {
      background-color: rgba(56, 189, 248, 0.08);
    }
    .history-row.active-history-row {
      background-color: rgba(56, 189, 248, 0.18) !important;
    }

    /* Profiles Table */
    .profiles-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .profiles-table th, .profiles-table td { text-align: left; padding: 0.65rem 0.75rem; border-bottom: 1px solid var(--card-border); }
    .profiles-table thead th { color: var(--text-muted); font-weight: 600; background: #090d16; }
    .default-star { color: #f59e0b; margin-left: 0.35rem; font-size: 0.9rem; }

    /* Header & Navigation Layout */
    .header-container { display: flex; align-items: center; justify-content: space-between; width: 100%; flex-wrap: wrap; gap: 0.75rem; }
    .header-left { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
    .header-right { display: flex; align-items: center; gap: 0.5rem; }
    .form-actions { display: flex; gap: 0.75rem; margin-top: 1.5rem; }

    /* Responsive Styles - Tablets and Mobile Phones */
    @media (max-width: 768px) {
      body { padding: 0.75rem; }
      header { margin-bottom: 1rem; padding-bottom: 0.75rem; }
      .grid { gap: 1rem; }
      .card { padding: 1rem; margin-bottom: 1rem; }
      .stepper { grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
      .step-item { padding: 0.5rem 0.25rem; font-size: 0.75rem; }
      .history-scroll-container, .profiles-table-container { -webkit-overflow-scrolling: touch; }
      .history-table { min-width: 520px; }
      .profiles-table { min-width: 620px; }
    }

    @media (max-width: 640px) {
      .header-container { flex-direction: column; align-items: flex-start; }
      .header-left { width: 100%; flex-direction: column; align-items: flex-start; gap: 0.5rem; }
      .header-right { width: 100%; justify-content: space-between; }
      .nav-tabs { width: 100%; display: flex; }
      .nav-tab { flex: 1; text-align: center; }
      .console-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
      .console-controls { width: 100%; display: flex; flex-wrap: wrap; justify-content: space-between; }
      .hist-logs-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
      #hist-logs-meta { width: 100%; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; }
      .console-box { height: 320px; min-height: 140px; font-size: 0.75rem; padding: 0.5rem 0.65rem; }
      input[type="text"], input[type="number"], select { font-size: 16px; padding: 0.55rem 0.65rem; }
      .btn { min-height: 44px; }
    }

    @media (max-width: 480px) {
      body { padding: 0.5rem; }
      h1 { font-size: 1.15rem; }
      .card { padding: 0.75rem; border-radius: 0.5rem; }
      .card h2 { font-size: 0.95rem; }
      .stepper { grid-template-columns: repeat(2, 1fr); gap: 0.35rem; }
      .step-item:last-child { grid-column: span 2; }
      .step-item { padding: 0.45rem 0.2rem; font-size: 0.7rem; }
      .nav-tab { padding: 0.45rem 0.4rem; font-size: 0.8rem; }
      .form-actions { flex-direction: column; }
      .form-actions .btn { width: 100% !important; }
      .console-box { height: 280px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-container">
      <div class="header-left">
        <h1 style="margin: 0;">Frigate Model Training Studio</h1>
        <nav class="nav-tabs">
          <button id="nav-btn-studio" class="nav-tab active" onclick="switchView('studio')">🚀 Training Studio</button>
          <button id="nav-btn-settings" class="nav-tab" onclick="switchView('settings')">⚙️ Settings &amp; Profiles</button>
        </nav>
      </div>
      <div class="header-right">
        <span id="status-badge" class="badge badge-idle">Idle</span>
        <span id="elapsed-time" style="font-size: 0.85rem; color: var(--text-muted); margin-left: 0.5rem;"></span>
      </div>
    </div>
  </header>

  <!-- Training Studio View -->
  <div id="view-studio" class="view-panel">
    <div class="grid">
      <!-- Left Column: Controls & Configuration -->
      <div id="studio-left-col">
        <div id="active-run-banner" style="display: none; background: rgba(56, 189, 248, 0.15); border: 1px solid var(--primary); padding: 0.65rem 0.85rem; border-radius: 0.5rem; font-size: 0.85rem; color: var(--primary); margin-bottom: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <strong>Active training job in progress</strong><span id="banner-settings-msg" style="display: none;"> &mdash; settings collapsed for log visibility</span>.
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <button id="banner-queue-btn" class="btn btn-sm" type="button" onclick="enterQueueMode()" style="background: rgba(16, 185, 129, 0.2); border: 1px solid var(--success); color: var(--success); font-weight: 600;">+ Setup Next Run</button>
              <button id="banner-toggle-settings-btn" class="btn btn-sm" type="button" onclick="toggleSettingsPanes()">Show Settings</button>
              <button id="banner-abort-btn" class="btn btn-sm btn-danger" type="button" onclick="abortTraining()">Abort Run</button>
            </div>
          </div>
        </div>

        <div id="queue-mode-banner" style="display: none; background: rgba(16, 185, 129, 0.15); border: 1px solid var(--success); padding: 0.65rem 0.85rem; border-radius: 0.5rem; font-size: 0.85rem; color: var(--success); margin-bottom: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <strong id="queue-mode-title">📋 Setup Next Model Run (Queued)</strong> &mdash; <span>Configure model options below. It will automatically start when the current run finishes.</span>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <button class="btn btn-sm btn-success" type="button" onclick="submitQueuedRun()">Save to Queue</button>
              <button class="btn btn-sm" type="button" onclick="exitQueueMode()">Back to Active Run</button>
            </div>
          </div>
        </div>

        <div id="queued-run-card" class="card" style="display: none; border: 1px solid var(--success); background: rgba(16, 185, 129, 0.08); margin-bottom: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span class="badge badge-queued">QUEUED NEXT</span>
              <h2 style="margin: 0; font-size: 0.95rem; border: none; padding: 0; display: inline;">
                Run <span id="queued-run-id-label">#q-1</span> (<span id="queued-run-profile-label">torq</span>)
              </h2>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <button id="queued-start-now-btn" class="btn btn-sm btn-success" type="button" style="display: none;" onclick="startQueuedRunNow()">▶️ Start Run Now</button>
              <button class="btn btn-sm" type="button" onclick="editQueuedRun()">✏️ Edit Options</button>
              <button class="btn btn-sm btn-danger" type="button" onclick="cancelQueuedRun()">🗑️ Cancel Queue</button>
            </div>
          </div>
          <div id="queued-run-summary" style="font-size: 0.825rem; color: #cbd5e1; line-height: 1.5;"></div>
          <div id="queued-run-status-notice" style="margin-top: 0.4rem; font-size: 0.8rem; color: var(--success); display: flex; align-items: center; gap: 0.4rem;">
            <span>⏳ Will automatically start when current pipeline execution completes.</span>
          </div>
        </div>

        <div id="studio-config-panes">
          <!-- Profile & Workflow -->
          <div class="card">
          <h2>Execution Profile &amp; Workflow</h2>
          <div class="form-group">
            <label for="profile-select">Target Profile</label>
            <select id="profile-select" onchange="onProfileSelectChange()"></select>
          </div>
          <div class="form-group">
            <label for="workflow-select">Pipeline Workflow</label>
            <select id="workflow-select" onchange="toggleWorkflowInputs()">
              <option value="full">Full Pipeline (Train, Compile, Deploy)</option>
              <option value="compile_only">Compile Only (--compile)</option>
              <option value="skip_compile">Train Only (--skip-compile)</option>
              <option value="pull_only">Pull Weights from Spark &amp; Compile (--pull)</option>
            </select>
          </div>
          <div id="custom-ckpt-group" class="form-group" style="display: none;">
            <label for="custom-checkpoint">Custom Checkpoint Path (optional)</label>
            <input type="text" id="custom-checkpoint" placeholder="staging/yolov8n/weights/best.pt">
          </div>
        </div>

        <!-- CVAT Task Selection & Export -->
        <div class="card">
          <h2>
            <span>CVAT Dataset Export</span>
            <button class="btn btn-sm" onclick="fetchCvatTasks()">Refresh Tasks</button>
          </h2>
          <div class="checkbox-row" style="margin-bottom: 0.5rem;">
            <input type="checkbox" id="do-export" onchange="toggleCvatTasksBox()">
            <label for="do-export" style="margin-bottom: 0; font-weight: 600; color: #e2e8f0;">Export CVAT Tasks Before Training (--export)</label>
          </div>
          <div id="cvat-container" style="display: none;">
            <div style="display: flex; gap: 0.4rem; margin-bottom: 0.4rem;">
              <button class="btn btn-sm" onclick="selectCvatQuick('latest')">Latest Only</button>
              <button class="btn btn-sm" onclick="selectCvatQuick('all')">Select All</button>
              <button class="btn btn-sm" onclick="selectCvatQuick('none')">Clear</button>
            </div>
            <div id="cvat-task-list" class="cvat-box">
              <div style="color: var(--text-muted); text-align: center; padding: 1rem;">Click "Refresh Tasks" to connect to CVAT.</div>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="replace-dataset">
              <label for="replace-dataset" style="margin-bottom: 0;">Wipe &amp; Replace Existing YOLO Dataset (--replace)</label>
            </div>
            <div class="form-group" style="margin-top: 0.6rem;">
              <label for="manual-task-ids">Or Enter Task IDs Manually</label>
              <input type="text" id="manual-task-ids" placeholder="e.g. 21 22 (leave blank to use checked tasks)">
            </div>
          </div>
        </div>

        <!-- Checkpoint & Architecture -->
        <div class="card">
          <h2>Model Checkpoint &amp; Architecture</h2>
          <div class="form-group">
            <label for="base-model">Base Model Checkpoint (-m)</label>
            <input type="text" id="base-model" placeholder="yolov8n.pt, yolov8m.pt, yolov9s.pt">
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="imgsz">Image Size (--imgsz)</label>
              <input type="number" id="imgsz" placeholder="640 or 320" value="640">
            </div>
            <div class="checkbox-row" style="margin-top: 1.5rem;">
              <input type="checkbox" id="resume">
              <label for="resume" style="margin-bottom: 0;">Resume Run (--resume)</label>
            </div>
          </div>
        </div>

        <!-- Training Hyperparameters -->
        <div class="card">
          <h2>Training Hyperparameters</h2>
          <div class="form-row">
            <div class="form-group">
              <label for="epochs">Epochs (--epochs)</label>
              <input type="number" id="epochs" value="80">
            </div>
            <div class="form-group">
              <label for="batch">Batch Size (--batch)</label>
              <input type="number" id="batch" value="16">
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="patience">Early Stopping Patience (--patience)</label>
              <input type="number" id="patience" value="30">
            </div>
            <div class="form-group">
              <label for="workers">Data Workers (--workers)</label>
              <input type="number" id="workers" value="8">
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="device">Device (--device)</label>
              <input type="text" id="device" value="0">
            </div>
            <div class="form-group">
              <label for="cache">Dataset Cache (--cache)</label>
              <select id="cache">
                <option value="ram">ram</option>
                <option value="disk">disk</option>
                <option value="false">false</option>
              </select>
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="optimizer">Optimizer (--optimizer)</label>
              <select id="optimizer">
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
                <option value="Adam">Adam</option>
                <option value="auto">auto</option>
              </select>
            </div>
            <div class="form-group">
              <label for="lr0">Initial LR (--lr0)</label>
              <input type="number" id="lr0" step="0.0001" value="0.0005">
            </div>
          </div>
          <div class="form-group">
            <label for="lrf">Final LR Ratio (--lrf)</label>
            <input type="number" id="lrf" step="0.001" value="0.01">
          </div>
          <div class="form-group">
            <label for="extra-args">Extra Train Args (--extra-args)</label>
            <input type="text" id="extra-args" placeholder="e.g. mosaic=0.5 mixup=0.1">
          </div>
        </div>

        <!-- Dataset & Co-Training -->
        <div class="card">
          <h2>Dataset &amp; Co-Training</h2>
          <div class="form-group">
            <label for="coco-dataset">COCO Subset (--coco-dataset)</label>
            <select id="coco-dataset">
              <option value="coco-val2017">coco-val2017 (surveillance &amp; objects)</option>
              <option value="coco128">coco128 (mini demo dataset)</option>
            </select>
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="frigate-plus" checked>
            <label for="frigate-plus" style="margin-bottom: 0;">Include Frigate+ Dataset (--frigate-plus)</label>
          </div>
        </div>

        <!-- Remote Spark Training -->
        <div class="card">
          <h2>Remote DGX Spark Training</h2>
          <div class="form-group">
            <label for="spark-host">Remote Spark Host (SSH, --host)</label>
            <input type="text" id="spark-host" placeholder="user@hostname">
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="clean-images">
            <label for="clean-images" style="margin-bottom: 0;">Clean Remote Images After Run (--clean-images)</label>
          </div>
        </div>

        <!-- Deployment & Logging -->
        <div class="card">
          <h2>Deployment &amp; Logging</h2>
          <div class="form-group">
            <label for="deploy-target">Deploy Target (--deploy user@host:/path or local dir)</label>
            <input type="text" id="deploy-target" placeholder="root@192.168.1.100:/opt/models">
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="skip-deploy">
            <label for="skip-deploy" style="margin-bottom: 0;">Skip Deployment (--skip-deploy)</label>
          </div>
          <div class="form-row" style="margin-top: 0.75rem;">
            <div class="checkbox-row">
              <input type="checkbox" id="log-console" checked>
              <label for="log-console" style="margin-bottom: 0;">Log Console</label>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="log-file" checked>
              <label for="log-file" style="margin-bottom: 0;">Log File</label>
            </div>
          </div>
        </div>
        </div><!-- end #studio-config-panes -->

        <!-- Action Buttons -->
        <div id="action-buttons-card" class="card">
          <div id="normal-action-buttons">
            <button id="start-btn" class="btn btn-primary" onclick="startTraining()">Start Pipeline Execution</button>
            <button id="queue-next-btn" class="btn" type="button" style="display: none; background: rgba(16, 185, 129, 0.2); border: 1px solid var(--success); color: var(--success); font-weight: 600; width: 100%; margin-top: 0.75rem;" onclick="enterQueueMode()">+ Setup &amp; Queue Next Run</button>
            <button id="abort-btn" class="btn btn-danger" style="display: none; margin-top: 0.75rem;" onclick="abortTraining()">Abort Active Run</button>
          </div>
          <div id="queue-action-buttons" style="display: none; flex-direction: column; gap: 0.5rem;">
            <button id="btn-submit-queue" class="btn btn-success" type="button" onclick="submitQueuedRun()">Queue Model Run (Starts After Current Run)</button>
            <button class="btn" type="button" style="background: transparent; border: 1px solid var(--card-border);" onclick="exitQueueMode()">Cancel &amp; Return to Active Run</button>
          </div>
        </div>
      </div>

      <!-- Right Column: Monitoring, Stepper & Logs -->
      <div>
        <div class="card">
          <h2>Pipeline Execution Stages</h2>
          <div class="stepper">
            <div id="step-export" class="step-item step-pending">1. CVAT Export</div>
            <div id="step-pretrain" class="step-item step-pending">2. Pretraining</div>
            <div id="step-train" class="step-item step-pending">3. Training</div>
            <div id="step-compile" class="step-item step-pending">4. Compilation</div>
            <div id="step-deploy" class="step-item step-pending">5. Deployment</div>
          </div>
        </div>

        <div class="card">
          <div class="console-header">
            <h2>Process Log Stream</h2>
            <div class="console-controls" style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
              <label style="margin: 0; font-size: 0.8rem; display: flex; align-items: center; gap: 0.25rem; cursor: pointer;">
                <input type="checkbox" id="stream-wrap-chk" checked onchange="toggleConsoleWrap('console-stream', this.checked)"> Wrap lines
              </label>
              <label style="margin: 0; font-size: 0.8rem; display: flex; align-items: center; gap: 0.25rem; cursor: pointer;">
                <input type="checkbox" id="autoscroll-chk" checked> Autoscroll
              </label>
              <button class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.75rem; width: auto;" onclick="clearConsole()">Clear</button>
            </div>
          </div>
          <div id="console-stream" class="console-box"></div>
        </div>

        <!-- Run Record & Artifact History -->
        <div class="card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <h2 style="margin: 0;">Run Record &amp; Artifact History</h2>
            <span style="font-size: 0.75rem; color: var(--text-muted);">Click any row to inspect execution logs</span>
          </div>
          <div class="history-scroll-container">
            <table class="history-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Profile</th>
                  <th>Status</th>
                  <th>Elapsed</th>
                  <th>Artifacts Generated</th>
                </tr>
              </thead>
              <tbody id="history-table-body">
                <tr><td colspan="5" style="color: var(--text-muted); text-align: center;">No runs recorded yet.</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Frame below run record & artifact history for clicked run logs -->
        <div id="historical-logs-card" class="card" style="margin-top: 1.25rem;">
          <div class="hist-logs-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
              <h2 style="margin: 0; font-size: 1.05rem;" id="hist-logs-title">Execution Log Viewer</h2>
              <span id="hist-logs-badge" class="badge" style="display: none;"></span>
            </div>
            <div id="hist-logs-meta" style="display: flex; gap: 0.75rem; align-items: center; font-size: 0.825rem; color: var(--text-muted);">
              <label style="margin: 0; font-size: 0.8rem; display: flex; align-items: center; gap: 0.25rem; cursor: pointer;">
                <input type="checkbox" id="hist-wrap-chk" checked onchange="toggleConsoleWrap('hist-console', this.checked)"> Wrap lines
              </label>
              <span id="hist-logs-hint">Click a run record above to inspect its execution logs</span>
              <button id="hist-close-btn" class="btn btn-sm" style="display: none;" onclick="closeHistoryLogs()">Close</button>
            </div>
          </div>
          <div id="hist-console" class="console-box" style="height: 320px; min-height: 160px;">
            <div style="color: var(--text-muted); text-align: center; padding: 3rem 1rem;">
              Select a past run from the table above to view its console log stream.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Settings & Profiles View -->
  <div id="view-settings" class="view-panel" style="display: none; max-width: 1200px; margin: 0 auto;">
    <!-- Profiles List Card -->
    <div class="card" style="margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem;">
        <div>
          <h2 style="margin: 0;">Configuration Profiles</h2>
          <div style="color: var(--text-muted); font-size: 0.85rem;">Manage execution targets, compilation settings, DGX Spark remotes, and hyperparameters in config.yaml</div>
        </div>
        <button class="btn btn-primary" style="width: auto; padding: 0.5rem 1rem;" onclick="createNewProfileForm()">+ Create New Profile</button>
      </div>
      <div class="profiles-table-container" style="overflow-x: auto;">
        <table class="profiles-table">
          <thead>
            <tr>
              <th>Profile Name</th>
              <th>Target</th>
              <th>Execution</th>
              <th>Base Model</th>
              <th>Epochs / Batch</th>
              <th>Device</th>
              <th>Status</th>
              <th style="text-align: right;">Actions</th>
            </tr>
          </thead>
          <tbody id="settings-profiles-tbody">
            <tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Loading profiles...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Profile Editor Form Card -->
    <div class="card" id="profile-editor-card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h2 style="margin: 0;" id="editor-heading">Create / Edit Profile</h2>
        <button class="btn btn-sm" onclick="resetProfileForm()">Clear / Reset</button>
      </div>
      <div id="profile-form-alert" style="display: none; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem;"></div>
      
      <form id="profile-edit-form" onsubmit="handleSaveProfile(event)">
        <input type="hidden" id="prof-original-name" value="">
        
        <div class="form-row">
          <div class="form-group">
            <label for="prof-name">Profile Name *</label>
            <input type="text" id="prof-name" placeholder="e.g. hailo-custom" required pattern="[a-zA-Z0-9_-]+">
          </div>
          <div class="form-group">
            <label for="prof-target">Target Hardware Architecture *</label>
            <select id="prof-target" onchange="handleTargetChange()">
              <option value="torq">Torq NPU (.vmfb / SL2610)</option>
              <option value="hailo">Hailo NPU (.hef / Hailo-10H)</option>
              <option value="nvidia">NVIDIA ONNX / TensorRT (GPU)</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="prof-base-model">Base Model Checkpoint *</label>
            <input type="text" id="prof-base-model" placeholder="e.g. yolov8n.pt or yolov8m.pt" required>
          </div>
          <div class="form-group" id="prof-hwarch-group">
            <label for="prof-hwarch">Hailo Architecture / HW Spec</label>
            <input type="text" id="prof-hwarch" value="hailo10h" placeholder="hailo10h or hailo8">
          </div>
        </div>

        <div style="margin: 1.25rem 0; padding: 1rem; background: var(--bg); border-radius: 6px; border: 1px solid var(--card-border);">
          <div class="checkbox-row" style="margin-bottom: 0.75rem;">
            <input type="checkbox" id="prof-remote" onchange="handleRemoteToggle()">
            <label for="prof-remote" style="margin-bottom: 0; font-weight: 600;">Remote DGX Spark Execution (via SSH)</label>
          </div>
          <div id="prof-remote-fields" style="display: none;" class="form-row">
            <div class="form-group">
              <label for="prof-host">SSH Remote Host</label>
              <input type="text" id="prof-host" placeholder="stephen@trident-canton.tridentlabs.internal">
            </div>
            <div class="form-group">
              <label for="prof-remote-dir">Remote Directory</label>
              <input type="text" id="prof-remote-dir" value="~/frigate-model-training" placeholder="~/frigate-model-training">
            </div>
            <div class="checkbox-row" style="margin-top: 1.75rem;">
              <input type="checkbox" id="prof-clean-images" checked>
              <label for="prof-clean-images" style="margin-bottom: 0;">Clean Remote Images</label>
            </div>
          </div>
        </div>

        <h3 style="font-size: 1rem; margin-bottom: 0.75rem; color: var(--primary);">Hyperparameters &amp; Training Settings</h3>
        <div class="form-row-4">
          <div class="form-group">
            <label for="prof-device">Device</label>
            <input type="text" id="prof-device" value="0" placeholder="0 or cpu" required>
          </div>
          <div class="form-group">
            <label for="prof-epochs">Epochs</label>
            <input type="number" id="prof-epochs" value="80" min="1" required>
          </div>
          <div class="form-group">
            <label for="prof-batch">Batch Size</label>
            <input type="number" id="prof-batch" value="16" min="1" required>
          </div>
          <div class="form-group">
            <label for="prof-patience">Patience</label>
            <input type="number" id="prof-patience" value="30" min="1" required>
          </div>
        </div>

        <div class="form-row-4">
          <div class="form-group">
            <label for="prof-workers">Workers</label>
            <input type="number" id="prof-workers" value="8" min="0" required>
          </div>
          <div class="form-group">
            <label for="prof-imgsz">Image Size</label>
            <input type="number" id="prof-imgsz" value="640" min="32" step="32" required>
          </div>
          <div class="form-group">
            <label for="prof-cache">Cache Mode</label>
            <select id="prof-cache">
              <option value="ram">ram (System Memory)</option>
              <option value="disk">disk (Filesystem)</option>
              <option value="false">false (Disabled)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="prof-optimizer">Optimizer</label>
            <select id="prof-optimizer">
              <option value="AdamW">AdamW</option>
              <option value="SGD">SGD</option>
              <option value="Adam">Adam</option>
              <option value="auto">auto</option>
            </select>
          </div>
        </div>

        <div class="form-row-3">
          <div class="form-group">
            <label for="prof-lr0">Initial LR (lr0)</label>
            <input type="number" id="prof-lr0" value="0.0005" step="0.0001" min="0.000001" required>
          </div>
          <div class="form-group">
            <label for="prof-lrf">Final LR (lrf)</label>
            <input type="number" id="prof-lrf" value="0.01" step="0.001" min="0.00001" required>
          </div>
          <div class="checkbox-row" style="margin-top: 1.75rem;">
            <input type="checkbox" id="prof-cos-lr" checked>
            <label for="prof-cos-lr" style="margin-bottom: 0;">Cosine LR Scheduler</label>
          </div>
        </div>

        <div class="form-row-3">
          <div class="form-group">
            <label for="prof-warmup-epochs">Warmup Epochs</label>
            <input type="number" id="prof-warmup-epochs" value="5.0" step="0.5" min="0">
          </div>
          <div class="form-group">
            <label for="prof-warmup-momentum">Warmup Momentum</label>
            <input type="number" id="prof-warmup-momentum" value="0.5" step="0.05" min="0">
          </div>
          <div class="form-group">
            <label for="prof-warmup-bias-lr">Warmup Bias LR</label>
            <input type="number" id="prof-warmup-bias-lr" value="0.0" step="0.01" min="0">
          </div>
        </div>

        <h3 style="font-size: 1rem; margin: 1rem 0 0.75rem 0; color: var(--primary);">Deployment &amp; Logging Settings</h3>
        <div class="form-row">
          <div class="form-group">
            <label for="prof-deploy-host">Deployment Host (e.g. root@coralboard.local or leave empty)</label>
            <input type="text" id="prof-deploy-host" placeholder="root@coralboard.local">
          </div>
          <div class="form-group">
            <label for="prof-deploy-dir">Deployment Directory</label>
            <input type="text" id="prof-deploy-dir" placeholder="/opt/models">
          </div>
        </div>

        <div class="form-row" style="margin-bottom: 1rem;">
          <div class="checkbox-row">
            <input type="checkbox" id="prof-log-console" checked>
            <label for="prof-log-console" style="margin-bottom: 0;">Console Logging</label>
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="prof-log-file" checked>
            <label for="prof-log-file" style="margin-bottom: 0;">File Logging</label>
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="prof-is-default">
            <label for="prof-is-default" style="margin-bottom: 0; font-weight: 600; color: var(--warning);">★ Set as Default Profile</label>
          </div>
        </div>

        <div class="form-actions" style="display: flex; gap: 0.75rem; margin-top: 1.5rem;">
          <button type="submit" class="btn btn-primary" style="width: auto; padding: 0.6rem 1.5rem;">💾 Save Profile</button>
          <button type="button" class="btn" style="width: auto; padding: 0.6rem 1.5rem; background: #334155; color: #e2e8f0;" onclick="resetProfileForm()">Cancel</button>
        </div>
      </form>
    </div>
  </div>

  <script>
    let profilesData = {};
    let isRunning = false;
    let wasRunning = false;
    let isSettingsCollapsed = false;
    let userManuallyToggledSettings = false;
    let sseSource = null;
    let cvatTasksList = [];
    let currentActiveRunId = null;
    let lastSeenRunId = null;
    let selectedHistoryRunId = null;
    let formMode = "normal";
    let editingQueueId = null;
    let cachedActiveRunOptions = null;
    let currentQueue = [];

    function isMobileWebFormat() {
      return window.matchMedia("(max-width: 1080px)").matches || window.innerWidth <= 1080;
    }

    function formatDuration(totalSec) {
      const s = Math.max(0, Math.floor(Number(totalSec) || 0));
      const hrs = Math.floor(s / 3600);
      const mins = Math.floor((s % 3600) / 60);
      const secs = s % 60;
      return `${hrs}h ${mins}m ${secs}s`;
    }

    function collapseSettingsPanes(collapse) {
      isSettingsCollapsed = collapse;
      const panes = document.getElementById("studio-config-panes");
      const actionsCard = document.getElementById("action-buttons-card");
      const bannerBtn = document.getElementById("banner-toggle-settings-btn");
      const bannerMsg = document.getElementById("banner-settings-msg");
      if (panes) {
        panes.style.display = collapse ? "none" : "block";
      }
      if (actionsCard) {
        actionsCard.style.display = (collapse && isRunning) ? "none" : "block";
      }
      if (bannerBtn) {
        bannerBtn.textContent = collapse ? "Show Settings" : "Hide Settings";
      }
      if (bannerMsg) {
        bannerMsg.style.display = collapse ? "inline" : "none";
      }
      if (collapse && isMobileWebFormat()) {
        const streamBox = document.getElementById("console-stream");
        if (streamBox) {
          streamBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }
    }

    function toggleSettingsPanes() {
      userManuallyToggledSettings = true;
      collapseSettingsPanes(!isSettingsCollapsed);
    }

    async function init() {
      setupTerminal();
      setupSSE();
      await fetchCvatTasks();
      await reloadProfiles();

      try {
        const statusRes = await fetch("/api/status");
        const statusData = await statusRes.json();
        if (statusData.running && isMobileWebFormat()) {
          collapseSettingsPanes(true);
        }
        if (statusData.running && statusData.options && Object.keys(statusData.options).length > 0) {
          currentActiveRunId = statusData.run_id;
          cachedActiveRunOptions = statusData.options;
          populateFormFromOptions(statusData.options);
          setFormInputsDisabled(true);
        } else if (profilesData.default_profile) {
          loadProfileDefaults(profilesData.default_profile);
        }
        if (statusData && statusData.run_id) {
          lastSeenRunId = statusData.run_id;
        }
        updateUIState(statusData);
      } catch (err) {
        console.error("Initial status check failed:", err);
      }

      // Hash routing
      window.addEventListener("hashchange", () => {
        const view = window.location.hash === "#settings" ? "settings" : "studio";
        switchView(view);
      });
      if (window.location.hash === "#settings") {
        switchView("settings");
      }

      window.addEventListener("resize", () => {
        if (isRunning && !userManuallyToggledSettings) {
          if (isMobileWebFormat() && !isSettingsCollapsed) {
            collapseSettingsPanes(true);
          } else if (!isMobileWebFormat() && isSettingsCollapsed) {
            collapseSettingsPanes(false);
          }
        }
      });

      setInterval(pollStatus, 3000);
    }

    function switchView(viewName) {
      const isStudio = viewName === "studio";
      document.getElementById("view-studio").style.display = isStudio ? "block" : "none";
      document.getElementById("view-settings").style.display = isStudio ? "none" : "block";
      document.getElementById("nav-btn-studio").classList.toggle("active", isStudio);
      document.getElementById("nav-btn-settings").classList.toggle("active", !isStudio);
      window.location.hash = isStudio ? "#studio" : "#settings";
      if (!isStudio) {
        renderSettingsProfilesTable();
      }
    }

    async function reloadProfiles() {
      try {
        const res = await fetch("/api/profiles");
        profilesData = await res.json();
        
        const select = document.getElementById("profile-select");
        const currentVal = select.value;
        select.innerHTML = "";
        Object.keys(profilesData.profiles || {}).forEach(name => {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name.toUpperCase() + (name === profilesData.default_profile ? " (Default)" : "");
          if (name === currentVal || (!currentVal && name === profilesData.default_profile)) {
            opt.selected = true;
          }
          select.appendChild(opt);
        });

        select.onchange = onProfileSelectChange;

        renderSettingsProfilesTable();
      } catch (err) {
        console.error("Failed to reload profiles:", err);
      }
    }

    function renderSettingsProfilesTable() {
      const tbody = document.getElementById("settings-profiles-tbody");
      if (!tbody) return;
      tbody.innerHTML = "";
      const profiles = profilesData.profiles || {};
      const defProf = profilesData.default_profile || "";

      if (Object.keys(profiles).length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-muted);">No profiles configured.</td></tr>';
        return;
      }

      Object.entries(profiles).forEach(([name, p]) => {
        const tr = document.createElement("tr");
        const isDefault = name === defProf;
        const targetBadge = `<span class="badge badge-idle">${p.target || "torq"}</span>`;
        const execMode = p.remote ? `Remote (${p.host ? (p.host.split("@")[1] || p.host) : "Spark"})` : "Local";
        const model = p.base_model || "-";
        const epBatch = `${p.epochs || 80} ep / b${p.batch || 16}`;
        const dev = p.device || "0";
        const statusHtml = isDefault 
          ? '<span style="color: var(--warning); font-weight: 600; font-size: 0.8rem;">★ Default</span>'
          : '<span style="color: var(--text-muted); font-size: 0.8rem;">Standard</span>';

        tr.innerHTML = `
          <td><strong>${name}</strong>${isDefault ? ' <span class="default-star">★</span>' : ''}</td>
          <td>${targetBadge}</td>
          <td>${execMode}</td>
          <td>${model}</td>
          <td>${epBatch}</td>
          <td>${dev}</td>
          <td>${statusHtml}</td>
          <td style="text-align: right; white-space: nowrap;">
            <div style="display: inline-flex; gap: 0.35rem;">
              <button class="btn btn-sm" onclick="editProfile('${name}')">Edit</button>
              <button class="btn btn-sm" onclick="duplicateProfile('${name}')">Duplicate</button>
              ${!isDefault ? `<button class="btn btn-sm" style="color: var(--warning);" onclick="setDefaultProfile('${name}')">Make Default</button>` : ''}
              <button class="btn btn-sm btn-danger" onclick="deleteProfile('${name}')">Delete</button>
            </div>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    function createNewProfileForm() {
      resetProfileForm();
      document.getElementById("editor-heading").textContent = "Create New Profile";
      document.getElementById("prof-name").focus();
      document.getElementById("profile-editor-card").scrollIntoView({ behavior: "smooth" });
    }

    function resetProfileForm() {
      document.getElementById("prof-original-name").value = "";
      document.getElementById("prof-name").value = "";
      document.getElementById("prof-name").disabled = false;
      document.getElementById("prof-target").value = "torq";
      document.getElementById("prof-base-model").value = "yolov8n.pt";
      document.getElementById("prof-hwarch").value = "hailo10h";
      document.getElementById("prof-device").value = "0";
      document.getElementById("prof-epochs").value = 80;
      document.getElementById("prof-batch").value = 16;
      document.getElementById("prof-patience").value = 30;
      document.getElementById("prof-workers").value = 8;
      document.getElementById("prof-imgsz").value = 640;
      document.getElementById("prof-cache").value = "ram";
      document.getElementById("prof-optimizer").value = "AdamW";
      document.getElementById("prof-lr0").value = 0.0005;
      document.getElementById("prof-lrf").value = 0.01;
      document.getElementById("prof-cos-lr").checked = true;
      document.getElementById("prof-warmup-epochs").value = 5.0;
      document.getElementById("prof-warmup-momentum").value = 0.5;
      document.getElementById("prof-warmup-bias-lr").value = 0.0;
      document.getElementById("prof-remote").checked = false;
      document.getElementById("prof-host").value = "";
      document.getElementById("prof-remote-dir").value = "~/frigate-model-training";
      document.getElementById("prof-clean-images").checked = true;
      document.getElementById("prof-deploy-host").value = "";
      document.getElementById("prof-deploy-dir").value = "";
      document.getElementById("prof-log-console").checked = true;
      document.getElementById("prof-log-file").checked = true;
      document.getElementById("prof-is-default").checked = false;
      hideProfileAlert();
      handleTargetChange();
      handleRemoteToggle();
      document.getElementById("editor-heading").textContent = "Create / Edit Profile";
    }

    function editProfile(name) {
      const p = (profilesData.profiles || {})[name];
      if (!p) return;

      document.getElementById("prof-original-name").value = name;
      document.getElementById("prof-name").value = name;
      document.getElementById("prof-name").disabled = false;
      document.getElementById("prof-target").value = p.target || "torq";
      document.getElementById("prof-base-model").value = p.base_model || "";
      document.getElementById("prof-hwarch").value = p.hw_arch || "hailo10h";
      document.getElementById("prof-device").value = p.device || "0";
      document.getElementById("prof-epochs").value = p.epochs || 80;
      document.getElementById("prof-batch").value = p.batch || 16;
      document.getElementById("prof-patience").value = p.patience || 30;
      document.getElementById("prof-workers").value = p.workers || 8;
      document.getElementById("prof-imgsz").value = p.imgsz || (p.target === "hailo" ? 640 : 320);
      document.getElementById("prof-cache").value = p.cache || "ram";
      document.getElementById("prof-optimizer").value = p.optimizer || "AdamW";
      document.getElementById("prof-lr0").value = p.lr0 || 0.0005;
      document.getElementById("prof-lrf").value = p.lrf || 0.01;
      document.getElementById("prof-cos-lr").checked = p.cos_lr !== false;
      document.getElementById("prof-warmup-epochs").value = p.warmup_epochs !== undefined ? p.warmup_epochs : 5.0;
      document.getElementById("prof-warmup-momentum").value = p.warmup_momentum !== undefined ? p.warmup_momentum : 0.5;
      document.getElementById("prof-warmup-bias-lr").value = p.warmup_bias_lr !== undefined ? p.warmup_bias_lr : 0.0;
      
      document.getElementById("prof-remote").checked = !!p.remote;
      document.getElementById("prof-host").value = p.host || "";
      document.getElementById("prof-remote-dir").value = p.remote_dir || "~/frigate-model-training";
      document.getElementById("prof-clean-images").checked = p.clean_images !== false;

      const dep = p.deployment || {};
      document.getElementById("prof-deploy-host").value = dep.host || "";
      document.getElementById("prof-deploy-dir").value = dep.dir || "";

      const log = p.logging || {};
      document.getElementById("prof-log-console").checked = log.console !== false;
      document.getElementById("prof-log-file").checked = log.file !== false;

      document.getElementById("prof-is-default").checked = (name === profilesData.default_profile);

      document.getElementById("editor-heading").textContent = "Edit Profile: " + name;
      hideProfileAlert();
      handleTargetChange();
      handleRemoteToggle();
      document.getElementById("profile-editor-card").scrollIntoView({ behavior: "smooth" });
    }

    function duplicateProfile(name) {
      editProfile(name);
      document.getElementById("prof-original-name").value = "";
      document.getElementById("prof-name").value = name + "-copy";
      document.getElementById("prof-is-default").checked = false;
      document.getElementById("editor-heading").textContent = "Duplicate Profile as: " + name + "-copy";
      document.getElementById("prof-name").focus();
    }

    async function setDefaultProfile(name) {
      try {
        const res = await fetch("/api/profiles/default", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (!res.ok) {
          alert("Failed to set default profile: " + (data.error || "Unknown error"));
        } else {
          await reloadProfiles();
        }
      } catch (err) {
        alert("Network error setting default profile: " + err);
      }
    }

    async function deleteProfile(name) {
      if (!confirm(`Are you sure you want to delete profile '${name}'? This action modifies config.yaml.`)) {
        return;
      }
      try {
        const res = await fetch(`/api/profiles?name=${encodeURIComponent(name)}`, {
          method: "DELETE"
        });
        const data = await res.json();
        if (!res.ok) {
          alert("Failed to delete profile: " + (data.error || "Unknown error"));
        } else {
          if (document.getElementById("prof-original-name").value === name) {
            resetProfileForm();
          }
          await reloadProfiles();
        }
      } catch (err) {
        alert("Network error deleting profile: " + err);
      }
    }

    async function handleSaveProfile(event) {
      event.preventDefault();
      const origName = document.getElementById("prof-original-name").value.trim();
      const name = document.getElementById("prof-name").value.trim().toLowerCase();
      if (!name) {
        showProfileAlert("Profile name is required.", "danger");
        return;
      }

      const isRemote = document.getElementById("prof-remote").checked;
      const target = document.getElementById("prof-target").value;

      const profilePayload = {
        target: target,
        remote: isRemote,
        device: document.getElementById("prof-device").value.trim() || "0",
        batch: parseInt(document.getElementById("prof-batch").value, 10) || 16,
        epochs: parseInt(document.getElementById("prof-epochs").value, 10) || 80,
        patience: parseInt(document.getElementById("prof-patience").value, 10) || 30,
        workers: parseInt(document.getElementById("prof-workers").value, 10) || 8,
        imgsz: parseInt(document.getElementById("prof-imgsz").value, 10) || 640,
        cache: document.getElementById("prof-cache").value,
        base_model: document.getElementById("prof-base-model").value.trim(),
        optimizer: document.getElementById("prof-optimizer").value,
        lr0: parseFloat(document.getElementById("prof-lr0").value) || 0.0005,
        lrf: parseFloat(document.getElementById("prof-lrf").value) || 0.01,
        cos_lr: document.getElementById("prof-cos-lr").checked,
        warmup_epochs: parseFloat(document.getElementById("prof-warmup-epochs").value) || 5.0,
        warmup_momentum: parseFloat(document.getElementById("prof-warmup-momentum").value) || 0.5,
        warmup_bias_lr: parseFloat(document.getElementById("prof-warmup-bias-lr").value) || 0.0,
        deployment: {
          host: document.getElementById("prof-deploy-host").value.trim() || null,
          dir: document.getElementById("prof-deploy-dir").value.trim() || null,
        },
        logging: {
          console: document.getElementById("prof-log-console").checked,
          file: document.getElementById("prof-log-file").checked,
        }
      };

      if (isRemote) {
        profilePayload.host = document.getElementById("prof-host").value.trim();
        profilePayload.remote_dir = document.getElementById("prof-remote-dir").value.trim() || "~/frigate-model-training";
        profilePayload.clean_images = document.getElementById("prof-clean-images").checked;
      }

      if (target === "hailo") {
        profilePayload.hw_arch = document.getElementById("prof-hwarch").value.trim() || "hailo10h";
      }

      const body = {
        name: name,
        original_name: origName,
        is_default: document.getElementById("prof-is-default").checked,
        profile: profilePayload,
      };

      try {
        const res = await fetch("/api/profiles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) {
          showProfileAlert(data.error || "Failed to save profile.", "danger");
        } else {
          showProfileAlert(`Profile '${name}' saved successfully to config.yaml.`, "success");
          document.getElementById("prof-original-name").value = name;
          document.getElementById("editor-heading").textContent = "Edit Profile: " + name;
          await reloadProfiles();
        }
      } catch (err) {
        showProfileAlert("Network error saving profile: " + err, "danger");
      }
    }

    function showProfileAlert(msg, type) {
      const alertEl = document.getElementById("profile-form-alert");
      alertEl.style.display = "block";
      alertEl.style.background = type === "success" ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)";
      alertEl.style.border = `1px solid var(--${type})`;
      alertEl.style.color = `var(--${type})`;
      alertEl.textContent = msg;
    }

    function hideProfileAlert() {
      const alertEl = document.getElementById("profile-form-alert");
      alertEl.style.display = "none";
    }

    function handleTargetChange() {
      const target = document.getElementById("prof-target").value;
      const hwGroup = document.getElementById("prof-hwarch-group");
      if (hwGroup) {
        hwGroup.style.display = (target === "hailo") ? "block" : "none";
      }
      const modelInput = document.getElementById("prof-base-model");
      if (modelInput && !modelInput.value) {
        if (target === "nvidia") modelInput.value = "yolov9s.pt";
        else if (target === "hailo") modelInput.value = "yolov8m.pt";
        else modelInput.value = "yolov8n.pt";
      }
    }

    function handleRemoteToggle() {
      const isRemote = document.getElementById("prof-remote").checked;
      const remoteFields = document.getElementById("prof-remote-fields");
      if (remoteFields) {
        remoteFields.style.display = isRemote ? "grid" : "none";
      }
    }

    function setFormInputsDisabled(disabled) {
      const banner = document.getElementById("active-run-banner");
      if (banner) banner.style.display = disabled ? "block" : "none";
      const inputs = document.querySelectorAll(".grid > div:first-child input, .grid > div:first-child select");
      inputs.forEach(el => {
        if (el.id !== "autoscroll-chk") {
          el.disabled = disabled;
        }
      });
    }

    function populateFormFromOptions(opt) {
      if (!opt) return;
      if (opt.profile) {
        const sel = document.getElementById("profile-select");
        if (sel) sel.value = opt.profile;
      }
      if (opt.workflow) {
        document.getElementById("workflow-select").value = opt.workflow;
        toggleWorkflowInputs();
      }
      if (opt.custom_checkpoint !== undefined) document.getElementById("custom-checkpoint").value = opt.custom_checkpoint || "";
      if (opt.do_export !== undefined) {
        document.getElementById("do-export").checked = !!opt.do_export;
        toggleCvatTasksBox();
      }
      if (opt.export_tasks) {
        const taskIds = Array.isArray(opt.export_tasks) ? opt.export_tasks.map(String) : String(opt.export_tasks).replace(",", " ").split(/\\s+/).filter(Boolean);
        const checkboxes = document.querySelectorAll("#cvat-task-list input[type='checkbox']");
        const found = [];
        checkboxes.forEach(cb => {
          cb.checked = taskIds.includes(cb.value);
          if (cb.checked) found.push(cb.value);
        });
        const missing = taskIds.filter(id => !found.includes(id));
        if (missing.length > 0) {
          document.getElementById("manual-task-ids").value = taskIds.join(" ");
        }
      }
      if (opt.replace_dataset !== undefined) document.getElementById("replace-dataset").checked = !!opt.replace_dataset;
      if (opt.base_model !== undefined) document.getElementById("base-model").value = opt.base_model || "";
      if (opt.imgsz !== undefined) document.getElementById("imgsz").value = opt.imgsz;
      if (opt.epochs !== undefined) document.getElementById("epochs").value = opt.epochs;
      if (opt.batch !== undefined) document.getElementById("batch").value = opt.batch;
      if (opt.patience !== undefined) document.getElementById("patience").value = opt.patience;
      if (opt.workers !== undefined) document.getElementById("workers").value = opt.workers;
      if (opt.device !== undefined) document.getElementById("device").value = opt.device || "0";
      if (opt.cache !== undefined) document.getElementById("cache").value = opt.cache || "ram";
      if (opt.optimizer !== undefined) document.getElementById("optimizer").value = opt.optimizer || "AdamW";
      if (opt.lr0 !== undefined) document.getElementById("lr0").value = opt.lr0;
      if (opt.lrf !== undefined) document.getElementById("lrf").value = opt.lrf;
      if (opt.spark_host !== undefined) document.getElementById("spark-host").value = opt.spark_host || "";
      if (opt.clean_images !== undefined) document.getElementById("clean-images").checked = !!opt.clean_images;
      if (opt.coco_dataset !== undefined) document.getElementById("coco-dataset").value = opt.coco_dataset;
      if (opt.frigate_plus !== undefined) document.getElementById("frigate-plus").checked = !!opt.frigate_plus;
      if (opt.resume !== undefined) document.getElementById("resume").checked = !!opt.resume;
      if (opt.deploy_target !== undefined) document.getElementById("deploy-target").value = opt.deploy_target || "";
      if (opt.skip_deploy !== undefined) document.getElementById("skip-deploy").checked = !!opt.skip_deploy;
      if (opt.extra_args !== undefined) document.getElementById("extra-args").value = opt.extra_args || "";
      if (opt.log_console !== undefined) document.getElementById("log-console").checked = !!opt.log_console;
      if (opt.log_file !== undefined) document.getElementById("log-file").checked = !!opt.log_file;
    }

    function toggleWorkflowInputs() {
      const wf = document.getElementById("workflow-select").value;
      document.getElementById("custom-ckpt-group").style.display = (wf === "compile_only") ? "block" : "none";
    }

    function toggleCvatTasksBox() {
      const enabled = document.getElementById("do-export").checked;
      document.getElementById("cvat-container").style.display = enabled ? "block" : "none";
    }

    async function fetchCvatTasks() {
      const container = document.getElementById("cvat-task-list");
      container.innerHTML = "<div style='color: var(--text-muted); padding: 0.5rem;'>Connecting to CVAT server...</div>";
      try {
        const res = await fetch("/api/cvat/tasks");
        const data = await res.json();
        if (data.error) {
          container.innerHTML = `<div style="color: var(--danger); padding: 0.5rem;">${data.error}</div>`;
          return;
        }
        cvatTasksList = data.tasks || [];
        renderCvatTasks();
      } catch (err) {
        container.innerHTML = `<div style="color: var(--danger); padding: 0.5rem;">Failed to fetch CVAT tasks: ${err}</div>`;
      }
    }

    function renderCvatTasks() {
      const container = document.getElementById("cvat-task-list");
      if (!cvatTasksList || cvatTasksList.length === 0) {
        container.innerHTML = "<div style='color: var(--text-muted); padding: 0.5rem;'>No CVAT tasks found on server.</div>";
        return;
      }
      container.innerHTML = "";
      cvatTasksList.forEach((t, idx) => {
        const div = document.createElement("div");
        div.className = "cvat-item";
        const jobDesc = t.jobs && t.jobs.length ? `[Job #${t.jobs[0].id}: ${t.jobs[0].status || t.status}]` : `[${t.status}]`;
        div.innerHTML = `
          <input type="checkbox" id="cvat-task-${t.id}" value="${t.id}" ${idx === 0 ? "checked" : ""}>
          <label for="cvat-task-${t.id}">
            <strong>Task #${t.id}:</strong> ${t.name}
            <div class="cvat-meta">${t.size} frames ${jobDesc}</div>
          </label>
        `;
        container.appendChild(div);
      });
    }

    function selectCvatQuick(mode) {
      const checkboxes = document.querySelectorAll("#cvat-task-list input[type='checkbox']");
      checkboxes.forEach((cb, idx) => {
        if (mode === "latest") cb.checked = (idx === 0);
        else if (mode === "all") cb.checked = true;
        else if (mode === "none") cb.checked = false;
      });
    }

    function onProfileSelectChange() {
      const sel = document.getElementById("profile-select");
      if (!isRunning && sel) {
        loadProfileDefaults(sel.value);
      }
    }

    function loadProfileDefaults(profName) {
      const p = (profilesData.profiles || {})[profName];
      if (!p) return;

      const target = p.target || (profName.includes("hailo") ? "hailo" : (profName.includes("nvidia") ? "nvidia" : "torq"));
      const defaultModel = target === "nvidia" ? "yolov9s.pt" : (target === "hailo" ? "yolov8m.pt" : "yolov8n.pt");
      document.getElementById("base-model").value = p.base_model || defaultModel;

      if (p.batch !== undefined) document.getElementById("batch").value = p.batch;
      if (p.epochs !== undefined) document.getElementById("epochs").value = p.epochs;
      if (p.patience !== undefined) document.getElementById("patience").value = p.patience;
      if (p.workers !== undefined) document.getElementById("workers").value = p.workers;
      if (p.device !== undefined) document.getElementById("device").value = p.device;
      if (p.cache !== undefined) document.getElementById("cache").value = p.cache;
      if (p.optimizer !== undefined) document.getElementById("optimizer").value = p.optimizer;
      if (p.lr0 !== undefined) document.getElementById("lr0").value = p.lr0;
      if (p.lrf !== undefined) document.getElementById("lrf").value = p.lrf;
      if (p.clean_images !== undefined) document.getElementById("clean-images").checked = !!p.clean_images;
      document.getElementById("spark-host").value = p.host || "";

      const defaultImg = p.imgsz || (target === "hailo" ? 640 : 320);
      document.getElementById("imgsz").value = defaultImg;

      const dep = p.deployment || {};
      if (dep.host || dep.dir) {
        document.getElementById("deploy-target").value = (dep.host ? dep.host + ":" : "") + (dep.dir || "");
      } else {
        document.getElementById("deploy-target").value = "";
      }

      const log = p.logging || {};
      if (log.console !== undefined) document.getElementById("log-console").checked = !!log.console;
      if (log.file !== undefined) document.getElementById("log-file").checked = !!log.file;
    }

    function toggleConsoleWrap(boxId, wrap) {
      const box = document.getElementById(boxId);
      if (!box) return;
      if (wrap) {
        box.classList.remove("nowrap");
      } else {
        box.classList.add("nowrap");
      }
      try {
        localStorage.setItem(`frigate_studio_${boxId}_wrap`, wrap ? "1" : "0");
      } catch (e) {}
    }

    function setupTerminal() {
      const streamBox = document.getElementById("console-stream");
      const histBox = document.getElementById("hist-console");

      try {
        const savedStreamHeight = localStorage.getItem("frigate_studio_stream_height");
        if (savedStreamHeight && streamBox) {
          streamBox.style.height = savedStreamHeight;
        }
        const savedHistHeight = localStorage.getItem("frigate_studio_hist_height");
        if (savedHistHeight && histBox) {
          histBox.style.height = savedHistHeight;
        }

        const streamWrap = localStorage.getItem("frigate_studio_console-stream_wrap");
        const streamWrapChk = document.getElementById("stream-wrap-chk");
        if (streamWrap !== null && streamWrapChk) {
          const isWrap = streamWrap === "1";
          streamWrapChk.checked = isWrap;
          toggleConsoleWrap("console-stream", isWrap);
        }
        const histWrap = localStorage.getItem("frigate_studio_hist-console_wrap");
        const histWrapChk = document.getElementById("hist-wrap-chk");
        if (histWrap !== null && histWrapChk) {
          const isWrap = histWrap === "1";
          histWrapChk.checked = isWrap;
          toggleConsoleWrap("hist-console", isWrap);
        }
      } catch (e) {
        console.warn("Could not load stored logging frame sizes or wrap preferences:", e);
      }

      if (window.ResizeObserver) {
        const ro = new ResizeObserver((entries) => {
          entries.forEach(entry => {
            const target = entry.target;
            if (target && target.style && target.style.height) {
              try {
                if (target.id === "console-stream") {
                  localStorage.setItem("frigate_studio_stream_height", target.style.height);
                } else if (target.id === "hist-console") {
                  localStorage.setItem("frigate_studio_hist_height", target.style.height);
                }
              } catch (e) {}
            }
          });
        });
        if (streamBox) ro.observe(streamBox);
        if (histBox) ro.observe(histBox);
      }

      if (streamBox) {
        streamBox.addEventListener("scroll", () => {
          const autoChk = document.getElementById("autoscroll-chk");
          if (!autoChk) return;
          const isAtBottom = streamBox.scrollHeight - streamBox.scrollTop - streamBox.clientHeight < 40;
          if (!isAtBottom && autoChk.checked) {
            autoChk.checked = false;
          } else if (isAtBottom && !autoChk.checked) {
            autoChk.checked = true;
          }
        });
      }
    }

    function setupSSE() {
      if (sseSource) sseSource.close();
      sseSource = new EventSource("/api/logs");

      sseSource.addEventListener("clear", () => {
        clearConsole();
      });

      sseSource.addEventListener("replace", (e) => {
        if (!e.data || e.data.startsWith(":")) return;
        updateOrAppendLogLine(e.data, true);
      });

      sseSource.onmessage = (e) => {
        if (!e.data || e.data.startsWith(":")) return;
        updateOrAppendLogLine(e.data, false);
      };

      sseSource.onerror = () => {
        setTimeout(setupSSE, 3000);
      };
    }

    function styleLogElement(el, line) {
      if (line.includes("[ERROR]") || line.includes("Traceback") || line.includes("Error:")) {
        el.style.color = "var(--danger)";
      } else if (line.includes("[WARN]")) {
        el.style.color = "var(--warning)";
      } else if (line.includes("[INFO] Step") || line.includes("COMPLETE")) {
        el.style.color = "var(--success)";
      } else {
        el.style.color = "";
      }
    }

    function updateOrAppendLogLine(line, replace) {
      const box = document.getElementById("console-stream");
      if (replace && box.lastElementChild) {
        const p = box.lastElementChild;
        p.textContent = line;
        styleLogElement(p, line);
      } else {
        const p = document.createElement("p");
        p.textContent = line;
        styleLogElement(p, line);
        box.appendChild(p);
      }

      if (document.getElementById("autoscroll-chk").checked) {
        box.scrollTop = box.scrollHeight;
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        updateUIState(data);
      } catch (err) {
        console.error("Poll status error:", err);
      }
    }

    function updateUIState(data) {
      const previouslyRunning = wasRunning;
      isRunning = !!data.running;
      wasRunning = isRunning;
      currentQueue = data.queue || [];

      if (data.run_id) {
        if (lastSeenRunId !== null && lastSeenRunId !== data.run_id) {
          clearConsole();
        }
        lastSeenRunId = data.run_id;
      }

      const badge = document.getElementById("status-badge");
      const startBtn = document.getElementById("start-btn");
      const abortBtn = document.getElementById("abort-btn");
      const elapsedSpan = document.getElementById("elapsed-time");
      const banner = document.getElementById("active-run-banner");
      const bannerQueueBtn = document.getElementById("banner-queue-btn");
      const queueNextBtn = document.getElementById("queue-next-btn");

      badge.className = "badge badge-" + (data.status || "idle");
      badge.textContent = (data.status || "idle").toUpperCase();

      // Update Queued Run Card
      const queuedCard = document.getElementById("queued-run-card");
      if (queuedCard) {
        if (currentQueue.length > 0) {
          const q = currentQueue[0];
          queuedCard.style.display = "block";
          const qIdEl = document.getElementById("queued-run-id-label");
          const qProfEl = document.getElementById("queued-run-profile-label");
          if (qIdEl) qIdEl.textContent = `#${q.queue_id}`;
          if (qProfEl) qProfEl.textContent = q.profile || "default";

          const opt = q.options || {};
          const wfText = opt.workflow === "compile_only" ? "Compile Only" : (opt.workflow === "skip_compile" ? "Train Only" : (opt.workflow === "pull_only" ? "Pull & Compile" : "Full Pipeline"));
          const summaryParts = [
            `<strong>Workflow:</strong> ${wfText}`,
            opt.base_model ? `<strong>Model:</strong> ${opt.base_model}` : null,
            opt.epochs ? `<strong>Epochs:</strong> ${opt.epochs}` : null,
            opt.batch ? `<strong>Batch:</strong> ${opt.batch}` : null,
            opt.imgsz ? `<strong>ImgSz:</strong> ${opt.imgsz}` : null,
            opt.device ? `<strong>Device:</strong> ${opt.device}` : null,
            opt.do_export ? `<strong>CVAT Export:</strong> ${Array.isArray(opt.export_tasks) && opt.export_tasks.length ? opt.export_tasks.length + ' task(s)' : 'Enabled'}` : null,
          ].filter(Boolean);

          const summaryEl = document.getElementById("queued-run-summary");
          if (summaryEl) summaryEl.innerHTML = summaryParts.join(" &bull; ");

          const startNowBtn = document.getElementById("queued-start-now-btn");
          const noticeEl = document.getElementById("queued-run-status-notice");
          if (!isRunning) {
            if (startNowBtn) startNowBtn.style.display = "inline-block";
            if (noticeEl) noticeEl.innerHTML = `<span>⚡ Current run finished. Queued run is ready to execute.</span>`;
          } else {
            if (startNowBtn) startNowBtn.style.display = "none";
            if (noticeEl) noticeEl.innerHTML = `<span>⏳ Will automatically start when current pipeline execution completes.</span>`;
          }
        } else {
          queuedCard.style.display = "none";
        }
      }

      // Update Active Banner & Queue action buttons
      if (isRunning) {
        if (!previouslyRunning) {
          userManuallyToggledSettings = false;
          if (isMobileWebFormat()) {
            collapseSettingsPanes(true);
          } else {
            collapseSettingsPanes(false);
          }
        }
        if (banner) banner.style.display = "block";
        if (bannerQueueBtn) {
          bannerQueueBtn.style.display = "inline-block";
          bannerQueueBtn.textContent = currentQueue.length > 0 ? "✏️ Edit Queued Run" : "+ Setup Next Run";
        }
        if (queueNextBtn) {
          queueNextBtn.style.display = "block";
          queueNextBtn.textContent = currentQueue.length > 0 ? "✏️ View / Edit Queued Next Run" : "+ Setup & Queue Next Run";
        }
        if (startBtn) startBtn.disabled = true;
        if (abortBtn) abortBtn.style.display = "block";
        if (elapsedSpan) elapsedSpan.textContent = `Running: ${formatDuration(data.elapsed_sec)}`;
      } else {
        if (previouslyRunning) {
          userManuallyToggledSettings = false;
          collapseSettingsPanes(false);
        }
        if (banner) banner.style.display = "none";
        if (bannerQueueBtn) bannerQueueBtn.style.display = "none";
        if (queueNextBtn) queueNextBtn.style.display = "none";
        if (startBtn) startBtn.disabled = false;
        if (abortBtn) abortBtn.style.display = "none";
        if (elapsedSpan) elapsedSpan.textContent = "";
      }

      // If user is currently editing or setting up a queued run, preserve user form input
      if (formMode !== "normal") {
        // Do not touch form inputs
      } else {
        if (isRunning) {
          if (currentActiveRunId !== data.run_id && data.options && Object.keys(data.options).length > 0) {
            currentActiveRunId = data.run_id;
            cachedActiveRunOptions = data.options;
            populateFormFromOptions(data.options);
            setFormInputsDisabled(true);
          }
        } else {
          if (currentActiveRunId !== null) {
            currentActiveRunId = null;
            cachedActiveRunOptions = null;
            setFormInputsDisabled(false);
          }
        }
      }

      const stages = data.stages || {};
      for (const [sid, state] of Object.entries(stages)) {
        const el = document.getElementById("step-" + sid);
        if (el) {
          el.className = "step-item step-" + (state || "pending");
        }
      }

      if (data.history && data.history.length > 0) {
        const tbody = document.getElementById("history-table-body");
        tbody.innerHTML = "";
        data.history.slice().reverse().forEach(run => {
          const tr = document.createElement("tr");
          tr.className = "history-row" + (selectedHistoryRunId === run.run_id ? " active-history-row" : "");
          tr.dataset.runId = run.run_id;
          tr.title = "Click to inspect execution logs for Run #" + run.run_id;
          tr.onclick = () => selectHistoryRun(run.run_id, run.profile, run.status, run.elapsed_sec);
          const artifactsHtml = (run.artifacts || []).map(a => `<div>${a.name} (${a.size})</div>`).join("") || "None";
          tr.innerHTML = `
            <td><strong>${run.run_id}</strong></td>
            <td>${run.profile}</td>
            <td><span class="badge badge-${run.status}">${run.status.toUpperCase()}</span></td>
            <td>${formatDuration(run.elapsed_sec)}</td>
            <td>${artifactsHtml}</td>
          `;
          tbody.appendChild(tr);
        });
      }
    }

    async function selectHistoryRun(runId, profile, status, elapsedSec) {
      selectedHistoryRunId = runId;
      document.querySelectorAll(".history-row").forEach(r => {
        r.classList.toggle("active-history-row", r.dataset.runId === runId);
      });

      const titleEl = document.getElementById("hist-logs-title");
      const badgeEl = document.getElementById("hist-logs-badge");
      const hintEl = document.getElementById("hist-logs-hint");
      const closeBtn = document.getElementById("hist-close-btn");
      const consoleEl = document.getElementById("hist-console");

      const durationStr = (elapsedSec !== undefined && elapsedSec !== null) ? ` [${formatDuration(elapsedSec)}]` : "";
      titleEl.textContent = `Logs: Run #${runId} (${profile})${durationStr}`;
      badgeEl.className = "badge badge-" + (status || "idle");
      badgeEl.textContent = (status || "unknown").toUpperCase();
      badgeEl.style.display = "inline-block";
      hintEl.style.display = "none";
      closeBtn.style.display = "inline-block";

      consoleEl.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Loading logs for run ${runId}...</div>`;

      try {
        const res = await fetch(`/api/history/logs?run_id=${encodeURIComponent(runId)}`);
        const data = await res.json();
        if (data.logs && data.logs.length > 0) {
          consoleEl.innerHTML = "";
          data.logs.forEach(line => {
            const p = document.createElement("p");
            p.textContent = line;
            styleLogElement(p, line);
            consoleEl.appendChild(p);
          });
          consoleEl.scrollTop = consoleEl.scrollHeight;
          requestAnimationFrame(() => {
            consoleEl.scrollTop = consoleEl.scrollHeight;
          });
        } else {
          consoleEl.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">No log output found for run ${runId}.</div>`;
        }
      } catch (err) {
        console.error("Failed to load logs:", err);
        consoleEl.innerHTML = `<div style="color: var(--danger); padding: 1rem;">Failed to load logs: ${err}</div>`;
      }
    }

    function closeHistoryLogs() {
      selectedHistoryRunId = null;
      document.querySelectorAll(".history-row").forEach(r => r.classList.remove("active-history-row"));
      document.getElementById("hist-logs-title").textContent = "Execution Log Viewer";
      document.getElementById("hist-logs-badge").style.display = "none";
      document.getElementById("hist-logs-hint").style.display = "inline";
      document.getElementById("hist-close-btn").style.display = "none";
      document.getElementById("hist-console").innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 3rem 1rem;">Select a past run from the table above to view its console log stream.</div>`;
    }

    function collectFormPayload() {
      const doExport = document.getElementById("do-export").checked;
      return {
        profile: document.getElementById("profile-select").value,
        workflow: document.getElementById("workflow-select").value,
        custom_checkpoint: document.getElementById("custom-checkpoint").value.trim() || null,
        do_export: doExport,
        export_tasks: getSelectedCvatTasks(),
        replace_dataset: document.getElementById("replace-dataset").checked,
        base_model: document.getElementById("base-model").value.trim() || null,
        imgsz: parseInt(document.getElementById("imgsz").value, 10) || null,
        epochs: parseInt(document.getElementById("epochs").value, 10) || null,
        batch: parseInt(document.getElementById("batch").value, 10) || null,
        patience: parseInt(document.getElementById("patience").value, 10) || null,
        workers: parseInt(document.getElementById("workers").value, 10) || null,
        device: document.getElementById("device").value.trim() || null,
        cache: document.getElementById("cache").value,
        optimizer: document.getElementById("optimizer").value,
        lr0: parseFloat(document.getElementById("lr0").value) || null,
        lrf: parseFloat(document.getElementById("lrf").value) || null,
        spark_host: document.getElementById("spark-host").value.trim() || null,
        clean_images: document.getElementById("clean-images").checked,
        coco_dataset: document.getElementById("coco-dataset").value,
        frigate_plus: document.getElementById("frigate-plus").checked,
        resume: document.getElementById("resume").checked,
        deploy_target: document.getElementById("deploy-target").value,
        skip_deploy: document.getElementById("skip-deploy").checked,
        extra_args: document.getElementById("extra-args").value,
        log_console: document.getElementById("log-console").checked,
        log_file: document.getElementById("log-file").checked,
      };
    }

    function enterQueueMode(queueId = null) {
      if (queueId) {
        editingQueueId = queueId;
        formMode = "queue_edit";
        const item = currentQueue.find(q => q.queue_id === queueId);
        if (item && item.options) {
          populateFormFromOptions(item.options);
        }
        const titleEl = document.getElementById("queue-mode-title");
        if (titleEl) titleEl.textContent = `✏️ Edit Queued Run (#${queueId})`;
        const submitBtn = document.getElementById("btn-submit-queue");
        if (submitBtn) submitBtn.textContent = `Update Queued Run (#${queueId})`;
      } else {
        editingQueueId = null;
        formMode = "queue_setup";
        const titleEl = document.getElementById("queue-mode-title");
        if (titleEl) titleEl.textContent = "📋 Setup Next Model Run (Queued)";
        const submitBtn = document.getElementById("btn-submit-queue");
        if (submitBtn) submitBtn.textContent = "Queue Model Run (Starts After Current Run)";
      }

      document.getElementById("queue-mode-banner").style.display = "block";
      document.getElementById("normal-action-buttons").style.display = "none";
      document.getElementById("queue-action-buttons").style.display = "flex";

      collapseSettingsPanes(false);
      setFormInputsDisabled(false);

      const panes = document.getElementById("studio-config-panes");
      if (panes) {
        panes.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    function exitQueueMode() {
      formMode = "normal";
      editingQueueId = null;

      document.getElementById("queue-mode-banner").style.display = "none";
      document.getElementById("queue-action-buttons").style.display = "none";
      document.getElementById("normal-action-buttons").style.display = "block";

      if (isRunning) {
        if (cachedActiveRunOptions) {
          populateFormFromOptions(cachedActiveRunOptions);
        }
        setFormInputsDisabled(true);
        if (isMobileWebFormat() && !userManuallyToggledSettings) {
          collapseSettingsPanes(true);
        }
      } else {
        setFormInputsDisabled(false);
      }
    }

    async function submitQueuedRun() {
      const payload = collectFormPayload();
      if (editingQueueId) {
        payload.queue_id = editingQueueId;
      }

      try {
        const res = await fetch("/api/queue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await res.json();
        if (!res.ok) {
          alert("Error queueing run: " + (result.error || "Failed to queue run"));
          return;
        }

        exitQueueMode();
        pollStatus();
      } catch (err) {
        alert("Network error while queueing run: " + err);
      }
    }

    function editQueuedRun(queueId = null) {
      const qid = queueId || (currentQueue[0] ? currentQueue[0].queue_id : null);
      if (qid) {
        enterQueueMode(qid);
      }
    }

    async function cancelQueuedRun(queueId = null) {
      const qid = queueId || (currentQueue[0] ? currentQueue[0].queue_id : null);
      if (!confirm(`Are you sure you want to remove queued run #${qid || ''} from the queue?`)) return;

      try {
        const res = await fetch("/api/queue/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ queue_id: qid })
        });
        const result = await res.json();
        if (!res.ok) {
          alert("Error: " + (result.error || "Failed to cancel queued run"));
        } else {
          pollStatus();
        }
      } catch (err) {
        alert("Network error: " + err);
      }
    }

    async function startQueuedRunNow() {
      if (!confirm("Start the queued run now?")) return;
      clearConsole();
      try {
        const res = await fetch("/api/queue/start", { method: "POST" });
        const result = await res.json();
        if (!res.ok) {
          alert("Error: " + (result.error || "Failed to start queued run"));
        } else {
          pollStatus();
        }
      } catch (err) {
        alert("Network error: " + err);
      }
    }

    async function startTraining() {
      clearConsole();
      const payload = collectFormPayload();

      try {
        const res = await fetch("/api/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await res.json();
        if (!res.ok) {
          alert("Error: " + (result.error || "Failed to start training"));
        } else {
          userManuallyToggledSettings = false;
          if (isMobileWebFormat()) {
            collapseSettingsPanes(true);
          }
          pollStatus();
        }
      } catch (err) {
        alert("Network error starting training: " + err);
      }
    }

    function getSelectedCvatTasks() {
      const manual = document.getElementById("manual-task-ids").value.trim();
      if (manual) {
        return manual.replace(",", " ").split(/\\s+/).filter(Boolean);
      }
      const checkedBoxes = document.querySelectorAll("#cvat-task-list input[type='checkbox']:checked");
      const ids = [];
      checkedBoxes.forEach(cb => ids.push(cb.value));
      return ids;
    }

    async function abortTraining() {
      if (!confirm("Are you sure you want to stop the current training run?")) return;
      try {
        const res = await fetch("/api/abort", { method: "POST" });
        const result = await res.json();
        if (!res.ok) {
          alert("Error: " + (result.error || "Failed to abort"));
        } else {
          pollStatus();
        }
      } catch (err) {
        alert("Network error: " + err);
      }
    }

    function clearConsole() {
      document.getElementById("console-stream").innerHTML = "";
    }

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""


class StudioHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Secure request handler for Frigate Studio API and Web UI."""

    server_version = "FrigateStudio/1.0"

    def send_security_headers(self, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; img-src 'self' data:;")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/settings"):
            content = get_html_content().encode("utf-8")
            self.send_response(200)
            self.send_security_headers("text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/profiles":
            cfg = config_utils.load_config(self.server.config_path)  # type: ignore[attr-defined]
            raw_profiles = cfg.get("profiles", {})
            resolved_profiles = {}
            for pname, pdata in raw_profiles.items():
                pcopy = dict(pdata) if isinstance(pdata, dict) else {}
                if "imgsz" not in pcopy or not pcopy["imgsz"]:
                    target = str(pcopy.get("target") or ("hailo" if "hailo" in pname.lower() else "torq")).lower()
                    pcopy["imgsz"] = 640 if target == "hailo" else 320
                resolved_profiles[pname] = pcopy

            data = {
                "default_profile": cfg.get("default_profile", "hailo-spark"),
                "profiles": resolved_profiles,
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_security_headers("application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/cvat/tasks":
            cvat_info = fetch_cvat_tasks(self.server.config_path)  # type: ignore[attr-defined]
            body = json.dumps(cvat_info).encode("utf-8")
            self.send_response(200)
            self.send_security_headers("application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/status":
            status = self.server.run_manager.get_status()  # type: ignore[attr-defined]
            body = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_security_headers("application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/queue":
            queue_data = self.server.run_manager.get_queued_runs()  # type: ignore[attr-defined]
            self._send_json({"ok": True, "queue": queue_data})

        elif path == "/api/history/logs":
            query = urllib.parse.parse_qs(parsed.query)
            run_id = query.get("run_id", [""])[0]
            if not run_id:
                self._send_json_error(400, "Query parameter 'run_id' is required.")
                return
            logs = self.server.run_manager.get_run_logs(run_id)  # type: ignore[attr-defined]
            self._send_json({"ok": True, "run_id": run_id, "logs": logs})

        elif path == "/api/logs":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            log_q, history = self.server.run_manager.subscribe()  # type: ignore[attr-defined]
            try:
                for line in history:
                    self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()

                while True:
                    try:
                        item = log_q.get(timeout=1.0)
                        if isinstance(item, tuple):
                            line, replace = item
                        else:
                            line, replace = item, False

                        if replace == "clear":
                            self.wfile.write(b"event: clear\ndata: {}\n\n")
                        elif replace:
                            self.wfile.write(f"event: replace\ndata: {line}\n\n".encode("utf-8"))
                        else:
                            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.server.run_manager.unsubscribe(log_q)  # type: ignore[attr-defined]

        else:
            self.send_response(404)
            self.send_security_headers("text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/train":
            length = int(self.headers.get("Content-Length", 0))
            if length > 1024 * 64:
                self.send_response(413)
                self.end_headers()
                return

            try:
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
            except Exception:
                self._send_json_error(400, "Invalid JSON payload")
                return

            cfg = config_utils.load_config(self.server.config_path)  # type: ignore[attr-defined]
            try:
                cmd_args, summary = build_validated_train_args(payload, cfg)
            except ValueError as e:
                self._send_json_error(400, str(e))
                return

            profile_name = payload.get("profile") or cfg.get("default_profile", "torq")
            success, msg = self.server.run_manager.start_run(cmd_args, profile_name, payload)  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(409, msg)
            else:
                self._send_json({"ok": True, "run_id": msg})

        elif path == "/api/abort":
            success, msg = self.server.run_manager.abort_run()  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(400, msg)
            else:
                self._send_json({"ok": True, "message": msg})

        elif path == "/api/queue":
            length = int(self.headers.get("Content-Length", 0))
            if length > 1024 * 64:
                self.send_response(413)
                self.end_headers()
                return

            try:
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
            except Exception:
                self._send_json_error(400, "Invalid JSON payload")
                return

            cfg = config_utils.load_config(self.server.config_path)  # type: ignore[attr-defined]
            try:
                cmd_args, summary = build_validated_train_args(payload, cfg)
            except ValueError as e:
                self._send_json_error(400, str(e))
                return

            profile_name = payload.get("profile") or cfg.get("default_profile", "torq")
            queue_id = payload.get("queue_id")
            success, action, item = self.server.run_manager.queue_run(cmd_args, profile_name, payload, queue_id=queue_id)  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(400, action)
            else:
                self._send_json({"ok": True, "action": action, "item": item})

        elif path in ("/api/queue/cancel", "/api/queue/delete"):
            length = int(self.headers.get("Content-Length", 0))
            queue_id = None
            if length > 0:
                try:
                    raw_body = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw_body)
                    queue_id = payload.get("queue_id")
                except Exception:
                    pass
            success, msg = self.server.run_manager.cancel_queued_run(queue_id)  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(404, msg)
            else:
                self._send_json({"ok": True, "message": msg})

        elif path == "/api/queue/start":
            success, msg = self.server.run_manager.start_next_queued_run()  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(400, msg)
            else:
                self._send_json({"ok": True, "run_id": msg})

        elif path == "/api/profiles":
            length = int(self.headers.get("Content-Length", 0))
            if length > 1024 * 64:
                self.send_response(413)
                self.end_headers()
                return

            try:
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
            except Exception:
                self._send_json_error(400, "Invalid JSON payload")
                return

            name = sanitize_input(payload.get("name", "")).strip().lower()
            if not name:
                self._send_json_error(400, "Profile name is required.")
                return

            original_name = sanitize_input(payload.get("original_name", "")).strip().lower()
            is_default = bool(payload.get("is_default", False))

            try:
                profile_data = validate_profile_payload(payload.get("profile", {}))
                if original_name and original_name != name:
                    try:
                        config_utils.delete_profile(original_name, self.server.config_path)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                result = config_utils.save_profile(name, profile_data, is_default, self.server.config_path)  # type: ignore[attr-defined]
                self._send_json({"ok": True, "message": f"Profile '{name}' saved successfully.", "result": result})
            except ValueError as e:
                self._send_json_error(400, str(e))
            except Exception as e:
                self._send_json_error(500, f"Failed to save profile: {e}")

        elif path == "/api/profiles/default":
            length = int(self.headers.get("Content-Length", 0))
            try:
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
            except Exception:
                self._send_json_error(400, "Invalid JSON payload")
                return

            name = sanitize_input(payload.get("name", "")).strip().lower()
            if not name:
                self._send_json_error(400, "Profile name is required.")
                return

            try:
                result = config_utils.set_default_profile(name, self.server.config_path)  # type: ignore[attr-defined]
                self._send_json({"ok": True, "default_profile": result})
            except ValueError as e:
                self._send_json_error(400, str(e))
            except Exception as e:
                self._send_json_error(500, f"Failed to set default profile: {e}")

        elif path in ("/api/profiles/delete",):
            length = int(self.headers.get("Content-Length", 0))
            try:
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
            except Exception:
                self._send_json_error(400, "Invalid JSON payload")
                return

            name = sanitize_input(payload.get("name", "")).strip().lower()
            if not name:
                self._send_json_error(400, "Profile name is required.")
                return

            try:
                result = config_utils.delete_profile(name, self.server.config_path)  # type: ignore[attr-defined]
                self._send_json({"ok": True, "message": f"Profile '{name}' deleted.", "result": result})
            except ValueError as e:
                self._send_json_error(400, str(e))
            except Exception as e:
                self._send_json_error(500, f"Failed to delete profile: {e}")

        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/queue":
            query = urllib.parse.parse_qs(parsed.query)
            queue_id = query.get("queue_id", [None])[0]
            success, msg = self.server.run_manager.cancel_queued_run(queue_id)  # type: ignore[attr-defined]
            if not success:
                self._send_json_error(404, msg)
            else:
                self._send_json({"ok": True, "message": msg})

        elif path == "/api/profiles":
            query = urllib.parse.parse_qs(parsed.query)
            name = query.get("name", [""])[0]
            clean_name = sanitize_input(name).strip().lower()
            if not clean_name:
                self._send_json_error(400, "Query parameter 'name' is required.")
                return

            try:
                result = config_utils.delete_profile(clean_name, self.server.config_path)  # type: ignore[attr-defined]
                self._send_json({"ok": True, "message": f"Profile '{clean_name}' deleted.", "result": result})
            except ValueError as e:
                self._send_json_error(400, str(e))
            except Exception as e:
                self._send_json_error(500, f"Failed to delete profile: {e}")
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_security_headers("application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        if len(args) > 0 and isinstance(args[0], str) and ("/api/status" in args[0] or "/api/logs" in args[0]):
            return
        super().log_message(format, *args)


class ThreadingStudioServer(http.server.ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass: Any, run_manager: RunManager, config_path: str):
        super().__init__(server_address, RequestHandlerClass)
        self.run_manager = run_manager
        self.config_path = config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frigate Model Training Studio - Web UI & API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind web server (default: 8080 or config.yaml web_ui.port)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host interface to bind web server (default: 127.0.0.1 or config.yaml web_ui.host)",
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = config_utils.load_config(args.config_path)
    web_cfg = cfg.get("web_ui", {}) if isinstance(cfg.get("web_ui"), dict) else {}

    host = args.host or web_cfg.get("host") or os.environ.get("HOST") or "127.0.0.1"
    port = args.port or web_cfg.get("port") or int(os.environ.get("PORT") or 8080)

    run_manager = RunManager(project_root=PROJECT_ROOT)
    server_address = (host, port)

    try:
        httpd = ThreadingStudioServer(server_address, StudioHTTPRequestHandler, run_manager, args.config_path)
    except OSError as e:
        print(f"[ERROR] Could not start server on {host}:{port} - {e}")
        return 1

    print("==========================================================================")
    print("  Frigate Model Training Studio Web Interface")
    print("==========================================================================")
    print(f"  Access URL : http://{host}:{port}/")
    print(f"  Bind Host  : {host} (Security: local loopback only by default)")
    print(f"  Config Path: {args.config_path}")
    print("  Press Ctrl+C to stop web server.")
    print("==========================================================================")

    def shutdown_handler(signum, frame):
        print("\n[WEB_UI] Shutting down studio server...")
        run_manager.abort_run()
        threading.Thread(target=httpd.shutdown).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
