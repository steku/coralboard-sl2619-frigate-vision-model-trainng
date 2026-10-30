import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import pytest

import web_ui
import config_utils


def test_build_validated_train_args_clean():
    cfg = {"default_profile": "torq", "profiles": {"torq": {}}}
    payload = {
        "profile": "torq",
        "workflow": "full",
        "base_model": "yolov8n.pt",
        "epochs": 50,
        "batch": 16,
        "device": "0",
        "clean_images": True,
        "frigate_plus": True,
        "spark_host": "user@spark-host",
        "coco_dataset": "coco-val2017",
    }
    args, summary = web_ui.build_validated_train_args(payload, cfg)

    assert "-p" in args
    assert "torq" in args
    assert "-m" in args
    assert "yolov8n.pt" in args
    assert "--epochs" in args
    assert "50" in args
    assert "--batch" in args
    assert "16" in args
    assert "--device" in args
    assert "0" in args
    assert "--clean-images" in args
    assert "--frigate-plus" in args
    assert "--host" in args
    assert "user@spark-host" in args
    assert "--coco-dataset" in args
    assert "coco-val2017" in args


def test_build_validated_train_args_security_sanitization():
    cfg = {"default_profile": "torq", "profiles": {"torq": {}}}
    # Attempt command injection in device and base_model
    payload = {
        "profile": "torq",
        "device": "0; rm -rf /",
        "base_model": "yolov8n.pt | whoami",
        "epochs": 10,
    }
    args, summary = web_ui.build_validated_train_args(payload, cfg)
    # Semicolons and pipes should be stripped
    assert ";" not in " ".join(args)
    assert "|" not in " ".join(args)
    assert summary["device"] == "0 rm -rf /"
    assert summary["base_model"] == "yolov8n.pt  whoami"


def test_build_validated_train_args_invalid_integers():
    cfg = {"default_profile": "torq", "profiles": {}}
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        web_ui.build_validated_train_args({"epochs": -5}, cfg)

    with pytest.raises(ValueError, match="Invalid integer value"):
        web_ui.build_validated_train_args({"batch": "not-a-number"}, cfg)


def test_run_manager_stage_progression(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    mgr.active_run = {
        "run_id": "test_run",
        "profile": "torq",
        "start_time": time.time(),
        "stages": {s["id"]: "pending" for s in web_ui.PIPELINE_STAGES},
    }

    # Simulate log lines matching steps
    mgr.add_log_line("[INFO] Step 1: Exporting CVAT Dataset to YOLO Format...")
    assert mgr.active_run["stages"]["export"] == "running"

    # Step 2 begins pretraining & dataset preparation
    mgr.add_log_line("[INFO] Step 2: Training Locally (Device: 0)...")
    assert mgr.active_run["stages"]["export"] == "completed"
    assert mgr.active_run["stages"]["pretrain"] == "running"

    # Training execution begins
    mgr.add_log_line("[INFO] Launching YOLO fine-tuning on host (device=0)...")
    assert mgr.active_run["stages"]["pretrain"] == "completed"
    assert mgr.active_run["stages"]["train"] == "running"

    mgr.add_log_line("[INFO] Step 3: Torq NPU Export, Quantization & Compilation...")
    assert mgr.active_run["stages"]["train"] == "completed"
    assert mgr.active_run["stages"]["compile"] == "running"

    mgr.add_log_line("[INFO] Step 4: Deploying artifacts to local directory...")
    assert mgr.active_run["stages"]["compile"] == "completed"
    assert mgr.active_run["stages"]["deploy"] == "running"

    mgr.add_log_line("[INFO] PIPELINE COMPLETE - MODEL READY FOR NPU INFERENCE!")
    assert mgr.active_run["stages"]["deploy"] == "completed"


def test_web_ui_server_http_endpoints(tmp_path):
    run_manager = web_ui.RunManager(project_root=tmp_path)
    server_address = ("127.0.0.1", 0)  # OS assigns ephemeral port
    httpd = web_ui.ThreadingStudioServer(server_address, web_ui.StudioHTTPRequestHandler, run_manager, "config.yaml")

    port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test GET /
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert "Content-Security-Policy" in resp.headers
            body = resp.read().decode("utf-8")
            assert "Frigate Model Training Studio" in body

        # 2. Test GET /api/profiles
        req = urllib.request.Request(f"{base_url}/api/profiles")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "profiles" in data
            assert "default_profile" in data

        # 3. Test GET /api/status
        req = urllib.request.Request(f"{base_url}/api/status")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            status_data = json.loads(resp.read().decode("utf-8"))
            assert status_data["running"] is False
            assert status_data["status"] == "idle"

        # 4. Test GET /api/cvat/tasks
        req = urllib.request.Request(f"{base_url}/api/cvat/tasks")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            cvat_data = json.loads(resp.read().decode("utf-8"))
            assert "tasks" in cvat_data

    finally:
        httpd.shutdown()
        httpd.server_close()


def test_build_validated_train_args_extended_options():
    cfg = {"default_profile": "torq", "profiles": {"torq": {}}}
    payload = {
        "profile": "torq",
        "do_export": True,
        "export_tasks": ["22", "21"],
        "replace_dataset": True,
        "imgsz": 640,
        "patience": 25,
        "workers": 4,
        "cache": "disk",
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.05,
        "extra_args": "cos_lr=True",
        "log_console": False,
        "log_file": True,
        "log_dir": "custom_logs",
    }
    args, summary = web_ui.build_validated_train_args(payload, cfg)

    assert "--export" in args
    assert "22" in args
    assert "21" in args
    assert "--replace" in args
    assert "--imgsz" in args
    assert "640" in args
    assert "--patience" in args
    assert "25" in args
    assert "--workers" in args
    assert "4" in args
    assert "--cache" in args
    assert "disk" in args
    assert "--optimizer" in args
    assert "SGD" in args
    assert "--lr0" in args
    assert "0.001" in args
    assert "--lrf" in args
    assert "0.05" in args
    assert "--extra-args" in args
    assert "cos_lr=True" in args
    assert "--no-log-console" in args
    assert "--log-file" in args
    assert "--log-dir" in args
    assert "custom_logs" in args


def test_get_status_active_run_options(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    mgr.active_run = {
        "run_id": "test_active",
        "profile": "hailo-spark",
        "start_time": time.time(),
        "status": "running",
        "options": {"epochs": 100, "batch": 32, "device": "0", "profile": "hailo-spark"},
        "stages": {},
    }
    status = mgr.get_status()
    assert status["run_id"] == "test_active"
    assert status["profile"] == "hailo-spark"
    assert status["options"]["epochs"] == 100
    assert status["options"]["batch"] == 32


def test_validate_profile_payload():
    valid_payload = {
        "target": "hailo",
        "remote": True,
        "host": "user@dgx",
        "remote_dir": "~/models",
        "clean_images": True,
        "device": "0",
        "batch": 16,
        "epochs": 80,
        "patience": 25,
        "workers": 8,
        "imgsz": 640,
        "base_model": "yolov8m.pt",
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "cos_lr": True,
        "hw_arch": "hailo10h",
        "deployment": {"host": "root@edge", "dir": "/opt"},
        "logging": {"console": True, "file": False},
    }
    cleaned = web_ui.validate_profile_payload(valid_payload)
    assert cleaned["target"] == "hailo"
    assert cleaned["remote"] is True
    assert cleaned["host"] == "user@dgx"
    assert cleaned["hw_arch"] == "hailo10h"
    assert cleaned["imgsz"] == 640
    assert cleaned["logging"]["console"] is True
    assert cleaned["logging"]["file"] is False

    with pytest.raises(ValueError, match="Target must be one of"):
        web_ui.validate_profile_payload({"target": "unknown", "base_model": "m.pt"})

    with pytest.raises(ValueError, match="base_model is required"):
        web_ui.validate_profile_payload({"target": "torq", "base_model": ""})


def test_run_manager_log_persistence_and_retrieval(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    run_id = "test_run_123"

    # Simulate log line written to disk
    log_file = mgr.logs_dir / f"{run_id}.log"
    log_file.write_text("[10:00:00] [INFO] Starting run\n[10:00:05] [INFO] Finished\n", encoding="utf-8")

    logs = mgr.get_run_logs(run_id)
    assert len(logs) == 2
    assert "[INFO] Starting run" in logs[0]
    assert "[INFO] Finished" in logs[1]

    # Nonexistent run returns notice
    missing_logs = mgr.get_run_logs("nonexistent_run")
    assert any("Detailed logs not available" in l for l in missing_logs)


def test_web_ui_profile_crud_http_endpoints(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """default_profile: torq
profiles:
  torq:
    target: torq
    device: "0"
    base_model: yolov8n.pt
  hailo:
    target: hailo
    device: "0"
    base_model: yolov8m.pt
""",
        encoding="utf-8",
    )

    run_manager = web_ui.RunManager(project_root=tmp_path)
    server_address = ("127.0.0.1", 0)
    httpd = web_ui.ThreadingStudioServer(server_address, web_ui.StudioHTTPRequestHandler, run_manager, str(config_file))

    port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. GET /settings and check HTML contains navigation and scroll container
        req = urllib.request.Request(f"{base_url}/settings")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Settings &amp; Profiles" in html or "Settings & Profiles" in html
            assert "history-scroll-container" in html
            assert "historical-logs-card" in html
            assert "view-settings" in html

        # 2. POST /api/profiles (Create new profile)
        create_payload = {
            "name": "nvidia-test",
            "is_default": True,
            "profile": {
                "target": "nvidia",
                "base_model": "yolov9s.pt",
                "device": "0",
                "batch": 16,
                "epochs": 80,
            },
        }
        post_req = urllib.request.Request(
            f"{base_url}/api/profiles",
            data=json.dumps(create_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(post_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True

        cfg = config_utils.load_config(str(config_file))
        assert "nvidia-test" in cfg["profiles"]
        assert cfg["default_profile"] == "nvidia-test"

        # 3. POST /api/profiles/default (Change default profile)
        default_req = urllib.request.Request(
            f"{base_url}/api/profiles/default",
            data=json.dumps({"name": "torq"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(default_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["default_profile"] == "torq"

        cfg = config_utils.load_config(str(config_file))
        assert cfg["default_profile"] == "torq"

        # 4. DELETE /api/profiles (Delete profile)
        del_req = urllib.request.Request(
            f"{base_url}/api/profiles?name=nvidia-test",
            method="DELETE",
        )
        with urllib.request.urlopen(del_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True

        cfg = config_utils.load_config(str(config_file))
        assert "nvidia-test" not in cfg["profiles"]

        # 5. GET /api/history/logs
        log_file = run_manager.logs_dir / "261030-999999.log"
        log_file.write_text("[12:00:00] [INFO] Sample history line\n", encoding="utf-8")

        hist_req = urllib.request.Request(f"{base_url}/api/history/logs?run_id=261030-999999")
        with urllib.request.urlopen(hist_req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["run_id"] == "261030-999999"
            assert len(data["logs"]) == 1
            assert "Sample history line" in data["logs"][0]

    finally:
        httpd.shutdown()
        httpd.server_close()


def test_run_manager_in_place_log_replacement_and_persistence(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    run_id = "test_run_cr"
    log_file_path = mgr.logs_dir / f"{run_id}.log"
    mgr.current_log_file = open(log_file_path, "w", encoding="utf-8")

    try:
        # 1. Normal line
        mgr.add_log_line("Starting training for 10 epochs...", replace=False)
        assert len(mgr.log_buffer) == 1
        assert "Starting training for 10 epochs..." in mgr.log_buffer[0]

        # 2. Epoch 1 updates in place
        mgr.add_log_line("1/10  10%", replace=False, pending=True)
        assert len(mgr.log_buffer) == 2
        assert "1/10  10%" in mgr.log_buffer[1]

        mgr.add_log_line("1/10  50%", replace=True, pending=True)
        assert len(mgr.log_buffer) == 2
        assert "1/10  50%" in mgr.log_buffer[1]

        mgr.add_log_line("1/10  100%", replace=True, pending=True)
        assert len(mgr.log_buffer) == 2
        assert "1/10  100%" in mgr.log_buffer[1]

        # Finalize epoch line
        mgr.finalize_pending_log()

        # 3. Epoch 1 summary line
        mgr.add_log_line("all 100 0.85", replace=False, pending=False)
        assert len(mgr.log_buffer) == 3
        assert "all 100 0.85" in mgr.log_buffer[2]
    finally:
        if mgr.current_log_file:
            mgr.current_log_file.close()

    # Verify log file on disk only has the finalized lines, not intermediate 10% or 50%
    disk_lines = log_file_path.read_text(encoding="utf-8").splitlines()
    assert len(disk_lines) == 3
    assert "Starting training for 10 epochs..." in disk_lines[0]
    assert "1/10  100%" in disk_lines[1]
    assert "all 100 0.85" in disk_lines[2]


def test_monitor_process_carriage_return_and_newline_in_place(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    run_id = "monitor_test_run"
    log_file_path = mgr.logs_dir / f"{run_id}.log"
    mgr.current_log_file = open(log_file_path, "w", encoding="utf-8")

    # Python process that outputs carriage returns followed by newlines
    script = (
        "import sys\n"
        "sys.stdout.write('Preparing dataset...\\n')\n"
        "sys.stdout.write('Epoch 1: 0/20\\r')\n"
        "sys.stdout.write('Epoch 1: 10/20\\r')\n"
        "sys.stdout.write('Epoch 1: 20/20\\n')\n"
        "sys.stdout.write('Epoch 2: 0/20\\r')\n"
        "sys.stdout.write('Epoch 2: 20/20\\n')\n"
        "sys.stdout.flush()\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    run_info = {"run_id": run_id, "start_time": time.time()}
    mgr._monitor_process(proc, run_info)

    # In log_buffer, Epoch 1 and Epoch 2 should each occupy exactly 1 line
    clean_buffer = [re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", line) for line in mgr.log_buffer]
    assert "Preparing dataset..." in clean_buffer
    assert "Epoch 1: 20/20" in clean_buffer
    assert "Epoch 2: 20/20" in clean_buffer
    # Intermediate 'Epoch 1: 0/20' and 'Epoch 1: 10/20' should have been replaced
    assert "Epoch 1: 0/20" not in clean_buffer
    assert "Epoch 1: 10/20" not in clean_buffer

    # Check file on disk
    disk_lines = [re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", l) for l in log_file_path.read_text(encoding="utf-8").splitlines()]
    assert disk_lines == ["Preparing dataset...", "Epoch 1: 20/20", "Epoch 2: 20/20"]


def test_monitor_process_yolo_tqdm_multi_epoch(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    run_id = "yolo_monitor_test"
    log_file_path = mgr.logs_dir / f"{run_id}.log"
    mgr.current_log_file = open(log_file_path, "w", encoding="utf-8")

    script = (
        "import sys\n"
        "sys.stdout.write('      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\\n')\n"
        "sys.stdout.write('\\r\\x1b[K       1/50      4.77G     0.9802     0.9108      1.098         33        640:  20% 100/500\\r')\n"
        "sys.stdout.write('\\r\\x1b[K       1/50      4.77G     0.9802     0.9108      1.098         33        640: 100% 500/500\\n')\n"
        "sys.stdout.write('\\r\\x1b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 33/33\\n')\n"
        "sys.stdout.write('                   all       1025       1854      0.823      0.834      0.814      0.627\\n')\n"
        "sys.stdout.write('      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\\n')\n"
        "sys.stdout.write('\\r\\x1b[K       2/50      4.77G     0.9777     0.8994      1.098         30        640:  40% 200/500\\r')\n"
        "sys.stdout.write('\\r\\x1b[K       2/50      4.77G     0.9777     0.8994      1.098         30        640: 100% 500/500\\n')\n"
        "sys.stdout.flush()\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    run_info = {"run_id": run_id, "start_time": time.time()}
    mgr._monitor_process(proc, run_info)

    clean_buffer = [re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", line) for line in mgr.log_buffer]
    # Verify Epoch 1 and Epoch 2 are BOTH in clean_buffer
    assert any("1/50" in line and "100%" in line for line in clean_buffer)
    assert any("2/50" in line and "100%" in line for line in clean_buffer)
    assert any("Class" in line for line in clean_buffer)
    assert any("all" in line and "0.823" in line for line in clean_buffer)
    # Intermediate 20% and 40% should NOT be present
    assert not any("20% 100/500" in line for line in clean_buffer)
    assert not any("40% 200/500" in line for line in clean_buffer)


def test_sse_replace_events_streaming(tmp_path):
    mgr = web_ui.RunManager(project_root=tmp_path)
    server_address = ("127.0.0.1", 0)
    httpd = web_ui.ThreadingStudioServer(server_address, web_ui.StudioHTTPRequestHandler, mgr, "config.yaml")
    port = httpd.server_address[1]

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # Pre-populate 1 normal line
        mgr.add_log_line("Initial Log Line", replace=False)

        # Open SSE stream in a thread
        received_chunks = []
        stop_event = threading.Event()

        def stream_reader():
            req = urllib.request.Request(f"{base_url}/api/logs")
            with urllib.request.urlopen(req) as resp:
                while not stop_event.is_set():
                    line = resp.readline().decode("utf-8")
                    if not line:
                        break
                    received_chunks.append(line)

        reader_thread = threading.Thread(target=stream_reader, daemon=True)
        reader_thread.start()

        # Allow SSE connection to initialize
        time.sleep(0.2)

        # Emit an in-place update
        mgr.add_log_line("Progress In-Place Line", replace=True)
        time.sleep(0.2)
        stop_event.set()

        joined = "".join(received_chunks)
        assert "data: " in joined
        assert "Initial Log Line" in joined
        assert "event: replace" in joined
        assert "Progress In-Place Line" in joined
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_profiles_resolves_imgsz_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "default_profile: hailo-spark\n"
        "profiles:\n"
        "  torq:\n"
        "    target: torq\n"
        "  hailo-spark:\n"
        "    target: hailo\n"
        "    imgsz: 640\n"
        "  custom:\n"
        "    target: torq\n"
        "    imgsz: 480\n",
        encoding="utf-8",
    )

    run_manager = web_ui.RunManager(project_root=tmp_path)
    server_address = ("127.0.0.1", 0)
    httpd = web_ui.ThreadingStudioServer(server_address, web_ui.StudioHTTPRequestHandler, run_manager, str(config_file))
    port = httpd.server_address[1]

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        req = urllib.request.Request(f"{base_url}/api/profiles")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            profiles = data["profiles"]
            # Torq defaults to 320 when omitted
            assert profiles["torq"]["imgsz"] == 320
            # Hailo defaults to 640
            assert profiles["hailo-spark"]["imgsz"] == 640
            # Explicit value is preserved
            assert profiles["custom"]["imgsz"] == 480

        # Also verify HTML content contains dynamic profile select event and correct resolution
        html_req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(html_req) as resp:
            html = resp.read().decode("utf-8")
            assert 'onchange="onProfileSelectChange()"' in html
            assert "p.imgsz || (target === \"hailo\" ? 640 : 320)" in html
            assert "resize: vertical;" in html
            assert "min-height: 160px;" in html
            assert "ResizeObserver" in html
            assert "frigate_studio_stream_height" in html
            assert "consoleEl.scrollTop = consoleEl.scrollHeight;" in html
            assert "streamBox.addEventListener(\"scroll\"" in html
            assert 'id="step-pretrain"' in html
    finally:
        httpd.shutdown()
        httpd.server_close()



