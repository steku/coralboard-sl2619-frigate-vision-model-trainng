from pathlib import Path
import pytest

import train


def test_format_duration():
    assert train.format_duration(45) == "45s"
    assert train.format_duration(125) == "2m 05s (125s)"
    assert train.format_duration(3665) == "1h 01m 05s (3665s)"


def test_relocate_root_checkpoints(tmp_path):
    root_dir = tmp_path
    base_dir = tmp_path / "models" / "base"

    ckpt1 = root_dir / "test1.pt"
    ckpt1.write_text("data1")

    other_file = root_dir / "other.txt"
    other_file.write_text("text")

    train.relocate_root_checkpoints(root_dir, base_dir)

    assert not ckpt1.exists()
    assert (base_dir / "test1.pt").is_file()
    assert other_file.is_file()


def test_parse_arguments_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py"])
    args = train.parse_arguments()
    assert args.profile is None
    assert args.compile_model is None
    assert args.export_tasks is None
    assert args.deploy_target is None
    assert args.skip_compile is False
    assert args.skip_deploy is False
    assert args.pull_only is False
    assert args.clean_images is None
    assert args.frigate_plus is None


def test_parse_arguments_overrides(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "-p", "torq-spark",
        "--compile", "staging/weights/best.pt",
        "--export", "10", "11",
        "--deploy", "root@192.168.1.50:/models",
        "--skip-deploy",
        "--clean-images",
        "--frigate-plus",
        "--epochs", "50",
        "--batch", "16",
        "--device", "1",
    ])
    args = train.parse_arguments()
    assert args.profile == "torq-spark"
    assert args.compile_model == "staging/weights/best.pt"
    assert args.export_tasks == ["10", "11"]
    assert args.deploy_target == "root@192.168.1.50:/models"
    assert args.skip_deploy is True
    assert args.clean_images is True
    assert args.frigate_plus is True
    assert args.epochs == 50
    assert args.batch == 16
    assert args.device == "1"


def test_main_missing_checkpoint(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--compile", "definitely_not_a_real_model.pt"])
    ret = train.main()
    assert ret == 1


def test_remote_train_cmd_syntax():
    import subprocess
    resolved_remote_dir = "/home/stephen/frigate-model-training"
    base_model = "yolov8n.pt"
    run_name = "yolov8n"
    epochs = 100
    imgsz = 320
    batch = 32
    workers = 8
    cache = "ram"
    patience = 30
    optimizer = "AdamW"
    lr0 = 0.0005
    lrf = 0.01
    cos_lr = True
    warmup_epochs = 5.0
    warmup_momentum = 0.5
    warmup_bias_lr = 0.0
    device = "0"
    extra_args = ""
    from_scratch = False

    cmd = (
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
        f"name='{run_name}' exist_ok=True {extra_args}"
    )

    # Verify bash parses command without syntax errors
    proc = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0, f"Syntax error in remote cmd: {proc.stderr}"


def test_remote_clean_images_preserves_coco(tmp_path):
    import subprocess
    resolved_remote_dir = tmp_path
    datasets_dir = resolved_remote_dir / "datasets"

    # Simulate remote dataset contents
    coco_dir = datasets_dir / "coco" / "images"
    coco_dir.mkdir(parents=True, exist_ok=True)
    (coco_dir / "val1.jpg").write_text("coco img")

    coco128_dir = datasets_dir / "coco128" / "images"
    coco128_dir.mkdir(parents=True, exist_ok=True)
    (coco128_dir / "img128.jpg").write_text("coco128 img")

    yolo_dir = datasets_dir / "yolo" / "images"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    (yolo_dir / "train1.jpg").write_text("yolo img")

    fp_dir = datasets_dir / "frigate-plus" / "images"
    fp_dir.mkdir(parents=True, exist_ok=True)
    (fp_dir / "fp1.jpg").write_text("fp img")

    data_yaml = datasets_dir / "data.yaml"
    data_yaml.write_text("train: yolo\n")

    # Command from train.py
    cleanup_cmd = f"rm -rf \"{resolved_remote_dir}/datasets/yolo\" \"{resolved_remote_dir}/datasets/frigate-plus\" \"{resolved_remote_dir}/datasets/data.yaml\" 2>/dev/null || true"

    proc = subprocess.run(["bash", "-c", cleanup_cmd], capture_output=True, text=True)
    assert proc.returncode == 0

    # Ensure yolo, frigate-plus, and data.yaml are deleted
    assert not yolo_dir.exists()
    assert not fp_dir.exists()
    assert not data_yaml.exists()

    # Ensure coco and coco128 datasets are preserved
    assert coco_dir.exists()
    assert (coco_dir / "val1.jpg").is_file()
    assert coco128_dir.exists()
    assert (coco128_dir / "img128.jpg").is_file()


def test_ensure_environment_defaults(monkeypatch):
    import os
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    train.ensure_environment_defaults()

    assert os.environ.get("USER") == "root"
    assert os.environ.get("USERNAME") == "root"
    assert os.environ.get("LOGNAME") == "root"
    assert os.environ.get("HOME") is not None


def test_update_model_symlinks_success(tmp_path):
    target_dir = tmp_path / "models" / "hailo"
    run_dir = target_dir / "261104-1200"
    run_dir.mkdir(parents=True, exist_ok=True)

    hef_file = run_dir / "hailo_yolov8m_261104-1200.hef"
    hef_file.write_text("binary content")
    json_file = run_dir / "hailo_yolov8m_261104-1200.json"
    json_file.write_text("{}")
    txt_file = run_dir / "hailo_yolov8m_261104-1200.txt"
    txt_file.write_text("0 person")

    links = train.update_model_symlinks(
        models_target_dir=target_dir,
        target_format="hailo",
        run_name_stem="yolov8m",
        base_model_name="yolov8m",
        primary_artifact=hef_file,
        labels_file=json_file,
        timestamp="261104-1200",
    )

    assert "model" in links
    assert (target_dir / "hailo_yolov8m.hef").is_symlink()
    assert (target_dir / "hailo_yolov8m.hef").resolve() == hef_file.resolve()
    assert (target_dir / "hailo_yolov8m.json").is_symlink()
    assert (target_dir / "hailo_yolov8m.json").resolve() == json_file.resolve()
    assert (target_dir / "hailo_yolov8m.txt").is_symlink()
    assert (target_dir / "hailo_yolov8m.txt").resolve() == txt_file.resolve()
    assert (target_dir / "labels.json").is_symlink()
    assert (target_dir / "labels.txt").is_symlink()


def test_update_model_symlinks_missing_artifacts(tmp_path):
    target_dir = tmp_path / "models" / "hailo"
    run_dir = target_dir / "261104-1200"
    run_dir.mkdir(parents=True, exist_ok=True)

    missing_hef = run_dir / "missing.hef"
    json_file = run_dir / "labels.json"
    json_file.write_text("{}")

    # If primary artifact doesn't exist (e.g. compilation failed), symlinks must not be created
    links = train.update_model_symlinks(
        models_target_dir=target_dir,
        target_format="hailo",
        run_name_stem="yolov8m",
        base_model_name="yolov8m",
        primary_artifact=missing_hef,
        labels_file=json_file,
        timestamp="261104-1200",
    )

    assert links == {}
    assert not (target_dir / "hailo_yolov8m.hef").exists()
    assert not (target_dir / "labels.json").exists()





