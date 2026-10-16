# Frigate Model Training & Coralboard SL2619 Torq NPU Pipeline

End-to-end pipeline to harvest detection events from **Frigate NVR**, annotate and review them in **[CVAT (Computer Vision Annotation Tool)](https://github.com/cvat-ai/cvat)**, export the dataset to **YOLO format**, train a **YOLOv8n ($320 \times 320$)** or **YOLO26n** model, and compile it for real-time hardware acceleration on the **Synaptics Coralboard SL2619 Torq NPU** running [`coralboard-sl2619-vision`](https://github.com/steku/coralboard-sl2619-vision.git).

Validated against official Synaptics documentation:
- [Deploy a Custom YOLO Model on the Synaptics Torq NPU](https://developer.synaptics.com/blog/custom-yolo-to-torq)
- [Synaptics Torq Compiler User Manual (v2.2.1+)](https://synaptics-torq.github.io/torq-compiler/v/latest/)

---

## Architecture Overview

```
┌─────────────────┐       frigate-to-cvat.py          ┌─────────────────┐       export-cvat-to-yolo.py        ┌─────────────────┐
│   Frigate NVR   │ ────────────────────────────────> │   Local CVAT    │ ──────────────────────────────────> │  YOLO Dataset   │
│  (Events API)   │   (Deficit-Weighted Harvest)      │ (Annotation/QA) │      (Export train/val/yaml)        │ (320x320 data)  │
└─────────────────┘                                   └────────┬────────┘                                     └────────┬────────┘
                                                               │                                                       │
                                                               │ report-cvat-balance.py                                │
                                                               ▼                                                       ▼
                                                      ┌─────────────────┐                             ┌────────────────┴────────────────┐
                                                      │  Balance Audit  │                             │            train.sh             │
                                                      │ (Health Ratios) │                             │(Train ──> ONNX ──> INT8 ──> VMFB│
                                                      └─────────────────┘                             └────────────────┬────────────────┘
                                                                                                                       │
                                                                                                                       │ deploy_to_coralboard.sh
                                                                                                                       ▼
┌─────────────────────────┐                            test_server_accuracy.py                        ┌─────────────────────────────────┐
│ Coralboard SL2619 Proxy │ <──────────────────────────────────────────────────────────────────────── │ Coralboard SL2619 Proxy Service │
│ (Precision/Recall/F1)   │                     (Benchmark Accuracy & Latency)                        │ (.vmfb binary & labels.json)    │
└─────────────────────────┘                                                                           └─────────────────────────────────┘
```

### What is CVAT?

**[CVAT (Computer Vision Annotation Tool)](https://github.com/cvat-ai/cvat)** is an open-source, web-based digital media annotation platform originally developed by Intel and maintained by CVAT.ai. In this training pipeline, CVAT acts as the central **human-in-the-loop quality assurance and annotation engine**:

* **Automated Task Ingestion**: [`frigate-to-cvat.py`](frigate-to-cvat.py) downloads harvested Frigate NVR snapshots and automatically uploads them as new review tasks pre-annotated with bounding boxes using the official Python [`cvat-sdk`](https://github.com/cvat-ai/cvat/tree/develop/cvat-sdk).
* **Interactive Web QA**: Annotators log into the CVAT web interface (`http://<CVAT_HOST>:8080`) to inspect bounding boxes, fix false positives, adjust spatial boundaries, and label previously undetected objects.
* **Distribution Auditing**: [`report-cvat-balance.py`](report-cvat-balance.py) inspects annotation counts and class balance across tasks via the CVAT REST API before training begins.
* **Direct YOLO Dataset Export**: [`export-cvat-to-yolo.py`](export-cvat-to-yolo.py) connects to CVAT's API, downloads verified annotations, partitions them into train/val splits, and formats them for YOLO model training.

### Hardware Acceleration & NPU Vector Slices (NSS)

The **Astra SL2610 / SL2619 Torq NPU** utilizes physical hardware vector slices (NSS) backed by high-speed Local SRAM (**LRAM**, ~256 KB per slice).
- **$320 \times 320$ INT8 Model**: The entire model executes natively inside the vector slices without LRAM overflow (~13 ms inference, ~78 FPS, pure INT8 I/O).
- **Zero Fallback**: Does **not** require `--torq-disable-slices`. Models that disable slices fall back to the RISC-V CSS microcontroller or host CPU, degrading latency from milliseconds to minutes.

---

## Prerequisites & Setup

All tasks across the entire lifecycle run in a **single unified Python 3.12 virtual environment**. Python 3.12 is strictly required for binary wheel and C++ extension compatibility with the Synaptics Torq compiler toolchain (`tosa-converter-for-tflite`, `torq-compile`, `iree-compiler`, and `onnx2tf`).

### 1. Install Python 3.12 & Create the Virtual Environment

Ensure Python 3.12 and its venv packages are installed on your host system:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Alternatively, using pyenv:
pyenv install 3.12.8
pyenv local 3.12.8
```

Create and activate the dedicated virtual environment in the repository root:

```bash
# Create the virtual environment using Python 3.12
python3.12 -m venv venv-torq-312

# Activate the virtual environment
source venv-torq-312/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Toolchain Installation

Confirm that PyTorch, Ultralytics, CVAT SDK, ONNX, onnx2tf, and the Torq compiler are functioning:

```bash
python -c "import torch, ultralytics, cvat_sdk, onnx, tensorflow, onnx2tf; print('All Python dependencies ready!')"
tosa-converter-for-tflite --version || tosa-converter-for-tflite -h
torq-compile --version
```

### 4. Run the Automated Test Suite

Run the unit test suite to verify configuration parsers, deficit balancing algorithms, server accuracy calculation, and dataset conversions:

```bash
pytest
```

---

## Central Configuration (`config.yaml`)

Configuration settings are stored in `config.yaml`. Copy the provided [`config.example.yaml`](config.example.yaml) to get started:

```bash
cp config.example.yaml config.yaml
```

> [!NOTE]
> `config.yaml` is ignored by Git to keep passwords and internal network endpoints private.

```yaml
# ------------------------------------------------------------------------------
# Frigate NVR Settings
# ------------------------------------------------------------------------------
frigate:
  url: "http://localhost:5000"
  camera: null                  # Camera name filter (e.g. "driveway") or null for all
  limit: 200                    # Maximum events to harvest per run
  clean: false                  # Delete existing images in output directory before downloading
  snapshot_endpoint: "snapshot-clean.webp" # "snapshot-clean.webp" or "snapshot.jpg"
  skip_processed: true          # Skip events already tracked in .processed_events.json
  balance_mode: "cvat-deficit"  # "cvat-deficit", "accuracy-deficit", "equal-split", "strict-equal", "natural"
  accuracy_test_images: 60      # Number of random images evaluated when balance_mode is accuracy-deficit
  accuracy_coco_images: 0       # Number of COCO images to include in accuracy benchmark
  accuracy_metric: "f1_score"   # Metric used for deficit quotas: "f1_score", "recall", "precision", "mean_iou"
  accuracy_report: null         # Optional path to cached JSON report to skip live benchmark
  timeout: 10

# ------------------------------------------------------------------------------
# CVAT Server Settings
# ------------------------------------------------------------------------------
cvat:
  url: "http://<CVAT_HOST>:8080"
  username: "your_cvat_username"
  password: "your_cvat_password"
  task_name_format: "Frigate Batch Review ({timestamp})"

# Target detection categories harvested from Frigate and labeled in CVAT
labels:
  - person
  - dog
  - car
  - horse
  - bird
  - cat
  - truck

storage:
  output_dir: "./datasets/staging"

# ------------------------------------------------------------------------------
# Model Deployment Settings
# ------------------------------------------------------------------------------
deployment:
  host: "root@<CORALBOARD_IP>:/home/root/coralboard-sl2619-vision/models"
  dir: null

# ------------------------------------------------------------------------------
# Execution Profiles (train.sh -p <name>)
# ------------------------------------------------------------------------------
default_profile: torq

profiles:
  torq:                         # Workstation local GPU training for Torq NPU (Default)
    target: "torq"
    remote: false
    device: "0"
    batch: 8
    workers: 8
    cache: "ram"
    epochs: 80
    patience: 50
    base_model: "yolov8n.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0

  spark-torq:                   # Remote NVIDIA DGX / Spark server training via SSH for Torq NPU
    target: "torq"
    remote: true
    host: "user@dgx-spark.local"
    remote_dir: "~/frigate-model-training"
    device: "0"
    batch: 64
    workers: 8
    cache: "ram"
    epochs: 300
    patience: 50
    base_model: "yolov8n.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0
    clean_images: false

  spark-torq-yolo26n:           # Remote Spark server training using yolo26n architecture for Torq NPU
    target: "torq"
    remote: true
    host: "user@dgx-spark.local"
    remote_dir: "~/frigate-model-training"
    device: "0"
    batch: 64
    workers: 8
    cache: "ram"
    epochs: 300
    patience: 50
    base_model: "yolo26n.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0
    clean_images: false

  cpu-torq:                     # Local CPU training fallback for Torq NPU
    target: "torq"
    remote: false
    device: "cpu"
    batch: 16
    workers: 4
    cache: "ram"
    epochs: 80
    patience: 50
    base_model: "yolov8n.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0

  nvidia:                       # Local NVIDIA GPU training & ONNX export (skips Torq .vmfb compilation)
    remote: false
    target: "nvidia"
    device: "0"
    batch: 16
    workers: 8
    cache: "ram"
    epochs: 80
    patience: 50
    base_model: "yolov9s.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0

  spark-nvidia:                 # Remote Spark GPU training & ONNX export for NVIDIA GPU deployment
    remote: true
    target: "nvidia"
    host: "user@dgx-spark.local"
    remote_dir: "~/frigate-model-training"
    device: "0"
    batch: 64
    workers: 8
    cache: "ram"
    epochs: 300
    patience: 50
    base_model: "yolov9s.pt"
    optimizer: "AdamW"
    lr0: 0.0005
    lrf: 0.01
    cos_lr: true
    warmup_epochs: 5.0
    warmup_momentum: 0.5
    warmup_bias_lr: 0.0
    clean_images: false
```

---

## Canonical COCO-80 Domain Adaptation & Gentle Fine-Tuning

Because the base checkpoints (`yolov8n.pt` and `yolo26n.pt`) are already pre-trained on the exact **canonical 80 COCO classes**, downloading and co-training on external COCO archives (`coco-val2017` / `coco128`) is unnecessary. Instead, the pipeline preserves 100% of the pre-trained backbone, neck, and classification head weights and gently adapts them to your Frigate camera angles, lighting conditions, and $320 \times 320$ resolution:

1. **Canonical COCO-80 ID Alignment (`nc = 80`)**:
   [`export-cvat-to-yolo.py`](export-cvat-to-yolo.py) and [`config_utils.prepare_training_data_yaml`](config_utils.py) map all CVAT labels to canonical COCO-80 indices (`person` = 0, `car` = 2, `dog` = 16, etc.) and emit the composite dataset manifest at `datasets/data.yaml` referencing your local CVAT splits, Frigate+ dataset, and optional COCO reference images:
   ```yaml
   train:
     - /path/to/datasets/yolo/images/train
     - /path/to/datasets/frigate-plus/images/train
     - /path/to/datasets/coco/images/val2017
   val:
     - /path/to/datasets/yolo/images/val
     - /path/to/datasets/frigate-plus/images/val
     - /path/to/datasets/coco/images/val2017
   names:
     0: person
     1: bicycle
     ...
     79: toothbrush
   ```
   This guarantees Ultralytics transfers the entire pre-trained detection head (`cv3` / `one2one_cv3`) without re-initializing output layers, while ensuring validation mAP, early stopping (`patience`), and INT8 PTQ calibration (`prepare_for_onnx2tf.py`) are computed purely on your real Frigate camera snapshots.

2. **Explicit Optimizer & Low Initial Learning Rate (`optimizer: AdamW`, `lr0: 0.0005`)**:
   In Ultralytics, leaving `optimizer: auto` causes `BaseTrainer.build_optimizer` to **ignore `lr0`** and switch to `MuSGD(lr=0.01)` with a $3\times$ classification head LR boost (`0.03`) when iterations exceed 10,000. Explicitly setting `optimizer: AdamW`, `lr0: 0.0005`, `lrf: 0.01`, and `cos_lr: true` enforces a $20\times$ lower initial learning rate with smooth cosine decay so pre-existing COCO-80 knowledge is not warped.

3. **Strong Warm-Up Phase (`warmup_epochs: 5.0`, `warmup_momentum: 0.5`, `warmup_bias_lr: 0.0`)**:
   Ramps all parameter groups linearly from `0.0` to `lr0` over 5 epochs (avoiding Ultralytics' default `warmup_bias_lr: 0.1`, which is $200\times$ higher than `lr0` and destabilizes pre-trained head biases during epoch 1).

---

## Step-by-Step Workflow

### Step 1: Harvest Events from Frigate & Upload to CVAT (`frigate-to-cvat.py`)

Harvest detection events and snapshots from Frigate NVR, generate COCO pre-annotations, and upload a review task directly to CVAT:

```bash
# Standard harvest using config.yaml defaults:
python frigate-to-cvat.py

# Accuracy-guided harvesting: tests vision server accuracy and allocates higher quotas to underperforming classes:
python frigate-to-cvat.py --balance-mode accuracy-deficit --limit 150 --accuracy-test-images 60

# Include COCO reference images in accuracy benchmark:
python frigate-to-cvat.py --balance-mode accuracy-deficit --limit 150 --accuracy-coco-images 20

# Use a precomputed JSON accuracy report to skip running the live benchmark:
python frigate-to-cvat.py --accuracy-report report.json --limit 100

# Equal-split harvest (divides limit equally across all configured classes):
python frigate-to-cvat.py --balance-mode equal-split --limit 140

# Strict balance: caps all classes to the count of the rarest class for exact 1:1 balance:
python frigate-to-cvat.py --balance-mode strict-equal --limit 150

# Filter by camera or override labels:
python frigate-to-cvat.py --camera driveway --labels person car --limit 100

# Local staging only (skip uploading to CVAT):
python frigate-to-cvat.py --no-upload

# Upload existing ./datasets/staging folder directly to CVAT:
python frigate-to-cvat.py --upload-only
```

Once uploaded, the script outputs the CVAT task URL:
```text
======================================================================
 [SUCCESS] CVAT Task Successfully Created!
 Task ID  : 4
 Task Name: Frigate Batch Review (2026-10-09 12:00)
 URL      : http://<CVAT_HOST>:8080/tasks/4
======================================================================
```

#### Accuracy-Guided Harvesting (Active Learning via `test_server_accuracy.py`)

Surveillance cameras routinely capture hundreds of repetitive snapshots of frequent scenes (such as static parked cars or familiar walkways). Blindly harvesting arbitrary events wastes human labeling effort without addressing edge cases or detection failures.

**Accuracy-guided harvesting** establishes a closed-loop feedback mechanism between deployed edge inference and dataset collection. By integrating [`test_server_accuracy.py`](test_server_accuracy.py) directly into [`frigate-to-cvat.py`](frigate-to-cvat.py), the pipeline evaluates the live Torq NPU proxy server, identifies classes with poor detection accuracy (lowest F1-score, precision, or recall), and dynamically allocates the largest Frigate harvesting quotas to those underperforming classes.

```
┌────────────────────────────────┐
│   Coralboard Torq NPU Server   │ <────────┐
│  http://<CORALBOARD_IP>:5000   │          │
└───────────────┬────────────────┘          │ 1. In-process benchmark
                │                           │    (test_server_accuracy.py)
                ▼                           │
┌────────────────────────────────┐          │
│   run_accuracy_benchmark()     │ ─────────┘
│   (Calculates per-class F1)    │
└───────────────┬────────────────┘
                │ 2. In-memory metric dictionary (score, gt, tested)
                ▼
┌────────────────────────────────┐
│  extract_accuracy_scores()     │ ───> Disregards untested/0-score classes (gt == 0)
└───────────────┬────────────────┘
                │ 3. Inverse deficit calculation: deficit = max(1.0, 100 - score)
                ▼
┌────────────────────────────────┐
│ compute_accuracy_balanced_     │ ───> Allocates largest quota to weakest classes
│ quotas()                       │ ───> Sorts query order: lowest score first
└───────────────┬────────────────┘
                │ 4. Targeted event queries with dynamic shortfall rollover
                ▼
┌────────────────────────────────┐
│       Frigate NVR Events       │ ───> Uploads weak-class batch to CVAT
└────────────────────────────────┘
```

##### How the Integration Works Under the Hood

1. **Direct In-Memory Benchmark Execution**:
   When `--balance-mode accuracy-deficit` is specified (or set as `frigate.balance_mode: "accuracy-deficit"` in `config.yaml`), [`frigate-to-cvat.py`](frigate-to-cvat.py) dynamically imports [`test_server_accuracy.py`](test_server_accuracy.py) and invokes `test_server_accuracy.run_accuracy_benchmark()` in-process:
   * Sends multipart detection requests to the Coralboard vision server (`--server-url`, default: `http://<CORALBOARD_IP>:5000/v1/vision/detection`).
   * Evaluates `--accuracy-test-images` (default: 60) against ground truth from `datasets/yolo/`.
   * Optionally blends `--accuracy-coco-images N` from standard COCO datasets (`coco128` or `coco-val2017`) to anchor evaluation on general object classes.
   * Prints the complete tabular accuracy report to `stdout` and returns structured per-class metrics directly in memory as a Python dictionary (no intermediate disk serialization required).

2. **Offline & Cached Report Ingestion**:
   You can provide `--accuracy-report <path/to/report.json>`:
   * **Existing Report**: If the file exists, `frigate-to-cvat.py` skips running the live network benchmark and ingests the saved class metrics directly. This allows running offline harvests or reusing comprehensive benchmarks (`python test_server_accuracy.py --all --output-json report.json`).
   * **Missing Report**: If the file does not exist, the live benchmark executes normally and automatically persists the evaluated metrics to that JSON path for subsequent runs.

3. **Disregarding Untested & Zero-Score Classes**:
   Any class with `gt == 0`, `tested == False`, or an accuracy score of `0.0` represents an unmeasured class in the test split. Instead of erroneously treating untested classes as 100% defective, the integration explicitly disregards them from the deficit pool. Only classes with real, measured performance gaps receive deficit quotas.

4. **Deficit Weighting & Prioritized Quota Calculation**:
   For each measured class with valid detections:
   * An inverse deficit is computed based on `--accuracy-metric` (options: `f1_score` [default], `recall`, `precision`, `mean_iou`):
     $$\text{deficit}_c = \max(1.0, 100.0 - \text{score}_c)$$
   * Classes are tagged with health statuses: `CRITICAL` (<40%), `WEAK` (<70%), or `GOOD` (≥70%).
   * The total harvest limit (`--limit`, e.g., 150) is partitioned proportionally to each class's deficit score. Underperforming classes receive the bulk of the harvest quota, while healthy classes receive smaller baseline maintenance quotas.
   * Target classes are queried from Frigate in **ascending order of score** (lowest-performing class queries Frigate first).

5. **Dynamic Shortfall Rollover**:
   If Frigate's event database does not contain enough unharvested events for a particular weak class to satisfy its allocated quota, the unused quota **automatically rolls over** and is distributed proportionally among the remaining deficit classes, ensuring that the total requested `--limit` is always fulfilled.

##### CLI Examples

```bash
# Live benchmark against NPU proxy server and harvest 150 events weighted towards weak classes:
python frigate-to-cvat.py --balance-mode accuracy-deficit --limit 150 --accuracy-test-images 60

# Target missed detections by calculating deficits from recall:
python frigate-to-cvat.py --balance-mode accuracy-deficit --accuracy-metric recall --limit 200

# Blend 20 COCO images with 40 local images to stabilize benchmark evaluation:
python frigate-to-cvat.py --balance-mode accuracy-deficit --accuracy-test-images 40 --accuracy-coco-images 20 --limit 150

# Two-step decoupled workflow (run deep evaluation first, then harvest):
python test_server_accuracy.py --all --output-json ./reports/accuracy_eval.json
python frigate-to-cvat.py --accuracy-report ./reports/accuracy_eval.json --limit 120
```

---

### Step 2: Audit Dataset Balance & Review in CVAT

#### 1. Audit Balance (`report-cvat-balance.py`)

Inspect annotation distributions across CVAT tasks/jobs to identify class deficits and potential misclassifications:

```bash
# Balance audit across all CVAT tasks:
python report-cvat-balance.py

# Include per-job breakdown:
python report-cvat-balance.py --by-job

# Audit a specific task:
python report-cvat-balance.py --task-id 4

# Export as JSON:
python report-cvat-balance.py --json
```

#### 2. Review Annotations in CVAT UI

1. Open the task in [CVAT](https://github.com/cvat-ai/cvat) at `http://<CVAT_HOST>:8080` (or visit the URL printed in Step 1).
2. Inspect bounding boxes, correct class labels, adjust coordinates, and label missed objects.
3. Save the annotations in CVAT.

---

### Step 3: Train, Quantize & Compile (`train.sh`)

[`train.sh`](train.sh) handles the complete post-annotation pipeline: exporting from CVAT, training YOLO, exporting ONNX, normalizing coordinates, quantizing to INT8, lowering to TOSA MLIR, compiling to Torq NPU `.vmfb`, and deploying to Coralboard.

#### Turnkey Execution Examples:

```bash
# 1. Turnkey run on default profile (torq local GPU), auto-exporting latest CVAT task:
./train.sh --export

# 2. Turnkey run on remote Spark GPU server for Torq NPU, exporting ALL CVAT tasks:
./train.sh -p spark-torq --export all

# 3. Train using YOLO26n architecture on Spark for Torq NPU:
./train.sh -p spark-torq-yolo26n --export all

# 4. Train with explicit local GPU device and custom epochs:
./train.sh --device 0 --epochs 100 --batch 16

# 5. Export a specific CVAT task ID:
./train.sh --export 4

# 6. Skip CVAT export and train on existing datasets/yolo/:
./train.sh

# 7. Fine-tune / resume from previous best.pt checkpoint:
./train.sh --resume

# 8. Train remotely on Spark only, pull weights, but skip local compilation:
./train.sh -p spark-torq --skip-compile

# 9. Pull latest trained weights from Spark and compile locally (skip training):
./train.sh -p spark-torq --pull

# 10. Fast compilation of an existing checkpoint (skip training):
./train.sh --compile runs/detect/frigate_coral/yolov8n/weights/best.pt

# 11. Wipe and replace datasets/yolo instead of appending:
./train.sh --export all --replace

# 12. Auto-clean remote training images on Spark after run completes:
./train.sh -p spark-torq --clean-images

# 13. Train YOLOv9s and export ONNX for NVIDIA GPU (exits early before Torq .vmfb compilation):
./train.sh -p nvidia

# 14. Train YOLOv9s remotely on Spark and export ONNX locally for NVIDIA GPU:
./train.sh -p spark-nvidia

# 15. Compile an existing checkpoint to ONNX only for NVIDIA GPU:
./train.sh -p nvidia --compile runs/detect/frigate_coral/yolov9s/weights/best.pt
```

#### Architecture Comparison: YOLOv8n vs. YOLO26n

| Feature | YOLOv8n (`yolov8n.pt`) | YOLO26n (`yolo26n.pt`) |
| :--- | :--- | :--- |
| **Recommended Profile** | `torq` / `spark-torq` (Default) | `spark-torq-yolo26n` |
| **Loss & Supervision** | Multi-anchor `TaskAlignedAssigner` + DFL | Single-anchor `end2end` scalar regression |
| **Small-Dataset Convergence** | Fast (~80 epochs, 84.6% mAP@50) | Requires 200–300 epochs (`patience: 50`) |
| **INT8 Output Scale** | Fine (`scale = 0.00616`), tight confidences | Coarser (`scale = 0.01012`), requires lower detection threshold (~0.15) |
| **Compiled Binary** | `custom_yolov8n.vmfb` (~3.4 MB, ~13 ms) | `custom_yolo26n.vmfb` (~3.5 MB, ~14 ms) |

---

### Step 4: Deployment to Coralboard Vision Service

The compiled model binary (`.vmfb`) and its paired label mapping (`labels_<name>.json`) must be deployed to the `coralboard-sl2619-vision` service.

#### Option A: Integrated Deployment via `train.sh`

Deploy automatically upon successful compilation:

```bash
# Remote deployment via SCP:
./train.sh --deploy root@<CORALBOARD_IP>:/home/root/coralboard-sl2619-vision/models

# Local directory copy:
./train.sh --deploy ../coralboard-sl2619-vision/models

# Deploy to destinations pre-configured in config.yaml:
./train.sh
```

#### Option B: Standalone Deployment Script (`deploy_to_coralboard.sh`)

[`deploy_to_coralboard.sh`](deploy_to_coralboard.sh) manages verification, SCP transfer, systemd service restart via SSH, and post-deployment HTTP health checks:

```bash
# Deploy, restart service, and verify health:
bash deploy_to_coralboard.sh \
  --host root@<CORALBOARD_IP>:/home/root/coralboard-sl2619-vision/models \
  --restart \
  --check

# Deploy to local directory:
bash deploy_to_coralboard.sh --dir ../coralboard-sl2619-vision/models
```

#### Why Matching `labels_<name>.json` is Critical

The Torq NPU binary outputs raw integer indices ($0 \dots C-1$). If `labels_<name>.json` is missing from the proxy's `models/` directory, the server falls back to standard COCO-80 indices:

| Class Index | Custom Dataset Label (`labels_<name>.json`) | Default COCO-80 Fallback Label | Consequence Without Matching Labels |
| :---: | :---: | :---: | :--- |
| `0` | `person` | `person` | Detected as `person` (matches by coincidence) |
| `1` | `dog` | `bicycle` | **Dog** misidentified as **`bicycle`** |
| `2` | `car` | `car` | Detected as `car` (matches by coincidence) |
| `3` | `horse` | `motorcycle` | **Horse** misidentified as **`motorcycle`** |
| `4` | `cat` | `airplane` | **Cat** misidentified as **`airplane`** |

> [!WARNING]
> **Frigate Silent Ingest Discard**: If the returned label (e.g. `bicycle` or `airplane`) is not explicitly listed in Frigate's `objects.track`, Frigate **silently discards the detection at ingest time**. No bounding box is tracked and no event is recorded.

---

### Step 5: Test & Benchmark Vision Server (`test_server_accuracy.py`)

#### 1. Quick Server Health Check

```bash
curl -s http://<CORALBOARD_IP>:5000/health | jq .
```
Expected response:
```json
{
  "status": "healthy",
  "npu_ready": true,
  "backend": "iree_runtime",
  "model_path": "models/custom_yolov8n.vmfb",
  "init_error": null
}
```

#### 2. Manual Test Detection Request

```bash
curl -X POST "http://<CORALBOARD_IP>:5000/v1/vision/detection" \
  -F "image=@/path/to/test.jpg" \
  -F "min_confidence=0.20"
```

#### 3. Automated Benchmark (`test_server_accuracy.py`)

Benchmark the running vision server against random images and ground truth annotations:

```bash
# Test 60 random images across the entire dataset (default):
python test_server_accuracy.py --server-url http://<CORALBOARD_IP>:5000/v1/vision/detection

# Test all images in dataset with per-image verbose logging:
python test_server_accuracy.py --all -v

# Evaluate validation split only:
python test_server_accuracy.py --split val

# Include 20 images from COCO dataset alongside local dataset images:
python test_server_accuracy.py --num-images 40 --coco-images 20 --coco-dataset coco-val2017

# Include 30 images from Frigate+ dataset alongside local images:
python test_server_accuracy.py --num-images 30 --frigate-plus-images 30

# Benchmark strictly against Frigate+ dataset images (0 local images):
python test_server_accuracy.py --num-images 0 --frigate-plus-images 50

# Benchmark strictly against COCO dataset images (0 local images):
python test_server_accuracy.py --num-images 0 --coco-images 50

# Custom confidence & IoU thresholds, exporting results to JSON:
python test_server_accuracy.py --threshold 0.20 --iou-threshold 0.45 --output-json report.json
```

#### Sample Benchmark Report

```text
==========================================================================================
                    CORALBOARD TORQ NPU ACCURACY REPORT
==========================================================================================
Endpoint URL    : http://<CORALBOARD_IP>:5000/v1/vision/detection
Dataset Split   : all (60 images evaluated)
Confidence Thresh: 0.20 | Matching IoU Thresh: 0.45
------------------------------------------------------------------------------------------
Class            GT   Pred     TP     FP     FN   Precision     Recall   F1-Score     mIoU
------------------------------------------------------------------------------------------
person           82     79     74      5      8       93.7%      90.2%      91.9%     0.78
dog              15     14     13      1      2       92.9%      86.7%      89.7%     0.81
cat               8      6      5      1      3       83.3%      62.5%      71.4%     0.74
car              24     22     20      2      4       90.9%      83.3%      87.0%     0.79
------------------------------------------------------------------------------------------
OVERALL         129    121    112      9     17       92.6%      86.8%      89.6%     0.78
==========================================================================================

Summary Statistics:
  Images Tested        : 60 (Failed requests: 0)
  Ground Truth Items   : 129
  Detections Produced  : 121 (True Positives: 112, False Positives: 9)
  Overall Precision    : 92.6%
  Overall Recall       : 86.8%
  Overall F1-Score     : 89.6%
  Mean Bounding-Box IoU: 0.781
  HTTP Roundtrip Time  : Avg: 22.4ms | Min: 18.2ms | Max: 41.5ms
  Server NPU Infer Time: Avg: 13.1ms (~76 FPS raw NPU capability)
```

#### Closing the Feedback Loop: Targeted Re-Harvesting

When the benchmark report highlights weak or degraded classes (for example, `cat` with 71.4% F1-score and 62.5% recall in the report above), close the active learning loop immediately by running accuracy-deficit harvesting in [`frigate-to-cvat.py`](frigate-to-cvat.py) to prioritize fresh training snapshots for those specific deficit categories:

```bash
# Prioritize harvesting weak classes identified by the benchmark:
python frigate-to-cvat.py --balance-mode accuracy-deficit --limit 150
```

---

### Step 6: Upstream Frigate NVR Configuration

In your Frigate `config.yaml`, configure the detector pointing to the Coralboard vision proxy:

```yaml
detectors:
  coral_torq:
    type: deepstack
    api_url: http://<CORALBOARD_IP>:5000/v1/vision/detection
    api_timeout: 0.25

model:
  width: 320
  height: 320

objects:
  track:
    - person
    - dog
    - cat
    - car
```

---

## Technical Pipeline Details (`train.sh`)

When `train.sh` executes compilation, it carries out the following steps:

1. **Training ($320 \times 320$)**: Fine-tunes the base model (`yolov8n.pt` or `yolo26n.pt`) at strictly $320 \times 320$ resolution using Ultralytics.
2. **ONNX Normalization ([`prepare_for_onnx2tf.py`](prepare_for_onnx2tf.py))**: Modifies the ONNX graph output channels `0..4` (bounding boxes `cx, cy, w, h`) by dividing them by `320.0` to yield normalized coordinates $[0.0, 1.0]$.
3. **Representative Calibration Dataset**: Samples images from `datasets/yolo/images/val` and `images/train` using class-balanced stratified sampling, letterboxes them to $320 \times 320$ RGB, and saves `calibration_video_300_nhwc_float32.npy`.
4. **Full INT8 Quantization (`onnx2tf`)**: Runs `onnx2tf` with `--tflite_backend tf_converter` and calibration data, producing full integer quantized TFLite (`[1, 320, 320, 3] int8` input, `[1, 4+N, 2100] int8` output).
5. **TOSA MLIR Lowering (`tosa-converter-for-tflite`)**: Lowers the quantized TFLite graph to TOSA MLIR dialect.
6. **Hardware Vector Compilation (`torq-compile`)**: Compiles TOSA MLIR with `--torq-hw=SL2610` into a hardware binary (`.vmfb`) executing natively inside Astra SL2610 NSS vector slices without LRAM overflow.
7. **Quantization Scale Extraction**: Extracts output quantization scales and zero-points from the TFLite model and generates [`models/labels/labels_<name>.json`](models/labels).

---

## Standalone CVAT Export (`export-cvat-to-yolo.py`)

If you prefer to export annotations separately outside of `train.sh`:

```bash
# Export latest task and append to datasets/yolo:
python export-cvat-to-yolo.py

# Export specific task:
python export-cvat-to-yolo.py --export 4

# Export multiple tasks:
python export-cvat-to-yolo.py --export 3 4

# Wipe existing dataset and do a clean export:
python export-cvat-to-yolo.py --export all --replace
```

---

## Summary of Key Scripts

| Script / Component | Description |
| :--- | :--- |
| [`frigate-to-cvat.py`](frigate-to-cvat.py) | **Frigate Harvest & CVAT Upload**: Downloads events from Frigate, creates COCO pre-annotations, and uploads review tasks to CVAT with dynamic deficit balancing. |
| [`report-cvat-balance.py`](report-cvat-balance.py) | **CVAT Dataset Balance Auditor**: Inspects annotation distributions across CVAT tasks/jobs, calculating class deficits, health status, and visual bars. |
| [`train.sh`](train.sh) | **Unified Pipeline Runner**: Automates CVAT export, local GPU or remote Spark training (`-p spark`, `-p spark-yolo26n`), ONNX export, INT8 quantization, TOSA lowering, Torq compilation, and deployment. |
| [`config_utils.py`](config_utils.py) | **Configuration Engine**: Parses `config.yaml`, resolves execution profiles, normalizes balance modes, maps COCO-80 classes, and dumps shell variables for `train.sh`. |
| [`export-cvat-to-yolo.py`](export-cvat-to-yolo.py) | **CVAT Exporter**: Connects to CVAT REST API and exports annotations to `datasets/yolo/` with train/val splits and `data.yaml`. |
| [`deploy_to_coralboard.sh`](deploy_to_coralboard.sh) | **Model Deployer**: Deploys compiled `.vmfb` and `labels.json` to Coralboard locally or via SCP, with optional SSH service restart and HTTP health checks. |
| [`test_server_accuracy.py`](test_server_accuracy.py) | **Server Accuracy Benchmark**: Evaluates running vision proxy against dataset images and COCO reference images, measuring precision, recall, F1, and NPU latency. |
| [`prepare_for_onnx2tf.py`](prepare_for_onnx2tf.py) | **ONNX Normalizer**: Divides ONNX box coordinates by 320.0, generates class-balanced INT8 calibration `.npy`, and writes `labels.json`. |
| [`tests/`](tests/) | **Automated Test Suite**: Pytest unit tests validating configuration resolution, balancing algorithms, accuracy calculations, and dataset conversions. |
