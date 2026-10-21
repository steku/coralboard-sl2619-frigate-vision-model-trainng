#!/usr/bin/env bash
# ==============================================================================
# Unified Frigate Model Training & Coralboard SL2619 Compilation Pipeline
# ==============================================================================
# Consolidates local GPU/CPU training and remote DGX Spark training into a
# single unified runner with local GPU training as the default.
#
# Workflows:
#   1. Export: Optional CVAT annotations export (--export)
#   2. Train: Local GPU (Default) or Remote DGX Spark (-p spark)
#   3. Compile: Local INT8 quantization (onnx2tf), TOSA MLIR lowering,
#               and Torq NPU compilation (torq-compile for Astra SL2610)
#   4. Deploy: Automated deployment to Coralboard SL2619 vision service
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ------------------------------------------------------------------------------
# Colors for logging
# ------------------------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()     { echo -e "${RED}[ERROR]${NC} $1"; }

relocate_root_checkpoints() {
    for pt in *.pt; do
        if [[ -f "${pt}" ]]; then
            mv "${pt}" "${MODELS_BASE_DIR}/"
        fi
    done
}

# ------------------------------------------------------------------------------
# Default Settings
# ------------------------------------------------------------------------------
CONFIG_FILE="config.yaml"
RUNNER="local"                    # Default: local GPU training
DEVICE="0"                        # Default: GPU '0'
DATA_YAML="datasets/data.yaml"
IMGSZ=320                         # Strictly 320 for Coralboard SL2619 NSS slices
EPOCHS=80
PATIENCE=50
BATCH=""                          # Evaluated based on runner if not provided
WORKERS=8
CACHE="ram"
OPTIMIZER="AdamW"
LR0="0.0005"
LRF="0.01"
COS_LR="True"
WARMUP_EPOCHS="5.0"
WARMUP_MOMENTUM="0.5"
WARMUP_BIAS_LR="0.0"
PROJECT="frigate"
TARGET_FORMAT="torq"              # 'torq' (.vmfb for Coralboard NPU) or 'nvidia' (.onnx for NVIDIA GPU)
BASE_MODEL="yolov8n.pt"
RUN_NAME=""
LABELS_FILE=""
OUTPUT_VMFB=""
OUTPUT_ONNX=""
FROM_SCRATCH=true
EXTRA_TRAIN_ARGS=""

# Remote Spark specific defaults
SPARK_HOST="${SPARK_HOST:-${DGX_HOST:-}}"
REMOTE_DIR="~/frigate-model-training"
CLEAN_IMAGES=false
PULL_ONLY=false

# Stage flow toggles
SKIP_TRAIN=false
CUSTOM_MODEL_PT=""
SKIP_COMPILE=false
SKIP_DEPLOY=false
DEPLOY_HOST=""
DEPLOY_DIR=""

# CVAT export options
DO_EXPORT=false
EXPORT_TASK_IDS=""
DO_APPEND=true
COCO_DATASET="coco128"
USE_FRIGATE_PLUS=""
FRIGATE_PLUS_DIR="datasets/frigate-plus"

# Output directories
OUTPUT_DIR="output"
MODELS_DIR="models"
MODELS_COMPILED_DIR="${MODELS_DIR}/compiled"
MODELS_BASE_DIR="${MODELS_DIR}/base"
MODELS_LABELS_DIR="${MODELS_DIR}/labels"
mkdir -p "${OUTPUT_DIR}" "${MODELS_COMPILED_DIR}" "${MODELS_LABELS_DIR}" "${MODELS_BASE_DIR}"

# ------------------------------------------------------------------------------
# Usage & Help
# ------------------------------------------------------------------------------
usage() {
    local exit_code="${1:-0}"
    echo "=========================================================================="
    echo "  Frigate Model Training & Compiler (Torq NPU & NVIDIA GPU ONNX)"
    echo "=========================================================================="
    echo "Usage: bash $0 [PROFILE] [WORKFLOW] [OPTIONS]"
    echo ""
    echo "Default (no flags): Runs default profile from config.yaml ('torq' local GPU),"
    echo "quantizes to INT8, compiles to Torq NPU binary, and deploys if configured."
    echo ""
    echo "Execution Profiles:"
    echo "  -p, --profile NAME      Select profile from config.yaml (e.g. 'torq', 'spark-torq', 'spark-torq-yolo26n', 'nvidia', 'spark-nvidia', 'cpu-torq')"
    echo ""
    echo "Primary Workflows:"
    echo "  --compile [MODEL.pt]    Skip training; compile checkpoint directly (or latest best.pt)"
    echo "  --export [TASK_ID|all]  Export CVAT task(s) to datasets/yolo/ before training (default: latest)"
    echo "  --deploy [DEST]         Deploy compiled model & labels to local dir or remote host (user@ip:/path)"
    echo ""
    echo "Workflow Controls:"
    echo "  --skip-compile          Train only; skip compilation/export"
    echo "  --skip-deploy           Skip deployment step even if configured in config.yaml"
    echo "  --pull                  Pull latest weights from Spark and compile locally (skip training)"
    echo ""
    echo "Profile Parameter Overrides:"
    echo "  -m, --base-model MODEL  Base checkpoint / architecture (default: yolov8n.pt, or yolov9s.pt for nvidia)"
    echo "  --epochs N              Training epochs"
    echo "  --batch N               Batch size"
    echo "  --device DEV            Training device ('0' local GPU, 'cpu', or '0,1' Spark)"
    echo "  --host USER@HOST        SSH destination for remote training"
    echo "  --resume                Resume/fine-tune from previous checkpoint (default: train from scratch)"
    echo "  --replace               Wipe and replace existing dataset instead of appending"
    echo "  --coco-dataset NAME     COCO subset for co-training ('coco128' or 'coco-val2017')"
    echo "  --frigate-plus          Include Frigate+ dataset in training & validation splits"
    echo "  --no-frigate-plus       Exclude Frigate+ dataset from training splits"
    echo "  --extra-args ARGS       Forward extra arguments directly to 'yolo detect train'"
    echo "  -c, --config PATH       Path to YAML configuration file (default: config.yaml)"
    echo "  -h, --help              Show this help message"
    echo ""
    exit "${exit_code}"
}

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
PROFILE=""
EXPLICIT_HOST=false
EXPLICIT_BATCH=false
EXPLICIT_DEVICE=false
EXPLICIT_EPOCHS=false
EXPLICIT_BASE_MODEL=false
EXPLICIT_COCO=false
EXPLICIT_DEPLOY_HOST=false
EXPLICIT_DEPLOY_DIR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        --host)
            SPARK_HOST="$2"
            EXPLICIT_HOST=true
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --export)
            DO_EXPORT=true
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                clean_ids="${1//,/ }"
                EXPORT_TASK_IDS="${EXPORT_TASK_IDS} ${clean_ids}"
                shift
            done
            ;;
        --replace)
            DO_APPEND=false
            shift
            ;;
        --coco-dataset)
            COCO_DATASET="$2"
            EXPLICIT_COCO=true
            shift 2
            ;;
        --frigate-plus)
            USE_FRIGATE_PLUS="true"
            shift
            ;;
        --no-frigate-plus)
            USE_FRIGATE_PLUS="false"
            shift
            ;;
        --compile)
            SKIP_TRAIN=true
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                CUSTOM_MODEL_PT="$2"
                shift
            fi
            shift
            ;;
        -m|--base-model)
            BASE_MODEL="$2"
            EXPLICIT_BASE_MODEL=true
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            EXPLICIT_EPOCHS=true
            shift 2
            ;;
        --batch)
            BATCH="$2"
            EXPLICIT_BATCH=true
            shift 2
            ;;
        --device)
            DEVICE="$2"
            EXPLICIT_DEVICE=true
            shift 2
            ;;
        --resume)
            FROM_SCRATCH=false
            shift
            ;;
        --extra-args)
            EXTRA_TRAIN_ARGS="$2"
            shift 2
            ;;
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        --deploy)
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                TARGET="$2"
                if [[ "${TARGET}" == *":"* ]]; then
                    DEPLOY_HOST="${TARGET}"
                    EXPLICIT_DEPLOY_HOST=true
                else
                    DEPLOY_DIR="${TARGET}"
                    EXPLICIT_DEPLOY_DIR=true
                fi
                shift
            fi
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --pull)
            PULL_ONLY=true
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            log_err "Unknown argument: $1"
            usage 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Virtual Environment Activation
# ------------------------------------------------------------------------------
log_info "Verifying local Python 3.12 environment..."
VIRTUAL_ENV_PATH=""
if [[ -d "./venv-torq-312" ]]; then
    VIRTUAL_ENV_PATH="./venv-torq-312"
elif [[ -d "./venv" ]]; then
    VIRTUAL_ENV_PATH="./venv"
else
    log_err "No virtual environment found. Please create one with 'python3.12 -m venv venv-torq-312'."
    exit 1
fi

log_info "Activating virtual environment: ${VIRTUAL_ENV_PATH}"
source "${VIRTUAL_ENV_PATH}/bin/activate"

# ------------------------------------------------------------------------------
# Load Configuration Profile & Apply Overrides
# ------------------------------------------------------------------------------
if [[ -f "${CONFIG_FILE}" ]]; then
    if ! PROFILE_ENV_OUT="$(python3 config_utils.py --profile-env "${PROFILE}" --config "${CONFIG_FILE}")"; then
        exit 1
    fi
    eval "${PROFILE_ENV_OUT}"
fi

PROFILE_NAME="${PROFILE_NAME:-torq}"
TARGET_FORMAT="${PROFILE_TARGET:-torq}"

if [[ "${PROFILE_REMOTE:-false}" == "true" || "${PULL_ONLY}" == "true" || "${EXPLICIT_HOST}" == "true" ]]; then
    RUNNER="spark"
else
    RUNNER="local"
fi

if [[ "${EXPLICIT_HOST}" == "false" && -n "${PROFILE_HOST}" ]]; then
    SPARK_HOST="${PROFILE_HOST}"
fi

if [[ "${EXPLICIT_BATCH}" == "false" ]]; then
    if [[ -n "${PROFILE_BATCH}" ]]; then
        BATCH="${PROFILE_BATCH}"
    elif [[ "${RUNNER}" == "spark" ]]; then
        BATCH=32
    else
        BATCH=8
    fi
fi

if [[ "${EXPLICIT_DEVICE}" == "false" && -n "${PROFILE_DEVICE}" ]]; then
    DEVICE="${PROFILE_DEVICE}"
fi

if [[ "${EXPLICIT_EPOCHS}" == "false" && -n "${PROFILE_EPOCHS}" ]]; then
    EPOCHS="${PROFILE_EPOCHS}"
fi

[[ -n "${PROFILE_PATIENCE}" ]] && PATIENCE="${PROFILE_PATIENCE}"
[[ -n "${PROFILE_WORKERS}" ]] && WORKERS="${PROFILE_WORKERS}"
[[ -n "${PROFILE_CACHE}" ]] && CACHE="${PROFILE_CACHE}"

if [[ "${EXPLICIT_BASE_MODEL}" == "false" && -n "${PROFILE_BASE_MODEL}" ]]; then
    BASE_MODEL="${PROFILE_BASE_MODEL}"
fi

if [[ "${EXPLICIT_COCO}" == "false" && -n "${PROFILE_COCO_DATASET}" ]]; then
    COCO_DATASET="${PROFILE_COCO_DATASET}"
fi

if [[ -z "${USE_FRIGATE_PLUS}" ]]; then
    USE_FRIGATE_PLUS="${PROFILE_FRIGATE_PLUS:-true}"
fi

[[ -n "${PROFILE_FRIGATE_PLUS_DIR}" ]] && FRIGATE_PLUS_DIR="${PROFILE_FRIGATE_PLUS_DIR}"

USE_FRIGATE_PLUS_PY="True"
[[ "${USE_FRIGATE_PLUS}" == "false" ]] && USE_FRIGATE_PLUS_PY="False"

[[ -n "${PROFILE_OPTIMIZER}" ]] && OPTIMIZER="${PROFILE_OPTIMIZER}"
[[ -n "${PROFILE_LR0}" ]] && LR0="${PROFILE_LR0}"
[[ -n "${PROFILE_LRF}" ]] && LRF="${PROFILE_LRF}"
[[ -n "${PROFILE_COS_LR}" ]] && COS_LR="${PROFILE_COS_LR}"
[[ -n "${PROFILE_WARMUP_EPOCHS}" ]] && WARMUP_EPOCHS="${PROFILE_WARMUP_EPOCHS}"
[[ -n "${PROFILE_WARMUP_MOMENTUM}" ]] && WARMUP_MOMENTUM="${PROFILE_WARMUP_MOMENTUM}"
[[ -n "${PROFILE_WARMUP_BIAS_LR}" ]] && WARMUP_BIAS_LR="${PROFILE_WARMUP_BIAS_LR}"
[[ "${PROFILE_CLEAN_IMAGES:-false}" == "true" ]] && CLEAN_IMAGES=true
[[ -n "${PROFILE_REMOTE_DIR}" ]] && REMOTE_DIR="${PROFILE_REMOTE_DIR}"

if [[ "${EXPLICIT_DEPLOY_HOST}" == "false" && -z "${DEPLOY_HOST}" && -n "${PROFILE_DEPLOY_HOST}" && "${TARGET_FORMAT}" != "nvidia" ]]; then
    DEPLOY_HOST="${PROFILE_DEPLOY_HOST}"
fi

if [[ "${EXPLICIT_DEPLOY_DIR}" == "false" && -z "${DEPLOY_DIR}" && -n "${PROFILE_DEPLOY_DIR}" && "${TARGET_FORMAT}" != "nvidia" ]]; then
    DEPLOY_DIR="${PROFILE_DEPLOY_DIR}"
fi

# ------------------------------------------------------------------------------
# Normalize Model Names & Dynamic Defaults
# ------------------------------------------------------------------------------
TIMESTAMP="$(date +%y%m%d-%H%M)"
BASE_MODEL_NAME="$(basename "${BASE_MODEL%.*}")"
RUN_NAME="${BASE_MODEL_NAME}"
RUN_NAME_STEM="$(basename "${RUN_NAME%.*}")"
LABELS_FILE="${MODELS_LABELS_DIR}/${TARGET_FORMAT}_${BASE_MODEL_NAME}_${TIMESTAMP}.json"
OUTPUT_VMFB="${MODELS_COMPILED_DIR}/${TARGET_FORMAT}_${RUN_NAME_STEM}_${TIMESTAMP}.vmfb"
OUTPUT_ONNX="${MODELS_COMPILED_DIR}/${TARGET_FORMAT}_${RUN_NAME_STEM}_${TIMESTAMP}.onnx"

# Resolve deployment overrides
if [[ "${SKIP_DEPLOY}" == "true" ]]; then
    DEPLOY_HOST=""
    DEPLOY_DIR=""
fi

echo "=========================================================================="
echo "  Frigate Training & Compilation Pipeline"
echo "=========================================================================="
if [[ "${RUNNER}" == "spark" ]]; then
    echo "  Profile:       ${PROFILE_NAME^^} (Remote Host: ${SPARK_HOST:-[Not Set]})"
else
    echo "  Profile:       ${PROFILE_NAME^^} (Local GPU, Device: ${DEVICE})"
fi
if [[ "${TARGET_FORMAT}" == "nvidia" ]]; then
    echo "  Target:        NVIDIA Video Card (ONNX -> ${OUTPUT_ONNX})"
else
    echo "  Target:        Coralboard SL2619 Torq NPU (VMFB -> ${OUTPUT_VMFB})"
fi
echo "  Hyperparams:   Batch: ${BATCH}, Workers: ${WORKERS}, Epochs: ${EPOCHS}, Patience: ${PATIENCE}, Cache: ${CACHE}"
echo "  Fine-Tuning:   Optimizer: ${OPTIMIZER}, lr0: ${LR0}, lrf: ${LRF}, cos_lr: ${COS_LR}"
echo "  Warm-Up:       Epochs: ${WARMUP_EPOCHS}, Momentum: ${WARMUP_MOMENTUM}, Bias LR: ${WARMUP_BIAS_LR}"
echo "  Base Model:    ${BASE_MODEL} (Run Name: ${RUN_NAME}, Co-Training: ${COCO_DATASET})"
echo "  Resolution:    ${IMGSZ}x${IMGSZ}"
echo "=========================================================================="

# ------------------------------------------------------------------------------
# Tool Availability Verification
# ------------------------------------------------------------------------------
REQUIRED_TOOLS=(python3 yolo)
if [[ "${SKIP_COMPILE}" == "false" && "${TARGET_FORMAT}" == "torq" ]]; then
    REQUIRED_TOOLS+=(onnx2tf tosa-converter-for-tflite torq-compile)
fi

for cmd in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "${cmd}" &>/dev/null; then
        log_err "Required tool '${cmd}' not found in active environment. Run 'pip install -r requirements.txt'."
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# Step 1: Optional CVAT Dataset Export
# ------------------------------------------------------------------------------
if [[ "${DO_EXPORT}" == "true" ]]; then
    echo ""
    echo "=========================================================================="
    echo "  Step 1: Exporting CVAT Dataset to YOLO Format"
    echo "=========================================================================="
    EXPORT_CMD=(python3 export-cvat-to-yolo.py)
    if [[ -n "${CONFIG_FILE}" ]]; then
        EXPORT_CMD+=(-c "${CONFIG_FILE}")
    fi
    if [[ -n "${EXPORT_TASK_IDS}" ]]; then
        EXPORT_CMD+=(--export ${EXPORT_TASK_IDS})
    else
        EXPORT_CMD+=(--export latest)
    fi
    if [[ "${DO_APPEND}" == "false" ]]; then
        EXPORT_CMD+=(--replace)
    fi
    if [[ -n "${COCO_DATASET}" ]]; then
        EXPORT_CMD+=(--coco-dataset "${COCO_DATASET}")
    fi
    if [[ "${USE_FRIGATE_PLUS}" == "true" ]]; then
        EXPORT_CMD+=(--frigate-plus)
    elif [[ "${USE_FRIGATE_PLUS}" == "false" ]]; then
        EXPORT_CMD+=(--no-frigate-plus)
    fi
    if [[ -n "${FRIGATE_PLUS_DIR}" ]]; then
        EXPORT_CMD+=(--frigate-plus-dir "${FRIGATE_PLUS_DIR}")
    fi
    DATA_DIR="$(dirname "${DATA_YAML}")"
    if [[ "${DATA_DIR}" != "." && "${DATA_DIR}" != "datasets" && "${DATA_DIR}" != "datasets/yolo" && "${DATA_DIR}" != "yolo_dataset" ]]; then
        EXPORT_CMD+=(--output-dir "${DATA_DIR}")
    fi
    log_info "Running: ${EXPORT_CMD[*]}"
    "${EXPORT_CMD[@]}"
fi

# ------------------------------------------------------------------------------
# Step 2: Training / Checkpoint Selection
# ------------------------------------------------------------------------------
WEIGHTS_PATH=""
LOCAL_WEIGHTS_DIR="runs/detect/${PROJECT}/${RUN_NAME}/weights"

if [[ -n "${CUSTOM_MODEL_PT}" ]]; then
    if [[ ! -f "${CUSTOM_MODEL_PT}" ]]; then
        log_err "Specified checkpoint '${CUSTOM_MODEL_PT}' does not exist!"
        exit 1
    fi
    WEIGHTS_PATH="${CUSTOM_MODEL_PT}"
    log_info "Using specified checkpoint: ${WEIGHTS_PATH}"

elif [[ "${SKIP_TRAIN}" == "true" ]]; then
    # Look for existing best.pt
    LATEST_BEST=$(find "runs/detect" -path "*${RUN_NAME}*" -name "best.pt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    if [[ -z "${LATEST_BEST}" ]]; then
        LATEST_BEST=$(find "runs/detect" -path "*${BASE_MODEL_NAME}*" -name "best.pt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    fi
    if [[ -z "${LATEST_BEST}" ]]; then
        log_err "No existing 'best.pt' found for '${RUN_NAME}'. Run without --compile to train first."
        exit 1
    fi
    WEIGHTS_PATH="${LATEST_BEST}"
    log_info "Using existing checkpoint: ${WEIGHTS_PATH}"

elif [[ "${RUNNER}" == "spark" ]]; then
    # --------------------------------------------------------------------------
    # Remote DGX Spark Training
    # --------------------------------------------------------------------------
    if [[ -z "${SPARK_HOST}" ]]; then
        log_err "No Spark host specified! Set --host USER@HOST or 'remote_training.host' in config.yaml."
        exit 1
    fi

    echo ""
    echo "=========================================================================="
    echo "  Step 2: Training Remotely on NVIDIA DGX Spark (${SPARK_HOST})"
    echo "=========================================================================="

    log_info "Testing SSH connectivity and resolving remote workspace on ${SPARK_HOST}..."
    REMOTE_DIR=$(ssh -o ConnectTimeout=5 "${SPARK_HOST}" "
        set -e
        eval \"TARGET_DIR=${REMOTE_DIR}\"
        mkdir -p \"\${TARGET_DIR}/models/base\"
        cd \"\${TARGET_DIR}\"
        pwd
    ") || {
        log_err "Could not connect to ${SPARK_HOST} via SSH or resolve remote directory."
        exit 1
    }
    log_info "Remote workspace verified: ${SPARK_HOST}:${REMOTE_DIR}"

    # Clean up stray literal '~' directory if created by earlier runs
    ssh "${SPARK_HOST}" "
        if [ -d \"\$HOME/~\" ]; then
            rm -rf \"\$HOME/~\"
        fi
    " 2>/dev/null || true

    if [[ "${PULL_ONLY}" == "true" ]]; then
        log_info "--pull specified: skipping remote execution, pulling existing weights from Spark..."
    else
        # Setup remote venv if missing
        ssh "${SPARK_HOST}" "
            set -e
            cd \"${REMOTE_DIR}\"
            if [ ! -d \"venv\" ]; then
                echo '[SPARK] Creating Python 3 virtual environment...'
                python3 -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install ultralytics onnx onnxruntime pyyaml requests cvat-sdk
            fi
        "

        # Sync dataset and training helpers
        log_info "Synchronizing dataset and code to ${SPARK_HOST}:${REMOTE_DIR}..."
        SCRIPTS_TO_SYNC=(config_utils.py)
        [[ -f "${CONFIG_FILE}" ]] && SCRIPTS_TO_SYNC+=("${CONFIG_FILE}")

        LOCAL_YOLO_DIR="datasets/yolo"
        [[ ! -d "${LOCAL_YOLO_DIR}" && -d "yolo_dataset" ]] && LOCAL_YOLO_DIR="yolo_dataset"

        if command -v rsync &>/dev/null; then
            rsync -avz --copy-links \
                "${SCRIPTS_TO_SYNC[@]}" \
                "${SPARK_HOST}:${REMOTE_DIR}/"

            ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/datasets'"
            if [[ -f "datasets/data.yaml" ]]; then
                rsync -avz "datasets/data.yaml" "${SPARK_HOST}:${REMOTE_DIR}/datasets/data.yaml"
            fi
            rsync -avz --delete \
                --exclude="coco*" \
                --exclude="*.zip" \
                --exclude="*.tar*" \
                --exclude="__pycache__" \
                "${LOCAL_YOLO_DIR}/" "${SPARK_HOST}:${REMOTE_DIR}/datasets/yolo/"

            if [[ "${USE_FRIGATE_PLUS}" == "true" && -d "${FRIGATE_PLUS_DIR}" ]]; then
                log_info "Synchronizing Frigate+ dataset to ${SPARK_HOST}:${REMOTE_DIR}/datasets/frigate-plus/..."
                ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/datasets/frigate-plus'"
                rsync -avz --delete \
                    --exclude="__pycache__" \
                    "${FRIGATE_PLUS_DIR}/" "${SPARK_HOST}:${REMOTE_DIR}/datasets/frigate-plus/"
            fi
        else
            scp "${SCRIPTS_TO_SYNC[@]}" "${SPARK_HOST}:${REMOTE_DIR}/"
            ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/datasets/yolo'/{images,labels}"
            scp -r "${LOCAL_YOLO_DIR}/images" "${LOCAL_YOLO_DIR}/labels" "${SPARK_HOST}:${REMOTE_DIR}/datasets/yolo/"
            if [[ "${USE_FRIGATE_PLUS}" == "true" && -d "${FRIGATE_PLUS_DIR}" ]]; then
                ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/datasets/frigate-plus'/{images,labels}"
                scp -r "${FRIGATE_PLUS_DIR}/images" "${FRIGATE_PLUS_DIR}/labels" "${SPARK_HOST}:${REMOTE_DIR}/datasets/frigate-plus/"
            fi
        fi

        # Sync previous checkpoint if resuming or base model if present locally
        if [[ "${FROM_SCRATCH}" == "false" && -f "${LOCAL_WEIGHTS_DIR}/best.pt" ]]; then
            ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/runs/detect/${PROJECT}/${RUN_NAME}/weights'"
            scp "${LOCAL_WEIGHTS_DIR}/best.pt" "${SPARK_HOST}:${REMOTE_DIR}/runs/detect/${PROJECT}/${RUN_NAME}/weights/best.pt"
        fi
        if [[ -f "models/base/${BASE_MODEL}" ]]; then
            scp "models/base/${BASE_MODEL}" "${SPARK_HOST}:${REMOTE_DIR}/models/base/"
        elif [[ -f "${BASE_MODEL}" ]]; then
            scp "${BASE_MODEL}" "${SPARK_HOST}:${REMOTE_DIR}/models/base/"
        fi

        # Prepare remote dataset manifest and canonical COCO-80 data.yaml
        log_info "Preparing remote dataset manifest on Spark..."
        ssh "${SPARK_HOST}" "
            set -e
            cd \"${REMOTE_DIR}\"
            source \"${REMOTE_DIR}/venv/bin/activate\"
            export PYTHONPATH=\"${REMOTE_DIR}:\${PYTHONPATH:-}\"
            python3 -c \"import config_utils; config_utils.prepare_training_data_yaml('${COCO_DATASET}', '${REMOTE_DIR}', include_frigate_plus=${USE_FRIGATE_PLUS_PY}, frigate_plus_dir='${FRIGATE_PLUS_DIR}')\"
        "

        # Launch training on Spark
        log_info "Launching YOLO fine-tuning on ${SPARK_HOST} (device=${DEVICE}, batch=${BATCH}, optimizer=${OPTIMIZER}, lr0=${LR0}, warmup=${WARMUP_EPOCHS})..."
        ssh -t "${SPARK_HOST}" "
            set -e
            cd \"${REMOTE_DIR}\"
            source \"${REMOTE_DIR}/venv/bin/activate\"
            export PYTHONPATH=\"${REMOTE_DIR}:\${PYTHONPATH:-}\"

            START_MODEL='${BASE_MODEL}'
            if [ -f \"models/base/${BASE_MODEL}\" ]; then
                START_MODEL=\"models/base/${BASE_MODEL}\"
            fi

            if [ \"${FROM_SCRATCH}\" = \"false\" ]; then
                LATEST_EXISTING=\$(find \"${REMOTE_DIR}/runs/detect/${PROJECT}\" -path \"*${RUN_NAME}*\" -name \"best.pt\" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
                if [ -n \"\${LATEST_EXISTING}\" ] && [ -f \"\${LATEST_EXISTING}\" ]; then
                    echo \"[SPARK] Resuming training from existing checkpoint: \${LATEST_EXISTING}\"
                    START_MODEL=\"\${LATEST_EXISTING}\"
                fi
            fi

            echo '=========================================================================='
            echo '  Executing: yolo detect train on NVIDIA DGX Spark'
            echo '=========================================================================='
            yolo detect train \
                data='datasets/data.yaml' \
                model=\"\${START_MODEL}\" \
                epochs=${EPOCHS} \
                imgsz=320 \
                batch=${BATCH} \
                workers=${WORKERS} \
                cache=${CACHE} \
                patience=${PATIENCE} \
                optimizer='${OPTIMIZER}' \
                lr0=${LR0} \
                lrf=${LRF} \
                cos_lr=${COS_LR} \
                warmup_epochs=${WARMUP_EPOCHS} \
                warmup_momentum=${WARMUP_MOMENTUM} \
                warmup_bias_lr=${WARMUP_BIAS_LR} \
                device='${DEVICE}' \
                project='${PROJECT}' \
                name='${RUN_NAME}' \
                ${EXTRA_TRAIN_ARGS}
        "
    fi

    # Retrieve trained weights directly from Spark to local system before local compilation
    log_info "Retrieving trained best.pt from ${SPARK_HOST} to local system..."
    mkdir -p "${LOCAL_WEIGHTS_DIR}" "models/base"

    REMOTE_BEST_PT=$(ssh "${SPARK_HOST}" "
        find \"${REMOTE_DIR}/runs/detect/${PROJECT}\" -path \"*${RUN_NAME}*\" -name \"best.pt\" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '
    ")

    if [[ -z "${REMOTE_BEST_PT}" ]]; then
        log_err "Could not locate best.pt on ${SPARK_HOST}:${REMOTE_DIR}/runs/detect/${PROJECT}/"
        exit 1
    fi

    scp "${SPARK_HOST}:${REMOTE_BEST_PT}" "${LOCAL_WEIGHTS_DIR}/best.pt"
    WEIGHTS_PATH="${LOCAL_WEIGHTS_DIR}/best.pt"
    log_success "Trained weights retrieved successfully to ${LOCAL_WEIGHTS_DIR}/best.pt"

    # Clean remote images if requested
    if [[ "${CLEAN_IMAGES}" == "true" ]]; then
        log_info "Cleaning up remote training images on ${SPARK_HOST}..."
        ssh "${SPARK_HOST}" "rm -rf \"${REMOTE_DIR}/datasets/yolo/images\" \"${REMOTE_DIR}/datasets/frigate-plus/images\" \"${REMOTE_DIR}/yolo_dataset/images\" \"${REMOTE_DIR}/datasets\"" || true
    fi

else
    # --------------------------------------------------------------------------
    # Local GPU/CPU Training (Default)
    # --------------------------------------------------------------------------
    if [[ ( "${DATA_YAML}" == "datasets/data.yaml" || "${DATA_YAML}" == "datasets/yolo/data.yaml" || "${DATA_YAML}" == "yolo_dataset/data.yaml" ) && ( -d "datasets/yolo/images/train" || -d "yolo_dataset/images/train" || -d "${FRIGATE_PLUS_DIR}/images/train" ) ]]; then
        python3 -c "import config_utils; config_utils.prepare_training_data_yaml('${COCO_DATASET}', '.', include_frigate_plus=${USE_FRIGATE_PLUS_PY}, frigate_plus_dir='${FRIGATE_PLUS_DIR}')"
    fi

    if [[ ! -f "${DATA_YAML}" ]]; then
        log_err "Dataset configuration file '${DATA_YAML}' not found. Run with --export first."
        exit 1
    fi

    echo ""
    echo "=========================================================================="
    echo "  Step 2: Training Locally (Device: ${DEVICE}, Batch: ${BATCH})"
    echo "=========================================================================="

    # Relocate any stray *.pt files in root to models/base/
    relocate_root_checkpoints

    START_MODEL="${BASE_MODEL}"
    if [[ -f "${MODELS_BASE_DIR}/${BASE_MODEL}" ]]; then
        START_MODEL="${MODELS_BASE_DIR}/${BASE_MODEL}"
    elif [[ -f "${MODELS_BASE_DIR}/${BASE_MODEL_NAME}.pt" ]]; then
        START_MODEL="${MODELS_BASE_DIR}/${BASE_MODEL_NAME}.pt"
    fi

    if [[ "${FROM_SCRATCH}" == "false" ]]; then
        LATEST_BEST=$(find "runs/detect" -path "*${RUN_NAME}*" -name "best.pt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        if [[ -n "${LATEST_BEST}" && -f "${LATEST_BEST}" ]]; then
            log_info "Found existing trained checkpoint: ${LATEST_BEST}. Fine-tuning from it..."
            START_MODEL="${LATEST_BEST}"
        fi
    fi

    log_info "Executing: yolo detect train on device='${DEVICE}' (optimizer=${OPTIMIZER}, lr0=${LR0}, warmup=${WARMUP_EPOCHS})..."
    yolo detect train \
        data="${DATA_YAML}" \
        model="${START_MODEL}" \
        epochs="${EPOCHS}" \
        imgsz="${IMGSZ}" \
        batch="${BATCH}" \
        workers="${WORKERS}" \
        cache="${CACHE}" \
        patience="${PATIENCE}" \
        optimizer="${OPTIMIZER}" \
        lr0="${LR0}" \
        lrf="${LRF}" \
        cos_lr="${COS_LR}" \
        warmup_epochs="${WARMUP_EPOCHS}" \
        warmup_momentum="${WARMUP_MOMENTUM}" \
        warmup_bias_lr="${WARMUP_BIAS_LR}" \
        device="${DEVICE}" \
        project="${PROJECT}" \
        name="${RUN_NAME}" \
        ${EXTRA_TRAIN_ARGS}

    # Relocate downloaded base weights
    relocate_root_checkpoints

    TRAINED_BEST="${LOCAL_WEIGHTS_DIR}/best.pt"
    if [[ ! -f "${TRAINED_BEST}" ]]; then
        TRAINED_BEST=$(find "runs/detect/${PROJECT}" -path "*${RUN_NAME}*" -name "best.pt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    fi
    if [[ -z "${TRAINED_BEST}" || ! -f "${TRAINED_BEST}" ]]; then
        log_err "Training completed but could not locate best.pt!"
        exit 1
    fi
    WEIGHTS_PATH="${TRAINED_BEST}"
    log_success "Training complete! Checkpoint: ${WEIGHTS_PATH}"
fi

# ------------------------------------------------------------------------------
# Step 3: Model Export & Target Compilation
# ------------------------------------------------------------------------------
if [[ "${SKIP_COMPILE}" == "true" ]]; then
    log_info "--skip-compile specified: exiting pipeline before compilation."
    exit 0
fi

if [[ -z "${WEIGHTS_PATH}" || ! -f "${WEIGHTS_PATH}" ]]; then
    log_err "Cannot compile: weights checkpoint '${WEIGHTS_PATH}' not found."
    exit 1
fi

echo ""
echo "=========================================================================="
if [[ "${TARGET_FORMAT}" == "nvidia" ]]; then
    echo "  Step 3: NVIDIA GPU ONNX Export (${IMGSZ}x${IMGSZ})"
else
    echo "  Step 3: Torq NPU Export, Quantization & Compilation (${IMGSZ}x${IMGSZ})"
fi
echo "=========================================================================="

EXPORTED_ONNX="${WEIGHTS_PATH%.pt}.onnx"
if [[ ! -f "${EXPORTED_ONNX}" || "${WEIGHTS_PATH}" -nt "${EXPORTED_ONNX}" ]]; then
    log_info "Exporting PyTorch model to ONNX (${IMGSZ}x${IMGSZ})..."
    yolo export \
        model="${WEIGHTS_PATH}" \
        format=onnx \
        imgsz="${IMGSZ}" \
        dynamic=False \
        simplify=True \
        opset=17
fi

if [[ ! -f "${EXPORTED_ONNX}" ]]; then
    log_err "Failed to locate exported ONNX at ${EXPORTED_ONNX}"
    exit 1
fi

cp -f "${EXPORTED_ONNX}" "${OUTPUT_DIR}/best.onnx"

if [[ "${TARGET_FORMAT}" == "nvidia" ]]; then
    mkdir -p "$(dirname "${OUTPUT_ONNX}")"
    cp -f "${EXPORTED_ONNX}" "${OUTPUT_ONNX}"
    python3 -c "from prepare_for_onnx2tf import step4_generate_labels_json; step4_generate_labels_json('${DATA_YAML}', '${LABELS_FILE}', '', '${BASE_MODEL_NAME}', '${BASE_MODEL_NAME}', target='${TARGET_FORMAT}')"
    log_success "NVIDIA GPU ONNX model generated successfully: ${OUTPUT_ONNX} ($(du -h "${OUTPUT_ONNX}" | cut -f1))"
    log_info "Target is NVIDIA GPU ('${TARGET_FORMAT}'): skipping Torq NPU INT8 quantization and .vmfb compilation."
    PRIMARY_ARTIFACT="${OUTPUT_ONNX}"
else
    log_info "Normalizing ONNX coordinates and generating calibration dataset..."
    python3 prepare_for_onnx2tf.py \
        --skip-export \
        --input-onnx "${OUTPUT_DIR}/best.onnx" \
        --output-onnx "${OUTPUT_DIR}/best_normalized_output.onnx" \
        --output-calib "${OUTPUT_DIR}/calibration_video_300_nhwc_float32.npy" \
        --output-labels "${LABELS_FILE}" \
        --data-yaml "${DATA_YAML}" \
        --imgsz "${IMGSZ}" \
        --base-model-name "${BASE_MODEL_NAME}" \
        --target "${TARGET_FORMAT}"

    log_info "Executing Full INT8 Quantization via onnx2tf..."
    rm -rf "${OUTPUT_DIR}/onnx2tf_normalized_full_int8"

    onnx2tf \
        -i "${OUTPUT_DIR}/best_normalized_output.onnx" \
        -o "${OUTPUT_DIR}/onnx2tf_normalized_full_int8" \
        -oiqt -iqd int8 -oqd int8 \
        --tflite_backend tf_converter \
        -cind images "${OUTPUT_DIR}/calibration_video_300_nhwc_float32.npy" \
        "[[[[0.0,0.0,0.0]]]]" "[[[[1.0,1.0,1.0]]]]"

    FULL_INT8_TFLITE="${OUTPUT_DIR}/onnx2tf_normalized_full_int8/best_normalized_output_full_integer_quant.tflite"
    if [[ ! -f "${FULL_INT8_TFLITE}" ]]; then
        log_err "onnx2tf failed to generate full integer quantized TFLite at ${FULL_INT8_TFLITE}"
        exit 1
    fi

    # Update labels with exact TFLite quantization scales/zero points
    python3 -c "from prepare_for_onnx2tf import step4_generate_labels_json; step4_generate_labels_json('${DATA_YAML}', '${LABELS_FILE}', '${FULL_INT8_TFLITE}', '${BASE_MODEL_NAME}', '${BASE_MODEL_NAME}', target='${TARGET_FORMAT}')"

    log_info "Converting TFLite to TOSA MLIR..."
    TOSA_MLIR="${OUTPUT_DIR}/${RUN_NAME_STEM}_normalized_full_integer.tosa.mlir"
    tosa-converter-for-tflite \
        "${FULL_INT8_TFLITE}" \
        --text \
        -o "${TOSA_MLIR}"

    log_info "Compiling TOSA MLIR for Synaptics Astra SL2610 Torq NPU..."
    torq-compile \
        --torq-hw=SL2610 \
        "${TOSA_MLIR}" \
        -o "${OUTPUT_VMFB}"

    if [[ ! -f "${OUTPUT_VMFB}" ]]; then
        log_err "torq-compile failed to produce VMFB binary at ${OUTPUT_VMFB}"
        exit 1
    fi
    log_success "Torq VMFB generated successfully: ${OUTPUT_VMFB} ($(du -h "${OUTPUT_VMFB}" | cut -f1))"
    PRIMARY_ARTIFACT="${OUTPUT_VMFB}"
fi

# ------------------------------------------------------------------------------
# Create Non-Timestamped Soft Links
# ------------------------------------------------------------------------------
MODEL_EXT="${PRIMARY_ARTIFACT##*.}"
MODEL_SYMLINK="${MODELS_COMPILED_DIR}/${TARGET_FORMAT}_${RUN_NAME_STEM}.${MODEL_EXT}"
ln -sf "$(basename "${PRIMARY_ARTIFACT}")" "${MODEL_SYMLINK}"
log_info "Created soft link: ${MODEL_SYMLINK} -> $(basename "${PRIMARY_ARTIFACT}")"

LABELS_JSON_SYMLINK="${MODELS_LABELS_DIR}/${TARGET_FORMAT}_${BASE_MODEL_NAME}.json"
ln -sf "$(basename "${LABELS_FILE}")" "${LABELS_JSON_SYMLINK}"
log_info "Created soft link: ${LABELS_JSON_SYMLINK} -> $(basename "${LABELS_FILE}")"

LABELS_TXT_FILE="${LABELS_FILE%.json}.txt"
LABELS_TXT_SYMLINK="${MODELS_LABELS_DIR}/${TARGET_FORMAT}_${BASE_MODEL_NAME}.txt"
if [[ -f "${LABELS_TXT_FILE}" ]]; then
    ln -sf "$(basename "${LABELS_TXT_FILE}")" "${LABELS_TXT_SYMLINK}"
    log_info "Created soft link: ${LABELS_TXT_SYMLINK} -> $(basename "${LABELS_TXT_FILE}")"
fi

ln -sf "$(basename "${LABELS_FILE}")" "${MODELS_LABELS_DIR}/labels.json"
if [[ -f "${LABELS_TXT_FILE}" ]]; then
    ln -sf "$(basename "${LABELS_TXT_FILE}")" "${MODELS_LABELS_DIR}/labels.txt"
fi

# ------------------------------------------------------------------------------
# Step 4: Optional Deployment
# ------------------------------------------------------------------------------
if [[ "${SKIP_DEPLOY}" == "false" ]]; then
    ARTIFACT_BASENAME="$(basename "${PRIMARY_ARTIFACT}")"
    MODEL_SYMLINK_BASENAME="$(basename "${MODEL_SYMLINK}")"
    LABELS_BASENAME="$(basename "${LABELS_FILE}")"
    LABELS_JSON_SYMLINK_BASENAME="$(basename "${LABELS_JSON_SYMLINK}")"
    LABELS_TXT_FILE="${LABELS_FILE%.json}.txt"
    LABELS_TXT_BASENAME="$(basename "${LABELS_TXT_FILE}")"
    LABELS_TXT_SYMLINK_BASENAME="$(basename "${LABELS_TXT_SYMLINK}")"

    if [[ -n "${DEPLOY_DIR}" ]]; then
        echo ""
        log_info "Deploying artifacts to local directory: ${DEPLOY_DIR}..."
        mkdir -p "${DEPLOY_DIR}"
        cp -f "${PRIMARY_ARTIFACT}" "${DEPLOY_DIR}/${ARTIFACT_BASENAME}"
        cp -f "${PRIMARY_ARTIFACT}" "${DEPLOY_DIR}/${MODEL_SYMLINK_BASENAME}"
        cp -f "${LABELS_FILE}" "${DEPLOY_DIR}/${LABELS_BASENAME}"
        cp -f "${LABELS_FILE}" "${DEPLOY_DIR}/${LABELS_JSON_SYMLINK_BASENAME}"
        cp -f "${LABELS_FILE}" "${DEPLOY_DIR}/labels.json"
        if [[ -f "${LABELS_TXT_FILE}" ]]; then
            cp -f "${LABELS_TXT_FILE}" "${DEPLOY_DIR}/${LABELS_TXT_BASENAME}"
            cp -f "${LABELS_TXT_FILE}" "${DEPLOY_DIR}/${LABELS_TXT_SYMLINK_BASENAME}"
            cp -f "${LABELS_TXT_FILE}" "${DEPLOY_DIR}/labels.txt"
        fi
        log_success "Deployed ${ARTIFACT_BASENAME}, ${LABELS_BASENAME}, and ${LABELS_TXT_BASENAME} to ${DEPLOY_DIR}/"
    fi

    if [[ -n "${DEPLOY_HOST}" ]]; then
        echo ""
        CLEAN_HOST="${DEPLOY_HOST%/}"
        log_info "Deploying artifacts via SCP to ${CLEAN_HOST}..."
        scp "${PRIMARY_ARTIFACT}" "${CLEAN_HOST}/${ARTIFACT_BASENAME}"
        scp "${PRIMARY_ARTIFACT}" "${CLEAN_HOST}/${MODEL_SYMLINK_BASENAME}"
        scp "${LABELS_FILE}" "${CLEAN_HOST}/${LABELS_BASENAME}"
        scp "${LABELS_FILE}" "${CLEAN_HOST}/${LABELS_JSON_SYMLINK_BASENAME}"
        scp "${LABELS_FILE}" "${CLEAN_HOST}/labels.json"
        if [[ -f "${LABELS_TXT_FILE}" ]]; then
            scp "${LABELS_TXT_FILE}" "${CLEAN_HOST}/${LABELS_TXT_BASENAME}"
            scp "${LABELS_TXT_FILE}" "${CLEAN_HOST}/${LABELS_TXT_SYMLINK_BASENAME}"
            scp "${LABELS_TXT_FILE}" "${CLEAN_HOST}/labels.txt"
        fi
        log_success "Deployed ${ARTIFACT_BASENAME}, ${LABELS_BASENAME}, and ${LABELS_TXT_BASENAME} via SCP to ${CLEAN_HOST}!"
    fi
fi

# ------------------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================================="
if [[ "${TARGET_FORMAT}" == "nvidia" ]]; then
    echo -e "${GREEN}  PIPELINE COMPLETE - ONNX MODEL READY FOR NVIDIA GPU INFERENCE!${NC}"
    echo "=========================================================================="
    echo "Artifacts Generated:"
    echo "  1. ONNX Model      : ${OUTPUT_ONNX} ($(du -h "${OUTPUT_ONNX}" | cut -f1))"
    echo "     Soft Link       : ${MODEL_SYMLINK}"
else
    echo -e "${GREEN}  PIPELINE COMPLETE - MODEL READY FOR NPU INFERENCE!${NC}"
    echo "=========================================================================="
    echo "Artifacts Generated:"
    echo "  1. Hardware Binary : ${OUTPUT_VMFB} ($(du -h "${OUTPUT_VMFB}" | cut -f1))"
    echo "     Soft Link       : ${MODEL_SYMLINK}"
fi
echo "  2. Label Metadata  : ${LABELS_FILE}"
echo "     Soft Link       : ${LABELS_JSON_SYMLINK}"
if [[ -f "${LABELS_FILE%.json}.txt" ]]; then
    echo "  3. Frigate Labelmap: ${LABELS_FILE%.json}.txt"
    echo "     Soft Link       : ${LABELS_TXT_SYMLINK}"
    echo "  4. PyTorch Weights : ${WEIGHTS_PATH}"
else
    echo "  3. PyTorch Weights : ${WEIGHTS_PATH}"
fi
echo "=========================================================================="
