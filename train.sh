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
DATA_YAML="yolo_dataset/data.yaml"
IMGSZ=320                         # Strictly 320 for Coralboard SL2619 NSS slices
EPOCHS=80
PATIENCE=50
BATCH=""                          # Evaluated based on runner if not provided
WORKERS=8
CACHE="ram"
PROJECT="frigate_coral"
BASE_MODEL="yolov8n.pt"
RUN_NAME=""
LABELS_FILE=""
OUTPUT_VMFB=""
FROM_SCRATCH=true
EXTRA_TRAIN_ARGS=""

# Remote Spark specific defaults
SPARK_HOST="${SPARK_HOST:-${DGX_HOST:-}}"
REMOTE_DIR="~/frigate-model-training"
COCO_DATASET="coco128"
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
    echo "  Frigate Model Training & Torq NPU Compiler (Coralboard SL2619)"
    echo "=========================================================================="
    echo "Usage: bash $0 [PROFILE] [WORKFLOW] [OPTIONS]"
    echo ""
    echo "Default (no flags): Runs default profile from config.yaml ('local' GPU),"
    echo "quantizes to INT8, compiles to Torq NPU binary, and deploys if configured."
    echo ""
    echo "Execution Profiles:"
    echo "  -p, --profile NAME      Select profile from config.yaml (e.g. 'local', 'spark', 'cpu')"
    echo ""
    echo "Primary Workflows:"
    echo "  --compile [MODEL.pt]    Skip training; compile checkpoint directly (or latest best.pt)"
    echo "  --export [TASK_ID|all]  Export CVAT task(s) to yolo_dataset/ before training (default: latest)"
    echo "  --deploy [DEST]         Deploy compiled VMFB & labels to local dir or remote host (user@ip:/path)"
    echo ""
    echo "Workflow Controls:"
    echo "  --skip-compile          Train only; skip local INT8 quantization & Torq compilation"
    echo "  --skip-deploy           Skip deployment step even if configured in config.yaml"
    echo "  --pull                  Pull latest weights from Spark and compile locally (skip training)"
    echo ""
    echo "Profile Parameter Overrides:"
    echo "  -m, --base-model MODEL  Base checkpoint / architecture (default: yolov8n.pt)"
    echo "  --epochs N              Training epochs"
    echo "  --batch N               Batch size"
    echo "  --device DEV            Training device ('0' local GPU, 'cpu', or '0,1' Spark)"
    echo "  --host USER@HOST        SSH destination for remote training"
    echo "  --resume                Resume/fine-tune from previous checkpoint (default: train from scratch)"
    echo "  --coco-dataset NAME     COCO subset for co-training ('coco128' or 'coco-val2017')"
    echo "  --replace               Wipe and replace existing dataset instead of appending"
    echo ""
    echo "Advanced & Environment:"
    echo "  -c, --config PATH       Path to YAML configuration file (default: config.yaml)"
    echo "  --remote-dir DIR        Remote workspace directory on server"
    echo "  --clean-images          Clean up training images on remote node after run completes"
    echo "  --workers N             Dataloader CPU workers"
    echo "  --cache MODE            Dataset cache mode ('ram', 'disk', 'false')"
    echo "  --run-name NAME         Ultralytics run name (default: derived from base model)"
    echo "  --patience N            Early stopping patience epochs (default: 50)"
    echo "  --extra-args ARGS       Forward extra arguments directly to 'yolo detect train'"
    echo "  -h, --help              Show this help message"
    echo ""
    exit "${exit_code}"
}

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
PROFILE=""
EXPLICIT_RUN_NAME=false
EXPLICIT_LABELS_FILE=false
EXPLICIT_HOST=false
EXPLICIT_BATCH=false
EXPLICIT_DEVICE=false
EXPLICIT_EPOCHS=false
EXPLICIT_PATIENCE=false
EXPLICIT_WORKERS=false
EXPLICIT_CACHE=false
EXPLICIT_BASE_MODEL=false
EXPLICIT_COCO=false
EXPLICIT_CLEAN_IMAGES=false
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
        --remote-dir)
            REMOTE_DIR="$2"
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
        --coco-dataset|--coco)
            COCO_DATASET="$2"
            EXPLICIT_COCO=true
            shift 2
            ;;
        --data-yaml)
            DATA_YAML="$2"
            shift 2
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
        --workers)
            WORKERS="$2"
            EXPLICIT_WORKERS=true
            shift 2
            ;;
        --cache)
            CACHE="$2"
            EXPLICIT_CACHE=true
            shift 2
            ;;
        --device)
            DEVICE="$2"
            EXPLICIT_DEVICE=true
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            EXPLICIT_RUN_NAME=true
            shift 2
            ;;
        --resume)
            FROM_SCRATCH=false
            shift
            ;;
        --patience)
            PATIENCE="$2"
            EXPLICIT_PATIENCE=true
            shift 2
            ;;
        --extra-args)
            EXTRA_TRAIN_ARGS="$2"
            shift 2
            ;;
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        --output-vmfb)
            OUTPUT_VMFB="$2"
            shift 2
            ;;
        --labels-file)
            LABELS_FILE="$2"
            EXPLICIT_LABELS_FILE=true
            shift 2
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
        --clean-images)
            CLEAN_IMAGES=true
            EXPLICIT_CLEAN_IMAGES=true
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
    eval "$(python3 config_utils.py --profile-env "${PROFILE}" --config "${CONFIG_FILE}" 2>/dev/null || true)"
fi

PROFILE_NAME="${PROFILE_NAME:-local}"

if [[ "${PROFILE_REMOTE:-false}" == "true" || "${PULL_ONLY}" == "true" || "${EXPLICIT_HOST}" == "true" ]]; then
    RUNNER="spark"
else
    RUNNER="local"
fi

if [[ "${EXPLICIT_HOST}" == "false" && -n "${PROFILE_HOST}" ]]; then
    SPARK_HOST="${PROFILE_HOST}"
fi

if [[ "${EXPLICIT_BATCH}" == "false" && -n "${PROFILE_BATCH}" ]]; then
    BATCH="${PROFILE_BATCH}"
fi

if [[ "${EXPLICIT_DEVICE}" == "false" && -n "${PROFILE_DEVICE}" ]]; then
    DEVICE="${PROFILE_DEVICE}"
fi

if [[ "${EXPLICIT_EPOCHS}" == "false" && -n "${PROFILE_EPOCHS}" ]]; then
    EPOCHS="${PROFILE_EPOCHS}"
fi

if [[ "${EXPLICIT_PATIENCE}" == "false" && -n "${PROFILE_PATIENCE}" ]]; then
    PATIENCE="${PROFILE_PATIENCE}"
fi

if [[ "${EXPLICIT_WORKERS}" == "false" && -n "${PROFILE_WORKERS}" ]]; then
    WORKERS="${PROFILE_WORKERS}"
fi

if [[ "${EXPLICIT_CACHE}" == "false" && -n "${PROFILE_CACHE}" ]]; then
    CACHE="${PROFILE_CACHE}"
fi

if [[ "${EXPLICIT_BASE_MODEL}" == "false" && -n "${PROFILE_BASE_MODEL}" ]]; then
    BASE_MODEL="${PROFILE_BASE_MODEL}"
fi

if [[ "${EXPLICIT_COCO}" == "false" && -n "${PROFILE_COCO_DATASET}" ]]; then
    COCO_DATASET="${PROFILE_COCO_DATASET}"
fi

if [[ "${EXPLICIT_CLEAN_IMAGES}" == "false" && "${PROFILE_CLEAN_IMAGES:-false}" == "true" ]]; then
    CLEAN_IMAGES=true
fi

if [[ -n "${PROFILE_REMOTE_DIR}" ]]; then
    REMOTE_DIR="${PROFILE_REMOTE_DIR}"
fi

if [[ "${EXPLICIT_DEPLOY_HOST}" == "false" && -z "${DEPLOY_HOST}" && -n "${PROFILE_DEPLOY_HOST}" ]]; then
    DEPLOY_HOST="${PROFILE_DEPLOY_HOST}"
fi

if [[ "${EXPLICIT_DEPLOY_DIR}" == "false" && -z "${DEPLOY_DIR}" && -n "${PROFILE_DEPLOY_DIR}" ]]; then
    DEPLOY_DIR="${PROFILE_DEPLOY_DIR}"
fi

# ------------------------------------------------------------------------------
# Normalize Model Names & Dynamic Defaults
# ------------------------------------------------------------------------------
BASE_MODEL_NAME="$(basename "${BASE_MODEL%.*}")"
if [[ "${EXPLICIT_RUN_NAME}" == "false" ]]; then
    RUN_NAME="${BASE_MODEL_NAME}"
fi
RUN_NAME_STEM="$(basename "${RUN_NAME%.*}")"

if [[ "${EXPLICIT_LABELS_FILE}" == "false" ]]; then
    LABELS_FILE="${MODELS_LABELS_DIR}/labels_${BASE_MODEL_NAME}.json"
fi

if [[ -z "${OUTPUT_VMFB}" ]]; then
    OUTPUT_VMFB="${MODELS_COMPILED_DIR}/custom_${RUN_NAME_STEM}.vmfb"
fi

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
echo "  Hyperparams:   Batch: ${BATCH}, Workers: ${WORKERS}, Epochs: ${EPOCHS}, Patience: ${PATIENCE}, Cache: ${CACHE}"
echo "  Base Model:    ${BASE_MODEL} (Run Name: ${RUN_NAME})"
echo "  Resolution:    ${IMGSZ}x${IMGSZ} (strictly required for Coralboard SL2619)"
echo "=========================================================================="

# ------------------------------------------------------------------------------
# Tool Availability Verification
# ------------------------------------------------------------------------------
REQUIRED_TOOLS=(python3 yolo)
if [[ "${SKIP_COMPILE}" == "false" ]]; then
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
    DATA_DIR="$(dirname "${DATA_YAML}")"
    if [[ "${DATA_DIR}" != "." && "${DATA_DIR}" != "yolo_dataset" ]]; then
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

        if command -v rsync &>/dev/null; then
            rsync -avz --copy-links \
                "${SCRIPTS_TO_SYNC[@]}" \
                "${SPARK_HOST}:${REMOTE_DIR}/"

            rsync -avz --delete \
                --exclude="coco*" \
                --exclude="*.zip" \
                --exclude="*.tar*" \
                --exclude="__pycache__" \
                yolo_dataset/ "${SPARK_HOST}:${REMOTE_DIR}/yolo_dataset/"
        else
            scp "${SCRIPTS_TO_SYNC[@]}" "${SPARK_HOST}:${REMOTE_DIR}/"
            ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/yolo_dataset'/{images,labels}"
            scp -r yolo_dataset/images yolo_dataset/labels "${SPARK_HOST}:${REMOTE_DIR}/yolo_dataset/"
        fi

        # Sync previous checkpoint if resuming or base model if present locally
        if [[ "${FROM_SCRATCH}" == "false" && -f "${LOCAL_WEIGHTS_DIR}/best.pt" ]]; then
            ssh "${SPARK_HOST}" "mkdir -p '${REMOTE_DIR}/runs/detect/frigate_coral/${RUN_NAME}/weights'"
            scp "${LOCAL_WEIGHTS_DIR}/best.pt" "${SPARK_HOST}:${REMOTE_DIR}/runs/detect/frigate_coral/${RUN_NAME}/weights/best.pt"
        fi
        if [[ -f "models/base/${BASE_MODEL}" ]]; then
            scp "models/base/${BASE_MODEL}" "${SPARK_HOST}:${REMOTE_DIR}/models/base/"
        elif [[ -f "${BASE_MODEL}" ]]; then
            scp "${BASE_MODEL}" "${SPARK_HOST}:${REMOTE_DIR}/models/base/"
        fi

        # Prepare remote COCO dataset
        log_info "Preparing remote COCO dataset '${COCO_DATASET}' on Spark..."
        ssh "${SPARK_HOST}" "
            set -e
            cd \"${REMOTE_DIR}\"
            source \"${REMOTE_DIR}/venv/bin/activate\"
            export PYTHONPATH=\"${REMOTE_DIR}:\${PYTHONPATH:-}\"
            python3 -c \"import config_utils; config_utils.prepare_training_data_yaml('${COCO_DATASET}', '${REMOTE_DIR}')\"
        "

        # Launch training on Spark
        log_info "Launching YOLOv8 training on ${SPARK_HOST} (device=${DEVICE}, batch=${BATCH}, workers=${WORKERS}, cache=${CACHE}, epochs=${EPOCHS})..."
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
                LATEST_EXISTING=\$(find \"${REMOTE_DIR}/runs/detect/frigate_coral\" -path \"*${RUN_NAME}*\" -name \"best.pt\" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
                if [ -n \"\${LATEST_EXISTING}\" ] && [ -f \"\${LATEST_EXISTING}\" ]; then
                    echo \"[SPARK] Resuming training from existing checkpoint: \${LATEST_EXISTING}\"
                    START_MODEL=\"\${LATEST_EXISTING}\"
                fi
            fi

            echo '=========================================================================='
            echo '  Executing: yolo detect train on NVIDIA DGX Spark'
            echo '=========================================================================='
            yolo detect train \
                data='yolo_dataset/data.yaml' \
                model=\"\${START_MODEL}\" \
                epochs=${EPOCHS} \
                imgsz=320 \
                batch=${BATCH} \
                workers=${WORKERS} \
                cache=${CACHE} \
                patience=${PATIENCE} \
                device='${DEVICE}' \
                project='frigate_coral' \
                name='${RUN_NAME}' \
                ${EXTRA_TRAIN_ARGS}
        "
    fi

    # Retrieve trained weights directly from Spark to local system before local compilation
    log_info "Retrieving trained best.pt from ${SPARK_HOST} to local system..."
    mkdir -p "${LOCAL_WEIGHTS_DIR}" "models/base"

    REMOTE_BEST_PT=$(ssh "${SPARK_HOST}" "
        find \"${REMOTE_DIR}/runs/detect/frigate_coral\" -path \"*${RUN_NAME}*\" -name \"best.pt\" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d' '
    ")

    if [[ -z "${REMOTE_BEST_PT}" ]]; then
        log_err "Could not locate best.pt on ${SPARK_HOST}:${REMOTE_DIR}/runs/detect/frigate_coral/"
        exit 1
    fi

    scp "${SPARK_HOST}:${REMOTE_BEST_PT}" "${LOCAL_WEIGHTS_DIR}/best.pt"
    WEIGHTS_PATH="${LOCAL_WEIGHTS_DIR}/best.pt"
    log_success "Trained weights retrieved successfully to ${LOCAL_WEIGHTS_DIR}/best.pt"

    # Clean remote images if requested
    if [[ "${CLEAN_IMAGES}" == "true" ]]; then
        log_info "Cleaning up remote training images on ${SPARK_HOST}..."
        ssh "${SPARK_HOST}" "rm -rf \"${REMOTE_DIR}/yolo_dataset/images\" \"${REMOTE_DIR}/datasets\"" || true
    fi

else
    # --------------------------------------------------------------------------
    # Local GPU/CPU Training (Default)
    # --------------------------------------------------------------------------
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

    log_info "Executing: yolo detect train on device='${DEVICE}'..."
    yolo detect train \
        data="${DATA_YAML}" \
        model="${START_MODEL}" \
        epochs="${EPOCHS}" \
        imgsz="${IMGSZ}" \
        batch="${BATCH}" \
        workers="${WORKERS}" \
        cache="${CACHE}" \
        patience="${PATIENCE}" \
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
# Step 3: Local INT8 Quantization & Torq Compilation
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
echo "  Step 3: Torq NPU Export, Quantization & Compilation (${IMGSZ}x${IMGSZ})"
echo "=========================================================================="

EXPORTED_ONNX="${WEIGHTS_PATH%.pt}.onnx"
if [[ ! -f "${EXPORTED_ONNX}" ]]; then
    log_info "Exporting PyTorch model to ONNX (${IMGSZ}x${IMGSZ})..."
    yolo export \
        model="${WEIGHTS_PATH}" \
        format=onnx \
        imgsz="${IMGSZ}" \
        dynamic=False \
        opset=17
fi

if [[ ! -f "${EXPORTED_ONNX}" ]]; then
    log_err "Failed to locate exported ONNX at ${EXPORTED_ONNX}"
    exit 1
fi

cp -f "${EXPORTED_ONNX}" "${OUTPUT_DIR}/best.onnx"

log_info "Normalizing ONNX coordinates and generating calibration dataset..."
python3 prepare_for_onnx2tf.py \
    --skip-export \
    --input-onnx "${OUTPUT_DIR}/best.onnx" \
    --output-onnx "${OUTPUT_DIR}/best_normalized_output.onnx" \
    --output-calib "${OUTPUT_DIR}/calibration_video_300_nhwc_float32.npy" \
    --output-labels "${LABELS_FILE}" \
    --data-yaml "${DATA_YAML}" \
    --imgsz "${IMGSZ}" \
    --base-model-name "${BASE_MODEL_NAME}"

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
python3 -c "from prepare_for_onnx2tf import step4_generate_labels_json; step4_generate_labels_json('${DATA_YAML}', '${LABELS_FILE}', '${FULL_INT8_TFLITE}', '${BASE_MODEL_NAME}', '${BASE_MODEL_NAME}')"

log_info "Converting TFLite to TOSA MLIR..."
TOSA_MLIR="${OUTPUT_DIR}/custom_${RUN_NAME_STEM}_normalized_full_integer.tosa.mlir"
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

# ------------------------------------------------------------------------------
# Step 4: Optional Deployment to Coralboard
# ------------------------------------------------------------------------------
if [[ "${SKIP_DEPLOY}" == "false" ]]; then
    OUTPUT_VMFB_BASENAME="$(basename "${OUTPUT_VMFB}")"
    LABELS_BASENAME="$(basename "${LABELS_FILE}")"

    if [[ -n "${DEPLOY_DIR}" ]]; then
        echo ""
        log_info "Deploying artifacts to local directory: ${DEPLOY_DIR}..."
        mkdir -p "${DEPLOY_DIR}"
        cp -f "${OUTPUT_VMFB}" "${DEPLOY_DIR}/${OUTPUT_VMFB_BASENAME}"
        cp -f "${LABELS_FILE}" "${DEPLOY_DIR}/${LABELS_BASENAME}"
        cp -f "${LABELS_FILE}" "${DEPLOY_DIR}/labels.json"
        log_success "Deployed ${OUTPUT_VMFB_BASENAME} and ${LABELS_BASENAME} to ${DEPLOY_DIR}/"
    fi

    if [[ -n "${DEPLOY_HOST}" ]]; then
        echo ""
        CLEAN_HOST="${DEPLOY_HOST%/}"
        log_info "Deploying artifacts via SCP to ${CLEAN_HOST}..."
        scp "${OUTPUT_VMFB}" "${CLEAN_HOST}/${OUTPUT_VMFB_BASENAME}"
        scp "${LABELS_FILE}" "${CLEAN_HOST}/${LABELS_BASENAME}"
        scp "${LABELS_FILE}" "${CLEAN_HOST}/labels.json"
        log_success "Deployed ${OUTPUT_VMFB_BASENAME} and ${LABELS_BASENAME} (and labels.json) via SCP to ${CLEAN_HOST}!"
    fi
fi

# ------------------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo -e "${GREEN}  PIPELINE COMPLETE - MODEL READY FOR NPU INFERENCE!${NC}"
echo "=========================================================================="
echo "Artifacts Generated:"
echo "  1. Hardware Binary : ${OUTPUT_VMFB} ($(du -h "${OUTPUT_VMFB}" | cut -f1))"
echo "  2. Label Metadata  : ${LABELS_FILE}"
echo "  3. PyTorch Weights : ${WEIGHTS_PATH}"
echo "=========================================================================="
