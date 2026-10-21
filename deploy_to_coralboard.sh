#!/usr/bin/env bash
# ==============================================================================
# Deploy Compiled YOLO Model & Labels to Coralboard SL2619 Torq Vision Service
# ==============================================================================
# Ensures BOTH the compiled .vmfb hardware binary and labels.json are distributed
# together to avoid class index mismatches (e.g. COCO-80 fallback misclassifications).
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
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()     { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------------------
# Default Settings & Configuration Discovery
# ------------------------------------------------------------------------------
MODEL_PATH=""
LABELS_PATH=""
DEPLOY_HOST="${DEPLOY_HOST:-}"
DEPLOY_DIR="${DEPLOY_DIR:-}"
RESTART_SERVICE=false
SERVICE_NAME="coralboard-sl2619-vision"
DO_HEALTH_CHECK=false

# Load settings from config.yaml if available
if [[ -f "config.yaml" ]]; then
    PYTHON_BIN="python3"
    if [[ -x "./venv-torq-312/bin/python" ]]; then
        PYTHON_BIN="./venv-torq-312/bin/python"
    elif [[ -x "./venv/bin/python" ]]; then
        PYTHON_BIN="./venv/bin/python"
    fi

    ${PYTHON_BIN} -c "import yaml" &>/dev/null && {
        CFG_HOST=$(${PYTHON_BIN} -c "import yaml; c=yaml.safe_load(open('config.yaml')).get('deployment',{}) or {}; print(c.get('host') or '')" 2>/dev/null || true)
        CFG_DIR=$(${PYTHON_BIN} -c "import yaml; c=yaml.safe_load(open('config.yaml')).get('deployment',{}) or {}; print(c.get('dir') or '')" 2>/dev/null || true)
        CFG_RESTART=$(${PYTHON_BIN} -c "import yaml; c=yaml.safe_load(open('config.yaml')).get('deployment',{}) or {}; print('true' if c.get('restart_service') else 'false')" 2>/dev/null || true)
        CFG_SERVICE=$(${PYTHON_BIN} -c "import yaml; c=yaml.safe_load(open('config.yaml')).get('deployment',{}) or {}; print(c.get('service_name') or '')" 2>/dev/null || true)

        if [[ -z "${DEPLOY_HOST}" && -n "${CFG_HOST}" ]]; then
            DEPLOY_HOST="${CFG_HOST}"
        fi
        if [[ -z "${DEPLOY_DIR}" && -n "${CFG_DIR}" ]]; then
            DEPLOY_DIR="${CFG_DIR}"
        fi
        if [[ "${CFG_RESTART}" == "true" ]]; then
            RESTART_SERVICE=true
        fi
        if [[ -n "${CFG_SERVICE}" ]]; then
            SERVICE_NAME="${CFG_SERVICE}"
        fi
    } || true
fi

# ------------------------------------------------------------------------------
# Usage & Help
# ------------------------------------------------------------------------------
usage() {
    echo "Usage: bash $0 [OPTIONS]"
    echo ""
    echo "Deployment Targets:"
    echo "  --host USER@HOST:PATH   Remote destination for SCP (e.g. root@192.168.100.2:/root/coralboard-sl2619-vision/models)"
    echo "  --dir PATH              Local directory destination (e.g. ../coralboard-sl2619-vision/models)"
    echo ""
    echo "Artifact Options:"
    echo "  --model PATH            Path to compiled .vmfb model (default: auto-detected models/compiled/torq_*.vmfb)"
    echo "  --labels PATH           Path to labels JSON mapping (default: auto-detected models/labels/torq_<name>.json)"
    echo ""
    echo "Service Management:"
    echo "  --restart               Restart the remote vision service via SSH after deployment"
    echo "  --service-name NAME     Name of systemd service on Coralboard (default: coralboard-sl2619-vision)"
    echo "  --check                 Perform HTTP health check on Coralboard after deployment"
    echo ""
    echo "Help:"
    echo "  -h, --help              Show this help message"
    echo ""
    exit 0
}

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            DEPLOY_HOST="$2"
            shift 2
            ;;
        --dir)
            DEPLOY_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --labels)
            LABELS_PATH="$2"
            shift 2
            ;;
        --restart)
            RESTART_SERVICE=true
            shift
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --check)
            DO_HEALTH_CHECK=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_err "Unknown argument: $1"
            usage
            ;;
    esac
done

echo "=========================================================================="
echo "  Deploy Model & Labels to Coralboard Vision Service"
echo "=========================================================================="

# ------------------------------------------------------------------------------
# Validate Model & Labels Artifacts
# ------------------------------------------------------------------------------
if [[ -z "${MODEL_PATH}" ]]; then
    # Auto-detect latest compiled *.vmfb by modification timestamp in models/compiled/ or root
    MODEL_PATH=$(find models/compiled models . -maxdepth 2 -name "torq_*.vmfb" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    MODEL_PATH="${MODEL_PATH#./}"
    if [[ -z "${MODEL_PATH}" ]]; then
        MODEL_PATH=$(find models/compiled models . -maxdepth 2 -name "*.vmfb" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        MODEL_PATH="${MODEL_PATH#./}"
    fi
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
    log_err "Compiled model (.vmfb) not found! Please compile with train.sh first or specify --model."
    exit 1
fi

if [[ -z "${LABELS_PATH}" ]]; then
    MODEL_STEM="$(basename "${MODEL_PATH%.*}")"
    CANDIDATE_NAME="${MODEL_STEM#torq_}"
    CANDIDATE_BASE="${CANDIDATE_NAME%_[0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]}"
    if [[ -f "models/labels/torq_${CANDIDATE_BASE}.json" ]]; then
        LABELS_PATH="models/labels/torq_${CANDIDATE_BASE}.json"
    elif [[ -f "models/labels/torq_${CANDIDATE_NAME}.json" ]]; then
        LABELS_PATH="models/labels/torq_${CANDIDATE_NAME}.json"
    elif [[ -f "models/labels/labels.json" ]]; then
        LABELS_PATH="models/labels/labels.json"
    else
        LATEST_LABELS=$(find models/labels -maxdepth 1 -name "*.json" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        if [[ -n "${LATEST_LABELS}" ]]; then
            LABELS_PATH="${LATEST_LABELS}"
        fi
    fi
fi

if [[ -z "${LABELS_PATH}" || ! -f "${LABELS_PATH}" ]]; then
    log_err "Matching labels file '${LABELS_PATH}' not found!"
    log_err "Coralboard proxy service requires 'labels_<model_name>.json' to decode detections and class names."
    exit 1
fi

MODEL_FILENAME="$(basename "${MODEL_PATH}")"
LABELS_FILENAME="$(basename "${LABELS_PATH}")"
log_info "Selected Model  : ${MODEL_PATH} ($(du -h "${MODEL_PATH}" | cut -f1))"
log_info "Selected Labels : ${LABELS_PATH} ($(grep -c '"' "${LABELS_PATH}" || echo "0") lines)"

# ------------------------------------------------------------------------------
# Verify at least one deployment target is specified
# ------------------------------------------------------------------------------
if [[ -z "${DEPLOY_HOST}" && -z "${DEPLOY_DIR}" ]]; then
    log_warn "No deployment target specified via flags or config.yaml!"
    echo ""
    echo "To deploy manually:"
    echo "  Remote: scp ${MODEL_PATH} root@<CORALBOARD_IP>:/path/to/coralboard-sl2619-vision/models/${MODEL_FILENAME}"
    echo "          scp ${LABELS_PATH} root@<CORALBOARD_IP>:/path/to/coralboard-sl2619-vision/models/${LABELS_FILENAME}"
    echo ""
    echo "Or set 'deployment.host' in config.yaml and re-run:"
    echo "  bash $0"
    exit 0
fi

# ------------------------------------------------------------------------------
# Local Directory Deployment
# ------------------------------------------------------------------------------
if [[ -n "${DEPLOY_DIR}" ]]; then
    log_info "Deploying to local directory: ${DEPLOY_DIR}..."
    mkdir -p "${DEPLOY_DIR}"
    cp -f "${MODEL_PATH}" "${DEPLOY_DIR}/${MODEL_FILENAME}"
    cp -f "${LABELS_PATH}" "${DEPLOY_DIR}/${LABELS_FILENAME}"
    log_success "Deployed ${MODEL_FILENAME} and ${LABELS_FILENAME} to ${DEPLOY_DIR}/"
fi

# ------------------------------------------------------------------------------
# Remote SCP Deployment
# ------------------------------------------------------------------------------
if [[ -n "${DEPLOY_HOST}" ]]; then
    CLEAN_HOST="${DEPLOY_HOST%/}"
    log_info "Deploying via SCP to remote destination: ${CLEAN_HOST}..."

    scp "${MODEL_PATH}" "${CLEAN_HOST}/${MODEL_FILENAME}"
    scp "${LABELS_PATH}" "${CLEAN_HOST}/${LABELS_FILENAME}"
    log_success "Successfully uploaded ${MODEL_FILENAME} and ${LABELS_FILENAME} to ${CLEAN_HOST}!"

    # Parse remote SSH host and port (e.g. root@192.168.100.2:/path -> root@192.168.100.2)
    REMOTE_SSH="${CLEAN_HOST%%:*}"
    REMOTE_IP=$(echo "${REMOTE_SSH}" | sed 's/.*@//')

    # Optional remote service restart
    if [[ "${RESTART_SERVICE}" == "true" ]]; then
        log_info "Restarting remote service '${SERVICE_NAME}' on ${REMOTE_SSH}..."
        if ssh -o ConnectTimeout=5 "${REMOTE_SSH}" "systemctl restart ${SERVICE_NAME}"; then
            log_success "Service '${SERVICE_NAME}' restarted successfully!"
        else
            log_warn "Could not restart '${SERVICE_NAME}' automatically via SSH. Please restart manually on the Coralboard."
        fi
    fi

    # Optional health check
    if [[ "${DO_HEALTH_CHECK}" == "true" ]]; then
        log_info "Performing health check at http://${REMOTE_IP}:5000/health..."
        sleep 2
        curl -s --connect-timeout 5 "http://${REMOTE_IP}:5000/health" || log_warn "Health check endpoint did not respond."
        echo ""
    fi
fi

echo "=========================================================================="
echo -e "${GREEN}  DEPLOYMENT COMPLETE!${NC}"
echo "=========================================================================="
