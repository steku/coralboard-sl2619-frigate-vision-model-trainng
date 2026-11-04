# ==============================================================================
# Frigate Model Training & Compilation Studio - Unified Optimized Container
# ==============================================================================
# Multi-stage lean image containing:
#   - Python 3.12 (/app/venv-torq-312): YOLO training, Torq NPU toolchain, Web Studio
#   - Python 3.10 (/app/venv-hailo-310): Hailo-10H Dataflow Compiler & Model Zoo
#   - CUDA 12.6 runtime (supports NVIDIA GPU and CPU execution)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build virtual environments and compile toolchains
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    USER=root \
    USERNAME=root \
    LOGNAME=root \
    HOME=/root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    build-essential \
    graphviz \
    libgraphviz-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libusb-1.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev

WORKDIR /app

# 1. Build Python 3.12 primary environment (/app/venv-torq-312)
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m venv /app/venv-torq-312 && \
    /app/venv-torq-312/bin/pip install --upgrade pip "setuptools<70" wheel && \
    /app/venv-torq-312/bin/pip install -r /app/requirements.txt

# 2. Build Python 3.10 Hailo environment (/app/venv-hailo-310)
COPY packages /tmp/packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends /tmp/packages/hailort_5.4.0_amd64.deb

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m venv /app/venv-hailo-310 && \
    /app/venv-hailo-310/bin/pip install --upgrade pip "setuptools<70" wheel cython "numpy<2.0" && \
    /app/venv-hailo-310/bin/pip install --no-build-isolation lap==0.5.12 && \
    /app/venv-hailo-310/bin/pip install \
        /tmp/packages/hailort-5.4.0-cp310-cp310-linux_x86_64.whl \
        /tmp/packages/hailo_dataflow_compiler-5.4.0-py3-none-linux_x86_64.whl \
        /tmp/packages/hailo_model_zoo-5.4.0-py3-none-any.whl && \
    rm -rf /app/venv-hailo-310/lib/python3.10/site-packages/hailo_tutorials && \
    mkdir -p /app/venv-hailo-310/nvvm/libdevice && \
    LIBDEVICE=$(find /app/venv-hailo-310/lib -name "libdevice.10.bc" 2>/dev/null | head -n 1); \
    if [ -n "$LIBDEVICE" ]; then ln -sf "$LIBDEVICE" /app/venv-hailo-310/nvvm/libdevice/libdevice.10.bc; fi && \
    /app/venv-hailo-310/bin/python3 -c "import pyximport; pyximport.install(); from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import nms" || true && \
    rm -rf /tmp/packages

# 3. Copy application code and strip unneeded build/source artifacts
COPY . /app
RUN rm -rf /app/packages /app/cvat /app/.git \
    && find /app/venv-* -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /app/venv-* -type f -name "*.pyc" -delete 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 2: Minimal runtime image
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    USER=root \
    USERNAME=root \
    LOGNAME=root \
    HOME=/root

# Install runtime shared libraries and development build tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    rsync \
    openssh-client \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0t64 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libusb-1.0-0 \
    graphviz \
    python3-tk \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev

WORKDIR /app

# Install HailoRT package cleanly via apt
COPY packages/hailort_5.4.0_amd64.deb /tmp/hailort.deb
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends /tmp/hailort.deb && \
    rm -f /tmp/hailort.deb

# Copy prepared application and virtual environments from builder stage
COPY --from=builder /app /app

# Setup default symlink, state files, and working volume directories
RUN ln -s /app/venv-torq-312 /app/venv && \
    mkdir -p /app/staging /app/logs /app/models /app/datasets /app/.web_ui_logs /app/packages && \
    touch /app/.web_ui_history.json /app/.processed_events.json /app/.last_cvat_task

# Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=9001 \
    USER=root \
    USERNAME=root \
    LOGNAME=root \
    HOME=/root \
    PATH="/app/venv-torq-312/bin:${PATH}" \
    PYTHONPATH="/app"

EXPOSE 9001

CMD ["python3", "web_ui.py", "--host", "0.0.0.0", "--port", "9001"]
