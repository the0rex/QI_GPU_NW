# ============================================================================
# Multi-stage Dockerfile for Alignment Pipeline with CUDA/GPU support
# ============================================================================

# Stage 1: Base with CUDA (for GPU builds) OR CPU-only
ARG USE_CUDA=false
ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.9

# Conditional base image selection
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 as cuda-builder
FROM python:${PYTHON_VERSION}-slim as cpu-builder
FROM ${USE_CUDA:+nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04} \
     ${USE_CUDA:?}${USE_CUDA:-python:${PYTHON_VERSION}-slim} as builder

# Set build arguments
ARG USE_CUDA
ARG TORCH_CUDA=cu121
ARG PYTORCH_VERSION=2.1.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libomp-dev \
    && if [ "$USE_CUDA" = "true" ]; then \
        apt-get install -y \
        cuda-toolkit-12-1 \
        cuda-cudart-dev-12-1 \
        libcublas-dev-12-1 \
        libcufft-dev-12-1 \
        libcurand-dev-12-1 \
        libcusolver-dev-12-1 \
        libcusparse-dev-12-1 \
        libcudnn8-dev; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel setuptools

# Copy requirements files
COPY requirements.txt .
COPY requirements-gpu.txt .

# Install Python dependencies conditionally
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing GPU-accelerated dependencies..." \
        && pip install --no-cache-dir \
            torch==${PYTORCH_VERSION}+${TORCH_CUDA} \
            torchvision==0.16.0+${TORCH_CUDA} \
            torchaudio==2.1.0+${TORCH_CUDA} \
            --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        && pip install --no-cache-dir -r requirements-gpu.txt; \
    else \
        echo "Installing CPU-only dependencies..." \
        && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r requirements.txt; \
    fi \
    && pip install --no-cache-dir \
        pybind11>=2.10.0 \
        ninja \
        scikit-build

# Copy source code
COPY . /app
WORKDIR /app

# Install the package with optional CUDA extensions
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Building with CUDA support..." \
        && pip install --no-cache-dir -e .[cuda]; \
    else \
        echo "Building CPU-only version..." \
        && pip install --no-cache-dir -e .; \
    fi

# Run tests to verify installation
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Running GPU tests..." \
        && python -m pytest tests/test_cuda.py -v || echo "GPU tests may require actual GPU hardware"; \
    else \
        echo "Running CPU tests..." \
        && python -m pytest tests/test_alignment.py tests/test_chunking.py tests/test_compression.py -v; \
    fi

# ============================================================================
# Stage 2: Runtime
# ============================================================================

# Conditional runtime base image
FROM ${USE_CUDA:+nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04} \
     ${USE_CUDA:?}${USE_CUDA:-python:${PYTHON_VERSION}-slim} as runtime

# Set labels
LABEL maintainer="Rowel Facunla <rowel.facunla@tip.edu.ph>"
LABEL org.opencontainers.image.description="Sequence Alignment Pipeline with optional GPU acceleration"
LABEL org.opencontainers.image.version="2.0.0"
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && if [ "$USE_CUDA" = "true" ]; then \
        apt-get install -y \
        libcublas-12-1 \
        libcufft-12-1 \
        libcurand-12-1 \
        libcusolver-12-1 \
        libcusparse-12-1 \
        libcudnn8; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 pipeline && \
    mkdir -p /home/pipeline/.cache/torch /home/pipeline/.cache/huggingface && \
    chown -R pipeline:pipeline /home/pipeline

USER pipeline

# Create working directory
WORKDIR /home/pipeline

# Copy application
COPY --from=builder --chown=pipeline:pipeline /app /home/pipeline/app

# Set environment variables
ENV PYTHONPATH=/home/pipeline/app/src
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Set CUDA-specific environment variables if using GPU
ARG USE_CUDA
ENV USE_CUDA=${USE_CUDA:-false}
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

# Create mount points
RUN mkdir -p /home/pipeline/data /home/pipeline/results /home/pipeline/config

# Copy default configuration
COPY --chown=pipeline:pipeline config/pipeline_config.yaml /home/pipeline/config/
COPY --chown=pipeline:pipeline config/pipeline_config_gpu.yaml /home/pipeline/config/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import alignment_pipeline; print('OK')" || exit 1

# Entrypoint script
COPY --chown=pipeline:pipeline docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command
CMD ["--help"]