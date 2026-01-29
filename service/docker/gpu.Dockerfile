# GPU Worker Dockerfile
# Requires NVIDIA GPU runtime

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.11
# (Python 3.11+ required for StrEnum support in Pydantic v2)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Install Python dependencies
# Install cffi first (required by cryptography, huggingface stack, etc.)
RUN pip3 install --no-cache-dir cffi

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip3 install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    pydantic>=2.0.0 \
    Pillow>=9.0.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    httpx>=0.24.0 \
    diffusers>=0.21.0 \
    transformers>=4.30.0 \
    accelerate>=0.20.0

# Copy source code
COPY src/ /app/src/
COPY service/gpu/ /app/service/gpu/

# Set Python path
ENV PYTHONPATH=/app

# Default environment
ENV GPU_HOST=0.0.0.0 \
    GPU_PORT=8001 \
    MODEL_ID=runwayml/stable-diffusion-v1-5 \
    DEVICE=cuda \
    STUB_MODE=false

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1

# Run the GPU worker
CMD ["python3", "-m", "service.gpu.main"]
