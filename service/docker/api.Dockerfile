# API Service Dockerfile
# CPU-only, lightweight

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx Pillow numpy python-multipart


# Copy source code
COPY src/ /app/src/
COPY service/api/ /app/service/api/

# Create necessary directories
RUN mkdir -p /app/data/keys /app/data/images /app/data/artifacts

# Set Python path
ENV PYTHONPATH=/app

# Default environment
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    GPU_WORKER_URL=http://gpu:8001 \
    STORAGE_BACKEND=local \
    STORAGE_PATH=/app/data/images \
    KEY_STORE_PATH=/app/data/keys.json \
    ARTIFACTS_PATH=/app/data/artifacts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API service
CMD ["python", "-m", "uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
