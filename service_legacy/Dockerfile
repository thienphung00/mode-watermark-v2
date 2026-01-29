FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and uvicorn
RUN pip install --no-cache-dir fastapi uvicorn[standard] cryptography python-multipart
RUN pip install --no-cache-dir google-cloud-pubsub

# Set HuggingFace cache directory (can be overridden via volume mount)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Copy source code
COPY src/ ./src/
COPY service/ ./service/

# Note: Detection artifacts (outputs/likelihood_models_exp/likelihood_params.json) are NOT copied by default.
# To enable detection in Docker, either:
# 1. Mount outputs as volume: docker run -v $(pwd)/outputs:/app/outputs ...
# 2. Set env var: docker run -e LIKELIHOOD_PARAMS_PATH=/path/to/likelihood_params.json ...
# 3. Uncomment the line below to copy outputs into the image (only if outputs/ exists in build context)
# COPY outputs/ ./outputs/

# Create data directory
RUN mkdir -p service_data

# NOTE: Models are lazy-loaded on first request to avoid blocking startup.
# To pre-download models, uncomment the following lines:
# RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"

# Expose port
EXPOSE 8000

# Run service
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

