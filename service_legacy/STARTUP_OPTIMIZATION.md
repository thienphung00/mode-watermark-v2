# Startup Optimization

## Issue
The service was taking a long time to start because it was pre-loading the Stable Diffusion pipeline at startup, which triggers a ~4GB model download.

## Solution
Pipeline loading is now **lazy-loaded** on first request instead of at startup.

### Changes Made
1. **Removed pre-loading at startup** (`service/app/main.py`)
   - Pipeline is now loaded only when first `/generate` or `/detect` request is made
   - Server starts immediately without waiting for model download

2. **Updated Dockerfile** (`service/Dockerfile`)
   - Added HuggingFace cache environment variables
   - Added commented instructions for pre-downloading models

## Current Behavior
- ✅ Server starts immediately (< 1 second)
- ✅ Health endpoint (`/api/v1/health`) works immediately
- ⏳ First `/generate` or `/detect` request will download models (~2-5 minutes)
- ✅ Subsequent requests are fast (models cached)

## Optional: Pre-download Models in Dockerfile

To avoid the first-request delay, uncomment this line in `Dockerfile`:

```dockerfile
RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

This will:
- Download models during Docker build (one-time cost)
- Make first request fast
- Increase Docker image size by ~4GB

## Using Volume Mounts for Model Cache

To persist models across container restarts:

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/.cache/huggingface:/app/.cache/huggingface \
  watermarking-service
```

This mounts the HuggingFace cache directory, so models are downloaded once and reused.

## Performance Comparison

### Before (Pre-loading at startup)
- Startup time: ~5-10 minutes (downloads models)
- First request: Fast
- Subsequent requests: Fast

### After (Lazy loading)
- Startup time: < 1 second ✅
- First request: ~2-5 minutes (downloads models)
- Subsequent requests: Fast ✅

