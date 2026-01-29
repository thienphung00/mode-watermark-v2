"""
GPU Worker FastAPI application.

Standalone service for running GPU-intensive inference operations.

Run with:
    uvicorn service.worker.main:app --host 0.0.0.0 --port 8080

Environment Variables:
    MODEL_ID: Hugging Face model ID (default: runwayml/stable-diffusion-v1-5)
    DEVICE: Device to use (cuda, mps, cpu, auto) (default: auto)
    USE_FP16: Use half precision (default: true)
    LIKELIHOOD_PARAMS_PATH: Path to likelihood parameters JSON
    MASK_PATH: Path to detection mask
    MAX_CONCURRENT_REQUESTS: Max concurrent requests (default: 4)
    MAX_QUEUE_SIZE: Max queue size (default: 16)
    GPU_SEMAPHORE_SIZE: Max parallel GPU operations (default: 2)
    ENABLE_WARMUP: Enable model warmup (default: true)
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from service.infra.logging import configure_logging, get_logger, get_request_id
from service.worker.model_loader import ModelLoader, WorkerSettings
from service.worker.routes import router

# Configure structured logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Worker lifespan management.
    
    Startup:
    1. Load settings from environment
    2. Initialize model loader
    3. Load models (pipeline, artifacts)
    4. Warmup models
    
    Shutdown:
    1. Cleanup model resources
    2. Clear GPU cache
    """
    logger.info("worker_starting")
    
    # Load settings
    settings = WorkerSettings()
    logger.info(
        "worker_settings_loaded",
        extra={
            "model_id": settings.model_id,
            "device": settings.device,
            "use_fp16": settings.use_fp16,
            "max_concurrent_requests": settings.max_concurrent_requests,
        }
    )
    
    # Initialize model loader
    model_loader = ModelLoader(settings)
    app.state.model_loader = model_loader
    
    # Load models
    try:
        await model_loader.load_models()
        logger.info("worker_ready")
    except Exception as e:
        logger.error(
            "worker_startup_failed",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
    
    yield
    
    # Shutdown
    logger.info("worker_stopping")
    await model_loader.cleanup()
    logger.info("worker_stopped")


# Create FastAPI app
app = FastAPI(
    title="GPU Worker Service",
    description="GPU inference worker for watermark detection and generation",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exception."""
    request_id = get_request_id()
    
    logger.error(
        "unhandled_exception",
        extra={
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
        exc_info=True,
    )
    
    response_data = {
        "error": "internal_error",
        "message": "An unexpected error occurred",
    }
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=500,
        content=response_data,
    )


# Include router
app.include_router(router)


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "GPU Worker",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/v1/detect",
            "generate": "/v1/generate",
            "health": "/v1/health",
            "ready": "/v1/ready",
        }
    }

