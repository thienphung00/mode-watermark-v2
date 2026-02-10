"""
FastAPI entrypoint for GPU worker service.

This service handles heavy GPU computations:
- Image generation with watermark embedding
- DDIM inversion for detection
- G-value computation

It is NOT user-facing - only called by the API service.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from service.gpu.schemas import (
    GenerateRequest,
    GenerateResponse,
    ReverseDDIMRequest,
    ReverseDDIMResponse,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)
from service.gpu.pipeline import GPUPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global state
pipeline: GPUPipeline = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline, start_time
    
    start_time = time.time()
    
    # Configuration from environment
    model_id = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    device = os.getenv("DEVICE", "cuda")
    stub_mode = os.getenv("STUB_MODE", "true").lower() == "true"
    
    logger.info(f"Starting GPU worker (stub_mode={stub_mode})")
    logger.info(f"Model: {model_id}, Device: {device}")
    
    # Initialize pipeline
    pipeline = GPUPipeline(
        model_id=model_id,
        device=device,
        stub_mode=stub_mode,
    )
    
    # Load models
    pipeline.load_models()
    
    yield
    
    # Cleanup
    logger.info("Shutting down GPU worker")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="GPU Worker",
        description="""
        Internal GPU worker for watermark operations.
        
        NOT user-facing. Only accepts requests from API service.
        
        ## Endpoints
        - POST /infer/generate - Generate watermarked image
        - POST /infer/reverse_ddim - DDIM inversion for detection
        - GET /health - Health check (internal)
        """,
        version="1.0.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


# =============================================================================
# Generation Endpoint
# =============================================================================


@app.post(
    "/infer/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate watermarked image.
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    """
    global pipeline
    
    if not pipeline or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        result = pipeline.generate(
            prompt=request.prompt,
            master_key=request.master_key,
            key_id=request.key_id,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            embedding_config=request.embedding_config,
        )
        
        processing_time_ms = (time.time() - start) * 1000
        
        logger.info(
            f"Generation completed: key_id={request.key_id}, "
            f"seed={result.seed_used}, time={processing_time_ms:.1f}ms"
        )
        
        return GenerateResponse(
            image_base64=base64.b64encode(result.image_bytes).decode(),
            seed_used=result.seed_used,
            request_id=request.request_id,
            processing_time_ms=processing_time_ms,
            key_fingerprint=request.key_fingerprint,
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Detection Endpoint
# =============================================================================


@app.post(
    "/infer/reverse_ddim",
    response_model=ReverseDDIMResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def reverse_ddim(request: ReverseDDIMRequest) -> ReverseDDIMResponse:
    """
    Perform DDIM inversion and detect watermark.
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    compute_g_values() uses (master_key, key_id) directly.
    """
    global pipeline
    
    if not pipeline or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        
        # Run detection with master_key only
        result = pipeline.invert_and_detect(
            image_bytes=image_bytes,
            master_key=request.master_key,
            key_id=request.key_id,
            g_field_config=request.g_field_config,
            detection_config=request.detection_config,
            inversion_config=request.inversion_config,
        )
        
        processing_time_ms = (time.time() - start) * 1000
        
        # Compute config hash
        import json
        config_json = json.dumps(request.g_field_config, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        logger.info(
            f"Detection completed: key_id={request.key_id}, "
            f"detected={result.detected}, score={result.score:.3f}, "
            f"time={processing_time_ms:.1f}ms"
        )

        return ReverseDDIMResponse(
            detected=result.detected,
            score=result.score,
            threshold=result.threshold,
            confidence=result.confidence,
            log_odds=result.log_odds,
            posterior=result.posterior,
            request_id=request.request_id,
            processing_time_ms=processing_time_ms,
            key_fingerprint=request.key_fingerprint,
            g_field_config_hash=config_hash,
            latent_shape=list(result.latent_shape),
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Internal health check with detailed metrics.
    
    WARNING: This endpoint exposes operational metrics.
    Protect via network policy in production.
    """
    global pipeline, start_time
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
        )
    
    gpu_info = pipeline.get_gpu_info()
    uptime = time.time() - start_time
    
    status = "healthy"
    if not pipeline.is_loaded:
        status = "unhealthy"
    elif gpu_info.get("memory_used_pct", 0) > 90:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=pipeline.is_loaded,
        gpu_available=gpu_info.get("gpu_available", False),
        gpu_memory_used_mb=gpu_info.get("memory_used_mb"),
        gpu_memory_total_mb=gpu_info.get("memory_total_mb"),
        active_requests=0,  # Could track this with middleware
        uptime_seconds=uptime,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """
    Public readiness probe.
    
    Safe for Kubernetes probes - minimal information.
    """
    global pipeline
    
    is_ready = pipeline is not None and pipeline.is_loaded
    gpu_info = pipeline.get_gpu_info() if pipeline else {}
    
    if not is_ready:
        raise HTTPException(status_code=503, detail="Worker not ready")
    
    return ReadyResponse(
        ready=is_ready,
        model_loaded=pipeline.is_loaded if pipeline else False,
        gpu_available=gpu_info.get("gpu_available", False),
    )


@app.get("/live")
async def liveness() -> dict:
    """
    Liveness probe.
    
    Returns 200 if process is running.
    """
    return {"live": True}


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("GPU_HOST", "0.0.0.0")
    port = int(os.getenv("GPU_PORT", "8001"))
    
    uvicorn.run(
        "service.gpu.main:app",
        host=host,
        port=port,
        reload=False,  # Never reload GPU workers
    )
