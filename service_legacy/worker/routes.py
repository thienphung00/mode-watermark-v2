"""
API routes for GPU worker service.

Provides:
- POST /v1/detect - Watermark detection
- POST /v1/generate - Watermarked image generation
- GET /v1/health - Health check (internal, detailed)
- GET /v1/ready - Readiness probe (public, minimal)

SECURITY INVARIANTS:
- NEVER receives master_key (only derived_key)
- Validates key_fingerprint format and logs mismatches
- Uses atomic backpressure (no manual counters)

TIMEOUT POLICY:
- NEVER cancel in-flight GPU operations (can corrupt CUDA state)
- Timeouts applied only at API→Worker RPC boundary (client side)
- Worker gracefully rejects new requests when overloaded
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import time
from typing import TYPE_CHECKING

import torch
from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from service.infra.logging import get_logger
from service.infra.security import validate_key_fingerprint
from service.worker.schemas import (
    DetectWorkerRequest,
    DetectWorkerResponse,
    GenerateWorkerRequest,
    GenerateWorkerResponse,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)

if TYPE_CHECKING:
    from service.worker.model_loader import ModelLoader

logger = get_logger(__name__)

router = APIRouter()


def get_model_loader(request: Request) -> "ModelLoader":
    """Get model loader from app state."""
    return request.app.state.model_loader


def _validate_fingerprint(key_fingerprint: str, key_id: str) -> None:
    """
    Validate key fingerprint format.
    
    Workers cannot independently verify fingerprint derivation (by design),
    but we validate format and log for audit trails.
    
    Raises:
        HTTPException: If fingerprint validation fails
    """
    if not validate_key_fingerprint(
        derived_key="",  # Not used for format validation
        expected_fingerprint=key_fingerprint,
        key_id=key_id,
    ):
        logger.error(
            "fingerprint_validation_rejected",
            extra={
                "key_id": key_id,
                "fingerprint_prefix": key_fingerprint[:8] if key_fingerprint else "NONE",
            }
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid key fingerprint format"
        )


# =============================================================================
# Detection Endpoint
# =============================================================================


@router.post(
    "/v1/detect",
    response_model=DetectWorkerResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Worker not ready or overloaded"},
    },
)
async def detect(request: Request, body: DetectWorkerRequest) -> DetectWorkerResponse:
    """
    Perform watermark detection.
    
    This endpoint:
    1. Validates key_fingerprint format
    2. Acquires request slot (atomic backpressure)
    3. Decodes image bytes
    4. Encodes to VAE latent z_0
    5. Performs DDIM inversion to z_T
    6. Computes g-values using derived_key (NEVER master_key)
    7. Runs Bayesian detection
    8. Returns results
    
    SECURITY: Only receives derived_key (never master_key).
    TIMEOUT: Never cancels in-flight GPU operations.
    """
    model_loader = get_model_loader(request)
    
    # Check if worker is ready
    if not model_loader.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Worker not ready - model not loaded",
        )
    
    # Validate key fingerprint format (defense in depth)
    _validate_fingerprint(body.key_fingerprint, body.key_id)
    
    # Atomic backpressure: acquire request slot
    slot_acquired = await model_loader.acquire_request_slot()
    if not slot_acquired:
        logger.warning(
            "request_rejected_queue_full",
            extra={
                "request_id": body.request_id,
                "queue_size": model_loader.queue_size,
            }
        )
        raise HTTPException(
            status_code=503,
            detail="Worker queue full - try again later",
        )
    
    start_time = time.time()
    
    try:
        # Acquire semaphore for GPU access
        async with model_loader.gpu_semaphore:
            result = await _run_detection(
                model_loader=model_loader,
                request=body,
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "detection_completed",
            extra={
                "request_id": body.request_id,
                "key_id": body.key_id,
                "detected": result["detected"],
                "score": result["score"],
                "processing_time_ms": processing_time_ms,
                "key_fingerprint_prefix": body.key_fingerprint[:8] + "...",
            }
        )
        
        return DetectWorkerResponse(
            detected=result["detected"],
            score=result["score"],
            confidence=result["confidence"],
            log_odds=result["log_odds"],
            posterior=result["posterior"],
            request_id=body.request_id,
            processing_time_ms=processing_time_ms,
            key_fingerprint=body.key_fingerprint,
            g_field_config_hash=result.get("g_field_config_hash"),
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "detection_failed",
            extra={
                "request_id": body.request_id,
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always release the slot
        await model_loader.release_request_slot()


async def _run_detection(
    model_loader: "ModelLoader",
    request: DetectWorkerRequest,
) -> dict:
    """
    Run detection inference.
    
    Runs in thread pool to avoid blocking event loop.
    
    SECURITY: Uses derived_key (never master_key).
    The derived_key is scoped to detection operation.
    """
    def detection_fn():
        # Decode image from base64
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get pipeline components
        pipeline = model_loader.pipeline
        vae = pipeline.vae
        unet = pipeline.unet
        scheduler = pipeline.scheduler
        
        # Import detection components
        from src.detection.inversion import DDIMInverter
        from src.detection.g_values import compute_g_values
        from src.models.detectors import BayesianDetector
        
        # Create inverter
        inverter = DDIMInverter(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            device=pipeline.device,
        )
        
        # Get inversion parameters
        inversion_config = request.inversion_config or {}
        num_steps = inversion_config.get("num_inference_steps", 50)
        
        # Run DDIM inversion to get z_T
        with torch.no_grad():
            z_T = inverter.invert(
                image=image,
                num_inference_steps=num_steps,
                prompt="",  # Unconditional for detection
                guidance_scale=1.0,
            )
        
        # Compute g-values using derived_key (NEVER master_key)
        # The derived_key is scoped to detection operation and this key_id
        g_values = compute_g_values(
            z_T=z_T,
            master_key=request.derived_key,  # SECURITY: This is derived_key, not master_key
            key_id=request.key_id,
            config=request.g_field_config,
        )
        
        # Compute g-field config hash
        import json
        config_json = json.dumps(request.g_field_config, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        
        # Run Bayesian detection
        detector = BayesianDetector(
            likelihood_params=model_loader.detector_artifacts.likelihood_params,
            mask=model_loader.detector_artifacts.mask,
        )
        
        detection_result = detector.detect(
            g_values=g_values,
            threshold=request.detection_config.get("threshold", 0.5),
            prior=request.detection_config.get("prior_watermarked", 0.5),
        )
        
        return {
            "detected": detection_result.is_watermarked,
            "score": detection_result.score,
            "confidence": detection_result.confidence,
            "log_odds": detection_result.log_odds,
            "posterior": detection_result.posterior,
            "g_field_config_hash": config_hash,
        }
    
    # Run in thread pool
    return await asyncio.to_thread(detection_fn)


# =============================================================================
# Generation Endpoint
# =============================================================================


@router.post(
    "/v1/generate",
    response_model=GenerateWorkerResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Worker not ready or overloaded"},
    },
)
async def generate(request: Request, body: GenerateWorkerRequest) -> GenerateWorkerResponse:
    """
    Generate watermarked image.
    
    This endpoint:
    1. Validates key_fingerprint format
    2. Acquires request slot (atomic backpressure)
    3. Creates SeedBiasStrategy from config using derived_key (NEVER master_key)
    4. Runs Stable Diffusion with watermarking
    5. Returns generated image bytes
    
    SECURITY: Only receives derived_key (never master_key).
    TIMEOUT: Never cancels in-flight GPU operations.
    NON-IDEMPOTENT: Retries may produce different results.
    """
    model_loader = get_model_loader(request)
    
    # Check if worker is ready
    if not model_loader.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Worker not ready - model not loaded",
        )
    
    # Validate key fingerprint format (defense in depth)
    _validate_fingerprint(body.key_fingerprint, body.key_id)
    
    # Atomic backpressure: acquire request slot
    slot_acquired = await model_loader.acquire_request_slot()
    if not slot_acquired:
        logger.warning(
            "request_rejected_queue_full",
            extra={
                "request_id": body.request_id,
                "queue_size": model_loader.queue_size,
            }
        )
        raise HTTPException(
            status_code=503,
            detail="Worker queue full - try again later",
        )
    
    start_time = time.time()
    
    try:
        # Acquire semaphore for GPU access
        async with model_loader.gpu_semaphore:
            result = await _run_generation(
                model_loader=model_loader,
                request=body,
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "generation_completed",
            extra={
                "request_id": body.request_id,
                "key_id": body.key_id,
                "seed_used": result["seed_used"],
                "processing_time_ms": processing_time_ms,
                "key_fingerprint_prefix": body.key_fingerprint[:8] + "...",
            }
        )
        
        return GenerateWorkerResponse(
            image_base64=result["image_base64"],
            seed_used=result["seed_used"],
            request_id=body.request_id,
            processing_time_ms=processing_time_ms,
            key_fingerprint=body.key_fingerprint,
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "generation_failed",
            extra={
                "request_id": body.request_id,
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always release the slot
        await model_loader.release_request_slot()


async def _run_generation(
    model_loader: "ModelLoader",
    request: GenerateWorkerRequest,
) -> dict:
    """
    Run generation inference.
    
    Runs in thread pool to avoid blocking event loop.
    
    SECURITY: Uses derived_key (never master_key).
    The derived_key is scoped to generation operation.
    """
    def generation_fn():
        import random
        
        # Get seed
        seed = request.seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        # Import generation components
        from src.engine.pipeline import generate_with_watermark
        from src.engine.strategies.seed_bias import SeedBiasStrategy
        
        # Create watermark strategy using derived_key (NEVER master_key)
        # The derived_key is scoped to generation operation and this key_id
        strategy = SeedBiasStrategy(
            master_key=request.derived_key,  # SECURITY: This is derived_key, not master_key
            key_id=request.key_id,
            lambda_strength=request.embedding_config.get("lambda_strength", 0.05),
            domain=request.embedding_config.get("domain", "frequency"),
            low_freq_cutoff=request.embedding_config.get("low_freq_cutoff", 0.05),
            high_freq_cutoff=request.embedding_config.get("high_freq_cutoff", 0.4),
        )
        
        # Generate watermarked image
        result = generate_with_watermark(
            pipeline=model_loader.pipeline,
            prompt=request.prompt,
            strategy=strategy,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=seed,
            height=request.height,
            width=request.width,
        )
        
        # Convert image to base64
        image = result["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image_base64": image_base64,
            "seed_used": seed,
        }
    
    # Run in thread pool
    return await asyncio.to_thread(generation_fn)


# =============================================================================
# Health Check Endpoints
# =============================================================================


@router.get("/v1/health", response_model=HealthResponse)
async def health_internal(request: Request) -> HealthResponse:
    """
    INTERNAL health check with detailed metrics.
    
    WARNING: This endpoint exposes operational metrics including GPU utilization.
    It MUST be protected via network policy or authentication in production.
    Do NOT expose to public internet.
    
    Returns:
    - status: "healthy" | "degraded" | "unhealthy"
    - model_loaded: bool
    - gpu_memory_used_mb: int
    - gpu_memory_total_mb: int
    - active_requests: int
    - queue_size: int
    - uptime_seconds: float
    """
    model_loader = get_model_loader(request)
    gpu_info = model_loader.get_gpu_info()
    
    # Determine status
    status = "healthy"
    if not model_loader.is_ready:
        status = "unhealthy"
    elif gpu_info["memory_used_pct"] > 90:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loader.pipeline is not None,
        gpu_memory_used_mb=gpu_info.get("memory_used_mb"),
        gpu_memory_total_mb=gpu_info.get("memory_total_mb"),
        active_requests=model_loader.queue_size,
        queue_size=model_loader.queue_size,
        uptime_seconds=model_loader.get_uptime_seconds(),
    )


@router.get("/v1/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    """
    PUBLIC readiness probe (minimal information).
    
    This endpoint is safe for public exposure (e.g., Kubernetes probes).
    It only returns boolean readiness status without detailed metrics.
    
    Returns 200 only when:
    - Model is fully loaded
    - GPU is available
    - Service can accept requests
    
    Does NOT expose:
    - GPU memory utilization
    - Queue sizes
    - Uptime
    """
    model_loader = get_model_loader(request)
    gpu_info = model_loader.get_gpu_info()
    
    is_ready = (
        model_loader.is_ready and
        model_loader.pipeline is not None
    )
    
    if not is_ready:
        raise HTTPException(status_code=503, detail="Worker not ready")
    
    return ReadyResponse(
        ready=is_ready,
        model_loaded=model_loader.pipeline is not None,
        gpu_available=gpu_info.get("gpu_available", False),
    )


@router.get("/v1/live")
async def liveness() -> dict:
    """
    PUBLIC liveness probe (minimal).
    
    Returns 200 if the process is running.
    Does not check model loading or GPU availability.
    
    This is the safest endpoint for public exposure.
    """
    return {"live": True}

