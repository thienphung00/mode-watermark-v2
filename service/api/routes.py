"""
API routes for watermarking service.

Endpoints:
- POST /keys/register - Register new watermark key
- POST /generate - Generate watermarked image
- POST /detect - Detect watermark in image
- GET /health - Health check
- GET /demo - Demo UI
"""
from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse

from service.api.schemas import (
    KeyRegisterRequest,
    KeyRegisterResponse,
    KeyListResponse,
    KeyInfo,
    GenerateRequest,
    GenerateResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ErrorResponse,
)
from service.api.key_store import get_key_store
from service.api.authority import get_authority
from service.api.detector import get_detector, get_stub_detector
from service.api.gpu_client import get_gpu_client, GPUClientError, GPUClientConnectionError
from service.api.storage import get_storage
from service.api.generation_store import get_generation_store

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Key Management
# =============================================================================


@router.post(
    "/keys/register",
    response_model=KeyRegisterResponse,
    tags=["keys"],
    summary="Register a new watermark key",
)
async def register_key(request: KeyRegisterRequest) -> KeyRegisterResponse:
    """
    Register a new watermark key.
    
    This creates a new unique key_id that can be used for:
    - Generating watermarked images
    - Detecting watermarks in images
    
    The key is persisted and can be used across sessions.
    """
    key_store = get_key_store()
    
    result = key_store.register_key(metadata=request.metadata)
    
    logger.info(f"Registered new key: {result['key_id']}")
    
    return KeyRegisterResponse(
        key_id=result["key_id"],
        fingerprint=result["fingerprint"],
        created_at=result["created_at"],
    )


@router.get(
    "/keys",
    response_model=KeyListResponse,
    tags=["keys"],
    summary="List all registered keys",
)
async def list_keys() -> KeyListResponse:
    """List all registered watermark keys."""
    key_store = get_key_store()
    keys = key_store.list_keys()
    
    return KeyListResponse(
        keys=[KeyInfo(**k) for k in keys],
        total=len(keys),
    )


# =============================================================================
# Image Generation
# =============================================================================


@router.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["generation"],
    summary="Generate a watermarked image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid key"},
        503: {"model": ErrorResponse, "description": "GPU worker unavailable"},
    },
)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a watermarked image using the specified key.
    
    The watermark is embedded during the diffusion process and
    is invisible to human observers but detectable by the API.
    
    Note: If GPU worker is unavailable, returns a stub response.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Validate key
    authority = get_authority()
    if not authority.validate_key(request.key_id):
        raise HTTPException(status_code=400, detail=f"Invalid or inactive key: {request.key_id}")
    
    # Get generation payload (with derived key)
    try:
        payload = authority.get_generation_payload(
            key_id=request.key_id,
            request_id=request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Try GPU worker, fall back to stub
    gpu_client = get_gpu_client()
    storage = get_storage()
    
    try:
        # Call GPU worker
        response = await gpu_client.generate(
            key_id=request.key_id,
            derived_key=payload["derived_key"],
            key_fingerprint=payload["key_fingerprint"],
            prompt=request.prompt,
            request_id=request_id,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            embedding_config=payload["embedding_config"],
        )
        
        # Decode and store image
        image_data = base64.b64decode(response.image_base64)
        filename = await storage.save_image(image_data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Construct browser-accessible URL
        image_url = f"/images/{filename}"
        
        # Persist generation record (non-blocking, failure-tolerant)
        try:
            generation_store = get_generation_store()
            generation_store.record_generation(
                key_id=request.key_id,
                filename=filename,
                seed_used=response.seed_used,
                processing_time_ms=processing_time_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to persist generation record: {e}")
        
        return GenerateResponse(
            image_url=image_url,
            key_id=request.key_id,
            seed_used=response.seed_used,
            processing_time_ms=processing_time_ms,
        )
        
    except GPUClientConnectionError:
        # GPU worker not available - return stub response
        logger.warning("GPU worker unavailable, returning stub response")
        
        import random
        seed_used = request.seed if request.seed else random.randint(0, 2**31 - 1)
        
        # Create a stub image path
        stub_path = f"stub_image_{request_id}.png"
        processing_time_ms = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            image_url=f"[STUB] {stub_path}",
            key_id=request.key_id,
            seed_used=seed_used,
            processing_time_ms=processing_time_ms,
        )
        
    except GPUClientError as e:
        logger.error(f"GPU generation failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# =============================================================================
# Detection
# =============================================================================


@router.post(
    "/detect",
    response_model=DetectResponse,
    tags=["detection"],
    summary="Detect watermark in an image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid key or missing image"},
        503: {"model": ErrorResponse, "description": "GPU worker unavailable"},
    },
)
async def detect_watermark(
    key_id: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
) -> DetectResponse:
    """
    Detect watermark in an uploaded image.
    
    The image can be provided either as:
    - A file upload (multipart form)
    - Base64-encoded data
    
    Returns detection result with confidence score.
    
    Note: If GPU worker is unavailable, returns a stub response.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Validate key
    authority = get_authority()
    if not authority.validate_key(key_id):
        raise HTTPException(status_code=400, detail=f"Invalid or inactive key: {key_id}")
    
    # Get image data
    if image is not None:
        image_bytes = await image.read()
        image_b64 = base64.b64encode(image_bytes).decode()
    elif image_base64 is not None:
        image_b64 = image_base64
    else:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Get detection payload
    try:
        payload = authority.get_detection_payload(
            key_id=key_id,
            request_id=request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Try GPU worker, fall back to stub
    gpu_client = get_gpu_client()
    
    try:
        # Call GPU worker
        response = await gpu_client.detect(
            key_id=key_id,
            derived_key=payload["derived_key"],
            master_key=payload["master_key"],
            key_fingerprint=payload["key_fingerprint"],
            image_base64=image_b64,
            request_id=request_id,
            g_field_config=payload["g_field_config"],
            detection_config=payload["detection_config"],
            inversion_config=payload["inversion_config"],
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return DetectResponse(
            detected=response.detected,
            confidence=response.confidence,
            key_id=key_id,
            score=response.score,
            threshold=0.5,  # Default threshold
            processing_time_ms=processing_time_ms,
            posterior=response.posterior,
            log_odds=response.log_odds,
        )
        
    except GPUClientConnectionError:
        # GPU worker not available - use stub detector
        logger.warning("GPU worker unavailable, using stub detector")
        
        stub_detector = get_stub_detector()
        result = stub_detector.detect_stub(key_id, simulate_watermarked=True)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return DetectResponse(
            detected=result.detected,
            confidence=result.confidence,
            key_id=key_id,
            score=result.score,
            threshold=result.threshold,
            processing_time_ms=processing_time_ms,
            posterior=result.posterior,
            log_odds=result.log_odds,
        )
        
    except GPUClientError as e:
        logger.error(f"GPU detection failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# =============================================================================
# Health
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """
    Check service health.
    
    Returns:
    - Service status
    - GPU worker connectivity
    - Number of registered keys
    """
    key_store = get_key_store()
    gpu_client = get_gpu_client()
    
    gpu_connected = await gpu_client.is_connected()
    
    status = "healthy"
    if not gpu_connected:
        status = "degraded"  # Can still work with stubs
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        gpu_worker_connected=gpu_connected,
        keys_loaded=key_store.count(),
    )


# =============================================================================
# Demo UI
# =============================================================================


@router.get(
    "/demo",
    response_class=HTMLResponse,
    tags=["demo"],
    summary="Demo UI",
)
async def demo_ui():
    """Serve the demo UI."""
    from pathlib import Path
    
    demo_path = Path(__file__).parent / "static" / "demo.html"
    
    if demo_path.exists():
        with open(demo_path) as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>Watermark Demo</title></head>
        <body>
            <h1>Watermark API Demo</h1>
            <p>Demo UI not found. See API docs at <a href="/docs">/docs</a>.</p>
        </body>
        </html>
        """)
