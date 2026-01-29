"""
Route for watermark detection.

POST /detect

Uses DetectionService (Bayesian-only) for watermark detection.
No research internals exposed to API layer.
"""
from __future__ import annotations

import asyncio
import base64
import io

from fastapi import APIRouter, HTTPException
from PIL import Image

from service.app.schemas import DetectRequest, DetectResponse
from service.app.dependencies import get_detection_service, is_detection_available
from service.infra.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def decode_base64_image(image_b64: str) -> Image.Image:
    """
    Decode base64-encoded image.
    
    Args:
        image_b64: Base64-encoded image string
    
    Returns:
        PIL Image
    """
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


@router.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest) -> DetectResponse:
    """
    Detect watermark in image (prompt-agnostic).
    
    This endpoint:
    1. Uses DetectionService (Bayesian-only) for detection
    2. Never exposes detector class names or research internals
    3. Returns product-level results
    4. Does not require or trust prompts (uses unconditional DDIM inversion)
    
    Detection flow:
    - Image → VAE encode → z_0
    - z_0 → DDIM invert (unconditional: prompt="", guidance_scale=1.0) → z_T
    - z_T → compute g-values (src logic)
    - g-values → mask + binarize (src logic)
    - binarized g-values → BayesianDetector (likelihood_params.json)
    - Returns: posterior, log_odds, decision
    
    Args:
        request: DetectRequest with image_base64 and key_id
    
    Returns:
        DetectResponse with detection results
    """
    try:
        # Step 1: Check if detection artifacts are available
        if not is_detection_available():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Detection artifacts not configured",
                    "hint": "Set LIKELIHOOD_PARAMS_PATH environment variable",
                    "mode": "production",
                }
            )
        
        # Step 2: Get detection service (will raise if artifacts unavailable)
        try:
            detection_service = get_detection_service()
        except RuntimeError as e:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Detection service unavailable",
                    "hint": "Detection artifacts are not configured",
                    "mode": "production",
                    "details": str(e),
                }
            ) from e
        
        # Step 3: Decode image
        image = decode_base64_image(request.image_base64)
        
        logger.info("detection_started", extra={"key_id": request.key_id})
        
        # Step 4: Run detection (prompt-agnostic)
        # If micro-batching worker is enabled, use it for batching
        # Otherwise, use asyncio.to_thread for async execution
        if detection_service._detection_worker is not None:
            result = await detection_service._detection_worker.detect(image, request.key_id)
        else:
            # Fallback to thread pool execution
            result = await asyncio.to_thread(
                detection_service.detect,
                image,
                request.key_id,
            )
        
        logger.info(
            "detection_completed",
            extra={
                "key_id": request.key_id,
                "detected": result["detected"],
                "score": round(result["score"], 4),
                "confidence": round(result["confidence"], 4),
            }
        )
        
        return DetectResponse(
            detected=result["detected"],
            score=result["score"],
            confidence=result["confidence"],
            policy_version=result["policy_version"],
            posterior=result["posterior"],
            log_odds=result["log_odds"],
            is_watermarked=result["is_watermarked"],
            watermark_version=result["watermark_version"],
            g_field_config_hash=result.get("g_field_config_hash"),
        )
    
    except ValueError as e:
        error_msg = str(e)
        # Check if this is a parameter validation error (400) vs watermark not found (404)
        if "not found" in error_msg.lower() or "revoked" in error_msg.lower():
            # Watermark not found or revoked
            raise HTTPException(status_code=404, detail=error_msg)
        else:
            # Validation error (config mismatch, shape mismatch, etc.)
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "detection_failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to detect watermark: {str(e)}")

