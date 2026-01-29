"""
Route for imperceptibility evaluation.

⚠️ EVALUATION-ONLY ENDPOINT ⚠️

POST /evaluate/imperceptibility

This endpoint allows users to compare watermarked vs baseline images under
controlled, deterministic conditions, without weakening the production watermark
authority model.

SECURITY NOTES:
- This endpoint is not part of watermark security guarantees
- Outputs must not be used for detection benchmarking
- Results do not imply attack feasibility
- This endpoint exists solely for imperceptibility evaluation
- Uses a fixed, hardcoded evaluation master key (not stored, not issued, not reusable)
- Evaluation images cannot be detected or validated with production detection
- This endpoint is isolated from production generation & detection paths

This endpoint does NOT:
- Use WatermarkAuthorityService
- Issue or accept key_id
- Register watermarks
- Produce images intended for detection
"""
from __future__ import annotations

import base64
import io
from typing import Optional

from fastapi import APIRouter, HTTPException

from service.app.schemas import (
    ImperceptibilityEvalRequest,
    ImperceptibilityEvalResponse,
    DifferenceMetrics,
    ModelInfo,
)
from service.evaluation import ImperceptibilityEvaluationService
from service.infra.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global evaluation service instance (lazy-loaded)
_evaluation_service: Optional[ImperceptibilityEvaluationService] = None


def get_evaluation_service() -> ImperceptibilityEvaluationService:
    """
    Get evaluation service singleton.
    
    Returns:
        ImperceptibilityEvaluationService instance
    """
    global _evaluation_service
    
    if _evaluation_service is None:
        _evaluation_service = ImperceptibilityEvaluationService(
            model_id="runwayml/stable-diffusion-v1-5",
            device=None,  # Auto-detect
            use_fp16=True,
        )
        logger.info(
            "evaluation_service_initialized",
            extra={"model_id": "runwayml/stable-diffusion-v1-5"}
        )
    
    return _evaluation_service


@router.post("/evaluate/imperceptibility", response_model=ImperceptibilityEvalResponse)
async def evaluate_imperceptibility(
    request: ImperceptibilityEvalRequest,
) -> ImperceptibilityEvalResponse:
    """
    Evaluate imperceptibility by comparing baseline vs watermarked images.
    
    ⚠️ EVALUATION-ONLY ENDPOINT ⚠️
    
    This endpoint generates two images with identical parameters:
    1. Baseline image (no watermark)
    2. Watermarked image (seed-bias applied with fixed evaluation key)
    
    It then computes difference metrics (L2, PSNR, SSIM) between the two images.
    
    SECURITY NOTES:
    - This endpoint is NOT part of the production watermarking system
    - Outputs are NOT valid for production detection
    - Uses a fixed, hardcoded evaluation master key (not stored, not issued)
    - Evaluation images cannot be detected or validated
    - Results do NOT imply attack feasibility
    
    This endpoint does NOT:
    - Use WatermarkAuthorityService
    - Issue or accept key_id
    - Register watermarks
    - Produce images intended for detection
    
    Args:
        request: ImperceptibilityEvalRequest with prompt, seed, and generation parameters
    
    Returns:
        ImperceptibilityEvalResponse with both images (base64) and difference metrics
    
    Raises:
        HTTPException: If generation fails
    """
    try:
        # Get evaluation service
        evaluation_service = get_evaluation_service()
        
        logger.info(
            "evaluation_started",
            extra={
                "prompt_length": len(request.prompt),
                "seed": request.seed,
            }
        )
        
        # Generate baseline and watermarked images with metrics
        result = evaluation_service.evaluate_imperceptibility(
            prompt=request.prompt,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps or 50,
            guidance_scale=request.guidance_scale or 1.0,
            height=request.height or 512,
            width=request.width or 512,
        )
        
        # Encode images to base64
        baseline_image = result["baseline_image"]
        watermarked_image = result["watermarked_image"]
        
        def encode_image(image) -> str:
            """Encode PIL Image to base64 string."""
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        baseline_base64 = encode_image(baseline_image)
        watermarked_base64 = encode_image(watermarked_image)
        
        # Build response
        difference_metrics = DifferenceMetrics(**result["difference_metrics"])
        model_info = ModelInfo(**result["model_info"])
        
        logger.info(
            "evaluation_completed",
            extra={
                "l2": round(difference_metrics.l2, 6),
                "psnr_db": round(difference_metrics.psnr, 2),
                "ssim": round(difference_metrics.ssim, 4),
            }
        )
        
        return ImperceptibilityEvalResponse(
            baseline_image_base64=baseline_base64,
            watermarked_image_base64=watermarked_base64,
            difference_metrics=difference_metrics,
            model_info=model_info,
            disclaimer="Evaluation-only output. Not valid for production use or detection.",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "evaluation_failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate imperceptibility: {str(e)}"
        )

