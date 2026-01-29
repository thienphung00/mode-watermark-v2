"""
Shared dependencies for FastAPI routes.

Provides dependency injection for:
- WatermarkAuthorityService
- GenerationAdapter (Phase-1: Stable Diffusion only)
- DetectionService (Bayesian-only)
- InferenceClient (local or remote)

Architecture:
- AppState holds all service instances
- Services are initialized in lifespan context manager
- Dependencies use request.app.state for access
- No global mutable state

DEPENDENCY PURITY RULES:
Dependencies (functions called by FastAPI `Depends()`) must be pure:

✅ ALLOWED in dependencies:
- Object retrieval from app state
- Lightweight validation
- In-memory access

❌ FORBIDDEN in dependencies:
- Network calls
- Database queries
- Secret fetching
- Health checks
- Locks (except for lightweight state access)
- Retries

Heavy initialization MUST be done in:
- Application startup lifecycle (lifespan)
- Background tasks
- Explicit service calls

No research-layer instantiation here.
No SD-specific logic here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any

import torch
from fastapi import Request, Depends

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from service.authority import WatermarkAuthorityService
from service.app.artifact_resolver import get_artifact_resolver
from service.infra.logging import get_logger

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from service.generation import GenerationAdapter, StableDiffusionSeedBiasAdapter
    from service.detection import DetectionService
    from service.inference.client import InferenceClient

logger = get_logger(__name__)


def detect_device() -> str:
    """
    Detect available device with priority: mps > cuda > cpu.
    
    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@dataclass
class AppState:
    """
    Application state holding all service instances.
    
    Initialized during FastAPI lifespan startup.
    Accessed via request.app.state.services.
    """
    authority: Optional[WatermarkAuthorityService] = None
    generation_adapter: Optional["GenerationAdapter"] = None
    detection_service: Optional["DetectionService"] = None
    pipeline: Optional["StableDiffusionPipeline"] = None
    inference_client: Optional["InferenceClient"] = None
    
    # State flags
    pipeline_initialized: bool = False
    is_ready: bool = False
    
    # Configuration
    device: str = field(default_factory=detect_device)
    model_id: str = "runwayml/stable-diffusion-v1-5"


# =============================================================================
# Legacy global state (for backward compatibility during migration)
# TODO: Remove after all routes are migrated to use request.app.state
# =============================================================================
_authority_service: Optional[WatermarkAuthorityService] = None
_generation_adapter: Optional[GenerationAdapter] = None
_pipeline_cache: Optional["StableDiffusionPipeline"] = None
_pipeline_initialized: bool = False
_detection_service: Optional["DetectionService"] = None


# =============================================================================
# New dependency functions using request.app.state (preferred)
# =============================================================================

def get_app_state(request: Request) -> AppState:
    """
    Get application state from request.
    
    Args:
        request: FastAPI request object
    
    Returns:
        AppState instance
    """
    return request.app.state.services


def get_authority(request: Request) -> WatermarkAuthorityService:
    """
    Get WatermarkAuthorityService from app state.
    
    Args:
        request: FastAPI request object
    
    Returns:
        WatermarkAuthorityService instance
    
    Raises:
        RuntimeError: If authority not initialized
    """
    state = get_app_state(request)
    if state.authority is None:
        raise RuntimeError("WatermarkAuthorityService not initialized")
    return state.authority


def get_adapter(request: Request) -> "GenerationAdapter":
    """
    Get GenerationAdapter from app state.
    
    Args:
        request: FastAPI request object
    
    Returns:
        GenerationAdapter instance
    
    Raises:
        RuntimeError: If adapter not initialized
    """
    state = get_app_state(request)
    if state.generation_adapter is None:
        raise RuntimeError("GenerationAdapter not initialized")
    return state.generation_adapter


def get_detector(request: Request) -> "DetectionService":
    """
    Get DetectionService from app state.
    
    Args:
        request: FastAPI request object
    
    Returns:
        DetectionService instance
    
    Raises:
        RuntimeError: If detection service not initialized
    """
    state = get_app_state(request)
    if state.detection_service is None:
        raise RuntimeError("DetectionService not initialized")
    return state.detection_service


# =============================================================================
# Legacy singleton functions (for backward compatibility)
# 
# WARNING: These functions violate DEPENDENCY PURITY rules.
# They may perform lazy initialization (I/O) on first call.
# 
# Use ONLY:
# - In startup lifecycle (lifespan context manager)
# - In explicit service calls (not FastAPI `Depends()`)
# 
# For FastAPI dependencies, use:
# - get_authority(request) 
# - get_adapter(request)
# - get_detector(request)
# =============================================================================

def get_watermark_authority() -> WatermarkAuthorityService:
    """
    Get WatermarkAuthorityService singleton (legacy).
    
    WARNING: This function may perform lazy initialization.
    DO NOT use as a FastAPI dependency.
    Prefer using get_authority(request) with dependency injection.
    
    Acceptable uses:
    - Startup lifecycle (lifespan)
    - Background tasks
    
    Returns:
        WatermarkAuthorityService instance
    """
    global _authority_service
    
    if _authority_service is None:
        _authority_service = WatermarkAuthorityService()
        logger.info(
            "authority_initialized",
            extra={"method": "legacy_singleton"}
        )
    
    return _authority_service


def preload_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None) -> None:
    """
    Preload Stable Diffusion pipeline at startup.
    
    This function should be called during FastAPI startup to ensure the pipeline
    is loaded once and frozen. This eliminates first-request latency spikes and
    prevents accidental re-initialization.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to load on (default: auto-detect with priority: mps > cuda > cpu)
    
    Raises:
        RuntimeError: If pipeline is already initialized
    """
    global _pipeline_cache, _pipeline_initialized
    
    if _pipeline_initialized:
        if _pipeline_cache is None:
            raise RuntimeError("Pipeline initialization state is inconsistent")
        logger.warning(
            "pipeline_already_preloaded",
            extra={"action": "skipping"}
        )
        return
    
    if device is None:
        device = detect_device()
    
    logger.info(
        "pipeline_preloading",
        extra={"model_id": model_id, "device": device}
    )
    
    # Load pipeline without dtype (dtype will be set via .to(device))
    # This avoids passing dtype to constructor which causes warnings
    from diffusers import StableDiffusionPipeline
    _pipeline_cache = StableDiffusionPipeline.from_pretrained(model_id)
    
    # Move to device and set appropriate dtype
    # MPS and CPU use float32, CUDA can use float16 for performance
    if device == "cuda":
        _pipeline_cache = _pipeline_cache.to(device, dtype=torch.float16)
    else:
        _pipeline_cache = _pipeline_cache.to(device, dtype=torch.float32)
    
    _pipeline_initialized = True
    logger.info(
        "pipeline_preloaded",
        extra={"model_id": model_id, "device": device, "status": "success"}
    )


def get_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None) -> "StableDiffusionPipeline":
    """
    Get preloaded Stable Diffusion pipeline (singleton).
    
    This function returns the pipeline that was preloaded at startup.
    It never initializes the pipeline itself to prevent hidden reloads.
    
    This is used internally by GenerationAdapter and DetectionService.
    FastAPI routes should not call this directly.
    
    Args:
        model_id: Hugging Face model identifier (must match preloaded model)
        device: Device string (must match preloaded device, ignored if already set)
    
    Returns:
        StableDiffusionPipeline instance
    
    Raises:
        RuntimeError: If pipeline was not preloaded at startup
    """
    global _pipeline_cache, _pipeline_initialized
    
    if not _pipeline_initialized or _pipeline_cache is None:
        raise RuntimeError(
            "Pipeline was not preloaded at startup. "
            "Ensure preload_pipeline() is called during FastAPI startup."
        )
    
    # Validate model_id matches (optional check, but helpful for debugging)
    if model_id != "runwayml/stable-diffusion-v1-5":
        logger.warning(
            "pipeline_model_mismatch",
            extra={
                "requested": model_id,
                "preloaded": "runwayml/stable-diffusion-v1-5",
            }
        )
    
    return _pipeline_cache


def get_generation_adapter() -> "GenerationAdapter":
    """
    Get GenerationAdapter singleton (Phase-1: Stable Diffusion only).
    
    NOTE: Prefer using get_adapter(request) with dependency injection.
    
    TODO (Phase-2): This adapter can be removed when moving to client-side generation.
    The API will only issue watermark credentials, not perform generation.
    
    Returns:
        GenerationAdapter instance (currently StableDiffusionSeedBiasAdapter)
    """
    global _generation_adapter
    
    if _generation_adapter is None:
        # Phase-1: Stable Diffusion only
        from service.generation import StableDiffusionSeedBiasAdapter
        _generation_adapter = StableDiffusionSeedBiasAdapter(
            model_id="runwayml/stable-diffusion-v1-5",
            device=None,  # Auto-detect
            use_fp16=True,
        )
        logger.info(
            "generation_adapter_initialized",
            extra={"type": "StableDiffusionSeedBiasAdapter", "method": "legacy_singleton"}
        )
    
    return _generation_adapter


def get_detection_service() -> "DetectionService":
    """
    Get DetectionService singleton (Bayesian-only).
    
    NOTE: Prefer using get_detector(request) with dependency injection.
    
    Detection uses detector_type="bayesian" only.
    Legacy detection modes (fast_only/hybrid/full_inversion) are deprecated.
    
    Returns:
        DetectionService instance
    
    Raises:
        RuntimeError: If detection artifacts are not available
    """
    global _detection_service
    
    if _detection_service is None:
        # Check artifact availability before creating service
        resolver = get_artifact_resolver()
        artifact_result = resolver.resolve()
        
        if not artifact_result.is_available:
            raise RuntimeError(
                f"Detection artifacts not available: {artifact_result.error_message}\n"
                f"Detection service cannot be initialized without valid artifacts."
            )
        
        from service.detection import DetectionService
        authority = get_watermark_authority()
        pipeline = get_pipeline()
        _detection_service = DetectionService(
            authority=authority,
            pipeline=pipeline,
        )
        logger.info(
            "detection_service_initialized",
            extra={"method": "legacy_singleton"}
        )
    
    return _detection_service


def is_detection_available() -> bool:
    """
    Check if detection artifacts are available.
    
    Returns:
        True if detection artifacts are available, False otherwise
    """
    resolver = get_artifact_resolver()
    artifact_result = resolver.resolve()
    return artifact_result.is_available


def get_detection_availability_status() -> dict:
    """
    Get detailed detection artifact availability status.
    
    Returns:
        Dictionary with availability status and paths
    """
    resolver = get_artifact_resolver()
    return resolver.get_availability_status()


# =============================================================================
# Initialization functions for lifespan management
# =============================================================================

async def initialize_app_state(app_state: AppState) -> None:
    """
    Initialize all services in app state.
    
    Called during FastAPI lifespan startup.
    
    Args:
        app_state: AppState instance to initialize
    """
    logger.info("app_state_initializing")
    
    # Initialize authority
    app_state.authority = WatermarkAuthorityService()
    logger.info("authority_initialized", extra={"method": "app_state"})
    
    # Preload pipeline (if not in remote inference mode)
    # In remote mode, pipeline is loaded on GPU workers
    inference_mode = "local"  # TODO: Get from settings
    
    if inference_mode == "local":
        device = app_state.device
        model_id = app_state.model_id
        
        logger.info(
            "pipeline_loading",
            extra={"model_id": model_id, "device": device}
        )
        
        from diffusers import StableDiffusionPipeline
        app_state.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        
        if device == "cuda":
            app_state.pipeline = app_state.pipeline.to(device, dtype=torch.float16)
        else:
            app_state.pipeline = app_state.pipeline.to(device, dtype=torch.float32)
        
        app_state.pipeline_initialized = True
        logger.info(
            "pipeline_loaded",
            extra={"model_id": model_id, "device": device}
        )
        
        # Initialize generation adapter
        from service.generation import StableDiffusionSeedBiasAdapter
        app_state.generation_adapter = StableDiffusionSeedBiasAdapter(
            model_id=model_id,
            device=device,
            use_fp16=(device == "cuda"),
        )
        logger.info("generation_adapter_initialized", extra={"method": "app_state"})
        
        # Initialize detection service if artifacts available
        if is_detection_available():
            from service.detection import DetectionService
            app_state.detection_service = DetectionService(
                authority=app_state.authority,
                pipeline=app_state.pipeline,
            )
            logger.info("detection_service_initialized", extra={"method": "app_state"})
    
    app_state.is_ready = True
    logger.info("app_state_initialized")


async def cleanup_app_state(app_state: AppState) -> None:
    """
    Cleanup app state during shutdown.
    
    Called during FastAPI lifespan shutdown.
    
    Args:
        app_state: AppState instance to cleanup
    """
    logger.info("app_state_cleanup_starting")
    
    app_state.is_ready = False
    
    # Disable micro-batching if enabled
    if app_state.detection_service is not None:
        if hasattr(app_state.detection_service, '_detection_worker') and \
           app_state.detection_service._detection_worker is not None:
            await app_state.detection_service.disable_micro_batching()
            logger.info("micro_batching_disabled")
    
    # Clear references
    app_state.detection_service = None
    app_state.generation_adapter = None
    app_state.pipeline = None
    app_state.authority = None
    
    logger.info("app_state_cleanup_completed")
