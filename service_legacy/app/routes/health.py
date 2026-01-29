"""
Health check routes.

PUBLIC endpoints (safe for internet exposure):
- GET /health - Liveness/readiness boolean only
- GET /live - Minimal liveness probe

INTERNAL endpoint (protect via network policy):
- GET /health/internal - Detailed operational metrics
"""
from fastapi import APIRouter, Request

from service.app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    PUBLIC health check endpoint.
    
    Returns ONLY readiness/liveness boolean.
    Safe for public exposure (Kubernetes probes, load balancers).
    
    Does NOT expose:
    - GPU utilization
    - Queue sizes
    - Memory usage
    - Internal metrics
    
    Returns:
        Minimal health status
    """
    return HealthResponse(status="ok")


@router.get("/live")
async def liveness() -> dict:
    """
    PUBLIC liveness probe (minimal).
    
    Returns 200 if the process is running.
    Does not check any dependencies.
    
    This is the safest endpoint for public exposure.
    """
    return {"live": True}


@router.get("/health/internal")
async def health_internal(request: Request) -> dict:
    """
    INTERNAL health check with detailed metrics.
    
    WARNING: This endpoint exposes operational metrics.
    It MUST be protected via network policy or authentication.
    Do NOT expose to public internet.
    
    Returns:
    - status: Overall status
    - detection_available: Whether detection artifacts are loaded
    - pipeline_ready: Whether ML pipeline is ready
    """
    from service.app.dependencies import is_detection_available, get_app_state
    
    state = get_app_state(request)
    
    return {
        "status": "ok" if state.is_ready else "initializing",
        "detection_available": is_detection_available(),
        "pipeline_ready": state.pipeline_initialized,
        # NOTE: No GPU metrics exposed here - those are worker-level
        # This is the API service, not the GPU worker
    }

