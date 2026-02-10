"""
Pydantic models for GPU worker request/response validation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Generation
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for image generation.
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    """
    
    key_id: str = Field(..., description="Key identifier (public PRF index)")
    master_key: str = Field(..., description="Master key for watermark embedding")
    key_fingerprint: str = Field(..., description="Key fingerprint for validation")
    
    prompt: str = Field(default="a beautiful landscape", description="Text prompt")
    seed: Optional[int] = Field(default=None, description="Random seed")
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    
    embedding_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Watermark embedding configuration"
    )
    request_id: str = Field(..., description="Request ID for tracing")


class GenerateResponse(BaseModel):
    """Response from image generation."""
    
    image_base64: str = Field(..., description="Base64-encoded generated image")
    seed_used: int = Field(..., description="Actual seed used")
    request_id: str = Field(..., description="Request ID")
    processing_time_ms: float = Field(..., description="Processing time")
    key_fingerprint: str = Field(..., description="Key fingerprint used")


# =============================================================================
# Detection / Reverse DDIM
# =============================================================================


class ReverseDDIMRequest(BaseModel):
    """Request for DDIM inversion (detection).
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    compute_g_values() uses (master_key, key_id) directly.
    """
    
    key_id: str = Field(..., description="Key identifier (public PRF index)")
    master_key: str = Field(..., description="Master key for g-value computation")
    key_fingerprint: str = Field(..., description="Key fingerprint for validation")
    
    image_base64: str = Field(..., description="Base64-encoded image to analyze")
    
    g_field_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="G-field configuration"
    )
    detection_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detection parameters"
    )
    inversion_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="DDIM inversion parameters"
    )
    request_id: str = Field(..., description="Request ID for tracing")


class ReverseDDIMResponse(BaseModel):
    """Response from DDIM inversion."""
    
    detected: bool = Field(..., description="Whether watermark was detected")
    score: float = Field(..., description="Normalized detection score")
    threshold: float = Field(..., description="Deployment threshold in score units")
    confidence: float = Field(..., description="Detection confidence")
    log_odds: float = Field(..., description="Log odds ratio")
    posterior: float = Field(..., description="Bayesian posterior")
    
    request_id: str = Field(..., description="Request ID")
    processing_time_ms: float = Field(..., description="Processing time")
    key_fingerprint: str = Field(..., description="Key fingerprint used")
    g_field_config_hash: Optional[str] = Field(default=None, description="Config hash")
    
    # Optional latent data (for debugging)
    latent_shape: Optional[list] = Field(default=None, description="Shape of inverted latent")


# =============================================================================
# Health
# =============================================================================


class HealthResponse(BaseModel):
    """GPU worker health response."""
    
    status: str = Field(..., description="Worker status: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(default=True, description="Whether GPU is available")
    gpu_memory_used_mb: Optional[int] = Field(default=None)
    gpu_memory_total_mb: Optional[int] = Field(default=None)
    active_requests: int = Field(default=0)
    uptime_seconds: Optional[float] = Field(default=None)


class ReadyResponse(BaseModel):
    """Simple readiness response."""
    
    ready: bool
    model_loaded: bool
    gpu_available: bool


# =============================================================================
# Errors
# =============================================================================


class ErrorResponse(BaseModel):
    """Error response."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None)
