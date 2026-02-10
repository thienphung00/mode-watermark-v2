"""
Pydantic models for API request/response validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Key Registration
# =============================================================================


class KeyRegisterRequest(BaseModel):
    """Request to register a new watermark key."""
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to associate with the key"
    )


class KeyRegisterResponse(BaseModel):
    """Response from key registration."""
    
    key_id: str = Field(..., description="Unique key identifier")
    fingerprint: str = Field(..., description="Key fingerprint for verification")
    created_at: str = Field(..., description="ISO timestamp of creation")


class KeyInfo(BaseModel):
    """Information about a registered key."""
    
    key_id: str
    fingerprint: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True


class KeyListResponse(BaseModel):
    """Response listing all keys."""
    
    keys: List[KeyInfo]
    total: int


# =============================================================================
# Image Generation
# =============================================================================


class GenerateRequest(BaseModel):
    """Request to generate a watermarked image."""
    
    key_id: str = Field(..., description="Key ID for watermarking")
    prompt: str = Field(
        default="a beautiful landscape",
        description="Text prompt for image generation"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed (optional, for reproducibility)"
    )
    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of diffusion steps"
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale"
    )
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)


class GenerateResponse(BaseModel):
    """Response from image generation."""
    
    image_url: str = Field(..., description="URL or path to generated image")
    key_id: str = Field(..., description="Key ID used")
    seed_used: int = Field(..., description="Actual seed used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Detection
# =============================================================================


class DetectRequest(BaseModel):
    """Request to detect watermark in an image."""
    
    key_id: str = Field(..., description="Key ID to check against")
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL to image (alternative to base64)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "wm_abc123",
                "image_base64": "<base64-encoded-image>"
            }
        }


class DetectResponse(BaseModel):
    """Response from watermark detection."""
    
    detected: bool = Field(..., description="Whether watermark was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    key_id: str = Field(..., description="Key ID checked")
    score: float = Field(..., description="Detection score (normalized)")
    threshold: float = Field(..., description="Threshold used for decision")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Health
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(default="1.0.0", description="API version")
    gpu_worker_connected: bool = Field(..., description="GPU worker connectivity")
    keys_loaded: int = Field(..., description="Number of keys in store")


# =============================================================================
# GPU Worker Schemas (Internal)
# =============================================================================


class GPUGenerateRequest(BaseModel):
    """Internal request to GPU worker for generation.
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    """
    
    key_id: str
    master_key: str  # Master key for watermark embedding
    key_fingerprint: str
    prompt: str
    seed: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    embedding_config: Dict[str, Any] = Field(default_factory=dict)
    request_id: str


class GPUGenerateResponse(BaseModel):
    """Internal response from GPU worker for generation."""
    
    image_base64: str
    seed_used: int
    request_id: str
    processing_time_ms: float
    key_fingerprint: str


class GPUDetectRequest(BaseModel):
    """Internal request to GPU worker for detection.
    
    ARCHITECTURAL REQUIREMENT: Uses master_key only.
    derived_key is NOT used - key_id is a public PRF index.
    compute_g_values() uses (master_key, key_id) directly.
    """
    
    key_id: str
    master_key: str  # Master key for g-value computation
    key_fingerprint: str
    image_base64: str
    g_field_config: Dict[str, Any]
    detection_config: Dict[str, Any]
    inversion_config: Dict[str, Any] = Field(default_factory=dict)
    request_id: str


class GPUDetectResponse(BaseModel):
    """Internal response from GPU worker for detection."""
    
    detected: bool
    score: float
    threshold: float
    confidence: float
    log_odds: float
    posterior: float
    request_id: str
    processing_time_ms: float
    key_fingerprint: str
    g_field_config_hash: Optional[str] = None


class GPUHealthResponse(BaseModel):
    """GPU worker health response."""
    
    status: str
    model_loaded: bool
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    active_requests: int = 0


# =============================================================================
# Error Responses
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
