"""
Pydantic schemas for inference client requests and responses.

These schemas define the contract between the API service and GPU workers.
They are used for both local and remote inference.

SECURITY INVARIANT:
- master_key is NEVER included in these schemas
- Workers receive only derived_key (scoped, non-reversible)
- key_fingerprint is used for cache keying and validation
"""
from __future__ import annotations

from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OperationType(str, Enum):
    """Operation types for idempotency classification."""
    GENERATION = "generation"  # Non-idempotent (stochastic)
    DETECTION = "detection"    # Idempotent (deterministic)


class InferenceBaseModel(BaseModel):
    """Base model for inference schemas."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


# =============================================================================
# Detection Schemas
# =============================================================================


class DetectInferenceRequest(InferenceBaseModel):
    """
    Request schema for detection inference.
    
    Contains all information needed for a GPU worker to perform detection.
    
    SECURITY INVARIANT:
    - master_key is NEVER included (workers receive only derived_key)
    - derived_key is scoped to detection operation
    - key_fingerprint is used for cache keying and validation
    """
    
    image_bytes: bytes = Field(..., description="Raw image bytes")
    key_id: str = Field(..., description="Watermark key identifier")
    derived_key: str = Field(
        ..., 
        description="Scoped derived key for g-value computation (never master_key)"
    )
    key_fingerprint: str = Field(
        ...,
        description="Canonical key fingerprint for cache keying and validation"
    )
    g_field_config: dict[str, Any] = Field(..., description="G-field configuration")
    detection_config: dict[str, Any] = Field(..., description="Detection configuration")
    request_id: str = Field(..., description="Request ID for tracing and idempotency")
    
    # Idempotency fields (detection is idempotent - retries are safe)
    operation_type: OperationType = Field(
        default=OperationType.DETECTION,
        description="Operation type for retry classification"
    )
    idempotency_key: Optional[str] = Field(
        None,
        description="Client-provided idempotency key for deduplication"
    )
    deterministic_seed: Optional[int] = Field(
        None,
        description="Deterministic seed for reproducible results"
    )
    
    # Optional parameters
    inversion_config: Optional[dict[str, Any]] = Field(
        None,
        description="DDIM inversion configuration"
    )


class DetectInferenceResponse(InferenceBaseModel):
    """
    Response schema for detection inference.
    
    Contains detection results from the GPU worker.
    """
    
    detected: bool = Field(..., description="Whether watermark was detected")
    score: float = Field(..., description="Detection score (log-odds)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    log_odds: float = Field(..., description="Log-odds ratio")
    posterior: float = Field(..., ge=0.0, le=1.0, description="Posterior probability")
    request_id: str = Field(..., description="Request ID for tracing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Key validation metadata
    key_fingerprint: Optional[str] = Field(
        None,
        description="Key fingerprint used for validation"
    )
    
    # Optional metadata
    g_field_config_hash: Optional[str] = Field(
        None,
        description="Hash of g-field config used"
    )


# =============================================================================
# Generation Schemas
# =============================================================================


class GenerateInferenceRequest(InferenceBaseModel):
    """
    Request schema for generation inference.
    
    Contains all information needed for a GPU worker to generate a watermarked image.
    
    SECURITY INVARIANT:
    - master_key is NEVER included (workers receive only derived_key)
    - derived_key is scoped to generation operation
    - key_fingerprint is used for cache keying and validation
    
    RETRY SEMANTICS:
    - Generation is NON-IDEMPOTENT (stochastic)
    - Retries are ONLY safe if request never reached worker
    - Once partial execution starts, DO NOT retry
    """
    
    prompt: str = Field(..., description="Text prompt for generation")
    derived_key: str = Field(
        ..., 
        description="Scoped derived key for watermarking (never master_key)"
    )
    key_fingerprint: str = Field(
        ...,
        description="Canonical key fingerprint for cache keying and validation"
    )
    key_id: str = Field(..., description="Watermark key identifier")
    embedding_config: dict[str, Any] = Field(..., description="Embedding configuration")
    request_id: str = Field(..., description="Request ID for tracing and idempotency")
    
    # Idempotency fields (generation is NON-idempotent)
    operation_type: OperationType = Field(
        default=OperationType.GENERATION,
        description="Operation type for retry classification"
    )
    idempotency_key: Optional[str] = Field(
        None,
        description="Client-provided idempotency key for deduplication"
    )
    deterministic_seed: Optional[int] = Field(
        None,
        description="Deterministic seed for reproducible generation"
    )
    
    # Generation parameters
    num_inference_steps: int = Field(50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(1.0, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    height: int = Field(512, ge=64, le=2048, description="Image height")
    width: int = Field(512, ge=64, le=2048, description="Image width")


class GenerateInferenceResponse(InferenceBaseModel):
    """
    Response schema for generation inference.
    
    Contains generated image from the GPU worker.
    """
    
    image_bytes: bytes = Field(..., description="Generated image bytes (PNG)")
    seed_used: int = Field(..., description="Random seed that was used")
    request_id: str = Field(..., description="Request ID for tracing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Key validation metadata
    key_fingerprint: Optional[str] = Field(
        None,
        description="Key fingerprint used for validation"
    )
    
    # Optional metadata
    g_values_computed: Optional[bool] = Field(
        None,
        description="Whether g-values were computed for validation"
    )


# =============================================================================
# Health Check Schemas
# =============================================================================


class HealthStatusEnum(str, Enum):
    """Health status values."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthStatus(InferenceBaseModel):
    """
    Health status response from GPU worker.
    
    Used for health checks and load balancing decisions.
    """
    
    status: HealthStatusEnum = Field(..., description="Overall health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    
    # Resource usage
    gpu_memory_used_mb: Optional[int] = Field(
        None,
        description="GPU memory usage in MB"
    )
    gpu_memory_total_mb: Optional[int] = Field(
        None,
        description="Total GPU memory in MB"
    )
    
    # Load metrics
    active_requests: Optional[int] = Field(
        None,
        description="Number of active requests"
    )
    queue_size: Optional[int] = Field(
        None,
        description="Number of queued requests"
    )
    
    # Uptime
    uptime_seconds: Optional[float] = Field(
        None,
        description="Worker uptime in seconds"
    )


class ReadyStatus(InferenceBaseModel):
    """
    Readiness status response from GPU worker.
    
    Returns 200 only when worker can accept requests.
    """
    
    ready: bool = Field(..., description="Whether worker is ready")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")

