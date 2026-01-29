"""
Pydantic schemas for GPU worker requests and responses.

These schemas define the API contract for the GPU worker service.

SECURITY INVARIANT:
- master_key is NEVER included in these schemas
- Workers receive only derived_key (scoped, non-reversible)
- key_fingerprint is used for cache keying and validation
- Workers must validate key_fingerprint format and log mismatches
"""
from __future__ import annotations

from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OperationType(str, Enum):
    """Operation types for idempotency classification."""
    GENERATION = "generation"  # Non-idempotent (stochastic)
    DETECTION = "detection"    # Idempotent (deterministic)


class WorkerBaseModel(BaseModel):
    """Base model for worker schemas."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


# =============================================================================
# Detection Schemas
# =============================================================================


class DetectWorkerRequest(WorkerBaseModel):
    """
    Request schema for POST /v1/detect endpoint.
    
    Contains all information needed for watermark detection.
    
    SECURITY INVARIANT:
    - NEVER contains master_key
    - derived_key is scoped to detection operation
    - key_fingerprint is validated before processing
    """
    
    image_base64: str = Field(..., description="Base64-encoded image bytes")
    key_id: str = Field(..., description="Watermark key identifier")
    derived_key: str = Field(
        ..., 
        description="Scoped derived key for g-value computation (NEVER master_key)"
    )
    key_fingerprint: str = Field(
        ...,
        description="Canonical key fingerprint for cache keying and validation"
    )
    g_field_config: dict[str, Any] = Field(..., description="G-field configuration")
    detection_config: dict[str, Any] = Field(..., description="Detection configuration")
    request_id: str = Field(..., description="Request ID for tracing and idempotency")
    
    # Idempotency fields
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
    
    # Optional inversion configuration
    inversion_config: Optional[dict[str, Any]] = Field(
        None,
        description="DDIM inversion configuration"
    )


class DetectWorkerResponse(WorkerBaseModel):
    """
    Response schema for POST /v1/detect endpoint.
    
    Contains detection results.
    """
    
    detected: bool = Field(..., description="Whether watermark was detected")
    score: float = Field(..., description="Detection score (log-odds)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    log_odds: float = Field(..., description="Log-odds ratio")
    posterior: float = Field(..., ge=0.0, le=1.0, description="Posterior probability")
    request_id: str = Field(..., description="Request ID for tracing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    key_fingerprint: Optional[str] = Field(None, description="Key fingerprint used for validation")
    g_field_config_hash: Optional[str] = Field(None, description="G-field config hash")


# =============================================================================
# Generation Schemas
# =============================================================================


class GenerateWorkerRequest(WorkerBaseModel):
    """
    Request schema for POST /v1/generate endpoint.
    
    Contains all information needed for watermarked image generation.
    
    SECURITY INVARIANT:
    - NEVER contains master_key
    - derived_key is scoped to generation operation
    - key_fingerprint is validated before processing
    
    RETRY SEMANTICS:
    - Generation is NON-IDEMPOTENT (stochastic)
    - Worker must reject retries after partial execution
    """
    
    prompt: str = Field(..., description="Text prompt for generation")
    derived_key: str = Field(
        ..., 
        description="Scoped derived key for watermarking (NEVER master_key)"
    )
    key_fingerprint: str = Field(
        ...,
        description="Canonical key fingerprint for cache keying and validation"
    )
    key_id: str = Field(..., description="Watermark key identifier")
    embedding_config: dict[str, Any] = Field(..., description="Embedding configuration")
    request_id: str = Field(..., description="Request ID for tracing and idempotency")
    
    # Idempotency fields
    operation_type: OperationType = Field(
        default=OperationType.GENERATION,
        description="Operation type for retry classification (generation = non-idempotent)"
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


class GenerateWorkerResponse(WorkerBaseModel):
    """
    Response schema for POST /v1/generate endpoint.
    
    Contains generated watermarked image.
    """
    
    image_base64: str = Field(..., description="Base64-encoded generated image")
    seed_used: int = Field(..., description="Random seed that was used")
    request_id: str = Field(..., description="Request ID for tracing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    key_fingerprint: Optional[str] = Field(None, description="Key fingerprint used for validation")


# =============================================================================
# Health Check Schemas
# =============================================================================


class HealthResponse(WorkerBaseModel):
    """
    Response schema for GET /v1/health endpoint.
    
    Returns health status and resource usage.
    """
    
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    gpu_memory_used_mb: Optional[int] = Field(None, description="GPU memory used (MB)")
    gpu_memory_total_mb: Optional[int] = Field(None, description="GPU memory total (MB)")
    active_requests: Optional[int] = Field(None, description="Number of active requests")
    queue_size: Optional[int] = Field(None, description="Number of queued requests")
    uptime_seconds: Optional[float] = Field(None, description="Worker uptime in seconds")


class ReadyResponse(WorkerBaseModel):
    """
    Response schema for GET /v1/ready endpoint.
    
    Returns 200 only when worker can accept requests.
    """
    
    ready: bool = Field(..., description="Whether worker is ready")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(WorkerBaseModel):
    """
    Error response schema.
    
    Returned for all error responses.
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(None, description="Request ID for correlation")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")

