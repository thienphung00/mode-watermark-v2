"""
Pydantic schemas for API requests and responses.

Product-level schemas only. No research internals exposed.

Architecture:
- ServiceBaseModel: Base class with shared configuration
- Type annotations: KeyIdField, Base64ImageField for validation
- All validation in schemas, not routes
"""
from __future__ import annotations

import base64
import re
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Base Model Configuration
# ============================================================================


class ServiceBaseModel(BaseModel):
    """
    Base model with shared configuration for all service schemas.
    
    Configuration:
    - Strip whitespace from strings
    - Validate assignments (mutation validation)
    - Forbid extra fields (strict schema)
    - Populate by name for serialization
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "description": "Watermarking Service API Schema"
        }
    )


# ============================================================================
# Custom Field Types
# ============================================================================

# Key ID pattern: wm_ followed by 10 hex characters
KEY_ID_PATTERN = r"^wm_[a-f0-9]{10}$"

# Maximum base64 image size (50MB decoded)
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024

# Minimum base64 image size (must have some content)
MIN_IMAGE_SIZE_BYTES = 100


def validate_key_id(value: str) -> str:
    """
    Validate key_id format.
    
    Args:
        value: Key ID string
    
    Returns:
        Validated key ID
    
    Raises:
        ValueError: If format is invalid
    """
    if not re.match(KEY_ID_PATTERN, value):
        raise ValueError(
            f"Invalid key_id format. Expected 'wm_' followed by 10 hex characters, "
            f"got '{value}'"
        )
    return value


def validate_base64_image(value: str) -> str:
    """
    Validate base64-encoded image.
    
    Args:
        value: Base64-encoded image string
    
    Returns:
        Validated base64 string
    
    Raises:
        ValueError: If base64 is invalid or image too large/small
    """
    # Strip data URL prefix if present
    if value.startswith("data:"):
        # Format: data:image/png;base64,XXXXX
        try:
            value = value.split(",", 1)[1]
        except IndexError:
            raise ValueError("Invalid data URL format")
    
    # Validate base64 encoding
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")
    
    # Check size limits
    size = len(decoded)
    if size < MIN_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too small: {size} bytes. Minimum: {MIN_IMAGE_SIZE_BYTES} bytes"
        )
    if size > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too large: {size} bytes. Maximum: {MAX_IMAGE_SIZE_BYTES} bytes (50MB)"
        )
    
    return value


# Type annotations for fields
KeyIdField = Annotated[
    str,
    Field(
        description="Watermark key identifier (format: wm_[10 hex chars])",
        examples=["wm_abc123def0"],
    )
]

Base64ImageField = Annotated[
    str,
    Field(
        description="Base64-encoded image (PNG, JPEG, WebP)",
        max_length=70_000_000,  # ~50MB in base64
    )
]


# ============================================================================
# Generate Schemas
# ============================================================================


class GenerateRequest(ServiceBaseModel):
    """Request schema for POST /generate endpoint."""
    
    prompt: str = Field(
        ...,
        description="Text prompt for image generation",
        min_length=1,
        max_length=1000,
        examples=["A beautiful sunset over mountains"]
    )
    key_id: Optional[str] = Field(
        None,
        description="Optional key identifier (if None, creates new)"
    )
    num_inference_steps: Optional[int] = Field(
        50,
        ge=1,
        le=100,
        description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(
        1.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale"
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2**31 - 1,
        description="Random seed (optional)"
    )
    height: Optional[int] = Field(
        None,
        ge=64,
        le=2048,
        description="Image height (64-2048 pixels)"
    )
    width: Optional[int] = Field(
        None,
        ge=64,
        le=2048,
        description="Image width (64-2048 pixels)"
    )
    
    @field_validator("key_id")
    @classmethod
    def validate_key_id_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate key_id format if provided."""
        if v is not None:
            return validate_key_id(v)
        return v
    
    @field_validator("height", "width")
    @classmethod
    def validate_dimensions_multiple_of_8(cls, v: Optional[int]) -> Optional[int]:
        """Validate that dimensions are multiples of 8 (required by SD)."""
        if v is not None and v % 8 != 0:
            raise ValueError(
                f"Dimension must be a multiple of 8 for Stable Diffusion, got {v}"
            )
        return v


class GenerationMetadata(ServiceBaseModel):
    """Generation metadata (product-level, no research internals)."""
    
    seed: Optional[int] = Field(None, description="Random seed used")
    num_inference_steps: int = Field(..., description="Number of inference steps")
    guidance_scale: float = Field(..., description="Guidance scale")
    model_version: str = Field(..., description="Model version identifier")
    height: Optional[int] = Field(None, description="Image height")
    width: Optional[int] = Field(None, description="Image width")


class GenerateResponse(ServiceBaseModel):
    """Response schema for POST /generate endpoint."""
    
    image_url: Optional[str] = Field(
        None,
        description="URL to generated image (if stored)"
    )
    image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded image (if returned directly)"
    )
    key_id: str = Field(..., description="Watermark key identifier")
    generation_metadata: GenerationMetadata = Field(
        ...,
        description="Generation metadata"
    )
    watermark_version: str = Field(..., description="Watermark policy version")
    request_id: Optional[str] = Field(
        None,
        description="Request ID for correlation"
    )


# ============================================================================
# Detect Schemas
# ============================================================================


class DetectRequest(ServiceBaseModel):
    """Request schema for POST /detect endpoint."""
    
    image_base64: Base64ImageField = Field(
        ...,
        description="Base64-encoded image"
    )
    key_id: KeyIdField = Field(..., description="Watermark key identifier")
    
    # NOTE: Detection is prompt-agnostic. The API does not require or trust prompts.
    # Detection uses unconditional DDIM inversion (prompt="", guidance_scale=1.0).
    
    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate base64 image."""
        return validate_base64_image(v)
    
    @field_validator("key_id")
    @classmethod
    def validate_key_id_format(cls, v: str) -> str:
        """Validate key_id format."""
        return validate_key_id(v)


class DetectResponse(ServiceBaseModel):
    """Response schema for POST /detect endpoint."""
    
    detected: bool = Field(..., description="Whether watermark was detected")
    score: float = Field(
        ...,
        description="Detection score (log-odds, higher = more confident)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (posterior probability, 0-1)"
    )
    policy_version: str = Field(..., description="Watermark policy version used")
    posterior: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Posterior probability P(watermarked | g)"
    )
    log_odds: float = Field(..., description="Log-odds ratio")
    is_watermarked: bool = Field(
        ...,
        description="Detection decision (same as detected)"
    )
    watermark_version: str = Field(
        ...,
        description="Watermark policy version (same as policy_version)"
    )
    g_field_config_hash: Optional[str] = Field(
        None,
        description="G-field config hash for validation"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for correlation"
    )


# ============================================================================
# Evaluation Schemas
# ============================================================================


class ImperceptibilityEvalRequest(ServiceBaseModel):
    """Request schema for POST /evaluate/imperceptibility endpoint."""
    
    prompt: str = Field(
        ...,
        description="Text prompt for image generation",
        min_length=1,
        max_length=1000
    )
    seed: int = Field(
        ...,
        ge=0,
        le=2**31 - 1,
        description="Random seed (required for determinism)"
    )
    num_inference_steps: Optional[int] = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale"
    )
    height: Optional[int] = Field(
        default=512,
        ge=64,
        le=2048,
        description="Image height"
    )
    width: Optional[int] = Field(
        default=512,
        ge=64,
        le=2048,
        description="Image width"
    )
    
    @field_validator("height", "width")
    @classmethod
    def validate_dimensions_multiple_of_8(cls, v: Optional[int]) -> Optional[int]:
        """Validate that dimensions are multiples of 8 (required by SD)."""
        if v is not None and v % 8 != 0:
            raise ValueError(
                f"Dimension must be a multiple of 8 for Stable Diffusion, got {v}"
            )
        return v


class DifferenceMetrics(ServiceBaseModel):
    """Difference metrics between baseline and watermarked images."""
    
    l2: float = Field(..., description="Normalized L2 distance (pixel space)")
    psnr: float = Field(..., description="Peak Signal-to-Noise Ratio (dB)")
    ssim: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Structural Similarity Index (0-1)"
    )


class ModelInfo(ServiceBaseModel):
    """Model information for evaluation response."""
    
    model_family: str = Field(..., description="Model family identifier")
    num_inference_steps: int = Field(
        ...,
        description="Number of inference steps used"
    )
    guidance_scale: float = Field(..., description="Guidance scale used")


class ImperceptibilityEvalResponse(ServiceBaseModel):
    """
    Response schema for POST /evaluate/imperceptibility endpoint.
    
    EVALUATION-ONLY: This endpoint is not part of the production watermarking system.
    Outputs must not be used for detection benchmarking.
    """
    
    baseline_image_base64: str = Field(
        ...,
        description="Base64-encoded baseline (unwatermarked) image"
    )
    watermarked_image_base64: str = Field(
        ...,
        description="Base64-encoded watermarked image (evaluation-only)"
    )
    difference_metrics: DifferenceMetrics = Field(
        ...,
        description="Difference metrics between images"
    )
    model_info: ModelInfo = Field(
        ...,
        description="Model and generation parameters"
    )
    disclaimer: str = Field(
        default="Evaluation-only output. Not valid for production use or detection.",
        description="Disclaimer about evaluation-only nature of output"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for correlation"
    )


# ============================================================================
# Health Check Schema
# ============================================================================


class HealthResponse(ServiceBaseModel):
    """Response schema for /health endpoint."""
    
    status: str = Field("ok", description="Service status")
    version: Optional[str] = Field(None, description="Service version")
    ready: Optional[bool] = Field(None, description="Whether service is ready")


class ReadyResponse(ServiceBaseModel):
    """Response schema for /ready endpoint."""
    
    ready: bool = Field(..., description="Whether service is ready to accept requests")
    pipeline_loaded: Optional[bool] = Field(
        None,
        description="Whether ML pipeline is loaded"
    )
    artifacts_available: Optional[bool] = Field(
        None,
        description="Whether detection artifacts are available"
    )


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorDetail(ServiceBaseModel):
    """Error detail for structured error responses."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")


class ErrorResponse(ServiceBaseModel):
    """Structured error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[list[ErrorDetail]] = Field(
        None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for correlation"
    )
