"""
Custom exceptions for the watermarking service.

Provides structured exception handling with:
- Service-level exceptions (not HTTP-specific)
- Automatic mapping to HTTP status codes
- Structured error responses
- Request ID correlation

Exception Hierarchy:
- ServiceError (base)
  - ValidationError (400)
  - NotFoundError (404)
  - ConflictError (409)
  - RateLimitError (429)
  - InferenceError (502)
  - ServiceUnavailableError (503)
  - TimeoutError (504)
"""
from __future__ import annotations

from typing import Any, Optional


class ServiceError(Exception):
    """
    Base exception for all service errors.
    
    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error details
        status_code: HTTP status code for the error
    """
    
    status_code: int = 500
    error_type: str = "internal_error"
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize service error.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code (defaults to error_type)
            details: Additional error details
        """
        self.message = message
        self.code = code or self.error_type
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        result = {
            "error": self.error_type,
            "message": self.message,
            "code": self.code,
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(ServiceError):
    """
    Validation error for invalid input.
    
    Maps to HTTP 400 Bad Request.
    """
    
    status_code: int = 400
    error_type: str = "validation_error"
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize validation error.
        
        Args:
            message: Human-readable error message
            field: Field that caused the validation error
            details: Additional error details
        """
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, code="invalid_input", details=details)


class NotFoundError(ServiceError):
    """
    Resource not found error.
    
    Maps to HTTP 404 Not Found.
    """
    
    status_code: int = 404
    error_type: str = "not_found"
    
    def __init__(
        self,
        resource: str,
        identifier: str,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize not found error.
        
        Args:
            resource: Type of resource (e.g., "watermark", "key")
            identifier: Resource identifier
            details: Additional error details
        """
        message = f"{resource} '{identifier}' not found"
        details = details or {}
        details["resource"] = resource
        details["identifier"] = identifier
        super().__init__(message, code=f"{resource}_not_found", details=details)


class ConflictError(ServiceError):
    """
    Resource conflict error (e.g., already exists).
    
    Maps to HTTP 409 Conflict.
    """
    
    status_code: int = 409
    error_type: str = "conflict"
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, code="resource_conflict", details=details)


class RateLimitError(ServiceError):
    """
    Rate limit exceeded error.
    
    Maps to HTTP 429 Too Many Requests.
    """
    
    status_code: int = 429
    error_type: str = "rate_limit_exceeded"
    
    def __init__(
        self,
        message: str = "Rate limit exceeded. Please try again later.",
        retry_after: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds to wait before retrying
            details: Additional error details
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, code="rate_limit", details=details)
        self.retry_after = retry_after


class InferenceError(ServiceError):
    """
    Inference/GPU worker error.
    
    Maps to HTTP 502 Bad Gateway.
    """
    
    status_code: int = 502
    error_type: str = "inference_error"
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, code="inference_failed", details=details)


class ServiceUnavailableError(ServiceError):
    """
    Service unavailable error.
    
    Maps to HTTP 503 Service Unavailable.
    """
    
    status_code: int = 503
    error_type: str = "service_unavailable"
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        reason: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize service unavailable error.
        
        Args:
            message: Human-readable error message
            reason: Reason for unavailability
            details: Additional error details
        """
        details = details or {}
        if reason:
            details["reason"] = reason
        super().__init__(message, code="service_unavailable", details=details)


class TimeoutError(ServiceError):
    """
    Request timeout error.
    
    Maps to HTTP 504 Gateway Timeout.
    """
    
    status_code: int = 504
    error_type: str = "timeout"
    
    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Human-readable error message
            timeout_seconds: Timeout duration
            details: Additional error details
        """
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, code="request_timeout", details=details)


class WatermarkRevokedError(ServiceError):
    """
    Watermark has been revoked error.
    
    Maps to HTTP 410 Gone.
    """
    
    status_code: int = 410
    error_type: str = "watermark_revoked"
    
    def __init__(
        self,
        key_id: str,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize watermark revoked error.
        
        Args:
            key_id: Revoked watermark key ID
            details: Additional error details
        """
        message = f"Watermark '{key_id}' has been revoked"
        details = details or {}
        details["key_id"] = key_id
        super().__init__(message, code="watermark_revoked", details=details)


class ArtifactsNotConfiguredError(ServiceUnavailableError):
    """
    Detection artifacts not configured error.
    
    Specialized ServiceUnavailableError for missing artifacts.
    """
    
    def __init__(
        self,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        details["hint"] = "Set LIKELIHOOD_PARAMS_PATH environment variable"
        super().__init__(
            message="Detection artifacts not configured",
            reason="missing_artifacts",
            details=details,
        )

