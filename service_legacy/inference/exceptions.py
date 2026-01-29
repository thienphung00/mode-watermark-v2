"""
Exceptions for inference client operations.

Provides specialized exceptions for:
- General inference failures
- Network/connection issues
- Timeout errors
- Circuit breaker state
"""
from __future__ import annotations

from typing import Any, Optional


class InferenceError(Exception):
    """
    Base exception for inference operations.
    
    Raised when inference fails for any reason.
    """
    
    def __init__(
        self,
        message: str,
        worker_url: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize inference error.
        
        Args:
            message: Human-readable error message
            worker_url: URL of the worker that failed (if applicable)
            details: Additional error details
        """
        self.message = message
        self.worker_url = worker_url
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            "error": "inference_error",
            "message": self.message,
        }
        if self.worker_url:
            result["worker_url"] = self.worker_url
        if self.details:
            result["details"] = self.details
        return result


class InferenceTimeoutError(InferenceError):
    """
    Inference operation timed out.
    
    Raised when a worker doesn't respond within the timeout period.
    """
    
    def __init__(
        self,
        message: str = "Inference request timed out",
        timeout_seconds: Optional[float] = None,
        worker_url: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Human-readable error message
            timeout_seconds: Timeout duration in seconds
            worker_url: URL of the worker that timed out
            details: Additional error details
        """
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, worker_url=worker_url, details=details)
        self.timeout_seconds = timeout_seconds
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = super().to_dict()
        result["error"] = "inference_timeout"
        return result


class InferenceConnectionError(InferenceError):
    """
    Failed to connect to inference worker.
    
    Raised when the worker is unreachable or connection fails.
    """
    
    def __init__(
        self,
        message: str = "Failed to connect to inference worker",
        worker_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize connection error.
        
        Args:
            message: Human-readable error message
            worker_url: URL of the unreachable worker
            original_error: Original exception that caused this error
            details: Additional error details
        """
        details = details or {}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(message, worker_url=worker_url, details=details)
        self.original_error = original_error
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = super().to_dict()
        result["error"] = "inference_connection_error"
        return result


class CircuitBreakerOpenError(InferenceError):
    """
    Circuit breaker is open, rejecting requests.
    
    Raised when too many consecutive failures have occurred and
    the circuit breaker has tripped to prevent cascading failures.
    """
    
    def __init__(
        self,
        message: str = "Circuit breaker is open, service temporarily unavailable",
        worker_url: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize circuit breaker error.
        
        Args:
            message: Human-readable error message
            worker_url: URL of the worker with open circuit breaker
            retry_after_seconds: Seconds until circuit breaker may close
            details: Additional error details
        """
        details = details or {}
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, worker_url=worker_url, details=details)
        self.retry_after_seconds = retry_after_seconds
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = super().to_dict()
        result["error"] = "circuit_breaker_open"
        return result


class WorkerOverloadedError(InferenceError):
    """
    Worker is overloaded and cannot accept more requests.
    
    Raised when backpressure is applied due to queue limits.
    """
    
    def __init__(
        self,
        message: str = "Worker is overloaded, try again later",
        worker_url: Optional[str] = None,
        queue_size: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize overload error.
        
        Args:
            message: Human-readable error message
            worker_url: URL of the overloaded worker
            queue_size: Current queue size
            max_queue_size: Maximum allowed queue size
            details: Additional error details
        """
        details = details or {}
        if queue_size is not None:
            details["queue_size"] = queue_size
        if max_queue_size is not None:
            details["max_queue_size"] = max_queue_size
        super().__init__(message, worker_url=worker_url, details=details)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = super().to_dict()
        result["error"] = "worker_overloaded"
        return result

