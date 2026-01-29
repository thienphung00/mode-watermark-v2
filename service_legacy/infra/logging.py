"""
Structured logging configuration for the watermarking service.

Provides:
- JSON-formatted structured logs
- Request ID correlation
- Sensitive data sanitization
- Configurable log levels
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import structlog

# Context variable for request ID propagation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Sensitive keys that should never appear in logs
SENSITIVE_KEYS: Set[str] = frozenset({
    "master_key",
    "secret_key",
    "encryption_key",
    "password",
    "token",
    "api_key",
    "secret",
    "credential",
    "private_key",
    "secret_key_encrypted",
})


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in context.
    
    Args:
        request_id: Optional request ID. If None, generates a new one.
    
    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear request ID from context."""
    request_id_var.set(None)


def sanitize_log_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive values from log context.
    
    Args:
        context: Log context dictionary
    
    Returns:
        Sanitized context with sensitive values redacted
    """
    sanitized = {}
    for key, value in context.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_context(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                sanitize_log_context(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def add_request_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Structlog processor to add request ID to all log entries."""
    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def sanitize_sensitive_data(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Structlog processor to sanitize sensitive data from logs."""
    return sanitize_log_context(event_dict)


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Structlog processor to add ISO timestamp."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Structlog processor to add service metadata."""
    event_dict["service"] = "watermarking-service"
    event_dict["version"] = os.getenv("SERVICE_VERSION", "1.0.0")
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    use_structured_logging: bool = True,
) -> None:
    """
    Configure structured logging for the service.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON logs. If False, use console format.
        use_structured_logging: If True, use structlog. If False, use basic logging.
    """
    # Check feature flag
    if os.getenv("USE_STRUCTURED_LOGGING", "true").lower() == "false":
        use_structured_logging = False
    
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    
    if not use_structured_logging:
        # Fall back to basic logging
        logging.basicConfig(
            level=log_level_num,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
        )
        return
    
    # Structlog processors
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_timestamp,
        add_request_id,
        add_service_info,
        sanitize_sensitive_data,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        # JSON output for production
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_num,
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Structured logger bound to the given name
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for binding additional context to logs.
    
    Usage:
        with LogContext(user_id="123", action="generate"):
            logger.info("Processing request")
    """
    
    def __init__(self, **kwargs: Any):
        """Initialize with context values to bind."""
        self.context = kwargs
        self._token = None
    
    def __enter__(self) -> "LogContext":
        """Bind context values."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unbind context values."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


# Pre-configured logger for import convenience
logger = get_logger(__name__)
