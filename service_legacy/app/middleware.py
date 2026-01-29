"""
Middleware for rate limiting, security, and observability.

This module provides:
- RateLimitMiddleware: In-memory rate limiting (use Redis in production)
- RequestIDMiddleware: Request correlation ID injection

Middleware order (first to last):
1. RequestIDMiddleware - Injects X-Request-ID for correlation
2. RateLimitMiddleware - Rate limiting per IP
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Dict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from service.infra.logging import (
    get_logger,
    get_request_id,
    set_request_id,
    clear_request_id,
)

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects a unique request ID into each request.
    
    The request ID is:
    - Generated if not provided in X-Request-ID header
    - Propagated to all log entries via context variable
    - Returned in X-Request-ID response header
    
    This enables correlation of all log entries within a single request,
    which is critical for debugging and observability.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with request ID injection.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
        
        Returns:
            Response with X-Request-ID header
        """
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store request ID in context for logging
        set_request_id(request_id)
        
        # Store request ID in request state for access in routes
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            "request_started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            }
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request completion
            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise
        
        finally:
            # Clear request ID from context
            clear_request_id()


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    In production, use Redis or similar for distributed rate limiting.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier (IP address)
        
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True


# Global rate limiters
_detect_rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
_evaluate_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Applies rate limiting to /detect endpoint.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Apply rate limiting to /detect endpoint
        if request.url.path.startswith("/api/v1/detect"):
            if not _detect_rate_limiter.is_allowed(client_ip):
                logger.warning(
                    "rate_limit_exceeded",
                    extra={
                        "client_ip": client_ip,
                        "endpoint": "detect",
                        "limit": _detect_rate_limiter.max_requests,
                        "window_seconds": _detect_rate_limiter.window_seconds,
                    }
                )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        # Apply rate limiting to /evaluate endpoint (evaluation-only, stricter limits)
        if request.url.path.startswith("/api/v1/evaluate"):
            if not _evaluate_rate_limiter.is_allowed(client_ip):
                logger.warning(
                    "rate_limit_exceeded",
                    extra={
                        "client_ip": client_ip,
                        "endpoint": "evaluate",
                        "limit": _evaluate_rate_limiter.max_requests,
                        "window_seconds": _evaluate_rate_limiter.window_seconds,
                    }
                )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        response = await call_next(request)
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Request timeout enforcement middleware.
    
    Configurable timeout for long-running requests.
    Default: 30 seconds for most endpoints, 120 seconds for generation.
    """
    
    def __init__(self, app, default_timeout: float = 30.0):
        """
        Initialize timeout middleware.
        
        Args:
            app: ASGI application
            default_timeout: Default timeout in seconds
        """
        super().__init__(app)
        self.default_timeout = default_timeout
        # Longer timeouts for compute-heavy endpoints
        self.endpoint_timeouts = {
            "/api/v1/generate": 120.0,
            "/api/v1/detect": 60.0,
            "/api/v1/evaluate": 120.0,
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply timeout to request."""
        import asyncio
        
        # Determine timeout for this endpoint
        path = request.url.path
        timeout = self.default_timeout
        for endpoint, endpoint_timeout in self.endpoint_timeouts.items():
            if path.startswith(endpoint):
                timeout = endpoint_timeout
                break
        
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout,
            )
            return response
        except asyncio.TimeoutError:
            logger.error(
                "request_timeout",
                extra={
                    "path": path,
                    "timeout_seconds": timeout,
                }
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request timeout after {timeout} seconds"
            )

