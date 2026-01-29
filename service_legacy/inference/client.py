"""
Inference client implementations.

Provides:
- InferenceClient: Protocol defining the interface
- LocalInferenceClient: In-process inference using local GPU
- RemoteInferenceClient: HTTP-based remote inference to GPU workers

Architecture:
- LocalInferenceClient wraps existing detection/generation code for dev/testing
- RemoteInferenceClient communicates with GPU workers via HTTP/gRPC
- Circuit breaker pattern protects against cascading failures
- Operation-aware retry logic (detection=idempotent, generation=non-idempotent)

SECURITY INVARIANT:
- master_key is NEVER transmitted to workers
- Only derived_key (scoped, non-reversible) is sent
- key_fingerprint is used for cache keying and validation

RETRY SEMANTICS:
- Detection: Idempotent, retries allowed with backoff
- Generation: Non-idempotent, retry ONLY if request never reached worker
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, TYPE_CHECKING

from PIL import Image

from service.infra.logging import get_logger
from service.inference.schemas import (
    DetectInferenceRequest,
    DetectInferenceResponse,
    GenerateInferenceRequest,
    GenerateInferenceResponse,
    HealthStatus,
    HealthStatusEnum,
    OperationType,
)
from service.inference.exceptions import (
    InferenceError,
    InferenceTimeoutError,
    InferenceConnectionError,
    CircuitBreakerOpenError,
    WorkerOverloadedError,
)

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from service.authority import WatermarkAuthorityService

logger = get_logger(__name__)


class RetryDecision(str, Enum):
    """Decision on whether to retry a request."""
    RETRY = "retry"           # Safe to retry
    NO_RETRY = "no_retry"     # Do not retry (would corrupt correctness)
    ABORT = "abort"           # Abort immediately (unrecoverable)


# =============================================================================
# Inference Client Protocol
# =============================================================================


class InferenceClient(Protocol):
    """
    Protocol defining the interface for inference operations.
    
    Implementations:
    - LocalInferenceClient: In-process inference
    - RemoteInferenceClient: HTTP-based remote inference
    """
    
    async def detect(
        self,
        request: DetectInferenceRequest,
    ) -> DetectInferenceResponse:
        """
        Perform watermark detection.
        
        Args:
            request: Detection request with image and configuration
        
        Returns:
            Detection response with results
        
        Raises:
            InferenceError: If detection fails
            InferenceTimeoutError: If request times out
        """
        ...
    
    async def generate(
        self,
        request: GenerateInferenceRequest,
    ) -> GenerateInferenceResponse:
        """
        Generate watermarked image.
        
        Args:
            request: Generation request with prompt and configuration
        
        Returns:
            Generation response with image bytes
        
        Raises:
            InferenceError: If generation fails
            InferenceTimeoutError: If request times out
        """
        ...
    
    async def health_check(self) -> HealthStatus:
        """
        Check health of inference service.
        
        Returns:
            Health status of the service/workers
        """
        ...


# =============================================================================
# Circuit Breaker
# =============================================================================


@dataclass
class CircuitBreakerState:
    """
    Circuit breaker state machine.
    
    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: After N consecutive failures, reject requests for timeout period
    - HALF_OPEN: Allow one test request to check if service recovered
    """
    
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    
    # Configuration
    failure_threshold: int = 5
    timeout_seconds: float = 30.0


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Protects against cascading failures by temporarily rejecting
    requests when a worker is unhealthy.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            timeout_seconds: Time to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._states: dict[str, CircuitBreakerState] = {}
    
    def _get_state(self, worker_url: str) -> CircuitBreakerState:
        """Get or create state for a worker."""
        if worker_url not in self._states:
            self._states[worker_url] = CircuitBreakerState(
                failure_threshold=self.failure_threshold,
                timeout_seconds=self.timeout_seconds,
            )
        return self._states[worker_url]
    
    def can_execute(self, worker_url: str) -> bool:
        """
        Check if requests can be sent to this worker.
        
        Args:
            worker_url: Worker URL to check
        
        Returns:
            True if requests are allowed
        """
        state = self._get_state(worker_url)
        
        if state.state == "closed":
            return True
        
        if state.state == "open":
            # Check if timeout has elapsed
            if time.time() - state.last_failure_time >= self.timeout_seconds:
                state.state = "half_open"
                logger.info(
                    "circuit_breaker_half_open",
                    extra={"worker_url": worker_url}
                )
                return True
            return False
        
        # half_open: allow one request
        return True
    
    def record_success(self, worker_url: str) -> None:
        """Record successful request."""
        state = self._get_state(worker_url)
        
        if state.state == "half_open":
            # Successful test request, close the circuit
            state.state = "closed"
            state.failure_count = 0
            logger.info(
                "circuit_breaker_closed",
                extra={"worker_url": worker_url}
            )
        elif state.state == "closed":
            state.failure_count = 0
    
    def record_failure(self, worker_url: str) -> None:
        """Record failed request."""
        state = self._get_state(worker_url)
        state.failure_count += 1
        state.last_failure_time = time.time()
        
        if state.state == "half_open":
            # Failed test request, reopen the circuit
            state.state = "open"
            logger.warning(
                "circuit_breaker_reopened",
                extra={"worker_url": worker_url}
            )
        elif state.failure_count >= self.failure_threshold:
            state.state = "open"
            logger.warning(
                "circuit_breaker_opened",
                extra={
                    "worker_url": worker_url,
                    "failure_count": state.failure_count,
                }
            )


# =============================================================================
# Local Inference Client
# =============================================================================


class LocalInferenceClient:
    """
    In-process inference client using local GPU.
    
    Wraps existing detection and generation code for development and testing.
    This client performs inference in the same process as the API service.
    
    Usage:
        client = LocalInferenceClient(pipeline, authority)
        response = await client.detect(request)
    """
    
    def __init__(
        self,
        pipeline: "StableDiffusionPipeline",
        authority: "WatermarkAuthorityService",
    ):
        """
        Initialize local inference client.
        
        Args:
            pipeline: Stable Diffusion pipeline
            authority: Watermark authority service
        """
        self.pipeline = pipeline
        self.authority = authority
        self._start_time = time.time()
        
        logger.info("local_inference_client_initialized")
    
    async def detect(
        self,
        request: DetectInferenceRequest,
    ) -> DetectInferenceResponse:
        """
        Perform watermark detection locally.
        
        Args:
            request: Detection request
        
        Returns:
            Detection response
        """
        start_time = time.time()
        
        try:
            # Import detection service lazily to avoid circular imports
            from service.detection import DetectionService
            
            # Create detection service for this request
            detection_service = DetectionService(
                authority=self.authority,
                pipeline=self.pipeline,
            )
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(request.image_bytes))
            
            # Run detection
            result = await asyncio.to_thread(
                detection_service.detect,
                image=image,
                key_id=request.key_id,
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return DetectInferenceResponse(
                detected=result["detected"],
                score=result["score"],
                confidence=result["confidence"],
                log_odds=result["log_odds"],
                posterior=result["posterior"],
                request_id=request.request_id,
                processing_time_ms=processing_time_ms,
                g_field_config_hash=result.get("g_field_config_hash"),
            )
        
        except Exception as e:
            logger.error(
                "local_detection_failed",
                extra={
                    "request_id": request.request_id,
                    "error": str(e),
                }
            )
            raise InferenceError(
                f"Local detection failed: {e}",
                details={"request_id": request.request_id},
            ) from e
    
    async def generate(
        self,
        request: GenerateInferenceRequest,
    ) -> GenerateInferenceResponse:
        """
        Generate watermarked image locally.
        
        Args:
            request: Generation request
        
        Returns:
            Generation response with image
        """
        start_time = time.time()
        
        try:
            # Import generation adapter lazily
            from service.generation import StableDiffusionSeedBiasAdapter
            
            # Create adapter for this request
            adapter = StableDiffusionSeedBiasAdapter(
                model_id="runwayml/stable-diffusion-v1-5",
                device=None,  # Auto-detect
                use_fp16=True,
            )
            
            # Build watermark payload
            watermark_payload = {
                "key_id": request.key_id,
                "master_key": request.master_key,
                "embedding_config": request.embedding_config,
                "watermark_version": "local",  # Placeholder
            }
            
            # Run generation
            result = await asyncio.to_thread(
                adapter.generate,
                prompt=request.prompt,
                watermark_payload=watermark_payload,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                height=request.height,
                width=request.width,
            )
            
            # Convert PIL Image to bytes
            image = result["image"]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return GenerateInferenceResponse(
                image_bytes=image_bytes,
                seed_used=result["generation_metadata"]["seed"],
                request_id=request.request_id,
                processing_time_ms=processing_time_ms,
            )
        
        except Exception as e:
            logger.error(
                "local_generation_failed",
                extra={
                    "request_id": request.request_id,
                    "error": str(e),
                }
            )
            raise InferenceError(
                f"Local generation failed: {e}",
                details={"request_id": request.request_id},
            ) from e
    
    async def health_check(self) -> HealthStatus:
        """Check health of local inference."""
        try:
            import torch
            
            gpu_memory_used_mb = None
            gpu_memory_total_mb = None
            
            if torch.cuda.is_available():
                gpu_memory_used_mb = torch.cuda.memory_allocated() // (1024 * 1024)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            
            return HealthStatus(
                status=HealthStatusEnum.HEALTHY,
                model_loaded=self.pipeline is not None,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                uptime_seconds=time.time() - self._start_time,
            )
        
        except Exception as e:
            logger.warning(
                "local_health_check_failed",
                extra={"error": str(e)}
            )
            return HealthStatus(
                status=HealthStatusEnum.UNHEALTHY,
                model_loaded=False,
            )


# =============================================================================
# Remote Inference Client
# =============================================================================


class RemoteInferenceClient:
    """
    HTTP-based remote inference client for GPU workers.
    
    Features:
    - Connection pooling for efficiency
    - Retry logic with exponential backoff
    - Circuit breaker for fault tolerance
    - Load balancing across multiple workers
    - Request signing for authentication
    
    Usage:
        client = RemoteInferenceClient(
            worker_urls=["http://worker-1:8080", "http://worker-2:8080"],
            auth_secret="secret-key",
        )
        response = await client.detect(request)
    """
    
    def __init__(
        self,
        worker_urls: list[str],
        auth_secret: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ):
        """
        Initialize remote inference client.
        
        Args:
            worker_urls: List of GPU worker URLs
            auth_secret: Secret for request signing (optional)
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Time before circuit half-opens
        """
        if not worker_urls:
            raise ValueError("At least one worker URL is required")
        
        self.worker_urls = worker_urls
        self.auth_secret = auth_secret
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout_seconds=circuit_breaker_timeout,
        )
        
        # HTTP client (lazy initialization)
        self._client = None
        self._start_time = time.time()
        
        logger.info(
            "remote_inference_client_initialized",
            extra={
                "worker_count": len(worker_urls),
                "timeout_seconds": timeout_seconds,
            }
        )
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
            )
        return self._client
    
    def _select_worker(self) -> str:
        """
        Select a healthy worker for the request.
        
        Uses random selection among workers with closed circuit breakers.
        
        Returns:
            Worker URL
        
        Raises:
            CircuitBreakerOpenError: If all workers have open circuits
        """
        healthy_workers = [
            url for url in self.worker_urls
            if self.circuit_breaker.can_execute(url)
        ]
        
        if not healthy_workers:
            raise CircuitBreakerOpenError(
                "All workers have open circuit breakers",
                retry_after_seconds=self.circuit_breaker.timeout_seconds,
            )
        
        return random.choice(healthy_workers)
    
    def _sign_request(self, body: bytes) -> str:
        """
        Sign request body for authentication.
        
        Args:
            body: Request body bytes
        
        Returns:
            HMAC signature
        """
        if not self.auth_secret:
            return ""
        
        return hmac.new(
            self.auth_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
    
    def _should_retry(
        self,
        operation_type: OperationType,
        error: Exception,
        request_started: bool,
    ) -> RetryDecision:
        """
        Determine if a request should be retried based on operation type and error.
        
        CRITICAL: Generation is stochastic and non-idempotent.
        Retrying after partial execution can produce different results,
        corrupting watermark correctness.
        
        Args:
            operation_type: Type of operation (detection or generation)
            error: The error that occurred
            request_started: Whether the request reached the worker
        
        Returns:
            RetryDecision indicating whether to retry
        """
        # Detection is idempotent - safe to retry
        if operation_type == OperationType.DETECTION:
            return RetryDecision.RETRY
        
        # Generation is non-idempotent
        # Only retry if request never reached worker (transport failure)
        if operation_type == OperationType.GENERATION:
            if isinstance(error, InferenceConnectionError):
                # Connection failed - request never reached worker
                logger.info(
                    "generation_retry_allowed",
                    extra={
                        "reason": "connection_failed",
                        "error": str(error),
                    }
                )
                return RetryDecision.RETRY
            
            if request_started:
                # Request reached worker but failed - DO NOT RETRY
                logger.warning(
                    "generation_retry_forbidden",
                    extra={
                        "reason": "partial_execution_possible",
                        "error": str(error),
                    }
                )
                return RetryDecision.NO_RETRY
            
            # Transport-level failure before request sent - safe to retry
            return RetryDecision.RETRY
        
        # Unknown operation type - be conservative, don't retry
        return RetryDecision.NO_RETRY
    
    async def _send_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any],
        worker_url: str,
        operation_type: OperationType = OperationType.DETECTION,
    ) -> dict[str, Any]:
        """
        Send HTTP request to worker with operation-aware retry logic.
        
        RETRY SEMANTICS:
        - Detection (idempotent): Retries allowed with backoff
        - Generation (non-idempotent): Retry ONLY if request never reached worker
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body
            worker_url: Target worker URL
            operation_type: Type of operation for retry classification
        
        Returns:
            Response data
        
        Raises:
            InferenceError: On request failure
        """
        import httpx
        import json
        
        client = await self._get_client()
        url = f"{worker_url}{path}"
        body_bytes = json.dumps(body).encode()
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.auth_secret:
            headers["X-Signature"] = self._sign_request(body_bytes)
        
        last_error = None
        request_started = False
        
        for attempt in range(self.max_retries):
            try:
                request_started = False  # Reset for each attempt
                
                if method == "POST":
                    # Mark request as started once we begin sending
                    request_started = True
                    response = await client.post(
                        url,
                        content=body_bytes,
                        headers=headers,
                    )
                else:
                    request_started = True
                    response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    self.circuit_breaker.record_success(worker_url)
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 503:
                    last_error = WorkerOverloadedError(worker_url=worker_url)
                else:
                    last_error = InferenceError(
                        f"Worker returned status {response.status_code}",
                        worker_url=worker_url,
                        details={"status_code": response.status_code},
                    )
            
            except httpx.TimeoutException as e:
                # Timeout - request may have been partially processed
                request_started = True
                last_error = InferenceTimeoutError(
                    timeout_seconds=self.timeout_seconds,
                    worker_url=worker_url,
                )
            
            except httpx.ConnectError as e:
                # Connection failed - request never reached worker
                request_started = False
                last_error = InferenceConnectionError(
                    worker_url=worker_url,
                    original_error=e,
                )
            
            except Exception as e:
                last_error = InferenceError(
                    str(e),
                    worker_url=worker_url,
                )
            
            # Record failure
            self.circuit_breaker.record_failure(worker_url)
            
            # Check if retry is allowed based on operation type
            retry_decision = self._should_retry(operation_type, last_error, request_started)
            
            if retry_decision == RetryDecision.NO_RETRY:
                logger.error(
                    "retry_forbidden",
                    extra={
                        "operation_type": operation_type.value,
                        "attempt": attempt + 1,
                        "reason": "non_idempotent_operation",
                    }
                )
                raise last_error
            
            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff = min(2 ** attempt, 10) * (0.5 + random.random())
                logger.warning(
                    "inference_request_retry",
                    extra={
                        "operation_type": operation_type.value,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "backoff_seconds": round(backoff, 2),
                        "worker_url": worker_url,
                        "retry_decision": retry_decision.value,
                    }
                )
                await asyncio.sleep(backoff)
        
        raise last_error
    
    async def detect(
        self,
        request: DetectInferenceRequest,
    ) -> DetectInferenceResponse:
        """
        Send detection request to GPU worker.
        
        Detection is IDEMPOTENT - retries are safe with backoff.
        
        SECURITY: master_key is NEVER sent to worker.
        Only derived_key and key_fingerprint are transmitted.
        
        Args:
            request: Detection request (contains derived_key, not master_key)
        
        Returns:
            Detection response
        """
        worker_url = self._select_worker()
        
        # Prepare request body (convert bytes to base64 for JSON)
        # SECURITY: NEVER include master_key - only derived_key
        import base64
        body = {
            "image_base64": base64.b64encode(request.image_bytes).decode(),
            "key_id": request.key_id,
            "derived_key": request.derived_key,
            "key_fingerprint": request.key_fingerprint,
            "g_field_config": request.g_field_config,
            "detection_config": request.detection_config,
            "request_id": request.request_id,
            "operation_type": request.operation_type.value,
        }
        if request.inversion_config:
            body["inversion_config"] = request.inversion_config
        if request.idempotency_key:
            body["idempotency_key"] = request.idempotency_key
        if request.deterministic_seed is not None:
            body["deterministic_seed"] = request.deterministic_seed
        
        logger.debug(
            "remote_detect_request",
            extra={
                "request_id": request.request_id,
                "worker_url": worker_url,
                "key_fingerprint_prefix": request.key_fingerprint[:8] + "...",
            }
        )
        
        response_data = await self._send_request(
            method="POST",
            path="/v1/detect",
            body=body,
            worker_url=worker_url,
            operation_type=OperationType.DETECTION,  # Idempotent - retries safe
        )
        
        return DetectInferenceResponse(
            detected=response_data["detected"],
            score=response_data["score"],
            confidence=response_data["confidence"],
            log_odds=response_data["log_odds"],
            posterior=response_data["posterior"],
            request_id=response_data["request_id"],
            processing_time_ms=response_data["processing_time_ms"],
            key_fingerprint=response_data.get("key_fingerprint"),
            g_field_config_hash=response_data.get("g_field_config_hash"),
        )
    
    async def generate(
        self,
        request: GenerateInferenceRequest,
    ) -> GenerateInferenceResponse:
        """
        Send generation request to GPU worker.
        
        Generation is NON-IDEMPOTENT (stochastic).
        Retries are ONLY safe if request never reached worker.
        After partial execution, DO NOT retry.
        
        SECURITY: master_key is NEVER sent to worker.
        Only derived_key and key_fingerprint are transmitted.
        
        Args:
            request: Generation request (contains derived_key, not master_key)
        
        Returns:
            Generation response with image
        """
        worker_url = self._select_worker()
        
        # SECURITY: NEVER include master_key - only derived_key
        body = {
            "prompt": request.prompt,
            "derived_key": request.derived_key,
            "key_fingerprint": request.key_fingerprint,
            "key_id": request.key_id,
            "embedding_config": request.embedding_config,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "height": request.height,
            "width": request.width,
            "request_id": request.request_id,
            "operation_type": request.operation_type.value,
        }
        if request.idempotency_key:
            body["idempotency_key"] = request.idempotency_key
        if request.deterministic_seed is not None:
            body["deterministic_seed"] = request.deterministic_seed
        
        logger.debug(
            "remote_generate_request",
            extra={
                "request_id": request.request_id,
                "worker_url": worker_url,
                "key_fingerprint_prefix": request.key_fingerprint[:8] + "...",
            }
        )
        
        response_data = await self._send_request(
            method="POST",
            path="/v1/generate",
            body=body,
            worker_url=worker_url,
            operation_type=OperationType.GENERATION,  # Non-idempotent - careful retries
        )
        
        # Decode base64 image from response
        import base64
        image_bytes = base64.b64decode(response_data["image_base64"])
        
        return GenerateInferenceResponse(
            image_bytes=image_bytes,
            seed_used=response_data["seed_used"],
            request_id=response_data["request_id"],
            processing_time_ms=response_data["processing_time_ms"],
            key_fingerprint=response_data.get("key_fingerprint"),
        )
    
    async def health_check(self) -> HealthStatus:
        """
        Check health of remote workers.
        
        Aggregates health from all workers.
        """
        healthy_count = 0
        total_count = len(self.worker_urls)
        
        for worker_url in self.worker_urls:
            if self.circuit_breaker.can_execute(worker_url):
                healthy_count += 1
        
        if healthy_count == total_count:
            status = HealthStatusEnum.HEALTHY
        elif healthy_count > 0:
            status = HealthStatusEnum.DEGRADED
        else:
            status = HealthStatusEnum.UNHEALTHY
        
        return HealthStatus(
            status=status,
            model_loaded=healthy_count > 0,
            active_requests=None,  # Not available for remote client
            uptime_seconds=time.time() - self._start_time,
        )
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Factory Function
# =============================================================================


def create_inference_client(
    mode: str = "local",
    pipeline: Optional["StableDiffusionPipeline"] = None,
    authority: Optional["WatermarkAuthorityService"] = None,
    worker_urls: Optional[list[str]] = None,
    auth_secret: Optional[str] = None,
) -> InferenceClient:
    """
    Create an inference client based on mode.
    
    Args:
        mode: "local" for in-process inference, "remote" for GPU workers
        pipeline: Stable Diffusion pipeline (required for local mode)
        authority: Watermark authority service (required for local mode)
        worker_urls: GPU worker URLs (required for remote mode)
        auth_secret: Authentication secret for remote workers
    
    Returns:
        InferenceClient instance
    
    Raises:
        ValueError: If required arguments are missing
    """
    if mode == "local":
        if pipeline is None:
            raise ValueError("Pipeline is required for local inference mode")
        if authority is None:
            raise ValueError("Authority is required for local inference mode")
        return LocalInferenceClient(pipeline=pipeline, authority=authority)
    
    elif mode == "remote":
        if not worker_urls:
            raise ValueError("Worker URLs are required for remote inference mode")
        return RemoteInferenceClient(
            worker_urls=worker_urls,
            auth_secret=auth_secret,
        )
    
    else:
        raise ValueError(f"Unknown inference mode: {mode}. Use 'local' or 'remote'.")

