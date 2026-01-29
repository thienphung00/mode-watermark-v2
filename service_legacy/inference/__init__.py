"""
Inference client abstraction for GPU worker communication.

Provides:
- InferenceClient: Protocol for inference operations
- LocalInferenceClient: In-process inference (dev/testing)
- RemoteInferenceClient: HTTP-based remote inference (production)

Usage:
    # Local inference (development)
    client = LocalInferenceClient(pipeline, authority)
    
    # Remote inference (production)
    client = RemoteInferenceClient(worker_urls=["http://worker-1:8080"])
    
    # Use in service
    response = await client.detect(request)
"""
from service.inference.client import (
    InferenceClient,
    LocalInferenceClient,
    RemoteInferenceClient,
    create_inference_client,
)
from service.inference.schemas import (
    DetectInferenceRequest,
    DetectInferenceResponse,
    GenerateInferenceRequest,
    GenerateInferenceResponse,
    HealthStatus,
)
from service.inference.exceptions import (
    InferenceError,
    InferenceTimeoutError,
    InferenceConnectionError,
    CircuitBreakerOpenError,
)

__all__ = [
    # Client classes
    "InferenceClient",
    "LocalInferenceClient",
    "RemoteInferenceClient",
    "create_inference_client",
    # Request/Response schemas
    "DetectInferenceRequest",
    "DetectInferenceResponse",
    "GenerateInferenceRequest",
    "GenerateInferenceResponse",
    "HealthStatus",
    # Exceptions
    "InferenceError",
    "InferenceTimeoutError",
    "InferenceConnectionError",
    "CircuitBreakerOpenError",
]

