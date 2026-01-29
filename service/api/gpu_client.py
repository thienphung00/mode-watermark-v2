"""
HTTP client for GPU worker communication.

Handles all communication between API and GPU worker:
- Image generation requests
- Detection (DDIM inversion) requests
- Health checks
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from service.api.config import get_config
from service.api.schemas import (
    GPUGenerateRequest,
    GPUGenerateResponse,
    GPUDetectRequest,
    GPUDetectResponse,
    GPUHealthResponse,
)

logger = logging.getLogger(__name__)


class GPUClientError(Exception):
    """Error communicating with GPU worker."""
    pass


class GPUClientTimeoutError(GPUClientError):
    """Timeout communicating with GPU worker."""
    pass


class GPUClientConnectionError(GPUClientError):
    """Connection error with GPU worker."""
    pass


class GPUClient:
    """
    Async HTTP client for GPU worker.
    
    Handles:
    - POST /infer/generate - Image generation
    - POST /infer/reverse_ddim - DDIM inversion for detection
    - GET /health - Health check
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize GPU client.
        
        Args:
            base_url: GPU worker URL (default: from config)
            timeout: Request timeout in seconds (default: from config)
        """
        config = get_config()
        self.base_url = base_url or config.gpu_worker_url
        self.timeout = timeout or config.gpu_worker_timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        key_id: str,
        derived_key: str,
        key_fingerprint: str,
        prompt: str,
        request_id: str,
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> GPUGenerateResponse:
        """
        Request image generation from GPU worker.
        
        Args:
            key_id: Key identifier
            derived_key: Scoped derived key (NOT master key)
            key_fingerprint: Key fingerprint for validation
            prompt: Text prompt
            request_id: Request ID for tracing
            seed: Optional random seed
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            width: Image width
            height: Image height
            embedding_config: Watermark embedding configuration
            
        Returns:
            GPUGenerateResponse with image data
            
        Raises:
            GPUClientError: On communication failure
        """
        request = GPUGenerateRequest(
            key_id=key_id,
            derived_key=derived_key,
            key_fingerprint=key_fingerprint,
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            embedding_config=embedding_config or {},
            request_id=request_id,
        )
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/infer/generate",
                json=request.model_dump(),
            )
            response.raise_for_status()
            
            data = response.json()
            return GPUGenerateResponse(**data)
            
        except httpx.TimeoutException as e:
            logger.error(f"GPU generate timeout: {e}")
            raise GPUClientTimeoutError(f"Generation timed out: {e}")
        except httpx.ConnectError as e:
            logger.error(f"GPU connect error: {e}")
            raise GPUClientConnectionError(f"Cannot connect to GPU worker: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"GPU HTTP error: {e.response.status_code}")
            raise GPUClientError(f"GPU worker error: {e.response.text}")
        except Exception as e:
            logger.error(f"GPU generate failed: {e}")
            raise GPUClientError(f"Generation failed: {e}")
    
    async def detect(
        self,
        key_id: str,
        derived_key: str,
        master_key: str,
        key_fingerprint: str,
        image_base64: str,
        request_id: str,
        g_field_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None,
        inversion_config: Optional[Dict[str, Any]] = None,
    ) -> GPUDetectResponse:
        """
        Request detection (DDIM inversion) from GPU worker.
        
        Args:
            key_id: Key identifier
            derived_key: Scoped derived key (for backward compat)
            master_key: Master key (required for compute_g_values)
            key_fingerprint: Key fingerprint for validation
            image_base64: Base64-encoded image
            request_id: Request ID for tracing
            g_field_config: G-field configuration
            detection_config: Detection parameters
            inversion_config: DDIM inversion parameters
            
        Returns:
            GPUDetectResponse with detection results
            
        Raises:
            GPUClientError: On communication failure
        """
        request = GPUDetectRequest(
            key_id=key_id,
            derived_key=derived_key,
            master_key=master_key,
            key_fingerprint=key_fingerprint,
            image_base64=image_base64,
            g_field_config=g_field_config or {},
            detection_config=detection_config or {},
            inversion_config=inversion_config or {},
            request_id=request_id,
        )
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/infer/reverse_ddim",
                json=request.model_dump(),
            )
            response.raise_for_status()
            
            data = response.json()
            return GPUDetectResponse(**data)
            
        except httpx.TimeoutException as e:
            logger.error(f"GPU detect timeout: {e}")
            raise GPUClientTimeoutError(f"Detection timed out: {e}")
        except httpx.ConnectError as e:
            logger.error(f"GPU connect error: {e}")
            raise GPUClientConnectionError(f"Cannot connect to GPU worker: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"GPU HTTP error: {e.response.status_code}")
            raise GPUClientError(f"GPU worker error: {e.response.text}")
        except Exception as e:
            logger.error(f"GPU detect failed: {e}")
            raise GPUClientError(f"Detection failed: {e}")
    
    async def health(self) -> GPUHealthResponse:
        """
        Check GPU worker health.
        
        Returns:
            GPUHealthResponse with worker status
            
        Raises:
            GPUClientError: On communication failure
        """
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            
            data = response.json()
            return GPUHealthResponse(**data)
            
        except httpx.ConnectError:
            return GPUHealthResponse(
                status="unavailable",
                model_loaded=False,
                active_requests=0,
            )
        except Exception as e:
            logger.warning(f"GPU health check failed: {e}")
            return GPUHealthResponse(
                status="error",
                model_loaded=False,
                active_requests=0,
            )
    
    async def is_connected(self) -> bool:
        """Check if GPU worker is reachable."""
        try:
            health = await self.health()
            return health.status not in ("unavailable", "error")
        except Exception:
            return False


# Global GPU client instance
_gpu_client: Optional[GPUClient] = None


def get_gpu_client() -> GPUClient:
    """Get the global GPU client instance."""
    global _gpu_client
    if _gpu_client is None:
        _gpu_client = GPUClient()
    return _gpu_client


async def close_gpu_client() -> None:
    """Close the global GPU client."""
    global _gpu_client
    if _gpu_client is not None:
        await _gpu_client.close()
        _gpu_client = None
