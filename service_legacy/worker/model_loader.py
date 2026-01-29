"""
Model lifecycle management for GPU workers.

Provides:
- Startup model loading with progress tracking
- GPU memory management
- Model warmup for consistent latency
- Graceful shutdown with resource cleanup
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import torch
from PIL import Image

from service.infra.logging import get_logger
from service.detector_artifacts import DetectorArtifacts

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline

logger = get_logger(__name__)


@dataclass
class WorkerSettings:
    """
    Settings for GPU worker.
    
    Loaded from environment variables at startup.
    
    SAFE DEFAULTS:
    - GPU_SEMAPHORE_SIZE=1: Only one GPU operation at a time (prevents OOM)
    - MAX_CONCURRENT_REQUESTS=2: Conservative to prevent memory oversubscription
    - Scale horizontally by adding more worker replicas, not by increasing concurrency
    """
    
    # Model settings
    model_id: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_ID",
            "runwayml/stable-diffusion-v1-5"
        )
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "auto")
    )
    use_fp16: bool = field(
        default_factory=lambda: os.getenv("USE_FP16", "true").lower() == "true"
    )
    
    # Artifact paths
    likelihood_params_path: Optional[str] = field(
        default_factory=lambda: os.getenv("LIKELIHOOD_PARAMS_PATH")
    )
    mask_path: Optional[str] = field(
        default_factory=lambda: os.getenv("MASK_PATH")
    )
    
    # Concurrency settings - SAFE DEFAULTS
    # GPU memory oversubscription causes instability
    # Scale horizontally (more workers) not vertically (higher concurrency)
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))  # Safe default: 2
    )
    max_queue_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUEUE_SIZE", "4"))  # Safe default: 4
    )
    gpu_semaphore_size: int = field(
        default_factory=lambda: int(os.getenv("GPU_SEMAPHORE_SIZE", "1"))  # Safe default: 1
    )
    
    # Warmup settings
    enable_warmup: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WARMUP", "true").lower() == "true"
    )
    
    def __post_init__(self):
        """Resolve auto device after initialization and validate settings."""
        if self.device == "auto":
            self.device = self._detect_device()
        
        # Warn if unsafe concurrency settings are used
        if self.gpu_semaphore_size > 1:
            logger.warning(
                "unsafe_gpu_concurrency",
                extra={
                    "gpu_semaphore_size": self.gpu_semaphore_size,
                    "recommendation": "Use GPU_SEMAPHORE_SIZE=1 and scale horizontally",
                }
            )
        
        if self.max_concurrent_requests > 4:
            logger.warning(
                "high_concurrent_requests",
                extra={
                    "max_concurrent_requests": self.max_concurrent_requests,
                    "recommendation": "Use MAX_CONCURRENT_REQUESTS<=4 and scale horizontally",
                }
            )
    
    @staticmethod
    def _detect_device() -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


class ModelLoader:
    """
    Manages model lifecycle for GPU worker.
    
    Responsibilities:
    - Load Stable Diffusion pipeline at startup
    - Load detector artifacts (likelihood params, masks)
    - Warmup models for consistent latency
    - Track GPU memory usage
    - Atomic backpressure control (race-free)
    - Cleanup on shutdown
    
    BACKPRESSURE:
    - Uses asyncio.Queue with maxsize for atomic admission control
    - No manual counters (race conditions in async code)
    - Request acquisition is atomic via queue.put_nowait() / queue.get()
    """
    
    def __init__(self, settings: WorkerSettings):
        """
        Initialize model loader.
        
        Args:
            settings: Worker settings
        """
        self.settings = settings
        
        # Model state
        self.pipeline: Optional["StableDiffusionPipeline"] = None
        self.detector_artifacts: Optional[DetectorArtifacts] = None
        
        # Status tracking
        self.is_ready = False
        self.load_start_time: Optional[float] = None
        self.load_end_time: Optional[float] = None
        self.start_time = time.time()
        
        # Concurrency control - using proper async primitives
        # GPU semaphore: limits parallel GPU operations (default: 1)
        self.gpu_semaphore = asyncio.Semaphore(settings.gpu_semaphore_size)
        
        # Request queue: bounded queue for atomic admission control
        # Uses asyncio.Queue instead of manual counter to avoid race conditions
        self._request_slots: asyncio.Queue[None] = asyncio.Queue(
            maxsize=settings.max_queue_size
        )
        
        # Pre-fill the queue with slots (each slot represents capacity for one request)
        # This inverts the queue logic: we take slots to process, put back when done
        # Note: We can't pre-fill in __init__ because event loop may not be running
        self._slots_initialized = False
        
        logger.info(
            "model_loader_initialized",
            extra={
                "model_id": settings.model_id,
                "device": settings.device,
                "use_fp16": settings.use_fp16,
                "max_queue_size": settings.max_queue_size,
                "gpu_semaphore_size": settings.gpu_semaphore_size,
            }
        )
    
    async def _ensure_slots_initialized(self) -> None:
        """Initialize request slots (must be called from async context)."""
        if not self._slots_initialized:
            # Pre-fill with None values to represent available slots
            for _ in range(self.settings.max_queue_size):
                await self._request_slots.put(None)
            self._slots_initialized = True
            logger.debug(
                "request_slots_initialized",
                extra={"max_queue_size": self.settings.max_queue_size}
            )
    
    async def acquire_request_slot(self) -> bool:
        """
        Try to acquire a request slot (atomic, race-free).
        
        Returns:
            True if slot acquired, False if queue is full
        """
        await self._ensure_slots_initialized()
        try:
            # Non-blocking get - fails immediately if no slots available
            self._request_slots.get_nowait()
            return True
        except asyncio.QueueEmpty:
            return False
    
    async def release_request_slot(self) -> None:
        """Release a request slot back to the pool."""
        await self._request_slots.put(None)
    
    @property
    def queue_size(self) -> int:
        """
        Get current queue size (number of active requests).
        
        Note: This is an approximation due to async nature, but safe for monitoring.
        """
        if not self._slots_initialized:
            return 0
        # Active requests = max_queue_size - available_slots
        return self.settings.max_queue_size - self._request_slots.qsize()
    
    @property
    def is_queue_full(self) -> bool:
        """Check if the queue is full (cannot accept new requests)."""
        if not self._slots_initialized:
            return False
        return self._request_slots.empty()
    
    async def load_models(self) -> None:
        """
        Load all models at startup.
        
        Order:
        1. Load Stable Diffusion pipeline
        2. Move to GPU with appropriate dtype
        3. Load detector artifacts
        4. Warm up with dummy inference
        
        Raises:
            RuntimeError: If model loading fails
        """
        self.load_start_time = time.time()
        
        logger.info("model_loading_started")
        
        try:
            # Load Stable Diffusion pipeline
            await self._load_pipeline()
            
            # Load detector artifacts (if paths configured)
            await self._load_artifacts()
            
            # Warmup models
            if self.settings.enable_warmup:
                await self._warmup()
            
            self.load_end_time = time.time()
            self.is_ready = True
            
            logger.info(
                "model_loading_completed",
                extra={
                    "load_time_seconds": self.load_end_time - self.load_start_time,
                }
            )
        
        except Exception as e:
            logger.error(
                "model_loading_failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    async def _load_pipeline(self) -> None:
        """Load Stable Diffusion pipeline."""
        from diffusers import StableDiffusionPipeline
        
        logger.info(
            "pipeline_loading",
            extra={
                "model_id": self.settings.model_id,
                "device": self.settings.device,
            }
        )
        
        # Load in thread to avoid blocking
        def load_fn():
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.settings.model_id,
            )
            
            # Move to device with appropriate dtype
            if self.settings.device == "cuda" and self.settings.use_fp16:
                pipeline = pipeline.to(self.settings.device, dtype=torch.float16)
            else:
                pipeline = pipeline.to(self.settings.device, dtype=torch.float32)
            
            return pipeline
        
        self.pipeline = await asyncio.to_thread(load_fn)
        
        logger.info(
            "pipeline_loaded",
            extra={
                "model_id": self.settings.model_id,
                "device": str(self.pipeline.device),
            }
        )
    
    async def _load_artifacts(self) -> None:
        """Load detector artifacts."""
        if not self.settings.likelihood_params_path:
            logger.warning(
                "artifacts_skipped",
                extra={"reason": "LIKELIHOOD_PARAMS_PATH not set"}
            )
            return
        
        logger.info(
            "artifacts_loading",
            extra={
                "likelihood_params_path": self.settings.likelihood_params_path,
                "mask_path": self.settings.mask_path,
            }
        )
        
        # Default g_field_config for artifact loading
        default_g_field_config = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        def load_fn():
            return DetectorArtifacts(
                likelihood_params_path=self.settings.likelihood_params_path,
                mask_path=self.settings.mask_path,
                g_field_config=default_g_field_config,
            )
        
        self.detector_artifacts = await asyncio.to_thread(load_fn)
        
        logger.info(
            "artifacts_loaded",
            extra={
                "num_positions": self.detector_artifacts.num_positions,
            }
        )
    
    async def _warmup(self) -> None:
        """
        Warm up models with dummy inference.
        
        This ensures CUDA kernels are compiled and cached,
        leading to consistent latency on subsequent requests.
        """
        logger.info("warmup_started")
        
        if self.pipeline is None:
            logger.warning("warmup_skipped", extra={"reason": "pipeline not loaded"})
            return
        
        try:
            # Create dummy image for encode/decode warmup
            dummy_image = Image.new("RGB", (512, 512), color="white")
            
            def warmup_fn():
                # Encode image to latent space
                if hasattr(self.pipeline, 'vae'):
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ])
                    image_tensor = transform(dummy_image).unsqueeze(0)
                    image_tensor = image_tensor.to(
                        self.pipeline.device,
                        dtype=self.pipeline.vae.dtype,
                    )
                    
                    with torch.no_grad():
                        _ = self.pipeline.vae.encode(image_tensor)
                
                logger.info("warmup_encode_completed")
            
            await asyncio.to_thread(warmup_fn)
            
            logger.info("warmup_completed")
        
        except Exception as e:
            logger.warning(
                "warmup_failed",
                extra={"error": str(e)}
            )
    
    def get_gpu_info(self) -> dict:
        """
        Get GPU memory information.
        
        Returns:
            Dictionary with GPU memory stats
        """
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_used_pct": 0.0,
            }
        
        try:
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "gpu_available": True,
                "memory_used_mb": memory_used // (1024 * 1024),
                "memory_total_mb": memory_total // (1024 * 1024),
                "memory_used_pct": (memory_used / memory_total) * 100,
            }
        except Exception as e:
            logger.warning(
                "gpu_info_failed",
                extra={"error": str(e)}
            )
            return {
                "gpu_available": False,
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_used_pct": 0.0,
            }
    
    def get_uptime_seconds(self) -> float:
        """Get worker uptime in seconds."""
        return time.time() - self.start_time
    
    def get_load_time_seconds(self) -> Optional[float]:
        """Get model load time in seconds."""
        if self.load_start_time and self.load_end_time:
            return self.load_end_time - self.load_start_time
        return None
    
    async def cleanup(self) -> None:
        """
        Cleanup resources on shutdown.
        
        - Clear model references
        - Clear CUDA cache
        """
        logger.info("cleanup_started")
        
        self.is_ready = False
        self.pipeline = None
        self.detector_artifacts = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("cleanup_completed")

