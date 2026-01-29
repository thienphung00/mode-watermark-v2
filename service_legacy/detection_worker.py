"""
Detection Worker for Micro-Batching.

This module provides a lightweight in-process worker that batches detection requests
to improve GPU utilization. The worker collects requests for 10-30ms, batches tensors,
and processes them together.

Single-process, single-GPU implementation using asyncio and standard Python tools.
"""
from __future__ import annotations

import asyncio
from service.infra.logging import get_logger
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = get_logger(__name__)


@dataclass
class DetectionRequest:
    """Single detection request item."""
    image: Image.Image
    key_id: str
    future: asyncio.Future[Dict[str, Any]]
    timestamp: float


class DetectionWorker:
    """
    In-process detection worker for micro-batching.
    
    Collects detection requests for a short time window (10-30ms), batches them,
    and processes them together to improve GPU utilization.
    
    The worker runs in a background asyncio task and processes batches continuously.
    """
    
    def __init__(
        self,
        detection_service: Any,  # DetectionService
        batch_window_ms: float = 20.0,
        max_batch_size: int = 8,
    ):
        """
        Initialize detection worker.
        
        Args:
            detection_service: DetectionService instance to use for detection
            batch_window_ms: Time window in milliseconds to collect requests (default: 20ms)
            max_batch_size: Maximum batch size before forcing processing (default: 8)
        """
        self.detection_service = detection_service
        self.batch_window_ms = batch_window_ms / 1000.0  # Convert to seconds
        self.max_batch_size = max_batch_size
        
        # Request queue
        self.request_queue: asyncio.Queue[Optional[DetectionRequest]] = asyncio.Queue()
        
        # Worker task
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the worker background task."""
        if self._running:
            logger.warning("Worker already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Detection worker started: batch_window={self.batch_window_ms*1000:.1f}ms, max_batch={self.max_batch_size}")
    
    async def stop(self) -> None:
        """Stop the worker background task."""
        if not self._running:
            return
        
        self._running = False
        # Send sentinel to wake up worker
        await self.request_queue.put(None)
        
        if self._worker_task:
            await self._worker_task
        
        logger.info("Detection worker stopped")
    
    async def detect(self, image: Image.Image, key_id: str) -> Dict[str, Any]:
        """
        Submit a detection request and wait for result.
        
        Args:
            image: Input image
            key_id: Public key identifier
        
        Returns:
            Detection result dictionary
        """
        future: asyncio.Future[Dict[str, Any]] = asyncio.Future()
        request = DetectionRequest(
            image=image,
            key_id=key_id,
            future=future,
            timestamp=time.time(),
        )
        
        await self.request_queue.put(request)
        
        # Wait for result
        return await future
    
    async def _worker_loop(self) -> None:
        """Main worker loop that processes batches."""
        while self._running:
            try:
                # Collect requests for batch_window_ms or until max_batch_size
                batch: List[DetectionRequest] = []
                batch_start_time = time.time()
                
                # Wait for first request
                first_request = await self.request_queue.get()
                if first_request is None:  # Sentinel
                    break
                
                batch.append(first_request)
                
                # Collect more requests until window expires or max size reached
                while len(batch) < self.max_batch_size:
                    elapsed = time.time() - batch_start_time
                    if elapsed >= self.batch_window_ms:
                        break
                    
                    try:
                        # Wait for next request with timeout
                        remaining_time = self.batch_window_ms - elapsed
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining_time,
                        )
                        
                        if request is None:  # Sentinel
                            break
                        
                        batch.append(request)
                    except asyncio.TimeoutError:
                        # Time window expired
                        break
                
                # Process batch
                if batch:
                    await self._process_batch(batch)
            
            except Exception as e:
                logger.error(f"Error in detection worker loop: {e}", exc_info=True)
                # Fail all pending requests in current batch
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(e)
    
    async def _process_batch(self, batch: List[DetectionRequest]) -> None:
        """
        Process a batch of detection requests.
        
        Processes requests in parallel using asyncio.gather to improve throughput.
        Each request uses the preloaded pipeline and cached detectors.
        
        Args:
            batch: List of detection requests to process
        """
        logger.debug(f"Processing detection batch: size={len(batch)}")
        
        # Process all requests in parallel using thread pool executor
        # This allows GPU operations to run concurrently while maintaining
        # thread safety (CPython GIL is released during I/O and GPU operations)
        loop = asyncio.get_event_loop()
        
        async def process_request(req: DetectionRequest) -> None:
            """Process a single detection request."""
            try:
                result = await loop.run_in_executor(
                    None,
                    self.detection_service.detect,
                    req.image,
                    req.key_id,
                )
                if not req.future.done():
                    req.future.set_result(result)
            except Exception as e:
                logger.error(f"Error processing detection request: {e}", exc_info=True)
                if not req.future.done():
                    req.future.set_exception(e)
        
        # Process all requests concurrently
        await asyncio.gather(*[process_request(req) for req in batch])

