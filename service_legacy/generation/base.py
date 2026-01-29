"""
Base generation adapter interface.

This abstraction allows the API to be generation-method agnostic.
Phase-1 uses Stable Diffusion, but Phase-2 may support client-side generation
where the API only issues watermark credentials.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from PIL import Image


class GenerationAdapter(ABC):
    """
    Abstract adapter for image generation with watermarking.
    
    This interface decouples the API layer from specific generation methods.
    Phase-1: Hosted Stable Diffusion generation
    Phase-2: Client-side generation (API only issues credentials)
    
    The adapter is responsible for:
    - Accepting generation parameters (prompt, watermark payload, etc.)
    - Applying watermark embedding (seed-bias at z_T for Phase-1)
    - Generating the image
    - Returning image and generation metadata
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        watermark_payload: Dict[str, Any],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a watermarked image.
        
        Args:
            prompt: Text prompt for generation
            watermark_payload: Watermark configuration from WatermarkAuthorityService
                Contains: key_id, master_key, embedding_config, etc.
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed (optional)
            height: Image height (optional)
            width: Image width (optional)
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary containing:
                - image: PIL Image
                - generation_metadata: Dict with seed, steps, model_version, etc.
                - watermark_version: Version identifier for watermark policy
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying generation model.
        
        Returns:
            Dictionary with model identifier, version, etc.
        """
        pass

