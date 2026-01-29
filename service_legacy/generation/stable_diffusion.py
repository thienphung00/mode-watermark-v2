"""
Stable Diffusion adapter for Phase-1 hosted generation.

This adapter wraps Stable Diffusion pipeline and seed-bias watermarking.
It isolates SD-specific assumptions from the API layer.
"""
from __future__ import annotations

from service.infra.logging import get_logger
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import (
    DiffusionConfig,
    SeedBiasConfig,
    WatermarkedConfig,
    KeySettings,
    AlgorithmParams,
    GFieldConfig,
    PRFConfig,
)
from src.detection.g_values import compute_g_values
from src.engine.pipeline import create_pipeline, generate_with_watermark
from src.engine.strategies.seed_bias import SeedBiasStrategy

from .base import GenerationAdapter

logger = get_logger(__name__)


def detect_device() -> str:
    """
    Detect available device with priority: mps > cuda > cpu.
    
    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class StableDiffusionSeedBiasAdapter(GenerationAdapter):
    """
    Stable Diffusion adapter with seed-bias watermarking.
    
    Phase-1 implementation: Hosted generation using Stable Diffusion.
    Applies seed-bias watermarking at z_T (initial latent).
    
    TODO (Phase-2): This adapter can be removed when moving to client-side generation.
    The API will only issue watermark credentials, not perform generation.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize Stable Diffusion adapter.
        
        Args:
            model_id: Hugging Face model identifier
            device: Device to use (default: auto-detect with priority: mps > cuda > cpu)
            use_fp16: Use FP16 precision (default: True, only used for CUDA)
        """
        if device is None:
            device = detect_device()
        
        logger.info(f"Using device: {device}")
        
        self.model_id = model_id
        self.device = device
        # Only use FP16 for CUDA; MPS and CPU use FP32
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Create diffusion config
        self.diffusion_config = DiffusionConfig(
            model_id=model_id,
            use_fp16=use_fp16,
            guidance_scale=1.0,
            inference_timesteps=50,
            trained_timesteps=1000,
        )
        
        # Pipeline is preloaded at startup via dependencies.get_pipeline()
        # We access it through the dependency injection system to ensure
        # we use the same singleton instance
        self._pipeline: Optional[Any] = None
    
    @property
    def pipeline(self):
        """
        Get preloaded Stable Diffusion pipeline.
        
        The pipeline is preloaded at FastAPI startup and accessed via
        dependencies.get_pipeline() to ensure we use the singleton instance.
        This eliminates per-request pipeline creation and ensures consistency.
        """
        if self._pipeline is None:
            # Import here to avoid circular dependency
            from service.app.dependencies import get_pipeline
            self._pipeline = get_pipeline(model_id=self.model_id, device=self.device)
            logger.debug(f"Retrieved preloaded pipeline for adapter: {self.model_id}")
        return self._pipeline
    
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
        Generate watermarked image using Stable Diffusion with seed-bias.
        
        Args:
            prompt: Text prompt
            watermark_payload: Watermark configuration from WatermarkAuthorityService
                Expected keys:
                    - key_id: Public key identifier
                    - master_key: Master key for PRF
                    - embedding_config: Seed-bias configuration
                    - watermark_version: Policy version
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            seed: Random seed (optional)
            height: Image height (optional)
            width: Image width (optional)
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with:
                - image: PIL Image
                - generation_metadata: Dict with seed, steps, model_version, etc.
                - watermark_version: Policy version
        """
        logger.info(f"Adapter received prompt: '{prompt}'")
        
        # Extract watermark configuration
        key_id = watermark_payload["key_id"]
        master_key = watermark_payload["master_key"]
        embedding_config = watermark_payload.get("embedding_config", {})
        watermark_version = watermark_payload.get("watermark_version", "1.0")
        
        # Create seed-bias strategy
        # ⚠️ GENERATION LAYER ⚠️
        # SeedBiasConfig controls watermark embedding during generation.
        # It does NOT configure detection algorithms.
        # Detection is configured separately in DetectionService (detector_type="bayesian").
        # Default config values (can be overridden by embedding_config)
        seed_bias_config = SeedBiasConfig(
            lambda_strength=embedding_config.get("lambda_strength", 0.05),
            domain=embedding_config.get("domain", "frequency"),
            low_freq_cutoff=embedding_config.get("low_freq_cutoff", 0.05),
            high_freq_cutoff=embedding_config.get("high_freq_cutoff", 0.4),
        )
        
        # Determine latent shape from pipeline
        pipeline = self.pipeline
        if height is None:
            height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        if width is None:
            width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        
        latent_height = height // pipeline.vae_scale_factor
        latent_width = width // pipeline.vae_scale_factor
        latent_channels = pipeline.unet.config.in_channels
        latent_shape = (latent_channels, latent_height, latent_width)
        
        # Create strategy
        strategy = SeedBiasStrategy(
            config=seed_bias_config,
            master_key=master_key,
            latent_shape=latent_shape,
            device=self.device,
        )
        
        # Prepare strategy for sample
        # VERIFICATION: key_id is used for deterministic G-field generation (watermark pattern)
        # seed is used for random epsilon (image variation)
        # Same key_id = same watermark (g-values), different seeds = different images
        strategy.prepare_for_sample(
            sample_id=key_id,  # Use key_id as sample_id
            prompt=prompt,
            seed=seed,
            key_id=key_id,  # key_id determines G-field (watermark), not seed
        )
        
        # Generate image
        logger.info(f"Pipeline being invoked with prompt: '{prompt}'")
        result = generate_with_watermark(
            pipeline=pipeline,
            strategy=strategy,
            prompt=prompt,
            sample_id=key_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
            **kwargs,
        )
        logger.info(f"Pipeline completed generation for prompt: '{prompt}'")
        
        # CRITICAL: Compute g-values from generated z_T using canonical function.
        # This ensures generation uses the same watermark statistic as detection.
        # Without this, the implicit statistic embedded during generation might not
        # match the explicit statistic evaluated during detection, breaking Bayesian validity.
        z_T = result.get("initial_latents")
        if z_T is not None:
            # Construct g_field_config from embedding_config (must match detection config)
            # This ensures the same G-field parameters are used for generation and detection
            g_field_config = {
                "mapping_mode": "binary",  # Seed bias uses binary mapping
                "domain": embedding_config.get("domain", "frequency"),
                "frequency_mode": "bandpass",  # Seed bias uses bandpass
                "low_freq_cutoff": embedding_config.get("low_freq_cutoff", 0.05),
                "high_freq_cutoff": embedding_config.get("high_freq_cutoff", 0.4),
                "normalize_zero_mean": True,
                "normalize_unit_variance": True,
            }
            
            # Compute g-values using canonical function
            # This validates that the watermark statistic matches what detection will compute
            g_values, mask = compute_g_values(
                z_T,
                key_id,
                master_key,
                return_mask=True,
                g_field_config=g_field_config,
                latent_type="zT",  # z_T is initial latent
            )
            
            # Log g-value statistics for validation (optional, for debugging)
            if g_values is not None:
                g_mean = float(g_values.mean().item())
                logger.debug(
                    f"Generated z_T g-value statistics: mean={g_mean:.4f}, "
                    f"shape={list(g_values.shape)}"
                )
        
        # Build generation metadata
        generation_metadata = {
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "model_id": self.model_id,
            "model_version": "stable-diffusion-v1-5",  # TODO: Extract from pipeline
            "height": height,
            "width": width,
            "zT_hash": result.get("zT_hash"),
        }
        
        return {
            "image": result["image"],
            "generation_metadata": generation_metadata,
            "watermark_version": watermark_version,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Stable Diffusion model information."""
        return {
            "model_id": self.model_id,
            "model_type": "stable-diffusion",
            "device": self.device,
            "use_fp16": self.use_fp16,
        }

