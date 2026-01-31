"""
DDIM Inversion for recovering latent representations from images.

Performs deterministic inversion of the diffusion process to recover
the initial latent tensor z_T from a generated image. This is the
first step in SynthID-style watermark detection.

Pipeline:
    Image → VAE Encode → z_0 → DDIM Invert → z_T

The recovered z_T can then be analyzed for watermark presence by
computing its correlation with the expected G-field.

Note:
    DDIM inversion is approximate. The reconstruction quality depends on:
    - Number of inversion steps (more = better but slower)
    - Guidance scale during original generation
    - Image quality and any post-processing
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DDIMInverter:
    """
    DDIM Inversion to recover latent z_T from an image.
    
    Uses the VAE encoder and DDIM scheduler to approximately invert
    the diffusion process. The recovered z_T should correlate with
    the watermark G-field if the image was watermarked.
    
    Example:
        >>> inverter = DDIMInverter(pipeline)
        >>> z_T = inverter.invert(image, num_steps=50)
        >>> print(z_T.shape)  # [1, 4, 64, 64]
    """
    
    def __init__(
        self,
        pipeline: Any,  # StableDiffusionPipeline
        device: Optional[str] = None,
    ):
        """
        Initialize DDIM inverter.
        
        Args:
            pipeline: Stable Diffusion pipeline with VAE and UNet
            device: Device for computation (default: pipeline device)
        """
        self.pipeline = pipeline
        self.device = device or str(pipeline.device)
        
        # Extract components
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        
        # VAE scaling factor (typically 0.18215 for SD 1.x)
        self.vae_scale_factor = getattr(pipeline, 'vae_scale_factor', 8)
        self.vae_scaling_factor = 0.18215  # Latent scaling
        
    @torch.no_grad()
    def invert(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        num_inference_steps: int = 50,
        prompt: str = "",
        guidance_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Invert an image to recover the initial latent z_T.
        
        Performs DDIM inversion: z_0 → z_1 → ... → z_T
        
        Args:
            image: Input image (PIL, tensor, or numpy array)
            num_inference_steps: Number of inversion steps
            prompt: Text prompt for guided inversion (optional)
            guidance_scale: Guidance scale (1.0 = unconditional)
            return_intermediates: If True, also return intermediate latents
            
        Returns:
            z_T: Recovered initial latent tensor [1, 4, H/8, W/8]
            intermediates: (optional) List of intermediate latents
        """
        # Step 1: Encode image to latent z_0
        z_0 = self.encode_image(image)
        
        # Step 2: Set up scheduler timesteps for inversion
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps.flip(0)  # Reverse for inversion
        
        # Step 3: Encode prompt if provided
        if prompt and guidance_scale > 1.0:
            prompt_embeds = self._encode_prompt(prompt)
        else:
            # Unconditional embedding
            prompt_embeds = self._encode_prompt("")
        
        # Step 4: DDIM inversion loop
        latent = z_0.clone()
        intermediates = [latent.clone()] if return_intermediates else []
        
        for i, t in enumerate(timesteps[:-1]):  # Don't process last timestep
            t_next = timesteps[i + 1]
            
            # Predict noise
            noise_pred = self._predict_noise(latent, t, prompt_embeds, guidance_scale)
            
            # DDIM inversion step: z_t → z_{t+1}
            latent = self._ddim_inversion_step(latent, noise_pred, t, t_next)
            
            if return_intermediates:
                intermediates.append(latent.clone())
        
        z_T = latent
        
        if return_intermediates:
            return z_T, intermediates
        return z_T
    
    @torch.no_grad()
    def perform_full_inversion(
        self,
        z_0: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        prompt: str = "",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform full DDIM inversion from z_0 to recover the initial latent z_T.
        
        This is the exact point where SeedBias watermark was embedded.
        Uses the FULL inference timeline (all steps) to ensure complete recovery
        of the initial noise state.
        
        CRITICAL: This must use the SAME parameters as generation:
        - Same num_inference_steps (scheduler timesteps)
        - Same guidance_scale
        - Same prompt (or empty for unconditional)
        - Same scheduler type and settings
        
        This is NOT partial inversion - it processes ALL inference steps.
        
        Args:
            z_0: VAE-encoded latent tensor [1, C, H, W]
            num_inference_steps: Number of inversion steps. MUST match generation.
                If None, raises an error (inversion requires explicit step count).
            prompt: Text prompt for guided inversion. MUST match generation.
            guidance_scale: Guidance scale. MUST match generation.
            
        Returns:
            z_T: Recovered initial latent tensor [1, C, H, W]
            
        Raises:
            ValueError: If num_inference_steps is None or guidance_scale != 1.0
        """
        # CRITICAL: Require explicit num_inference_steps for mathematical correctness
        if num_inference_steps is None:
            raise ValueError(
                "num_inference_steps is required for DDIM inversion. "
                "It MUST match the exact value used during generation. "
                "Partial or default inversion will produce incorrect results."
            )
        
        # CRITICAL: Only support unconditional inversion for now
        # Guided inversion requires careful handling of classifier-free guidance
        if guidance_scale != 1.0:
            raise ValueError(
                f"Only unconditional inversion (guidance_scale=1.0) is currently supported. "
                f"Got guidance_scale={guidance_scale}. "
                f"Guided inversion requires additional implementation for mathematical correctness."
            )
        
        # CRITICAL: Verify DDIM scheduler configuration matches generation
        # This assertion guarantees inversion uses identical scheduler settings
        if hasattr(self.scheduler, 'config') and hasattr(self.scheduler.config, 'timestep_spacing'):
            scheduler_type = self.scheduler.__class__.__name__
            if scheduler_type == "DDIMScheduler":
                assert self.scheduler.config.timestep_spacing == "leading", (
                    f"DDIM scheduler timestep_spacing must be 'leading' for inversion correctness. "
                    f"Got: {self.scheduler.config.timestep_spacing}. "
                    f"Generation and inversion schedulers must match exactly."
                )
                assert self.scheduler.config.clip_sample == False, (
                    f"DDIM scheduler clip_sample must be False. Got: {self.scheduler.config.clip_sample}"
                )
                assert self.scheduler.config.set_alpha_to_one == False, (
                    f"DDIM scheduler set_alpha_to_one must be False. Got: {self.scheduler.config.set_alpha_to_one}"
                )
        
        # Set up scheduler timesteps for FULL inversion
        # This MUST match the num_inference_steps used during generation
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps.flip(0)  # Reverse for inversion
        
        # Cache unconditional embedding once (no need to recompute in loop)
        # For unconditional: prompt = "", guidance_scale = 1.0
        prompt_embeds = self._encode_prompt("")
        
        # Full DDIM inversion loop: z_0 → z_1 → ... → z_T
        # Processes ALL timesteps (full inversion, not partial)
        # timesteps[:-1] excludes the last timestep to avoid going beyond z_T
        latent = z_0.clone()
        
        for i, t in enumerate(timesteps[:-1]):  # Process all but last (full inversion)
            t_next = timesteps[i + 1]
            
            # Predict noise (using cached prompt_embeds)
            noise_pred = self._predict_noise(latent, t, prompt_embeds, guidance_scale)
            
            # DDIM inversion step: z_t → z_{t+1}
            latent = self._ddim_inversion_step(latent, noise_pred, t, t_next)
        
        z_T = latent
        return z_T
    
    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        allow_resize: bool = False,
    ) -> torch.Tensor:
        """
        Encode image to latent space z_0.
        
        For DDIM inversion, images must be exactly 512x512 to ensure
        mathematical correctness of the inversion process.
        
        Args:
            image: Input image
            allow_resize: If False (default), require exact 512x512.
                Set to True only for fast-only detection mode.
            
        Returns:
            Latent tensor z_0 [1, 4, H/8, W/8]
            
        Raises:
            ValueError: If image is not 512x512 and allow_resize=False
        """
        from .utils import encode_image_to_latent
        return encode_image_to_latent(
            image, 
            self.vae, 
            device=self.device, 
            vae_scaling_factor=self.vae_scaling_factor,
            allow_resize=allow_resize,
        )
    
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        # Tokenize
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode
        prompt_embeds = self.pipeline.text_encoder(text_input_ids)[0]
        
        return prompt_embeds
    
    def _predict_noise(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Predict noise using UNet.
        
        Args:
            latent: Latent tensor
            t: Timestep
            prompt_embeds: Pre-computed prompt embeddings (reused across loop)
            guidance_scale: Guidance scale
            
        Returns:
            Predicted noise tensor
        """
        # For unconditional (guidance_scale == 1.0), use prompt_embeds directly
        # prompt_embeds should already be the unconditional embedding ("")
        if guidance_scale == 1.0:
            latent_model_input = latent
            prompt_embeds_input = prompt_embeds
        else:
            # Guided inversion: expand for classifier-free guidance
            latent_model_input = torch.cat([latent] * 2)
            prompt_embeds_input = torch.cat([
                self._encode_prompt(""),  # Unconditional
                prompt_embeds,            # Conditional
            ])
        
        # Ensure timestep is correct shape
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=self.device)
        t = t.expand(latent_model_input.shape[0])
        
        # Predict noise
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds_input,
        ).sample
        
        # Apply classifier-free guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        return noise_pred
    
    def _ddim_inversion_step(
        self,
        latent: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform single DDIM inversion step: z_t → z_{t+1}.
        
        DDIM inversion formula:
            z_{t+1} = √(α_{t+1}) * x_0_pred + √(1 - α_{t+1}) * ε_pred
        
        where x_0_pred = (z_t - √(1 - α_t) * ε_pred) / √(α_t)
        """
        # Get alpha values
        alpha_t = self.scheduler.alphas_cumprod[t.long().item()]
        alpha_t_next = self.scheduler.alphas_cumprod[t_next.long().item()]
        
        sqrt_alpha_t = alpha_t ** 0.5
        sqrt_one_minus_alpha_t = (1 - alpha_t) ** 0.5
        sqrt_alpha_t_next = alpha_t_next ** 0.5
        sqrt_one_minus_alpha_t_next = (1 - alpha_t_next) ** 0.5
        
        # Predict x_0
        x_0_pred = (latent - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Compute z_{t+1}
        z_next = sqrt_alpha_t_next * x_0_pred + sqrt_one_minus_alpha_t_next * noise_pred
        
        return z_next


class SimpleLatentEncoder:
    """
    Simple VAE-based latent encoding without full DDIM inversion.
    
    For many detection scenarios, encoding the image directly to latent
    space (z_0) is sufficient. Full DDIM inversion to z_T is only needed
    if the watermark was embedded in the initial noise.
    
    This provides a faster alternative when detection operates on z_0.
    """
    
    def __init__(
        self,
        vae: torch.nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize simple encoder.
        
        Args:
            vae: VAE encoder module
            device: Device for computation
        """
        self.vae = vae.to(device)
        self.device = device
        self.vae_scaling_factor = 0.18215
        
        self.vae.eval()
    
    @torch.no_grad()
    def encode(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        allow_resize: bool = True,
    ) -> torch.Tensor:
        """
        Encode image to latent z_0.
        
        Args:
            image: Input image
            allow_resize: If True, allow resizing to 512x512. If False, require exact 512x512.
            
        Returns:
            Latent tensor [1, 4, H/8, W/8]
        """
        from .utils import encode_image_to_latent
        return encode_image_to_latent(
            image, 
            self.vae, 
            device=self.device, 
            vae_scaling_factor=self.vae_scaling_factor,
            allow_resize=allow_resize,
        )
    


def invert_image(
    image: Union[Image.Image, str],
    pipeline: Any,
    num_inference_steps: int = 50,
    prompt: str = "",
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Convenience function to invert an image to z_T.
    
    Args:
        image: PIL image or path to image
        pipeline: Stable Diffusion pipeline
        num_inference_steps: Number of inversion steps
        prompt: Optional prompt for guided inversion
        guidance_scale: Guidance scale
        
    Returns:
        Recovered latent z_T
        
    Example:
        >>> z_T = invert_image("watermarked.png", pipeline)
        >>> print(z_T.shape)  # [1, 4, 64, 64]
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Create inverter and invert
    inverter = DDIMInverter(pipeline)
    z_T = inverter.invert(
        image,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        guidance_scale=guidance_scale,
    )
    
    return z_T



