"""
High-level detection API functions.

This module provides convenient functions for detecting watermarks in images
and latents without requiring detailed knowledge of the detection pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .statistics import DetectionResult, detect_watermark
from ..algorithms.g_field import compute_g_expected


def _extract_g_observed(latent_np: np.ndarray, method: str = "normalized") -> np.ndarray:
    """
    Extract observed G values from latent using simple numpy-based methods.
    
    Args:
        latent_np: Latent tensor as numpy array (C, H, W)
        method: Extraction method ("normalized", "sign", "raw")
        
    Returns:
        Extracted G observed array
    """
    if method == "normalized":
        # Zero-mean, unit-variance normalization
        return (latent_np - latent_np.mean()) / (latent_np.std() + 1e-8)
    elif method == "sign":
        # Sign extraction
        return np.sign(latent_np)
    elif method == "raw":
        # Raw values
        return latent_np.copy()
    elif method == "whitened":
        # Simple whitening via normalization (simplified version)
        return (latent_np - latent_np.mean()) / (latent_np.std() + 1e-8)
    else:
        # Default to normalized
        return (latent_np - latent_np.mean()) / (latent_np.std() + 1e-8)


def detect_image(
    image,  # PIL.Image, str (path), or np.ndarray
    key_id: str,
    master_key: str,
    pipeline=None,  # StableDiffusionPipeline (optional, for full DDIM inversion)
    vae=None,  # VAE encoder (alternative to pipeline)
    extraction_method: str = "whitened",
    threshold: Optional[float] = None,
    alpha: float = 0.01,
    use_ddim_inversion: bool = False,
    num_inversion_steps: int = 50,
    device: str = "cuda",
) -> DetectionResult:
    """
    High-level API to detect watermark in an image.
    
    This is the main entry point for detection. It:
    1. Encodes image to latent space
    2. Computes expected G-field from master_key + key_id
    3. Extracts observed G from latent
    4. Computes S-statistic and makes decision
    
    NO METADATA REQUIRED - only image and key_id.
    
    Args:
        image: Input image (PIL, path string, or numpy array)
        key_id: Public key identifier (stored with image or known)
        master_key: Secret master key (kept by detector)
        pipeline: SD pipeline for DDIM inversion (optional)
        vae: VAE encoder if not using full pipeline
        extraction_method: How to extract G_observed ("whitened", "normalized", etc.)
        threshold: Detection threshold (default: z_{1-alpha} from N(0,1))
        alpha: FPR for default threshold
        use_ddim_inversion: If True, perform full DDIM inversion to z_T
        num_inversion_steps: Number of DDIM inversion steps
        device: Device for computation
        
    Returns:
        DetectionResult with s_statistic, p_value, is_watermarked, etc.
        
    Example:
        >>> result = detect_image(
        ...     image="test.png",
        ...     key_id="img_001",
        ...     master_key="secret_master_key",
        ...     vae=sd_pipeline.vae,
        ... )
        >>> if result.is_watermarked:
        ...     print(f"Watermark detected! S={result.s_statistic:.3f}")
    """
    try:
        import torch
    except ImportError:
        raise ImportError("detect_image requires PyTorch. Install with: pip install torch")
    
    from PIL import Image as PILImage
    from .inversion import DDIMInverter, SimpleLatentEncoder
    
    # Load image if path
    if isinstance(image, str):
        image = PILImage.open(image).convert("RGB")
    
    # Step 1: Encode image to latent
    if use_ddim_inversion and pipeline is not None:
        # Full DDIM inversion to z_T
        inverter = DDIMInverter(pipeline, device=device)
        latent = inverter.invert(image, num_inference_steps=num_inversion_steps)
    else:
        # Simple VAE encoding to z_0
        if vae is None:
            if pipeline is not None:
                vae = pipeline.vae
            else:
                raise ValueError("Either pipeline or vae must be provided")
        encoder = SimpleLatentEncoder(vae, device=device)
        latent = encoder.encode(image)
    
    # Convert to numpy for detection
    latent_np = latent.cpu().numpy()
    if latent_np.ndim == 4:
        latent_np = latent_np[0]  # Remove batch dim -> (C, H, W)
    
    # Step 2: Compute expected G-field
    g_expected = compute_g_expected(
        master_key=master_key,
        key_id=key_id,
        shape=latent_np.shape,
    )
    
    # Step 3: Extract observed G using simple numpy-based methods
    g_observed = _extract_g_observed(latent_np, method=extraction_method)
    
    # Step 4: Detect watermark
    result = detect_watermark(
        g_observed=g_observed,
        g_expected=g_expected,
        threshold=threshold,
        alpha=alpha,
    )
    
    return result


def detect_latent(
    latent,  # torch.Tensor or np.ndarray
    key_id: str,
    master_key: str,
    extraction_method: str = "whitened",
    threshold: Optional[float] = None,
    alpha: float = 0.01,
) -> DetectionResult:
    """
    Detect watermark from a pre-computed latent tensor.
    
    Use this when you've already encoded/inverted the image.
    
    Args:
        latent: Latent tensor (C, H, W) or (B, C, H, W)
        key_id: Public key identifier
        master_key: Secret master key
        extraction_method: G_observed extraction method
        threshold: Detection threshold
        alpha: FPR for default threshold
        
    Returns:
        DetectionResult
    """
    # Convert to numpy
    try:
        import torch
        if isinstance(latent, torch.Tensor):
            latent_np = latent.cpu().numpy()
        else:
            latent_np = np.asarray(latent)
    except ImportError:
        latent_np = np.asarray(latent)
    
    # Remove batch dimension if present
    if latent_np.ndim == 4:
        latent_np = latent_np[0]
    
    # Compute G_expected
    g_expected = compute_g_expected(
        master_key=master_key,
        key_id=key_id,
        shape=latent_np.shape,
    )
    
    # Extract G_observed
    g_observed = _extract_g_observed(latent_np, method=extraction_method)
    
    # Detect
    return detect_watermark(g_observed, g_expected, threshold, alpha)

