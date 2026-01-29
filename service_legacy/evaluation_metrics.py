"""
Evaluation metrics for imperceptibility comparison.

This module provides metrics to compare baseline (unwatermarked) vs watermarked images.
These metrics are used ONLY for evaluation purposes and do not affect watermark detection.

Metrics:
    - L2: Normalized L2 distance (pixel space)
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

# Quality metrics from scikit-image
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def image_to_array(image: Image.Image, normalize: bool = True) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        normalize: If True, normalize to [0, 1]
    
    Returns:
        Image as numpy array [H, W, C]
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)
    
    if normalize:
        arr = arr / 255.0
    
    return arr


def compute_l2(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute normalized L2 distance (Euclidean norm) between two images.
    
    L2 is normalized by the number of pixels to give per-pixel average distance.
    
    Args:
        image1: First image array [H, W, C] in range [0, 1]
        image2: Second image array [H, W, C] in range [0, 1]
    
    Returns:
        Normalized L2 distance (float)
    """
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    l2 = np.sqrt(np.mean(diff ** 2))
    return float(l2)


def compute_psnr(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Higher PSNR indicates less distortion. Typical good values:
    - >40 dB: Excellent (nearly invisible changes)
    - 30-40 dB: Good
    - <30 dB: Visible distortion
    
    Args:
        image1: First image array [H, W, C] in range [0, data_range]
        image2: Second image array [H, W, C] in range [0, data_range]
        data_range: Data range (default: 1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    if HAS_SKIMAGE:
        return float(peak_signal_noise_ratio(image1, image2, data_range=data_range))
    
    # Fallback implementation
    mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(image1: np.ndarray, image2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index.
    
    SSIM measures perceptual similarity. Range [0, 1]:
    - >0.95: Nearly identical
    - 0.90-0.95: Very similar
    - <0.90: Noticeable differences
    
    Args:
        image1: First image array [H, W, C] in range [0, data_range]
        image2: Second image array [H, W, C] in range [0, data_range]
        data_range: Data range (default: 1.0 for normalized images)
    
    Returns:
        SSIM value (0-1)
    
    Raises:
        ImportError: If scikit-image is not available
    """
    if not HAS_SKIMAGE:
        raise ImportError("SSIM requires scikit-image: pip install scikit-image")
    
    if image1.ndim == 3:
        # Multi-channel (RGB)
        return float(
            structural_similarity(
                image1, image2, data_range=data_range, channel_axis=-1
            )
        )
    else:
        # Single channel
        return float(structural_similarity(image1, image2, data_range=data_range))


def compute_all_metrics(
    baseline_image: Image.Image,
    watermarked_image: Image.Image,
) -> Dict[str, float]:
    """
    Compute all difference metrics between baseline and watermarked images.
    
    Args:
        baseline_image: Baseline (unwatermarked) PIL Image
        watermarked_image: Watermarked PIL Image
    
    Returns:
        Dictionary with:
            - l2: Normalized L2 distance
            - psnr: PSNR in dB
            - ssim: SSIM value (0-1)
    """
    # Convert to normalized arrays
    arr_baseline = image_to_array(baseline_image, normalize=True)
    arr_watermarked = image_to_array(watermarked_image, normalize=True)
    
    # Compute metrics
    l2 = compute_l2(arr_baseline, arr_watermarked)
    psnr = compute_psnr(arr_baseline, arr_watermarked, data_range=1.0)
    ssim = compute_ssim(arr_baseline, arr_watermarked, data_range=1.0)
    
    return {
        "l2": l2,
        "psnr": psnr,
        "ssim": ssim,
    }

