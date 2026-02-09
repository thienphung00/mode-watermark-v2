"""
Canonical g-value computation for key-dependent watermark detection.

This module provides the single source of truth for computing g-values
from latents using a secret key. All g-values must be computed using
this function to ensure consistency across generation, training, and detection.

Design Principles:
- Key-dependent: All g-values depend on a secret key
- No heuristics: No latent statistics or extraction methods
- Explicit: Simple, readable computation
- SynthID-aligned: Follows SynthID-style key → signal → likelihood test
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..algorithms.g_field import GFieldGenerator
from .prf import PRFKeyDerivation
from ..core.config import PRFConfig, GFieldConfig


def g_field_config_to_dict(g_field_config: GFieldConfig) -> dict:
    """
    Convert GFieldConfig (Pydantic model) to dict format expected by compute_g_values().
    
    This helper function ensures consistent conversion between the config model
    used in generation and the dict format used in g-value computation.
    
    Args:
        g_field_config: GFieldConfig instance from config
    
    Returns:
        Dictionary with keys expected by GFieldGenerator
    """
    # Extract normalization settings
    # Priority: top-level fields > nested normalize dict > defaults
    normalize_dict = g_field_config.normalize if isinstance(g_field_config.normalize, dict) else {}
    
    # Use top-level fields if provided, otherwise fall back to nested dict
    if g_field_config.normalize_zero_mean is not None:
        normalize_zero_mean = g_field_config.normalize_zero_mean
    else:
        normalize_zero_mean = normalize_dict.get("zero_mean_per_timestep", True) or normalize_dict.get("zero_mean_per_channel", True)
    
    if g_field_config.normalize_unit_variance is not None:
        normalize_unit_variance = g_field_config.normalize_unit_variance
    else:
        normalize_unit_variance = normalize_dict.get("unit_variance", False)
    
    # Build dict
    result = {
        "mapping_mode": g_field_config.mapping_mode,
        "domain": g_field_config.domain,
        "frequency_mode": g_field_config.frequency_mode,
        "low_freq_cutoff": g_field_config.low_freq_cutoff,
        "normalize_zero_mean": normalize_zero_mean,
        "normalize_unit_variance": normalize_unit_variance,
    }
    
    # Add optional fields
    if g_field_config.continuous_range is not None:
        result["continuous_range"] = g_field_config.continuous_range
    
    # Add high_freq_cutoff if frequency_mode is bandpass
    if g_field_config.frequency_mode == "bandpass":
        if g_field_config.high_freq_cutoff is not None:
            result["high_freq_cutoff"] = g_field_config.high_freq_cutoff
        else:
            # Default: use 0.4 as high cutoff (common in seed bias configs)
            result["high_freq_cutoff"] = 0.4
    
    return result


def compute_g_values(
    x0: torch.Tensor,
    key: str,
    master_key: str,
    *,
    return_mask: bool = True,
    g_field_config: Optional[dict] = None,
    prf_config: Optional[PRFConfig] = None,
    latent_type: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute key-dependent g-values from latent.
    
    This function defines the canonical watermark statistic.
    All generation, detection, and calibration must use it.
    
    This is the canonical function for computing g-values. It:
    1. Generates expected G-field from key using PRF
    2. Compares observed latent against expected G-field
    3. Outputs g-values: binary {0,1} (mapping_mode="binary") or real-valued (mapping_mode="continuous")
    
    This function must be used:
    - During watermark generation (to compute ground truth g-values)
    - During training (to compute training g-values)
    - During detection (to compute detection g-values)
    
    Args:
        x0: Observed latent tensor [B, 4, 64, 64] or [4, 64, 64]
        key: Key identifier (key_id) for PRF-based G-field generation
        master_key: Master key for PRF
        return_mask: If True, return mask tensor (default: True)
            The mask is a binary mask identifying structurally valid positions based on
            frequency domain filtering logic. It is deterministic and key-independent.
        g_field_config: REQUIRED G-field configuration dict. Must match the config
            used during generation. Raises ValueError if None.
            Required keys:
            - mapping_mode: "binary" or "continuous"
            - domain: "spatial" or "frequency"
            - frequency_mode: "lowpass", "highpass", or "bandpass" (if domain="frequency")
            - low_freq_cutoff: float (if frequency_mode requires it)
            - high_freq_cutoff: float (if frequency_mode="bandpass")
            - normalize_zero_mean: bool
            - normalize_unit_variance: bool
        prf_config: Optional PRF configuration (default: ChaCha20, 64-bit)
        latent_type: Optional latent type identifier ("z0" or "zT") for validation.
            If provided, validates that the correct latent space is being used.
            - "z0": VAE-encoded latent (after encoding image)
            - "zT": Initial noise latent (before diffusion process)
            If None, no validation is performed (not recommended).
    
    Returns:
        Tuple of (g, mask):
            - g: G-values [B, N] or [N]. If mapping_mode="binary": values in {0, 1}.
                 If mapping_mode="continuous": real-valued floats (element-wise x0*G), no thresholding.
            - mask: Optional mask [B, N] or [N] with 1 for valid positions, 0 for invalid
                   If return_mask=False, mask is None
                   The mask identifies structurally valid positions based on frequency
                   domain filtering logic (deterministic and key-independent).
    
    Raises:
        ValueError: If g_field_config is None (configuration drift prevention)
        ValueError: If latent_type validation fails
    
    Example:
        >>> g_field_config = {
        ...     "mapping_mode": "binary",
        ...     "domain": "frequency",
        ...     "frequency_mode": "bandpass",
        ...     "low_freq_cutoff": 0.05,
        ...     "high_freq_cutoff": 0.4,
        ...     "normalize_zero_mean": True,
        ...     "normalize_unit_variance": True,
        ... }
        >>> x0 = torch.randn(1, 4, 64, 64)  # VAE-encoded latent
        >>> g, mask = compute_g_values(
        ...     x0, "image_001", "master_key_32bytes!",
        ...     g_field_config=g_field_config, latent_type="z0"
        ... )
        >>> # g is [1, 16384] binary tensor
        >>> # mask is [1, 16384] binary mask with 1 for valid positions, 0 for invalid
    """
    # Handle input shape
    squeeze_batch = False
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)  # [1, C, H, W]
        squeeze_batch = True
    elif x0.dim() == 4:
        pass  # Already [B, C, H, W]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x0.dim()}D")
    
    B, C, H, W = x0.shape
    device = x0.device
    
    # LOCK PRECISION: Ensure bit-stable computation across phases
    # Convert input to float32 if needed (ensures consistent precision)
    original_dtype = x0.dtype
    x0 = x0.float()  # Always use float32 for computation
    
    # ENFORCE: g_field_config must be provided explicitly
    # This prevents silent configuration drift between generation and detection
    if g_field_config is None:
        raise ValueError(
            "g_field_config is REQUIRED and cannot be None. "
            "This prevents configuration drift between generation and detection. "
            "You must pass the same g_field_config used during generation. "
            "Example:\n"
            "  g_field_config = {\n"
            "      'mapping_mode': 'binary',\n"
            "      'domain': 'frequency',\n"
            "      'frequency_mode': 'bandpass',\n"
            "      'low_freq_cutoff': 0.05,\n"
            "      'high_freq_cutoff': 0.4,\n"
            "      'normalize_zero_mean': True,\n"
            "      'normalize_unit_variance': True,\n"
            "  }"
        )
    
    # Validate latent_type if provided
    if latent_type is not None:
        latent_type = latent_type.lower()
        if latent_type not in ("z0", "zt"):
            raise ValueError(
                f"latent_type must be 'z0' or 'zT', got '{latent_type}'. "
                "Use 'z0' for VAE-encoded latents (after encoding image), "
                "or 'zT' for initial noise latents (before diffusion process)."
            )
        # Note: We cannot automatically detect which latent space we're in,
        # but we can document the expectation for debugging
    
    # Create PRF key derivation
    if prf_config is None:
        prf_config = PRFConfig(algorithm="chacha20", output_bits=64)
    
    prf = PRFKeyDerivation(master_key, prf_config)
    
    # Generate expected G-field from key
    shape = (C, H, W)
    num_elements = C * H * W
    prf_seeds = prf.generate_seeds(key, num_elements)
    
    # Generate G-field and mask from generator
    # This ensures the mask geometry exactly matches what was used during generation
    g_gen = GFieldGenerator(**g_field_config)
    if return_mask:
        G_expected_np, mask_np = g_gen.generate_g_field(shape, prf_seeds, return_mask=True)
        # mask_np is [H, W] for frequency domain or [H, W] all-ones for spatial
        # Broadcast to [C, H, W] to match G-field shape
        if mask_np.ndim == 2:
            # Frequency domain: mask is [H, W], broadcast to all channels
            mask_np = np.broadcast_to(mask_np[np.newaxis, :, :], (C, H, W)).copy()
        # mask_np is now [C, H, W]
    else:
        G_expected_np = g_gen.generate_g_field(shape, prf_seeds, return_mask=False)
        mask_np = None
    
    # Convert to tensor with locked precision (float32)
    # G_expected_np is already float32 from GFieldGenerator
    G_expected = torch.from_numpy(G_expected_np).to(device=device, dtype=torch.float32)  # [C, H, W]
    
    # Expand G_expected to batch dimension
    G_expected = G_expected.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, H, W]
    
    # -------------------------------------------------------------------------
    # Binary vs continuous semantics (mapping_mode)
    # -------------------------------------------------------------------------
    # - binary:     g_i ∈ {0, 1} from sign agreement (Bernoulli likelihood).
    #   Thresholding is intentional and required for binary mode.
    # - continuous: g_i ∈ ℝ from element-wise product x0_i * G_i (Gaussian likelihood).
    #   No thresholding, sign, or rounding — real-valued only.
    # -------------------------------------------------------------------------
    mapping_mode = (g_field_config.get("mapping_mode") or "binary").lower()
    
    if mapping_mode == "binary":
        # Binary: sign agreement → g_i = 1 if sign(x0_i) == sign(G_i), else 0
        sign_x0 = torch.sign(x0)  # [B, C, H, W]
        sign_G = torch.sign(G_expected)  # [B, C, H, W]
        g = (sign_x0 == sign_G).float()  # [B, C, H, W]
    elif mapping_mode == "continuous":
        # Continuous: preserve real-valued alignment (additive/affine in magnitude).
        # g_i = x0_i * G_i (element-wise product). No sign, no threshold — forbidden in continuous mode.
        g = (x0 * G_expected)  # [B, C, H, W], real-valued
    else:
        raise ValueError(
            f"mapping_mode must be 'binary' or 'continuous', got '{mapping_mode}'. "
            "Binary uses sign agreement {0,1}; continuous uses real-valued product."
        )
    
    # Flatten to [B, N]
    g = g.flatten(start_dim=1)  # [B, N]
    
    # Process mask from generator
    # The mask identifies valid watermark positions using the exact same geometry
    # as used during generation (derived from the same DCT mask in apply_frequency_bandpass)
    mask = None
    if return_mask and mask_np is not None:
        # Flatten mask to match g shape
        mask_flat = mask_np.flatten()  # [N]
        # Convert to torch tensor and expand to batch
        mask = torch.from_numpy(mask_flat).to(device)  # [N]
        mask = mask.unsqueeze(0).expand(B, -1)  # [B, N]
    
    # Remove batch dimension if input was 3D
    if squeeze_batch:
        g = g.squeeze(0)  # [N]
        if mask is not None:
            mask = mask.squeeze(0)  # [N]
    
    # Ensure output dtype is float32 (locked precision)
    g = g.float()
    if mask is not None:
        mask = mask.float()
    
    return g, mask


def compute_g_values_from_latent(
    x0: torch.Tensor,
    key: str,
    master_key: str,
    *,
    return_mask: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Alias for compute_g_values() for backward compatibility.
    
    This function name matches the requirement to extract g-value computation
    logic from HybridDetector.
    """
    return compute_g_values(
        x0,
        key,
        master_key,
        return_mask=return_mask,
        **kwargs,
    )


def validate_g_consistency(
    latent: torch.Tensor,
    key: str,
    master_key: str,
    g_field_config: dict,
    prf_config: Optional[PRFConfig] = None,
    tolerance: float = 1e-6,
) -> bool:
    """
    Validate that g-values computed in two separate calls are identical.
    
    This function ensures that compute_g_values() is deterministic and
    produces bit-stable results across phases (Phase 1 vs Phase 3).
    
    Args:
        latent: Input latent tensor [B, 4, 64, 64] or [4, 64, 64]
        key: Key identifier
        master_key: Master key
        g_field_config: G-field configuration dict
        prf_config: Optional PRF configuration
        tolerance: Maximum allowed absolute difference (default: 1e-6)
        
    Returns:
        True if g-values are identical within tolerance
        
    Raises:
        AssertionError: If g-values differ by more than tolerance
    """
    # Compute g-values twice
    g1, mask1 = compute_g_values(
        latent,
        key,
        master_key,
        return_mask=True,
        g_field_config=g_field_config,
        prf_config=prf_config,
    )
    
    g2, mask2 = compute_g_values(
        latent,
        key,
        master_key,
        return_mask=True,
        g_field_config=g_field_config,
        prf_config=prf_config,
    )
    
    # Check g-values are identical
    max_abs_diff_g = torch.abs(g1 - g2).max().item()
    if max_abs_diff_g > tolerance:
        raise AssertionError(
            f"❌ G-VALUE CONSISTENCY VIOLATION: "
            f"max_abs_diff={max_abs_diff_g} > tolerance={tolerance}. "
            f"This indicates non-deterministic computation or precision drift. "
            f"g1 shape: {g1.shape}, g2 shape: {g2.shape}"
        )
    
    # Check masks are identical (if both are not None)
    if mask1 is not None and mask2 is not None:
        max_abs_diff_mask = torch.abs(mask1 - mask2).max().item()
        if max_abs_diff_mask > tolerance:
            raise AssertionError(
                f"❌ MASK CONSISTENCY VIOLATION: "
                f"max_abs_diff={max_abs_diff_mask} > tolerance={tolerance}. "
                f"This indicates non-deterministic mask computation."
            )
    elif mask1 is not None or mask2 is not None:
        raise AssertionError(
            f"❌ MASK CONSISTENCY VIOLATION: "
            f"One mask is None while the other is not. "
            f"mask1 is None: {mask1 is None}, mask2 is None: {mask2 is None}"
        )
    
    return True

