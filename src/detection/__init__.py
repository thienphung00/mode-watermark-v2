"""
SynthID-style watermark detection pipeline.

This module provides a complete PRF-based detection system that requires
only the image and key_id (no metadata like zT_hash, sample_id, etc.).

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  SynthID-Style Detection Pipeline               │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Image Input                                                 │
    │     └── PNG/JPG image + key_id                                  │
    │                                                                 │
    │  2. Latent Recovery (inversion.py)                             │
    │     └── Image → VAE Encode → z_0 → DDIM Invert → z_T           │
    │                                                                 │
    │  3. G_expected Computation (algorithms/g_field.py)              │
    │     └── G_expected[i] = g(PRF(master_key, key_id, i))          │
    │     └── Uses ChaCha20/AES-CTR, NOT XOR-shift                   │
    │                                                                 │
    │  4. G_observed Extraction (g_values.py)                         │
    │     └── G_observed = compute_g_values(z_T, key, master_key)    │
    │     └── Key-dependent g-value computation                      │
    │                                                                 │
    │  5. S-statistic Computation (statistics.py)                    │
    │     └── S = (1/√n) * Σ G_observed[i] * G_expected[i]          │
    │     └── Direct dot-product, NO Pearson correlation             │
    │                                                                 │
    │  6. Threshold Decision (calibration.py)                        │
    │     └── is_watermarked = (S > threshold)                       │
    │     └── p-value from N(0,1) distribution                       │
    │                                                                 │
    │  7. Quality Metrics (metrics.py) - OPTIONAL                    │
    │     └── PSNR, SSIM, LPIPS for quality evaluation               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Key Design Principles:
    - NO reliance on zT_hash, sample_id, base_seed, or per-image metadata
    - Detection requires ONLY: image + key_id (decoder has master_key)
    - Pure, stateless functions
    - Cryptographically secure PRF (ChaCha20/AES-CTR)

Reference:
    Dathathri et al. "Scalable watermarking for identifying large language
    model outputs." Nature 634, 818-823 (2024).
    https://www.nature.com/articles/s41586-024-08025-4

Example:
    >>> from src.detection import detect_image
    >>> result = detect_image(
    ...     image="watermarked.png",
    ...     key_id="abc123",
    ...     master_key="secret_key",
    ...     pipeline=sd_pipeline,
    ... )
    >>> print(f"Watermarked: {result.is_watermarked}, S={result.s_statistic:.3f}")
"""
from __future__ import annotations

# ============================================================================
# Core modules (no torch/diffusers dependency)
# ============================================================================

# PRF - Cryptographically secure pseudorandom function
from ..core.config import PRFConfig
from .prf import (
    PRFKeyDerivation,
    create_prf,
    generate_prf_seeds,
    seeds_to_rademacher,
    seeds_to_gaussian,
)

# G-field computation (unified module from algorithms)
from ..algorithms.g_field import (
    compute_g_expected,
    compute_g_expected_flat,
    GFieldGenerator,
    apply_frequency_bandpass,
    apply_frequency_mask,  # Backward compatibility alias
)

# S-statistic computation
from .statistics import (
    DetectionResult,
    compute_s_statistic,
    compute_s_statistic_batch,
    compute_p_value,
    detect_watermark,
    threshold_from_fpr,
    expected_tpr,
    diagnose_detection,
)

# Calibration
from .calibration import (
    CalibrationResult,
    calibrate_theoretical,
    calibrate_empirical,
    calibrate_from_labeled_data,
    compute_roc_curve,
    find_threshold_at_fpr,
    find_threshold_at_fpr_from_labeled_data,
    find_threshold_at_tpr,
    compute_detection_metrics_at_threshold,
    summarize_calibration,
)

# Quality metrics (optional, does not affect detection)
from .metrics import (
    QualityMetrics,
    compute_detection_metrics,
    compute_confusion_matrix,
    load_image_as_array,
)

# High-level API
from .api import detect_image, detect_latent

# ============================================================================
# Lazy imports for torch-dependent modules
# ============================================================================

# Try to import torch-dependent modules
try:
    import torch
    
    from .inversion import (
        DDIMInverter,
        SimpleLatentEncoder,
        invert_image,
    )
    from .utils import (
        encode_image_to_latent,
        decode_latent_to_image,
        invert_latent_ddim,
        extract_latent,
    )
    from .g_values import (
        compute_g_values,
        compute_g_values_from_latent,
        g_field_config_to_dict,
    )
except ImportError:
    # Placeholder functions that raise helpful errors
    def _torch_required(*args, **kwargs):
        raise ImportError(
            "This function requires PyTorch. Install with: pip install torch"
        )
    
    DDIMInverter = _torch_required
    SimpleLatentEncoder = _torch_required
    invert_image = _torch_required
    encode_image_to_latent = _torch_required
    decode_latent_to_image = _torch_required
    invert_latent_ddim = _torch_required
    extract_latent = _torch_required
    compute_g_values = _torch_required
    compute_g_values_from_latent = _torch_required
    g_field_config_to_dict = _torch_required


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # High-level API
    "detect_image",
    "detect_latent",
    
    # PRF
    "PRFKeyDerivation",
    "create_prf",
    "generate_prf_seeds",
    "seeds_to_rademacher",
    "seeds_to_gaussian",
    
    # G-field
    "GFieldGenerator",  # Unified G-field generator
    "compute_g_expected",
    "compute_g_expected_flat",
    "apply_frequency_bandpass",
    "apply_frequency_mask",  # Backward compatibility alias
    
    # Inversion
    "DDIMInverter",
    "SimpleLatentEncoder",
    "invert_image",
    
    # Detection utilities
    "encode_image_to_latent",
    "decode_latent_to_image",
    "invert_latent_ddim",
    "extract_latent",
    
    # G-value computation (canonical function for detection)
    "compute_g_values",
    "compute_g_values_from_latent",
    "g_field_config_to_dict",
    
    # Statistics
    "DetectionResult",
    "compute_s_statistic",
    "compute_s_statistic_batch",
    "compute_p_value",
    "detect_watermark",
    "threshold_from_fpr",
    "expected_tpr",
    "diagnose_detection",
    
    # Calibration
    "CalibrationResult",
    "calibrate_theoretical",
    "calibrate_empirical",
    "calibrate_from_labeled_data",
    "compute_roc_curve",
    "find_threshold_at_fpr",
    "find_threshold_at_fpr_from_labeled_data",
    "find_threshold_at_tpr",
    "compute_detection_metrics_at_threshold",
    "summarize_calibration",
    
    # Metrics
    "QualityMetrics",
    "compute_detection_metrics",
    "compute_confusion_matrix",
    "load_image_as_array",
]
