"""
SynthID-style S-statistic computation for watermark detection.

Implements the core detection statistic from SynthID:
    S = (1 / √n) * Σ G_observed[i] * G_expected[i]

This is a direct dot-product alignment score, NOT Pearson correlation.
The normalization by √n ensures the statistic follows a standard
normal distribution under the null hypothesis (no watermark).

Key Properties:
    - Under H0 (no watermark): S ~ N(0, 1)
    - Under H1 (watermarked): S ~ N(μ, 1) where μ > 0
    - Higher S indicates stronger watermark presence
    - p-value can be computed as 1 - Φ(S) for one-sided test

Reference:
    Dathathri et al. "Scalable watermarking for identifying large language
    model outputs." Nature 634, 818-823 (2024).
    https://www.nature.com/articles/s41586-024-08025-4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    
    s_statistic: float  # S-statistic value
    p_value: float  # p-value (one-sided)
    is_watermarked: bool  # Detection decision
    threshold: float  # Threshold used
    confidence: float  # Confidence level (1 - p_value)
    n_elements: int  # Number of elements used
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "s_statistic": self.s_statistic,
            "p_value": self.p_value,
            "is_watermarked": self.is_watermarked,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "n_elements": self.n_elements,
        }


def compute_s_statistic(
    g_observed: np.ndarray,
    g_expected: np.ndarray,
) -> float:
    """
    Compute SynthID-style S-statistic.
    
    S = (1 / √n) * Σ G_observed[i] * G_expected[i]
    
    This is NOT Pearson correlation. It's a normalized dot product
    that measures alignment between observed and expected signals.
    
    Under the null hypothesis (no watermark, G_observed is random):
        - E[S] = 0
        - Var[S] = 1 (when G_expected is ±1 Rademacher)
        - S ~ N(0, 1) by CLT
    
    Under the alternative (watermarked):
        - E[S] = √n * ρ where ρ is correlation
        - S > 0 with high probability
    
    Args:
        g_observed: Observed signal from latent (any shape)
        g_expected: Expected watermark pattern (same shape as g_observed)
        
    Returns:
        S-statistic value
        
    Example:
        >>> g_obs = np.random.randn(4, 64, 64)  # Random
        >>> g_exp = np.sign(np.random.randn(4, 64, 64))  # ±1
        >>> S = compute_s_statistic(g_obs, g_exp)
        >>> print(f"S = {S:.3f}")  # Should be near 0
    """
    # Flatten both arrays
    obs_flat = g_observed.flatten().astype(np.float64)
    exp_flat = g_expected.flatten().astype(np.float64)
    
    if len(obs_flat) != len(exp_flat):
        raise ValueError(
            f"Shape mismatch: g_observed has {len(obs_flat)} elements, "
            f"g_expected has {len(exp_flat)}"
        )
    
    n = len(obs_flat)
    
    # Compute dot product
    dot_product = np.dot(obs_flat, exp_flat)
    
    # Normalize by √n
    s_statistic = dot_product / np.sqrt(n)
    
    return float(s_statistic)


def compute_s_statistic_batch(
    g_observed_batch: List[np.ndarray],
    g_expected_batch: List[np.ndarray],
) -> np.ndarray:
    """
    Compute S-statistics for a batch of samples.
    
    Args:
        g_observed_batch: List of observed signals
        g_expected_batch: List of expected patterns
        
    Returns:
        Array of S-statistics
    """
    if len(g_observed_batch) != len(g_expected_batch):
        raise ValueError("Batch sizes must match")
    
    s_stats = []
    for g_obs, g_exp in zip(g_observed_batch, g_expected_batch):
        s = compute_s_statistic(g_obs, g_exp)
        s_stats.append(s)
    
    return np.array(s_stats)


def compute_p_value(s_statistic: float, alternative: str = "greater") -> float:
    """
    Compute p-value for S-statistic.
    
    Under H0 (no watermark), S ~ N(0, 1).
    
    Args:
        s_statistic: S-statistic value
        alternative: Test type ("greater", "less", "two-sided")
        
    Returns:
        p-value
    """
    if alternative == "greater":
        # P(S > s) under H0
        return float(1 - stats.norm.cdf(s_statistic))
    elif alternative == "less":
        # P(S < s) under H0
        return float(stats.norm.cdf(s_statistic))
    elif alternative == "two-sided":
        # P(|S| > |s|) under H0
        return float(2 * (1 - stats.norm.cdf(abs(s_statistic))))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")


def detect_watermark(
    g_observed: np.ndarray,
    g_expected: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = 0.01,
) -> DetectionResult:
    """
    Perform watermark detection with statistical decision.
    
    Computes S-statistic and compares to threshold. If threshold
    is not provided, uses the quantile corresponding to FPR=alpha.
    
    Args:
        g_observed: Observed signal
        g_expected: Expected watermark pattern
        threshold: Detection threshold (default: z_{1-alpha})
        alpha: False positive rate for default threshold
        
    Returns:
        DetectionResult with decision and statistics
        
    Example:
        >>> result = detect_watermark(g_observed, g_expected, alpha=0.01)
        >>> if result.is_watermarked:
        ...     print(f"Watermark detected! S={result.s_statistic:.3f}")
    """
    # Compute S-statistic
    s_stat = compute_s_statistic(g_observed, g_expected)
    
    # Determine threshold
    if threshold is None:
        # z_{1-alpha} quantile of standard normal
        threshold = float(stats.norm.ppf(1 - alpha))
    
    # Compute p-value
    p_value = compute_p_value(s_stat, alternative="greater")
    
    # Make decision
    is_watermarked = s_stat > threshold
    confidence = 1 - p_value
    
    return DetectionResult(
        s_statistic=s_stat,
        p_value=p_value,
        is_watermarked=is_watermarked,
        threshold=threshold,
        confidence=confidence,
        n_elements=g_observed.size,
    )


def threshold_from_fpr(target_fpr: float) -> float:
    """
    Compute detection threshold for target false positive rate.
    
    Under H0, S ~ N(0, 1), so threshold = z_{1-FPR}.
    
    Args:
        target_fpr: Target false positive rate (e.g., 0.01 for 1%)
        
    Returns:
        Threshold value
        
    Example:
        >>> threshold = threshold_from_fpr(0.01)  # 1% FPR
        >>> print(f"Threshold: {threshold:.3f}")  # ~2.326
    """
    return float(stats.norm.ppf(1 - target_fpr))


def expected_tpr(
    correlation: float,
    n_elements: int,
    threshold: float,
) -> float:
    """
    Compute expected true positive rate for given parameters.
    
    Under H1, if true correlation is ρ, then:
        S ~ N(√n * ρ, 1)
    
    TPR = P(S > threshold | H1) = 1 - Φ(threshold - √n * ρ)
    
    Args:
        correlation: True correlation ρ between observed and expected
        n_elements: Number of elements n
        threshold: Detection threshold
        
    Returns:
        Expected TPR
    """
    mean_h1 = np.sqrt(n_elements) * correlation
    return float(1 - stats.norm.cdf(threshold - mean_h1))


# ============================================================================
# Diagnostic Functions
# ============================================================================


def diagnose_detection(
    g_observed: np.ndarray,
    g_expected: np.ndarray,
) -> Dict:
    """
    Provide diagnostic information about detection.
    
    Useful for debugging and understanding detection behavior.
    
    Args:
        g_observed: Observed signal
        g_expected: Expected pattern
        
    Returns:
        Dictionary of diagnostic values
    """
    obs_flat = g_observed.flatten()
    exp_flat = g_expected.flatten()
    
    n = len(obs_flat)
    
    # Basic statistics
    s_stat = compute_s_statistic(g_observed, g_expected)
    p_value = compute_p_value(s_stat)
    
    # Pearson correlation (for comparison)
    pearson_corr, pearson_p = stats.pearsonr(obs_flat, exp_flat)
    
    # Signal statistics
    obs_mean = obs_flat.mean()
    obs_std = obs_flat.std()
    exp_mean = exp_flat.mean()
    exp_std = exp_flat.std()
    
    # Dot product components
    dot_product = np.dot(obs_flat, exp_flat)
    
    return {
        # Main statistic
        "s_statistic": float(s_stat),
        "p_value": float(p_value),
        
        # Sample size
        "n_elements": n,
        "sqrt_n": np.sqrt(n),
        
        # Raw dot product
        "dot_product": float(dot_product),
        
        # Pearson comparison
        "pearson_correlation": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
        
        # Signal properties
        "observed_mean": float(obs_mean),
        "observed_std": float(obs_std),
        "expected_mean": float(exp_mean),
        "expected_std": float(exp_std),
        
        # Expected S under perfect match
        "expected_s_if_identical": float(np.sqrt(n)),
    }


