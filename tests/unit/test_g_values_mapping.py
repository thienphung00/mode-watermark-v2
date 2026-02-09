"""
Unit tests for g-value mapping_mode: binary vs continuous.

Verifies that compute_g_values produces:
- binary mode: g in {0, 1}, exactly two unique values (Bernoulli likelihood).
- continuous mode: real-valued g, many unique values, not confined to {0, 1} (Gaussian likelihood).
"""
import torch
import pytest

from src.detection.g_values import compute_g_values
from src.detection.prf import PRFConfig


# Minimal configs (spatial domain to avoid frequency params)
BINARY_CONFIG = {
    "mapping_mode": "binary",
    "domain": "spatial",
    "frequency_mode": "lowpass",
    "low_freq_cutoff": 0.12,
    "high_freq_cutoff": None,
    "normalize_zero_mean": True,
    "normalize_unit_variance": False,
}
CONTINUOUS_CONFIG = {
    "mapping_mode": "continuous",
    "domain": "spatial",
    "frequency_mode": "lowpass",
    "low_freq_cutoff": 0.12,
    "high_freq_cutoff": None,
    "normalize_zero_mean": True,
    "normalize_unit_variance": False,
    "continuous_range": [-1.0, 1.0],
}

MASTER_KEY = "test_master_key_32bytes_long!!!!!!!"
KEY_ID = "test_key_id"


class TestGValuesMappingMode:
    """Binary vs continuous g-value semantics."""

    def test_binary_mode_values_in_zero_one(self):
        """Binary mode must produce g in {0, 1} only."""
        x0 = torch.randn(2, 4, 64, 64)  # batch of 2
        g, mask = compute_g_values(
            x0, KEY_ID, MASTER_KEY,
            g_field_config=BINARY_CONFIG,
            return_mask=True,
        )
        assert g.dtype == torch.float32
        unique = torch.unique(g)
        assert unique.numel() <= 2, f"Binary mode should have at most 2 unique values, got {unique.tolist()}"
        assert torch.all((g == 0) | (g == 1)), (
            f"Binary mode must be confined to {{0, 1}}, got min={g.min().item()}, max={g.max().item()}"
        )

    def test_continuous_mode_real_valued_not_binary(self):
        """Continuous mode must produce real-valued g, not confined to {0, 1}."""
        x0 = torch.randn(2, 4, 64, 64)
        g, mask = compute_g_values(
            x0, KEY_ID, MASTER_KEY,
            g_field_config=CONTINUOUS_CONFIG,
            return_mask=True,
        )
        assert g.dtype == torch.float32
        unique_count = torch.unique(g).numel()
        assert unique_count > 2, (
            f"Continuous mode must have many unique values (>>2), got {unique_count}"
        )
        assert not torch.all((g == 0) | (g == 1)), (
            "Continuous mode must not be confined to {0, 1}; Gaussian likelihood requires ℝ."
        )
        assert g.std().item() > 0.01, (
            f"Continuous g must be non-degenerate (std > 0.01), got std={g.std().item()}"
        )

    def test_continuous_mode_many_unique_values(self):
        """Continuous mode histogram should not be two-point mass."""
        x0 = torch.randn(1, 4, 64, 64)
        g, _ = compute_g_values(
            x0, KEY_ID, MASTER_KEY,
            g_field_config=CONTINUOUS_CONFIG,
            return_mask=False,
        )
        g_flat = g.flatten()
        assert g_flat.numel() > 0
        assert torch.unique(g_flat).numel() >> 2

    def test_binary_mode_unchanged_backward_compat(self):
        """Binary mode behavior unchanged: sign agreement -> {0,1}."""
        # Deterministic: same x0 and G should give same g
        x0 = torch.randn(1, 4, 64, 64)
        g1, _ = compute_g_values(x0, KEY_ID, MASTER_KEY, g_field_config=BINARY_CONFIG, return_mask=False)
        g2, _ = compute_g_values(x0, KEY_ID, MASTER_KEY, g_field_config=BINARY_CONFIG, return_mask=False)
        torch.testing.assert_close(g1, g2)
        assert set(g1.unique().tolist()).issubset({0.0, 1.0})
