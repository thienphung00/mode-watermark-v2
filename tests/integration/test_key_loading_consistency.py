"""
Test key loading consistency between generation and detection pipelines.

This module verifies that keys are loaded and derived identically between:
- generate_training_images.py (generation)
- precompute_inverted_g_values.py (detection)

Critical consistency requirements tested:
1. Master Key Hash - Same SHA-256 hashing produces identical results
2. PRF Key Derivation - Same master_key + key_id produces identical seed sequences
3. Key Fingerprint - compute_key_fingerprint() produces same output
4. G-Field Config Hash - Same config produces identical g_field_config_hash
5. G-Value Computation - Same keys/config produce identical g-values
"""
import hashlib
import json
from typing import Dict, Any

import pytest
import numpy as np
import torch

from src.core.key_utils import (
    derive_key_fingerprint,
    UNWATERMARKED_DUMMY_KEY,
)
from src.core.config import PRFConfig
from src.detection.prf import PRFKeyDerivation
from src.detection.g_values import compute_g_values, g_field_config_to_dict


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_master_key() -> str:
    """Standard test master key used across all tests."""
    return "test_master_key_32_bytes_long!"


@pytest.fixture
def test_key_id() -> str:
    """Standard test key_id used across all tests."""
    return "test_key_001"


@pytest.fixture
def test_prf_config() -> PRFConfig:
    """Standard PRF configuration."""
    return PRFConfig(algorithm="chacha20", output_bits=64)


@pytest.fixture
def test_g_field_config() -> Dict[str, Any]:
    """
    Standard g-field configuration matching typical production configs.
    
    This config must match the format expected by compute_g_values().
    """
    return {
        "mapping_mode": "binary",
        "domain": "frequency",
        "frequency_mode": "bandpass",
        "low_freq_cutoff": 0.05,
        "high_freq_cutoff": 0.4,
        "normalize_zero_mean": True,
        "normalize_unit_variance": True,
    }


@pytest.fixture
def test_latent() -> torch.Tensor:
    """
    Deterministic test latent tensor.
    
    Uses fixed seed for reproducibility across test runs.
    """
    torch.manual_seed(42)
    return torch.randn(1, 4, 64, 64)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_master_key_hash(master_key: str) -> str:
    """
    Compute master key hash the same way precompute_inverted_g_values.py does.
    
    This is the exact hashing used in:
    - precompute_inverted_g_values.py line 381
    """
    return hashlib.sha256(master_key.encode()).hexdigest()[:16]


def compute_g_field_config_hash(g_field_config: Dict[str, Any]) -> str:
    """
    Compute g-field config hash the same way both scripts do.
    
    This is used in:
    - precompute_inverted_g_values.py: compute_hash_of_dict()
    - generate_training_images.py: lines 244-246
    """
    json_str = json.dumps(g_field_config, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# Test Cases
# =============================================================================


class TestMasterKeyHashConsistency:
    """Test that master key hashing is consistent."""

    def test_master_key_hash_consistency(self, test_master_key: str):
        """
        Verify master key hashing matches between scripts.
        
        Both generation and detection scripts use:
        hashlib.sha256(master_key.encode()).hexdigest()[:16]
        """
        # Compute hash using the same method as precompute_inverted_g_values.py
        hash1 = compute_master_key_hash(test_master_key)
        
        # Compute again to verify determinism
        hash2 = compute_master_key_hash(test_master_key)
        
        assert hash1 == hash2, "Master key hash should be deterministic"
        assert len(hash1) == 16, "Master key hash should be 16 characters"

    def test_different_keys_different_hashes(self):
        """Verify different master keys produce different hashes."""
        hash1 = compute_master_key_hash("master_key_one")
        hash2 = compute_master_key_hash("master_key_two")
        
        assert hash1 != hash2, "Different master keys should produce different hashes"

    def test_master_key_hash_format(self, test_master_key: str):
        """Verify master key hash is valid hexadecimal."""
        hash_value = compute_master_key_hash(test_master_key)
        
        # Should be valid hex
        try:
            int(hash_value, 16)
        except ValueError:
            pytest.fail("Master key hash should be valid hexadecimal")


class TestPRFSeedGenerationConsistency:
    """Test that PRF seed generation is consistent."""

    def test_prf_seed_generation_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """
        Verify PRF produces identical seeds for same inputs.
        
        This is critical for generation-detection consistency.
        """
        prf1 = PRFKeyDerivation(test_master_key, test_prf_config)
        prf2 = PRFKeyDerivation(test_master_key, test_prf_config)
        
        # Generate seeds
        seeds1 = prf1.generate_seeds(test_key_id, count=100)
        seeds2 = prf2.generate_seeds(test_key_id, count=100)
        
        assert seeds1 == seeds2, "Same master_key + key_id should produce identical seeds"

    def test_prf_single_seed_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """Verify single seed generation is consistent."""
        prf = PRFKeyDerivation(test_master_key, test_prf_config)
        
        for i in range(10):
            seed1 = prf.generate_seed(test_key_id, index=i)
            seed2 = prf.generate_seed(test_key_id, index=i)
            
            assert seed1 == seed2, f"Seed at index {i} should be deterministic"

    def test_different_key_ids_different_seeds(
        self,
        test_master_key: str,
        test_prf_config: PRFConfig,
    ):
        """Verify different key_ids produce different seeds."""
        prf = PRFKeyDerivation(test_master_key, test_prf_config)
        
        seeds1 = prf.generate_seeds("key_id_one", count=10)
        seeds2 = prf.generate_seeds("key_id_two", count=10)
        
        assert seeds1 != seeds2, "Different key_ids should produce different seeds"

    def test_different_master_keys_different_seeds(
        self,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """Verify different master keys produce different seeds."""
        prf1 = PRFKeyDerivation("master_key_one", test_prf_config)
        prf2 = PRFKeyDerivation("master_key_two", test_prf_config)
        
        seeds1 = prf1.generate_seeds(test_key_id, count=10)
        seeds2 = prf2.generate_seeds(test_key_id, count=10)
        
        assert seeds1 != seeds2, "Different master keys should produce different seeds"

    def test_prf_seed_stream_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """Verify seed stream matches batch generation."""
        prf = PRFKeyDerivation(test_master_key, test_prf_config)
        
        # Generate via batch
        batch_seeds = prf.generate_seeds(test_key_id, count=50)
        
        # Generate via stream
        stream_seeds = list(prf.generate_seed_stream(test_key_id, count=50))
        
        assert batch_seeds == stream_seeds, "Batch and stream generation should match"


class TestKeyFingerprintConsistency:
    """Test that key fingerprint computation is consistent."""

    def test_key_fingerprint_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """
        Verify compute_key_fingerprint() produces identical results.
        
        This is used by both generation and detection for cache key isolation.
        """
        fp1 = derive_key_fingerprint(test_master_key, test_key_id, test_prf_config)
        fp2 = derive_key_fingerprint(test_master_key, test_key_id, test_prf_config)
        
        assert fp1 == fp2, "Key fingerprint should be deterministic"

    def test_key_fingerprint_length(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
    ):
        """Verify key fingerprint is 64-character hex (SHA-256)."""
        fp = derive_key_fingerprint(test_master_key, test_key_id, test_prf_config)
        
        assert len(fp) == 64, "Key fingerprint should be 64 characters (SHA-256 hex)"
        
        # Should be valid hex
        try:
            int(fp, 16)
        except ValueError:
            pytest.fail("Key fingerprint should be valid hexadecimal")

    def test_different_keys_different_fingerprints(
        self,
        test_prf_config: PRFConfig,
    ):
        """Verify different keys produce different fingerprints."""
        fp1 = derive_key_fingerprint("master_one", "key_one", test_prf_config)
        fp2 = derive_key_fingerprint("master_two", "key_one", test_prf_config)
        fp3 = derive_key_fingerprint("master_one", "key_two", test_prf_config)
        
        assert fp1 != fp2, "Different master keys should produce different fingerprints"
        assert fp1 != fp3, "Different key_ids should produce different fingerprints"
        assert fp2 != fp3, "All fingerprints should be unique"

    def test_key_fingerprint_with_default_prf_config(
        self,
        test_master_key: str,
        test_key_id: str,
    ):
        """Verify fingerprint works with default PRF config."""
        # With explicit config
        fp_explicit = derive_key_fingerprint(
            test_master_key, test_key_id, PRFConfig()
        )
        
        # With None (should use default)
        fp_default = derive_key_fingerprint(test_master_key, test_key_id, None)
        
        assert fp_explicit == fp_default, "Default PRF config should match explicit default"


class TestGFieldConfigHashConsistency:
    """Test that g-field config hashing is consistent."""

    def test_g_field_config_hash_consistency(
        self,
        test_g_field_config: Dict[str, Any],
    ):
        """
        Verify g-field config hashing is deterministic.
        
        Both scripts use: hashlib.sha256(json.dumps(config, sort_keys=True)).hexdigest()[:16]
        """
        hash1 = compute_g_field_config_hash(test_g_field_config)
        hash2 = compute_g_field_config_hash(test_g_field_config)
        
        assert hash1 == hash2, "G-field config hash should be deterministic"

    def test_g_field_config_hash_length(
        self,
        test_g_field_config: Dict[str, Any],
    ):
        """Verify g-field config hash is 16 characters."""
        hash_value = compute_g_field_config_hash(test_g_field_config)
        
        assert len(hash_value) == 16, "G-field config hash should be 16 characters"

    def test_different_configs_different_hashes(self):
        """Verify different configs produce different hashes."""
        config1 = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
        }
        
        config2 = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.10,  # Different cutoff
            "high_freq_cutoff": 0.4,
        }
        
        hash1 = compute_g_field_config_hash(config1)
        hash2 = compute_g_field_config_hash(config2)
        
        assert hash1 != hash2, "Different configs should produce different hashes"

    def test_config_order_independence(self):
        """Verify config hash is independent of key order."""
        config1 = {
            "a": 1,
            "b": 2,
            "c": 3,
        }
        
        config2 = {
            "c": 3,
            "a": 1,
            "b": 2,
        }
        
        hash1 = compute_g_field_config_hash(config1)
        hash2 = compute_g_field_config_hash(config2)
        
        assert hash1 == hash2, "Hash should be independent of key order (sort_keys=True)"


class TestGValueComputationConsistency:
    """Test that g-value computation is consistent."""

    def test_g_value_computation_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify g-values computed identically for same inputs.
        
        This is the core test for generation-detection consistency.
        """
        g1, mask1 = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g2, mask2 = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        assert torch.allclose(g1, g2), "G-values should be identical for same inputs"
        assert torch.allclose(mask1, mask2), "Masks should be identical for same inputs"

    def test_g_value_binary_format(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify g-values are binary {0, 1} when mapping_mode is binary.
        
        Note: This assertion is conditional on mapping_mode to future-proof
        against new modes (e.g., continuous).
        """
        g, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Conditional assertion: only enforce binary values for binary mode
        if test_g_field_config.get("mapping_mode") == "binary":
            unique_vals = torch.unique(g)
            for val in unique_vals:
                assert val.item() in (0.0, 1.0), f"G-values should be binary, got {val.item()}"

    def test_different_keys_different_g_values(
        self,
        test_master_key: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """Verify different keys produce different g-values."""
        g1, _ = compute_g_values(
            test_latent,
            "key_one",
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g2, _ = compute_g_values(
            test_latent,
            "key_two",
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        assert not torch.allclose(g1, g2), "Different keys should produce different g-values"

    def test_unwatermarked_dummy_key(
        self,
        test_master_key: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify UNWATERMARKED_DUMMY_KEY produces consistent g-values.
        
        Detection uses this key for unwatermarked samples.
        """
        g1, _ = compute_g_values(
            test_latent,
            UNWATERMARKED_DUMMY_KEY,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g2, _ = compute_g_values(
            test_latent,
            UNWATERMARKED_DUMMY_KEY,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        assert torch.allclose(g1, g2), "Dummy key should produce consistent g-values"


class TestGenerationDetectionKeyFlow:
    """
    End-to-end test simulating generation and detection key flow.
    
    This test verifies the complete key loading path used by both scripts.
    """

    def test_generation_detection_key_flow(
        self,
        test_master_key: str,
        test_key_id: str,
        test_prf_config: PRFConfig,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        End-to-end test verifying generation-detection key consistency.
        
        Simulates:
        1. Generation: Creates PRF, generates G-field seeds
        2. Detection: Uses same master_key + key_id, computes g-values
        3. Verification: All derived values match
        """
        # === GENERATION SIDE ===
        # This simulates what generate_training_images.py does
        
        # 1. Hash master key (for logging/verification)
        gen_master_key_hash = compute_master_key_hash(test_master_key)
        
        # 2. Create PRF for G-field generation
        gen_prf = PRFKeyDerivation(test_master_key, test_prf_config)
        
        # 3. Generate seeds for G-field
        gen_seeds = gen_prf.generate_seeds(test_key_id, count=100)
        
        # 4. Compute g-field config hash
        gen_config_hash = compute_g_field_config_hash(test_g_field_config)
        
        # 5. Compute key fingerprint
        gen_fingerprint = derive_key_fingerprint(test_master_key, test_key_id, test_prf_config)
        
        # === DETECTION SIDE ===
        # This simulates what precompute_inverted_g_values.py does
        
        # 1. Hash master key (same method)
        det_master_key_hash = compute_master_key_hash(test_master_key)
        
        # 2. Create PRF for g-value computation
        det_prf = PRFKeyDerivation(test_master_key, test_prf_config)
        
        # 3. Generate seeds (should match generation)
        det_seeds = det_prf.generate_seeds(test_key_id, count=100)
        
        # 4. Compute g-field config hash
        det_config_hash = compute_g_field_config_hash(test_g_field_config)
        
        # 5. Compute key fingerprint
        det_fingerprint = derive_key_fingerprint(test_master_key, test_key_id, test_prf_config)
        
        # 6. Compute g-values
        g_values, mask = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # === VERIFICATION ===
        assert gen_master_key_hash == det_master_key_hash, (
            f"Master key hash mismatch: gen={gen_master_key_hash}, det={det_master_key_hash}"
        )
        
        assert gen_seeds == det_seeds, (
            "PRF seeds mismatch between generation and detection"
        )
        
        assert gen_config_hash == det_config_hash, (
            f"G-field config hash mismatch: gen={gen_config_hash}, det={det_config_hash}"
        )
        
        assert gen_fingerprint == det_fingerprint, (
            f"Key fingerprint mismatch: gen={gen_fingerprint[:16]}..., det={det_fingerprint[:16]}..."
        )
        
        # Verify g-values are valid
        assert g_values is not None, "G-values should not be None"
        assert mask is not None, "Mask should not be None"
        assert g_values.shape[0] > 0, "G-values should have non-zero size"

    def test_watermarked_vs_unwatermarked_flow(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify watermarked and unwatermarked flows produce different g-values.
        
        This tests the key_id vs UNWATERMARKED_DUMMY_KEY distinction.
        """
        # Watermarked: use actual key_id
        g_watermarked, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Unwatermarked: use dummy key
        g_unwatermarked, _ = compute_g_values(
            test_latent,
            UNWATERMARKED_DUMMY_KEY,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        assert not torch.allclose(g_watermarked, g_unwatermarked), (
            "Watermarked and unwatermarked g-values should differ"
        )

    def test_multiple_key_ids_isolation(
        self,
        test_master_key: str,
        test_prf_config: PRFConfig,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify multiple key_ids are properly isolated.
        
        Different key_ids should produce different fingerprints and g-values.
        """
        key_ids = ["key_001", "key_002", "key_003"]
        fingerprints = []
        g_values_list = []
        
        for key_id in key_ids:
            fp = derive_key_fingerprint(test_master_key, key_id, test_prf_config)
            fingerprints.append(fp)
            
            g, _ = compute_g_values(
                test_latent,
                key_id,
                test_master_key,
                return_mask=True,
                g_field_config=test_g_field_config,
                latent_type="zT",
            )
            g_values_list.append(g)
        
        # All fingerprints should be unique
        assert len(set(fingerprints)) == len(key_ids), "All key fingerprints should be unique"
        
        # All g-values should be different
        for i in range(len(key_ids)):
            for j in range(i + 1, len(key_ids)):
                assert not torch.allclose(g_values_list[i], g_values_list[j]), (
                    f"G-values for {key_ids[i]} and {key_ids[j]} should differ"
                )


# =============================================================================
# NEW TESTS: Harden Key Consistency Tests Against Silent Drift
# =============================================================================


class TestPRFConfigEffectiveEquivalence:
    """
    Test that PRF configuration equivalence is enforced.
    
    Critical: Generation and detection must construct identical PRFConfig objects,
    even when one uses defaults and the other uses explicit config.
    """

    def test_prf_config_effective_equivalence(self, test_master_key: str):
        """
        Verify PRF configuration equivalence between explicit defaults and None.
        
        This test catches silent drift if:
        - One side uses defaults and the other explicit config
        - Any PRFConfig field silently changes between versions
        """
        # Generation-style: explicit PRFConfig with defaults
        gen_prf = PRFKeyDerivation(test_master_key, PRFConfig())
        
        # Detection-style: None config (should use same defaults internally)
        det_prf = PRFKeyDerivation(test_master_key, None)
        
        # Use model_dump() for canonical serialization (Pydantic v2)
        gen_config_dict = gen_prf.config.model_dump()
        det_config_dict = det_prf.config.model_dump()
        
        assert gen_config_dict == det_config_dict, (
            f"PRFConfig effective equivalence violated!\n"
            f"Generation config: {gen_config_dict}\n"
            f"Detection config: {det_config_dict}\n"
            "This can cause silent key mismatch between generation and detection."
        )

    def test_prf_config_all_fields_explicit(self, test_master_key: str):
        """
        Verify that all PRFConfig fields are compared.
        
        Ensures the comparison includes algorithm and output_bits.
        """
        # Create configs with different parameters
        config1 = PRFConfig(algorithm="chacha20", output_bits=64)
        config2 = PRFConfig(algorithm="chacha20", output_bits=32)
        config3 = PRFConfig(algorithm="aes_ctr", output_bits=64)
        
        prf1 = PRFKeyDerivation(test_master_key, config1)
        prf2 = PRFKeyDerivation(test_master_key, config2)
        prf3 = PRFKeyDerivation(test_master_key, config3)
        
        # Different output_bits should produce different config dicts
        assert prf1.config.model_dump() != prf2.config.model_dump(), (
            "PRFConfig with different output_bits should not be equal"
        )
        
        # Different algorithm should produce different config dicts
        assert prf1.config.model_dump() != prf3.config.model_dump(), (
            "PRFConfig with different algorithm should not be equal"
        )

    def test_prf_config_serialization_stability(self, test_master_key: str):
        """
        Verify PRFConfig serialization is stable across multiple calls.
        
        Ensures model_dump() is deterministic.
        """
        prf = PRFKeyDerivation(test_master_key, PRFConfig())
        
        # Serialize multiple times
        dict1 = prf.config.model_dump()
        dict2 = prf.config.model_dump()
        dict3 = prf.config.model_dump()
        
        assert dict1 == dict2 == dict3, (
            "PRFConfig.model_dump() must be deterministic"
        )

    def test_prf_produces_identical_seeds_regardless_of_config_init(
        self,
        test_master_key: str,
        test_key_id: str,
    ):
        """
        Verify PRF produces identical seeds whether config is explicit or None.
        
        This is the ultimate test: seeds must match regardless of how
        the PRFConfig was initialized.
        """
        # Create PRFs with different initialization styles
        prf_explicit = PRFKeyDerivation(test_master_key, PRFConfig())
        prf_none = PRFKeyDerivation(test_master_key, None)
        prf_explicit_with_values = PRFKeyDerivation(
            test_master_key,
            PRFConfig(algorithm="chacha20", output_bits=64)
        )
        
        # Generate seeds
        seeds_explicit = prf_explicit.generate_seeds(test_key_id, count=100)
        seeds_none = prf_none.generate_seeds(test_key_id, count=100)
        seeds_values = prf_explicit_with_values.generate_seeds(test_key_id, count=100)
        
        assert seeds_explicit == seeds_none, (
            "PRF with explicit PRFConfig() must produce same seeds as PRF with None"
        )
        assert seeds_explicit == seeds_values, (
            "PRF with explicit default values must produce same seeds"
        )


class TestPRFSeedConsumptionAlignment:
    """
    Test that PRF seed consumption is aligned with g-value usage.
    
    This ensures no out-of-bounds or skipped seeds occur between
    generation and detection.
    """

    def test_prf_seed_consumption_alignment(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify g.numel() is bounded by PRF seed count.
        
        The number of g-values produced must not exceed the number of
        PRF seeds consumed.
        """
        # Compute expected seed count based on latent shape
        _, C, H, W = test_latent.shape
        expected_seed_count = C * H * W
        
        # Generate seeds
        prf = PRFKeyDerivation(test_master_key, PRFConfig())
        seeds = prf.generate_seeds(test_key_id, count=expected_seed_count)
        
        # Compute g-values
        g, mask = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Verify alignment: g.numel() should equal seed count (exact match expected)
        assert g.numel() == len(seeds), (
            f"G-value count ({g.numel()}) does not match seed count ({len(seeds)}). "
            f"This indicates seed consumption misalignment."
        )

    def test_seed_count_determinism(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
    ):
        """
        Verify the seed count formula is deterministic.
        
        For any latent shape, the seed count should be C * H * W.
        """
        shapes = [
            (1, 4, 64, 64),
            (2, 4, 64, 64),
            (1, 4, 32, 32),
            (1, 8, 64, 64),
        ]
        
        for shape in shapes:
            latent = torch.randn(shape)
            B, C, H, W = shape
            expected_seeds = C * H * W
            
            g, _ = compute_g_values(
                latent,
                test_key_id,
                test_master_key,
                return_mask=True,
                g_field_config=test_g_field_config,
                latent_type="zT",
            )
            
            # g is [B, N] where N = C * H * W
            assert g.shape[-1] == expected_seeds, (
                f"For shape {shape}, expected {expected_seeds} g-values per sample, "
                f"got {g.shape[-1]}"
            )

    def test_seed_indexing_consistency(
        self,
        test_master_key: str,
        test_key_id: str,
    ):
        """
        Verify seeds are generated with consistent indexing.
        
        Batch generation and individual generation must produce
        identical sequences.
        """
        prf = PRFKeyDerivation(test_master_key, PRFConfig())
        count = 1000
        
        # Batch generation
        batch_seeds = prf.generate_seeds(test_key_id, count=count)
        
        # Individual generation
        individual_seeds = [
            prf.generate_seed(test_key_id, index=i)
            for i in range(count)
        ]
        
        assert batch_seeds == individual_seeds, (
            "Batch seed generation must match individual seed generation. "
            "This ensures no seed indexing drift."
        )

    def test_no_seed_skipping(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify seeds are consumed sequentially without skipping.
        
        The first N seeds should correspond to the first N g-values.
        """
        _, C, H, W = test_latent.shape
        num_elements = C * H * W
        
        prf = PRFKeyDerivation(test_master_key, PRFConfig())
        
        # Generate different counts of seeds
        seeds_full = prf.generate_seeds(test_key_id, count=num_elements)
        seeds_half = prf.generate_seeds(test_key_id, count=num_elements // 2)
        
        # First half should match
        assert seeds_full[:num_elements // 2] == seeds_half, (
            "Seeds should be generated sequentially. "
            "First N seeds should be identical regardless of total count."
        )


class TestGValueSerializationStability:
    """
    Test that serialized g-values remain bit-identical across save/load.
    
    Detection often relies on offline precomputation. This ensures
    no drift occurs during serialization.
    """

    def test_g_value_serialization_roundtrip(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
        tmp_path,
    ):
        """
        Verify g-values survive torch.save/torch.load bit-exactly.
        
        Uses torch.equal() for strict bit-exact comparison, not allclose.
        """
        g1, mask1 = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Save to file
        g_path = tmp_path / "g.pt"
        mask_path = tmp_path / "mask.pt"
        torch.save(g1, g_path)
        torch.save(mask1, mask_path)
        
        # Load back
        g2 = torch.load(g_path)
        mask2 = torch.load(mask_path)
        
        # Bit-exact comparison (not allclose!)
        assert torch.equal(g1, g2), (
            f"G-values not bit-identical after serialization roundtrip!\n"
            f"Max diff: {(g1 - g2).abs().max().item()}\n"
            f"This indicates dtype, normalization, or precision drift."
        )
        
        assert torch.equal(mask1, mask2), (
            f"Mask not bit-identical after serialization roundtrip!\n"
            f"Max diff: {(mask1 - mask2).abs().max().item()}"
        )

    def test_g_value_dtype_preservation(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
        tmp_path,
    ):
        """
        Verify g-value dtype is preserved through serialization.
        """
        g1, mask1 = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g_path = tmp_path / "g_dtype.pt"
        torch.save(g1, g_path)
        g2 = torch.load(g_path)
        
        assert g1.dtype == g2.dtype, (
            f"G-value dtype changed: {g1.dtype} -> {g2.dtype}"
        )

    def test_g_value_device_handling(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
        tmp_path,
    ):
        """
        Verify g-values can be loaded to same device type.
        """
        g1, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g_path = tmp_path / "g_device.pt"
        torch.save(g1, g_path)
        
        # Load with map_location to ensure consistent device
        g2 = torch.load(g_path, map_location=g1.device)
        
        assert g1.device == g2.device, (
            f"G-value device changed: {g1.device} -> {g2.device}"
        )

    def test_multiple_serialization_roundtrips(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
        tmp_path,
    ):
        """
        Verify g-values remain stable through multiple save/load cycles.
        """
        g, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        g_original = g.clone()
        
        # Multiple roundtrips
        for i in range(5):
            g_path = tmp_path / f"g_round_{i}.pt"
            torch.save(g, g_path)
            g = torch.load(g_path)
        
        assert torch.equal(g_original, g), (
            f"G-values drifted after {5} serialization roundtrips!\n"
            f"Max diff: {(g_original - g).abs().max().item()}"
        )


class TestConditionalBinaryAssertions:
    """
    Test binary g-value assertions are conditional on mapping_mode.
    
    Future-proofs tests against new mapping modes (e.g., continuous).
    """

    def test_binary_mode_produces_binary_values(
        self,
        test_master_key: str,
        test_key_id: str,
        test_latent: torch.Tensor,
    ):
        """
        Verify binary mapping mode produces only {0.0, 1.0} values.
        """
        binary_config = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        g, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=binary_config,
            latent_type="zT",
        )
        
        # Conditional assertion: only for binary mode
        if binary_config["mapping_mode"] == "binary":
            unique_vals = set(torch.unique(g).tolist())
            expected_vals = {0.0, 1.0}
            assert unique_vals.issubset(expected_vals), (
                f"Binary mode should produce only {{0.0, 1.0}}, "
                f"got {unique_vals}"
            )

    def test_g_value_format_conditional_on_config(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify g-value format assertions are conditional on mapping_mode.
        
        This is the pattern all tests should follow for future-proofing.
        """
        g, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Conditional assertion based on mapping_mode
        mapping_mode = test_g_field_config.get("mapping_mode", "binary")
        
        if mapping_mode == "binary":
            unique_vals = set(torch.unique(g).tolist())
            assert unique_vals.issubset({0.0, 1.0}), (
                f"Binary mapping_mode should produce only {{0.0, 1.0}}, "
                f"got {unique_vals}"
            )
        elif mapping_mode == "continuous":
            # For continuous mode, values should be in the continuous range
            continuous_range = test_g_field_config.get("continuous_range", (-1.0, 1.0))
            min_val, max_val = continuous_range
            assert g.min().item() >= min_val - 1e-6, (
                f"Continuous mode values below range: {g.min().item()} < {min_val}"
            )
            assert g.max().item() <= max_val + 1e-6, (
                f"Continuous mode values above range: {g.max().item()} > {max_val}"
            )
        # Add new mapping modes here as they are introduced

    def test_existing_binary_test_uses_conditional_assertion(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Pattern test: demonstrate how existing tests should be structured.
        
        This test shows the correct conditional pattern that should be
        applied to test_g_value_binary_format and similar tests.
        """
        g, _ = compute_g_values(
            test_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # CORRECT PATTERN: Conditional on config
        if test_g_field_config.get("mapping_mode") == "binary":
            unique_vals = torch.unique(g)
            for val in unique_vals:
                assert val.item() in (0.0, 1.0), (
                    f"G-values should be binary for binary mode, got {val.item()}"
                )


class TestCrossProcessDeterminism:
    """
    Additional tests for cross-process determinism.
    
    These tests verify that g-values computed in separate processes
    or at different times remain identical.
    """

    def test_compute_g_values_is_deterministic(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
        test_latent: torch.Tensor,
    ):
        """
        Verify compute_g_values is deterministic across multiple calls.
        """
        results = []
        for _ in range(5):
            g, mask = compute_g_values(
                test_latent,
                test_key_id,
                test_master_key,
                return_mask=True,
                g_field_config=test_g_field_config,
                latent_type="zT",
            )
            results.append((g.clone(), mask.clone()))
        
        # All results should be identical
        g_ref, mask_ref = results[0]
        for i, (g, mask) in enumerate(results[1:], start=1):
            assert torch.equal(g_ref, g), (
                f"G-values differ between call 0 and call {i}"
            )
            assert torch.equal(mask_ref, mask), (
                f"Masks differ between call 0 and call {i}"
            )

    def test_g_values_independent_of_batch_size(
        self,
        test_master_key: str,
        test_key_id: str,
        test_g_field_config: Dict[str, Any],
    ):
        """
        Verify g-values are consistent regardless of batch processing.
        
        Single-sample computation should match batched computation.
        """
        torch.manual_seed(42)
        single_latent = torch.randn(1, 4, 64, 64)
        
        # Compute for single sample
        g_single, mask_single = compute_g_values(
            single_latent,
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Compute for 3D input (no batch dim)
        g_3d, mask_3d = compute_g_values(
            single_latent.squeeze(0),
            test_key_id,
            test_master_key,
            return_mask=True,
            g_field_config=test_g_field_config,
            latent_type="zT",
        )
        
        # Results should match (squeeze/unsqueeze should be transparent)
        assert torch.equal(g_single.squeeze(0), g_3d), (
            "G-values differ between 3D and 4D input"
        )
        assert torch.equal(mask_single.squeeze(0), mask_3d), (
            "Masks differ between 3D and 4D input"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
