"""
Key registration and policy management.

The Authority is responsible for:
- Key registration and validation
- Deriving scoped keys for GPU workers
- Managing embedding and detection configurations
- Computing policy versions for consistency
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from service.api.artifacts import get_artifact_loader
from service.api.key_store import get_key_store

logger = logging.getLogger(__name__)

# Environment variable for YAML config path
WATERMARK_CONFIG_PATH_ENV = "WATERMARK_CONFIG_PATH"

# Module-level cache for likelihood params (loaded once at runtime)
_ARTIFACT_LOADER = None
_LIKELIHOOD_PARAMS = None


def _load_likelihood_once():
    """
    Load likelihood parameters once and cache them.
    
    Returns:
        Dictionary of likelihood parameters, or None if not available
    """
    global _ARTIFACT_LOADER, _LIKELIHOOD_PARAMS

    if _LIKELIHOOD_PARAMS is not None:
        return _LIKELIHOOD_PARAMS

    loader = get_artifact_loader()
    likelihood = loader.load_likelihood_params()

    _ARTIFACT_LOADER = loader
    _LIKELIHOOD_PARAMS = likelihood

    if likelihood is not None:
        print(f"[authority] Loaded likelihood params from {loader.likelihood_params_path}")
    else:
        print(f"[authority] No likelihood params available")

    return likelihood


class OperationType(str, Enum):
    """Operation types for scoped key derivation."""
    GENERATION = "generation"
    DETECTION = "detection"


def derive_scoped_key(
    master_key: str,
    key_id: str,
    operation: OperationType,
    request_id: Optional[str] = None,
) -> str:
    """
    Derive a scoped ephemeral key from the master key.
    
    SECURITY:
    - Master key never leaves the API boundary
    - Workers only receive derived keys
    - Derived keys are scoped to specific operations
    
    Args:
        master_key: The master key (hex string)
        key_id: Public key identifier
        operation: Operation type (generation or detection)
        request_id: Optional request ID for logging
        
    Returns:
        64-character hex derived key
    """
    context = f"watermark_derived_key_v1:{operation.value}:{key_id}"
    
    # HKDF-like construction with HMAC-SHA256
    extract_key = hmac.new(
        key=b"watermark_extract_salt_v1",
        msg=bytes.fromhex(master_key),
        digestmod=hashlib.sha256,
    ).digest()
    
    derived_bytes = hmac.new(
        key=extract_key,
        msg=context.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    
    return derived_bytes.hex()


class Authority:
    """
    Watermark authority for key management and policy.
    
    Responsibilities:
    - Key validation
    - Scoped key derivation
    - Embedding configuration
    - Detection configuration
    """
    
    # Default embedding configuration (seed-bias watermarking)
    DEFAULT_EMBEDDING_CONFIG = {
        "lambda_strength": 0.075,
        "domain": "frequency",
        "low_freq_cutoff": 0.05,
        "high_freq_cutoff": 0.4,
    }
    
    # Default detection configuration
    DEFAULT_DETECTION_CONFIG = {
        "detector_type": "bayesian",
        "threshold": 0.5,
        "prior_watermarked": 0.5,
    }
    
    # Fallback G-field configuration (used when WATERMARK_CONFIG_PATH is not set)
    FALLBACK_G_FIELD_CONFIG = {
        "mapping_mode": "continuous",
        "continuous_range": [-0.6, 0.6],
        "domain": "frequency",
        "frequency_mode": "bandpass",
        "low_freq_cutoff": 0.08,
        "high_freq_cutoff": 0.32,
        "normalize_zero_mean": True,
        "normalize_unit_variance": False,
    }
    
    # Cached g_field_config loaded from YAML (None = not yet attempted)
    _loaded_g_field_config: Optional[Dict[str, Any]] = None
    _g_field_config_loaded: bool = False
    
    # Default inversion configuration
    # NOTE: Keep this aligned with the train–detect pipeline to avoid
    # distribution shift between precompute/training and API detection.
    DEFAULT_INVERSION_CONFIG = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
    
    @classmethod
    def load_g_field_config(cls) -> Dict[str, Any]:
        """
        Load g_field_config from the YAML config file specified by the
        WATERMARK_CONFIG_PATH environment variable.
        
        Falls back to FALLBACK_G_FIELD_CONFIG if the env var is not set
        or loading fails, logging a warning.
        
        Returns:
            Dictionary with g_field configuration parameters.
        """
        if cls._g_field_config_loaded:
            return cls._loaded_g_field_config if cls._loaded_g_field_config is not None else cls.FALLBACK_G_FIELD_CONFIG.copy()
        
        cls._g_field_config_loaded = True
        
        config_path = os.environ.get(WATERMARK_CONFIG_PATH_ENV)
        if config_path is None:
            logger.warning(
                f"{WATERMARK_CONFIG_PATH_ENV} not set. "
                f"Using hardcoded FALLBACK_G_FIELD_CONFIG defaults. "
                f"Set {WATERMARK_CONFIG_PATH_ENV} to the experiment YAML to ensure "
                f"train-detect parity."
            )
            return cls.FALLBACK_G_FIELD_CONFIG.copy()
        
        try:
            from src.core.config import AppConfig
            from src.detection.g_values import g_field_config_to_dict
            
            app_config = AppConfig.from_yaml(config_path)
            g_field = app_config.watermark.algorithm_params.g_field
            loaded = g_field_config_to_dict(g_field)
            cls._loaded_g_field_config = loaded
            
            logger.info(
                f"Loaded g_field_config from {config_path}: {loaded}"
            )
            return loaded
        except Exception as e:
            logger.warning(
                f"Failed to load g_field_config from {config_path}: {e}. "
                f"Falling back to FALLBACK_G_FIELD_CONFIG defaults."
            )
            return cls.FALLBACK_G_FIELD_CONFIG.copy()
    
    def __init__(self):
        """Initialize authority."""
        self.key_store = get_key_store()
    
    def validate_key(self, key_id: str) -> bool:
        """
        Validate that a key exists and is active.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if valid and active
        """
        return self.key_store.is_active(key_id)
    
    def get_generation_payload(
        self,
        key_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Get payload for generation request to GPU worker.
        
        ARCHITECTURAL REQUIREMENT: Returns master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        
        Args:
            key_id: Key identifier (public PRF index)
            request_id: Request ID for tracing
            
        Returns:
            Dictionary with master_key, key_id, fingerprint, embedding_config
            
        Raises:
            ValueError: If key not found or inactive
        """
        master_key = self.key_store.get_master_key(key_id)
        if master_key is None:
            raise ValueError(f"Key {key_id} not found or inactive")
        
        fingerprint = self.key_store.get_fingerprint(key_id)
        
        return {
            "key_id": key_id,
            "master_key": master_key,
            "key_fingerprint": fingerprint,
            "embedding_config": self.DEFAULT_EMBEDDING_CONFIG.copy(),
        }
    
    def _get_likelihood_path(self) -> Optional[str]:
        """Get path to likelihood params file."""
        from service.api.config import get_config
        config = get_config()
        return config.likelihood_params_path
    
    def _get_normalization_path(self) -> Optional[str]:
        """Get path to normalization params file."""
        from service.api.config import get_config
        config = get_config()
        return config.normalization_params_path
    
    def _get_calibration_path(self) -> Optional[str]:
        """Get path to calibration params file."""
        from service.api.config import get_config
        config = get_config()
        return config.calibration_params_path
    
    def get_detection_payload(
        self,
        key_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Get payload for detection request to GPU worker.
        
        ARCHITECTURAL REQUIREMENT: Returns master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        compute_g_values() uses (master_key, key_id) directly.
        
        Args:
            key_id: Key identifier (public PRF index)
            request_id: Request ID for tracing
            
        Returns:
            Dictionary with master_key, key_id, fingerprint, detection configs
            
        Raises:
            ValueError: If key not found or inactive
        """
        master_key = self.key_store.get_master_key(key_id)
        if master_key is None:
            raise ValueError(f"Key {key_id} not found or inactive")
        
        fingerprint = self.key_store.get_fingerprint(key_id)
        
        # Build detection config with all artifact paths (not the params themselves)
        detection_config = self.DEFAULT_DETECTION_CONFIG.copy()
        
        # Add likelihood params path
        likelihood_params_path = self._get_likelihood_path()
        if likelihood_params_path is not None:
            detection_config["likelihood_params_path"] = likelihood_params_path
        
        # Add normalization params path
        normalization_params_path = self._get_normalization_path()
        if normalization_params_path is not None:
            detection_config["normalization_params_path"] = normalization_params_path
        
        # Add calibration params path
        calibration_params_path = self._get_calibration_path()
        if calibration_params_path is not None:
            detection_config["calibration_params_path"] = calibration_params_path
        
        return {
            "key_id": key_id,
            "master_key": master_key,
            "key_fingerprint": fingerprint,
            "g_field_config": self.load_g_field_config(),
            "detection_config": detection_config,
            "inversion_config": self.DEFAULT_INVERSION_CONFIG.copy(),
        }
    
    @staticmethod
    def compute_policy_version(
        embedding_config: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Compute deterministic policy version from configurations.
        
        Args:
            embedding_config: Embedding configuration
            detection_config: Detection configuration
            
        Returns:
            16-character policy version hash
        """
        policy_dict = {
            "embedding": dict(sorted(embedding_config.items())),
            "detection": dict(sorted(detection_config.items())),
        }
        policy_json = json.dumps(policy_dict, sort_keys=True, separators=(',', ':'))
        hash_hex = hashlib.sha256(policy_json.encode('utf-8')).hexdigest()
        return hash_hex[:16]


# Global authority instance
_authority: Optional[Authority] = None


def get_authority() -> Authority:
    """Get the global authority instance."""
    global _authority
    if _authority is None:
        _authority = Authority()
    return _authority


def reset_authority() -> None:
    """Reset the global authority (useful for testing)."""
    global _authority
    _authority = None
