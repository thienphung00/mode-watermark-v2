"""
Detector Artifacts Loader.

Loads and caches immutable artifacts required for Bayesian detection:
- likelihood_params.json (trained likelihood parameters)
- mask tensor (structural mask geometry)
- g-field config (G-field generation parameters)

These artifacts are:
- Read-only (never modified)
- Loaded once at startup or first request
- Cached in memory
- Validated for consistency (config hash, mask shape)
"""
from __future__ import annotations

import hashlib
import json
from service.infra.logging import get_logger
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = get_logger(__name__)


class DetectorArtifacts:
    """
    Loader and cache for detector artifacts.
    
    This class provides a clean abstraction for loading and validating
    the immutable artifacts required for Bayesian detection:
    - likelihood_params.json: Trained likelihood parameters
    - mask: Structural mask tensor (geometry)
    - g_field_config: G-field configuration dict
    
    Artifacts are validated for consistency:
    - Config hash must match likelihood metadata
    - Mask shape must match likelihood num_positions
    """
    
    def __init__(
        self,
        likelihood_params_path: str,
        mask_path: Optional[str] = None,
        g_field_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize detector artifacts loader.
        
        Args:
            likelihood_params_path: Path to likelihood_params.json (should be absolute)
            mask_path: Optional path to mask tensor file (.pt) (should be absolute)
            g_field_config: G-field configuration dict (required for config hash)
        
        Raises:
            FileNotFoundError: If likelihood_params_path doesn't exist
            ValueError: If config hash mismatch or mask shape mismatch
        """
        # Resolve to absolute paths (handles relative paths, symlinks, etc.)
        # Paths should already be absolute from startup validation, but normalize here for safety
        self.likelihood_params_path = Path(likelihood_params_path).resolve()
        self.mask_path = Path(mask_path).resolve() if mask_path else None
        self.g_field_config = g_field_config
        
        # Validate paths
        if not self.likelihood_params_path.exists():
            raise FileNotFoundError(
                f"Likelihood parameters not found: {self.likelihood_params_path}\n"
                f"  Original path: {likelihood_params_path}\n"
                f"  Resolved absolute path: {self.likelihood_params_path}"
            )
        
        if self.mask_path and not self.mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {self.mask_path}\n"
                f"  Original path: {mask_path}\n"
                f"  Resolved absolute path: {self.mask_path}"
            )
        
        # Load artifacts
        self._load_artifacts()
        
        # Validate consistency
        self._validate_consistency()
    
    def _load_artifacts(self) -> None:
        """Load all artifacts from disk."""
        # Load likelihood parameters
        with open(self.likelihood_params_path, "r") as f:
            self.likelihood_data = json.load(f)
        
        self.num_positions = self.likelihood_data.get("num_positions")
        self.probs_watermarked = torch.tensor(
            self.likelihood_data["watermarked"]["probs"],
            dtype=torch.float32,
        )
        self.probs_unwatermarked = torch.tensor(
            self.likelihood_data["unwatermarked"]["probs"],
            dtype=torch.float32,
        )
        
        # Load mask if provided
        self.mask = None
        if self.mask_path:
            self.mask = torch.load(self.mask_path, map_location="cpu")
            # Ensure mask is 1D
            if self.mask.dim() > 1:
                self.mask = self.mask.flatten()
            self.mask = (self.mask > 0.5).float()
        
        # Extract config hash from likelihood metadata if present
        self.config_hash_from_likelihood = self.likelihood_data.get(
            "g_field_config_hash"
        )
        
        logger.info(
            f"Loaded artifacts: num_positions={self.num_positions}, "
            f"mask_shape={list(self.mask.shape) if self.mask is not None else None}"
        )
    
    def _validate_consistency(self) -> None:
        """
        Validate artifact consistency.
        
        Checks:
        1. Config hash matches (if both are provided)
        2. Mask shape matches num_positions (if mask is provided)
        
        Raises:
            ValueError: If validation fails
        """
        # Check 1: Config hash consistency
        if self.g_field_config and self.config_hash_from_likelihood:
            computed_hash = self._compute_config_hash(self.g_field_config)
            if computed_hash != self.config_hash_from_likelihood:
                raise ValueError(
                    f"Config hash mismatch!\n"
                    f"  Likelihood metadata hash: {self.config_hash_from_likelihood}\n"
                    f"  Computed hash: {computed_hash}\n"
                    f"  This indicates the g-field config used for detection "
                    f"does not match the config used during training."
                )
            logger.info(f"Config hash validated: {computed_hash}")
        
        # Check 2: Mask shape consistency
        if self.mask is not None and self.num_positions is not None:
            mask_sum = int(self.mask.sum().item())
            if mask_sum != self.num_positions:
                raise ValueError(
                    f"Mask shape mismatch!\n"
                    f"  mask.sum()={mask_sum}\n"
                    f"  likelihood num_positions={self.num_positions}\n"
                    f"  This indicates the mask used for detection does not match "
                    f"the mask used during training."
                )
            logger.info(
                f"Mask shape validated: mask.sum()={mask_sum} == num_positions={self.num_positions}"
            )
    
    @staticmethod
    def _compute_config_hash(g_field_config: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of g-field config.
        
        Args:
            g_field_config: G-field configuration dict
        
        Returns:
            16-character hex hash string
        """
        json_str = json.dumps(g_field_config, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    @property
    def config_hash(self) -> Optional[str]:
        """Get config hash (computed from g_field_config if available)."""
        if self.g_field_config:
            return self._compute_config_hash(self.g_field_config)
        return self.config_hash_from_likelihood
    
    @property
    def num_positions(self) -> Optional[int]:
        """Get number of positions from likelihood model."""
        return self._num_positions
    
    @num_positions.setter
    def num_positions(self, value: Optional[int]) -> None:
        """Set number of positions."""
        self._num_positions = value

