"""
Artifact loader and cache management.

Handles loading of model artifacts:
- Likelihood parameters (for Bayesian detection)
- Masks (for watermark regions)
- Other calibration data
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """
    Loader for detection artifacts.
    
    Artifacts include:
    - likelihood_params: Parameters for Bayesian likelihood computation
    - mask: Region mask for detection
    """
    
    def __init__(
        self,
        artifacts_path: str = "./data/artifacts",
        likelihood_params_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ):
        """
        Initialize artifact loader.
        
        Args:
            artifacts_path: Base path for artifacts
            likelihood_params_path: Override path for likelihood params
            mask_path: Override path for mask
        """
        self.artifacts_path = Path(artifacts_path)
        self._likelihood_params_path = likelihood_params_path
        self._mask_path = mask_path
        
        # Cached artifacts
        self._likelihood_params: Optional[Dict[str, Any]] = None
        self._mask: Optional[np.ndarray] = None
    
    @property
    def likelihood_params_path(self) -> Optional[Path]:
        """Get resolved path to likelihood params."""
        if self._likelihood_params_path:
            return Path(self._likelihood_params_path)
        default = self.artifacts_path / "likelihood_params.json"
        return default if default.exists() else None
    
    @property
    def mask_path(self) -> Optional[Path]:
        """Get resolved path to mask."""
        if self._mask_path:
            return Path(self._mask_path)
        default = self.artifacts_path / "mask.npy"
        return default if default.exists() else None
    
    def load_likelihood_params(self) -> Optional[Dict[str, Any]]:
        """
        Load likelihood parameters.
        
        Returns:
            Dictionary of likelihood parameters, or None if not available
        """
        if self._likelihood_params is not None:
            return self._likelihood_params
        
        path = self.likelihood_params_path
        if path is None:
            logger.warning("Likelihood params path not set")
            return None
        # Allow path without .json extension (e.g. best_model -> best_model.json)
        if not path.exists() and path.suffix != ".json":
            path_json = path.with_suffix(".json")
            if path_json.exists():
                path = path_json
        if not path.exists():
            logger.warning(f"Likelihood params not found at {path}")
            return None

        try:
            with open(path, "r") as f:
                self._likelihood_params = json.load(f)
            logger.info(f"Loaded likelihood params from {path}")
            return self._likelihood_params
        except Exception as e:
            logger.error(f"Failed to load likelihood params: {e}")
            return None
    
    def load_mask(self) -> Optional[np.ndarray]:
        """
        Load detection mask.
        
        Returns:
            Numpy array mask, or None if not available
        """
        if self._mask is not None:
            return self._mask
        
        path = self.mask_path
        if path is None or not path.exists():
            logger.warning(f"Mask not found at {path}")
            return None
        
        try:
            self._mask = np.load(path)
            logger.info(f"Loaded mask from {path} (shape: {self._mask.shape})")
            return self._mask
        except Exception as e:
            logger.error(f"Failed to load mask: {e}")
            return None
    
    def get_detector_config(self) -> Dict[str, Any]:
        """
        Get detector configuration with artifact paths.
        
        Returns:
            Configuration dictionary
        """
        return {
            "likelihood_params_path": str(self.likelihood_params_path) if self.likelihood_params_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "artifacts_available": self.is_ready(),
        }
    
    def is_ready(self) -> bool:
        """Check if essential artifacts are available."""
        # For now, artifacts are optional (stub mode allowed)
        return True
    
    def clear_cache(self) -> None:
        """Clear cached artifacts."""
        self._likelihood_params = None
        self._mask = None


# Global artifact loader instance
_artifact_loader: Optional[ArtifactLoader] = None


def get_artifact_loader() -> ArtifactLoader:
    """Get the global artifact loader instance."""
    global _artifact_loader
    if _artifact_loader is None:
        from service.api.config import get_config
        config = get_config()
        _artifact_loader = ArtifactLoader(
            artifacts_path=config.artifacts_path,
            likelihood_params_path=config.likelihood_params_path,
            mask_path=config.mask_path,
        )
    return _artifact_loader


def reset_artifact_loader() -> None:
    """Reset the global artifact loader (useful for testing)."""
    global _artifact_loader
    _artifact_loader = None
