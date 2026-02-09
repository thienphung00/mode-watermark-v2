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
    - normalization_params: Score normalization parameters (mean, std)
    - calibration_params: Calibrated detection threshold (fixed FPR)
    - mask: Region mask for detection
    """
    
    def __init__(
        self,
        artifacts_path: str = "./data/artifacts",
        likelihood_params_path: Optional[str] = None,
        normalization_params_path: Optional[str] = None,
        calibration_params_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ):
        """
        Initialize artifact loader.
        
        Args:
            artifacts_path: Base path for artifacts
            likelihood_params_path: Override path for likelihood params
            normalization_params_path: Override path for normalization params
            calibration_params_path: Override path for calibration params
            mask_path: Override path for mask
        """
        self.artifacts_path = Path(artifacts_path)
        self._likelihood_params_path = likelihood_params_path
        self._normalization_params_path = normalization_params_path
        self._calibration_params_path = calibration_params_path
        self._mask_path = mask_path
        
        # Cached artifacts
        self._likelihood_params: Optional[Dict[str, Any]] = None
        self._normalization_params: Optional[Dict[str, Any]] = None
        self._calibration_params: Optional[Dict[str, Any]] = None
        self._mask: Optional[np.ndarray] = None
    
    @property
    def likelihood_params_path(self) -> Optional[Path]:
        """Get resolved path to likelihood params."""
        if self._likelihood_params_path:
            return Path(self._likelihood_params_path)
        default = self.artifacts_path / "likelihood_params.json"
        return default if default.exists() else None
    
    @property
    def normalization_params_path(self) -> Optional[Path]:
        """Get resolved path to normalization params."""
        if self._normalization_params_path:
            return Path(self._normalization_params_path)
        default = self.artifacts_path / "normalization_098.json"
        return default if default.exists() else None
    
    @property
    def calibration_params_path(self) -> Optional[Path]:
        """Get resolved path to calibration params."""
        if self._calibration_params_path:
            return Path(self._calibration_params_path)
        default = self.artifacts_path / "calibration_098.json"
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
    
    def load_normalization_params(self) -> Optional[Dict[str, Any]]:
        """
        Load normalization parameters for score standardization.
        
        Returns:
            Dictionary with 'mean' and 'std' for score normalization, or None if not available
        """
        if self._normalization_params is not None:
            return self._normalization_params
        
        path = self.normalization_params_path
        if path is None:
            logger.error("Normalization params path not set - NORMALIZATION_PARAMS_PATH env var required")
            return None
        
        if not path.exists():
            logger.error(f"Normalization params not found at {path}")
            return None
        
        try:
            with open(path, "r") as f:
                self._normalization_params = json.load(f)
            logger.info(f"Loaded normalization params from {path}")
            return self._normalization_params
        except Exception as e:
            logger.error(f"Failed to load normalization params: {e}")
            return None
    
    def load_calibration_params(self) -> Optional[Dict[str, Any]]:
        """
        Load calibration parameters for deployment threshold.
        
        Returns:
            Dictionary with 'deployment_threshold' and 'target_fpr', or None if not available
        """
        if self._calibration_params is not None:
            return self._calibration_params
        
        path = self.calibration_params_path
        if path is None:
            logger.error("Calibration params path not set - CALIBRATION_PARAMS_PATH env var required")
            return None
        
        if not path.exists():
            logger.error(f"Calibration params not found at {path}")
            return None
        
        try:
            with open(path, "r") as f:
                self._calibration_params = json.load(f)
            logger.info(f"Loaded calibration params from {path}")
            return self._calibration_params
        except Exception as e:
            logger.error(f"Failed to load calibration params: {e}")
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
            "normalization_params_path": str(self.normalization_params_path) if self.normalization_params_path else None,
            "calibration_params_path": str(self.calibration_params_path) if self.calibration_params_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "artifacts_available": self.is_ready(),
        }
    
    def is_ready(self) -> bool:
        """
        Check if essential artifacts are available.
        
        For calibrated Bayesian detection, all three artifacts are required:
        - likelihood_params
        - normalization_params
        - calibration_params
        """
        # In stub mode, artifacts are optional
        # In full detection mode, all three are required
        has_likelihood = self.likelihood_params_path is not None and self.likelihood_params_path.exists()
        has_normalization = self.normalization_params_path is not None and self.normalization_params_path.exists()
        has_calibration = self.calibration_params_path is not None and self.calibration_params_path.exists()
        
        # If any are configured, all must be configured
        if has_likelihood or has_normalization or has_calibration:
            if not (has_likelihood and has_normalization and has_calibration):
                logger.warning(
                    f"Incomplete artifact configuration: "
                    f"likelihood={has_likelihood}, normalization={has_normalization}, calibration={has_calibration}. "
                    f"All three are required for calibrated detection."
                )
                return False
        
        return True
    
    def clear_cache(self) -> None:
        """Clear cached artifacts."""
        self._likelihood_params = None
        self._normalization_params = None
        self._calibration_params = None
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
            normalization_params_path=config.normalization_params_path,
            calibration_params_path=config.calibration_params_path,
            mask_path=config.mask_path,
        )
    return _artifact_loader


def reset_artifact_loader() -> None:
    """Reset the global artifact loader (useful for testing)."""
    global _artifact_loader
    _artifact_loader = None
