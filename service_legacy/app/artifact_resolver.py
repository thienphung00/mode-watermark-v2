"""
Centralized artifact resolution and availability tracking.

This module provides a single source of truth for detection artifact resolution:
- Environment variable resolution (LIKELIHOOD_PARAMS_PATH, MASK_PATH)
- Startup-time validation and caching
- Availability state tracking
- No path guessing or fallback searching

This replaces scattered path resolution logic throughout the codebase.
"""
from __future__ import annotations

import json
from service.infra.logging import get_logger
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = get_logger(__name__)


@dataclass
class ArtifactResolutionResult:
    """Result of artifact resolution attempt."""
    
    likelihood_params_path: Optional[Path]
    mask_path: Optional[Path]
    is_available: bool
    error_message: Optional[str] = None
    
    @property
    def likelihood_params_path_str(self) -> Optional[str]:
        """Get likelihood params path as string."""
        return str(self.likelihood_params_path) if self.likelihood_params_path else None
    
    @property
    def mask_path_str(self) -> Optional[str]:
        """Get mask path as string."""
        return str(self.mask_path) if self.mask_path else None


class ArtifactResolver:
    """
    Centralized artifact resolution with startup validation and caching.
    
    This class:
    - Resolves artifact paths from environment variables ONLY
    - Validates paths exist and are readable at startup
    - Caches resolved paths and availability state
    - Provides clear error messages for missing artifacts
    - Never guesses paths or searches filesystem
    
    Usage:
        resolver = ArtifactResolver()
        result = resolver.resolve()  # Called once at startup
        if result.is_available:
            # Use result.likelihood_params_path, result.mask_path
        else:
            # Handle missing artifacts gracefully
    """
    
    def __init__(self):
        """Initialize artifact resolver."""
        self._cached_result: Optional[ArtifactResolutionResult] = None
        self._initialized = False
    
    def resolve(self, force_refresh: bool = False) -> ArtifactResolutionResult:
        """
        Resolve artifact paths from environment variables.
        
        This method:
        1. Reads LIKELIHOOD_PARAMS_PATH and MASK_PATH environment variables
        2. Resolves paths to absolute paths
        3. Validates files exist and are readable
        4. Validates JSON schema for likelihood_params.json
        5. Caches result for subsequent calls
        
        Args:
            force_refresh: If True, force re-resolution (default: False)
        
        Returns:
            ArtifactResolutionResult with resolved paths and availability status
        
        Note:
            This should be called once at startup. Subsequent calls return cached result.
        """
        if self._cached_result is not None and not force_refresh:
            return self._cached_result
        
        self._initialized = True
        
        # Resolve likelihood params path
        likelihood_params_path = self._resolve_likelihood_params_path()
        
        # Resolve mask path (optional)
        mask_path = self._resolve_mask_path()
        
        # Determine availability
        if likelihood_params_path is None:
            error_msg = (
                "Detection artifacts not configured. "
                "Set LIKELIHOOD_PARAMS_PATH environment variable to the absolute path of likelihood_params.json. "
                "For Docker: mount outputs directory or set env var: docker run -e LIKELIHOOD_PARAMS_PATH=/path/to/likelihood_params.json ..."
            )
            result = ArtifactResolutionResult(
                likelihood_params_path=None,
                mask_path=None,
                is_available=False,
                error_message=error_msg,
            )
        else:
            # Validate likelihood_params.json schema
            try:
                self._validate_likelihood_params(likelihood_params_path)
                result = ArtifactResolutionResult(
                    likelihood_params_path=likelihood_params_path,
                    mask_path=mask_path,
                    is_available=True,
                )
            except Exception as e:
                error_msg = (
                    f"Likelihood params file is invalid: {e}\n"
                    f"  Path: {likelihood_params_path}\n"
                    f"  Please ensure the file is a valid JSON with expected schema."
                )
                result = ArtifactResolutionResult(
                    likelihood_params_path=likelihood_params_path,
                    mask_path=mask_path,
                    is_available=False,
                    error_message=error_msg,
                )
        
        # Cache result
        self._cached_result = result
        
        # Log result
        if result.is_available:
            logger.info(
                f"✓ Artifact resolution successful:\n"
                f"  Likelihood params: {result.likelihood_params_path}\n"
                f"  Mask: {result.mask_path_str or 'not set (optional)'}"
            )
        else:
            logger.error(f"✗ Artifact resolution failed: {result.error_message}")
        
        return result
    
    def _resolve_likelihood_params_path(self) -> Optional[Path]:
        """Resolve likelihood params path from LIKELIHOOD_PARAMS_PATH env var."""
        raw_path = os.getenv("LIKELIHOOD_PARAMS_PATH")
        
        if not raw_path:
            logger.debug("LIKELIHOOD_PARAMS_PATH not set")
            return None
        
        try:
            resolved_path = Path(raw_path).resolve()
        except Exception as e:
            logger.warning(f"Failed to resolve LIKELIHOOD_PARAMS_PATH '{raw_path}': {e}")
            return None
        
        if not resolved_path.exists():
            logger.warning(f"Likelihood params not found: {resolved_path} (original: {raw_path})")
            return None
        
        if not os.access(resolved_path, os.R_OK):
            logger.warning(f"Likelihood params not readable: {resolved_path}")
            return None
        
        logger.info(
            f"✓ Resolved likelihood params path:\n"
            f"  Env var: LIKELIHOOD_PARAMS_PATH\n"
            f"  Original: {raw_path}\n"
            f"  Resolved: {resolved_path}"
        )
        
        return resolved_path
    
    def _resolve_mask_path(self) -> Optional[Path]:
        """Resolve mask path from MASK_PATH env var (optional)."""
        raw_path = os.getenv("MASK_PATH")
        
        if not raw_path:
            logger.debug("MASK_PATH not set (optional)")
            return None
        
        try:
            resolved_path = Path(raw_path).resolve()
        except Exception as e:
            logger.warning(f"Failed to resolve MASK_PATH '{raw_path}': {e}")
            return None
        
        if not resolved_path.exists():
            logger.warning(f"Mask not found: {resolved_path} (original: {raw_path})")
            return None
        
        if not os.access(resolved_path, os.R_OK):
            logger.warning(f"Mask not readable: {resolved_path}")
            return None
        
        logger.info(
            f"✓ Resolved mask path:\n"
            f"  Env var: MASK_PATH\n"
            f"  Original: {raw_path}\n"
            f"  Resolved: {resolved_path}"
        )
        
        return resolved_path
    
    def _validate_likelihood_params(self, path: Path) -> None:
        """
        Validate likelihood_params.json schema.
        
        Args:
            path: Path to likelihood_params.json
        
        Raises:
            ValueError: If file is invalid JSON or missing required fields
            json.JSONDecodeError: If file is not valid JSON
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        
        # Basic schema validation (ensure it's a dict/object)
        if not isinstance(data, dict):
            raise ValueError("Likelihood params must be a JSON object/dict")
        
        # Log validation success
        logger.debug(f"Likelihood params schema validation passed for {path}")
    
    def get_availability_status(self) -> dict:
        """
        Get current artifact availability status (for diagnostics).
        
        Returns:
            Dictionary with availability status and paths
        """
        if not self._initialized:
            return {
                "initialized": False,
                "is_available": False,
                "error": "Artifact resolver not initialized. Call resolve() first.",
            }
        
        result = self.resolve()
        return {
            "initialized": True,
            "is_available": result.is_available,
            "likelihood_params_path": result.likelihood_params_path_str,
            "mask_path": result.mask_path_str,
            "error": result.error_message,
        }


# Global singleton instance
_global_resolver: Optional[ArtifactResolver] = None


def get_artifact_resolver() -> ArtifactResolver:
    """
    Get global ArtifactResolver singleton.
    
    Returns:
        ArtifactResolver instance
    """
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = ArtifactResolver()
    return _global_resolver

