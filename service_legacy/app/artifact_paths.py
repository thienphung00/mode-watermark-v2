"""
Artifact path resolution and validation utilities.

Provides robust path resolution for detection artifacts that works
regardless of project layout, Docker mounts, or external artifact locations.
"""
from __future__ import annotations

from service.infra.logging import get_logger
import os
from pathlib import Path
from typing import Optional

logger = get_logger(__name__)

# Module-level cache for resolved paths (set at startup)
_resolved_likelihood_params_path: Optional[Path] = None
_resolved_mask_path: Optional[Path] = None


def resolve_artifact_path(
    env_var_name: str,
    description: str = "artifact",
    required: bool = False,
) -> Optional[Path]:
    """
    Resolve and validate an artifact path from environment variable.
    
    This function:
    1. Reads the path from environment variable
    2. Resolves it to an absolute path
    3. Validates that the file exists and is readable
    4. Logs the resolved absolute path
    
    Args:
        env_var_name: Name of environment variable containing the path
        description: Human-readable description of the artifact (for logging)
        required: If True, raises ValueError if path is not set or invalid
    
    Returns:
        Resolved absolute Path if valid, None if not set and not required
    
    Raises:
        ValueError: If required=True and path is not set or invalid
        FileNotFoundError: If path is set but file doesn't exist
        PermissionError: If path is set but file is not readable
    """
    raw_path = os.getenv(env_var_name)
    
    if not raw_path:
        if required:
            raise ValueError(
                f"{env_var_name} is required but not set. "
                f"Please set {env_var_name} to the absolute path of the {description}."
            )
        logger.debug(f"{env_var_name} not set, {description} will not be loaded")
        return None
    
    # Resolve to absolute path (handles relative paths, symlinks, etc.)
    try:
        resolved_path = Path(raw_path).resolve()
    except Exception as e:
        if required:
            raise ValueError(
                f"Failed to resolve {description} path '{raw_path}': {e}"
            ) from e
        logger.warning(f"Failed to resolve {description} path '{raw_path}': {e}")
        return None
    
    # Validate file exists
    if not resolved_path.exists():
        if required:
            raise FileNotFoundError(
                f"{description.capitalize()} not found at resolved path: {resolved_path}\n"
                f"  Original path: {raw_path}\n"
                f"  Environment variable: {env_var_name}\n"
                f"  Please ensure the file exists and the path is correct."
            )
        logger.warning(
            f"{description.capitalize()} not found at resolved path: {resolved_path} "
            f"(original: {raw_path})"
        )
        return None
    
    # Validate file is readable
    if not os.access(resolved_path, os.R_OK):
        if required:
            raise PermissionError(
                f"{description.capitalize()} is not readable: {resolved_path}\n"
                f"  Please check file permissions."
            )
        logger.warning(f"{description.capitalize()} is not readable: {resolved_path}")
        return None
    
    # Log successful resolution
    logger.info(
        f"✓ Resolved {description} path:\n"
        f"  Environment variable: {env_var_name}\n"
        f"  Original path: {raw_path}\n"
        f"  Resolved absolute path: {resolved_path}"
    )
    
    return resolved_path


def get_likelihood_params_path(use_cache: bool = True) -> Optional[Path]:
    """
    Get resolved likelihood parameters path from LIKELIHOOD_PARAMS_PATH.
    
    Args:
        use_cache: If True, use cached resolved path from startup (default: True)
    
    Returns:
        Resolved absolute Path if valid, None if not set
    
    Raises:
        ValueError: If path is set but invalid
        FileNotFoundError: If path is set but file doesn't exist
        PermissionError: If path is set but file is not readable
    """
    global _resolved_likelihood_params_path
    
    if use_cache and _resolved_likelihood_params_path is not None:
        return _resolved_likelihood_params_path
    
    resolved = resolve_artifact_path(
        env_var_name="LIKELIHOOD_PARAMS_PATH",
        description="likelihood parameters",
        required=False,
    )
    
    if resolved is not None:
        _resolved_likelihood_params_path = resolved
    
    return resolved


def get_mask_path(use_cache: bool = True) -> Optional[Path]:
    """
    Get resolved mask path from MASK_PATH.
    
    Args:
        use_cache: If True, use cached resolved path from startup (default: True)
    
    Returns:
        Resolved absolute Path if valid, None if not set
    
    Raises:
        ValueError: If path is set but invalid
        FileNotFoundError: If path is set but file doesn't exist
        PermissionError: If path is set but file is not readable
    """
    global _resolved_mask_path
    
    if use_cache and _resolved_mask_path is not None:
        return _resolved_mask_path
    
    resolved = resolve_artifact_path(
        env_var_name="MASK_PATH",
        description="mask tensor",
        required=False,
    )
    
    if resolved is not None:
        _resolved_mask_path = resolved
    
    return resolved


def get_resolved_likelihood_params_path_str() -> Optional[str]:
    """
    Get resolved likelihood parameters path as string (for use in configs).
    
    This returns the cached resolved absolute path from startup.
    Use this in authority.py to get the resolved path that was validated at startup.
    
    Returns:
        Resolved absolute path as string, or None if not set
    """
    global _resolved_likelihood_params_path
    if _resolved_likelihood_params_path is None:
        # Try to resolve if not cached (shouldn't happen if startup validation ran)
        _resolved_likelihood_params_path = get_likelihood_params_path(use_cache=False)
    return str(_resolved_likelihood_params_path) if _resolved_likelihood_params_path else None


def get_resolved_mask_path_str() -> Optional[str]:
    """
    Get resolved mask path as string (for use in configs).
    
    This returns the cached resolved absolute path from startup.
    Use this in authority.py to get the resolved path that was validated at startup.
    
    Returns:
        Resolved absolute path as string, or None if not set
    """
    global _resolved_mask_path
    if _resolved_mask_path is None:
        # Try to resolve if not cached (shouldn't happen if startup validation ran)
        _resolved_mask_path = get_mask_path(use_cache=False)
    return str(_resolved_mask_path) if _resolved_mask_path else None

