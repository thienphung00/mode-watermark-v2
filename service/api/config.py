"""
Environment configuration for API service.

Loads configuration from environment variables with sensible defaults
for local development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """API service configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # GPU Worker settings
    gpu_worker_url: str = "http://localhost:8001"
    gpu_worker_timeout: float = 120.0  # seconds
    
    # Storage settings
    storage_backend: str = "local"  # "local" or "gcs"
    storage_path: str = "./data/images"
    gcs_bucket: Optional[str] = None
    
    # Key store settings
    key_store_path: str = "./data/keys.json"
    
    # Artifacts settings
    artifacts_path: str = "./data/artifacts"
    likelihood_params_path: Optional[str] = None
    normalization_params_path: Optional[str] = None
    calibration_params_path: Optional[str] = None
    mask_path: Optional[str] = None
    
    # Security
    encryption_key: str = "development-key-not-for-production"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            
            gpu_worker_url=os.getenv("GPU_WORKER_URL", "http://localhost:8001"),
            gpu_worker_timeout=float(os.getenv("GPU_WORKER_TIMEOUT", "120.0")),
            
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            storage_path=os.getenv("STORAGE_PATH", "./data/images"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
            
            key_store_path=os.getenv("KEY_STORE_PATH", "./data/keys.json"),
            
            artifacts_path=os.getenv("ARTIFACTS_PATH", "./data/artifacts"),
            likelihood_params_path=os.getenv("LIKELIHOOD_PARAMS_PATH"),
            normalization_params_path=os.getenv("NORMALIZATION_PARAMS_PATH"),
            calibration_params_path=os.getenv("CALIBRATION_PARAMS_PATH"),
            mask_path=os.getenv("MASK_PATH"),
            
            encryption_key=os.getenv("ENCRYPTION_KEY", "development-key-not-for-production"),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
