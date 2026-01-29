"""
Centralized configuration management using Pydantic Settings.

Provides:
- Environment variable loading with validation
- Secret management with SecretStr
- Feature flags
- Deployment-specific configuration

Configuration Sources (in order of precedence):
1. Environment variables
2. .env file
3. Default values

Usage:
    from service.infra.settings import get_settings
    
    settings = get_settings()
    print(settings.ENVIRONMENT)
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings are loaded from environment variables.
    Secrets use SecretStr to prevent accidental logging.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )
    
    # =========================================================================
    # Environment
    # =========================================================================
    
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="production",
        description="Deployment environment"
    )
    SERVICE_VERSION: str = Field(
        default="1.0.0",
        description="Service version"
    )
    
    # =========================================================================
    # Database
    # =========================================================================
    
    DATABASE_URL: Optional[SecretStr] = Field(
        default=None,
        description="PostgreSQL connection URL (not required for JSON file storage)"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    DATABASE_POOL_MAX_OVERFLOW: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Max overflow connections"
    )
    
    # JSON file storage (for development/testing)
    STORAGE_PATH: str = Field(
        default="service_data/watermark_keys.json",
        description="Path to JSON file storage (used when DATABASE_URL not set)"
    )
    
    # =========================================================================
    # Encryption
    # =========================================================================
    
    ENCRYPTION_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Master encryption key for key storage (required in production)"
    )
    ENCRYPTION_KEY_ROTATION_ENABLED: bool = Field(
        default=False,
        description="Enable dual-key decryption during rotation"
    )
    ENCRYPTION_KEY_OLD: Optional[SecretStr] = Field(
        default=None,
        description="Previous encryption key for rotation"
    )
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    INFERENCE_MODE: Literal["local", "remote"] = Field(
        default="local",
        description="Inference mode: local (in-process) or remote (GPU workers)"
    )
    WORKER_URLS: list[str] = Field(
        default_factory=list,
        description="List of GPU worker URLs (required for remote mode)"
    )
    WORKER_AUTH_SECRET: Optional[SecretStr] = Field(
        default=None,
        description="Secret for worker request signing"
    )
    INFERENCE_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Inference request timeout"
    )
    
    # =========================================================================
    # Model Configuration
    # =========================================================================
    
    MODEL_ID: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="Hugging Face model ID"
    )
    DEVICE: str = Field(
        default="auto",
        description="Device for inference (cuda, mps, cpu, auto)"
    )
    USE_FP16: bool = Field(
        default=True,
        description="Use half precision on CUDA"
    )
    
    # =========================================================================
    # Detection Artifacts
    # =========================================================================
    
    LIKELIHOOD_PARAMS_PATH: Optional[str] = Field(
        default=None,
        description="Path to likelihood parameters JSON"
    )
    MASK_PATH: Optional[str] = Field(
        default=None,
        description="Path to detection mask"
    )
    
    # =========================================================================
    # Feature Flags
    # =========================================================================
    
    ENABLE_DOCS: bool = Field(
        default=False,
        description="Enable Swagger/ReDoc documentation"
    )
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    ENABLE_MICRO_BATCHING: bool = Field(
        default=True,
        description="Enable micro-batching for detection"
    )
    REGISTER_TEST_KEYS: bool = Field(
        default=False,
        description="Register test keys at startup"
    )
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    RATE_LIMIT_DETECT_MAX: int = Field(
        default=50,
        ge=1,
        description="Max detection requests per minute per IP"
    )
    RATE_LIMIT_DETECT_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Detection rate limit window in seconds"
    )
    RATE_LIMIT_EVALUATE_MAX: int = Field(
        default=20,
        ge=1,
        description="Max evaluation requests per minute per IP"
    )
    RATE_LIMIT_EVALUATE_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Evaluation rate limit window in seconds"
    )
    
    # =========================================================================
    # Logging
    # =========================================================================
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FORMAT: Literal["json", "console", "auto"] = Field(
        default="auto",
        description="Log output format"
    )
    USE_STRUCTURED_LOGGING: bool = Field(
        default=True,
        description="Use structlog for structured logging"
    )
    
    # =========================================================================
    # CORS
    # =========================================================================
    
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    
    # =========================================================================
    # Validators
    # =========================================================================
    
    @field_validator("WORKER_URLS", mode="before")
    @classmethod
    def parse_worker_urls(cls, v):
        """Parse worker URLs from comma-separated string."""
        if isinstance(v, str):
            return [url.strip() for url in v.split(",") if url.strip()]
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @model_validator(mode="after")
    def validate_production_settings(self):
        """Validate that required settings are set in production."""
        if self.ENVIRONMENT == "production":
            # Encryption key is required in production
            if self.ENCRYPTION_KEY is None:
                raise ValueError(
                    "ENCRYPTION_KEY is required in production environment"
                )
            
            # Worker URLs required for remote inference
            if self.INFERENCE_MODE == "remote" and not self.WORKER_URLS:
                raise ValueError(
                    "WORKER_URLS is required for remote inference mode"
                )
        
        return self
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def get_encryption_key(self) -> str:
        """
        Get decrypted encryption key.
        
        Returns:
            Encryption key string
        
        Raises:
            ValueError: If encryption key not configured
        """
        if self.ENCRYPTION_KEY is None:
            # Use default for development only
            if self.ENVIRONMENT == "development":
                return "development-key-not-for-production"
            raise ValueError("ENCRYPTION_KEY not configured")
        return self.ENCRYPTION_KEY.get_secret_value()
    
    def get_old_encryption_key(self) -> Optional[str]:
        """
        Get old encryption key for rotation.
        
        Returns:
            Old encryption key or None
        """
        if self.ENCRYPTION_KEY_OLD is None:
            return None
        return self.ENCRYPTION_KEY_OLD.get_secret_value()
    
    def get_database_url(self) -> Optional[str]:
        """
        Get database URL.
        
        Returns:
            Database URL or None for JSON file storage
        """
        if self.DATABASE_URL is None:
            return None
        return self.DATABASE_URL.get_secret_value()
    
    def get_worker_auth_secret(self) -> Optional[str]:
        """
        Get worker authentication secret.
        
        Returns:
            Auth secret or None
        """
        if self.WORKER_AUTH_SECRET is None:
            return None
        return self.WORKER_AUTH_SECRET.get_secret_value()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Settings are loaded once and cached for the lifetime of the application.
    
    Returns:
        Settings instance
    """
    return Settings()


def clear_settings_cache() -> None:
    """
    Clear the settings cache.
    
    Useful for testing or reloading configuration.
    """
    get_settings.cache_clear()

