"""
Simple in-memory database for watermark key storage.

In production, this should be replaced with a proper database (PostgreSQL, etc.)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .security import KeyEncryption
from .logging import get_logger

logger = get_logger(__name__)


class WatermarkKeyDB:
    """
    In-memory database for watermark keys.
    
    Stores:
        - watermark_id -> secret_key mapping
        - Metadata (model, strategy, status, created_at)
    
    In production, replace with proper database.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, encryption: Optional[KeyEncryption] = None):
        """
        Initialize key database.
        
        Args:
            storage_path: Optional path to persistent storage file
            encryption: Optional key encryption instance
        """
        self.storage_path = storage_path
        self.encryption = encryption or KeyEncryption()
        self._db: Dict[str, Dict] = {}
        
        # Load from disk if path provided
        if storage_path and storage_path.exists():
            self._load_from_disk()
    
    def _load_from_disk(self) -> None:
        """Load database from disk."""
        if self.storage_path is None:
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self._db = data
        except Exception as e:
            # If loading fails, start with empty DB
            logger.warning(
                "db_load_failed",
                extra={"error": str(e), "path": str(self.storage_path)}
            )
            self._db = {}
    
    def _save_to_disk(self) -> None:
        """Save database to disk."""
        if self.storage_path is None:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self._db, f, indent=2)
        except Exception as e:
            logger.warning(
                "db_save_failed",
                extra={"error": str(e), "path": str(self.storage_path)}
            )
    
    def create_watermark(
        self,
        watermark_id: str,
        secret_key: str,
        model: str,
        strategy: str = "seed_bias",
        config_hash: Optional[str] = None,
    ) -> None:
        """
        Create a new watermark record.
        
        Args:
            watermark_id: Unique watermark identifier
            secret_key: Secret master key (will be encrypted)
            model: Model identifier (e.g., "sdxl")
            strategy: Strategy type (e.g., "seed_bias")
            config_hash: Optional config hash for validation
        """
        encrypted_key = self.encryption.encrypt_key(secret_key)
        
        self._db[watermark_id] = {
            "watermark_id": watermark_id,
            "secret_key_encrypted": encrypted_key,
            "strategy": strategy,
            "model": model,
            "config_hash": config_hash,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        self._save_to_disk()
    
    def get_watermark(self, watermark_id: str) -> Optional[Dict]:
        """
        Get watermark record by ID.
        
        Args:
            watermark_id: Watermark identifier
        
        Returns:
            Watermark record dict or None if not found
        """
        if watermark_id not in self._db:
            return None
        
        record = self._db[watermark_id].copy()
        
        # Decrypt secret key
        encrypted_key = record.pop("secret_key_encrypted")
        record["secret_key"] = self.encryption.decrypt_key(encrypted_key)
        
        return record
    
    def revoke_watermark(self, watermark_id: str) -> bool:
        """
        Revoke a watermark (mark as inactive).
        
        Args:
            watermark_id: Watermark identifier
        
        Returns:
            True if revoked, False if not found
        """
        if watermark_id not in self._db:
            return False
        
        self._db[watermark_id]["status"] = "revoked"
        self._save_to_disk()
        return True
    
    def is_active(self, watermark_id: str) -> bool:
        """
        Check if watermark is active.
        
        Args:
            watermark_id: Watermark identifier
        
        Returns:
            True if active, False otherwise
        """
        if watermark_id not in self._db:
            return False
        return self._db[watermark_id]["status"] == "active"


# Global database instance (singleton)
_db_instance: Optional[WatermarkKeyDB] = None


def get_db() -> WatermarkKeyDB:
    """
    Get global database instance.
    
    Returns:
        WatermarkKeyDB instance
    """
    global _db_instance
    if _db_instance is None:
        storage_path = Path("service_data/watermark_keys.json")
        _db_instance = WatermarkKeyDB(storage_path=storage_path)
    return _db_instance

