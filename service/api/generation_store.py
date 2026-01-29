"""
Lightweight generation record persistence.

Stores generation records (key_id, timestamp, filename, seed) in JSON.
Non-blocking and failure-tolerant - logging errors only.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GenerationStore:
    """
    JSON-based generation record storage.
    
    Records are stored with:
    - key_id: Key used for generation
    - timestamp: ISO timestamp of generation
    - filename: Generated image filename
    - seed_used: Seed used for generation
    """
    
    def __init__(self, store_path: str = "./data/generations.json"):
        """
        Initialize generation store.
        
        Args:
            store_path: Path to JSON file for persistence
        """
        self.store_path = Path(store_path)
        self._records: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self) -> None:
        """Load records from persistent storage."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                    self._records = data.get("generations", [])
                logger.info(f"Loaded {len(self._records)} generation records")
            except Exception as e:
                logger.warning(f"Failed to load generation records: {e}")
                self._records = []
        else:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self._records = []
    
    def _save(self) -> None:
        """Save records to persistent storage (non-blocking intent)."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "w") as f:
                json.dump({"generations": self._records}, f, indent=2)
        except Exception as e:
            # Log warning only - don't break generation flow
            logger.warning(f"Failed to save generation record: {e}")
    
    def record_generation(
        self,
        key_id: str,
        filename: str,
        seed_used: int,
        processing_time_ms: Optional[float] = None,
    ) -> None:
        """
        Record a successful generation.
        
        Args:
            key_id: Key ID used for generation
            filename: Generated image filename
            seed_used: Seed used for generation
            processing_time_ms: Optional processing time
        """
        try:
            record = {
                "key_id": key_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filename": filename,
                "seed_used": seed_used,
            }
            if processing_time_ms is not None:
                record["processing_time_ms"] = processing_time_ms
            
            self._records.append(record)
            self._save()
            logger.debug(f"Recorded generation for key {key_id}")
        except Exception as e:
            # Log warning only - don't break generation flow
            logger.warning(f"Failed to record generation: {e}")
    
    def get_records_by_key(self, key_id: str) -> List[Dict[str, Any]]:
        """Get all generation records for a key."""
        return [r for r in self._records if r.get("key_id") == key_id]
    
    def count(self) -> int:
        """Get total number of generation records."""
        return len(self._records)


# Global generation store instance
_generation_store: Optional[GenerationStore] = None


def get_generation_store() -> GenerationStore:
    """Get the global generation store instance."""
    global _generation_store
    if _generation_store is None:
        from service.api.config import get_config
        config = get_config()
        # Store generations.json alongside keys.json
        store_dir = Path(config.key_store_path).parent
        store_path = store_dir / "generations.json"
        _generation_store = GenerationStore(store_path=str(store_path))
    return _generation_store


def reset_generation_store() -> None:
    """Reset the global generation store (useful for testing)."""
    global _generation_store
    _generation_store = None
