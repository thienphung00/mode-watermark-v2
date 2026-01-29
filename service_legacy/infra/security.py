"""
Security utilities for key generation and encryption.

Provides functions for:
- Generating secure watermark keys
- Encrypting/decrypting secret keys
- Key derivation (including scoped ephemeral keys)
- Secret rotation support
- Key fingerprint computation
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from enum import Enum
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from service.infra.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Key Scoping and Derivation (CRITICAL SECURITY BOUNDARY)
# =============================================================================


class OperationType(str, Enum):
    """
    Operation types for scoped key derivation.
    
    Each operation type produces a unique derived key from the master key.
    This ensures that a leaked derived key cannot be used for other operations.
    """
    GENERATION = "generation"
    DETECTION = "detection"


def compute_key_fingerprint(master_key: str) -> str:
    """
    Compute a canonical fingerprint for a master key.
    
    The fingerprint is:
    - Deterministic (same key always produces same fingerprint)
    - Non-reversible (cannot recover master_key from fingerprint)
    - Unique (different keys produce different fingerprints)
    
    This fingerprint is used for cache keying and validation without
    exposing the master key itself.
    
    Args:
        master_key: The master key (hex string)
    
    Returns:
        32-character hex fingerprint (truncated SHA-256)
    """
    # Use HMAC with a domain-specific key to prevent oracle attacks
    fingerprint_bytes = hmac.new(
        key=b"watermark_fingerprint_v1",
        msg=master_key.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    
    # Return truncated hex (32 chars = 128 bits, sufficient for uniqueness)
    return fingerprint_bytes.hex()[:32]


def derive_scoped_key(
    master_key: str,
    key_id: str,
    operation: OperationType,
    request_id: Optional[str] = None,
) -> str:
    """
    Derive a scoped ephemeral key from the master key.
    
    The derived key is:
    - Deterministic: Same inputs always produce same output
    - Non-reversible: Cannot recover master_key from derived key
    - Scoped: Cannot be used for other operations or key_ids
    - Safe if leaked: Cannot be used to derive other keys
    
    CRITICAL SECURITY PROPERTY:
    - The master_key never leaves the API boundary
    - Workers only receive derived keys
    - A leaked derived key cannot compromise other operations
    
    Args:
        master_key: The master key (hex string, never transmitted)
        key_id: Public key identifier (scopes to specific watermark)
        operation: Operation type (generation or detection)
        request_id: Optional request ID for additional entropy (not used for
                   determinism, but included in context for audit trails)
    
    Returns:
        64-character hex derived key (256 bits)
    """
    # Build context string for key derivation
    # Format: "watermark_derived_key_v1:{operation}:{key_id}"
    # Note: request_id is NOT included in derivation to maintain determinism
    # across retries with the same request_id
    context = f"watermark_derived_key_v1:{operation.value}:{key_id}"
    
    # Use HKDF-like construction with HMAC-SHA256
    # First, extract: derive intermediate key from master_key
    extract_key = hmac.new(
        key=b"watermark_extract_salt_v1",
        msg=bytes.fromhex(master_key),
        digestmod=hashlib.sha256,
    ).digest()
    
    # Then, expand: derive scoped key from context
    derived_bytes = hmac.new(
        key=extract_key,
        msg=context.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    
    derived_key = derived_bytes.hex()
    
    # Log derivation (without exposing keys)
    logger.debug(
        "scoped_key_derived",
        extra={
            "key_id": key_id,
            "operation": operation.value,
            "request_id": request_id,
            "derived_key_prefix": derived_key[:8] + "...",
        }
    )
    
    return derived_key


def validate_key_fingerprint(
    derived_key: str,
    expected_fingerprint: str,
    key_id: str,
) -> bool:
    """
    Validate that a derived key matches the expected fingerprint.
    
    IMPORTANT: Workers use this to validate incoming keys.
    The derived key cannot be used to compute the original fingerprint
    directly, but we can verify consistency using a known mapping.
    
    In practice, the API computes both the derived_key and fingerprint
    from master_key, and the worker trusts these are consistent.
    This validation is a defense-in-depth check against payload tampering.
    
    Args:
        derived_key: The derived key received by worker
        expected_fingerprint: The fingerprint sent with the request
        key_id: Public key identifier for logging
    
    Returns:
        True if validation passes (fingerprint matches expected format)
    
    Note:
        The worker cannot independently verify the fingerprint was derived
        from the same master_key as the derived_key. This is by design:
        the worker should never have access to master_key. The fingerprint
        serves as an audit trail and cache key, not a cryptographic proof.
    """
    # Basic format validation
    if not expected_fingerprint or len(expected_fingerprint) != 32:
        logger.error(
            "fingerprint_validation_failed",
            extra={
                "key_id": key_id,
                "reason": "invalid_format",
                "fingerprint_length": len(expected_fingerprint) if expected_fingerprint else 0,
            }
        )
        return False
    
    # Verify fingerprint is valid hex
    try:
        bytes.fromhex(expected_fingerprint)
    except ValueError:
        logger.error(
            "fingerprint_validation_failed",
            extra={
                "key_id": key_id,
                "reason": "invalid_hex",
            }
        )
        return False
    
    logger.debug(
        "fingerprint_validation_passed",
        extra={
            "key_id": key_id,
            "fingerprint_prefix": expected_fingerprint[:8] + "...",
        }
    )
    
    return True


def generate_watermark_id() -> str:
    """
    Generate a unique watermark identifier.
    
    Returns:
        Watermark ID in format "wm_xxxxx" where xxxxx is a random hex string
    """
    random_bytes = secrets.token_bytes(8)
    hex_str = random_bytes.hex()[:10]
    return f"wm_{hex_str}"


def generate_master_key() -> str:
    """
    Generate a cryptographically secure master key.
    
    Returns:
        32-byte key encoded as hex string (64 characters)
    """
    key_bytes = secrets.token_bytes(32)
    return key_bytes.hex()


def derive_encryption_key(password: str, salt: Optional[bytes] = None) -> bytes:
    """
    Derive an encryption key from a password using PBKDF2.
    
    Args:
        password: Password string
        salt: Optional salt (if None, uses a deterministic salt)
    
    Returns:
        32-byte encryption key suitable for Fernet
    """
    # Use deterministic salt for consistency
    # In production, consider using per-key salt stored with the ciphertext
    if salt is None:
        salt = b"watermarking_service_salt_v1"
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


class KeyEncryption:
    """
    Encryption wrapper for storing secret keys.
    
    Uses Fernet (symmetric encryption) for key storage.
    Supports key rotation with dual-key decryption.
    
    Usage:
        # With settings
        encryption = KeyEncryption.from_settings()
        
        # Manual initialization
        encryption = KeyEncryption(encryption_key=derived_key)
        
        # Encrypt/decrypt
        encrypted = encryption.encrypt_key(master_key)
        decrypted = encryption.decrypt_key(encrypted)
    """
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        old_encryption_key: Optional[bytes] = None,
    ):
        """
        Initialize key encryption.
        
        Args:
            encryption_key: Encryption key (if None, loads from settings)
            old_encryption_key: Previous encryption key for rotation
        """
        if encryption_key is None:
            encryption_key = self._get_key_from_settings()
        
        self.cipher = Fernet(encryption_key)
        
        # Support key rotation
        self.old_cipher: Optional[Fernet] = None
        if old_encryption_key is not None:
            self.old_cipher = Fernet(old_encryption_key)
            logger.info("key_rotation_enabled")
    
    @classmethod
    def from_settings(cls) -> "KeyEncryption":
        """
        Create KeyEncryption instance from settings.
        
        Returns:
            KeyEncryption instance configured from environment
        """
        from service.infra.settings import get_settings
        
        settings = get_settings()
        
        # Get current encryption key
        encryption_key = derive_encryption_key(settings.get_encryption_key())
        
        # Get old encryption key for rotation
        old_encryption_key = None
        if settings.ENCRYPTION_KEY_ROTATION_ENABLED:
            old_key_str = settings.get_old_encryption_key()
            if old_key_str:
                old_encryption_key = derive_encryption_key(old_key_str)
        
        return cls(
            encryption_key=encryption_key,
            old_encryption_key=old_encryption_key,
        )
    
    def _get_key_from_settings(self) -> bytes:
        """
        Get encryption key from settings.
        
        Returns:
            Derived encryption key
        """
        try:
            from service.infra.settings import get_settings
            settings = get_settings()
            key_str = settings.get_encryption_key()
        except Exception:
            # Fall back to development default
            logger.warning(
                "encryption_key_fallback",
                extra={"reason": "settings_load_failed"}
            )
            key_str = "development-key-not-for-production"
        
        return derive_encryption_key(key_str)
    
    def encrypt_key(self, master_key: str) -> str:
        """
        Encrypt a master key for storage.
        
        Args:
            master_key: Master key string to encrypt
        
        Returns:
            Encrypted key as base64 string
        """
        return self.cipher.encrypt(master_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """
        Decrypt a stored master key.
        
        Supports key rotation by trying the old key if the current key fails.
        
        Args:
            encrypted_key: Encrypted key string
        
        Returns:
            Decrypted master key string
        
        Raises:
            InvalidToken: If decryption fails with all available keys
        """
        # Try current key first
        try:
            return self.cipher.decrypt(encrypted_key.encode()).decode()
        except InvalidToken:
            # Try old key for rotation
            if self.old_cipher is not None:
                logger.debug("decrypt_with_old_key")
                return self.old_cipher.decrypt(encrypted_key.encode()).decode()
            raise
    
    def rotate_encrypted_key(self, encrypted_key: str) -> str:
        """
        Re-encrypt a key with the current encryption key.
        
        Used during key rotation to update stored keys.
        
        Args:
            encrypted_key: Key encrypted with old key
        
        Returns:
            Key encrypted with current key
        """
        decrypted = self.decrypt_key(encrypted_key)
        return self.encrypt_key(decrypted)


# Convenience function for getting a configured KeyEncryption instance
_encryption_instance: Optional[KeyEncryption] = None


def get_encryption() -> KeyEncryption:
    """
    Get a configured KeyEncryption instance.
    
    Returns a cached singleton instance for efficiency.
    
    Returns:
        KeyEncryption instance
    """
    global _encryption_instance
    
    if _encryption_instance is None:
        _encryption_instance = KeyEncryption.from_settings()
    
    return _encryption_instance


def reset_encryption() -> None:
    """
    Reset the cached encryption instance.
    
    Useful for testing or after key rotation.
    """
    global _encryption_instance
    _encryption_instance = None
