"""
Watermark Authority Service.

This service is the cryptographic and statistical authority of the system.
It owns:
- master_key (never exposed to clients or workers)
- key_id → watermark policy mapping
- Embedding configuration
- Detection configuration (Bayesian parameters)
- Calibration version
- Scoped key derivation (workers receive derived keys, never master_key)

All watermark decisions flow from this authority.

CRITICAL SECURITY INVARIANT:
- master_key NEVER leaves this service
- Workers receive only derived_key (scoped, non-reversible)
- Fingerprints are computed here and passed for cache keying
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from service.infra.db import get_db
from service.infra.security import (
    generate_watermark_id,
    generate_master_key,
    compute_key_fingerprint,
    derive_scoped_key,
    OperationType,
)
from service.infra.logging import get_logger

logger = get_logger(__name__)


class WatermarkAuthorityService:
    """
    Watermark authority service.
    
    This service is the root of trust for watermarking:
    - Owns master_key (never exposed)
    - Resolves key_id → watermark policy
    - Decides embedding and detection configuration
    - Produces watermark payloads for generation
    - Produces detection configuration for detection
    """
    
    def __init__(self):
        """Initialize watermark authority service."""
        self.db = get_db()
    
    @staticmethod
    def _compute_policy_version(
        embedding_config: Dict[str, Any],
        g_field_config: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Compute deterministic policy version from statistical parameters.
        
        policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        It is a deterministic hash of:
        - embedding_config (watermark embedding parameters)
        - g_field_config (G-field generation parameters)
        - detection_config (Bayesian detector parameters: threshold, prior, likelihood config)
        
        It does NOT depend on:
        - key_id (key-specific, not policy-specific)
        - user input (runtime values)
        - master_key (cryptographic, not statistical)
        
        Args:
            embedding_config: Embedding configuration dict
            g_field_config: G-field configuration dict
            detection_config: Detection configuration dict (Bayesian parameters)
        
        Returns:
            Deterministic policy version string (SHA-256 hash, truncated to 16 chars for readability)
        """
        # Create a stable, sorted representation of all statistical parameters
        # Sort keys to ensure deterministic ordering
        policy_dict = {
            "embedding_config": dict(sorted(embedding_config.items())),
            "g_field_config": dict(sorted(g_field_config.items())),
            # Only include statistical parameters from detection_config, not runtime values
            "detection_config": {
                "detector_type": detection_config.get("detector_type"),
                "threshold": detection_config.get("threshold"),
                "prior_watermarked": detection_config.get("prior_watermarked"),
                "likelihood_params_path": detection_config.get("likelihood_params_path"),
            },
        }
        
        # Serialize to JSON with sorted keys for deterministic hashing
        policy_json = json.dumps(policy_dict, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(policy_json.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Truncate to 16 characters for readability (still provides 64 bits of entropy)
        policy_version = hash_hex[:16]
        
        return policy_version
    
    def create_watermark_policy(
        self,
        key_id: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new watermark policy.
        
        This generates a new master_key and key_id, and stores the policy.
        
        Args:
            key_id: Optional key identifier (if None, generates new one)
            embedding_config: Optional embedding configuration override
            detection_config: Optional detection configuration override
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - watermark_version: Policy version
                - embedding_config: Embedding configuration
                - detection_config: Detection configuration
        """
        # Generate key_id and master_key
        if key_id is None:
            key_id = generate_watermark_id()
        master_key = generate_master_key()
        
        # Default embedding config (seed-bias for Phase-1)
        default_embedding_config = {
            "lambda_strength": 0.05,
            "domain": "frequency",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
        }
        embedding_config = embedding_config or default_embedding_config
        
        # Default detection config (Bayesian for Phase-1)
        default_detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        detection_config = detection_config or default_detection_config
        
        # G-field config must match generation config (for policy version computation)
        g_field_config = {
            "mapping_mode": "binary",
            "domain": embedding_config.get("domain", "frequency"),
            "frequency_mode": "bandpass",
            "low_freq_cutoff": embedding_config.get("low_freq_cutoff", 0.05),
            "high_freq_cutoff": embedding_config.get("high_freq_cutoff", 0.4),
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        watermark_version = self._compute_policy_version(
            embedding_config,
            g_field_config,
            detection_config,
        )
        
        # Store in database
        self.db.create_watermark(
            watermark_id=key_id,
            secret_key=master_key,
            model="stable-diffusion-v1-5",  # Phase-1: SD only
            strategy="seed_bias",
        )
        
        logger.info(
            "watermark_policy_created",
            extra={"key_id": key_id, "watermark_version": watermark_version}
        )
        
        return {
            "key_id": key_id,
            "watermark_version": watermark_version,
            "embedding_config": embedding_config,
            "detection_config": detection_config,
        }
    
    def get_watermark_payload(
        self,
        key_id: str,
        request_id: Optional[str] = None,
        for_local_use: bool = True,
    ) -> Dict[str, Any]:
        """
        Get watermark payload for generation.
        
        This returns the configuration needed by GenerationAdapter to embed
        the watermark. For remote workers, only derived keys are included.
        For local use (in-process), master_key can be optionally included.
        
        SECURITY INVARIANT:
        - master_key is NEVER sent to remote workers
        - derived_key is scoped to this specific operation and key_id
        - key_fingerprint is used for cache keying and audit trails
        
        Args:
            key_id: Public key identifier
            request_id: Optional request ID for audit trails
            for_local_use: If True, include master_key for in-process use.
                          MUST be False for any remote worker transmission.
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - derived_key: Scoped derived key (for workers)
                - key_fingerprint: Canonical fingerprint for cache keying
                - embedding_config: Embedding configuration
                - watermark_version: Policy version
                - master_key: (ONLY if for_local_use=True)
        
        Raises:
            ValueError: If key_id not found or revoked
        """
        # Load watermark record
        record = self.db.get_watermark(key_id)
        if record is None:
            raise ValueError(f"Watermark {key_id} not found")
        
        if not self.db.is_active(key_id):
            raise ValueError(f"Watermark {key_id} is revoked")
        
        master_key = record["secret_key"]
        
        # Derive scoped key for generation (NEVER send master_key to workers)
        derived_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.GENERATION,
            request_id=request_id,
        )
        
        # Compute fingerprint for cache keying
        key_fingerprint = compute_key_fingerprint(master_key)
        
        # Get policy (for now, use defaults; in future, store in DB)
        # TODO: Store policy in database for versioning
        embedding_config = {
            "lambda_strength": 0.05,
            "domain": "frequency",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
        }
        
        # G-field config must match generation config (for policy version computation)
        g_field_config = {
            "mapping_mode": "binary",
            "domain": embedding_config["domain"],
            "frequency_mode": "bandpass",
            "low_freq_cutoff": embedding_config["low_freq_cutoff"],
            "high_freq_cutoff": embedding_config["high_freq_cutoff"],
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Default detection config (for policy version computation)
        detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        watermark_version = self._compute_policy_version(
            embedding_config,
            g_field_config,
            detection_config,
        )
        
        result = {
            "key_id": key_id,
            "derived_key": derived_key,
            "key_fingerprint": key_fingerprint,
            "embedding_config": embedding_config,
            "watermark_version": watermark_version,
        }
        
        # SECURITY: Only include master_key for local use (never for remote workers)
        if for_local_use:
            result["master_key"] = master_key
            logger.debug(
                "master_key_included_for_local_use",
                extra={"key_id": key_id, "request_id": request_id}
            )
        
        return result
    
    def get_detection_config(
        self,
        key_id: str,
        request_id: Optional[str] = None,
        for_local_use: bool = True,
    ) -> Dict[str, Any]:
        """
        Get detection configuration for detection.
        
        This returns the configuration needed by DetectionService to detect
        the watermark. For remote workers, only derived keys are included.
        For local use (in-process), master_key can be optionally included.
        
        SECURITY INVARIANT:
        - master_key is NEVER sent to remote workers
        - derived_key is scoped to detection operation and key_id
        - key_fingerprint is used for cache keying and audit trails
        
        Args:
            key_id: Public key identifier
            request_id: Optional request ID for audit trails
            for_local_use: If True, include master_key for in-process use.
                          MUST be False for any remote worker transmission.
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - derived_key: Scoped derived key (for workers)
                - key_fingerprint: Canonical fingerprint for cache keying
                - detection_config: Detection configuration (Bayesian parameters)
                - watermark_version: Policy version
                - g_field_config: G-field configuration (must match generation)
                - master_key: (ONLY if for_local_use=True)
        
        Raises:
            ValueError: If key_id not found or revoked
        """
        # Load watermark record
        record = self.db.get_watermark(key_id)
        if record is None:
            raise ValueError(f"Watermark {key_id} not found")
        
        if not self.db.is_active(key_id):
            raise ValueError(f"Watermark {key_id} is revoked")
        
        master_key = record["secret_key"]
        
        # Derive scoped key for detection (NEVER send master_key to workers)
        derived_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.DETECTION,
            request_id=request_id,
        )
        
        # Compute fingerprint for cache keying
        key_fingerprint = compute_key_fingerprint(master_key)
        
        # Get policy (for now, use defaults; in future, store in DB)
        # TODO: Store policy in database for versioning
        detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        
        # G-field config must match generation config
        g_field_config = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Inversion parameters are part of watermark policy, not user preference.
        # These must match research calibration assumptions to ensure accurate detection.
        # DetectionService will enforce these parameters and reject mismatched user input.
        # 
        # CRITICAL: These parameters must remain aligned with research calibration assumptions in src/.
        # Research layer requires guidance_scale=1.0 for DDIM inversion (see src/detection/inversion.py).
        # This ensures detection reliability and Bayesian calibration correctness.
        inversion_config = {
            "num_inference_steps": 50,
            "guidance_scale": 1.0,  # Must be 1.0 for DDIM inversion correctness (research requirement)
            "prompt_required": False,  # Prompt is optional but recommended for accuracy
        }
        
        # Calibration is scoped per (model_family, embedding_schema_version).
        # For Phase-1: stable-diffusion-v1-5 with seed_bias_v1
        # This will later be resolved dynamically.
        model_family = "stable-diffusion-v1-5"
        embedding_schema_version = "seed_bias_v1"
        
        # Get resolved paths from startup validation cache
        # Use centralized artifact resolver (validated at startup)
        # This provides a single source of truth for artifact paths
        # NO path guessing or fallback searching - paths come from env vars only
        from service.app.artifact_resolver import get_artifact_resolver
        
        resolver = get_artifact_resolver()
        artifact_result = resolver.resolve()
        
        # Get resolved paths from centralized resolver
        calibration_path = artifact_result.likelihood_params_path_str
        mask_path = artifact_result.mask_path_str
        
        # Update detection_config with resolved paths
        detection_config["likelihood_params_path"] = calibration_path
        detection_config["mask_path"] = mask_path
        
        # Log resolved paths for debugging
        logger.debug(
            "detection_config_paths_resolved",
            extra={
                "likelihood_params_path": calibration_path,
                "mask_path": mask_path,
            }
        )
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        # This ensures results are reproducible, auditable, and scientifically defensible.
        watermark_version = self._compute_policy_version(
            embedding_config={
                "lambda_strength": 0.05,  # Match generation defaults
                "domain": "frequency",
                "low_freq_cutoff": 0.05,
                "high_freq_cutoff": 0.4,
            },
            g_field_config=g_field_config,
            detection_config=detection_config,
        )
        
        result = {
            "key_id": key_id,
            "derived_key": derived_key,
            "key_fingerprint": key_fingerprint,
            "detection_config": detection_config,
            "g_field_config": g_field_config,
            "inversion": inversion_config,
            "watermark_version": watermark_version,
        }
        
        # SECURITY: Only include master_key for local use (never for remote workers)
        if for_local_use:
            result["master_key"] = master_key
            logger.debug(
                "master_key_included_for_local_use",
                extra={"key_id": key_id, "request_id": request_id}
            )
        
        return result
    
    def revoke_watermark(self, key_id: str) -> bool:
        """
        Revoke a watermark (mark as inactive).
        
        Args:
            key_id: Public key identifier
        
        Returns:
            True if revoked, False if not found
        """
        return self.db.revoke_watermark(key_id)

