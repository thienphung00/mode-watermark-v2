"""
Detection Service (Bayesian-only).

This service provides watermark detection using BayesianDetector only.
It does not expose detector class names or research internals to the API layer.

Phase-1: Bayesian detection only (statistical, defensible, calibration-friendly)
Phase-2: Same (Bayesian remains the authority)

IMPORTANT: Detection currently assumes Stable Diffusion generation.
This includes:
- VAE encoding (image → z_0)
- DDIM inversion to z_T (initial latent)
Future generation models may not require inversion or may use different latent spaces.

Detection is prompt-agnostic: uses unconditional DDIM inversion (prompt="", guidance_scale=1.0).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.g_values import compute_g_values
from src.detection.inversion import DDIMInverter
from src.models.detectors import BayesianDetector

from .authority import WatermarkAuthorityService
from .detector_artifacts import DetectorArtifacts
from .infra.logging import get_logger

logger = get_logger(__name__)


class StableDiffusionDetectionBackend:
    """
    Stable Diffusion-specific detection backend.
    
    This backend handles SD-specific operations:
    - VAE encoding (image → z_0)
    - DDIM inversion (z_0 → z_T)
    
    NOTE: Detection currently assumes Stable Diffusion generation.
    This includes:
    - VAE encoding
    - DDIM inversion to z_T
    Future generation models may not require inversion or may use different latent spaces.
    
    TODO: Future detection backends may bypass inversion or use different latent spaces.
    """
    
    def __init__(self, pipeline: Any, device: str):
        """
        Initialize Stable Diffusion detection backend.
        
        Args:
            pipeline: StableDiffusionPipeline (for VAE encoding and DDIM inversion)
            device: Device string (e.g., "cuda" or "cpu")
        """
        self.pipeline = pipeline
        self.device = device
        self._inverter: Optional[DDIMInverter] = None
    
    @property
    def inverter(self) -> DDIMInverter:
        """
        Lazy-load DDIM inverter.
        
        The inverter is only created when needed (on first encode/invert call),
        not during DetectionService initialization. This ensures detection
        remains lightweight until actually performing detection operations.
        """
        if self._inverter is None:
            self._inverter = DDIMInverter(self.pipeline, device=self.device)
        return self._inverter
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to VAE latent z_0 (Stable Diffusion specific).
        
        Args:
            image: Input PIL Image
        
        Returns:
            VAE-encoded latent tensor z_0
        """
        return self.inverter.encode_image(image, allow_resize=False)
    
    def invert_to_zT(
        self,
        z_0: torch.Tensor,
        num_inference_steps: int,
        prompt: str = "",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Invert VAE latent z_0 to initial latent z_T using DDIM (Stable Diffusion specific).
        
        Detection uses prompt-free (unconditional) inversion:
        - prompt="" (unconditional embedding)
        - guidance_scale=1.0 (required for DDIM inversion correctness)
        
        Args:
            z_0: VAE-encoded latent tensor
            num_inference_steps: Number of DDIM inversion steps
            prompt: Text prompt (default: "" for unconditional)
            guidance_scale: Guidance scale (default: 1.0 for unconditional)
        
        Returns:
            Initial latent tensor z_T (where watermark was embedded)
        """
        return self.inverter.perform_full_inversion(
            z_0,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            guidance_scale=guidance_scale,
        )


class DetectionService:
    """
    Detection service using Bayesian detector only.
    
    This service:
    - Fetches detection configuration from WatermarkAuthorityService
    - Loads detector artifacts (likelihood_params.json, mask, g-field config)
    - Delegates Stable Diffusion-specific operations to StableDiffusionDetectionBackend
    - Instantiates BayesianDetector (never HybridDetector)
    - Performs prompt-free DDIM inversion (unconditional: prompt="", guidance_scale=1.0)
    - Computes g-values from inverted z_T using exact same code path as precompute
    - Applies mask + binarization
    - Runs Bayesian detection
    - Returns product-level results (no research internals)
    
    The service never exposes:
    - Detector class names
    - Bayesian internals
    - Research terminology
    
    IMPORTANT: This service currently depends on Stable Diffusion.
    Detection assumes:
    - Stable Diffusion VAE encoding (image → z_0)
    - DDIM inversion (z_0 → z_T) with unconditional embedding
    - SD-specific latent spaces
    This is acceptable in Phase-1, but must be clearly labeled.
    
    Detection is prompt-agnostic: never requires or trusts prompts.
    """
    
    def __init__(
        self,
        authority: WatermarkAuthorityService,
        pipeline: Any,  # StableDiffusionPipeline (for SD-specific operations)
    ):
        """
        Initialize detection service.
        
        Args:
            authority: WatermarkAuthorityService instance
            pipeline: Stable Diffusion pipeline (for SD-specific encoding and inversion)
        """
        self.authority = authority
        self.device = str(pipeline.device)
        
        # Detector type: Only Bayesian detection is supported
        # Detection mode selection (fast_only/hybrid/full_inversion) is deprecated.
        self.detector_type = "bayesian"
        
        logger.info(
            "detection_service_initialized",
            extra={"detector_type": self.detector_type, "device": self.device}
        )
        
        # Delegate SD-specific operations to backend
        # NOTE: Detection currently assumes Stable Diffusion generation.
        # This includes VAE encoding and DDIM inversion.
        # Future generation models may not require inversion or may use different latent spaces.
        self.sd_backend = StableDiffusionDetectionBackend(pipeline, self.device)
        
        # Artifacts will be loaded on first detection request
        self._artifacts: Optional[DetectorArtifacts] = None
        
        # Detector cache: keyed by (policy_version, device) to reuse stateless detectors
        # Detectors are stateless and can be safely reused across requests
        self._detector_cache: Dict[tuple, BayesianDetector] = {}
        
        # Micro-batching worker (optional, enabled via enable_micro_batching())
        self._detection_worker: Optional[Any] = None  # DetectionWorker
    
    def _load_artifacts(self, detection_config: Dict[str, Any], g_field_config: Dict[str, Any]) -> DetectorArtifacts:
        """
        Load detector artifacts (lazy loading on first request).
        
        Args:
            detection_config: Detection configuration dict
            g_field_config: G-field configuration dict
        
        Returns:
            DetectorArtifacts instance
        
        Raises:
            ValueError: If likelihood_params_path is not set or invalid
            FileNotFoundError: If artifact files don't exist
        """
        if self._artifacts is None:
            likelihood_params_path = detection_config.get("likelihood_params_path")
            if not likelihood_params_path:
                raise ValueError(
                    "likelihood_params_path is required in detection_config. "
                    "Please set LIKELIHOOD_PARAMS_PATH environment variable. "
                    "Artifacts must be precomputed by the offline pipeline."
                )
            
            # Resolve to absolute path (handles relative paths, symlinks, etc.)
            # Paths should already be absolute from startup validation, but normalize here for safety
            likelihood_params_path = str(Path(likelihood_params_path).resolve())
            
            # Mask path is optional (may be embedded in likelihood metadata)
            mask_path = detection_config.get("mask_path")
            if mask_path:
                mask_path = str(Path(mask_path).resolve())
            
            self._artifacts = DetectorArtifacts(
                likelihood_params_path=likelihood_params_path,
                mask_path=mask_path,
                g_field_config=g_field_config,
            )
            logger.info("detector_artifacts_loaded")
        
        return self._artifacts
    
    async def enable_micro_batching(
        self,
        batch_window_ms: float = 20.0,
        max_batch_size: int = 8,
    ) -> None:
        """
        Enable micro-batching for detection requests.
        
        This starts exactly one DetectionWorker per process.
        Calling this multiple times must be safe and idempotent.
        
        Args:
            batch_window_ms: Time window in milliseconds to collect requests (default: 20ms)
            max_batch_size: Maximum batch size before forcing processing (default: 8)
        """
        if self._detection_worker is not None:
            logger.warning("micro_batching_already_enabled")
            return
        
        from service.detection_worker import DetectionWorker
        
        self._detection_worker = DetectionWorker(
            detection_service=self,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
        )
        
        await self._detection_worker.start()
    
    async def disable_micro_batching(self) -> None:
        """Disable micro-batching and stop the worker."""
        if self._detection_worker is not None:
            await self._detection_worker.stop()
            self._detection_worker = None
    
    def detect(
        self,
        image: Image.Image,
        key_id: str,
    ) -> Dict[str, Any]:
        """
        Detect watermark in image (prompt-agnostic).
        
        This method:
        1. Fetches detection configuration from authority (for_local_use=True)
        2. Loads detector artifacts (likelihood_params.json, mask, g-field config)
        3. Encodes image to z_0 (VAE encoding)
        4. Performs prompt-free DDIM inversion to z_T (unconditional: prompt="", guidance_scale=1.0)
        5. Computes g-values from z_T using exact same code path as precompute_inverted_g_values.py
        6. Applies mask + binarization
        7. Runs Bayesian detection
        8. Returns product-level results
        
        LOCAL EXECUTION: This service runs in-process and uses master_key directly.
        For remote workers, use the inference client which transmits derived_key only.
        
        Args:
            image: Input image (PIL Image)
            key_id: Public key identifier
        
        Returns:
            Dictionary with:
                - detected: bool (watermark detected or not)
                - score: float (log-odds detection score)
                - confidence: float (posterior probability, 0-1)
                - policy_version: str (watermark policy version)
                - posterior: float (posterior probability)
                - log_odds: float (log-odds ratio)
                - is_watermarked: bool (detection decision)
                - watermark_version: str (watermark policy version)
                - g_field_config_hash: str (G-field config hash for validation)
        
        Raises:
            ValueError: If artifacts cannot be loaded or validation fails
        """
        # Step 1: Get detection configuration from authority
        # for_local_use=True because this is in-process detection (not remote worker)
        config = self.authority.get_detection_config(key_id, for_local_use=True)
        master_key = config["master_key"]  # Available because for_local_use=True
        detection_config = config["detection_config"]
        g_field_config = config["g_field_config"]
        inversion_config = config.get("inversion", {})
        watermark_version = config["watermark_version"]
        
        # Step 2: Load and validate artifacts
        artifacts = self._load_artifacts(detection_config, g_field_config)
        
        # Step 3: Encode image to z_0 (Stable Diffusion specific: VAE encoding)
        logger.debug(
            "encoding_image",
            extra={"image_size": list(image.size), "image_mode": image.mode}
        )
        z_0 = self.sd_backend.encode_image(image)
        logger.debug(
            "z_0_stats",
            extra={
                "shape": list(z_0.shape),
                "min": round(z_0.min().item(), 4),
                "max": round(z_0.max().item(), 4),
                "mean": round(z_0.mean().item(), 4),
                "std": round(z_0.std().item(), 4),
            }
        )
        
        # Step 4: Perform prompt-free DDIM inversion to z_T
        # CRITICAL: Detection is prompt-agnostic. We use unconditional inversion:
        # - prompt="" (unconditional embedding)
        # - guidance_scale=1.0 (required for DDIM inversion correctness)
        # - num_inference_steps from policy (typically 50)
        policy_num_steps = inversion_config.get("num_inference_steps", 50)
        policy_guidance_scale = inversion_config.get("guidance_scale", 1.0)
        
        logger.debug(
            "ddim_inversion_config",
            extra={
                "num_steps": policy_num_steps,
                "guidance_scale": policy_guidance_scale,
                "prompt": "",
            }
        )
        
        # Enforce unconditional inversion
        if policy_guidance_scale != 1.0:
            raise ValueError(
                f"Detection requires unconditional inversion (guidance_scale=1.0). "
                f"Policy specifies guidance_scale={policy_guidance_scale}, which is invalid."
            )
        
        z_T = self.sd_backend.invert_to_zT(
            z_0,
            num_inference_steps=policy_num_steps,
            prompt="",  # Unconditional: prompt-agnostic detection
            guidance_scale=1.0,  # Unconditional: required for DDIM inversion correctness
        )
        logger.debug(
            "z_T_stats",
            extra={
                "shape": list(z_T.shape),
                "min": round(z_T.min().item(), 4),
                "max": round(z_T.max().item(), 4),
                "mean": round(z_T.mean().item(), 4),
                "std": round(z_T.std().item(), 4),
            }
        )
        
        # Step 5: Compute g-values from z_T using exact same code path as precompute
        # This uses the canonical compute_g_values function.
        # This function defines the canonical watermark statistic.
        # All generation, detection, and calibration must use it.
        logger.debug("computing_g_values", extra={"key_id": key_id, "latent_type": "zT"})
        g, mask = compute_g_values(
            z_T,
            key_id,
            master_key,
            return_mask=True,
            g_field_config=g_field_config,
            latent_type="zT",  # z_T is initial latent
        )
        logger.debug(
            "raw_g_values_stats",
            extra={
                "shape": list(g.shape) if g.dim() > 0 else "scalar",
                "min": round(g.min().item(), 4) if g.numel() > 0 else 0,
                "max": round(g.max().item(), 4) if g.numel() > 0 else 0,
                "mean": round(g.mean().item(), 4) if g.numel() > 0 else 0,
                "mask_shape": list(mask.shape) if mask is not None else None,
                "mask_sum": int(mask.sum().item()) if mask is not None else None,
            }
        )
        
        # Step 6: Apply masking + binarization (exact same logic as detect_bayesian_test.py)
        # Ensure batch dimension for consistency
        if g.dim() == 1:
            g = g.unsqueeze(0)  # [1, N]
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)  # [1, N]
        
        # Apply selection-based masking (exact same as precompute)
        if mask is not None:
            # Select only valid positions (mask == 1)
            g_selected = g[0][mask[0] == 1]  # [N_eff]
            mask_selected = mask[0][mask[0] == 1]  # [N_eff] (all ones, but keep for consistency)
            
            # Remove batch dimension
            g = g_selected  # [N_eff]
            mask = mask_selected  # [N_eff]
        else:
            # No mask: just remove batch dimension
            g = g[0]  # [N]
            mask = None
        
        # CRITICAL: Normalize and binarize g-values (exact same logic as detect_bayesian_test.py)
        # This matches the single source of truth for g-value normalization
        # Convert to float if needed
        if g.dtype in (torch.long, torch.int64):
            g = g.float()
        
        # Handle {-1, 1} format: convert to {0, 1}
        unique_vals = torch.unique(g)
        unique_set = set(unique_vals.cpu().tolist())
        if unique_set.issubset({-1.0, 1.0}):
            g = (g + 1) / 2
        
        # Ensure values are in [0, 1] range
        g = torch.clamp(torch.round(g), 0, 1)
        
        # Binarize: convert to binary {0, 1}
        g = (g > 0).float()
        
        # Log g-value statistics for debugging
        logger.debug(
            "g_values_after_normalization",
            extra={
                "shape": list(g.shape),
                "min": round(g.min().item(), 4),
                "max": round(g.max().item(), 4),
                "mean": round(g.mean().item(), 4),
                "unique_values": torch.unique(g).cpu().tolist(),
            }
        )
        
        # Add batch dimension for detector: [N] -> [1, N]
        g = g.unsqueeze(0)  # [1, N]
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, N]
        
        # Step 7: Validate g-value shape consistency
        if artifacts.num_positions is not None:
            g_length = g.shape[1]
            if g_length != artifacts.num_positions:
                raise ValueError(
                    f"G-value shape mismatch: g.shape[1]={g_length} != "
                    f"artifacts.num_positions={artifacts.num_positions}. "
                    f"This indicates the g-value computation does not match training."
                )
        
        # Step 8: Get or create BayesianDetector (cached by policy_version and device)
        # Detectors are stateless and can be safely reused across requests
        threshold = detection_config.get("threshold", 0.5)
        prior_watermarked = detection_config.get("prior_watermarked", 0.5)
        
        # Cache key: (policy_version, device) ensures detectors are reused per policy
        cache_key: Tuple[str, str] = (watermark_version, self.device)
        
        if cache_key not in self._detector_cache:
            logger.debug(
                "detector_created",
                extra={
                    "cache_key": str(cache_key),
                    "threshold": threshold,
                    "prior_watermarked": prior_watermarked,
                }
            )
            self._detector_cache[cache_key] = BayesianDetector(
                likelihood_params_path=str(artifacts.likelihood_params_path),
                threshold=threshold,
                prior_watermarked=prior_watermarked,
                mask=artifacts.mask,  # Use mask from artifacts (validated during loading)
            )
        else:
            logger.debug("detector_cache_hit", extra={"cache_key": str(cache_key)})
        
        detector = self._detector_cache[cache_key]
        
        logger.debug(
            "detector_using",
            extra={
                "key_id": key_id,
                "detector_type": self.detector_type,
                "threshold": threshold,
                "prior_watermarked": prior_watermarked,
            }
        )
        
        # Step 9: Run detection
        result = detector.score(g, mask=mask)
        
        # Extract results
        posterior = result["posterior"][0].item()  # [1] -> scalar
        log_odds = result["log_odds"][0].item()
        
        # CRITICAL: Use log_odds > 0 for decision (exact same as detect_bayesian_test.py)
        # This matches the research script decision logic exactly
        is_watermarked = log_odds > 0
        detected = bool(is_watermarked)
        
        # Use log-odds as score (higher = more confident)
        score = float(log_odds)
        
        # Use posterior as confidence
        confidence = float(posterior)
        
        # Get config hash for validation
        g_field_config_hash = artifacts.config_hash
        
        # Structured logging matching research script output
        logger.info(
            "detection_result",
            extra={
                "key_id": key_id,
                "detected": detected,
                "log_odds": round(log_odds, 6),
                "posterior": round(posterior, 6),
                "score": round(score, 6),
                "confidence": round(confidence, 6),
                "policy_version": watermark_version,
            }
        )
        logger.debug(
            "detection_stats",
            extra={
                "g_shape": list(g.shape),
                "mask_shape": list(mask.shape) if mask is not None else None,
                "num_positions": artifacts.num_positions,
                "threshold": threshold,
                "prior_watermarked": prior_watermarked,
            }
        )
        
        return {
            "detected": detected,
            "score": score,
            "confidence": confidence,
            "policy_version": watermark_version,
            "posterior": posterior,
            "log_odds": log_odds,
            "is_watermarked": is_watermarked,
            "watermark_version": watermark_version,
            "g_field_config_hash": g_field_config_hash,
        }
    
    def detect_from_g_values(
        self,
        g: torch.Tensor,
        mask: Optional[torch.Tensor],
        key_id: Optional[str] = None,
        detection_config_override: Optional[Dict[str, Any]] = None,
        g_field_config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect watermark from precomputed g-values (for testing and equivalence validation).
        
        This method accepts precomputed g-values and performs the same normalization,
        masking, binarization, and Bayesian detection as the full detect() method.
        
        This is the core detection logic extracted from detect() to enable:
        - Testing with precomputed g-values
        - Equivalence validation against research scripts
        - Direct comparison with detect_bayesian_test.py logic
        
        Args:
            g: Precomputed g-values tensor [N] or [1, N] (raw, will be normalized/binarized)
            mask: Optional mask tensor [N] or [1, N] (structural mask from g-value computation)
            key_id: Public key identifier (required when detection_config_override is not provided)
            detection_config_override: Optional detection config dict to bypass Authority lookup.
                Must contain: likelihood_params_path, mask_path (optional), threshold, prior_watermarked
            g_field_config_override: Optional g-field config dict to bypass Authority lookup.
                Defaults to empty dict if not provided when using override mode.
        
        Returns:
            Dictionary with:
                - posterior: float (posterior probability)
                - log_odds: float (log-odds ratio)
                - is_watermarked: bool (detection decision: log_odds > 0)
        
        Raises:
            ValueError: If artifacts cannot be loaded or validation fails
        """
        # Step 1: Resolve detection configuration (override-aware)
        if detection_config_override is not None:
            detection_config = detection_config_override
            g_field_config = g_field_config_override or {}
            logger.warning("authority_bypass_enabled", extra={"mode": "testing/demo"})
        else:
            if key_id is None:
                raise ValueError(
                    "key_id is required when detection_config_override is not provided"
                )
            config = self.authority.get_detection_config(key_id)
            detection_config = config["detection_config"]
            g_field_config = config["g_field_config"]
        
        # Step 2: Load and validate artifacts
        artifacts = self._load_artifacts(detection_config, g_field_config)
        
        # Step 3: Apply masking + binarization (exact same logic as detect())
        # Ensure batch dimension for consistency
        if g.dim() == 1:
            g = g.unsqueeze(0)  # [1, N]
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)  # [1, N]
        
        # Apply selection-based masking (exact same as precompute)
        if mask is not None:
            # Select only valid positions (mask == 1)
            g_selected = g[0][mask[0] == 1]  # [N_eff]
            mask_selected = mask[0][mask[0] == 1]  # [N_eff] (all ones, but keep for consistency)
            
            # Remove batch dimension
            g = g_selected  # [N_eff]
            mask = mask_selected  # [N_eff]
        else:
            # No mask: just remove batch dimension
            g = g[0]  # [N]
            mask = None
        
        # CRITICAL: Normalize and binarize g-values (exact same logic as detect_bayesian_test.py)
        # This matches the single source of truth for g-value normalization
        # Convert to float if needed
        if g.dtype in (torch.long, torch.int64):
            g = g.float()
        
        # Handle {-1, 1} format: convert to {0, 1}
        unique_vals = torch.unique(g)
        unique_set = set(unique_vals.cpu().tolist())
        if unique_set.issubset({-1.0, 1.0}):
            g = (g + 1) / 2
        
        # Ensure values are in [0, 1] range
        g = torch.clamp(torch.round(g), 0, 1)
        
        # Binarize: convert to binary {0, 1}
        g = (g > 0).float()
        
        # Log g-value statistics for debugging
        logger.debug(
            "g_values_from_precomputed_normalized",
            extra={
                "shape": list(g.shape),
                "min": round(g.min().item(), 4),
                "max": round(g.max().item(), 4),
                "mean": round(g.mean().item(), 4),
                "unique_values": torch.unique(g).cpu().tolist(),
            }
        )
        
        # Add batch dimension for detector: [N] -> [1, N]
        g = g.unsqueeze(0)  # [1, N]
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, N]
        
        # Step 4: Validate g-value shape consistency
        if artifacts.num_positions is not None:
            g_length = g.shape[1]
            if g_length != artifacts.num_positions:
                raise ValueError(
                    f"G-value shape mismatch: g.shape[1]={g_length} != "
                    f"artifacts.num_positions={artifacts.num_positions}. "
                    f"This indicates the g-value computation does not match training."
                )
        
        # Step 5: Get or create BayesianDetector (cached by policy_version and device)
        # For detect_from_g_values, we need to construct a cache key
        # If using override mode, we can't cache reliably, so create a temporary detector
        threshold = detection_config.get("threshold", 0.5)
        prior_watermarked = detection_config.get("prior_watermarked", 0.5)
        
        if detection_config_override is not None:
            # Override mode: create temporary detector (can't cache reliably)
            logger.debug("detector_override_mode", extra={"cached": False})
            detector = BayesianDetector(
                likelihood_params_path=str(artifacts.likelihood_params_path),
                threshold=threshold,
                prior_watermarked=prior_watermarked,
                mask=artifacts.mask,
            )
        else:
            # Normal mode: use cache (same logic as detect())
            # Note: watermark_version not available in this method, use a default key
            # This is a limitation, but detect_from_g_values is mainly for testing
            cache_key: Tuple[str, str] = ("default", self.device)
            if cache_key not in self._detector_cache:
                logger.debug(
                    "detector_created_from_g_values",
                    extra={"cache_key": str(cache_key)}
                )
                self._detector_cache[cache_key] = BayesianDetector(
                    likelihood_params_path=str(artifacts.likelihood_params_path),
                    threshold=threshold,
                    prior_watermarked=prior_watermarked,
                    mask=artifacts.mask,
                )
            else:
                logger.debug(
                    "detector_cache_hit_from_g_values",
                    extra={"cache_key": str(cache_key)}
                )
            detector = self._detector_cache[cache_key]
        
        logger.debug(
            "detector_using_from_g_values",
            extra={
                "key_id": key_id,
                "detector_type": self.detector_type,
                "threshold": threshold,
                "prior_watermarked": prior_watermarked,
            }
        )
        
        # Step 6: Run detection
        result = detector.score(g, mask=mask)
        
        # Extract results
        posterior = result["posterior"][0].item()  # [1] -> scalar
        log_odds = result["log_odds"][0].item()
        
        # CRITICAL: Use log_odds > 0 for decision (exact same as detect_bayesian_test.py)
        # This matches the research script decision logic exactly
        is_watermarked = log_odds > 0
        
        logger.debug(
            "detection_result_from_g_values",
            extra={
                "log_odds": round(log_odds, 6),
                "posterior": round(posterior, 6),
                "detected": is_watermarked,
            }
        )
        
        return {
            "posterior": posterior,
            "log_odds": log_odds,
            "is_watermarked": is_watermarked,
        }

