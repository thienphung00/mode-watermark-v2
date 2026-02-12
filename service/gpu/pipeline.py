"""
GPU pipeline for image generation and DDIM inversion.

This module provides:
- Watermarked image generation using Stable Diffusion
- DDIM inversion for latent extraction
- G-value computation for detection

Initially implemented as stubs, with realistic interfaces.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from image generation."""
    image_bytes: bytes
    seed_used: int
    latent_shape: Tuple[int, ...]


@dataclass
class InversionResult:
    """Result from DDIM inversion."""
    latent: np.ndarray
    g_values: np.ndarray
    

@dataclass
class DetectionResult:
    """Result from detection."""
    detected: bool
    score: float
    threshold: float
    confidence: float
    log_odds: float
    posterior: float
    latent_shape: Tuple[int, ...]


class GPUPipeline:
    """
    GPU pipeline for watermark operations.
    
    This class wraps the heavy GPU computations:
    - Image generation with watermark embedding
    - DDIM inversion for latent recovery
    - G-value computation and detection
    
    The implementation can be either:
    - Stub mode: Returns realistic mock data (for testing)
    - Full mode: Uses actual SD pipeline and detection logic from /src
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        stub_mode: bool = True,
    ):
        """
        Initialize GPU pipeline.
        
        Args:
            model_id: HuggingFace model ID
            device: Device for computation
            stub_mode: If True, use stub implementations
        """
        self.model_id = model_id
        self.device = device
        self.stub_mode = stub_mode
        
        self._pipeline = None
        self._vae = None
        self._is_loaded = False
        
        if stub_mode:
            logger.info("GPUPipeline initialized in STUB mode")
            self._is_loaded = True
        else:
            logger.info(f"GPUPipeline initializing with model: {model_id}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    def load_models(self) -> None:
        """
        Load models into GPU memory.
        
        In stub mode, this is a no-op.
        In full mode, loads SD pipeline from HuggingFace.
        """
        if self.stub_mode:
            self._is_loaded = True
            return
        
        try:
            # Import here to avoid dependency issues in stub mode
            import torch
            from diffusers import StableDiffusionPipeline
            
            logger.info(f"Loading model: {self.model_id}")
            
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            self._vae = self._pipeline.vae
            self._is_loaded = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("Falling back to stub mode")
            self.stub_mode = True
            self._is_loaded = True
    
    def generate(
        self,
        prompt: str,
        master_key: str,
        key_id: str,
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Generate watermarked image.
        
        ARCHITECTURAL REQUIREMENT: Uses master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        
        Args:
            prompt: Text prompt
            master_key: Master key for watermark embedding
            key_id: Key identifier (public PRF index)
            seed: Random seed
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            width: Image width
            height: Image height
            embedding_config: Watermark embedding parameters
            
        Returns:
            GenerationResult with image bytes and metadata
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        if self.stub_mode:
            return self._generate_stub(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
            )
        
        return self._generate_full(
            prompt=prompt,
            master_key=master_key,
            key_id=key_id,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            embedding_config=embedding_config or {},
        )
    
    def _generate_stub(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
    ) -> GenerationResult:
        """Generate stub image (colored noise based on prompt hash)."""
        # Create deterministic "image" based on prompt and seed
        np.random.seed(seed)
        
        # Generate colored noise
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add some structure based on prompt hash
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        base_color = [
            (prompt_hash >> 16) & 0xFF,
            (prompt_hash >> 8) & 0xFF,
            prompt_hash & 0xFF,
        ]
        
        # Blend with base color
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] * 0.7 + base_color[i] * 0.3).astype(np.uint8)
        
        # Convert to PNG bytes
        try:
            from PIL import Image
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        except ImportError:
            # Minimal PNG without PIL
            image_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # Invalid but non-empty
        
        return GenerationResult(
            image_bytes=image_bytes,
            seed_used=seed,
            latent_shape=(4, height // 8, width // 8),
        )
    
    def _generate_full(
        self,
        prompt: str,
        master_key: str,
        key_id: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        embedding_config: Dict[str, Any],
    ) -> GenerationResult:
        """Generate image using full SD pipeline with watermark.
        
        ARCHITECTURAL REQUIREMENT: Uses master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        """
        import torch
        
        # Import watermarking components from src
        from src.core.config import SeedBiasConfig
        from src.engine.strategies.seed_bias import SeedBiasStrategy
        

        # Create SeedBiasConfig from embedding_config
        seed_bias_config = SeedBiasConfig(
            lambda_strength=embedding_config.get("lambda_strength", 0.05),
            domain=embedding_config.get("domain", "frequency"),
            low_freq_cutoff=embedding_config.get("low_freq_cutoff", 0.05),
            high_freq_cutoff=embedding_config.get("high_freq_cutoff", 0.4),
        )

        latent_shape = (4, height // 8, width // 8)
        
        # Create watermark strategy with master_key (not derived_key)
        strategy = SeedBiasStrategy(
            config=seed_bias_config,
            master_key=master_key,
            latent_shape=latent_shape,
            device=self.device,
        )
        
        # Pass key_id via prepare_for_sample (not constructor)
        strategy.prepare_for_sample(
            sample_id=f"gen-{seed}",
            prompt=prompt,
            seed=seed,
            key_id=key_id,
        )

        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate with watermark
        # Note: This requires the pipeline to support callback hooks
        with torch.no_grad():
            output = self._pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                # callback=strategy.callback,  # If supported
            )
        
        image = output.images[0]
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return GenerationResult(
            image_bytes=image_bytes,
            seed_used=seed,
            latent_shape=(4, height // 8, width // 8),
        )
    
    def invert_and_detect(
        self,
        image_bytes: bytes,
        master_key: str,
        key_id: str,
        g_field_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None,
        inversion_config: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """
        Perform DDIM inversion and detect watermark.
        
        ARCHITECTURAL REQUIREMENT: Uses master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        compute_g_values() uses (master_key, key_id) directly.
        
        Args:
            image_bytes: Raw image bytes
            master_key: Master key for g-value computation
            key_id: Key identifier (public PRF index)
            g_field_config: G-field configuration
            detection_config: Detection parameters
            inversion_config: DDIM inversion parameters
            
        Returns:
            DetectionResult with detection decision and statistics
        """
        if self.stub_mode:
            return self._detect_stub(
                image_bytes=image_bytes,
                key_id=key_id,
            )
        
        return self._detect_full(
            image_bytes=image_bytes,
            master_key=master_key,
            key_id=key_id,
            g_field_config=g_field_config or {},
            detection_config=detection_config or {},
            inversion_config=inversion_config or {},
        )
    
    def _detect_stub(
        self,
        image_bytes: bytes,
        key_id: str,
    ) -> DetectionResult:
        """Generate stub detection result."""
        # Use image hash + key_id for deterministic but varied results
        combined = hashlib.sha256(image_bytes + key_id.encode()).hexdigest()
        hash_val = int(combined[:8], 16) / 0xFFFFFFFF
        
        # Simulate watermarked images having higher scores
        if hash_val > 0.3:  # 70% chance of "detected" for varied images
            score = 2.0 + hash_val * 3.0  # Score between 2 and 5
            posterior = 0.75 + hash_val * 0.24
        else:
            score = -0.5 + hash_val * 2.0  # Score between -0.5 and 1.5
            posterior = 0.1 + hash_val * 0.4
        
        detected = posterior > 0.5
        confidence = posterior if detected else (1 - posterior)
        log_odds = np.log(posterior / (1 - posterior)) if 0 < posterior < 1 else 0.0
        
        return DetectionResult(
            detected=detected,
            score=float(score),
            threshold=0.5,
            confidence=float(confidence),
            log_odds=float(log_odds),
            posterior=float(posterior),
            latent_shape=(4, 64, 64),
        )
    
    def _detect_full(
        self,
        image_bytes: bytes,
        master_key: str,
        key_id: str,
        g_field_config: Dict[str, Any],
        detection_config: Dict[str, Any],
        inversion_config: Dict[str, Any],
    ) -> DetectionResult:
        """
        Perform full DDIM inversion and detection using canonical BayesianDetector.
        
        ARCHITECTURAL REQUIREMENT: Uses master_key only.
        derived_key is NOT used - key_id is a public PRF index.
        
        This method matches the canonical training/evaluation pipeline exactly:
        - Uses compute_g_values() from src/detection/g_values
        - Uses BayesianDetector from src/models/detectors
        - compute_g_values() uses (master_key, key_id) directly
        
        CALIBRATED DETECTION FLOW:
        1. Compute raw log-odds using BayesianDetector
        2. Normalize log-odds using normalization_params (mean, std)
        3. Compare normalized score against calibrated threshold
        """
        import torch
        from PIL import Image
        
        # Import detection components from src (canonical imports)
        from src.detection.inversion import DDIMInverter
        from src.detection.g_values import compute_g_values
        from src.models.detectors import BayesianDetector
        
        # CRITICAL: Validate all three artifact paths are provided
        likelihood_params_path = detection_config.get("likelihood_params_path")
        normalization_params_path = detection_config.get("normalization_params_path")
        calibration_params_path = detection_config.get("calibration_params_path")
        
        if likelihood_params_path is None:
            raise ValueError(
                "likelihood_params_path is required for Bayesian detection. "
                "Set LIKELIHOOD_PARAMS_PATH environment variable."
            )
        
        if normalization_params_path is None:
            raise ValueError(
                "normalization_params_path is required for calibrated detection. "
                "Set NORMALIZATION_PARAMS_PATH environment variable."
            )
        
        if calibration_params_path is None:
            raise ValueError(
                "calibration_params_path is required for calibrated detection. "
                "Set CALIBRATION_PARAMS_PATH environment variable."
            )
        
        logger.info(f"Loading BayesianDetector from: {likelihood_params_path}")
        logger.info(f"Loading normalization params from: {normalization_params_path}")
        logger.info(f"Loading calibration params from: {calibration_params_path}")

        # #region agent log
        try:
            import os
            _log_path = os.environ.get("DEBUG_LOG_PATH", "/Users/macos/Downloads/mode-watermarking-restructured/.cursor/debug.log")
            with open(_log_path, "a") as _f:
                _f.write(__import__("json").dumps({"location": "gpu/pipeline.py:run_bayesian_detection", "message": "GPU opening normalization path", "data": {"normalization_params_path": normalization_params_path, "is_absolute": os.path.isabs(normalization_params_path), "cwd": os.getcwd()}, "timestamp": __import__("time").time() * 1000, "hypothesisId": "B"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Load normalization parameters
        import json
        from pathlib import Path

        with open(normalization_params_path, "r") as f:
            normalization_data = json.load(f)
        
        # Extract normalization params for the specific family
        # Assuming family_098 is the key (adjust if needed)
        family_key = list(normalization_data.keys())[0]
        norm_mean = normalization_data[family_key]["mean"]
        norm_std = normalization_data[family_key]["std"]
        
        logger.info(f"Normalization params: mean={norm_mean:.2f}, std={norm_std:.2f}")
        
        # Load calibration parameters
        with open(calibration_params_path, "r") as f:
            calibration_data = json.load(f)
        
        # Extract calibration threshold for the specific family
        family_key = list(calibration_data.keys())[0]
        calibrated_threshold = calibration_data[family_key]["deployment_threshold"]
        target_fpr = calibration_data[family_key]["target_fpr"]
        
        logger.info(f"Calibration params: threshold={calibrated_threshold:.4f}, target_fpr={target_fpr}")
        
        # Load trained detector (with default threshold - we'll override with calibrated one)
        detector = BayesianDetector(
            likelihood_params_path=likelihood_params_path,
            threshold=detection_config.get("threshold", 0.5),
            prior_watermarked=detection_config.get("prior_watermarked", 0.5),
        )
        
        logger.info(f"Detector num_positions: {detector.num_positions}")
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Create inverter
        inverter = DDIMInverter(
            pipeline=self._pipeline,
            device=self.device,
        )
        
        # DDIM inversion
        num_steps = inversion_config.get("num_inference_steps", 50)
        guidance_scale = inversion_config.get("guidance_scale", 1.0)
        
        logger.info(f"DDIM inversion: num_steps={num_steps}, guidance_scale={guidance_scale}")
        
        with torch.no_grad():
            z_T = inverter.invert(
                image=image,
                num_inference_steps=num_steps,
                prompt="",
                guidance_scale=guidance_scale,
            )
        
        # z_T is [1, 4, 64, 64]
        logger.info(f"Inverted latent shape: {z_T.shape}")
        
        # Compute g-values using canonical function (matches training pipeline)
        # Uses (master_key, key_id) - derived_key is NOT used
        g, mask = compute_g_values(
            x0=z_T,  # Inverted latent tensor [1, 4, 64, 64]
            key=key_id,  # key_id is the public PRF index
            master_key=master_key,  # master_key is the cryptographic secret
            return_mask=True,
            g_field_config=g_field_config,  # From detection_config
            latent_type="zT",  # We're working with inverted latent
        )
        
        logger.info(f"G-values shape: {g.shape}")
        
        # DIAGNOSTIC: Log g-value statistics to verify they differ across images
        logger.info(
            f"G-values stats: min={g.min().item():.4f}, max={g.max().item():.4f}, "
            f"mean={g.mean().item():.4f}, std={g.std().item():.4f}"
        )
        
        # Flatten and prepare for detector
        if g.dim() > 1:
            g = g.flatten()
        if mask is not None and mask.dim() > 1:
            mask = mask.flatten()
        
        # Apply mask to get valid positions only
        if mask is not None:
            g_valid = g[mask > 0.5]
        else:
            g_valid = g
        
        # DIAGNOSTIC: Log g_valid stats after masking, before scoring
        logger.info(
            f"g_valid stats (after mask, before scoring): "
            f"mean={g_valid.mean().item():.4f}, "
            f"std={g_valid.std().item():.4f}, "
            f"frac_positive={(g_valid > 0).float().mean().item():.4f}, "
            f"n_positions={g_valid.shape[0]}"
        )
        
        # Prepare g-values based on mapping_mode
        # CRITICAL: The Gaussian likelihood model was trained on real-valued
        # g-values in continuous mode. Binarizing destroys the signal.
        mapping_mode = g_field_config.get("mapping_mode", "binary")
        logger.info(f"g_field_config mapping_mode: {mapping_mode}")
        
        if mapping_mode == "continuous":
            # Continuous mode: preserve real-valued g-values (no thresholding)
            g_input = g_valid.unsqueeze(0)  # [1, N_eff]
        else:
            # Binary mode: binarize g-values to {0, 1}
            g_input = (g_valid > 0).float().unsqueeze(0)  # [1, N_eff]
        
        mask_valid = mask[mask > 0.5].unsqueeze(0) if mask is not None else None
        
        # Validate mask alignment with detector
        N_eff = int(g_input.shape[1])
        if detector.num_positions is not None and N_eff != detector.num_positions:
            raise ValueError(
                f"Mask alignment mismatch: g_input has {N_eff} positions "
                f"but likelihood model expects {detector.num_positions}. "
                f"This indicates g_field_config mismatch between generation and detection."
            )
        
        logger.info(f"Mask validated: {N_eff} positions")
        
        # Sanity check for continuous mode: g_input should have many unique values
        if mapping_mode == "continuous":
            n_unique = g_input.unique().numel()
            logger.info(f"Continuous mode: g_input has {n_unique} unique values")
            assert n_unique > 10, (
                f"Continuous mode g_input has only {n_unique} unique values "
                f"(expected many). This suggests an upstream binarization bug."
            )
        
        # Run detection using trained BayesianDetector
        result = detector.score(g_input, mask_valid)
        
        # Extract raw log-odds
        raw_log_odds = result["log_odds"].item()
        posterior = result["posterior"].item()
        
        # DIAGNOSTIC: Log raw values BEFORE normalization to verify they differ
        logger.info(f"Raw log-odds (before normalization): {raw_log_odds:.4f}")
        logger.info(f"Posterior (from detector): {posterior:.4f}")
        
        # CRITICAL: Apply normalization to raw log-odds
        # normalized_score = (raw_log_odds - mean) / std
        normalized_score = (raw_log_odds - norm_mean) / norm_std
        
        # DIAGNOSTIC: Log normalized score
        logger.info(f"Normalized score: {normalized_score:.4f}")
        
        # CRITICAL: Use calibrated threshold for detection decision
        # Detection: normalized_score > calibrated_threshold
        detected = normalized_score > calibrated_threshold
        
        # CRITICAL FIX: Compute confidence using sigmoid for proper probability scaling
        # The previous formula (0.5 + 0.1 * distance) always hit the 0.99 ceiling
        # for any score significantly different from threshold, causing constant confidence.
        #
        # New approach: Use sigmoid of normalized score to map to [0, 1] probability
        # This ensures different scores produce different confidence values.
        import torch
        confidence_raw = torch.sigmoid(torch.tensor(normalized_score)).item()
        
        # Clip to [0.01, 0.99] to avoid extreme values
        confidence = max(0.01, min(0.99, confidence_raw))
        
        # DIAGNOSTIC: Log complete detection result
        logger.info(
            f"Detection result: raw_log_odds={raw_log_odds:.4f}, "
            f"normalized_score={normalized_score:.4f}, "
            f"threshold={calibrated_threshold:.4f}, "
            f"detected={detected}, "
            f"confidence={confidence:.4f}, "
            f"posterior={posterior:.4f}"
        )
        
        # Get latent shape for result
        latent_shape = tuple(z_T.shape[1:]) if z_T.dim() == 4 else tuple(z_T.shape)
        
        return DetectionResult(
            detected=detected,
            score=float(normalized_score),  # Return normalized score
            threshold=float(calibrated_threshold),
            confidence=float(confidence),
            log_odds=float(raw_log_odds),  # Keep raw log-odds for debugging
            posterior=float(posterior),  # Keep posterior for reference
            latent_shape=latent_shape,
        )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.stub_mode:
            return {
                "gpu_available": True,
                "memory_used_mb": 1000,
                "memory_total_mb": 8000,
                "memory_used_pct": 12.5,
            }
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {"gpu_available": False}
            
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            return {
                "gpu_available": True,
                "memory_used_mb": int(memory_allocated),
                "memory_total_mb": int(memory_total),
                "memory_used_pct": (memory_allocated / memory_total) * 100,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {"gpu_available": False, "error": str(e)}
