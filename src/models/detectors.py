"""
Watermark detector models.

Implements UNet-based and Bayesian detectors for watermark detection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DoubleConv, DownBlock, UpBlock


class UNetDetector(nn.Module):
    """
    UNet-based binary classifier for watermark detection.

    Processes latent representations [B, 4, 64, 64] → logits [B, 1]

    Input Contract:
        Input: VAE-encoded latent z_0
        Shape: [B, 4, 64, 64]
        Statistics: approximately zero-mean, scaled by VAE factor (≈0.18215)
        Domain: spatial (not frequency)
    
    NOTE: This model assumes preprocessing is already applied.
    Do NOT perform extraction or normalization here.
    """

    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 64,
        num_classes: int = 1,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize UNet detector.

        Args:
            input_channels: Input latent channels (default: 4)
            base_channels: Base channel count (default: 64)
            num_classes: Output classes (default: 1 for binary)
            depth: Number of downsampling levels (default: 4)
            use_batch_norm: Use batch normalization (default: True)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()

        # Defensive assertions - ensure all channel parameters are plain integers
        assert isinstance(input_channels, int), f"input_channels must be int, got {type(input_channels)}"
        assert isinstance(base_channels, int), f"base_channels must be int, got {type(base_channels)}"
        assert isinstance(depth, int), f"depth must be int, got {type(depth)}"
        assert input_channels > 0, f"input_channels must be positive, got {input_channels}"
        assert base_channels > 0, f"base_channels must be positive, got {base_channels}"
        assert depth > 0, f"depth must be positive, got {depth}"

        # Initial convolution
        self.inc = DoubleConv(input_channels, base_channels, batch_norm=use_batch_norm, dropout=dropout)

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** (i + 1))
            assert isinstance(out_ch, int), f"Calculated out_ch must be int, got {type(out_ch)}"
            self.down_blocks.append(DownBlock(in_ch, out_ch, use_batch_norm, dropout))
            in_ch = out_ch

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            out_ch = base_channels * (2 ** (depth - i - 1))
            assert isinstance(out_ch, int), f"Calculated out_ch must be int, got {type(out_ch)}"
            self.up_blocks.append(UpBlock(in_ch, out_ch, use_batch_norm, dropout))
            in_ch = out_ch

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(base_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor, validate: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input latent tensor [B, 4, 64, 64]
                - Input: VAE-encoded latent z_0
                - Shape: [B, 4, 64, 64]
                - Statistics: approximately zero-mean, scaled by VAE factor (≈0.18215)
                - Domain: spatial (not frequency)
            validate: If True, perform lightweight validation checks (default: False)

        Returns:
            Logits [B, 1]
        """
        # Shape assertion (hard error)
        assert x.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {x.dim()}D"
        assert x.shape[1:] == (4, 64, 64), f"Expected shape [B, 4, 64, 64], got {list(x.shape)}"
        
        # Distribution check (soft warning if validate=True)
        if validate:
            import warnings
            with torch.no_grad():
                mean_val = x.mean().item()
                
                # Warn if mean is significantly non-zero (VAE latents should be ~0 mean)
                if abs(mean_val) > 0.2:
                    warnings.warn(
                        f"Input mean {mean_val:.3f} is far from expected 0. "
                        "Expected VAE-style latent input.",
                        UserWarning,
                    )
                
                # Warn if absolute mean magnitude is abnormally large (detects discrete/bimodal inputs)
                mean_abs = x.abs().mean().item()
                if mean_abs > 1.5:
                    warnings.warn(
                        f"Input mean absolute value {mean_abs:.3f} "
                        "is inconsistent with expected VAE latent distribution.",
                        UserWarning,
                    )
        
        # Initial convolution
        x = self.inc(x)

        # Encoder with skip connections
        skips = []
        for down in self.down_blocks:
            skips.append(x)
            x = down(x)

        # Decoder with skip connections
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        # Classifier
        logits = self.classifier(x)

        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step (Lightning-style interface).

        Args:
            batch: Batch dictionary with 'image' and 'label'
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss' and metrics
        """
        images = batch["image"]
        labels = batch["label"].float()

        # Forward pass
        logits = self(images).squeeze(-1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step (Lightning-style interface).

        Args:
            batch: Batch dictionary with 'image' and 'label'
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss' and metrics
        """
        # Same as training step but without gradient computation
        return self.training_step(batch, batch_idx)


class BayesianDetector(nn.Module):
    """
    Likelihood-based Bayesian detector for watermark detection.
    
    Operates on precomputed g-values and uses trained likelihood models:
    - P(g | watermarked)
    - P(g | unwatermarked)
    
    Computes posterior probability P(watermarked | g) using Bayes' rule.
    
    This detector is:
        - Stateless: No internal state, pure computation
        - Metadata-free: Does not depend on manifests, seeds, or keys
        - G-value agnostic: Works with any g-values regardless of how computed
        - Trained: Uses learned likelihood parameters from train_g_likelihoods.py
    
    Input Contract:
        g: Tensor[B, N]        # Binary g-values per sample (values in {0, 1})
        mask: Optional[B, N]   # Optional mask for valid positions
    
    The detector must be initialized with trained likelihood parameters.
    """

    def __init__(
        self,
        likelihood_params_path: Optional[str] = None,
        threshold: float = 0.5,
        prior_watermarked: float = 0.5,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Bayesian detector with trained likelihood parameters.
        
        Args:
            likelihood_params_path: Path to JSON file with trained parameters.
                                  If None, uses default uniform model.
            threshold: Detection threshold for posterior probability.
                      Default: 0.5 (P(watermarked | g) > 0.5 means watermarked)
            prior_watermarked: Prior probability P(watermarked).
                              Default: 0.5 (uniform prior)
            mask: Optional mask tensor for validation during initialization.
                  If provided, validates that mask.sum() == model.num_positions
                  and logs mask checksum for alignment verification.
        """
        super().__init__()
        
        self.threshold = threshold
        self.prior_watermarked = prior_watermarked
        self.prior_unwatermarked = 1.0 - prior_watermarked
        
        # Initialize key fingerprint storage (will be set during load)
        self._stored_key_fingerprint = None
        self._stored_key_id = None
        self._stored_prf_algorithm = None
        
        # Load trained parameters
        if likelihood_params_path is not None:
            self._load_likelihood_params(likelihood_params_path)
        else:
            # Default: uniform model (P(g_i = 1) = 0.5 for all positions)
            self.num_positions = None
            self.probs_watermarked = None
            self.probs_unwatermarked = None
            self.use_trained = False
        
        # Validate mask alignment if provided during initialization
        if mask is not None and self.num_positions is not None:
            self._validate_mask_alignment(mask)
    
    def _load_likelihood_params(self, params_path: str) -> None:
        """
        Load trained likelihood parameters from JSON file.
        
        Args:
            params_path: Path to likelihood_params.json
        """
        import json
        from pathlib import Path
        
        path = Path(params_path)
        # Allow path without .json extension (e.g. best_model -> best_model.json)
        if not path.exists() and path.suffix != ".json":
            path_json = path.with_suffix(".json")
            if path_json.exists():
                path = path_json
        if not path.exists():
            raise FileNotFoundError(f"Likelihood parameters not found: {params_path}")

        with open(path, "r") as f:
            data = json.load(f)
        
        self.num_positions = data["num_positions"]
        self.probs_watermarked = torch.tensor(
            data["watermarked"]["probs"], dtype=torch.float32
        )
        self.probs_unwatermarked = torch.tensor(
            data["unwatermarked"]["probs"], dtype=torch.float32
        )
        self.use_trained = True
        
        # CRITICAL: Store key fingerprint from model file for verification
        self._stored_key_fingerprint = data.get("key_fingerprint")
        self._stored_key_id = data.get("key_id")
        self._stored_prf_algorithm = data.get("prf_algorithm")
    
    def _validate_mask_alignment(self, mask: torch.Tensor) -> None:
        """
        Validate that mask aligns with model's num_positions.
        
        This provides hard guarantees against silent mismatch between
        the mask used during training and the mask used during detection.
        
        Args:
            mask: Mask tensor to validate
        
        Raises:
            AssertionError: If mask.sum() != model.num_positions
        """
        import hashlib
        
        # Ensure mask is 1D for validation
        if mask.dim() > 1:
            mask_flat = mask.flatten()
        else:
            mask_flat = mask
        
        # Compute mask checksum for logging
        mask_np = mask_flat.detach().cpu().numpy()
        mask_bytes = mask_np.tobytes()
        mask_checksum = hashlib.sha256(mask_bytes).hexdigest()[:16]
        
        # Hard assertion: mask must match model geometry
        mask_sum = int(mask_flat.sum().item())
        assert mask_sum == self.num_positions, (
            f"Mask alignment mismatch: mask.sum()={mask_sum} != model.num_positions={self.num_positions}. "
            f"This indicates the mask used during detection does not match the mask used during training. "
            f"Mask checksum: {mask_checksum}"
        )
        
        # Log mask checksum for verification
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Mask alignment validated: mask.sum()={mask_sum} == model.num_positions={self.num_positions}, "
            f"mask_checksum: {mask_checksum}"
        )
    
    def score(
        self,
        g: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        expected_key_fingerprint: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection score from g-values.
        
        Computes:
        - log P(g | watermarked)
        - log P(g | unwatermarked)
        - log-odds: log P(watermarked | g) - log P(unwatermarked | g)
        - posterior: P(watermarked | g)
        
        Args:
            g: Binary g-values [B, N] with values in {0, 1}
            mask: Optional mask [B, N] with 1 for valid positions, 0 for invalid
            expected_key_fingerprint: Optional key fingerprint to verify against stored fingerprint.
                                     If provided and model has stored fingerprint, will raise error on mismatch.
        
        Returns:
            Dictionary with:
                - "log_likelihood_watermarked": log P(g | watermarked) [B]
                - "log_likelihood_unwatermarked": log P(g | unwatermarked) [B]
                - "log_odds": log-odds ratio [B]
                - "posterior": P(watermarked | g) [B]
                - "score": Detection score (log-odds) [B]
                - "decision": Binary decision (posterior > threshold) [B]
        """
        # CRITICAL: Verify key fingerprint if both are provided
        if expected_key_fingerprint is not None and self._stored_key_fingerprint is not None:
            if expected_key_fingerprint != self._stored_key_fingerprint:
                raise RuntimeError(
                    f"KEY MISMATCH: Likelihood model was trained with different key. "
                    f"Model key_fingerprint: {self._stored_key_fingerprint[:16]}..., "
                    f"Expected key_fingerprint: {expected_key_fingerprint[:16]}.... "
                    f"This model cannot score g-values from a different key. "
                    f"Use the correct key or retrain the model."
                )
        
        B, N = g.shape
        device = g.device
        
        # Normalize g to {0, 1} if needed
        if g.dtype in (torch.long, torch.int64):
            g = g.float()
        
        # Convert {-1, +1} to {0, 1} if needed
        unique_vals = torch.unique(g)
        unique_set = set(unique_vals.cpu().tolist())
        if unique_set.issubset({-1, 1}):
            g = (g + 1) / 2
        g = torch.clamp(torch.round(g), 0, 1)
        
        # Use trained parameters if available
        if self.use_trained:
            if N != self.num_positions:
                raise ValueError(
                    f"G-values length {N} does not match trained model "
                    f"positions {self.num_positions}"
                )
            
            probs_w = self.probs_watermarked.to(device)  # [N]
            probs_u = self.probs_unwatermarked.to(device)  # [N]
        else:
            # Default: uniform model
            probs_w = torch.full((N,), 0.5, device=device)
            probs_u = torch.full((N,), 0.5, device=device)
        
        # Expand to batch
        probs_w = probs_w.unsqueeze(0).expand(B, -1)  # [B, N]
        probs_u = probs_u.unsqueeze(0).expand(B, -1)  # [B, N]
        
        # Compute log-likelihoods per position
        # log P(g_i | class) = g_i * log(p_i) + (1 - g_i) * log(1 - p_i)
        log_p_w = g * torch.log(probs_w + 1e-10) + (1 - g) * torch.log(1 - probs_w + 1e-10)
        log_p_u = g * torch.log(probs_u + 1e-10) + (1 - g) * torch.log(1 - probs_u + 1e-10)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(device)
            
            # Validate mask alignment if using trained model
            if self.use_trained:
                # Ensure mask is 1D for validation
                if mask.dim() > 1:
                    mask_flat = mask.flatten()
                else:
                    mask_flat = mask
                
                # Hard assertion: mask must match model geometry
                mask_sum = int(mask_flat.sum().item())
                assert mask_sum == self.num_positions, (
                    f"Mask alignment mismatch in score(): mask.sum()={mask_sum} != model.num_positions={self.num_positions}. "
                    f"This indicates the mask used during detection does not match the mask used during training."
                )
            
            log_p_w = log_p_w * mask
            log_p_u = log_p_u * mask
        
        # Sum over positions
        log_likelihood_w = log_p_w.sum(dim=1)  # [B]
        log_likelihood_u = log_p_u.sum(dim=1)  # [B]
        
        # Compute log-odds (log posterior ratio)
        # log P(watermarked | g) - log P(unwatermarked | g)
        # = log P(g | watermarked) + log P(watermarked) - log P(g | unwatermarked) - log P(unwatermarked)
        log_odds = (
            log_likelihood_w
            + np.log(self.prior_watermarked)
            - log_likelihood_u
            - np.log(self.prior_unwatermarked)
        )
        
        # Compute posterior P(watermarked | g)
        # Using log-sum-exp trick for numerical stability
        log_posterior_w = log_likelihood_w + np.log(self.prior_watermarked)
        log_posterior_u = log_likelihood_u + np.log(self.prior_unwatermarked)
        
        # Normalize
        log_sum = torch.logsumexp(
            torch.stack([log_posterior_w, log_posterior_u], dim=0), dim=0
        )
        posterior = torch.exp(log_posterior_w - log_sum)  # [B]
        
        # Make decision
        decision = (posterior > self.threshold).long()  # [B]
        
        return {
            "log_likelihood_watermarked": log_likelihood_w,
            "log_likelihood_unwatermarked": log_likelihood_u,
            "log_odds": log_odds,
            "posterior": posterior,
            "score": log_odds,  # Use log-odds as main score
            "decision": decision,
        }
    
    def forward(
        self,
        g: torch.Tensor,
        key: Any = None,
        threshold: Optional[float] = None,
        master_key: Optional[str] = None,
        validate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute detection statistics from g-values.
        
        NOTE: This method is kept for backward compatibility but key-dependent
        logic is deprecated. Use score() method instead.
        
        Args:
            g: Binary g-values [B, N]
            key: Deprecated - not used
            threshold: Detection threshold (overrides self.threshold if provided)
            master_key: Deprecated - not used
            validate: If True, perform validation checks (default: False)
            
        Returns:
            Dictionary with detection results
        """
        # Use score() method
        result = self.score(g, mask=None)
        
        # Override threshold if provided
        if threshold is not None:
            decision = (result["posterior"] > threshold).long()
            result["decision"] = decision
        
        # Rename for backward compatibility
        result["matches"] = torch.zeros_like(result["score"])  # Placeholder
        result["num_bits"] = torch.tensor(g.shape[1], device=g.device)
        
        return result


