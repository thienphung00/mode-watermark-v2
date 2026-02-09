#!/usr/bin/env python3
"""
Train Bayesian likelihood models for g-values (one per detector family).

This script trains two likelihood models per family:
- P(g | watermarked)
- P(g | unwatermarked)

Samples are pooled across all transforms (no per-transform likelihoods). Model is
p(score | watermark, T ~ P_real) and p(score | clean, T ~ P_real).

MAPPING MODE SUPPORT:
- Binary (mapping_mode="binary"): Uses Bernoulli likelihood
  log p(g | wm) = Σ [ g·log(p) + (1−g)·log(1−p) ]
  
- Continuous (mapping_mode="continuous"): Uses Gaussian likelihood
  g ~ Normal(μ_wm, σ_wm²) vs Normal(μ_clean, σ_clean²)
  log p(g | wm) = Σ [ -0.5 * log(2π) - log(σ) - 0.5 * ((g - μ) / σ)² ]

The likelihood type is determined from metadata.json (created by precompute_inverted_g_values.py)
and persisted in the output likelihood_params.json for detection-time validation.

Inputs:
- Precomputed inverted g-values (created by precompute_inverted_g_values.py),
  optionally with "transform" in manifest entries (all transforms are pooled).
- Corresponding labels: watermarked / unwatermarked

This script only loads precomputed g-values - no DDIM inversion, no image loading.
For Tier-2 training, use precompute_inverted_g_values.py first to precompute g-values
from images (including transformed views) using DDIM inversion.

Usage:
    # Step 1: Precompute inverted g-values
    python scripts/precompute_inverted_g_values.py \
        --manifest path/to/train_manifest.jsonl \
        --output-dir path/to/precomputed_g_values \
        --config-path configs/experiments/seedbias.yaml \
        --master-key "your_secret_key" \
        --num-inversion-steps 20
    
    # Step 2: Train likelihood models (train split only)
    python scripts/train_g_likelihoods.py \
        --g-manifest path/to/precomputed_g_values/g_manifest_train.jsonl \
        --output-dir outputs/likelihood_models_train \
        --phase train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# Constants
# ==============================================================================
# Minimum standard deviation floor for Gaussian likelihood to prevent numerical issues
GAUSSIAN_STD_FLOOR = 1e-3


# ==============================================================================
# Mapping Mode Validation Utilities
# ==============================================================================
def validate_mapping_mode(mapping_mode: str) -> None:
    """
    Validate that mapping_mode is a valid value.
    
    Args:
        mapping_mode: The mapping mode to validate
        
    Raises:
        ValueError: If mapping_mode is invalid
    """
    valid_modes = {"binary", "continuous"}
    if mapping_mode not in valid_modes:
        raise ValueError(
            f"Invalid mapping_mode: '{mapping_mode}'. "
            f"Must be one of: {valid_modes}"
        )


def validate_g_values_for_mapping_mode(
    g: torch.Tensor,
    mapping_mode: str,
    epsilon: float = 1e-3,
) -> None:
    """
    Validate that g-values are consistent with the declared mapping_mode.
    
    For binary mode: Assert all g ∈ {0, 1}
    For continuous mode: Raise if std(g) < epsilon (indicates accidental binarization)
    
    Args:
        g: G-values tensor [N] or [B, N]
        mapping_mode: "binary" or "continuous"
        epsilon: Minimum std threshold for continuous mode
        
    Raises:
        ValueError: If g-values are inconsistent with mapping_mode
    """
    g_flat = g.flatten()
    
    if mapping_mode == "binary":
        # For binary mode: all values must be in {0, 1}
        unique_vals = torch.unique(g_flat)
        unique_set = set(unique_vals.cpu().tolist())
        
        # Check for {0, 1} or {-1, 1} format
        valid_binary = unique_set.issubset({0.0, 1.0}) or unique_set.issubset({-1.0, 1.0})
        if not valid_binary:
            raise ValueError(
                f"mapping_mode is 'binary' but g-values contain non-binary values. "
                f"Unique values: {sorted(unique_set)[:10]}{'...' if len(unique_set) > 10 else ''}. "
                f"Expected all values in {{0, 1}} or {{-1, 1}}."
            )
    
    elif mapping_mode == "continuous":
        # For continuous mode: std must be above threshold
        g_std = g_flat.std().item()
        if g_std < epsilon:
            # Additional check: if all values are in {0, 1}, this is definitely wrong
            unique_vals = torch.unique(g_flat)
            unique_set = set(unique_vals.cpu().tolist())
            if unique_set.issubset({0.0, 1.0}):
                raise ValueError(
                    f"mapping_mode is 'continuous' but g-values appear to be binarized. "
                    f"All values are in {{0, 1}} (std={g_std:.6f} < {epsilon}). "
                    f"This indicates accidental binarization in the pipeline. "
                    f"Check precompute_inverted_g_values.py and ensure mapping_mode is correctly propagated."
                )
            else:
                raise ValueError(
                    f"mapping_mode is 'continuous' but g-values have near-zero variance. "
                    f"std(g) = {g_std:.6f} < {epsilon}. "
                    f"This may indicate degenerate g-values or pipeline misconfiguration."
                )


class GValueLikelihoodModel(nn.Module):
    """
    Simple likelihood model for g-values.
    
    Models P(g | class) as independent Bernoulli per position:
        P(g_i = 1 | class) = sigmoid(bias_i)
    
    For unwatermarked: bias_i ≈ 0 (P ≈ 0.5)
    For watermarked: bias_i can be learned (P can deviate from 0.5)
    
    Note: This model assumes a single global mask geometry. The model is only
    valid for detectors using the same mask pattern. G-values passed to forward()
    should already be masked (only valid positions).
    """
    
    def __init__(self, num_positions: int):
        """
        Initialize likelihood model.
        
        Args:
            num_positions: Number of g-value positions (N)
        """
        super().__init__()
        # Per-position bias parameters (logits)
        # Initialize near 0 (P ≈ 0.5 for unwatermarked)
        self.biases = nn.Parameter(torch.zeros(num_positions))
    
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood log P(g | class).
        
        Args:
            g: Binary g-values [B, N_eff] with values in {0, 1}
               Must already be masked (only valid positions)
        
        Returns:
            Log-likelihood [B]
        """
        B, N_eff = g.shape
        
        # Strict shape validation: g must match model size
        if N_eff != len(self.biases):
            raise ValueError(
                f"G-values length {N_eff} does not match model positions {len(self.biases)}"
            )
        
        # Get per-position probabilities
        probs = torch.sigmoid(self.biases)  # [N_eff]
        
        # Expand to batch
        probs = probs.unsqueeze(0).expand(B, -1)  # [B, N_eff]
        
        # Compute log-likelihood per position
        # log P(g_i | class) = g_i * log(p_i) + (1 - g_i) * log(1 - p_i)
        log_probs = g * torch.log(probs + 1e-10) + (1 - g) * torch.log(1 - probs + 1e-10)
        
        # Sum over positions (all positions are valid since g is already masked)
        log_likelihood = log_probs.sum(dim=1)  # [B]
        
        return log_likelihood
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get learned parameters as numpy arrays.
        
        Returns:
            Dictionary with 'biases' (logits) and 'probs' (probabilities)
        """
        with torch.no_grad():
            biases = self.biases.cpu().numpy()
            probs = torch.sigmoid(self.biases).cpu().numpy()
        
        return {
            "biases": biases.tolist(),
            "probs": probs.tolist(),
            "likelihood_type": "bernoulli",
        }


class GaussianLikelihoodModel(nn.Module):
    """
    Gaussian likelihood model for continuous g-values.
    
    Models P(g | class) as independent Gaussian per position:
        g_i ~ Normal(μ_i, σ_i²)
    
    For unwatermarked: μ_i ≈ 0, σ_i ≈ 1 (approximately standard normal)
    For watermarked: μ_i and σ_i are learned (can deviate from standard normal)
    
    CRITICAL: This model assumes continuous g-values in ℝ.
    Do NOT use with binarized g-values.
    
    Note: This model assumes a single global mask geometry. The model is only
    valid for detectors using the same mask pattern. G-values passed to forward()
    should already be masked (only valid positions).
    """
    
    def __init__(self, num_positions: int, std_floor: float = GAUSSIAN_STD_FLOOR):
        """
        Initialize Gaussian likelihood model.
        
        Args:
            num_positions: Number of g-value positions (N)
            std_floor: Minimum allowed standard deviation (prevents numerical issues)
        """
        super().__init__()
        self.std_floor = std_floor
        
        # Per-position mean parameters
        # Initialize to 0 (approximately standard normal for unwatermarked)
        self.means = nn.Parameter(torch.zeros(num_positions))
        
        # Per-position log-std parameters (use log for positivity constraint)
        # Initialize to log(1) = 0 (std ≈ 1 for approximately unit variance)
        self.log_stds = nn.Parameter(torch.zeros(num_positions))
    
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood log P(g | class).
        
        Args:
            g: Continuous g-values [B, N_eff] with values in ℝ
               Must already be masked (only valid positions)
        
        Returns:
            Log-likelihood [B]
        """
        B, N_eff = g.shape
        
        # Strict shape validation: g must match model size
        if N_eff != len(self.means):
            raise ValueError(
                f"G-values length {N_eff} does not match model positions {len(self.means)}"
            )
        
        # Get per-position parameters
        means = self.means  # [N_eff]
        stds = torch.exp(self.log_stds).clamp(min=self.std_floor)  # [N_eff]
        
        # Expand to batch
        means = means.unsqueeze(0).expand(B, -1)  # [B, N_eff]
        stds = stds.unsqueeze(0).expand(B, -1)  # [B, N_eff]
        
        # Compute Gaussian log-likelihood per position
        # log P(g_i | class) = -0.5 * log(2π) - log(σ_i) - 0.5 * ((g_i - μ_i) / σ_i)²
        log_probs = (
            -0.5 * np.log(2 * np.pi)
            - torch.log(stds)
            - 0.5 * ((g - means) / stds) ** 2
        )
        
        # Sum over positions (all positions are valid since g is already masked)
        log_likelihood = log_probs.sum(dim=1)  # [B]
        
        return log_likelihood
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get learned parameters as numpy arrays.
        
        Returns:
            Dictionary with 'means', 'stds', and 'likelihood_type'
        """
        with torch.no_grad():
            means = self.means.cpu().numpy()
            stds = torch.exp(self.log_stds).clamp(min=self.std_floor).cpu().numpy()
        
        return {
            "means": means.tolist(),
            "stds": stds.tolist(),
            "likelihood_type": "gaussian",
        }


class GValueDataset(Dataset):
    """
    Dataset for loading precomputed g-values and labels.
    
    This dataset only loads precomputed g-values from disk.
    DDIM inversion should be performed separately using precompute_inverted_g_values.py.
    
    MAPPING MODE HANDLING:
    - Binary: G-values are binarized to {0, 1} for Bernoulli likelihood
    - Continuous: G-values are preserved as-is (real-valued) for Gaussian likelihood
    """
    
    def __init__(
        self,
        manifest_path: Path,
        g_key: str = "g_path",
        label_key: str = "label",
        mapping_mode: str = "binary",
    ):
        """
        Initialize dataset.
        
        Args:
            manifest_path: Path to manifest.jsonl file
            g_key: Key in manifest for g-values path
            label_key: Key in manifest for label
            mapping_mode: "binary" or "continuous" - determines g-value handling
        """
        self.manifest_path = manifest_path
        self.g_key = g_key
        self.label_key = label_key
        self.mapping_mode = mapping_mode
        
        # Validate mapping mode
        validate_mapping_mode(mapping_mode)
        
        self.samples = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest.jsonl file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        samples = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Loads precomputed g-values from disk (no DDIM inversion).
        G-values should be precomputed using precompute_inverted_g_values.py.
        
        Handling depends on mapping_mode:
        - Binary: G-values are binarized to {0, 1}
        - Continuous: G-values are preserved as real-valued
        
        Returns:
            Dictionary with:
                - 'g': g-values tensor [N] (format depends on mapping_mode)
                - 'label': binary label (1 for watermarked, 0 for unwatermarked)
        """
        sample = self.samples[idx]
        
        # Load precomputed g-values from disk
        g_path_str = sample.get(self.g_key) or sample.get("g_path")
        if not g_path_str:
            raise ValueError(f"Sample {idx} missing '{self.g_key}' field")
        
        g_path = Path(g_path_str)
        if not g_path.is_absolute():
            g_path = self.manifest_path.parent / g_path
        
        if not g_path.exists():
            raise FileNotFoundError(f"G-values file not found: {g_path}")
        
        # Load g-values
        g_data = torch.load(g_path, map_location="cpu")
        
        # Extract g tensor
        if isinstance(g_data, dict):
            g = g_data.get("g", g_data.get("g_values"))
            if g is None:
                raise ValueError(f"G-values file {g_path} missing 'g' key")
        else:
            g = g_data
        
        # Ensure 1D
        if g.dim() == 0:
            raise ValueError(f"G-values must be 1D, got scalar")
        if g.dim() > 1:
            g = g.flatten()
        
        # Convert to float if needed
        if g.dtype in (torch.long, torch.int64):
            g = g.float()
        
        # Handle g-value processing based on mapping_mode
        if self.mapping_mode == "binary":
            # Binary mode: Binarize g-values for Bernoulli likelihood
            # Handle {-1, 1} format: convert to {0, 1}
            unique_vals = torch.unique(g)
            if set(unique_vals.cpu().tolist()).issubset({-1.0, 1.0}):
                g = (g + 1) / 2
            
            # Ensure binary {0, 1} format
            g = torch.clamp(torch.round(g), 0, 1)
            
            # Binarize g-values for Bernoulli likelihood model
            g = (g > 0).float()
            
        elif self.mapping_mode == "continuous":
            # Continuous mode: Preserve real-valued g-values exactly
            # Only ensure float32 dtype, no binarization
            if g.dtype != torch.float32:
                g = g.float()
        
        # Extract label
        label = self._extract_label(sample)
        
        return {
            "g": g,
            "label": torch.tensor(label, dtype=torch.long),
        }
    
    def _extract_label(self, sample: Dict) -> int:
        """Extract binary label from sample."""
        label = sample.get(self.label_key) or sample.get("is_watermarked") or sample.get("watermarked")
        
        if label is None:
            return 0
        
        # Convert to int
        if isinstance(label, bool):
            return 1 if label else 0
        elif isinstance(label, str):
            label_lower = label.lower()
            if label_lower in ("true", "1", "watermarked", "yes"):
                return 1
            elif label_lower in ("false", "0", "unwatermarked", "no"):
                return 0
            else:
                return int(label)
        else:
            return int(label)


def train_likelihood_models(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_positions: int,
    mapping_mode: str = "binary",
    num_epochs: int = 10,
    lr: float = 0.01,
    device: str = "cpu",
    early_stop: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train two likelihood models: P(g | watermarked) and P(g | unwatermarked).
    
    Model type is selected based on mapping_mode:
    - Binary: Uses GValueLikelihoodModel (Bernoulli likelihood)
    - Continuous: Uses GaussianLikelihoodModel (Gaussian likelihood)
    
    Args:
        train_loader: Training data loader
        val_loader: Optional validation data loader
        num_positions: Number of effective g-value positions (N_eff)
        mapping_mode: "binary" or "continuous" - determines likelihood model type
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        early_stop: Whether to enable early stopping based on validation loss
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum improvement in validation loss to reset patience counter
    
    Returns:
        Tuple of (watermarked_model, unwatermarked_model)
    """
    # Validate mapping mode
    validate_mapping_mode(mapping_mode)
    
    # Create models based on mapping_mode
    if mapping_mode == "binary":
        model_w = GValueLikelihoodModel(num_positions).to(device)
        model_u = GValueLikelihoodModel(num_positions).to(device)
        likelihood_type = "bernoulli"
    elif mapping_mode == "continuous":
        model_w = GaussianLikelihoodModel(num_positions).to(device)
        model_u = GaussianLikelihoodModel(num_positions).to(device)
        likelihood_type = "gaussian"
    else:
        raise ValueError(f"Invalid mapping_mode: {mapping_mode}")
    
    # Optimizers
    optimizer_w = torch.optim.Adam(model_w.parameters(), lr=lr)
    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=lr)
    
    # Early stopping setup
    # Automatically disable early stopping if no validation loader is provided
    if early_stop and val_loader is None:
        print("Warning: Early stopping requested but no validation loader provided. Disabling early stopping.")
        early_stop = False
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_w_state = None
    best_model_u_state = None
    
    print(f"Training likelihood models...")
    print(f"  Mapping mode: {mapping_mode}")
    print(f"  Likelihood type: {likelihood_type}")
    print(f"  Effective positions (N_eff): {num_positions}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    if early_stop:
        print(f"  Early stopping: enabled (patience={patience}, min_delta={min_delta})")
    
    for epoch in range(num_epochs):
        # Train both models in single pass through data
        model_w.train()
        model_u.train()
        train_loss_w = 0.0
        train_loss_u = 0.0
        train_count_w = 0
        train_count_u = 0
        
        for batch in train_loader:
            g = batch["g"].to(device)
            labels = batch["label"].to(device)
            
            # Train watermarked model on watermarked samples
            wm_mask = (labels == 1)
            if wm_mask.sum() > 0:
                g_w = g[wm_mask]
                log_likelihood = model_w(g_w)
                loss_w = -log_likelihood.mean()
                
                optimizer_w.zero_grad()
                loss_w.backward()
                optimizer_w.step()
                
                train_loss_w += loss_w.item()
                train_count_w += 1
            
            # Train unwatermarked model on unwatermarked samples
            clean_mask = (labels == 0)
            if clean_mask.sum() > 0:
                g_u = g[clean_mask]
                log_likelihood = model_u(g_u)
                loss_u = -log_likelihood.mean()
                
                optimizer_u.zero_grad()
                loss_u.backward()
                optimizer_u.step()
                
                train_loss_u += loss_u.item()
                train_count_u += 1
        
        # Validation
        val_loss_w = 0.0
        val_loss_u = 0.0
        val_count = 0
        
        if val_loader is not None:
            model_w.eval()
            model_u.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    g = batch["g"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Watermarked
                    wm_mask = (labels == 1)
                    if wm_mask.sum() > 0:
                        g_w = g[wm_mask]
                        log_likelihood = model_w(g_w)
                        val_loss_w += (-log_likelihood.mean()).item()
                    
                    # Unwatermarked
                    clean_mask = (labels == 0)
                    if clean_mask.sum() > 0:
                        g_u = g[clean_mask]
                        log_likelihood = model_u(g_u)
                        val_loss_u += (-log_likelihood.mean()).item()
                    
                    val_count += 1
        
        # Log
        avg_train_loss_w = train_loss_w / max(train_count_w, 1)
        avg_train_loss_u = train_loss_u / max(train_count_u, 1)
        
        if val_count > 0:
            avg_val_loss_w = val_loss_w / val_count
            avg_val_loss_u = val_loss_u / val_count
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_w={avg_train_loss_w:.4f}, train_u={avg_train_loss_u:.4f}, "
                f"val_w={avg_val_loss_w:.4f}, val_u={avg_val_loss_u:.4f}"
            )
            
            # Early stopping check
            if early_stop:
                avg_val_loss = avg_val_loss_w + avg_val_loss_u
                
                # Check if validation loss improved by more than min_delta
                if best_val_loss - avg_val_loss > min_delta:
                    # Improvement found: reset patience counter and save best model states
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_model_w_state = {k: v.clone() for k, v in model_w.state_dict().items()}
                    best_model_u_state = {k: v.clone() for k, v in model_u.state_dict().items()}
                else:
                    # No improvement: increment patience counter
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                        break
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_w={avg_train_loss_w:.4f}, train_u={avg_train_loss_u:.4f}"
            )
    
    # Restore best model weights if early stopping was used and we have saved states
    if early_stop and best_model_w_state is not None and best_model_u_state is not None:
        model_w.load_state_dict(best_model_w_state)
        model_u.load_state_dict(best_model_u_state)
        print(f"Restored best model weights from epoch {best_epoch}")
    
    return model_w, model_u


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian likelihood models for g-values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--g-manifest",
        type=str,
        required=True,
        help="Path to g-values manifest.jsonl (created by precompute_inverted_g_values.py)",
    )
    
    parser.add_argument(
        "--val-g-manifest",
        type=str,
        default=None,
        help="Path to validation g-values manifest.jsonl (optional)",
    )
    
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to metadata.json (created by precompute_inverted_g_values.py). "
             "If not provided, will look for metadata.json in the same directory as --g-manifest.",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for likelihood model JSON (default: {output-dir}/likelihood_params.json)",
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset phase: only 'train' is valid for likelihood training. Use g_manifest_train.jsonl.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/likelihood_models_train",
        help="Directory to save trained models (default: outputs/likelihood_models_train for phase=train)",
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    
    # Early stopping arguments
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="Enable early stopping based on validation loss",
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before stopping (only used with --early-stop)",
    )
    
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in validation loss to reset patience counter (only used with --early-stop)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("Training Bayesian Likelihood Models")
    print("=" * 60)
    
    # Load manifest
    train_manifest_path = Path(args.g_manifest)
    
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    else:
        metadata_path = train_manifest_path.parent / "metadata.json"
    
    training_metadata = None
    mapping_mode = "binary"  # Default, will be overridden by metadata
    
    if metadata_path.exists():
        print(f"\nLoading metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            training_metadata = json.load(f)
        print(f"  Latent type: {training_metadata.get('latent_type', 'unknown')}")
        print(f"  Inversion steps: {training_metadata.get('num_inversion_steps', 'unknown')}")
        print(f"  G-field config hash: {training_metadata.get('g_field_config_hash', 'unknown')}")
        
        # CRITICAL: Extract mapping_mode from metadata (required for correct likelihood)
        mapping_mode = training_metadata.get("mapping_mode")
        if mapping_mode is None:
            # Try to get from g_field_config
            g_field_config = training_metadata.get("g_field_config", {})
            if isinstance(g_field_config, dict):
                mapping_mode = g_field_config.get("mapping_mode", "binary")
            else:
                mapping_mode = "binary"
        
        print(f"  Mapping mode: {mapping_mode}")
    else:
        print(f"\n⚠️  Warning: Metadata file not found at {metadata_path}")
        print("  Train-detect symmetry checks may fail.")
        print("  Defaulting to mapping_mode='binary'")
    
    # Validate mapping_mode once after loading metadata
    validate_mapping_mode(mapping_mode)
    print(f"\n✓ Mapping mode: {mapping_mode}")
    
    train_dataset = GValueDataset(train_manifest_path, mapping_mode=mapping_mode)
    print(f"Training samples: {len(train_dataset)}")
    
    # Get num_positions from first sample
    first_sample = train_dataset[0]
    num_positions = first_sample["g"].shape[-1]
    
    print(f"Effective masked positions (N_eff): {num_positions}")
    print("  (G-values are already masked from precomputation)")
    
    # Validate g-values once on first batch only
    print("Validating g-value consistency on first batch...")
    validate_g_values_for_mapping_mode(first_sample["g"], mapping_mode)
    print("✓ G-value consistency validated")
    
    # Validation dataset
    val_dataset = None
    if args.val_g_manifest:
        val_dataset = GValueDataset(Path(args.val_g_manifest), mapping_mode=mapping_mode)
        print(f"Validation samples: {len(val_dataset)}")
        
        val_first = val_dataset[0]
        val_g_shape = val_first["g"].shape[-1]
        if val_g_shape != num_positions:
            raise ValueError(
                f"Validation set has {val_g_shape} g-value positions, "
                f"but training set has {num_positions}. Shapes must match."
            )
    
    # Create dataloaders (default collation works since g-values are fixed size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device != "cpu"),
        )
    
    # Train models
    model_w, model_u = train_likelihood_models(
        train_loader,
        val_loader,
        num_positions,
        mapping_mode=mapping_mode,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    
    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "likelihood_params.json"
    
    # Save as JSON (simple format)
    params_w = model_w.get_parameters()
    params_u = model_u.get_parameters()
    
    # Determine likelihood_type from model parameters
    likelihood_type = params_w.get("likelihood_type", "bernoulli")
    
    output_data = {
        "num_positions": num_positions,
        "mapping_mode": mapping_mode,  # CRITICAL: Required for detection-time validation
        "likelihood_type": likelihood_type,  # "bernoulli" or "gaussian"
        "watermarked": params_w,
        "unwatermarked": params_u,
    }
    
    # CRITICAL: Extract and store key fingerprint from training data
    key_fingerprint = None
    key_id = None
    prf_algorithm = None
    
    # Check if key info is in metadata (from Phase-1 exports)
    if training_metadata is not None:
        key_fingerprint = training_metadata.get("key_fingerprint")
        key_id = training_metadata.get("key_id")
        prf_algorithm = training_metadata.get("prf_algorithm")
    
    # Store key fingerprint in output
    output_data["key_fingerprint"] = key_fingerprint
    output_data["key_id"] = key_id
    output_data["prf_algorithm"] = prf_algorithm
    
    # Save training metadata for train-detect symmetry verification
    if training_metadata is not None:
        output_data["training_metadata"] = {
            "latent_type": training_metadata.get("latent_type", "unknown"),
            "num_inversion_steps": training_metadata.get("num_inversion_steps"),
            "g_field_config_hash": training_metadata.get("g_field_config_hash"),
            "g_field_config": training_metadata.get("g_field_config"),
        }
    else:
        output_data["training_metadata"] = {
            "latent_type": "unknown",
            "num_inversion_steps": None,
            "g_field_config_hash": None,
            "g_field_config": None,
        }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved likelihood parameters to: {output_path}")
    
    # Also save as torch checkpoint
    checkpoint_path = output_dir / "likelihood_models.pt"
    torch.save({
        "watermarked": model_w.state_dict(),
        "unwatermarked": model_u.state_dict(),
        "num_positions": num_positions,
    }, checkpoint_path)
    
    print(f"✓ Saved model checkpoint to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
