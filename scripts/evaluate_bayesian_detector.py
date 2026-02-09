#!/usr/bin/env python3
"""
Bayesian Detector Controlled Evaluation & Signal-Strength Tuning

This script provides controlled evaluation and signal-strength tuning for the
Bayesian watermark detector without changing detector math or likelihood logic.

Features:
- ROC curve & AUC computation at fixed inversion steps (25)
- Adjustable detection threshold (evaluation-side only)
- Comprehensive logging (N_eff, config hash, threshold, AUC)
- Signal-strength tuning via config (mask density, frequency band, normalization)

This script is generation-side + evaluation-side only. Detector correctness
and likelihood logic remain unchanged.

Usage:
    python scripts/evaluate_bayesian_detector.py \
        --g-manifest outputs/test_data/g_manifest.jsonl \
        --likelihood-params outputs/likelihood_models/likelihood_params.json \
        --output-dir outputs/evaluation \
        --log-odds-threshold 0.0 \
        --plot-roc

For signal-strength tuning, adjust parameters in seedbias.yaml:
- Mask density: algorithm_params.mask.strength, cutoff_freq, bandwidth_fraction
- Frequency band: algorithm_params.g_field.low_freq_cutoff, high_freq_cutoff (via seed_bias)
- G-field normalization: algorithm_params.g_field.normalize.unit_variance
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.models.detectors import BayesianDetector
from src.detection.calibration import find_threshold_at_fpr


def load_manifest(manifest_path: Path) -> List[Dict]:
    """
    Load manifest.jsonl file.
    
    Args:
        manifest_path: Path to manifest.jsonl file
    
    Returns:
        List of manifest entries
    """
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def extract_key_id(entry: Dict) -> str:
    """Extract key_id from manifest entry (for logging only)."""
    return (
        entry.get("key_id") or
        entry.get("sample_id") or
        entry.get("image_id") or
        entry.get("id") or
        "default_key"
    )


def extract_label(entry: Dict) -> int:
    """Extract binary label from manifest entry."""
    label = entry.get("label") or entry.get("is_watermarked") or 0
    if isinstance(label, bool):
        return 1 if label else 0
    return int(label)


def compute_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from scores and labels.
    
    Args:
        scores: Detection scores (log_odds) [N]
        labels: True labels (0 or 1) [N]
        num_thresholds: Number of threshold points for ROC curve
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
        - fpr: False Positive Rate at each threshold
        - tpr: True Positive Rate at each threshold
        - thresholds: Threshold values used
    """
    # Count positives and negatives
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        # Edge case: all labels are same class
        thresholds = np.linspace(scores.max() + 1.0, scores.min() - 1.0, num_thresholds)
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), thresholds
    
    # Sort scores in descending order (higher scores first)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Get unique score values to use as thresholds
    unique_scores = np.unique(scores)
    
    if len(unique_scores) <= num_thresholds:
        # Use all unique scores plus boundaries
        thresholds = np.concatenate([
            [scores.max() + 1.0],  # Above maximum (all negative)
            np.sort(unique_scores)[::-1],  # Descending order
            [scores.min() - 1.0],  # Below minimum (all positive)
        ])
    else:
        # Use evenly spaced thresholds
        thresholds = np.linspace(scores.max() + 1.0, scores.min() - 1.0, num_thresholds)
    
    # Compute TPR and FPR at each threshold using efficient cumulative counts
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    
    # Use binary search to find threshold positions for efficiency
    for i, threshold in enumerate(thresholds):
        # Count samples with score > threshold
        # Since sorted_scores is in descending order, count from the start
        above_threshold = sorted_scores > threshold
        tp = np.sum(sorted_labels[above_threshold] == 1)
        fp = np.sum(sorted_labels[above_threshold] == 0)
        
        tpr[i] = tp / n_pos if n_pos > 0 else 0.0
        fpr[i] = fp / n_neg if n_neg > 0 else 0.0
    
    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve (AUC) using trapezoidal rule.
    
    Args:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        
    Returns:
        AUC score (0.0 to 1.0)
    """
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    
    return float(auc)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: Path,
    inversion_steps: int = 25,
    latent_type: str = "zT",
    threshold: float = 0.0,
):
    """
    Plot ROC curve and save to disk.
    
    Args:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        auc: AUC score
        output_path: Path to save plot
        inversion_steps: Number of inversion steps (for title)
        latent_type: Latent type (for title)
        threshold: Detection threshold used (for title)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping ROC plot")
        return
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        f"ROC Curve (Inversion Steps: {inversion_steps}, Latent Type: {latent_type}, Threshold: {threshold:.2f})",
        fontsize=14,
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ ROC curve saved to: {output_path}")


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    scores: List[float],
) -> Dict[str, float]:
    """
    Compute detection metrics.
    
    Args:
        predictions: Predicted labels (0 or 1)
        labels: True labels (0 or 1)
        scores: Detection scores (log_odds)
    
    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Basic metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Score statistics
    watermarked_scores = scores[labels == 1]
    unwatermarked_scores = scores[labels == 0]
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(labels),
        "score_mean_watermarked": float(np.mean(watermarked_scores)) if len(watermarked_scores) > 0 else 0.0,
        "score_std_watermarked": float(np.std(watermarked_scores)) if len(watermarked_scores) > 0 else 0.0,
        "score_mean_unwatermarked": float(np.mean(unwatermarked_scores)) if len(unwatermarked_scores) > 0 else 0.0,
        "score_std_unwatermarked": float(np.std(unwatermarked_scores)) if len(unwatermarked_scores) > 0 else 0.0,
    }


def determine_mapping_mode(
    manifest_path: Path,
    entries: List[Dict],
) -> str:
    """
    Determine mapping_mode from metadata or manifest entries.
    
    Precedence:
    1. entry["mapping_mode"] (if present in any manifest entry)
    2. metadata.json["g_field_config"]["mapping_mode"]
    3. Default to "binary"
    
    Args:
        manifest_path: Path to manifest.jsonl file
        entries: List of manifest entries
        
    Returns:
        mapping_mode: "binary" or "continuous"
    """
    # First, check manifest entries for mapping_mode
    for entry in entries:
        if "mapping_mode" in entry:
            mapping_mode = entry["mapping_mode"]
            if mapping_mode in {"binary", "continuous"}:
                return mapping_mode
    
    # Second, check metadata.json
    metadata_path = manifest_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                g_field_config = metadata.get("g_field_config", {})
                if isinstance(g_field_config, dict):
                    mapping_mode = g_field_config.get("mapping_mode")
                    if mapping_mode in {"binary", "continuous"}:
                        return mapping_mode
        except Exception:
            pass  # Fall back to default
    
    # Default to binary for backward compatibility
    return "binary"


def detect_watermark_bayesian_from_g_values(
    g: torch.Tensor,
    mask: Optional[torch.Tensor],
    detector: BayesianDetector,
    mapping_mode: str = "binary",
) -> dict:
    """
    Detect watermark using Bayesian detector with precomputed g-values.
    
    Args:
        g: Precomputed g-values tensor [N] or [1, N] (raw, will be normalized/binarized if binary)
        mask: Optional mask tensor [N] or [1, N]
        detector: Pre-loaded BayesianDetector
        mapping_mode: "binary" or "continuous" - determines whether to binarize g-values
        
    Returns:
        Detection result dictionary with log_odds
    """
    # Ensure batch dimension
    if g.dim() == 1:
        g = g.unsqueeze(0)  # [1, N]
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)  # [1, N]
    
    # Convert to float32 if needed
    if g.dtype in (torch.long, torch.int64):
        g = g.float()
    
    # Handle mapping mode
    if mapping_mode == "binary":
        # Binary mapping: binarize g-values to {0, 1}
        # Handle {-1, 1} format: convert to {0, 1}
        unique_vals = torch.unique(g)
        if set(unique_vals.cpu().tolist()).issubset({-1.0, 1.0}):
            g = (g + 1) / 2
        
        # Ensure values are in [0, 1] range and binarize
        g = torch.clamp(torch.round(g), 0, 1)
        g = (g > 0).float()
    elif mapping_mode == "continuous":
        # Continuous mapping: preserve floating-point values exactly as stored
        # Only ensure float32 dtype
        if g.dtype != torch.float32:
            g = g.float()
        # No binarization or rounding - preserve exact values
        
        # Runtime consistency check: warn if g-values appear binary
        unique_vals = torch.unique(g)
        unique_set = set(unique_vals.cpu().tolist())
        if unique_set.issubset({0.0, 1.0}):
            import warnings
            warnings.warn(
                f"mapping_mode is 'continuous' but g-values appear binary (all values in {{0, 1}}). "
                f"This may indicate a pipeline misconfiguration."
            )
    else:
        raise ValueError(f"Invalid mapping_mode: {mapping_mode}. Must be 'binary' or 'continuous'")
    
    # Run detection
    result = detector.score(g, mask)
    
    # Extract log_odds
    log_odds = result["log_odds"].item() if result["log_odds"].numel() == 1 else result["log_odds"][0].item()
    
    return {
        "log_odds": float(log_odds),
    }


def compute_n_eff(mask: Optional[np.ndarray]) -> int:
    """
    Compute effective number of positions (N_eff) from mask.
    
    Args:
        mask: Optional mask array [N] or [C, H, W]
        
    Returns:
        Number of effective positions (mask.sum() if mask provided, else 0)
    """
    if mask is None:
        return 0
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask_flat = mask.flatten()
    n_eff = int(np.sum(mask_flat > 0.5))  # Count positions with mask > 0.5
    
    return n_eff


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Compute hash of configuration for logging/tracking.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Hexadecimal hash string (first 16 characters)
    """
    # Convert to JSON string and compute hash
    config_str = json.dumps(config_dict, sort_keys=True)
    config_bytes = config_str.encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]
    
    return config_hash


def evaluate_bayesian_detector(
    manifest_path: Path,
    likelihood_params_path: str,
    log_odds_threshold: float = 0.0,
    fixed_fpr: Optional[float] = None,
    plot_roc: bool = False,
    output_dir: Optional[Path] = None,
    inversion_steps: int = 25,
    latent_type: str = "zT",
    normalization_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    family_id: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run controlled evaluation on Bayesian detector. Optionally uses normalized
    scores and calibrated (worst-case-safe) threshold. Reports worst-case TPR at
    fixed FPR and TPR variance across transforms when transform metadata present.

    Args:
        manifest_path: Path to g-manifest.jsonl file
        likelihood_params_path: Path to trained likelihood parameters
        log_odds_threshold: Detection threshold in log-odds space (default: 0.0)
        fixed_fpr: If set, override threshold by ROC point closest to this FPR
        plot_roc: Whether to generate and save ROC curve plot (diagnostics only)
        output_dir: Directory to save results (default: same as manifest parent)
        inversion_steps: Number of inversion steps (for logging/plotting)
        latent_type: Latent type (for logging/plotting)
        normalization_path: Optional JSON of normalization params per family
        calibration_path: Optional JSON of deployment threshold per family
        family_id: Family ID for normalization/calibration lookup when using both

    Returns:
        Tuple of (metrics_dict, evaluation_info_dict)
    """
    # Load manifest
    entries = load_manifest(manifest_path)
    
    # CRITICAL: Load inversion_steps from metadata.json (single source of truth)
    metadata_path = manifest_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                metadata_inversion_steps = metadata.get("num_inversion_steps")
                
                if metadata_inversion_steps is not None:
                    # Validate consistency if user provided --inversion-steps
                    if inversion_steps != 25:  # 25 is the default, so user explicitly set it
                        if inversion_steps != metadata_inversion_steps:
                            raise RuntimeError(
                                f"INVERSION STEPS MISMATCH: "
                                f"CLI argument --inversion-steps={inversion_steps} does not match "
                                f"precomputed G-values metadata (num_inversion_steps={metadata_inversion_steps}). "
                                f"The G-values in {manifest_path.parent} were precomputed with "
                                f"{metadata_inversion_steps} inversion steps. "
                                f"Either omit --inversion-steps to use the correct value from metadata, "
                                f"or provide the matching value: --inversion-steps {metadata_inversion_steps}"
                            )
                    
                    # Override with authoritative value from metadata
                    inversion_steps = metadata_inversion_steps
                    print(f"Inversion steps (from precomputed G-values metadata): {inversion_steps}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse metadata.json at {metadata_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata.json at {metadata_path}: {e}") from e
    else:
        # Metadata file missing - warn user but continue with provided value
        import warnings
        warnings.warn(
            f"metadata.json not found at {metadata_path}. "
            f"Using --inversion-steps={inversion_steps} from CLI, but this cannot be validated. "
            f"For reproducible experiments, ensure metadata.json exists alongside g-manifest.jsonl.",
            UserWarning,
        )
    
    if len(entries) == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    
    # Validate g-manifest
    first_entry = entries[0]
    assert "g_path" in first_entry, \
        "This script only supports precomputed g-manifests. " \
        "Every manifest entry must contain 'g_path'."
    
    print(f"Processing {len(entries)} samples from g-manifest: {manifest_path}")
    
    # Determine mapping_mode once per evaluation run
    mapping_mode = determine_mapping_mode(manifest_path, entries)
    print(f"Evaluation mapping_mode: {mapping_mode}")
    
    # Optional assertion: verify continuous g-values are not all in {0, 1}
    if mapping_mode == "continuous":
        # Check first entry to validate continuous g-values
        first_g_path_str = entries[0].get("g_path")
        if first_g_path_str:
            first_g_path = Path(first_g_path_str)
            if not first_g_path.is_absolute():
                first_g_path = manifest_path.parent / first_g_path
            if first_g_path.exists():
                try:
                    first_g_data = torch.load(first_g_path, map_location="cpu")
                    if isinstance(first_g_data, dict):
                        first_g = first_g_data.get("g", first_g_data.get("g_values"))
                    else:
                        first_g = first_g_data
                    if first_g is not None:
                        if first_g.dim() > 1:
                            first_g = first_g.flatten()
                        unique_vals = torch.unique(first_g.float())
                        unique_set = set(unique_vals.cpu().tolist())
                        # Check if all values are in {0, 1} (which would indicate binary data)
                        if unique_set.issubset({0.0, 1.0}):
                            print(
                                f"Warning: mapping_mode is 'continuous' but g-values appear to be binary "
                                f"(all values in {{0, 1}}). This may indicate a configuration mismatch."
                            )
                except Exception:
                    pass  # Skip validation if loading fails
    
    # Load Bayesian detector
    detector = BayesianDetector(
        likelihood_params_path=likelihood_params_path,
    )
    
    # Verify mapping_mode consistency between data and detector
    if detector.mapping_mode != mapping_mode:
        raise RuntimeError(
            f"MAPPING MODE MISMATCH: Data has mapping_mode='{mapping_mode}' but "
            f"detector was trained with mapping_mode='{detector.mapping_mode}'. "
            f"The likelihood model must match the g-value format. "
            f"Retrain the likelihood model with the correct mapping_mode."
        )
    
    print(f"Detector likelihood_type: {detector.likelihood_type}")
    
    # Collect scores and labels
    log_odds_scores = []
    labels = []
    detailed_results = []
    n_eff_values = []
    
    # Process each entry
    for i, entry in enumerate(tqdm(entries, desc="Evaluating detector")):
        try:
            # Extract information
            label = extract_label(entry)
            key_id = extract_key_id(entry)
            
            # Load precomputed g-values
            g_path_str = entry.get("g_path")
            if not g_path_str:
                raise ValueError(f"Entry {i} missing 'g_path' field")
            
            g_path = Path(g_path_str)
            if not g_path.is_absolute():
                g_path = manifest_path.parent / g_path
            
            if not g_path.exists():
                raise FileNotFoundError(f"G-values file not found: {g_path}")
            
            # Load g-values
            g_data = torch.load(g_path, map_location="cpu")
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
            
            # Load mask if present
            mask = None
            if "mask_path" in entry:
                mask_path_str = entry["mask_path"]
                mask_path = Path(mask_path_str)
                if not mask_path.is_absolute():
                    mask_path = manifest_path.parent / mask_path
                if mask_path.exists():
                    mask = torch.load(mask_path, map_location="cpu")
                    if mask.dim() > 1:
                        mask = mask.flatten()
                    mask = (mask > 0.5).float()
            
            # Compute N_eff
            n_eff = compute_n_eff(mask if mask is not None else g)
            n_eff_values.append(n_eff)
            
            # Run detection
            result = detect_watermark_bayesian_from_g_values(
                g=g,
                mask=mask,
                detector=detector,
                mapping_mode=mapping_mode,
            )
            
            # Extract log_odds
            log_odds = result["log_odds"]

            log_odds_scores.append(log_odds)
            labels.append(label)
            
            detailed_results.append({
                "index": i,
                "g_path": str(g_path),
                "key_id": key_id,
                "label": label,
                "log_odds": log_odds,
                "n_eff": n_eff,
                "transform": entry.get("transform", "identity"),
            })
            
        except Exception as e:
            raise RuntimeError(f"Failed to process entry {i}: {e}") from e
    
    # Compute ROC curve and AUC (diagnostics only)
    scores_array = np.array(log_odds_scores)
    labels_array = np.array(labels)

    # Optional: normalize scores using family-level normalization
    if normalization_path is not None and family_id is not None and normalization_path.exists():
        with open(normalization_path, "r") as f:
            norm_by_family = json.load(f)
        norm = norm_by_family.get(family_id, {})
        mean = norm.get("mean", 0.0)
        std = norm.get("std", 1.0)
        if std <= 0:
            std = 1.0
        scores_array = (scores_array - mean) / std

    fpr, tpr, thresholds = compute_roc_curve(scores_array, labels_array)
    auc = compute_auc(fpr, tpr)

    # Choose evaluation threshold: calibrated (worst-case-safe) or fixed-fpr or fixed value
    threshold_source = "fixed_log_odds_threshold"
    achieved_fpr_from_roc = None
    achieved_fpr_roc_index = None
    target_fpr = None

    if calibration_path is not None and family_id is not None and calibration_path.exists():
        with open(calibration_path, "r") as f:
            cal_by_family = json.load(f)
        cal = cal_by_family.get(family_id, {})
        log_odds_threshold = float(cal.get("deployment_threshold", log_odds_threshold))
        threshold_source = "calibrated_worst_case_safe"
        target_fpr = cal.get("target_fpr")
    elif fixed_fpr is not None:
        target_fpr = float(fixed_fpr)
        selected_threshold, achieved_fpr_from_roc, achieved_fpr_roc_index = find_threshold_at_fpr(
            fpr=fpr,
            thresholds=thresholds,
            target_fpr=target_fpr,
        )
        log_odds_threshold = float(selected_threshold)
        threshold_source = "roc_closest_fpr"

    # Apply threshold and compute metrics at this operating point
    predictions_array = (scores_array > float(log_odds_threshold)).astype(int)
    metrics = compute_metrics(predictions_array.tolist(), labels, log_odds_scores)
    metrics["auc"] = auc
    metrics["log_odds_threshold"] = float(log_odds_threshold)
    metrics["threshold_source"] = threshold_source

    # Per-transform and worst-case TPR (after metrics so we can use metrics in fallback)
    transforms = [row["transform"] for row in detailed_results]
    unique_transforms = list(dict.fromkeys(transforms))
    per_transform_tpr = {}
    for t in unique_transforms:
        mask = np.array([row["transform"] == t for row in detailed_results])
        lab_t = labels_array[mask]
        pred_t = predictions_array[mask]
        n_pos_t = np.sum(lab_t == 1)
        if n_pos_t > 0:
            per_transform_tpr[t] = float(np.sum((lab_t == 1) & (pred_t == 1)) / n_pos_t)
        else:
            per_transform_tpr[t] = 0.0
    tpr_values = list(per_transform_tpr.values())
    worst_case_tpr = float(np.min(tpr_values)) if tpr_values else float(metrics.get("recall", 0.0))
    tpr_variance = float(np.var(tpr_values)) if len(tpr_values) > 1 else 0.0
    metrics["worst_case_tpr"] = worst_case_tpr
    metrics["tpr_variance"] = tpr_variance
    metrics["per_transform_tpr"] = per_transform_tpr

    if target_fpr is not None:
        neg_mask = labels_array == 0
        achieved_fpr = float(np.mean(predictions_array[neg_mask] == 1)) if np.any(neg_mask) else 0.0
        assert 0.0 <= achieved_fpr <= 1.0

        metrics["target_fpr"] = float(target_fpr)
        metrics["achieved_fpr"] = achieved_fpr
        metrics["achieved_fpr_from_roc"] = float(achieved_fpr_from_roc) if achieved_fpr_from_roc is not None else None
        metrics["achieved_fpr_roc_index"] = int(achieved_fpr_roc_index) if achieved_fpr_roc_index is not None else None

        if achieved_fpr_from_roc is not None and abs(achieved_fpr - float(achieved_fpr_from_roc)) > 1e-6:
            print(
                f"Warning: achieved FPR differs between ROC point ({float(achieved_fpr_from_roc):.6f}) "
                f"and recomputed predictions ({achieved_fpr:.6f})."
            )
        if abs(achieved_fpr - float(target_fpr)) > 0.005:
            print(
                f"Warning: achieved FPR {achieved_fpr:.4f} deviates from target {target_fpr:.4f} by > 0.005."
            )

    # Fill in per-sample correctness after threshold selection
    for j, row in enumerate(detailed_results):
        pred = int(predictions_array[j])
        row["prediction"] = pred
        row["correct"] = bool(pred == int(row["label"]))

    # Per-transform TPR, worst-case TPR at fixed FPR, and TPR variance (when transform present)
    transforms = [row["transform"] for row in detailed_results]
    unique_transforms = list(dict.fromkeys(transforms))
    per_transform_tpr: Dict[str, float] = {}
    for t in unique_transforms:
        mask = np.array([row["transform"] == t for row in detailed_results])
        lab_t = labels_array[mask]
        pred_t = predictions_array[mask]
        n_pos_t = np.sum(lab_t == 1)
        if n_pos_t > 0:
            per_transform_tpr[t] = float(np.sum((lab_t == 1) & (pred_t == 1)) / n_pos_t)
        else:
            per_transform_tpr[t] = 0.0
    tpr_values = list(per_transform_tpr.values())
    worst_case_tpr = float(np.min(tpr_values)) if tpr_values else float(metrics.get("recall", 0.0))
    tpr_variance = float(np.var(tpr_values)) if len(tpr_values) > 1 else 0.0

    # Compute average N_eff
    avg_n_eff = float(np.mean(n_eff_values)) if n_eff_values else 0.0

    # Extract g_field_config_hash from manifest entries or metadata.json (if available)
    g_field_config_hash = None
    g_field_config = None
    
    # First, try to load from metadata.json (created by precompute_inverted_g_values.py)
    metadata_path = manifest_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                g_field_config_hash = metadata.get("g_field_config_hash")
                g_field_config = metadata.get("g_field_config")
        except Exception:
            pass  # Fall back to checking manifest entries
    
    # If not found, check manifest entries
    if g_field_config_hash is None:
        for entry in entries:
            if "g_field_config_hash" in entry:
                g_field_config_hash = entry["g_field_config_hash"]
            if "g_field_config" in entry:
                g_field_config = entry["g_field_config"]
                # Compute hash if not provided
                if g_field_config_hash is None and g_field_config is not None:
                    g_field_config_hash = compute_config_hash(g_field_config)
            # Use first entry's config info
            if g_field_config_hash is not None:
                break
    
    # Build evaluation info
    evaluation_info = {
        "inversion_steps": inversion_steps,  # From precomputed G-values metadata
        "latent_type": latent_type,
        "log_odds_threshold": log_odds_threshold,
        "threshold_source": threshold_source,
        "target_fpr": target_fpr,
        "achieved_fpr": metrics.get("achieved_fpr", None),
        "auc": auc,
        "avg_n_eff": avg_n_eff,
        "total_samples": len(entries),
        "g_field_config_hash": g_field_config_hash,
        "mapping_mode": mapping_mode,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
        "worst_case_tpr": metrics.get("worst_case_tpr"),
        "tpr_variance": metrics.get("tpr_variance"),
        "per_transform_tpr": metrics.get("per_transform_tpr"),
    }
    
    # Save ROC plot if requested
    if plot_roc:
        if output_dir is None:
            output_dir = manifest_path.parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        roc_plot_path = output_dir / f"roc_{inversion_steps}_steps.png"
        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            auc=auc,
            output_path=roc_plot_path,
            inversion_steps=inversion_steps,
            latent_type=latent_type,
            threshold=log_odds_threshold,
        )
        evaluation_info["roc_plot_path"] = str(roc_plot_path)
    
    return metrics, evaluation_info


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Controlled evaluation and signal-strength tuning for Bayesian detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--g-manifest",
        type=str,
        required=True,
        help="Path to g-values manifest.jsonl (created by precompute_inverted_g_values.py)",
    )
    
    parser.add_argument(
        "--likelihood-params",
        type=str,
        required=True,
        help="Path to trained likelihood parameters JSON file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: same directory as manifest)",
    )
    
    parser.add_argument(
        "--log-odds-threshold",
        type=float,
        default=0.0,
        help="Detection threshold in log-odds space (prediction = log_odds > threshold)",
    )

    parser.add_argument(
        "--fixed-fpr",
        type=float,
        default=None,
        help="If set, override --log-odds-threshold by selecting the ROC threshold closest to this target FPR",
    )
    
    parser.add_argument(
        "--plot-roc",
        action="store_true",
        help="Generate and save ROC curve plot",
    )
    
    parser.add_argument(
        "--inversion-steps",
        type=int,
        default=25,
        help="Number of inversion steps (for logging/plotting, default: 25)",
    )
    
    parser.add_argument(
        "--latent-type",
        type=str,
        default="zT",
        help="Latent type (for logging/plotting, default: zT)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        help="Optional path to normalization JSON per family (from compute_score_normalization)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Optional path to calibration JSON per family (from calibrate_threshold)",
    )
    parser.add_argument(
        "--family-id",
        type=str,
        default=None,
        help="Family ID for normalization/calibration lookup when using --normalization/--calibration",
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Bayesian Detector Controlled Evaluation")
    print("=" * 60)
    
    # Validate paths
    manifest_path = Path(args.g_manifest)
    if not manifest_path.exists():
        print(f"⚠️  G-manifest file not found: {manifest_path}")
        return 1
    
    likelihood_params_path = Path(args.likelihood_params)
    if not likelihood_params_path.exists():
        print(f"⚠️  Likelihood params file not found: {likelihood_params_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = manifest_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  G-manifest: {manifest_path}")
    print(f"  Likelihood params: {likelihood_params_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Log-odds threshold: {args.log_odds_threshold}")
    print(f"  Fixed FPR: {args.fixed_fpr}")
    print(f"  Plot ROC: {args.plot_roc}")
    print(f"  Inversion steps: {args.inversion_steps}")
    print(f"  Latent type: {args.latent_type}")
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    metrics, evaluation_info = evaluate_bayesian_detector(
        manifest_path=manifest_path,
        likelihood_params_path=str(likelihood_params_path),
        log_odds_threshold=args.log_odds_threshold,
        fixed_fpr=args.fixed_fpr,
        plot_roc=args.plot_roc,
        output_dir=output_dir,
        inversion_steps=args.inversion_steps,
        latent_type=args.latent_type,
        normalization_path=Path(args.normalization) if args.normalization else None,
        calibration_path=Path(args.calibration) if args.calibration else None,
        family_id=args.family_id,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {metrics['total']}")
    if args.fixed_fpr is not None:
        print("\nFixed-FPR Operating Point:")
        print(f"  Target FPR:   {metrics.get('target_fpr', float(args.fixed_fpr)):.4f}")
        if metrics.get("achieved_fpr", None) is not None:
            print(f"  Achieved FPR: {float(metrics['achieved_fpr']):.4f}")
        print(f"  Threshold:    {metrics['log_odds_threshold']:.6f} (source: {metrics.get('threshold_source')})")

    print(f"\nDetection Metrics (threshold = {metrics['log_odds_threshold']}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"\nROC/AUC Metrics (diagnostics):")
    print(f"  AUC: {metrics['auc']:.4f}")
    if "worst_case_tpr" in metrics:
        print(f"\nTransform-robust metrics (at fixed FPR):")
        print(f"  Worst-case TPR: {metrics['worst_case_tpr']:.4f}")
        print(f"  TPR variance across transforms: {metrics.get('tpr_variance', 0):.4f}")
        if metrics.get("per_transform_tpr"):
            for t, tpr in metrics["per_transform_tpr"].items():
                print(f"    {t}: TPR={tpr:.4f}")
    print(f"\nScore Statistics (log_odds):")
    print(f"  Watermarked mean:   {metrics['score_mean_watermarked']:.4f} ± {metrics['score_std_watermarked']:.4f}")
    print(f"  Unwatermarked mean: {metrics['score_mean_unwatermarked']:.4f} ± {metrics['score_std_unwatermarked']:.4f}")
    print(f"\nExperiment Hygiene:")
    print(f"  Average N_eff: {evaluation_info['avg_n_eff']:.1f}")
    print(f"  G-field config hash: {evaluation_info.get('g_field_config_hash', 'N/A')}")
    print(f"  Inversion steps (from precomputed G-values): {evaluation_info['inversion_steps']}")
    print(f"  Latent type: {evaluation_info['latent_type']}")
    print(f"  Log-odds threshold: {evaluation_info['log_odds_threshold']}")
    print(f"  AUC: {evaluation_info['auc']:.4f}")
    
    # Save results
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    evaluation_info_path = output_dir / "evaluation_info.json"
    with open(evaluation_info_path, "w") as f:
        json.dump(evaluation_info, f, indent=2)
    print(f"✓ Evaluation info saved to: {evaluation_info_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

