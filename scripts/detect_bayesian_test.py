#!/usr/bin/env python3
"""
Bayesian Detection with Precomputed G-Values (Step 7)

This script performs Bayesian watermark detection using precomputed g-values.
It does NOT perform DDIM inversion or load images - it only consumes g-values
that were precomputed by precompute_inverted_g_values.py.

This is the ONLY valid detection path for BayesianDetector when using precomputed g-values.

Outputs:
- metrics.json, detailed_results.json: full detection metrics and per-sample details.
- {family_id}.json: family_id + per_image (label, log_odds, transform) for steps 8 and 9
  (compute_score_normalization, calibrate_threshold).

Usage:
    python scripts/detect_bayesian_test.py \
        --g-manifest path/to/g_manifest.jsonl \
        --likelihood-params outputs/likelihood_models/likelihood_params.json \
        --output-dir outputs/detection_results \
        --family-id default \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.models.detectors import BayesianDetector


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
    """
    Extract key_id from manifest entry.
    
    Note: key_id is extracted for logging/tracking purposes only.
    The Bayesian detector does not require key_id at detection time because:
    - G-values are already key-conditioned (computed with the key during precomputation)
    - The detector operates on precomputed g-values, not raw images
    - Key-dependent processing happens upstream in precompute_inverted_g_values.py
    """
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


def detect_watermark_bayesian_from_g_values(
    g: torch.Tensor,
    mask: Optional[torch.Tensor],
    detector: BayesianDetector,
    mapping_mode: str = "binary",
) -> dict:
    """
    Detect watermark using Bayesian detector with precomputed g-values.
    
    This function uses precomputed g-values directly - no DDIM inversion,
    no image loading, no g-value computation.
    
    G-value handling depends on mapping_mode:
    - Binary: Normalize and binarize to {0, 1} for Bernoulli likelihood
    - Continuous: Preserve real-valued g-values for Gaussian likelihood
    
    Args:
        g: Precomputed g-values tensor [N] or [1, N]
        mask: Optional mask tensor [N] or [1, N] (typically None since g is already masked)
        detector: Pre-loaded BayesianDetector
        mapping_mode: "binary" or "continuous" - determines g-value handling
        
    Returns:
        Detection result dictionary
    """
    # Ensure batch dimension
    if g.dim() == 1:
        g = g.unsqueeze(0)  # [1, N]
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)  # [1, N]
    
    # Convert to float if needed
    if g.dtype in (torch.long, torch.int64):
        g = g.float()
    
    # Handle g-value processing based on mapping_mode
    if mapping_mode == "binary":
        # Binary mode: Normalize and binarize g-values
        # Handle {-1, 1} format: convert to {0, 1}
        unique_vals = torch.unique(g)
        if set(unique_vals.cpu().tolist()).issubset({-1.0, 1.0}):
            g = (g + 1) / 2
        
        # Ensure values are in [0, 1] range
        g = torch.clamp(torch.round(g), 0, 1)
        
        # Binarize: convert to binary {0, 1}
        g = (g > 0).float()
        
    elif mapping_mode == "continuous":
        # Continuous mode: Preserve real-valued g-values exactly
        # Only ensure float32 dtype, NO binarization
        if g.dtype != torch.float32:
            g = g.float()
        
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
    
    # Run detection (detector will use appropriate likelihood based on its trained type)
    result = detector.score(g, mask)
    
    # Extract log_odds
    log_odds = result["log_odds"].item() if result["log_odds"].numel() == 1 else result["log_odds"][0].item()
    is_watermarked = log_odds > 0
    
    return {
        "is_watermarked": bool(is_watermarked),
        "posterior": float(result["posterior"].item()),
        "log_odds": float(log_odds),
        "score": float(log_odds),
    }


def determine_mapping_mode(manifest_path: Path, entries: list) -> str:
    """
    Determine mapping_mode from metadata or manifest entries.
    
    Precedence:
    1. metadata.json["mapping_mode"]
    2. metadata.json["g_field_config"]["mapping_mode"]
    3. entry["mapping_mode"] (if present in any manifest entry)
    4. Default to "binary"
    
    Args:
        manifest_path: Path to manifest.jsonl file
        entries: List of manifest entries
        
    Returns:
        mapping_mode: "binary" or "continuous"
    """
    # First, check metadata.json
    metadata_path = manifest_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
                # Direct mapping_mode field
                mapping_mode = metadata.get("mapping_mode")
                if mapping_mode in {"binary", "continuous"}:
                    return mapping_mode
                
                # From g_field_config
                g_field_config = metadata.get("g_field_config", {})
                if isinstance(g_field_config, dict):
                    mapping_mode = g_field_config.get("mapping_mode")
                    if mapping_mode in {"binary", "continuous"}:
                        return mapping_mode
        except Exception:
            pass  # Fall back to checking entries
    
    # Second, check manifest entries
    for entry in entries:
        if "mapping_mode" in entry:
            mapping_mode = entry["mapping_mode"]
            if mapping_mode in {"binary", "continuous"}:
                return mapping_mode
    
    # Default to binary for backward compatibility
    return "binary"


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
        "score_mean_unwatermarked": float(np.mean(unwatermarked_scores)) if len(unwatermarked_scores) > 0 else 0.0,
        "score_std_watermarked": float(np.std(watermarked_scores)) if len(watermarked_scores) > 0 else 0.0,
        "score_std_unwatermarked": float(np.std(unwatermarked_scores)) if len(unwatermarked_scores) > 0 else 0.0,
    }


def batch_detect_watermark_bayesian(
    manifest_path: Path,
    likelihood_params_path: str,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run batch detection on entire test dataset from g-manifest.
    
    This function ONLY supports g-manifests with precomputed g-values.
    It does NOT support image-based detection.
    
    Args:
        manifest_path: Path to g-manifest.jsonl file (must contain g_path)
        likelihood_params_path: Path to trained likelihood parameters
        batch_size: Batch size for processing (used only for tqdm batching/logging)
    
    Returns:
        Tuple of (metrics_dict, detailed_results_list)
    """
    # Load manifest
    entries = load_manifest(manifest_path)
    
    if len(entries) == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    
    # SAFETY CHECK: Assert that this is a g-manifest
    first_entry = entries[0]
    assert "g_path" in first_entry, \
        "detect_bayesian_test.py only supports precomputed g-manifests. " \
        "Every manifest entry must contain 'g_path'. " \
        "Use precompute_inverted_g_values.py to create g-manifests from images."
    
    # SAFETY CHECK: Fail immediately if image_path or path is encountered
    if "image_path" in first_entry or "path" in first_entry:
        raise RuntimeError(
            "Image-based detection is no longer supported. "
            "This script only accepts g-manifests with precomputed g-values. "
            "Use precompute_inverted_g_values.py to precompute g-values first."
        )
    
    print(f"Processing {len(entries)} samples from g-manifest: {manifest_path}")
    
    # Determine mapping_mode from metadata
    mapping_mode = determine_mapping_mode(manifest_path, entries)
    print(f"Detected mapping_mode: {mapping_mode}")
    
    # Load Bayesian detector (once for all samples)
    # Note: threshold parameter is not used - decision is based on log_odds > 0
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
    
    predictions = []
    labels = []
    scores = []
    detailed_results = []
    
    # Process each entry
    for i, entry in enumerate(tqdm(entries, desc="Detecting watermarks")):
        try:
            # SAFETY CHECK: Fail if image_path or path is encountered
            if "image_path" in entry or "path" in entry:
                raise RuntimeError(
                    f"Entry {i} contains 'image_path' or 'path'. "
                    "Image-based detection is no longer supported. "
                    "This script only accepts g-manifests with precomputed g-values."
                )
            
            # Extract information
            label = extract_label(entry)
            # key_id is extracted for logging/tracking only (not used in detection)
            key_id = extract_key_id(entry)
            
            # Load precomputed g-values (raw, will be normalized/binarized in detection function)
            g_path_str = entry.get("g_path")
            if not g_path_str:
                raise ValueError(f"Entry {i} missing 'g_path' field (g-manifest)")
            
            g_path = Path(g_path_str)
            if not g_path.is_absolute():
                g_path = manifest_path.parent / g_path
            
            if not g_path.exists():
                raise FileNotFoundError(f"G-values file not found: {g_path}")
            
            # Load g-values (raw, no processing here)
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
            
            # Run detection on precomputed g-values
            result = detect_watermark_bayesian_from_g_values(
                g=g,
                mask=mask,
                detector=detector,
                mapping_mode=mapping_mode,
            )
            
            # Extract results - use log_odds as the detection score
            log_odds = result["log_odds"]
            prediction = 1 if log_odds > 0 else 0
            score = log_odds
            
            predictions.append(prediction)
            labels.append(label)
            scores.append(score)
            
            detailed_results.append({
                "index": i,
                "g_path": str(g_path),
                "key_id": key_id,
                "label": label,
                "prediction": prediction,
                "is_watermarked": result["is_watermarked"],
                "posterior": result["posterior"],
                "log_odds": result["log_odds"],
                "score": score,
                "correct": prediction == label,
                "transform": entry.get("transform", "identity"),
            })
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to process entry {i}: {e}"
            ) from e
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, scores)
    
    return metrics, detailed_results


def main():
    """Main detection function."""
    parser = argparse.ArgumentParser(
        description="Bayesian watermark detection with precomputed g-values",
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
        help="Directory to save detection results (default: same directory as manifest)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (used only for tqdm batching/logging)",
    )
    parser.add_argument(
        "--family-id",
        type=str,
        default="default",
        help="Family identifier for result JSON consumed by compute_score_normalization and calibrate_threshold (step 8/9)",
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Bayesian Watermark Detection (Precomputed G-Values)")
    print("=" * 60)
    
    # Validate paths
    manifest_path = Path(args.g_manifest)
    if not manifest_path.exists():
        print(f"⚠️  G-manifest file not found: {manifest_path}")
        print(f"   Hint: Use precompute_inverted_g_values.py to create g-manifests")
        return 1
    
    likelihood_params_path = Path(args.likelihood_params)
    if not likelihood_params_path.exists():
        print(f"⚠️  Likelihood params file not found: {likelihood_params_path}")
        print(f"   Hint: Train the model first using train_g_likelihoods.py")
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
    print(f"  Family ID: {args.family_id}")
    print(f"  Batch size: {args.batch_size}")
    
    # Run batch detection
    print(f"\nRunning batch detection...")
    metrics, detailed_results = batch_detect_watermark_bayesian(
        manifest_path=manifest_path,
        likelihood_params_path=str(likelihood_params_path),
        batch_size=args.batch_size,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    print(f"Total samples: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"\nScore Statistics (log_odds):")
    print(f"  Watermarked mean:   {metrics['score_mean_watermarked']:.4f} ± {metrics['score_std_watermarked']:.4f}")
    print(f"  Unwatermarked mean: {metrics['score_mean_unwatermarked']:.4f} ± {metrics['score_std_unwatermarked']:.4f}")
    
    # Save results
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    detailed_results_path = output_dir / "detailed_results.json"
    with open(detailed_results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "family_id": args.family_id,
            "detailed_results": detailed_results,
        }, f, indent=2)
    print(f"✓ Detailed results saved to: {detailed_results_path}")

    # Write family result JSON for steps 8 and 9 (compute_score_normalization, calibrate_threshold)
    per_image = [
        {
            "label": r["label"],
            "log_odds": r["log_odds"],
            "transform": r.get("transform", "identity"),
        }
        for r in detailed_results
    ]
    family_result_path = output_dir / f"{args.family_id}.json"
    with open(family_result_path, "w") as f:
        json.dump({
            "family_id": args.family_id,
            "per_image": per_image,
            "num_samples": len(per_image),
        }, f, indent=2)
    print(f"✓ Family result saved to: {family_result_path} (for step 8/9)")

    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
