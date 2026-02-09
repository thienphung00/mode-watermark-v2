#!/usr/bin/env python3
"""
Phase 8: Compute score normalization statistics per detector family.

Uses clean (unwatermarked) images across all transforms to compute family-level
mean and std of detection scores (log_odds). Normalization is reusable across
configs in the same family. Uses existing score definitions (log_odds).

Input: Detection results (JSON per config with per_image list: label, log_odds, transform)
       from run_ablation_detection.py.
Output: Normalization params per family (mean, std, n_clean) for use by evaluation
        and threshold calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils import setup_logging


def load_per_image_scores_by_family(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all result JSONs from results_dir; group per_image entries by family_id.
    Each per_image entry must have label, log_odds, and optionally transform.
    """
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for p in results_dir.glob("*.json"):
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        family_id = data.get("family_id")
        per_image = data.get("per_image")
        if not family_id or not per_image:
            continue
        if family_id not in by_family:
            by_family[family_id] = []
        by_family[family_id].extend(per_image)
    return by_family


def compute_normalization_for_family(
    per_image: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute mean and std of log_odds over clean (label==0) samples.
    Uses all transforms (no per-transform stats; family-level only).
    """
    clean_scores = [
        float(e["log_odds"])
        for e in per_image
        if e.get("label", 1) == 0
    ]
    if not clean_scores:
        return {"mean": 0.0, "std": 1.0, "n_clean": 0}
    arr = np.array(clean_scores)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0:
        std = 1.0
    return {"mean": mean, "std": std, "n_clean": len(clean_scores)}


def normalize_score(score: float, mean: float, std: float) -> float:
    """Normalize score using family-level mean and std."""
    return (score - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute score normalization per detector family (Phase 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory of detection result JSONs (from run_ablation_detection, with per_image)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path: normalization params per family (mean, std, n_clean)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    logger = setup_logging(level=args.log_level)

    by_family = load_per_image_scores_by_family(args.results_dir)
    if not by_family:
        raise FileNotFoundError(
            f"No result JSONs with per_image and family_id found in {args.results_dir}"
        )

    out: Dict[str, Dict[str, float]] = {}
    for family_id, per_image in by_family.items():
        out[family_id] = compute_normalization_for_family(per_image)
        logger.info(
            "family %s: mean=%.4f std=%.4f n_clean=%d",
            family_id,
            out[family_id]["mean"],
            out[family_id]["std"],
            out[family_id]["n_clean"],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote normalization params to %s", args.output)


if __name__ == "__main__":
    main()
