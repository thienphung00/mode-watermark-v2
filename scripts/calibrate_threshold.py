#!/usr/bin/env python3
"""
Phase 9: Calibrate a single global (worst-case-safe) threshold per detector family.

Targets a user-specified FPR. Computes per-transform thresholds on clean scores,
then selects the worst-case-safe threshold (max over transforms) so that FPR
is at or below target for every transform. This threshold is the deployment threshold.

Input: Detection results (per_image with label, log_odds, transform) and optional
       normalization params (if scores are normalized, use normalized for calibration).
Output: One deployment threshold per family and per-transform thresholds for audit.
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

from src.detection.calibration import calibrate_from_labeled_data
from scripts.compute_score_normalization import (
    load_per_image_scores_by_family,
    normalize_score,
)
from scripts.utils import setup_logging


def calibrate_family(
    per_image: List[Dict[str, Any]],
    target_fpr: float,
    normalization: Dict[str, float],
) -> Dict[str, Any]:
    """
    For one family: group by transform; for each transform compute threshold at target_fpr
    on clean scores; deployment threshold = max over transforms (worst-case-safe).
    Optionally normalize scores first using normalization (mean, std).
    """
    by_transform: Dict[str, List[Dict[str, Any]]] = {}
    for e in per_image:
        t = e.get("transform", "identity")
        if t not in by_transform:
            by_transform[t] = []
        by_transform[t].append(e)

    mean = normalization.get("mean", 0.0) if normalization else 0.0
    std = normalization.get("std", 1.0) if normalization else 1.0
    use_norm = bool(normalization)

    per_transform_thresholds: Dict[str, float] = {}
    per_transform_achieved_fpr: Dict[str, float] = {}
    for t, entries in by_transform.items():
        clean_scores = [
            float(e["log_odds"]) for e in entries if e.get("label", 1) == 0
        ]
        wm_scores = [
            float(e["log_odds"]) for e in entries if e.get("label", 0) == 1
        ]
        if use_norm:
            clean_scores = [normalize_score(s, mean, std) for s in clean_scores]
            wm_scores = [normalize_score(s, mean, std) for s in wm_scores]
        if not clean_scores:
            per_transform_thresholds[t] = np.inf
            per_transform_achieved_fpr[t] = 0.0
            continue
        result = calibrate_from_labeled_data(
            np.array(wm_scores),
            np.array(clean_scores),
            target_fpr=target_fpr,
        )
        per_transform_thresholds[t] = result.threshold
        per_transform_achieved_fpr[t] = result.achieved_fpr

    if not per_transform_thresholds:
        deployment_threshold = 0.0
    else:
        deployment_threshold = float(max(per_transform_thresholds.values()))

    return {
        "target_fpr": target_fpr,
        "deployment_threshold": deployment_threshold,
        "per_transform_threshold": per_transform_thresholds,
        "per_transform_achieved_fpr": per_transform_achieved_fpr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate worst-case-safe threshold per family (Phase 9)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory of detection result JSONs (from detect_bayesian_test.py or run_ablation_detection; per_image or detailed_results)",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help="Optional JSON of normalization params per family (from compute_score_normalization)",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target false positive rate (default 0.01)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON: deployment threshold and per-transform thresholds per family",
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
            f"No result JSONs with per_image (or detailed_results from detect_bayesian_test) and family_id found in {args.results_dir}"
        )

    norm_by_family: Dict[str, Dict[str, float]] = {}
    if args.normalization and args.normalization.exists():
        with open(args.normalization, "r") as f:
            norm_by_family = json.load(f)

    calibration_by_family: Dict[str, Any] = {}
    for family_id, per_image in by_family.items():
        norm = norm_by_family.get(family_id) or {}
        calibration_by_family[family_id] = calibrate_family(per_image, args.target_fpr, norm)
        d = calibration_by_family[family_id]
        logger.info(
            "family %s: deployment_threshold=%.4f (target_fpr=%.4f)",
            family_id,
            d["deployment_threshold"],
            d["target_fpr"],
        )
        for t, thr in d["per_transform_threshold"].items():
            logger.info("  transform %s: threshold=%.4f achieved_fpr=%.4f", t, thr, d["per_transform_achieved_fpr"].get(t, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(calibration_by_family, f, indent=2)
    logger.info("Wrote calibration to %s", args.output)


if __name__ == "__main__":
    main()
