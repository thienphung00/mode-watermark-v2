# Watermark Detection Pipeline Audit Report

**Date:** 2025-02-11  
**Scope:** Likelihood training, score normalization, calibration, detection, and evaluation scripts.  
**Objective:** Correctness, leakage, calibration validity, score-space consistency, threshold integrity.

---

## Script Mapping (Your Names → Repo)

| You specified     | Actual script/path |
|-------------------|--------------------|
| train_likelihood.py | `scripts/train_g_likelihoods.py` |
| detect.py         | `scripts/detect_bayesian_test.py` (+ `src/models/detectors.py`) |
| score_norm.py     | `scripts/compute_score_normalization.py` |
| calibrate.py      | `scripts/calibrate_threshold.py` |
| evaluate_detection.py | `scripts/evaluate_bayesian_detector.py` |

---

## 1. Leakage Risk Report

### 1.1 Summary

| Risk category              | Level  | Notes |
|----------------------------|--------|--------|
| Train/test contamination  | **Medium** | Scripts do not enforce splits; procedural only. |
| Cross-key leakage         | **Low**  | Single-key fingerprint in likelihood params; no key mixing in code. |
| Transform leakage         | **Low**  | Likelihoods pooled across transforms; no transform-specific params from test. |
| Threshold leakage         | **High**  | If calibration/normalization use same set as evaluation → leakage. |
| Normalization leakage     | **High**  | Normalization stats from `results_dir`; if that is test set → leakage. |

### 1.2 Likelihood training (`train_g_likelihoods.py`)

- **Watermarked vs unwatermarked:** Strictly separated. `model_w` is trained only on `labels == 1`, `model_u` only on `labels == 0` (e.g. lines 424–438, 449–463; MLE 453–461).
- **Test set in likelihood:** Script does **not** load a test set. It only uses `--g-manifest` (and optional `--val-g-manifest`). So no test data in likelihood parameter estimation **provided** the user passes a train-only manifest.
- **Transform leakage:** All samples are pooled across transforms (docstring and single model per class). No per-transform likelihoods; no transform-specific use of test data.
- **Gaussian std floor:** Applied consistently: `GAUSSIAN_STD_FLOOR` in training (e.g. `GaussianLikelihoodModel`, MLE `std_w`/`std_u` clamp), and `detectors.py` uses the same floor when loading and in `_compute_gaussian_log_likelihood` (clamp to `std_floor`).

**Verdict:** No in-script train/test or transform leakage. Risk is **procedural**: using a manifest that contains test data would leak.

### 1.3 Score normalization (`compute_score_normalization.py`)

- **Data source:** Stats are computed from whatever is in `--results-dir`: it loads all `*.json` with `family_id` and `per_image` and uses those entries.
- **Statistic used:** Mean/std (actually median and 1.4826×MAD) of **clean (unwatermarked) log_odds only** (lines 61–66). So no direct use of watermarked labels for the stats.
- **Leakage:** If `results_dir` contains detection results from the **test** set, then normalization is estimated from test clean scores → **test-set leakage**. The script does not know or enforce “train” vs “val” vs “test”; it is the caller’s responsibility to point to a non-test set (e.g. validation only).

**Verdict:** **High** leakage risk if normalization is run on the same set used for final evaluation. Must use a held-out set (e.g. validation) for `--results-dir`.

### 1.4 Calibration (`calibrate_threshold.py`)

- **Data source:** Same as normalization: `load_per_image_scores_by_family(args.results_dir)`. So calibration thresholds are computed from whatever run produced the JSONs in that directory.
- **Validation vs test:** Script does not distinguish validation from test. If the same 2000 samples are used for calibration and then for evaluation in `evaluate_bayesian_detector.py`, threshold and FPR/TPR are optimized on the evaluation set → **calibration leakage**.

**Verdict:** **High** leakage risk if calibration is run on the same set used for evaluation. Calibration (and normalization) must be run on a **validation** (or train) set only; evaluation must be on a **separate test** set.

### 1.5 Detection (`detect_bayesian_test.py`)

- No access to calibration or normalization; it only outputs raw `log_odds` and uses a **hard-coded threshold 0** for its own metrics (line 399: `prediction = 1 if log_odds > 0`). So no threshold leakage from this script. Calibrated threshold is only applied in `evaluate_bayesian_detector.py`.

### 1.6 Evaluation (`evaluate_bayesian_detector.py`)

- **Test reuse:** Evaluation runs on whatever `--g-manifest` is passed. If that manifest is the same set used to compute normalization and calibration, then:
  - Normalization was fit on (a subset of) that set.
  - Calibration was fit on that set.
  - Metrics (accuracy, FPR, TPR, AUC) are then reported on the same set → **optimistic bias**.

**Verdict:** No in-script reuse of a dedicated “test” variable; leakage is entirely **procedural**. Required discipline: **train** → likelihood; **validation** → normalization + calibration; **test** → evaluation only.

---

## 2. Score-Space Consistency Report

### 2.1 Pipeline order

- **Intended order:** likelihood → log-odds → (optional) normalization → threshold.
- **Implementation:**
  - **Likelihood → log-odds:** In `src/models/detectors.py`, `score()` computes `log_odds = log_likelihood_w + log_prior_w - log_likelihood_u - log_prior_u` (lines 406–411). So score space is **log-odds**.
  - **Normalization:** Applied to **log-odds** in both:
    - `compute_score_normalization.py`: `clean_scores = [float(e["log_odds"]) ...]` (lines 61–63).
    - `evaluate_bayesian_detector.py`: `scores_array = (scores_array - mean) / std` (line 447), where `scores_array` is from `log_odds_scores`.
  - **Threshold:** Applied in the **same** space as the score used for decision:
    - With calibration: threshold comes from `calibrate_threshold.py`, which (when `--normalization` is used) normalizes scores first, so the stored threshold is in **normalized log-odds**; evaluation normalizes then compares → consistent.
    - Without normalization: threshold is in raw log-odds; evaluation compares raw `scores_array` (or normalized only if `normalization_path` is given) → must match.

So: **likelihood → log-odds → optional normalization → threshold** is respected.

### 2.2 Raw vs normalized vs log-odds

- **Detector output:** Always **log-odds** (and optionally posterior; decision in evaluation uses log-odds vs threshold).
- **Normalization:** Applied to log-odds only; no normalization of raw likelihoods.
- **Double normalization:** In `evaluate_bayesian_detector.py`, normalization is applied once to `scores_array` (lines 441–448). No second application found.

### 2.3 Log-odds formula

- **Definition:** log P(watermarked | g) − log P(unwatermarked | g) = log P(g|wm) − log P(g|clean) + log(prior_wm) − log(prior_clean). With equal priors this is log P(g|wm) − log P(g|clean).
- **Code:** `log_odds = log_likelihood_w + log_prior_w - log_likelihood_u - log_prior_u` in `detectors.py` (lines 406–411) → correct.
- **Numerical stability:** Posterior uses log-sum-exp (lines 417–419); log-odds is a difference of log-probs; Bernoulli uses `+ 1e-10` in log (detectors and training). No obvious instability.

### 2.4 Mean/std used and when normalization is applied

- **Normalization stats:** In `results/normalization_098.json`: `mean = -3838.23`, `std = 573.72` (robust median/MAD on clean only).
- **When applied:** Normalization is **optional** in evaluation: only if `normalization_path` and `family_id` are provided (lines 441–448). So it is not “always enforced”; it is conditional on CLI/API arguments.
- **Printing:** The normalization script logs mean/std per family (e.g. lines 131–136 in `compute_score_normalization.py`). The evaluation script does not re-print the loaded mean/std; it only uses them. Recommendation: log or print loaded norm params in evaluation when normalization is used.

---

## 3. Threshold Integrity Report

### 3.1 Space and source

- **Space:** Threshold is in **log-odds** space (raw or normalized, depending on whether normalization was used when calibrating and when evaluating). No use of posterior space for threshold selection in calibration or evaluation.
- **Source:** In evaluation with `method="worst_case_safe"`, threshold is loaded from calibration JSON: `log_odds_threshold = float(cal.get("deployment_threshold", log_odds_threshold))` (e.g. `evaluate_bayesian_detector.py` lines 458–461). So **threshold is loaded from calibration**, not hard-coded for that method.
- **Hard-coded threshold:**  
  - In **detection** script `detect_bayesian_test.py`, line 399: `prediction = 1 if log_odds > 0` → **hard-coded 0**. That script does not load calibration; it is intended for producing `per_image` for steps 8/9. For **evaluation** with calibrated threshold, the evaluation script correctly uses the calibration file.
  - In **detector** `BayesianDetector.forward`, `decision = (result["posterior"] > threshold)` with default `threshold=0.5` (posterior space). The evaluation and calibration path consistently use **log-odds** and a log-odds threshold, not posterior.

### 3.2 Calibration logic

- **calibrate_from_labeled_data** (`src/detection/calibration.py` 115–156): Threshold = (1 − target_fpr) quantile of **unwatermarked** scores. So threshold is in the same space as the input scores (raw or normalized log-odds). FPR/TPR are computed from `score > threshold` → correct for “predict 1 when score > threshold”.
- **Worst-case safe:** In `calibrate_threshold.py`, per-transform thresholds are computed, then `deployment_threshold = max(per_transform_thresholds)` so FPR is at or below target for every transform → correct.
- **TPR variance:** In evaluation, `per_transform_tpr` and `tpr_variance` are computed from predictions at the chosen threshold (lines 461–472, 449–471). So TPR variance is consistent with the threshold used.

### 3.3 Confusion with posterior vs log-odds

- Decision in evaluation is **log-odds vs threshold**: `predictions_array = (scores_array > float(log_odds_threshold)).astype(int)` (line 453). So no confusion with posterior > 0.5 in the evaluation path.
- The detector’s internal `decision` uses posterior > 0.5, but the scripts that report metrics and use calibration use log-odds and the calibrated threshold.

### 3.4 Metadata

- Evaluation info includes `log_odds_threshold`, `threshold_source`, `threshold_method` (e.g. `evaluation_info` and `metrics`). So **threshold_used** and **threshold_source** are reported. **score_space** is not explicitly set in the saved JSON; it is implicit (log-odds, and normalized if normalization was used). Recommendation: add an explicit `score_space` (e.g. `"log_odds"` or `"normalized_log_odds"`) to evaluation output.

---

## 4. Calibration Validity and Statistical Assumptions

### 4.1 Data used for calibration

- Calibration uses whatever is in `--results-dir` (validation intended). It does **not** use a dedicated “test” set inside the script; again, separation is procedural.
- Threshold is computed in **log-odds** (or normalized log-odds when normalization is supplied) → correct.

### 4.2 Worst-case-safe and TPR variance

- **Worst-case threshold:** max over per-transform thresholds → correct for capping FPR per transform.
- **TPR variance:** Computed from per-transform TPRs at the deployment threshold; no separate “worst_case_safe” TPR variance estimation in calibration; variance is reported in evaluation. No obvious optimistic bias in the formula.

### 4.3 ROC-based path

- When using `expected_tpr` or `roc_closest_fpr`, the ROC is built from the **evaluation** run’s scores and labels (the same manifest). So if that manifest is the test set, ROC is on test (no calibration leakage from a separate calibration set). If the manifest is val, then threshold is chosen on val. So ROC is always “on the set we’re evaluating on” in the current code; for a proper report, ROC/AUC on test should use only test data and a threshold chosen on val.

### 4.4 Hard-coded thresholds and overwriting

- No hard-coded threshold in calibration or in evaluation when using `worst_case_safe` with calibration file.
- In evaluation, when calibration is used, `log_odds_threshold` is overwritten from the calibration file (lines 458–461). That is intended; no later overwrite found.

---

## 5. Evaluation Script Checks

### 5.1 Confusion matrix and labels

- **TP/TN/FP/FN:**  
  `tp = (predictions == 1) & (labels == 1)`, `tn = (predictions == 0) & (labels == 0)`, etc. (`evaluate_bayesian_detector.py` and `compute_metrics`). So 1 = watermarked, 0 = unwatermarked; prediction 1 when score > threshold → correct.
- **Your counts:** TP=999, TN=971, FP=29, FN=1 → 999+971+29+1 = 2000 = total. Labels are not flipped.

### 5.2 Threshold used

- Predictions use `scores_array > log_odds_threshold`, with `log_odds_threshold` from calibration when `method="worst_case_safe"` and calibration is provided. So the **same** calibrated threshold is used for the reported metrics.

### 5.3 AUC

- AUC is computed from the same `scores_array` and `labels_array` used for the ROC (lines 449–451). When normalization is used, `scores_array` is normalized; so AUC is on (possibly normalized) **log-odds**, not on thresholded 0/1 values. So **AUC is from raw (or normalized) log-odds**, as required.

### 5.4 Test reuse

- No explicit “test” split in code. If the same 2000 samples were used for normalization, calibration, **and** evaluation, then test data were reused. This must be avoided by using separate val (for norm + calibration) and test (for evaluation only).

---

## 6. Extreme Score Separation and AUC

### 6.1 Score means (664 vs -3930)

- With **continuous** Gaussian likelihood over **904** positions, log-odds is a sum of 904 terms; each term can be on the order of several units when class means/stds differ, so sums in the hundreds or thousands are plausible.
- Such separation can still be **legitimate** if:
  - Train and test share the same key and same g-field/inversion setup.
  - No test data were used for likelihood, normalization, or calibration.
- It becomes **suspicious** if:
  - Normalization or calibration (or likelihood) used the same 2000 samples as evaluation → would inflate separation and metrics.
  - Recommendation: Confirm that normalization and calibration were run on a **validation** set only and evaluation on a **held-out test** set.

### 6.2 AUC ≈ 0.9995

- AUC is computed on the evaluation run’s scores and labels. If that run is on an independent test set (no overlap with train/val for likelihood, normalization, or calibration), the AUC is a valid test AUC. If the evaluation set was used for normalization or calibration, AUC is optimistically biased.

### 6.3 Gaussian variance floor

- Std floor (e.g. 0.01) avoids near-zero variances and stabilizes log-likelihoods. It can **slightly** reduce separation when true variance is very small, but it does not artificially **increase** separation. So no indication that the floor inflates performance.

### 6.4 Likelihood pooling across transforms

- Single likelihood model per class, pooled over transforms. So one set of parameters for “watermarked” and one for “unwatermarked” across all transforms. No per-transform parameters are estimated from test; pooling is consistent.

---

## 7. Suspected Bugs and Inconsistencies (with line references)

### 7.1 High / medium severity

1. **Normalization/calibration on same set as evaluation (procedural)**  
   - **Where:** `compute_score_normalization.py` and `calibrate_threshold.py` take `--results-dir`; `evaluate_bayesian_detector.py` takes `--g-manifest`.  
   - **Issue:** If the same 2000 samples are used to (a) run detection and write results to `results_dir`, (b) compute normalization and calibration from that `results_dir`, and (c) run evaluation on the same 2000 via `--g-manifest`, then normalization and threshold are fit on the evaluation set.  
   - **Fix:** Document and enforce: normalization and calibration from **validation** (or train) only; evaluation on **test** only. Optionally add a `--split` or `phase` to manifests and have scripts reject or warn when phase is inconsistent (e.g. warn if calibration results_dir looks like “test”).

2. **detect_bayesian_test.py uses hard-coded threshold 0**  
   - **Where:** `scripts/detect_bayesian_test.py` line 399: `prediction = 1 if log_odds > 0`.  
   - **Issue:** Standalone detection always uses 0; calibrated threshold is only used in the evaluation script. So metrics in `detect_bayesian_test.py` output are at threshold 0, not at deployment threshold.  
   - **Fix:** Either add optional `--calibration` (and `--normalization`) to the detection script and use the same threshold as evaluation, or clearly document that detection script metrics are at threshold 0 and that deployment decisions should use the evaluation pipeline with calibration.

3. **Calibration docstring mentions “detailed_results” but loader does not use it**  
   - **Where:** `calibrate_threshold.py` help text (line 102) says “per_image or detailed_results from detect_bayesian_test”. `load_per_image_scores_by_family` (in `compute_score_normalization.py`, used by calibrate) only loads `data.get("per_image")`; `detailed_results.json` has `detailed_results`, not top-level `per_image`.  
   - **Issue:** If a directory contains only `detailed_results.json`, calibration (and normalization) get no data and fail with “No result JSONs... found”.  
   - **Fix:** Either extend the loader to accept files that have `family_id` and `detailed_results` and map each entry to `{label, log_odds, transform}` for `per_image`, or change the help text to state that only JSONs with top-level `per_image` and `family_id` are supported.

### 7.2 Low severity / recommendations

4. **Evaluation does not print which normalization mean/std were used**  
   - **Where:** `evaluate_bayesian_detector.py` lines 441–448 load norm and apply it but do not log mean/std.  
   - **Fix:** When `normalization_path` and `family_id` are set, log or print the loaded `mean` and `std` (and optionally `normalization_method`) for reproducibility.

5. **No explicit `score_space` in evaluation output**  
   - **Where:** `evaluation_info` and metrics do not include a field like `score_space`.  
   - **Fix:** Add e.g. `"score_space": "log_odds"` or `"normalized_log_odds"` when normalization is used, so downstream and audits know exactly which space the threshold and scores are in.

6. **Transform name mismatch in results**  
   - **Where:** `calibration_098.json` has `per_transform_threshold` with key `gaussian_blur_1`; `evaluation_info`/metrics have `per_transform_tpr` with `gaussian_blur_2`.  
   - **Issue:** Suggests calibration was run on a different run (e.g. val with different transforms) than the evaluation run. Not a code bug, but worth confirming that val and test use the same transform set or that worst-case threshold is still valid for test transforms.

---

## 8. Recommended Fixes (ranked by severity)

| Priority | Fix |
|----------|-----|
| **P0** | Enforce or strictly document data split: **train** → likelihood only; **validation** → normalization + calibration (and optionally detection for norm/cal); **test** → evaluation only. Add a short “Experiment hygiene” section in README or runbook. |
| **P1** | In `evaluate_bayesian_detector.py`, when normalization is used: log/print the loaded normalization mean and std (and method). Add `score_space` to saved evaluation info (e.g. `"log_odds"` or `"normalized_log_odds"`). |
| **P2** | In `detect_bayesian_test.py`: either support optional `--calibration` and `--normalization` and use the same threshold as evaluation, or document that metrics are at threshold 0 and that production should use the evaluation path with calibration. |
| **P3** | In `load_per_image_scores_by_family`: support JSONs that have `detailed_results` and `family_id` by building `per_image` from `detailed_results` (label, log_odds, transform), or update calibrate/normalization help text to say only `per_image` + `family_id` are supported. |
| **P4** | When writing calibration output, optionally store which `results_dir` or run was used (e.g. path or manifest hash) so that audits can verify calibration was not run on the test set. |

---

## 9. Information You Should Confirm (as requested)

To fully close leakage and interpret the 0.985 accuracy and AUC ≈ 0.9995:

1. **Dataset split:** How many keys are there, and are keys (or image IDs) strictly separated across train / val / test (no key in both train and test)?  
2. **Transforms:** Are the same transform names and definitions used in train, val, and test (e.g. same `gaussian_blur_1` vs `gaussian_blur_2`)?  
3. **Cross-key evaluation:** Was evaluation done on keys that were never used for training (or for normalization/calibration)?  
4. **Normalization:** Are normalization stats stored per key or one global per family? (Current code: one global per family from whatever `results_dir` contained.)  
5. **Calibration:** Was calibration done per key or one global per family? (Current code: one global per family.)  
6. **Posterior:** Is posterior still used for any **decision** in production? (In the reviewed code, evaluation and calibration use log-odds vs threshold; detector’s internal `decision` uses posterior > 0.5, which is equivalent to log_odds > 0 for equal priors.)

---

## 10. Summary

- **Likelihood training:** Correct separation of watermarked/unwatermarked; no test set in script; Gaussian floor applied consistently; log-odds formula correct.  
- **Score space:** Log-odds throughout; normalization applied to log-odds once; order likelihood → log-odds → normalization → threshold is correct.  
- **Thresholds:** Calibration threshold is in log-odds (or normalized log-odds); evaluation uses it correctly; detection script uses 0 unless calibration is added.  
- **Main risk:** **Data leakage** if the same set is used for normalization, calibration, and evaluation. The scripts do not enforce splits; you must use a dedicated validation set for norm/cal and a held-out test set for evaluation to trust the reported metrics and AUC.

Implementing the P0 and P1 recommendations and confirming the split and key/transform usage will make the pipeline auditable and the reported numbers trustworthy.
