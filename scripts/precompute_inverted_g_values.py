#!/usr/bin/env python3
"""
Precompute inverted g-values from images for Tier-2 Bayesian watermark training.

This script:
1. Loads images from a manifest
2. Performs DDIM inversion on each image to get latent_T (zT)
3. Computes g-values from inverted latents using the same logic as detection
4. Saves g-values and masks to disk
5. Creates a new manifest with paths to saved data
6. Saves metadata for train-detect symmetry checks

CRITICAL FEATURES:
- Config validation: Fails fast if watermark config is missing or malformed
- Train/val/test symmetry: Validates existing metadata matches current config
- G-field provenance: Logs and persists config hash for alignment verification
- Detection parity: Ensures g-value computation matches detection exactly

Usage:
    python scripts/precompute_inverted_g_values.py \
        --manifest path/to/manifest.jsonl \
        --output-dir path/to/output \
        --config-path configs/experiments/seedbias.yaml \
        --master-key "your_secret_key" \
        --num-inversion-steps 20

The script will fail loudly if:
- Config is missing required watermark parameters
- Existing metadata indicates different config was used (train/val/test mismatch)
- Manifest entries violate strict contract (missing label, invalid key_id)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig
from src.core.key_utils import UNWATERMARKED_DUMMY_KEY
from src.engine.pipeline import create_pipeline
from src.detection.inversion import DDIMInverter
from src.detection.g_values import compute_g_values, g_field_config_to_dict
from scripts.utils import setup_logging, get_device


def load_manifest_entries(manifest_path: str) -> list[Dict[str, Any]]:
    """
    Load and flatten manifest file (JSON or JSONL).
    
    Supports multiple manifest schemas:
    - JSONL: Each line is a sample entry (legacy)
    - JSON list: Flat list of sample entries (legacy)
    - JSON dict: Grouped format with 'watermarked' and/or 'clean' keys (new)
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        List of manifest entries (flattened). Each entry must have 'image_path' field.
        The 'prompt' field is automatically stripped to prevent leakage.
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest schema is invalid or unsupported
        RuntimeError: If entries are missing required fields
    """
    manifest_path_obj = Path(manifest_path)
    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path_obj}")
    
    entries = []
    schema_type = None
    
    # Detect and load based on file extension
    if manifest_path_obj.suffix == ".jsonl":
        # JSONL format: stream line-by-line
        schema_type = "JSONL (line-by-line)"
        with open(manifest_path_obj, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON on line {line_num} of {manifest_path_obj}: {e}"
                        )
    
    elif manifest_path_obj.suffix == ".json":
        # JSON format: load entire file
        with open(manifest_path_obj, "r") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Legacy format: flat list of entries
            schema_type = "JSON (flat list)"
            entries = data
        
        elif isinstance(data, dict):
            # New grouped format: dict with 'watermarked' and/or 'clean' keys
            detected_keys = list(data.keys())
            
            # Check if this looks like the grouped format
            if "watermarked" in data or "clean" in data:
                schema_type = f"JSON (grouped: {detected_keys})"
                entries = []
                
                # Flatten entries from grouped structure
                if "watermarked" in data:
                    watermarked = data["watermarked"]
                    if isinstance(watermarked, list):
                        entries.extend(watermarked)
                    else:
                        raise ValueError(
                            f"Invalid manifest schema. "
                            f"'watermarked' key must contain a list of entries, got {type(watermarked)}. "
                            f"Manifest: {manifest_path_obj}"
                        )
                
                if "clean" in data:
                    clean = data["clean"]
                    if isinstance(clean, list):
                        entries.extend(clean)
                    else:
                        raise ValueError(
                            f"Invalid manifest schema. "
                            f"'clean' key must contain a list of entries, got {type(clean)}. "
                            f"Manifest: {manifest_path_obj}"
                        )
            
            # Legacy fallback: check for "samples" key
            elif "samples" in data:
                schema_type = "JSON (dict with 'samples' key)"
                entries = data["samples"]
                if not isinstance(entries, list):
                    raise ValueError(
                        f"Invalid manifest schema. "
                        f"'samples' key must contain a list of entries, got {type(entries)}. "
                        f"Manifest: {manifest_path_obj}"
                    )
            
            else:
                # Unknown dict structure
                raise ValueError(
                    f"Invalid manifest schema. "
                    f"Expected sample entry with field 'image_path'. "
                    f"Detected manifest root keys: {detected_keys}. "
                    f"Manifest: {manifest_path_obj}\n"
                    f"Supported formats:\n"
                    f"  - Flat list: [{{sample_entry}}, ...]\n"
                    f"  - Grouped dict: {{'watermarked': [...], 'clean': [...]}}\n"
                    f"  - Dict with 'samples' key: {{'samples': [...]}}\n"
                    f"Did you forget to flatten grouped manifest format?"
                )
        
        else:
            raise ValueError(
                f"Invalid manifest schema. "
                f"Expected list or dict at root, got {type(data)}. "
                f"Manifest: {manifest_path_obj}"
            )
    
    else:
        raise ValueError(
            f"Unsupported manifest format: {manifest_path_obj.suffix}. "
            f"Supported formats: .json, .jsonl"
        )
    
    # Validate entries
    if not entries:
        raise RuntimeError(
            f"Manifest contains no entries. "
            f"Manifest: {manifest_path_obj}, "
            f"Schema type: {schema_type or 'unknown'}"
        )
    
    # Validate each entry has required fields
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Invalid manifest entry {i}. "
                f"Expected dict, got {type(entry)}. "
                f"Manifest: {manifest_path_obj}, "
                f"Schema type: {schema_type or 'unknown'}"
            )
        
        if "image_path" not in entry:
            available_keys = list(entry.keys())
            raise RuntimeError(
                f"Entry {i} has no 'image_path' field.\n"
                f"Manifest: {manifest_path_obj}\n"
                f"Schema type: {schema_type or 'unknown'}\n"
                f"Entry keys: {available_keys}\n"
                f"Expected sample entry with field 'image_path'. "
                f"Did you forget to flatten grouped manifest format?"
            )
    
    # CRITICAL: Strip 'prompt' field immediately to prevent leakage into detection
    # This must happen AFTER validation to ensure we've loaded valid entries
    # but BEFORE returning entries so they never contain prompt data
    prompt_stripped_count = 0
    for entry in entries:
        if "prompt" in entry:
            del entry["prompt"]
            prompt_stripped_count += 1
    
    # Log warning if prompt was found and stripped (once)
    if prompt_stripped_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Stripped 'prompt' field from {prompt_stripped_count} manifest entries "
            f"to prevent information leakage into detection pipeline. "
            f"Manifest: {manifest_path_obj}"
        )
    
    return entries


# Alias for backward compatibility
def load_manifest(manifest_path: str) -> list[Dict[str, Any]]:
    """
    Load manifest file (JSON or JSONL).
    
    DEPRECATED: Use load_manifest_entries() instead.
    This function is kept for backward compatibility.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        List of manifest entries
    """
    return load_manifest_entries(manifest_path)


def extract_label(entry: Dict[str, Any], entry_index: int) -> int:
    """
    Extract binary label from manifest entry.
    
    STRICT: Only uses explicit 'label' field. No inference from paths or metadata.
    
    Args:
        entry: Manifest entry dictionary
        entry_index: Index of entry (for error messages)
        
    Returns:
        Label (0=unwatermarked, 1=watermarked)
        
    Raises:
        ValueError: If 'label' field is missing or invalid
    """
    if "label" not in entry:
        raise ValueError(
            f"Entry {entry_index} missing required 'label' field. "
            f"Manifest entries must explicitly specify label: 0 (unwatermarked) or 1 (watermarked)."
        )
    
    label = entry["label"]
    
    # Convert to int
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        label_int = int(label)
        if label_int not in (0, 1):
            raise ValueError(
                f"Entry {entry_index} has invalid label value: {label}. "
                f"Label must be 0 (unwatermarked) or 1 (watermarked)."
            )
        return label_int
    if isinstance(label, str):
        label_lower = label.lower()
        if label_lower in ["watermarked", "1", "true", "yes"]:
            return 1
        elif label_lower in ["unwatermarked", "0", "false", "no"]:
            return 0
        else:
            raise ValueError(
                f"Entry {entry_index} has invalid label string: {label}. "
                f"Label must be '0', '1', 'watermarked', or 'unwatermarked'."
            )
    
    raise ValueError(
        f"Entry {entry_index} has invalid label type: {type(label)}. "
        f"Label must be int (0 or 1), bool, or string."
    )


def extract_key_id(entry: Dict[str, Any], label: int, entry_index: int) -> str | None:
    """
    Extract key_id from manifest entry.
    
    STRICT: Only uses explicit 'key_id' field. No fallback logic.
    
    Args:
        entry: Manifest entry dictionary
        label: Binary label (0 or 1)
        entry_index: Index of entry (for error messages)
        
    Returns:
        key_id string if label==1, None if label==0
        
    Raises:
        ValueError: If label==1 and key_id is None or missing
        ValueError: If label==0 and key_id is not None
    """
    key_id = entry.get("key_id")
    
    # Enforce strict key_id consistency
    if label == 1:
        # Watermarked: key_id is REQUIRED
        if key_id is None:
            raise ValueError(
                f"Entry {entry_index} has label=1 (watermarked) but key_id is None or missing. "
                f"Watermarked entries must have a non-null key_id."
            )
        # Ensure it's a string
        return str(key_id)
    else:
        # Unwatermarked: key_id must be None
        if key_id is not None:
            raise ValueError(
                f"Entry {entry_index} has label=0 (unwatermarked) but key_id is not None: {key_id}. "
                f"Unwatermarked entries must have key_id=None."
            )
        return None


def compute_hash_of_dict(d: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of a dictionary.
    
    Args:
        d: Dictionary to hash
        
    Returns:
        Hexadecimal hash string
    """
    # Convert to JSON string with sorted keys for deterministic hashing
    json_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def precompute_inverted_g_values(
    manifest_path: str,
    output_dir: str,
    config: AppConfig,
    config_path: str,
    master_key: str,
    num_inversion_steps: int,
    device: str,
    seed: int = 42,
) -> None:
    """
    Precompute inverted g-values for all images in manifest.
    
    Args:
        manifest_path: Path to input manifest file
        output_dir: Directory to save g-values and output manifest
        config: Application configuration
        config_path: Path to config file (for metadata)
        master_key: Master key for PRF key derivation
        num_inversion_steps: Number of DDIM inversion steps
        device: Device to run inversion on
        seed: Random seed for deterministic processing
    """
    logger = setup_logging()
    
    # ============================================================================
    # Explicit Startup Logging (Required Parameters)
    # ============================================================================
    guidance_scale = config.diffusion.guidance_scale
    master_key_hash = hashlib.sha256(master_key.encode()).hexdigest()[:16]
    
    logger.info("=" * 80)
    logger.info("Precompute Pipeline Configuration")
    logger.info("=" * 80)
    logger.info(f"guidance_scale: {guidance_scale}")
    logger.info(f"num_inversion_steps: {num_inversion_steps}")
    logger.info(f"master_key_hash: {master_key_hash}")
    logger.info(f"UNWATERMARKED_DUMMY_KEY: {UNWATERMARKED_DUMMY_KEY}")
    logger.info("=" * 80)
    
    # Set seed for deterministic processing
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Infer dataset root from manifest location
    manifest_path_obj = Path(manifest_path).resolve()
    dataset_root = manifest_path_obj.parent
    
    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    entries = load_manifest(manifest_path)
    logger.info(f"Found {len(entries)} images to process")
    
    # Log dataset root and example path for debugging
    logger.info(f"Using inferred dataset root: {dataset_root}")
    if entries:
        first_image_rel_path = entries[0].get("image_path") or entries[0].get("path")
        if first_image_rel_path:
            first_example_path = dataset_root / first_image_rel_path
            logger.info(f"Example resolved image path: {first_example_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # CRITICAL: Enforce Correct Config Loading
    # ============================================================================
    # The script MUST only use the YAML config passed via --config-path.
    # No defaults, no fallbacks, no hardcoded parameters.
    # This ensures g-values are generated using the exact same watermark config
    # as image generation and detection.
    
    # Validate config structure (fail fast if missing)
    if not hasattr(config, 'watermark'):
        raise ValueError(
            "Config missing 'watermark' section. "
            "The provided config file must contain watermark configuration."
        )
    if not hasattr(config.watermark, 'algorithm_params'):
        raise ValueError(
            "Config missing 'watermark.algorithm_params' section. "
            "The provided config file must contain algorithm_params."
        )
    if not hasattr(config.watermark.algorithm_params, 'g_field'):
        raise ValueError(
            "Config missing 'watermark.algorithm_params.g_field' section. "
            "The provided config file must contain g_field configuration. "
            "This is required for g-value computation."
        )
    
    # Extract and validate g_field_config
    g_field_config = g_field_config_to_dict(config.watermark.algorithm_params.g_field)
    
    # Validate g_field_config is not empty or malformed
    if not g_field_config:
        raise ValueError(
            "G-field config is empty. "
            "The config.watermark.algorithm_params.g_field section must contain valid parameters."
        )
    
    # Validate required fields are present
    required_fields = ['mapping_mode', 'domain']
    missing_fields = [f for f in required_fields if f not in g_field_config]
    if missing_fields:
        raise ValueError(
            f"G-field config is missing required fields: {missing_fields}. "
            f"Full config: {json.dumps(g_field_config, indent=2)}"
        )
    
    # ============================================================================
    # Explicitly Log & Persist G-Field Provenance
    # ============================================================================
    # We log the fully resolved g-field config and compute a deterministic hash.
    # This hash + full config is saved to metadata.json to enable later assertion:
    # "These g-values were computed with the same config used during generation and detection."
    # The hash enables fast comparison across train/val/test datasets.
    
    logger.info("=" * 80)
    logger.info("G-Field Configuration (from --config-path)")
    logger.info("=" * 80)
    logger.info(f"Fully resolved g-field config:\n{json.dumps(g_field_config, indent=2)}")
    
    # Compute deterministic hash of g_field_config for provenance tracking
    # This hash is used to verify train/val/test symmetry (same config = same hash)
    g_field_config_hash = compute_hash_of_dict(g_field_config)
    logger.info(f"G-field config hash: {g_field_config_hash}")
    logger.info("=" * 80)
    
    # ============================================================================
    # Train/Val/Test Symmetry Enforcement (Comprehensive Validation)
    # ============================================================================
    # If metadata.json already exists, validate ALL symmetry parameters:
    # 1. g_field_config_hash (same watermark config)
    # 2. config_path (same config file)
    # 3. guidance_scale (same inversion guidance)
    # 4. num_inversion_steps (same inversion steps)
    # 5. master_key_hash (same master key for PRF)
    # This ensures all datasets (train/val/test) use identical parameters.
    
    metadata_path = output_dir / "metadata.json"
    config_path_obj = Path(config_path).resolve()
    
    # Required metadata fields for validation
    required_metadata_fields = [
        "latent_type",
        "num_inversion_steps",
        "guidance_scale",
        "g_field_config_hash",
        "master_key_hash",
        "config_path",
        "num_samples",
    ]
    
    if metadata_path.exists():
        logger.info(f"Existing metadata found at {metadata_path}")
        logger.info("Validating train/val/test symmetry...")
        
        with open(metadata_path, "r") as f:
            existing_metadata = json.load(f)
        
        # Check for missing required fields (fail immediately, no migration)
        missing_fields = [f for f in required_metadata_fields if f not in existing_metadata]
        if missing_fields:
            raise ValueError(
                f"Existing metadata is missing required fields: {missing_fields}.\n"
                f"Metadata path: {metadata_path}\n"
                f"Existing metadata keys: {list(existing_metadata.keys())}\n"
                f"This metadata format is incompatible. Please regenerate."
            )
        
        existing_hash = existing_metadata.get("g_field_config_hash")
        existing_config_path = existing_metadata.get("config_path")
        existing_guidance_scale = existing_metadata.get("guidance_scale")
        existing_num_inversion_steps = existing_metadata.get("num_inversion_steps")
        existing_master_key_hash = existing_metadata.get("master_key_hash")
        
        # Validate g_field_config_hash matches
        if existing_hash != g_field_config_hash:
            raise ValueError(
                f"G-field config hash mismatch!\n"
                f"  Existing hash (from {metadata_path}): {existing_hash}\n"
                f"  Current hash (from {config_path_obj}): {g_field_config_hash}\n"
                f"  This indicates different watermark configs were used.\n"
                f"  All datasets (train/val/test) must use the same config.\n"
                f"  Existing config path: {existing_config_path}\n"
                f"  Current config path: {config_path_obj}"
            )
        
        # Validate config path matches
        if existing_config_path != str(config_path_obj):
            raise ValueError(
                f"Config path mismatch!\n"
                f"  Existing config path: {existing_config_path}\n"
                f"  Current config path: {config_path_obj}\n"
                f"  Even though hashes match, config paths differ.\n"
                f"  This may indicate config file was moved/renamed.\n"
                f"  All datasets must use the same config file path."
            )
        
        # Validate guidance_scale matches
        if existing_guidance_scale != guidance_scale:
            raise ValueError(
                f"Guidance scale mismatch!\n"
                f"  Existing guidance_scale (from {metadata_path}): {existing_guidance_scale}\n"
                f"  Current guidance_scale (from config): {guidance_scale}\n"
                f"  Inversion parameters must match exactly for symmetry.\n"
                f"  All datasets (train/val/test) must use the same guidance_scale."
            )
        
        # Validate num_inversion_steps matches
        if existing_num_inversion_steps != num_inversion_steps:
            raise ValueError(
                f"Inversion steps mismatch!\n"
                f"  Existing num_inversion_steps (from {metadata_path}): {existing_num_inversion_steps}\n"
                f"  Current num_inversion_steps (from CLI): {num_inversion_steps}\n"
                f"  Inversion parameters must match exactly for symmetry.\n"
                f"  All datasets (train/val/test) must use the same num_inversion_steps."
            )
        
        # Validate master_key_hash matches
        if existing_master_key_hash != master_key_hash:
            raise ValueError(
                f"Master key hash mismatch!\n"
                f"  Existing master_key_hash (from {metadata_path}): {existing_master_key_hash}\n"
                f"  Current master_key_hash: {master_key_hash}\n"
                f"  Master key must be identical for PRF consistency.\n"
                f"  All datasets (train/val/test) must use the same master key."
            )
        
        logger.info("✓ All symmetry parameters validated")
        logger.info(f"  - g_field_config_hash: {g_field_config_hash}")
        logger.info(f"  - config_path: {config_path_obj}")
        logger.info(f"  - guidance_scale: {guidance_scale}")
        logger.info(f"  - num_inversion_steps: {num_inversion_steps}")
        logger.info(f"  - master_key_hash: {master_key_hash}")
    
    # Create subdirectories for g-values and masks
    g_values_dir = output_dir / "g_values"
    masks_dir = output_dir / "masks"
    g_values_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Stable Diffusion pipeline once
    logger.info("Loading Stable Diffusion pipeline...")
    pipeline = create_pipeline(config.diffusion, device=device)
    logger.info("Pipeline loaded and ready")
    
    # Create DDIM inverter
    inverter = DDIMInverter(pipeline, device=device)
    
    # Process each image
    g_manifest_entries = []
    
    # Track failures for dataset contamination detection
    total_images = len(entries)
    failed_images = 0
    
    logger.info("Processing images...")
    for i, entry in enumerate(tqdm(entries, desc="Precomputing g-values")):
        # Resolve image path
        image_rel_path = entry.get("image_path") or entry.get("path")
        if image_rel_path is None:
            raise RuntimeError(
                f"Entry {i} has no image_path field.\n"
                f"Manifest: {manifest_path_obj}\n"
                f"Entry keys: {list(entry.keys())}"
            )
        
        # Enforce strict path resolution: dataset_root / image_path
        image_path = dataset_root / image_rel_path
        
        # Fail fast if image doesn't exist
        if not image_path.exists():
            raise RuntimeError(
                f"Image not found while precomputing g-values.\n"
                f"Manifest: {manifest_path_obj}\n"
                f"Dataset root (inferred): {dataset_root}\n"
                f"image_path (manifest): {image_rel_path}\n"
                f"resolved path: {image_path}"
            )
        
        # Extract label and key_id with strict validation
        label = extract_label(entry, i)
        key_id = extract_key_id(entry, label, i)
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # ========================================================================
            # Detection Parity Guardrail: DDIM Inversion to zT
            # ========================================================================
            # CRITICAL: We MUST use latent_type="zT" (DDIM-inverted latent).
            # This is the exact point where SeedBias watermark was embedded during generation.
            # Using z0 (VAE-encoded) or any other latent type will produce incorrect g-values.
            # This must match detection code path exactly.
            
            # Perform DDIM inversion to get latent_T (zT)
            latent_T = inverter.invert(
                image,
                num_inference_steps=num_inversion_steps,
                prompt="",  # Unconditional inversion
                guidance_scale=guidance_scale,
            )  # [1, 4, 64, 64]
            
            # ========================================================================
            # Deterministic G-Value Computation
            # ========================================================================
            # For watermarked (label==1): Use master_key + key_id for PRF derivation.
            # For unwatermarked (label==0): Use fixed dummy key for computation only.
            # The dummy key ensures consistent computation, but key_id in manifest remains None.
            # This matches detection logic: unwatermarked samples have no semantic key.
            
            computation_key = key_id if key_id is not None else UNWATERMARKED_DUMMY_KEY
            g, mask = compute_g_values(
                latent_T,
                computation_key,
                master_key,
                return_mask=True,
                g_field_config=g_field_config,
                latent_type="zT",  # CRITICAL: Must be "zT" to match detection
            )  # g: [1, N] or [N], mask: [1, N] or [N]
            
            # ========================================================================
            # Detection Parity Guardrail: Masking Logic
            # ========================================================================
            # CRITICAL: This masking logic MUST match detection code path exactly.
            # Detection uses selection-based masking: g_selected = g[mask == 1]
            # Any deviation from this will cause train-detect mismatch.
            # If you need to change this, you must also update detection code.
            
            # Ensure batch dimension for consistency
            if g.dim() == 1:
                g = g.unsqueeze(0)  # [1, N]
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [1, N]
            
            # Apply masking exactly as done in detection (selection-based)
            # This is selection-based masking: we select only positions where mask == 1
            # The mask is structural (frequency-domain filtering) and key-independent
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
            
            # ========================================================================
            # Detection Parity Guardrail: Config-Driven G-Value Mapping
            # ========================================================================
            # mapping_mode is applied inside compute_g_values (binary: sign agreement → {0,1};
            # continuous: element-wise x0*G → real-valued). Here we only enforce binary
            # output for binary mode (idempotent) and never threshold in continuous mode.
            # Thresholding in continuous mode is forbidden — Gaussian likelihood requires ℝ.
            
            mapping_mode = g_field_config.get("mapping_mode")
            
            if mapping_mode == "binary":
                g = (g > 0).float()  # Ensure {0, 1} (idempotent with compute_g_values output)
            elif mapping_mode == "continuous":
                # No thresholding, sign, or rounding — g is already real-valued from compute_g_values
                pass
            else:
                raise ValueError(
                    f"Unknown mapping_mode '{mapping_mode}'. "
                    "Expected 'binary' or 'continuous'."
                )
            
            # Runtime assertion: continuous mode must yield real-valued g (required for Gaussian likelihood)
            if mapping_mode == "continuous":
                uniq = torch.unique(g)
                assert uniq.numel() > 10, (
                    f"[FATAL] Continuous mode collapsed to binary: unique={uniq[:10]}"
                )
                assert g.dtype == torch.float32, (
                    f"Continuous g-values must be float32, got {g.dtype}"
                )
                assert torch.unique(g).numel() > 2, (
                    f"Continuous mode must have many unique values (>>2), got {torch.unique(g).numel()}. "
                    "This indicates binarization or upstream bug."
                )
                assert not torch.all((g == 0) | (g == 1)), (
                    "Continuous mode must not be confined to {0, 1}. "
                    "Gaussian likelihood requires real-valued g."
                )
                assert g.std().item() > 0.01, (
                    f"Continuous g-values must be non-degenerate (std > 0.01), got std={g.std().item()}"
                )
            
            # Generate unique filenames
            g_filename = f"{uuid.uuid4().hex[:16]}.pt"
            g_path = g_values_dir / g_filename
            
            mask_filename = f"{uuid.uuid4().hex[:16]}.pt"
            mask_path = masks_dir / mask_filename if mask is not None else None
            
            # Save g-values and mask
            torch.save(g, g_path)
            if mask_path is not None:
                torch.save(mask, mask_path)
            
            # Create manifest entry (strict contract: g_path, label, key_id, mask_path; optional transform)
            g_manifest_entry = {
                "g_path": str(g_path.relative_to(output_dir)),
                "label": label,
                "key_id": key_id,  # None for unwatermarked, string for watermarked
            }

            if mask_path is not None:
                g_manifest_entry["mask_path"] = str(mask_path.relative_to(output_dir))

            # Preserve transform label when present (transform-aware pipeline)
            if "transform" in entry:
                g_manifest_entry["transform"] = entry["transform"]

            g_manifest_entries.append(g_manifest_entry)
            
        except Exception as e:
            failed_images += 1
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            continue
    
    # Log failure summary
    failure_ratio = failed_images / total_images if total_images > 0 else 0.0
    logger.info("=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"total_images: {total_images}")
    logger.info(f"failed_images: {failed_images}")
    logger.info(f"failure_ratio: {failure_ratio:.4f} ({failure_ratio * 100:.2f}%)")
    if failure_ratio > 0.01:
        logger.warning(
            f"Failure ratio exceeds 1% ({failure_ratio * 100:.2f}%). "
            f"This may indicate dataset contamination or configuration issues."
        )
    logger.info("=" * 80)
    
    # Save g-values manifest
    g_manifest_path = output_dir / "g_manifest.jsonl"
    logger.info(f"Saving g-values manifest to {g_manifest_path}")
    with open(g_manifest_path, "w") as f:
        for entry in g_manifest_entries:
            f.write(json.dumps(entry) + "\n")
    
    # ============================================================================
    # Save Metadata for Train-Detect Symmetry Checks
    # ============================================================================
    # This metadata enables later verification that:
    # 1. Generation, precompute, and detection used the same watermark config
    # 2. All datasets (train/val/test) used identical config (via hash comparison)
    # 3. The exact config file path is preserved for debugging
    # 4. All inversion parameters (guidance_scale, num_inversion_steps) match
    # 5. Master key is consistent (via hash comparison)
    # 
    # The g_field_config_hash is the key to proving config alignment:
    # - Same config → same hash → provably aligned
    # - Different hash → different config → mismatch detected
    
    # Resolve mapping_mode once for metadata (same source as per-image logic)
    mapping_mode = g_field_config.get("mapping_mode")
    
    metadata = {
        "latent_type": "zT",  # CRITICAL: Must match detection (zT, not z0)
        "num_inversion_steps": num_inversion_steps,
        "guidance_scale": guidance_scale,
        "g_field_config_hash": g_field_config_hash,  # Deterministic hash for fast comparison
        "master_key_hash": master_key_hash,  # Deterministic hash of master key (never store raw key)
        "g_field_config": g_field_config,  # Full config for detailed inspection
        "config_path": str(config_path_obj),  # Actual config file path
        "num_samples": len(g_manifest_entries),
        "mapping_mode": mapping_mode,  # Explicit: aids downstream debugging and experiment hygiene
    }
    
    logger.info(f"Saving metadata to {metadata_path}")
    logger.info(f"  G-field config hash: {g_field_config_hash}")
    logger.info(f"  Config path: {config_path_obj}")
    logger.info(f"  Latent type: zT (DDIM-inverted)")
    logger.info(f"  Guidance scale: {guidance_scale}")
    logger.info(f"  Inversion steps: {num_inversion_steps}")
    logger.info(f"  Master key hash: {master_key_hash}")
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Precomputation complete!")
    logger.info(f"  Processed: {len(g_manifest_entries)} images")
    logger.info(f"  Failed: {failed_images} images")
    logger.info(f"  G-values saved to: {g_values_dir}")
    logger.info(f"  Masks saved to: {masks_dir}")
    logger.info(f"  Manifest saved to: {g_manifest_path}")
    logger.info(f"  Metadata saved to: {metadata_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Precompute inverted g-values for Tier-2 Bayesian training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input manifest file (JSON or JSONL)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save g-values and output manifest",
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. seedbias.yaml)",
    )
    
    parser.add_argument(
        "--master-key",
        type=str,
        required=True,
        help="Master key for PRF key derivation",
    )
    
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        required=True,
        help="Number of DDIM inversion steps",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic processing (default: 42)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    logger.info("=" * 80)
    logger.info("Precompute Inverted G-Values")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_path}")
    config = AppConfig.from_yaml(args.config_path)
    
    # Get device
    device = get_device(args.device, use_fp16=config.diffusion.use_fp16)
    logger.info(f"Device: {device}")
    
    # Run precomputation
    precompute_inverted_g_values(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=config,
        config_path=args.config_path,
        master_key=args.master_key,
        num_inversion_steps=args.num_inversion_steps,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

