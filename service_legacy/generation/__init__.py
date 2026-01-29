"""
Generation adapters for watermarking.

Phase-1: Stable Diffusion only (hosted generation)
Phase-2: Client-side generation (credential-only watermark issuance)
"""
from .base import GenerationAdapter
from .stable_diffusion import StableDiffusionSeedBiasAdapter

__all__ = ["GenerationAdapter", "StableDiffusionSeedBiasAdapter"]

