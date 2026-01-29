"""
Route for generating watermark seeds.

POST /generate_seed
"""
from __future__ import annotations

import base64
import json
from service.infra.logging import get_logger
from typing import Dict

import torch
from fastapi import APIRouter, HTTPException

from service.app.schemas import GenerateSeedRequest, GenerateSeedResponse
from service.infra.db import get_db
from service.infra.security import generate_watermark_id, generate_master_key
from service.app.dependencies import create_seed_bias_strategy_for_service, get_pipeline

logger = get_logger(__name__)

router = APIRouter()


@router.post("/generate_seed", response_model=GenerateSeedResponse)
async def generate_seed(request: GenerateSeedRequest) -> GenerateSeedResponse:
    """
    Generate a watermark seed for client-side image generation.
    
    This endpoint:
    1. Creates a new watermark internally (generates watermark_id and secret_key)
    2. Uses seed bias strategy to create watermarked initial latent z_T
    3. Serializes the seed payload (seed + key_id) for client use
    4. Stores the secret key internally (never returned to client)
    
    The client will use the seed_payload to initialize their diffusion process
    with the watermarked latent.
    """
    try:
        # Step 1: Generate watermark_id and secret_key
        watermark_id = generate_watermark_id()
        secret_key = generate_master_key()
        
        logger.info(f"Generated watermark_id: {watermark_id}")
        
        # Step 2: Get pipeline
        pipeline = get_pipeline(model_id="runwayml/stable-diffusion-v1-5")
        device = str(pipeline.device)
        
        # Step 3: Create seed bias strategy
        strategy = create_seed_bias_strategy_for_service(
            master_key=secret_key,
            model=request.model,
            device=device,
        )
        
        # Step 4: Generate seed payloads for each image
        # For seed bias, we need to generate the initial latent z_T
        # The client will use this to initialize their diffusion
        
        # Generate a random seed for the noise component
        import secrets
        seed = secrets.randbelow(2**31)  # Random seed for epsilon
        
        # Prepare strategy for this sample
        strategy.prepare_for_sample(
            sample_id=watermark_id,
            prompt="",  # Not used for seed bias
            seed=seed,
            key_id=watermark_id,  # Use watermark_id as key_id
        )
        
        # Generate initial latent z_T with watermark
        latent_shape = strategy.latent_shape
        z_T = strategy.get_initial_latent(
            shape=latent_shape,
            seed=seed,
            key_id=watermark_id,
        )
        
        # Step 5: Serialize seed payload
        # For client-side generation, we need to provide:
        # - seed: Random seed for epsilon
        # - key_id: Public key identifier (watermark_id)
        # - z_T: Pre-computed watermarked initial latent (optional, can be recomputed)
        
        # Serialize z_T to base64
        z_T_np = z_T.detach().cpu().numpy()
        z_T_bytes = z_T_np.tobytes()
        z_T_b64 = base64.b64encode(z_T_bytes).decode('utf-8')
        
        seed_payload_data = {
            "seed": seed,
            "key_id": watermark_id,
            "z_T_b64": z_T_b64,
            "shape": list(latent_shape),
        }
        
        seed_payload = base64.b64encode(
            json.dumps(seed_payload_data).encode('utf-8')
        ).decode('utf-8')
        
        # Step 6: Store watermark record
        db = get_db()
        db.create_watermark(
            watermark_id=watermark_id,
            secret_key=secret_key,
            model=request.model,
            strategy="seed_bias",
        )
        
        logger.info(f"Created watermark {watermark_id} for {request.num_images} images")
        
        return GenerateSeedResponse(
            watermark_id=watermark_id,
            seed_payload=seed_payload,
        )
    
    except Exception as e:
        logger.error(f"Error generating seed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate seed: {str(e)}")

