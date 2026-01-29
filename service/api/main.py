"""
FastAPI entrypoint for the public API service.

This service handles:
- Key registration
- Image generation requests (delegates to GPU worker)
- Detection requests (delegates to GPU worker)
- Health checks
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from service.api.config import get_config
from service.api.routes import router
from service.api.gpu_client import close_gpu_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()
    logger.info(f"Starting API service on {config.host}:{config.port}")
    logger.info(f"GPU worker URL: {config.gpu_worker_url}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API service")
    await close_gpu_client()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title="Watermark API",
        description="""
        GPU-backed watermarking service for image generation and detection.
        
        ## Features
        - **Key Registration**: Create unique watermark keys
        - **Image Generation**: Generate images with invisible watermarks
        - **Detection**: Detect watermarks in images
        
        ## Architecture
        This API service handles business logic and delegates heavy computation
        to a GPU worker service.
        """,
        version="1.0.0",
        lifespan=lifespan,
        debug=config.debug,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    # Mount static files for generated images
    # Images saved to storage_path will be accessible at /images/<filename>
    from pathlib import Path
    config = get_config()
    images_path = Path(config.storage_path)
    images_path.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(images_path)), name="images")
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "service.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
