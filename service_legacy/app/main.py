"""
FastAPI application entrypoint.

Run with:
    uvicorn service.app.main:app --host 0.0.0.0 --port 8000

Environment Variables:
    USE_STRUCTURED_LOGGING: Set to "false" to disable structured logging (default: true)
    LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
    LOG_FORMAT: "json" for JSON output, "console" for colored output (default: auto-detect)
    ENABLE_DOCS: Set to "true" to enable Swagger/ReDoc docs (default: false in production)
    ENVIRONMENT: "production" or "development" (default: production)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from google.cloud import pubsub_v1
import json

from .routes import generate, detect, health, evaluate, demo, jobs
from .middleware import RateLimitMiddleware, RequestIDMiddleware
# NOTE: TimeoutMiddleware REMOVED - cancelling GPU operations corrupts CUDA state
# Timeouts are enforced at API→Worker RPC boundary only (in RemoteInferenceClient)
from .exceptions import (
    ServiceError,
    ValidationError as ServiceValidationError,
    NotFoundError,
    RateLimitError,
    InferenceError,
    ServiceUnavailableError,
    TimeoutError as ServiceTimeoutError,
    WatermarkRevokedError,
)

# Configure structured logging BEFORE importing anything else that uses logging
from service.infra.logging import configure_logging, get_logger, get_request_id




configure_logging()
logger = get_logger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"

# In development mode, enable docs by default
if ENVIRONMENT == "development":
    ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"




PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "data-scraper-pipeline")
TOPIC_ID = "watermark-jobs"


publisher = None
topic_path = None

if os.getenv("ENABLE_PUBSUB", "false").lower() == "true":
    try:
        from google.cloud import pubsub_v1

        PROJECT_ID = os.environ["PROJECT_ID"]
        TOPIC_ID = os.environ["TOPIC_ID"]

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

        print(f"✅ PubSub enabled → {topic_path}")

    except Exception as e:
        publisher = None
        topic_path = None
        print(f"⚠️ PubSub failed to initialize — disabling. Reason: {e}")
else:
    print("⚠️ PubSub disabled (ENABLE_PUBSUB=false)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.
    
    Loads models and configs at startup.
    Validates artifact paths and consistency (config hash, mask shape).
    Preloads Stable Diffusion pipeline to eliminate first-request latency.
    
    Architecture:
    - Creates AppState to hold all service instances
    - Stores AppState in app.state.services
    - Routes access services via request.app.state.services
    """
    import os
    from .dependencies import (
        AppState,
        preload_pipeline,
        get_detection_service,
        is_detection_available,
        get_watermark_authority,
    )
    from .artifact_resolver import get_artifact_resolver
    from service.detector_artifacts import DetectorArtifacts
    
    # Startup
    logger.info("service_starting", extra={"service": "watermarking"})
    
    # Create AppState for explicit lifecycle management
    app_state = AppState()
    app.state.services = app_state
    
    # Preload Stable Diffusion pipeline at startup
    # This eliminates first-request latency spikes and prevents accidental re-initialization
    try:
        preload_pipeline()
        logger.info("pipeline_preloaded", extra={"status": "success"})
    except Exception as e:
        logger.error("pipeline_preload_failed", extra={"error": str(e)}, exc_info=True)
        raise RuntimeError(f"Pipeline preload failed: {e}") from e
    
    # Enable micro-batching for detection if artifacts are available
    # This improves GPU utilization by batching detection requests
    if is_detection_available():
        try:
            detection_service = get_detection_service()
            # Start worker asynchronously (lifespan is async context)
            await detection_service.enable_micro_batching(
                batch_window_ms=20.0,  # 20ms collection window
                max_batch_size=8,      # Max 8 requests per batch
            )
            logger.info(
                "micro_batching_enabled",
                extra={"batch_window_ms": 20.0, "max_batch_size": 8}
            )
        except Exception as e:
            logger.warning(
                "micro_batching_failed",
                extra={"error": str(e)}
            )
            # Don't fail startup if micro-batching fails - detection will still work
    
    # Register test keys at startup if enabled
    # This makes key registration deterministic, version-controlled, and environment-aware
    # Controlled by REGISTER_TEST_KEYS environment variable (explicit opt-in)
    if os.getenv("REGISTER_TEST_KEYS") == "true":
        logger.info("test_keys_registration_starting", extra={"enabled": True})
        try:
            authority = get_watermark_authority()
            key_id = "test_batch_001"
            
            # Check if key already exists (idempotent)
            # Try to get the key, but handle decryption errors gracefully
            # (keys encrypted with different encryption keys will fail to decrypt)
            try:
                existing_record = authority.db.get_watermark(key_id)
                if existing_record is not None:
                    logger.info(
                        "test_key_exists",
                        extra={"key_id": key_id, "action": "skipped"}
                    )
                    # Key exists and can be decrypted, no action needed
                else:
                    # Key doesn't exist, register it
                    logger.info(
                        "test_key_registering",
                        extra={"key_id": key_id}
                    )
                    policy = authority.create_watermark_policy(key_id=key_id)
                    logger.info(
                        "test_key_registered",
                        extra={
                            "key_id": key_id,
                            "watermark_version": policy["watermark_version"],
                        }
                    )
            except Exception as e:
                # If decryption fails (e.g., encryption key mismatch), check if key exists in DB
                # If it exists but can't be decrypted, we'll overwrite it with a new registration
                if key_id in authority.db._db:
                    logger.warning(
                        "test_key_decryption_failed",
                        extra={
                            "key_id": key_id,
                            "action": "will_overwrite",
                            "reason": "encryption_key_mismatch",
                        }
                    )
                # Register key (will overwrite if exists, create if doesn't)
                logger.info(
                    "test_key_registering",
                    extra={"key_id": key_id}
                )
                policy = authority.create_watermark_policy(key_id=key_id)
                logger.info(
                    "test_key_registered",
                    extra={
                        "key_id": key_id,
                        "watermark_version": policy["watermark_version"],
                    }
                )
        except Exception as e:
            # Don't fail startup if registration fails - log and continue
            logger.warning(
                "test_keys_registration_failed",
                extra={"error": str(e)}
            )
    
    # Resolve and validate artifact paths at startup using centralized resolver
    
    # Log environment variable status at startup
    env_likelihood_path = os.getenv("LIKELIHOOD_PARAMS_PATH")
    logger.info(
        "artifact_env_vars",
        extra={
            "LIKELIHOOD_PARAMS_PATH": env_likelihood_path or "NOT_SET",
        }
    )
    
    # Use centralized artifact resolver (validates and caches at startup)
    resolver = get_artifact_resolver()
    artifact_result = resolver.resolve()
    
    if artifact_result.is_available:
        # Validate artifacts can be loaded (fail fast if invalid)
        logger.info("artifact_validation_starting")
        try:
            authority = get_watermark_authority()
            
            # Create a test policy to get g_field_config
            test_policy = authority.create_watermark_policy()
            test_key_id = test_policy["key_id"]
            config = authority.get_detection_config(test_key_id)
            g_field_config = config["g_field_config"]
            
            # Try to load artifacts (this will validate consistency)
            # This will raise if artifacts are invalid
            artifacts = DetectorArtifacts(
                likelihood_params_path=artifact_result.likelihood_params_path_str,
                mask_path=artifact_result.mask_path_str,
                g_field_config=g_field_config,
            )
            
            logger.info(
                "artifact_validation_passed",
                extra={
                    "num_positions": artifacts.num_positions,
                    "mask_shape": list(artifacts.mask.shape) if artifacts.mask is not None else None,
                }
            )
            
        except Exception as e:
            # Fail fast if artifacts are invalid
            logger.error(
                "artifact_validation_failed",
                extra={
                    "error": str(e),
                    "hint": "Check LIKELIHOOD_PARAMS_PATH and MASK_PATH environment variables",
                }
            )
            raise RuntimeError(
                f"Failed to validate detector artifacts at startup: {e}"
            ) from e
    else:
        # Artifacts not available - log warning but don't fail startup (demo mode may work)
        logger.warning(
            "artifacts_not_available",
            extra={
                "error": artifact_result.error_message,
                "hint": "Detection endpoints will be unavailable until artifacts are configured",
            }
        )
    
    # Mark app as ready
    app_state.is_ready = True
    logger.info("service_ready", extra={"service": "watermarking"})
    
    yield
    
    # Shutdown
    logger.info("service_stopping", extra={"service": "watermarking"})
    
    # Mark app as not ready
    app_state.is_ready = False
    
    # Disable micro-batching if it was enabled
    if is_detection_available():
        try:
            detection_service = get_detection_service()
            if detection_service._detection_worker is not None:
                await detection_service.disable_micro_batching()
                logger.info("micro_batching_disabled", extra={"status": "success"})
        except Exception as e:
            logger.warning(
                "micro_batching_disable_failed",
                extra={"error": str(e)}
            )
    
    logger.info("service_stopped", extra={"service": "watermarking"})


# Create FastAPI app with conditional docs
app = FastAPI(
    title="Watermarking Service",
    description="Custodial API for watermark generation and detection",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production unless explicitly enabled
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Handle all ServiceError subclasses."""
    request_id = get_request_id()
    
    logger.warning(
        "service_error",
        extra={
            "error_type": exc.error_type,
            "error_code": exc.code,
            "message": exc.message,
            "status_code": exc.status_code,
        }
    )
    
    response_data = exc.to_dict()
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors from request parsing."""
    request_id = get_request_id()
    
    # Extract error details
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(
        "validation_error",
        extra={
            "errors": errors,
            "error_count": len(errors),
        }
    )
    
    response_data = {
        "error": "validation_error",
        "message": "Request validation failed",
        "details": errors,
    }
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=422,
        content=response_data,
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError from service layer (e.g., watermark not found)."""
    request_id = get_request_id()
    message = str(exc)
    
    # Determine status code based on error message
    status_code = 400
    error_type = "validation_error"
    
    if "not found" in message.lower():
        status_code = 404
        error_type = "not_found"
    elif "revoked" in message.lower():
        status_code = 410
        error_type = "watermark_revoked"
    
    logger.warning(
        "value_error",
        extra={
            "error_type": error_type,
            "message": message,
            "status_code": status_code,
        }
    )
    
    response_data = {
        "error": error_type,
        "message": message,
    }
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Handle RuntimeError from service layer."""
    request_id = get_request_id()
    message = str(exc)
    
    # Determine status code based on error message
    status_code = 500
    error_type = "internal_error"
    
    if "artifacts" in message.lower() or "not configured" in message.lower():
        status_code = 503
        error_type = "service_unavailable"
    elif "pipeline" in message.lower():
        status_code = 503
        error_type = "service_unavailable"
    
    logger.error(
        "runtime_error",
        extra={
            "error_type": error_type,
            "message": message,
            "status_code": status_code,
        }
    )
    
    response_data = {
        "error": error_type,
        "message": message,
    }
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exception."""
    request_id = get_request_id()
    
    logger.error(
        "unhandled_exception",
        extra={
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
        exc_info=True,
    )
    
    response_data = {
        "error": "internal_error",
        "message": "An unexpected error occurred",
    }
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=500,
        content=response_data,
    )


# =============================================================================
# Middleware Configuration
# =============================================================================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: TimeoutMiddleware REMOVED
# REASON: Cancelling in-flight GPU operations via asyncio.wait_for() can corrupt
# CUDA state, leading to memory corruption and undefined behavior.
# 
# Timeouts are now enforced ONLY at:
# - API → Worker RPC boundary (RemoteInferenceClient timeout_seconds)
# - Worker gracefully rejects new requests when overloaded (503)
# - Worker NEVER cancels in-flight GPU kernels

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Add request ID middleware for correlation
# This MUST be added last so it executes first (middleware stack is LIFO)
app.add_middleware(RequestIDMiddleware)

# Include routers
app.include_router(generate.router, prefix="/api/v1", tags=["generation"])
app.include_router(detect.router, prefix="/api/v1", tags=["detection"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(demo.router, prefix="/api/v1", tags=["demo"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])

# Path to demo.html
DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """
    Serve the Phase-1 demo page at /demo (user-friendly URL).
    
    This is a presentation-only demo for non-technical users (e.g., recruiters).
    It wraps the existing API endpoints without modifying any watermark logic.
    """
    try:
        with open(DEMO_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Demo not found</h1><p>demo.html is missing from service/app/static/</p>",
            status_code=404
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Watermarking Service",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "detect": "/api/v1/detect",
            "evaluate": "/api/v1/evaluate/imperceptibility",
            "health": "/api/v1/health",
            "demo": "/demo",
            "demo_verify": "/api/v1/demo/verify",
        }
    }

