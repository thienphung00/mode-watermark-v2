# Service Folder Structure

This document describes the folder structure and modules within the `service/` directory, which contains the FastAPI service layer for the watermarking engine.

## Overview

The service layer provides a custodial API for watermark generation and detection, wrapping the existing watermarking engine in `src/`.

```
service/
├── __init__.py
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── artifact_paths.py
│   ├── artifact_resolver.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── main.py
│   ├── middleware.py
│   ├── schemas.py
│   ├── routes/                   # API route handlers
│   │   ├── __init__.py
│   │   ├── demo.py
│   │   ├── detect.py
│   │   ├── evaluate.py
│   │   ├── generate_seed.py
│   │   ├── generate.py
│   │   ├── health.py
│   │   ├── jobs.py
│   │   └── pubsub.py
│   └── static/
│       └── demo.html
├── configs/
│   └── runtime.yaml
├── generation/                   # Image generation adapters
│   ├── __init__.py
│   ├── base.py
│   └── stable_diffusion.py
├── inference/                    # Inference client implementations
│   ├── __init__.py
│   ├── client.py
│   ├── exceptions.py
│   └── schemas.py
├── infra/                        # Infrastructure utilities
│   ├── __init__.py
│   ├── db.py
│   ├── logging.py
│   ├── security.py
│   └── settings.py
├── worker/                       # GPU worker service
│   ├── __init__.py
│   ├── Dockerfile
│   ├── main.py
│   ├── model_loader.py
│   ├── routes.py
│   └── schemas.py
├── ARCHITECTURE.md
├── authority.py
├── detection_worker.py
├── detection.py
├── detector_artifacts.py
├── Dockerfile
├── evaluation_metrics.py
├── evaluation.py
├── README.md
└── STARTUP_OPTIMIZATION.md
```

---

## Root-Level Modules

### `__init__.py`
Package initialization for the service layer. Contains docstring describing the service as a FastAPI wrapper for the watermarking engine.

### `authority.py`
**Watermark Authority Service** - The cryptographic and statistical authority of the system.

**Key Responsibilities:**
- Owns the `master_key` (never exposed to clients or workers)
- Manages `key_id` → watermark policy mapping
- Handles embedding configuration and detection configuration (Bayesian parameters)
- Manages calibration versions
- Provides scoped key derivation (workers receive derived keys, never `master_key`)

**Main Class: `WatermarkAuthorityService`**
- `create_watermark_policy()` - Creates new watermark policies with key generation
- `get_watermark_payload()` - Gets watermark configuration for generation (includes security: master_key only for local use)
- `get_detection_config()` - Gets detection configuration for watermark detection
- `revoke_watermark()` - Revokes a watermark (marks as inactive)
- `_compute_policy_version()` - Computes deterministic policy version from statistical parameters

**Security Invariant:** `master_key` NEVER leaves this service; workers only receive `derived_key`.

### `detection.py`
**Detection Service** - Bayesian-only watermark detection service.

**Key Responsibilities:**
- Fetches detection configuration from `WatermarkAuthorityService`
- Loads detector artifacts (likelihood_params.json, mask, g-field config)
- Performs prompt-free DDIM inversion (unconditional detection)
- Computes g-values and runs Bayesian detection
- Returns product-level results (no research internals exposed)

**Main Classes:**
- `StableDiffusionDetectionBackend` - SD-specific operations (VAE encoding, DDIM inversion)
- `DetectionService` - Main detection service with micro-batching support
  - `detect()` - Full detection pipeline from image to result
  - `detect_from_g_values()` - Detection from precomputed g-values (for testing)
  - `enable_micro_batching()` / `disable_micro_batching()` - Batch processing control

### `detection_worker.py`
**Detection Worker** - Background worker for micro-batched detection requests. Improves GPU utilization by batching detection requests within a configurable time window.

### `detector_artifacts.py`
**Detector Artifacts Loader** - Loads and caches immutable artifacts for Bayesian detection.

**Artifacts Managed:**
- `likelihood_params.json` - Trained likelihood parameters
- Mask tensor - Structural mask geometry
- G-field config - G-field generation parameters

**Features:**
- Read-only artifacts (never modified)
- Loaded once at startup or first request
- Cached in memory
- Validated for consistency (config hash, mask shape)

### `evaluation.py`
**Imperceptibility Evaluation Service** - Evaluation-only service for generating baseline and watermarked images for comparison.

**⚠️ EVALUATION-ONLY** - Not part of production watermarking system.

**Main Class: `ImperceptibilityEvaluationService`**
- `generate_baseline()` - Generates unwatermarked baseline image
- `generate_watermarked()` - Generates watermarked image using fixed evaluation key
- `evaluate_imperceptibility()` - Generates both images and computes difference metrics (L2, PSNR, SSIM)
- Uses optimized paired-diffusion approach (~40-50% compute reduction)

### `evaluation_metrics.py`
Metrics computation for imperceptibility evaluation (L2, PSNR, SSIM calculations).

### `Dockerfile`
Docker configuration for the API service container.

---

## `app/` - FastAPI Application

### `main.py`
**FastAPI Application Entrypoint**

**Features:**
- Lifespan context manager for startup/shutdown
- Model preloading to eliminate first-request latency
- Micro-batching enablement for detection
- Artifact validation at startup
- Exception handlers for all error types
- CORS, rate limiting, and request ID middleware

**Environment Variables:**
- `USE_STRUCTURED_LOGGING` - Enable/disable structured logging
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_DOCS` - Enable Swagger/ReDoc documentation
- `REGISTER_TEST_KEYS` - Register test keys at startup

### `dependencies.py`
**Shared Dependencies for FastAPI Routes**

**Provides Dependency Injection for:**
- `WatermarkAuthorityService`
- `GenerationAdapter` (Phase-1: Stable Diffusion)
- `DetectionService` (Bayesian-only)
- `InferenceClient` (local or remote)

**Main Class: `AppState`** - Holds all service instances, initialized during lifespan.

**Key Functions:**
- `get_watermark_authority()` - Returns singleton authority service
- `get_generation_adapter()` - Returns generation adapter instance
- `get_detection_service()` - Returns detection service instance
- `preload_pipeline()` - Preloads Stable Diffusion pipeline

### `schemas.py`
**Pydantic Schemas for API Requests/Responses**

**Base Class: `ServiceBaseModel`** - Shared configuration (strip whitespace, validate assignments, forbid extra fields)

**Request/Response Models:**
- `GenerateRequest` / `GenerateResponse`
- `DetectRequest` / `DetectResponse`
- `EvaluateRequest` / `EvaluateResponse`
- Field validators for key_id format, base64 image validation, etc.

### `exceptions.py`
**Custom Exception Hierarchy**

**Base: `ServiceError`**
- `ValidationError` (400) - Invalid input
- `NotFoundError` (404) - Resource not found
- `ConflictError` (409) - Resource conflict
- `RateLimitError` (429) - Rate limit exceeded
- `InferenceError` (502) - GPU worker error
- `ServiceUnavailableError` (503) - Service unavailable
- `TimeoutError` (504) - Request timeout
- `WatermarkRevokedError` (410) - Watermark revoked
- `ArtifactsNotConfiguredError` (503) - Missing detection artifacts

### `middleware.py`
**Middleware for Rate Limiting, Security, and Observability**

**Middleware Classes:**
- `RequestIDMiddleware` - Injects unique request ID for correlation (X-Request-ID header)
- `RateLimitMiddleware` - In-memory rate limiting per IP (use Redis in production)

### `artifact_paths.py` / `artifact_resolver.py`
Utilities for resolving and validating artifact file paths (likelihood parameters, masks).

---

## `app/routes/` - API Route Handlers

### `generate.py`
**POST /api/v1/generate** - Generate watermarked images.
- Creates/retrieves watermark policy from authority
- Uses GenerationAdapter for watermarked image generation
- Returns image and metadata

### `detect.py`
**POST /api/v1/detect** - Watermark detection.
- Uses DetectionService (Bayesian-only)
- Returns detection result (detected, score, confidence, policy_version)

### `evaluate.py`
**POST /api/v1/evaluate/imperceptibility** - Imperceptibility evaluation.
- Generates baseline and watermarked images
- Returns both images with difference metrics

### `health.py`
**GET /api/v1/health** - Health check endpoints.
- Liveness and readiness probes
- Service status information

### `demo.py`
Demo endpoints for presentation purposes.

### `jobs.py`
Job management endpoints for async operations.

### `pubsub.py`
Google Cloud Pub/Sub integration for event handling.

### `generate_seed.py`
Seed generation utilities for reproducible image generation.

---

## `generation/` - Image Generation Adapters

### `base.py`
**Abstract GenerationAdapter Interface**

Decouples API layer from specific generation methods.
- Phase-1: Hosted Stable Diffusion generation
- Phase-2: Client-side generation (API only issues credentials)

**Abstract Methods:**
- `generate()` - Generate watermarked image
- `get_model_info()` - Get model information

### `stable_diffusion.py`
**Stable Diffusion Adapter with Seed-Bias Watermarking**

**Main Class: `StableDiffusionSeedBiasAdapter`**
- Wraps Stable Diffusion pipeline
- Applies seed-bias watermarking at z_T (initial latent)
- Handles SD-specific configuration (FP16, device selection)

---

## `inference/` - Inference Client Implementations

### `client.py`
**Inference Client Protocol and Implementations**

**Protocol: `InferenceClient`**
- `detect()` - Perform watermark detection
- `generate()` - Generate watermarked image
- `health_check()` - Check worker health

**Implementations:**
- `LocalInferenceClient` - In-process inference using local GPU
- `RemoteInferenceClient` - HTTP-based remote inference to GPU workers
  - Circuit breaker pattern for cascading failure protection
  - Operation-aware retry logic (detection=idempotent, generation=non-idempotent)

**Security:** `master_key` is NEVER transmitted; only `derived_key` is sent.

### `schemas.py`
Pydantic schemas for inference requests/responses.
- `DetectInferenceRequest` / `DetectInferenceResponse`
- `GenerateInferenceRequest` / `GenerateInferenceResponse`
- `HealthStatus` / `HealthStatusEnum`

### `exceptions.py`
Inference-specific exceptions:
- `InferenceError` - Base inference error
- `InferenceTimeoutError` - Request timeout
- `InferenceConnectionError` - Connection failure
- `CircuitBreakerOpenError` - Circuit breaker is open
- `WorkerOverloadedError` - Worker overloaded (503)

---

## `infra/` - Infrastructure Utilities

### `db.py`
**Watermark Key Database**

**Main Class: `WatermarkKeyDB`** - In-memory database for watermark keys (replace with PostgreSQL in production).

**Features:**
- Stores `watermark_id` → `secret_key` mapping with metadata
- Encrypted key storage via `KeyEncryption`
- Persistent storage to JSON file
- Key lifecycle management (create, get, revoke, is_active)

### `security.py`
**Security Utilities for Key Generation and Encryption**

**Key Operations:**
- `OperationType` enum - GENERATION, DETECTION
- `generate_watermark_id()` - Generate unique watermark ID (wm_xxxxx)
- `generate_master_key()` - Generate 32-byte master key
- `compute_key_fingerprint()` - Non-reversible key fingerprint for cache keying
- `derive_scoped_key()` - Derive scoped ephemeral key from master key
- `validate_key_fingerprint()` - Validate key fingerprint format

**Encryption Class: `KeyEncryption`**
- Fernet (symmetric) encryption for key storage
- Key rotation support with dual-key decryption
- PBKDF2 key derivation from password

### `settings.py`
**Centralized Configuration Management**

**Main Class: `Settings`** (Pydantic BaseSettings)

**Configuration Categories:**
- Environment (ENVIRONMENT, SERVICE_VERSION)
- Database (DATABASE_URL, STORAGE_PATH)
- Encryption (ENCRYPTION_KEY, key rotation)
- Inference (INFERENCE_MODE: local/remote, WORKER_URLS)
- Model Configuration (MODEL_ID, DEVICE, USE_FP16)
- Detection Artifacts (LIKELIHOOD_PARAMS_PATH, MASK_PATH)
- Feature Flags (ENABLE_DOCS, ENABLE_METRICS, ENABLE_MICRO_BATCHING)
- Rate Limiting (per-endpoint limits)
- Logging (LOG_LEVEL, LOG_FORMAT)
- CORS (CORS_ORIGINS, credentials)

### `logging.py`
**Structured Logging Configuration**

**Features:**
- JSON-formatted structured logs via `structlog`
- Request ID correlation (context variable)
- Sensitive data sanitization (redacts master_key, secret_key, etc.)
- Configurable log levels

**Key Functions:**
- `get_logger()` - Get configured logger instance
- `get_request_id()` / `set_request_id()` / `clear_request_id()` - Request ID management
- `configure_logging()` - Configure logging processors and format

---

## `worker/` - GPU Worker Service

Standalone FastAPI service for GPU-intensive inference operations.

### `main.py`
**GPU Worker FastAPI Application**

**Lifespan Management:**
- Load settings from environment
- Initialize model loader
- Load and warmup models
- Cleanup on shutdown

**Environment Variables:**
- `MODEL_ID` - Hugging Face model ID
- `DEVICE` - Device (cuda, mps, cpu, auto)
- `MAX_CONCURRENT_REQUESTS` - Concurrent request limit
- `MAX_QUEUE_SIZE` - Queue size limit
- `GPU_SEMAPHORE_SIZE` - Parallel GPU operations limit

### `model_loader.py`
**Model Loading and Management**

**Main Class: `ModelLoader`**
- Loads Stable Diffusion pipeline
- Loads detector artifacts
- Handles model warmup
- GPU memory management

**Settings Class: `WorkerSettings`** - Worker-specific configuration.

### `routes.py`
**Worker API Routes**

- `/detect` - Watermark detection endpoint
- `/generate` - Image generation endpoint
- `/health` - Health check endpoint
- Request queue management
- GPU semaphore for parallel operation control

### `schemas.py`
Pydantic schemas for worker requests/responses.

### `Dockerfile`
Docker configuration for GPU worker container.

---

## `configs/`

### `runtime.yaml`
Runtime configuration file for the service.

---

## Documentation Files

- **ARCHITECTURE.md** - Service architecture documentation
- **README.md** - Service overview and usage
- **STARTUP_OPTIMIZATION.md** - Startup optimization details
- **FOLDER_STRUCTURE.md** - This file
