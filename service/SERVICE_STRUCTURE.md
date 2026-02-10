# Service Directory Structure & Documentation

This document provides a comprehensive overview of the `/service` directory, which contains a production-ready GPU-backed watermarking system for image generation and detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Component Details](#component-details)
   - [API Service (`/api`)](#api-service-api)
   - [GPU Worker (`/gpu`)](#gpu-worker-gpu)
   - [Docker Configuration (`/docker`)](#docker-configuration-docker)
   - [Scripts (`/scripts`)](#scripts-scripts)
5. [File-by-File Documentation](#file-by-file-documentation)
6. [Data Flow](#data-flow)
7. [Security Model](#security-model)
8. [Configuration Reference](#configuration-reference)

---

## Overview

The service implements a **two-tier architecture** for watermarking images:

| Component | Role | Hardware | Port |
|-----------|------|----------|------|
| **API Service** | Public-facing REST API, business logic, key management | CPU | 8000 |
| **GPU Worker** | Heavy computation (image generation, DDIM inversion) | GPU | 8001 |

This separation allows:
- Horizontal scaling of the API tier
- GPU resource isolation
- Security boundary between public endpoints and sensitive operations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          External Clients                           │
│                     (Web apps, CLI, integrations)                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ HTTPS (port 8000)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API Service (CPU)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │   Routes    │  │  Authority  │  │  Key Store  │  │  Storage   │  │
│  │  (FastAPI)  │  │  (Security) │  │   (JSON)    │  │(Local/GCS) │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ Internal HTTP (port 8001)
                                 │ (derived keys for generation; master keys for detection)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Worker (CUDA)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Pipeline   │  │  SD Model   │  │   DDIM      │                  │
│  │  Manager    │  │  (HF/local) │  │  Inverter   │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
service/
├── __init__.py                  # Package initialization, version info
├── .env.example                 # Environment variable template
├── README.md                    # Quick start guide
├── SERVICE_STRUCTURE.md         # This file
│
├── api/                         # Public API Service (CPU)
│   ├── __init__.py              # API package exports
│   ├── main.py                  # FastAPI application entrypoint
│   ├── routes.py                # REST endpoint definitions
│   ├── schemas.py               # Pydantic request/response models
│   ├── config.py                # Environment configuration loader
│   ├── authority.py             # Key validation & derivation
│   ├── detector.py              # Detection logic wrapper
│   ├── artifacts.py             # Model artifact loader
│   ├── key_store.py             # Persistent key registry (JSON)
│   ├── generation_store.py      # Generation record persistence (JSON)
│   ├── gpu_client.py            # HTTP client for GPU worker
│   ├── storage.py               # Image storage abstraction
│   └── static/
│       └── demo.html            # Demo web interface
│
├── gpu/                         # GPU Worker Service (CUDA)
│   ├── __init__.py              # GPU package exports
│   ├── main.py                  # FastAPI application entrypoint
│   ├── pipeline.py              # SD + watermark operations
│   ├── schemas.py               # Internal request/response models
│   ├── requirements.txt         # GPU-specific dependencies
│   └── Dockerfile               # Legacy Dockerfile location
│
├── docker/                      # Docker configurations
│   ├── api.Dockerfile           # API service container
│   └── gpu.Dockerfile           # GPU worker container
│
├── scripts/                     # Operational scripts
│   ├── smoke_test.sh            # End-to-end service test
│   └── deploy_gpu_gcp.sh        # GCP deployment helper
│
├── docker-compose.yml           # Full deployment config
└── docker-compose.stub.yml      # Override for stub/testing mode
```

---

## Component Details

### API Service (`/api`)

The API service is the **public-facing** component that handles all external requests.

#### Core Responsibilities:
- **Key Management**: Registration, validation, listing, revocation
- **Request Routing**: Delegates heavy work to GPU worker
- **Security Enforcement**: Master keys stay within API for generation; passed to GPU only for detection (required by g-value computation)
- **Storage Management**: Image persistence (local or GCS)
- **Generation Tracking**: Records all generations for audit
- **Health Monitoring**: Service status and GPU connectivity

#### Key Files:

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | FastAPI app creation | `create_app()`, `lifespan()` |
| `routes.py` | REST endpoints | `register_key()`, `generate_image()`, `detect_watermark()` |
| `schemas.py` | Data validation | Request/Response Pydantic models |
| `authority.py` | Key security | `derive_scoped_key()`, `get_generation_payload()`, `get_detection_payload()` |
| `key_store.py` | Key persistence | `register_key()`, `get_master_key()`, `revoke_key()` |
| `generation_store.py` | Generation records | `record_generation()`, `get_records_by_key()` |
| `gpu_client.py` | GPU communication | `generate()`, `detect()`, `health()` |
| `detector.py` | Detection logic | `detect_from_score()`, `StubDetector` |
| `storage.py` | Image storage | `LocalStorage`, `GCSStorage` |
| `artifacts.py` | Model artifacts | `load_likelihood_params()`, `load_mask()` |
| `config.py` | Configuration | `Config.from_env()`, `get_config()` |

---

### GPU Worker (`/gpu`)

The GPU worker handles **compute-intensive operations** that require GPU acceleration.

#### Core Responsibilities:
- **Image Generation**: Stable Diffusion with watermark embedding
- **DDIM Inversion**: Latent space recovery from images
- **G-Value Computation**: Watermark detection statistics
- **Model Management**: Loading/unloading SD models

#### Key Files:

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | FastAPI app for GPU | `generate()`, `reverse_ddim()`, `health()` |
| `pipeline.py` | SD + watermark ops | `GPUPipeline.generate()`, `invert_and_detect()` |
| `schemas.py` | Internal models | `GenerateRequest`, `ReverseDDIMResponse` |

#### Operating Modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Full Mode** | Uses actual SD pipeline, `SeedBiasStrategy`, `BayesianDetector` | Production with GPU |
| **Stub Mode** | Returns deterministic mock data based on input hashes | Testing, CI/CD, development |

Stub mode generates:
- Images: Colored noise based on prompt hash and seed
- Detection: Deterministic results based on image hash + key_id (70% chance of "detected")

---

### Docker Configuration (`/docker`)

#### `api.Dockerfile`
- **Base**: `python:3.10-slim`
- **Size**: ~200MB
- **Dependencies**: FastAPI, httpx, Pillow, numpy
- **Resources**: CPU only

#### `gpu.Dockerfile`
- **Base**: `nvidia/cuda:12.1-runtime-ubuntu22.04`
- **Size**: ~8GB (with models)
- **Dependencies**: PyTorch, diffusers, transformers
- **Resources**: NVIDIA GPU required

---

### Scripts (`/scripts`)

#### `smoke_test.sh`
End-to-end test that:
1. Checks service health
2. Registers a new key
3. Generates a watermarked image
4. Runs detection on a test image
5. Reports results

Usage:
```bash
# Basic run
./service/scripts/smoke_test.sh

# With verbose output
VERBOSE=true ./service/scripts/smoke_test.sh

# Custom API URL
API_URL=http://myserver:8000 ./service/scripts/smoke_test.sh
```

#### `deploy_gpu_gcp.sh`
Helper script for GCP deployment (Compute Engine with GPU).

---

## File-by-File Documentation

### `/api/main.py`
**FastAPI Application Entrypoint**

```python
def create_app() -> FastAPI:
    """Creates configured FastAPI application with:
    - CORS middleware
    - Lifespan management (startup/shutdown)
    - Route registration
    """
```

Key features:
- Lifespan context manager for clean startup/shutdown
- CORS configuration for web clients
- Debug mode support

---

### `/api/routes.py`
**REST API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/keys/register` | POST | Create new watermark key |
| `/keys` | GET | List all registered keys |
| `/generate` | POST | Generate watermarked image |
| `/detect` | POST | Detect watermark in image |
| `/health` | GET | Service health status |
| `/demo` | GET | Demo web interface |

Each endpoint includes:
- Pydantic validation
- Error handling with appropriate HTTP codes
- GPU worker fallback (stub mode)

---

### `/api/schemas.py`
**Pydantic Models for API**

**Public API Models:**
- `KeyRegisterRequest/Response` - Key registration
- `KeyInfo`, `KeyListResponse` - Key listing
- `GenerateRequest/Response` - Image generation (with inference params)
- `DetectRequest/Response` - Watermark detection (single normalized score + threshold)
- `HealthResponse` - Service health
- `ErrorResponse` - Standard error format

**Internal GPU Models:**
- `GPUGenerateRequest/Response` - GPU generation calls
- `GPUDetectRequest/Response` - GPU detection calls (includes master_key)
- `GPUHealthResponse` - GPU worker health

---

### `/api/config.py`
**Environment Configuration**

```python
@dataclass
class Config:
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # GPU Worker
    gpu_worker_url: str = "http://localhost:8001"
    gpu_worker_timeout: float = 120.0
    
    # Storage
    storage_backend: str = "local"  # "local" or "gcs"
    storage_path: str = "./data/images"
    gcs_bucket: Optional[str] = None
    
    # Key Store
    key_store_path: str = "./data/keys.json"
    
    # Artifacts (for detection)
    artifacts_path: str = "./data/artifacts"
    likelihood_params_path: Optional[str] = None
    mask_path: Optional[str] = None
    
    # Security
    encryption_key: str = "development-key-not-for-production"
```

---

### `/api/authority.py`
**Security & Key Derivation**

The Authority manages the **security boundary** between API and GPU:

```python
def derive_scoped_key(master_key, key_id, operation, request_id):
    """
    SECURITY: Creates operation-specific derived keys
    - Master key NEVER leaves API boundary (except for detection - see note)
    - Derived keys are scoped to generation OR detection
    - Uses HKDF-like construction with HMAC-SHA256
    """
```

**Note**: For detection, the `master_key` is now passed to the GPU worker because `compute_g_values()` requires it to match the training pipeline exactly.

Default configurations managed:
- `DEFAULT_EMBEDDING_CONFIG` - Watermark embedding parameters (lambda_strength, domain, freq cutoffs)
- `DEFAULT_G_FIELD_CONFIG` - G-field for detection (mapping_mode, frequency settings)
- `DEFAULT_DETECTION_CONFIG` - Bayesian detection settings (threshold, prior)
- `DEFAULT_INVERSION_CONFIG` - DDIM inversion parameters (num_inference_steps, guidance_scale)

---

### `/api/key_store.py`
**Persistent Key Registry**

Key record structure:
```json
{
  "key_id": "wm_abc123def4",
  "master_key": "64-char hex string (secret)",
  "fingerprint": "32-char hex (public)",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {},
  "is_active": true
}
```

Operations:
- `register_key()` - Creates new key with cryptographic randomness
- `get_master_key()` - Retrieves secret (internal use only)
- `revoke_key()` - Deactivates key
- `list_keys()` - Returns public key info (no secrets)

---

### `/api/generation_store.py`
**Generation Record Persistence**

Lightweight JSON-based storage for generation records:

```python
class GenerationStore:
    def record_generation(key_id, filename, seed_used, processing_time_ms):
        """Record a successful generation."""
    
    def get_records_by_key(key_id) -> List[Dict]:
        """Get all generation records for a key."""
    
    def count() -> int:
        """Get total number of generation records."""
```

Record structure:
```json
{
  "key_id": "wm_abc123def4",
  "timestamp": "2024-01-15T10:30:00Z",
  "filename": "20240115_103000_abc123.png",
  "seed_used": 42,
  "processing_time_ms": 1234.5
}
```

Features:
- Non-blocking and failure-tolerant (logs errors only)
- Stored alongside `keys.json` in the data directory
- Global instance via `get_generation_store()`

---

### `/api/gpu_client.py`
**HTTP Client for GPU Worker**

Async client using `httpx`:

```python
class GPUClient:
    async def generate(...) -> GPUGenerateResponse:
        """POST /infer/generate"""
    
    async def detect(...) -> GPUDetectResponse:
        """POST /infer/reverse_ddim"""
    
    async def health() -> GPUHealthResponse:
        """GET /health"""
    
    async def is_connected() -> bool:
        """Check GPU worker reachability"""
```

Error handling:
- `GPUClientConnectionError` - Network/connection issues
- `GPUClientTimeoutError` - Request timeout
- `GPUClientError` - HTTP/processing errors

---

### `/api/detector.py`
**Watermark Detection Logic**

```python
class Detector:
    def detect_from_score(score, n_elements) -> DetectionResult:
        """Bayesian inference from S-statistic"""
        # Computes posterior probability
        # Uses likelihood ratio from normal approximation
    
    def detect_from_gpu_response(gpu_response) -> DetectionResult:
        """Create DetectionResult from GPU worker response"""

class StubDetector(Detector):
    def detect_stub(key_id, simulate_watermarked) -> DetectionResult:
        """Generate deterministic stub result based on key_id hash"""
```

`StubDetector` provides deterministic mock results for testing when GPU worker is unavailable.

---

### `/api/storage.py`
**Image Storage Abstraction**

Interface:
```python
class StorageBackend(ABC):
    async def save_image(image_data, filename, content_type) -> str
    async def get_image(path) -> Optional[bytes]
    async def delete_image(path) -> bool
```

Implementations:
- `LocalStorage` - Filesystem storage with timestamp-based naming
- `GCSStorage` - Google Cloud Storage (stub implementation)

---

### `/api/artifacts.py`
**Model Artifact Loader**

Loads pre-computed artifacts for detection:
- `likelihood_params.json` - Bayesian likelihood parameters
- `mask.npy` - Detection region mask

Features:
- Lazy loading with caching
- Graceful degradation if artifacts missing

---

### `/gpu/main.py`
**GPU Worker FastAPI Application**

Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer/generate` | POST | Generate watermarked image (accepts derived_key only) |
| `/infer/reverse_ddim` | POST | DDIM inversion + detection (accepts master_key for g-values) |
| `/health` | GET | Detailed health metrics (internal - protect in production) |
| `/ready` | GET | Kubernetes readiness probe (safe for external probes) |
| `/live` | GET | Kubernetes liveness probe |

**Note**: The `/infer/reverse_ddim` endpoint now requires `master_key` to compute g-values that match the training pipeline exactly.

---

### `/gpu/pipeline.py`
**GPU Pipeline for Watermark Operations**

```python
class GPUPipeline:
    def generate(prompt, derived_key, key_id, ...) -> GenerationResult:
        """Generate image with embedded watermark using SeedBiasStrategy"""
    
    def invert_and_detect(image_bytes, derived_key, master_key, key_id, ...) -> DetectionResult:
        """DDIM inversion and watermark detection using BayesianDetector"""
```

Two modes:
- **Stub Mode**: Fast mock responses for testing (deterministic based on input hashes)
- **Full Mode**: Actual SD pipeline with diffusers, uses canonical detection from `/src`

Full mode detection flow:
1. Load image and create `DDIMInverter`
2. Invert to latent `z_T`
3. Compute g-values using `compute_g_values(master_key, key_id, ...)`
4. Apply mask and binarize
5. Run `BayesianDetector.score()` for log_odds and posterior

---

### `/gpu/schemas.py`
**Internal Request/Response Models**

Models for GPU-API communication:
- `GenerateRequest/Response` - Generation with embedding_config
- `ReverseDDIMRequest/Response` - Detection with master_key, g_field_config, detection_config, inversion_config
- `HealthResponse` - Detailed health with GPU memory stats
- `ReadyResponse` - Kubernetes readiness probe response
- `ErrorResponse` - Error format

---

## Data Flow

### Image Generation Flow

```
1. Client → POST /generate {key_id, prompt, seed, num_inference_steps, guidance_scale, width, height}
   
2. API validates key_id via KeyStore
   
3. Authority derives scoped key:
   derived_key = HKDF(master_key, "generation", key_id)
   
4. API → GPU Worker: POST /infer/generate
   {derived_key, key_id, prompt, seed, embedding_config, ...}
   
5. GPU Pipeline:
   a. Load/create SD pipeline
   b. Create SeedBiasStrategy with embedding_config
   c. Generate latents with watermark bias
   d. Decode to image
   
6. GPU → API: {image_base64, seed_used, processing_time_ms}
   
7. API saves image via Storage
   
8. API records generation via GenerationStore (non-blocking)
   
9. Client ← {image_url, key_id, seed_used, processing_time_ms}
```

### Detection Flow

```
1. Client → POST /detect {key_id, image}
   
2. API validates key_id
   
3. Authority prepares detection payload:
   - derived_key = HKDF(master_key, "detection", key_id)
   - master_key (required for compute_g_values() to match training)
   
4. API → GPU Worker: POST /infer/reverse_ddim
   {master_key, derived_key, image_base64, g_field_config, detection_config, inversion_config}
   
5. GPU Pipeline:
   a. DDIM inversion → recover latents (z_T)
   b. Compute G-values using master_key
   c. Run BayesianDetector → raw log_odds and posterior
   d. Normalize log-odds and apply calibrated threshold for detection
   
6. GPU → API: {detected, score, threshold, confidence, log_odds, posterior}
   
7. Client ← {detected, score, threshold, key_id, processing_time_ms}
```

---

## Security Model

### Key Hierarchy

```
Master Key (256-bit)
    │
    ├──[HKDF]──▶ Generation Derived Key
    │               └── Used only for watermark embedding
    │
    └──[HKDF]──▶ Detection Derived Key
                    └── Used only for G-field computation
```

### Security Properties

| Property | Implementation |
|----------|----------------|
| **Key Isolation** | Master keys stay within API boundary for generation; passed to GPU only for detection (required by `compute_g_values()`) |
| **Operation Scoping** | Derived keys are operation-specific |
| **Forward Secrecy** | Request ID included in key derivation |
| **Auditability** | Fingerprints enable tracing without exposing secrets |
| **Revocation** | Keys can be deactivated without deletion |
| **Internal Network** | GPU worker should only be accessible via internal network |

---

## Configuration Reference

### Environment Variables

#### API Service

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Port number |
| `API_DEBUG` | `false` | Enable debug mode |
| `GPU_WORKER_URL` | `http://localhost:8001` | GPU worker URL |
| `GPU_WORKER_TIMEOUT` | `120.0` | Request timeout (seconds) |
| `STORAGE_BACKEND` | `local` | `local` or `gcs` |
| `STORAGE_PATH` | `./data/images` | Local storage path |
| `GCS_BUCKET` | - | GCS bucket (if using gcs) |
| `KEY_STORE_PATH` | `./data/keys.json` | Key store location |
| `ARTIFACTS_PATH` | `./data/artifacts` | Artifacts directory |
| `LIKELIHOOD_PARAMS_PATH` | - | Path to trained likelihood params (required for Bayesian detection) |
| `MASK_PATH` | - | Path to detection mask |
| `ENCRYPTION_KEY` | `development-key-...` | Key encryption (change in prod!) |

#### GPU Worker

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_HOST` | `0.0.0.0` | Bind address |
| `GPU_PORT` | `8001` | Port number |
| `MODEL_ID` | `runwayml/stable-diffusion-v1-5` | HuggingFace model |
| `DEVICE` | `cuda` | PyTorch device |
| `STUB_MODE` | `true` | Use stub implementations |

---

## Quick Start

### Local Development (Stub Mode)

```bash
# Terminal 1: Start API service
cd service
python -m service.api.main

# Terminal 2: Start GPU worker (stub)
STUB_MODE=true python -m service.gpu.main
```

### Docker Compose

```bash
# With GPU
docker-compose -f service/docker-compose.yml up

# Without GPU (stub mode)
docker-compose -f service/docker-compose.yml -f service/docker-compose.stub.yml up
```

### Test the Service

```bash
# Run smoke test
./service/scripts/smoke_test.sh

# Or manually:
# 1. Register key
curl -X POST http://localhost:8000/keys/register -H "Content-Type: application/json" -d '{}'

# 2. Generate image
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"key_id": "wm_xxx", "prompt": "a cat", "seed": 42}'

# 3. Detect watermark
curl -X POST http://localhost:8000/detect -F "key_id=wm_xxx" -F "image=@image.png"
```

---

## API Documentation

When the service is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Demo UI**: http://localhost:8000/demo
