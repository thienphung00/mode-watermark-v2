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
                                 │ (derived keys only - never master keys)
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
- **Security Enforcement**: Master keys never leave this boundary
- **Storage Management**: Image persistence (local or GCS)
- **Health Monitoring**: Service status and GPU connectivity

#### Key Files:

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | FastAPI app creation | `create_app()`, `lifespan()` |
| `routes.py` | REST endpoints | `register_key()`, `generate_image()`, `detect_watermark()` |
| `schemas.py` | Data validation | Request/Response Pydantic models |
| `authority.py` | Key security | `derive_scoped_key()`, `get_generation_payload()` |
| `key_store.py` | Key persistence | `register_key()`, `get_master_key()`, `revoke_key()` |
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
| **Full Mode** | Uses actual SD pipeline | Production with GPU |
| **Stub Mode** | Returns mock data | Testing, CI/CD, development |

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
- `GenerateRequest/Response` - Image generation
- `DetectRequest/Response` - Watermark detection
- `HealthResponse` - Service health

**Internal GPU Models:**
- `GPUGenerateRequest/Response` - GPU generation calls
- `GPUDetectRequest/Response` - GPU detection calls

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
    
    # Key Store
    key_store_path: str = "./data/keys.json"
    
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
    - Master key NEVER leaves API boundary
    - Derived keys are scoped to generation OR detection
    - Uses HKDF-like construction with HMAC-SHA256
    """
```

Default configurations managed:
- `DEFAULT_EMBEDDING_CONFIG` - Watermark embedding parameters
- `DEFAULT_G_FIELD_CONFIG` - G-field for detection
- `DEFAULT_DETECTION_CONFIG` - Bayesian detection settings

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
```

`StubDetector` provides deterministic mock results for testing.

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
| `/infer/generate` | POST | Generate watermarked image |
| `/infer/reverse_ddim` | POST | DDIM inversion + detection |
| `/health` | GET | Detailed health metrics |
| `/ready` | GET | Kubernetes readiness probe |
| `/live` | GET | Kubernetes liveness probe |

---

### `/gpu/pipeline.py`
**GPU Pipeline for Watermark Operations**

```python
class GPUPipeline:
    def generate(prompt, derived_key, ...) -> GenerationResult:
        """Generate image with embedded watermark"""
    
    def invert_and_detect(image_bytes, derived_key, ...) -> DetectionResult:
        """DDIM inversion and watermark detection"""
```

Two modes:
- **Stub Mode**: Fast mock responses for testing
- **Full Mode**: Actual SD pipeline with diffusers

---

### `/gpu/schemas.py`
**Internal Request/Response Models**

Models for GPU-API communication:
- `GenerateRequest/Response`
- `ReverseDDIMRequest/Response`
- `HealthResponse`, `ReadyResponse`

---

## Data Flow

### Image Generation Flow

```
1. Client → POST /generate {key_id, prompt, seed}
   
2. API validates key_id via KeyStore
   
3. Authority derives scoped key:
   derived_key = HKDF(master_key, "generation", key_id)
   
4. API → GPU Worker: POST /infer/generate
   {derived_key, prompt, seed, embedding_config}
   
5. GPU Pipeline:
   a. Load/create SD pipeline
   b. Generate latents with watermark bias
   c. Decode to image
   
6. GPU → API: {image_base64, seed_used}
   
7. API saves image via Storage
   
8. Client ← {image_url, key_id, seed_used}
```

### Detection Flow

```
1. Client → POST /detect {key_id, image}
   
2. API validates key_id
   
3. Authority derives detection key:
   derived_key = HKDF(master_key, "detection", key_id)
   
4. API → GPU Worker: POST /infer/reverse_ddim
   {derived_key, image_base64, g_field_config, detection_config}
   
5. GPU Pipeline:
   a. DDIM inversion → recover latents
   b. Compute G-values
   c. Statistical test → score
   d. Bayesian inference → posterior
   
6. GPU → API: {detected, score, confidence, posterior}
   
7. Client ← {detected, confidence, score}
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
| **Key Isolation** | Master keys never leave API boundary |
| **Operation Scoping** | Derived keys are operation-specific |
| **Forward Secrecy** | Request ID included in key derivation |
| **Auditability** | Fingerprints enable tracing without exposing secrets |
| **Revocation** | Keys can be deactivated without deletion |

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
