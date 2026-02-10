# Watermark Service

GPU-backed watermarking service for image generation and detection.

## Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│                 │ ───────────▶  │                 │
│   API Service   │               │   GPU Worker    │
│   (CPU, Public) │  ◀───────────  │   (GPU, Private)│
│                 │               │                 │
└─────────────────┘               └─────────────────┘
        │                                  │
        │                                  │
        ▼                                  ▼
┌─────────────────┐               ┌─────────────────┐
│  Key Store      │               │  SD Pipeline    │
│  (JSON file)    │               │  + Watermark    │
└─────────────────┘               └─────────────────┘
```

### API Service (Port 8000)
- **Public-facing** endpoints for users
- Key registration and management
- Delegates heavy computation to GPU worker
- Handles storage and business logic

### GPU Worker (Port 8001)
- **Internal only** - not user-facing
- Image generation with watermark embedding
- DDIM inversion for detection
- Runs on GPU-enabled hardware

## Quick Start

### Local Development (Stub Mode)

Run without GPU using stub implementations:

```bash
# Start API service
cd service
python -m service.api.main

# In another terminal, start GPU worker (stub mode)
STUB_MODE=true python -m service.gpu.main
```

### Docker Compose

```bash
# With GPU
docker-compose -f service/docker-compose.yml up

# Without GPU (stub mode)
docker-compose -f service/docker-compose.yml -f service/docker-compose.stub.yml up
```

## API Endpoints

### Key Management

#### Register a new key
```bash
curl -X POST http://localhost:8000/keys/register \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{
  "key_id": "wm_abc123def4",
  "fingerprint": "a1b2c3d4e5f6...",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List all keys
```bash
curl http://localhost:8000/keys
```

### Image Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "key_id": "wm_abc123def4",
    "prompt": "a beautiful mountain landscape",
    "seed": 42
  }'
```

Response:
```json
{
  "image_url": "./data/images/img_20240115_103000_abc12345.png",
  "key_id": "wm_abc123def4",
  "seed_used": 42,
  "processing_time_ms": 5234.5
}
```

### Watermark Detection

```bash
# Using file upload
curl -X POST http://localhost:8000/detect \
  -F "key_id=wm_abc123def4" \
  -F "image=@my_image.png"

# Using base64
curl -X POST http://localhost:8000/detect \
  -F "key_id=wm_abc123def4" \
  -F "image_base64=$(base64 -i my_image.png)"
```

Response:
```json
{
  "detected": true,
  "key_id": "wm_abc123def4",
  "score": 1.8,
  "threshold": 0.0,
  "processing_time_ms": 2341.2
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_DEBUG` | `false` | Enable debug mode |
| `GPU_WORKER_URL` | `http://localhost:8001` | GPU worker URL |
| `GPU_WORKER_TIMEOUT` | `120.0` | GPU request timeout (seconds) |
| `STORAGE_BACKEND` | `local` | Storage backend: `local` or `gcs` |
| `STORAGE_PATH` | `./data/images` | Local storage path |
| `GCS_BUCKET` | - | GCS bucket name (if using GCS) |
| `KEY_STORE_PATH` | `./data/keys.json` | Key store file path |
| `LIKELIHOOD_PARAMS_PATH` | `./experiments/likelihood_models_50/best_model` | Path to likelihood model (for Bayesian detection) |
| `ENCRYPTION_KEY` | - | Key encryption password |

### GPU Worker Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_HOST` | `0.0.0.0` | GPU worker bind address |
| `GPU_PORT` | `8001` | GPU worker port |
| `MODEL_ID` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `DEVICE` | `cuda` | PyTorch device |
| `STUB_MODE` | `true` | Use stub implementations |

## Testing

### Smoke Test

```bash
./service/scripts/smoke_test.sh

# With verbose output
VERBOSE=true ./service/scripts/smoke_test.sh

# Custom API URL
API_URL=http://localhost:8000 ./service/scripts/smoke_test.sh
```

### Manual Testing

```bash
# 1. Register key
KEY_ID=$(curl -s -X POST http://localhost:8000/keys/register | jq -r '.key_id')
echo "Key: $KEY_ID"

# 2. Generate image
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"key_id\": \"$KEY_ID\", \"prompt\": \"test image\"}"

# 3. Detect watermark
curl -X POST http://localhost:8000/detect \
  -F "key_id=$KEY_ID" \
  -F "image=@test.png"
```

## Directory Structure

```
service/
├── api/                     # Public API Service
│   ├── main.py              # FastAPI entrypoint
│   ├── routes.py            # API endpoints
│   ├── schemas.py           # Pydantic models
│   ├── config.py            # Configuration
│   ├── authority.py         # Key management
│   ├── detector.py          # Detection logic
│   ├── artifacts.py         # Artifact loading
│   ├── key_store.py         # Key persistence
│   ├── gpu_client.py        # GPU worker client
│   ├── storage.py           # Storage abstraction
│   └── static/demo.html     # Demo UI
│
├── gpu/                     # GPU Worker (Private)
│   ├── main.py              # FastAPI entrypoint
│   ├── pipeline.py          # SD + watermark ops
│   ├── schemas.py           # Request models
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker/
│   ├── api.Dockerfile
│   └── gpu.Dockerfile
│
├── scripts/
│   ├── smoke_test.sh
│   └── deploy_gpu_gcp.sh
│
├── data/
│   ├── keys.json            # Key persistence
│   └── artifacts/           # Model artifacts
│
├── docker-compose.yml
├── docker-compose.stub.yml
├── .env.example
└── README.md
```

## Security Notes

1. **Master keys** never leave the API service boundary
2. GPU workers only receive **derived keys** scoped to specific operations
3. Key fingerprints are used for cache keying and audit trails
4. The encryption key should be set via environment variable in production

## Demo UI

A simple demo UI is available at:
```
http://localhost:8000/demo
```

Features:
- Register new keys
- Generate watermarked images
- Upload and detect watermarks
- Health status monitoring

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
