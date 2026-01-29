# Watermarking Service

FastAPI service layer for the watermarking engine. This service provides a custodial API for watermark generation and detection.

## Architecture

The service layer wraps the existing watermarking engine in `src/` without modifying it:

- **Key Custody**: Secret keys are generated and stored internally
- **Minimal API**: Only exposes necessary endpoints
- **Existing Engine**: Uses all existing strategies, detection pipelines, and algorithms from `src/`

## API Endpoints

### POST `/api/v1/generate_seed`

Generate a watermark seed for client-side image generation.

**Request:**
```json
{
  "model": "sdxl",
  "num_images": 4
}
```

**Response:**
```json
{
  "watermark_id": "wm_xxxxx",
  "seed_payload": "<opaque blob>"
}
```

The `seed_payload` contains:
- Random seed for noise generation
- Key identifier (watermark_id)
- Pre-computed watermarked initial latent (optional)

### POST `/api/v1/detect`

Detect watermark in provided images.

**Request:**
```json
{
  "watermark_id": "wm_xxxxx",
  "images": ["<base64-encoded-image-1>", "<base64-encoded-image-2>"]
}
```

**Response:**
```json
{
  "detected": true,
  "score": 0.65,
  "n_eff": 2,
  "confidence": 0.85
}
```

### GET `/api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Running the Service

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn[standard] cryptography python-multipart

# Run service
uvicorn service.app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t watermarking-service -f service/Dockerfile .

# Run container
docker run -p 8000:8000 watermarking-service
```

## Configuration

Runtime configuration is in `service/configs/runtime.yaml`. Key settings:

- `model.default_model_id`: Stable Diffusion model to use
- `seed_bias.lambda_strength`: Watermark injection strength
- `detection.threshold_high`: Detection threshold
- `storage.key_db_path`: Path to key storage file

## Security Notes

1. **Key Storage**: Keys are encrypted at rest using Fernet (symmetric encryption)
2. **Key Generation**: Uses cryptographically secure random number generation
3. **Rate Limiting**: `/detect` endpoint is rate-limited (50 requests/minute per IP)
4. **Production**: Set `WATERMARK_ENCRYPTION_KEY` environment variable for key encryption

## Key Storage

Watermark keys are stored in `service_data/watermark_keys.json` (configurable). Each record contains:

- `watermark_id`: Public identifier
- `secret_key_encrypted`: Encrypted master key
- `strategy`: Strategy type (e.g., "seed_bias")
- `model`: Model identifier
- `status`: "active" or "revoked"
- `created_at`: Timestamp

## Integration with Existing Engine

The service uses existing modules from `src/`:

- `src.engine.strategies.seed_bias.SeedBiasStrategy`: For watermark generation
- `src.detection.pipeline.HybridDetector`: For watermark detection
- `src.detection.prf.PRFKeyDerivation`: For key derivation
- `src.algorithms.g_field.GFieldGenerator`: For G-field generation

**Important**: The service layer does NOT modify any code in `src/`. All watermark logic comes from the existing engine.

## Development

### Project Structure

```
service/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── routes/              # API routes
│   │   ├── generate_seed.py
│   │   ├── detect.py
│   │   └── health.py
│   ├── schemas.py           # Pydantic schemas
│   ├── dependencies.py      # Shared dependencies
│   └── middleware.py        # Rate limiting
├── infra/
│   ├── db.py                # Key storage
│   └── security.py          # Key encryption
├── configs/
│   └── runtime.yaml         # Runtime config
├── Dockerfile
└── README.md
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test generate_seed
curl -X POST http://localhost:8000/api/v1/generate_seed \
  -H "Content-Type: application/json" \
  -d '{"model": "sdxl", "num_images": 1}'
```

## Notes

- The service is stateless except for key storage
- Pipeline is loaded once at startup (singleton)
- Detection uses the existing hybrid detection pipeline
- Rate limiting is applied to `/detect` endpoint only

