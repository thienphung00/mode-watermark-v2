# Service Layer Architecture Summary

## Overview

The `service/` directory implements a Phase-1 watermarking API that provides:
- **Generation**: Hosted Stable Diffusion image generation with seed-bias watermarking
- **Detection**: Bayesian-only watermark detection

The architecture strictly separates concerns:
- **FastAPI layer** (`app/`): Request/response handling, dependency injection, middleware
- **Service layer** (`service/`): Business logic, watermark authority, orchestration
- **Research layer** (`src/`): Embedding strategies, Bayesian detection, diffusion internals

**Critical constraint**: FastAPI routes never directly construct or reference research objects.

---

## Directory Structure

```
service/
├── app/                          # FastAPI application layer
│   ├── main.py                   # FastAPI app entrypoint
│   ├── dependencies.py           # Dependency injection providers
│   ├── schemas.py                 # Product-level Pydantic schemas
│   ├── middleware.py              # Rate limiting, CORS, etc.
│   └── routes/                    # API route handlers
│       ├── generate.py            # POST /generate (hosted generation)
│       ├── detect.py              # POST /detect (Bayesian detection)
│       └── health.py              # Health check endpoint
│
├── authority.py                   # WatermarkAuthorityService (root of trust)
├── detection.py                   # DetectionService (Bayesian-only)
├── generation/                    # Generation adapters
│   ├── base.py                    # GenerationAdapter interface
│   └── stable_diffusion.py       # StableDiffusionSeedBiasAdapter (Phase-1)
│
└── infra/                         # Infrastructure
    ├── db.py                      # Watermark key database
    └── security.py                # Key generation, encryption
```

---

## Core Services

### 1. WatermarkAuthorityService (`authority.py`)

**Purpose**: Cryptographic and statistical authority of the system.

**Responsibilities**:
- Owns `master_key` (never exposed to clients)
- Manages `key_id` → watermark policy mapping
- Decides embedding configuration (seed-bias parameters)
- Decides detection configuration (Bayesian parameters)
- Produces watermark payloads for generation
- Produces detection configuration for detection

**Key Methods**:
- `create_watermark_policy()`: Creates new watermark with master_key and key_id
- `get_watermark_payload(key_id)`: Returns configuration for generation (includes master_key internally)
- `get_detection_config(key_id)`: Returns configuration for detection (includes master_key internally)
- `revoke_watermark(key_id)`: Revokes a watermark

**Design Decisions**:
- Master key is stored encrypted in database (`infra/db.py`)
- Policy configuration is currently hardcoded (TODO: store in DB for versioning)
- All watermark decisions flow from this service (single source of truth)

---

### 2. GenerationAdapter (`generation/base.py`, `generation/stable_diffusion.py`)

**Purpose**: Abstract adapter for image generation with watermarking.

**Interface** (`GenerationAdapter`):
- `generate(prompt, watermark_payload, ...)`: Generate watermarked image
- `get_model_info()`: Get model information

**Implementation** (`StableDiffusionSeedBiasAdapter`):
- Phase-1: Hosted generation using Stable Diffusion
- Applies seed-bias watermarking at z_T (initial latent)
- Uses `src.engine.pipeline.generate_with_watermark()` internally
- Creates `SeedBiasStrategy` from watermark payload

**Design Decisions**:
- Adapter pattern isolates SD-specific assumptions from API layer
- FastAPI routes don't know SD exists (they only know `GenerationAdapter`)
- TODO (Phase-2): This adapter can be removed when moving to client-side generation

**Flow**:
1. Receives `watermark_payload` from `WatermarkAuthorityService`
2. Extracts `master_key`, `key_id`, `embedding_config`
3. Creates `SeedBiasStrategy` with seed-bias parameters
4. Calls `generate_with_watermark()` from research layer
5. Returns image + generation metadata

---

### 3. DetectionService (`detection.py`)

**Purpose**: Bayesian-only watermark detection.

**Responsibilities**:
- Fetches detection configuration from `WatermarkAuthorityService`
- Instantiates `BayesianDetector` (never `HybridDetector`)
- Computes g-values from image using DDIM inversion
- Runs Bayesian detection
- Returns product-level results (no research internals)

**Key Methods**:
- `detect(image, key_id, ...)`: Detect watermark in image

**Detection Pipeline**:
1. Get detection config from authority (includes `master_key`, `g_field_config`)
2. Encode image to z_0 (VAE encoding)
3. Invert to z_T (DDIM inversion to initial latent)
4. Compute g-values from z_T using `compute_g_values()` (canonical function)
5. Instantiate `BayesianDetector` with likelihood parameters
6. Run `detector.score(g, mask)` to get posterior probability
7. Return `{detected, score, confidence, policy_version}`

**Design Decisions**:
- **Bayesian-only**: Never uses `HybridDetector`
- **No research internals exposed**: Returns product-level results only
- **G-field config must match generation**: Ensures consistency via authority service
- Uses canonical `compute_g_values()` function (same as generation)

---

## API Layer (`app/`)

### Dependencies (`app/dependencies.py`)

**Purpose**: Dependency injection providers for FastAPI routes.

**Providers**:
- `get_watermark_authority()` → `WatermarkAuthorityService` (singleton)
- `get_generation_adapter()` → `GenerationAdapter` (singleton, Phase-1: SD only)
- `get_detection_service()` → `DetectionService` (singleton)
- `get_pipeline()` → `StableDiffusionPipeline` (internal use only)

**Design Decisions**:
- No research-layer instantiation in dependencies
- No SD-specific logic in dependencies
- Services are singletons (lazy-loaded)

---

### Routes

#### POST `/api/v1/generate` (`routes/generate.py`)

**Purpose**: Generate watermarked image (Phase-1: hosted generation).

**Flow**:
1. Get `WatermarkAuthorityService` and `GenerationAdapter` from dependencies
2. If `key_id` provided: get existing watermark payload
3. If `key_id` not provided: create new watermark policy
4. Call `adapter.generate(prompt, watermark_payload, ...)`
5. Encode image to base64
6. Return `GenerateResponse` with image, key_id, metadata, watermark_version

**Request Schema**:
```python
{
    "prompt": str,
    "key_id": Optional[str],  # If None, creates new watermark
    "num_inference_steps": Optional[int],
    "guidance_scale": Optional[float],
    "seed": Optional[int],
    "height": Optional[int],
    "width": Optional[int]
}
```

**Response Schema**:
```python
{
    "image_base64": str,
    "key_id": str,
    "generation_metadata": {
        "seed": Optional[int],
        "num_inference_steps": int,
        "guidance_scale": float,
        "model_version": str,
        "height": Optional[int],
        "width": Optional[int]
    },
    "watermark_version": str
}
```

**Design Decisions**:
- Thin route handler (delegates to services)
- No research objects in route
- Product-level schemas only

---

#### POST `/api/v1/detect` (`routes/detect.py`)

**Purpose**: Detect watermark in image (Bayesian-only).

**Flow**:
1. Get `DetectionService` from dependencies
2. Decode base64 image
3. Call `detection_service.detect(image, key_id, ...)`
4. Return `DetectResponse` with detected, score, confidence, policy_version

**Request Schema**:
```python
{
    "image": str,  # Base64-encoded image
    "key_id": str,
    "num_inference_steps": Optional[int],
    "prompt": Optional[str],  # For accurate inversion
    "guidance_scale": Optional[float]  # For accurate inversion
}
```

**Response Schema**:
```python
{
    "detected": bool,
    "score": float,  # Log-odds (higher = more confident)
    "confidence": float,  # Posterior probability (0-1)
    "policy_version": str
}
```

**Design Decisions**:
- Never exposes detector class names
- Never exposes research internals (g-values, z_T, etc.)
- Returns product-level results only

---

### Schemas (`app/schemas.py`)

**Purpose**: Product-level Pydantic schemas (no research internals).

**Key Schemas**:
- `GenerateRequest` / `GenerateResponse`
- `DetectRequest` / `DetectResponse`
- `GenerationMetadata`

**Design Decisions**:
- No mention of: `HybridDetector`, `BayesianDetector`, `StableDiffusion`, `z_T`, `diffusion steps`
- Product-level terminology only
- Clear separation from research layer

---

## Data Flow

### Generation Flow

```
Client Request
    ↓
POST /api/v1/generate
    ↓
routes/generate.py
    ↓
get_watermark_authority() → WatermarkAuthorityService
    ├─ create_watermark_policy() OR get_watermark_payload(key_id)
    └─ Returns: {key_id, master_key, embedding_config, watermark_version}
    ↓
get_generation_adapter() → StableDiffusionSeedBiasAdapter
    ├─ adapter.generate(prompt, watermark_payload, ...)
    ├─ Creates SeedBiasStrategy from payload
    ├─ Calls src.engine.pipeline.generate_with_watermark()
    └─ Returns: {image, generation_metadata, watermark_version}
    ↓
Encode image to base64
    ↓
GenerateResponse (product-level schema)
```

**Key Points**:
- `master_key` never leaves service layer
- FastAPI route never sees research objects
- Adapter isolates SD-specific logic

---

### Detection Flow

```
Client Request
    ↓
POST /api/v1/detect
    ↓
routes/detect.py
    ↓
get_detection_service() → DetectionService
    ├─ authority.get_detection_config(key_id)
    │   └─ Returns: {master_key, detection_config, g_field_config, watermark_version}
    ├─ Invert image: Image → z_0 → z_T
    ├─ compute_g_values(z_T, key_id, master_key, g_field_config)
    ├─ Instantiate BayesianDetector
    ├─ detector.score(g, mask) → posterior probability
    └─ Returns: {detected, score, confidence, policy_version}
    ↓
DetectResponse (product-level schema)
```

**Key Points**:
- `master_key` never leaves service layer
- Only `BayesianDetector` is used (never `HybridDetector`)
- G-field config matches generation (ensured by authority)
- No research internals exposed to API

---

## Design Principles

### 1. Watermark Authority (Root of Trust)

- **API owns**: `master_key`, `key_id`, watermark policy
- **Clients never see**: `master_key`
- **All decisions flow from**: `WatermarkAuthorityService`

### 2. Explicit Phase-1 Scope

- **Stable Diffusion only**: Seed-bias watermarking only
- **Model dependence is explicit**: Isolated in `StableDiffusionSeedBiasAdapter`
- **No false model-agnosticism**: Phase-1 is explicitly SD-specific

### 3. Strict Separation of Concerns

- **FastAPI layer**: Request/response, dependency injection, auth/middleware
- **Service layer**: Watermark authority, generation orchestration, detection orchestration
- **Research layer** (`src/`): Embedding strategies, Bayesian detection, diffusion internals
- **FastAPI must never**: Directly construct or reference research objects

### 4. Adapter-Based Generation

- **GenerationAdapter abstraction**: Decouples API from generation method
- **SD-specific assumptions**: Contained in `StableDiffusionSeedBiasAdapter`
- **FastAPI routes**: Don't know SD exists

### 5. Bayesian Detection Only

- **HybridDetector is out of scope**: Must not be used
- **BayesianDetector is the only method**: Statistical, defensible, calibration-friendly
- **Detector choice is policy decision**: Never a route decision

---

## Phase-1 vs Phase-2

### Phase-1 (Current)

- **Hosted generation**: API performs image generation using Stable Diffusion
- **Seed-bias watermarking**: Applied at z_T (initial latent)
- **Bayesian detection**: Only detection method
- **Stable Diffusion only**: Explicitly SD-specific

### Phase-2 (Future)

**TODO markers in code**:
- `service/generation/stable_diffusion.py`: "This adapter can be removed when moving to client-side generation"
- `service/app/routes/generate.py`: "This endpoint will be deprecated when moving to client-side generation"
- `service/app/dependencies.py`: "This adapter can be removed when moving to client-side generation"

**Phase-2 Architecture**:
- **Client-side generation**: Clients generate images themselves
- **Credential-only issuance**: API only issues watermark credentials (key_id + embedding config)
- **Remove GenerationAdapter**: No hosted generation needed
- **Keep**: `WatermarkAuthorityService` + `DetectionService` (still needed for detection)

**Migration Path**:
1. Keep `GenerationAdapter` for backward compatibility (deprecated)
2. Add credential issuance endpoint (returns watermark payload without generation)
3. Remove `GenerationAdapter` when all clients migrated
4. Keep detection service (unchanged)

---

## Security Considerations

### Key Management

- **Master keys**: Stored encrypted in database (`infra/db.py`)
- **Key encryption**: Uses Fernet (symmetric encryption) via `infra/security.py`
- **Key derivation**: PRF-based (ChaCha20) for deterministic key derivation
- **Never exposed**: `master_key` never returned to clients

### Watermark Policy

- **Policy versioning**: Currently hardcoded (TODO: store in DB)
- **Revocation**: Watermarks can be revoked via `authority.revoke_watermark()`
- **Active check**: Detection/generation checks if watermark is active

---

## Configuration

### Default Values

**Embedding Config** (seed-bias):
- `lambda_strength`: 0.05
- `domain`: "frequency"
- `low_freq_cutoff`: 0.05
- `high_freq_cutoff`: 0.4

**Detection Config** (Bayesian):
- `detector_type`: "bayesian"
- `threshold`: 0.5
- `prior_watermarked`: 0.5
- `likelihood_params_path`: None (TODO: Load from calibration)

**G-Field Config** (must match generation):
- `mapping_mode`: "binary"
- `domain`: "frequency"
- `frequency_mode`: "bandpass"
- `low_freq_cutoff`: 0.05
- `high_freq_cutoff`: 0.4
- `normalize_zero_mean`: True
- `normalize_unit_variance`: True

---

## Error Handling

### Generation Errors

- **Watermark not found**: 404 if `key_id` provided but not found
- **Watermark revoked**: 404 if watermark is revoked
- **Generation failure**: 500 with error message

### Detection Errors

- **Watermark not found**: 404 if `key_id` not found
- **Watermark revoked**: 404 if watermark is revoked
- **Image decode failure**: 400 if base64 decode fails
- **Detection failure**: 500 with error message

---

## Testing Considerations

### Unit Tests

- **Service layer**: Test `WatermarkAuthorityService`, `DetectionService`, `GenerationAdapter` in isolation
- **Mock dependencies**: Mock database, pipeline, etc.

### Integration Tests

- **End-to-end**: Test full generation → detection flow
- **Key management**: Test watermark creation, retrieval, revocation
- **Error cases**: Test watermark not found, revoked, etc.

### Research Layer Tests

- **Separate from service**: Research layer (`src/`) has its own tests
- **Service layer doesn't test research**: Service layer tests focus on orchestration

---

## Future Improvements

### Short-term

1. **Policy versioning**: Store policy configuration in database
2. **Calibration loading**: Load likelihood parameters from calibration files
3. **Model version extraction**: Extract actual model version from pipeline

### Long-term (Phase-2)

1. **Client-side generation**: Remove hosted generation
2. **Credential issuance**: Add endpoint for watermark credential issuance
3. **Remove GenerationAdapter**: When all clients migrated to client-side generation

---

## Summary

The service layer implements a clean, maintainable architecture that:

✅ **Strictly separates concerns**: FastAPI, service, and research layers
✅ **Isolates SD-specific assumptions**: Contained in adapter
✅ **Uses Bayesian detection only**: No HybridDetector
✅ **Maintains watermark authority**: Single source of truth for keys and policy
✅ **Future-proofs for Phase-2**: Clear migration path to client-side generation
✅ **Never exposes research internals**: Product-level schemas only

The architecture is designed to be:
- **Maintainable**: Clear separation of concerns
- **Testable**: Services can be tested in isolation
- **Extensible**: Easy to add new generation methods or detection strategies
- **Secure**: Master keys never exposed, encrypted storage
- **Future-proof**: Clear path to Phase-2 client-side generation

