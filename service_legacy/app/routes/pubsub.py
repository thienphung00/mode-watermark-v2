import os
import json
import base64
import asyncio
from fastapi import APIRouter, Request, HTTPException, Header

router = APIRouter()

# Simple shared secret
PUBSUB_SECRET = os.getenv("PUBSUB_SECRET", None)

# GPU safety: allow only 1 job at a time
gpu_semaphore = asyncio.Semaphore(1)


def decode_pubsub_message(body: dict) -> dict:
    try:
        message = body["message"]["data"]
        decoded = base64.b64decode(message).decode("utf-8")
        return json.loads(decoded)
    except Exception as e:
        raise ValueError(f"Invalid PubSub payload: {e}")


@router.post("/pubsub/job")
async def handle_pubsub_job(
    request: Request,
    x_pubsub_secret: str | None = Header(default=None),
):
    # --- Security check ---
    if PUBSUB_SECRET:
        if x_pubsub_secret != PUBSUB_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()

    try:
        payload = decode_pubsub_message(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    print("📩 PubSub job received:", payload)

    # --- GPU protected execution ---
    async with gpu_semaphore:
        # TODO: wire this into your real generation service
        # Example placeholder:
        prompt = payload.get("prompt", "test prompt")
        steps = payload.get("steps", 20)

        print(f"🚀 Running generation: prompt={prompt}, steps={steps}")

        # You will plug in:
        # result = await generation_service.generate(...)
        await asyncio.sleep(1)  # placeholder

    return {"status": "ok"}
