from fastapi import APIRouter
from google.cloud import pubsub_v1
import os
import json

router = APIRouter(prefix="/jobs", tags=["jobs"])

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "data-scraper-pipeline")
TOPIC_ID = "watermark-jobs"

_publisher = None
_topic_path = None


def get_publisher():
    global _publisher, _topic_path
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
        _topic_path = _publisher.topic_path(PROJECT_ID, TOPIC_ID)
    return _publisher, _topic_path


@router.post("/submit")
async def submit_job(payload: dict):
    """
    Submit a job to GPU worker via Pub/Sub.
    """
    publisher, topic_path = get_publisher()

    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(topic_path, data)
    message_id = future.result()

    return {
        "status": "submitted",
        "message_id": message_id,
    }
