"""
GPU Worker Service for watermark inference.

This module provides a standalone FastAPI application for running
GPU-intensive inference operations on dedicated GPU workers.

Architecture:
- Stateless workers that can be horizontally scaled
- Model preloading at startup
- Health checks for load balancer integration
- Structured logging for observability

Run with:
    uvicorn service.worker.main:app --host 0.0.0.0 --port 8080
"""

