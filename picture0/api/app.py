"""
FastAPI application entry point for the picture compliance engine.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from picture.api.routes import router as picture_router
from picture.infra.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    setup_logging()
    logger.info("Picture compliance service starting")
    yield
    logger.info("Picture compliance service stopping")


app = FastAPI(
    title="Picture Data Compliance Checker",
    description=(
        "Image compliance processing engine with OCR, PII detection, "
        "safety moderation, vision detection, and configurable policy evaluation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register picture compliance routes
app.include_router(picture_router)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "picture-compliance-checker",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn
    from picture.infra.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "picture.api.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
