"""
FastAPI application entry point for the picture compliance engine.
"""
# 中文说明：本文件负责创建 FastAPI 应用、注册生命周期钩子，并挂载 picture 路由。

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
    # 中文说明：启动阶段先初始化统一日志，这样后续各层日志格式保持一致。
    setup_logging()
    logger.info("Picture compliance service starting")
    # 中文说明：yield 之前的代码属于启动阶段，yield 之后的代码属于关闭阶段。
    yield
    logger.info("Picture compliance service stopping")


# 中文说明：这里创建整个 picture 模块对外暴露的 FastAPI 应用对象。
# title / description / version 会直接出现在 OpenAPI 文档中。
app = FastAPI(
    title="Picture Data Compliance Checker",
    description=(
        "Image compliance processing engine with OCR, PII detection, "
        "safety moderation, vision detection, and configurable policy evaluation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# 中文说明：统一开启 CORS，方便前端和调试环境跨域访问。
# 当前配置较宽松，更偏向开发或内网环境；若生产上线通常需要收紧 allow_origins。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 中文说明：把 picture 模块的所有业务路由挂载到主应用上。
app.include_router(picture_router)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    # 中文说明：这里只返回最基础的存活信息。
    # 如果后续要增强健康检查，可以把 provider 状态、模型状态和存储状态也加入进来。
    return {
        "status": "healthy",
        "service": "picture-compliance-checker",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn
    from picture.infra.config import get_settings

    # 中文说明：直接运行本文件时按配置启动开发服务。
    # 真正生产部署时通常由 uvicorn / gunicorn / 容器入口接管。
    settings = get_settings()
    uvicorn.run(
        "picture.api.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
