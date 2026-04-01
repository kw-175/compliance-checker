"""
Structured logging setup for the picture compliance engine.
"""
# 中文说明：该文件负责 picture 模块的统一日志初始化。
# 把日志格式和第三方库噪音控制集中到这里，能避免入口文件里重复配置。
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the picture module."""
    # 中文说明：统一输出时间、等级、logger 名称和消息，便于排查跨模块问题。
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        stream=sys.stdout,
        force=True,
    )

    # 中文说明：这些库通常会输出较多底层日志，默认压到 WARNING 减少噪音。
    for name in ("PIL", "urllib3", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
