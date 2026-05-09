"""Logging setup using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str | Path = "outputs/logs", level: str = "INFO") -> None:
    """Configure loguru with console and file sinks.

    Args:
        log_dir: Directory for log files.
        level: Minimum log level.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True,
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        Path(log_dir) / "train_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
    )


def get_logger(name: str):
    """Get a logger with a bound module name."""
    return logger.bind(name=name)
