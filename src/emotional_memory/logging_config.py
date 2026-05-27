"""Standardised logging setup for the ``emotional_memory`` package.

Provides a single convenience entry-point so users and integrations don't
have to configure ``logging.basicConfig`` manually every time.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(
    level: str | int | None = None,
    *,
    json_format: bool = False,
) -> None:
    """Configure the root ``emotional_memory`` logger.

    Parameters
    ----------
    level:
        Log level.  When *None* (default) the value of the environment
        variable ``EMOTIONAL_MEMORY_LOG_LEVEL`` is used, falling back to
        ``WARNING``.
    json_format:
        When ``True``, emit single-line JSON objects with ``timestamp``,
        ``level``, ``logger``, and ``message`` fields.  Useful for
        production deployments that feed logs into structured ingestion
        pipelines.
    """
    if level is None:
        level = os.environ.get("EMOTIONAL_MEMORY_LOG_LEVEL", "WARNING")
    if isinstance(level, str):
        numeric_level: int = int(getattr(logging, level.upper(), logging.WARNING))
    else:
        numeric_level = level

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        import json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, default=str)

        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger = logging.getLogger("emotional_memory")
    logger.setLevel(numeric_level)
    # Avoid duplicate handlers if configure_logging is called twice.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(handler)
