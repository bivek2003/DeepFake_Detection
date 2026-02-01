"""
Structured JSON logging configuration.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

from pythonjsonlogger import jsonlogger

from app.settings import get_settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add log level
        log_record["level"] = record.levelname

        # Add service name
        log_record["service"] = "deepfake-detection"

        # Add source location
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add extra fields
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id


def setup_logging() -> None:
    """Configure structured JSON logging."""
    settings = get_settings()

    # Create formatter
    formatter = CustomJsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "name": "logger"},
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stdout handler with JSON formatting
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Set log levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
