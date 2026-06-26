"""Logging helpers for the credit risk project."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a simple configured logger."""
    logger = logging.getLogger(name)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    return logger


if __name__ == "__main__":
    demo_logger = get_logger(__name__)
    demo_logger.info("Logger is ready.")
