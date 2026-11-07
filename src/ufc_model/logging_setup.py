"""Logging configuration shared by all entry points."""

from __future__ import annotations

import logging


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
