"""Utility functions for AO-Shaping system.

The ``utils`` package is organized into domain sub-packages:

- :mod:`ao_shaping.utils.image` — spot analysis, beam metrics, target masks.
- :mod:`ao_shaping.utils.optics` — Zernike basis, wavefront, phase unwrapping,
  pattern generation, matrix helpers.
- :mod:`ao_shaping.utils.slm` — SLM phase patterns and LUT handling.
- :mod:`ao_shaping.utils.hardware` — device config, network, camera helpers.
- :mod:`ao_shaping.utils.filesystem` — file paths, timestamps, error handlers.
- :mod:`ao_shaping.utils.cli` — argument parsing, coredumpy setup.
- :mod:`ao_shaping.utils.viz` — Pygame visualization displays and plotting.

Import from the deep path (e.g. ``from ao_shaping.utils.image.spots_calc
import centroid``).  Only :data:`logger` and :func:`configure_error_logging`
live in this package ``__init__``.
"""

from loguru import logger

__all__ = ["logger", "configure_error_logging"]


def configure_error_logging(
    log_file: str = "logs/error.log",
    rotation: str = "500 MB",
    level: str = "ERROR",
) -> int:
    """Configure error logging to file.

    This function should be called explicitly if error logging is needed.
    It is NOT called automatically on import to avoid side effects.

    Args:
        log_file: Path to log file.
        rotation: Log rotation policy.
        level: Log level.

    Returns:
        Handler ID for removal.
    """
    return logger.add(
        log_file,
        rotation=rotation,
        encoding="utf-8",
        level=level,
        backtrace=True,
        diagnose=True,
    )