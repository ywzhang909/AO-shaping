"""CLI helper functions for AO-Shaping runners.

Common utilities used across multiple CLI runner scripts.
"""

from __future__ import annotations

import os
import re
import click
from datetime import datetime
from pathlib import Path


def get_debug_mode() -> bool:
    """从环境变量读取DEBUG模式

    Returns:
        bool: DEBUG环境变量为1/true/yes时返回True，否则返回False
    """
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    return debug_mode


def _get_wfs_res(res_str: str) -> type:
    """Convert WFS resolution string to :class:`MlaRes` enum value.

    Args:
        res_str: Resolution string, one of ``'320'``, ``'512'``,
            ``'768'``, ``'1024'``, ``'1280'``.

    Returns:
        The matching :class:`MlaRes` member; falls back to
        :attr:`MlaRes.Res1024` on unrecognised input.
    """
    from ao_shaping.drivers import MlaRes  # lazy import, avoid circular

    res_map = {
        "320": MlaRes.Res320,
        "512": MlaRes.Res512,
        "768": MlaRes.Res768,
        "1024": MlaRes.Res1024,
        "1280": MlaRes.Res1280,
    }
    return res_map.get(res_str, MlaRes.Res1024)


def parse_tuple(ctx, param, value):
    """Parse tuple format parameter supporting 'x,y' or '(x,y)' formats."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ["mass", "max", "shape"]:
        return value.lower()

    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception:
        raise click.BadParameter(
            f"Invalid center format: {value}. Expected formats: 'x,y' or '(x,y)'"
        )


def setup_coredumpy(directory: str = "logs/debug/error"):
    """Initialize coredumpy for exception debugging.
    
    Args:
        directory: Directory to save core dumps
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import coredumpy
        coredumpy.patch_except(directory=directory)
        return True
    except Exception:
        from loguru import logger
        logger.error("coredumpy initialization failed")
        return False


def get_date_dir_name() -> str:
    """Get current date as directory name (YYYYMMDD format)."""
    return datetime.now().strftime("%Y%m%d")


def get_timestamp_str() -> str:
    """Get current timestamp string (YYYYMMDD_HHMMSS format)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_save_dir(base_dir: str | Path, subdir: str) -> Path:
    """Create and return save directory with date subdirectory.
    
    Args:
        base_dir: Base directory path
        subdir: Subdirectory name
        
    Returns:
        Path object for the created directory
    """
    path = Path(base_dir) / subdir / get_date_dir_name()
    path.mkdir(parents=True, exist_ok=True)
    return path
