"""CLI helper functions for AO-Shaping runners.

Common utilities used across multiple CLI runner scripts.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import click


def get_debug_mode() -> bool:
    """从环境变量读取DEBUG模式

    Returns:
        bool: DEBUG环境变量为1/true/yes时返回True，否则返回False
    """
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    return debug_mode


def resolve_debug(ctx, flag: bool | None = None) -> bool:
    """Resolve debug mode from any of: group ``--debug``, command ``--debug``, ``DEBUG`` env.

    Enabling debug by *any* of the three means debug on. The group flag is read
    from ``ctx.parent.obj`` (set by ``main.py``); it is absent when a runner is
    invoked standalone (``python -m ao_shaping.runners...``), in which case the
    command flag / env var still work.

    Args:
        ctx: click context of the command (``ctx.parent`` is the ``main.py`` group).
        flag: value of the command's own ``--debug`` option (``None`` = not passed).

    Returns:
        True when debug mode should be enabled.
    """
    group_flag = False
    parent = getattr(ctx, "parent", None) if ctx is not None else None
    if parent is not None and getattr(parent, "obj", None):
        group_flag = bool(parent.obj.get("debug", False))
    return group_flag or get_debug_mode() or bool(flag)


def parse_tuple(ctx, param, value):
    """Parse tuple format parameter supporting 'x,y' or '(x,y)' formats.

    数值按 **float** 解析 —— WFS pupil 中心单位是 mm (实测常为小数, 如
    ``(-0.14, 0.18)``), 旧的 ``int()`` 解析会拒绝这类值并迫使用户退回到
    ``(0,0)``, 而构造函数传入的 pupil **会覆盖配置文件中的实测值**
    (``thorlab_wfs.py:440-453``) → 用整数近似会造成 pupil 偏移、污染
    ``WFS_ZernikeLsf`` 拟合。整数输入 (如像素中心) 仍可用, 只是返回 float。
    """
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ["mass", "max", "shape"]:
        return value.lower()

    s_clean = re.sub(r"[()\s]", "", str(value))
    try:
        parts = s_clean.split(",")
        if len(parts) != 2:
            raise ValueError("Must have exactly two numbers")
        x, y = map(float, parts)
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
