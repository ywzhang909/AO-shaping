"""数据目录 / 权重目录的解析 (环境变量 → 配置文件 → 默认值)。

优先级 (高 → 低):

- 数据目录: ``AO_SHAPING_DATA_DIR`` → 配置文件 ``[paths] data_dir`` → ``data``
- 权重目录: ``SLM_MODELS_DIR`` → ``AO_SHAPING_WEIGHTS_DIR`` → 配置文件 ``[paths] weights_dir``
  → ``<数据目录>/slm_models``

配置文件默认为 ``ao_shaping.toml``: 源码检出时位于仓库根, 打包 exe 时位于 exe 同目录;
可用 ``AO_SHAPING_CONFIG`` 指定其它路径。配置文件中的相对路径相对于该文件所在目录解析。

本模块在 import 时不做任何磁盘 I/O: 配置文件在每次调用时惰性读取 (文件很小)。
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from loguru import logger

ENV_DATA_DIR = "AO_SHAPING_DATA_DIR"
ENV_WEIGHTS_DIR = "AO_SHAPING_WEIGHTS_DIR"
ENV_LEGACY_WEIGHTS_DIR = "SLM_MODELS_DIR"
ENV_CONFIG_FILE = "AO_SHAPING_CONFIG"
CONFIG_FILE_NAME = "ao_shaping.toml"

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_WEIGHTS_SUBDIR = "slm_models"


def _anchor_dir() -> Path:
    """配置文件默认所在目录: 打包 exe 为 exe 目录, 源码检出为仓库根。"""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.executable).resolve().parent
    # src/ao_shaping/utils/io/paths.py -> 仓库根
    return Path(__file__).resolve().parents[4]


def config_file_path() -> Path:
    """配置文件路径 (不要求存在)。"""
    override = os.environ.get(ENV_CONFIG_FILE, "").strip()
    if override:
        return Path(override)
    return _anchor_dir() / CONFIG_FILE_NAME


def _read_paths_section() -> dict[str, str]:
    """读取配置文件 ``[paths]`` 段; 文件缺失或格式错误时返回空 dict 并记日志。"""
    path = config_file_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("配置文件读取失败, 忽略: {} ({})", path, exc)
        return {}
    section = data.get("paths", {})
    if not isinstance(section, dict):
        logger.warning("配置文件 [paths] 段不是表, 忽略: {}", path)
        return {}
    return {str(k): str(v) for k, v in section.items() if isinstance(v, str)}


def _resolve_config_path(raw: str) -> Path:
    """配置文件中的路径: 绝对路径原样, 相对路径相对于配置文件所在目录。"""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_file_path().parent / candidate).resolve()


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def data_dir() -> Path:
    """数据目录 (运行输出、会话目录等)。"""
    env = _env_path(ENV_DATA_DIR)
    if env is not None:
        return env
    cfg = _read_paths_section().get("data_dir", "").strip()
    if cfg:
        return _resolve_config_path(cfg)
    return _DEFAULT_DATA_DIR


def weights_dir() -> Path:
    """权重目录 (按 SLM 序列号保存的拟合模型 ``<serial>/latest``)。"""
    for name in (ENV_LEGACY_WEIGHTS_DIR, ENV_WEIGHTS_DIR):
        env = _env_path(name)
        if env is not None:
            return env
    cfg = _read_paths_section().get("weights_dir", "").strip()
    if cfg:
        return _resolve_config_path(cfg)
    base = data_dir()
    if not base.is_absolute():
        base = (_anchor_dir() / base).resolve()
    return base / _DEFAULT_WEIGHTS_SUBDIR
