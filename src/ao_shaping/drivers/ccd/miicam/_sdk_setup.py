"""MIICAM SDK 路径查找与设置工具。"""

import os
import sys
import ctypes

from loguru import logger

from ao_shaping.utils.io.file import ROOT_DIR


def _find_miicam_sdk_path() -> str | None:
    """通过检查多个可能位置来查找 MIICAM SDK 路径。

    检查顺序:
    1. 环境变量 MIICAM_SDK_PATH (可由用户配置)
    2. 随项目自带 (src/ao_shaping/drivers/ccd/_miicam_sdk)
    3. 外部 libs 目录 (libs/miicamsdk.20240728/python)

    返回:
        str | None: 找到时返回 SDK 路径, 否则返回 None。
    """
    # 方案 0: 环境变量 (最高优先级, 可由用户配置)
    env_sdk_path = os.environ.get("MIICAM_SDK_PATH")
    if env_sdk_path and os.path.isdir(env_sdk_path):
        return env_sdk_path

    _project_root = str(ROOT_DIR)

    # 去重并保持顺序
    seen: set[str] = set()
    _MII_SDK_PATHS: list[str] = []
    for path in [
        # 方案 1: 随项目自带 (供开发使用)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_miicam_sdk"),
        # 方案 2: 外部 libs 目录
        os.path.join(_project_root, "libs", "miicamsdk.20240728", "python"),
    ]:
        if path not in seen:
            seen.add(path)
            _MII_SDK_PATHS.append(path)

    for path in _MII_SDK_PATHS:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "miicam.py")):
            return path
    return None


def _setup_miicam_sdk() -> bool:
    """通过把其路径加入 sys.path 与 DLL 搜索路径来设置 MIICAM SDK。

    返回:
        bool: SDK 被找到且设置成功返回 True, 否则返回 False。
    """
    sdk_path = _find_miicam_sdk_path()
    if sdk_path is None:
        logger.error("MIICAM SDK not found")
        return False

    if sdk_path not in sys.path:
        sys.path.append(sdk_path)

    # 把 SDK 路径加入 DLL 搜索路径 (Windows)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(sdk_path)
        except Exception as e:
            logger.warning(f"Failed to add DLL directory: {e}")

    # 若可用则预加载 SDK DLL
    dll_path = os.path.join(sdk_path, "MIIUSB.dll")
    if os.path.exists(dll_path):
        try:
            ctypes.CDLL(dll_path)
        except Exception as e:
            logger.warning(f"Failed to pre-load MIIUSB.dll: {e}")

    logger.debug(f"MIICAM SDK set up successfully from: {sdk_path}")
    return True
