"""数据目录 / 权重目录 / 设备配置目录的解析 (环境变量 → 配置文件 → 默认值)。

三类路径的优先级 (高 → 低) 都是 **环境变量 → 配置文件 → 默认值**:

============  ==========================  ==========================  ==========================
用途          环境变量                    配置文件 ``[paths]`` 键      默认值
============  ==========================  ==========================  ==========================
数据目录      ``AO_SHAPING_DATA_DIR``      ``data_dir``               ``data`` (相对 CWD)
权重目录      ``SLM_MODELS_DIR`` /        ``weights_dir``            ``<锚定目录>/data/slm_models``
              ``AO_SHAPING_WEIGHTS_DIR``
设备配置目录  ``CCD_CONFIG_DIR`` /        ``ccd_config_dir`` 等      ``<锚定目录>/data/<kind>_configs``
              ``SLM_CONFIG_DIR`` / ...
============  ==========================  ==========================  ==========================

**锚定目录** (anchor): 源码检出为仓库根, 打包 exe 为 exe 所在目录。设备配置目录
默认锚定在它下面 —— 与改造前 ``PROJECT_ROOT / "data" / "*_configs"`` 完全一致,
所以源码检出下的行为不变, 而 exe 里自动指向 exe 旁的 ``data/``。

**数据目录** (``data_dir``) 存的是运行*输出* (会话目录、快照、调试产物), 刻意保持
CWD 相对 —— 仓库里 ``PATHS.root_dir`` 的约定就是这样, 换机器不用改代码。一旦用户
**显式**配置了数据目录 (环境变量或配置文件), 设备配置目录与权重目录就跟着它走:
显式配置的数据目录是一个整体, 拆开会让 "把数据放到另一块盘" 变成半拉子。

配置文件默认名为 ``ao_shaping.toml``: 源码检出时在仓库根, 打包 exe 时在 exe 同目录;
用 ``AO_SHAPING_CONFIG`` 可指定其它路径。配置文件里的相对路径相对于**配置文件所在
目录**解析, 不是 CWD。示例见仓库根 ``ao_shaping.toml.example``。

除 ``[paths]`` 外, 文件里还可以有别的段 (如 ``[slm] two_pi_grayscale``)。本模块不解释
它们的语义, 只提供 :func:`read_config_section` 原样取出整段, 由各模块自己校验类型
并给缺省值 —— 这样新增一个配置项不需要再改这里。

本模块在 import 时不做任何磁盘 I/O: 配置文件在每次调用时惰性读取 (文件很小),
按 (路径, mtime) 缓存, 改完配置立即生效。
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Callable

from loguru import logger

ENV_DATA_DIR = "AO_SHAPING_DATA_DIR"
ENV_WEIGHTS_DIR = "AO_SHAPING_WEIGHTS_DIR"
ENV_LEGACY_WEIGHTS_DIR = "SLM_MODELS_DIR"
ENV_CONFIG_FILE = "AO_SHAPING_CONFIG"
CONFIG_FILE_NAME = "ao_shaping.toml"

#: 设备配置目录的种类 → (环境变量名, 默认子目录名, 配置文件键前缀)。
#:
#: 键前缀拼出 ``[paths]`` 里的键 (如 ``ccd`` → ``ccd_config_dir``) 与默认目录名
#: (如 ``ccd`` → ``ccd_configs``)。
DEVICE_CONFIG_KINDS: dict[str, tuple[str, str]] = {
    "ccd": ("CCD_CONFIG_DIR", "ccd_configs"),
    "slm": ("SLM_CONFIG_DIR", "slm_configs"),
    "dm": ("DM_CONFIG_DIR", "dm_configs"),
    "micro_dm": ("MICRO_DM_CONFIG_DIR", "micro_dm_configs"),
    "wfs": ("WFS_CONFIG_DIR", "wfs_configs"),
}

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_WEIGHTS_SUBDIR = "slm_models"
_DEFAULT_DATA_SUBDIR = "data"

#: ``(配置文件路径, mtime) → 整份配置``。mtime 变了就重读, 改配置立即生效。
#:
#: 缓存的是**整份文档**而不是单个段: 除 ``[paths]`` 之外还有 ``[slm]`` 等段
#: (见 :func:`read_config_section`), 按段各存一份会让同一份文件被解析多遍。
_CONFIG_CACHE: tuple[str, float, dict[str, object]] | None = None


def anchor_dir() -> Path:
    """锚定目录: 打包 exe 为 exe 所在目录, 源码检出为仓库根。"""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    # src/ao_shaping/utils/io/paths.py -> 仓库根
    return Path(__file__).resolve().parents[4]


def _is_frozen() -> bool:
    """是否运行在 Nuitka 打包的 exe 里。"""
    return getattr(sys, "frozen", False) or "__compiled__" in globals()


def config_file_path() -> Path:
    """配置文件路径 (不要求存在)。"""
    override = os.environ.get(ENV_CONFIG_FILE, "").strip()
    if override:
        return Path(override).expanduser()
    return anchor_dir() / CONFIG_FILE_NAME


def _load_config_document() -> dict[str, object]:
    """读取并缓存整份配置文件; 文件缺失或格式错误时返回空 dict 并记日志。"""
    global _CONFIG_CACHE

    path = config_file_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _CONFIG_CACHE = None
        return {}

    if _CONFIG_CACHE is not None:
        cached_path, cached_mtime, cached_doc = _CONFIG_CACHE
        if cached_path == str(path) and cached_mtime == mtime:
            return cached_doc

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("配置文件读取失败, 忽略: {} ({})", path, exc)
        return {}

    _CONFIG_CACHE = (str(path), mtime, data)
    return data


def read_config_section(section: str) -> dict[str, object]:
    """读取配置文件里任意 ``[section]`` 段的**原始值**; 段缺失时返回空 dict。

    与 :func:`_read_paths_section` 的区别: 后者只服务路径解析, 于是把值强转成
    ``str`` 并丢掉非字符串; 这里的调用方要的是**带类型的数值/布尔** (如
    ``[slm] two_pi_grayscale = 1023``), 所以原样返回, 由调用方自己校验类型。

    文件不存在、无法解析、或该段不是表时都返回空 dict 并记一条日志 —— 调用方
    据此走各自缺省值, 而不是分支判"配置有没有"。

    Args:
        section: 段名 (TOML 顶层键), 如 ``"slm"``。

    Returns:
        该段的键 → 原始值 (``bool`` / ``int`` / ``float`` / ``str`` / 嵌套表)。
    """
    data = _load_config_document()
    value = data.get(section, {})
    if not isinstance(value, dict):
        logger.warning(
            "配置文件 [{}] 段不是表, 忽略: {}", section, config_file_path()
        )
        return {}
    return dict(value)


def _read_paths_section() -> dict[str, str]:
    """读取配置文件 ``[paths]`` 段 (只取字符串值); 文件缺失或格式错误时返回空 dict。"""
    section = read_config_section("paths")
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


def explicit_data_dir() -> Path | None:
    """用户**显式**配置的数据目录 (环境变量或配置文件); 没配置时返回 ``None``。

    与 :func:`data_dir` 的区别: 后者在没配置时返回 CWD 相对的 ``data`` (运行输出
    的约定), 而这里只回答 "用户指定过吗" —— 设备配置目录与权重目录据此决定是留在
    锚定目录下, 还是跟着显式配置走。
    """
    env = _env_path(ENV_DATA_DIR)
    if env is not None:
        return env
    cfg = _read_paths_section().get("data_dir", "").strip()
    if cfg:
        return _resolve_config_path(cfg)
    return None


def data_dir() -> Path:
    """数据目录 (运行输出、会话目录、快照等)。

    源码检出下默认是 CWD 相对的 ``data`` —— 仓库约定 ``PATHS.root_dir`` 存的是运行
    *输出*, CWD 相对是对的 (换机器不用改代码)。打包 exe 下默认落在 exe 旁的
    ``data/``: 双击 exe 时 CWD 未必可写, 落在安装目录旁更可预期。
    """
    explicit = explicit_data_dir()
    if explicit is not None:
        return explicit
    if _is_frozen():
        return anchor_dir() / _DEFAULT_DATA_SUBDIR
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
    base = explicit_data_dir()
    if base is None:
        base = anchor_dir() / _DEFAULT_DATA_SUBDIR
    return base / _DEFAULT_WEIGHTS_SUBDIR


def device_config_dir(kind: str) -> Path:
    """设备配置目录 (``ccd`` / ``slm`` / ``dm`` / ``micro_dm`` / ``wfs``)。

    优先级: 环境变量 → 配置文件 → (显式配置了数据目录 ? 数据目录 : 锚定目录/data)。

    Args:
        kind: :data:`DEVICE_CONFIG_KINDS` 里的键。

    Returns:
        该类设备的配置根目录 (``<dir>/<device_type>/<serial>.json`` 里的 ``<dir>``)。
    """
    try:
        env_name, default_name = DEVICE_CONFIG_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"未知设备配置目录种类 {kind!r}; 可用: {sorted(DEVICE_CONFIG_KINDS)}"
        ) from None

    env = _env_path(env_name)
    if env is not None:
        return env

    cfg = _read_paths_section().get(f"{kind}_config_dir", "").strip()
    if cfg:
        return _resolve_config_path(cfg)

    base = explicit_data_dir()
    if base is None:
        base = anchor_dir() / _DEFAULT_DATA_SUBDIR
    return base / default_name


def resolved_device_config_dirs() -> dict[str, Path]:
    """所有设备配置目录的当前解析结果 (给 CLI / 诊断页用)。"""
    return {kind: device_config_dir(kind) for kind in DEVICE_CONFIG_KINDS}


def data_subdir(*parts: str) -> Path:
    """数据目录下的某个子路径 (运行输出: 会话目录、快照、调试产物)。

    与 :func:`data_dir` 的关系: 数据目录的**约定**是 CWD 相对 (源码检出) 或 exe 同目录
    (冻结), 所以运行时产物都在它下面。历史代码把这条约定写成了字面量
    ``Path("data") / "debug" / ...`` —— 打包成 exe 后 CWD 未必是 exe 目录, 于是产物
    落到意料之外的地方。这个函数把同一份约定收敛到一处: 源码检出下返回值与旧的
    字面量逐位相同 (``Path("data") / "debug" / ...``), 冻结下落在 exe 旁的 ``data/``。

    Args:
        parts: 相对数据目录的路径片段, 如 ``("debug", "slm_transfer")``。

    Returns:
        ``<数据目录>/<parts...>``; 片段为空时返回数据目录本身。
    """
    return data_dir().joinpath(*parts)


#: 供 :class:`~ao_shaping.utils.io.file.DeviceConfigManager` 使用的惰性目录工厂。
def device_config_dir_factory(kind: str) -> Callable[[], Path]:
    """返回 ``device_config_dir(kind)`` 的惰性工厂。

    驱动模块用它在**调用时**取目录, 而不是 import 时把路径冻在模块常量里 ——
    import 期不得读盘, 而且配置改完不必重启进程。
    """

    def _factory() -> Path:
        return device_config_dir(kind)

    return _factory
