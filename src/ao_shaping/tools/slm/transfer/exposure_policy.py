"""SLM 传输采集的**开场曝光策略**（``auto`` / ``default`` / 显式毫秒）。

为什么需要这个模块
------------------
``capture.py`` 的采集阶梯、``capture_response.py`` 的标定阶梯、
``optical_probe.py`` 的单帧平场，过去都写死同一句
``create_camera(..., exposure_time_ms=0.02)``。0.02 ms 既不是这些设备
的物理下限，也不是任何标定的结果，只是 2026-10-26 第一轮传输实验里一个
够用的数量级。之后它被复制了五次。

把它硬编码的真实代价是：

* 换相机或换光路（加了中性密度滤光片、换了 12-bit 取图模式）之后，0.02 ms
  可能让阶梯从欠曝 100 倍的量级起步，白烧掉一轮采集的帧预算；或者反过来，
  一开场就把相机顶到饱和。
* 上一次会话真正收住的那个曝光（阶梯末端、设备实际生效的那一档）就躺在
  会话产物里，是下一次会话最好的起点，可没有代码去读它。
* 想直接用相机自己配置的曝光起步时（"别替我设，我先看看你现在是多少"），
  没有开关可用。

三态语义
--------
原始词汇完全复用 :mod:`ao_shaping.utils.cli.exposure`：``auto`` /
``default`` / 正数毫秒 / ``None``（等同 ``auto``）。本模块只回答两个问题，
而且这两个问题的答案不一样：

1. :func:`resolve_seed_exposure` —— 开场到底要不要给 ``create_camera`` 传
   ``exposure_time_ms``，传多少？

   ``auto``      → 记住的生效曝光，没有记录就回落到 :data:`FALLBACK_EXPOSURE_MS`
                   （= 原来那个 0.02）。行为与改动前逐位一致。
   ``default``   → **None 表示"别碰相机"**：调用方必须整个省掉这个关键字，
                   让 ``create_camera`` / 驱动自己的配置生效，阶梯从回读值起步。
                   这里最容易被做错成"回落 0.02"，那不是 ``default`` 的意思。
   一个数字      → 就是它（种子），钳进设备可用区间后交给阶梯。

2. :func:`resolve_effective_exposure` —— 这一帧**实际**用了多少曝光？
   永远是设备回读值，与原始 token 无关。平场帧、冻结参数、形状相似度判据
   都按它标定：种子可能没被接受（驱动限幅、设备配置、阶梯改动），落盘的
   数字必须是真正曝光的那一个。

它不是什么
----------
* 不是**固定曝光**。种子只决定阶梯的**起点**，之后的 :func:`adapt_exposure`
  乘法阶梯照旧跑，会话真正用的曝光写在 ``manifest.json`` 每一行里。
  ``--exposure-ms 1.0`` 的含义是"从 1 ms 起找"，不是"整场会话用 1 ms"。
* 不是**设备下限**。:data:`FALLBACK_EXPOSURE_MS` 只是"没有历史记录时用什么
  种子"，与 ``get_camera_exposure_range`` 的下界无关；调用方拿到种子之后
  自己还会再钳一次。
* 不是**可靠存储**。状态文件是优化：读不到、JSON 坏掉、数据目录不可写，
  一律退化成"没有记录"并记日志，绝不抛异常、绝不打断硬件在跑的会话。

* 不参与**指纹复用**的判定：:func:`ao_shaping.tools.slm.transfer.auto_reuse.probe_for_fingerprint`
  自己的 5.0 ms 探针曝光是光路复用判据的一部分（"同一条件下再探一次"），
  与本策略无关，本模块不改它。

相机与 SDK 一律在函数内部惰性解析：模块顶层不 import 驱动、不做磁盘 I/O
（只有 ``state_path()`` 在调用时才算路径），于是导入本模块永远安全。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from click import BadParameter
from loguru import logger

from ao_shaping.utils.cli.exposure import (
    AUTO_EXPOSURE,
    AUTO_EXPOSURE_TOKEN,
    DEFAULT_EXPOSURE,
    DEFAULT_EXPOSURE_TOKEN,
    ExposureChoice,
    resolve_exposure,
)

#: 没有历史记录时 ``auto`` 用的种子 (ms)。就是被替换掉的那个硬编码 0.02，
#: 于是 ``auto`` 的默认行为与改动前逐位一致。
FALLBACK_EXPOSURE_MS = 0.02

#: 每台相机共享的曝光记录文件名（落在 ``data/.cache/slm_transfer_exposure/``）。
STATE_FILE = "cameras.json"

#: 两个非数字策略的**规范字符串写法**（``resolve_cli_exposure`` 的返回值，也是
#: ``TransferSession.exposure_ms`` / config.json 里存的东西）。枚举成员本身由词汇
#: 模块给出，这里只把它的 ``value`` 转出来给需要字符串的调用方。
AUTO_TOKEN = AUTO_EXPOSURE.value
DEFAULT_TOKEN = DEFAULT_EXPOSURE.value

__all__ = [
    "AUTO_EXPOSURE",
    "AUTO_EXPOSURE_TOKEN",
    "AUTO_TOKEN",
    "DEFAULT_EXPOSURE",
    "DEFAULT_EXPOSURE_TOKEN",
    "DEFAULT_TOKEN",
    "FALLBACK_EXPOSURE_MS",
    "STATE_FILE",
    "argparse_exposure_type",
    "clear_exposure_state",
    "normalize_exposure",
    "policy_source",
    "read_effective_exposure",
    "record_effective_exposure",
    "remember_exposure",
    "resolve_cli_exposure",
    "resolve_effective_exposure",
    "resolve_seed_exposure",
    "state_path",
]


# ---------------------------------------------------------------------------
# 原始 token：解析 / 归一化
# ---------------------------------------------------------------------------


def normalize_exposure(raw: str | float | int | None) -> object:
    """把原始值整成 :func:`ao_shaping.utils.cli.exposure.resolve_exposure` 收的形状。

    只做「字符串去空格 / 空串当没给」这一件事，别的都原样交给词汇解析器：
    它本来就认 ``ExposureChoice``（已经解析过的值，可往返）、数字，并且会拒绝
    bool / 非有限 / 非正数 —— 语义不在本模块重讲一遍。

    Args:
        raw: CLI / ``TransferSession`` 里存的那个原始值。

    Returns:
        可直接喂给 :func:`resolve_exposure` 的值 (str / 数字 / ``ExposureChoice`` / None)。
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip() or None
    return raw


def resolve_cli_exposure(
    raw: str | float | int | None,
    *,
    param_hint: str = "--exposure-ms",
) -> str:
    """CLI 层：把用户给的原始值归一成 ``auto`` / ``default`` / 数字字符串。

    只做词汇校验，**不碰状态文件、不碰相机** —— 非法值必须在设备被打开之前
    就报错（一个多余的磁盘/相机往返换一次报废的会话不划算）。
    ``None`` 归一成 ``auto``（默认值），于是 ``args.exposure_ms`` 永远是字符串，
    写进 config.json 的 ``requested`` 字段也永远是可比的词汇。

    Args:
        raw: 用户给的原值（``--exposure-ms`` 的参数字符串、数字、或 None）。
        param_hint: 报错文案里的选项名。

    Returns:
        ``AUTO_EXPOSURE`` / ``DEFAULT_EXPOSURE`` / 数字字符串。

    Raises:
        click.BadParameter: 非正数、非有限、或根本不是个曝光 token。
    """
    choice = resolve_exposure(normalize_exposure(raw), param_hint=param_hint)
    if choice is ExposureChoice.AUTO:
        return AUTO_TOKEN
    if choice is ExposureChoice.DEFAULT:
        return DEFAULT_TOKEN
    return _text(float(choice))


def argparse_exposure_type(param_hint: str = "--exposure-ms"):
    """给 argparse ``type=`` 用的解析器（同一个词汇表的 argparse 侧入口）。

    传输工具链的 CLI 一半是 click（``slm`` 入口）、一半是 argparse
    （``capture`` / ``capture_response`` / ``validate`` / ``replay_phases`` 各自的
    ``__main__``，以及 workflow 的那套子命令表），而曝光词汇只该有一份。
    click 直接用 :func:`resolve_cli_exposure`；argparse
    需要的是「报错要变成退出码 2 而不是一屏 traceback」，故有本适配层。
    它不做第三种语义：校验仍由 :func:`resolve_cli_exposure` 完成。

    Args:
        param_hint: 报错文案里的选项名。

    Returns:
        一个 ``str -> str`` 的解析函数，可直接传给 ``add_argument(type=...)``。
    """

    def parse(raw: str) -> str:
        try:
            return resolve_cli_exposure(raw, param_hint=param_hint)
        except BadParameter as exc:
            # argparse 只认 ValueError/TypeError/ArgumentError，其余会变成 traceback。
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return parse


def policy_source(raw: str | float | int | None) -> str:
    """原始值对应哪种策略，只读词汇、**绝不抛异常**（日志/记录用）。

    Args:
        raw: CLI / 会话里存的原始值。

    Returns:
        ``"auto"`` / ``"default"`` / ``"explicit"``；认不出的值报 ``"auto"``
        （与「非法即回默认」的降级口径一致）。
    """
    try:
        choice = resolve_exposure(normalize_exposure(raw), param_hint="--exposure-ms")
    except BadParameter:  # 记录用途的读取绝不能打断采集: 非法值退 auto, 不抛
        return "auto"
    if choice is ExposureChoice.AUTO:
        return "auto"
    if choice is ExposureChoice.DEFAULT:
        return "default"
    return "explicit"


# ---------------------------------------------------------------------------
# 每台相机的生效曝光记录
# ---------------------------------------------------------------------------


def _camera_key(camera: Any, cam_id: Any = 0) -> str:
    """``(后端, 序号)`` 的稳定文件名。

    后端名与 cam_id 都是用户给的字符串/整数，可能带空格、冒号、反斜杠。这里把
    非 ``[a-z0-9._-]`` 的字符换成 ``_`` 并加一个**跨进程稳定**的 CRC32 后缀：
    后缀必须稳定，否则 ``hash()`` 的随机化会让同一台相机每次开机换一个文件，
    「记住上次的生效曝光」当场失效（这是本模块最容易写错的一行）。
    """
    label = f"{str(camera).strip().lower()}-{cam_id}"
    slug = re.sub(r"[^a-z0-9._-]+", "_", label).strip("_")[:48] or "camera"
    return f"{slug}_{zlib.crc32(label.encode('utf-8')):08x}.json"


def state_path(camera: Any = "", cam_id: Any = 0) -> Path:
    """这台相机（或整个后端）的曝光记录路径。

    走 :func:`ao_shaping.utils.io.paths.data_subdir`，于是测试里 monkeypatch
    ``AO_DATA_ROOT`` 就能整体改址（``data_subdir`` 每次调用都重读 env，
    没有缓存问题）。

    Args:
        camera: 相机后端名（``"daheng"`` / ``"miicam"`` / ...）。留空则返回
            后端目录本身（用于整表读取/清理）。
        cam_id: 同一后端里的第几台相机。

    Returns:
        记录文件的 :class:`Path`（不保证存在）。
    """
    from ao_shaping.utils.io.paths import data_subdir

    root = data_subdir(".cache", "slm_transfer_exposure")
    if str(camera) == "":
        return root
    return root / _camera_key(camera, cam_id)


def _usable(value: Any) -> float | None:
    """正有限数 → float；其它一律 None（「不可用」）。"""
    if isinstance(value, bool):  # bool 是 int 的子类，True 绝不是 1 ms 曝光
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def _text(value: float) -> str:
    """浮点转字符串，去掉 ``repr`` 的尾零噪音（``0.5`` 而不是 ``0.5`` 之外的写法）。"""
    return repr(float(value))


def read_effective_exposure(camera: Any = None, cam_id: Any = 0) -> float | None:
    """这台相机上次被接受的**生效**曝光 (ms)，没有记录就 None。

    Args:
        camera: 相机后端名（字符串）。
        cam_id: 后端内的相机序号。

    Returns:
        正有限毫秒数，或 None（没有记录 / 记录不可用 / 读不到）。
    """
    if camera is None or (isinstance(camera, str) and not str(camera).strip()):
        return None
    try:
        payload = json.loads(state_path(camera, cam_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return _usable(payload.get("exposure_ms"))


def remember_exposure(
    camera: Any = None,
    cam_id: Any = 0,
    exposure_ms: Any = None,
    *,
    backend: Any = None,
) -> float | None:
    """记住这台相机上次被接受的**生效**曝光 (ms)；best-effort，返回写入的值。

    先写同目录的临时文件再 ``Path.replace``：Windows 上这是同一卷内的原子改名，
    因此并发会话（campaign 并行 / GUI 与 CLI 同时跑）不会读到半截 JSON，也不会
    出现「A 的写入把 B 的文件截断」。``exposure_ms=None``（= ``default`` 策略，
    相机没被我们碰过）什么都不记，返回 None。

    Args:
        camera: 相机后端名（字符串），或一个相机实例。
        cam_id: 后端内的相机序号。
        exposure_ms: 要记的生效曝光；None/不可用的值不写。
        backend: 显式指定后端名（当第一个参数是相机实例时用它做 key）。

    Returns:
        写入的曝光 (ms)，或 None（没写 / 值不可用 / 目录不可写）。
    """
    key = str(backend) if backend is not None else str(camera)
    value = _usable(exposure_ms)
    if value is None:
        return None
    path = state_path(key, cam_id)
    payload = {
        "camera": str(key),
        "cam_id": cam_id,
        "exposure_ms": value,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 写的是 UTF-8 字节 + 原子改名：Windows 文本模式会把 \n 翻成 \r\n，
        # 而 JSON 不需要那个，逐字节写更省心。
        tmp.write_bytes(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
        tmp.replace(path)
    except OSError as exc:
        # policy: 记录是优化。只读 / 满盘 / 权限不足都只该让下次少一个种子。
        logger.warning(f"曝光策略: 记不下 {path} 的生效曝光 ({exc!r})")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    logger.debug(f"曝光策略: 记住 {key}[{cam_id}] 生效曝光 {value:g} ms")
    return value


def resolve_seed_exposure(
    raw: str | float | int | None,
    *,
    camera: Any = None,
    cam_id: Any = 0,
    floor_ms: float | None = None,
) -> float | None:
    """开场种子 (ms) 给 ``create_camera(exposure_time_ms=...)``；**None = 别碰相机**。

    这是本模块唯一的「原始 token → 相机行为」映射。三态里 ``default`` 与
    ``None``（回落默认）走的是**不同**分支，别在最外层把它们折叠成一个值。

    Args:
        raw: 原始策略值（``auto`` / ``default`` / 正数毫秒 / None / 相机实例以外的
            任何 CLI 字符串）。
        camera: 相机后端名，``auto`` 查记录时用。
        cam_id: 同上，后端内的序号。
        floor_ms: 设备可用区间下界；给了就把种子钳到不低于它（阶梯起点至少是
            设备能做到的最短曝光），同时 ``auto`` 无记录时用 :data:`FALLBACK_EXPOSURE_MS`
            后再钳。

    Returns:
        种子毫秒数；或 None 表示**不要**给 ``create_camera`` 传 ``exposure_time_ms``。

    Raises:
        click.BadParameter: ``raw`` 不是可理解的曝光 token（"0" / "-3" / "nonsense"）。
    """
    choice = resolve_exposure(normalize_exposure(raw), param_hint="--exposure-ms")
    if choice is ExposureChoice.DEFAULT:
        # ``default`` = 相机自己的配置说了算：连关键字都不给，阶梯从回读起步。
        return None
    if choice is ExposureChoice.AUTO:
        seed = _usable(read_effective_exposure(camera, cam_id))
        if seed is None:
            seed = FALLBACK_EXPOSURE_MS
        return _clamped(seed, floor_ms)
    return _clamped(float(choice), floor_ms)


def _clamped(seed: float, floor_ms: float | None) -> float:
    """把种子抬到设备下界之上（调用方拿到后还会再钳一次，此处只为不白跑阶梯）。"""
    if floor_ms is not None and seed < floor_ms:
        return floor_ms
    return seed


def resolve_effective_exposure(camera: Any, requested_ms: Any = None) -> float | None:
    """这一帧**实际**用的曝光 (ms)：优先设备回读，读不到退化成请求值。

    落盘（平场帧 sidecar、config.json、frozen parameters）永远用这个，不用种子
    —— 种子可能没被相机接受。

    Args:
        camera: 相机实例（有 ``get_camera_exposure_ms`` 即可）。None 表示设备没开。
        requested_ms: 回读失败时的兜底值（种子 / 请求值）。

    Returns:
        正有限毫秒数，或 None（两头都拿不到）。
    """
    if camera is not None:
        try:
            from ao_shaping.drivers.ccd.common import get_camera_exposure_ms

            read_back = _usable(get_camera_exposure_ms(camera))
        except Exception as exc:  # policy: 回读失败只是少一个更准的数字
            logger.debug(f"曝光策略: 生效曝光回读失败 ({exc!r})，用请求值")
            read_back = None
        if read_back is not None:
            return read_back
    return _usable(requested_ms)


def record_effective_exposure(
    camera: Any,
    cam_id: Any = 0,
    exposure_ms: Any = None,
    *,
    source: str | None = None,
    backend: Any = None,
) -> float | None:
    """阶梯收住之后记下设备**回读**到的生效曝光；返回该生效值。

    Args:
        camera: 相机实例（用于回读生效曝光），或直接给后端名字符串。
        cam_id: 后端内的相机序号。
        exposure_ms: 回读失败时的兜底值（阶梯最终请求值）。
        source: 策略来源标注（``"auto"`` / ``"default"`` / ``"explicit"``），
            只进日志，用于诊断「种子是从哪一级来的」。
        backend: 记录 key 用的后端名；不给就从实例类名猜，猜不到就退回
            :data:`FALLBACK_EXPOSURE_MS` 所属的通用 key。

    Returns:
        记下的生效曝光 (ms)，或 None（完全没记成）。
    """
    is_instance = not isinstance(camera, str)
    # 只有拿到实例才回读得到；给的是后端名字符串时，生效值就等于调用方给的兜底。
    effective = resolve_effective_exposure(camera if is_instance else None, exposure_ms)
    if backend is not None:
        key = backend
    elif is_instance:
        key = type(camera).__name__.lower()
    else:
        key = camera
    remembered = remember_exposure(key, cam_id, effective)
    logger.debug(
        f"曝光策略: 来源 {source or 'unknown'} / 后端 {key}[{cam_id}] 生效 "
        f"{'无' if effective is None else f'{effective:g} ms'}"
    )
    return remembered


def clear_exposure_state(camera: Any = None, cam_id: Any = 0) -> None:
    """删掉这台相机的记录（没有就静默）。记录是优化，删不掉也不报错。"""
    try:
        state_path(camera, cam_id).unlink(missing_ok=True)
    except OSError:
        pass
