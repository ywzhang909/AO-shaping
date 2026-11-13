"""2π 对应灰度 (相位编码满量程) 的唯一解析入口。

**这是"弧度 → 灰度"的标度, 不是设备回报的物理 2π 灰度。** 两者在本仓库严格分开:

============  ==============================  ==================================
概念          来源                            用途
============  ==============================  ==================================
编码标度      本模块 (配置 → 缺省 1023)        ``gray = φ / 2π × max_grayscale``
物理 2π 灰度  ``Santec._max_gray`` (相位表)    只作诊断, 不参与编码
============  ==============================  ==================================

编码标度是**用户可配**的: 面板满量程 (10 bit ⇒ 1023) 是缺省值, 但不同批次/型号的
面板, 或与厂商软件对齐标度时, 需要换一个值。改动只需要写配置文件, 不必改代码:

.. code-block:: toml

    [slm]
    two_pi_grayscale = 1023   # 缺省值; 省略该键即 1023

优先级: 调用方显式传入 > ``[slm] two_pi_grayscale`` > :data:`DEFAULT_MAX_GRAYSCALE`。
配置文件缺失、键缺失、类型不对或越界时一律记一条 warning 后回落缺省值 ——
编码永远拿得到一个可用的标度, 不会因为配置写错而让整条发送链失败。

**不要在调用点写死 1023。** 需要标度的地方一律走 :func:`resolve_max_grayscale`
(驱动层还应暴露 ``max_grayscale`` 属性转发到它)。

哪些 ``1023`` **不是**编码标度 (改配置时不要动它们):

* ``Santec.MAX_GRAYSCALE_VALUE`` / ``slm200_constants.get_max_grayscale()`` ——
  硬件灰阶满量程, 用于钳制与 ``uint16`` 合法性检查;
* 厂商矫正 CSV 的 0~1023 约定 (``wavefront_correction`` 的 ``calc`` /
  ``correction_gray_offsets`` / ``csv_to_phase``) —— 官方软件按满量程解释 CSV,
  改成配置值会让导出的矫正文件在厂商软件里整体缩放;
* ``tools/slm/cartographer`` 的 Hartmann 标定图案 —— 那是对**灰度域**做扫描
  (输出正好喂上面那份 CSV), 起点就是满量程, 不是相位编码。

本模块在 import 时不做任何 I/O; 配置在调用时经
:func:`ao_shaping.utils.io.paths.read_config_section` 惰性读取 (按 mtime 缓存)。
"""

from __future__ import annotations

from loguru import logger

from ao_shaping.utils.io import paths

__all__ = [
    "CONFIG_SECTION",
    "CONFIG_KEY",
    "DEFAULT_MAX_GRAYSCALE",
    "configured_max_grayscale",
    "resolve_max_grayscale",
]

#: 配置文件里承载 SLM 编码参数的段名。
CONFIG_SECTION = "slm"

#: 该段里"2π 对应灰度"的键名。
CONFIG_KEY = "two_pi_grayscale"

#: 2π 对应的缺省灰度 (10 bit 面板满量程 = 2**10 - 1)。
DEFAULT_MAX_GRAYSCALE: int = 1023

#: 灰度是 uint16, 超过这个值的"标度"没有物理意义 (也说明配置写错了)。
_MAX_SENSIBLE_GRAYSCALE = 65535


def _coerce_grayscale(raw: object) -> int | None:
    """把配置里的原始值转成合法灰度标度; 不可用返回 ``None``。

    TOML 里写 ``1023.0`` 是常见笔误, 整数值的浮点照收; ``True``/``False``
    在 TOML 里是 bool, 不能当 1/0 用 (那会静默把相位压成一个像素级标度)。
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and raw.is_integer():
        value = int(raw)
    else:
        return None
    if not 1 <= value <= _MAX_SENSIBLE_GRAYSCALE:
        return None
    return value


def configured_max_grayscale() -> int:
    """配置文件里 ``[slm] two_pi_grayscale`` 的值; 未配置或不可用时返回缺省 1023。"""
    raw = paths.read_config_section(CONFIG_SECTION).get(CONFIG_KEY)
    if raw is None:
        return DEFAULT_MAX_GRAYSCALE
    value = _coerce_grayscale(raw)
    if value is None:
        logger.warning(
            "配置文件 [{}] {} 的值 {!r} 不可用 (需要 1..{} 的整数), 改用缺省 {}",
            CONFIG_SECTION,
            CONFIG_KEY,
            raw,
            _MAX_SENSIBLE_GRAYSCALE,
            DEFAULT_MAX_GRAYSCALE,
        )
        return DEFAULT_MAX_GRAYSCALE
    return value


def resolve_max_grayscale(explicit: int | None = None) -> int:
    """解析 2π 对应灰度: 显式值 > 配置值 > :data:`DEFAULT_MAX_GRAYSCALE`。

    Args:
        explicit: 调用方传入的标度; ``None`` 表示"用配置"。

    Returns:
        实际用于 弧度 → 灰度 编码的灰度标度 (``1..65535``)。

    Raises:
        ValueError: ``explicit`` 不是正整数时 —— 这是**调用方代码错误**, 不是配置
            错误, 所以不静默回落 (静默回落会让错标度的图案上屏, 而且无人察觉)。
    """
    if explicit is None:
        return configured_max_grayscale()
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
        raise ValueError(f"max_grayscale 必须是正整数, 收到 {explicit!r}")
    return explicit
