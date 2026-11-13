"""HDR 预览的阶梯规则与模式文案: 「HDR 获取」开关与帧源都要用同一套说法。

放在独立模块而不是 ``app.py`` 里, 是为了让**离线**两头都能用它: 帧源
(:mod:`.frame_source`) 校验操作员填的阶梯, 测试和脚本解析界面文案 —— 两头都不
必把 Qt 或相机原生 SDK 拖进来 (``capture_hdr`` 那条路本身要驱动, 界面上的校验不
需要: 阶梯合不合法, 只看数本身)。

阶梯的**语义**不在这里定义: 升序逐档曝光、最后一档作融合参考、失败退回单曝光,
都由 :meth:`ao_shaping.drivers.ccd.base.BaseCamera.capture_hdr` 与
:func:`ao_shaping.utils.image.hdr.hdr_fuse` 决定。这里只负责把它们翻译成操作员
看得懂的一行字。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ao_shaping.tools.slm.transfer.hdr_probe import DEFAULT_HDR_EXPOSURES_MS

__all__ = [
    "DEFAULT_HDR_LADDER_MS",
    "MAX_LADDER_RUNGS",
    "MIN_LADDER_RUNGS",
    "default_ladder_text",
    "format_ladder_ms",
    "ladder_to_text",
    "parse_exposure_ladder",
    "preview_mode_text",
    "validate_exposure_ladder",
]

#: 分隔符: 中英文逗号/分号/空白/顿号 —— 抄仪表盘 ``parse_grid_sizes`` 那一套写法。
_LADDER_SEP_RE = re.compile(r"[,，;；\s、]+")

#: 一轮 bracket 最多几档。再多的话每轮预览的耗时 (逐档改曝光 + 读出) 会让画面明显
#: 跟不上手, 而收益早就饱和了; 超限直接判非法, 让操作员删几个数而不是默默截断。
MAX_LADDER_RUNGS = 10

#: 最少几档。一档就只是"改个曝光", 谈不上融合。
MIN_LADDER_RUNGS = 2

#: 界面与帧源共用的默认曝光阶梯 (ms)。**数字取自 transfer 侧的 HDR 探测**
#  (``tools.slm.transfer.hdr_probe``), 那里是按真实相机试过的一组; 两边共用同一份,
#  免得"预览默认阶梯"和"探测默认阶梯"长成两个数。该模块是 Qt-free 且纯 NumPy 的,
#  离线机器 import 它不会拉起 SDK。
DEFAULT_HDR_LADDER_MS: tuple[float, ...] = tuple(DEFAULT_HDR_EXPOSURES_MS)


def default_ladder_text() -> str:
    """默认阶梯的文本框写法 (``"1, 4, 16"``): 占位、回写、恢复默认都用它。"""
    return ladder_to_text(DEFAULT_HDR_LADDER_MS)


def ladder_to_text(ladder_ms: Sequence[float]) -> str:
    """阶梯 → 文本框里能直接再解析回去的写法 (``"1, 4, 16"``)。"""
    return ", ".join(f"{float(value):g}" for value in ladder_ms)


def parse_exposure_ladder(raw: str) -> tuple[float, ...]:
    """把文本框里的曝光阶梯解析成浮点元组 (不做合法性判断)。

    空串/只有分隔符 → 空元组; 解析不出的片段静默跳过 (界面上会按解析结果回写文本,
    操作员看得见自己漏了什么)。
    """
    values: list[float] = []
    for part in _LADDER_SEP_RE.split((raw or "").strip()):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return tuple(values)


def validate_exposure_ladder(
    ladder_ms: Sequence[float], *, max_ms: float | None = None
) -> tuple[float, ...]:
    """校验并整理 HDR 曝光阶梯, 返回**升序去重**后的元组。

    规则与 :meth:`BaseCamera.capture_hdr` 的入参约定一致 (至少 2 档、有限正数),
    再补两条界面上的红线: 档数上限 :data:`MAX_LADDER_RUNGS`; 给了 ``max_ms`` 就不
    许超过相机当前量程上限。

    Raises:
        ValueError: 阶梯不可用, 文案直接能贴到状态栏 (界面据此回滚「HDR 获取」开关)。
    """
    values = [float(value) for value in ladder_ms]
    if len(values) < MIN_LADDER_RUNGS:
        raise ValueError(
            f"HDR 曝光阶梯至少 {MIN_LADDER_RUNGS} 档 "
            f"(默认 {default_ladder_text()}), 现在只有 {len(values)} 档"
        )
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"HDR 曝光阶梯里有 {value:g} ms: 每档都得是有限正数 (毫秒)"
            )
    if max_ms is not None and max_ms > 0.0:
        over = [value for value in values if value > max_ms]
        if over:
            raise ValueError(
                "HDR 曝光阶梯超出相机量程上限 "
                f"{max_ms:g} ms: {' '.join(f'{v:g}' for v in over)}"
            )
    if len(values) > MAX_LADDER_RUNGS:
        raise ValueError(
            f"HDR 曝光阶梯最多 {MAX_LADDER_RUNGS} 档, 现在 {len(values)} 档 —— "
            "每多一档预览就慢一轮, 请删掉中间那些"
        )
    ordered = sorted(set(values))
    if len(ordered) < MIN_LADDER_RUNGS:
        raise ValueError(
            f"HDR 曝光阶梯去重后只剩 {len(ordered)} 档 "
            f"({format_ladder_ms(ordered)}), "
            f"至少要有 {MIN_LADDER_RUNGS} 档才谈得上融合"
        )
    return tuple(ordered)


def format_ladder_ms(ladder_ms: Sequence[float]) -> str:
    """``(1.0, 4.0, 16.0)`` → ``"1/4/16"``: 控件文案与日志都用这个写法。"""
    return "/".join(f"{float(v):g}" for v in ladder_ms)


def preview_mode_text(
    hdr: bool,
    ladder_ms: Sequence[float] = (),
    *,
    tone_map: str = "",
    reference_ms: float | None = None,
    stale: str = "",
) -> str:
    """预览模式一行的文案: 现在屏上是单曝光还是多曝光融合、阶梯是多少、有没有降级。

    ``hdr`` 为假时只说"单曝光取流" —— 这时候阶梯和色调映射都不作数, 说了反而误导。
    ``stale`` 是降级原因 (例如某次 bracket 失败), 非空就把它带在末尾, 让"预览看着
    正常但已经不是 HDR"这件事在界面上留下痕迹。
    """
    if not hdr:
        return "预览模式: 单曝光取流"
    parts = ["预览模式: HDR 多曝光融合"]
    if len(ladder_ms) >= 1:
        parts.append(f"阶梯 {format_ladder_ms(ladder_ms)} ms")
    if reference_ms is not None:
        parts.append(f"参考 {reference_ms:g} ms")
    if tone_map:
        parts.append(f"显示 {tone_map}")
    text = " · ".join(parts)
    return f"{text} · {stale}" if stale else text
