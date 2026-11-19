"""Beam Quality 窗口 → 整形会话的中心口径偏好 (进程内交接)。

Beam Quality、传输整形、静态整形三个页面同处一个 Qt 进程
(``DesktopShellApp``), 传输页经 ``QThreadPool`` 在本进程内跑 ``run_session``,
而非另起子进程。因此这里用模块级状态做交接: 操作员在 Beam Quality 里实际
打开过的相机 (kind, cam_id) 与点选的中心计算方法, 会被后续整形链路优先采用。

口径约定:

* 从未点过 (``None``) ⇒ 回落到 :data:`DEFAULT_CENTER_METHOD` (椭圆中心),
  与 :mod:`ao_shaping.utils.image.zero_order` 和
  :mod:`ao_shaping.gui.ccd.beam_quality.analyzer` 的默认口径一致。
* 本模块只存 *界面键* (``centroid`` / ``box_centroid`` / ``ellipse`` /
  ``peak``); 取中心时把 ``peak`` 映射为 ``argmax``, 使
  :func:`~ao_shaping.utils.image.zero_order.zero_order_center_by_method`
  永远不会收到界面侧的 ``peak``。
* 仅在单进程内有效 (桌面控制台), 不持久化、不跨进程 —— 有意为之: 一份陈旧
  偏好不该悄悄改变下一次实验的定位方式。

坐标序: :func:`resolve_zero_order_center` 返回 ``(x, y)``, 与
``zero_order_center`` 一致; 需要行优先的行内调用方用
``optimizer.wfless._measure.locate_zero_order_preferred``, 它返回
``(cy, cx)``, 与本模块的 ``(x, y)`` 严格区分。
"""

from __future__ import annotations

import numpy as np

from ao_shaping.utils.image.zero_order import zero_order_center_by_method

__all__ = [
    "DEFAULT_CENTER_METHOD",
    "get_center_method",
    "get_opened_camera",
    "reset",
    "resolve_zero_order_center",
    "set_center_method",
    "set_opened_camera",
]


#: 未点选时的默认口径; 必须与 ``zero_order.DEFAULT_ZERO_ORDER_METHOD``、
#: ``analyzer.DEFAULT_CENTER_METHOD`` 保持一致 (椭圆中心)。
DEFAULT_CENTER_METHOD = "ellipse"

#: Beam Quality 下拉框的全部界面键 (见 ``analyzer.CENTER_METHODS``)。
_ANALYZER_METHODS: frozenset[str] = frozenset(
    {"centroid", "box_centroid", "ellipse", "peak"}
)

#: 界面键 → ``zero_order_center_by_method`` 的算法键。界面用 ``peak`` 指代
#: 最大值法, 而后者只认 ``argmax``; 其余三键同名直通。
_ANALYZER_TO_ZERO_ORDER: dict[str, str] = {
    "centroid": "centroid",
    "box_centroid": "box_centroid",
    "ellipse": "ellipse",
    "peak": "argmax",
}

_center_method: str | None = None
_opened_camera: tuple[str, int] | None = None


def set_center_method(method: str) -> None:
    """记录操作员在 Beam Quality 里点选的中心计算方法 (界面键)。

    Args:
        method: ``centroid`` / ``box_centroid`` / ``ellipse`` / ``peak`` 之一。

    Raises:
        ValueError: 方法键不在界面键集合内。
    """
    if method not in _ANALYZER_METHODS:
        raise ValueError(
            f"unknown center method {method!r}; expected one of "
            f"{sorted(_ANALYZER_METHODS)}"
        )
    global _center_method
    _center_method = method


def get_center_method() -> str:
    """返回当前中心计算方法 (界面键); 从未点选时返回默认椭圆中心。"""
    return _center_method if _center_method is not None else DEFAULT_CENTER_METHOD


def set_opened_camera(kind: str, cam_id: int) -> None:
    """记录 Beam Quality 成功打开的帧源 (相机类型与编号)。"""
    global _opened_camera
    _opened_camera = (str(kind), int(cam_id))


def get_opened_camera() -> tuple[str, int] | None:
    """返回成功打开过的 ``(kind, cam_id)``; 从未打开过为 ``None``。"""
    return _opened_camera


def resolve_zero_order_center(frame: np.ndarray) -> tuple[int, int]:
    """按偏好口径取 0 级中心, 返回 ``(x, y)``。

    等价于 ``zero_order_center_by_method(frame, method)``, 其中 ``method`` 来自
    :func:`get_center_method`, 并把界面键 ``peak`` 映射为 ``argmax``。
    """
    method = _ANALYZER_TO_ZERO_ORDER[get_center_method()]
    return zero_order_center_by_method(frame, method)


def reset() -> None:
    """清空全部偏好 (测试用; 让模块状态不跨用例泄漏)。"""
    global _center_method, _opened_camera
    _center_method = None
    _opened_camera = None
