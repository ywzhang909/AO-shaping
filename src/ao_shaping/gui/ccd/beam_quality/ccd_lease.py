"""进程内 CCD 租约：同一台相机同时只允许一个采集者。

为什么需要它
------------
桌面外壳里 BeamQuality 页与动态整形页各建各的
:class:`~ao_shaping.gui.ccd.beam_quality.frame_source.CameraFrameSource`。
两个源同时 ``start`` 同一台 CCD 时，底层 SDK 会报重复打开（更糟的是两个
句柄抢流），而驱动层的错误信息通常看不出「另一页正在用」。本模块把「谁在
用哪台相机」变成进程内原子状态：``try_acquire`` 失败即明确占用，由帧源给出
可读错误，而不是让冲突拖到 SDK。

口径
----
* 键是 ``(camera_type, cam_id)``；``camera_type`` 统一去空白、转小写，
  数字形式的 ``cam_id`` 统一成 ``int``（``"1"`` 与 ``1`` 是同一台相机）。
* 只租约**真实相机**。离屏仿真源 (``SimBeamSource``) 不占租约，可随时与
  实验系统租赁并存。
* 允许同时租不同相机（多相机场景）；``current()`` 返回最近一次成功获取的
  那台。**同键冲突绝不覆盖旧租约**（失败方不写入任何状态）。
* 租约不绑定线程：帧源对象在 ``start()`` 成功后持有、``stop()`` 里释放；
  释放是幂等的，释放未持有的键是无操作。
"""

from __future__ import annotations

import threading

from loguru import logger

__all__ = [
    "current",
    "held_keys",
    "release",
    "reset_for_tests",
    "try_acquire",
]

#: 租约键：``(camera_type, cam_id)``。
LeaseKey = tuple[str, int | str]

_lock = threading.Lock()
#: 键 → 获取序号（越大越新）。
_held: dict[LeaseKey, int] = {}
_seq = 0
#: 最近一次成功获取的键；``None`` 表示进程内没有租约。
_current: LeaseKey | None = None


def _normalize(camera_type: str, cam_id: int | str) -> LeaseKey:
    """把调用方写法统一成比较用的键。"""
    kind = str(camera_type).strip().lower()
    if not kind:
        raise ValueError("camera_type 不能为空")
    if isinstance(cam_id, bool):
        raise ValueError(f"cam_id 不能是 bool: {cam_id!r}")
    try:
        return (kind, int(cam_id))
    except (TypeError, ValueError):
        text = str(cam_id).strip()
        if not text:
            raise ValueError("cam_id 不能为空") from None
        return (kind, text)


def try_acquire(camera_type: str, cam_id: int | str) -> bool:
    """尝试租用一台相机；成功返回 ``True``。

    已被占用时返回 ``False`` 且**不改变**现有租约（不覆盖持有者）。
    """
    global _seq, _current
    key = _normalize(camera_type, cam_id)
    with _lock:
        if key in _held:
            return False
        _seq += 1
        _held[key] = _seq
        _current = key
    logger.debug("CCD 租约获取: {}#{}", key[0], key[1])
    return True


def release(camera_type: str, cam_id: int | str) -> None:
    """释放租约；未持有该键时是无操作（幂等）。"""
    global _current
    key = _normalize(camera_type, cam_id)
    with _lock:
        if _held.pop(key, None) is None:
            return
        if _current == key:
            _current = max(_held, key=_held.__getitem__) if _held else None
    logger.debug("CCD 租约释放: {}#{}", key[0], key[1])


def current() -> LeaseKey | None:
    """最近一次成功获取、尚未释放的租约；没有则 ``None``。"""
    with _lock:
        return _current


def held_keys() -> tuple[LeaseKey, ...]:
    """当前全部租约（按获取先后排序），用于诊断与冲突提示。"""
    with _lock:
        items = sorted(_held.items(), key=lambda item: item[1])
    return tuple(key for key, _seq_value in items)


def reset_for_tests() -> None:
    """清空进程内租约。**仅供测试夹具**，生产代码不要调用。"""
    global _seq, _current
    with _lock:
        _held.clear()
        _seq = 0
        _current = None
