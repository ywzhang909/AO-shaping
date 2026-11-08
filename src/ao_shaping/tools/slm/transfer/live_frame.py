"""CCD 实时帧: 迁移学习桌面端 ⑥ 实时画面的后端一半。

Qt-free by construction: 本模块在**测量线程**里被 ``capture`` / ``validate``
调用, 所以不能 (也不会) 引入任何 GUI 工具包。负载是 ``ndarray`` + int + str 的
普通 dict, 因此可以跨 ``QThreadPool`` 边界传递而不共享任何 QObject。

刷新粒度
--------
**每一次 ``get_numpy_image`` 都发一帧**, 不再是每块一张快照。CCD 的读出周期约
0.9 s (``session_plan.READ_SECONDS_PER_FRAME``, 取自 2026-10-28 实测会话),
也就是 ~1.1 帧/秒: 这仍然不是视频流, 所以逐帧发帧不会把重绘压进采集关键路径。

测量线程为每帧付出的代价只有一次 ``ndarray.copy()`` (2592x1944 uint8 ≈ 1.3 ms,
~0.15% 的帧周期) 加一次 emit —— copy 是必须的: 相机缓冲区会被下一次读出复用,
GUI 拿着的数组不能被改写。归一化、降采样与分析全部留在 GUI 线程, 见
``gui.slm.transfer_desktop.live_view``。

观察者抛异常绝不能让会话失败: 与 ``workflow._notify`` 对 ``on_stage`` 的策略一致,
这里只记一条 warning。journal 与落盘的采集始终是事实来源。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

__all__ = [
    "block_frame_payload",
    "capture_frames",
    "emit_block_frame",
    "frame_payload",
    "notify_frame",
    "remember_capture",
]


#: Captures whose raw frames are still in memory, newest last.
#:
#: The anchored path re-opened the npz to take its mean (~145 ms per 46 MB capture
#: here, measured 2026-10-29). Keeping the array for the few blocks that still
#: need it removes a disk round trip without changing what is written or what any
#: later consumer reads -- the file remains the record of the measurement.
_CAPTURE_FRAMES: "OrderedDict[str, np.ndarray]" = OrderedDict()
#: Four covers the live needs (an anchored block reads back only its own A1;
#: the paired direction probe reads the three captures it just took) and holds
#: ~120 MB of uint16 frames rather than a session's worth.
_CAPTURE_FRAMES_MAX = 4


def remember_capture(file_name: str, raw: np.ndarray) -> None:
    """Keep one capture's raw frames available to :func:`capture_frames`."""
    _CAPTURE_FRAMES[file_name] = raw
    _CAPTURE_FRAMES.move_to_end(file_name)
    while len(_CAPTURE_FRAMES) > _CAPTURE_FRAMES_MAX:
        _CAPTURE_FRAMES.popitem(last=False)


def capture_frames(directory: Path, file_name: str) -> np.ndarray:
    """Raw frames of a capture: from memory when this process still has them.

    Falls back to the npz so a resumed run (a fresh process) still reads the
    measurement it did not take.
    """
    cached = _CAPTURE_FRAMES.get(file_name)
    if cached is not None:
        return cached
    with np.load(Path(directory) / file_name) as payload:
        return payload["raw_frames"]


def _raw_2d(frame) -> np.ndarray:
    """Coerce one camera frame to a copied 2-D array.

    The copy is the thread-safety contract, not an optimisation: the camera
    reuses its buffer for the next read, and the GUI keeps this array after the
    call returns.
    """
    raw = np.asarray(frame)
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"live frame must be 2-D, got shape {raw.shape}")
    return raw.copy()


def frame_payload(
    *,
    frame,
    label: str = "",
    phase: str = "validate",
    shape: str = "",
    block: int = 0,
    block_index: int = 0,
    blocks_total: int = 0,
    iteration: int = 0,
    iterations_total: int = 0,
    frame_index: int = 1,
    frames_total: int = 1,
    remaining: int = 0,
    xy=None,
    phase_grid: Any = None,
    predicted: Any = None,
    target: Any = None,
    hdr: bool = False,
    hdr_image: Any = None,
    selfcheck: Any = None,
    zero_order: Any = None,
    ccd_profile: Any = None,
) -> dict:
    """One live refresh: a single CCD read, plus whatever progress is known.

    ``phase`` (a stage name str) says which stage produced the read --
    ``"capture"`` (探针族), ``"exposure"`` (曝光阶梯, 逐档亮度变化是预期的) or
    ``"validate"`` (锚定块). The GUI labels the frame with it so an exposure
    sweep is not mistaken for a wavefront change.

    Optional ``phase_grid`` (current SLM phase mask, e.g. 128x128) and
    ``predicted`` (far-field intensity from ``forward.predict(phase_grid)`` on the
    camera sampling grid) let the GUI show the three-panel live view. Both are
    ``None``-safe and copied so a reused backing buffer is never rewritten.

    Deliberately no ``profile_x`` / ``profile_y``: computing those means cost
    8.7 ms per frame on the measurement thread and ``LiveView`` re-derives them
    from the image anyway. Per-frame emission is the whole point here, so the
    payload stays at one copy and a handful of ints.
    """
    return dict(
        frame=_raw_2d(frame),
        xy=None if xy is None else np.array(xy, copy=True),
        label=str(label),
        phase=str(phase),
        shape=str(shape),
        block=int(block),
        block_index=int(block_index),
        blocks_total=int(blocks_total),
        iteration=int(iteration),
        iterations_total=int(iterations_total),
        frame_index=int(frame_index),
        frames_total=int(frames_total),
        remaining=int(remaining),
        phase_grid=None if phase_grid is None else np.array(phase_grid, copy=True),
        predicted=None if predicted is None else np.array(predicted, copy=True),
        target=None if target is None else np.array(target, copy=True),
        hdr=bool(hdr),
        hdr_image=None if hdr_image is None else np.array(hdr_image, copy=True),
        selfcheck=None if selfcheck is None else dict(selfcheck),
        zero_order=None if zero_order is None else dict(zero_order),
        ccd_profile=None if ccd_profile is None else dict(ccd_profile),
    )


def block_frame_payload(
    *,
    frame,
    shape: str,
    block: int,
    block_index: int,
    blocks_total: int,
    iteration: int,
    iterations_total: int,
    remaining: int,
    xy=None,
    label: str = "",
    target: Any = None,
    hdr: bool = False,
    hdr_image: Any = None,
    selfcheck: Any = None,
    zero_order: Any = None,
    ccd_profile: Any = None,
) -> dict:
    """Plain-data payload for one anchored block's last frame.

    Kept alongside :func:`frame_payload` because a block also carries the
    optimizer progress (``block_index`` / ``iteration``) that a bare CCD read
    cannot know. ``remaining`` counts optimizer iterations until the next block.
    """
    return dict(
        frame=_raw_2d(frame),
        xy=None if xy is None else np.array(xy, copy=True),
        profile_x=np.asarray(frame).mean(axis=0),
        profile_y=np.asarray(frame).mean(axis=1),
        shape=str(shape),
        block=int(block),
        block_index=int(block_index),
        blocks_total=int(blocks_total),
        iteration=int(iteration),
        iterations_total=int(iterations_total),
        frame_index=1,
        frames_total=1,
        remaining=int(remaining),
        label=str(label),
        phase="validate",
        phase_grid=None,
        predicted=None,
        target=None if target is None else np.array(target, copy=True),
        hdr=bool(hdr),
        hdr_image=None if hdr_image is None else np.array(hdr_image, copy=True),
        selfcheck=None if selfcheck is None else dict(selfcheck),
        zero_order=None if zero_order is None else dict(zero_order),
        ccd_profile=None if ccd_profile is None else dict(ccd_profile),
    )


def notify_frame(on_frame, payload: dict) -> None:
    """Best-effort live observer, mirroring ``workflow._notify`` for stages.

    A raising GUI hook must never abort a measured session: the journal and the
    captures remain the source of truth, the exception is only logged.
    """
    if on_frame is None:
        return
    try:
        on_frame(payload)
    except Exception as exc:  # a UI hook must never abort a measured session
        logger.warning("transfer live frame hook failed: {}", exc)


def _capture_target(directory: Path, file_name: str):
    """Target mask from a capture npz (``target`` field), or None if absent.

    The live block payload wants the same target the block was optimised
    against; the npz stores it next to the frames so a resumed run can still
    show the contour. A missing field is tolerated (older captures).
    """
    try:
        with np.load(Path(directory) / file_name) as payload:
            if "target" not in payload:
                return None
            return np.array(payload["target"], copy=True)
    except Exception:  # policy: 存档 npz 可能缺失或损坏，返回 None 让 UI 显示「无目标」
        return None


def emit_block_frame(
    on_frame,
    *,
    directory: Path,
    capture_row: dict | None,
    shape: str,
    block: int,
    block_index: int,
    blocks_total: int,
    iteration: int,
    iterations_total: int,
    remaining: int,
    xy=None,
    target=None,
) -> None:
    """Build and deliver one block's last-frame payload. Never raises.

    Runs once per block, after the block's acceptance / anchor capture, and only
    when an observer was supplied. The per-frame reads inside the block were
    already delivered by :func:`frame_payload`; this one adds the block's
    optimizer progress to the same measurement. ``target`` is the target mask;
    when omitted it is loaded from the capture npz.
    """
    if on_frame is None or capture_row is None:
        return
    try:
        raw = capture_frames(directory, capture_row["file"])
        if target is None:
            target = _capture_target(directory, capture_row["file"])
        payload = block_frame_payload(
            frame=raw[-1],
            xy=xy,
            shape=shape,
            block=block,
            block_index=block_index,
            blocks_total=blocks_total,
            iteration=iteration,
            iterations_total=iterations_total,
            remaining=remaining,
            label=str(capture_row.get("label", "")),
            target=target,
        )
    except Exception as exc:
        # policy: 块级画面只是观察者; 读不到帧 (缓存被逐出 / npz 缺失) 就跳过这一条
        # 刷新, 记一条 warning。测量与落盘的采集仍是事实来源, 不能因为 GUI 少一帧
        # 就让整场会话失败。
        logger.warning("transfer live block frame skipped: {}", exc)
        return
    notify_frame(on_frame, payload)


def _resolve_observer(args, on_frame):
    """Explicit keyword wins; otherwise honour ``args.on_frame`` if present.

    Two call shapes exist (a direct ``run(args, on_frame=...)`` and a forwarded
    ``SimpleNamespace``), so both must reach the same observer.
    """
    return on_frame if on_frame is not None else getattr(args, "on_frame", None)
