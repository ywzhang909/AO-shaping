"""SLM 句柄交接（GUI 与优化器 / 标定器谁持有设备）。

为什么要单独成模块
------------------
这一族职责独立: ``shaping_controls`` 是闭环整形页面的编排层, 而在一次运行或一次
侧栏标定期间「把 SLM 交出去、跑完再拿回来」是设备所有权的通用协议 —— 整形运行、
会话标定、多形状 campaign 三条入口都走同一份实现。原先它们挤在
``shaping_controls`` 里, 让那个文件同时承担页面编排与设备句柄生命周期两件事;
主文件已到模块规模棘轮线 (``scripts/check_module_size.py``) 的基线, 因此按职责
整体搬到这里, 行为与文案逐字未改。

不变量
------
- Santec SDK 句柄不可重入: 交出前必须先停后台相位切换线程, 再用 ``finally``
  收回, 崩溃或中止都不会让页面失去设备。
- 整个 ``with`` 体跑在会话级 ``bench_busy`` 标志下, 这是运行期间禁用侧栏光路
  控件的唯一开关。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import streamlit as st
from loguru import logger

from ao_shaping.drivers.slm.santec import Santec, describe_santec_error
from ao_shaping.gui.slm import bench_settings

__all__ = [
    "SlmOwnership",
    "with_slm_ownership",
]

# ---------------------------------------------------------------------------
# Device lifecycle — the optimizer owns the handles while a run is in flight
# ---------------------------------------------------------------------------


def _stop_background_toggle(slm_num: int) -> None:
    """Stop any phase-A/B toggle thread bound to this SLM.

    Lazy import: ``multi_slm_controller`` imports this module at module level,
    so the reverse edge has to be deferred.
    """
    from ao_shaping.gui.slm.multi_slm_controller import _stop_toggle

    _stop_toggle(slm_num)


def _release_slm_handle(slm_num: int) -> None:
    """Close the GUI's SLM handle so the optimizer can open its own.

    Only the target SLM is released; the other SLM (dual layout) stays
    connected.  Session state is cleared the same way ``disconnect_slm`` does
    (minus the rerun), so the sidebar reflects reality during the run.
    """
    prefix = f"slm{slm_num}"
    slm = st.session_state.get(prefix)
    if slm is not None and getattr(slm, "is_open", False):
        try:
            slm.close()
        except Exception as exc:  # policy: must not block the reacquire: the handle slot is cleared either way, so a refusing close cannot be allowed to abort the handover
            logger.exception("SLM {} 整形前释放句柄失败: {}", slm_num, exc)
    st.session_state[prefix] = None
    st.session_state[f"{prefix}_connected"] = False
    st.session_state[f"{prefix}_phase_preview"] = None
    st.session_state[f"{prefix}_phase_source"] = "闭环整形运行中"


def _reacquire_slm_handle(slm_num: int) -> Any:
    """Re-open the SLM after the optimizer closed its own handle.

    ``Santec.open()`` reloads the per-serial config file itself (wavelength,
    shift, correction), so the reacquired device carries the same settings the
    sidebar had before the run.  The fresh :class:`Santec` is wrapped in the
    controller-layer overlay proxy (same as ``connect_slm``) so the send
    pipeline overlay (defocus + base phase) is carried on the session handle,
    and the proxy is both stored and returned (the session holds exactly what
    the caller gets back).
    """
    from ao_shaping.gui.slm.multi_slm_controller import _sync_slm_session_state
    from ao_shaping.gui.utils.phase_overlay import OverlaidSantec

    prefix = f"slm{slm_num}"
    try:
        raw = Santec(slm_number=slm_num)
        raw.open()
    except Exception as exc:  # 保留宽泛捕获: 内部调用链 Santec 的失败类型不可枚举
        logger.exception("SLM {} 整形后重新连接失败: {}", slm_num, exc)
        # policy: 与侧边栏「连接」按钮同一套提示: 说明是设备被占用还是 USB 断链,
        # 而不是把原始错误码直接甩给操作员。
        st.error(
            f"SLM {slm_num} 整形后重新连接失败。\n\n"
            f"{describe_santec_error(exc)}\n\n"
            "请在侧边栏手动重连。"
        )
        return None
    slm = OverlaidSantec(raw)
    st.session_state[prefix] = slm
    st.session_state[f"{prefix}_connected"] = True
    _sync_slm_session_state(slm_num, slm)
    return slm


@dataclass
class SlmOwnership:
    """The SLM-handle handover slot yielded by :func:`with_slm_ownership`.

    Inside the ``with`` body the GUI owns **no** handle (``slm`` is ``None`` by
    construction — the optimizer or the calibrator has it).  On exit the
    re-acquired handle lands in :attr:`slm`, so the caller can adopt a phase
    afterwards — the only point at which the GUI owns the device again.

    ``st.session_state`` lives on the ``Streamlit`` object, not on this module,
    so the tests can swap in a plain dict and drive the whole handover offline.
    """

    slm_num: int
    slm: Any = None


@contextmanager
def with_slm_ownership(
    slm_num: int, slot: SlmOwnership | None = None
) -> Iterator[SlmOwnership]:
    """Hand the SLM to whoever needs it for the whole body, then take it back.

    Three steps, in this order:

    1. stop the background phase toggle (it would fight for the panel);
    2. release the GUI's handle — Santec SDK handles are not re-entrant, and
       the optimizers *and* the session calibrator open their own;
    3. ``finally``: re-acquire, so a crash or an abort cannot strand the page
       without a device.

    The whole body also runs under the session ``bench_busy`` flag, which is what
    disables the sidebar optical-path controls while a run or a calibration is
    in flight.

    Both entry points share it: :func:`shaping_session` (a shaping run) and the
    sidebar calibration button.  ``slot`` is optional, so a caller can hold the
    reference *before* entering — that is what lets the ``finally`` of
    :func:`shaping_session` read :attr:`SlmOwnership.slm` even when step 1 or 2
    raised.  Exceptions from the body propagate untouched — surfacing them once
    is the caller's job.
    """
    owned = slot if slot is not None else SlmOwnership(slm_num)
    _stop_background_toggle(slm_num)
    bench_settings.set_bench_busy(True)
    _release_slm_handle(slm_num)
    try:
        yield owned
    finally:
        # Nested, not sequential: a re-acquire that raised would otherwise leave
        # ``bench_busy`` stuck True, which disables the sidebar optical-path
        # controls for the rest of the session with no way to clear them.
        try:
            owned.slm = _reacquire_slm_handle(slm_num)
        finally:
            bench_settings.set_bench_busy(False)
