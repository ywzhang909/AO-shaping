"""Preserve the SLM display while a bench probe uses temporary memory slots."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def slm_self_check(slm) -> dict:
    """开机后立刻核对 SLM 处于干净状态, 返回可进 ``summary.json`` 的回读记录。

    ``get_current_grayscale()`` 必须读到 0 (平场/关), 否则相位下发没有意义。
    ``get_displayed_memory_number()`` 抛 ``SantecError`` 且 ``code == 1`` 在
    grayscale 模式下是**正常**的 (没有 slot); ``get_displayed_phase()`` 只是驱动侧
    写缓存, 不是设备回读, 读失败不影响自检结论。

    Args:
        slm: 已 ``open()`` 的 SLM 驱动句柄。

    Returns:
        ``{"grayscale", "memory_slot", "displayed_phase_source"}``; 驱动缺某个
        getter 时对应项为 ``None``。

    Raises:
        RuntimeError: 灰度回读不是 0。
    """
    from ao_shaping.drivers.slm.santec import SantecError

    get_gray = getattr(slm, "get_current_grayscale", None)
    if get_gray is None:
        gray = None
    else:
        gray = float(get_gray())
        if gray != 0.0:
            raise RuntimeError(f"SLM grayscale not 0 after open (read {gray})")

    slot = None
    get_slot = getattr(slm, "get_displayed_memory_number", None)
    if get_slot is not None:
        try:
            slot = get_slot()
        except SantecError as exc:
            if exc.code != 1:
                raise
            slot = None  # grayscale mode: no memory slot (normal)

    phase_source = None
    get_phase = getattr(slm, "get_displayed_phase", None)
    if get_phase is not None:
        try:
            _, phase_source = get_phase()
        except Exception as exc:  # cache read is best-effort
            phase_source = f"unavailable: {exc}"

    return {
        "grayscale": gray,
        "memory_slot": slot,
        "displayed_phase_source": phase_source,
    }


@contextmanager
def preserve_slm_display(slm) -> Iterator[None]:
    """Restore either the displayed memory slot or grayscale mode on exit."""
    from ao_shaping.drivers.slm.santec import SantecError

    try:
        original_slot = slm.get_displayed_memory_number()
    except SantecError as exc:
        if exc.code != 1:
            raise
        original_slot = None
    original_gray = slm.get_current_grayscale() if original_slot is None else None
    try:
        yield
    finally:
        if original_slot is None:
            slm.set_grayscale(original_gray)
        else:
            slm._display_memory(original_slot)
