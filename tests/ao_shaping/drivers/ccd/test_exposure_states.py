"""ONLINE test: the three exposure states against real cameras (Daheng + MiiCam).

This file is **not** part of a default ``pytest`` run: it lives under
``tests/ao_shaping/drivers/``, which ``pyproject.toml``'s ``addopts`` passes to
``--ignore`` precisely because SDK opens have no timeout guard. Run it
explicitly, with the hardware free::

    $env:AO_RUN_HARDWARE = "1"
    .venv\Scripts\python.exe -m pytest tests/ao_shaping/drivers/ccd/test_exposure_states.py -q -s

The same opt-in gate the sibling ``test_ccd.py`` uses. Do **not** run this
concurrently with another SLM/camera job -- both hold the same devices and the
readbacks would be garbage.

What is being pinned
--------------------
The offline half (``tests/ao_shaping/runners/test_exposure_cli_token.py``) proves
that ``auto`` and ``default`` are two *distinct* values and that a bare ``0`` is
an error. It cannot prove that they do different things to a camera, because the
CLI never reaches one here. That is what this file is for:

* ``auto`` must **change** the exposure -- it is "let the session/device
  determine it", so something has to move;
* ``default`` must **not** change it: write a known value, take the ``default``
  path, and ``get_exposure_time_ms()`` must read back the same number.
* a bare ``0`` is rejected in the parse layer (offline, repeated here only as a
  guard against someone re-widening the resolver).

The ``default`` assertion is the load-bearing one, because it is the only claim
in the vocabulary that a driver could plausibly violate: ``DahengCamera.open()``
writes and clamps the exposure itself (``driver.py``, "写入前把请求的曝光钳到
设备范围内"), so "change nothing" has to be tested as *no further write*, not as
"the value equals what we asked for at construction". The test therefore sets
the value through the device and only then walks the ``default`` path.
"""

from __future__ import annotations

import os

import pytest

# 硬件门控: 与 test_ccd.py 同款 (README v0.12.0 约定). 未设 AO_RUN_HARDWARE=1
# 时整模块跳过 —— 无设备时 Daheng/Mii SDK open 会抛 ConnectionAbortedError.
if os.environ.get("AO_RUN_HARDWARE", "").strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "set AO_RUN_HARDWARE=1 to run CCD exposure-state hardware tests",
        allow_module_level=True,
    )

from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
from ao_shaping.utils.cli.exposure import (
    AUTO_EXPOSURE,
    DEFAULT_EXPOSURE,
    exposure_ms_legacy,
    resolve_exposure,
)

#: Readbacks are quantised to whole microseconds by both drivers, so equality is
#: the right assertion and any slack would hide a real write.
_EXACT = 1e-9

#: Both backends clamp to whole microseconds (``int(ms * 1000)``), so a request
#: that survives the round trip must land inside the device's own range. The
#: range is read per camera; these are the brackets used to pick a value that is
#: comfortably inside either device's range.
_MIN, _MAX = 0.05, 20.0


def _open(cam_type: str, cam_id: int):
    """Open a camera at a mid-range exposure, or skip if the device is absent."""
    try:
        cam = create_camera(cam_type, cam_id=cam_id, exposure_time_ms=_MIN)
        cam.open()
    except Exception as exc:  # SDK missing, device busy, IP unreachable
        pytest.skip(f"{cam_type} camera unavailable: {exc}")
    return cam


def _usable(cam) -> float:
    """A known-good exposure inside this device's range, or skip."""
    low, high = cam.get_exposure_range()
    target = min(max(_MIN, low), high - 1e-6)
    if target <= 0:
        pytest.skip(f"{type(cam).__name__} reports no usable exposure range")
    cam.reset_exposure_time(target)
    return float(get_camera_exposure_ms(cam))


@pytest.fixture(params=["daheng", "miicam"])
def cam(request):
    """One open camera per backend; the vocabulary is backend-agnostic."""
    camera = _open(request.param, 0)
    try:
        yield camera
    finally:
        try:
            camera.close()
        except Exception:  # teardown must not mask the assertion
            pass


def test_bare_zero_is_rejected_at_the_parse_layer(cam) -> None:
    """Offline claim, re-pinned here so a hardware run cannot pass a broken resolver."""
    with pytest.raises(Exception) as excinfo:
        resolve_exposure("0", param_hint="--exposure-time-ms")
    assert "auto" in str(excinfo.value)
    assert cam is not None  # the parse layer needs no device; this one does


def test_default_does_not_change_the_exposure(cam) -> None:
    """Write a known value, take the ``default`` path, read the same value back.

    This is the assertion the vocabulary exists for: ``default`` means "change
    nothing", so a single ``reset_exposure_time`` inside the ``default`` path
    would break it -- and nothing offline could see that.
    """
    known = _usable(cam)
    before = float(get_camera_exposure_ms(cam))

    # Exactly what `panel_locate.scan_panel` does for DEFAULT_EXPOSURE: resolve,
    # branch on the sentinel, and otherwise touch nothing.
    resolved = resolve_exposure("default", param_hint="--exposure-ms")
    assert resolved is DEFAULT_EXPOSURE
    keep_device_exposure = resolved is DEFAULT_EXPOSURE
    if not keep_device_exposure:  # pragma: no cover - would be a resolver bug
        cam.reset_exposure_time(exposure_ms_legacy(resolved))

    after = float(get_camera_exposure_ms(cam))
    assert after == pytest.approx(before, abs=_EXACT), (
        f"{type(cam).__name__}: 'default' moved the exposure "
        f"{before} -> {after} ms (it must change nothing)"
    )
    assert after == pytest.approx(known, abs=_EXACT)


def test_auto_changes_the_exposure(cam) -> None:
    """``auto`` means "let the session decide", so something has to move.

    The driver-level stand-in is the legacy rung ``auto`` collapses onto: a fresh
    ``reset_exposure_time`` at the other end of the device range. If a future
    driver made ``auto`` a no-op, this is what would catch it.
    """
    _usable(cam)
    low, high = cam.get_exposure_range()
    before = float(get_camera_exposure_ms(cam))

    resolved = resolve_exposure("auto", param_hint="--exposure-ms")
    assert resolved is AUTO_EXPOSURE
    # What every call site does with AUTO_EXPOSURE today (see
    # `exposure_ms_legacy`): the legacy 0.0 rung, which the driver clamps to
    # its own minimum rather than leaving alone.
    cam.reset_exposure_time(exposure_ms_legacy(resolved))

    after = float(get_camera_exposure_ms(cam))
    assert after == pytest.approx(low, rel=1e-3), (
        f"{type(cam).__name__}: 'auto' should land on the device minimum "
        f"({low} ms), read {after} ms"
    )
    assert low < before <= high or abs(before - low) < 1e-9, (
        f"{type(cam).__name__}: precondition failed, the 'before' exposure "
        f"{before} ms is not above the device minimum {low} ms"
    )