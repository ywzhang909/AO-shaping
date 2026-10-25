"""Offline proof for :meth:`ThorlabWFS._ensure_open` (R-64 / WFS device guard).

``ThorlabWFS`` had **no** open-state guard at all: every public method passed
``self._instrument_handle`` straight to ``WFS_64.dll``.  With a null handle the
DLL does not reliably fail -- ``select_mla`` discards its return codes and the
auto-exposure branch of ``take_image`` ignores the error code -- so an unopened
call returned garbage instead of raising.  The only thing standing between a
caller and that garbage was ``assert wfs.is_connected()`` written in each call
site, and ``assert`` is stripped wholesale under ``python -O``.

The guard lives in the driver (mechanism), not in the caller (policy): see
``tests/ao_shaping/drivers/ccd/test_ensure_open.py`` for the same paradigm on
the camera side and ``Santec._ensure_open`` on the SLM side.

No hardware and no vendor SDK: construction is offline by design (the DLL load
is deferred to the ``_lib`` property), and ``_lib`` is patched here with an
object that explodes on *any* attribute access, so a test failure distinguishes
"guard fired" (``WfsError``) from "guard missing, reached the DLL".
"""

from __future__ import annotations

from ctypes import c_ulong

import pytest

from ao_shaping.drivers import MlaRes, ThorlabWFS
from ao_shaping.drivers.wfs.thorlab_wfs import WfsError


class _ExplodingLib:
    """Stand-in for the DLL handle: touching it at all is a test failure."""

    def __getattr__(self, name: str):
        raise AssertionError(f"driver reached the DLL through WFS_{name}")


@pytest.fixture()
def wfs(monkeypatch: pytest.MonkeyPatch) -> ThorlabWFS:
    """A real driver instance that was never opened, with the DLL booby-trapped."""
    monkeypatch.setattr(ThorlabWFS, "_lib", property(lambda self: _ExplodingLib()))
    return ThorlabWFS(MlaRes.Res768)


def test_construction_never_touches_the_dll(wfs: ThorlabWFS) -> None:
    """The lazy-DLL contract: building an instance must not load anything."""
    assert wfs.is_connected() is False


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda w: w.select_mla(MlaRes.Res768), id="select_mla"),
        pytest.param(lambda w: w.take_image(), id="take_image"),
        pytest.param(lambda w: w.get_spotfiled_image(), id="get_spotfiled_image"),
        pytest.param(lambda w: w.get_spots_statics(), id="get_spots_statics"),
        pytest.param(lambda w: w.get_wavefront(), id="get_wavefront"),
        pytest.param(lambda w: w.get_zernike(), id="get_zernike"),
        pytest.param(lambda w: w.get_spot_deviation(), id="get_spot_deviation"),
        pytest.param(lambda w: w.optimize_pupil(), id="optimize_pupil"),
        pytest.param(lambda w: w.set_ref_plane(False), id="set_ref_plane"),
        pytest.param(lambda w: w.build_subaperture_mask(), id="build_subaperture_mask"),
        pytest.param(lambda w: w.get_mla_name(), id="get_mla_name"),
        pytest.param(lambda w: w.get_exposure_time_range(), id="get_exposure_time_range"),
        pytest.param(lambda w: w.exposure_time, id="exposure_time_get"),
        pytest.param(lambda w: setattr(w, "exposure_time", 5.0), id="exposure_time_set"),
        pytest.param(lambda w: w.pupil, id="pupil_get"),
        pytest.param(lambda w: setattr(w, "pupil", (0.0, 0.0, 2.0, 2.0)), id="pupil_set"),
        pytest.param(lambda w: w.high_speed, id="high_speed_get"),
        pytest.param(lambda w: setattr(w, "high_speed", True), id="high_speed_set"),
    ],
)
def test_unopened_call_raises_before_reaching_the_dll(
    wfs: ThorlabWFS, call
) -> None:
    """Every public DLL entry point refuses a null handle with ``WfsError``."""
    with pytest.raises(WfsError, match="not open"):
        call(wfs)


def test_guard_accepts_a_live_handle_while_still_connecting(
    wfs: ThorlabWFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard keys on the handle, **not** on ``DeviceState.READY``.

    ``open()`` fills the handle and then calls ``select_mla`` / ``set_ref_plane``
    / ``optimize_exposure_time_and_gain`` / ``exposure_time`` / ``high_speed`` /
    ``pupil`` while the state is still ``CONNECTING``; ``READY`` is set only after
    they return.  A guard requiring ``is_connected()`` would make ``open()``
    refuse itself, so this pins the deliberate distinction.
    """
    wfs._instrument_handle = c_ulong(0x1234)
    assert wfs._state is not None and not wfs.is_connected()
    wfs._ensure_open()


def test_out_of_range_order_raises_after_the_guard(wfs: ThorlabWFS) -> None:
    """Parameter boundaries stay explicit raises too, never stripped asserts."""
    wfs._instrument_handle = c_ulong(0x1234)
    # get_zernike carries @require_take_image: with no captured frame the
    # decorator auto-captures first, so the check below would never be reached.
    wfs._image_captured = True
    with pytest.raises(WfsError, match="zernike order"):
        wfs.get_zernike(zernike_order=11)
