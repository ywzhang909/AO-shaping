"""Offline tests for the controller-layer send-overlay proxy (gui/utils/phase_overlay).

Upstream (acb97b1f) implements the defocus / base-phase overlay as *driver
slots* applied inside ``Santec.create_phase_from_array``. The optimizer,
however, closes the GUI handle and opens its *own* fresh ``Santec`` during
closed-loop shaping, so those slots are silently lost on that path. This module
adds a controller-layer transparent proxy :class:`OverlaidSantec` plus the
single dispatch point :func:`apply_overlays_to_handle`.

Load-bearing properties — all fail silently when broken:

1. **bit-equivalence** — the proxy's ``create_phase_from_array`` matches the
   driver-slot reference pipeline (resize -> +defocus -> rad->gray -> +base ->
   mod) exactly, so an operator sees no change to the on-panel bytes.
2. **single-overlay invariant** — :func:`apply_overlays_to_handle` writes the
   proxy layer for a proxy (clearing its driver slot) or the driver slot for a
   raw handle, so a single send never stacks the layer twice.
3. **display_phase interception** — ``display_phase`` must route through the
   proxy's ``create_phase_from_array`` (with overlay) rather than the raw
   driver's (without), else the overlay is lost on the display path.
4. **transparency** — every other attribute/method/dunder passes through to the
   wrapped driver; ``copy`` / ``deepcopy`` / ``__getstate__`` do not recurse.

No Streamlit server, no hardware: the Santec SDK (``_slm_win``) is stubbed in
``sys.modules`` (the driver imports it at module load and would call
``ctypes.WinDLL`` on non-Windows), and ``Santec.create_phase_from_array`` is a
pure-numpy pipeline that never touches the device.
"""

from __future__ import annotations

import copy
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.drivers.slm.santec.wavefront_correction import (
    WavefrontCorrection,
)
from ao_shaping.gui.utils.phase_overlay import (
    PhaseOverlay,
    OverlaidSantec,
    apply_overlays_to_handle,
)

MG = Santec.MAX_GRAYSCALE_VALUE
H, W = Santec.Panel_Res[1], Santec.Panel_Res[0]
PANEL = (H, W)
RNG = np.random.default_rng(0)


def _defo() -> np.ndarray:
    a = np.zeros(PANEL)
    a[H // 2, W // 2] = 0.7
    return a


def _base() -> np.ndarray:
    a = np.zeros(PANEL, dtype=np.uint16)
    a[0, 0] = 100
    a[-1, -1] = 50
    return a


def _reference(small: np.ndarray, defo: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Driver-slot reference: resize -> +defo -> nan_to_num -> rad->gray -> +base -> mod."""
    resized = WavefrontCorrection.resize_to_panel(small, (W, H))
    ref = resized + defo
    np.nan_to_num(ref, copy=False, nan=0)
    ref_gray = np.mod(ref / (2 * np.pi) * MG, MG) + base
    return np.mod(ref_gray, MG).astype(np.uint16)


def test_proxy_matches_driver_slot_reference() -> None:
    small = RNG.uniform(0, 2 * np.pi, size=(H // 2, W // 2))
    defo, base = _defo(), _base()
    proxy = OverlaidSantec(Santec(slm_number=1), PhaseOverlay(defo, base))
    out = proxy.create_phase_from_array(small)
    ref = _reference(small, defo, base)
    diff = np.abs(out.astype(int) - ref.astype(int))
    assert diff.max() <= 1, f"proxy not bit-equivalent to driver-slot path: max {diff.max()}"


def test_empty_overlay_is_a_noop() -> None:
    small = RNG.uniform(0, 2 * np.pi, size=PANEL)
    raw = Santec(slm_number=1)
    proxy = OverlaidSantec(raw, PhaseOverlay())
    assert proxy.create_phase_from_array(small).shape == PANEL
    # No overlay -> identical to the raw driver's own transform.
    np.testing.assert_array_equal(proxy.create_phase_from_array(small),
                                  raw.create_phase_from_array(small))


class TestApplyOverlaysToHandle:
    """``apply_overlays_to_handle`` enforces the single-overlay invariant."""

    def test_proxy_writes_proxy_layer_and_clears_driver_slot(self) -> None:
        raw = Santec(slm_number=1)
        raw.set_phase_overlay(np.full(PANEL, 0.5))
        raw.set_base_phase(np.full(PANEL, 7, dtype=np.uint16))
        proxy = OverlaidSantec(raw, PhaseOverlay())
        defo, base = _defo(), _base()
        apply_overlays_to_handle(proxy, PhaseOverlay(defo, base))
        # proxy layer set
        assert np.allclose(proxy.get_phase_overlay(), defo)
        assert np.array_equal(proxy.get_base_phase(), base)
        # driver slot cleared (invariant: proxy path overlays only via proxy)
        assert raw.get_phase_overlay() is None
        assert raw.get_base_phase() is None

    def test_raw_writes_driver_slot(self) -> None:
        raw = Santec(slm_number=2)
        defo, base = _defo(), _base()
        apply_overlays_to_handle(raw, PhaseOverlay(defo, base))
        assert np.allclose(raw.get_phase_overlay(), defo)
        assert np.array_equal(raw.get_base_phase(), base)

    def test_none_overlay_clears_both(self) -> None:
        raw = Santec(slm_number=3)
        raw.set_phase_overlay(np.full(PANEL, 0.5))
        raw.set_base_phase(np.full(PANEL, 7, dtype=np.uint16))
        apply_overlays_to_handle(raw, None)
        assert raw.get_phase_overlay() is None
        assert raw.get_base_phase() is None

    def test_none_handle_is_a_noop(self) -> None:
        apply_overlays_to_handle(None, PhaseOverlay(_defo(), _base()))  # must not raise

    def test_handle_without_setters_is_a_noop(self) -> None:
        # A sim SLM (SimSLMPib) exposes no set_phase_overlay/set_base_phase, so the
        # optimizer applying an overlay to it must be a silent no-op, not an error
        # (the hasattr guard). This is what keeps the sim closed loop untouched.
        class _NoSetters:
            pass  # deliberately has no set_phase_overlay / set_base_phase

        apply_overlays_to_handle(_NoSetters(), PhaseOverlay(_defo(), _base()))  # must not raise


def test_display_phase_is_intercepted() -> None:
    calls: dict = {}

    class FakeRaw(Santec):
        def display_data(self, phase_gray, wait_time_s=None,
                         memory_number=None, memory_mode=0):
            calls["gray"] = phase_gray
            return 1

    fake = FakeRaw(slm_number=1)
    proxy = OverlaidSantec(fake, PhaseOverlay(_defo(), _base()))
    small = RNG.uniform(0, 2 * np.pi, size=(H // 2, W // 2))
    assert proxy.display_phase(small) == 1
    plain = fake.create_phase_from_array(small)  # no overlay
    # The gray handed to display_data must carry the overlay (differs from plain).
    assert not np.array_equal(calls["gray"], plain)
    np.testing.assert_array_equal(calls["gray"], proxy.create_phase_from_array(small))


def test_set_get_aliases_write_proxy_layer_not_driver_slot() -> None:
    raw = Santec(slm_number=1)
    proxy = OverlaidSantec(raw, PhaseOverlay())
    defo, base = _defo(), _base()
    proxy.set_phase_overlay(defo)
    proxy.set_base_phase(base)
    # written to the proxy layer, not the driver slot
    assert np.allclose(proxy.get_phase_overlay(), defo)
    assert np.array_equal(proxy.get_base_phase(), base)
    assert raw.get_phase_overlay() is None
    assert raw.get_base_phase() is None
    # clearing
    proxy.set_phase_overlay(None)
    proxy.set_base_phase(None)
    assert proxy.get_phase_overlay() is None
    assert proxy.get_base_phase() is None
    assert proxy.overlay.empty


def test_transparency_and_copy() -> None:
    raw = Santec(slm_number=1)
    proxy = OverlaidSantec(raw, PhaseOverlay(_defo(), _base()))
    assert proxy.slm_number == raw.slm_number
    assert proxy.is_open == raw.is_open
    assert proxy.overlay is proxy.overlay
    # copy / deepcopy must not recurse
    c = copy.copy(proxy)
    assert isinstance(c, OverlaidSantec)
    d = copy.deepcopy(proxy)
    assert isinstance(d, OverlaidSantec)
    assert isinstance(proxy.__getstate__(), dict)


def test_context_manager_delegates_to_raw_driver() -> None:
    # The optimizer uses its SLM handle as ``with slm:`` (Santec.__enter__ opens
    # the device, __exit__ closes it). The proxy must delegate lifecycle to the
    # wrapped driver, else the closed-loop entry points break with TypeError.
    events: list[str] = []

    class FakeRaw(Santec):
        def open(self) -> None:  # hardware stub
            events.append("open")

        def close(self) -> None:  # hardware stub
            events.append("close")

    raw = FakeRaw(slm_number=1)
    proxy = OverlaidSantec(raw)
    with proxy as p:
        assert p is raw  # Santec.__enter__ returns self -> proxy returns raw
        events.append("body")
    assert events == ["open", "body", "close"]
    with proxy:
        pass  # open() twice stays a no-op in this stub; must not raise
