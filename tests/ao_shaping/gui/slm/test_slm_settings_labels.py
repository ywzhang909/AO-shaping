"""Unit tests for the pure label helpers behind the SLM settings sidebar.

Issue #167 splits ``_render_slm_settings`` into per-section render functions.
The parts that are pure text formatting -- no Streamlit calls -- are pulled out
so they can be tested directly with plain pytest (per the Streamlit guidance:
pure logic does not need ``AppTest``).
"""

from __future__ import annotations

from ao_shaping.gui.slm.multi_slm_controller import (
    _slm_pitch_label,
    _slm_resolution_label,
    _slm_serial_label,
)


class _StubSLM:
    def __init__(self, serial: str) -> None:
        self._serial_number = serial


def test_resolution_label_joins_width_and_height() -> None:
    assert _slm_resolution_label(1920, 1200) == "分辨率: 1920×1200"


def test_pitch_label_without_clear_aperture() -> None:
    assert _slm_pitch_label(8.0, None) == "像素间距: 8.0 μm"


def test_pitch_label_appends_clear_aperture() -> None:
    """有效口径 and 间距 are different physical quantities -- both are shown."""
    assert _slm_pitch_label(8.0, 7.8) == "像素间距: 8.0 μm · 有效口径: 7.8 μm"


def test_serial_label_falls_back_to_dash() -> None:
    assert _slm_serial_label(_StubSLM("SN-0001")) == "SLM序列号: SN-0001"
    assert _slm_serial_label(_StubSLM("")) == "SLM序列号: -"
