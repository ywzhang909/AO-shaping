"""SLM state is restored after a panel-location probe."""

import pytest
import numpy as np
import json
from contextlib import nullcontext
from types import SimpleNamespace

from ao_shaping.tools.slm.panel import panel_locate
from ao_shaping.tools.slm.panel.panel_locate import (
    PanelLocateParams, _paired_delta, _restore_slm_display, scan_panel,
)


class _SlmStub:
    def __init__(self, slot):
        self.slot = slot
        self.restored = None

    def get_displayed_memory_number(self):
        return self.slot

    def get_current_grayscale(self):
        return 17

    def set_grayscale(self, gray):
        self.restored = ("gray", gray)

    def _display_memory(self, slot):
        self.restored = ("slot", slot)


@pytest.mark.parametrize(
    ("slot", "expected"), [(None, ("gray", 17)), (5, ("slot", 5))]
)
def test_restores_original_display_even_on_error(slot, expected):
    slm = _SlmStub(slot)
    with pytest.raises(RuntimeError, match="probe failed"):
        with _restore_slm_display(slm):
            raise RuntimeError("probe failed")
    assert slm.restored == expected


def test_paired_flat_removes_linear_power_drift():
    box = np.ones((2, 2), dtype=bool)
    delta, drift = _paired_delta(np.full((2, 2), 100.0),
                                 np.full((2, 2), 90.0),
                                 np.full((2, 2), 80.0), box, 0.25)
    assert delta == 0
    assert drift == pytest.approx(2 / 9)


def test_paired_flat_rejects_power_step():
    box = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="drift"):
        _paired_delta(np.ones((2, 2)), np.ones((2, 2)),
                      np.full((2, 2), 0.1), box, 0.2)


@pytest.mark.parametrize("last_flat, expected_error", [(100.0, None), (110.0, "session flat drift")])
def test_scan_reports_session_drift_and_rejects_unstable_run(
    monkeypatch, tmp_path, last_flat, expected_error,
):
    from ao_shaping.drivers.ccd import common as camera_common
    from ao_shaping.drivers.slm import santec

    class Device:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def reset_exposure_time(self, value):
            return value

        def get_numpy_image(self, n_sample: int):
            # Dark raw frame: the ceiling probe walks its whole ladder
            # without saturating and returns a generous ceiling.
            return np.full((8, 8), 10, dtype=np.uint8)

    frames = iter([100.0] * 18 + [last_flat])

    def flat_reference(*_args, **_kwargs):
        value = next(frames)
        return np.full((8, 8), value), SimpleNamespace(
            centroid_x=4, centroid_y=4, as_row=lambda: "flat"
        )

    monkeypatch.setattr(camera_common, "create_camera", lambda *_a, **_k: Device())
    monkeypatch.setattr(santec, "Santec", lambda **_k: Device())
    monkeypatch.setattr(panel_locate, "_restore_slm_display", lambda _slm: nullcontext())
    monkeypatch.setattr(panel_locate, "SLM_PANEL_H", 16)
    monkeypatch.setattr(panel_locate, "SLM_PANEL_W", 16)
    monkeypatch.setattr(panel_locate, "measure_flat_reference", flat_reference)
    monkeypatch.setattr(
        panel_locate, "display_and_average",
        lambda *_a, **_k: np.full((8, 8), 80.0),
    )
    params = PanelLocateParams(
        grid_xs="4,8,12", grid_ys="4,8,12", patch_radius=2,
        exposure_ms=0.02, frames=2, max_flat_drift=0.2,
        debug_dir=str(tmp_path / "scan"),
    )

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            scan_panel(params, strict=True, max_session_drift=0.05)
    else:
        result = scan_panel(params, strict=True, max_session_drift=0.05)
        assert result.center == (12, 12)
        assert result.rejected_count == 0
        assert result.session_drift == 0
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["best_center"] == [12, 12]
    assert summary["session_drift"] == pytest.approx(
        abs(last_flat - 100) / ((last_flat + 100) / 2)
    )
