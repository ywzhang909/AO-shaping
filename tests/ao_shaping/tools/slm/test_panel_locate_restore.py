"""SLM state is restored after a panel-location probe."""

import pytest
import numpy as np
import json
import dataclasses
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


@pytest.mark.parametrize("last_flat, expected_error", [(100.0, None), (110.0, None)])
def test_scan_reports_session_drift_and_rejects_unstable_run(
    monkeypatch, tmp_path, last_flat, expected_error,
):
    """Intensity drift across the session is **reported**, not enforced.

    Changed 2026-10-17 after the real run refused with *"session flat drift
    7.9% exceeds 5.0%"* while the flat peak had moved 63 -> 90 over the session
    and the centroid moved <1 px. Every grid point normalises its own reading
    against the flat measured immediately before and after it
    (:func:`_paired_delta`), so a session-wide power change cannot bias the
    located centre; what invalidates a locate is the beam **moving**. The gate
    moved to :func:`test_scan_rejects_a_beam_that_moved`.

    The 110% case is kept as a case that must now *pass* -- that is the whole
    point: it used to be the failure that blocked the bench.
    """
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
        assert result.session_shift_px == 0.0
        assert result.session_drift == pytest.approx(
            abs(last_flat - 100) / ((last_flat + 100) / 2)
        )
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["best_center"] == [12, 12]
    assert summary["session_drift"] == pytest.approx(
        abs(last_flat - 100) / ((last_flat + 100) / 2)
    )
    assert summary["session_shift_px"] == 0.0


def test_the_spot_measurement_field_name_is_what_the_scanner_reads():
    """A hand-made stub mirrored my own wrong attribute name and hid the bug.

    The scanner reads ``SpotMeasurement.fwhm_px``. An earlier version read
    ``.fwhm``, which does not exist, so ``getattr(..., None)`` returned ``None``
    and the spot-scaled limit silently never engaged -- while the test's
    ``SimpleNamespace(fwhm=...)`` matched the mistake and stayed green.
    """
    from ao_shaping.utils.image.frame_analysis import SpotMeasurement

    names = {f.name for f in dataclasses.fields(SpotMeasurement)}
    assert "fwhm_px" in names
    assert "fwhm" not in names


@pytest.mark.parametrize(
    ("fwhm", "shift", "should_pass"),
    [
        # Broad, dim spot: the centroid's own wander is several px, so a shift
        # that is a small fraction of the spot must not refuse the scan.
        (62.9, 3.47, True),
        # On a tight spot the same absolute shift buys a much smaller budget
        # (0.5 x 13 = 6.5 px), so 10 px IS a real move.
        (13.0, 10.0, False),
        # Beyond half the spot width is a move at any scale.
        (62.9, 40.0, False),
    ],
)
def test_the_shift_limit_scales_with_the_spot_size(
    monkeypatch, tmp_path, fwhm, shift, should_pass
):
    """A fixed pixel limit cannot serve both a 13 px and a 63 px spot.

    Measured 2026-10-17: a 3.47 px session shift on a FWHM 62.9 px spot at
    peak 18/255 was refused by an absolute 3 px limit, although the scan's own
    answer (`best_center 960,980`) beat the runner-up by 9% -- far more than a
    3 px move could explain. The spot size is the scale that makes the limit
    mean the same thing on both benches.
    """
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
            return np.full((8, 8), 10, dtype=np.uint8)

    calls = {"n": 0}

    def flat_reference(*_args, **_kwargs):
        calls["n"] += 1
        dx = 0.0 if calls["n"] <= 9 else float(shift)
        return np.full((8, 8), 100.0), SimpleNamespace(
            centroid_x=4.0 + dx,
            centroid_y=4.0,
            fwhm_px=float(fwhm),
            as_row=lambda: "flat",
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

    if should_pass:
        result = scan_panel(
            params, strict=True, max_session_drift=0.05,
            max_session_shift_px=3.0, max_session_shift_fwhm=0.5,
        )
        assert result.session_shift_px == pytest.approx(shift, abs=0.1)
        summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
        assert summary["session_shift_limit_px"] == pytest.approx(
            max(3.0, 0.5 * fwhm)
        )
    else:
        with pytest.raises(ValueError, match="beam shift"):
            scan_panel(
                params, strict=True, max_session_drift=0.05,
                max_session_shift_px=3.0, max_session_shift_fwhm=0.5,
            )


def test_scan_rejects_a_beam_that_moved(monkeypatch, tmp_path):
    """The enforced session criterion: the centroid must not walk.

    A flat whose intensity is perfectly steady but whose centroid drifts 10 px
    describes no single centre, so the locate must refuse -- the opposite
    verdict to the intensity case above.
    """
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
            return np.full((8, 8), 10, dtype=np.uint8)

    # First flats sit at x=4, later ones at x=14: a pure translation.
    calls = {"n": 0}

    def flat_reference(*_args, **_kwargs):
        calls["n"] += 1
        cx = 4 if calls["n"] <= 9 else 14
        return np.full((8, 8), 100.0), SimpleNamespace(
            centroid_x=cx, centroid_y=4, as_row=lambda: "flat"
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

    with pytest.raises(ValueError, match="beam shift"):
        scan_panel(params, strict=True, max_session_drift=0.05, max_session_shift_px=3.0)

    # The same scan passes when the shift gate is wide enough to admit it.
    calls["n"] = 0
    result = scan_panel(
        params, strict=True, max_session_drift=0.05, max_session_shift_px=20.0
    )
    assert result.session_shift_px == pytest.approx(10.0, abs=0.1)
