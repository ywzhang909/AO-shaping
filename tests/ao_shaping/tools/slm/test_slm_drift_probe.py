"""Offline tests for the flat-field drift + exposure-ladder probe.

The pure analysis functions take a ``capture() -> frame`` callable, so the whole
probe is testable with synthetic frames and no hardware. Only :func:`main`
touches a device.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_drift_probe import (
    LADDER_DEFAULT,
    ROW_KEYS,
    drift_series,
    exposure_ladder,
    main,
)

H = 120
W = 160
CENTRE = (W // 2, H // 2)


def _frame(seed: int = 0) -> np.ndarray:
    """A deterministic, reproducible far-field-like frame."""
    rng = np.random.default_rng(seed)
    img = rng.normal(500.0, 20.0, size=(H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    img += 4000.0 * np.exp(-(((xx - CENTRE[0]) ** 2 + (yy - CENTRE[1]) ** 2) / 40.0))
    return img


class _Cam:
    """Minimal fake camera: fixed noise-free frame, settable exposure."""

    def __init__(self, base: float = 1000.0) -> None:
        self.base = base
        self.exposure_ms = 1.0
        self.reads = 0

    def set_exposure(self, exposure_ms: float) -> None:
        self.exposure_ms = float(exposure_ms)

    def read(self) -> np.ndarray:
        self.reads += 1
        # Brightness scales with exposure, so an increasing ladder must give an
        # increasing box sum -- the property the ladder test relies on.
        return _frame(0) * (self.exposure_ms / 1.0)


def test_drift_series_rows_have_exactly_row_keys():
    cam = _Cam()
    rows, summary = drift_series(cam.read, n=5)

    assert len(rows) == 5
    for row in rows:
        assert set(row) == set(ROW_KEYS)
        assert row["repeat"] is None  # None during drift
        assert isinstance(row["_epoch"], int)
    assert summary["n"] == 5
    assert len(summary["drift_l2_series"]) == 4  # n-1 consecutive pairs


def test_exposure_ladder_rows_have_exactly_row_keys():
    cam = _Cam()
    rows, summary = exposure_ladder(cam.read, cam.set_exposure, LADDER_DEFAULT, repeats=2)

    assert len(rows) == len(LADDER_DEFAULT) * 2
    for row in rows:
        assert set(row) == set(ROW_KEYS)
        assert row["repeat"] is not None  # populated on the ladder
    assert summary["repeats"] == 2
    assert summary["ladder"] == [float(v) for v in LADDER_DEFAULT]


def test_constant_frame_gives_zero_drift_floor():
    frame = _frame(3)
    rows, summary = drift_series(lambda: frame.copy(), n=6)

    assert summary["drift_floor_l2"] == pytest.approx(0.0, abs=1e-9)
    assert summary["box_sum_cv_pct"] == pytest.approx(0.0, abs=1e-9)
    # Identical reads => identical every row observable.
    assert all(row["sum"] == pytest.approx(rows[0]["sum"]) for row in rows)


def test_ramping_frame_gives_nonzero_increasing_drift():
    # Amplitude accelerates, so each consecutive difference is strictly larger
    # than the last. A *linear* ramp would give a constant step by construction
    # and could not show this.
    frames = [_frame(0) * float(1 + k) ** 2 for k in range(6)]
    it = iter(frames)
    rows, summary = drift_series(lambda: next(it), n=6)

    assert summary["drift_floor_l2"] > 0.0
    series = summary["drift_l2_series"]
    assert all(b > a for a, b in zip(series, series[1:])), series
    assert summary["box_sum_cv_pct"] > 0.0


def test_ladder_increasing_exposure_increases_box_sum():
    ladder = (0.2, 0.5, 1.0, 2.0)
    cam = _Cam()
    rows, summary = exposure_ladder(cam.read, cam.set_exposure, ladder, repeats=1)

    sums = [row["sum"] for row in rows]
    assert all(b > a for a, b in zip(sums, sums[1:])), sums
    level_sums = summary["level_sums"]
    assert all(b > a for a, b in zip(level_sums, level_sums[1:])), level_sums


def test_ladder_verdict_is_monotonic_for_clean_ramp():
    ladder = (0.25, 0.5, 1.0, 2.0)
    cam = _Cam()
    _, summary = exposure_ladder(cam.read, cam.set_exposure, ladder, repeats=2)

    assert summary["verdict"] == "monotonic"
    assert summary["saturated"] is False


def test_ladder_verdict_flags_non_monotonic_peaks():
    """The measured 194/193/80/117 sequence must not pass as monotonic."""
    peaks = (194.0, 193.0, 80.0, 117.0)
    # `measure_spot` reads the raw frame max, so scaling to an exact peak is
    # exact -- no need to reverse-engineer the smoothing for these values.
    base = _frame(0)
    base = base - base.min() + 1.0  # strictly positive => no clipping asymmetry
    base_max = float(base.max())
    frames = iter(base * (p / base_max) for p in peaks)

    rows, summary = exposure_ladder(lambda: next(frames), lambda ms: None,
                                    (1.0, 2.0, 3.0, 4.0), repeats=1)

    assert [row["peak"] for row in rows] == pytest.approx(list(peaks), rel=1e-9)
    assert summary["verdict"] == "non_monotonic"
    assert summary["level_peaks"] == pytest.approx(list(peaks), rel=1e-9)


def test_ladder_flags_saturation_when_level_reached():
    cam = _Cam()
    _, summary = exposure_ladder(
        cam.read, cam.set_exposure, (0.5, 1.0, 2.0), repeats=1,
        saturation_level=1e-9,  # anything above this counts as full scale
    )

    assert summary["saturated"] is True


def test_drift_series_rejects_single_frame():
    with pytest.raises(ValueError, match="n >= 2"):
        drift_series(lambda: _frame(0), n=1)


def test_exposure_ladder_rejects_short_ladder():
    cam = _Cam()
    with pytest.raises(ValueError, match="at least two exposures"):
        exposure_ladder(cam.read, cam.set_exposure, (1.0,), repeats=1)


def test_exposure_ladder_rejects_zero_repeats():
    cam = _Cam()
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        exposure_ladder(cam.read, cam.set_exposure, (0.5, 1.0), repeats=0)


def test_main_no_hw_is_offline_and_creates_no_hardware(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--no-hw", "--out", str(tmp_path / "unused")]) == 0
    assert not (tmp_path / "unused").exists()  # no dir written on the dry path