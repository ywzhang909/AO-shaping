"""Exposure-sweep summary contracts for the light-level probe.

The probe exists to tell "no light" apart from "wrong phase" before any SLM phase is
blamed, and the classification that keeps it honest is the false-alarm guard: a
genuine but dim spot must never be reported as a dark field. Everything here is
offline -- synthetic frames in, summary dicts out, no camera and no device.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.slm.diagnose_light_level import _verdict, stats


def _pin_peak(frame: np.ndarray, value: float) -> np.ndarray:
    """Copy ``frame`` with its brightest pixel pinned to exactly ``value``."""
    pinned = np.array(frame, dtype=np.float64, copy=True)
    row, col = np.unravel_index(int(np.argmax(pinned)), pinned.shape)
    pinned[row, col] = value
    return pinned


def _dim_genuine_frame() -> np.ndarray:
    """The frame the probe was written for: peak 17/255 at 20 ms on a faint pedestal.

    Reproduces the recorded summary (peak 17, mean 0.376, std 0.605) to within 5%
    on mean and std -- the peak is exact, since that is the quantity the verdict
    turns on.
    """
    shape = (120, 160)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    rng = np.random.default_rng(11)
    frame = np.clip(rng.normal(0.10, 0.72, shape), 0.0, None)
    spot_row, spot_col = 74, 46
    spot = np.exp(-(((rows - spot_row) ** 2 + (cols - spot_col) ** 2) / (2 * 1.6**2)))
    return frame + (17.0 - frame[spot_row, spot_col]) * spot


def test_stats_summarises_a_frame() -> None:
    """``max_position_xy`` is ``[x, y]``, not ``[row, col]`` -- consumers index with it."""
    frame = np.linspace(0.0, 10.0, 48 * 64).reshape(48, 64)
    summary = stats(frame)

    assert summary["shape"] == [48, 64]
    assert summary["dtype"] == "float64"
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(10.0)
    assert summary["range"] == pytest.approx(10.0)
    assert summary["mean"] == pytest.approx(5.0)
    assert summary["median"] == pytest.approx(5.0)
    assert summary["p99"] == pytest.approx(9.9, abs=0.01)
    # A discrete uniform on 48*64 points: std = 10 * sqrt((N**2 - 1) / (12 * (N - 1) ** 2)).
    assert summary["std"] == pytest.approx(10.0 / np.sqrt(12.0), rel=1e-3)
    # The peak is at row 47, column 63; the pair is reported as [x, y].
    assert summary["max_position_xy"] == [63, 47]


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (2.0, "no_light"),
        (3.0, "no_light"),  # the boundary itself still counts as no light
        (3.5, "weak"),
        (17.0, "weak"),
        (19.9, "weak"),
        (20.0, "ok"),
        (200.0, "ok"),
    ],
)
def test_verdict_thresholds(peak: float, expected: str) -> None:
    rng = np.random.default_rng(17)
    frame = _pin_peak(1.0 + rng.random((32, 32)), peak)

    verdict = _verdict([], stats(frame))
    assert verdict["state"] == expected
    assert verdict["reasons"]


def test_dim_but_genuine_spot_is_weak_not_no_light() -> None:
    """The regression: a real spot at peak 17/255 must never read as "no light".

    Reading the exposure ladder and concluding the beam never arrived would blame
    the phase for a bench that was merely dim, so the ``no_light`` state is gated on
    the peak alone and the dim case has to fall through to ``weak``.
    """
    summary = stats(_dim_genuine_frame())
    assert summary["max"] == pytest.approx(17.0)
    assert summary["mean"] == pytest.approx(0.376, abs=0.05)
    assert summary["std"] == pytest.approx(0.605, abs=0.05)

    verdict = _verdict([], summary)
    assert verdict["state"] == "weak"
    assert verdict["state"] != "no_light"


def test_uniform_dark_field_is_no_light() -> None:
    """Peak 2 with a sub-count standard deviation: nothing reached the camera."""
    rng = np.random.default_rng(5)
    frame = _pin_peak(np.clip(rng.normal(1.8, 0.12, (120, 160)), 0.0, 2.0), 2.0)

    summary = stats(frame)
    assert summary["max"] == pytest.approx(2.0)
    assert summary["std"] < 1.0

    verdict = _verdict([], summary)
    assert verdict["state"] == "no_light"
    # Three notes: the peak, the missing structure, and the uniform-dark-field tell.
    assert len(verdict["reasons"]) == 3
    assert all(reason for reason in verdict["reasons"])


def test_no_light_threshold_ignores_frame_structure() -> None:
    """The peak alone gates ``no_light``; a structured frame at the same peak adds nothing.

    This pins the boundary the guard above leans on -- if the classification ever
    starts folding structure into it, the dim-but-genuine case can be dragged down
    with it.
    """
    rng = np.random.default_rng(5)
    dark = _pin_peak(np.clip(rng.normal(1.8, 0.12, (120, 160)), 0.0, 2.0), 2.0)
    alternating = np.where(np.arange(120 * 160).reshape(120, 160) % 2 == 0, 0.0, 2.5)
    structured = _pin_peak(alternating, 3.0)

    dark_summary = stats(dark)
    structured_summary = stats(structured)
    assert dark_summary["max"] == pytest.approx(2.0)
    assert structured_summary["max"] == pytest.approx(3.0)
    assert structured_summary["std"] >= 1.0

    assert _verdict([], dark_summary)["state"] == "no_light"
    # 3 is the boundary, so the structured frame is no light with one note fewer.
    assert _verdict([], structured_summary)["state"] == "no_light"
    assert len(_verdict([], structured_summary)["reasons"]) == 2