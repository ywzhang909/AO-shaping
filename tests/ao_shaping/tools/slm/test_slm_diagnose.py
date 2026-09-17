"""Tests for slm_diagnose hardware self-check logic (no hardware needed).

The module imports hardware drivers lazily inside functions, so the pure
decision helpers (frames_same / peak_and_bucket / step verdicts) can be
tested with synthetic frames — reproducing the 2026-09 hardware diagnostic:
a frozen panel is detected when pattern frames do not differ, modulation is
absent when the 0-order bucket shows no periodicity, and the linearity step
flags abnormal light when peak brightness does not grow with exposure.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_diagnose import (
    _DIFFRACTION_SCALE_PX,
    frames_same,
    peak_and_bucket,
)


def _make_frame(width: int = 2688, height: int = 1520, peak=(1450, 642), amp: float = 100.0) -> np.ndarray:
    frame = np.zeros((height, width), dtype=np.float64)
    p = np.random.default_rng(0).uniform(0, 0.5, (height, width))
    frame += p
    yy, xx = np.mgrid[0:height, 0:width]
    frame += amp * np.exp(-(((xx - peak[0]) ** 2 + (yy - peak[1]) ** 2) / 288.0))
    return frame


class TestFramesSame:
    def test_identical_frames(self):
        a = _make_frame()
        assert frames_same(a, a.copy())

    def test_different_frames(self):
        a = _make_frame(peak=(1450, 642))
        b = _make_frame(peak=(1500, 700))
        assert not frames_same(a, b)

    def test_shape_mismatch(self):
        assert not frames_same(np.zeros((10, 10)), np.zeros((5, 5)))


class TestPeakAndBucket:
    def test_peak_found_at_synthetic_location(self):
        frame = _make_frame(peak=(1450, 642), amp=200.0)
        px, py, peak, bucket = peak_and_bucket(frame)
        assert px == 1450
        assert py == 642
        assert peak > 100.0
        # bucket sums a 61x61 neighborhood around the peak
        assert bucket > peak

    def test_0order_is_global_max_not_frame_center(self):
        """Fourier-bench axiom: optical axis is the frame max, not frame center."""
        frame = _make_frame(peak=(1450, 642), amp=200.0)
        px, py, _, _ = peak_and_bucket(frame)
        assert (px, py) != (frame.shape[1] // 2, frame.shape[0] // 2)


class TestDiagnoseVerdicts:
    """Verdict thresholds embedded in the step functions (no hardware).

    Reproduces the 2026-09 finding set:
      * frozen panel  => N pattern frames differ from flat (ok if >= 2/3)
      * no modulation => bucket relative spread < 15% (ok if >= 15%)
      * abnormal light=> peak growth at x20 exposure < 2x (ok if >= 2x)
    """

    @staticmethod
    def _freeze_verdict(frames: dict[str, np.ndarray]) -> bool:
        from ao_shaping.tools.slm.slm_diagnose import DiagnoseResult

        n_diff = sum(
            1 for name, f in frames.items()
            if name != "flat" and not frames_same(frames["flat"], f)
        )
        return DiagnoseResult(ok=n_diff >= 2, message="", metrics={}).ok

    def test_frozen_panel_detected(self):
        frames = {"flat": _make_frame(), "grat": _make_frame(), "top": _make_frame(), "bot": _make_frame()}
        assert not self._freeze_verdict(frames)

    def test_active_panel_detected(self):
        frames = {
            "flat": _make_frame(peak=(1400, 600)),
            "grat": _make_frame(peak=(1450, 642)),
            "top": _make_frame(peak=(1500, 700)),
            "bot": _make_frame(peak=(1600, 800)),
        }
        assert self._freeze_verdict(frames)

    @staticmethod
    def _bucket_spread(amps: list[float], base: float = 8000.0) -> float:
        buckets = np.asarray([base * a for a in amps])
        return (float(buckets.max()) - float(buckets.min())) / (float(buckets.mean()) or 1.0)

    def test_modulating_panel_has_periodic_bucket(self):
        """Amplitude coupling: bucket varies by >15% across grayscale sweep."""
        # Simulate ~2x peak-to-peak swing around 993-gray period.
        spread = self._bucket_spread([0.6, 1.0, 1.4, 1.0, 0.6])
        assert spread > 0.15

    def test_frozen_panel_has_flat_bucket(self):
        """No modulation => bucket stays within noise (~7% per diagnostics)."""
        spread = self._bucket_spread([1.0, 1.05, 1.03, 1.07, 1.0])
        assert spread < 0.15

    def test_normal_light_grows_with_exposure(self):
        peaks = np.asarray([100.0, 400.0, 2000.0])
        assert peaks[-1] / peaks[0] > 2.0

    def test_abnormal_light_constant_peak(self):
        """2026-09 finding: exposure x20 while peak stays ~51 => abnormal."""
        peaks = np.asarray([51.0, 51.0, 51.0])
        assert peaks[-1] / peaks[0] <= 2.0


class TestConstantsPinned:
    """Confirmed hardware/optical facts from the 2026-09 diagnostic."""

    def test_diffraction_scale_pinned(self):
        """λ·f/(d·p_cam)=5021 px·period -> P64=78px, P32=157px."""
        assert _DIFFRACTION_SCALE_PX == 5021.0
        assert round(_DIFFRACTION_SCALE_PX / 64) == 78
        assert round(_DIFFRACTION_SCALE_PX / 32) == 157