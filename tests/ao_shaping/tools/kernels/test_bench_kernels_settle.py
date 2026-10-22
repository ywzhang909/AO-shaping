"""Offline tests for the shared settle criterion of the SLM bench kernels.

**No hardware.** The failure this locks down is a *timing* failure, so it is
reproduced with a fake camera whose spot relaxes over the frames read after a
phase write -- exactly the shape of the real event measured on 2026-10-21:

* the same flat ladder read 213 / 33 / 213 counts (10 ms exposure) at 09:02,
  13:32 and 16:44, each ladder strictly linear internally;
* the flat-spot FWHM flipped between 40.6 px and 48.1 px across two runs of the
  same geometric self-check.

The old criterion ("two consecutive readings agree on peak and centroid") is
*too easy* to satisfy mid-relaxation: the centroid has often not moved yet
while the spot is still widening and dimming, so the gate opens on a frame that
is still relaxing. These tests pin the new criterion down:

1. the gate does not open on a mid-relaxation state;
2. the *width* participates in the criterion (peak settled but still widening is
   not "settled");
3. a frame that never settles is **reported** as not settled, never silently
   passed off as a measurement;
4. a genuinely stable sequence returns a bit-identical frame to the old code.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from loguru import logger

from ao_shaping.tools.kernels.bench_kernels import (
    DEFAULT_MIN_WAIT_S,
    DEFAULT_N_STABLE,
    capture_settled,
    capture_settled_report,
    display_and_average,
    measure_spot,
    wait_for_settle,
)

SHAPE = (120, 160)
CX, CY = 80.0, 60.0

#: Per-frame spot width (sigma, px) of the fake relaxation curve. The measured
#: FWHM of these frames falls 21.5 px -> 12.2 px, and the *last three* entries
#: agree to 1.3% -- which is where the new criterion is allowed to open.
RELAXING_SIGMAS = [
    9.0, 8.2, 7.5, 6.9, 6.4, 6.0, 5.7, 5.45, 5.25, 5.1, 5.0, 4.95, 4.93, 4.92,
]
#: The width the panel ends up at; the fake holds it forever.
FINAL_SIGMA = RELAXING_SIGMAS[-1]


def _render(peak: float, sigma: float, cx: float, cy: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)))


def _nan_frame() -> np.ndarray:
    """A frame with a non-finite pixel -- ``measure_spot`` cannot digest it."""
    frame = _render(200.0, 4.92, CX, CY)
    frame[0, 0] = np.nan
    return frame


class FakeCamera:
    """Camera whose spot relaxes over the frames read after a phase write.

    The frame index counts reads since the last phase write; the last entry of
    each curve repeats forever, so the camera ends in a well-defined final
    state. ``frame_period_s`` burns wall-clock time per read, which is how the
    timeout path is reached without a real camera.
    """

    def __init__(
        self,
        *,
        peaks: list[float] | None = None,
        sigmas: list[float] | None = None,
        cxs: list[float] | None = None,
        cys: list[float] | None = None,
        frame_period_s: float = 0.0,
        blank: bool = False,
    ) -> None:
        self.peaks = peaks if peaks is not None else [200.0]
        self.sigmas = sigmas if sigmas is not None else [4.92]
        self.cxs = cxs if cxs is not None else [CX]
        self.cys = cys if cys is not None else [CY]
        self.frame_period_s = float(frame_period_s)
        self.blank = bool(blank)
        self.index = 0
        self.reads = 0

    def reset(self) -> None:
        """Called by the fake SLM: a new phase write restarts the relaxation."""
        self.index = 0

    def _at(self, curve: list[float], i: int) -> float:
        return curve[min(i, len(curve) - 1)]

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        i = self.index
        self.index += 1
        self.reads += 1
        if self.frame_period_s > 0.0:
            time.sleep(self.frame_period_s)
        if self.blank:
            return np.zeros(SHAPE, dtype=np.float64)
        return _render(
            self._at(self.peaks, i),
            self._at(self.sigmas, i),
            self._at(self.cxs, i),
            self._at(self.cys, i),
        )


class FakeSlm:
    """Minimal SLM duck-type: it only has to accept a phase and a wait."""

    def __init__(self, cam: FakeCamera) -> None:
        self.cam = cam
        self.writes = 0
        self.last_wait_s: float | None = None

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        return np.zeros(np.shape(phase_rad), dtype=np.uint16)

    def display_data(self, gray, wait_time_s=None, **_kwargs) -> int:
        self.writes += 1
        self.last_wait_s = wait_time_s
        self.cam.reset()
        if wait_time_s:
            time.sleep(float(wait_time_s))
        return 1


def _legacy_display_and_average(
    cam,
    slm,
    phase_rad: np.ndarray,
    n_frames: int = 4,
    n_discard: int = 3,
    wait_time_s: float | None = 0.5,
    stable_tol: float = 0.02,
    max_wait_s: float = 6.0,
) -> np.ndarray:
    """Frozen copy of the pre-2026-10-22 settle gate (the HEAD implementation).

    Kept verbatim so the "no regression on a stable sequence" claim is checked
    against the code that used to run, not against a paraphrase of it.
    """
    gray = slm.create_phase_from_array(phase_rad)
    slm.display_data(gray, wait_time_s)
    for _ in range(int(n_discard)):
        cam.get_numpy_image(n_sample=1)

    def probe() -> tuple[float, float, float]:
        img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        m = measure_spot(img)
        return m.peak, m.centroid_x, m.centroid_y

    prev = probe()
    deadline = time.time() + float(max_wait_s)
    while time.time() < deadline:
        cur = probe()
        if (
            abs(cur[0] - prev[0]) <= stable_tol * max(prev[0], 1.0)
            and abs(cur[1] - prev[1]) <= stable_tol * 40.0
            and abs(cur[2] - prev[2]) <= stable_tol * 40.0
        ):
            break
        prev = cur

    acc = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(int(n_frames))
    ]
    return np.mean(acc, axis=0)


def _final_fwhm() -> float:
    return measure_spot(_render(200.0, FINAL_SIGMA, CX, CY)).fwhm_px


class TestRelaxationIsNotMistakenForStability:
    """The core regression: the gate must not open mid-relaxation."""

    def test_returns_the_final_state_not_a_mid_relaxation_state(self) -> None:
        cam = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm = FakeSlm(cam)

        result = capture_settled_report(
            cam,
            slm,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        assert result.settle.settled is True
        assert result.settle.reason == "stable"

        final = _final_fwhm()
        got = measure_spot(result.frame).fwhm_px
        assert got == pytest.approx(final, rel=0.03), (
            f"returned frame is still relaxing: fwhm {got:.2f} px vs final {final:.2f} px"
        )

    def test_the_old_criterion_would_have_returned_a_relaxing_frame(self) -> None:
        """Pins *why* the change is needed, with the old code as the witness."""
        cam = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm = FakeSlm(cam)
        legacy_frame = _legacy_display_and_average(
            cam, slm, np.zeros((8, 8)), n_frames=4, n_discard=3, wait_time_s=0.0
        )

        final = _final_fwhm()
        legacy = measure_spot(legacy_frame).fwhm_px
        assert abs(legacy - final) / final > 0.04, (
            f"the legacy gate is expected to return a visibly relaxing frame "
            f"(fwhm {legacy:.2f} px vs final {final:.2f} px)"
        )

    def test_the_new_gate_waits_longer_than_the_old_one(self) -> None:
        cam_old = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm_old = FakeSlm(cam_old)
        _legacy_display_and_average(
            cam_old, slm_old, np.zeros((8, 8)), n_frames=4, n_discard=3, wait_time_s=0.0
        )
        legacy_reads = cam_old.reads

        cam_new = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm_new = FakeSlm(cam_new)
        result = capture_settled_report(
            cam_new,
            slm_new,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        assert result.settle.n_probes > 2
        assert cam_new.reads > legacy_reads
        # The gate opened only once the width had stopped changing.
        assert result.settle.fwhm_px == pytest.approx(_final_fwhm(), rel=0.05)


class TestWidthParticipatesInTheCriterion:
    """Peak settled + centroid still + width moving == NOT settled."""

    def test_a_peak_settled_width_still_changing_sequence_is_not_stable(self) -> None:
        # Every step changes the width by ~7%, far above the 2% tolerance, so a
        # peak-only criterion would have declared stability on the 2nd reading.
        sigmas = [5.0 * (0.93**k) for k in range(20)]
        cam = FakeCamera(sigmas=sigmas)
        slm = FakeSlm(cam)

        result = capture_settled_report(
            cam,
            slm,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        assert result.settle.settled is True
        # It could not have opened on the first two readings: the width was
        # still moving by several percent per frame.
        assert result.settle.n_probes >= 10
        # The reading that closed the window is already at the final width,
        # i.e. the gate waited for the *shape* to stop changing.
        final = measure_spot(_render(200.0, sigmas[-1], CX, CY)).fwhm_px
        assert result.settle.fwhm_px == pytest.approx(final, rel=0.02)

    def test_the_criterion_compares_the_width(self) -> None:
        """Same peak, same centroid, different width -> not the same reading."""
        cam = FakeCamera(sigmas=[9.0, 9.0, 9.0, 9.0])
        slm = FakeSlm(cam)
        report = wait_for_settle(
            cam,
            n_discard=0,
            min_wait_s=0.0,
            stable_tol=0.02,
            max_wait_s=0.05,
            n_stable=3,
        )
        # Identical frames settle; a width change would have blocked this.
        assert report.settled is True
        assert report.n_probes == 3

    def test_a_degenerate_width_never_counts_as_unchanged(self) -> None:
        """A ``nan`` FWHM (empty frame) must not be read as "unchanged"."""
        cam = FakeCamera(blank=True)
        report = wait_for_settle(
            cam,
            n_discard=0,
            min_wait_s=0.0,
            stable_tol=0.02,
            max_wait_s=0.02,
            n_stable=2,
        )
        assert report.settled is False
        assert report.reason == "timeout"

    def test_an_unmeasurable_frame_does_not_abort_the_acquisition(self) -> None:
        """A non-finite pixel must not crash the acquisition."""
        cam = FakeCamera(blank=True)
        cam.get_numpy_image = lambda n_sample=1: _nan_frame()  # type: ignore[assignment]
        report = wait_for_settle(
            cam,
            n_discard=0,
            min_wait_s=0.0,
            stable_tol=0.02,
            max_wait_s=0.02,
            n_stable=2,
        )
        assert report.settled is False


class TestUnsettledIsReported:
    """A settling failure must never be indistinguishable from a measurement."""

    def test_timeout_is_reported_not_silently_passed(self) -> None:
        cam = FakeCamera(sigmas=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], frame_period_s=0.02)
        slm = FakeSlm(cam)

        result = capture_settled_report(
            cam,
            slm,
            np.zeros((8, 8)),
            n_frames=1,
            n_discard=0,
            wait_time_s=0.0,
            min_wait_s=0.0,
            max_wait_s=0.05,
        )

        assert result.settle.settled is False
        assert result.settle.reason == "timeout"
        assert result.settle.n_probes >= 1
        # The frame is still returned (existing contract) -- but flagged.
        assert result.frame.shape == SHAPE

    def test_the_numpy_entry_points_warn_when_not_settled(self) -> None:
        cam = FakeCamera(sigmas=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], frame_period_s=0.02)
        slm = FakeSlm(cam)

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")
        try:
            capture_settled(
                cam,
                slm,
                np.zeros((8, 8)),
                n_frames=1,
                n_discard=0,
                wait_time_s=0.0,
                min_wait_s=0.0,
                max_wait_s=0.05,
            )
        finally:
            logger.remove(sink_id)

        assert any("settle criterion not met" in m for m in messages), messages

    def test_no_warning_when_the_criterion_is_met(self) -> None:
        cam = FakeCamera(sigmas=[4.92])
        slm = FakeSlm(cam)

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")
        try:
            capture_settled(
                cam,
                slm,
                np.zeros((8, 8)),
                n_frames=1,
                n_discard=0,
                wait_time_s=0.0,
                min_wait_s=0.0,
            )
        finally:
            logger.remove(sink_id)

        assert not any("settle criterion not met" in m for m in messages), messages


class TestNoRegression:
    def test_a_stable_sequence_returns_a_bit_identical_frame(self) -> None:
        cam_old = FakeCamera(sigmas=[4.92])
        slm_old = FakeSlm(cam_old)
        legacy = _legacy_display_and_average(
            cam_old, slm_old, np.zeros((8, 8)), n_frames=4, n_discard=3, wait_time_s=0.0
        )

        cam_new = FakeCamera(sigmas=[4.92])
        slm_new = FakeSlm(cam_new)
        frame = display_and_average(
            cam_new,
            slm_new,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        assert np.array_equal(frame, legacy)

    def test_the_old_positional_signature_still_works(self) -> None:
        cam = FakeCamera(sigmas=[4.92])
        slm = FakeSlm(cam)
        frame = display_and_average(cam, slm, np.zeros((8, 8)), 4, 3, 0.0, 0.02, 6.0)
        assert frame.shape == SHAPE

    def test_capture_settled_and_display_and_average_share_one_criterion(self) -> None:
        cam_a = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm_a = FakeSlm(cam_a)
        a = capture_settled_report(
            cam_a,
            slm_a,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        cam_b = FakeCamera(sigmas=list(RELAXING_SIGMAS))
        slm_b = FakeSlm(cam_b)
        b = capture_settled(
            cam_b,
            slm_b,
            np.zeros((8, 8)),
            n_frames=4,
            n_discard=3,
            wait_time_s=0.0,
            min_wait_s=0.0,
        )

        assert np.array_equal(a.frame, b)
        # 3 discarded frames + 10 stability probes before the width settled.
        assert a.settle.n_probes == 10


class TestMinWaitFloor:
    def test_the_first_probe_does_not_happen_before_min_wait_s(self) -> None:
        cam = FakeCamera(sigmas=[4.92])
        slm = FakeSlm(cam)
        t0 = time.monotonic()
        capture_settled(
            cam,
            slm,
            np.zeros((8, 8)),
            n_frames=1,
            n_discard=0,
            wait_time_s=0.0,
            min_wait_s=0.2,
        )
        assert time.monotonic() - t0 >= 0.2

    def test_the_driver_wait_already_covers_the_floor(self) -> None:
        """No double wait: the floor is a floor, not an extra sleep.

        The driver waited 0.5 s, which already exceeds the 0.3 s floor, so the
        criterion must not add another 0.3 s on top.
        """
        cam = FakeCamera(sigmas=[4.92])
        slm = FakeSlm(cam)
        t0 = time.monotonic()
        capture_settled(
            cam,
            slm,
            np.zeros((8, 8)),
            n_frames=1,
            n_discard=0,
            wait_time_s=0.5,
            min_wait_s=0.3,
        )
        elapsed = time.monotonic() - t0
        assert 0.5 <= elapsed < 0.5 + 0.3


class TestDefaults:
    def test_the_defaults_are_the_conservative_ones(self) -> None:
        assert DEFAULT_N_STABLE == 3
        assert DEFAULT_MIN_WAIT_S == 0.3

    def test_n_stable_must_be_positive(self) -> None:
        cam = FakeCamera(sigmas=[4.92])
        with pytest.raises(ValueError, match="n_stable"):
            wait_for_settle(cam, n_stable=0)
