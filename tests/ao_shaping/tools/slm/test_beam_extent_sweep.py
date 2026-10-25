"""Offline tests for the extracted beam-extent sweep (panel/beam_extent).

``sweep_axis`` was split out of the Click command so the GUI can call the
measurement directly -- it is the probe the model-in-loop geometry warning names
when its two focal-scale routes disagree. These tests pin the edge rule that
the split had to preserve, because that rule is what makes the number usable:

* the edge is where ``rel`` stops falling (the *last* index still above the
  fully-randomised floor), not the first point that dropped;
* ``edge_px`` is a far edge from the panel origin, so ``radius_px`` is an upper
  bound rather than a radius.

The devices are stubbed: ``Santec`` and ``create_camera`` are replaced, so no
hardware and no SDK import happens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.tools.slm.panel import beam_extent as be


class _StubSLM:
    """Records the displayed phase and reports the panel half-plane it covered."""

    def __init__(self, *a: Any, **k: Any) -> None:
        self.displayed: list[np.ndarray] = []

    def __enter__(self) -> "_StubSLM":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _StubCamera:
    """Camera stub: peak falls with how much of the beam is still un-randomised."""

    def __init__(self, profile: list[float] | None = None) -> None:
        self.profile = profile
        self.flat_rel = 1.0
        self.current_ms = 0.0
        self.n_calls = 0

    def __enter__(self) -> "_StubCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def reset_exposure_time(self, ms: float) -> float:
        self.current_ms = float(ms)
        return self.current_ms

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        self.n_calls += 1
        peak = 200
        if self.profile is not None:
            idx = min(self.n_calls - 1, len(self.profile) - 1)
            peak = max(1, int(200 * self.profile[idx]))
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[32, 32] = peak
        return frame

    def next_rel(self) -> float:
        """The ``rel`` value the next frame will score, or 0.0 on a dark bench.

        ``core_fraction`` is a disc-energy ratio, so a stub that divides by the
        flat peak does not reproduce it. This drives the sweep's *scoring* seam
        directly, which is what the edge rule actually reads.
        """
        if self.profile is None:
            return 1.0
        # n_calls was already incremented by the display this score belongs to.
        idx = min(self.n_calls - 1, len(self.profile) - 1)
        return self.profile[idx]


@pytest.fixture
def stub_devices(monkeypatch):
    """Replace Santec / create_camera so no SDK import or device access happens."""
    slm = _StubSLM()
    cam = _StubCamera()
    monkeypatch.setattr(
        "ao_shaping.drivers.slm.santec.Santec", lambda *a, **k: slm
    )
    monkeypatch.setattr(
        "ao_shaping.drivers.ccd.common.create_camera", lambda *a, **k: cam
    )
    # Deterministic scoring: the flat reference scores 1.0, and each swept frame
    # scores whatever the stub camera says next.
    monkeypatch.setattr(
        be, "measure_flat_reference",
        lambda cam, slm, n_frames, panel_shape: (
            np.full((64, 64), 200, dtype=np.uint8),
            type("F", (), {
                "centroid_x": 32.0, "centroid_y": 32.0, "peak": 200.0,
                "as_row": staticmethod(lambda: "flat"),
            })(),
        ),
    )
    def _score(img, cx, cy, r):
        # n_calls == 0 means this is the flat reference, not a swept frame.
        return cam.flat_rel if not cam.n_calls else cam.next_rel()

    monkeypatch.setattr(be, "core_fraction", _score)
    monkeypatch.setattr(
        be, "display_and_average",
        lambda cam, slm, phase, n_frames: cam.get_numpy_image(),
    )
    return slm, cam


def _run(**over: Any) -> be.BeamExtentResult:
    kwargs: dict[str, Any] = {
        "axis": "x",
        "steps": 8,
        "frames": 1,
        "repeats": 1,
        "core_radius": 8.0,
        "seed": 2024,
        "slm_number": 1,
        "slm_wavelength": 1064,
        "cam_type": "daheng",
        "cam_id": 0,
        "exposure_ms": 0.1,
    }
    kwargs.update(over)
    return be.sweep_axis(**kwargs)


class TestEdgeRule:
    def test_edge_is_where_the_response_stops_falling(self, stub_devices) -> None:
        """rel falls to a floor and stays there; the edge is the last step before it."""
        profile = [1.0, 1.0, 0.95, 0.9, 0.7, 0.5, 0.5, 0.5]
        stub_devices[1].profile = profile
        result = _run(steps=8)
        assert result.profile_rel[0] == pytest.approx(1.0)
        assert result.floor_rel == pytest.approx(0.5)
        # 1.10 * floor = 0.55; the first index at/below that is index 5 (rel 0.5).
        assert result.edge_px == result.positions_px[5]

    def test_a_monotonic_fall_reports_the_last_step(
        self, stub_devices
    ) -> None:
        """A profile that never flattens still yields a usable far edge."""
        profile = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        stub_devices[1].profile = profile
        result = _run(steps=8)
        assert result.edge_px == result.positions_px[-1]


class TestCentreFreeWidth:
    """Radius from the fall's own width — no ``panel_locate`` in the loop.

    The measured 2026-10-17 x profile is the reference case: it produced 180 px
    when the centre was subtracted and 300 px for a different centre, while its
    fall width says ~330 px on its own.
    """

    # positions 120..1920 step 120; rel holds ~1, falls, then pins at 0.48
    MEASURED_X = (
        [0.99, 1.0, 1.0, 0.99, 0.97, 0.92, 0.84, 0.72,
         0.61, 0.52, 0.49, 0.48, 0.48, 0.48, 0.48, 0.49],
        120,
    )

    def test_the_measured_profile_yields_about_330_px(self, stub_devices) -> None:
        profile, step = self.MEASURED_X
        stub_devices[1].profile = profile
        result = _run(steps=16)
        assert result.radius_px == pytest.approx(330, abs=2 * step)
        assert result.centre_px == pytest.approx(990, abs=step)

    def test_the_width_needs_no_centre_at_all(self, stub_devices) -> None:
        """The whole point: two centres must give the same radius."""
        profile, _ = self.MEASURED_X
        stub_devices[1].profile = profile
        result = _run(steps=16)
        width = result.fall_hi_px - result.fall_lo_px
        assert result.radius_px == pytest.approx(width / 2.0)

    def test_onset_ignores_the_non_monotonic_first_points(
        self, stub_devices
    ) -> None:
        """One random draw per point makes the head of a scan noisy; a tight
        threshold would latch onto it and report a radius far too large."""
        profile, _ = self.MEASURED_X
        noisy = [1.0, 0.985, 1.0, 1.0] + profile[4:]
        stub_devices[1].n_calls = 0
        stub_devices[1].profile = noisy
        noisy_result = _run(steps=16)
        stub_devices[1].n_calls = 0
        stub_devices[1].profile = profile
        clean_result = _run(steps=16)
        assert noisy_result.radius_px == clean_result.radius_px

    def test_the_uncertainty_is_one_step(self, stub_devices) -> None:
        profile, step = self.MEASURED_X
        stub_devices[1].profile = profile
        result = _run(steps=16)
        assert result.step_px == pytest.approx(step)


class TestAxisHandling:
    def test_x_and_y_use_different_panel_spans(self, stub_devices) -> None:
        stub_devices[1].profile = [1.0] * 4 + [0.5] * 4
        x = _run(axis="x", steps=8)
        y = _run(axis="y", steps=8)
        assert x.positions_px[-1] == pytest.approx(1920, abs=1)
        assert y.positions_px[-1] == pytest.approx(1200, abs=1)

    def test_flat_centroid_is_carried_through(self, stub_devices) -> None:
        stub_devices[1].profile = [1.0] * 4 + [0.5] * 4
        result = _run(steps=8)
        assert result.flat_centroid == (32.0, 32.0)


class TestFailureModes:
    def test_a_dark_bench_raises_instead_of_scoring_noise(
        self, stub_devices
    ) -> None:
        """The first beam-extent run of 2026-10-17 hit exactly this: the camera was
        disconnected, every frame was flat noise, and the tool printed a confident
        'extent ends near 120 px'. It must refuse instead."""
        cam = stub_devices[1]
        cam.flat_rel = 0.0
        cam.profile = [0.0] * 8
        with pytest.raises(RuntimeError, match="not reaching the camera"):
            _run(steps=8)


class TestResultShape:
    def test_fields_are_plain_data(self, stub_devices) -> None:
        """The GUI renders these straight into st.success / st.caption."""
        stub_devices[1].profile = [1.0] * 4 + [0.5] * 4
        result = _run(steps=8)
        assert isinstance(result.axis, str)
        assert isinstance(result.edge_px, int)
        assert isinstance(result.radius_px, int)
        assert isinstance(result.profile_rel, tuple)
        assert isinstance(result.positions_px, tuple)
        assert len(result.profile_rel) == len(result.positions_px)
        assert result.axis == "x"