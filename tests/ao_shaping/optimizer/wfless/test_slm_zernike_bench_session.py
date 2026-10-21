"""Unit contract for the shared ``BenchSession`` (R-6 / issue #43).

Both search branches in ``slm_zernike_pib.py`` evaluate a candidate the same
way: clip coefficients -> synthesize phase -> display -> wait -> grab ->
guard saturation -> re-centre the ROI -> score. The session single-homes that
sequence; these tests pin each method's contract in isolation with fake
devices (no SLM, no camera, no simulation bench). The end-to-end simulation
suite only pins the *order* the branches call these methods in
(``test_slm_zernike_pib_production_sim.py``); the per-method semantics live
here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ao_shaping.optimizer.wfless import slm_zernike_pib as engine


class _FakeCam:
    """Camera stub: records every grab's frame count, returns a canned frame."""

    def __init__(self, frame: np.ndarray, events: list[str] | None = None) -> None:
        self.frame = frame
        self.grabbed: list[int] = []
        self.events = events if events is not None else []

    def get_numpy_image(self, n_frames: int) -> np.ndarray:
        self.grabbed.append(n_frames)
        self.events.append("grab")
        return self.frame


class _FakeShaping:
    """Objective stub: records re-centre/score events in call order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.centers: list[tuple[float, float]] = []

    def set_reference_center(self, center: tuple[float, float]) -> None:
        self.centers.append(center)
        self.events.append(("center", center))

    def __call__(self, img: np.ndarray) -> str:
        self.events.append(("score", img))
        return "SCORED"


def _session(**overrides):
    """Build a ``BenchSession`` with fake devices and harmless defaults."""
    kwargs: dict = dict(
        slm=object(),
        cam=_FakeCam(np.zeros((4, 4), dtype=np.uint8)),
        to_gray=lambda phase: phase,
        shaping=_FakeShaping(),
        n_max=2,
        pattern_helper=object(),
        zernike_radius=300.0,
        exposure_time_ms=0.0,
        target_max_brightness=0.0,
        n_eval_frames=3,
        reference_center=(7.0, 9.0),
        blaze_choice=None,
    )
    kwargs.update(overrides)
    return engine.BenchSession(**kwargs)


# --------------------------------------------------------------------------
# capture: clip -> z2p -> to_gray -> display -> sleep -> grab, in that order
# --------------------------------------------------------------------------
def test_capture_clips_coeffs_and_orders_events(monkeypatch):
    """One candidate goes through the full sequence exactly once.

    The coefficient clip to ±``ZERNIKE_CLIP`` must happen *before* phase
    synthesis, and the frame must be grabbed only after the display write and
    the response-time wait.
    """
    events: list[str] = []
    seen_coeffs: list[np.ndarray] = []

    def fake_z2p(coeffs, n_max, pattern_helper, radius, *, default_radius):
        events.append("z2p")
        seen_coeffs.append(np.asarray(coeffs, dtype=float))
        return np.zeros((4, 4), dtype=float)

    def fake_to_gray(phase):
        events.append("to_gray")
        return np.zeros((4, 4), dtype=np.uint16)

    def fake_display(slm, gray, rotator):
        events.append("display")

    monkeypatch.setattr(engine, "zernike_to_phase", fake_z2p)
    monkeypatch.setattr(engine, "display_rotating", fake_display)
    monkeypatch.setattr(
        engine, "time", SimpleNamespace(sleep=lambda seconds: events.append("sleep"))
    )

    cam = _FakeCam(np.full((4, 4), 100, dtype=np.uint8), events)
    session = _session(cam=cam, to_gray=fake_to_gray, n_eval_frames=5)

    clipped, phase_gray, img = session.capture(np.array([6.0, -7.0, 0.5]))

    assert events == ["z2p", "to_gray", "display", "sleep", "grab"]
    np.testing.assert_array_equal(seen_coeffs[0], [5.0, -5.0, 0.5])
    np.testing.assert_array_equal(clipped, [5.0, -5.0, 0.5])
    assert cam.grabbed == [5]
    assert img is cam.frame


# --------------------------------------------------------------------------
# resample_if_saturated: one guard for both branches
# --------------------------------------------------------------------------
def _record_resample(monkeypatch) -> list[tuple]:
    """Replace ``resample_on_saturation`` with a recorder returning its input."""
    calls: list[tuple] = []

    def fake(img, cam, exposure_time_ms, target_max_brightness):
        calls.append((img, cam, exposure_time_ms, target_max_brightness))
        return img

    monkeypatch.setattr(engine, "resample_on_saturation", fake)
    return calls


def test_resample_fixed_exposure_never_fires(monkeypatch):
    """Fixed exposure short-circuits before any peak check (both branches)."""
    calls = _record_resample(monkeypatch)
    session = _session(exposure_time_ms=5.0)
    hot = np.full((4, 4), 255, dtype=np.uint8)

    out, fired = session.resample_if_saturated(hot, trigger_peak=999.0)

    assert fired is False
    assert out is hot
    assert calls == []


def test_resample_auto_cool_frame_does_not_fire(monkeypatch):
    """Auto exposure alone is not enough: the peak must reach full scale."""
    calls = _record_resample(monkeypatch)
    session = _session()
    cool = np.full((4, 4), 100, dtype=np.uint8)

    out, fired = session.resample_if_saturated(cool)

    assert out is cool
    assert fired is False
    assert calls == []


def test_resample_saturated_frame_fires_with_target_fallback(monkeypatch):
    """Target 0 falls back to 220, and ``fired`` stays True even if the
    resampler keeps the frame (SPGD's ``scale_momentum`` keys off ``fired``)."""
    calls = _record_resample(monkeypatch)
    session = _session(target_max_brightness=0)
    hot = np.full((4, 4), 255, dtype=np.uint8)

    out, fired = session.resample_if_saturated(hot)

    assert fired is True
    assert out is hot
    assert len(calls) == 1
    _, _, exposure, target = calls[0]
    assert exposure == 0.0
    assert target == 220


def test_resample_saturated_frame_passes_explicit_target(monkeypatch):
    """An explicit target is forwarded untouched (no fallback substitution)."""
    calls = _record_resample(monkeypatch)
    session = _session(target_max_brightness=180)
    hot = np.full((4, 4), 255, dtype=np.uint8)

    session.resample_if_saturated(hot)

    assert len(calls) == 1
    assert calls[0][3] == 180


def test_resample_trigger_peak_gates_on_explicit_peak(monkeypatch):
    """``trigger_peak`` overrides the frame's own peak (SPGD batch max), and
    NaN never fires (comparison is False, matching the pre-refactor guard)."""
    calls = _record_resample(monkeypatch)
    session = _session()
    cool = np.full((4, 4), 100, dtype=np.uint8)

    _, fired = session.resample_if_saturated(cool, trigger_peak=255.0)
    assert fired is True
    assert len(calls) == 1

    _, fired = session.resample_if_saturated(cool, trigger_peak=254.0)
    assert fired is False
    assert len(calls) == 1

    _, fired = session.resample_if_saturated(cool, trigger_peak=float("nan"))
    assert fired is False
    assert len(calls) == 1


# --------------------------------------------------------------------------
# recenter / score: ROI follows the spot, centre set before scoring
# --------------------------------------------------------------------------
def test_score_uses_blaze_center_when_probe_ran(monkeypatch):
    """With a blaze probe the fixed centre is used and argmax never runs."""

    def boom(img):
        raise AssertionError(
            "zero_order_center must not run when blaze_choice is set"
        )

    monkeypatch.setattr(engine, "zero_order_center", boom)
    shaping = _FakeShaping()
    session = _session(
        shaping=shaping, blaze_choice=object(), reference_center=(7.0, 9.0)
    )
    frame = np.zeros((4, 4), dtype=np.uint8)

    result = session.score(frame)

    assert result == "SCORED"
    assert shaping.centers == [(7.0, 9.0)]
    assert [name for name, _ in shaping.events] == ["center", "score"]
    assert shaping.events[1][1] is frame


def test_score_recenters_from_frame_when_no_blaze(monkeypatch):
    """Without a blaze probe the frame's own 0-order position becomes the
    centre, and it is set *before* the score call."""
    monkeypatch.setattr(engine, "zero_order_center", lambda img: (3.0, 4.0))
    shaping = _FakeShaping()
    session = _session(shaping=shaping)
    frame = np.zeros((4, 4), dtype=np.uint8)

    session.score(frame)

    assert shaping.centers == [(3.0, 4.0)]
    assert [name for name, _ in shaping.events] == ["center", "score"]
