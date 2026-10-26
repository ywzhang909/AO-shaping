"""Offline tests for the sidebar exposure probe (bench_settings).

The probe is the only sidebar control that **reaches a device**, so it is
tested against a stub camera rather than by asserting on rendered widgets
alone: what matters is that it derives the working exposure from the measured
ladder, re-arms the camera afterwards, records the evidence on disk, and
refuses rather than inventing a number when the measurement cannot be made.

No Streamlit server and no hardware: ``st.session_state`` is a plain dict and
the camera is a stub, same convention as ``test_bench_settings.py``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.utils.image import exposure_control as ec


@pytest.fixture(autouse=True)
def state(monkeypatch):
    """Patch ``st.session_state`` with a plain dict and stub the window probe.

    Same convention as ``test_bench_settings.py``: bare-mode Streamlit needs a
    real dict, and the window probe would otherwise open the SDK.
    """
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(bs, "probe_camera_window", lambda *a, **k: None)
    return state


class _StubCamera:
    """Camera stub whose peak rises linearly with exposure, saturating at 0.2 ms.

    Mirrors the shape the real bench showed on 2026-10-17: dim at the 0.02 ms
    driver clamp, a comfortable peak in between, clipped well before the
    camera's 1000 ms hardware maximum. ``peak = 255 * ms / SATURATES_AT`` is the
    same linear-response assumption ``estimate_exposure_ceiling_ms`` makes.
    """

    SATURATES_AT = 0.2

    def __init__(self) -> None:
        self.current_ms = 0.02
        self.applied: list[float] = []
        self.closed = False

    def reset_exposure_time(self, ms: float) -> float:
        self.current_ms = float(ms)
        self.applied.append(float(ms))
        return self.current_ms

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        peak = int(round(255.0 * min(1.0, self.current_ms / self.SATURATES_AT)))
        frame = np.zeros((8, 8), dtype=np.uint8)
        frame[4, 4] = peak
        return frame

    def __enter__(self) -> "_StubCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


@pytest.fixture
def stub_camera(monkeypatch):
    """Install the stub camera and an unset env ceiling (so the probe runs)."""
    cam = _StubCamera()
    monkeypatch.delenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
    monkeypatch.setattr(
        "ao_shaping.drivers.ccd.common.create_camera",
        lambda *a, **k: cam,
    )
    return cam


class TestWorkingExposureFormula:
    """``working_exposure_ms`` is the shared 'probe said X, use Y' formula."""

    def test_half_the_ceiling(self) -> None:
        assert ec.working_exposure_ms(0.4) == pytest.approx(0.2)

    def test_fraction_is_configurable(self) -> None:
        assert ec.working_exposure_ms(0.4, fraction=0.75) == pytest.approx(0.3)

    def test_bright_bench_still_lands_on_the_driver_floor(self) -> None:
        """A ceiling below the driver's own clamp must not go under it."""
        assert ec.working_exposure_ms(0.01, floor_ms=0.02) == pytest.approx(0.02)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_a_nonsense_ceiling(self, bad: float) -> None:
        with pytest.raises(ValueError, match="ceiling_ms"):
            ec.working_exposure_ms(bad)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_a_nonsense_fraction(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            ec.working_exposure_ms(0.4, fraction=bad)


class TestProbeSessionExposure:
    def test_working_exposure_is_half_the_probed_ceiling(
        self, stub_camera, tmp_path
    ) -> None:
        working, evidence = bs.probe_session_exposure_ms(
            debug_dir=tmp_path / "probe"
        )
        ceiling = evidence["ceiling_ms"]
        assert ceiling > 0
        assert working == pytest.approx(ec.working_exposure_ms(ceiling))

    def test_camera_is_rearmed_to_the_working_exposure(
        self, stub_camera, tmp_path
    ) -> None:
        """The ladder leaves the camera at its last step; the caller owns re-arming.

        Without this the *next* measurement silently runs at the saturated step,
        which is exactly the 1.0 ms reading that made the first beam-extent scan
        on 2026-10-17 look like a uniformly bright, meaningless field.
        """
        working, _ = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert stub_camera.applied[-1] == pytest.approx(working)

    def test_confirming_frame_is_clipped(
        self, stub_camera, tmp_path
    ) -> None:
        """The returned peak must be the confirming frame, read at the working exposure."""
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["peak"] > 0
        assert 0.0 < evidence["peak_fraction"] < 1.0
        assert evidence["peak"] == pytest.approx(
            255.0 * min(1.0, evidence["working_ms"] / _StubCamera.SATURATES_AT), abs=2
        )

    def test_evidence_is_written_next_to_the_other_probes(
        self, stub_camera, tmp_path
    ) -> None:
        target = tmp_path / "exposure_probe"
        _, evidence = bs.probe_session_exposure_ms(debug_dir=target)
        files = list(target.glob("*.json"))
        assert len(files) == 1, files
        assert json.loads(files[0].read_text(encoding="utf-8")) == evidence

    def test_evidence_default_dir_is_under_data_debug(self) -> None:
        parts = bs.EXPOSURE_PROBE_DEBUG_DIR.parts
        assert parts[0] == "data" and parts[1] == "debug"
        assert parts[-1] == "exposure_probe"

    def test_a_clipping_working_exposure_is_stepped_down(
        self, stub_camera, tmp_path
    ) -> None:
        """The ladder's ceiling is a *model*; the returned value must be verified.

        Measured 2026-10-17: the ladder put the ceiling at 0.243 ms, and half of
        it -- the "working" exposure -- read 255/255 on the confirming frame. The
        first version returned it anyway, so the GUI would have shipped a clipped
        exposure described as a working one.
        """
        cam = stub_camera
        cam.SATURATES_AT = 0.05  # ladder will think the bench is very bright
        working, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        fractions = [c["peak_fraction"] for c in evidence["confirmations"]]
        assert evidence["peak_fraction"] < bs._CONFIRM_SATURATION
        assert fractions[-1] < bs._CONFIRM_SATURATION
        assert working < evidence["ceiling_ms"]

    def test_an_unsalvageable_bench_raises_instead_of_returning_a_clip(
        self, stub_camera, tmp_path
    ) -> None:
        cam = stub_camera
        cam.SATURATES_AT = 0.02  # saturated even at the 0.02 ms floor
        with pytest.raises(bs.ExposureProbeFailure, match="仍然饱和|曝光探测失败"):
            bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")

    def test_camera_is_closed_even_when_the_probe_fails(
        self, monkeypatch, tmp_path
    ) -> None:
        """A probe that leaves the handle open blocks the next run (USB exclusive)."""
        cam = _StubCamera()
        monkeypatch.delenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
        monkeypatch.setattr(
            "ao_shaping.drivers.ccd.common.create_camera", lambda *a, **k: cam
        )
        monkeypatch.setattr(
            ec, "probe_exposure_ceiling_ms", lambda *a, **k: (_ for _ in ()).throw(
                ValueError("no ladder frame carries signal")
            )
        )
        with pytest.raises(bs.ExposureProbeFailure, match="no ladder frame"):
            bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert cam.closed is True

    def test_refuses_a_camera_type_it_cannot_serve(self, tmp_path) -> None:
        """Exposure semantics differ per camera; guessing would be worse than asking."""
        with pytest.raises(bs.ExposureProbeFailure, match="只支持大恒"):
            bs.probe_session_exposure_ms("miicam", 0, debug_dir=tmp_path / "probe")

    def test_configured_ceiling_wins_over_probing(
        self, stub_camera, monkeypatch, tmp_path
    ) -> None:
        """``AO_DAHENG_MAX_EXPOSURE_MS`` is an operator decision and outranks the probe."""
        monkeypatch.setenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, "0.08")
        working, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["ceiling_ms"] == pytest.approx(0.08)
        assert evidence["ceiling_source"] == "env"
        assert working == pytest.approx(0.04)

    def test_the_ladder_is_walked_exactly_once(
        self, stub_camera, monkeypatch, tmp_path
    ) -> None:
        """Two walks cost double and can contradict each other.

        The first version called ``resolve_daheng_exposure_limit(camera=...)``
        *and* probed itself. On a bench whose light drifts -- measured
        2026-10-17, peak at 0.1 ms moved 63 -> 79 within 20 minutes -- the two
        walks can return different ceilings, and the evidence recorded would not
        be the evidence that decided the value.
        """
        walks: list[int] = []
        real_probe = ec.probe_exposure_ceiling_ms

        def counting_probe(*a, **k):
            walks.append(1)
            return real_probe(*a, **k)

        monkeypatch.setattr(ec, "probe_exposure_ceiling_ms", counting_probe)
        bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert len(walks) == 1

    def test_env_ceiling_skips_the_ladder_entirely(
        self, stub_camera, monkeypatch, tmp_path
    ) -> None:
        """An operator-set ceiling is a decision, not a measurement to redo."""
        monkeypatch.setenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, "0.08")
        monkeypatch.setattr(
            ec,
            "probe_exposure_ceiling_ms",
            lambda *a, **k: pytest.fail("must not walk the ladder when env sets one"),
        )
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["ceiling_source"] == "env"


class _UnboundedStubCamera(_StubCamera):
    """A dark bench: the sensor integrates to 0.5 ms but never saturates there.

    ``255 * ms / 1.0`` means clipping would need 1 ms, past the camera's own
    maximum, so the ladder has **no** measured saturation point at all -- the
    camera maximum is the only bound available.
    """

    SATURATES_AT = 1.0

    def get_exposure_range(self) -> tuple[float, float]:
        return 0.02, 0.5


def _install_camera(monkeypatch, cam) -> None:
    """Route ``create_camera`` to ``cam`` and leave the env ceiling unset."""
    monkeypatch.delenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
    monkeypatch.setattr(
        "ao_shaping.drivers.ccd.common.create_camera", lambda *a, **k: cam
    )


class TestCameraMaximumBoundsTheCeiling:
    """The GUI path must ask the camera how far it can integrate.

    ``probe_exposure_ceiling_ms`` caps at ``hard_max_ms``, but the sidebar probe
    used to call it without one, so on a bench that never saturates inside the
    camera's reachable range the reported ceiling could exceed the camera
    maximum.  ``working_exposure_ms`` would then hand the driver an exposure it
    silently clamps, and the measured peak would not belong to the requested
    exposure.
    """

    def test_the_probe_is_capped_at_the_cameras_maximum_exposure(
        self, monkeypatch, tmp_path
    ) -> None:
        cam = _UnboundedStubCamera()
        _install_camera(monkeypatch, cam)
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        # The fit alone extrapolates to ~0.90 ms, well past the 0.5 ms maximum.
        assert evidence["fit_ms"] > 0.5
        assert evidence["ceiling_ms"] == pytest.approx(0.5 * 0.9)
        assert evidence["bound_ms"] == pytest.approx(0.5)
        assert evidence["camera_max_ms"] == pytest.approx(0.5)
        assert max(cam.applied) <= 0.5 + 1e-9

    def test_a_saturation_below_the_camera_maximum_is_still_measured(
        self, stub_camera, monkeypatch, tmp_path
    ) -> None:
        """The cap must not cost the saturation measurement.

        The walk breaks on the first frame at the saturation fraction *before*
        it can reach the cap, so passing ``hard_max_ms`` cannot hide a
        saturation the ladder can actually see: here the 0.32 ms rung is the
        first at 90 % of full scale, on a camera that goes to 1000 ms.
        """
        stub_camera.get_exposure_range = lambda: (0.02, 1000.0)
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["camera_max_ms"] == pytest.approx(1000.0)
        assert evidence["saturated_ms"] == pytest.approx(0.32)
        # Stopped on the saturation, far short of the cap -- not at the cap.
        assert max(stub_camera.applied) == pytest.approx(0.32)

    def test_a_configured_ceiling_is_still_capped_by_the_camera(
        self, monkeypatch, tmp_path
    ) -> None:
        """The env value stays the tighter of (operator decision, sensor limit)."""
        cam = _UnboundedStubCamera()
        _install_camera(monkeypatch, cam)
        monkeypatch.setenv(ec.DAHENG_EXPOSURE_LIMIT_ENV, "5.0")
        working, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["ceiling_source"] == "env"
        assert evidence["ceiling_ms"] == pytest.approx(0.5)
        assert evidence["camera_max_ms"] == pytest.approx(0.5)
        assert working == pytest.approx(0.25)

    def test_a_camera_without_a_range_leaves_the_probe_unbounded(
        self, monkeypatch, tmp_path
    ) -> None:
        """A backend that cannot report its range must not fail the probe."""
        cam = _UnboundedStubCamera()

        def no_range() -> tuple[float, float]:
            raise NotImplementedError("this backend has no exposure range")

        cam.get_exposure_range = no_range
        _install_camera(monkeypatch, cam)
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")
        assert evidence["camera_max_ms"] is None
        # Uncapped, exactly as before: the ceiling runs past the 0.5 ms the
        # camera would have imposed had it reported a range.
        assert evidence["ceiling_ms"] > 0.5 * 0.9



class TestSidebarButtonWiring:
    """The button has to behave like the pixel-size probe next to it."""

    def test_button_writes_a_pending_value_not_a_widget_key(
        self, state, monkeypatch, stub_camera
    ) -> None:
        """Widget keys are locked once instantiated, so the write must be queued."""
        calls: list[dict[str, Any]] = []

        def fake_button(label, **kwargs):
            calls.append({"label": label, **kwargs})
            return False

        monkeypatch.setattr(st, "button", fake_button, raising=False)
        state["bench_cam_type"] = "daheng"
        state["bench_cam_id"] = 0
        state["_cam_pixel_checked"] = True  # skip the pixel auto-check warning

        bs.render_bench_settings()

        exposure_btn = next(
            c for c in calls if c.get("key") == "detect_exposure_btn"
        )
        assert exposure_btn["disabled"] is False
        # No click happened, so nothing may be queued yet.
        assert bs.pending_key("") not in state

    def test_button_is_disabled_for_a_non_daheng_camera(
        self, state, monkeypatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_button(label, **kwargs):
            calls.append({"label": label, **kwargs})
            return False

        monkeypatch.setattr(st, "button", fake_button, raising=False)
        state["bench_cam_type"] = "miicam"
        state["bench_cam_id"] = 0
        state["_cam_pixel_checked"] = True

        bs.render_bench_settings()

        exposure_btn = next(
            c for c in calls if c.get("key") == "detect_exposure_btn"
        )
        assert exposure_btn["disabled"] is True