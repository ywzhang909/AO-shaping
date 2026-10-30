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
import re
from datetime import datetime
from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs
from ao_shaping.gui.slm import exposure_status as es
from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.utils.image import exposure_control as ec


class _Rerun(Exception):
    """``st.rerun()`` 在裸模式没有脚本运行上下文,所以用异常代替「重跑一次」。"""


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
        assert evidence["peak_fraction"] < es._CONFIRM_SATURATION
        assert fractions[-1] < es._CONFIRM_SATURATION
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


class TestExposureStaleness:
    """L2：实测曝光带取样时刻，过期只上黄点，绝不自动改值（#215 C3）。"""

    def test_a_fresh_probe_is_not_stale(self, state) -> None:
        state[bs.detected_key("")] = {
            "exposure_ms": 0.15,
            "probed_at": datetime.now().timestamp(),
        }

        info = bs.exposure_state("")

        assert info["stale"] is False
        assert info["age_s"] == pytest.approx(0.0, abs=5.0)
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", info["probed_at_text"])

    def test_three_hours_old_is_stale(self, state) -> None:
        """2 h 是分界：实测 1.5 h 后光强漂 ×1.35,所以 3 h 必然该重测。"""
        state[bs.detected_key("")] = {
            "exposure_ms": 0.15,
            "probed_at": datetime.now().timestamp() - 3 * 3600,
        }

        info = bs.exposure_state("")

        assert info["stale"] is True
        assert info["age_s"] == pytest.approx(3 * 3600, abs=5.0)

    def test_a_hand_typed_exposure_is_not_called_stale(self, state) -> None:
        """没有取样时刻 ≠ 陈旧：手填值不该被点黄点催着「重测」。"""
        state[bs.detected_key("")] = {"exposure_ms": 0.3}

        info = bs.exposure_state("")

        assert info["exposure_ms"] == pytest.approx(0.3)
        assert info["probed_at"] is None
        assert info["probed_at_text"] == ""
        assert info["age_s"] is None
        assert info["stale"] is False

    def test_per_slm_status_falls_back_to_the_experiment_record(self, state) -> None:
        """曝光是实验级字段：某台 SLM 的探测表没有它也要读出同一条记录。"""
        state[bs.detected_key("")] = {
            "exposure_ms": 0.15,
            "probed_at": datetime.now().timestamp() - 3 * 3600,
        }

        assert bs.exposure_state("slm2")["exposure_ms"] == pytest.approx(0.15)
        assert bs.exposure_state("slm2")["stale"] is True

    def test_record_exposure_probe_stamps_the_sample_moment(self, state) -> None:
        moment = datetime.now().timestamp() - 60

        written = bs.record_exposure_probe(
            "", 0.21, {"probed_at": moment, "probed_at_text": "12:00:01"}
        )

        assert written["exposure_ms"] == pytest.approx(0.21)
        assert written["probed_at"] == pytest.approx(moment)
        assert written["probed_at_text"] == "12:00:01"
        assert bs.exposure_state("")["stale"] is False

    def test_probe_evidence_carries_the_session_and_sample_moment(
        self, stub_camera, tmp_path
    ) -> None:
        """证据 JSON 只**追加** session_id / probed_at，不改既有键与返回值。"""
        _, evidence = bs.probe_session_exposure_ms(debug_dir=tmp_path / "probe")

        assert evidence["probed_at"] == pytest.approx(
            datetime.now().timestamp(), abs=60.0
        )
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", evidence["probed_at_text"])
        assert evidence["session_id"] == ""  # 还没标定过 ⇒ 空串,不伪造编号
        assert "ceiling_ms" in evidence  # 既有键一个不少


class TestExposureStatusBar:
    """状态栏：说清「当前多少 ms / 实测什么时候 / 是否该重测」，从不自己改值。"""

    def _captions(self, monkeypatch) -> list[str]:
        lines: list[str] = []

        def fake_caption(text, *a, **k):
            lines.append(str(text))

        monkeypatch.setattr(st, "caption", fake_caption, raising=False)
        monkeypatch.setattr(
            st, "button", lambda *a, **k: False, raising=False
        )
        return lines

    def test_a_fresh_probe_shows_its_time_and_no_yellow_dot(
        self, state, monkeypatch
    ) -> None:
        lines = self._captions(monkeypatch)
        state["bench_exposure_ms"] = 0.15
        state[bs.detected_key("")] = {
            "exposure_ms": 0.15,
            "probed_at": datetime.now().timestamp(),
            "probed_at_text": "12:31:05",
        }

        sc.render_exposure_status("slm1")

        assert "重测于 12:31:05" in lines[-1]
        assert "0.150 ms" in lines[-1]
        assert "🟡" not in lines[-1]

    def test_a_stale_probe_adds_a_yellow_dot_without_touching_the_value(
        self, state, monkeypatch
    ) -> None:
        lines = self._captions(monkeypatch)
        state["bench_exposure_ms"] = 0.15
        state[bs.detected_key("")] = {
            "exposure_ms": 0.15,
            "probed_at": datetime.now().timestamp() - 3 * 3600,
        }

        sc.render_exposure_status("slm1")

        assert "🟡" in lines[-1]
        # 不自动改：控件值与探测记录都还在原处。
        assert state["bench_exposure_ms"] == pytest.approx(0.15)
        assert state[bs.detected_key("")]["exposure_ms"] == pytest.approx(0.15)

    def test_the_button_runs_the_sidebar_probe_path(
        self, state, monkeypatch, stub_camera, tmp_path
    ) -> None:
        self._captions(monkeypatch)
        monkeypatch.setattr(st, "button", lambda *a, **k: True, raising=False)
        monkeypatch.setattr(
            st, "rerun", lambda *a, **k: (_ for _ in ()).throw(_Rerun()), raising=False
        )
        monkeypatch.setattr(es, "EXPOSURE_PROBE_DEBUG_DIR", tmp_path / "probe")
        state["bench_exposure_ms"] = 0.02
        state["bench_cam_type"] = "daheng"

        with pytest.raises(_Rerun):
            sc.render_exposure_status("slm1")

        assert state[bs.pending_key("")]["bench_exposure_ms"] > 0.02
        recorded = state[bs.detected_key("")]["exposure_ms"]
        assert recorded == pytest.approx(stub_camera.applied[-1])
        assert bs.exposure_state("")["probed_at"] is not None