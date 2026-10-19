"""Offline tests for the defocus/focus trim over an accepted GS phase.

The trim is a *manual* adjustment layered on an already-accepted GS solution, so
three properties are load-bearing — and all three fail silently when broken:

1. **no accumulation** — every application recomputes from the cached base
   phase, so dragging −40 → +40 → −20 cannot leave the phase where −20 alone
   would have put it. A naive in-place ``phase += defocus`` would drift by the
   sum of every step, and the drift is invisible on screen.
2. **zero is exactly the base phase** — the restore path must be bit-identical
   to what GS produced, not merely close.
3. **no full-page rerun, no dispatch on drag** — the slider must not touch the
   SLM and must not re-render the page; only the buttons dispatch. This is also
   the regression that motivated the rewrite: the block used to hang off
   ``show_phase``, which runs once per accepted round, so the first slider
   interaction re-rendered the page, the block was never re-rendered, and the
   operator lost it until the next run.

The trim is centred on the *measured* pupil centre and uses the canonical
Zernike generator (``zernike_panel``), not a hand-rolled defocus.

No Streamlit server, no hardware: the SLM is a stub, ``zernike_panel`` is the
real (pure-numpy) function, and ``st`` is patched to a plain-dict session state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import shaping_controls as sc
from ao_shaping.tools.kernels.bench_kernels import fresnel_panel, zernike_panel

PREFIX = "slm1"
SLM_NUM = 1


class _StubSLM:
    """Records what was displayed; no device, no driver pipeline."""

    def __init__(self, shape: tuple[int, int] = (120, 160)) -> None:
        self.shape = shape
        self.displayed: list[np.ndarray] = []

    def create_phase_from_array(self, phase: np.ndarray) -> np.ndarray:
        a = np.asarray(phase, dtype=np.float64)
        # The driver wraps to 2pi; mimic only enough to type-check the call.
        return np.mod(a, 2 * np.pi).astype(np.float32)

    def display_data(self, phase_gray: np.ndarray, *, memory_mode: int = 0) -> None:
        self.displayed.append(np.asarray(phase_gray))


def _same_mod2pi(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> bool:
    """Compare two phases on the circle, not on the real line.

    The stub wraps with ``mod 2*pi`` like the driver; two phases differing by
    1e-9 rad can straddle that boundary and compare as 0 vs 6.28.
    """
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + np.pi) % (
        2 * np.pi
    ) - np.pi
    return bool(np.allclose(d, 0.0, atol=atol))


def _base(seed: int = 0, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=shape)


def _combined(base: np.ndarray, amount: float, centre, radius: float) -> np.ndarray:
    if abs(amount) < 1e-9:
        return np.asarray(base, dtype=np.float64)
    defocus = zernike_panel(
        {(2, 0): float(amount)},
        radius=int(round(radius)),
        pupil_center=centre,
        panel_shape=base.shape,
    )
    return np.asarray(base, dtype=np.float64) + defocus


@pytest.fixture
def bench(monkeypatch):
    """Patch Streamlit to plain-dict state and scriptable widgets.

    Returns a handle with ``state`` (the session dict), ``slm`` (a stub), and
    ``reruns`` / ``errors`` / ``successes`` recorders.
    """
    import contextlib

    h: dict[str, Any] = {
        "state": {},
        "amount": 0.0,
        "manual_focal": 0.0,
        "click": None,
        "reruns": [],
        "errors": [],
        "successes": [],
        "sliders": [],
    }
    monkeypatch.setattr(st, "session_state", h["state"])
    monkeypatch.setattr(sc, "refresh_phase_preview", lambda n: None)
    monkeypatch.setattr(st, "expander", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: h["successes"].append(a[0]))
    monkeypatch.setattr(st, "error", lambda *a, **k: h["errors"].append(a[0]))
    monkeypatch.setattr(st, "columns", lambda n, **k: [contextlib.nullcontext() for _ in range(n)])
    monkeypatch.setattr(
        st, "slider", lambda label, **k: h["sliders"].append(label) or h["amount"]
    )
    monkeypatch.setattr(
        st, "number_input", lambda label, **k: h["manual_focal"]
    )
    monkeypatch.setattr(st, "button", lambda label, **k: label == h["click"])
    monkeypatch.setattr(
        st, "rerun", lambda *a, **k: h["reruns"].append("rerun"), raising=False
    )
    h["slm"] = _StubSLM()
    return h


def _arm(bench, base_phase=None) -> None:
    """Connect the stub SLM and cache a base phase (i.e. a run has happened)."""
    bench["state"][PREFIX] = bench["slm"]
    bench["state"][f"{PREFIX}_gs_base_phase"] = (
        _base() if base_phase is None else base_phase
    )
    # The trim derives its aperture and its centre from the sidebar; pin both so
    # the expected phase is unambiguous (the sidebar default is 2x span, and the
    # fallback centre 960,690 does not exist on this 120x160 test panel).
    bench["state"][f"{PREFIX}_bench_beam_radius_px"] = 40.0
    bench["state"][f"{PREFIX}_bench_pupil_center"] = "80,60"


def _render() -> None:
    """Drive the fragment's body.

    ``st.fragment`` wraps the callable so it only runs inside a live script run;
    called bare it is a no-op. Offline that makes the wrapper untestable *and*
    unexercised, so the logic tests drive ``__wrapped__`` directly, and a
    separate test asserts the entry point really is fragment-wrapped (that is
    what keeps a slider drag off the full-page rerun).
    """
    body = getattr(sc.defocus_trim_fragment, "__wrapped__", sc.defocus_trim_fragment)
    body(PREFIX, SLM_NUM)


class TestFragmentContract:
    def test_the_entry_point_is_a_fragment(self) -> None:
        """This is the property that stops a slider drag re-rendering the page.

        It is also the one thing offline tests can only check structurally --
        the fragment wrapper is a no-op outside a script run, so its runtime
        behaviour needs a real Streamlit run to verify.
        """
        assert hasattr(sc.defocus_trim_fragment, "__wrapped__")

    def test_it_is_a_module_level_function_not_a_method(self) -> None:
        """A fragment created per render would re-register itself every pass."""
        import inspect

        assert inspect.isfunction(sc.defocus_trim_fragment)
        assert not inspect.ismethod(sc.defocus_trim_fragment)


class TestTrimArithmetic:
    def test_zero_is_exactly_the_base_phase(self) -> None:
        base = _base()
        assert np.array_equal(_combined(base, 0.0, (80, 60), 40.0), base)

    def test_the_same_amount_always_gives_the_same_phase(self) -> None:
        """The property an in-place ``+=`` implementation would violate."""
        base = _base()
        centre, radius = (80.0, 60.0), 40.0
        first = _combined(base, -20.0, centre, radius)
        _ = _combined(base, 40.0, centre, radius)  # the operator dragged past
        again = _combined(base, -20.0, centre, radius)
        assert np.allclose(first, again, atol=1e-12)

    def test_positive_and_negative_defocus_are_opposite(self) -> None:
        base = np.zeros((120, 160))
        pos = _combined(base, 12.0, (80, 60), 40.0)
        neg = _combined(base, -12.0, (80, 60), 40.0)
        assert np.allclose(pos, -neg, atol=1e-12)
        assert np.any(pos > 0)

    def test_defocus_is_centred_on_the_measured_centre(self) -> None:
        """Moving the centre must move the pattern — it is not panel-centred."""
        a = _combined(np.zeros((120, 160)), 10.0, (80, 60), 30.0)
        b = _combined(np.zeros((120, 160)), 10.0, (100, 60), 30.0)
        assert not np.allclose(a, b)

    def test_the_defocus_is_confined_to_the_aperture(self) -> None:
        out = _combined(np.zeros((120, 160)), 10.0, (80, 60), 20.0)
        assert out[0, 0] == 0.0
        assert out[60, 80] != 0.0


class TestProbedFocalLength:
    """The Fresnel term is the *coarse* defocus; the Zernike slider is the fine one.

    The camera moving behind the 2f plane is a bench fact measured once, so the
    trim must consume the probe's number by default and let an operator override
    it — while never letting a stale probe result silently apply.
    """

    def test_the_probe_result_is_used_when_the_manual_field_is_zero(
        self, bench
    ) -> None:
        _arm(bench)
        bench["state"]["slm1_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["manual_focal"], bench["amount"] = 0.0, 0.0
        bench["click"] = "叠加并下发"

        _render()

        slm = bench["slm"]
        assert len(slm.displayed) == 1
        expected = _base() + fresnel_panel(
            -1680.0, radius=40, pupil_center=(80, 60), panel_shape=(120, 160)
        )
        assert _same_mod2pi(slm.displayed[0], expected)
        assert any("菲涅尔" in m for m in bench["successes"])

    def test_a_manual_focal_length_overrides_the_probe(self, bench) -> None:
        _arm(bench)
        bench["state"]["slm1_bench_focus_probe"] = {"best_focal_mm": -1680.0}
        bench["manual_focal"], bench["amount"] = 900.0, 0.0
        bench["click"] = "叠加并下发"

        _render()

        slm = bench["slm"]
        expected = _base() + fresnel_panel(
            900.0, radius=40, pupil_center=(80, 60), panel_shape=(120, 160)
        )
        assert _same_mod2pi(slm.displayed[0], expected)

    def test_without_a_probe_and_without_a_manual_value_nothing_is_added(
        self, bench
    ) -> None:
        _arm(bench)
        bench["manual_focal"], bench["amount"] = 0.0, 0.0
        bench["click"] = "叠加并下发"

        _render()

        slm = bench["slm"]
        assert np.allclose(
            slm.displayed[0], slm.create_phase_from_array(_base()), atol=1e-6
        )

    def test_the_fresnel_and_zernike_terms_compose(self, bench) -> None:
        """Both are quadratic phases over the same disc, so they add."""
        _arm(bench)
        bench["manual_focal"], bench["amount"] = -1680.0, 4.0
        bench["click"] = "叠加并下发"

        _render()

        slm = bench["slm"]
        expected = (
            _base()
            + fresnel_panel(
                -1680.0, radius=40, pupil_center=(80, 60), panel_shape=(120, 160)
            )
            + zernike_panel(
                {(2, 0): 4.0},
                radius=40,
                pupil_center=(80, 60),
                panel_shape=(120, 160),
            )
        )
        assert _same_mod2pi(slm.displayed[0], expected)
        assert any("菲涅尔" in m and "Zernike" in m for m in bench["successes"])


class TestCentreParsing:
    def test_parses_the_sidebar_format(self) -> None:
        assert sc._parse_centre("960,720") == (960.0, 720.0)

    def test_falls_back_on_garbage(self) -> None:
        got = sc._parse_centre("not a point")
        assert len(got) == 2 and all(isinstance(v, float) for v in got)

    def test_falls_back_on_the_wrong_arity(self) -> None:
        assert len(sc._parse_centre("960")) == 2


class TestNoRerunContract:
    def test_dragging_does_not_dispatch_and_does_not_rerun(self, bench) -> None:
        _arm(bench)
        bench["amount"], bench["click"] = -40.0, None

        _render()

        assert bench["slm"].displayed == [], "dragging must not touch the SLM"
        assert bench["reruns"] == [], "dragging must not rerun the page"

    def test_apply_dispatches_exactly_once_without_rerunning(self, bench) -> None:
        _arm(bench)
        bench["amount"], bench["click"] = -40.0, "叠加并下发"

        _render()

        assert len(bench["slm"].displayed) == 1
        assert bench["reruns"] == []

    def test_restore_dispatches_the_untrimmed_phase(self, bench) -> None:
        base = _base()
        _arm(bench, base)
        bench["amount"], bench["click"] = -40.0, "恢复原始 GS 相位"

        _render()

        slm = bench["slm"]
        assert len(slm.displayed) == 1
        assert np.allclose(slm.displayed[0], slm.create_phase_from_array(base), atol=1e-6)
        assert bench["reruns"] == []

    def test_the_block_survives_reruns_without_a_new_run(self, bench) -> None:
        """The regression that started all this: the block vanished on rerun."""
        _arm(bench)
        bench["click"] = None

        for _ in range(3):
            _render()

        assert len(bench["sliders"]) == 3, "the block must re-render every pass"

    def test_nothing_renders_before_a_run_has_produced_a_base_phase(
        self, bench
    ) -> None:
        bench["state"][PREFIX] = bench["slm"]  # connected, but never ran
        _render()
        assert bench["sliders"] == []

    def test_a_disconnected_slm_is_reported_not_skipped(self, bench) -> None:
        """Silently doing nothing would leave the operator thinking it shipped."""
        bench["state"][f"{PREFIX}_gs_base_phase"] = _base()  # no handle
        bench["click"] = "叠加并下发"

        _render()

        assert bench["slm"].displayed == []
        assert bench["errors"] and "未连接" in bench["errors"][0]


class TestDispatch:
    def test_a_centre_outside_the_panel_does_not_crash(self, bench) -> None:
        """The sidebar fallback centre can exceed a small panel; it must clamp.

        ``zernike_panel`` computes an empty target slice when the centre is
        outside, then assigns the whole patch into it — a broadcast error. The
        first version of this trim hit exactly that and silently displayed
        nothing at all.
        """
        _arm(bench)
        sc._dispatch_defocus(SLM_NUM, 10.0, (960.0, 690.0), 40.0)
        assert bench["errors"] == []
        assert len(bench["slm"].displayed) == 1

    def test_a_driver_failure_is_surfaced_not_raised(self, bench) -> None:
        class _Broken(_StubSLM):
            def create_phase_from_array(self, phase):
                raise RuntimeError("SLM handle is closed")

        _arm(bench)
        bench["state"][PREFIX] = _Broken()

        sc._dispatch_defocus(SLM_NUM, 10.0, (80.0, 60.0), 40.0)

        assert bench["errors"] and "SLM handle is closed" in bench["errors"][0]

    def test_no_base_phase_is_reported(self, bench) -> None:
        bench["state"][PREFIX] = bench["slm"]
        sc._dispatch_defocus(SLM_NUM, 10.0, (80.0, 60.0), 40.0)
        assert bench["errors"] and "基准相位" in bench["errors"][0]

    def test_a_zero_amount_reports_the_restore_wording(self, bench) -> None:
        _arm(bench)
        sc._dispatch_defocus(SLM_NUM, 0.0, (80.0, 60.0), 40.0)
        assert any("恢复" in m for m in bench["successes"])

    def test_a_nonzero_amount_reports_the_signed_amount(self, bench) -> None:
        _arm(bench)
        sc._dispatch_defocus(SLM_NUM, -40.0, (80.0, 60.0), 40.0)
        assert any("-40" in m for m in bench["successes"])