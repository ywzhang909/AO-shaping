"""Tests for the MIL record report's shaped-phase handling.

Lane B fix: the report must distinguish the phase FINALLY ADOPTED
(``best_phase.npy``, which is a flat zero field when bake-off rejected the
shaping phase) from the phase Step B actually synthesised and dropped
(``shaped_phase.npy``).  The reader prefers the new root ``shaped_phase.npy``
over the last ``split == "shaped"`` probe, and never invents a shaped phase
when none exists.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slm import generate_model_in_loop_record_report as gen


def _load_generator():
    """Import the report generator by path. (Helps catch import trouble
    separate from the package import above.)"""
    script = ROOT / "scripts" / "slm" / "generate_model_in_loop_record_report.py"
    spec = importlib.util.spec_from_file_location("_mil_report", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probe(split: str, phase: np.ndarray) -> gen.ProbeEntry:
    return gen.ProbeEntry(
        name=f"round_01_{split}_00.npz",
        round=1,
        split=split,
        index=0,
        family=None,
        exposure_ms=None,
        timestamp_utc=None,
        manifest_correlation=None,
        phase=np.asarray(phase, dtype=np.float64),
        frame=np.zeros((8, 8), dtype=np.float64),
        raw_frame=None,
        model_grid=None,
        prediction=None,
    )


def _bundle(**overrides) -> gen.RunBundle:
    """A minimal RunBundle with an aberration present so phases are usable."""
    shape = (16, 16)
    # Coeffs are a VECTOR (Noll-indexed); aberration phase is built inside.
    coefficients_vec = np.zeros(12, dtype=np.float64)
    coefficients_vec[2] = 0.5  # defocus-ish nonzero
    base: dict[str, object] = {
        "run_dir": Path("/nonexistent/run"),
        "stamp": "test",
        "variant": "baseline_a",
        "config": {"region": 16},
        "geometry": {},
        "handoff": {},
        "status": "bake_off_rejected",
        "error_terms": None,
        "failure": None,
        "rows": [],
        "coefficients": coefficients_vec,
        "best_phase": None,
        "shaped_phase": None,
        "source_amplitude": None,
        "flat_frame": None,
        "probes": [],
        "trace_a": gen.Trace(),
        "trace_b": gen.Trace(),
        "rotation_quarters": 0,
        "rotation_residual": 0.0,
        "roi_center": (0, 0),
        "cell_scale": None,
        "far_field_size": None,
    }
    return _make_bundle(base, overrides)


def _make_bundle(base: dict[str, object], overrides: dict[str, object]) -> gen.RunBundle:
    # Build RunBundle positionally-safe via keyword matching on the dataclass.
    merged = {**base, **overrides}
    return gen.RunBundle(**merged)


class TestShapedPhaseResolution:
    """``_shaped_phase`` picks the new root file first, then the probe."""

    def test_prefers_root_file_over_probe(self):
        shaped = np.ones((16, 16)) * 0.9
        probe_phase = np.ones((16, 16)) * 0.3
        bundle = _bundle(
            shaped_phase=shaped,
            probes=[_probe("shaped", probe_phase)],
        )
        out = gen._shaped_phase(bundle)
        np.testing.assert_allclose(out, shaped)

    def test_falls_back_to_last_shaped_probe(self):
        probe_phase = np.ones((16, 16)) * 0.3
        bundle = _bundle(
            shaped_phase=None,
            probes=[_probe("shaped", probe_phase)],
        )
        out = gen._shaped_phase(bundle)
        np.testing.assert_allclose(out, probe_phase)

    def test_none_when_neither(self):
        bundle = _bundle(shaped_phase=None, probes=[])
        assert gen._shaped_phase(bundle) is None

    def test_source_label_root_file(self):
        bundle = _bundle(shaped_phase=np.ones((16, 16)), probes=[])
        assert gen._shaped_source_label(bundle) == "shaped_phase.npy"

    def test_source_label_probe_fallback(self):
        bundle = _bundle(
            shaped_phase=None,
            probes=[_probe("shaped", np.ones((16, 16)))],
        )
        assert gen._shaped_source_label(bundle) == "round_01_shaped_00.npz"

    def test_source_label_unrecorded(self):
        bundle = _bundle(shaped_phase=None, probes=[])
        assert gen._shaped_source_label(bundle) == "未记录"


class TestReportRejectedWording:
    """The §5 report prose differentiates best vs shaped under reject."""

    def test_rejected_report_distinguishes_phases(self):
        shaped = np.ones((16, 16)) * 0.7
        bundle = _bundle(
            status="bake_off_rejected",
            best_phase=np.zeros((16, 16)),
            shaped_phase=shaped,
            probes=[_probe("shaped", shaped)],
        )
        text = gen.build_report(
            bundle,
            figure_names={},
            report_key="test",
            command="replay",
        )
        assert "`shaped_phase.npy`" in text
        assert "`best_phase.npy` 是**全零平场**" in text
        assert "两者不同" in text
        assert "真正合成并下发的整形相位" in text

    def test_report_without_shaped_has_no_shaped_mention(self):
        bundle = _bundle(
            status="bake_off_rejected",
            best_phase=np.zeros((16, 16)),
            shaped_phase=None,
            probes=[],
        )
        text = gen.build_report(
            bundle,
            figure_names={},
            report_key="test",
            command="replay",
        )
        assert "`shaped_phase.npy`" not in text

    def test_probe_fallback_reports_probe_source_not_file(self):
        """shaped_phase=None + shaped probe: the report must name the probe as
        the source and must NOT claim it lives in shaped_phase.npy."""
        probe_phase = np.ones((16, 16)) * 0.4
        bundle = _bundle(
            status="bake_off_rejected",
            best_phase=np.zeros((16, 16)),
            shaped_phase=None,
            probes=[_probe("shaped", probe_phase)],
        )
        text = gen.build_report(
            bundle,
            figure_names={},
            report_key="test",
            command="replay",
        )
        # Names the actual probe, not a fabricated root file.
        assert "来自历史目录的探针" in text
        assert "`round_01_shaped_00.npz`" in text
        assert "来自探针" in text
        # No claim that the shaped phase is a root file.
        assert "落在 `shaped_phase.npy`" not in text
        assert "`shaped_phase.npy` 才是 Step B" not in text

    def test_accepted_run_labels_best_same_as_shaped(self):
        shaped = np.ones((16, 16)) * 0.7
        bundle = _bundle(
            status="completed",
            best_phase=shaped.copy(),
            shaped_phase=shaped,
            probes=[_probe("shaped", shaped)],
        )
        text = gen.build_report(
            bundle,
            figure_names={},
            report_key="test",
            command="replay",
        )
        assert "内容一致" in text
        assert "`best_phase.npy` 就是真正下发" in text