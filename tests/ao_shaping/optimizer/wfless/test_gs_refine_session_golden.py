"""Golden: the numeric trace of a ``cam_type="sim"`` GS-refine session.

Issue #167 splits ``_run_gs_refine_session``; the split must not move a
number. So the numbers are pinned: every recorded row's metrics (stage, score,
pib, cv, shape_iou, per-epoch probe values) for a fixed seed.

The camera is a stub that returns a deterministic frame derived from the
displayed phase, so the run is fully offline and reproducible. ``timestamp_utc``
is deliberately excluded -- it is wall-clock time and would make the golden
machine-dependent.

The golden is regenerated with::

    AO_GS_REFINE_GOLDEN_UPDATE=1 pytest tests/ao_shaping/optimizer/wfless/test_gs_refine_session_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_gs_refine as gsr
from ao_shaping.optimizer.wfless.slm_gs_refine import SlmGsRefineConfig
from ao_shaping.utils.io.file import Recorder

GOLDEN = Path(__file__).with_name("gs_refine_session_golden.json")
UPDATE = os.environ.get("AO_GS_REFINE_GOLDEN_UPDATE") == "1"

#: Recorder row keys that are deterministic. ``timestamp_utc`` is wall-clock
#: time and is left out on purpose.
PINNED_KEYS = (
    "stage",
    "score",
    "pib",
    "cv",
    "shape_iou",
    "target_side",
    "_epoch",
    "epoch",
    "lr",
    "pos",
    "neg",
    "probe_score",
    "confirmed",
    "dof_score",
)


class _StubSLM:
    """Just enough SLM for the session: a panel size and a phase encoder."""

    def __init__(self) -> None:
        self.Panel_Res = (64, 48)

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        return np.asarray(phase_rad, dtype=np.float64)


def _stub_display_and_average(cam: Any, slm: Any, phase_rad: np.ndarray, **kwargs: Any):
    """A deterministic camera whose spot *moves* with the phase.

    The spot is displaced by the phase at the panel centre -- inside the pupil
    square, where the optimiser's DOF is embedded. Moving it (rather than
    scaling it) is what makes the measurement metrics respond: an
    amplitude-only change leaves ``pib`` and ``cv`` inside the target square
    invariant, which would pin a trace that cannot tell a real numeric drift
    from a no-op.
    """
    phase = np.asarray(phase_rad, dtype=np.float64)
    height, width = phase.shape
    offset = 8.0 * float(phase[height // 2, width // 2])
    centre_y = height // 2 + offset
    centre_x = width // 2 + offset
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(-(((yy - centre_y) ** 2 + (xx - centre_x) ** 2) / 18.0))
    return 100.0 * blob


def _config(**overrides: Any) -> SlmGsRefineConfig:
    base: dict[str, Any] = dict(
        cam_type="sim",
        epochs=3,
        phase_grid=4,
        delta=0.4,
        lr=0.05,
        seed=7,
        beam_radius_px=12.0,
        cam_size=0,
        target_side=12,
        gs_warm_start=False,
        n_eval_frames=1,
        settle_wait_s=0.001,
        settle_max_wait_s=0.001,
        settle_max_discard=1,
        progress_every=1,
        focal_length_m=0.1,
        camera_pixel_um=8.0,
        panel_pixel_um=8.0,
    )
    base.update(overrides)
    return SlmGsRefineConfig(**base)


def _run(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[dict[str, Any]]:
    monkeypatch.setattr(gsr, "display_and_average", _stub_display_and_average)
    # Do NOT stub sync_device_wavelength: _bench_probe reads the post-sync
    # slm_wavelength (2aa39d86); the real sim fallback pins it to
    # DEFAULT_WAVELENGTH_NM deterministically, a no-op stub leaves 0.
    recorder = Recorder(mark="score", mode="max")
    gsr._run_gs_refine_session(
        _config(**overrides),
        _StubSLM(),
        object(),
        np.random.default_rng(7),
        recorder,
    )
    rows = []
    for row in recorder.history:
        rows.append({k: row[k] for k in PINNED_KEYS if k in row})
    return rows


def test_session_trace_matches_the_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    scenarios = {
        "plain": _run(monkeypatch),
        "confirm_best": _run(monkeypatch, confirm_best=True),
    }

    if UPDATE:
        GOLDEN.write_text(
            json.dumps(scenarios, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = scenarios
    else:
        assert GOLDEN.exists(), (
            f"missing golden {GOLDEN.name}; regenerate with "
            "AO_GS_REFINE_GOLDEN_UPDATE=1"
        )
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    assert scenarios == expected


def test_the_trace_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    """A golden that only reproduces once pins nothing."""
    assert _run(monkeypatch) == _run(monkeypatch)
