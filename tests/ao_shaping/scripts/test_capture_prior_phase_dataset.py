"""Offline checks for the prior-phase capture script.

No bench is touched here: the plan mode never opens a device, and the capture
loop is exercised with fake camera/SLM objects.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.slm.capture_defocus_forward_dataset import probe_grid as baseline_probe_grid
from scripts.slm.capture_prior_phase_dataset import (
    MANIFEST_KEYS,
    NPZ_KEYS,
    CaptureContext,
    build_steps,
    load_module_by_path,
    load_prior_candidates,
    load_probe_grid,
    main,
    resolve_step,
)

#: Panel geometry of the bench (the canonical constants, not a local copy).
PANEL_SHAPE = (1200, 1920)
PUPIL_CENTER = (960, 600)
PUPIL_RADIUS = 450


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate_entry(index: int, region: int, name: str) -> dict:
    return {
        "run_path": f"data/debug/slm_model_in_loop/2026101{index}_000000",
        "campaign": "slm_model_in_loop",
        "stamp": f"2026101{index}_000000",
        "round": index + 1,
        "region": region,
        "relative_improvement": 0.1 * (index + 1),
        "cv_after": 0.8 + 0.01 * index,
        "ee_after": 0.7 + 0.01 * index,
        "phase_file": f"candidates/{name}",
    }


def _write_focus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "focus": {"best_focal_mm": 1783.5418257111967, "best_diopter_m1": 0.5606821133007323},
        "settings": {
            "center_x": PUPIL_CENTER[0],
            "center_y": PUPIL_CENTER[1],
            "radius": PUPIL_RADIUS,
            "wavelength": 1064,
            "slm_number": 1,
            "cam_type": "daheng",
            "cam_id": 0,
        },
    }
    (directory / "focus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def _write_candidates(directory: Path) -> Path:
    """Two candidates on a 64-cell grid plus two on a 128-cell grid."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "candidates").mkdir(exist_ok=True)
    main_group, secondary = [], []
    for index, region in enumerate((64, 64)):
        name = f"main{index:02d}.npy"
        rng = np.random.default_rng(index)
        np.save(directory / "candidates" / name,
                rng.normal(0, 1.5, (region, region)).astype(np.float64))
        main_group.append(_candidate_entry(index, region, name))
    for index, region in enumerate((128, 128)):
        name = f"secondary{index:02d}.npy"
        rng = np.random.default_rng(100 + index)
        np.save(directory / "candidates" / name,
                rng.normal(0, 1.5, (region, region)).astype(np.float64))
        secondary.append(_candidate_entry(index, region, name))
    payload = {
        "candidates": main_group,
        "secondary_groups": [{"region": 128, "candidates": secondary}],
    }
    (directory / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    return _write_focus(tmp_path / "focus"), _write_candidates(tmp_path / "audit")


def _plan(workspace, capsys, monkeypatch, tmp_path, extra=()) -> dict:
    """Run the script in plan mode and return the plan it printed."""
    focus, candidates = workspace
    argv = [
        "capture_prior_phase_dataset.py",
        "--output", str(tmp_path / "out"),
        "--focus-directory", str(focus),
        "--candidates", str(candidates),
        "--count", "6",
        "--seed", "20261022",
        "--region", "128",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    printed = capsys.readouterr().out
    body = printed.split("=== plan (JSON) ===", 1)[1].split("=== capture list ===", 1)[0]
    return json.loads(body)


def _forbid_devices(monkeypatch) -> None:
    """Make every device entry point explode if the plan mode touches it."""
    def boom(*args, **kwargs):
        raise AssertionError("plan mode must not open a device")

    monkeypatch.setattr("ao_shaping.drivers.ccd.common.create_camera", boom)
    monkeypatch.setattr("ao_shaping.drivers.slm.santec.Santec", boom)
    monkeypatch.setattr("ao_shaping.tools.slm.panel.focus_probe.adapt_exposure", boom)
    monkeypatch.setattr("ao_shaping.tools.kernels.bench_kernels.capture_settled_report", boom)


# ---------------------------------------------------------------------------
# Plan mode
# ---------------------------------------------------------------------------


def test_plan_mode_never_opens_a_device(monkeypatch, capsys, workspace, tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path)
    assert plan["random_probe_count"] == 6
    assert not (tmp_path / "out").exists(), "plan mode must not create the output directory"


def test_plan_reports_probe_prior_counts_pairing_and_splits(monkeypatch, capsys, workspace,
                                                             tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path)
    assert plan["prior_count"] == 4
    assert plan["capture_count"] == len(plan["samples"])
    # Every probe and every prior phase is captured on both planes.
    paired = [s for s in plan["samples"] if s["source"]["kind"] in ("probe", "prior")]
    labels = {s["label"] for s in paired}
    assert len(labels) == 6 + 4
    for label in labels:
        branches = sorted(s["branch"] for s in paired if s["label"] == label)
        assert branches == ["C", "U"]
    # The order alternates so that the plane order cannot bias the pair.
    probe_labels = sorted({s["label"] for s in paired if s["source"]["kind"] == "probe"},
                          key=lambda text: int(text.split("_")[1]))
    for index, label in enumerate(probe_labels):
        rows = [s for s in paired if s["label"] == label]
        assert rows[0]["branch"] == ("U" if index % 2 == 0 else "C")
    # Split distribution: 2 flat/comp references, 8 ramps, 6 probes (3 train,
    # 3 test), 1 probe drift pair, 4 prior phases and 4 prior drift pairs.
    assert plan["split_counts"] == {"calibration": 20, "train": 14, "test": 6}
    # The ramps only exist on the compensated plane, so C outnumbers U.
    assert plan["branch_counts"] == {"U": 16, "C": 24}


def test_plan_lists_every_prior_source(monkeypatch, capsys, workspace, tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path)
    sources = plan["prior_sources"]
    assert len(sources) == 4
    for entry in sources:
        assert set(entry) >= {"label", "split", "source_region", "resampled_from_region",
                              "run_path", "stamp", "round", "relative_improvement",
                              "cv_after", "ee_after"}
        assert entry["run_path"].startswith("data/debug/slm_model_in_loop/")
        assert entry["relative_improvement"] > 0
    assert [entry["label"] for entry in sources] == [f"prior_{i:03d}" for i in range(4)]


def test_plan_mode_works_as_a_standalone_file(tmp_path, workspace):
    """The documented command form must run without a device."""
    focus, candidates = workspace
    script = Path(__file__).resolve().parents[3] / "scripts" / "slm" / "capture_prior_phase_dataset.py"
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(script),
         "--output", str(tmp_path / "out"),
         "--focus-directory", str(focus),
         "--candidates", str(candidates),
         "--count", "6", "--seed", "20261022", "--region", "128"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "out").exists()
    assert "=== capture list ===" in completed.stdout


# ---------------------------------------------------------------------------
# Prior phases and the resampling loss
# ---------------------------------------------------------------------------


def test_prior_phases_are_resampled_to_the_target_region(workspace):
    _, candidates = workspace
    priors = load_prior_candidates(candidates, 128, 0)
    assert len(priors) == 4
    resampled = [p for p in priors if p.resampled_from_region is not None]
    native = [p for p in priors if p.resampled_from_region is None]
    assert len(resampled) == 2 and len(native) == 2
    for prior in resampled:
        assert prior.source_region == 64
        assert prior.grid.shape == (128, 128)
        assert np.isfinite(prior.grid).all()
        # A phase, not a normalised image: radians, so several multiples of pi.
        assert np.abs(prior.grid).max() > 1.0
        assert np.abs(prior.grid).max() < 100.0
    for prior in native:
        assert prior.source_region == 128
        assert prior.resampled_from_region is None
        assert prior.grid.shape == (128, 128)


def test_the_resampled_grid_is_the_linear_upsample_of_the_source(workspace):
    _, candidates = workspace
    priors = load_prior_candidates(candidates, 128, 0)
    original = np.load(candidates / "candidates" / "main00.npy")
    prior = priors[0]
    assert prior.resampled_from_region == 64
    from scipy.ndimage import zoom

    np.testing.assert_allclose(prior.grid, zoom(original, 2.0, order=1), atol=1e-12)


def test_the_plan_annotates_the_resampled_prior_phases(monkeypatch, capsys, workspace,
                                                        tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path)
    annotated = {entry["label"]: entry for entry in plan["prior_sources"]}
    assert annotated["prior_000"]["resampled_from_region"] == 64
    assert annotated["prior_000"]["source_region"] == 64
    assert annotated["prior_002"]["resampled_from_region"] is None
    assert annotated["prior_002"]["source_region"] == 128


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_probe_split_rule_matches_the_baseline_group():
    steps = build_steps(12, [])
    probes = [s for s in steps if s["source"]["kind"] == "probe"]
    labels = sorted({s["label"] for s in probes}, key=lambda text: int(text.split("_")[1]))
    assert len(labels) == 12
    for index, label in enumerate(labels):
        rows = [s for s in probes if s["label"] == label]
        assert {r["split"] for r in rows} == {"test" if index % 6 >= 3 else "train"}
        family, _ = baseline_probe_grid(index, 128, 20261022)
        assert {r["family"] for r in rows} == {family}


def test_resolved_probe_phase_is_bit_identical_to_the_baseline_group():
    from ao_shaping.utils.slm.panel_geometry import phase_to_panel

    ctx = CaptureContext(
        shape=PANEL_SHAPE, corrected=np.zeros(PANEL_SHAPE), radius=PUPIL_RADIUS,
        center=PUPIL_CENTER, region=128, seed=20261022, probe_grid=baseline_probe_grid,
        priors=[],
    )
    steps = build_steps(6, [])
    checked = 0
    for step in steps:
        if step["source"]["kind"] != "probe" or step["branch"] != "U":
            continue
        index = step["source"]["index"]
        family, model = baseline_probe_grid(index, 128, 20261022)
        panel, resolved = resolve_step(step, ctx)
        np.testing.assert_array_equal(panel, phase_to_panel(model, PUPIL_RADIUS, PUPIL_CENTER))
        np.testing.assert_array_equal(resolved, model)
        checked += 1
    assert checked == 6


def test_prior_phases_default_to_the_training_split(monkeypatch, capsys, workspace,
                                                      tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path)
    assert {entry["split"] for entry in plan["prior_sources"]} == {"train"}
    prior_rows = [s for s in plan["samples"] if s["family"] == "prior"]
    assert len(prior_rows) == 8
    assert {s["split"] for s in prior_rows} == {"train"}


def test_prior_holdout_moves_exactly_n_phases_to_the_validation_split(monkeypatch, capsys,
                                                                      workspace, tmp_path):
    _forbid_devices(monkeypatch)
    plan = _plan(workspace, capsys, monkeypatch, tmp_path, extra=["--prior-holdout", "2"])
    held = [entry for entry in plan["prior_sources"] if entry["split"] == "test"]
    assert len(held) == 2
    assert [entry["label"] for entry in held] == ["prior_002", "prior_003"]
    rows = [s for s in plan["samples"] if s["family"] == "prior" and s["split"] == "test"]
    assert len({s["label"] for s in rows}) == 2
    assert len(rows) == 4, "a held-out prior phase is captured on both planes"


# ---------------------------------------------------------------------------
# Capture loop with fake devices
# ---------------------------------------------------------------------------


class _FakeCamera:
    """A camera that returns a synthetic spot; it never touches hardware."""

    def __init__(self, *args, **kwargs):
        self.exposure_time = 1.0
        self.frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reset_exposure_time(self, milliseconds: float) -> float:
        self.exposure_time = float(milliseconds)
        return self.exposure_time

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        self.frames += 1
        yy, xx = np.mgrid[0:64, 0:64]
        return 100.0 * np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / 32.0)


class _FakeSlm:
    """An SLM that only records what it was asked to display."""

    def __init__(self, *args, **kwargs):
        self.writes: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def create_phase_from_array(self, phase):
        return np.asarray(phase)

    def display_data(self, gray, wait_time_s: float = 0.5) -> None:
        self.writes.append(float(wait_time_s))


def _fake_adapt_exposure(set_exposure_ms, read_frame, *, start_ms, floor_ms, ceiling_ms,
                         full_scale, **kwargs):
    set_exposure_ms(start_ms)
    read_frame()
    from ao_shaping.tools.slm.panel.focus_probe import ExposureSetting

    return ExposureSetting(float(start_ms), 100.0, 100.0 / full_scale, "ok", 1)


def _fake_capture_settled_report(cam, slm, phase, **kwargs):
    from ao_shaping.tools.kernels.bench_kernels import SettleReport

    frame = cam.get_numpy_image(n_sample=1)
    settle = SettleReport(True, "stable", 3, 3, 0.5, 100.0, 4.0, 32.0, 32.0)
    return SimpleNamespace(frame=frame, settle=settle)


def _install_fake_devices(monkeypatch) -> None:
    monkeypatch.setattr("ao_shaping.drivers.ccd.common.create_camera",
                        lambda *args, **kwargs: _FakeCamera())
    monkeypatch.setattr("ao_shaping.drivers.slm.santec.Santec",
                        lambda *args, **kwargs: _FakeSlm())
    monkeypatch.setattr("ao_shaping.tools.slm.panel.focus_probe.adapt_exposure",
                        _fake_adapt_exposure)
    monkeypatch.setattr("ao_shaping.tools.kernels.bench_kernels.capture_settled_report",
                        _fake_capture_settled_report)


def test_capture_loop_writes_the_format_contract_of_the_baseline_group(monkeypatch, capsys,
                                                                        workspace, tmp_path):
    _install_fake_devices(monkeypatch)
    focus, candidates = workspace
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "capture_prior_phase_dataset.py",
        "--output", str(out),
        "--focus-directory", str(focus),
        "--candidates", str(candidates),
        "--count", "6", "--seed", "20261022", "--region", "128", "--execute",
    ])
    assert main() == 0

    assert (out / "run_config.json").is_file()
    assert (out / "focus_phase.npy").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "capture_records.pkl").is_file()

    rows = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert len(rows) == plan["capture_count"] == 40
    for row in rows:
        assert set(MANIFEST_KEYS) <= set(row), sorted(set(MANIFEST_KEYS) - set(row))
        assert (out / row["file"]).is_file()
        with np.load(out / row["file"]) as data:
            assert set(data.files) == set(NPZ_KEYS)
            assert data["raw_frames"].shape[0] == 3
            assert data["panel_phase"].shape == PANEL_SHAPE
            assert data["panel_phase"].dtype == np.float32
            if row["family"] in ("zernike", "smooth", "bandlimited", "prior"):
                assert data["model_phase"].shape == (128, 128)
            else:
                assert data["model_phase"].shape == (0, 0)

    prior_rows = [row for row in rows if row["family"] == "prior"]
    assert len(prior_rows) == 8
    for row in prior_rows:
        assert set(row["prior_source"]) == {"run_path", "stamp", "round",
                                             "relative_improvement", "cv_after", "ee_after"}
        assert row["resampled_from_region"] in (64, None)

    import pickle

    with (out / "capture_records.pkl").open("rb") as stream:
        assert pickle.load(stream) == rows


def test_capture_loop_restores_a_flat_panel_and_reports_a_summary(monkeypatch, capsys,
                                                                   workspace, tmp_path):
    _install_fake_devices(monkeypatch)
    focus, candidates = workspace
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "capture_prior_phase_dataset.py",
        "--output", str(out),
        "--focus-directory", str(focus),
        "--candidates", str(candidates),
        "--count", "6", "--seed", "20261022", "--region", "128", "--execute",
    ])
    assert main() == 0
    printed = capsys.readouterr().out
    assert "=== capture summary ===" in printed
    assert "branch U:" in printed and "branch C:" in printed
    assert "prior_000" in printed


def test_an_unusable_exposure_aborts_the_run(monkeypatch, capsys, workspace, tmp_path):
    """A clipped exposure must never be mixed into the dataset."""
    _install_fake_devices(monkeypatch)

    def clipped(set_exposure_ms, read_frame, *, start_ms, floor_ms, ceiling_ms, full_scale,
                **kwargs):
        from ao_shaping.tools.slm.panel.focus_probe import ExposureSetting

        set_exposure_ms(start_ms)
        return ExposureSetting(float(start_ms), 255.0, 1.0, "clipped", 1)

    monkeypatch.setattr("ao_shaping.tools.slm.panel.focus_probe.adapt_exposure", clipped)
    focus, candidates = workspace
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "capture_prior_phase_dataset.py",
        "--output", str(out),
        "--focus-directory", str(focus),
        "--candidates", str(candidates),
        "--count", "6", "--seed", "20261022", "--region", "128", "--execute",
    ])
    with pytest.raises(RuntimeError, match="unusable exposure"):
        main()
    rows = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert rows[-1]["exposure_status"] == "clipped"


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def test_load_module_by_path_registers_the_module_before_executing_it(tmp_path):
    """A module-level dataclass needs its module in ``sys.modules``.

    ``dataclasses`` resolves annotations such as ``ClassVar`` against
    ``sys.modules[cls.__module__].__dict__``; when the module was never
    registered, that lookup yields ``None`` and the import dies with
    "NoneType object has no attribute '__dict__'".
    """
    path = tmp_path / "annotated_module.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "\n"
        "@dataclass\n"
        "class Record:\n"
        "    value: int\n"
        "    tag: ClassVar[str] = 'x'\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("annotated_module", path)
    assert spec is not None and spec.loader is not None
    unregistered = importlib.util.module_from_spec(spec)
    with pytest.raises(AttributeError, match="__dict__"):
        spec.loader.exec_module(unregistered)

    module = load_module_by_path("annotated_module", path)
    assert sys.modules["annotated_module"] is module
    assert module.Record(3).value == 3


def test_the_reused_probe_grid_reproduces_the_baseline_group_bit_for_bit():
    """The random probes must be the baseline's, not a look-alike.

    The baseline script is loaded by path, so it is a *second* module object;
    what makes the pairing valid is that it computes the same phases, not that
    it is the same function object.
    """
    reused = load_probe_grid()
    for index in range(6):
        family, phase = reused(index, 128, 20261022)
        expected_family, expected_phase = baseline_probe_grid(index, 128, 20261022)
        assert family == expected_family
        np.testing.assert_array_equal(phase, expected_phase)
