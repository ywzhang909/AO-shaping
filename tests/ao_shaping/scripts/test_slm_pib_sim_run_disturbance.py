"""The ``slm_pib_sim_run`` disturbance CLI and its companion artifacts.

The simulation report generator is a pure offline reader of the run's saved
artifacts, so it can only describe the wavefront disturbance if the harness
records one. These tests pin that contract: the CLI surface, the run-directory
discovery (which must not be fooled by a concurrently-written directory), and
the ``disturbance.json`` / ``disturbance.npz`` pair the report consumes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.drivers.sim.disturbance import DisturbanceConfig, SimDisturbance  # noqa: E402

_MODULE_NAME = "slm_pib_sim_run_under_test"
_runner = None


def _load_runner():
    """Import the harness script by path (it lives outside the package)."""
    global _runner
    if _runner is None:
        spec = importlib.util.spec_from_file_location(
            _MODULE_NAME, ROOT / "scripts" / "slm_pib_sim_run.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _runner = module
    return _runner


# --- CLI surface -------------------------------------------------------------


def test_cli_defaults_match_the_frozen_contract() -> None:
    args = _load_runner().parse_args([])
    assert args.disturbance == "none"
    assert args.cn2 == 2e-13
    assert args.distance_m == 500.0
    assert args.l_max == 30.0
    assert args.l_min == 2e-3
    assert args.pixel_pitch_um == 8.0
    assert args.halo_pv_waves == 0.30
    assert args.halo_radius_px == 600.0
    assert args.dist_seed == 20261001
    assert args.dist_archive_factor == 8
    assert args.dist_archive_max == 12
    assert args.dist_tag is None
    assert args.epochs == 300
    assert args.objective == "shape"
    assert args.target_shape == "square"
    assert args.algorithm is None
    assert Path(args.data_root) == Path("data")


def test_disturbance_mode_choices_are_exactly_the_three() -> None:
    mod = _load_runner()
    for mode in ("none", "static", "dynamic"):
        assert mod.parse_args(["--disturbance", mode]).disturbance == mode
    with pytest.raises(SystemExit):
        mod.parse_args(["--disturbance", "bogus"])


def test_parse_args_builds_the_expected_disturbance_config() -> None:
    mod = _load_runner()
    args = mod.parse_args(
        [
            "--disturbance", "dynamic",
            "--cn2", "1e-13",
            "--distance-m", "750",
            "--l-max", "25",
            "--l-min", "1e-3",
            "--pixel-pitch-um", "12.5",
            "--halo-pv-waves", "0.6",
            "--halo-radius-px", "500",
            "--dist-seed", "7",
        ]
    )
    cfg = mod.build_disturbance_config(args)
    assert cfg.mode == "dynamic"
    assert cfg.cn2 == 1e-13
    assert cfg.distance_m == 750.0
    assert cfg.l_max == 25.0
    assert cfg.l_min == 1e-3
    assert cfg.pixel_pitch_m == pytest.approx(12.5e-6)
    assert cfg.thermal_halo_pv_waves == 0.6
    assert cfg.thermal_halo_radius_px == 500.0
    assert cfg.seed == 7


def test_tag_is_explicit_or_derived_from_the_mode() -> None:
    mod = _load_runner()
    assert mod.resolve_tag(mod.parse_args(["--disturbance", "static"])) == "static"
    assert mod.resolve_tag(mod.parse_args(["--disturbance", "dynamic"])) == "dynamic"
    assert (
        mod.resolve_tag(mod.parse_args(["--disturbance", "dynamic", "--dist-tag", "turb-x"]))
        == "turb-x"
    )


# --- run-directory discovery -------------------------------------------------


def test_artifact_dir_discovery_finds_the_newly_created_dir(tmp_path: Path) -> None:
    mod = _load_runner()
    debug = tmp_path / "debug"
    debug.mkdir(parents=True)
    (debug / "slm_pib_shape_20200101_000000").mkdir()

    before = mod._artifact_roots(tmp_path)
    fresh = debug / "slm_pib_shape_20200101_000001"
    fresh.mkdir()

    assert mod.find_new_artifact_dir(tmp_path, before) == fresh.resolve()


def test_artifact_dir_discovery_ignores_unrelated_pre_existing_dirs(tmp_path: Path) -> None:
    mod = _load_runner()
    debug = tmp_path / "debug"
    debug.mkdir(parents=True)
    (debug / "slm_pib_shape_old").mkdir()
    before = mod._artifact_roots(tmp_path)
    # A non-matching dir appearing must not be selected.
    (debug / "something_else").mkdir()
    # No new match: falls back to an existing match rather than returning junk.
    found = mod.find_new_artifact_dir(tmp_path, before)
    assert found == (debug / "slm_pib_shape_old").resolve()


def test_artifact_dir_discovery_returns_none_when_empty(tmp_path: Path) -> None:
    mod = _load_runner()
    assert mod.find_new_artifact_dir(tmp_path, set()) is None


def test_run_dir_of_picks_the_subdir_holding_the_pickle(tmp_path: Path) -> None:
    mod = _load_runner()
    root = tmp_path / "slm_pib_shape_x"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "b" / "run.pkl").write_bytes(b"")

    assert mod.run_dir_of(root) == root / "b"


# --- companion artifacts -----------------------------------------------------


def test_companion_artifacts_are_written(tmp_path: Path) -> None:
    mod = _load_runner()
    dist = SimDisturbance(DisturbanceConfig(mode="static", cn2=2e-13), (240, 384))
    for _ in range(3):
        dist.phase()

    run_dir = tmp_path / "run"
    json_path, npz_path = mod.write_companion(
        dist,
        run_dir,
        run_meta={"tag": "static", "epochs": 3},
        archive_factor=8,
        archive_max=12,
    )

    assert json_path.exists() and npz_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["enabled"] is True
    assert payload["mode"] == "static"
    assert payload["run"]["tag"] == "static"
    assert payload["run"]["epochs"] == 3
    assert payload["measured"]["streaks_used"] == 1
    assert payload["run"]["streaks_total"] == 1
    assert payload["run"]["screens_archived"] == 1

    with np.load(npz_path) as archive:
        assert archive["screens"].dtype == np.float32
        assert archive["screens"].shape[0] == 1
        assert archive["call_rms"].shape == (3,)
        assert archive["call_streak_index"].dtype == np.int32
        assert int(archive["archive_factor"]) == 8
        # Downsampled, never the full-resolution screen.
        assert archive["screens"].shape[1:] == (240 // 8, 384 // 8)
        assert archive["screens"].shape[1] < 240
        assert archive["screens"].shape[2] < 384


def test_companion_archive_is_capped_but_trace_is_complete(tmp_path: Path) -> None:
    """Dynamic runs draw one screen per evaluation; only a sample is archived."""
    mod = _load_runner()
    dist = SimDisturbance(DisturbanceConfig(mode="dynamic", cn2=2e-13), (120, 192))
    calls = 5
    for _ in range(calls):
        dist.phase()

    json_path, npz_path = mod.write_companion(
        dist, tmp_path / "run", run_meta={"tag": "dynamic"}, archive_factor=4, archive_max=3
    )
    payload = json.loads(json_path.read_text())
    assert payload["run"]["streaks_total"] == calls
    assert payload["run"]["screens_archived"] == 3

    with np.load(npz_path) as archive:
        assert archive["screens"].shape[0] == 3
        # Every evaluation is still recorded, even the ones not archived.
        assert archive["call_rms"].shape == (calls,)
        assert archive["streak_indices"].shape == (3,)


def test_none_mode_writes_enabled_false(tmp_path: Path) -> None:
    mod = _load_runner()
    dist = SimDisturbance(DisturbanceConfig(mode="none"), (120, 192))
    json_path, npz_path = mod.write_companion(
        dist, tmp_path / "run", run_meta={"tag": "none"}
    )
    payload = json.loads(json_path.read_text())
    assert payload["enabled"] is False
    assert payload["mode"] == "none"
    assert payload["measured"]["streaks_used"] == 0
    with np.load(npz_path) as archive:
        assert archive["screens"].shape[0] == 0
        assert archive["call_rms"].shape == (0,)


def test_companion_writing_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    mod = _load_runner()

    def write(where: Path) -> tuple[bytes, bytes]:
        dist = SimDisturbance(DisturbanceConfig(mode="static", cn2=2e-13, seed=5), (120, 192))
        dist.phase()
        json_path, npz_path = mod.write_companion(
            dist, where, run_meta={"tag": "static"}, archive_factor=4, archive_max=12
        )
        return json_path.read_bytes(), npz_path.read_bytes()

    first_json, first_npz = write(tmp_path / "a")
    second_json, second_npz = write(tmp_path / "b")
    assert first_json == second_json
    assert first_npz == second_npz


def test_runner_reset_system_does_not_discard_the_disturbance() -> None:
    """Regression: ``slm_pib_runner`` resets the system while sim-patching.

    ``_maybe_sim_patch`` imports ``reset_system`` locally and calls it with a
    seed only. Without re-attaching the disturbance, the run silently executes
    disturbance-free while the companion manifest claims otherwise.
    """
    mod = _load_runner()
    import ao_shaping.drivers.sim.slm_pib_sim as sim_module

    original = sim_module.reset_system
    previous_injected = mod._INJECTED_DISTURBANCE
    try:
        # Match the default panel: reset_system builds a full-size SimPibSystem.
        dist = SimDisturbance(DisturbanceConfig(mode="static"), (1200, 1920))
        mod._install_disturbance_reset(dist)

        # Exactly what the runner does: a function-local import, then a bare call.
        from ao_shaping.drivers.sim.slm_pib_sim import reset_system as local_reset

        system = local_reset(seed=42)
        assert system.disturbance is dist
        system.far_field()
        assert dist.stats()["calls"] == 1.0
    finally:
        sim_module.reset_system = original
        mod._INJECTED_DISTURBANCE = previous_injected
        original(seed=42)


def test_install_disturbance_reset_is_idempotent() -> None:
    mod = _load_runner()
    import ao_shaping.drivers.sim.slm_pib_sim as sim_module

    original = sim_module.reset_system
    previous_injected = mod._INJECTED_DISTURBANCE
    try:
        dist = SimDisturbance(DisturbanceConfig(mode="none"), (120, 192))
        mod._install_disturbance_reset(dist)
        once = sim_module.reset_system
        mod._install_disturbance_reset(dist)
        assert sim_module.reset_system is once, "must not stack wrappers"
    finally:
        sim_module.reset_system = original
        mod._INJECTED_DISTURBANCE = previous_injected
        original(seed=42)


def test_manifest_is_json_round_trippable(tmp_path: Path) -> None:
    mod = _load_runner()
    dist = SimDisturbance(DisturbanceConfig(mode="static"), (120, 192))
    json_path, _ = mod.write_companion(dist, tmp_path / "run", run_meta={"tag": "static"})
    payload = json.loads(json_path.read_text())
    assert payload["config"]["mode"] == "static"
    assert payload["config"]["halo_noll"] == [[4, -1.0], [11, -1.0]]
    assert json.dumps(payload)  # must survive a second round trip
