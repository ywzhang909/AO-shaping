"""Offline guard for the checkpoint-aware direction-gate wrapper (issue #237).

The self-check is deliberately small (one mode, two pass1/pass2 pairs): it proves
that the wrapper reuses ``phase_scope_direction_20261107.py`` statistics exactly
without requiring the full 14-pair archived gate in unit tests.  No hardware or
``ao_shaping.drivers`` module is touched.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "slm" / "registration_direction_gate_20261108.py"
PHASE_SCOPE = ROOT / "scripts" / "slm" / "phase_scope_direction_20261107.py"
RUN = ROOT / "data" / "debug" / "slm_transfer" / "issue234_single_mode_screen_20261107"
FIT = ROOT / "data" / "debug" / "slm_transfer" / "defocus_forward_20261107_focus_r2_fit" / "forward_fit.json"
CHECKPOINT = ROOT / "data" / "debug" / "slm_transfer" / "phase_scope_20261107" / "checkpoint_command" / "frozen_parameters.json"


def _load_gate() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.skip(f"wrapper under review is not present: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("registration_direction_gate_20261108", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary_from_stats(payload: dict, *, registration: tuple[float, float] | None) -> dict:
    stats = payload["stats"]
    return {
        "median": stats["median"],
        "mean": stats["mean"],
        "std": stats["std_ddof1"],
        "n_ge_0.5": stats["n_ge_0p5"],
        "n": stats["n"],
        "corrs": [float(row["corr"]) for row in payload["results"]],
        "registration_shift_ccd_px": None if registration is None else list(registration),
    }


def test_fit_mode_reuses_old_phase_scope_statistics_exactly(tmp_path: Path) -> None:
    old_out = tmp_path / "old.json"
    gate_out = tmp_path / "gate.json"
    subprocess.run(
        [
            sys.executable, str(PHASE_SCOPE), "--run", str(RUN), "--amp", "0.10",
            "--fit", str(FIT), "--modes", "5", "--origin", "model", "--out", str(old_out),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [
            sys.executable, str(SCRIPT), "--run", str(RUN), "--amp", "0.10",
            "--fit", str(FIT), "--modes", "5", "--origin", "model", "--out", str(gate_out),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    old = json.loads(old_out.read_text(encoding="utf-8"))
    gate_result = json.loads(gate_out.read_text(encoding="utf-8"))
    assert gate_result["regime"] == "fit"
    assert gate_result["registration_shift_ccd_px"] is None
    assert gate_result["groups"]["0.10"] == _summary_from_stats(old, registration=None)


def test_checkpoint_registration_is_reported_and_cli_override_wins(tmp_path: Path) -> None:
    gate = _load_gate()
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    checkpoint["registration_shift_ccd_px"] = [1.5, 2.5]
    checkpoint_path = tmp_path / "frozen_parameters.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    loaded = gate._load_checkpoint(checkpoint_path)

    from_checkpoint = gate.run_one_amp(
        amp=0.10, run=RUN, fit=FIT, model="M1_defocus", modes=[5],
        phase_sign=-1.0, phase_sign_scope="command", origin="model",
        checkpoint=loaded,
    )
    from_cli = gate.run_one_amp(
        amp=0.10, run=RUN, fit=FIT, model="M1_defocus", modes=[5],
        phase_sign=-1.0, phase_sign_scope="command", origin="model",
        checkpoint=loaded, registration_override=(3.5, 4.5),
    )
    assert from_checkpoint["registration_shift_ccd_px"] == [1.5, 2.5]
    assert from_cli["registration_shift_ccd_px"] == [3.5, 4.5]


def test_checkpoint_physics_actually_drives_predictions(tmp_path: Path) -> None:
    """checkpoint 的物理参数（defocus/origin 等）必须真的进入前向。

    若 ``replace`` 丢了 defocus/gain 等字段，checkpoint 模式会退化成 fit 模式，
    本测试通过对比两者的逐对 ``corrs`` 抓住这一点。同时核对非零配准会改变预测。
    """
    gate = _load_gate()
    loaded = gate._load_checkpoint(CHECKPOINT)
    kwargs = dict(
        amp=0.10, run=RUN, fit=FIT, model="M1_defocus", modes=[5],
        phase_sign=-1.0, phase_sign_scope="command", origin="model",
    )
    fit_mode = gate.run_one_amp(**kwargs)
    checkpoint_mode = gate.run_one_amp(**kwargs, checkpoint=loaded)
    fit_corrs = fit_mode["corrs"]
    checkpoint_corrs = checkpoint_mode["corrs"]
    assert len(fit_corrs) == len(checkpoint_corrs) and fit_corrs
    assert any(a != b for a, b in zip(fit_corrs, checkpoint_corrs)), (
        "checkpoint 模式与 fit 模式逐对 corr 完全一致：物理参数未进入前向"
    )

    shifted = gate.run_one_amp(**kwargs, registration_override=(1.5, 2.5))
    shifted_corrs = shifted["corrs"]
    assert any(a != b for a, b in zip(fit_corrs, shifted_corrs)), (
        "非零 registration_shift 未改变预测"
    )


def test_wrapper_does_not_import_drivers() -> None:
    _load_gate()
    assert not any(
        name == "ao_shaping.drivers" or name.startswith("ao_shaping.drivers.")
        for name in sys.modules
    )
