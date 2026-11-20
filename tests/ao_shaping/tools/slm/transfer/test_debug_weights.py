"""Offline tests for the opt-in ``--debug-weights`` trace in the transfer fit stage.

The trace is purely additive: three files under ``<fit_out>/debug_weights`` and the
canonical ``iterations.pkl`` untouched. These tests cover the two pure helpers plus
one bounded end-to-end run of :func:`fit.fit` with the model and loss mocked, so the
loop wiring (history JSONL + summary) is exercised without a dataset or a device.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import fit as fit_module


def _params(iteration: float = 0.0) -> dict:
    """One iteration's parameter dict, shaped exactly like ``fit_loss`` emits it."""
    return {
        "defocus_diopter": 0.10 + 0.01 * iteration,
        "origin": [960.0, 600.0 + 0.01 * iteration],
        "amplitude_center": [960.0, 600.0],
        "amplitude_width": [400.0, 300.0],
        "residual_noll_radians": {"5": 0.001 * iteration, "6": 0.0},
        "gain": 1.0,
        "background": 0.5,
        "sigma_ccd_px": 2.0,
        "bias_counts": 3.0,
    }


def test_flatten_parameters_expands_lists_and_dicts() -> None:
    flat = fit_module._flatten_parameters(_params(iteration=2.0))
    assert flat["defocus_diopter"] == pytest.approx(0.12)
    assert flat["origin_0"] == pytest.approx(960.0)
    assert flat["origin_1"] == pytest.approx(600.02)
    assert flat["amplitude_width_1"] == pytest.approx(300.0)
    assert flat["residual_noll_5"] == pytest.approx(0.002)
    assert flat["gain"] == pytest.approx(1.0)
    assert flat["background"] == pytest.approx(0.5)
    assert flat["sigma_ccd_px"] == pytest.approx(2.0)
    # Non-trainable radiometry extras are not promoted to the trace.
    assert "bias_counts" not in flat
    # Every value is a plain float, ready for json.dumps.
    assert all(isinstance(value, float) for value in flat.values())


def test_write_weight_trajectory_reports_first_last_delta(tmp_path) -> None:
    trajectory = [
        fit_module._flatten_parameters(_params(iteration=0.0)),
        fit_module._flatten_parameters(_params(iteration=1.0)),
        fit_module._flatten_parameters(_params(iteration=3.0)),
    ]
    fit_module._write_weight_trajectory(tmp_path, trajectory)

    npz = np.load(tmp_path / "weights_trajectory.npz")
    assert list(npz["iterations"]) == [0, 1, 2]
    assert npz["defocus_diopter"].tolist() == pytest.approx([0.10, 0.11, 0.13])

    summary = json.loads((tmp_path / "weights_summary.json").read_text(encoding="utf-8"))
    row = summary["defocus_diopter"]
    assert row["first"] == pytest.approx(0.10)
    assert row["last"] == pytest.approx(0.13)
    assert row["delta"] == pytest.approx(0.03)
    assert row["max_abs_step"] == pytest.approx(0.02)


def test_write_weight_trajectory_ignores_empty(tmp_path) -> None:
    fit_module._write_weight_trajectory(tmp_path, [])
    assert list(tmp_path.iterdir()) == []


def test_run_parser_debug_weights_reaches_the_session() -> None:
    """``run --debug-weights`` must survive the parser into the session config."""
    from ao_shaping.tools.slm.transfer import workflow

    parser = workflow._wf_build_parser()
    argv = [
        "run", "--out", "out", "--dataset", "d", "--checkpoint", "c",
        "--debug-weights",
    ]
    args = parser.parse_args(argv)
    assert args.debug_weights is True
    config = workflow.TransferSession(
        **{k: v for k, v in vars(args).items() if k not in {"command", "offline", "execute"}}
    )
    assert config.debug_weights is True

    default = parser.parse_args(["run", "--out", "out", "--dataset", "d", "--checkpoint", "c"])
    assert default.debug_weights is False
    assert workflow.TransferSession(
        **{k: v for k, v in vars(default).items() if k not in {"command", "offline", "execute"}}
    ).debug_weights is False


def _fake_comps() -> SimpleNamespace:
    import torch

    model = SimpleNamespace(
        _device=SimpleNamespace(type="cpu"),
        optimizer=SimpleNamespace(zero_grad=lambda: None, step=lambda: None),
        d=torch.zeros(1),
        center=torch.tensor([960.0, 600.0]),
        log_width=torch.tensor([np.log(400.0), np.log(300.0)]),
        registration_value=0.0,
    )
    old = SimpleNamespace(
        grid_size=64, fft_size=256, pixel_samples=2, origin_xy=[960.0, 600.0]
    )
    return SimpleNamespace(
        model=model,
        matrix=np.zeros((2, 2)),
        old=old,
        measured_origin=np.zeros(2),
        phases=np.zeros((1, 2)),
        images=np.zeros((1, 2, 2)),
        xy=np.zeros((1, 2)),
        train=np.array([True]),
        ramp_samples={},
        rows=[],
    )


def _run_fit(tmp_path, monkeypatch, *, debug_weights: bool):
    torch = pytest.importorskip("torch")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    out = tmp_path / "fit_out"

    comps = _fake_comps()

    def fake_build(*_args, **_kwargs):
        return comps

    def fake_fit_loss(_model, _comps, _radiometry_mode, iteration):
        entry = {
            "iteration": iteration,
            "loss": 10.0 - iteration,
            "parameters": _params(iteration=float(iteration)),
        }
        loss = torch.tensor(10.0 - iteration, requires_grad=True)
        return loss, entry, np.zeros((2, 2))

    monkeypatch.setattr(fit_module, "build_fit_model", fake_build)
    monkeypatch.setattr(fit_module, "fit_loss", fake_fit_loss)

    args = SimpleNamespace(
        dataset=dataset,
        checkpoint=checkpoint,
        out=out,
        iterations=2,
        patience=8,
        min_delta=1e-4,
        adaptation="full",
        debug_weights=debug_weights,
        span=10,
        device="cpu",
        amplitude_center_lr=None,
        radiometry="exposure",
        samples=None,
        train_all=False,
    )
    fit_module.fit(args)
    return out


def test_fit_writes_debug_weight_artifacts(tmp_path, monkeypatch) -> None:
    out = _run_fit(tmp_path, monkeypatch, debug_weights=True)
    debug_dir = out / "debug_weights"
    assert (debug_dir / "weights_history.jsonl").is_file()
    assert (debug_dir / "weights_trajectory.npz").is_file()
    assert (debug_dir / "weights_summary.json").is_file()

    records = [
        json.loads(line)
        for line in (debug_dir / "weights_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["iteration"] for record in records] == [0, 1, 2]
    assert "delta" not in records[0]
    assert records[1]["delta"]["defocus_diopter"] == pytest.approx(0.01)

    summary = json.loads((debug_dir / "weights_summary.json").read_text(encoding="utf-8"))
    assert summary["defocus_diopter"]["last"] == pytest.approx(0.12)


def test_fit_without_debug_weights_writes_no_debug_dir(tmp_path, monkeypatch) -> None:
    out = _run_fit(tmp_path, monkeypatch, debug_weights=False)
    assert not (out / "debug_weights").exists()
    # The canonical products still land.
    assert (out / "iterations.pkl").is_file()
    assert (out / "frozen_parameters.json").is_file()
