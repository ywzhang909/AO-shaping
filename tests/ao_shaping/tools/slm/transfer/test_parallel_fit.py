"""Tests for the headless parallel-fit worker (no device imports required)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ao_shaping.tools.slm.transfer.parallel_fit import (
    EarlyStopper, LRScheduler, ParallelFitWorker,
)


# --------------------------------------------------------------------------- #
# EarlyStopper
# --------------------------------------------------------------------------- #
def test_early_stopper_trials_without_improvement():
    es = EarlyStopper(patience=3, min_delta=0.01)
    assert es.update(1.00) is False  # establishes best, wait=0
    assert es.update(0.99) is False  # no improvement, wait=1
    assert es.update(0.99) is False  # wait=2
    assert es.update(0.99) is True   # wait=3 -> stop_fit
    assert es.stop_fit and not es.good_enough


def test_early_stopper_good_enough_fires_immediately():
    es = EarlyStopper(good_enough_loss=0.5)
    assert es.update(0.4) is True
    assert es.good_enough and es.stop_fit is False


def test_early_stopper_good_enough_disabled_when_zero():
    es = EarlyStopper(good_enough_loss=0.0, patience=100)
    for _ in range(50):
        assert es.update(0.001) is False
    assert not es.good_enough


# --------------------------------------------------------------------------- #
# LRScheduler
# --------------------------------------------------------------------------- #
def _fake_optimizer(groups):
    return SimpleNamespace(param_groups=[dict(name=n, lr=l) for n, l in groups])


def test_lr_scheduler_warmup_ramps_to_one():
    sched = LRScheduler(warmup_polls=3, warmup_start=0.1)
    scales = [sched.step(0.5) for _ in range(3)]
    assert scales == pytest.approx([0.4, 0.7, 1.0], abs=1e-9)


def test_lr_scheduler_plateau_reduces_and_floors():
    sched = LRScheduler(warmup_polls=1, warmup_start=0.1, reduce_patience=2,
                        reduce_factor=0.5, min_scale=0.05)
    assert sched.step(0.5) == pytest.approx(1.0)        # warmup done
    sched.step(0.5)                                      # plateau wait=1
    sched.step(0.5)                                      # plateau wait=2 -> reduce
    assert sched.scale == pytest.approx(0.5)
    # Keep plateauing; scale should floor at min_scale, never below.
    for _ in range(20):
        sched.step(0.5)
    assert sched.scale >= 0.05


def test_lr_scheduler_improvement_resets_reduction():
    sched = LRScheduler(warmup_polls=1, reduce_patience=1, reduce_factor=0.5)
    sched.step(0.5)            # warmup -> 1.0
    sched.step(0.5)            # plateau wait=1 -> reduce to 0.5
    assert sched.scale == pytest.approx(0.5)
    sched.step(0.49)           # improved -> reset wait, scale unchanged
    assert sched.scale == pytest.approx(0.5)


def test_lr_scheduler_apply_multiplies_base_by_scale():
    sched = LRScheduler(warmup_polls=0, warmup_start=1.0)
    sched.step(0.5)
    sched.scale = 0.5
    opt = _fake_optimizer([("d", 0.01), ("center", 0.5)])
    sched.apply(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.005)
    assert opt.param_groups[1]["lr"] == pytest.approx(0.25)
    # A second apply must keep using the captured base LR, not compound.
    sched.scale = 0.25
    sched.apply(opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0025)


# --------------------------------------------------------------------------- #
# Worker integration (matched-model synthetic capture)
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_capture(tmp_path):
    pytest.importorskip("torch")
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset

    dataset = tmp_path / "synthetic"
    checkpoint = generate_dataset(dataset, power=1.0, count=6)
    return dataset, Path(checkpoint)


def test_worker_poll_writes_trace_and_provenance(synthetic_capture):
    dataset, checkpoint = synthetic_capture
    worker = ParallelFitWorker(
        dataset, checkpoint, span=120, out=dataset / "parallel_fit",
        device="cpu", poll_iterations=25, signal_stop=False)
    exit_info = worker.run(once=True)
    assert exit_info["reason"] == "once"
    trace = dataset / "parallel_fit" / "loss_trace.jsonl"
    prov = dataset / "parallel_fit" / "parallel_fit_provenance.json"
    assert trace.is_file() and prov.is_file()
    lines = trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] in {"poll", "early_stop"}
    assert float(row["best_loss"]) < 2.0
    assert "best_prediction.npz" in [p.name for p in (dataset / "parallel_fit").iterdir()]


def test_worker_signals_stop_when_good_enough(synthetic_capture):
    dataset, checkpoint = synthetic_capture
    # good_enough_loss far above any achievable loss => first poll triggers.
    worker = ParallelFitWorker(
        dataset, checkpoint, span=120, out=dataset / "parallel_fit",
        device="cpu", poll_iterations=25,
        early_stopper=EarlyStopper(good_enough_loss=10.0))
    row = worker.poll_once()
    assert row["status"] == "early_stop"
    assert (dataset / "stop_acquisition.request").is_file()


def test_worker_respects_existing_stop_request(synthetic_capture):
    dataset, checkpoint = synthetic_capture
    (dataset / "stop_acquisition.request").write_text("{}")
    worker = ParallelFitWorker(
        dataset, checkpoint, span=120, out=dataset / "parallel_fit",
        device="cpu", poll_iterations=10, signal_stop=False)
    exit_info = worker.run(once=True)
    assert exit_info["reason"] == "user_requested_stop"


def test_worker_finalizes_when_capture_done(synthetic_capture, tmp_path):
    dataset, checkpoint = synthetic_capture
    (dataset / ".capture_done").write_text(json.dumps(dict(early_stopped=False, probes=6)))
    out = tmp_path / "canonical"
    worker = ParallelFitWorker(
        dataset, checkpoint, span=120, out=dataset / "parallel_fit",
        device="cpu", poll_iterations=5, signal_stop=False)
    exit_info = worker.run(finalize=True, finalize_out=out, finalize_iterations=10, once=False)
    assert exit_info["reason"] == "capture_done"
    assert (out / "frozen_parameters.json").is_file()
    assert (out / "result.json").is_file()
    assert (out / "iterations.pkl").is_file()
