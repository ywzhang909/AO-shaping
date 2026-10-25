"""Tests for :mod:`ml.sim2f.evaluate` -- the flat / net / teacher table.

A checkpoint is produced by really training on the toy cache from
``test_train.py`` (six samples, a stubbed bench), then scored.  Nothing depends
on ``ml.sim2f.optics``: the bench enters through the same ``_build_sim`` seam the
production code uses, patched in both modules that call it.

The stub bench carries a constant pedestal, so these tests exercise the
*background-subtracted* path the calibrated twin needs rather than an ideal
energy-conserving frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from pathlib import Path

from ml.sim2f.evaluate import EVAL_NAME, evaluate, format_table
from ml.sim2f.train import CHECKPOINT_LAST, TrainConfig, train
from tests.ao_shaping.ml.sim2f.test_train import (
    SAMPLES,
    StubBench,
    build_toy_cache,
    _tiny_config,
)


@pytest.fixture()
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, TrainConfig]:
    """Train a tiny checkpoint on a toy cache with the bench stubbed everywhere.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patcher, undone automatically.

    Returns:
        ``(cache_dir, checkpoint_path, train_config)``.
    """
    cache = build_toy_cache(tmp_path / "cache")
    bench = StubBench()
    # `evaluate` imports `_build_sim` into its own namespace, so patching only
    # `train._build_sim` would leave the evaluator on the real twin.
    monkeypatch.setattr("ml.sim2f.train._build_sim", lambda sim_cfg, device: bench)
    monkeypatch.setattr("ml.sim2f.evaluate._build_sim", lambda sim_cfg, device: bench)

    # 30 steps is the same overfit budget `test_train.py` checks: enough that
    # the `net` row is a trained network rather than a random one, and cheap
    # because the bench is the stub.
    cfg = _tiny_config(steps=30, val_every=15, out_dir=str(tmp_path / "run"))
    train(cfg, cache_dir=cache, device=torch.device("cpu"))
    checkpoint = Path(cfg.out_dir) / CHECKPOINT_LAST
    return cache, checkpoint, cfg


def test_evaluate_writes_the_flat_net_teacher_table(trained) -> None:
    """``eval.json`` sits next to the checkpoint and holds the three rows."""
    cache, checkpoint, _cfg = trained
    report = evaluate(checkpoint, cache_dir=cache, split="test", device="cpu")

    written = checkpoint.parent / EVAL_NAME
    assert written.is_file()
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["split"] == "test"
    assert on_disk["n_samples"] == 1
    assert on_disk["prepare"] is True

    aggregate = report["aggregate"]
    for row in ("flat", "net", "teacher"):
        assert row in aggregate, row
        assert math.isfinite(aggregate[row]["objective"])
        assert 0.0 <= aggregate[row]["ee"] <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= aggregate[row]["ncc"] <= 1.0 + 1e-6
    assert math.isfinite(report["aggregate"]["phasor_corr"])
    assert report["n_parameters"] > 0

    # The teacher phase is an exact optimum of the stub bench for every cached
    # sample, so no single forward pass may beat it.
    assert aggregate["teacher"]["objective"] <= aggregate["net"]["objective"] + 1e-6


def test_evaluate_reports_per_kind_with_counts(trained) -> None:
    """Every kind row carries its own sample count; counts sum to the split."""
    cache, checkpoint, _cfg = trained
    report = evaluate(checkpoint, cache_dir=cache, split="all", device="cpu")

    per_kind = report["per_kind"]
    assert per_kind
    assert sum(entry["n"] for entry in per_kind.values()) == SAMPLES
    for kind, entry in per_kind.items():
        assert entry["n"] >= 1, kind
        for row in ("flat", "net", "teacher"):
            assert math.isfinite(entry[row]["objective"])
    assert len(report["per_sample"]) == SAMPLES


def test_evaluate_honours_an_explicit_out_path(trained, tmp_path: Path) -> None:
    """``out`` writes the report elsewhere instead of beside the checkpoint."""
    cache, checkpoint, _cfg = trained
    target = tmp_path / "reports" / "custom.json"
    evaluate(checkpoint, cache_dir=cache, split="val", device="cpu", out=target)

    assert target.is_file()
    assert not (checkpoint.parent / EVAL_NAME).exists()
    assert json.loads(target.read_text(encoding="utf-8"))["split"] == "val"


def test_evaluate_raw_and_prepared_differ(trained) -> None:
    """The background switch changes the numbers, so a table must name it."""
    cache, checkpoint, _cfg = trained
    prepared = evaluate(checkpoint, cache_dir=cache, split="test", device="cpu")
    raw = evaluate(
        checkpoint, cache_dir=cache, split="test", device="cpu", prepare=False
    )
    assert prepared["prepare"] is True
    assert raw["prepare"] is False
    assert (
        abs(prepared["aggregate"]["net"]["objective"] - raw["aggregate"]["net"]["objective"])
        > 1e-6
    )


def test_evaluate_rejects_a_mismatched_bench(trained) -> None:
    """A checkpoint from another bench is refused, not silently mis-scored."""
    cache, checkpoint, _cfg = trained
    payload = torch.load(checkpoint, weights_only=False)
    payload["sim_config"] = replace(payload["sim_config"], ccd_px=32)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="does not match the cache"):
        evaluate(checkpoint, cache_dir=cache, split="test", device="cpu")


def test_evaluate_rejects_an_unknown_split(trained) -> None:
    """A misspelled split is an error, not an empty table."""
    cache, checkpoint, _cfg = trained
    with pytest.raises(ValueError, match="unknown split"):
        evaluate(checkpoint, cache_dir=cache, split="holdout", device="cpu")


def test_format_table_has_a_row_per_baseline() -> None:
    """The printed table shows one row per baseline and every column."""
    rows = {
        "flat": {"objective": 0.5, "l1": 0.4, "ee": 0.2},
        "net": {"objective": 0.2},
    }
    table = format_table(rows)
    lines = table.splitlines()
    assert len(lines) == 3
    assert "flat" in lines[1] and "net" in lines[2]
    assert "objective" in lines[0] and "cv_in_support" in lines[0]
    # A missing column prints as a dash rather than silently vanishing.
    assert "-" in lines[2]


def test_cli_eval_subcommand(trained, monkeypatch) -> None:
    """``python -m ml.sim2f eval`` runs end-to-end and writes the report."""
    from ml.sim2f.__main__ import main

    cache, checkpoint, _cfg = trained
    code = main(
        [
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--cache",
            str(cache),
            "--split",
            "test",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert (checkpoint.parent / EVAL_NAME).is_file()
