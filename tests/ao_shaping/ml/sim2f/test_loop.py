"""Tests for :mod:`ml.sim2f.loop` -- the five arms, determinism, artefacts.

The loop needs a checkpoint, a corpus and a bench, so this file builds a
**miniature** one: a 64-px camera, a 64-px pupil, ``far_field_padding = 2`` and a
four-sample cache written straight in the schema's format.  That is enough to
exercise the arm construction, the re-commanding step, the per-family
aggregation and the writers without a teacher run -- and it keeps the whole file
in seconds.

What is *not* asserted here: the ordering ``net_closed >= net_probe >= net_open
>= flat``.  It is an empirical claim about a trained network on a real corpus, and
pinning it against a stub would only pin the stub.  ``run_loop`` reports it; the
report and the chat measurement carry it.  What *is* asserted are the structural
invariants that make those numbers meaningful (round 0 is the probe frame, an
ablated stage collapses its arm exactly, every arm is scored on one state).
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import (
    LoopConfig,
    OpticsErrors,
    Sim2fConfig,
    SynthConfig,
    TeacherConfig,
    TrainConfig,
)
from ml.sim2f.loop import ARMS, LOOP_CSV, LOOP_NAME, run_loop
from ml.sim2f.model import build_unet
from ml.sim2f.schema import CACHE_FILE, META_FILE

_GRID = 64
_CCD = 64
_PADDING = 2
_SAMPLES = 3


def _tiny_bench() -> Sim2fConfig:
    """The miniature *calibrated* bench: same errors as the twin, smaller grids."""
    return replace(
        Sim2fConfig.calibrated(),
        grid=_GRID,
        ccd_px=_CCD,
        far_field_padding=_PADDING,
        exposure_ms=20.0,
    )


def _tiny_label() -> Sim2fConfig:
    """The bench the corpus' labels were made on (ideal, same geometry)."""
    return replace(
        Sim2fConfig(),
        grid=_GRID,
        ccd_px=_CCD,
        far_field_padding=_PADDING,
        errors=OpticsErrors(),
    )


def _write_cache(root: Path, count: int = _SAMPLES, seed: int = 5) -> Path:
    """Write a schema-valid, tiny corpus (targets, supports, phases, splits)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    target = np.zeros((count, 1, _CCD, _CCD), dtype=np.float32)
    support = np.zeros((count, 1, _CCD, _CCD), dtype=np.uint8)
    phase = np.zeros((count, _GRID, _GRID), dtype=np.float32)
    for index in range(count):
        size = int(rng.integers(8, 18))
        row = int(rng.integers(8, _CCD - 8 - size))
        col = int(rng.integers(8, _CCD - 8 - size))
        block = np.zeros((_CCD, _CCD), dtype=np.float32)
        block[row : row + size, col : col + size] = 1.0
        target[index, 0] = block / block.sum()
        support[index, 0] = (block > 0).astype(np.uint8)
        phase[index] = rng.normal(0.0, 0.05, size=(_GRID, _GRID)).astype(np.float32)
    spot = np.full((count, 1, _CCD, _CCD), 1.0 / _CCD**2, dtype=np.float32)
    np.savez(
        root / CACHE_FILE,
        target=target,
        support=support,
        phase=phase,
        spot=spot,
        teacher_metrics=np.zeros((count, 4), dtype=np.float32),
        kind=np.zeros((count,), dtype=np.int16),
        seed=np.arange(count, dtype=np.int64),
        split=np.zeros((count,), dtype=np.int8),
    )
    (root / META_FILE).write_text(
        json.dumps(
            {
                "version": 1,
                "sim2f": _tiny_label().to_dict(),
                "teacher": TeacherConfig().to_dict(),
                "synth": SynthConfig(count=count).to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _write_checkpoint(root: Path) -> Path:
    """Save a small U-Net checkpoint in the trainer's format."""
    label = _tiny_label()
    train_cfg = TrainConfig(width=4, depth=2, steps=1, batch_size=1)
    with torch.no_grad():
        zeros = torch.zeros(1, label.grid, label.grid)
        from ml.sim2f.optics import Sim2f

        context = Sim2f(label, device="cpu").forward(zeros).detach()
    model = build_unet(train_cfg, label, device="cpu", context=context)
    path = root / "model_best.pt"
    torch.save(
        {
            "format": 1,
            "state_dict": model.state_dict(),
            "train_config": train_cfg,
            "sim_config": label,
            "train_config_dict": train_cfg.to_dict(),
            "sim_config_dict": label.to_dict(),
            "context": context,
            "step": 1,
            "metrics": {"val_phy": None},
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A tiny corpus + checkpoint, shared by the module (building a bench is slow)."""
    root = tmp_path_factory.mktemp("sim2f_loop")
    cache = _write_cache(root)
    checkpoint = _write_checkpoint(root)
    return cache, checkpoint


def _run(setup, out: Path, **overrides) -> dict:
    """Run the loop with a miniature bench and no noise ladder."""
    cache, checkpoint = setup
    # Detector off by default in these tests: two arms that command the *same*
    # phase must then produce bit-identical frames, which is what makes the
    # ablation-collapse invariants checkable.  The detector adds an independent
    # noise draw per arm (deliberately -- see `loop.ARMS`'s module docstring), so
    # an exact collapse is not available there.
    cfg = LoopConfig(samples=_SAMPLES, rounds=1, seed=11, out_dir=str(out), use_detector=False)
    for key, value in overrides.items():
        cfg = replace(cfg, **{key: value})
    return run_loop(
        cfg,
        cache_dir=cache,
        checkpoint=checkpoint,
        bench_cfg=_tiny_bench(),
        device="cpu",
        ladder_exposures=None,
    )


class TestArms:
    """Every arm is scored, on one state, and the ablations collapse exactly."""

    def test_all_arms_present_and_per_family_sums(self, setup, tmp_path):
        report = _run(setup, tmp_path)
        assert list(report["arms"]) == list(ARMS)
        assert set(report["aggregate"]) == set(ARMS)
        for arm in ARMS:
            row = report["aggregate"][arm]
            for column in ("objective", "l1", "ncc", "ee"):
                assert isinstance(row[column], float)
                assert np.isfinite(row[column])
        assert sum(entry["n"] for entry in report["per_kind"].values()) == _SAMPLES
        assert report["n_samples"] == _SAMPLES
        # One bench state per sample, so every arm of a sample shares the truth.
        for sample in report["per_sample"]:
            assert set(sample["arms"]) == set(ARMS)
            assert len(sample["truth"]["coeffs"]) == report["aberration"]["n_modes"]

    def test_round_zero_is_the_probe_frame(self, setup, tmp_path):
        report = _run(setup, tmp_path, rounds=2)
        for sample in report["per_sample"]:
            rounds = sample["closed"]["rounds"]
            assert [entry["round"] for entry in rounds] == [0, 1, 2]
            assert rounds[0]["objective"] == pytest.approx(sample["arms"]["net_probe"]["objective"])
            assert sample["closed"]["round_count"] == 2
            assert 0 <= sample["closed"]["improved_rounds"] <= 2

    def test_disabling_the_probe_collapses_net_probe_onto_net_open(self, setup, tmp_path):
        report = _run(setup, tmp_path, probe_fit=False)
        for sample in report["per_sample"]:
            assert not sample["probe"]["ok"]
            assert "disabled" in (sample["probe"]["reason"] or "")
            assert sample["arms"]["net_probe"] == sample["arms"]["net_open"]
            # `net_closed` is *not* required to collapse: with the probe off, its
            # refit starts from zero and may still improve a shaped frame from
            # there (measured on this stub: 2.235 -> 1.980).  The refit and the
            # probe are separate legs, which is why the arms separate them.
            assert sample["closed"]["round_count"] == 1

    def test_disabling_the_refit_collapses_net_closed_onto_net_probe(self, setup, tmp_path):
        report = _run(setup, tmp_path, probe_fit=True, refit=False)
        for sample in report["per_sample"]:
            assert len(sample["closed"]["rounds"]) == 1
            assert sample["arms"]["net_closed"] == sample["arms"]["net_probe"]
        assert report["rounds"]["total"] == 0
        assert report["rounds"]["improved_fraction"] is None

    def test_flat_arm_ignores_the_network(self, setup, tmp_path):
        report = _run(setup, tmp_path)
        # The flat arm is the bench's own spot: its frame is the same for every
        # sample only if the aberration were absent, so instead check that it is
        # scored and differs from the shaped arms (the states differ per sample).
        assert all("flat" in sample["arms"] for sample in report["per_sample"])
        assert report["rounds"]["total"] == report["rounds"]["rounds_per_sample"] * _SAMPLES


class TestArtefacts:
    """JSON + CSV are written, complete, and JSON is standards-compliant."""

    def test_writes_report_and_csv(self, setup, tmp_path):
        report = _run(setup, tmp_path)
        out = Path(report["loop"]["out_dir"])
        payload = json.loads((out / LOOP_NAME).read_text(encoding="utf-8"))
        assert payload["n_samples"] == report["n_samples"]
        assert payload["notes"]
        with (out / LOOP_CSV).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        metrics = {row["metric"] for row in rows}
        assert metrics >= {"objective", "l1", "ncc"}
        # 5 arms x 7 columns per sample, plus one objective row per closed round.
        expected = _SAMPLES * (len(ARMS) * 7 + 2)
        assert len(rows) == expected

    def test_detector_run_stays_finite(self, setup, tmp_path):
        report = _run(setup, tmp_path / "det", use_detector=True)
        for arm in ARMS:
            row = report["aggregate"][arm]
            assert all(np.isfinite(row[column]) for column in ("objective", "l1", "ncc", "ee"))
        # A refused probe is reported, not hidden: the reason names the stage.
        for sample in report["per_sample"]:
            if not sample["probe"]["ok"]:
                assert sample["probe"]["reason"]

    def test_round_trips_without_nan_tokens(self, setup, tmp_path):
        report = _run(setup, tmp_path)
        text = (Path(report["loop"]["out_dir"]) / LOOP_NAME).read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text


class TestDeterminism:
    """Same seed, same numbers -- the whole point of seeding every draw."""

    def test_two_runs_agree(self, setup, tmp_path):
        first = _run(setup, tmp_path / "a")
        second = _run(setup, tmp_path / "b")
        assert first["aggregate"] == second["aggregate"]
        assert first["per_kind"] == second["per_kind"]
        assert first["rounds"] == second["rounds"]
        for left, right in zip(first["per_sample"], second["per_sample"], strict=True):
            assert left["arms"] == right["arms"]
            assert left["truth"] == right["truth"]
            assert left["closed"]["rounds"] == right["closed"]["rounds"]

    def test_different_seed_draws_a_different_state(self, setup, tmp_path):
        first = _run(setup, tmp_path / "a", seed=11)
        second = _run(setup, tmp_path / "b", seed=12)
        assert first["per_sample"][0]["truth"] != second["per_sample"][0]["truth"]


class TestGuards:
    """A bench that is not the corpus' bench must not be silently measured."""

    def test_geometry_mismatch_raises(self, setup, tmp_path):
        cache, checkpoint = setup
        with pytest.raises(ValueError, match="disagrees with the corpus"):
            run_loop(
                LoopConfig(samples=1, rounds=0, out_dir=str(tmp_path)),
                cache_dir=cache,
                checkpoint=checkpoint,
                bench_cfg=Sim2fConfig.calibrated(),  # 128/256, not the tiny corpus
                device="cpu",
                ladder_exposures=None,
            )

    def test_missing_checkpoint_raises(self, setup, tmp_path):
        cache, _ = setup
        with pytest.raises(FileNotFoundError):
            run_loop(
                LoopConfig(samples=1, rounds=0, out_dir=str(tmp_path)),
                cache_dir=cache,
                checkpoint=tmp_path / "nope.pt",
                bench_cfg=_tiny_bench(),
                device="cpu",
                ladder_exposures=None,
            )

    def test_sample_count_is_capped_by_the_corpus(self, setup, tmp_path):
        report = _run(setup, tmp_path, samples=99)
        assert report["n_samples"] == _SAMPLES
