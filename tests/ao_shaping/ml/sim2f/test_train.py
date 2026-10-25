"""Tests for :mod:`ml.sim2f.train` -- the training loop, its artefacts, overfit.

The corpus is a tiny cache written into ``tmp_path`` that satisfies
:data:`ml.sim2f.schema.CACHE_FIELDS` exactly, and the bench is a hand-written
forward model (``|FFT2(exp(i*phase))|^2`` plus a constant pedestal).  Neither
depends on ``ml.sim2f.optics`` or on the real twin, so this file runs while
those are still being changed; the stub is monkeypatched in through
``train._build_sim``, which is the same seam the real bench enters by.

The pedestal in the stub is deliberate: the production bench is the *calibrated*
2f twin, whose frames carry a dark level and a 0-order background, so the
training path has to be background-robust rather than energy-conserving.  No
test here asserts that a frame's total equals the modulated energy.

``build_toy_cache`` and ``StubBench`` are imported by ``test_evaluate.py`` (it
needs a checkpoint to score), so they are written to be reusable.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from dataclasses import fields, replace
from pathlib import Path

from ml.sim2f.config import Sim2fConfig, TARGET_KINDS, TeacherConfig, TrainConfig
from ml.sim2f.evaluate import evaluate
from ml.sim2f.losses import objective_terms
from ml.sim2f.model import Sim2fUNet, build_unet
from ml.sim2f.schema import CACHE_FILE, META_FILE
from ml.sim2f.train import (
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    CONFIG_NAME,
    HISTORY_NAME,
    METRICS_NAME,
    _build_sim,
    _evaluate_split,
    build_parser,
    config_from_args,
    context_from_corpus,
    load_corpus,
    seed_everything,
    train,
)

GRID = 64
CCD = 64
SAMPLES = 6
#: 4 train / 1 val / 1 test -- enough for a validation pass and a test split.
SPLITS = (0, 0, 0, 0, 1, 2)
#: Constant pedestal standing in for dark level + 0-order background.
BACKGROUND = 40.0


def _sim_config(**overrides: object) -> Sim2fConfig:
    """Build a :class:`Sim2fConfig` from field overrides only.

    Args:
        **overrides: Field values to apply.

    Returns:
        The configuration, with unknown field names dropped.
    """
    known = {item.name for item in fields(Sim2fConfig)}
    values: dict[str, object] = {"grid": GRID, "ccd_px": CCD}
    values.update(overrides)
    return replace(
        Sim2fConfig(), **{key: value for key, value in values.items() if key in known}
    )


def _pupil_disc(size: int, fill: float = 0.75) -> torch.Tensor:
    """The aperture disc, built the same way the network's mask is.

    Args:
        size: Grid side.
        fill: Aperture diameter as a fraction of ``size``.

    Returns:
        ``(size, size)`` with 1 inside the disc.
    """
    coordinates = torch.arange(size, dtype=torch.float32) + 0.5 - 0.5 * size
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return (torch.hypot(yy, xx) <= 0.5 * fill * size).to(torch.float32)


class StubBench(torch.nn.Module):
    """Differentiable 2f stand-in: an apertured field, then a Fourier transform.

    ``I = |FFT2(disc * exp(i*phase))|^2 + background``, i.e. the same convention
    as ``optics.Sim2f.forward`` (a ``(B, grid, grid)`` phase in radians gives a
    ``(B, 1, ccd_px, ccd_px)`` frame) but without a 0-order mix or a detector.

    The aperture matters for the tests' *validity*, not for realism: without it a
    flat phase focuses into a single pixel and every target in the corpus looks
    like the flat spot, which would make the flat baseline unbeatable by
    construction and the overfit check vacuous.
    """

    def __init__(self, size: int = GRID, background: float = BACKGROUND) -> None:
        """Build the stateless bench.

        Args:
            size: Grid side of the pupil.
            background: Constant added to every pixel (dark level + 0-order
                pedestal stand-in).
        """
        super().__init__()
        self.background = float(background)
        self.register_buffer("aperture", _pupil_disc(size))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Propagate a phase map.

        Args:
            phase: ``(B, G, G)`` or ``(B, 1, G, G)`` radians.

        Returns:
            ``(B, 1, G, G)`` intensity plus the pedestal.
        """
        if phase.ndim == 4:
            phase = phase[:, 0]
        field = self.aperture * torch.exp(1j * phase.to(torch.complex64))
        image = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1)).abs().pow(2)
        return image[:, None] + self.background


def _toy_phase(size: int, seed: int, amplitude: float) -> torch.Tensor:
    """A low-order ripple inside the pupil disc the network can express.

    Two properties are load-bearing, and both are about the *objective*, not
    about realism:

    * the ripple stays inside the pupil disc, because the U-Net parks its phasor
      at ``(1, 0)`` outside ``pupil_fill`` -- a label that is non-zero there
      would be unreachable by construction;
    * the ripple is low-order and modest, so the target still **overlaps** the
      flat spot.  ``EE = sum(p * support)`` has an exactly zero derivative while
      the shaped spot and the support are disjoint, so a corpus whose targets
      sit far from the flat spot is a plateau for the physics leg: 30 steps then
      cannot move it, and the overfit check would be measuring the plateau
      rather than the loop.

    Args:
        size: Grid side.
        seed: RNG seed.
        amplitude: Peak ripple amplitude, radians.

    Returns:
        ``(size, size)`` radians, zero outside the pupil.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, 1, size, size, generator=generator)
    padded = torch.nn.functional.pad(noise, (4, 4, 4, 4), mode="replicate")
    ripple = torch.nn.functional.avg_pool2d(padded, kernel_size=9, stride=1)[0, 0]
    return (amplitude * ripple * _pupil_disc(size)).detach()


def build_toy_cache(
    root: Path, *, samples: int = SAMPLES, splits: tuple[int, ...] = SPLITS
) -> Path:
    """Write a schema-valid cache whose labels are exactly achievable.

    Each sample's target is the stub bench's own frame for that sample's phase,
    so the teacher phase is an exact optimum of the physics leg; that is what
    makes the overfit check meaningful (a network that cannot reach the label is
    a broken network, not a hard problem).

    Args:
        root: Directory to create.
        samples: Number of pairs.
        splits: One split code per sample (0 train / 1 val / 2 test).  Pass a
            table with no ``1`` to build the empty-validation corpus that a
            smoke run of a handful of pairs produces.

    Returns:
        ``root``, containing ``pairs.npz`` and ``meta.json``.

    Raises:
        ValueError: If ``samples`` and ``splits`` disagree in length.
    """
    if samples != len(splits):
        raise ValueError(f"the split table has {len(splits)} entries, got {samples}")
    root.mkdir(parents=True, exist_ok=True)
    sim_cfg = _sim_config()
    teacher_cfg = TeacherConfig()
    bench = StubBench()

    target = np.zeros((samples, 1, CCD, CCD), dtype=np.float32)
    support = np.zeros((samples, 1, CCD, CCD), dtype=np.uint8)
    phase = np.zeros((samples, GRID, GRID), dtype=np.float32)
    teacher_metrics = np.zeros((samples, 4), dtype=np.float32)
    kind = np.zeros((samples,), dtype=np.int16)
    seeds = np.zeros((samples,), dtype=np.int64)
    split = np.asarray(splits, dtype=np.int8)

    for index in range(samples):
        theta = _toy_phase(GRID, 1000 + index, amplitude=0.35 + 0.05 * index)
        with torch.no_grad():
            image = bench(theta[None])
        unit = image / image.sum()
        mask = (unit >= 0.15 * unit.max()).to(torch.float32)
        loss, l1, ee, ncc = objective_terms(unit, unit, mask, w_ee=teacher_cfg.w_ee)
        del loss
        target[index] = unit[0].numpy()
        support[index] = mask[0].numpy().astype(np.uint8)
        phase[index] = theta.numpy()
        teacher_metrics[index] = [
            0.0,  # the teacher phase is the exact optimum of this stub
            float(l1),
            float(ee),
            float(ncc),
        ]
        kind[index] = index % len(TARGET_KINDS)
        seeds[index] = 1000 + index

    spot = torch.zeros(1, 1, CCD, CCD)
    with torch.no_grad():
        spot = bench(torch.zeros(1, GRID, GRID))

    np.savez(
        root / CACHE_FILE,
        target=target,
        support=support,
        phase=phase,
        spot=np.repeat(spot.numpy().astype(np.float32), samples, axis=0),
        teacher_metrics=teacher_metrics,
        kind=kind,
        seed=seeds,
        split=split,
    )
    meta = {
        "sim": sim_cfg.to_dict(),
        "teacher": teacher_cfg.to_dict(),
        "synth": {"count": samples, "note": "toy cache for tests"},
    }
    (root / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


@pytest.fixture()
def toy_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny cache with the optics seam patched to a stub bench.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest patcher, undone automatically.

    Returns:
        The cache directory.
    """
    cache = build_toy_cache(tmp_path / "cache")
    bench = StubBench()
    monkeypatch.setattr("ml.sim2f.train._build_sim", lambda sim_cfg, device: bench)
    return cache


def _tiny_config(**overrides: object) -> TrainConfig:
    """A configuration small enough to overfit six samples on CPU in seconds.

    Args:
        **overrides: Field values to apply.

    Returns:
        The configuration.
    """
    values: dict[str, object] = {
        "width": 8,
        "depth": 2,
        "batch_size": 4,
        "steps": 30,
        "lr": 1.0e-2,
        "val_every": 10,
        "seed": 0,
    }
    values.update(overrides)
    return TrainConfig(**values)  # type: ignore[arg-type]


def _split_objective(
    model: Sim2fUNet, corpus: object, indices: torch.Tensor, *, cfg: TrainConfig
) -> dict[str, float]:
    """Score the training objective on a split with the stubbed bench.

    Args:
        model: The network.
        corpus: The loaded corpus.
        indices: Sample indices.
        cfg: Loss weights.

    Returns:
        Mean objective terms for the split.
    """
    context = context_from_corpus(corpus)  # type: ignore[arg-type]
    return _evaluate_split(
        model,
        corpus,  # type: ignore[arg-type]
        indices,
        context=context,
        sim=StubBench(),
        cfg=cfg,
        teacher=TeacherConfig(),
        mask=model.pupil_mask,
        batch_size=cfg.batch_size,
        device=torch.device("cpu"),
    )


# ----------------------------------------------------------------------
# The corpus reader
# ----------------------------------------------------------------------
def test_load_corpus_validates_and_exposes_splits(toy_cache: Path) -> None:
    """The reader round-trips the archive and reports per-split counts."""
    corpus = load_corpus(toy_cache)
    assert corpus.n_samples == SAMPLES
    assert int(corpus.indices("train").numel()) == 4
    assert int(corpus.indices("val").numel()) == 1
    assert int(corpus.indices("test").numel()) == 1
    assert int(corpus.indices("all").numel()) == SAMPLES
    assert corpus.sim_cfg.grid == GRID
    assert corpus.sim_cfg.ccd_px == CCD
    assert len(corpus.kinds()) == SAMPLES
    assert corpus.teacher_metrics.shape == (SAMPLES, 4)
    with pytest.raises(ValueError, match="unknown split"):
        corpus.indices("holdout")


def test_load_corpus_reports_a_missing_archive(tmp_path: Path) -> None:
    """A missing ``pairs.npz`` names the command that would create it."""
    with pytest.raises(FileNotFoundError, match="synth"):
        load_corpus(tmp_path / "nowhere")


def test_context_channel_is_the_cached_flat_frame(toy_cache: Path) -> None:
    """The PSF-context channel is the cache's own flat frame, peak-normalised."""
    corpus = load_corpus(toy_cache)
    context = context_from_corpus(corpus)
    assert context.shape == (1, 1, CCD, CCD)
    assert float(context.amax()) == pytest.approx(1.0)
    # The stub's flat frame is a single hot pixel plus the pedestal; preparing
    # is not applied here, so the peak pixel keeps its dominance.
    assert float(context[0, 0].min()) < float(context.amax()) * 0.1


# ----------------------------------------------------------------------
# The parser
# ----------------------------------------------------------------------
def test_parser_defaults_come_from_the_dataclass() -> None:
    """Every CLI default is the dataclass value, so the two cannot drift."""
    parser = build_parser()
    args = parser.parse_args(["--cache", "somewhere"])
    cfg = config_from_args(args)
    default = TrainConfig()
    for name in ("steps", "batch_size", "lr", "width", "depth", "w_sup", "w_phy",
                 "w_tv", "val_every", "seed", "device", "out_dir", "amp",
                 "cudnn_deterministic"):
        assert getattr(cfg, name) == getattr(default, name), name


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def test_train_overfits_a_six_sample_cache(toy_cache: Path, tmp_path: Path) -> None:
    """30 steps on six samples more than halve the objective (overfit check)."""
    cfg = _tiny_config(out_dir=str(tmp_path / "run"))
    corpus = load_corpus(toy_cache)
    device = torch.device("cpu")

    # Baseline: a fresh network from the same seed, i.e. exactly what the loop
    # starts from. `seed_everything` is the same call `train` makes first.
    seed_everything(cfg.seed)
    context = context_from_corpus(corpus)
    initial = build_unet(cfg, corpus.sim_cfg, device=device, context=context)
    train_indices = corpus.indices("train")
    before = _split_objective(initial, corpus, train_indices, cfg=cfg)

    result = train(cfg, cache_dir=toy_cache, device=device)

    payload = torch.load(Path(cfg.out_dir) / CHECKPOINT_LAST, weights_only=False)
    trained = build_unet(cfg, corpus.sim_cfg, device=device, context=context)
    trained.load_state_dict(payload["state_dict"])
    after = _split_objective(trained, corpus, train_indices, cfg=cfg)

    assert result["n_parameters"] > 0
    assert before["total"] > 0.0
    assert after["total"] < 0.5 * before["total"], (before, after)
    # The physics leg must improve as well -- otherwise the drop could come from
    # the supervised term alone, which says nothing about shaping quality.
    assert after["phy"] < before["phy"]
    print(
        f"overfit: objective {before['total']:.4f} -> {after['total']:.4f}; "
        f"physics {before['phy']:.4f} -> {after['phy']:.4f}"
    )


def test_train_writes_every_artefact(toy_cache: Path, tmp_path: Path) -> None:
    """The run leaves the four documents plus two checkpoints on disk."""
    cfg = _tiny_config(steps=4, val_every=2, out_dir=str(tmp_path / "run"))
    result = train(cfg, cache_dir=toy_cache, device=torch.device("cpu"))
    out_dir = Path(cfg.out_dir)

    for name in (CONFIG_NAME, METRICS_NAME, HISTORY_NAME, CHECKPOINT_BEST, CHECKPOINT_LAST):
        assert (out_dir / name).is_file(), name
    assert set(result["paths"]) == {"config", "metrics", "history", "best", "last"}

    payload = torch.load(out_dir / CHECKPOINT_BEST, weights_only=False)
    assert isinstance(payload["train_config"], TrainConfig)
    assert isinstance(payload["sim_config"], Sim2fConfig)
    assert payload["context"].shape == (1, 1, CCD, CCD)
    assert any(key.startswith("encoder.0") for key in payload["state_dict"])

    metrics = json.loads((out_dir / METRICS_NAME).read_text(encoding="utf-8"))
    assert metrics["n_parameters"] == result["n_parameters"]
    assert metrics["counts"] == {"train": 4, "val": 1, "test": 1}
    assert metrics["steps"] == 4
    assert metrics["val_enabled"] is True
    assert result["val_enabled"] is True
    assert isinstance(result["best_val_phy"], float)


def test_train_is_bit_reproducible_for_the_same_seed(
    toy_cache: Path, tmp_path: Path
) -> None:
    """Two runs with the same seed write byte-identical ``history.csv``."""
    first = _tiny_config(steps=6, val_every=3, out_dir=str(tmp_path / "a"))
    second = _tiny_config(steps=6, val_every=3, out_dir=str(tmp_path / "b"))
    train(first, cache_dir=toy_cache, device=torch.device("cpu"))
    train(second, cache_dir=toy_cache, device=torch.device("cpu"))

    left = (Path(first.out_dir) / HISTORY_NAME).read_bytes()
    right = (Path(second.out_dir) / HISTORY_NAME).read_bytes()
    assert left == right
    rows = left.decode("utf-8").strip().splitlines()
    assert rows[0].split(",")[0] == "step"
    assert len(rows) == 3  # header + two validation events


def test_train_records_the_objective_terms(toy_cache: Path, tmp_path: Path) -> None:
    """``history.csv`` carries both legs, so an ablation can be read off it."""
    cfg = _tiny_config(steps=4, val_every=2, out_dir=str(tmp_path / "run"))
    train(cfg, cache_dir=toy_cache, device=torch.device("cpu"))
    rows = (Path(cfg.out_dir) / HISTORY_NAME).read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    assert {"train_total", "train_sup", "train_phy", "val_phy", "val_total"} <= set(header)
    for row in rows[1:]:
        values = [float(part) for part in row.split(",")]
        assert all(math.isfinite(value) for value in values)


def test_train_rejects_an_empty_train_split(
    toy_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache with no training samples fails with an actionable message."""
    corpus = load_corpus(toy_cache)
    corpus.split = torch.full_like(corpus.split, 1)  # everything is validation
    cfg = _tiny_config(steps=1, val_every=1, out_dir=str(tmp_path / "run"))
    monkeypatch.setattr("ml.sim2f.train.load_corpus", lambda _: corpus)
    with pytest.raises(ValueError, match="train split"):
        train(cfg, cache_dir=toy_cache, device=torch.device("cpu"))


def test_build_sim_is_the_only_optics_seam() -> None:
    """``_build_sim`` constructs the real bench (lazily imported)."""
    pytest.importorskip("ml.sim2f.optics", reason="optics.py is written by a parallel change")
    bench = _build_sim(_sim_config(), torch.device("cpu"))
    assert bench.grid == GRID
    assert bench.ccd_px == CCD


def test_train_without_a_val_split_writes_both_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-train corpus disables validation instead of tabulating NaN.

    This is the smoke-run shape: a handful of pairs, every one of them train.  A
    loop that "validates" the empty split writes a row of ``nan`` every
    ``val_every`` steps and never lets a checkpoint qualify as best, so
    ``evaluate`` finds no ``model_best.pt``.  The run must instead say so once,
    keep the table numeric, and leave a best checkpoint that equals the final
    state.
    """
    cache = build_toy_cache(tmp_path / "cache", splits=(0, 0, 0, 0, 2, 2))
    bench = StubBench()
    monkeypatch.setattr("ml.sim2f.train._build_sim", lambda sim_cfg, device: bench)
    monkeypatch.setattr("ml.sim2f.evaluate._build_sim", lambda sim_cfg, device: bench)

    cfg = _tiny_config(steps=10, val_every=5, out_dir=str(tmp_path / "run"))
    result = train(cfg, cache_dir=cache, device=torch.device("cpu"))
    out_dir = Path(cfg.out_dir)

    # The invariant, read off the returned metrics rather than off a log line.
    assert result["val_enabled"] is False
    assert result["best_val_phy"] is None
    assert result["best_step"] == cfg.steps
    assert result["counts"] == {"train": 4, "val": 0, "test": 2}

    text = (out_dir / HISTORY_NAME).read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    rows = [line.split(",") for line in text.strip().splitlines()]
    assert len(rows) == 3  # header + the step-5 and step-10 rows
    header = rows[0]
    for row in rows[1:]:
        assert len(row) == len(header)
        for column, cell in zip(header, row):
            if column.startswith("val_"):
                assert cell == "", f"{column} should be empty, got {cell!r}"
            else:
                assert math.isfinite(float(cell)), column

    metrics = json.loads((out_dir / METRICS_NAME).read_text(encoding="utf-8"))
    assert metrics["val_enabled"] is False
    assert metrics["best_val_phy"] is None
    assert metrics["best_step"] == cfg.steps
    assert all(row.get("val_phy") is None for row in metrics["history"])

    # Both checkpoints exist and carry the same (final) weights.
    best_path = out_dir / CHECKPOINT_BEST
    last_path = out_dir / CHECKPOINT_LAST
    assert best_path.is_file() and last_path.is_file()
    best = torch.load(best_path, weights_only=False)
    last = torch.load(last_path, weights_only=False)
    assert best["step"] == cfg.steps == last["step"]
    assert best["metrics"]["val_phy"] is None
    for key, value in last["state_dict"].items():
        assert torch.equal(best["state_dict"][key], value), key

    # And the checkpoint the run left behind is usable downstream.
    report = evaluate(best_path, cache_dir=cache, split="test", device="cpu")
    assert report["n_samples"] == 2
    assert math.isfinite(report["aggregate"]["net"]["objective"])
