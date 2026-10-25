"""Draw the supervised corpus, run the teacher, write the cache.

One sample is one ``(target, teacher phase)`` pair.  The teacher label depends only
on the target -- the bench it is computed on is the *aberration-free* one, with
nominal illumination -- which is what makes the corpus reusable across bench
states: aberrations are handled at evaluation time by the Zernike probe fit and
the closed-loop refit, not by the labels.

Reproducibility
===============
Three layers, in order:

1. the master seed produces one **per-sample seed** per pair (stored as ``seed`` in
   the archive), so any single sample can be regenerated alone;
2. splits are **stratified by target kind** -- round-robin over the shuffled kind
   list, then ``train / val / test`` inside each kind.  A plain global shuffle
   would routinely leave one kind entirely out of validation, and the corpus's
   shapes are heterogeneous enough that this showed up as fold-to-fold noise in the
   earlier `zernike_phase2amp` work;
3. the teacher itself has no stochastic step (flat initial phase, Adam), so the
   same three configs produce byte-identical arrays.

What is written
===============
``pairs.npz`` exactly as declared by :data:`ml.sim2f.schema.CACHE_FIELDS`, plus
``meta.json`` with the three configs, the derived optics
(:meth:`ml.sim2f.config.Sim2fConfig.describe`), the split/kind counts, the wall
clock and the tool versions.  Every write is verified by reloading the file and
running :func:`ml.sim2f.schema.validate_cache`, so a schema drift fails here
rather than in the trainer.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ml.sim2f.config import TARGET_KINDS, Sim2fConfig, SynthConfig, TeacherConfig
from ml.sim2f.optics import Sim2f
from ml.sim2f.schema import (
    CACHE_FILE,
    CACHE_VERSION,
    META_FILE,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    validate_cache,
)
from ml.sim2f.targets import draw_target
from ml.sim2f.teacher import Teacher

__all__ = ["build_pairs", "load_pairs"]


def build_pairs(
    sim_cfg: Sim2fConfig,
    synth_cfg: SynthConfig,
    teacher_cfg: TeacherConfig,
    *,
    out_dir: str | Path,
    device: str = "auto",
    progress: bool = False,
) -> Path:
    """Generate the corpus and write ``pairs.npz`` / ``meta.json``.

    Args:
        sim_cfg: Bench the labels are computed on. The default (ideal) config is
            the intended choice; a calibrated one is accepted, in which case the
            labels and the reference ``spot`` carry its optical errors.
        synth_cfg: How many pairs, which kinds, the master seed and the split
            fractions.
        teacher_cfg: Teacher objective and budget.
        out_dir: Directory to write into (created if missing).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        progress: Log every sample's progress. Off by default because the corpus
            is small and the log would dominate.

    Returns:
        The output directory (holding :data:`~ml.sim2f.schema.CACHE_FILE` and
        :data:`~ml.sim2f.schema.META_FILE`).

    Raises:
        ValueError: If the written archive does not satisfy
            :func:`ml.sim2f.schema.validate_cache`.
    """
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = int(synth_cfg.count)
    ccd_px = int(sim_cfg.ccd_px)
    grid = int(sim_cfg.grid)

    # 1. per-sample seeds, drawn from the master seed so each sample is
    #    reproducible on its own.
    master = np.random.default_rng(int(synth_cfg.seed))
    seeds = master.integers(0, 2**31 - 1, size=count, dtype=np.int64)

    # 2. kinds round-robin over the shuffled kind list, then stratified splits.
    shuffled = [str(kind) for kind in master.permutation(np.array(synth_cfg.target_kinds))]
    kinds = [shuffled[index % len(shuffled)] for index in range(count)]
    splits = _assign_splits(kinds, synth_cfg, master)

    sim = Sim2f(sim_cfg, device=device)
    teacher = Teacher(sim, teacher_cfg)

    target = np.empty((count, 1, ccd_px, ccd_px), dtype=np.float32)
    support = np.empty((count, 1, ccd_px, ccd_px), dtype=np.uint8)
    phase = np.empty((count, grid, grid), dtype=np.float32)
    metrics = np.empty((count, 4), dtype=np.float32)
    kind_index = np.empty((count,), dtype=np.int16)
    seed = np.empty((count,), dtype=np.int64)
    split = np.empty((count,), dtype=np.int8)

    zeros = np.zeros((1, grid, grid), dtype=np.float32)
    spot = (
        sim.forward(zeros, detector=False)[0, 0].detach().to("cpu").numpy().astype(np.float32)
    )

    for index in range(count):
        kind = kinds[index]
        seed_value = int(seeds[index])
        rng = np.random.default_rng(seed_value)
        sample = draw_target(
            kind, ccd_px, rng, size_frac=tuple(synth_cfg.size_frac)
        )
        result = teacher.optimise(
            torch.as_tensor(sample.intensity)[None],
            support=torch.as_tensor(sample.support),
            seed=seed_value,
        )
        target[index, 0] = sample.intensity
        support[index, 0] = sample.support.astype(np.uint8)
        phase[index] = result.phase
        metrics[index] = (result.loss, result.l1, result.ee, result.ncc)
        kind_index[index] = TARGET_KINDS.index(kind)
        seed[index] = seed_value
        split[index] = splits[index]
        if progress:
            logger.info(
                "[{}/{}] {} seed={} loss={:.3f} ee={:.3f} ncc={:.3f} split={}",
                index + 1,
                count,
                kind,
                seed_value,
                result.loss,
                result.ee,
                result.ncc,
                SPLIT_NAMES[int(splits[index])],
            )

    # 3. one archive, every field from the schema.
    archive = out_path / CACHE_FILE
    np.savez(
        archive,
        target=target,
        support=support,
        phase=phase,
        spot=np.repeat(spot[None, None], count, axis=0),
        teacher_metrics=metrics,
        kind=kind_index,
        seed=seed,
        split=split,
    )

    wall_s = time.perf_counter() - started
    meta = _build_meta(
        sim_cfg=sim_cfg,
        synth_cfg=synth_cfg,
        teacher_cfg=teacher_cfg,
        kinds=kinds,
        splits=splits,
        wall_s=wall_s,
    )
    (out_path / META_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 4. verify what landed on disk, not what we think we wrote.
    with np.load(archive) as stored:
        validate_cache(stored, grid=grid, ccd_px=ccd_px)

    logger.info(
        "wrote {} pairs to {} in {:.1f}s (train/val/test = {}/{}/{})",
        count,
        archive,
        wall_s,
        int((splits == SPLIT_TRAIN).sum()),
        int((splits == SPLIT_VAL).sum()),
        int((splits == SPLIT_TEST).sum()),
    )
    return out_path


def load_pairs(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a corpus and check it against the schema.

    Args:
        path: The corpus directory (or the ``pairs.npz`` file itself).

    Returns:
        ``(arrays, meta)`` where ``arrays`` holds every
        :data:`~ml.sim2f.schema.CACHE_FIELDS` entry as a plain numpy array and
        ``meta`` is the parsed ``meta.json``.

    Raises:
        FileNotFoundError: If the archive or the sidecar is missing.
        ValueError: If the archive violates
            :func:`ml.sim2f.schema.validate_cache`, or the sidecar does not carry
            the ``grid`` / ``ccd_px`` the check needs.
    """
    location = Path(path)
    archive_path = location / CACHE_FILE if location.is_dir() else location
    meta_path = archive_path.parent / META_FILE
    if not archive_path.is_file():
        raise FileNotFoundError(f"no archive at {archive_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"no sidecar at {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sim_cfg = meta.get("sim2f") or {}
    grid = sim_cfg.get("grid")
    ccd_px = sim_cfg.get("ccd_px")
    if grid is None or ccd_px is None:
        raise ValueError(
            f"{meta_path} must record sim2f.grid and sim2f.ccd_px to validate the archive"
        )

    with np.load(archive_path) as stored:
        validate_cache(stored, grid=int(grid), ccd_px=int(ccd_px))
        arrays = {name: stored[name] for name in stored.files}
    return arrays, meta


def _assign_splits(
    kinds: list[str], synth_cfg: SynthConfig, rng: np.random.Generator
) -> np.ndarray:
    """Assign train / val / test **inside each target kind**.

    Args:
        kinds: Per-sample kind, in sample order.
        synth_cfg: Split fractions.
        rng: Master generator (the shuffle is part of the corpus identity).

    Returns:
        ``(N,)`` int8 array of ``SPLIT_TRAIN`` / ``SPLIT_VAL`` / ``SPLIT_TEST``.
    """
    splits = np.full(len(kinds), SPLIT_TRAIN, dtype=np.int8)
    for kind in dict.fromkeys(kinds):
        members = rng.permutation([i for i, value in enumerate(kinds) if value == kind])
        n_members = len(members)
        n_train = int(round(float(synth_cfg.train_frac) * n_members))
        n_val = int(round(float(synth_cfg.val_frac) * n_members))
        for position, index in enumerate(members):
            if position < n_train:
                splits[index] = SPLIT_TRAIN
            elif position < n_train + n_val:
                splits[index] = SPLIT_VAL
            else:
                splits[index] = SPLIT_TEST
    return splits


def _build_meta(
    *,
    sim_cfg: Sim2fConfig,
    synth_cfg: SynthConfig,
    teacher_cfg: TeacherConfig,
    kinds: list[str],
    splits: np.ndarray,
    wall_s: float,
) -> dict:
    """Assemble ``meta.json``.

    Everything here is either a config dump or a number derived from one, so a
    cached corpus can be described -- and its teacher reproduced -- without the
    code that wrote it.

    Args:
        sim_cfg: Bench config.
        synth_cfg: Corpus config.
        teacher_cfg: Teacher config.
        kinds: Per-sample kind name.
        splits: Per-sample split index.
        wall_s: Wall clock of the whole build.

    Returns:
        JSON-serialisable metadata.
    """
    split_names = [SPLIT_NAMES[int(value)] for value in splits]
    return {
        "version": int(CACHE_VERSION),
        "sim2f": sim_cfg.to_dict(),
        "synth": synth_cfg.to_dict(),
        "teacher": teacher_cfg.to_dict(),
        "derived": sim_cfg.describe(),
        "counts": {
            "split": {name: int(split_names.count(name)) for name in SPLIT_NAMES},
            "kind": {kind: int(kinds.count(kind)) for kind in dict.fromkeys(kinds)},
        },
        "wall_s": float(wall_s),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    """Current ``HEAD``, or ``None`` when git is unavailable.

    Returns:
        The commit hash, or ``None`` (a build outside a checkout is legitimate).
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
