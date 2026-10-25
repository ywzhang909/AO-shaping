"""``python -m ml.sim2f {synth,train,eval,loop}`` -- the package's entry points.

``synth`` builds the supervised corpus (targets, teacher labels, cache archive);
``train`` fits the U-Net on it; ``eval`` scores a checkpoint on a split and
writes the flat / net / teacher table; ``loop`` runs the five-arm closed-loop
ablation (probe + refit) on the calibrated twin.  Each sub-command owns its own
parser and every one of its defaults is read from a frozen dataclass in
:mod:`ml.sim2f.config` -- the repository shipped a parser literal that silently
overrode a calibrated default on every run
(``report/zernike_phase2amp/report.md``), so "the CLI default" and "the code
default" are made the same object here.

Flags are not duplicated between this module and the sub-command modules:
``train`` simply forwards its argument vector to :func:`ml.sim2f.train.main`.
That is deliberate -- two parsers for one run is exactly how a default drifts.

``--grid`` / ``--ccd-px`` / ``--far-field-padding`` overrides
------------------------------------------------
These describe the optics, and only ``synth`` is free to choose them: it *makes*
the labels.  ``train`` and ``eval`` read the bench from the cache (``meta.json``
and the array shapes), because a network trained at one ``grid`` cannot consume
a phase map at another, and evaluating on a different ``far_field_padding`` than the teacher
used would silently turn the "amortisation gap" into a comparison of two
different benches.  For the same reason the evaluator rejects a checkpoint whose
bench disagrees with the cache instead of quietly running it.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from ml.sim2f.config import Sim2fConfig, SynthConfig, TeacherConfig

__all__ = ["main"]

_COMMANDS = ("synth", "train", "eval", "loop")


def _sim_overrides(sim_cfg: Sim2fConfig, args: argparse.Namespace) -> Sim2fConfig:
    """Apply the ``--grid`` / ``--ccd-px`` / ``--far-field-padding`` overrides.

    ``dataclasses.replace`` re-runs ``__post_init__``, so an illegal combination
    (e.g. ``--far-field-padding`` zero) fails here with the dataclass's own
    message rather than deep inside the generator.

    Args:
        sim_cfg: Base configuration (the dataclass defaults).
        args: Parsed arguments; a ``None`` value means "not given".

    Returns:
        The configuration with the given overrides applied.
    """
    candidates = {"grid": args.grid, "ccd_px": args.ccd_px,
                  "far_field_padding": args.far_field_padding}
    known = {item.name for item in fields(Sim2fConfig)}
    overrides = {
        key: value
        for key, value in candidates.items()
        if value is not None and key in known
    }
    return replace(sim_cfg, **overrides) if overrides else sim_cfg


def _synth_parser() -> argparse.ArgumentParser:
    """Build the ``synth`` parser, every default read from a dataclass.

    Returns:
        The configured parser.
    """
    synth = SynthConfig()
    teacher = TeacherConfig()
    parser = argparse.ArgumentParser(
        prog="python -m ml.sim2f synth",
        description=(
            "Draw target/teacher pairs on the 2f twin and write the corpus cache "
            "(pairs.npz + meta.json)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default="data/sim2f", help="output directory for the cache"
    )
    parser.add_argument("--count", type=int, default=synth.count)
    parser.add_argument("--seed", type=int, default=synth.seed)
    parser.add_argument(
        "--kinds",
        default=",".join(synth.target_kinds),
        help=(
            "comma-separated target families; the on-disk `kind` index follows "
            f"{list(synth.target_kinds)}"
        ),
    )
    parser.add_argument(
        "--size-frac",
        type=float,
        nargs=2,
        default=list(synth.size_frac),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--train-frac", type=float, default=synth.train_frac)
    parser.add_argument("--val-frac", type=float, default=synth.val_frac)
    parser.add_argument("--batch-size", type=int, default=synth.batch_size)

    parser.add_argument("--grid", type=int, default=None, help="pupil grid side")
    parser.add_argument("--ccd-px", type=int, default=None, help="camera window side")
    parser.add_argument(
        "--far-field-padding",
        type=int,
        default=None,
        help="focal-plane FFT zero-padding factor (the angular-scale knob)",
    )
    parser.add_argument("--w-ee", type=float, default=teacher.w_ee)
    parser.add_argument("--iters", type=int, default=teacher.iters)
    parser.add_argument("--lr", type=float, default=teacher.lr)
    parser.add_argument("--metric", choices=("l1", "mse"), default=teacher.metric)
    parser.add_argument(
        "--sim-field",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "extra Sim2fConfig field override (repeatable), e.g. "
            "--sim-field phase_blur_px=0.0. Needed because the twin-error block "
            "is wider than the three geometry flags above and must be driven "
            "from the dataclass rather than hard-coded here."
        ),
    )
    return parser


def _parse_sim_fields(values: list[str], sim_cfg: Sim2fConfig) -> Sim2fConfig:
    """Apply ``--sim-field NAME=VALUE`` overrides with dataclass type coercion.

    Args:
        values: ``["name=value", ...]`` as parsed from the CLI.
        sim_cfg: Base configuration.

    Returns:
        The overridden configuration.

    Raises:
        ValueError: If a name is not a ``Sim2fConfig`` field, or the value cannot
            be coerced to that field's type (reported through
            :class:`argparse.ArgumentTypeError` machinery so the CLI exits 2).
    """
    if not values:
        return sim_cfg
    known = {item.name: item for item in fields(Sim2fConfig)}
    overrides: dict[str, Any] = {}
    for item in values:
        name, separator, raw = str(item).partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"--sim-field expects NAME=VALUE, got {item!r}")
        if name not in known:
            raise ValueError(
                f"--sim-field {name!r} is not a Sim2fConfig field; known: "
                f"{sorted(known)}"
            )
        current = getattr(sim_cfg, name)
        if isinstance(current, bool):
            parsed: Any = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            parsed = int(raw)
        elif isinstance(current, float):
            parsed = float(raw)
        else:
            parsed = raw.strip()
        overrides[name] = parsed
    return replace(sim_cfg, **overrides)


def _call_build_pairs(
    builder: Callable[..., Any],
    *,
    sim_cfg: Sim2fConfig,
    teacher_cfg: TeacherConfig,
    synth_cfg: SynthConfig,
    out_dir: Path,
) -> Any:
    """Call ``synth.build_pairs`` by name, tolerating its parameter names.

    ``synth`` is written by a parallel change, so this adapter passes the four
    values it owns and lets the callee's own signature decide the keywords.  It
    is deliberately not a bare positional call: a positional call would bind
    silently to the wrong parameter if ``build_pairs`` orders them differently,
    and that failure would show up as a corpus generated with the wrong teacher
    settings rather than as an error.

    Args:
        builder: ``ml.sim2f.synth.build_pairs``.
        sim_cfg: Bench geometry.
        teacher_cfg: Teacher objective settings.
        synth_cfg: Corpus drawing settings.
        out_dir: Destination directory.

    Returns:
        Whatever ``build_pairs`` returns (the cache directory, by convention).

    Raises:
        TypeError: If a required parameter of ``build_pairs`` has no counterpart
            here, which is a contract mismatch worth failing loudly on.
    """
    signature = inspect.signature(builder)
    candidates = {
        "sim_cfg": sim_cfg,
        "sim_config": sim_cfg,
        "sim": sim_cfg,
        "teacher_cfg": teacher_cfg,
        "teacher_config": teacher_cfg,
        "teacher": teacher_cfg,
        "synth_cfg": synth_cfg,
        "synth_config": synth_cfg,
        "synth": synth_cfg,
        "config": synth_cfg,
        "cfg": synth_cfg,
        "out_dir": out_dir,
        "out": out_dir,
        "root": out_dir,
    }
    provided: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in candidates:
            provided[name] = candidates[name]
        elif parameter.default is parameter.empty:
            raise TypeError(
                f"build_pairs requires parameter {name!r}, which this CLI cannot "
                "supply; the synth contract changed"
            )
        # An unknown optional parameter keeps its own default: guessing a value
        # for it would be worse than leaving the callee's calibrated choice in
        # place.
    logger.debug("calling build_pairs with {}", sorted(provided))
    return builder(**provided)


def _synth_main(argv: list[str]) -> int:
    """Run the ``synth`` sub-command.

    Args:
        argv: Arguments after the sub-command name.

    Returns:
        Process exit code.
    """
    args = _synth_parser().parse_args(argv)
    try:
        sim_cfg = _parse_sim_fields(args.sim_field, _sim_overrides(Sim2fConfig(), args))
        kinds = tuple(part.strip() for part in str(args.kinds).split(",") if part.strip())
        synth_cfg = SynthConfig(
            count=args.count,
            seed=args.seed,
            target_kinds=kinds,
            size_frac=(float(args.size_frac[0]), float(args.size_frac[1])),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            batch_size=args.batch_size,
        )
        teacher_cfg = TeacherConfig(
            iters=args.iters, lr=args.lr, w_ee=args.w_ee, metric=args.metric
        )
    except ValueError as exc:
        logger.error("invalid configuration: {}", exc)
        return 2

    out_dir = Path(args.out)
    logger.info(
        "synth: {} pairs from {} kinds, grid={} ccd_px={} padding={} -> {}",
        synth_cfg.count, list(synth_cfg.target_kinds), sim_cfg.grid,
        sim_cfg.ccd_px, sim_cfg.far_field_padding, out_dir,
    )
    # Imported here so `--help` and the parser tests never pay for torch/numpy.
    from ml.sim2f.synth import build_pairs

    try:
        _call_build_pairs(
            build_pairs,
            sim_cfg=sim_cfg,
            teacher_cfg=teacher_cfg,
            synth_cfg=synth_cfg,
            out_dir=out_dir,
        )
    except (ImportError, TypeError, ValueError, FileNotFoundError) as exc:
        logger.error("synth failed: {}", exc)
        return 2
    return 0


def _eval_parser() -> argparse.ArgumentParser:
    """Build the ``eval`` parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.sim2f eval",
        description=(
            "Score a sim2f checkpoint on one split and write eval.json "
            "(flat / net / teacher rows)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="model_best.pt / model_last.pt")
    parser.add_argument(
        "--cache", required=True, help="corpus directory (must be the one it trained on)"
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--out",
        default=None,
        help="report path; defaults to eval.json next to the checkpoint",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help=(
            "score raw frames instead of background-subtracted ones; the "
            "ablation number, not comparable with a prepared teacher"
        ),
    )
    return parser


def _eval_main(argv: list[str]) -> int:
    """Run the ``eval`` sub-command.

    Args:
        argv: Arguments after the sub-command name.

    Returns:
        Process exit code.
    """
    args = _eval_parser().parse_args(argv)
    from ml.sim2f.evaluate import evaluate

    try:
        evaluate(
            args.checkpoint,
            cache_dir=args.cache,
            split=args.split,
            device=args.device,
            out=args.out,
            prepare=not args.no_prepare,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("eval failed: {}", exc)
        return 2
    return 0


def _print_help(stream: object = None) -> None:
    """Print the top-level usage summary.

    Args:
        stream: Destination, defaulting to ``sys.stdout`` (an explicit ``--help``
            is not an error, so it must be pipeable); the dispatch error path
            passes ``sys.stderr``.
    """
    print(
        "usage: python -m ml.sim2f {synth,train,eval,loop} ...\n\n"
        "  synth   draw target/teacher pairs on the 2f twin -> pairs.npz\n"
        "  train   fit the U-Net on a corpus (see `... train --help`)\n"
        "  eval    score a checkpoint, write eval.json (see `... eval --help`)\n"
        "  loop    five-arm closed-loop ablation (see `... loop --help`)\n",
        file=sys.stdout if stream is None else stream,
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a sub-command.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success, ``2`` on a usage or config error.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _print_help()
        return 2
    command, rest = arguments[0], arguments[1:]
    if command in ("-h", "--help", "help"):
        _print_help()
        return 0
    if command == "synth":
        return _synth_main(rest)
    if command == "train":
        # No duplicated flags: the trainer owns its parser and its defaults.
        from ml.sim2f.train import main as train_main

        return train_main(rest)
    if command in ("eval", "evaluate"):
        return _eval_main(rest)
    if command == "loop":
        # Same rule as `train`: one parser, owned by the module that runs it.
        from ml.sim2f.loop import main as loop_main

        return loop_main(rest)
    logger.error("unknown sub-command {!r}; expected one of {}", command, list(_COMMANDS))
    _print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
