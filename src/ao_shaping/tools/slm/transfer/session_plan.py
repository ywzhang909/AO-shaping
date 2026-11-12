"""Device-free session planning, resume bookkeeping and model-only acceptance.

These helpers back ``validate.run`` but never open a device and never import
torch: ``session_plan`` prices what a session still has to measure,
``_frozen_groups`` counts the block groups a stage directory already holds,
``skips_shape`` gates a resume that has nothing left, and ``model_only_decision``
accepts on the model's own loss when the A–B–B–A CCD verdict is switched off.

They live apart from ``validate`` so the device-execution module stays under the
module-size gate (``scripts/check_module_size.py``); ``validate`` re-imports the
names its CLI and callers still reach by the old ``transfer.validate`` path.
"""
from __future__ import annotations

import json
from pathlib import Path


#: Order-of-magnitude device-time model used only to pick a run order.
#: 2026-10-28 post-mortem of ``multishape_scale_20261028``: a full-voltage CCD
#: read costs ~0.9 s at this session's frame count, and the fixed ``time.sleep``
#: before a capture is 0.6 s (the settle inside ``validate.run``). Both were read
#: off measured sessions, not benchmarked per camera: they are an *estimate*, so
#: a caller can rank "one group now" against "one group later", nothing more.
SETTLE_SECONDS = 0.6
READ_SECONDS_PER_FRAME = 0.9


def _group_tuple(bit_depth, window_step, groups) -> tuple[int, float, int]:
    """Coerce one ``(bit_depth, window_step, groups)`` triple, rejecting nonsense.

    A plan is read by an operator before a bench session, so a typo in a block
    count must fail here with the offending value rather than silently plan a
    zero-capture stage.
    """
    depth, step, count = int(bit_depth), float(window_step), int(groups)
    if depth not in (8, 12):
        raise ValueError(f"plan bit depth must be 8 or 12, got {bit_depth!r}")
    if step <= 0:
        raise ValueError(f"plan window step must be positive, got {window_step!r}")
    if count < 0:
        raise ValueError(f"plan group count must be nonnegative, got {groups!r}")
    return depth, step, count


def _frozen_groups(directory: Path) -> int:
    """How many CCD blocks a stage directory already holds.

    The block groups file is authoritative when present; an older directory
    without it still reports what ``summary.json`` recorded. The file is
    *recorded* data, so a shape mismatch is a broken archive and raises
    :class:`ValueError` instead of an ``IndexError`` from a half-read row.

    A group is one ``(shape, block)`` pair: several shapes share one summary, and
    each block is what the stage's capture loop iterates over.
    """
    groups_path = directory / f"{directory.name}_groups.json"
    summary_path = directory / "summary.json"
    if groups_path.is_file():
        try:
            payload = json.loads(groups_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable block-groups file {groups_path}: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("groups")
        if not isinstance(payload, list):
            raise ValueError(f"block-groups file {groups_path} must hold a list of groups")
        return len(payload)
    if not summary_path.is_file():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable summary {summary_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"summary {summary_path} must be a JSON list of block entries")
    blocks: set[tuple[str, int]] = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or "shape" not in entry or "block" not in entry:
            raise ValueError(f"summary {summary_path} entry {index} lacks shape/block keys")
        blocks.add((str(entry["shape"]), int(entry["block"])))
    return len(blocks)


def remaining_blocks(directory: Path, blocks: int, *, resume: bool) -> list[int]:
    """Return this shape directory's outstanding block indices.

    A stage may contain several shapes, but the runner invokes ``validate``
    with one shape-specific directory at a time. The checkpoint offset is
    therefore local to ``directory``; it must never be shared across shapes.
    """
    recorded = _frozen_groups(directory) if resume else 0
    return list(range(min(recorded, int(blocks)), int(blocks)))


def stage_complete(directory: Path, blocks: int) -> bool:
    """Whether a shape stage has capture evidence and every block verdict."""
    return ((directory / "manifest.json").is_file()
            and _frozen_groups(directory) >= int(blocks))


def skips_shape(*, resume: bool, blocks) -> bool:
    """Whether this shape has nothing left to do -- which is a *resume-only* state.

    An empty block list is legitimate without ``resume``: that is how the seed
    bake-off asks for a seed-only pass (``blocks=0, updates=0``), and the run
    still measures the seed and final rows. Gating on the empty list alone (as
    the resume gate in ``c1c7aeed`` did) made that pass write no
    ``manifest.json`` at all, so every square/triangle/star lane died reading a
    file that was never created.
    """
    return bool(resume) and not list(blocks)


def session_plan(args) -> dict:
    """Work out, without opening a device, what a session would still have to measure.

    A session is one long process, so an abort in its middle (a saturation guard,
    a crash) leaves finished stage directories behind that today cannot be
    re-entered at all: the run root is created with ``exist_ok=False``. This
    helper answers, before any device opens, which ``(shape, bit_depth,
    window_step, groups)`` tuples are already on disk and which are still needed,
    so the operator can plan the order -- the stage whose predecessor is already
    on disk first, the stage that needs the widest re-measurement last.

    A shape with a stage directory that already holds CCD blocks lands in
    ``frozen`` as ``(shape, bit_depth, window_step, groups)``, where the bit
    depth / window step are the *last* requested tuple's: on resume an earlier
    tuple is finished, so only the unfinished one is relevant, and its ``groups``
    is where the stage left off. Everything else lands in ``todo``.

    ``args.session_groups`` (optional) is a list of ``dict(bit_depth=...,
    window_step=..., groups=...)`` overrides; without it every shape is planned
    at the session's own ``bit_depth`` / ``window_step`` / ``blocks``, which is
    the single-tuple behaviour the caller had before.

    ``estimated_seconds`` is an *order of magnitude* (``groups * (0.6 + frames *
    0.9)``, see ``SETTLE_SECONDS``): it exists to choose a run order, never to
    promise wall clock, and it deliberately ignores the optimizer's CPU time.
    """
    out = Path(args.out)
    overrides = getattr(args, "session_groups", None)
    if overrides:
        wanted = [_group_tuple(g["bit_depth"], g["window_step"], g["groups"]) for g in overrides]
    else:
        wanted = [_group_tuple(getattr(args, "bit_depth", 8), getattr(args, "window_step", 2.),
                               getattr(args, "blocks", 1))]
    frozen, todo = [], []
    for shape in args.shapes:
        directory = out / str(shape)
        groups = _frozen_groups(directory) if directory.is_dir() else None
        if not groups:
            todo.extend((str(shape), *triple) for triple in wanted)
            continue
        frozen.append((str(shape), *wanted[-1][:2], groups))
        if len(wanted) > 1:
            todo.extend((str(shape), *triple) for triple in wanted[1:])
    estimated = sum(groups for *_, groups in todo) * (SETTLE_SECONDS
                                                      + float(getattr(args, "frames", 1))*READ_SECONDS_PER_FRAME)
    return dict(shapes=[str(shape) for shape in args.shapes], frozen=frozen, todo=todo,
                estimated_seconds=round(estimated, 1))


def model_only_decision(history) -> dict:
    """Accept a block on the model's own loss, with no CCD verdict measured.

    The A–B–B–A blocks exist to let the *measurement* decide. When they are
    switched off (``abba=False``) the only evidence left is the optimizer's loss
    history, so the rule is deliberately weak and explicit: accept when the
    block's last update beat the block's **first** one — i.e. "did this block
    improve on the phase it started from".

    Why not the last two updates (the previous rule): ``history`` is one block's
    trace, the step cap is applied to the increment, so once ``lr`` is large
    enough to saturate ``step_rms_cap`` (deployed default: ``lr=0.30``) every
    update after the first 2–3 lands on the same phase and the loss just dithers
    on the last digit. Comparing the final two entries then reports a coin flip:
    measured on the frozen checkpoint, ``lr=0.30``/``updates=10`` rejected all six
    blocks (0.256636 vs 0.256611, a 2.5e-5 uptick) and the loop re-ran the
    identical block forever, CV frozen at 0.5066. Block-end vs block-start is the
    block-level form of the grid probe's ``gated_oracle`` arm and accepts 6/6 on
    the same data (CV 0.648 → 0.089).

    With a single update there is no first value to compare, hence no evidence,
    hence no acceptance. Keys keep their historical names (``previous``/``last``)
    because the GUI and the recordings read them; ``previous`` is now the block's
    first loss, not the second-to-last.

    The returned dict carries ``acceptance="model_only"`` so a summary entry can
    never be mistaken for a paired CCD acceptance, and the two loss values so an
    audit can recompute ``accepted`` without a device.
    """
    losses = [float(entry["loss"]) for entry in history]
    previous = losses[0] if len(losses) >= 2 else None
    last = losses[-1] if losses else None
    accepted = previous is not None and last is not None and last < previous
    return dict(acceptance="model_only", accept_metric="model", accepted=bool(accepted),
                model_loss_previous=previous, model_loss_last=last)
