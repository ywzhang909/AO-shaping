"""Shared click parameter groups for the ``tools/slm`` bench probes.

Why this module exists
----------------------
Every probe in this package drives the *same* two pieces of hardware over the
*same* two kinds of knob, and each one used to re-declare them by hand:

* which device to open — SLM index / wavelength, camera backend / id / exposure;
* how to read a frame — frame count, discard count, and the three settle-criterion
  numbers that :func:`~ao_shaping.tools.slm.slm_bench_probe.display_and_average`
  needs to tell an *unsettled* frame from a noisy one.

Those knobs are not cosmetic. ``settle_s`` / ``stable_tol`` / ``max_wait_s``
encode the measurement in :mod:`ao_shaping.tools.slm.slm_bench_probe`'s module
docstring (waiting a fixed duration instead of waiting for *stability* recorded a
3.3x slope error), so a probe that re-declares them with different defaults is a
silent correctness hazard, not just duplication. Lifting them here makes the
defaults single-sourced.

The two groups
--------------
:class:`SlmBenchParams` is the **device/session** role (what to open);
:class:`SlmAcquireParams` is the **acquisition** role (how to read a frame).
They are separate classes because a consumer may need one without the other —
an offline renderer still needs the acquisition numbers but opens nothing — and
because keeping them contiguous lets a parent splice them into its own option
list *at the position the original hand-written declarations occupied*, so
migrating a probe does not silently reorder its ``--help``.

Compose them with :class:`~ao_shaping.utils.cli.params.ClickGroup`::

    @dataclass
    class MyParams:
        out: Annotated[str, option("--out")] = "data/out"
        bench: Annotated[SlmBenchParams, ClickGroup()] = field(
            default_factory=SlmBenchParams
        )
        pupil_center: Annotated[
            str | tuple[float, float], option("--pupil-center", callback=parse_tuple)
        ] = "960,600"
        acquire: Annotated[SlmAcquireParams, ClickGroup()] = field(
            default_factory=SlmAcquireParams
        )

Layering invariant (do not relax)
---------------------------------
**This module MUST NOT import ``ao_shaping.runners``, directly or indirectly.**
``runners/slm/zernike_matrix_runner.py`` imports ``ao_shaping.tools.slm.*`` at
module level, so any ``runners`` edge here would close the cycle
``tools.slm.params -> runners.runner_common -> utils.cli.params`` and make the
parameter plumbing (which is deliberately pure metadata) depend on hardware
orchestration. It may import ``click`` and the ``utils`` leaf layers only; a
regression test enforces this by AST.

Deliberately absent: a ``seed`` field
-------------------------------------
Several probes (``slm_panel_locate``, ``slm_beam_extent``, ...) take
``--seed``, but this sweep does not — its point list is fully deterministic, so
a seed would be a flag that provably cannot change the output. Adding it here
would also widen the probe's frozen 21-option surface (``TODO.md`` R5) by one
flag that no consumer could exercise. A probe that genuinely needs randomness
should declare its own ``seed`` field next to its other options rather than
forcing it onto every consumer of these groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import click

from ao_shaping.utils.cli.params import option

#: Camera backends the bench probes can open.
CAM_TYPE_CHOICES: tuple[str, ...] = ("daheng", "miicam")


@dataclass
class SlmBenchParams:
    """Device/session knobs shared by the SLM bench probes.

    Which panel to drive and which camera to read through. The exposure time is
    part of this group rather than the acquisition group because it is applied
    by the *camera* (``reset_exposure_time``) before any frame is counted, and
    two different exposures silently make two runs incomparable.
    """

    slm_number: Annotated[int, option("--slm-number")] = 1
    slm_wavelength: Annotated[int, option("--slm-wavelength")] = 1064
    cam_type: Annotated[
        str, option("--cam-type", type=click.Choice(CAM_TYPE_CHOICES))
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id")] = 0
    exposure_ms: Annotated[float, option("--exposure-ms")] = 3.0


@dataclass
class SlmAcquireParams:
    """Frame-reading and settle-criterion knobs shared by the bench probes.

    ``settle_s`` / ``stable_tol`` / ``max_wait_s`` are the settle criterion, not
    a sleep: the first is the *initial* wait, the second the relative agreement
    required between two consecutive readings, the third the ceiling on waiting.
    See :func:`~ao_shaping.tools.slm.slm_bench_probe.display_and_average`.
    """

    frames: Annotated[int, option("--frames")] = 4
    discard: Annotated[int, option("--discard")] = 3
    settle_s: Annotated[float, option("--settle-s")] = 0.5
    stable_tol: Annotated[float, option("--stable-tol")] = 0.02
    max_wait_s: Annotated[float, option("--max-wait-s")] = 6.0