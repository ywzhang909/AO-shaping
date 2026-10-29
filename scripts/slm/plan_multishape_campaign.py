#!/usr/bin/env python
"""Build the device-online campaign plan for the no-GS multishape matrix.

Why this script exists
----------------------

The 2026-10-28 review of ``report/shape_trials/multishape_scale/`` left one
structural problem that no amount of re-analysis can fix: the campaign's lanes
were written as free-form PowerShell, so

* a lane could silently drift from the protocol (the older
  ``plan_transfer_next_run.py`` emitted ``--initial-phase gs`` while the stated
  goal was "no GS at all");
* **严格冷启动** (plane / random, no analytic structure) and **解析初始化**
  (back-projection / energy mapping, *does* use the forward model) ended up in
  the same table, which lets an analytic seed be read as "cold start worked";
* nothing allocated the repeat budget, so a "win" was a single measurement
  against a 3-repeat spread and every conclusion rested on one frame;
* the preflight that gates a whole session (``probe_optical_path_validity.py``)
  was a line an operator could forget.

This script turns all of that into data: one ``plan.json`` plus one ready-to-run
lane per condition, each lane carrying the preflight, the budget, the required
result classes and the phase provenance. It writes files and opens no device.

Two routes that must never be merged
------------------------------------

``strict_cold``  -- initial phase = flat plane or seeded random on the carrier.
                   No forward model, no target, no back-projection. This is the
                   only route that can answer "does the closed loop work from
                   nothing".
``analytic_init``-- initial phase = back-projection of the target through the
                   fitted optics (or an energy-mapping seed). Uses the fitted
                   physical model even though it never calls GS.

Both may consume the same **fitted model**; the model is not the initial phase.
Every emitted lane names its route in ``--initial-phase`` and in the plan, so a
report cannot pool them without saying so.

The reuse of that model is conditional
--------------------------------------

A checkpoint carries the radiometry of the bench it was fitted on (exposure
response gain, background, the panel's response at that optical power), so
reusing one across a laser or attenuator change rescales every predicted
intensity while nothing fails loudly. Lane 00 therefore *fits the model on the
current bench* (it omits ``--model-dir`` and writes the directory lanes 01+ point
at), and ``model_power_check`` in the plan records what must be compared before
an old-power number is quoted as a baseline. ``--no-refit`` exists only for the
case where the model dir was fitted in this same bench session.

Usage
-----

    python scripts/slm/plan_multishape_campaign.py                    # write plan.json + lanes
    python scripts/slm/plan_multishape_campaign.py --check            # validate only
    python scripts/slm/plan_multishape_campaign.py --session s600 --sizes 600
    python scripts/slm/plan_multishape_campaign.py --power-note "laser output changed since s1200"
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # allow `python scripts/...` from a bare checkout
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.transfer.protocol import plan_reference_blocks

CAMPAIGN_ROOT = ROOT / "data" / "debug" / "slm_transfer"
RUNNER = "src/ao_shaping/main.py"
PROBE = "scripts/slm/probe_optical_path_validity.py"

#: Placeholder the *plan* carries for the campaign root, so "plan.json stays
#: portable: it can be copied to another machine whose data/ lives elsewhere.
#: The rendered .ps1 must not carry it -- a lane is executed, not read, and an
#: unexpanded token makes --model-dir a literal "<campaign>/..." directory.
CAMPAIGN_TOKEN = "<campaign>"

#: Sizes the campaign measures, in native CCD pixels, and the camera window each
#: needs. The window must be wider than the target and the zero-order disk: the
#: 200 px box overlaps the zero-order disk by 5.61 % (measured), which is why the
#: aperture metrics exclude it rather than shrinking the window.
SIZE_WINDOWS = {200: 420.0, 600: 620.0, 1200: 900.0}

#: Initial-phase provenance per route. ``initial_phase`` is the CLI value;
#: ``gs_calls`` is what the route is allowed to spend (0 for both).
ROUTES = {
    "strict_cold": dict(initial_phase="random", gs_calls=0,
                        description="flat plane / seeded random on the carrier; no forward model, no target"),
    "analytic_init": dict(initial_phase="backprojection", gs_calls=0,
                          description="target back-projected through the fitted optics; no GS, but the model is used"),
}

#: The three result classes every condition must report, and where each comes
#: from. Written into every lane so a run cannot produce "CV only" evidence.
REQUIRED_RESULTS = {
    "fixed_core_cv": "protocol.aperture_metrics: frozen sensor-pixel aperture, not the stage's eroded core",
    "shape_agreement": "protocol.shape_agreement: IoU + boundary mean/p95 + coverage of the target mask",
    "relative_energy": "protocol.aperture_metrics.ee_rel: in-box / whole-frame, same exposure",
}

#: Auxiliary read-outs the review made mandatory (saturation, provenance, model).
REQUIRED_AUDIT = ("zero_excluded_fraction", "saturated_fraction", "phase_provenance",
                  "model_checkpoint_sha256", "exposure_readback_ms")

DEFAULT_MINUTES_PER_BLOCK = 0.4  # measured: ~4 captures x ~6 s with settle


def plan_refit_lane(*, sizes: list[int], shapes: list[str], model_session: str,
                    campaign_dir: Path) -> dict:
    """Lane 00: fit the model on the *current* bench before any lane reuses one.

    A fitted checkpoint carries the radiometry of the bench it was fitted on --
    exposure response gain, background and the panel's response at that optical
    power. Reusing a checkpoint across a laser or attenuator change silently
    rescales every predicted intensity, and nothing in the reused run fails
    loudly: the shaping loop just chases a model that no longer describes the
    bench. So the reuse is made conditional on a fit at the current power,
    written into the same directory the other lanes point ``--model-dir`` at
    (which is why this lane omits ``--model-dir`` and thereby takes the
    from-zero branch).

    The lane also drags one shape through the four stages at the current power,
    so the session ends with at least one anchor that is *not* a replay of an
    old-power number.
    """
    size = max(sizes)
    return dict(
        session=f"{model_session}_refit",
        size=size,
        shape=shapes[0],
        route="refit",
        model_session=None,
        reuse_session=model_session,
        out=str(Path(campaign_dir) / model_session),
        window_span=SIZE_WINDOWS.get(size),
        initial_phase=ROUTES["strict_cold"]["initial_phase"],
        gs_calls_allowed=0,
        description=("fit the physical model on the current bench (no --model-dir); "
                     "lanes 01+ reuse this directory"),
        required_results=dict(REQUIRED_RESULTS),
        required_audit=list(REQUIRED_AUDIT),
    )


def plan_conditions(*, sizes: list[int], shapes: list[str], route: str,
                    model_session: str, campaign_dir: Path) -> list[dict]:
    """One lane per (size, shape): the same recipe, separated by size.

    Size is kept as a whole lane rather than a loop inside a lane because a size
    change also changes the camera window, the zero-order overlap and therefore
    the aperture: re-configuring the camera mid-lane is how the earlier session
    ended up with frames it could not compare.
    """
    lanes = []
    for size in sizes:
        for shape in shapes:
            lanes.append(dict(
                session=f"{model_session}_{route}_s{size}_{shape}",
                size=size, shape=shape, route=route,
                model_session=model_session,
                out=str(campaign_dir / f"{model_session}_{route}_s{size}" / shape),
                window_span=SIZE_WINDOWS.get(size),
                initial_phase=ROUTES[route]["initial_phase"],
                gs_calls_allowed=ROUTES[route]["gs_calls"],
                required_results=dict(REQUIRED_RESULTS),
                required_audit=list(REQUIRED_AUDIT),
            ))

    return lanes

def render_lane(lane: dict, *, frames: int,
                replay_exposures: list[float] | None) -> list[str]:
    """The exact command lines for one lane: preflight, then the runner.

    The preflight is a separate line whose exit code ends the lane, because the
    probe's verdict is the gate -- a nonzero exit means "do not spend the
    session", and a runner flag cannot express that.

    ``replay_exposures`` defaults to ``None``, which asks the runner to derive the
    ladder from the lane's own measured peaks. A ladder fixed across lanes is
    wrong: the 2026-10-29 lane 01 aborted because 125 ms drove a 200 px core to
    full scale while a 1200 px core was still clean at 225 ms.
    """
    preflight = f'python {PROBE} --session {lane["session"]}'
    if replay_exposures is None:
        replay = ["--replay-exposures-auto"]
    else:
        # click's ``multiple=True`` takes the flag once per value; a
        # space-separated list is parsed as extra positional arguments.
        replay = [part for value in replay_exposures
                  for part in ("--replay-exposures", str(value))]
    runner = [
        "python", RUNNER, "slm-transfer-shapes",
        "--out", lane["out"],
        "--shapes", lane["shape"],
        "--side", str(lane["size"]),
        "--window-span", str(lane["window_span"]),
        # No model dir = the from-zero branch. That is the whole point of the
        # refit lane: a lane that reuses a checkpoint must be preceded by a fit
        # on the current bench, or the reuse is silent.
        *(["--model-dir", f"{CAMPAIGN_TOKEN}/{lane['model_session']}"]
          if lane.get("model_session") else []),
        "--initial-phase", lane["initial_phase"],
        "--no-gs-seed",
        *replay,
        "--record-responses",
        "--frames", str(frames),
        "--execute",
    ]
    return [preflight, 'if ($LASTEXITCODE -ne 0) { Write-Error "preflight refused"; exit 1 }',
            " ".join(shlex.quote(part) for part in runner)]


def expand_campaign_token(commands: list[str], campaign_dir: Path) -> list[str]:
    """Resolve the plan's portable campaign token into a runnable absolute path.

    The plan keeps ``<campaign>`` so it survives being copied to a machine whose
    ``data/`` sits elsewhere; a lane script is executed, not copied, so it must
    carry the real root. Every lane but the refit has a ``--model-dir`` that
    would otherwise point at a literal ``<campaign>`` directory.

    Args:
        commands: Lane commands as stored in ``plan.json``.
        campaign_dir: The campaign root to substitute.

    Returns:
        The same commands with the token replaced by ``str(campaign_dir)``.
    """
    root = str(campaign_dir)
    return [part.replace(CAMPAIGN_TOKEN, root) for part in commands]


def build_plan(*, sizes: list[int], shapes: list[str], routes: list[str],
               model_session: str, campaign_dir: Path, frames: int,
               replay_exposures: list[float] | None, repeats: int,
               power_note: str = "") -> dict:
    """The whole campaign: lanes, ordering, repeat budget and phase provenance."""
    lanes = [plan_refit_lane(sizes=sizes, shapes=shapes, model_session=model_session,
                             campaign_dir=campaign_dir)]
    for route in routes:
        lanes.extend(plan_conditions(sizes=sizes, shapes=shapes, route=route,
                                     model_session=model_session, campaign_dir=campaign_dir))
    # The refit lane stays at index 0 no matter how the rest sorts: every other
    # lane's ``--model-dir`` points at the directory it writes.
    ordered = [lanes[0]]
    rest = lanes[1:]
    # Stage-major, size-major: every lane of one size and one route runs back to
    # back so the camera window and the exposure land once, and a mid-campaign
    # abort costs the rest of that size rather than scattering half-finished work
    # across three windows.
    rest.sort(key=lambda lane: (lane["route"], lane["size"], shapes.index(lane["shape"])))
    ordered.extend(rest)
    for index, lane in enumerate(ordered):
        lane["order_index"] = index
        lane["commands"] = render_lane(lane, frames=frames, replay_exposures=replay_exposures)
        lane["plan_reference_blocks"] = plan_reference_blocks(
            [dict(name=f'ref_s{lane["size"]}', block_seconds=DEFAULT_MINUTES_PER_BLOCK * 60,
                  planned_repeats=repeats)], block_seconds=DEFAULT_MINUTES_PER_BLOCK * 60)
    return dict(
        campaign_dir=str(campaign_dir), model_session=model_session,
        routes={name: ROUTES[name] for name in routes},
        sizes=list(sizes), shapes=list(shapes), frames=frames,
        replay_exposures=("auto" if replay_exposures is None else list(replay_exposures)),
        repeats_per_condition=repeats,
        required_results=dict(REQUIRED_RESULTS),
        required_audit=list(REQUIRED_AUDIT),
        model_power_check=dict(
            refit_lane=ordered[0]["session"],
            reuse_session=model_session,
            note=power_note or (
                "state what changed on the bench (laser power, attenuator, coupling) "
                "and whether the reused checkpoint was fitted before or after it"),
            compare=[
                "fit/result.json holdout error of the refit dir vs the reused dir, "
                "on the same held-out phases",
                "the preflight's recommended exposure vs the exposure the reused capture was taken at",
            ],
            rule=("lanes 01+ may only cite the reused checkpoint once the refit's holdout "
                  "error on the current bench is compared with it; if the refit wins, the "
                  "campaign's model is the refit and every old-power number is a different "
                  "optical power, not a baseline"),
        ),
        phases=dict(
            offline_first=[
                "python scripts/slm/audit_replay_blocks.py --session <previous> --json <out>",
                "read the clipped/withheld list before trusting any replay comparison",
            ],
            device=[
                "python scripts/slm/probe_optical_path_validity.py  (gate; exit code is the verdict)",
                "run lanes in order_index order; each lane repeats its preflight",
                "on any abort: relaunch the same lane with --resume, never re-measure finished stages",
            ],
            offline_last=[
                "python scripts/slm/audit_replay_blocks.py --session <new> --json <out>",
                "compare only within one (exposure, bit depth, shape, route) condition",
            ]),
        lanes=ordered)


def validate_plan(plan: dict) -> list[str]:
    """Reject a plan that would break the protocol. Returns the problems found.

    These are the mistakes that actually happened: a GS lane, a lane without the
    preflight, two routes pooled into one session name, a missing repeat budget,
    and a missing result class. The last block is the 2026-10-29 one: a campaign
    that reused a checkpoint fitted at a different optical power.
    """
    problems = []
    seen_sessions = set()
    refit_lanes = [lane for lane in plan["lanes"] if lane.get("route") == "refit"]
    for lane in plan["lanes"]:
        if lane["session"] in seen_sessions:
            problems.append(f'duplicate session name {lane["session"]}')
        seen_sessions.add(lane["session"])
        if lane["gs_calls_allowed"] != 0 or "gs" in lane["initial_phase"].split("_"):
            problems.append(f'{lane["session"]}: a no-GS campaign lane may not use GS')
        if lane.get("route") != "refit" and lane["route"] not in plan["routes"]:
            problems.append(f'{lane["session"]}: unknown route {lane["route"]}')
        if not lane["commands"] or PROBE not in lane["commands"][0]:
            problems.append(f'{lane["session"]}: preflight missing')
        if set(lane["required_results"]) != set(REQUIRED_RESULTS):
            problems.append(f'{lane["session"]}: result classes incomplete')
        if lane["plan_reference_blocks"]["min_repeats_for_p005"] > plan["repeats_per_condition"]:
            problems.append(
                f'{lane["session"]}: {plan["repeats_per_condition"]} repeats cannot reach p<0.05 '
                f'(needs {lane["plan_reference_blocks"]["min_repeats_for_p005"]}) -- '
                "report an effect size, not a verdict")
        if lane["window_span"] is None:
            problems.append(f'{lane["session"]}: unknown target size {lane["size"]}')

    # -- the model may not be reused across an optical-power change -----------
    reused = [lane for lane in plan["lanes"] if lane.get("model_session")]
    if len(refit_lanes) > 1:
        problems.append(f'{len(refit_lanes)} refit lanes: exactly one model fit per campaign')
    if refit_lanes:
        refit = refit_lanes[0]
        if refit["order_index"] != 0:
            problems.append(f'{refit["session"]}: the refit must run first, it writes the model dir others reuse')
        if "--model-dir" in " ".join(refit["commands"]):
            problems.append(f'{refit["session"]}: the refit lane must not reuse a model dir')
        expected = str(Path(plan["campaign_dir"]) / plan["model_session"])
        if Path(refit["out"]) != Path(expected):
            problems.append(f'{refit["session"]}: the refit must write {expected}, which is what lanes reuse')
        if (plan.get("model_power_check") or {}).get("refit_lane") != refit["session"]:
            problems.append("model_power_check does not name the refit lane")
    elif reused:
        problems.append("lanes reuse a model dir but no refit lane establishes it on the current bench")
    for lane in reused:
        if f'<campaign>/{plan["model_session"]}' not in " ".join(lane["commands"]):
            problems.append(f'{lane["session"]}: --model-dir is not the campaign model session')
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path,
                        default=CAMPAIGN_ROOT / "multishape_nogs_20261029")
    parser.add_argument("--model-session", default="s1200",
                        help="campaign session whose fitting run is reused (model-dir)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 600, 1200])
    parser.add_argument("--shapes", nargs="+", default=["circle", "square", "triangle", "star"])
    parser.add_argument("--routes", nargs="+", default=["strict_cold", "analytic_init"],
                        choices=sorted(ROUTES))
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--replay-exposures", type=float, nargs="+", default=None,
                        help="explicit replay ladder (ms); default: let each lane derive it from "
                             "its own measured core peak -- a fixed ladder cannot serve both "
                             "200 px and 1200 px")
    parser.add_argument("--repeats", type=int, default=6,
                        help="repeats per condition (6 = the fewest that can reach p<0.05)")
    parser.add_argument("--power-note", default="",
                        help="what changed on the bench since the reused model was fitted "
                             "(recorded in the plan; the refit lane is the check, this is the memo)")
    parser.add_argument("--check", action="store_true", help="validate and print, write nothing")
    args = parser.parse_args(argv)

    plan = build_plan(sizes=args.sizes, shapes=args.shapes, routes=args.routes,
                      model_session=args.model_session, campaign_dir=args.campaign_dir,
                      frames=args.frames, replay_exposures=args.replay_exposures,
                      repeats=args.repeats, power_note=args.power_note)
    problems = validate_plan(plan)
    text = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.check:
        print(text)
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        return 1 if problems else 0
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    (args.campaign_dir / "plan.json").write_text(text, encoding="utf-8")
    lanes_dir = args.campaign_dir / "lanes"
    lanes_dir.mkdir(exist_ok=True)
    for lane in plan["lanes"]:
        body = "\n".join([f'# {lane["session"]}  route={lane["route"]}  '
                          f'shape={lane["shape"]}  size={lane["size"]} px',
                          *expand_campaign_token(lane["commands"], args.campaign_dir), ""])
        (lanes_dir / f'{lane["order_index"]:02d}_{lane["session"]}.ps1').write_text(
            body, encoding="utf-8")
    print(f'wrote {args.campaign_dir / "plan.json"} and {len(plan["lanes"])} lanes')
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
