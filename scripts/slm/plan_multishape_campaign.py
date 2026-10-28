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

Usage
-----

    python scripts/slm/plan_multishape_campaign.py                    # write plan.json + lanes
    python scripts/slm/plan_multishape_campaign.py --check            # validate only
    python scripts/slm/plan_multishape_campaign.py --session s600 --sizes 600
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

def render_lane(lane: dict, *, frames: int, replay_exposures: list[float]) -> list[str]:
    """The exact command lines for one lane: preflight, then the runner.

    The preflight is a separate line whose exit code ends the lane, because the
    probe's verdict is the gate -- a nonzero exit means "do not spend the
    session", and a runner flag cannot express that.
    """
    preflight = f'python {PROBE} --session {lane["session"]}'
    # click's ``multiple=True`` takes the flag once per value; a space-separated
    # list is parsed as extra positional arguments, so expand the flag here.
    replay = [part for value in replay_exposures
              for part in ("--replay-exposures", str(value))]
    runner = [
        "python", RUNNER, "slm-transfer-shapes",
        "--out", lane["out"],
        "--shapes", lane["shape"],
        "--side", str(lane["size"]),
        "--window-span", str(lane["window_span"]),
        "--model-dir", f'<campaign>/{lane["model_session"]}',
        "--initial-phase", lane["initial_phase"],
        "--no-gs-seed",
        *replay,
        "--record-responses",
        "--frames", str(frames),
        "--execute",
    ]
    return [preflight, 'if ($LASTEXITCODE -ne 0) { Write-Error "preflight refused"; exit 1 }',
            " ".join(shlex.quote(part) for part in runner)]


def build_plan(*, sizes: list[int], shapes: list[str], routes: list[str],
               model_session: str, campaign_dir: Path, frames: int,
               replay_exposures: list[float], repeats: int) -> dict:
    """The whole campaign: lanes, ordering, repeat budget and phase provenance."""
    lanes = []
    for route in routes:
        lanes.extend(plan_conditions(sizes=sizes, shapes=shapes, route=route,
                                     model_session=model_session, campaign_dir=campaign_dir))
    # Stage-major, size-major: every lane of one size and one route runs back to
    # back so the camera window and the exposure land once, and a mid-campaign
    # abort costs the rest of that size rather than scattering half-finished work
    # across three windows.
    lanes.sort(key=lambda lane: (lane["route"], lane["size"], shapes.index(lane["shape"])))
    for index, lane in enumerate(lanes):
        lane["order_index"] = index
        lane["commands"] = render_lane(lane, frames=frames, replay_exposures=replay_exposures)
        lane["plan_reference_blocks"] = plan_reference_blocks(
            [dict(name=f'ref_s{lane["size"]}', block_seconds=DEFAULT_MINUTES_PER_BLOCK * 60,
                  planned_repeats=repeats)], block_seconds=DEFAULT_MINUTES_PER_BLOCK * 60)
    return dict(
        campaign_dir=str(campaign_dir), model_session=model_session,
        routes={name: ROUTES[name] for name in routes},
        sizes=list(sizes), shapes=list(shapes), frames=frames,
        replay_exposures=list(replay_exposures), repeats_per_condition=repeats,
        required_results=dict(REQUIRED_RESULTS),
        required_audit=list(REQUIRED_AUDIT),
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
        lanes=lanes)


def validate_plan(plan: dict) -> list[str]:
    """Reject a plan that would break the protocol. Returns the problems found.

    These are the mistakes that actually happened: a GS lane, a lane without the
    preflight, two routes pooled into one session name, a missing repeat budget,
    and a missing result class.
    """
    problems = []
    seen_sessions = set()
    for lane in plan["lanes"]:
        if lane["session"] in seen_sessions:
            problems.append(f'duplicate session name {lane["session"]}')
        seen_sessions.add(lane["session"])
        if lane["gs_calls_allowed"] != 0 or "gs" in lane["initial_phase"].split("_"):
            problems.append(f'{lane["session"]}: a no-GS campaign lane may not use GS')
        if lane["route"] not in plan["routes"]:
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
    parser.add_argument("--replay-exposures", type=float, nargs="+", default=[20.0, 50.0, 90.0])
    parser.add_argument("--repeats", type=int, default=6,
                        help="repeats per condition (6 = the fewest that can reach p<0.05)")
    parser.add_argument("--check", action="store_true", help="validate and print, write nothing")
    args = parser.parse_args(argv)

    plan = build_plan(sizes=args.sizes, shapes=args.shapes, routes=args.routes,
                      model_session=args.model_session, campaign_dir=args.campaign_dir,
                      frames=args.frames, replay_exposures=args.replay_exposures,
                      repeats=args.repeats)
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
                          f'shape={lane["shape"]}  size={lane["size"]} px', *lane["commands"], ""])
        (lanes_dir / f'{lane["order_index"]:02d}_{lane["session"]}.ps1').write_text(
            body, encoding="utf-8")
    print(f'wrote {args.campaign_dir / "plan.json"} and {len(plan["lanes"])} lanes')
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
