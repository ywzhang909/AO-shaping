# -*- coding: utf-8 -*-
"""Emit the next hardware run of the multishape transfer campaign.

This is the run-plan builder, not a runner: it writes a ``plan.json`` plus one
PowerShell lane per shape into ``data/debug/slm_transfer/multishape_scale_20261028/r2_next``.
Everything it decides is derived from measured evidence, and every claim it
makes about the CLI is checked against the option list the runner actually
registers.

Why each choice is what it is (report §6.5 / §7 / §9):

* ``--model-dir`` reuses ``s600_fixed``'s capture/fit/model640/model_wide640.
  That skips capture (173.2 s) + fit (31.2 s) + geometric re-export. It is only
  valid while the optical path is unchanged, so the emitted lane runs
  ``probe_optical_path_validity.py`` first and aborts on any non-zero exit.
  ``s600_fixed`` re-captured its model precisely because the intensity had
  moved ×1.35; that re-capture is done, so this run consumes it.
* ``--seed-bakeoff`` is skipped per shape whose measured winner is already in
  ``s600_fixed/journal.json`` (the ``seed_bakeoff`` entries carry the scores).
  Re-baking would repeat 2–3 CCD captures per shape for a result already known.
  ``--initial-phase`` is pinned to that winner, which is what the runner uses
  anyway when the bake-off is off.
* ``--replay-exposures 20 50 90`` matches the campaign's common-bit-depth
  replay set; the replay is acceptance evidence (§7 ③), not optional polish.

Deliberately NOT emitted:

* ``--blocks`` / ``--updates``: the runner does not expose them — the stage
  table owns those numbers (``STAGE_RECIPES``) and ``_validate_args`` only
  fills the recipe values. Changing them is a source edit with an offline
  comparison first, not a run flag.
* the ``exposure_read_*.npy`` suppression (§7 ①): it is the largest remaining
  I/O redundancy (393 MB/stage) but it removes the mid-ladder crash evidence.
  It is a code change to evaluate offline (Recorder corpus), not a run flag.

Usage::

    python scripts/slm/plan_transfer_next_run.py --help
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "data" / "debug" / "slm_transfer" / "multishape_scale_20261028"
DEFAULT_MODEL_SESSION = CAMPAIGN / "s600_fixed"
DEFAULT_OUT = CAMPAIGN / "r2_next"
PROBE = ROOT / "scripts" / "slm" / "probe_optical_path_validity.py"
RUNNER = ROOT / "src" / "ao_shaping" / "main.py"

STAGE_ORDER = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")

#: The first rung of every stage's exposure ladder starts at the camera floor
#: and doubles: 6 rungs x (2 discarded + 1 kept) = 18 reads, measured 88 s in
#: ``square/transport_cv`` of ``s600_fixed`` (09:08:15.118 -> 09:08:34.1).
COLD_LADDER_S = 88.0


def stage_durations(session: Path) -> dict[tuple[str, str], float]:
    """Per-stage wall clock, paired by journal ``status``.

    Adjacency pairing fabricates durations when a stage is interrupted and
    resumed (``star/mono12_cv`` in session A: 1227.9 s interrupted + 210.9 s
    resumed, never one 1438.8 s call), so only ``running`` -> ``complete``
    pairs are counted.
    """
    entries = json.loads((session / "journal.json").read_text(encoding="utf-8"))
    running: dict[str, dt.datetime] = {}
    out: dict[tuple[str, str], float] = {}
    for entry in entries:
        key = entry["stage"]
        if "/" not in key:
            continue
        stamp = dt.datetime.fromisoformat(entry["timestamp_utc"])
        if entry.get("status") == "running":
            running[key] = stamp
        elif entry.get("status") == "complete" and key in running:
            out[tuple(key.split("/", 1))] = (stamp - running.pop(key)).total_seconds()
    return out


def seed_winners(session: Path) -> dict[str, dict]:
    """Measured seed 对比择优 outcome per shape, straight from the journal."""
    entries = json.loads((session / "journal.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for entry in entries:
        if entry.get("stage") != "seed_bakeoff":
            continue
        shape = entry.get("shape")
        if not shape:
            continue
        if entry.get("status") == "complete":
            out[shape] = dict(kind=entry["chosen"], scores=entry["scores"])
        elif entry.get("status") == "skipped":
            out[shape] = dict(kind=entry["chosen"], scores=None)
    return out


def lane_script(*, shape: str, out: Path, model_dir: Path, side: float,
                initial_phase: str, seed_bakeoff: bool, gate: Path,
                replay: list[float]) -> str:
    flags = ["--seed-bakeoff"] if seed_bakeoff else ["--no-seed-bakeoff"]
    replay_flags = [f"--replay-exposures {value:g}" for value in replay]
    command = " ".join([
        r".venv\Scripts\python.exe", f'"{RUNNER.relative_to(ROOT)}"',
        "slm-transfer-shapes",
        f'--out "{out}"',
        f"--shapes {shape}",
        f"--side {side:g}",
        f'--model-dir "{model_dir}"',
        f"--initial-phase {initial_phase}",
        *flags,
        *replay_flags,
        "--execute",
    ])
    return f"""# lane {shape} -- generated by scripts/slm/plan_transfer_next_run.py
# One SLM display buffer: lanes must not overlap, so a lock file guards them.
$ErrorActionPreference = "Stop"
Set-Location "{ROOT}"
if (Test-Path "{gate}") {{
    $held = Get-Content "{gate}" -Raw
    Write-Error "another lane holds lane.lock ($held); wait for it before starting {shape}"
    exit 1
}}
Set-Content -Path "{gate}" -Value "{shape} started $((Get-Date).ToString('o'))" -Encoding UTF8

try {{
    # Gate the model reuse: --model-dir is only sound while the path is unchanged.
    # Exit code is the verdict (0 = ok); low_snr / no_response abort the lane.
    .venv\\Scripts\\python.exe "{PROBE.relative_to(ROOT)}"
    if ($LASTEXITCODE -ne 0) {{
        Write-Error "optical-path probe did not return ok (exit $LASTEXITCODE); do not reuse the model"
        exit 1
    }}

    {command}
    if ($LASTEXITCODE -ne 0) {{
        Write-Error "shaping run for {shape} failed (exit $LASTEXITCODE); see the stage journal"
        exit $LASTEXITCODE
    }}
}}
finally {{
    Remove-Item -Path "{gate}" -ErrorAction SilentlyContinue
}}
"""


def replay_only(session: Path, out: Path, *, lane_dir: Path, gate: Path,
                exposures: list[float]) -> Path:
    """A lane that only re-runs the replay of an already-finished session.

    ``s600_fixed`` completed all 18 shaping stages and then died in the replay
    step (``replayed runs disagree on the camera grid``): stage 4 samples the
    window at the finer recipe step, so its ``camera_xy`` differs from stages
    1-3 by construction, not by drift. Re-running the whole session would
    re-measure the finished shaping work (the four-shape shaping stages sum to
    2675.0 s) to obtain evidence the recorded phases already contain, so this
    lane replays the phases that are on disk.
    """
    shapes = [p.name for p in sorted((session / "shape").iterdir()) if p.is_dir()]
    runs = [session / "shape" / shape / stage
            for shape in shapes for stage in STAGE_ORDER
            if (session / "shape" / shape / stage / "manifest.json").is_file()]
    if not runs:
        raise FileNotFoundError(f"no finished stage runs under {session}")
    seeds = [p for p in runs if (p / "manifest.json").is_file()
             and any(row["label"].endswith("_seed")
                     for row in json.loads((p / "manifest.json").read_text(encoding="utf-8")))]
    command = " ".join([
        r".venv\Scripts\python.exe", "-m",
        "ao_shaping.tools.slm.transfer.replay_phases",
        f'--initial-run "{seeds[0]}"',
        "--runs", *[f'"{run}"' for run in runs],
        f'--out "{out}"',
        *[f"--exposures {value:g}" for value in exposures],
        "--camera daheng --cam-id 0",
    ])
    path = lane_dir / "lane_replay_s600_fixed.ps1"
    path.write_text(f"""# lane replay_s600_fixed -- generated by scripts/slm/plan_transfer_next_run.py
# Re-runs ONLY the replay step of the finished s600_fixed session.
$ErrorActionPreference = "Stop"
Set-Location "{ROOT}"
if (Test-Path "{gate}") {{
    $held = Get-Content "{gate}" -Raw
    Write-Error "another lane holds lane.lock ($held); wait for it before the replay"
    exit 1
}}
Set-Content -Path "{gate}" -Value "replay_s600_fixed started $((Get-Date).ToString('o'))" -Encoding UTF8

try {{
    .venv\\Scripts\\python.exe "{PROBE.relative_to(ROOT)}"
    if ($LASTEXITCODE -ne 0) {{
        Write-Error "optical-path probe did not return ok (exit $LASTEXITCODE); the replay anchors to this path"
        exit 1
    }}
    {command}
    if ($LASTEXITCODE -ne 0) {{
        Write-Error "replay failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }}
}}
finally {{
    Remove-Item -Path "{gate}" -ErrorAction SilentlyContinue
}}
""", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-session", type=Path, default=DEFAULT_MODEL_SESSION,
                        help="会话目录 (含 capture/ fit/ model640/ model_wide640/) 供 --model-dir 复用")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="本次运行的根目录")
    parser.add_argument("--side", type=float, default=600.0, help="目标边长 (CCD 原生像元)")
    parser.add_argument("--shapes", nargs="+", default=["square", "triangle", "star", "circle"])
    parser.add_argument("--replay-exposures", type=float, nargs="+", default=[20.0, 50.0, 90.0])
    parser.add_argument("--dry-run", action="store_true", help="只打印计划, 不落盘")
    args = parser.parse_args()

    durations = stage_durations(args.model_session)
    winners = seed_winners(args.model_session)

    plan = []
    measured_shaping_s = 0.0
    for shape in args.shapes:
        stages = [name for name in STAGE_ORDER if (shape, name) in durations]
        if not stages:
            continue
        winner = winners.get(shape)
        measured = sum(durations[(shape, name)] for name in stages)
        measured_shaping_s += measured
        # The cold ladder appears once per run (stage 1, from the camera floor);
        # later stages reuse the exposure that stage 1 landed on.
        predicted = measured - (0.0 if winner is None else COLD_LADDER_S)
        plan.append(dict(
            shape=shape,
            stages=stages,
            measured_s=round(measured, 1),
            initial_phase=(winner or {}).get("kind") or "auto",
            seed_bakeoff=winner is None,
            predicted_s=round(predicted, 1),
            complete_on_disk=len(stages) == len(STAGE_ORDER),
        ))

    payload = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        model_session=str(args.model_session.resolve()),
        out=str(args.out.resolve()),
        side=args.side,
        replay_exposures=args.replay_exposures,
        cold_ladder_s=COLD_LADDER_S,
        skipped_by_model_reuse=["capture", "fit", "model", "capture_geometry"],
        # The same four-shape shaping stages the report quotes as 2675.0 s; the
        # plan's saving is measured against this, so the two numbers of the same
        # session stay in one unit (see the campaign report, §6.5 / §8.5).
        measured_shaping_s=round(measured_shaping_s, 1),
        shapes=plan,
        excluded_by_design=dict(
            blocks_updates="runner exposes no such option; the stage table owns them",
            exposure_read_npy="largest I/O redundancy but part of the crash evidence; needs an offline comparison first",
        ),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    args.out.mkdir(parents=True, exist_ok=False)
    lanes = args.out / "lanes"
    lanes.mkdir()
    gate = args.out / "lane.lock"
    written = []
    for item in plan:
        path = lanes / f"lane_{item['shape']}.ps1"
        path.write_text(lane_script(
            shape=item["shape"], out=args.out / "run" / item["shape"],
            model_dir=args.model_session, side=args.side,
            initial_phase=item["initial_phase"], seed_bakeoff=item["seed_bakeoff"],
            gate=gate, replay=args.replay_exposures), encoding="utf-8")
        written.append(str(path))
    run_all = args.out / "run_all.ps1"
    run_all.write_text("\n".join(
        f'& "{path}"\nif ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}' for path in written),
        encoding="utf-8")
    written.append(str(replay_only(args.model_session, args.out / "replay_s600_fixed",
                                   lane_dir=lanes, gate=gate,
                                   exposures=args.replay_exposures)))
    (args.out / "plan.json").write_text(json.dumps(dict(payload, lanes=written),
                                                   indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(dict(out=str(args.out), lanes=written, run_all=str(run_all),
                          predicted_total_s=round(sum(item["predicted_s"] for item in plan), 1)),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
