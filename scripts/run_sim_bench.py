"""Run the optical runners headlessly under simulation and record a manifest.

Every optical runner can now be exercised without hardware: the camera and SLM
resolve through the device registries (``--cam_type`` / ``--slm_type``) and the
DM through ``--dm_type sim``. This script drives those runs with the smallest
useful budget and records what each one produced, so a report generator can read
artefacts offline instead of re-running anything.

Each runner is isolated: a failure is captured into the manifest rather than
aborting the sweep, because a partially-completed matrix is still worth
reporting on.

Physical validity differs by family, and the manifest records it explicitly:

* ``spgd-square`` / ``slm-gsnet`` / ``slm-pib`` drive the **SLM**, which
  ``SimPibSystem.far_field()`` actually models, so their numbers mean something.
* ``pib`` / ``combined`` drive the **DM**, and nothing maps DM voltage to phase
  in that model. Their objective is uncoupled noise — the run proves the control
  loop executes, nothing more. See ``drivers/sim/AGENTS.md``.

Usage:
    python scripts/run_sim_bench.py --out data/sim_bench
    python scripts/run_sim_bench.py --only slm-pib spgd-square --epochs 40
    python scripts/run_sim_bench.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

#: Family -> whether the simulation model actually couples the runner's actuator.
#: "dm" runners are smoke tests only; see module docstring.
FAMILY = {
    "pib": "dm",
    "combined": "dm",
    "spgd-square": "slm",
    "slm-gsnet": "slm",
    "slm-pib": "slm",
}


@dataclass
class RunResult:
    """One runner's outcome, as written into the manifest."""

    runner: str
    family: str
    ok: bool
    command: list[str]
    returncode: int | None = None
    wall_s: float = 0.0
    artefacts: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""


def _newest_glob(root: Path, pattern: str) -> Path | None:
    """Most recently modified path matching ``pattern``, or ``None``."""
    hits = [p for p in root.glob(pattern) if p.exists()]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _pib_artefacts(out: Path) -> list[str]:
    """The ``data/flatten_voltages/<date>/*.csv`` written by pib / combined."""
    flat = ROOT / "data" / "flatten_voltages"
    if not flat.is_dir():
        return []
    newest = max(
        (p for p in flat.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if newest is None:
        return []
    return [str(p) for p in sorted(newest.glob("*.csv"))][-4:]


def _slm_square_artefacts(out: Path) -> list[str]:
    """The ``data/slm_square/<ts>/`` history + best coefficients."""
    root = ROOT / "data" / "slm_square"
    newest = _newest_glob(root, "*") if root.is_dir() else None
    if newest is None or not newest.is_dir():
        return []
    return [str(p) for p in sorted(newest.glob("*.csv"))]


def _debug_artefacts(prefix: str) -> list[str]:
    """The ``data/debug/<prefix>_<ts>/<ts>/*`` recorder dumps (needs ``--debug``)."""
    root = ROOT / "data" / "debug"
    if not root.is_dir():
        return []
    newest = _newest_glob(root, f"{prefix}_*")
    if newest is None:
        return []
    inner = _newest_glob(newest, "*")
    if inner is None or not inner.is_dir():
        return []
    return [str(p) for p in sorted(inner.iterdir()) if p.is_file()]


def _collect(runner: str) -> list[str]:
    if runner in ("pib", "combined"):
        return _pib_artefacts(ROOT)
    if runner == "spgd-square":
        return _slm_square_artefacts(ROOT)
    if runner == "slm-pib":
        return _debug_artefacts("slm_pib")
    if runner == "slm-gsnet":
        return _debug_artefacts("slm_gsnet")
    return []


def _metrics_from_csv(paths: list[str]) -> dict[str, float]:
    """Best-effort headline metrics from whichever CSV the runner wrote."""
    import csv

    out: dict[str, float] = {}
    for raw in paths:
        if not raw.endswith(".csv"):
            continue
        try:
            with open(raw, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, csv.Error):
            continue
        if not rows:
            continue
        # History CSVs carry one row per epoch; take the last as the final state.
        row = rows[-1]
        for key in ("J", "pib", "quality", "cv", "ee", "ar", "_p%"):
            if key not in row:
                continue
            try:
                out.setdefault(key, float(row[key]))
            except (TypeError, ValueError):
                continue
    return out


def _build_command(runner: str, epochs: int, out: Path) -> list[str]:
    """The exact argv for one simulated run."""
    base = [sys.executable, str(ROOT / "src" / "ao_shaping" / "main.py"), runner]
    if runner == "pib":
        return base + [
            "--cam_type", "sim", "--dm_type", "sim",
            "-e", str(epochs), "--root_dir", str(out / "pib"),
        ]
    if runner == "combined":
        return base + [
            "--cam_type", "sim", "--dm_type", "sim", "-e", str(epochs),
            "-d", str(out / "combined"),
        ]
    if runner == "spgd-square":
        return base + [
            "--cam_type", "sim", "--slm_type", "sim",
            "-e", str(epochs), "--cam-size", "256", "--zernike-radius", "200",
        ]
    if runner == "slm-gsnet":
        return base + [
            "spgd", "--cam_type", "sim", "-e", str(epochs),
            "--cam_size", "256", "--debug",
        ]
    if runner == "slm-pib":
        return base + [
            "spgd", "--cam_type", "sim", "-e", str(epochs),
            "--cam_size", "256", "--debug",
        ]
    raise ValueError(f"unknown runner {runner!r}")


def _run_one(runner: str, epochs: int, out: Path, timeout: int) -> RunResult:
    import subprocess

    family = FAMILY[runner]
    cmd = _build_command(runner, epochs, out)
    result = RunResult(
        runner=runner,
        family=family,
        ok=False,
        command=cmd[1:],
        note=(
            "DM voltage is not mapped to phase by SimPibSystem, so the objective "
            "is uncoupled; control-loop smoke test only"
            if family == "dm"
            else ""
        ),
    )
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
        result.returncode = proc.returncode
        result.ok = proc.returncode == 0
        if not result.ok:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            result.note = (result.note + " | " if result.note else "") + " | ".join(tail)
    except subprocess.TimeoutExpired:
        result.note = f"timeout after {timeout}s"
    except OSError as exc:
        result.note = f"spawn failed: {exc}"
    result.wall_s = round(time.perf_counter() - started, 2)
    result.artefacts = _collect(runner) if result.ok else []
    result.metrics = _metrics_from_csv(result.artefacts)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "sim_bench"),
        help="Directory for the manifest (default: data/sim_bench)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Epochs per run (default: 20)"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-run timeout in seconds"
    )
    parser.add_argument(
        "--only", nargs="*", choices=sorted(FAMILY),
        help="Run only these runners (default: all five)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the runners and exit"
    )
    args = parser.parse_args()

    if args.list:
        for name, family in sorted(FAMILY.items()):
            kind = "smoke-test only" if family == "dm" else "physically modelled"
            print(f"{name:14s} family={family:4s} {kind}")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runners = args.only or sorted(FAMILY)

    results: list[RunResult] = []
    for runner in runners:
        print(f"--- {runner} (epochs={args.epochs}) ---", flush=True)
        res = _run_one(runner, args.epochs, out, args.timeout)
        results.append(res)
        status = "ok" if res.ok else f"FAILED rc={res.returncode}"
        print(
            f"    {status} in {res.wall_s}s; {len(res.artefacts)} artefact(s); "
            f"metrics={res.metrics or '{}'}",
            flush=True,
        )
        if res.note:
            print(f"    note: {res.note}", flush=True)

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "seed_note": "sim runs are seeded per runner; re-running reproduces them",
        "families": FAMILY,
        "runs": [asdict(r) for r in results],
        "ok_count": sum(1 for r in results if r.ok),
        "total": len(results),
    }
    path = out / "summary.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nmanifest -> {path}")
    print(f"ok {manifest['ok_count']}/{manifest['total']}")
    return 0 if manifest["ok_count"] == manifest["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
