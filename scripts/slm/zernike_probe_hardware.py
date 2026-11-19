"""Issue #229 hardware probe acquisition: physically-interpretable ±Zernike vs random.

One-off experiment (2026-11-04). Reuses production ``transfer/capture.py`` for all
hardware, exposure, manifest and safe-close behaviour by patching ``capture.probes``
to emit a single superset pool. The pool is captured once so the ±Zernike arm, the
random arm and the baseline can be sliced **offline at equal training-frame budgets**
without re-occupying the SLM/CCD.

Dataset layout matches ``transfer/fit.load_dataset``: calibration rows
(``reference_start`` + signed ``ramp_*`` + ``reference_end``) for geometry, then
labelled probe rows whose NPZ carries ``model_phase`` and ``raw_frames``:

- ``ztrain_{i}_Z{j}_{amp}``  split=train  single-mode Noll-j at ±amplitude (rad)
- ``rtrain_{i}_{family}``    split=train  random probes (baseline/random arm)
- ``holdout_{i}_{family}``   split=test   unseen phases, never used for fitting

Zernike rows are nested-ordered (all modes at ±0.1 first, then ±0.2) so any prefix
of length n is a balanced training subset for a sample-efficiency learning curve.
Default is plan-only; ``--execute`` is required to open hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ao_shaping.tools.slm.transfer import capture
from ao_shaping.tools.slm.transfer.probes import probe_grid
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

LOCK_EVIDENCE_NAME = "device_lock.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_lock_evidence(
    out: Path,
    wait_until_idle_s: float,
    device_wait: dict[str, object],
    lock_record: dict[str, object],
) -> Path | None:
    """Persist lock wait/acquire/release evidence next to ``config.json``.

    ``capture.capture`` owns ``config.json`` (``exist_ok=False``); this writes a
    separate file so the capture config is never touched. Best-effort: an
    unwritable/missing output directory is reported, never raised, so a capture
    failure keeps its own exception.
    """
    doc = {
        "schema": "zernike_probe_hardware/device_lock/1",
        "issue": 229,
        "wait_until_idle_s": wait_until_idle_s,
        "device_wait": device_wait,
        "lock": lock_record,
        "written_utc": _utc_now(),
    }
    path = out / LOCK_EVIDENCE_NAME
    try:
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except OSError as exc:
        print(
            f"WARNING: could not write lock evidence {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    return path


def build_plan(
    modes: list[int], amplitudes: list[float], n_random: int, n_holdout: int
) -> list[tuple[str, str, str, float | None]]:
    """Return (label, split, kind, noll_or_nan) plan; kind in {zernike,random,holdout}."""
    plan: list[tuple[str, str, str, float | None]] = []
    i = 0
    for amp in amplitudes:  # nested: every mode at the small amplitude first
        for j in modes:
            for sign in (1.0, -1.0):
                plan.append(
                    (
                        f"ztrain_{i:03d}_Z{j}_{sign * amp:+.2f}",
                        "train",
                        "zernike",
                        float(j),
                    )
                )
                i += 1
    for k in range(n_random):
        plan.append((f"rtrain_{k:03d}", "train", "random", None))
    for k in range(n_holdout):
        plan.append((f"holdout_{k:03d}", "test", "holdout", None))
    return plan


def make_probes(plan, seed: int):
    original = capture.probes

    def probes(size: int, count: int, seed_arg: int, periods=(64, 128)):
        # reference_start + signed ramps (drop the trailing reference_end).
        yield from list(original(size, 0, seed_arg, periods))[:-1]
        zero = np.zeros((size, size), np.float32)
        random_index = 0
        holdout_index = 0
        for label, split, kind, noll in plan:
            if kind == "zernike":
                # label encodes Z{j} and signed amplitude; recover amplitude from label.
                amp = float(label.rsplit("_", 1)[1])
                phase = generate_zernike_phase({int(noll): amp}, (size, size))
                phase = np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)
                yield label, split, phase
            elif kind == "random":
                family, phase = probe_grid(random_index, size, seed + 777)
                yield f"{label}_{family}", split, phase
                random_index += 1
            else:  # holdout
                family, phase = probe_grid(1000 + holdout_index, size, seed + 5)
                yield f"{label}_{family}", split, phase
                holdout_index += 1
        yield "reference_end", "calibration", zero

    return probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(4, 12)))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument("--n-holdout", type=int, default=12)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1104229)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help=(
            "Before opening hardware (--execute only), wait up to SECONDS for the "
            "device lock / other hardware processes to clear, then hold the "
            "device_mutex lock over the whole capture (default: 120). "
            "Pass 0 to bypass both the wait and the lock (repo convention: "
            "0 = off)."
        ),
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        metavar="MS",
        help=(
            "Seed exposure (ms) handed to capture as its opening exposure; the "
            "capture's own ladder still adapts from it. Omit to use the camera's "
            "remembered effective exposure (capture's 'auto' rule)."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be a finite non-negative number")

    plan = build_plan(args.modes, args.amplitudes, args.n_random, args.n_holdout)
    plan_doc = {
        "schema": "zernike_probe_hardware/1",
        "issue": 229,
        "grid": args.grid,
        "modes": args.modes,
        "amplitudes_rad": args.amplitudes,
        "frames": args.frames,
        "seed": args.seed,
        "n_zernike": sum(1 for p in plan if p[2] == "zernike"),
        "n_random": args.n_random,
        "n_holdout": args.n_holdout,
        "plan": [{"label": p[0], "split": p[1], "kind": p[2]} for p in plan],
    }

    if not args.execute:
        for label, split, kind, _ in plan:
            print(f"{split:12s} {kind:8s} {label}")
        print(
            f"\nPLAN ONLY: {len(plan)} probes + calibration (reference_start, ramps, reference_end)."
        )
        print("Re-run with --execute to open hardware and acquire.")
        print(
            f"Device lock: --wait-until-idle {args.wait_until_idle:g}s "
            f"(default {120:g}s; 0 disables the wait and the lock)."
        )
        print(
            "NOTE: --execute requires a fresh --out directory (capture uses exist_ok=False)."
        )
        return

    # capture() creates args.out itself (exist_ok=False); do not pre-create it.
    namespace = SimpleNamespace(
        out=args.out,
        execute=True,
        camera=args.camera,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        grid=args.grid,
        ramp_periods=(64, 128),
        count=0,
        frames=args.frames,
        seed=args.seed,
        exposure_max_steps=2,
        exposure_ms=args.exposure_ms,
        save_dataset=True,
        device=None,
        checkpoint=None,
    )

    # ---- lazy device imports (this is the ONLY place drivers are imported) -- #
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    # ---- device lock (held for the whole capture) --------------------------- #
    lock = None
    device_wait: dict[str, object]
    if args.wait_until_idle > 0:
        wait_kwargs: dict[str, object] = {
            "slm_number": args.slm_number,
            "cam_id": args.cam_id,
            "cam_type": args.camera,
            "own_paths": [Path(__file__).resolve()],
        }
        # Probe at the seed exposure when one was asked for; otherwise let
        # ``wait_until_idle`` keep its own default.
        if args.exposure_ms is not None:
            wait_kwargs["exposure_ms"] = float(args.exposure_ms)
        verdict = device_mutex.wait_until_idle(args.wait_until_idle, **wait_kwargs)
        device_wait = {
            "budget_s": args.wait_until_idle,
            "status": verdict.status,
            "waited_s": round(verdict.waited_s, 3),
            "own_process_tree": verdict.holder_is_me,
            "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
            "holder_cmdline": (
                verdict.holder.cmdline if verdict.holder is not None else None
            ),
            "detail": verdict.detail,
            "remedy": verdict.remedy,
        }
        if not verdict.ok:
            # Master ``WaitVerdict`` has no ``headline()``; compose the same
            # one-liner from its fields (reason is the short compatibility
            # summary, detail the root cause).
            headline = verdict.reason or verdict.detail or verdict.status
            raise SystemExit(
                f"device unavailable ({verdict.kind or verdict.status}): "
                f"{headline} -- {verdict.remedy}"
            )
        # ``wait_until_idle`` hands back the lock it already took when it cleared
        # us; otherwise take the shared bench lock ourselves.
        if verdict.lock is not None:
            lock = verdict.lock
        else:
            lock = DeviceLock.default()
            if not lock.acquire(timeout=0.0):
                raise SystemExit(
                    "device lock could not be acquired right after the idle wait; "
                    "another session took it — rerun (or wait for it to finish)"
                )
    else:
        device_wait = {
            "budget_s": 0.0,
            "status": "skipped",
            "waited_s": 0.0,
            "own_process_tree": False,
            "holder_pid": None,
            "holder_cmdline": None,
            "detail": "waiting disabled (--wait-until-idle 0)",
            "remedy": "",
        }

    lock_record: dict[str, object] = {
        "path": str(lock.path) if lock is not None else None,
        "held": lock.is_locked() if lock is not None else False,
        "acquired_utc": _utc_now() if lock is not None else None,
        "held_s": None,
        "released": None,
        "released_utc": None,
        "release_error": None,
    }

    def release_lock() -> None:
        if lock_record["acquired_utc"] is not None:
            lock_record["held_s"] = round(
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(lock_record["acquired_utc"]))
                ).total_seconds(),
                3,
            )
        if lock is None:
            lock_record["released"] = True
            lock_record["released_utc"] = _utc_now()
            return
        try:
            lock.release()
            lock_record["released"] = True
        except Exception as exc:  # pragma: no cover - operator-facing guard
            lock_record["release_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: device lock release failed: "
                f"{lock_record['release_error']}",
                file=sys.stderr,
            )
        finally:
            lock_record["released_utc"] = _utc_now()

    try:
        with patch.object(capture, "probes", make_probes(plan, args.seed)):
            capture.capture(namespace)
    finally:
        release_lock()
        write_lock_evidence(
            args.out, args.wait_until_idle, device_wait, lock_record
        )
    (args.out / "probe_plan.json").write_text(
        json.dumps(plan_doc, indent=2), encoding="utf-8"
    )
    print(f"Acquisition complete -> {args.out}")


if __name__ == "__main__":
    main()
