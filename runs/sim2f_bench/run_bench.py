"""CLI for the pre-registered 2f bench measurement (``runs/sim2f_bench/README.md``).

``--backend sim`` is the **default**, so a bare run measures the calibrated twin
and never touches hardware:

    .venv\\Scripts\\python.exe runs\\sim2f_bench\\run_bench.py --samples loop

That dry run must reproduce ``runs/sim2f_fair/loop/a_no_noise/eval_loop.json``
(the published §5 ``flat`` / ``net_open`` rows) before any hardware session is
trusted -- the measurement code is then proven against the report it is meant to
extend, not merely reviewed.

Hardware needs the exposure ceiling in the environment first (README §8.3):

    $env:AO_DAHENG_MAX_EXPOSURE_MS="0.8"
    .venv\\Scripts\\python.exe runs\\sim2f_bench\\run_bench.py --backend hw `
        --exposure-ms 0.02 --settle-s 0.3 --samples loop --out-dir runs/sim2f_bench/session

Exit codes: ``0`` a verdict was written, ``2`` a Gate-0 check failed (so the main
criterion was deliberately not reported), ``1`` any error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo-root bootstrap: this file lives in ``runs/``, outside ``src/``, so the
# package is importable without a PYTHONPATH (the repo's scripts/ convention).
ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / "src"), str(ROOT / "libs")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from loguru import logger  # noqa: E402

from ml.sim2f.bench import (  # noqa: E402
    LOOP_COUNT,
    LOOP_POPULATION,
    LOOP_SEED,
    MIN_DELTA,
    MIN_WINS,
    NOISE_FACTOR,
    PUPIL_RADIUS_PX,
    SETTLE_S,
    BenchmarkError,
    format_gate0,
    format_rows,
    run_session,
)

#: Defaults of README §1 (the frozen measurement target).
DEFAULT_CHECKPOINT = "runs/sim2f_fair/train/model_best.pt"
DEFAULT_CACHE = "runs/sim2f_fair/corpus"
DEFAULT_OUT = "runs/sim2f_bench/session"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser, defaults equal to the protocol's."""
    parser = argparse.ArgumentParser(
        prog="run_bench.py",
        description=(
            "Measure a trained 2f U-Net phase on the calibrated twin (default) or on "
            "the real SLM+CCD bench, and apply the pre-registered criterion of "
            "runs/sim2f_bench/README.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("sim", "hw"),
        default="sim",
        help="sim = calibrated twin (dry run, no hardware); hw = Santec SLM + CCD",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="trained U-Net checkpoint"
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="corpus directory (pairs.npz)")
    parser.add_argument(
        "--samples",
        choices=("test", "loop"),
        default="test",
        help=(
            "test = the corpus' test split; "
            f"loop = first {LOOP_COUNT} of randperm({LOOP_POPULATION}, seed={LOOP_SEED})"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="measure at most N samples")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="artefact directory")
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        help=(
            "fixed camera exposure in ms; an explicit value always wins and skips the "
            "ladder (required only if you want to pin it)"
        ),
    )
    parser.add_argument(
        "--exposure-ladder",
        default=None,
        help=(
            "comma-separated exposures (ms) to probe when --exposure-ms is absent: the "
            "flat-phase peak is measured at each and the step closest to half of full "
            "scale wins. Omit for the default geometric ladder cap/64..cap where cap is "
            "AO_DAHENG_MAX_EXPOSURE_MS"
        ),
    )
    parser.add_argument("--settle-s", type=float, default=SETTLE_S, help="wait after a write")
    parser.add_argument("--ccd-px", type=int, default=256, help="camera window side")
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="skip frames.npz (single-sample debugging)",
    )
    parser.add_argument("--cam-type", default="daheng", help="camera backend name")
    parser.add_argument("--cam-id", type=int, default=0, help="camera index")
    parser.add_argument(
        "--panel-shape",
        default=None,
        help="HxW panel pixels; omit to ask the SLM (SLM-200: 1200x1920)",
    )
    parser.add_argument(
        "--pupil-center-xy",
        default=None,
        help=(
            "centre X,Y (panel pixels) of the ILLUMINATED pupil -- i.e. of the 900 px "
            "square the 128-grid phase is embedded into; omit only if the beam is "
            "centred on the panel (the default assumes it is, which is the 2f bench's "
            "alignment target). A wrong value puts the phase on the wrong pixels, so "
            "the far field is wrong rather than shifted"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="torch device for the network and the twin",
    )
    parser.add_argument("--seed", type=int, default=LOOP_SEED, help="master seed")
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "also persist process data (measured frame, command phase, grayscale) "
            "under <out-dir>/process_data/ via BenchRecorder; off by default "
            "(~4.4 MiB/row)"
        ),
    )
    return parser


def _panel_shape(text: str | None) -> tuple[int, int] | None:
    """Parse ``--panel-shape HxW`` (``x``, ``,`` or space separated)."""
    if text is None:
        return None
    cleaned = str(text).lower().replace("x", " ").replace(",", " ").split()
    if len(cleaned) != 2:
        raise ValueError(f"--panel-shape expects HxW, got {text!r}")
    return int(cleaned[0]), int(cleaned[1])


def _center_xy(text: str | None) -> tuple[int, int] | None:
    """Parse ``--pupil-center-xy X,Y`` (``x``, ``,`` or space separated)."""
    if text is None or not str(text).strip():
        return None
    cleaned = str(text).lower().replace("x", " ").replace(",", " ").split()
    if len(cleaned) != 2:
        raise ValueError(f"--pupil-center-xy expects X,Y, got {text!r}")
    return int(cleaned[0]), int(cleaned[1])


def _ladder(text: str | None) -> tuple[float, ...] | None:
    """Parse ``--exposure-ladder a,b,c`` into a tuple of ms values."""
    if text is None or not str(text).strip():
        return None
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        return None
    return tuple(float(item) for item in parts)


def main(argv: list[str] | None = None) -> int:
    """Run one session.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` a verdict was written, ``2`` a Gate-0 check failed, ``1`` an error.
    """
    args = build_parser().parse_args(argv)
    try:
        panel_shape = _panel_shape(args.panel_shape)
        pupil_center = _center_xy(args.pupil_center_xy)
        ladder = _ladder(args.exposure_ladder)
    except ValueError as exc:
        logger.error("invalid geometry option: {}", exc)
        return 1
    panel_text = (
        "device" if panel_shape is None else f"{panel_shape[0]}x{panel_shape[1]}"
    )
    center_text = (
        "panel centre (assumed)"
        if pupil_center is None
        else f"{pupil_center[0]},{pupil_center[1]}"
    )
    if args.exposure_ms is not None:
        exposure_text = f"{args.exposure_ms} ms (explicit; ladder skipped)"
    elif ladder is None:
        exposure_text = "ladder cap/64..cap (chosen by measurement)"
    else:
        exposure_text = "ladder " + ",".join(f"{v:g}" for v in ladder) + " (measured)"

    print("=== 2f bench session ===")
    print(f"backend       : {args.backend}")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"cache         : {args.cache}")
    print(f"samples       : {args.samples}"
          + ("" if args.limit is None else f" (limit {args.limit})"))
    print(f"out-dir       : {args.out_dir}")
    print(f"exposure      : {exposure_text}   settle {args.settle_s} s")
    print(f"ccd window    : {args.ccd_px} px   camera {args.cam_type}#{args.cam_id}")
    print(f"panel         : {panel_text}   pupil radius "
          f"{PUPIL_RADIUS_PX:g} px ({int(round(2 * PUPIL_RADIUS_PX))} px square)")
    print(f"pupil centre  : {center_text}  (panel px)")
    print(f"criterion     : wins >= {MIN_WINS}, mean(ncc_net - ncc_flat) > {MIN_DELTA}, "
          f"noise floor {NOISE_FACTOR:g} x |ncc_A1 - ncc_A2|")
    print()

    try:
        payload = run_session(
            checkpoint=args.checkpoint,
            cache=args.cache,
            samples=args.samples,
            limit=args.limit,
            out_dir=args.out_dir,
            backend=args.backend,
            exposure_ms=args.exposure_ms,
            exposure_ladder=ladder,
            settle_s=args.settle_s,
            ccd_px=args.ccd_px,
            save_frames=not args.no_save_frames,
            cam_type=args.cam_type,
            cam_id=args.cam_id,
            panel_shape=panel_shape,
            pupil_center_xy=pupil_center,
            device=args.device,
            seed=args.seed,
            record=args.record,
        )
    except (FileNotFoundError, ValueError, KeyError, BenchmarkError) as exc:
        logger.error("session failed: {}", exc)
        return 1
    except Exception:
        logger.exception("session failed")
        return 1

    print("--- Gate 0 ---")
    print(format_gate0(payload["gate0"]))
    print()
    exposure = payload.get("exposure_ladder") or {}
    if exposure.get("chosen_ms") is not None:
        print(f"exposure chosen    : {exposure['chosen_ms']} ms "
              f"({exposure.get('source')}, target peak "
              f"{exposure.get('target_fraction')} of full scale)")
        print()
    print("--- per-sample rows ---")
    print(format_rows(payload["rows"]))
    print()
    criterion = payload["criterion"]
    print("--- criterion ---")
    print(f"counted            : {criterion['counted']} (excluded "
          f"{criterion['excluded']}: {criterion['indistinguishable'] or '-'})")
    print(f"wins (net > flat)  : {criterion['wins']} / {criterion['min_wins']} required")
    delta = criterion["counted_mean_delta"]
    print(f"mean(ncc_net-flat) : {'-' if delta is None else f'{delta:+.4f}'} "
          f"(> {criterion['min_delta']})")
    print()
    if payload["verdict"] is None:
        print("verdict            : (not reported: Gate 0 failed)")
        print(f"artefacts          : {Path(args.out_dir).resolve()}")
        return 2
    print(f"verdict            : {payload['verdict']}")
    print(f"artefacts          : {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
