"""Interleaved hardware replay of GS-refine phases saved in one run directory."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEQUENCE = ("flat", "gs", "best", "gs", "flat", "best", "flat")


@dataclass(frozen=True)
class ReplayPlan:
    debug_pkl: Path
    best_record_key: int
    phases: dict[str, np.ndarray]
    output_dir: Path
    exposure_ms: float
    cam_size: int
    target_center: tuple[int, int]
    target_side: int


def parse_xy(value: str) -> tuple[int, int]:
    """Parse a camera coordinate supplied as ``x,y``."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("coordinate must be x,y")
    return int(parts[0]), int(parts[1])


def load_replay_plan(
    source: Path,
    *,
    exposure_ms: float,
    output_dir: Path | None = None,
    cam_size: int = 250,
    target_center: tuple[int, int] | None = None,
    target_side: int | None = None,
) -> ReplayPlan:
    """Resolve a run directory or PKL into validated phases without opening hardware."""
    from ao_shaping.utils.image.exposure_control import (
        DAHENG_EXPOSURE_LIMIT_ENV,
        configured_daheng_exposure_limit,
    )

    if exposure_ms <= 0:
        raise ValueError("Daheng exposure must be positive")
    env_ceiling = configured_daheng_exposure_limit()
    if env_ceiling > 0 and exposure_ms > env_ceiling:
        raise ValueError(
            f"Daheng exposure {exposure_ms} ms exceeds the {env_ceiling} ms "
            f"ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
        )
    if cam_size < 1:
        raise ValueError("cam_size must be positive")
    source = Path(source)
    if source.is_dir():
        files = list(source.glob("*.pkl"))
        if len(files) != 1:
            raise ValueError(f"expected one GS debug PKL in {source}, got {len(files)}")
        debug_pkl = files[0]
        run_dir = source
    else:
        debug_pkl = source
        run_dir = source.parent
    with debug_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, dict) or 0 not in records or 1 not in records:
        raise ValueError("GS debug PKL must contain flat (0) and GS (1) records")
    if any(not math.isfinite(float(record["score"])) for record in records.values()):
        raise ValueError("GS debug scores must be finite")
    best_key = max(records, key=lambda key: float(records[key]["score"]))
    phases = {
        name: np.asarray(records[key]["_phase"])
        for name, key in (("flat", 0), ("gs", 1), ("best", best_key))
    }
    shapes = {phase.shape for phase in phases.values()}
    if len(shapes) != 1 or any(phase.ndim != 2 for phase in phases.values()):
        raise ValueError("replay phases must be matching 2D panel arrays")
    if any(not np.isfinite(phase).all() for phase in phases.values()):
        raise ValueError("replay phases contain non-finite values")
    if any(phase.dtype != np.uint16 for phase in phases.values()):
        raise ValueError("replay phases must be recorded uint16 SLM gray arrays")
    flat = records[0]
    if target_center is None:
        target_center = (int(flat["cx"]), int(flat["cy"]))
    if target_side is None:
        target_side = int(flat["target_side"])
    if target_side < 1:
        raise ValueError("target_side must be positive")
    return ReplayPlan(
        debug_pkl=debug_pkl,
        best_record_key=int(best_key),
        phases=phases,
        output_dir=Path(output_dir) if output_dir is not None else run_dir / "replay",
        exposure_ms=exposure_ms,
        cam_size=cam_size,
        target_center=target_center,
        target_side=target_side,
    )


def run_replay(
    plan: ReplayPlan,
    *,
    slm_serial: str,
    cam_center: tuple[int, int],
    expected_wavelength_nm: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Replay while saving each raw CCD pair and its scored image."""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SantecError
    from ao_shaping.optimizer.wfless.slm_gs_refine import _metrics
    from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
    from ao_shaping.utils.image.beam_metrics import prepare_frame
    from ao_shaping.utils.image.exposure_control import (
        resolve_daheng_exposure_limit,
    )
    from ao_shaping.utils.image.targets import square_mask

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "replay_config.json").write_text(
        json.dumps(
            {
                "debug_pkl": str(plan.debug_pkl.resolve()),
                "best_record_key": plan.best_record_key,
                "sequence": SEQUENCE,
                "exposure_ms": plan.exposure_ms,
                "slm_serial": slm_serial,
                "expected_wavelength_nm": expected_wavelength_nm,
                "cam_center": cam_center,
                "cam_size": plan.cam_size,
                "target_center": plan.target_center,
                "target_side": plan.target_side,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, float | int | str]] = []
    with (
        Santec(slm_number=1, wavelength=None, video_mode=0) as slm,
        create_camera("daheng", 0, exposure_time_ms=plan.exposure_ms) as cam,
        preserve_slm_display(slm),
    ):
        if slm.get_serial_number() != slm_serial:
            raise ValueError("SLM serial changed; refusing phase replay")
        if getattr(slm, "_wavelength_phase_unset", False):
            raise ValueError("SLM phase table is unset; refusing gray-phase replay")
        if (expected_wavelength_nm is not None
                and slm.wavelength != expected_wavelength_nm):
            raise ValueError("SLM wavelength changed; saved gray phases may be invalid")
        ceiling = resolve_daheng_exposure_limit(camera=cam)
        if plan.exposure_ms > ceiling + 1e-9:
            raise ValueError(
                f"Daheng exposure {plan.exposure_ms} ms exceeds the resolved "
                f"ceiling {ceiling} ms"
            )
        cam.reset_exposure_time(plan.exposure_ms)
        if cam.exposure_time > ceiling + 1e-9:
            raise ValueError(
                f"camera exposure readback {cam.exposure_time} ms exceeds the "
                f"{ceiling} ms ceiling"
            )
        cam.reset_window(center=cam_center, size=(plan.cam_size, plan.cam_size))
        try:
            protected = slm.get_displayed_memory_number()
        except SantecError as exc:
            if exc.code != 1:
                raise
            protected = None
        work_slots = [slot for slot in (126, 127, 124) if slot != protected][:2]
        for index, name in enumerate(SEQUENCE):
            slm.display_data(plan.phases[name], memory_number=work_slots[index % 2])
            time.sleep(0.8)
            for _ in range(3):
                cam.get_numpy_image(n_sample=1)
            raw_frames = [
                np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                for _ in range(2)
            ]
            raw_peak = max(float(np.max(sample)) for sample in raw_frames)
            if raw_peak >= 250:
                raise ValueError(f"CCD saturated during replay at {name}")
            frame = prepare_frame(np.mean(raw_frames, axis=0))
            mask = square_mask(
                frame.shape, plan.target_center[1], plan.target_center[0],
                plan.target_side,
            )
            score, pib, cv = _metrics(frame, mask, 0.5, 0.5)
            np.save(plan.output_dir / f"{index:02d}_{name}.npy", frame)
            np.save(plan.output_dir / f"{index:02d}_{name}_raw.npy", np.stack(raw_frames))
            row = dict(
                index=index, phase=name, score=float(score), pib=float(pib),
                cv=float(cv), peak=float(np.max(frame)), raw_peak=raw_peak,
                total=float(np.sum(frame)),
            )
            rows.append(row)
            (plan.output_dir / "measurements.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            print(row, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="GS run directory or debug PKL")
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--exposure-ms", type=float, required=True)
    parser.add_argument("--slm-serial", required=True)
    parser.add_argument("--expected-wavelength", type=int)
    parser.add_argument("--cam-center", required=True, help="full-frame zero order x,y")
    parser.add_argument("--cam-size", type=int, default=250)
    parser.add_argument("--target-center", help="ROI center x,y (default: saved flat)")
    parser.add_argument("--target-side", type=int, help="default: saved target side")
    args = parser.parse_args()
    try:
        plan = load_replay_plan(
            args.source,
            exposure_ms=args.exposure_ms,
            output_dir=args.out_dir,
            cam_size=args.cam_size,
            target_center=parse_xy(args.target_center) if args.target_center else None,
            target_side=args.target_side,
        )
        cam_center = parse_xy(args.cam_center)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    run_replay(
        plan,
        slm_serial=args.slm_serial,
        cam_center=cam_center,
        expected_wavelength_nm=args.expected_wavelength,
    )


if __name__ == "__main__":
    main()
