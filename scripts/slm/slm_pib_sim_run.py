"""Run the ``slm-pib`` SPGD pipeline in the simulation environment.

Wires the pure-numpy 2f-Fourier sim (``drivers/sim/slm_pib_sim.py``) into the
real ``slm-pib`` CLI path:

* registers the ``"sim"`` camera type so ``create_camera("sim", ...)`` returns a
  :class:`SimPibCCD` that reads the shared far-field state;
* monkeypatches ``ao_shaping.optimizer.wfless.slm_zernike_pib.Santec`` to
  :class:`SimSLMPib` so the optimizer's SLM context manager instantiates the
  sim instead of the real Santec driver (no hardware, no DVI hang);
* invokes the genuine ``slm_shaping_runner.run`` Click entry with ``--cam-type sim``
  and ``--debug`` so the standard debug artifacts (PNG / PKL / JSON) are written.

An optional wavefront disturbance -- atmospheric turbulence plus a thermal halo
-- can be injected through ``--disturbance``. Because the report generator is a
pure offline reader, everything the report needs to describe the disturbance is
written next to the run as a companion ``disturbance.json`` + ``disturbance.npz``
pair.

Usage:
    python scripts/slm/slm_pib_sim_run.py --epochs 300 --target-shape square
    python scripts/slm/slm_pib_sim_run.py --epochs 300 --disturbance static
    python scripts/slm/slm_pib_sim_run.py --epochs 300 --disturbance dynamic --cn2 2e-13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.disturbance import (
    DISTURBANCE_MODES,
    DisturbanceConfig,
    SimDisturbance,
)
from ao_shaping.drivers.sim.slm_pib_sim import (
    SLM_H,
    SLM_W,
    SimSLMPib,
    register_sim_camera,
    reset_system,
)

#: Fixed optical-system seed (the report's baseline runs use the same value).
SYSTEM_SEED = 42

#: 伴随产物文件名与注入状态的唯一来源在 drivers.sim.turbulence_harness
#: (#253 收口: 脚本与新 runner 共用同一实现, 此处仅为旧引用面保留转发)。
from ao_shaping.drivers.sim.turbulence_harness import (
    COMPANION_JSON,
    COMPANION_NPZ,
    artifact_roots as _artifact_roots,
    find_new_artifact_dir,
    install_disturbance_system,
    run_dir_of,
    write_disturbance_companion,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the harness command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the slm-pib SPGD pipeline on the 2f-Fourier sim (no hardware)."
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="SPGD epochs (default: 300)"
    )
    parser.add_argument(
        "--objective",
        default="shape",
        help="Shaping objective passed through to the runner (default: shape)",
    )
    parser.add_argument(
        "--target-shape", default="square", help="Target shape (default: square)"
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Search driver; omit to use the runner's default (spgd)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root dir the runner writes debug artifacts under (default: data)",
    )

    parser.add_argument(
        "--disturbance",
        choices=DISTURBANCE_MODES,
        default="none",
        help=(
            "Wavefront disturbance regime (default: none). 'static' generates one "
            "frozen screen (the repo's 'closed' analogue); 'dynamic' draws an "
            "independent screen on every optical evaluation (the 'open'/sliding "
            "analogue, i.e. the fully-decorrelated limit -- NOT a wind model)."
        ),
    )
    parser.add_argument(
        "--dist-tag",
        default=None,
        help="Report tag for this run (default: derived from --disturbance)",
    )
    parser.add_argument(
        "--cn2",
        type=float,
        default=2e-13,
        help="Refractive-index structure constant (default: 2e-13)",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=500.0,
        help="Generator path-length knob [m] (default: 500). Degenerate with --cn2.",
    )
    parser.add_argument(
        "--l-max", type=float, default=30.0, help="Outer scale [m] (default: 30)"
    )
    parser.add_argument(
        "--l-min", type=float, default=2e-3, help="Inner scale [m] (default: 2e-3)"
    )
    parser.add_argument(
        "--pixel-pitch-um",
        type=float,
        default=8.0,
        help="SLM pixel pitch [um] (default: 8); sets the screen's physical extent",
    )
    parser.add_argument(
        "--halo-pv-waves",
        type=float,
        default=0.30,
        help="Thermal-halo peak-to-valley [waves] (default: 0.30; 0 disables it)",
    )
    parser.add_argument(
        "--halo-radius-px",
        type=float,
        default=600.0,
        help=(
            "Thermal-halo radius [SLM px] (default: 600). The beam waist is 400 px, "
            "so a halo much beyond ~800 px is attenuated to invisibility."
        ),
    )
    parser.add_argument(
        "--dist-seed",
        type=int,
        default=20261001,
        help="Disturbance seed (default: 20261001)",
    )
    parser.add_argument(
        "--dist-archive-factor",
        type=int,
        default=8,
        help="Spatial decimation factor for archived screen thumbnails (default: 8)",
    )
    parser.add_argument(
        "--dist-archive-max",
        type=int,
        default=12,
        help="Maximum distinct screens archived (default: 12)",
    )
    return parser.parse_args(argv)


def build_disturbance_config(args: argparse.Namespace) -> DisturbanceConfig:
    """Build a :class:`DisturbanceConfig` from parsed CLI arguments."""
    return DisturbanceConfig(
        mode=args.disturbance,
        cn2=args.cn2,
        distance_m=args.distance_m,
        l_max=args.l_max,
        l_min=args.l_min,
        pixel_pitch_m=args.pixel_pitch_um * 1e-6,
        thermal_halo_pv_waves=args.halo_pv_waves,
        thermal_halo_radius_px=args.halo_radius_px,
        seed=args.dist_seed,
    )


def resolve_tag(args: argparse.Namespace) -> str:
    """Report tag for this run: explicit ``--dist-tag`` or the disturbance mode."""
    return args.dist_tag or args.disturbance


def _install_disturbance_reset(disturbance: SimDisturbance) -> None:
    """转发到 drivers.sim.turbulence_harness.install_disturbance_system。"""
    install_disturbance_system(disturbance)


def write_companion(
    disturbance: SimDisturbance,
    run_dir: Path,
    *,
    run_meta: dict,
    archive_factor: int = 8,
    archive_max: int = 12,
) -> tuple[Path, Path]:
    """转发到 harness 的 write_disturbance_companion (旧调用面不变)。"""
    return write_disturbance_companion(
        disturbance,
        run_dir,
        run_meta=run_meta,
        archive_factor=archive_factor,
        archive_max=archive_max,
    )


def _patch_santec() -> None:
    """Replace the Santec SLM with the sim in the optimizer module."""
    import ao_shaping.optimizer.wfless.slm_zernike_pib as opt
    from ao_shaping.drivers.sim.sim_bench_patch import install_sim_slm

    install_sim_slm(opt)


def main() -> None:
    """Entry point: run one simulated ``slm-pib`` search and record the disturbance."""
    args = parse_args()
    config = build_disturbance_config(args)
    disturbance = SimDisturbance(config, (SLM_H, SLM_W))

    if args.disturbance != "none" and args.halo_pv_waves <= 0.0:
        logger.warning(
            "--disturbance {} with --halo-pv-waves 0: thermal halo DISABLED, "
            "turbulence only",
            args.disturbance,
        )

    _install_disturbance_reset(disturbance)
    reset_system(seed=SYSTEM_SEED, disturbance=disturbance)
    register_sim_camera()
    _patch_santec()

    from ao_shaping.runners.slm.slm_shaping_runner import run as slm_pib_run

    before = _artifact_roots(args.data_root)

    click_args = [
        "spgd",
        "-d",
        str(args.data_root),
        "--debug",
        "--cam-type",
        "sim",
        "--cam-id",
        "0",
        "--exposure-time-ms",
        "80",
        "--cam-size",
        "512",
        "-c",
        "shape",
        "--slm-number",
        "1",
        "--slm-wavelength",
        "1064",
        "-n",
        "4",
        "--objective",
        args.objective,
        "--target-shape",
        args.target_shape,
        "--target-size",
        "120",
        "-e",
        str(args.epochs),
        "--delta",
        "0.5",
        "--optimizer-type",
        "adamod",
        "--w-uniformity",
        "2.0",
        "--w-peak",
        "0.5",
    ]
    if args.algorithm:
        click_args += ["--algorithm", args.algorithm]

    exit_code = 0
    try:
        slm_pib_run.main(args=click_args, standalone_mode=True)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
        if exit_code != 0:
            logger.warning("slm-pib runner exited with status {}", exit_code)

    artifact_root = find_new_artifact_dir(args.data_root, before)
    if artifact_root is None:
        logger.error(
            "no artifact dir found under {}/debug; companion files NOT written",
            args.data_root,
        )
        return

    run_meta = {
        "tag": resolve_tag(args),
        "epochs": args.epochs,
        "objective": args.objective,
        "target_shape": args.target_shape,
        "cam_type": "sim",
        "seed": SYSTEM_SEED,
        "exit_code": exit_code,
    }
    json_path, npz_path = write_companion(
        disturbance,
        run_dir_of(artifact_root),
        run_meta=run_meta,
        archive_factor=args.dist_archive_factor,
        archive_max=args.dist_archive_max,
    )

    measured = disturbance.stats()
    logger.info(
        "disturbance mode={} sigma_total={:.4f} rad ({:.4f} waves) "
        "sigma_turb={:.4f} sigma_halo={:.4f} streaks={} evals={}",
        config.mode,
        measured["sigma_total_rad"],
        measured["sigma_total_rad"] / (2.0 * np.pi),
        measured["sigma_turb_rad"],
        measured["sigma_halo_rad"],
        int(measured["streaks_used"]),
        int(measured["calls"]),
    )
    logger.info("companion manifest -> {}", json_path)
    logger.info("companion archive  -> {}", npz_path)


if __name__ == "__main__":
    main()
