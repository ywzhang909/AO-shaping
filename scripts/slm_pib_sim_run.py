"""Run the ``slm-pib`` SPGD pipeline in the simulation environment.

Wires the pure-numpy 2f-Fourier sim (``drivers/sim/slm_pib_sim.py``) into the
real ``slm_pib_runner`` CLI path:

* registers the ``"sim"`` camera type so ``create_camera("sim", ...)`` returns a
  :class:`SimPibCCD` that reads the shared far-field state;
* monkeypatches ``ao_shaping.optimizer.wfless.slm_zernike_pib.Santec`` to
  :class:`SimSLMPib` so the optimizer's SLM context manager instantiates the
  sim instead of the real Santec driver (no hardware, no DVI hang);
* invokes the genuine ``slm_pib_runner.run`` Click entry with ``--cam_type sim``
  and ``--debug`` so the standard debug artifacts (PNG / PKL / JSON) are written.

The output is a full SPGD search history (objective, per-mode coefficients,
sent phase and far-field image per epoch) — exactly what a hardware run would
record — ready for report generation.

Usage:
    python scripts/slm_pib_sim_run.py --epochs 300 --target-shape square
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import numpy as np  # noqa: E402

from ao_shaping.drivers.sim.slm_pib_sim import (  # noqa: E402
    register_sim_camera,
    reset_system,
    SimSLMPib,
)


def _patch_santec() -> None:
    """Replace the Santec SLM with the sim in the optimizer module."""
    import ao_shaping.optimizer.wfless.slm_zernike_pib as opt

    opt.Santec = SimSLMPib


def main() -> None:
    import click

    # Fresh optical system per run (so repeated runs don't share state).
    reset_system(seed=42)
    register_sim_camera()
    _patch_santec()

    from ao_shaping.runners.slm_pib_runner import run as slm_pib_run

    # Build the argument list for the Click entry. We invoke the ``spgd``
    # subcommand (the default) with a small epoch count suitable for a CPU sim.
    epochs = 300
    click_args = [
        "spgd",
        "-d", "data",
        "--debug",
        "--cam_type", "sim",
        "--cam-id", "0",
        "--exposure_time_ms", "80",
        "--cam_size", "512",
        "-c", "shape",
        "--slm_number", "1",
        "--slm_wavelength", "1064",
        "-n", "4",
        "--objective", "shape",
        "--target_shape", "square",
        "--target_size", "120",
        "-e", str(epochs),
        "--delta", "0.5",
        "--optimizer_type", "adamod",
        "--w_uniformity", "2.0",
        "--w_peak", "0.5",
    ]
    slm_pib_run.main(args=click_args, standalone_mode=True)


if __name__ == "__main__":
    main()
