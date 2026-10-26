"""Build the spatially correlated (von Karman) Zernike covariance for ``ml.sim2f``.

Why this file exists
====================
``ml.sim2f.optics.draw_aberration`` used to draw the high-order Zernike
coefficients **i.i.d.** ``N(0, 1)`` and only rescale the total RMS.  Real
turbulence -- and any physically motivated aberration ensemble -- has strongly
correlated modes with variance falling steeply with radial order.  An i.i.d.
draw puts far too much energy in high order, so a model trained on it faces a
*different* aberration family than the one the physics implies.

This script produces the missing ingredient: the ``(K, K)`` covariance of the
Noll coefficients ``4 .. n_modes + 1``, in ``rad**2``, **for the sim2f pupil**
(``Sim2fConfig.grid`` samples across, ``Sim2fConfig.pupil_radius_px`` modelling
disc).  ``AberrationConfig(aberration_model="kolmogorov")`` reads it.

How it is computed
==================
``K`` Zernike doesn't fit in an atmosphere, so the two ingredients come from
opposite ends:

1. **Spatial covariance of the phase** from OOPAO's own von Karman formula,
   :func:`OOPAO.phaseStats.makeCovarianceMatrix`.  It is evaluated at the pupil
   sample coordinates that :class:`ml.sim2f.optics.Sim2f` actually uses, so the
   discrete pupil is the same one the model sees.  OOPAO takes the coordinates
   as **complex numbers** ``x + 1j*y`` (``|z_i - z_j|`` *is* the separation);
   passing two real arrays instead silently builds an x-vs-y cross matrix that
   is not even positive semi-definite.  That is measured, not guessed: the real
   coordinates give a matrix whose minimum eigenvalue is ``-10.6`` while the
   phase variance is ``436 rad**2``.
2. **Projection onto the canonical basis.**  ``a = pinv(B) phi`` with ``B`` the
   Noll ``4 .. n_modes+1`` basis sampled on the same pupil samples, taken from
   :meth:`ml.sim2f.optics.Sim2f.zernike_basis` -- i.e. the *model's* basis, not a
   second Zernike implementation (the repo forbids one).  Then
   ``Sigma = A C A^T`` with ``A = pinv(B)``.

Two details that are deliberate:

* **Piston is removed from ``C``** before projecting (rank-1 projection onto the
  orthogonal complement of the constant).  Piston is in the phase but not in the
  basis, and its variance diverges as ``L0 -> inf``; removing it is what makes
  the projected covariance mean the Zernike coefficients and not "coefficients
  plus whatever piston leaked through the discrete non-orthogonality".
* **The matrix is expressed at the sim2f wavelength.**  OOPAO's convention is a
  fixed 500 nm; phase is ``2*pi*OPD/lam``, so ``rad**2`` scales by
  ``(lam_ref/lam)**2``.  Applying it makes the stored numbers honest radians at
  the wavelength the bench runs at.  It is a **global scalar**, so it cannot
  change the drawn shape, and ``draw_aberration`` renormalises the amplitude
  anyway -- the knob the corpus was generated with is untouched.

Determinism
===========
The default estimator is **analytic** (OOPAO's closed-form von Karman
covariance), so there is no Monte Carlo noise and no dependence on the random
state: rerunning this script is byte-identical -- both the ``.npz`` payload and
the provenance JSON -- which is why ``--check`` can compare a SHA-256 of the
float64 payload.  No wall-clock or timestamp is written into the JSON for the
same reason (timings go to stdout).  ``--estimator sampled`` is the alternative: it
draws ``--n-samples`` OOPAO phase screens (``Atmosphere.generateNewPhaseScreen``)
and estimates the same matrix empirically.  It exists as a cross-check, not as
the asset source: its output carries sampling noise and depends on the
FFT backend, so its SHA is not reproducible across CPU/GPU hosts.

Usage
=====
    python scripts/ml/build_sim2f_aberration_covariance.py
    python scripts/ml/build_sim2f_aberration_covariance.py --check

``--check`` recomputes the matrix and compares it against the SHA recorded in
the provenance JSON, and also verifies the ``.npz`` on disk still hashes to that
SHA.  The SHA is byte-exact within one BLAS/platform; on a mismatch the two
matrices' maximum relative deviation is printed so numerical drift (last-ulp
LAPACK differences) can be told apart from a real divergence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ao_shaping.drivers.sim._oopao_compat import (
    Atmosphere,
    Source,
    Telescope,
    as_numpy,
    phaseStats,
)
from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.optics import Sim2f

#: Where the asset lives.  Anchored to the package (not the CWD) because
#: ``draw_aberration`` loads it through ``Path(__file__).parent``.
ASSET_DIR = ROOT / "src" / "ml" / "sim2f" / "assets"
ASSET_STEM = "zernike_cov_kolmogorov"

#: OOPAO's hard-wired wavelength convention for ``r0`` / ``L0`` / the von Karman
#: spectrum (``Atmosphere.wavelength = 500 nm``).  Not a knob.
LAMBDA_REF_M = 500e-9

#: Fried parameter of the ensemble, metres at 500 nm.  OOPAO's own ``r0_def``;
#: it is a pure multiplicative factor on the whole covariance, so it cannot
#: change the drawn *shape*, only the reported rad**2 scale.
DEFAULT_R0_M = 0.15

#: Outer scale, metres.  ``L0/D = 4600`` at the sim2f geometry, so the ensemble
#: is the Kolmogorov limit of von Karman; a smaller ``L0`` would suppress the
#: lowest orders.
DEFAULT_L0_M = 25.0


def _pkg_version(name: str) -> str | None:
    """Installed version of ``name``, or ``None`` when it is not installed.

    Reads metadata rather than importing: importing ``cupy`` costs ~1 s and
    prints a ``cupyx.jit`` experimental warning, and the only thing wanted here
    is the version string.

    Args:
        name: Distribution name.

    Returns:
        The version string, or ``None``.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cupy_version() -> str | None:
    """Version of the installed CuPy wheel, whatever it is called.

    There is no distribution named ``cupy``: the wheel is ``cupy-cuda12x`` /
    ``cupy-cuda13x`` / ``cupy-rocm-*`` depending on the build, so
    :func:`_pkg_version` alone reports nothing even though CuPy is installed and
    OOPAO is using it.

    Returns:
        The version string, or ``None`` when no CuPy wheel is present.
    """
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name == "cupy" or name.startswith("cupy-"):
            return dist.version
    return None


def _backend_record() -> dict[str, object]:
    """The array backend OOPAO actually bound in this process.

    ``OOPAO_BACKEND`` and ``OOPAO_GPU_RESIDENT`` are read once, at OOPAO's first
    use, so which module did the maths is a property of the **process**, not of
    this script's arguments -- and ``auto`` silently selects CuPy on a CUDA host.
    Recording that next to the CuPy *version* is the difference between "CuPy is
    installed" and "CuPy ran": for the analytic estimator the answer is always
    ``numpy`` (``makeCovarianceMatrix`` is closed-form NumPy), and knowing that is
    what stops a future reader from blaming the GPU for a numerical drift.

    Returns:
        ``array_module`` / ``gpu`` / ``gpu_resident`` of the live process, plus
        the two environment values and the CuPy wheel version. The three runtime
        fields are ``None`` when ``OOPAO.runtime`` was never imported (should not
        happen here -- the module-level ``_oopao_compat`` import loads it).
    """
    record: dict[str, object] = {
        "oopao_backend_env": os.environ.get("OOPAO_BACKEND", "<unset = auto>"),
        "oopao_gpu_resident_env": os.environ.get("OOPAO_GPU_RESIDENT", "<unset = 0>"),
        "cupy": _cupy_version(),
        "array_module": "unknown",
        "gpu": None,
        "gpu_resident": None,
    }
    runtime = sys.modules.get("OOPAO.runtime")
    if runtime is None:  # pragma: no cover - _oopao_compat imports it
        return record
    xp, gpu = runtime.array_backend()
    record["array_module"] = xp.__name__
    record["gpu"] = bool(gpu)
    record["gpu_resident"] = bool(runtime.gpu_resident())
    return record


def _oopao_commit() -> str:
    """``git rev-parse HEAD`` of the ``libs/OOPAO`` submodule.

    Returns:
        The full commit hash, or ``"unknown"`` when git cannot answer (tarball
        checkout, no git binary).
    """
    checkout = ROOT / "libs" / "OOPAO"
    if not checkout.is_dir():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _shim_atmosphere(
    *, grid: int, aperture_m: float, r0_m: float, l0_m: float
) -> Atmosphere:
    """A minimal OOPAO ``Atmosphere`` that only carries the von Karman parameters.

    :func:`OOPAO.phaseStats.makeCovarianceMatrix` reads ``atm.L0``,
    ``atm.r0_def`` and ``atm.param`` and nothing else, so the layers are never
    built here: ``initializeAtmosphere`` would generate phase screens (and, with
    ``compute_covariance=True``, a dense ``layer x layer`` matrix pair) that this
    script does not use.  The ``Telescope``/``Source`` exist because OOPAO's
    constructor insists on them.

    OOPAO prints a banner from both constructors; stdout is captured so a
    regeneration does not look like a crash report.

    Args:
        grid: Pupil grid side, model samples.
        aperture_m: Diameter of the pupil the OOPAO telescope represents.  For a
            *sampled* estimator this must be the **grid span**
            (``_grid_span_m``), because OOPAO's layer pixel pitch is
            ``aperture_m / grid`` while the screens are indexed with the model's
            grid; only the analytic estimator is indifferent (it reads
            ``atm.r0_def`` / ``atm.L0`` / ``atm.param`` and nothing else).
        r0_m: Fried parameter, metres at 500 nm.
        l0_m: Outer scale, metres.

    Returns:
        The atmosphere carrier.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        telescope = Telescope(
            resolution=int(grid),
            diameter=float(aperture_m),
            fov=0.0,
            samplingTime=0.001,
        )
        source = Source(optBand="R", magnitude=0.0, display_properties=False)
        source * telescope
        atm = Atmosphere(
            telescope,
            r0=float(r0_m),
            L0=float(l0_m),
            windSpeed=[10.0],
            fractionalR0=[1.0],
            windDirection=[0.0],
            altitude=[0.0],
            src=source,
        )
    return atm


def _grid_span_m(sim: Any) -> float:
    """Physical extent of the model's pupil grid, metres.

    ``Sim2fConfig`` states the pupil pitch **indirectly**
    (``pupil_pitch_m = region_span_px / grid * panel_pixel``), and the Zernike disc
    fills only part of it (``pupil_radius_px`` of ``grid/2``).  That distinction is
    load-bearing for the sampled estimator: OOPAO's ``Telescope(resolution=n,
    diameter=D)`` gives its phase screens a pixel pitch of ``D/n``, and the screens
    are then indexed with the **model's** grid, so ``D`` must be the grid span --
    ``grid * pupil_pitch_m`` -- not the disc diameter.  Feeding the disc diameter
    samples the screens 4/3 finer than the basis they are projected onto and
    underestimates every Zernike variance by ``(3/4)**(5/3) = 0.62`` (measured:
    diagonal ratio 0.622 before, 1.005 after).

    Args:
        sim: The bench.

    Returns:
        ``grid * pupil_pitch_m``, metres.
    """
    return float(sim.grid) * float(sim.cfg.pupil_pitch_m)


def _pupil_geometry(
    sim: Sim2f, n_modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the model's Noll basis ``4 .. n_modes+1`` on the pupil samples.

    Args:
        sim: The bench whose grid / support / basis define the geometry.
        n_modes: ``AberrationConfig.n_modes``; the covariance covers Noll
            ``4 .. n_modes + 1``, i.e. ``coeffs[2:]``.

    Returns:
        ``(coords, basis, noll, support)``: complex ``x + 1j*y`` pupil
        coordinates in metres ``(n_pix,)``, the sampled basis ``(n_pix, K)``,
        the Noll indices ``(K,)``, and the boolean support mask ``(grid, grid)``.
    """
    grid = int(sim.grid)
    if n_modes <= 2:
        raise ValueError(f"n_modes must be > 2 to have high-order modes, got {n_modes}")

    support = np.asarray(sim.pupil_mask.cpu().numpy()[0, 0] > 0.5)
    rows, cols = np.nonzero(support)
    pitch_m = float(sim.cfg.pupil_pitch_m)
    axis_px = np.arange(grid, dtype=np.float64) - (grid - 1) / 2.0
    axis_m = axis_px * pitch_m
    # OOPAO's makeCovarianceMatrix takes the points as complex numbers: its
    # `bsxfunMinus` + `abs` is a Euclidean distance only for z = x + 1j*y.
    coords = axis_m[cols] + 1j * axis_m[rows]

    # Rows of `zernike_basis` are Noll 1..count; `aberration()` maps coefficient
    # row i to basis row i + 1, so coeffs[2:] (Noll 4..) is basis rows 3..3+K.
    noll = np.arange(4, n_modes + 2, dtype=np.int64)
    raw = sim.zernike_basis(int(noll[-1])).cpu().numpy()
    basis = np.ascontiguousarray(raw[noll - 1][:, rows, cols].T, dtype=np.float64)
    return coords, basis, noll, support


def _remove_piston(cov: np.ndarray) -> np.ndarray:
    """Project a phase covariance onto the orthogonal complement of piston.

    ``C_c = M C M`` with ``M = I - 11^T/N``, expanded so no ``(N, N)`` matrix is
    ever formed.

    Args:
        cov: ``(N, N)`` phase covariance, ``rad**2``.

    Returns:
        The piston-removed covariance, same shape.
    """
    n = cov.shape[0]
    row_mean = cov.sum(axis=1) / n
    grand = cov.sum() / (n * n)
    return cov - row_mean[:, None] - row_mean[None, :] + grand


def _phase_covariance(coords: np.ndarray, atm: Atmosphere) -> np.ndarray:
    """OOPAO's von Karman phase covariance at ``coords``, ``rad**2`` at 500 nm.

    Args:
        coords: Complex pupil sample coordinates, metres.
        atm: The parameter carrier from :func:`_shim_atmosphere`.

    Returns:
        ``(N, N)`` covariance.
    """
    return np.asarray(phaseStats.makeCovarianceMatrix(coords, coords, atm))


def _project(phase_cov: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Zernike coefficient covariance ``A C A^T`` with ``A = pinv(B)``.

    ``pinv`` *is* the least-squares coefficient recovery, so this is the exact
    discrete projection the model's own ``fit`` would perform on a phase map.

    Args:
        phase_cov: ``(N, N)`` piston-removed phase covariance, ``rad**2``.
        basis: ``(N, K)`` basis samples.

    Returns:
        ``(K, K)`` symmetric coefficient covariance.
    """
    projector = np.linalg.pinv(basis)
    sigma = projector @ phase_cov @ projector.T
    return 0.5 * (sigma + sigma.T)


def analytic_covariance(
    sim: Sim2f, n_modes: int, *, r0_m: float, l0_m: float, lam_m: float
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Closed-form von Karman Zernike covariance for this bench.

    Args:
        sim: The bench (defines grid, support and basis).
        n_modes: Modes covered, Noll ``4 .. n_modes + 1``.
        r0_m: Fried parameter, metres at 500 nm.
        l0_m: Outer scale, metres.
        lam_m: Wavelength the returned ``rad**2`` are expressed at.

    Returns:
        ``(sigma, noll, info)`` -- the ``(K, K)`` covariance, the Noll indices,
        and a dict of diagnostic numbers for the provenance file.
    """
    coords, basis, noll, _ = _pupil_geometry(sim, n_modes)
    atm = _shim_atmosphere(
        grid=int(sim.grid),
        aperture_m=_grid_span_m(sim),
        r0_m=r0_m,
        l0_m=l0_m,
    )
    start = time.perf_counter()
    phase_cov = _phase_covariance(coords, atm)
    cov_seconds = time.perf_counter() - start

    phase_cov = _remove_piston(phase_cov)
    sigma = _project(phase_cov, basis) * (LAMBDA_REF_M / float(lam_m)) ** 2
    print(f"phase covariance {phase_cov.shape} in {cov_seconds:.2f} s")
    return sigma, noll, {
        "n_pupil_samples": int(coords.size),
        "phase_variance_rad2_at_500nm_piston_removed": float(phase_cov[0, 0]),
        # A least-squares round trip: inject a known coefficient vector, rebuild
        # its phase, recover it.  Not a formality -- it is what proves the
        # sampled basis and the sampled coordinates describe the same pupil.
        "basis_roundtrip_max_abs_err": _roundtrip_error(basis),
    }


def _roundtrip_error(basis: np.ndarray) -> float:
    """Recover an injected coefficient vector and report the worst error.

    Args:
        basis: ``(N, K)`` basis samples.

    Returns:
        ``max |a_recovered - a_injected|`` in radians.
    """
    injected = np.linspace(-1.0, 1.0, basis.shape[1])
    recovered = np.linalg.pinv(basis) @ (basis @ injected)
    return float(np.max(np.abs(recovered - injected)))


def sampled_covariance(
    sim: Sim2f,
    n_modes: int,
    *,
    r0_m: float,
    l0_m: float,
    lam_m: float,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate the same covariance from ``n_samples`` OOPAO phase screens.

    This is the cross-check mode.  It draws screens with
    ``Atmosphere.generateNewPhaseScreen`` (the same generator the repository's
    OOPAO backend uses), projects each onto the same basis, and accumulates the
    second moment.  Sampling noise scales as ``sqrt(2 / n_samples)`` per entry,
    so this output is *not* what the asset is built from.

    Args:
        sim: The bench.
        n_modes: Modes covered.
        r0_m: Fried parameter, metres at 500 nm.
        l0_m: Outer scale, metres.
        lam_m: Wavelength the returned ``rad**2`` are expressed at.
        n_samples: Number of independent screens.
        seed: Seed of the first screen; screen ``i`` uses ``seed + i``.

    Returns:
        ``(sigma, noll, info)``.
    """
    if n_samples < 2 * n_modes:
        raise ValueError(
            f"n_samples={n_samples} is too small to estimate a {n_modes - 2}x"
            f"{n_modes - 2} covariance; use at least {2 * n_modes}"
        )
    coords, basis, noll, support = _pupil_geometry(sim, n_modes)
    atm = _shim_atmosphere(
        grid=int(sim.grid),
        aperture_m=_grid_span_m(sim),
        r0_m=r0_m,
        l0_m=l0_m,
    )
    projector = np.linalg.pinv(basis)

    with contextlib.redirect_stdout(io.StringIO()):
        atm.initializeAtmosphere(atm.telescope, compute_covariance=False)

    acc = np.zeros((noll.size, noll.size), dtype=np.float64)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(int(n_samples)):
            atm.generateNewPhaseScreen(seed=int(seed) + i)
            # CuPy 数组不能隐式转 NumPy (``OOPAO_GPU_RESIDENT=1`` 时层相位可能留在
            # 设备上), 所以经 as_numpy 统一取回主机 —— 见 drivers/sim/AGENTS.md。
            screen = as_numpy(atm.layer_1.OPD)[2:-2, 2:-2]
            phase = np.asarray(screen, dtype=np.float64)[support]
            coeffs = projector @ (phase - phase.mean())
            acc += np.outer(coeffs, coeffs)
    elapsed = time.perf_counter() - start

    sigma = acc / float(n_samples) * (LAMBDA_REF_M / float(lam_m)) ** 2
    print(f"{n_samples} screens in {elapsed:.2f} s")
    return 0.5 * (sigma + sigma.T), noll, {
        "n_pupil_samples": int(coords.size),
        "basis_roundtrip_max_abs_err": _roundtrip_error(basis),
    }


def _sha256(sigma: np.ndarray) -> str:
    """SHA-256 of the float64 payload, little-endian canonical form.

    Args:
        sigma: The covariance.

    Returns:
        Hexadecimal digest.
    """
    payload = np.ascontiguousarray(sigma, dtype="<f8")
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    Every geometric default is read from the config dataclasses rather than
    written as a literal here: this repository once shipped a CLI literal that
    silently overrode a calibrated dataclass default.

    Returns:
        The parser.
    """
    bench = Sim2fConfig()
    aberration = AberrationConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--n-modes",
        type=int,
        default=aberration.n_modes,
        help="modes covered, Noll 4..n_modes+1 (default: AberrationConfig.n_modes)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=bench.grid,
        help="pupil grid side (default: Sim2fConfig.grid)",
    )
    parser.add_argument(
        "--pupil-radius-px",
        type=float,
        default=bench.pupil_radius_px,
        help="Zernike support radius in grid samples (default: Sim2fConfig)",
    )
    parser.add_argument(
        "--pupil-pitch-m",
        type=float,
        default=bench.pupil_pitch_m,
        help="model sample pitch in metres (default: Sim2fConfig)",
    )
    parser.add_argument("--r0-m", type=float, default=DEFAULT_R0_M, help="Fried parameter, m at 500 nm")
    parser.add_argument("--l0-m", type=float, default=DEFAULT_L0_M, help="outer scale, m")
    parser.add_argument(
        "--estimator",
        choices=("analytic", "sampled"),
        default="analytic",
        help="'analytic' (default, deterministic) or 'sampled' (Monte Carlo cross-check)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="screens to draw for --estimator sampled (0 for analytic)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first screen seed for 'sampled'")
    parser.add_argument(
        "--wavelength-m",
        type=float,
        default=bench.wavelength_m,
        help="wavelength the stored rad**2 are expressed at (default: Sim2fConfig)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare against the SHA in the provenance JSON; write nothing",
    )
    return parser


def _build_bench(args: argparse.Namespace) -> Sim2f:
    """Rebuild the bench from the CLI overrides.

    ``Sim2fConfig`` states the pupil pitch *indirectly*
    (``pupil_pitch_m = region_span_px / grid * panel_pixel``), so overriding the
    pitch means inverting that relation rather than assigning a field.

    Args:
        args: Parsed arguments.

    Returns:
        A float64 CPU bench at the requested geometry.
    """
    base = Sim2fConfig()
    span_px = float(args.grid) * float(args.pupil_pitch_m) / (
        float(base.panel_pixel_um) * 1e-6
    )
    cfg = Sim2fConfig(
        grid=int(args.grid),
        region_span_px=span_px,
        pupil_fill=2.0 * float(args.pupil_radius_px) / float(args.grid),
    )
    return Sim2f(cfg, device="cpu", dtype=torch.float64)


def main(argv: list[str] | None = None) -> int:
    """Regenerate (or verify) the covariance asset.

    Args:
        argv: Arguments after the script name; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 when ``--check`` fails).
    """
    args = _build_parser().parse_args(argv)
    reference = Sim2fConfig()
    if (int(args.grid), float(args.pupil_radius_px)) != (
        reference.grid,
        reference.pupil_radius_px,
    ):
        # The asset name does not encode the geometry, so a non-reference run
        # overwrites the packaged file.  Say so: the runner-up mistake here is a
        # silent clobber that only shows up as a slightly wrong pupil.
        print(
            f"note: grid={args.grid}/radius={args.pupil_radius_px} differs from the "
            f"reference geometry (grid={reference.grid}, "
            f"radius={reference.pupil_radius_px}); this rewrites the packaged asset "
            f"{ASSET_STEM}_n{int(args.n_modes)}.npz, and only the provenance JSON "
            "records the geometry it was built for",
            file=sys.stderr,
        )
    if args.estimator == "analytic" and args.n_samples:
        print("note: --n-samples is ignored by --estimator analytic", file=sys.stderr)
    if args.estimator == "sampled" and args.n_samples < 1:
        print("error: --estimator sampled needs --n-samples > 0", file=sys.stderr)
        return 2

    sim = _build_bench(args)
    started = time.perf_counter()
    if args.estimator == "analytic":
        sigma, noll, info = analytic_covariance(
            sim,
            int(args.n_modes),
            r0_m=float(args.r0_m),
            l0_m=float(args.l0_m),
            lam_m=float(args.wavelength_m),
        )
    else:
        sigma, noll, info = sampled_covariance(
            sim,
            int(args.n_modes),
            r0_m=float(args.r0_m),
            l0_m=float(args.l0_m),
            lam_m=float(args.wavelength_m),
            n_samples=int(args.n_samples),
            seed=int(args.seed),
        )
    elapsed = time.perf_counter() - started

    digest = _sha256(sigma)
    asset = ASSET_DIR / f"{ASSET_STEM}_n{int(args.n_modes)}.npz"
    provenance = ASSET_DIR / f"{ASSET_STEM}_n{int(args.n_modes)}.json"
    record: dict[str, object] = {
        "asset": asset.name,
        "covariance_sha256": digest,
        "estimator": "analytic-von-karman" if args.estimator == "analytic" else "monte-carlo",
        "n_samples": int(args.n_samples) if args.estimator == "sampled" else 0,
        "seed": int(args.seed) if args.estimator == "sampled" else None,
        "turbulence": {
            "r0_m_at_500nm": float(args.r0_m),
            "L0_m": float(args.l0_m),
            "wavelength_rad2_expressed_at_m": float(args.wavelength_m),
            "oopao_reference_wavelength_m": LAMBDA_REF_M,
        },
        "sim2f": {
            "n_modes": int(args.n_modes),
            "noll": [int(v) for v in noll],
            "grid": int(args.grid),
            "pupil_radius_px": float(args.pupil_radius_px),
            "pupil_pitch_m": float(args.pupil_pitch_m),
            "aperture_m": float(sim.cfg.aperture_m),
            **info,
        },
        "oopao": {
            "commit": _oopao_commit(),
            "path": "libs/OOPAO",
            "imported_via": "ao_shaping.drivers.sim._oopao_compat",
        },
        "backend": _backend_record(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": _pkg_version("numpy"),
            "cupy": _cupy_version(),
            "platform": sys.platform,
        },
    }

    if args.check:
        return _check(asset, provenance, sigma, digest, record)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        asset,
        covariance=sigma,
        noll=noll,
        grid=np.int64(args.grid),
        pupil_radius_px=np.float64(args.pupil_radius_px),
        wavelength_m=np.float64(args.wavelength_m),
    )
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(sigma, noll, elapsed, record["backend"])
    print(f"wrote {asset.relative_to(ROOT)}")
    print(f"wrote {provenance.relative_to(ROOT)}")
    print(f"sha256 {digest}")
    return 0


def _check(
    asset: Path,
    provenance: Path,
    sigma: np.ndarray,
    digest: str,
    record: dict[str, object],
) -> int:
    """Compare a fresh computation against the recorded asset.

    Args:
        asset: The ``.npz`` on disk.
        provenance: The ``.json`` on disk.
        sigma: Freshly computed covariance.
        digest: SHA of ``sigma``.
        record: The provenance dict a rewrite would produce.

    Returns:
        Exit code: 0 when everything matches, 1 otherwise.
    """
    if not provenance.is_file():
        print(f"FAIL: provenance missing: {provenance}", file=sys.stderr)
        return 1
    stored_record = json.loads(provenance.read_text(encoding="utf-8"))
    if stored_record.get("covariance_sha256") != digest:
        print(
            f"FAIL: recomputed covariance differs from the SHA in {provenance.name}\n"
            f"      stored {stored_record.get('covariance_sha256')}\n"
            f"      fresh  {digest}",
            file=sys.stderr,
        )
        return 1
    if not asset.is_file():
        print(f"FAIL: asset missing: {asset}", file=sys.stderr)
        return 1
    with np.load(asset) as data:
        on_disk = np.asarray(data["covariance"], dtype=np.float64)
    disk_digest = _sha256(on_disk)
    if disk_digest != digest:
        worst = float(np.max(np.abs(on_disk - sigma)) / np.max(np.abs(sigma)))
        print(
            f"FAIL: {asset.name} does not hash to the recorded SHA "
            f"({disk_digest} vs {digest}); max relative deviation {worst:.3e}",
            file=sys.stderr,
        )
        return 1

    differing = sorted(
        key
        for key, value in record.items()
        if key not in ("asset", "covariance_sha256") and stored_record.get(key) != value
    )
    if differing:
        print(
            "note: provenance fields differ from a fresh run (re-run without "
            f"--check to refresh): {', '.join(differing)}"
        )
    print(f"OK: {asset.relative_to(ROOT)} matches sha256 {digest}")
    return 0


def _print_summary(
    sigma: np.ndarray, noll: np.ndarray, elapsed: float, backend: dict[str, object]
) -> None:
    """Print the mode-variance ladder, the strongest correlations and the backend.

    Args:
        sigma: The covariance.
        noll: Noll indices.
        elapsed: Wall clock of the construction, seconds.
        backend: The :func:`_backend_record` block, so the summary says which array
            module actually did the maths in this run.
    """
    diag = np.diag(sigma)
    off = sigma / np.sqrt(np.outer(diag, diag))
    order = np.argsort(np.abs(off - np.eye(noll.size)).ravel())[::-1]
    print(f"K={noll.size} Noll {int(noll[0])}..{int(noll[-1])}  built in {elapsed:.2f} s")
    print(
        "backend: array module {} (gpu={}, gpu_resident={}; OOPAO_BACKEND={}, "
        "OOPAO_GPU_RESIDENT={}; cupy {})".format(
            backend.get("array_module"),
            backend.get("gpu"),
            backend.get("gpu_resident"),
            backend.get("oopao_backend_env"),
            backend.get("oopao_gpu_resident_env"),
            backend.get("cupy"),
        )
    )
    print("mode variance (rad^2) and share of the diagonal:")
    total = float(diag.sum())
    for index, value in enumerate(diag):
        print(f"  Noll {int(noll[index]):3d}  {value:.4e}  {100.0 * value / total:5.1f}%")
    print("strongest |correlation|:")
    seen: set[tuple[int, int]] = set()
    for flat in order:
        i, j = divmod(int(flat), noll.size)
        if i >= j or (i, j) in seen:
            continue
        seen.add((i, j))
        if abs(off[i, j]) < 0.05:
            break
        print(f"  Noll {int(noll[i]):3d} / Noll {int(noll[j]):3d}  {off[i, j]:+.3f}")
        if len(seen) >= 12:
            break


if __name__ == "__main__":
    raise SystemExit(main())
