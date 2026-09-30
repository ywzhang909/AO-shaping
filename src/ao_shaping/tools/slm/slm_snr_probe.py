"""Device-agnostic perturbation-sensitivity (SNR) measurement for the SLM bench.

**Pure measurement + math, no device construction.** Every entry point takes
already-opened device *instances* (anything with the ``BaseCamera`` / ``Santec``
duck-type), so the same code serves the Daheng bench, a MiiCam, a simulation
camera, or a fake in a unit test. Nothing here imports a driver, opens a
device, or reads ``.env``.

Why this exists
---------------
SPGD can only optimise if it can *see* its own gradient. That requires

    SNR = ΔJ_signal / σ_noise  ≳ 2–3

where ``σ_noise`` is the objective's fluctuation at a **fixed** phase. Three
bench-measured facts (Daheng MER2-507-23GM NIR + Santec SLM-200, 2026-09-29)
shaped the API:

1. **The noise floor is not a constant of the bench.** Three sweeps on one day,
   same spot, gave ``σ = 3.7e-4``, ``8.2e-5`` and ``1.7e-5`` — a 21x spread.
   It must be measured per session, never assumed or hard-coded.
2. **A single-mode SNR does not transfer to SPGD.** SPGD perturbs *every* mode
   by a random ±1 pattern, which dilutes the coherent signal. Measured at
   ``delta=0.0005``: single-mode SNR 2.25 (usable) vs 54-DOF SNR 1.34
   (unusable) — which is exactly why a run gated ~95% of its epochs.
   :func:`snr_sweep` therefore measures both.
3. **Drift, not shot noise, dominates.** Averaging 40 frames instead of 10 did
   *not* reduce σ (2.8e-3 vs 1.7e-3). Plain frame averaging cannot fix it, so
   the signal is measured with **ABBA pairs** that return to the flat phase
   between signs, cancelling drift common to both.

Public symbols
--------------
- ``snr_verdict``        — threshold → strong / usable / unusable
- ``measure_noise_floor``— σ of a score at a fixed state
- ``abba_signal``        — drift-cancelled ΔJ for one perturbation direction
- ``snr_sweep``          — noise floor + per-delta SNR, single- and multi-mode
- ``SnrSweepResult``     — dataclass returned by :func:`snr_sweep`

Typical use::

    with create_camera("daheng", cam_id=0, exposure_time_ms=1.2) as cam, \\
         Santec(slm_number=1, wavelength=1064) as slm:
        result = snr_sweep(cam, slm, n_max=9, radius=480.0)
        print(result.verdicts)
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.targets import roi_pib_metric, target_shape_roi
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_calc import zernike_modes
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

#: SNR thresholds. ``>= 3`` is comfortable, ``>= 2`` is workable, below that the
#: gradient is buried and a run will gate nearly every epoch.
SNR_STRONG = 3.0
SNR_USABLE = 2.0

#: σ at or below this is treated as *no measurable noise* (a perfectly static
#: bench, or a synthetic camera). Without the floor, ``signal / sigma`` would
#: divide by ~1e-16 and report an absurd SNR from float round-off. Such a
#: measurement carries no information about a real bench.
SIGMA_FLOOR = 1e-12

#: Default perturbation amplitudes (rad) probed when the caller does not pass any.
DEFAULT_DELTAS: tuple[float, ...] = (0.0005, 0.001, 0.01, 0.05, 0.1, 0.2)

#: Reference Zernike mode for the single-mode probe: defocus (2, 0).
REFERENCE_MODE: tuple[int, int] = (2, 0)


def snr_verdict(snr: float) -> str:
    """Classify an SNR into ``strong`` / ``usable`` / ``unusable``."""
    if not np.isfinite(snr):
        return "unusable"
    if snr >= SNR_STRONG:
        return "strong"
    if snr >= SNR_USABLE:
        return "usable"
    return "unusable"


def measure_noise_floor(
    score: Callable[[], float],
    n_frames: int = 15,
) -> tuple[float, list[float]]:
    """Standard deviation of ``score`` at a *fixed* device state.

    The caller is responsible for leaving the devices untouched between calls
    (e.g. a flat phase already written). This measures the combined shot noise
    **and** slow drift, which is what a single SPGD step actually sees.

    Args:
        score: Zero-arg callable returning one scalar objective value.
        n_frames: Number of consecutive evaluations.

    Returns:
        ``(sigma, values)`` — the sample standard deviation and the raw series.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    values = [float(score()) for _ in range(n_frames)]
    return float(np.std(values)), values


def abba_signal(
    score: Callable[[], float],
    write_phase: Callable[[float], None],
    write_reference: Callable[[], None],
    amplitude: float,
    pairs: int = 3,
) -> float:
    """Drift-cancelled ``ΔJ`` for one perturbation amplitude.

    Each repetition uses the palindrome ``+ - - +``. That ordering is what makes
    the drift cancellation work: the two ``+`` samples sit at the outer time
    positions and the two ``-`` samples at the inner ones, so for *any* drift
    that is locally linear in time the two means carry an identical drift term
    and it drops out of ``|mean(+) - mean(-)|``.

    The superficially similar ``+ - + -`` ordering does **not** cancel: with a
    drift of ``c·t`` it leaves a residual of ``c`` per pair, which on this bench
    is the same order as the signal being measured. Frame averaging cannot fix
    this either, because the dominant residual is slow intensity drift rather
    than per-frame shot noise.

    Args:
        score: Zero-arg callable returning one scalar objective value.
        write_phase: ``amplitude -> None``; writes that coefficient value.
        write_reference: ``-> None``; restores the flat/reference state.
        amplitude: Perturbation magnitude (rad).
        pairs: Number of repetitions to average.

    Returns:
        Mean ``|mean(+) - mean(-)|`` over the repetitions.
    """
    if pairs < 1:
        raise ValueError(f"pairs must be >= 1, got {pairs}")
    # Palindrome, not ABAB: see the docstring.
    signs = (1.0, -1.0, -1.0, 1.0)
    diffs: list[float] = []
    for _ in range(pairs):
        plus: list[float] = []
        minus: list[float] = []
        for sign in signs:
            write_phase(sign * amplitude)
            (plus if sign > 0 else minus).append(float(score()))
            write_reference()
        diffs.append(abs(float(np.mean(plus)) - float(np.mean(minus))))
    return float(np.mean(diffs))


@dataclass
class SnrSweepResult:
    """Outcome of :func:`snr_sweep`."""

    #: 0-order centre (x, y) the ROI was anchored on, window-local.
    center: tuple[float, float]
    #: σ of the objective at a fixed state.
    sigma: float
    #: Raw noise-floor series.
    noise_values: list[float] = field(default_factory=list)
    #: ``str(amplitude) -> ΔJ`` for the single-mode probe.
    single_signals: dict[str, float] = field(default_factory=dict)
    #: ``str(amplitude) -> ΔJ`` for the multi-mode (SPGD-style) probe.
    multi_signals: dict[str, float] = field(default_factory=dict)
    #: Number of perturbed DOF in the multi-mode probe.
    n_dof: int = 0
    seed: int | None = None
    #: Fraction of the frame's light inside the scored ROI. A very small value
    #: means the ROI geometry does not match the bench (typically the camera was
    #: never windowed around the spot) and the SNRs above are not meaningful.
    roi_fraction: float = 0.0

    @property
    def _denom(self) -> float:
        """σ with a floor, so a perfectly static bench cannot yield inf SNR."""
        return self.sigma if self.sigma > SIGMA_FLOOR else 0.0

    @property
    def single_snrs(self) -> dict[str, float]:
        return {
            k: (v / self._denom if self._denom else 0.0)
            for k, v in self.single_signals.items()
        }

    @property
    def multi_snrs(self) -> dict[str, float]:
        return {
            k: (v / self._denom if self._denom else 0.0)
            for k, v in self.multi_signals.items()
        }

    @property
    def verdicts(self) -> dict[str, str]:
        """Multi-mode (SPGD-relevant) verdict per amplitude."""
        return {k: snr_verdict(v) for k, v in self.multi_snrs.items()}

    def usable_deltas(self, minimum: float = SNR_USABLE) -> list[float]:
        """Amplitudes whose *multi-mode* SNR clears ``minimum``.

        This is the list to feed to ``--delta``: the single-mode column is a
        trap, since it over-reports what SPGD can actually resolve.
        """
        return sorted(
            float(k) for k, v in self.multi_snrs.items() if v >= minimum
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (used by the report generator)."""
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "sigma": self.sigma,
            "noise_mean": float(np.mean(self.noise_values)) if self.noise_values else 0.0,
            "n_dof": self.n_dof,
            "seed": self.seed,
            "roi_fraction": self.roi_fraction,
            "single_signals": self.single_signals,
            "multi_signals": self.multi_signals,
            "single_snrs": self.single_snrs,
            "multi_snrs": self.multi_snrs,
            "verdicts": self.verdicts,
            "thresholds": {"strong": SNR_STRONG, "usable": SNR_USABLE},
            "sigma_floor": SIGMA_FLOOR,
        }


def _phase_to_gray(phase: np.ndarray, slm: Any) -> np.ndarray:
    """Radian phase -> uint16 grayscale, tolerating a minimal SLM stub.

    Prefers the driver's own pipeline (``create_phase_from_array``: mod 2π +
    wavefront correction + LUT + shift, with the device's wavelength-dependent
    2π gray) so the measurement sees exactly what the optimizer would send.
    An instance that only implements ``display_data`` (a test double, or a
    reduced driver) falls back to the pure-math conversion rather than raising —
    that is what keeps this module usable without hardware.
    """
    if hasattr(slm, "create_phase_from_array"):
        return phase_to_slm_grayscale(phase, slm=slm)
    return phase_to_slm_grayscale(phase, slm=None)


def _write_zernike(
    slm: Any,
    amps: Mapping[Any, float],
    n_max: int,
    resolution: tuple[int, int],
    radius: float | None,
) -> None:
    """Write a Zernike phase (raw radians -> grayscale) through the driver."""
    # generate_zernike_phase takes a concrete dict of {(n, m) | noll | name: amp};
    # normalise at the boundary so callers may pass any mapping.
    coefficients: dict[Any, float] = {k: float(v) for k, v in amps.items()}
    phase = generate_zernike_phase(
        coefficients, resolution=resolution, n_max=n_max, radius=radius
    )
    # generate_zernike_phase returns NaN outside its inscribed aperture; the
    # optimizer's aperture-limited phase is flat there.
    phase = np.nan_to_num(phase, nan=0.0)
    slm.display_data(_phase_to_gray(phase, slm))


def snr_sweep(
    cam: Any,
    slm: Any,
    *,
    n_max: int = 9,
    radius: float | None = 480.0,
    resolution: tuple[int, int] = (1920, 1200),
    target_shape: str = "square",
    target_size: float = 50.0,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    n_frames: int = 15,
    pairs: int = 3,
    seed: int | None = 0,
    measure_single: bool = True,
    measure_multi: bool = True,
    score_fn: Callable[[np.ndarray], float] | None = None,
    window: tuple[Any, Any] | None = None,
) -> SnrSweepResult:
    """Measure the objective's noise floor and per-amplitude SNR on a live bench.

    Both device arguments are **already-open instances** — this function never
    constructs, opens or closes a device, and never reads a camera type, so it
    is equally valid for Daheng, MiiCam, a simulation camera or a test double.

    Two probes are run per amplitude:

    * **single-mode** — only :data:`REFERENCE_MODE` (defocus) is perturbed.
      This is what a naive sensitivity check measures.
    * **multi-mode** — every non-piston mode is perturbed by a random ±1
      pattern, which is what SPGD actually does.

    The gap between the two is the practical dilution factor; judge
    ``--delta`` by the multi-mode column.

    Args:
        cam: Open camera exposing ``get_numpy_image(n_sample=...)``.
        slm: Open SLM exposing ``display_data(gray)``.
        n_max: Max Zernike radial order (sets the DOF count).
        radius: Zernike aperture radius (px) on the SLM panel.
        resolution: SLM panel resolution as ``(width, height)``.
        target_shape: ROI shape for the built-in score.
        target_size: ROI size (px) for the built-in score.
        deltas: Perturbation amplitudes to probe (rad).
        n_frames: Frames for the noise floor.
        pairs: ABBA repetitions per amplitude.
        seed: RNG seed for the ±1 pattern (``None`` -> nondeterministic).
        measure_single: Run the single-mode probe.
        measure_multi: Run the multi-mode probe.
        score_fn: ``frame -> float`` overriding the built-in ``roi_pib`` score.

            **Pass the objective you actually care about.** The built-in score
            anchors a fixed ``target_size`` box on the flat-state 0-order, which
            is only meaningful once the caller has windowed the camera around the
            spot. ``create_camera(..., cam_size=N)`` does *not* window: the driver
            only calls ``reset_window`` when asked, so on a full 2592x1944 frame
            a 50 px box can hold <1 % of the light and the measured SNR collapses
            to ~1 for every amplitude. Bench-verified: that misconfiguration
            reported ``usable_deltas() == []`` while the optimizer at the same
            delta reached +18.8 % sustained improvement. When ``score_fn`` is
            given, ``target_shape``/``target_size`` are ignored.
        window: Optional ``(size, center)`` forwarded to ``cam.reset_window``
            before measuring, so the probe can window the bench itself. The
            caller still owns device configuration; this is a convenience.

    Returns:
        A populated :class:`SnrSweepResult`.
    """
    if n_max < 0:
        raise ValueError(f"n_max must be >= 0, got {n_max}")
    if deltas and all(d <= 0 for d in deltas):
        raise ValueError(f"deltas must contain a positive amplitude, got {deltas!r}")

    if window is not None and hasattr(cam, "reset_window"):
        cam.reset_window(*window)

    def write(amps: dict[tuple[int, int], float]) -> None:
        _write_zernike(slm, amps, n_max, resolution, radius)

    def write_flat() -> None:
        write({})

    # Piston (0,0) is a constant phase offset the detector cannot see; including
    # it would waste perturbation budget and dilute the multi-mode signal.
    modes = [(n, m) for n, m in zernike_modes(n_max) if (n, m) != (0, 0)]
    pattern = None
    if measure_multi and modes:
        rng = np.random.default_rng(seed)
        pattern = rng.choice([-1.0, 1.0], size=len(modes))

    # Anchor the ROI once on the flat state. Re-deriving it per frame is fine
    # (measured cost: 1.1x more noise) but a fixed anchor is what the optimizer
    # uses, so measure what the optimizer sees.
    write_flat()
    first = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
    cx, cy = (float(v) for v in zero_order_center(first))
    center: tuple[float, float] = (cx, cy)

    def score() -> float:
        frame = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
        if score_fn is not None:
            return float(score_fn(frame))
        return float(
            roi_pib_metric(
                frame, center, target_shape, target_size
            )[0]
        )

    sigma, noise_values = measure_noise_floor(score, n_frames=n_frames)

    single: dict[str, float] = {}
    multi: dict[str, float] = {}
    for delta in deltas:
        key = str(delta)
        if measure_single:
            single[key] = abba_signal(
                score,
                lambda a, _d=delta: write({REFERENCE_MODE: a}),
                write_flat,
                delta,
                pairs=pairs,
            )
        if measure_multi and pattern is not None:
            multi[key] = abba_signal(
                score,
                lambda a, _d=delta, _p=pattern: write(
                    {k: a * _d * float(v) for k, v in zip(modes, _p, strict=True)}
                ),
                write_flat,
                delta,
                pairs=pairs,
            )
    write_flat()  # leave the bench flat, not on a perturbation

    # Diagnostic: what fraction of the frame's light does the scored ROI actually
    # hold? A tiny value means the geometry is wrong (e.g. the camera was never
    # windowed around the spot) and every SNR below is meaningless, however
    # plausible the numbers look. This is the check that would have caught the
    # misconfiguration documented on ``score_fn``.
    roi_fraction = 0.0
    if score_fn is None:
        probe = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
        total = float(probe.sum())
        if total > 0:
            box = target_shape_roi(
                probe.shape, center, target_shape, target_size
            )
            roi_fraction = float(probe[box].sum()) / total

    return SnrSweepResult(
        center=center,
        sigma=sigma,
        noise_values=noise_values,
        single_signals=single,
        multi_signals=multi,
        n_dof=len(modes),
        seed=seed,
        roi_fraction=roi_fraction,
    )
