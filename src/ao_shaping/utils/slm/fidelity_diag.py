"""Seven-factor fidelity decomposition for closed-loop shaping diagnostics.

Answers "why is the measured enhancement below the theoretical ceiling?" by
splitting the gap into the seven independent factors of Mastiani, Cox &
Vellekoop, *J. Phys.: Photonics* 6(3):033003 (2024), DOI
``10.1088/2515-7647/ad5775`` (Eq. 15-44).

Pure numpy leaf module: no device, no camera, no panel geometry, no I/O — the
same contract as ``utils/math/stability.py``. Callers feed it either raw
phase-stepping frame stacks (the DFT is done here) or already-reduced scalars.

Assumptions (stated by the source paper, *not* universally true):

- ``theoretical_ceiling`` uses ``π/4·(N-1)+1``, the phase-only bound for a
  circular-Gaussian ``t_ba`` (strong diffuse scattering). For weak forward
  scattering the bound degrades; the number stays a useful upper limit only.
- ``gamma_unmodulated`` takes the small-defect root of ``r² = γ0²(1-γ0²)``;
  the complementary root is the degenerate large-defect branch.
- ``gamma_calibration`` uses the phase-response curve form. It equals the
  paper's higher-harmonic pseudo-inverse when the response is sampled at the
  stepped phases; both measure the same deviation from an ideal linear ramp.

Factors are squared moduli (``|γ|²``), each in ``[0, 1]``; ``1.0`` means "this
mechanism costs nothing". ``diagnose`` multiplies them into the coherent part
of the enhancement and reports every factor that fell below
``DEFAULT_THRESHOLD`` together with a one-line mitigation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_THRESHOLD",
    "MITIGATIONS",
    "FidelityReport",
    "diagnose",
    "gamma_amplitude_only",
    "gamma_calibration",
    "gamma_decorrelation",
    "gamma_jitter",
    "gamma_mode_coupling",
    "gamma_noise",
    "gamma_unmodulated",
    "step_dft",
    "theoretical_ceiling",
]

#: Below this a factor is reported as a real contributor to the gap. Chosen
#: from the paper's worked example (each defect there costs 5 % or more).
DEFAULT_THRESHOLD = 0.95

#: One-line remedy per factor, used by ``diagnose``. Keys must stay in sync
#: with ``FidelityReport`` field names (guarded by the test-suite).
MITIGATIONS: dict[str, str] = {
    "gamma_mode_coupling": "增大受控模式数或收紧瞳面（瞳面）区域，让更多光落入受控子空间",
    "gamma_amplitude_only": "纯相位调制的固有上限；需要复振幅调制才能继续提升",
    "gamma_unmodulated": "检查 LCoS 前表面反射与零级泄漏，必要时楔形移位或光阑遮挡",
    "gamma_calibration": "重跑 LUT 标定（slm-lut），并核对灰度→相位线性度",
    "gamma_jitter": "降低刷新闪烁：提高曝光平均帧数，或避开半帧刷新时刻采样",
    "gamma_noise": "提高信噪比：增加平均帧数、提高曝光，先减背景再算指标",
    "gamma_decorrelation": "缩短单轮耗时或做样品/光路固定，测量前后各拍参考帧监控漂移",
}


def theoretical_ceiling(n_modes: int) -> float:
    """Phase-only enhancement bound for ``n_modes`` controlled modes.

    ``π/4·(N-1)+1`` — about 21 % below ``N``. Valid for a circular-Gaussian
    ``t_ba`` (strong diffuse scattering); for weak forward scattering treat it
    as an upper limit only.

    Args:
        n_modes: Number of controlled modes (``N >= 1``).

    Returns:
        The theoretical enhancement ceiling (``1.0`` when ``N == 1``).
    """
    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be >= 1")
    return 1.0 + (np.pi / 4.0) * (n - 1)


def _unit(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(np.clip(v, 0.0, 1.0))


def gamma_mode_coupling(fraction_in_controlled_modes: float) -> float:
    """``|γ_N|²``: fraction of the focus energy inside the controlled modes.

    Args:
        fraction_in_controlled_modes: Measured (or simulated) energy fraction.

    Returns:
        The fraction itself, clipped to ``[0, 1]``.
    """
    return _unit(fraction_in_controlled_modes, "fraction_in_controlled_modes")


def gamma_amplitude_only(transmissions: Sequence[float]) -> float:
    """``|γ_A|² = ⟨|t|⟩²/⟨|t|²⟩`` for phase-only modulation.

    Args:
        transmissions: Per-pixel transmitted amplitude magnitudes ``|t|``.

    Returns:
        The amplitude-only fidelity; ``1.0`` for uniform transmission.

    Raises:
        ValueError: If the sequence is empty or holds non-finite values.
    """
    t = np.abs(np.asarray(list(transmissions), dtype=np.float64))
    if t.size == 0:
        raise ValueError("transmissions must not be empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("transmissions must be finite")
    mean_sq = float(np.mean(t) ** 2)
    sq_mean = float(np.mean(t**2))
    if sq_mean <= 0.0:
        raise ValueError("transmissions are all zero")
    return float(mean_sq / sq_mean)


def gamma_unmodulated(f0: complex, f1: complex) -> float:
    """``|γ_0|²`` from the flat + stepping DFT components (Eq. 27-30).

    Unmodulated front-surface reflection contributes ``|F1|/|F0| = γ0·√(1-γ0²)``.
    Solving for the small-defect root gives ``γ0² = (1-√(1-4r²))/2``.

    Args:
        f0: Zeroth-order DFT component (time average) of the stepped series.
        f1: First-order DFT component.

    Returns:
        The unmodulated-light fidelity ``|γ_0|²``.

    Raises:
        ValueError: If ``|F0|`` is zero or the ratio exceeds ``0.5`` (no real
            solution — the series is not a phase-stepping measurement).
    """
    a0 = abs(complex(f0))
    if a0 <= 0.0:
        raise ValueError("|F0| must be > 0")
    r = abs(complex(f1)) / a0
    if r > 0.5 + 1e-12:
        raise ValueError(f"|F1|/|F0| = {r:.4f} exceeds 0.5; not phase stepping")
    r = min(r, 0.5)
    return float((1.0 - np.sqrt(max(0.0, 1.0 - 4.0 * r * r))) / 2.0)


def gamma_calibration(
    measured_phase_rad: Sequence[float],
    ideal_phase_rad: Sequence[float] | None = None,
) -> float:
    """``|γ_φ|²``: penalty from a non-ideal grayscale→phase response.

    When the response is sampled at the stepped phases this equals the paper's
    higher-harmonic fit: both reduce to ``|⟨exp(i·Δφ)⟩|²`` over the step
    series. A constant offset is *not* a loss (it is a global phase).

    Args:
        measured_phase_rad: Measured phase at each applied step.
        ideal_phase_rad: Ideal linear ramp; defaults to ``measured`` itself
            (useful only to verify the ``1.0`` identity).

    Returns:
        The calibration fidelity in ``[0, 1]``; ``1.0`` for an ideal ramp.

    Raises:
        ValueError: On empty, mismatched or non-finite inputs.
    """
    meas = np.asarray(list(measured_phase_rad), dtype=np.float64)
    if meas.size == 0:
        raise ValueError("measured_phase_rad must not be empty")
    ideal = meas if ideal_phase_rad is None else np.asarray(list(ideal_phase_rad), dtype=np.float64)
    if ideal.shape != meas.shape:
        raise ValueError("measured_phase_rad and ideal_phase_rad must match")
    if not (np.all(np.isfinite(meas)) and np.all(np.isfinite(ideal))):
        raise ValueError("phase values must be finite")
    phasor = np.exp(1j * (meas - ideal))
    return float(min(1.0, abs(np.mean(phasor)) ** 2))


def gamma_jitter(phase_error_rad: float) -> float:
    """``|γ_J|² = (sin Δφ/Δφ)²`` for an RMS phase error ``Δφ`` (Eq. 39).

    Args:
        phase_error_rad: RMS phase error from LCOS refresh flicker, radians.

    Returns:
        The jitter fidelity in ``[0, 1]``; ``1.0`` at zero error.

    Raises:
        ValueError: If the error is negative or non-finite.
    """
    d = float(phase_error_rad)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("phase_error_rad must be finite and >= 0")
    if d == 0.0:
        return 1.0
    return float((np.sin(d) / d) ** 2)


def gamma_noise(snr: float) -> float:
    """``|γ_S|² = SNR/(1+SNR)`` for measurement noise (Eq. 40-43).

    Args:
        snr: Signal-to-noise ratio of the stepped measurement (linear).

    Returns:
        The noise fidelity in ``[0, 1)``.
    """
    s = float(snr)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("snr must be finite and >= 0")
    return float(min(1.0, s / (1.0 + s)))


def gamma_decorrelation(before: np.ndarray, after: np.ndarray) -> float:
    """``|γ_D|²`` from the Pearson correlation of reference frames (Eq. 44).

    Args:
        before: Frame captured before the measurement block.
        after: Frame captured after it, same shape.

    Returns:
        Squared Pearson coefficient in ``[0, 1]``.

    Raises:
        ValueError: On shape mismatch, empty input or a constant frame.
    """
    a = np.asarray(before, dtype=np.float64).ravel()
    b = np.asarray(after, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("before and after must have the same shape")
    if a.size == 0:
        raise ValueError("frames must not be empty")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        raise ValueError("frames are constant; correlation undefined")
    return float(min(1.0, (float(np.sum(a * b)) / denom) ** 2))


def step_dft(frames: np.ndarray) -> np.ndarray:
    """DFT of a phase-stepping frame stack along the step axis.

    Args:
        frames: Array of shape ``(n_steps, *pixel_shape)``.

    Returns:
        Complex array with the same shape; ``[0]`` is the time average,
        ``[1]`` the first harmonic carrying the modulated signal.

    Raises:
        ValueError: If fewer than 2 steps are supplied.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] < 2:
        raise ValueError("frames must be (n_steps>=2, *pixel_shape)")
    return np.fft.fft(arr, axis=0)


@dataclass(frozen=True)
class FidelityReport:
    """Per-factor result of :func:`diagnose`.

    Attributes:
        n_modes: Controlled mode count used for the ceiling.
        factors: ``|γ|²`` per mechanism, keyed like :data:`MITIGATIONS`.
        combined: Product of all factors.
        expected_enhancement: ``1 + π/4·(N-1)·combined``.
        ceiling: :func:`theoretical_ceiling` for ``n_modes``.
        warnings: Factors strictly below the threshold, with mitigations.
    """

    n_modes: int
    factors: dict[str, float] = field(default_factory=dict)
    combined: float = 1.0
    expected_enhancement: float = 1.0
    ceiling: float = 1.0
    warnings: tuple[str, ...] = ()

    @property
    def shortfall(self) -> float:
        """Enhancement points missing versus the ceiling."""
        return self.ceiling - self.expected_enhancement


def diagnose(
    n_modes: int,
    *,
    mode_fraction: float = 1.0,
    transmissions: Sequence[float] | None = None,
    f0: complex | None = None,
    f1: complex | None = None,
    measured_phase_rad: Sequence[float] | None = None,
    ideal_phase_rad: Sequence[float] | None = None,
    phase_error_rad: float = 0.0,
    snr: float | None = None,
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> FidelityReport:
    """Compose the seven factors into one enhancement budget.

    Every input is optional: a factor whose inputs are not supplied stays at
    ``1.0`` ("no evidence of a loss"), so the report is usable with whatever
    a given measurement actually provides.

    Args:
        n_modes: Controlled mode count.
        mode_fraction: ``|γ_N|²`` input.
        transmissions: ``|γ_A|²`` input.
        f0: Zeroth-order DFT component for ``|γ_0|²``.
        f1: First-order DFT component for ``|γ_0|²``.
        measured_phase_rad: ``|γ_φ|²`` input.
        ideal_phase_rad: Optional ideal ramp for ``|γ_φ|²``.
        phase_error_rad: ``|γ_J|²`` input.
        snr: ``|γ_S|²`` input.
        before: Pre-measurement reference frame for ``|γ_D|²``.
        after: Post-measurement reference frame for ``|γ_D|²``.
        threshold: Report factors below this value.

    Returns:
        The populated :class:`FidelityReport`.

    Raises:
        ValueError: If only one of ``f0``/``f1`` or of ``before``/``after`` is
            supplied, or if any factor input is invalid.
    """
    if (f0 is None) != (f1 is None):
        raise ValueError("pass both f0 and f1, or neither")
    if (before is None) != (after is None):
        raise ValueError("pass both before and after, or neither")

    factors = {"gamma_mode_coupling": gamma_mode_coupling(mode_fraction)}
    if transmissions is not None:
        factors["gamma_amplitude_only"] = gamma_amplitude_only(transmissions)
    if f0 is not None and f1 is not None:
        factors["gamma_unmodulated"] = gamma_unmodulated(f0, f1)
    if measured_phase_rad is not None:
        factors["gamma_calibration"] = gamma_calibration(measured_phase_rad, ideal_phase_rad)
    factors["gamma_jitter"] = gamma_jitter(phase_error_rad)
    if snr is not None:
        factors["gamma_noise"] = gamma_noise(snr)
    if before is not None and after is not None:
        factors["gamma_decorrelation"] = gamma_decorrelation(before, after)

    combined = float(np.prod(list(factors.values())))
    ceiling = theoretical_ceiling(n_modes)
    expected = 1.0 + (ceiling - 1.0) * combined
    warnings = tuple(
        f"{name}={value:.3f} < {threshold:.2f} — {MITIGATIONS[name]}"
        for name, value in factors.items()
        if value < threshold
    )
    return FidelityReport(
        n_modes=int(n_modes),
        factors=factors,
        combined=combined,
        expected_enhancement=float(expected),
        ceiling=float(ceiling),
        warnings=warnings,
    )
