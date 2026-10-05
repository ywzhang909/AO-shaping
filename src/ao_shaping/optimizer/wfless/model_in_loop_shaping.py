"""SLM 方形整形的 model-in-the-loop 迭代校正 (仅仿真)。

每一轮在两步之间交替:

* **Step A —— 波前拟合。** :class:`ZernikeCoefficientOptimizer` 由 (测量, 瞳孔相位)
  这一对数据拟合出静态像差。它的正向模型是规范形式
  ``|fftshift(fft2(ifftshift(U), norm="ortho))|**2``, 与
  :class:`~ao_shaping.drivers.sim.fouriergsnet_env.SimFourierGSNetEnv` **共用同一个**
  前向模型, 因此拟合出的系数可与注入的真值直接逐位比较。
* **Step B —— 自由相位整形。** 冻结拟合出的系数, 用
  :mod:`ao_shaping.algorithm.signal_processing.differentiable_shaping` 的规范可微
  损失优化整幅逐像素 SLM 相位, 使远场逼近方形目标。

由于 Step A 与数字孪生共享同一个前向模型, 在同网格下模型对孪生是**观测等价**的:
孪生的原生振幅等于 :meth:`ZernikeCoefficientOptimizer.native_amplitude`, piston 模式
的有限支撑恰好就是孪生的孔径掩模。以原生几何 (``BeamParams(native=True)``) 运行孪生
即可让 SLM 面板本身就是模型网格, 环路内不引入任何重采样。

本模块**不复刻**任何 Zernike 数学、前向模型、指标或损失: 全部委派给上述规范实现,
自身只做薄编排层。

与其他核心模块的关系
--------------------
* 上游物理: ``drivers/sim/fouriergsnet_env.py`` —— 唯一的 SLM+CCD 数字孪生, 也是
  本模块"仅仿真"声明的依据 (全文不 import 任何硬件驱动)。
* 上游数学: ``algorithm/signal_processing/zernike_coefficient_optimizer.py`` (Step A)
  与 ``.../differentiable_shaping.py`` (Step B); 指标来自
  ``utils/image/beam_metrics.py``, Zernike 计数来自 ``utils/wavefront/zernike_calc.py``。
* 硬件对应路径: 同一整形目标在真机上由 ``runners/slm/gsnet_runner.py``
  (``slm-gsnet``) 与 ``runners/slm/slm_shaping_runner.py`` (``spgd-square``) 驱动,
  但那两条走 Zernike/自由相位搜索而非本模块的迭代正向模型闭环。

本模块是**仅仿真**的: 它从不 import 硬件驱动, 唯一接触的"设备"是
:class:`SimFourierGSNetEnv`。
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Mapping, Sequence

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar


import numpy as np
from loguru import logger
from scipy.ndimage import zoom

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    create_target_mask,
    total_loss,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    TWIN_REGION,
    TWIN_W0,
    ZernikeCoefficientOptimizer,
)
from ao_shaping.drivers.sim.fouriergsnet_env import BeamParams, SimFourierGSNetEnv
from ao_shaping.utils.image.beam_metrics import compute_square_metrics, zero_order_center
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

#: The twin's Zernike generator is hard-wired to ``n_orders=6``
#: (``fouriergsnet_env._zernike_generator``) and ``generate_polynomial`` silently
#: drops any mode whose 0-based Noll index exceeds ``RZern.nk``. A higher index
#: would therefore be fitted by the model yet ignored by the twin, so the
#: injected ground truth is validated against this bound.
ENV_MAX_NOLL = calc_n_zernike_terms(6)

# Below this Pearson correlation between the model and a real capture, the
# speckle patterns do not correspond and the geometry solve has not converged on
# anything physical. Measured against a synthetic bench the correct geometry
# scores >0.95, so 0.75 leaves ample margin while still rejecting the
# "smooth near-flat pupil" case that cannot discriminate at all.
_MIN_CORRELATION = 0.75
# Above this relative RMS the defocus sweep no longer constrains the waist. The
# tilt-derived scale is unaffected -- it comes from the centroid slope alone.
_MAX_DEFOCUS_FIT_RMS = 0.10
# Rejected as a width-validity criterion: a mode that genuinely responds strongly
# also runs its width far past the narrowest point in its own sweep, so "w > K *
# min(w)" drops exactly the large-|coefficient| points that carry the curvature
# signal (it removed astig_y from a synthetic sweep whose widths were correct).
# Hollowness below measures single-lobedness directly and is what actually
# catches a ring. Kept as documentation of the rejected alternative.
_DEFOCUS_SINGLE_LOBE_FACTOR = 1.8
# Below this centroid/peak ratio the focal plane is a ring pattern, not one lobe.
_DEFOCUS_MIN_HOLLOWNESS = 0.60
# Minimum phase-correlation peak height for a tilt shift to be trusted over the
# spot centroid.
_MIN_SHIFT_CORRELATION = 0.30

def _has_both_signs(values: np.ndarray) -> bool:
    """True when the coefficients straddle zero, so an offset is observable."""
    v = np.asarray(values, dtype=np.float64)
    return bool(np.any(v > 0) and np.any(v < 0))


#: Directory searched by :func:`calibrate_shared_aberration` when no explicit
#: records and no ``search_root`` are given.
DEFAULT_SEARCH_ROOT = Path("data") / "debug"


def _require_torch() -> Any:
    """Return the ``torch`` module, raising a helpful error when absent.

    Returns:
        The ``torch`` top-level module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "model_in_loop_shaping requires PyTorch. Install with: uv sync --group ml"
        ) from None
    return _t


def _auto_device(torch: Any) -> str:
    """Return the auto-selected torch device string.

    Args:
        torch: The ``torch`` module.

    Returns:
        ``"cuda"`` when a GPU is visible, else ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def native_w0(region: int) -> float:
    """Return the digital twin's native Gaussian waist for a ``region`` grid.

    The waist scales linearly with the grid, so ``region == TWIN_REGION`` gives
    the twin's ``BeamParams.w0 = 250.0`` and the model's pupil amplitude matches
    the twin's exactly.

    Args:
        region: Side length of the square grid in pixels.

    Returns:
        The Gaussian waist in grid pixels (``31.25`` at the default
        ``region=64``).
    """
    return TWIN_W0 * (int(region) / TWIN_REGION)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationEnvParams:
    """Digital-twin parameters for the model-in-the-loop measurement.

    The twin runs in **native** geometry, so the SLM panel is the model grid and
    the pupil phase and the far field share the model's pixel pitch. ``w0`` must
    equal :func:`native_w0` for the same ``region`` and ``k_px`` must equal
    ``region``; both are enforced by :func:`_validate_config`.

    Attributes:
        region: Side length of the square SLM panel / far-field grid, pixels.
        w0: Gaussian beam waist on the panel, pixels. ``None`` derives it from
            ``region`` via :func:`native_w0`.
        k_px: Far-field sampling grid. ``None`` means ``region``.
        d_eff: Effective pixel size of the sinc envelope, metres. ``0.0``
            disables the envelope so the twin's forward reduces to the model's.
        noise_enabled: Add Poisson (and optional read) noise to the measurement.
        peak_photons: Full-well scale of the simulated CCD, electrons.
        n_frames: Frames averaged per measurement.
        seed: Seed for the twin's RNG, and therefore for camera noise.
    """

    region: int = 64
    w0: float | None = None
    k_px: int | None = None
    d_eff: float = 0.0
    noise_enabled: bool = False
    peak_photons: float = 60000.0
    n_frames: int = 1
    seed: int = 0

    def resolved_w0(self) -> float:
        """Return the effective waist, deriving the twin default when unset.

        Returns:
            The beam waist in panel pixels.
        """
        if self.w0 is not None:
            return float(self.w0)
        return native_w0(self.region)

    def resolved_k_px(self) -> int:
        """Return the far-field grid size, defaulting to the panel size.

        ``SimFourierGSNetEnv`` would otherwise derive ``K_px`` from the optical
        prescription (~7557 px here), which is not the model grid. Defaulting to
        ``region`` is what makes the twin 1:1 with the model.

        Returns:
            The far-field sampling grid side length.
        """
        if self.k_px is not None:
            return int(self.k_px)
        return int(self.region)


@dataclass
class ModelParams:
    """Step A model parameters.

    Attributes:
        region: Model grid side length. Must equal ``env.region``.
        n_orders: Maximum Zernike radial order; ``n_orders=10`` yields 66 modes.
        dtype: Torch dtype name for the model. ``"float64"`` supports the tight
            model/twin equivalence check; ``"float32"`` is faster.
        device: Torch device, or ``None`` to auto-detect.
        seed: Seed forwarded to the Step A optimizer.
    """

    region: int = 64
    n_orders: int = 10
    dtype: str = "float64"
    device: str | None = None
    seed: int = 0

    @property
    def n_coefficients(self) -> int:
        """Return the number of Zernike modes implied by ``n_orders``."""
        return calc_n_zernike_terms(self.n_orders)


@dataclass
class StepAConfig:
    """Step A (wavefront fit) budget.

    Attributes:
        iterations: Adam steps per round. The optimizer warm-starts across
            rounds, so each round polishes the previous fit.
        lr: Adam learning rate.
        initial_coefficients: Optional starting coefficient vector in radians.
            Supplying a deliberately wrong guess makes the fit -> shape
            iteration observable; ``None`` starts from zeros.
        probe_spread: Standard deviation, radians, of the zero-mean Gaussian
            pupil phase Step A projects through before fitting.

            Step A needs a *strong* probe. A flat pupil leaves the focal-plane
            intensity of a smooth low-order aberration nearly invariant -- on a
            ``region=32`` grid the peak-normalised MSE between the true and the
            unaberrated frame is only ~4.6e-4, the same order as the twin's
            16-bit quantisation pedestal (half a count at a 60000-count peak is
            8.3e-6). Above ~3 rad the signal clears the pedestal by two orders
            of magnitude and the fit becomes well posed.
        probe_count: Number of distinct probes cycled within a round.

            Focal-plane intensity is a non-convex measurement of pupil phase, so
            a single probe has many stationary points and Adam stalls in
            whichever one it happens to fall into. Cycling several independent
            probes through one Adam state supplies the phase diversity that
            removes them: measured on the twin, one probe leaves a residual
            coefficient error of ~0.6-1.6 rad while eight probes reach
            0.10-0.28 rad and drop the loss by 20-250x, repeatably across
            seeds.
        frozen_modes: Noll indices held at zero while fitting. Defaults to
            piston and tilt, which a far-field intensity cannot see; see
            :func:`calibrate_shared_aberration` for the measurement.
    """

    iterations: int = 500
    lr: float = 0.1
    initial_coefficients: tuple[float, ...] | None = None
    probe_spread: float = 5.0
    probe_count: int = 8
    frozen_modes: tuple[int, ...] = (1, 2, 3)


@dataclass
class StepBConfig:
    """Step B (freeform square shaping) budget.

    Attributes:
        iterations: Adam steps per round.
        lr: Adam learning rate on the full-pixel phase.
        target_shape: Target mask shape accepted by ``create_target_mask``.
        target_side: Target characteristic size, pixels.
        w_uniformity: Weight of the in-box coefficient-of-variation term.
        w_efficiency: Weight of the ``1 - encircled energy`` term. Must stay
            non-zero: optimising ``-CV`` alone empties the target box.
        w_zero_order: Weight of the zero-order penalty. Kept at ``0.0`` because
            DC suppression destroys a target centred on the beam origin.
        w_smoothness: Weight of the smoothness penalty; ``0.0`` keeps the sharp
            high-frequency content that square edges require.
    """

    iterations: int = 600
    lr: float = 0.05
    target_shape: str = "square"
    target_side: int = 6
    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    w_zero_order: float = 0.0
    w_smoothness: float = 0.0


@dataclass
class ModelInLoopConfig:
    """Full configuration of :func:`simulate_iterative_shaping`.

    Attributes:
        env: Digital-twin parameters.
        model: Step A model parameters.
        step_a: Step A budget.
        step_b: Step B budget.
        aberrations: Injected ground-truth aberration as ``{noll: radians}``,
            limited to Noll ``<= ENV_MAX_NOLL``.
        n_rounds: Number of fit -> shape -> re-measure rounds.
        seed: Seed for Step B's phase initialisation.
    """

    env: SimulationEnvParams = field(default_factory=SimulationEnvParams)
    model: ModelParams = field(default_factory=ModelParams)
    step_a: StepAConfig = field(default_factory=StepAConfig)
    step_b: StepBConfig = field(default_factory=StepBConfig)
    aberrations: Mapping[int, float] = field(default_factory=dict)
    n_rounds: int = 3
    seed: int = 0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Per-round record of one fit -> shape -> re-measure iteration.

    Attributes:
        round_index: Zero-based round index.
        coefficients: Fitted Zernike coefficients after Step A, radians.
        coeff_error: L2 error of ``coefficients`` against the ground truth, rad.
        step_a_loss_initial: Peak-normalised MSE before Step A's fit.
        step_a_loss_final: Peak-normalised MSE after Step A's fit.
        step_a_iterations: Adam steps actually taken by Step A.
        step_b_loss_initial: Weighted shaping loss before Step B.
        step_b_loss_final: Weighted shaping loss after Step B.
        step_b_iterations: Adam steps actually taken by Step B.
        center_before: ``(x, y)`` 0-order location of the pre-ShapeB frame.
        center_after: ``(x, y)`` 0-order location of the re-measured frame.
        metrics_before: ``compute_square_metrics`` on the pre-ShapeB frame.
        metrics_after: ``compute_square_metrics`` on the re-measured frame.
    """

    round_index: int
    coefficients: np.ndarray
    coeff_error: float
    step_a_loss_initial: float
    step_a_loss_final: float
    step_a_iterations: int
    step_b_loss_initial: float
    step_b_loss_final: float
    step_b_iterations: int
    center_before: tuple[int, int]
    center_after: tuple[int, int]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]


@dataclass
class ModelInLoopResult:
    """Outcome of :func:`simulate_iterative_shaping`.

    Attributes:
        rounds: One :class:`RoundMetrics` per round, in order.
        true_coefficients: Ground-truth coefficient vector injected into the
            twin, radians.
        initial_coefficients: The Step A starting guess, radians.
        initial_error: L2 error of the starting guess against the truth, rad.
        final_error: L2 error of the last fit against the truth, rad.
        final_phase: The last full-pixel SLM phase, raw unwrapped radians.
        final_coefficients: The last fitted coefficient vector, radians.
    """

    rounds: list[RoundMetrics]
    true_coefficients: np.ndarray
    initial_coefficients: np.ndarray
    initial_error: float
    final_error: float
    final_phase: np.ndarray
    final_coefficients: np.ndarray

    @property
    def coefficient_errors(self) -> list[float]:
        """Return the per-round coefficient L2 errors, in order."""
        return [r.coeff_error for r in self.rounds]

    @property
    def encircled_energy(self) -> list[float]:
        """Return the per-round encircled energy after each re-measurement."""
        return [r.metrics_after["encircled_energy"] for r in self.rounds]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_config(config: ModelInLoopConfig) -> None:
    """Validate a :class:`ModelInLoopConfig` eagerly.

    Args:
        config: The configuration to validate.

    Raises:
        ValueError: On any inconsistent or out-of-range setting.
    """
    if config.n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {config.n_rounds}")
    if config.step_a.iterations < 1:
        raise ValueError(
            f"step_a.iterations must be >= 1, got {config.step_a.iterations}"
        )
    if config.step_b.iterations < 1:
        raise ValueError(
            f"step_b.iterations must be >= 1, got {config.step_b.iterations}"
        )
    if not np.isfinite(config.step_a.probe_spread) or config.step_a.probe_spread <= 0.0:
        raise ValueError(
            f"step_a.probe_spread must be finite and > 0, got "
            f"{config.step_a.probe_spread}; a flat or zero-amplitude probe leaves "
            "Step A blind to a smooth low-order aberration"
        )
    if config.step_b.target_side < 1:
        raise ValueError(
            f"step_b.target_side must be >= 1, got {config.step_b.target_side}"
        )
    if config.step_a.probe_count < 1:
        raise ValueError(
            f"step_a.probe_count must be >= 1, got {config.step_a.probe_count}; "
            "Step A needs at least one probe frame to fit against"
        )
    if config.model.n_orders < 1:
        raise ValueError(f"model.n_orders must be >= 1, got {config.model.n_orders}")
    if config.env.n_frames < 1:
        raise ValueError(f"env.n_frames must be >= 1, got {config.env.n_frames}")
    if config.model.region != config.env.region:
        raise ValueError(
            f"model.region ({config.model.region}) must equal env.region "
            f"({config.env.region}): the model and the twin only observe the same "
            "grid when both run at the same native pitch"
        )
    k_px = config.env.resolved_k_px()
    if k_px != config.model.region:
        raise ValueError(
            f"env.k_px ({k_px}) must equal model.region ({config.model.region}) so "
            "the twin's far field lands on the model grid"
        )
    expected = native_w0(config.model.region)
    actual = config.env.resolved_w0()
    if not np.isclose(actual, expected, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"env.w0 ({actual}) must equal the model's native waist {expected} for "
            f"region {config.model.region} (TWIN_W0 * region / TWIN_REGION)"
        )
    initial = config.step_a.initial_coefficients
    if initial is not None and np.asarray(initial).shape != (
        config.model.n_coefficients,
    ):
        raise ValueError(
            f"step_a.initial_coefficients must have shape "
            f"({config.model.n_coefficients},), got {np.asarray(initial).shape}"
        )
    for noll, value in config.aberrations.items():
        if int(noll) < 1 or int(noll) > ENV_MAX_NOLL:
            raise ValueError(
                f"aberration Noll index {noll} is outside 1..{ENV_MAX_NOLL}; the "
                "twin's Zernike generator is limited to n_orders=6 and would "
                "silently drop higher modes"
            )
        if not np.isfinite(value):
            raise ValueError(f"aberration Noll {noll} must be finite, got {value}")


def coefficients_from_noll(
    n_coefficients: int, values: Mapping[int, float]
) -> np.ndarray:
    """Build a Noll-ordered coefficient vector from ``{noll: radians}``.

    Args:
        n_coefficients: Length of the dense vector.
        values: Mapping of 1-based Noll index to amplitude in radians.

    Returns:
        A ``(n_coefficients,)`` float64 vector, zero outside ``values``.

    Raises:
        ValueError: If a Noll index is outside ``[1, n_coefficients]`` or a value
            is not finite.
    """
    vector = np.zeros(int(n_coefficients), dtype=np.float64)
    for noll, radians in values.items():
        index = int(noll)
        if not 1 <= index <= int(n_coefficients):
            raise ValueError(f"Noll index {noll} is outside [1, {n_coefficients}]")
        if not np.isfinite(radians):
            raise ValueError(f"Noll index {noll} has a non-finite value {radians!r}")
        vector[index - 1] = float(radians)
    return vector


def noll_coefficients_to_dict(coefficients: np.ndarray) -> dict[int, float]:
    """Convert a Noll-ordered vector to the ``{noll: radians}`` twin mapping.

    Zero entries are dropped, which is the form
    :attr:`SimFourierGSNetEnv.aberrations` expects.

    Args:
        coefficients: 1D coefficient vector in radians.

    Returns:
        Mapping of 1-based Noll index to non-zero amplitude.
    """
    values = np.asarray(coefficients, dtype=np.float64).ravel()
    return {
        index + 1: float(value)
        for index, value in enumerate(values)
        if value != 0.0
    }


def make_native_env(
    params: SimulationEnvParams,
    aberrations: Mapping[int, float] | None = None,
) -> SimFourierGSNetEnv:
    """Build the digital twin described by ``params`` in native geometry.

    Args:
        params: Validated twin parameters.
        aberrations: Optional ``{noll: radians}`` ground truth to inject.

    Returns:
        A twin whose SLM panel is the model grid.

    Raises:
        ValueError: If an injected Noll index exceeds :data:`ENV_MAX_NOLL`.
    """
    env = SimFourierGSNetEnv(
        d_eff=params.d_eff,
        K_px=params.resolved_k_px(),
        beam=BeamParams(
            region=params.region, w0=params.resolved_w0(), native=True
        ),
        peak_photons=params.peak_photons,
        noise_enabled=params.noise_enabled,
        seed=params.seed,
    )
    if aberrations:
        for noll in aberrations:
            if int(noll) > ENV_MAX_NOLL:
                raise ValueError(
                    f"aberration Noll index {noll} exceeds the twin's maximum "
                    f"{ENV_MAX_NOLL} and would be silently dropped"
                )
        env.aberrations = {int(k): float(v) for k, v in aberrations.items()}
    return env


def display_and_measure(
    env: SimFourierGSNetEnv, phase: np.ndarray, n_frames: int = 1
) -> np.ndarray:
    """Display ``phase`` on the twin and return the measured far field.

    Args:
        env: The digital twin.
        phase: Full-pixel SLM phase, raw unwrapped radians.
        n_frames: Frames averaged per measurement.

    Returns:
        A float64 far-field frame on the model grid (the twin is native, so no
        resampling is needed).
    """
    env.slm.display_phase(np.asarray(phase, dtype=np.float64))
    frame = env.ccd.get_numpy_image(n_sample=n_frames)
    return np.asarray(frame, dtype=np.float64)


def square_metrics_at_zero_order(
    intensity: np.ndarray, target_side: int
) -> tuple[dict[str, float], tuple[int, int]]:
    """Compute the square metrics with the 0-order located by global argmax.

    The optical axis is the frame's global maximum, never the geometric frame
    centre, so the anchor comes from
    :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center` with
    ``refine=False`` (literally ``np.unravel_index(np.argmax(frame))``) and is
    handed straight to
    :func:`~ao_shaping.utils.image.beam_metrics.compute_square_metrics`, which
    shares the ``(x, y)`` convention.

    Args:
        intensity: 2D far-field intensity frame.
        target_side: Target square side, pixels.

    Returns:
        A tuple of the metric dict and the ``(x, y)`` centre used.
    """
    center = zero_order_center(intensity, refine=False)
    return compute_square_metrics(intensity, target_side, center), center


def _aperture_from_basis(basis: np.ndarray) -> np.ndarray:
    """Recover the twin's aperture mask from a cached Zernike basis.

    Every mode is evaluated on the same circular aperture and the generator
    emits NaN outside it, so the piston mode's finite support *is* the mask --
    the same derivation :class:`ZernikeCoefficientOptimizer` uses internally,
    which keeps the model and the twin consistent by construction.

    Args:
        basis: ``(n_coefficients, region, region)`` basis from
            :meth:`ZernikeCoefficientOptimizer.generate_basis`.

    Returns:
        A float64 ``(region, region)`` array, 1 inside the aperture and 0
        outside.
    """
    return (np.isfinite(basis[0]) & (np.abs(basis[0]) > 0.0)).astype(np.float64)


def _probe_phase(region: int, spread: float, seed: int) -> np.ndarray:
    """Build the deterministic pupil probe phase Step A fits through.

    A flat pupil leaves the focal intensity almost unchanged by a smooth
    low-order aberration, so the fit has to interrogate the beam with a known,
    spatially rich phase. The probe is drawn from a seeded generator so a run
    is bit-for-bit reproducible, and each round gets a different draw which
    both re-probes the wavefront and adds the phase diversity that decorrelates
    the Zernike modes.

    Args:
        region: Side length of the pupil grid, pixels.
        spread: Standard deviation of the zero-mean Gaussian, radians.
        seed: Seed for the probe.

    Returns:
        A float64 ``(region, region)`` phase in raw unwrapped radians.
    """
    return np.random.default_rng(int(seed)).normal(
        0.0, float(spread), (int(region), int(region))
    )


def _to_model_units(
    optimizer: ZernikeCoefficientOptimizer, frame: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """Rescale a measured frame onto the forward model's intensity scale.

    Step A's loss is ``mean((I_model / peak_anchor - I_meas / peak_peak) ** 2)``
    with ``peak_anchor`` taken from the *measurement*, so the two sides are only
    commensurate when the measurement already shares the model's units. The
    simulated CCD does not: it re-normalises every frame to ``peak_photons``
    counts (60000 by default) while the model's own far field peaks in the
    thousands, so a raw frame is ~47x brighter than any model intensity. Feeding
    that to Step A silently changes the objective from "match the normalised
    pattern" into "maximise model brightness under the measured core", whose
    optimum is a focusing aberration rather than the true one -- the fit then
    wanders to a coefficient error of ~0.7 rad no matter how long it runs.

    The gain is recovered from the model itself: the twin is a linear detector,
    so the measured frame is the model's far field times a constant, and scaling
    by ``model_peak / measured_peak`` restores model units exactly. The scale is
    recomputed from the *current* coefficients so it tightens as the fit
    converges, and any degenerate frame (zero peak) is passed through untouched.

    Args:
        optimizer: The Step A optimizer, whose current coefficients set the scale.
        frame: Measured far field, CCD units.
        phase: The pupil phase the frame was measured through, radians.

    Returns:
        The frame rescaled to model units, or the input unchanged when either
        peak is non-positive.
    """
    measured_peak = float(np.max(frame))
    if not np.isfinite(measured_peak) or measured_peak <= 0.0:
        return frame
    model_peak = float(
        np.max(optimizer.forward_intensity(optimizer.coefficients, phase))
    )
    if not np.isfinite(model_peak) or model_peak <= 0.0:
        return frame
    return frame * (model_peak / measured_peak)


class _StepAOptimizer(Protocol):
    """The slice of the Step A optimizer that the fitting loop actually drives.

    :class:`ZernikeCoefficientOptimizer` satisfies this structurally, so the
    public behaviour is unchanged; stating the dependency explicitly documents
    what one fitting pass needs (``update`` per step, and the coefficients, loss
    and convergence latch to read back) and keeps the loop drivable by a test
    double that has already tripped its latch.
    """

    @property
    def coefficients(self) -> np.ndarray:
        """The current coefficient vector, radians."""
        ...

    @property
    def last_loss(self) -> float | None:
        """Loss at the last step, or ``None`` before the first one."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the optimizer's plateau detector has latched."""
        ...

    def update(self, i_meas: np.ndarray, phase_slm: np.ndarray) -> np.ndarray:
        """Take one Adam step against a measured far field and its probe phase."""
        ...


#: Bound so the helper stays generic over anything satisfying the protocol while
#: still handing the caller back the *same* concrete type it passed in. Without
#: this the loop's own optimizer would widen to the protocol and stop being
#: acceptable to the Step B helpers that need the real class.
_StepAOptimizerT = TypeVar("_StepAOptimizerT", bound=_StepAOptimizer)


def _fit_aberration_at_probes(
    optimizer: _StepAOptimizerT,
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    rearm: Callable[[np.ndarray], _StepAOptimizerT] | None = None,
) -> tuple[np.ndarray, list[float], int, _StepAOptimizerT]:

    """Run Step A for a fixed number of Adam steps over cycling probe frames.

    The probes are cycled inside a *single* Adam state rather than fitted one
    after another. That is the whole point: focal-plane intensity is a non-convex
    function of pupil phase, so one probe leaves a large set of stationary points
    and Adam settles into whichever one it enters from the start guess. Feeding
    several independent probes to one optimizer means no single measurement can
    satisfy the fit on its own terms, which removes the local minima; measured on
    the twin this cuts the residual coefficient error from ~0.6-1.6 rad with one
    probe to ~0.1-0.3 rad with eight, and drives the loss down 20-250x instead of
    leaving it flat or rising.

    The frames are driven through consecutive :meth:`update` calls rather than
    one :meth:`~ZernikeCoefficientOptimizer.run` per frame. Driving them jointly
    is what makes the fit work: ``run`` optimises each frame to its own optimum
    and restarts Adam, so a set of records degenerates into a chain of
    independent fits in which the last frame wins and the shared vector ends up
    fitting nothing. Measured on the twin at ``region=64``/``n_orders=10`` (66
    coefficients), sequential per-frame ``run`` reduced the coefficient error in
    only 4 of 5 seeds and diverged to ~19 rad on the fifth, while cycling the
    frames through one ``update`` loop improved 5 of 5 and stayed within
    0.11-0.50 rad.

    ``update`` latches ``_converged`` after ``PLATEAU_PATIENCE`` non-improving
    steps and then returns without moving the coefficients, and nothing clears
    that latch. :meth:`reset` is not a usable re-arm: it calls
    ``_restart(restore_initial=True)`` and so rewinds the coefficients to the
    values handed to ``__init__``. ``rearm`` therefore rebuilds the optimizer
    around its *current* coefficients, which clears the latch and the Adam
    moments while keeping the progress made so far.

    Args:
        optimizer: The Step A optimizer, warm-started in place.
        rearm: Factory returning a fresh optimizer seeded with the coefficients
            it is given, used to clear a tripped convergence latch. ``None``
            disables re-arming, in which case a tripped latch ends the fit early.
        frames: One or more ``(measured_far_field, probe_phase)`` pairs. Each

            probe is the pupil phase its frame was measured through, radians.
        iterations: Total number of Adam steps to take across all frames.

    Returns:
        A tuple of the fitted coefficients (radians), the loss history produced
        by *this* call, the number of steps actually taken, and the optimizer
        the call ended on. That last element matters: a tripped latch makes this
        function re-arm into a *fresh* optimizer, so a caller that keeps using
        the instance it passed in would silently resume the next round from the
        stale latch point and throw away everything learned since. It is the
        same object that was passed in when no re-arm fires, so adopting it costs
        nothing and never resets Adam on its own.


    Raises:
        ValueError: If ``frames`` is empty or ``iterations`` is not positive.
    """
    if not frames:
        raise ValueError("Step A needs at least one probe frame, got none")
    total = int(iterations)
    if total <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    count = len(frames)
    losses: list[float] = []
    taken = 0
    for index in range(total):
        image, probe = frames[index % count]
        optimizer.update(image, probe)
        if optimizer.converged and rearm is not None:
            optimizer = rearm(optimizer.coefficients)
        # ``last_loss`` is populated by every ``update``; the ``None`` branch only
        # exists on the type, since the attribute is Optional-typed.
        loss = optimizer.last_loss
        if loss is not None:
            losses.append(float(loss))
        taken += 1
    return (
        np.asarray(optimizer.coefficients, dtype=np.float64),
        losses,
        taken,
        optimizer,
    )




def shape_phase_with_frozen_aberration(
    optimizer: ZernikeCoefficientOptimizer,
    coefficients: np.ndarray,
    target: np.ndarray,
    initial_phase: np.ndarray,
    config: StepBConfig,
    dtype: str = "float64",
    device: str | None = None,
    seed: int = 0,
    source_amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run Step B: optimise a full-pixel phase under a frozen aberration.

    The forward mirrors the twin -- and therefore
    :meth:`ZernikeCoefficientOptimizer.forward_intensity` -- exactly, with the
    coefficients frozen: the SLM phase and the aberration are summed first and
    the *sum* is masked, because the twin applies
    ``nan_to_num(phi_slm + aberration)`` and the aberration is NaN outside the
    aperture. Gradients therefore flow only into the SLM phase.

    Args:
        optimizer: The Step A optimizer, used for its cached basis and native
            amplitude. It is not mutated.
        coefficients: Frozen Zernike coefficients, radians.
        target: Target intensity mask on the model grid.
        initial_phase: Starting full-pixel phase, radians.
        config: Step B budget and loss weights.
        dtype: Torch dtype name for the model.
        device: Torch device, or ``None`` to auto-detect.
        seed: Seed for the phase initialisation.
        source_amplitude: Optional ``(region, region)`` source amplitude. When
            ``None`` the native twin Gaussian is used, which is right for the
            digital twin. On hardware the beam waist is a *measured* quantity
            (see :func:`calibrate_bench_geometry` and
            ``report/slm/model_in_loop_bench_calibration.md``), so the calibrated
            amplitude must be passed in or Step B optimises for a beam that is
            not the real one.

    Returns:
        A tuple of the optimised phase (raw unwrapped radians) and the
        per-iteration loss history.

    Raises:
        ImportError: If PyTorch is unavailable.
    """
    torch = _require_torch()
    tdtype = getattr(torch, dtype)
    dev = torch.device(device) if device is not None else _auto_device(torch)

    region = initial_phase.shape[0]
    basis_np = optimizer.generate_basis()
    basis = torch.from_numpy(basis_np).to(device=dev, dtype=tdtype)
    aperture = torch.from_numpy(_aperture_from_basis(basis_np)).to(
        device=dev, dtype=tdtype
    )
    amplitude_grid = (
        ZernikeCoefficientOptimizer.native_amplitude(region)
        if source_amplitude is None
        else np.asarray(source_amplitude, dtype=np.float64)
    )
    if amplitude_grid.shape != (region, region):
        raise ValueError(
            f"source_amplitude must have shape {(region, region)}, "
            f"got {amplitude_grid.shape}"
        )
    if not np.all(np.isfinite(amplitude_grid)):
        raise ValueError("source_amplitude must be finite")
    amplitude = torch.from_numpy(
        np.ascontiguousarray(amplitude_grid)
    ).to(device=dev, dtype=tdtype)
    aberration = torch.einsum(
        "j,jhw->hw",
        torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.float64)).to(
            device=dev, dtype=tdtype
        ),
        basis,
    )
    target_t = torch.from_numpy(np.ascontiguousarray(target)).to(
        device=dev, dtype=tdtype
    )
    # The target is expressed on the *pupil* grid, but the model's far field is
    # far_field_size pixels wide. When those differ (any zero-padded run, e.g.
    # far_field_size=4096 on a 256 pupil) a target pixel is a different physical
    # size from a far-field pixel, so comparing them directly makes the loss
    # measure a box that is far_field_size/region times too small. The optimizer
    # then satisfies it by squeezing energy into that tiny box: measured on the
    # twin at the bench's sampling, EE collapsed 0.82 -> 0.011 at every learning
    # rate. Rescale the target onto the far-field grid.
    scale = float(getattr(optimizer, "far_field_size", region)) / float(region)
    if abs(scale - 1.0) > 1e-9:
        target_t = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(
                    zoom(target, (scale, scale), order=1), dtype=np.float64
                )
            )
        ).to(device=dev, dtype=tdtype)

    torch.manual_seed(int(seed))
    # A small random start escapes the degenerate zero-gradient point where the
    # pupil field is purely real and the intensity is quadratic in the phase.
    start_phase = torch.from_numpy(
        np.ascontiguousarray(np.asarray(initial_phase, dtype=np.float64))
    ).to(device=dev, dtype=tdtype)
    start = start_phase + 0.1 * torch.randn(
        (region, region), device=dev, dtype=tdtype
    )
    phase = torch.nn.Parameter(start.clone())
    opt = torch.optim.Adam([phase], lr=config.lr)

    def closure() -> Any:
        opt.zero_grad()
        patch = (phase + aberration) * aperture
        field = amplitude * torch.exp(1j * patch)
        # Honour the optimizer's zero-padding so the far field is sampled at the
        # same pitch as Step A and as the measurement. Without this the FFT
        # always ran at `region`, so a padded run compared a `region`-sized
        # intensity against a `far_field_size`-sized target.
        pad = int(getattr(optimizer, "far_field_size", region))
        if pad > field.shape[0]:
            half = (pad - field.shape[0]) // 2
            field = torch.nn.functional.pad(
                field.unsqueeze(0),
                (half, pad - field.shape[0] - half,
                 half, pad - field.shape[1] - half),
            ).squeeze(0)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field), norm="ortho")
        )
        intensity = spectrum.real**2 + spectrum.imag**2
        loss = total_loss(
            intensity,
            phase,
            target_t,
            w_uniformity=config.w_uniformity,
            w_efficiency=config.w_efficiency,
            w_zero_order=config.w_zero_order,
            w_smoothness=config.w_smoothness,
        )
        loss.backward()
        return loss

    history: list[float] = []
    for _ in range(config.iterations):
        history.append(float(closure().detach().cpu().item()))
        opt.step()
    logger.debug(
        "Step B done: iterations={} loss {} -> {}",
        len(history),
        history[0],
        history[-1],
    )
    return phase.detach().cpu().numpy().astype(np.float64), history


# ---------------------------------------------------------------------------
# The model-in-the-loop simulation
# ---------------------------------------------------------------------------


def simulate_iterative_shaping(config: ModelInLoopConfig) -> ModelInLoopResult:
    """Run the fit -> shape -> re-measure loop on the digital twin.

    Each round measures the current phase through
    :class:`~ao_shaping.drivers.sim.fouriergsnet_env.SimFourierGSNetEnv`, fits
    the aberration with Step A, optimises a full-pixel phase with Step B under
    that frozen aberration, re-measures, and records the square metrics.

    Args:
        config: Loop configuration, validated eagerly.

    Returns:
        A :class:`ModelInLoopResult`.

    Raises:
        ValueError: If ``config`` is inconsistent (see :func:`_validate_config`).
        ImportError: If PyTorch is unavailable.
    """
    _validate_config(config)
    region = config.model.region
    n_coeffs = config.model.n_coefficients

    env = make_native_env(config.env, config.aberrations)
    true_c = coefficients_from_noll(n_coeffs, config.aberrations)

    initial = config.step_a.initial_coefficients
    initial_vector = (
        np.zeros(n_coeffs, dtype=np.float64)
        if initial is None
        else np.asarray(initial, dtype=np.float64)
    )

    optimizer = ZernikeCoefficientOptimizer(
        n_orders=config.model.n_orders,
        region=region,
        initial_coefficients=initial_vector,
        lr=config.step_a.lr,
        max_iterations=config.step_a.iterations,
        device=config.model.device,
        seed=config.model.seed,
        dtype=config.model.dtype,
        frozen_modes=config.step_a.frozen_modes,
    )

    def rearm(coefficients: np.ndarray) -> ZernikeCoefficientOptimizer:
        """Rebuild the optimizer around ``coefficients`` to clear a tripped latch.

        ``ZernikeCoefficientOptimizer.reset`` cannot do this: it restores the
        coefficients handed to ``__init__`` rather than the current ones, so
        re-arming with it would snap the fit back to the start guess roughly
        every ``PLATEAU_PATIENCE`` steps. A fresh instance seeded with the live
        coefficients keeps the progress while discarding the stale latch and the
        Adam moments that stopped moving.
        """
        return ZernikeCoefficientOptimizer(
            n_orders=config.model.n_orders,
            region=region,
            initial_coefficients=np.asarray(coefficients, dtype=np.float64),
            lr=config.step_a.lr,
            max_iterations=config.step_a.iterations,
            device=config.model.device,
            seed=config.model.seed,
            dtype=config.model.dtype,
            frozen_modes=config.step_a.frozen_modes,
        )

    target = create_target_mask(
        config.step_b.target_shape, (region, region), config.step_b.target_side
    )

    def error(vector: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(vector) - true_c))

    rounds: list[RoundMetrics] = []
    initial_error = error(initial_vector)
    phase = np.zeros((region, region), dtype=np.float64)
    image = display_and_measure(env, phase, config.env.n_frames)

    for index in range(config.n_rounds):
        metrics_before, center_before = square_metrics_at_zero_order(
            image, config.step_b.target_side
        )

        # Step A interrogates the wavefront with a *set* of strong pupil probes
        # rather than the shaped phase: a flat pupil carries almost no
        # information about a smooth low-order aberration, so fitting it would be
        # blind, and any single probe leaves the non-convex intensity fit stuck
        # in whichever stationary point it happens to start from. Cycling
        # ``probe_count`` independent probes through one Adam state supplies the
        # phase diversity that removes them. Each round draws a fresh family
        # (``config.seed`` + round), so consecutive rounds are not re-fitting the
        # same degenerate direction.
        frames = []
        for probe_index in range(config.step_a.probe_count):
            probe = _probe_phase(
                region,
                config.step_a.probe_spread,
                config.seed + index * config.step_a.probe_count + probe_index,
            )
            frames.append((display_and_measure(env, probe, config.env.n_frames), probe))
        fitted, losses, step_a_iterations, optimizer = _fit_aberration_at_probes(
            optimizer, frames, config.step_a.iterations, rearm
        )

        logger.info(
            "round {}/{}: Step A loss {} -> {} over {} steps (coeff error {:.4g} rad)",
            index + 1,
            config.n_rounds,
            losses[0] if losses else float("nan"),
            losses[-1] if losses else float("nan"),
            step_a_iterations,
            error(fitted),
        )

        shaped, shaping_loss = shape_phase_with_frozen_aberration(
            optimizer,
            fitted,
            target,
            phase,
            config.step_b,
            dtype=config.model.dtype,
            device=config.model.device,
            seed=config.seed + index,
        )

        image = display_and_measure(env, shaped, config.env.n_frames)
        metrics_after, center_after = square_metrics_at_zero_order(
            image, config.step_b.target_side
        )

        rounds.append(
            RoundMetrics(
                round_index=index,
                coefficients=np.array(fitted, dtype=np.float64),
                coeff_error=error(fitted),
                step_a_loss_initial=float(losses[0]) if losses else float("nan"),
                step_a_loss_final=float(losses[-1]) if losses else float("nan"),
                step_a_iterations=step_a_iterations,
                step_b_loss_initial=shaping_loss[0],
                step_b_loss_final=shaping_loss[-1],
                step_b_iterations=len(shaping_loss),
                center_before=center_before,
                center_after=center_after,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
        )
        phase = shaped

    return ModelInLoopResult(
        rounds=rounds,
        true_coefficients=true_c,
        initial_coefficients=initial_vector,
        initial_error=initial_error,
        final_error=rounds[-1].coeff_error,
        final_phase=phase,
        final_coefficients=rounds[-1].coefficients,
    )


# ---------------------------------------------------------------------------
# Bench geometry calibration (offline, from real captures)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchGeometry:
    """Model-to-camera calibration derived from real far-field captures.

    The digital twin hides three quantities that a real bench fixes by
    experiment: how much of the SLM panel the illuminated pupil covers, how wide
    that beam is, and how finely the far field has to be sampled before it can
    be compared with a camera frame. This record carries the answer so a
    hardware run does not have to re-derive it.

    Attributes:
        panel_disc_radius: Panel pixels from the panel centre to the edge of the
            region that was mapped onto the model grid.
        region: Model grid side length in pupil pixels.
        beam_waist_panel_px: Gaussian waist of the illumination in panel pixels.
        far_field_size: Zero-padding size the forward model needs for its far
            field to be sampled at the camera's pitch.
        camera_px_per_model_px: Far-field scale: camera pixels per model pixel.
        spot_fwhm_camera_px: Measured spot FWHM in camera pixels.
        spot_fwhm_model_px: Spot FWHM the model reproduces, in model pixels.
        correlation: Pearson correlation between the model and the measurement
            over the calibration records; the discriminative goodness metric.
            For ``method="sweep"`` this instead carries the defocus-response
            goodness of fit (1 - normalised RMS), because that method has no
            speckle to correlate against.
        method: How the geometry was obtained -- ``"speckle"`` (default) or
            ``"sweep"``.
        calibration_notes: Free-form provenance, e.g. the measured tilt slopes
            and the panel/camera axis mapping.
    """

    panel_disc_radius: int
    region: int
    beam_waist_panel_px: float
    far_field_size: int
    camera_px_per_model_px: float
    spot_fwhm_camera_px: float
    spot_fwhm_model_px: float
    correlation: float
    method: str = "speckle"
    calibration_notes: str = ""

    def model_side_for_camera(self, camera_px: float) -> int:
        """Convert a target square side from camera pixels to model pixels.

        Args:
            camera_px: Desired target side in camera pixels.

        Returns:
            The equivalent side in model far-field pixels, at least 1.
        """
        return max(1, int(round(float(camera_px) / self.camera_px_per_model_px)))


def _spot_fwhm(image: np.ndarray) -> float:
    """Spot full-width-at-half-maximum in pixels, averaged over two axes.

    FWHM rather than a second moment on purpose. A speckle pattern has heavy
    tails, and the second moment weights them by ``r**2``, so it is dominated by
    a few far-out pixels: on a 2048-px model grid it read ~70x the value the
    same spot has on a 160-px camera crop, which made the two sides
    incomparable. FWHM is local and bounded, so it measures the same width on
    any grid that contains the half-maximum region.

    Args:
        image: 2D intensity frame.

    Returns:
        The mean horizontal/vertical FWHM in pixels, or ``nan`` for a
        degenerate frame.
    """
    frame = np.asarray(image, dtype=np.float64)
    peak_index = np.unravel_index(int(np.argmax(frame)), frame.shape)
    peak = float(frame[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")
    cy, cx = int(peak_index[0]), int(peak_index[1])
    widths = []
    for line in (frame[cy, :], frame[:, cx]):
        half = peak / 2.0
        idx = np.flatnonzero(line >= half)
        if idx.size < 2:
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        # Linear interpolation at the two half-maximum crossings.
        left = lo
        if lo > 0:
            y0, y1 = line[lo - 1], line[lo]
            left = lo - 1 + (half - y0) / (y1 - y0) if y1 != y0 else float(lo)
        right = hi
        if hi < line.size - 1:
            y0, y1 = line[hi], line[hi + 1]
            right = hi + (y0 - half) / (y0 - y1) if y0 != y1 else float(hi)
        widths.append(abs(right - left))
    if not widths:
        return float("nan")
    return float(np.mean(widths))


def _model_window(
    far_field: np.ndarray, camera_shape: tuple[int, int], camera_px_per_model_px: float
) -> np.ndarray:
    """Crop the model's far field to the extent a camera frame covers.

    ``camera_px_per_model_px`` is how many camera pixels one model pixel spans,
    so a frame of ``n`` camera pixels covers ``n / camera_px_per_model_px`` model
    pixels.
    """
    rows = max(1, int(round(camera_shape[0] / camera_px_per_model_px)))
    cols = max(1, int(round(camera_shape[1] / camera_px_per_model_px)))
    rows, cols = min(rows, far_field.shape[0]), min(cols, far_field.shape[1])
    top = (far_field.shape[0] - rows) // 2
    left = (far_field.shape[1] - cols) // 2
    return far_field[top : top + rows, left : left + cols]


def calibrate_bench_geometry(
    records: Sequence[CalibrationRecord],
    *,
    wavelength: float = 1064e-9,
    focal_length: float = 0.125,
    slm_pitch: float = 8e-6,
    camera_pixel: float = 2.2e-6,
    region: int = 256,
    far_field_size: int = 4096,
    disc_candidates: Sequence[int] = (600, 450, 300, 250, 200, 150),
    panel_span_px: float | None = None,
) -> BenchGeometry:
    """Derive the model-to-camera geometry from real captures, offline.

    Two quantities are solved in sequence, both against the *measured* spot
    rather than a loss:

    1. **Beam waist.** The model's spot sigma scales as ``1 / w0``, so one probe
       evaluation gives the scale factor needed to match the measured sigma.
    2. **Panel extent.** For each candidate radius the panel phase is cropped to
       that disc, mapped onto the model grid, and the resulting far field is
       compared with the measurement by Pearson correlation.

    Correlation, not MSE, is the scoring metric on purpose: with a peaked spot
    the peak-normalised MSE is dominated by the spot's tail energy and reports
    ~0.007 for patterns that do not actually correspond, whereas the speckle
    correlation only peaks when the phase mapping is right.

    Args:
        records: Real captures, each with the applied panel ``phase`` and the
            measured far-field ``image``. Two or more distinct phases are needed
            for the speckle patterns to be comparable; a single record is
            accepted but scores less reliably.
        wavelength: Laser wavelength, metres.
        focal_length: 2f lens focal length, metres.
        slm_pitch: SLM pixel pitch, metres.
        camera_pixel: Camera pixel pitch, metres.
        region: Model grid side length in pupil pixels.
        far_field_size: Zero-padding size for the forward model.
        disc_candidates: Aperture semi-axes to try, in **model** pixels. Values
            above ``region / 2`` are clipped to it: the aperture is the disc
            inscribed in the model grid, because the runbook maps the model grid
            one-to-one onto the illuminated panel box. (Interpreting these as
            *panel* pixels, as earlier revisions did, made every candidate above
            ``region / 2`` collapse onto the same array, so the search could only
            ever report a saturated edge.)
        panel_span_px: Width of the panel region the model grid covers, in panel
            pixels (twice the runbook's ``--collect-disc``). **Required for real
            captures.** The model grid samples the pupil at
            ``panel_span_px / region`` times the SLM pitch, not at the SLM pitch
            itself; using ``slm_pitch`` directly makes the modelled aperture
            physically smaller than the real beam (2.05 mm vs 7.2 mm on this
            bench, a 3.5x error) and the predicted focal spot ~200x wider than
            the measured one, so no aperture can correlate.

    Returns:
        The best-scoring :class:`BenchGeometry`.

    Raises:
        ValueError: If ``records`` is empty or no disc radius yields a finite
            spot sigma.
    """
    if not records:
        raise ValueError("calibrate_bench_geometry needs at least one record")

    # Camera-side reference: sigma of the brightest real capture.
    measured = [np.asarray(r.image, dtype=np.float64) for r in records]
    widths = [_spot_fwhm(img) for img in measured]
    finite = [w for w in widths if np.isfinite(w) and w > 0]
    if not finite:
        raise ValueError("no record has a measurable spot (all zero or degenerate)")
    fwhm_meas = float(np.median(finite))
    reference = measured[int(np.nanargmax(widths))]

    # The model grid spans `panel_span_px` panel pixels, so its physical pitch is
    # that span times the SLM pitch divided by the number of samples. Using
    # `slm_pitch` here (as this did originally) assumes one model pixel per SLM
    # pixel, which is only true when region == panel_span_px.
    if panel_span_px is not None and float(panel_span_px) > 0:
        model_pitch = float(panel_span_px) * slm_pitch / float(region)
    else:
        model_pitch = slm_pitch
    p_fft = wavelength * focal_length / (far_field_size * model_pitch)
    # How many camera pixels one model pixel spans (1.85 at far_field_size=4096
    # for a 2.2 um camera and 8 um SLM pitch).
    camera_px_per_model_px = p_fft / camera_pixel
    # Model pixels -> panel pixels, for reporting sizes on the panel.
    model_to_panel = (float(panel_span_px) / float(region)) if panel_span_px else 1.0

    def pupil(semi_axis_model_px: int) -> np.ndarray | None:
        """The pupil phase over the aperture, mapped onto the model grid.

        The aperture spans ``2 * semi_axis_model_px`` model pixels. How that maps
        onto the record's ``phase`` array depends on what the array represents:

        * ``region x region`` -- it is already the model grid (the runbook builds
          each probe on the model grid and scales it onto the panel box), so it
          is used unchanged;
        * anything else -- it is a panel-sized phase, and the aperture covers
          ``2 * semi_axis_model_px * model_to_panel`` panel pixels of it, which is
          cropped out and rescaled onto the grid.

        This used to always crop a centred sub-window and rescale, which
        double-counted the mapping in the first case and made every candidate at
        or above ``region / 2`` produce the same array -- a search that could
        only ever return its own edge.
        """
        a = int(min(int(semi_axis_model_px), region // 2))
        if a < 8:
            return None
        span_panel = 2.0 * a * model_to_panel
        for record in records:
            phase = np.asarray(record.phase, dtype=np.float64)
            h, w = phase.shape
            if (h, w) == (region, region):
                return phase
            r_y, r_x = span_panel / 2.0, span_panel / 2.0
            y0, y1 = int(round(h / 2.0 - r_y)), int(round(h / 2.0 + r_y))
            x0, x1 = int(round(w / 2.0 - r_x)), int(round(w / 2.0 + r_x))
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                return None  # aperture does not fit; skip this candidate
            sub = phase[y0:y1, x0:x1]
            if sub.size == 0:
                return None
            return np.asarray(
                zoom(sub, (region / sub.shape[0], region / sub.shape[1]), order=1),
                dtype=np.float64,
            )
        return None

    def amplitude(waist_model_px: float, semi_axis_model_px: int) -> np.ndarray:
        """Gaussian illumination masked to the aperture inscribed in the grid."""
        yy, xx = np.mgrid[0:region, 0:region]
        r = int(min(int(semi_axis_model_px), region // 2))
        r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
        amp = np.exp(-r2 / (2.0 * max(float(waist_model_px), 1e-6) ** 2))
        amp[r2 > float(r) ** 2] = 0.0
        return amp

    best: tuple[float, BenchGeometry] | None = None
    # (disc, waist) are searched jointly against the correlation. Matching a
    # spot *size* first was tried and abandoned: the FWHM of a speckle cut
    # through the global max is a noisy, non-monotonic function of the beam
    # waist, so a bisection on it converged to ~60% error, whereas the speckle
    # correlation peaks exactly when the pupil mapping is right.
    waist_grid_candidates = [
        region / float(2**k) for k in range(1, 7)
    ]  # region/2 .. region/64
    for semi_axis in disc_candidates:
        semi_axis = int(min(int(semi_axis), region // 2))
        grid = pupil(semi_axis)
        if grid is None:
            continue
        opt = ZernikeCoefficientOptimizer(
            n_orders=1,
            region=region,
            dtype="float64",
            device="cpu",
            far_field_size=far_field_size,
        )
        zeros = np.zeros(opt.n_coefficients)
        for waist_grid in waist_grid_candidates:
            model = opt.forward_intensity(
                zeros, grid, amplitude(waist_grid, semi_axis)
            )
            window = _model_window(model, reference.shape, camera_px_per_model_px)
            if window.shape != reference.shape:
                # Real frames are 250x248, not square, so the window is resampled
                # per axis rather than through the square-only _resize_grid.
                window = np.asarray(
                    zoom(
                        window,
                        (
                            reference.shape[0] / window.shape[0],
                            reference.shape[1] / window.shape[1],
                        ),
                        order=1,
                    ),
                    dtype=np.float64,
                )
                if window.shape != reference.shape:
                    continue
            a = window / (window.max() + 1e-30)
            b = reference / (reference.max() + 1e-30)
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            if not np.isfinite(corr):
                continue
            disc_px = int(round(semi_axis * model_to_panel))
            geo = BenchGeometry(
                panel_disc_radius=disc_px,
                region=region,
                beam_waist_panel_px=float(waist_grid * model_to_panel),
                far_field_size=far_field_size,
                camera_px_per_model_px=float(camera_px_per_model_px),
                spot_fwhm_camera_px=fwhm_meas,
                spot_fwhm_model_px=float(_spot_fwhm(model)),
                correlation=corr,
            )
            if best is None or geo.correlation > best[1].correlation:
                best = (corr, geo)

    if best is None:
        raise ValueError(
            "no candidate panel radius produced a usable model; check the record "
            "phase shapes and far_field_size"
        )
    # A low correlation, or a winner sitting on the edge of the search grid,
    # means the data cannot discriminate the geometry -- not that the geometry
    # was found. Returning the argmax silently would dress that up as a
    # calibration, so say it plainly.
    waist_grid_candidates_all = [region / float(2**k) for k in range(1, 7)]
    saturated = (
        best[1].beam_waist_panel_px
        >= waist_grid_candidates_all[0] * model_to_panel - 1e-9
    )
    if best[1].correlation < _MIN_CORRELATION or saturated:
        logger.warning(
            "bench geometry is NOT trustworthy: correlation {:.4f}, aperture {} px, "
            "w0 {:.1f} panel px. The pupil phase must vary across the illuminated "
            "area for this to work; Zernike runs built over a large radius leave "
            "the beam nearly flat and cannot pin the geometry. Also check that "
            "panel_span_px matches the panel box the probes were displayed on, and "
            "that the displayed phase actually covered the beam (the runbook "
            "writes one phase/CCD image per probe for exactly this).",
            best[1].correlation,
            best[1].panel_disc_radius,
            best[1].beam_waist_panel_px,
        )
    logger.info(
        "bench geometry: aperture={} px, w0={:.1f} panel px, {:.3f} cam px per "
        "model px, sigma {:.2f} cam px, correlation {:.4f}",
        best[1].panel_disc_radius,
        best[1].beam_waist_panel_px,
        best[1].camera_px_per_model_px,
        best[1].spot_fwhm_camera_px,
        best[1].correlation,
    )
    return best[1]


@dataclass
class SweepRecord:
    """One point of a tilt / defocus calibration sweep.

    Attributes:
        label: Identifier, echoed into error messages.
        mode: ``"flat"``, ``"tilt"`` or ``"defocus"``.
        coefficient: Applied Zernike coefficient in radians.
        axis: ``"x"`` / ``"y"`` for tilt, empty otherwise.
        fwhm_px: Measured spot FWHM in camera pixels.
        centroid_x: Measured spot centroid x, camera pixels.
        centroid_y: Measured spot centroid y, camera pixels.
        peak: Measured peak, for a saturation cross-check.
        hollowness: Intensity at the centroid divided by the peak. ~1 for a
            single spot lobe, well below 1 once the focal plane breaks into rings.
            Used to keep ring points out of the defocus fit, which width cannot
            do: a -4 rad ring measured *narrower* than the -2.5 rad lobe, so a
            width threshold selects exactly the wrong points.
        shift_x: Sub-pixel pattern shift in camera px, from phase correlation
            against the flat reference. **Preferred over the centroid for tilt**:
            a tilt large enough to give signal deforms the spot, and at +/-1.5 rad
            the centroid readings jumped +/-8 px between repeats of the same
            phase, halving the fitted slope. ``nan`` when not measured.
        shift_y: As ``shift_x``, along camera y.
        shift_corr: Height of the phase-correlation peak (1.0 = perfect match).
    """

    label: str
    mode: str
    coefficient: float
    axis: str
    fwhm_px: float
    centroid_x: float
    centroid_y: float
    peak: float
    hollowness: float = 1.0
    shift_x: float = float("nan")
    shift_y: float = float("nan")
    shift_corr: float = float("nan")


def _linear_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)``; ``(nan, nan)`` if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or not np.isfinite(y).all():
        return float("nan"), float("nan")
    var = float(np.sum((x - x.mean()) ** 2))
    if var <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / var)
    return slope, float(y.mean() - slope * x.mean())


def _pupil_amplitude(
    region: int, waist_model_px: float, semi_axis_model_px: int
) -> np.ndarray:
    """Gaussian illumination masked to the aperture inscribed in the grid."""
    yy, xx = np.mgrid[0:region, 0:region]
    r = int(min(int(semi_axis_model_px), region // 2))
    r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
    amp = np.exp(-r2 / (2.0 * max(float(waist_model_px), 1e-6) ** 2))
    amp[r2 > float(r) ** 2] = 0.0
    return amp


def calibrate_bench_geometry_from_sweep(
    records: Sequence[SweepRecord],
    *,
    region: int,
    far_field_size: int,
    model_to_panel: float,
    semi_axis_model_px: int | None = None,
    waist_candidates: Sequence[float] | None = None,
    aperture_m: float | None = None,
    camera_pixel_m: float | None = 2.2e-6,
    focal_length_m: float | None = 0.125,
    wavelength_m: float = 1064e-9,
) -> BenchGeometry:
    """Calibrate the bench from a tilt + defocus sweep instead of speckle.

    Speckle correlation is unusable on this bench: the panel's effective phase
    resolution is far coarser than one pixel, so a random pupil phase leaves the
    far field a single tight focus instead of a speckle field, and the model has
    nothing to correlate against. A sweep of *smooth* Zernike modes does produce
    a measurable response, and two of them are enough to fix the geometry without
    assuming anything about the bench:

    **Tilt -> far-field scale.** A unit-RMS Zernike tilt of coefficient ``t``
    over an aperture of radius ``a`` imposes a phase gradient ``2t/a``, which
    deflects the beam by ``2t/(k a)`` and shifts the focal spot by
    ``f * t * lambda / (pi * a)``. The model needs
    ``camera_px_per_model_px = lambda f / (P * d_model * camera_pixel)``.
    Dividing the two cancels lambda, f, the model pitch and the camera pixel
    outright and leaves

        ``camera_px_per_model_px = k_tilt * pi * a / P``

    where ``k_tilt`` is the *measured* centroid shift in camera pixels per radian
    of tilt coefficient. That is why this measurement is the right one: it is
    immune to every geometric constant that was wrong before.

    **Defocus -> beam waist.** The spot width grows as the square root of a
    quadratic in the defocus coefficient, whose curvature depends on the
    illumination profile. The waist is fitted by matching the model's width
    response to the measured one, not by a spot-size bisection (which is noisy
    and non-monotonic).

    The two tilt axes are also compared, which doubles as a check on the
    panel/camera axis mapping -- on this bench the axes are swapped 90 degrees,
    and a run whose slopes disagree by more than a few percent says the panel
    was indexed inconsistently.

    Args:
        records: Sweep points, at least two non-collinear tilts and two
            non-zero defocuses.
        region: Model grid side length.
        far_field_size: Zero-padding size the forward model will use.
        model_to_panel: Model pixels to panel pixels (the model grid spans
            ``model_to_panel * region`` panel pixels).
        semi_axis_model_px: Aperture semi-axis in model pixels. Defaults to the
            inscribed circle, ``region // 2``.
        waist_candidates: Gaussian waists to try, in model pixels.
        aperture_m: Illuminated aperture radius in metres, used for the physical
            cross-check on the tilt slope. Pass ``None`` to skip the check.
        camera_pixel_m: Camera pixel pitch, metres, for the same check.
        focal_length_m: 2f lens focal length, metres, for the same check.
        wavelength_m: Laser wavelength, metres, for the same check.

    Returns:
        The best-fitting :class:`BenchGeometry`, with ``method="sweep"``.

    Raises:
        ValueError: If the records do not contain a usable tilt or defocus sweep.
    """
    tilts = [r for r in records if r.mode == "tilt"]
    ramps = [r for r in records if r.mode == "ramp"]
    defocuses = [r for r in records if r.mode == "defocus"]
    flats = [r for r in records if r.mode == "flat"]
    if len(tilts) < 2:
        raise ValueError(f"need >= 2 tilt points, got {len(tilts)}")
    if len(defocuses) < 2:
        raise ValueError(f"need >= 2 defocus points, got {len(defocuses)}")

    a = int(min(semi_axis_model_px or (region // 2), region // 2))
    if a < 8:
        raise ValueError(f"semi_axis_model_px must be >= 8, got {a}")

    # --- tilt -> far-field scale, per axis -----------------------------------
    # Prefer the phase-correlation shift over the centroid: a tilt big enough to
    # give a usable signal also deforms the spot, and the centroid of a
    # split/multi-lobed spot jumps by several pixels between repeats of the same
    # phase. Measured here: +/-1.5 rad gave centroid readings scattered over
    # 1020.8..1035.9 and a slope of 1.72 cam px/rad, against 3.3-3.6 from the
    # smaller single-lobed range -- a 2x error in the number the whole
    # calibration rests on.
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    sources: dict[str, str] = {}
    for axis in ("x", "y"):
        pts = [r for r in tilts if r.axis == axis]
        if len(pts) < 2:
            continue
        coeffs = np.array([r.coefficient for r in pts], dtype=np.float64)
        use_shift = all(
            np.isfinite(r.shift_x) and np.isfinite(r.shift_y) for r in pts
        ) and all(r.shift_corr >= _MIN_SHIFT_CORRELATION for r in pts)
        if use_shift:
            sx, ix = _linear_slope(
                coeffs, np.array([r.shift_x for r in pts], dtype=np.float64)
            )
            sy, iy = _linear_slope(
                coeffs, np.array([r.shift_y for r in pts], dtype=np.float64)
            )
            sources[axis] = "phase-correlation"
        else:
            sx, ix = _linear_slope(
                coeffs, np.array([r.centroid_y for r in pts], dtype=np.float64)
            )
            sy, iy = _linear_slope(
                coeffs, np.array([r.centroid_x for r in pts], dtype=np.float64)
            )
            sources[axis] = "centroid (no usable phase correlation)"
        if np.isfinite(sx) and abs(sx) >= abs(sy):
            slopes[axis], intercepts[axis] = sx, ix
        else:
            slopes[axis], intercepts[axis] = sy, iy
    if not slopes:
        raise ValueError("no tilt axis had two usable points")
    k_tilt = float(np.mean([abs(v) for v in slopes.values()]))
    if not np.isfinite(k_tilt) or k_tilt <= 0.0:
        raise ValueError(f"tilt slope is not usable: {slopes}")

    # --- focal scale ---------------------------------------------------------
    # Preferred source: a **phase ramp** sweep. A 2*pi ramp over P panel px moves
    # the spot by `S/P` camera px, measured cleanly and repeatably (2-3 repeats
    # agreeing to 0.2 px, cleanly 1/P, consistent with theory). A Zernike tilt is
    # equivalent to a ramp of period pi*R/c, so the two must agree -- but on this
    # bench they did not: the Zernike route returned 1.63 cam px/rad where the
    # ramp says 5.25, because a Zernike tilt large enough for signal deforms the
    # spot and the centroid stops tracking the peak. The ramp has no such
    # failure mode, so it wins.
    #
    # With k_tilt = S/(pi*R) and a model semi-axis of `a` model px covering the
    # same physical R = a*model_to_panel panel px,
    #     camera_px_per_model_px = k_tilt * pi * a / P_far_field
    #                              = S / (model_to_panel * far_field_size)
    ramp_scale = float("nan")
    ramp_axis = ""
    if len(ramps) >= 2:
        for ax in ("x", "y"):
            pts = [r for r in ramps if r.axis == ax]
            if len(pts) < 2:
                continue
            periods = np.array([r.coefficient for r in pts], dtype=np.float64)
            if np.any(periods <= 0):
                continue
            inv = 1.0 / periods
            sx, _ = _linear_slope(
                inv, np.array([r.centroid_x for r in pts], dtype=np.float64)
            )
            sy, _ = _linear_slope(
                inv, np.array([r.centroid_y for r in pts], dtype=np.float64)
            )
            s = sx if abs(sx) >= abs(sy) else sy
            if abs(s) > abs(ramp_scale) or not np.isfinite(ramp_scale):
                ramp_scale, ramp_axis = float(s), ax
    if np.isfinite(ramp_scale) and ramp_scale != 0.0:
        camera_px_per_model_px = abs(ramp_scale) / (model_to_panel * float(far_field_size))
        focal_source = f"ramp sweep (S={abs(ramp_scale):.0f} cam px per 1/period, axis {ramp_axis})"
    else:
        camera_px_per_model_px = k_tilt * np.pi * a / float(far_field_size)
        focal_source = (
            f"Zernike tilt slope {k_tilt:.2f} cam px/rad via {sorted(set(sources.values()))}"
        )
        logger.warning(
            "no usable ramp sweep, so the focal scale comes from the Zernike tilt "
            "slope -- which has measured 3x too small on this bench when the spot "
            "deforms. Run --stage sweep with a ramp sweep included, or calibrate "
            "with `python -m ao_shaping.tools.slm.slm_tilt_probe`."
        )

    # Independent cross-check, derived a completely different way: the model's
    # far-field field of view against the camera's.
    #
    #     p_fft = lambda*f / (P_far_field * model_pitch)   [m per model px]
    #     model FOV  = far_field_size * p_fft
    #     camera FOV = camera_px * camera_pixel
    #     -> camera_px_per_model_px = FOV_model / FOV_camera
    #
    # Disagreement with the ramp route beyond ~25% means one of the two is wrong,
    # and the honest response is to say so rather than pick one.
    fov_ratio = float("nan")
    if aperture_m and camera_pixel_m and focal_length_m and a > 0:
        # One model pixel spans aperture_m / a metres, since `a` model pixels
        # cover the illuminated radius. (Multiplying by model_to_panel here was a
        # bug: it double-counts the panel->model scale and inflated the pitch 3.5x,
        # which is most of why the two routes looked 124% apart.)
        model_pitch_m = aperture_m / a
        p_fft = wavelength_m * focal_length_m / (far_field_size * model_pitch_m)
        fov_ratio = (far_field_size * p_fft) / (2592.0 * camera_pixel_m)
        if np.isfinite(fov_ratio) and fov_ratio > 0:
            rel = abs(camera_px_per_model_px / fov_ratio - 1.0)
            logger.info(
                "focal scale {:.4f} cam px per model px (from {}) vs {:.4f} from "
                "the independent field-of-view ratio -- {:.0%} apart",
                camera_px_per_model_px, focal_source, fov_ratio, rel,
            )
            if rel > 0.25:
                logger.warning(
                    "the two focal-scale routes disagree by {:.0%} ({} vs {:.4f} "
                    "from the FOV ratio). At least one assumption is wrong -- most "
                    "likely the illuminated aperture radius, which is what both "
                    "routes scale with. Re-measure it with "
                    "`python -m ao_shaping.tools.slm.slm_beam_extent` before "
                    "trusting either number.",
                    rel, focal_source, fov_ratio,
                )

    # Physical cross-check on the Zernike tilt slope, kept as a diagnostic: the
    # ramp route is authoritative, but if a Zernike sweep is present and its slope
    # is nowhere near the ramp's equivalent, say so rather than silently ignoring
    # it.

    # Physical cross-check on the tilt measurement. A unit-RMS Zernike tilt of
    # coefficient c over an aperture of radius R imposes a phase gradient 2c/R,
    # which deflects by (1/k)(2c/R) = c*lambda/(pi*R) and shifts the focal spot by
    # f times that. So the predicted slope is
    #     k_pred = f * lambda / (pi * R * camera_pixel)   [camera px per rad]
    # This is what makes the measurement falsifiable. The wide-angle runs measured
    # 1.63-1.74 cam px/rad where 5.34 was predicted -- a ratio of 0.31, which
    # back-solves to an effective aperture of 11 mm, larger than the panel's short
    # side, so those runs were corrupted (the spot had deformed and the centroid
    # stopped tracking the peak) rather than revealing a huge beam.
    if aperture_m and camera_pixel_m and focal_length_m:
        k_pred = (
            focal_length_m * wavelength_m
            / (np.pi * aperture_m * camera_pixel_m)
        )
        ratio = k_tilt / k_pred if k_pred > 0 else float("nan")
        tilt_ratio = float(ratio)
        implied_aperture_mm = (
            focal_length_m * wavelength_m / (np.pi * camera_pixel_m * k_tilt) * 1e3
            if k_tilt > 0 else float("nan")
        )
        if not (0.4 <= ratio <= 1.6):
            logger.warning(
                "tilt slope {:.3f} cam px/rad is {:.2f}x the {:.3f} predicted for a "
                "{:.1f} mm illuminated aperture -- OUTSIDE the credible band "
                "0.4-1.6, so the focal scale derived from it ({:.4f} cam px per "
                "model px) is NOT trustworthy. It back-solves to an effective "
                "aperture of {:.1f} mm; the panel's short side is 9.6 mm, so a "
                "measurement implying more than that is corrupted, not a discovery. "
                "Re-run the tilt sweep with smaller coefficients.",
                k_tilt, ratio, k_pred, 1e3 * aperture_m,
                camera_px_per_model_px, implied_aperture_mm,
            )
    else:
        tilt_ratio = float("nan")
        implied_aperture_mm = float("nan")

    # --- width responses: defocus + astigmatism, fitted jointly ---------------
    # Defocus alone cannot identify the waist on this bench: its width response
    # stayed 16% asymmetric after the bench's own defocus offset was removed,
    # which is astigmatism/coma the defocus sweep does not contain. Sweeping both
    # quadratic modes and fitting them together separates the two contributions.
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical"):
        pts = [r for r in records if r.mode == mode]
        if len(pts) < 2:
            continue
        c = np.array([r.coefficient for r in pts], dtype=np.float64)
        w = np.array([r.fwhm_px for r in pts], dtype=np.float64)
        h = np.array([r.hollowness for r in pts], dtype=np.float64)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(h)
        if int(ok.sum()) < 2:
            continue
        c, w, h = c[ok], w[ok], h[ok]

        # A width of `w` is only meaningful while the focal plane is a SINGLE
        # lobe. The direct measure of that is hollowness (intensity at the
        # centroid / peak): a spherical -1.2 rad point read 38.2 px wide with
        # hollowness 0.43, i.e. a ring whose "width" is a ring diameter, and
        # fitting it as a Gaussian lobe is meaningless. The constant below was
        # defined for this but never applied; it is used *alone* here, and only
        # after an alternative was measured and rejected -- see the note on
        # _DEFOCUS_SINGLE_LOBE_FACTOR.
        single_lobe = h >= _DEFOCUS_MIN_HOLLOWNESS
        if not single_lobe.all() and int(single_lobe.sum()) >= 2 and _has_both_signs(
            c[single_lobe]
        ):
            logger.info(
                "{} sweep: dropped {} point(s) that are not a single spot lobe "
                "(hollowness < {:.2f}): {}",
                mode, int((~single_lobe).sum()), _DEFOCUS_MIN_HOLLOWNESS,
                ", ".join(
                    f"c={cc:+.2f} w={ww:.1f} hollow={hh:.2f}"
                    for cc, ww, hh in zip(
                        c[~single_lobe], w[~single_lobe], h[~single_lobe]
                    )
                ),
            )
            c, w, h = c[single_lobe], w[single_lobe], h[single_lobe]

        # A single spot lobe under growing |coefficient| must broaden
        # monotonically. Enforce that per side and drop the violators: a real
        # sweep had a -4 rad point read *narrower* (12.5 px) than the -2.5 rad one
        # (30.9 px), and that single outlier is enough to drive the fitted
        # curvature negative. A width threshold and hollowness both fail to catch
        # that one -- the outlier's hollowness (0.80) and width both sit inside
        # the single-lobe range -- so the two filters are complementary and both
        # are needed.
        keep = np.ones(c.shape, dtype=bool)
        for side in (np.sign(c) < 0, np.sign(c) > 0):
            idx = np.flatnonzero(side)
            if idx.size == 0:
                continue
            running = -np.inf
            for j in idx[np.argsort(np.abs(c[idx]))]:
                if w[j] >= running:
                    running = w[j]
                else:
                    keep[j] = False
        if int(keep.sum()) >= 4 and _has_both_signs(c[keep]):
            dropped = int((~keep).sum())
            if dropped:
                logger.info(
                    "{} sweep: dropped {} non-monotonic width point(s); "
                    "fitting the remaining {}",
                    mode, dropped, int(keep.sum()),
                )
            c, w = c[keep], w[keep]
        groups[mode] = (c, w)
    if "defocus" not in groups:
        raise ValueError("defocus sweep has fewer than two usable FWHM points")

    # Per-mode bench offset: the width minimum is where the bench's own
    # aberration cancels the command, so fit it rather than assuming it sits at
    # zero (measured -0.171 rad here; assuming zero gave -1.7 rad).
    offsets: dict[str, float] = {}
    curvature_report: dict[str, float] = {}
    for mode, (c, w) in list(groups.items()):
        grid = np.linspace(c.min() - 1.0, c.max() + 1.0, 241)
        best_q: tuple[float, float, float] | None = None
        for c0 in grid:
            basis = (c - c0) ** 2
            design = np.stack([np.ones_like(basis), basis], axis=1)
            sol, *_ = np.linalg.lstsq(design, w**2, rcond=None)
            resid = design @ sol - w**2
            rms = float(np.sqrt(np.mean(resid**2)))
            if best_q is None or rms < best_q[0]:
                best_q = (rms, float(c0), float(sol[1]))
        assert best_q is not None
        rms_q, c0_q, curv_q = best_q
        # A width response that does not open upward has no physical
        # interpretation, and an offset pinned to the scan edge means the
        # minimum was never bracketed. Both happen when a mode simply does not
        # respond -- emitting a number anyway is how an astig_y "offset" of
        # +5.0 rad (the grid edge) with curvature -22 appeared. Drop the mode and
        # keep the ones that do respond.
        edge = (c0_q <= grid[0] + 1e-9) or (c0_q >= grid[-1] - 1e-9)
        if curv_q <= 0.0 or edge:
            logger.warning(
                "{} width response is degenerate (curvature {:.1f}, offset {:+.3f} "
                "{}) -- dropping this mode from the fit rather than reporting a "
                "meaningless offset. Check that the mode is actually reachable on "
                "this bench: its spot widths were {}",
                mode, curv_q, c0_q, "at the scan edge" if edge else "",
                ", ".join(f"{v:.1f}" for v in w),
            )
            groups.pop(mode)
            continue
        offsets[mode] = c0_q
        curvature_report[mode] = curv_q
    if not groups:
        raise ValueError(
            "no swept mode produced a usable width response; the spot width does "
            "not respond to any of them on this bench"
        )

    # Width asymmetry of the defocus response at matched |c| is a direct read of
    # residual non-defocus aberration.
    c_d, w_d = groups["defocus"]
    asym: list[float] = []
    for c_pos in c_d[c_d > 0]:
        neg = c_d[np.isclose(c_d, -c_pos)]
        if neg.size:
            w_p = float(w_d[c_d == c_pos][0])
            w_n = float(w_d[c_d == neg[0]][0])
            if w_p > 0 and w_n > 0:
                asym.append((w_p - w_n) / (0.5 * (w_p + w_n)))
    asymmetry = float(np.mean(asym)) if asym else float("nan")

    if waist_candidates is None:
        waist_candidates = [region / float(2**k) for k in range(0, 8)]
    opt = ZernikeCoefficientOptimizer(
        n_orders=1, region=region, dtype="float64", device="cpu",
        far_field_size=far_field_size,
    )
    zeros = np.zeros(opt.n_coefficients)
    span = max(float(np.max(np.abs(c))) for c, _ in groups.values()) + 1.0
    resp_grid = np.linspace(-span, span, _SWEEP_RESPONSE_POINTS)

    # Per waist: precompute each mode's model width response once, then fit the
    # per-mode offsets by interpolating against it. Evaluating the model at every
    # (waist, mode, coefficient, offset) would be ~500x more forward passes.
    best: tuple[float, float, dict[str, float]] | None = None
    for waist in waist_candidates:
        amp = _pupil_amplitude(region, float(waist), a)
        total = 0.0
        n_pts = 0
        per_mode: dict[str, float] = {}
        ok_waist = True
        for mode, (c, w) in groups.items():
            nm = _SWEEP_MODE_NM[mode]
            curve = np.array(
                [
                    _spot_fwhm(
                        opt.forward_intensity(
                            zeros, _zernike_phase(region, a, nm, float(g)), amp
                        )
                    )
                    * camera_px_per_model_px
                    for g in resp_grid
                ],
                dtype=np.float64,
            )
            good = np.isfinite(curve) & (curve > 0)
            if good.sum() < 3:
                ok_waist = False
                break
            best_off, best_rms = 0.0, np.inf
            for c0 in resp_grid:
                pred = np.interp(c - c0, resp_grid, curve)
                okp = np.isfinite(pred) & (pred > 0)
                if int(okp.sum()) < 2:
                    continue
                rms = float(
                    np.sqrt(np.mean(((pred[okp] - w[okp]) / w[okp]) ** 2))
                )
                if rms < best_rms:
                    best_rms, best_off = rms, float(c0)
            if not np.isfinite(best_rms):
                ok_waist = False
                break
            per_mode[mode] = best_off
            total += best_rms * c.size
            n_pts += c.size
        if not ok_waist or n_pts == 0:
            continue
        joint = total / n_pts
        if best is None or joint < best[0]:
            best = (joint, float(waist), per_mode)
    if best is None:
        raise ValueError("no waist candidate produced a finite width response")
    fit_rms, waist_fit, mode_offsets = best
    for mode, off in mode_offsets.items():
        offsets[mode] = off

    # The illumination profile is only identified when the joint fit is good.
    # Otherwise the best-fit waist is an artefact of the model/bench mismatch, so
    # emit the flat-top default and say so rather than a number that looks
    # calibrated.
    waist_determined = fit_rms <= _MAX_DEFOCUS_FIT_RMS
    if waist_determined:
        waist_model = waist_fit
    else:
        waist_model = float(a)
        logger.warning(
            "width-response fit is poor (joint rel-RMS {:.1%}; defocus asymmetry "
            "{:+.0%}), so the illumination profile is NOT identifiable: the bench "
            "carries aberration the swept modes do not cover. Emitting the flat-top "
            "default w0={:.0f} model px instead of the best fit {:.0f}. The "
            "tilt-derived scale ({:.4f} cam px per model px) is unaffected -- it "
            "comes from the centroid slope alone. Add coma/spherical sweeps to "
            "identify the waist.",
            fit_rms, asymmetry, waist_model, waist_fit, camera_px_per_model_px,
        )

    # --- reported spot widths ------------------------------------------------
    flat_fwhm = float(np.median([r.fwhm_px for r in flats])) if flats else float("nan")
    if not np.isfinite(flat_fwhm):
        flat_fwhm = float(np.min(w_d))
    amp = _pupil_amplitude(region, waist_model, a)
    flat_model = opt.forward_intensity(zeros, np.zeros((region, region)), amp)
    flat_model_fwhm = float(_spot_fwhm(flat_model))

    modes_used = "+".join(sorted(groups))
    notes = (
        f"tilt slope {k_tilt:.3f} cam px/rad ("
        + ", ".join(f"{ax}:{v:+.3f}" for ax, v in sorted(slopes.items()))
        + f", via {sorted(set(sources.values()))}), semi-axis {a} model px, "
        f"P={far_field_size}"
        + (
        f"; physical tilt check {tilt_ratio:.2f}x predicted"
        if np.isfinite(tilt_ratio)
        else ""
    )
        + f"; focal scale from {focal_source}"
        + (
            f", FOV cross-check {fov_ratio:.3f} ({abs(camera_px_per_model_px / fov_ratio - 1.0):.0%} apart)"
            if np.isfinite(fov_ratio) and fov_ratio > 0
            else ""
        )
        + f"; modes fitted {modes_used}; joint rel-RMS {fit_rms:.1%}; bench offsets "
        + ", ".join(f"{m} {o:+.3f}" for m, o in sorted(offsets.items()))
        + f" rad; curvatures "
        + ", ".join(f"{m} {v:.1f}" for m, v in sorted(curvature_report.items()))
        + f"; defocus asymmetry {asymmetry:+.0%}; waist "
        + ("fitted" if waist_determined else "NOT identifiable, flat-top default")
    )
    logger.info(
        "bench geometry from sweep: aperture={} panel px, w0={:.1f} panel px, "
        "{:.4f} cam px per model px, flat FWHM {:.1f} cam px, joint fit {:.1%} "
        "over {}, asymmetry {:+.0%}, waist {}{}",
        int(round(a * model_to_panel)),
        waist_model * model_to_panel,
        camera_px_per_model_px,
        flat_fwhm,
        fit_rms,
        modes_used,
        asymmetry,
        "fitted" if waist_determined else "provisional",
        (
            f", tilt slope {tilt_ratio:.2f}x the physical prediction"
            if np.isfinite(tilt_ratio)
            else ""
        ),
    )
    for mode in sorted(offsets):
        logger.info(
            "  bench {}: offset {:+.3f} rad, width^2 curvature {:.1f}, {} points",
            mode, offsets[mode], curvature_report[mode], groups[mode][0].size,
        )
    return BenchGeometry(
        panel_disc_radius=int(round(a * model_to_panel)),
        region=region,
        beam_waist_panel_px=float(waist_model * model_to_panel),
        far_field_size=far_field_size,
        camera_px_per_model_px=float(camera_px_per_model_px),
        spot_fwhm_camera_px=flat_fwhm,
        spot_fwhm_model_px=flat_model_fwhm,
        correlation=float(max(0.0, 1.0 - fit_rms)),
        method="sweep",
        calibration_notes=notes,
    )

def _defocus_phase(region: int, semi_axis_model_px: int, coefficient: float) -> np.ndarray:
    """A unit-RMS Zernike defocus of the given coefficient on the model grid."""
    return _zernike_phase(region, semi_axis_model_px, (2, 0), coefficient)


# Zernike modes the width sweep drives, and the grid the model response is
# precomputed on. The grid must span the measured coefficients plus a margin so
# the per-mode offset can be fitted by interpolation.
_SWEEP_MODE_NM: dict[str, tuple[int, int]] = {
    "defocus": (2, 0),
    "astig_x": (2, -2),
    "astig_y": (2, 2),
    # Coma and spherical are what make the waist identifiable. Defocus alone
    # cannot separate "wide beam" from "residual quadratic aberration", and the
    # quadratic pair is degenerate with it on this bench; the cubic/quartic pair
    # has a different width-vs-coefficient signature, so fitting them together is
    # what breaks the degeneracy.
    "coma_x": (3, -1),
    "coma_y": (3, 1),
    "spherical": (4, 0),
}
_SWEEP_RESPONSE_POINTS = 21


def _zernike_phase(
    region: int, semi_axis_model_px: int, nm: tuple[int, int], coefficient: float
) -> np.ndarray:
    """A unit-RMS Zernike mode of the given coefficient on the model grid."""
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    r = int(min(int(semi_axis_model_px), region // 2))
    gen = ZernikeGenerator((region, region), radius=r, n_orders=nm[0])
    mode = np.asarray(gen.generate_polynomial({nm: float(coefficient)}), dtype=np.float64)
    if mode.shape != (region, region):  # generator may return (W, H)
        mode = mode.T
    out = np.zeros((region, region), dtype=np.float64)
    finite = np.isfinite(mode)
    out[finite] = mode[finite]
    return out


# ---------------------------------------------------------------------------
# Shared-aberration calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationRecord:
    """One ``(far field, pupil phase)`` pair for the shared fit.

    Attributes:
        image: Measured far field. Any non-empty 2D shape; cropped to the model
            grid by the calibration.
        phase: Pupil phase that produced ``image``, raw unwrapped radians. Any
            non-empty 2D shape; resampled to the model grid.
        label: Human-readable provenance, echoed in the result.
        coefficients: Optional per-record ground truth in Noll order, used only
            to report the per-record fit error.
    """

    image: np.ndarray
    phase: np.ndarray
    label: str = "explicit"
    coefficients: np.ndarray | None = None


@dataclass
class FitAberrationResult:
    """Outcome of :func:`calibrate_shared_aberration`.

    Attributes:
        coefficients: The shared fitted coefficients, Noll-ordered, radians.
        loss_history: Peak-normalised MSE per Adam step across all records.
        labels: Provenance of each record consumed.
        source: Where the records came from (``"explicit"``, a dataset path, or
            ``"none"`` when no usable data was found).
        n_records: Number of records the fit consumed.
        epochs: Optimization steps granted to each record.
        iterations: Adam steps actually taken.
        record_errors: Per-record L2 error against ``record.coefficients``,
            empty when no ground truth was supplied.
    """

    coefficients: np.ndarray
    loss_history: list[float]
    labels: list[str]
    source: str
    n_records: int
    epochs: int
    iterations: int
    record_errors: list[float] = field(default_factory=list)


def _resize_grid(array: np.ndarray, grid: int) -> np.ndarray:
    """Resample a 2D array onto a ``(grid, grid)`` grid by nearest neighbour.

    Kept local rather than imported from ``ml.gsnet_debug.offline`` so the
    optimizer layer never depends on the runner layer (and never drags its
    plotting/CLI imports into an algorithm import).

    Args:
        array: Any non-empty 2D array.
        grid: Target side length.

    Returns:
        A ``(grid, grid)`` float64 array.

    Raises:
        ValueError: If ``array`` is empty or not 2D, or ``grid < 1``.
    """
    frame = np.asarray(array, dtype=np.float64)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"expected a non-empty 2D array, got shape {frame.shape}")
    if int(grid) < 1:
        raise ValueError(f"grid must be a positive integer, got {grid}")
    rows = np.clip(
        np.round(np.linspace(0, frame.shape[0] - 1, grid)).astype(int),
        0,
        frame.shape[0] - 1,
    )
    cols = np.clip(
        np.round(np.linspace(0, frame.shape[1] - 1, grid)).astype(int),
        0,
        frame.shape[1] - 1,
    )
    return frame[np.ix_(rows, cols)]


def _crop_to_grid(image: np.ndarray, grid: int) -> np.ndarray:
    """Crop a far field around its global ``argmax`` onto a ``grid`` window.

    The 0-order is the frame's global maximum, never the geometric centre, so
    the window is anchored by
    :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center` with
    ``refine=False``. The window is clamped inside the frame, so a frame smaller
    than ``grid`` yields a zero-padded result rather than an exception.

    Args:
        image: 2D far-field intensity frame.
        grid: Output side length.

    Returns:
        A ``(grid, grid)`` float64 window centred on the 0-order.
    """
    frame = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0)
    side = int(grid)
    center_x, center_y = zero_order_center(frame, refine=False)
    out = np.zeros((side, side), dtype=np.float64)
    top = center_y - side // 2
    left = center_x - side // 2
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1 = min(src_y0 + side, frame.shape[0])
    src_x1 = min(src_x0 + side, frame.shape[1])
    # A window whose anchor is off-centre hangs off the far edge as well as the
    # near one, so the destination span can run past ``out`` (rows 10:34 of a
    # 32-row buffer). Clamp the destination and copy only the overlapping part
    # of the source; the remainder of ``out`` stays zero padding.
    dst_y0, dst_x0 = src_y0 - top, src_x0 - left
    dst_y1 = min(dst_y0 + (src_y1 - src_y0), side)
    dst_x1 = min(dst_x0 + (src_x1 - src_x0), side)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[
        src_y0 : src_y0 + (dst_y1 - dst_y0),
        src_x0 : src_x0 + (dst_x1 - dst_x0),
    ]
    return out


def _coerce_to_record(
    item: Any, region: int, n_coeffs: int, label: str
) -> CalibrationRecord | None:
    """Coerce a loaded dataset entry into a :class:`CalibrationRecord`.

    Accepts either a mapping with ``_img`` / ``_phase`` (and optionally ``_c``)
    or an object exposing ``image`` / ``phase`` attributes.

    Args:
        item: The loaded entry.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.
        label: Fallback provenance label.

    Returns:
        A record, or ``None`` when the entry lacks an image or a phase.
    """
    if isinstance(item, Mapping):
        image = item.get("_img", item.get("image"))
        phase = item.get("_phase", item.get("phase"))
        truth = item.get("_c", item.get("coefficients"))
    else:
        image = getattr(item, "image", None)
        phase = getattr(item, "phase", None)
        truth = getattr(item, "coefficients", None)
    if image is None or phase is None:
        return None
    vector: np.ndarray | None = None
    if truth is not None:
        vector = np.zeros(n_coeffs, dtype=np.float64)
        raw = np.asarray(truth, dtype=np.float64).ravel()
        keep = min(raw.size, n_coeffs)
        vector[:keep] = raw[:keep]
    return CalibrationRecord(
        image=_crop_to_grid(image, region),
        phase=_resize_grid(phase, region),
        label=label,
        coefficients=vector,
    )


def _records_from_pkl(
    path: Path, region: int, n_coeffs: int
) -> list[CalibrationRecord]:
    """Build records from a fat GSNet ``.pkl`` file.

    Args:
        path: Path to the pickle.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        The parsed records, in file order.

    Raises:
        ValueError: If the file holds no entry with both an image and a phase.
    """
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local dataset
    entries = payload if isinstance(payload, (list, tuple)) else [payload]
    records: list[CalibrationRecord] = []
    for index, entry in enumerate(entries):
        record = _coerce_to_record(entry, region, n_coeffs, f"{path.name}#{index}")
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError(f"{path} holds no entry with both an image and a phase")
    return records


def _records_from_cache(
    path: Path, region: int, n_coeffs: int
) -> list[CalibrationRecord]:
    """Build records from a lean ``.gsnet_cache`` directory.

    The lean cache stores flattened coefficients and images plus a
    ``meta.json`` but **no** pupil phase, so the phase defaults to zeros (a flat
    SLM) and the resulting fit is correspondingly weaker. Prefer the fat
    ``.pkl`` when both are present.

    Args:
        path: The cache directory.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        The parsed records.

    Raises:
        ValueError: If the cache arrays or the metadata describing their shape
            are missing or inconsistent.
    """
    coeffs = np.load(path / "c_flat.npy")
    images = np.load(path / "img_flat.npy")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in meta["img_shape"])
    per_record = int(np.prod(shape))
    if per_record <= 0 or images.size % per_record:
        raise ValueError(
            f"{path}/img_flat.npy holds {images.size} entries, not a multiple of "
            f"{per_record}"
        )
    # Reshape once rather than slicing per record: the on-disk layout is
    # ``(n_records, prod(img_shape))`` (the real cache is 2-D), so slicing a
    # *pixel* count off axis 0 would hand back whole rows instead of one frame.
    # ``reshape`` is order preserving, so flat, 2-D and 3-D caches all work.
    frames = images.reshape(images.size // per_record, *shape)
    flat_phase = np.zeros((region, region), dtype=np.float64)
    records: list[CalibrationRecord] = []
    for index in range(len(frames)):
        vector: np.ndarray | None = None
        if index < len(coeffs):
            vector = np.zeros(n_coeffs, dtype=np.float64)
            raw = np.asarray(coeffs[index], dtype=np.float64).ravel()
            keep = min(raw.size, n_coeffs)
            vector[:keep] = raw[:keep]
        records.append(
            CalibrationRecord(
                image=_crop_to_grid(frames[index], region),
                phase=flat_phase.copy(),
                label=f"{path.name}#{index}",
                coefficients=vector,
            )
        )
    if not records:
        raise ValueError(f"{path} contains no cached frames")
    return records


def _discover_dataset(
    search_root: Path, region: int, n_coeffs: int
) -> tuple[list[CalibrationRecord], str]:
    """Locate the newest usable calibration dataset under ``search_root``.

    The fat ``.pkl`` is preferred over the lean ``.gsnet_cache`` (it carries the
    pupil phase); ties on modification time prefer the ``.pkl``.

    Args:
        search_root: Directory searched recursively.
        region: Model grid side length.
        n_coeffs: Length of the coefficient vector.

    Returns:
        A tuple of the records and their source description; the list is empty
        when nothing usable was found.
    """
    if not search_root.exists():
        return [], "none"
    candidates: list[tuple[float, int, Path]] = []
    candidates.extend((p.stat().st_mtime, 0, p) for p in search_root.rglob("*.pkl"))
    candidates.extend(
        (p.stat().st_mtime, 1, p) for p in search_root.rglob("*.gsnet_cache") if p.is_dir()
    )
    if not candidates:
        return [], "none"
    _, _, newest = max(candidates)
    try:
        if newest.is_dir():
            records = _records_from_cache(newest, region, n_coeffs)
        else:
            records = _records_from_pkl(newest, region, n_coeffs)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        json.JSONDecodeError,
        pickle.UnpicklingError,
    ) as error:
        logger.warning("calibration dataset {} unusable: {}", newest, error)
        return [], "none"
    return records, str(newest)


def calibrate_shared_aberration(
    records: Sequence[CalibrationRecord] | None = None,
    *,
    search_root: Path | str | None = None,
    region: int = 64,
    n_orders: int = 10,
    epochs: int = 50,
    lr: float = 0.05,

    dtype: str = "float64",
    device: str | None = None,
    seed: int = 0,
    frozen_modes: tuple[int, ...] = (1, 2, 3),
    far_field_size: int | None = None,
) -> FitAberrationResult:
    """Fit ONE aberration shared by a whole set of records.

    A single :class:`ZernikeCoefficientOptimizer` is reused for every record so
    all records inform the same coefficient vector instead of each receiving an
    independent fit. Each record is driven by one
    :meth:`~ZernikeCoefficientOptimizer.run` call, which warm-starts from the
    coefficients accumulated so far.

    All records are fitted by **one** optimizer over consecutive
    :meth:`~ZernikeCoefficientOptimizer.update` calls, cycling through the
    records, so every record informs the same coefficient vector at every step.
    This is what makes the shared fit work. Issuing one
    :meth:`~ZernikeCoefficientOptimizer.run` per record instead optimises each
    record to its own optimum and restarts Adam in between, so the sequence
    degenerates into a chain of independent fits in which the last record wins
    and the shared vector ends up fitting none of them: on the twin at
    ``region=64``/``n_orders=10`` (66 coefficients) that per-record ``run``
    chain *raised* the loss and left the injected-mode error above 1.2 rad, and
    got worse as the budget grew.

    Cycling needs the convergence latch handled. ``update`` stops moving the
    coefficients once the plateau detector has seen ``PLATEAU_PATIENCE``
    non-improving steps, and nothing clears the latch, so a shared best-loss
    tracker across heterogeneous records trips it easily. A tripped latch is
    re-armed by rebuilding the optimizer around its *current* coefficients.
    :meth:`~ZernikeCoefficientOptimizer.reset` is deliberately not used: it
    restores the coefficients passed to ``__init__`` and would discard the whole
    fit.

    Identifiability: the fit is only well posed when the records sample

    **varied SLM phases**. A far field measured under a flat SLM phase focuses to
    a near-delta spot, so its normalized intensity barely responds to the pupil
    aberration and the coefficients are not recoverable. Datasets whose records
    all share one flat (or otherwise identical) phase will return a poor fit
    even though the optimizer behaved correctly; add records with different
    SLM phases, as the canonical recovery fixture does.

    Args:
        records: Explicit records. When ``None``, the newest dataset under
            ``search_root`` is loaded.
        search_root: Directory searched when ``records`` is ``None``. Defaults
            to ``data/debug``.
        region: Model grid side length.
        n_orders: Maximum Zernike radial order.
        epochs: Optimization steps granted to each record, so the total budget
            is ``epochs * len(records)``. Values at or above
            ``PLATEAU_PATIENCE`` may stop a record early on its own plateau.
        lr: Adam learning rate. ``0.05`` is the measured sweet spot for the
            joint cycling loop on the twin: at ``region=64``/``n_orders=10`` with
            16 records it reached a 0.19 rad full-vector error against 0.30 at
            ``0.1`` and 0.94 at ``0.01``, while ``0.2`` diverges outright (73 rad).

        dtype: Torch dtype name for the model.
        device: Torch device, or ``None`` to auto-detect.
        seed: Seed for the Step A optimizer.
        frozen_modes: Noll indices held at zero for the whole fit. Defaults to
            ``(1, 2, 3)`` because piston and tilt are unidentifiable from a
            far-field intensity; on real captures, leaving them free put 56% of
            the fitted coefficient norm into those degenerate directions without
            improving the fit at all. Pass ``()`` to fit every mode.
        far_field_size: Zero-padded FFT size for the forward model, or ``None``
            for the native ``region``-sized transform. **Must** be set when the
            records come from a real bench: the focal-plane sampling is set by
            the optical geometry, not by the model grid, so fitting on the
            native transform rescales every recovered coefficient. On this bench
            the mismatch is ~29.5x, which is larger than any coefficient being
            measured.

    Returns:
        A :class:`FitAberrationResult`. When no data is available the
        coefficients are zeros and ``source`` is ``"none"`` -- the function
        degrades gracefully instead of raising.

    Raises:
        ValueError: If ``epochs < 1``, if ``records`` is given but empty, or if
            a record is missing its image or phase.
        ImportError: If PyTorch is unavailable.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(region) < 1:
        raise ValueError(f"region must be a positive integer, got {region}")
    n_coeffs = calc_n_zernike_terms(n_orders)

    if records is None:
        root = Path(search_root) if search_root is not None else DEFAULT_SEARCH_ROOT
        found, source = _discover_dataset(root, int(region), n_coeffs)
        if not found:
            logger.warning(
                "no usable calibration dataset under {}; returning zeros", root
            )
            return FitAberrationResult(
                coefficients=np.zeros(n_coeffs, dtype=np.float64),
                loss_history=[],
                labels=[],
                source="none",
                n_records=0,
                epochs=epochs,
                iterations=0,
            )
        records = found
        logger.info("calibrating on {} record(s) from {}", len(records), source)
    else:
        source = "explicit"
        records = list(records)

    if not records:
        raise ValueError("records must contain at least one entry")

    optimizer = ZernikeCoefficientOptimizer(
        n_orders=n_orders,
        region=region,
        lr=lr,
        max_iterations=max(1, epochs),
        device=device,
        seed=seed,
        dtype=dtype,
        frozen_modes=frozen_modes,
        far_field_size=far_field_size,
    )
    history: list[float] = []
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for record in records:
        image = np.asarray(record.image, dtype=np.float64)
        if image.size == 0:
            raise ValueError(f"record {record.label!r} has an empty image")
        if image.shape != (int(region), int(region)):
            image = _crop_to_grid(image, int(region))
        phase = np.asarray(record.phase, dtype=np.float64)
        if phase.size == 0:
            raise ValueError(f"record {record.label!r} has an empty phase")
        if phase.shape != (int(region), int(region)):
            phase = _resize_grid(phase, int(region))
        # Rescaled to model units: a CCD frame peaks at ``peak_photons`` counts
        # while the model peaks in the thousands, and that mismatch would turn
        # the fit into a brightness maximisation rather than a pattern match
        # (see :func:`_to_model_units`).
        frames.append((_to_model_units(optimizer, image, phase), phase))

    def rearm(coefficients: np.ndarray) -> ZernikeCoefficientOptimizer:
        """Rebuild the optimizer around ``coefficients`` to clear a tripped latch.

        ``reset`` cannot do this: it restores the coefficients handed to
        ``__init__`` rather than the current ones, so re-arming with it would
        discard the whole fit. A fresh instance keeps the progress and discards
        only the stale latch and the Adam moments that stopped moving.
        """
        return ZernikeCoefficientOptimizer(
            n_orders=n_orders,
            region=region,
            initial_coefficients=np.asarray(coefficients, dtype=np.float64),
            lr=lr,
            max_iterations=max(1, epochs),
            device=device,
            seed=seed,
            dtype=dtype,
            frozen_modes=frozen_modes,
            far_field_size=far_field_size,
        )

    # All records are driven through consecutive ``update`` calls on ONE
    # optimizer, cycling. Fitting them one after another with ``run`` instead
    # degenerates into a chain of independent fits -- each ``run`` optimises its
    # record to that record's own optimum and restarts Adam -- so the last record
    # wins and the shared vector ends up fitting none of them. ``epochs`` keeps
    # its per-record meaning: the loop runs ``epochs * len(records)`` steps.
    total = max(1, int(epochs)) * len(frames)
    count = len(frames)
    for index in range(total):
        image, phase = frames[index % count]
        optimizer.update(image, phase)
        if optimizer.converged:
            optimizer = rearm(optimizer.coefficients)
        # ``last_loss`` is populated by every ``update``; the ``None`` branch only
        # exists on the type, since the attribute is Optional-typed.
        loss = optimizer.last_loss
        if loss is not None:
            history.append(float(loss))


    coefficients = optimizer.coefficients
    errors = [
        float(np.linalg.norm(coefficients - np.asarray(record.coefficients)))
        for record in records
        if record.coefficients is not None
    ]
    logger.info(
        "shared calibration: records={} epochs={} steps={} loss {} -> {}",
        len(records),
        epochs,
        len(history),
        history[0] if history else float("nan"),
        history[-1] if history else float("nan"),
    )
    return FitAberrationResult(
        coefficients=np.array(coefficients, dtype=np.float64),
        loss_history=history,
        labels=[record.label for record in records],
        source=source,
        n_records=len(records),
        epochs=epochs,
        iterations=len(history),
        record_errors=errors,
    )
