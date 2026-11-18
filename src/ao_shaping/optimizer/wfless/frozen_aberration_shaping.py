"""冻结像差下的逐像素可微整形（原 model-in-loop 的 Step B 内核）。

调用方把已标定的像差系数冻结，只用可微损失优化整幅逐像素 SLM 相位以逼近目标。
它不含 Step A 拟合、不确定性探针或任何会话循环，因此与 model-in-the-loop 的
闭环判决无关；保留它的理由是 ``scripts/slm/model_in_loop_hw_runbook.py`` 的
``--stage shape`` 与 ``report/sim_tue_methods`` 的复现脚本仍直接调用。

物理预测模型来自 ``algorithm/signal_processing/zernike_coefficient_optimizer.py``
与 ``algorithm/signal_processing/differentiable_shaping.py``。
"""

from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy.ndimage import zoom

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    total_loss,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)

# ``_linear_slope`` was a verbatim copy of the canonical leaf kernel; alias it so the
# fitting math keeps one home (utils/math/fitting.py, R-49/#65) and the two cannot drift.


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
            "frozen_aberration_shaping requires PyTorch. Install with: uv sync --group ml"
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
        noise_inject_every: Inject zero-mean Gaussian pupil noise into the
            evolving SLM phase every N iterations (``0`` disables injection --
            the default, which keeps the un-noised Step B bit-for-bit). This is
            the system-relaxation hook (issue R-83): a per-step random perturbation
            of the candidate phase that, in simulation, can make the solution
            robust to the mismatch between the frozen aberration and the true
            one, at the cost of a rougher phase.
        noise_sigma_rad: Standard deviation (radians) of the injected noise.
            Only applied inside the aperture (outside is masked by the pupil).
            ``0.0`` (the default) injects nothing even when
            ``noise_inject_every`` is set.
    """

    iterations: int = 150
    lr: float = 0.05
    target_shape: str = "square"
    target_side: int = 6
    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    w_zero_order: float = 0.0
    w_smoothness: float = 0.0
    noise_inject_every: int = 0
    noise_sigma_rad: float = 0.0


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


def _mis_build_phase_objective_tensors(background_intensity, coefficients, dev, model_gain, optimizer, psf_sigma, region, source_amplitude, target, tdtype, torch):
    """Build the torch-side objective tensors (aperture, amplitude, PSF, target)."""
    aperture = torch.from_numpy(_aperture_from_basis(optimizer.generate_basis())).to(
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
    amplitude = torch.from_numpy(np.ascontiguousarray(amplitude_grid)).to(
        device=dev, dtype=tdtype
    )
    background = None
    if background_intensity is not None:
        background_array = np.asarray(background_intensity, dtype=np.float64)
        expected = int(getattr(optimizer, "far_field_size", region))
        if background_array.shape != (expected, expected):
            raise ValueError("background_intensity must match the far-field grid")
        if not np.all(np.isfinite(background_array)) or np.any(background_array < 0):
            raise ValueError("background_intensity must be finite and nonnegative")
        background = torch.from_numpy(np.ascontiguousarray(background_array)).to(
            device=dev, dtype=tdtype
        )
    if not np.isfinite(model_gain) or model_gain <= 0:
        raise ValueError("model_gain must be finite and positive")
    if not np.isfinite(psf_sigma) or psf_sigma < 0:
        raise ValueError("psf_sigma must be finite and nonnegative")
    gaussian_kernel = None
    if psf_sigma > 0:
        radius = int(4 * psf_sigma + 0.5)
        coords = torch.arange(-radius, radius + 1, device=dev, dtype=tdtype)
        weights = torch.exp(-0.5 * (coords / psf_sigma) ** 2)
        weights /= weights.sum()
        gaussian_kernel = torch.outer(weights, weights)[None, None]
    coefficient_tensor = torch.from_numpy(
        np.ascontiguousarray(coefficients, dtype=np.float64)
    ).to(device=dev, dtype=tdtype)
    target_t = torch.from_numpy(np.ascontiguousarray(target)).to(
        device=dev, dtype=tdtype
    )
    return amplitude, aperture, background, coefficient_tensor, gaussian_kernel, target_t


def _mis_run_phase_refinement(aperture, closure, config, dev, noise_rng, on_iteration, opt, phase, region, tdtype, torch):
    """Drive the optimiser for ``config.iterations`` steps, recording each loss."""
    history: list[float] = []
    for index in range(config.iterations):
        loss_value = float(closure().detach().cpu().item())
        history.append(loss_value)
        opt.step()
        if (
            config.noise_inject_every > 0
            and config.noise_sigma_rad > 0
            and (index + 1) % config.noise_inject_every == 0
        ):
            with torch.no_grad():
                inject = torch.from_numpy(
                    np.ascontiguousarray(
                        noise_rng.normal(
                            0.0,
                            float(config.noise_sigma_rad),
                            (region, region),
                        ).astype(np.float64)
                    )
                ).to(device=dev, dtype=tdtype)
                phase.data += inject * aperture
        if on_iteration is not None:
            on_iteration(index, loss_value)
    logger.debug(
        "Step B done: iterations={} loss {} -> {}",
        len(history),
        history[0],
        history[-1],
    )
    return history


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
    model_gain: float = 1.0,
    background_intensity: np.ndarray | None = None,
    psf_sigma: float = 0.0,
    on_iteration: Callable[[int, float], None] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run Step B: optimise a full-pixel phase under a frozen aberration.

    The forward mirrors the twin -- and therefore
    :meth:`ZernikeCoefficientOptimizer.forward_intensity` -- exactly, with the
    coefficients frozen: the SLM phase and the aberration are summed first and
    the *sum* is masked, because the twin applies
    ``nan_to_num(phi_slm + aberration)`` and the aberration is NaN outside the
    aperture. Gradients therefore flow only into the SLM phase.

    The **error terms** (:attr:`ZernikeCoefficientOptimizer.error_terms`) are
    applied by the optimizer's own forward model, which this function calls
    directly: the LUT nonlinearity acts on the *command* phase (before the
    pupil phase is formed), the defocus and the tilt act on the *focal complex
    field* (before the intensity is taken). Step A's fit and Step B's
    synthesis therefore evaluate the *same* chain by construction, not by
    convention.

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
    # Function-local, like every other torch touch in this module: the package
    # must stay importable without PyTorch so the CPU-only paths still work.

    tdtype = getattr(torch, dtype)
    dev = torch.device(device) if device is not None else _auto_device(torch)

    region = initial_phase.shape[0]
    # The aperture mask is still needed here: the relaxation noise is injected
    # inside the pupil only. The *forward* itself (the aberration, the error
    # terms) belongs to the optimizer, so Step A and Step B cannot drift.
    amplitude, aperture, background, coefficient_tensor, gaussian_kernel, target_t = _mis_build_phase_objective_tensors(background_intensity, coefficients, dev, model_gain, optimizer, psf_sigma, region, source_amplitude, target, tdtype, torch)
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
                np.asarray(zoom(target, (scale, scale), order=1), dtype=np.float64)
            )
        ).to(device=dev, dtype=tdtype)

    torch.manual_seed(int(seed))
    # A small random start escapes the degenerate zero-gradient point where the
    # pupil field is purely real and the intensity is quadratic in the phase.
    start_phase = torch.from_numpy(
        np.ascontiguousarray(np.asarray(initial_phase, dtype=np.float64))
    ).to(device=dev, dtype=tdtype)
    start = start_phase + 0.1 * torch.randn((region, region), device=dev, dtype=tdtype)
    phase = torch.nn.Parameter(start.clone())
    opt = torch.optim.Adam([phase], lr=config.lr)
    # Dedicated RNG for the relaxation noise so it is independent of the torch
    # start-phase draw but still reproducible from ``seed``.
    noise_rng = np.random.default_rng(int(seed) + 0x9E37)

    def closure() -> Any:
        opt.zero_grad()
        # The SAME forward the Step A fit uses: the optimizer owns the chain
        # (LUT nonlinearity on the command, defocus and tilt on the focal
        # field), so the calibration and the synthesis cannot drift apart.
        intensity = optimizer.far_field_intensity_tensor(
            coefficient_tensor, phase, amplitude
        )
        if gaussian_kernel is not None:
            import torch.nn.functional as F

            radius = gaussian_kernel.shape[-1] // 2
            intensity = F.conv2d(
                F.pad(intensity[None, None], (radius,) * 4, mode="reflect"),
                gaussian_kernel,
            )[0, 0]
        if background is not None:
            intensity = float(model_gain) * intensity + background
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

    history = _mis_run_phase_refinement(aperture, closure, config, dev, noise_rng, on_iteration, opt, phase, region, tdtype, torch)
    return phase.detach().cpu().numpy().astype(np.float64), history
