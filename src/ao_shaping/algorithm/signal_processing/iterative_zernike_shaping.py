"""Iterative Zernike + phase-only SLM beam-shaping optimizer (torch-differentiable).

Implements a two-stage iterative scheme on the 2f-Fourier bench:

* **Stage A — Zernike calibration**: with an initial random SLM phase, learn a
  small set (≤ n_max) of Zernike coefficients (raw radians) so that the
  *simulated* far-field matches a *reference* far-field (the "actual" target).
* **Stage B — Phase-only shaping**: freeze the calibrated Zernike coefficients,
  optimize the free-form SLM phase so the far-field matches a square target.
* **Iteration A↔B**: alternate A and B (early-stop) until the square converges.

The forward model is a zero-padded Fraunhofer FFT (float64):

    field = gauss_pupil · exp(1j·(zernike_phase + slm_phase))
    far_field = fftshift(fft2(ifftshift(pad(field))))

Zernike basis maps come from the canonical ``ZernikeGenerator`` and are
converted to constant torch tensors; only their coefficients are trainable.

All phase generators return **raw unwrapped radians**; the only mod-2π wrap is
the SLM driver ``create_phase_from_array`` (not exercised in simulation).

Class-based optimizer convention (AGENTS.md): ``__init__`` validates + sets
state; ``update()`` performs one step and returns the next state; ``run()``
returns a result dataclass.

Requires torch; import is lazy (``_torch()``) so the package imports without
torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, zernike_modes

__all__ = [
    "IterativeZernikeShapingConfig",
    "IterativeZernikeShapingResult",
    "IterativeZernikeShapingOptimizer",
]


# ---------------------------------------------------------------------------
# Result / config containers
# ---------------------------------------------------------------------------
@dataclass
class IterativeZernikeShapingConfig:
    """Parameters for the iterative Zernike + phase-only shaping optimizer.

    Attributes:
        n_grid: SLM grid edge length (square) -- the pupil grid.
        n_zernike: Maximum Zernike radial order (n_max). 0 = skip calibration.
        target_side_px: Target square side length in far-field (camera) pixels,
            i.e. pixels of the zero-padded far-field grid (``far_field_size``).
        seed: RNG seed for reproducibility.
        zernike_lr: Learning rate for Zernike calibration (Adam). Keep this small
            (1e-3..1e-2); the far-field MSE landscape is rugged and a larger value
            diverges to non-finite coefficients.
        slm_lr: Learning rate for SLM phase shaping (cosine-decayed per pass).
        calib_iters: Number of Adam steps per Zernike calibration pass.
        shaping_iters: Number of Adam steps per SLM phase shaping pass.
        max_outer_iters: Maximum number of A↔B outer iterations.
        early_stop_patience: Stop outer loop after this many non-improving passes.
        early_stop_min_delta: Minimum score improvement to count as "improved".
        far_field_padding: Zero-padding factor for the Fraunhofer FFT. The
            far-field grid is ``n_grid * far_field_padding``; a same-size FFT
            undersamples the focal plane at ~1.1 px per waist radius (a model
            constant) and aliases into a lattice. Must match the bench config
            used to build the reference far-field.
    """

    n_grid: int = 64
    n_zernike: int = 4
    target_side_px: int = 16
    seed: int = 0
    zernike_lr: float = 0.005
    slm_lr: float = 0.02
    calib_iters: int = 50
    shaping_iters: int = 100
    max_outer_iters: int = 5
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.005
    far_field_padding: int = 8

    def __post_init__(self) -> None:
        """Validate fields early (mirrors the optimizer constructor checks)."""
        if self.n_grid < 4:
            raise ValueError(f"n_grid must be >= 4, got {self.n_grid}")
        if self.n_zernike < 0:
            raise ValueError(f"n_zernike must be >= 0, got {self.n_zernike}")
        if self.shaping_iters <= 0:
            raise ValueError(f"shaping_iters must be > 0, got {self.shaping_iters}")
        if self.max_outer_iters < 1:
            raise ValueError(
                f"max_outer_iters must be >= 1, got {self.max_outer_iters}"
            )
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )

    @property
    def far_field_size(self) -> int:
        """Edge length of the zero-padded far-field (camera) grid."""
        return self.n_grid * self.far_field_padding


@dataclass
class IterativeZernikeShapingResult:
    """Outcome of one full run of the iterative optimizer."""

    zernike_coeffs: dict[tuple[int, int], float]
    slm_phase: npt.NDArray[np.floating]
    far_field: npt.NDArray[np.floating]
    target: npt.NDArray[np.floating]
    score_history: list[dict[str, Any]]
    n_outer_iters: int
    converged: bool
    final_score: float


# ---------------------------------------------------------------------------
# Torch lazy import
# ---------------------------------------------------------------------------
_torch_module: Any = None


def _torch() -> Any:
    """Lazily import and cache torch.

    Returns:
        The ``torch`` module.

    Raises:
        RuntimeError: If torch is not installed.
    """
    global _torch_module
    if _torch_module is None:
        try:
            import torch as _t
        except ImportError as exc:
            raise RuntimeError(
                "IterativeZernikeShapingOptimizer requires torch. "
                "Install with: uv sync --group ml"
            ) from exc
        _torch_module = _t
    return _torch_module


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class IterativeZernikeShapingOptimizer:
    """Two-stage iterative Zernike + phase-only SLM beam-shaping optimizer.

    The forward model is:

        field = gauss_pupil · exp(1j·(zernike_phase + slm_phase))
        far_field = fftshift(fft2(ifftshift(pad(field))))   (float64)

    Zernike phase uses the canonical RZern grid convention:
    - Grid: (np.arange(n) - (n-1)/2) / radius, radius = n/2 (unit circle at grid edge)
    - R = sqrt(x² + y²), θ = atan2(y, x)
    - Z_n^m(r,θ) = R_n^|m|(r) · (cos(mθ) if m≥0 else sin(|m|θ))
    - Basis maps come from the canonical ``ZernikeGenerator``.

    The class follows the class-based optimizer convention:
    - ``__init__``: validate config, precompute static tensors (gauss pupil,
      target, Zernike basis modes).
    - ``calibrate_zernike``: one pass of Stage A (returns updated coeffs).
    - ``shape_phase``: one pass of Stage B (returns updated slm phase).
    - ``update``: one outer iteration (A then B, or B only if n_zernike=0).
    - ``run``: full loop with early stopping; returns result dataclass.
    """

    def __init__(self, config: IterativeZernikeShapingConfig) -> None:
        """Initialize the optimizer.

        Args:
            config: Optimizer parameters.

        Raises:
            ValueError: If n_grid < 4 or n_zernike < 0.
        """
        if config.n_grid < 4:
            raise ValueError(f"n_grid must be >= 4, got {config.n_grid}")
        if config.n_zernike < 0:
            raise ValueError(f"n_zernike must be >= 0, got {config.n_zernike}")
        if config.n_zernike > 0 and config.calib_iters <= 0:
            raise ValueError("calib_iters must be > 0 when n_zernike > 0")
        if config.shaping_iters <= 0:
            raise ValueError(f"shaping_iters must be > 0, got {config.shaping_iters}")
        if config.max_outer_iters < 1:
            raise ValueError(
                f"max_outer_iters must be >= 1, got {config.max_outer_iters}"
            )

        self._config = config
        t = _torch()

        # --- Static tensors (precomputed once) ---
        n = config.n_grid
        m = config.far_field_size
        radius = n / 2.0

        # Normalized coordinate grid (canonical RZern convention)
        coords = (t.arange(n, dtype=t.float64) - (n - 1) / 2.0) / radius
        y, x = t.meshgrid(coords, coords, indexing="ij")  # shape (n, n)
        self._x = x
        self._y = y
        self._r = t.sqrt(x**2 + y**2)
        self._theta = t.atan2(y, x)
        self._far_field_size = m
        self._pad = (m - n) // 2

        # Gaussian pupil on the aperture, same convention as the bench
        # (exp(-r^2/w0^2) with w0 = aperture/3.5, truncated at the edge r = 1).
        w0_norm = 2.0 / 3.5
        aperture_mask = (self._r <= 1.0).to(t.float64)
        self._gauss = t.exp(-(self._r**2) / (w0_norm**2)) * aperture_mask

        # Zernike basis modes (n=1..n_zernike, piston excluded)
        self._zernike_modes: list[tuple[int, int]] = []
        if config.n_zernike > 0:
            self._zernike_modes = [
                mode for mode in zernike_modes(config.n_zernike) if mode != (0, 0)
            ]
            generator = ZernikeGenerator(
                (n, n), radius=radius, n_orders=config.n_zernike
            )
            self._basis = [
                (
                    nn,
                    mm,
                    t.as_tensor(
                        np.nan_to_num(
                            generator.generate_polynomial({(nn, mm): 1.0}), nan=0.0
                        ),
                        dtype=t.float64,
                    ),
                )
                for nn, mm in self._zernike_modes
            ]
        else:
            self._basis = []
        self._n_zernike_params = len(self._zernike_modes)

        # Target (square, normalized to sum=1)
        target = t.zeros((m, m), dtype=t.float64)
        half = m // 2
        s = config.target_side_px
        target[half - s // 2 : half + s // 2, half - s // 2 : half + s // 2] = 1.0
        target = target / target.sum()
        self._target = target
        self._target_support = (target > 0).to(t.float64)

        # --- Trainable parameters (initialized to zero; reset in run()) ---
        self._zernike_coeffs: t.nn.Parameter | None = None
        self._slm_phase: t.nn.Parameter | None = None

        logger.debug(
            "IterativeZernikeShapingOptimizer initialized: n_grid={}, n_zernike={}, "
            "zernike_params={}, target_side={}",
            n,
            config.n_zernike,
            self._n_zernike_params,
            config.target_side_px,
        )

    def _zernike_basis(self) -> list[tuple[int, int, Any]]:
        """Return the cached basis tensors built by the canonical generator.

        The maps come from :class:`ZernikeGenerator` (``generate_polynomial``,
        NaN outside the aperture replaced by 0) and are converted to constant
        torch tensors once in ``__init__``; only the coefficients are trainable.
        Re-deriving the radial polynomial here instead would be a second,
        drifting copy of the canonical Zernike math.

        Returns:
            List of (n, m, basis_tensor) tuples.
        """
        return self._basis

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def _far_field(self, zernike_coeffs: Any, slm_phase: Any) -> Any:
        """Compute far-field intensity from Zernike + SLM phase.

        The pupil field is zero-padded to the far-field grid and transformed by
        a centred Fraunhofer FFT (``fftshift(fft2(ifftshift(...)))``), matching
        ``slm_shaping_bench.forward_intensity``.

        Args:
            zernike_coeffs: 1-D tensor of Zernike coefficients (raw radians),
                length = n_zernike_params.
            slm_phase: 2-D tensor of SLM phase (raw radians), shape (n, n).

        Returns:
            Normalized far-field intensity tensor (sum=1), shape (M, M).
        """
        t = _torch()
        if zernike_coeffs is not None and self._n_zernike_params > 0:
            zernike_phase = t.zeros_like(slm_phase)
            for i, (_, _, basis) in enumerate(self._zernike_basis()):
                zernike_phase = zernike_phase + zernike_coeffs[i] * basis
        else:
            zernike_phase = t.zeros_like(slm_phase)

        total_phase = zernike_phase + slm_phase
        field = self._gauss * t.exp(1j * total_phase)
        n = field.shape[0]
        m = self._far_field_size
        if m > n:
            pad = self._pad
            padded = t.zeros((m, m), dtype=t.complex128, device=field.device)
            padded[pad : pad + n, pad : pad + n] = field
            field = padded
        focal = t.fft.fftshift(t.fft.fft2(t.fft.ifftshift(field)))
        intensity = focal.real**2 + focal.imag**2
        intensity = intensity / (intensity.sum() + 1e-12)
        return intensity

    def _zernike_phase(self, coeffs: dict[tuple[int, int], float]) -> npt.NDArray[np.floating]:
        """Zernike phase (raw radians) for a coefficient dict, shape (n, n)."""
        t = _torch()
        n = self._config.n_grid
        phase = t.zeros((n, n), dtype=t.float64)
        if coeffs and self._n_zernike_params > 0:
            for nm, _, basis in self._zernike_basis():
                if nm in coeffs:
                    phase = phase + float(coeffs[nm]) * basis
        return phase.numpy()

    def _score(self, intensity: Any) -> Any:
        """Compute composite quality score (PIB + uniformity).

        Score = 0.5 * PIB + 0.5 * (1 - min(CV/0.3, 1))

        Args:
            intensity: Normalized far-field intensity tensor.

        Returns:
            Scalar tensor (composite score, higher is better).
        """
        t = _torch()
        # Fixed centred support (matches compute_metrics with center=None). An
        # argmax-rolled box is discontinuous: for speckle-like fields the argmax
        # hops between near-equal grains under a ~1e-3 model change, so the
        # optimizer chases a box that no longer covers the beam.
        support = self._target_support
        pib = (intensity * support).sum()
        vals = intensity[support > 0]
        if vals.numel() == 0 or vals.mean() <= 0:
            cv = t.tensor(float("inf"), dtype=t.float64, device=intensity.device)
        else:
            # Population std (unbiased=False) to match numpy's np.std, which the
            # bench composite_score uses -- otherwise the two objectives differ.
            cv = vals.std(unbiased=False) / (vals.mean() + 1e-12)
        # Keep in sync with slm_shaping_bench.composite_from_pib_cv.
        return 0.5 * pib + 0.5 * (1.0 / (1.0 + cv))

    # ------------------------------------------------------------------
    # Stage A: Zernike calibration
    # ------------------------------------------------------------------
    def calibrate_zernike(
        self,
        actual_far_field: Any,
        initial_slm_phase: Any,
    ) -> dict[tuple[int, int], float]:
        """Run one Zernike calibration pass (Stage A).

        Learns Zernike coefficients (raw radians) to minimize the difference
        between the simulated far-field and the reference "actual" far-field,
        given a fixed initial SLM phase.

        Args:
            actual_far_field: Reference far-field (numpy or torch), shape (n, n),
                normalized to sum=1.
            initial_slm_phase: Initial SLM phase (numpy or torch), shape (n, n),
                raw radians (not wrapped).

        Returns:
            Dictionary mapping (n, m) to calibrated coefficient (raw radians).
        """
        t = _torch()
        if self._n_zernike_params == 0:
            return {}

        if isinstance(actual_far_field, np.ndarray):
            target_ff = t.as_tensor(actual_far_field, dtype=t.float64)
        else:
            target_ff = actual_far_field.to(t.float64)

        if isinstance(initial_slm_phase, np.ndarray):
            slm_init = t.as_tensor(initial_slm_phase, dtype=t.float64)
        else:
            slm_init = initial_slm_phase.to(t.float64)

        # Trainable Zernike coefficients
        zernike_coeffs = t.nn.Parameter(
            t.zeros(self._n_zernike_params, dtype=t.float64)
        )
        opt = t.optim.Adam([zernike_coeffs], lr=self._config.zernike_lr)

        # Sum-normalised far field => per-pixel values ~1/m^2 and raw MSE ~1e-10,
        # where Adam's default eps=1e-8 dominates the gradient. Scaling the loss
        # by the target keeps it O(1) so the learning rate actually applies.
        scale = t.mean(target_ff**2) + 1e-12
        for _ in range(self._config.calib_iters):
            opt.zero_grad()
            ff = self._far_field(zernike_coeffs, slm_init)
            loss = t.mean((ff - target_ff) ** 2) / scale
            loss.backward()
            opt.step()
            if not bool(t.isfinite(zernike_coeffs).all()):
                logger.warning(
                    "Zernike calibration diverged to non-finite coefficients at "
                    "iter {} (zernike_lr={}); returning the last finite estimate.",
                    _,
                    self._config.zernike_lr,
                )
                break

        # Extract calibrated coefficients
        with t.no_grad():
            coeffs_dict: dict[tuple[int, int], float] = {}
            for i, (n, m, _) in enumerate(self._zernike_basis()):
                coeffs_dict[(n, m)] = float(zernike_coeffs[i].item())
        return coeffs_dict

    # ------------------------------------------------------------------
    # Stage B: SLM phase shaping
    # ------------------------------------------------------------------
    def shape_phase(
        self,
        zernike_coeffs: dict[tuple[int, int], float] | None,
        initial_slm_phase: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Run one SLM phase shaping pass (Stage B).

        Freezes the Zernike coefficients and optimizes the free-form SLM phase
        to maximize the composite score (PIB + uniformity) against the square
        target.

        Args:
            zernike_coeffs: Frozen Zernike coefficients (raw radians), or None.
            initial_slm_phase: Initial SLM phase (raw radians), shape (n, n).
                If None, initialized to zeros.

        Returns:
            Optimized SLM phase (numpy, raw radians), shape (n, n).
        """
        t = _torch()
        n = self._config.n_grid

        if zernike_coeffs is None or self._n_zernike_params == 0:
            zernike_vec = t.zeros(self._n_zernike_params, dtype=t.float64)
        else:
            zernike_vec = t.zeros(self._n_zernike_params, dtype=t.float64)
            for i, (nm, _) in enumerate(self._zernike_modes):
                if nm in zernike_coeffs:
                    zernike_vec[i] = zernike_coeffs[nm]

        if initial_slm_phase is None:
            slm_phase = t.nn.Parameter(t.zeros((n, n), dtype=t.float64))
        elif isinstance(initial_slm_phase, np.ndarray):
            slm_phase = t.nn.Parameter(
                t.as_tensor(initial_slm_phase, dtype=t.float64).clone()
            )
        else:
            slm_phase = t.nn.Parameter(initial_slm_phase.to(t.float64).clone())

        base_lr = self._config.slm_lr
        opt = t.optim.Adam([slm_phase], lr=base_lr)
        iters = self._config.shaping_iters
        best_score = float("-inf")
        best_phase = slm_phase.detach().clone()

        for i in range(iters):
            # Cosine decay: a flat slm_lr makes Adam overshoot and the score
            # oscillates, so the pass would end on an arbitrary (poor) iterate.
            for group in opt.param_groups:
                group["lr"] = base_lr * 0.5 * (1.0 + np.cos(np.pi * i / max(iters, 1)))
            opt.zero_grad()
            ff = self._far_field(zernike_vec, slm_phase)
            score = self._score(ff)
            score_value = float(score.item())
            if score_value > best_score:
                best_score = score_value
                best_phase = slm_phase.detach().clone()
            (-score).backward()
            opt.step()

        return best_phase.numpy()

    # ------------------------------------------------------------------
    # One outer iteration
    # ------------------------------------------------------------------
    def update(
        self,
        actual_far_field: Any,
        zernike_coeffs: dict[tuple[int, int], float],
        slm_phase: npt.NDArray[np.floating],
    ) -> tuple[dict[tuple[int, int], float], npt.NDArray[np.floating]]:
        """Perform one outer iteration: Stage A (if n_zernike > 0) then Stage B.

        Args:
            actual_far_field: Reference far-field (for Stage A).
            zernike_coeffs: Current Zernike coefficients.
            slm_phase: Current SLM phase (raw radians).

        Returns:
            Tuple of (updated zernike_coeffs, updated slm_phase).
        """
        if self._n_zernike_params > 0:
            zernike_coeffs = self.calibrate_zernike(actual_far_field, slm_phase)
        slm_phase = self.shape_phase(zernike_coeffs, slm_phase)
        return zernike_coeffs, slm_phase

    # ------------------------------------------------------------------
    # Full run with early stopping
    # ------------------------------------------------------------------
    def run(
        self,
        actual_far_field: npt.NDArray[np.floating],
        initial_slm_phase: npt.NDArray[np.floating] | None = None,
    ) -> IterativeZernikeShapingResult:
        """Run the full iterative optimization loop.

        Args:
            actual_far_field: Reference far-field (the "actual" target), shape
                (n, n), normalized to sum=1.
            initial_slm_phase: Initial SLM phase (raw radians), shape (n, n).
                If None, initialized to zeros (flat).

        Returns:
            IterativeZernikeShapingResult with all intermediates.
        """
        t = _torch()
        cfg = self._config
        n = cfg.n_grid

        # Shaping start.
        if initial_slm_phase is None:
            slm_phase = np.zeros((n, n), dtype=np.float64)
        else:
            slm_phase = np.asarray(initial_slm_phase, dtype=np.float64)

        # Stage A must be evaluated with the same SLM phase the reference was
        # built with (flat); using the shaping warm start would make the fit
        # inconsistent with the reference.
        calibration_phase = np.zeros((n, n), dtype=np.float64)

        zernike_coeffs: dict[tuple[int, int], float] = {}
        score_history: list[dict[str, Any]] = []
        converged = False

        # Compute initial score
        t_arr = _torch()
        actual_t = t_arr.as_tensor(actual_far_field, dtype=t_arr.float64)
        slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
        ff = self._far_field(
            t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64)
            if self._n_zernike_params > 0
            else None,
            slm_t,
        )
        init_score = float(self._score(ff).item())
        score_history.append({"outer_iter": 0, "score": init_score, "stage": "init"})

        best_score = init_score
        best_slm_phase = slm_phase
        best_ff = ff.detach().numpy()
        best_zernike: dict[tuple[int, int], float] = dict(zernike_coeffs)
        patience_counter = 0
        first_shape = True

        for outer in range(1, cfg.max_outer_iters + 1):
            # Stage A (if applicable)
            if self._n_zernike_params > 0:
                zernike_coeffs = self.calibrate_zernike(
                    actual_far_field, calibration_phase
                )

            # The model adds the frozen Zernike to the SLM phase, so on the first
            # shaping pass subtract it from the initial phase: otherwise the
            # warm start is corrupted by a Zernike the model then re-adds.
            if first_shape and self._n_zernike_params > 0:
                slm_phase = slm_phase - self._zernike_phase(zernike_coeffs)
                first_shape = False

            # Stage B
            slm_phase = self.shape_phase(zernike_coeffs, slm_phase)

            # Compute score
            zernike_vec = t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64)
            for i, nm in enumerate(self._zernike_modes):
                if nm in zernike_coeffs:
                    zernike_vec[i] = zernike_coeffs[nm]
            slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
            ff = self._far_field(zernike_vec, slm_t)
            score = float(self._score(ff).item())
            score_history.append({"outer_iter": outer, "score": score, "stage": "A+B"})

            # Early stopping
            if score > best_score + cfg.early_stop_min_delta:
                best_score = score
                best_slm_phase = slm_phase.copy()
                best_ff = ff.detach().numpy()
                best_zernike = dict(zernike_coeffs)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    converged = True
                    logger.info(
                        "Early stop at outer iter {} (patience={}, score={:.4f})",
                        outer,
                        cfg.early_stop_patience,
                        score,
                    )
                    break

        # Return the best-scoring state (not the last), so ``final_score``,
        # ``slm_phase`` and ``far_field`` all describe the same artifact.
        return IterativeZernikeShapingResult(
            zernike_coeffs=best_zernike,
            slm_phase=best_slm_phase,
            far_field=best_ff,
            target=self._target.detach().numpy(),
            score_history=score_history,
            n_outer_iters=len(score_history) - 1,
            converged=converged,
            final_score=best_score,
        )
