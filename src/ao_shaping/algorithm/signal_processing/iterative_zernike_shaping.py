"""Iterative Zernike + phase-only SLM beam-shaping optimizer (torch-differentiable).

Implements a two-stage iterative scheme on the 2f-Fourier bench:

* **Stage A — Zernike calibration**: with an initial random SLM phase, learn a
  small set (≤ n_max) of Zernike coefficients (raw radians) so that the
  *simulated* far-field matches a *reference* far-field (the "actual" target).
* **Stage B — Phase-only shaping**: freeze the calibrated Zernike coefficients,
  optimize the free-form SLM phase so the far-field matches a square target.
* **Iteration A↔B**: alternate A and B (early-stop) until the square converges.

The forward model is a pure FFT (Fraunhofer) propagation:

    field = gauss_pupil · exp(1j·(zernike_phase + slm_phase))
    far_field = fft2(field)          # no fftshift; float64

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
        n_grid: SLM grid edge length (square).
        n_zernike: Maximum Zernike radial order (n_max). 0 = skip calibration.
        target_side_px: Target square side length in camera pixels.
        seed: RNG seed for reproducibility.
        zernike_lr: Learning rate for Zernike calibration (Adam).
        slm_lr: Learning rate for SLM phase shaping (Adam).
        calib_iters: Number of Adam steps per Zernike calibration pass.
        shaping_iters: Number of Adam steps per SLM phase shaping pass.
        max_outer_iters: Maximum number of A↔B outer iterations.
        early_stop_patience: Stop outer loop after this many non-improving passes.
        early_stop_min_delta: Minimum score improvement to count as "improved".
    """

    n_grid: int = 64
    n_zernike: int = 4
    target_side_px: int = 16
    seed: int = 0
    zernike_lr: float = 0.05
    slm_lr: float = 0.02
    calib_iters: int = 50
    shaping_iters: int = 100
    max_outer_iters: int = 5
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.005

    def __post_init__(self) -> None:
        """Validate fields early (mirrors the optimizer constructor checks)."""
        if self.n_grid < 4:
            raise ValueError(f"n_grid must be >= 4, got {self.n_grid}")
        if self.n_zernike < 0:
            raise ValueError(f"n_zernike must be >= 0, got {self.n_zernike}")
        if self.shaping_iters <= 0:
            raise ValueError(f"shaping_iters must be > 0, got {self.shaping_iters}")
        if self.max_outer_iters < 1:
            raise ValueError(f"max_outer_iters must be >= 1, got {self.max_outer_iters}")


@dataclass
class IterativeZernikeShapingResult:
    """Outcome of one full run of the iterative optimizer."""

    zernike_coeffs: dict[tuple[int, int], float]
    slm_phase: np.ndarray
    far_field: np.ndarray
    target: np.ndarray
    score_history: list[dict[str, float]]
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
        far_field = fft2(field)   (pure FFT, no fftshift, float64)

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
            raise ValueError(f"max_outer_iters must be >= 1, got {config.max_outer_iters}")

        self._config = config
        t = _torch()

        # --- Static tensors (precomputed once) ---
        n = config.n_grid
        radius = n / 2.0

        # Normalized coordinate grid (canonical RZern convention)
        coords = (t.arange(n, dtype=t.float64) - (n - 1) / 2.0) / radius
        y, x = t.meshgrid(coords, coords, indexing="ij")  # shape (n, n)
        # Gaussian pupil (matched to aperture)
        self._gauss = t.exp(-(x**2 + y**2) / (2 * (n / 4.0) ** 2))

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
                (nn, mm, t.as_tensor(
                    np.nan_to_num(generator.generate_polynomial({(nn, mm): 1.0}), nan=0.0),
                    dtype=t.float64,
                ))
                for nn, mm in self._zernike_modes
            ]
        else:
            self._basis = []
        self._n_zernike_params = len(self._zernike_modes)

        # Target (square, normalized to sum=1)
        target = t.zeros((n, n), dtype=t.float64)
        half = n // 2
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
            n, config.n_zernike, self._n_zernike_params, config.target_side_px,
        )

    def _zernike_basis(self) -> list[tuple[int, int, Any]]:
        """Return cached basis tensors from the canonical Zernike generator.

        Returns:
            List of (n, m, basis_tensor) tuples.
        """
        return self._basis

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def _far_field(self, zernike_coeffs: Any, slm_phase: Any) -> Any:
        """Compute far-field intensity from Zernike + SLM phase.

        Args:
            zernike_coeffs: 1-D tensor of Zernike coefficients (raw radians),
                length = n_zernike_params.
            slm_phase: 2-D tensor of SLM phase (raw radians), shape (n, n).

        Returns:
            Normalized far-field intensity tensor (sum=1), shape (n, n).
        """
        t = _torch()
        # Build Zernike phase
        if zernike_coeffs is not None and self._n_zernike_params > 0:
            zernike_phase = t.zeros_like(slm_phase)
            for i, (_, _, basis) in enumerate(self._zernike_basis()):
                zernike_phase = zernike_phase + zernike_coeffs[i] * basis
        else:
            zernike_phase = t.zeros_like(slm_phase)

        total_phase = zernike_phase + slm_phase
        field = self._gauss * t.exp(1j * total_phase)
        ff = t.fft.fft2(field)  # pure FFT, no fftshift
        intensity = ff.real**2 + ff.imag**2
        intensity = intensity / (intensity.sum() + 1e-12)
        return intensity

    def _score(self, intensity: Any) -> Any:
        """Compute composite quality score (PIB + uniformity).

        Score = 0.5 * PIB + 0.5 * (1 - min(CV/0.3, 1))

        Args:
            intensity: Normalized far-field intensity tensor.

        Returns:
            Scalar tensor (composite score, higher is better).
        """
        t = _torch()
        # PIB: power in target support
        pib = (intensity * self._target_support).sum()
        # CV: coefficient of variation within target support
        vals = intensity[self._target_support > 0]
        if vals.numel() == 0 or vals.mean() <= 0:
            cv = t.tensor(1.0, dtype=t.float64, device=intensity.device)
        else:
            cv = vals.std() / (vals.mean() + 1e-12)
        cv_clamped = t.min(cv / 0.3, t.tensor(1.0, dtype=t.float64, device=intensity.device))
        score = 0.5 * pib + 0.5 * (1.0 - cv_clamped)
        return score

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
        zernike_coeffs = t.nn.Parameter(t.zeros(self._n_zernike_params, dtype=t.float64))
        opt = t.optim.Adam([zernike_coeffs], lr=self._config.zernike_lr)

        for _ in range(self._config.calib_iters):
            opt.zero_grad()
            ff = self._far_field(zernike_coeffs, slm_init)
            # MSE loss between simulated and actual far-field
            loss = t.mean((ff - target_ff) ** 2)
            loss.backward()
            opt.step()

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
        initial_slm_phase: np.ndarray | None = None,
    ) -> np.ndarray:
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
            slm_phase = t.nn.Parameter(t.as_tensor(initial_slm_phase, dtype=t.float64).clone())
        else:
            slm_phase = t.nn.Parameter(initial_slm_phase.to(t.float64).clone())

        opt = t.optim.Adam([slm_phase], lr=self._config.slm_lr)

        for _ in range(self._config.shaping_iters):
            opt.zero_grad()
            ff = self._far_field(zernike_vec, slm_phase)
            score = self._score(ff)
            loss = -score  # maximize score
            loss.backward()
            opt.step()

        with t.no_grad():
            return slm_phase.detach().numpy()

    # ------------------------------------------------------------------
    # One outer iteration
    # ------------------------------------------------------------------
    def update(
        self,
        actual_far_field: Any,
        zernike_coeffs: dict[tuple[int, int], float],
        slm_phase: np.ndarray,
    ) -> tuple[dict[tuple[int, int], float], np.ndarray]:
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
        actual_far_field: np.ndarray,
        initial_slm_phase: np.ndarray | None = None,
    ) -> IterativeZernikeShapingResult:
        """Run the full iterative optimization loop.

        Args:
            actual_far_field: Reference far-field (the "actual" target), shape
                (n, n), normalized to sum=1.
            initial_slm_phase: Initial SLM phase (raw radians), shape (n, n).
                If None, initialized to small random noise (seeded).

        Returns:
            IterativeZernikeShapingResult with all intermediates.
        """
        t = _torch()
        cfg = self._config
        n = cfg.n_grid

        # Initial SLM phase
        if initial_slm_phase is None:
            rng = np.random.default_rng(cfg.seed)
            slm_phase = rng.normal(0, 0.1, size=(n, n)).astype(np.float64)
        else:
            slm_phase = np.asarray(initial_slm_phase, dtype=np.float64)

        zernike_coeffs: dict[tuple[int, int], float] = {}
        score_history: list[dict[str, float]] = []
        converged = False

        # Compute initial score
        t_arr = _torch()
        actual_t = t_arr.as_tensor(actual_far_field, dtype=t_arr.float64)
        slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
        ff = self._far_field(
            t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64) if self._n_zernike_params > 0 else None,
            slm_t,
        )
        init_score = float(self._score(ff).item())
        score_history.append({"outer_iter": 0, "score": init_score, "stage": "init"})

        best_score = init_score
        patience_counter = 0

        for outer in range(1, cfg.max_outer_iters + 1):
            # Stage A (if applicable)
            if self._n_zernike_params > 0:
                zernike_coeffs = self.calibrate_zernike(actual_far_field, slm_phase)

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
            score_history.append({"outer_iter": outer, "score": score, "stage": f"A+B"})

            # Early stopping
            if score > best_score + cfg.early_stop_min_delta:
                best_score = score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    converged = True
                    logger.info(
                        "Early stop at outer iter {} (patience={}, score={:.4f})",
                        outer, cfg.early_stop_patience, score,
                    )
                    break

        # Final far-field
        zernike_vec = t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64)
        for i, nm in enumerate(self._zernike_modes):
            if nm in zernike_coeffs:
                zernike_vec[i] = zernike_coeffs[nm]
        slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
        ff = self._far_field(zernike_vec, slm_t)

        return IterativeZernikeShapingResult(
            zernike_coeffs=zernike_coeffs,
            slm_phase=slm_phase,
            far_field=ff.detach().numpy(),
            target=self._target.detach().numpy(),
            score_history=score_history,
            n_outer_iters=len(score_history) - 1,
            converged=converged,
            final_score=best_score,
        )
