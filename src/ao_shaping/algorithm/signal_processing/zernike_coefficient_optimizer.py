"""Fit Zernike aberration coefficients to a measured far-field intensity.

"Step A" of the iterative model-in-the-loop SLM beam-shaping pipeline: with the
SLM phase map held **fixed**, fit the Zernike aberration coefficients ``c`` so
that a differentiable FFT forward model reproduces the far-field intensity that
was actually measured.  The digital twin then re-measures with the updated
phase and the loop re-fits, one round at a time.

Model
-----
On the ``region x region`` pupil grid::

    U        = A * exp(i * (phi_slm + sum_j c_j * Z_j))
    I_model  = |fftshift(fft2(ifftshift(U), norm="ortho"))|**2

The FFT convention is the one used by the digital twin
(:meth:`SimFourierGSNetEnv.render_intensity`), so with the native Gaussian beam
(``w0 = 250 * region / 512``) this model's far field is bit-identical to the
twin's on the same grid.

``A`` defaults to that native Gaussian and may be overridden per run.
``phi_slm`` is consumed as **raw unwrapped radians**: phase generators in this
repo return raw radians and the only ``mod 2*pi`` wrap lives in the SLM driver,
so nothing is wrapped here.

Loss and gradient
-----------------
The objective mirrors :meth:`DifferentiableBeamOptimizer._intensity_loss`: a
*peak-normalized* MSE against the measured far field, ``i / (i.max() + 1e-8)``.
Peak rather than energy normalization is required because the FFT concentrates
the energy in a few pixels, which underflows in float32.

The step is **measurement-anchored** in the sense of
``differentiable_beam.py``: the target *and* the normalization scale of the
loss are taken from the MEASURED image, so the upstream intensity gradient
handed to the model Jacobian ``dI_model/dc`` is anchored on real data rather
than on the model's own image.

.. note::
   In :class:`DifferentiableBeamOptimizer` the anchor point and the target are
   *different* images (hardware frame vs. desired pattern), so the loss can be
   evaluated at the measurement to obtain ``dL/dI`` and pushed through the
   model Jacobian with ``i_sim.backward(gradient=i_meas.grad)``.  Here the
   measurement *is* the fit target, so that literal form evaluates the loss at
   the target itself and yields an identically zero upstream gradient.  The
   anchor is therefore carried by the normalization scale (``peak_anchor``,
   taken from the measurement) and the residual is taken on the model image,
   which is algebraically the same anchored gradient and is non-degenerate.

Attributes:
    n_coefficients: Number of fitted Zernike coefficients.
    n_orders: Maximum radial order of the Zernike basis.
    region: Side length of the square pupil/far-field grid.
    radius: Zernike aperture radius in pixels.
    device: The torch device the optimization runs on.
    coefficient_tensor: The live coefficient parameter tensor (requires_grad).
    coefficients: Detached numpy copy of the current coefficients (radians).
    loss_history: Loss value per optimization step.
    last_loss: The loss at the last step, or ``None`` before the first update.
    converged: Whether the early-stopping criterion was met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import torch

#: Reference region of the digital twin (``SimFourierGSNetEnv.BeamParams``).
TWIN_REGION = 512
#: Native Gaussian waist of the digital twin, in twin-region pixels.
TWIN_W0 = 250.0
#: Guard added to every peak-normalization divisor, mirroring
#: :meth:`DifferentiableBeamOptimizer._intensity_loss`.
PEAK_EPS = 1e-8
#: Plateau detection: stop once the best loss has not improved by more than
#: ``RELATIVE_TOL`` (or ``ABSOLUTE_FLOOR``) for ``PLATEAU_PATIENCE`` steps.
RELATIVE_TOL = 1e-5
ABSOLUTE_FLOOR = 1e-12
PLATEAU_PATIENCE = 30
#: Never stop before this many steps, whatever the plateau heuristic says.
MIN_ITERATIONS = 5


def _torch():
    """Return the ``torch`` module, raising :class:`ImportError` when absent.

    Returns:
        The ``torch`` top-level module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "zernike_coefficient_optimizer requires PyTorch. "
            "Install with: uv sync --group ml"
        ) from None
    return _t


def _numpy_dtype(dtype_name: str) -> Any:
    """Map a working-precision name to the matching numpy dtype."""
    return np.float64 if dtype_name == "float64" else np.float32


def _to_tensor(
    x: np.ndarray,
    device: torch.device,
    dtype: str = "float32",
) -> torch.Tensor:
    """Convert a real 2D numpy array to a torch tensor on ``device``.

    Args:
        x: Source array.
        device: Target torch device.
        dtype: Working precision, ``"float32"`` or ``"float64"``.
    """
    torch = _torch()
    arr = np.ascontiguousarray(x, dtype=_numpy_dtype(dtype))
    return torch.from_numpy(arr).to(device)


@dataclass
class ZernikeCoefficientResult:
    """Outcome of a :meth:`ZernikeCoefficientOptimizer.run` call.

    Attributes:
        coefficients: Final fitted coefficients in radians, shape
            ``(calc_n_zernike_terms(n_orders),)`` in Noll order.
        history: Per-iteration records; always contains ``"loss"``.
        iterations: Number of optimization steps actually performed.
        converged: Whether the stopping criterion was met before the budget
            ran out.
    """

    coefficients: np.ndarray
    history: dict[str, list] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False


class ZernikeCoefficientOptimizer(IterativeOptimizer):
    """Fit Zernike aberration coefficients to a measured far field.

    The optimizer is stateful: construct it once, then call :meth:`update`
    repeatedly for manual control, or :meth:`run` for the full loop.

    Digital-twin contract
    ---------------------
    The far field reproduces ``SimFourierGSNetEnv.render_intensity``'s core
    convention, ``|fftshift(fft2(ifftshift(U), norm="ortho"))|**2`` with
    ``U = A * exp(1j * patch)``, including two details that are easy to miss and
    that this class matches deliberately:

    * **The patch is masked, not the aberration.** The twin computes
      ``nan_to_num(phi_slm + aberration, nan=0.0)``, so the *sum* is flat
      outside the circular Zernike aperture. Applying the SLM phase everywhere
      instead makes the model disagree with the twin by O(100%).
    * **The amplitude still applies outside the aperture**, because the twin
      masks phase only, never amplitude.

    With ``dtype="float64"`` the far field matches the float64 twin to ~1 ulp
    (2.7e-16 relative); the float32 default agrees to ~7e-8 relative, which is
    float32 round-off rather than a modelling difference.

    Loss
    ----
    The recorded loss is the peak-normalized MSE between the model and measured
    far fields. The literal two-stage "measurement-anchored" trick of
    ``differentiable_beam.py`` (backpropagating the loss into the measured
    intensity) is degenerate here, because the measured image is a constant with
    respect to the coefficients -- ``dL/dI_meas == 0`` at the target, so it
    contributes no gradient. This class therefore uses the measured *peak* as a
    fixed normalization anchor, which is equivalent to standard peak-normalized
    MSE while keeping the target and the normalizer constant in ``c``, so the
    gradient flows through the model intensity alone.

    Learning rate
    -------------
    With a high-order basis (``n_orders=10``, 66 coefficients) the objective has
    a shallow local basin that can absorb the fit while leaving the coefficients
    wrong. Measured on a ``region=64`` grid with a three-mode truth, the same
    problem converges to loss ~9e-13 and ``|c - c_true| < 1e-4`` at
    ``lr in {0.02, 0.03, 0.1}`` but stalls at loss ~6.9e-6 and ``|c - c_true|``
    ~0.49 at ``lr in {0.05, 0.08}``. The Jacobian of the normalized intensity is
    well conditioned at that operating point (condition number ~5.6,
    ``sigma_min`` ~12.5), so the basin is an optimization artifact rather than an
    identifiability limit -- a direct least-squares solve from the same start
    recovers ``c_true``. Callers using a 10-order basis should therefore validate
    the *coefficient* error, not just the loss: a low loss alone does not prove
    the coefficients were found.
    """

    def __init__(
        self,
        n_orders: int = 10,
        region: int = 64,
        radius: int | None = None,
        initial_coefficients: np.ndarray | None = None,
        lr: float = 0.1,
        max_iterations: int = 500,
        device: str | None = None,
        seed: int | None = None,
        dtype: str = "float32",
        far_field_size: int | None = None,
        frozen_modes: tuple[int, ...] = (),
    ) -> None:
        """Initialize the optimizer and validate all inputs.

        Args:
            n_orders: Maximum radial order of the Zernike basis. Must be >= 1.
                The basis holds ``calc_n_zernike_terms(n_orders)`` modes in
                Noll 1976 order (Noll 4 = defocus, Noll 5 = astigmatism).
            region: Side length of the square grid. Must be >= 16 and even.
            radius: Zernike aperture radius in pixels. Defaults to
                ``region // 2``. Modes outside the aperture evaluate to zero.
            initial_coefficients: Optional starting coefficients in radians,
                shape ``(n_coefficients,)``. Defaults to zeros, or to a small
                seeded random draw when ``seed`` is given.
            lr: Adam learning rate. Must be > 0.
            max_iterations: Iteration budget for :meth:`run`. Must be >= 1.
            device: Torch device string. Defaults to ``"cpu"`` so that runs
                are reproducible and never touch a GPU implicitly; pass
                ``"cuda"`` explicitly to use one.
            seed: Optional RNG seed for the random initial coefficients.
            dtype: Working precision, ``"float32"`` (default) or
                ``"float64"``. The digital twin evaluates its far field in
                float64, so ``"float64"`` is what makes this model
                bit-identical to ``SimFourierGSNetEnv``; float32 keeps the
                default fast and matches the repo's differentiable-beam
                convention, agreeing with the twin to float32 round-off
                (~1e-7 relative).
            far_field_size: Optional zero-padding size for the pupil field
                before the FFT. ``None`` (default) keeps the historical
                behaviour of transforming at ``region``, which is what the
                digital twin's native 1:1 mode does. Pass a larger power of two
                to sample the *same* field of view more finely -- required when
                comparing against a real camera, whose pixels are far smaller
                than the unpadded ``lambda * f / (region * d_slm)`` pitch.
                Must be ``>= region`` when given.
            frozen_modes: Noll indices (1-based) to hold at zero for the whole
                fit. Defaults to empty, i.e. every mode moves. Pass
                ``(1, 2, 3)`` when fitting to a measured far-field intensity:
                piston and tilt are invisible to ``|E|**2`` (see
                :meth:`_apply_mode_mask`), so leaving them free lets the fit
                absorb unmodellable residual into degenerate directions.

        Raises:
            ValueError: If any argument is out of range or any array has the
                wrong shape.
        """
        torch = _torch()
        super().__init__(max_iterations=max_iterations)

        if not isinstance(n_orders, int) or isinstance(n_orders, bool) or n_orders < 1:
            raise ValueError(f"n_orders must be an integer >= 1, got {n_orders!r}")
        if not isinstance(region, int) or isinstance(region, bool):
            raise ValueError(f"region must be an integer, got {region!r}")
        if region < 16 or region % 2 != 0:
            raise ValueError(f"region must be >= 16 and even, got {region!r}")
        if radius is None:
            radius = region // 2
        elif not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
            raise ValueError(f"radius must be None or an integer >= 1, got {radius!r}")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be a positive finite float, got {lr!r}")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed must be None or an integer, got {seed!r}")
        if dtype not in ("float32", "float64"):
            raise ValueError(
                f"dtype must be 'float32' or 'float64', got {dtype!r}"
            )
        if far_field_size is None:
            far_field_size = int(region)
        elif (
            not isinstance(far_field_size, int)
            or isinstance(far_field_size, bool)
            or far_field_size < region
        ):
            raise ValueError(
                f"far_field_size must be None or an integer >= region "
                f"({region}), got {far_field_size!r}"
            )
        self._far_field_size = int(far_field_size)

        mask = np.ones(calc_n_zernike_terms(n_orders), dtype=np.float64)
        for noll in frozen_modes:
            index = int(noll)
            if not 1 <= index <= mask.size:
                raise ValueError(
                    f"frozen_modes entries must be Noll indices in 1..{mask.size}, "
                    f"got {noll!r}"
                )
            mask[index - 1] = 0.0
        # Kept as numpy and turned into a tensor on first use: the device is not
        # resolved yet at this point in __init__.
        self._frozen_mask: np.ndarray | None = None if mask.all() else mask
        self._mode_mask: Any = None

        n_coeffs = calc_n_zernike_terms(n_orders)
        if initial_coefficients is None:
            if seed is None:
                initial = np.zeros(n_coeffs, dtype=np.float64)
            else:
                rng = np.random.default_rng(seed)
                initial = rng.normal(0.0, 0.1, n_coeffs)
        else:
            initial = np.asarray(initial_coefficients, dtype=np.float64)
            if initial.ndim != 1 or initial.shape[0] != n_coeffs:
                raise ValueError(
                    f"initial_coefficients must have shape ({n_coeffs},) for "
                    f"n_orders={n_orders}, got {initial.shape}"
                )
            if not np.all(np.isfinite(initial)):
                raise ValueError("initial_coefficients must be finite")
        self._initial_coefficients = initial.copy()

        self._n_orders = n_orders
        self._region = region
        self._radius = radius
        self._n_coeffs = n_coeffs
        self._lr = float(lr)
        self._seed = seed
        self._dev = torch.device(device if device is not None else "cpu")
        self._dtype_name = dtype
        self._np_dtype = _numpy_dtype(dtype)

        # Precompute the Zernike basis once, for both the numpy and the torch path.
        self._basis_np = self._build_basis()
        self._basis_t = _to_tensor(self._basis_np, self._dev, dtype)
        # Aperture mask (1 inside, 0 outside), mirroring the twin's
        # ``nan_to_num(patch, nan=0.0)`` on the summed phase.
        self._aperture_t = _to_tensor(self._aperture_np, self._dev, dtype)

        self._default_amplitude = self.native_amplitude(region)
        self._source_amplitude = self._default_amplitude
        self._loss_history: list[float] = []
        self._converged = False
        self._no_improve = 0
        self._restart(restore_initial=True)

        logger.debug(
            "ZernikeCoefficientOptimizer ready: n_coefficients={} region={} "
            "radius={} lr={} device={}",
            n_coeffs,
            region,
            radius,
            self._lr,
            self._dev,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_basis(self) -> np.ndarray:
        """Build the Noll-ordered Zernike basis, zero outside the aperture.

        Returns:
            2D-array of shape ``(n_coeffs, region, region)``; mode ``j`` is
            Noll index ``j + 1``. Outside-aperture NaNs are mapped to zero.
        """
        generator = ZernikeGenerator(
            (self._region, self._region),
            radius=self._radius,
            n_orders=self._n_orders,
        )
        basis = np.zeros((self._n_coeffs, self._region, self._region))
        for index in range(self._n_coeffs):
            weights = np.zeros(self._n_coeffs)
            weights[index] = 1.0
            mode = np.asarray(generator.generate_noll(weights), dtype=np.float64)
            if index == 0:
                # The generator evaluates every mode on the same circular
                # aperture, so the piston mode's finite support *is* that
                # aperture. Deriving the mask here (instead of re-deriving the
                # geometry) keeps it exactly consistent with the twin, which
                # masks via the same NaN-outside-aperture convention.
                self._aperture_np = np.isfinite(mode)
            basis[index] = np.nan_to_num(mode, nan=0.0)
        return basis

    def _restart(self, restore_initial: bool = False) -> None:
        """Reset the coefficient tensor, Adam state and loop bookkeeping.

        Args:
            restore_initial: When ``True`` the coefficients are reset to the
                values passed to ``__init__``; otherwise the current
                coefficients are kept and only the optimizer/loop state is
                rebuilt, which warm-starts a repeated :meth:`run`.
        """
        torch = _torch()
        start = (
            self._initial_coefficients if restore_initial else self.coefficients
        )
        coefficients = torch.from_numpy(
            np.ascontiguousarray(start, dtype=self._np_dtype)
        ).to(self._dev)
        coefficients.requires_grad_(True)
        self._coefficients = coefficients
        self._opt = torch.optim.Adam([coefficients], lr=self._lr)
        self._iteration = 0
        self._convergence_history = []
        self._best_value = float("inf")
        self._loss_history = []
        self._converged = False
        self._no_improve = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @staticmethod
    def native_amplitude(region: int) -> np.ndarray:
        """Return the digital twin's native Gaussian beam for a ``region`` grid.

        The waist scales linearly with the grid so that ``region=TWIN_REGION``
        reproduces ``BeamParams.w0 = 250.0`` exactly and the resulting far
        field matches the digital twin's.

        Args:
            region: Side length of the square grid in pixels.

        Returns:
            2D float64 amplitude map, unit on-axis and Gaussian off-axis.
        """
        w0 = TWIN_W0 * (region / TWIN_REGION)
        yy, xx = np.mgrid[0:region, 0:region]
        r2 = (yy - region / 2) ** 2 + (xx - region / 2) ** 2
        return np.exp(-r2 / (2 * w0**2))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_coefficients(self) -> int:
        """Number of fitted Zernike coefficients."""
        return self._n_coeffs

    @property
    def n_orders(self) -> int:
        """Maximum radial order of the Zernike basis."""
        return self._n_orders

    @property
    def region(self) -> int:
        """Side length of the square grid in pixels."""
        return self._region

    @property
    def radius(self) -> int:
        """Zernike aperture radius in pixels."""
        return self._radius

    @property
    def device(self) -> str:
        """The torch device the optimization runs on."""
        return str(self._dev)

    @property
    def iterations(self) -> int:
        """Number of optimization steps performed so far."""
        return self._iteration

    @property
    def coefficient_tensor(self) -> torch.Tensor:
        """The live coefficient parameter tensor (requires_grad)."""
        return self._coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """Detached numpy copy of the current coefficients (radians)."""
        return self._coefficients.detach().cpu().numpy().astype(np.float64)

    @property
    def loss_history(self) -> list[float]:
        """Loss value per optimization step."""
        return self._loss_history

    @property
    def last_loss(self) -> float | None:
        """The loss at the last step, or ``None`` before the first update."""
        return self._loss_history[-1] if self._loss_history else None

    @property
    def converged(self) -> bool:
        """Whether the early-stopping criterion was met."""
        return self._converged

    @property
    def is_converged(self) -> bool:
        """True when the iteration budget is exhausted or the fit converged.

        Combines the base class' budget check with this class' own criteria:
        an absolute loss floor, or a plateau in which the best loss has not
        improved for :data:`PLATEAU_PATIENCE` consecutive steps.

        Returns:
            ``True`` when :meth:`run` should stop.
        """
        if self._iteration >= self.max_iterations:
            return True
        if self._iteration < MIN_ITERATIONS or not self._loss_history:
            return False
        if self._loss_history[-1] <= ABSOLUTE_FLOOR:
            self._converged = True
            return True
        if self._no_improve >= PLATEAU_PATIENCE:
            # The plateau is a real convergence criterion, not a budget
            # exhaustion, so the result must report it as converged.
            self._converged = True
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_basis(self) -> np.ndarray:
        """Return a copy of the cached Zernike basis.

        Returns:
            Array of shape ``(n_coefficients, region, region)`` in Noll order;
            entries outside the aperture are exactly zero. The returned array
            is a copy, so mutating it does not affect the optimizer.
        """
        return self._basis_np.copy()

    def set_source_amplitude(self, source_amplitude: np.ndarray) -> None:
        """Set the source-plane (pupil) amplitude used by subsequent steps.

        Args:
            source_amplitude: 2D amplitude map of shape
                ``(region, region)``.

        Raises:
            ValueError: If the shape is wrong or the values are not finite.
        """
        amplitude = np.asarray(source_amplitude, dtype=np.float64)
        if amplitude.shape != (self._region, self._region):
            raise ValueError(
                f"source_amplitude shape {amplitude.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("source_amplitude must be finite")
        self._source_amplitude = amplitude.copy()

    def forward_intensity(
        self,
        coefficients: np.ndarray,
        phase_slm: np.ndarray,
        source_amplitude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the far-field intensity of a coefficient vector.

        This is the single source of truth for the forward model; it is the
        same computation :meth:`update` differentiates, evaluated without a
        graph. Use it to synthesize a reference measurement.

        Args:
            coefficients: 1D coefficient vector in radians, shape
                ``(n_coefficients,)``.
            phase_slm: 2D fixed SLM phase in **raw unwrapped radians**, shape
                ``(region, region)``.
            source_amplitude: Optional 2D amplitude override for this call.

        Returns:
            2D float64 far-field intensity on the model grid.

        Raises:
            ValueError: If any array has the wrong shape or holds non-finite
                values.
        """
        torch = _torch()
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != self._n_coeffs:
            raise ValueError(
                f"coefficients must have shape ({self._n_coeffs},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficients must be finite")
        phase_t = _to_tensor(
            self._validate_phase(phase_slm), self._dev, self._dtype_name
        )
        amplitude = self._resolve_amplitude(source_amplitude)
        with torch.no_grad():
            intensity = self._far_field_intensity(
                torch.from_numpy(
                    np.ascontiguousarray(values, dtype=self._np_dtype)
                ).to(self._dev),
                phase_t,
                _to_tensor(amplitude, self._dev, self._dtype_name),
            )
        return intensity.detach().cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        """Reset the coefficients to their initial values and clear history."""
        self._restart(restore_initial=True)
        logger.debug("ZernikeCoefficientOptimizer reset to initial coefficients")

    def update(self, i_meas: np.ndarray, phase_slm: np.ndarray) -> np.ndarray:
        """Perform one Adam step and return the next coefficient vector.

        The step is measurement-anchored: the peak-normalized MSE is taken
        between the model far field and the measured one, with both the target
        and the normalization scale taken from the measurement, and the
        gradient is backpropagated through the FFT into the coefficients.

        If the optimizer has already converged, this is a no-op that returns
        the current coefficients without stepping.

        Args:
            i_meas: 2D measured far-field intensity. Resized to the model grid
                when its shape differs (real CCD frames are 250x248).
            phase_slm: 2D fixed SLM phase in **raw unwrapped radians**, shape
                ``(region, region)``.

        Returns:
            The updated coefficients as a detached numpy array (radians).

        Raises:
            ValueError: If the measurement or the phase map is invalid.
        """
        if self._converged:
            return self.coefficients

        torch = _torch()
        target, peak_anchor = self._prepare_measurement(i_meas)
        phase_t = _to_tensor(
            self._validate_phase(phase_slm), self._dev, self._dtype_name
        )
        amplitude_t = _to_tensor(
            self._source_amplitude, self._dev, self._dtype_name
        )

        self._opt.zero_grad(set_to_none=True)
        intensity = self._far_field_intensity(
            self._coefficients, phase_t, amplitude_t
        )
        # Measurement-anchored gradient: target and normalization scale both
        # come from the measured image; the residual lives on the model image.
        self._anchored_intensity_loss(intensity, target, peak_anchor).backward()
        self._opt.step()
        self._apply_mode_mask()

        with torch.no_grad():
            recorded = float(self._intensity_loss(intensity, target).detach().cpu())
        self._loss_history.append(recorded)
        self._update_stagnation(recorded)
        self._record(recorded)
        return self.coefficients

    def run(
        self,
        i_meas: np.ndarray,
        phase_slm: np.ndarray,
        source_amplitude: np.ndarray | None = None,
    ) -> ZernikeCoefficientResult:
        """Run the fitting loop until convergence or the iteration budget.

        The Adam state and the loss history are rebuilt first, but the current
        coefficients are kept, so repeated calls warm-start from the previous
        fit. Call :meth:`reset` to return to the initial coefficients.

        Args:
            i_meas: 2D measured far-field intensity, any shape.
            phase_slm: 2D fixed SLM phase in **raw unwrapped radians**, shape
                ``(region, region)``.
            source_amplitude: Optional 2D amplitude override of shape
                ``(region, region)``; restored to the native Gaussian only by
                constructing a new optimizer.

        Returns:
            A :class:`ZernikeCoefficientResult` holding the fitted
            coefficients, the per-iteration ``"loss"`` history, the number of
            steps and whether the stopping criterion was met.
        """
        self._restart()
        if source_amplitude is not None:
            self.set_source_amplitude(source_amplitude)
        logger.debug(
            "ZernikeCoefficientOptimizer.run start: budget={} n_coefficients={}",
            self.max_iterations,
            self._n_coeffs,
        )
        for _ in range(self.max_iterations):
            self.update(i_meas, phase_slm)
            if self.is_converged:
                break
        result = ZernikeCoefficientResult(
            coefficients=self.coefficients,
            history={"loss": list(self._loss_history)},
            iterations=self._iteration,
            converged=self._converged,
        )
        logger.info(
            "ZernikeCoefficientOptimizer.run done: iterations={} converged={} "
            "loss {} -> {}",
            result.iterations,
            result.converged,
            result.history["loss"][0] if result.history["loss"] else float("nan"),
            result.history["loss"][-1] if result.history["loss"] else float("nan"),
        )
        return result

    # ------------------------------------------------------------------
    # Model and loss
    # ------------------------------------------------------------------
    def _apply_mode_mask(self) -> None:
        """Zero the frozen coefficients in place after an optimizer step.

        Piston and tilt (Noll 1-3) are **unidentifiable** from a far-field
        *intensity*: piston is a global phase that leaves ``|E|**2`` exactly
        unchanged, and tilt only displaces the spot, which a same-window or
        argmax-aligned comparison barely sees. Leaving them free is not
        harmless -- with no identifiable signal the optimizer parks the
        residual in those degenerate directions instead of reporting "no
        aberration". Measured on real hardware captures, 56% of the fitted
        coefficient norm landed in Noll 1-3 while the fit quality did not
        improve at all. Freezing them follows the repo's existing convention
        (``slm_square_shaping --zernike-mask`` also forces Noll 1-3 to zero).
        """
        if self._frozen_mask is None:
            return
        torch = _torch()
        if self._mode_mask is None:
            self._mode_mask = torch.from_numpy(self._frozen_mask).to(self._dev)
        with torch.no_grad():
            self._coefficients.mul_(self._mode_mask)

    @property
    def far_field_size(self) -> int:
        """Side length of the far-field grid produced by the forward model.

        Equals ``region`` unless zero-padding was requested via
        ``far_field_size``. Callers that build a target on the *pupil* grid must
        rescale it by ``far_field_size / region`` before comparing it with a
        model far field, because a far-field pixel covers a different physical
        extent than a pupil pixel once padding is in play.
        """
        return self._far_field_size

    def _far_field_intensity(
        self,
        coefficients: torch.Tensor,
        phase_slm: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable far-field intensity of the current coefficients.

        Mirrors the digital twin's convention:
        ``|fftshift(fft2(ifftshift(U), norm="ortho"))|**2`` with
        ``U = A * exp(1j * patch)``.

        ``patch`` follows the twin exactly: the SLM phase and the aberration
        are summed first and the *sum* is then masked, because the twin applies
        ``nan_to_num(phi_slm + aberration, nan=0.0)`` and the aberration is NaN
        outside the circular aperture. The total phase is therefore flat
        (zero) outside the aperture while the Gaussian amplitude still applies
        there -- multiplying only the aberration by the mask would instead
        leave ``exp(1j * phi_slm)`` there and disagree with the twin.

        When ``far_field_size`` exceeds ``region`` the pupil field is zero-padded
        to that size before the FFT. Padding does **not** change the field of
        view -- the sampled extent stays ``lambda * f / d_slm`` -- it only
        samples it more finely, by ``far_field_size / region``. That matters on
        real benches: the far-field pixel pitch is ``lambda * f / (P * d_slm)``,
        so an unpadded ``P = region = 256`` grid has a ~65 um pitch and renders
        this bench's ~27 um spot as a sub-pixel delta (sigma ~0.4 px), which
        cannot be compared with - let alone shaped against - a camera whose
        pixels are ~2.2 um. The digital twin exposes the same knob through its
        zero-padding size ``P``; see also ``generate_zernike_farfield_sim_report``
        (FAR_N = 8192) for the canonical padded sampling.
        """
        torch = _torch()
        aberration = torch.einsum("j,jhw->hw", coefficients, self._basis_t)
        patch = (phase_slm + aberration) * self._aperture_t
        field = amplitude * torch.exp(1j * patch)
        pad = self._far_field_size
        if pad > field.shape[0]:
            half = (pad - field.shape[0]) // 2
            field = torch.nn.functional.pad(
                field.unsqueeze(0), (half, pad - field.shape[0] - half,
                                    half, pad - field.shape[1] - half)
            ).squeeze(0)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field), norm="ortho")
        )
        return spectrum.real**2 + spectrum.imag**2

    def _intensity_loss(
        self, intensity: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Peak-normalized MSE between an intensity map and the target.

        Mirrors :meth:`DifferentiableBeamOptimizer._intensity_loss` exactly:
        peak-normalize to a maximum of 1.0 and take the mean squared
        difference. Peak normalization in float32 avoids the underflow that
        energy normalization suffers when the FFT concentrates energy in a few
        pixels; the ``+ PEAK_EPS`` term guards a zero peak.

        Args:
            intensity: 2D intensity tensor, normalized by its own peak.
            target: 2D peak-normalized target tensor.

        Returns:
            Scalar MSE loss tensor, differentiable w.r.t. ``intensity``.
        """
        torch = _torch()
        normalized = intensity / (intensity.max() + PEAK_EPS)
        return torch.mean((normalized - target) ** 2)

    def _anchored_intensity_loss(
        self,
        intensity: torch.Tensor,
        target: torch.Tensor,
        peak_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Measurement-anchored loss used for the gradient step.

        The target and the normalization scale both come from the measured
        image; only the residual is evaluated on the model. This is the
        non-degenerate form of the measurement-anchored gradient of
        ``differentiable_beam.py``: evaluating the peak-normalized loss at the
        measurement against the measurement as its own target would give an
        identically zero upstream gradient.

        Args:
            intensity: 2D model intensity tensor.
            target: 2D peak-normalized measured tensor.
            peak_anchor: Scalar tensor, the measured peak used as the
                normalization scale.

        Returns:
            Scalar loss tensor, differentiable w.r.t. ``intensity``.
        """
        torch = _torch()
        return torch.mean((intensity / peak_anchor - target) ** 2)

    # ------------------------------------------------------------------
    # Input preparation and bookkeeping
    # ------------------------------------------------------------------
    def _validate_phase(self, phase_slm: np.ndarray) -> np.ndarray:
        """Validate the fixed SLM phase map and return it as a 2D array.

        The phase is consumed as **raw unwrapped radians**; this method never
        wraps or rescales it.
        """
        phase = np.asarray(phase_slm, dtype=np.float64)
        if phase.ndim != 2:
            raise ValueError(f"phase_slm must be 2D, got {phase.ndim}D")
        if phase.shape != (self._region, self._region):
            raise ValueError(
                f"phase_slm shape {phase.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(phase)):
            raise ValueError("phase_slm must be finite")
        return phase

    def _resolve_amplitude(self, source_amplitude: np.ndarray | None) -> np.ndarray:
        """Return the amplitude to use, defaulting to the stored one."""
        if source_amplitude is None:
            return self._source_amplitude
        amplitude = np.asarray(source_amplitude, dtype=np.float64)
        if amplitude.shape != (self._region, self._region):
            raise ValueError(
                f"source_amplitude shape {amplitude.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("source_amplitude must be finite")
        return amplitude

    def _prepare_measurement(
        self, i_meas: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize, peak-normalize and tensorize a measured far field.

        Args:
            i_meas: 2D measured intensity, any shape (real CCD frames are
                250x248 while the model grid is square).

        Returns:
            Tuple of the peak-normalized target tensor and the scalar
            measured-peak tensor used as the gradient anchor.

        Raises:
            ValueError: If the measurement is not 2D, is not finite, has no
                positive peak, or cannot be resized to the model grid.
        """
        torch = _torch()
        measured = np.asarray(i_meas, dtype=np.float64)
        if measured.ndim != 2:
            raise ValueError(f"i_meas must be 2D, got {measured.ndim}D")
        if not np.all(np.isfinite(measured)):
            raise ValueError("i_meas must be finite")

        grid = (self._region, self._region)
        if measured.shape != grid:
            factors = (self._region / measured.shape[0], self._region / measured.shape[1])
            measured = np.asarray(
                ndimage.zoom(measured, factors, order=1), dtype=np.float64
            )
            if measured.shape != grid:
                raise ValueError(
                    f"i_meas shape {i_meas.shape} could not be resized to "
                    f"{grid} (got {measured.shape})"
                )
            logger.debug(
                "Resized measured far field {} -> {}", i_meas.shape, grid
            )

        peak = float(measured.max())
        if peak <= 0:
            raise ValueError("i_meas must have a positive peak")
        target = _to_tensor(measured / peak, self._dev, self._dtype_name)
        return target, torch.tensor(
            peak + PEAK_EPS, dtype=self._basis_t.dtype, device=self._dev
        )

    def _update_stagnation(self, loss_value: float) -> None:
        """Advance the plateau counter used by :attr:`is_converged`."""
        previous_best = self._best_value
        if loss_value <= ABSOLUTE_FLOOR:
            self._converged = True
        if not np.isfinite(previous_best):
            self._no_improve = 0
            return
        tolerance = max(ABSOLUTE_FLOOR, RELATIVE_TOL * abs(previous_best))
        if previous_best - loss_value > tolerance:
            self._no_improve = 0
        else:
            self._no_improve += 1
