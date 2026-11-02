"""Physics-informed residual refinement of the 2f forward model.

What is estimated
=================
A residual pupil phase ``delta`` that is **added to the commanded phase** ``phi``
before the (unchanged) propagation of :class:`ml.sim2f.optics.Sim2f`.  The
forward physics is therefore exact -- ``|FFT(A e^{i(phi + delta)})|^2`` resampled
to the camera -- and nothing here is a second propagator (rule 1 of
:mod:`ml.sim2f`).  ``delta`` is fitted **only** on pixels inside a
:class:`~ml.sim2f.window.CameraWindow`, so every out-of-window prediction is a
genuine extrapolation; :mod:`ml.sim2f.extrapolate` is what scores it.

Why physics terms, and what each one states
===========================================
A windowed fit is under-determined: many ``delta`` reproduce the same window
pixels and disagree completely outside it.  The regularisers below are not
cosmetic -- each one states something the hardware really cannot do, so it
removes exactly the freedom the bench does not have:

* ``support`` -- the SLM addresses a pupil disc, so a residual phase **outside**
  that disc is not realisable.  ``mean((1 - pupil) * delta^2)``, in rad^2.  (The
  literal ``||(1 - pupil) * A||^2`` of the brief is *independent of* ``delta``:
  the amplitude is set by the illumination, never by the phase.  The only
  delta-dependent quantity the aperture can forbid is the phase itself, which is
  what is penalised here.)
* ``band`` -- the panel + LUT cannot produce arbitrary spatial frequency, and a
  low-order optical surface is what a residual physically is.  The in-band part
  is the least-squares projection onto Zernike Noll 4..20 (the *fitted band*);
  the term is the illumination-weighted energy of the part that projection
  throws away, in rad^2.
* ``exposure`` -- detector linearity.  Two frames of the *same* phase at
  exposures ``e1 < e2`` must have intensity ratio ``e2/e1``; the term is the
  squared relative error of the predicted ratio against the measured one.  It
  needs no full-frame label.  On a perfectly linear detector with the same
  photometric constants on both benches it is identically zero and carries no
  gradient w.r.t. ``delta`` -- that is what the term can and cannot do, stated
  rather than hidden.
* ``smooth`` -- total variation of ``delta``: an optical surface, not pixel
  noise.

``physics=False`` keeps ``data`` plus ``smooth`` only (an ordinary
regulariser), which is the ablation that isolates the three physics terms.

Why the physics terms are absolute energies, not ratios
=======================================================
``support`` and ``band`` are mean square radians, not "fraction of the residual
energy".  A ratio is invariant to ``|delta|``, so it cannot bound the residual:
``data`` is a relative L1 that saturates once the window is already badly wrong,
and a scale-free regulariser then leaves the amplitude free to run away (that is
exactly what a first draft of this module did).  An absolute energy grows like
``|delta|^2``, so a stationary point exists and the residual amplitude is pinned
by the physics rather than by the optimiser's patience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ml.sim2f.losses import as_frame, prepare_frame

__all__ = [
    "DEFAULT_WEIGHTS",
    "ResidualFit",
    "ResidualPhase",
    "fit_residual",
    "pinn_terms",
    "zernike_band_basis",
]

#: Term weights of :func:`pinn_terms`.  ``data`` is a relative L1 in ``[0, 2]``;
#: ``support``/``band`` are mean square radians; ``exposure`` is a squared
#: relative ratio error.  Those four are all dimensionless-or-rad^2 violations of
#: a *density*, so weight 1 means "one rad^2 of violation costs one unit of
#: relative L1".  ``smooth`` is the only term measured per pixel (rad/px), so it
#: needs a length scale: 0.05 keeps the TV of a smooth residual (~0.05 rad/px) a
#: ~10% correction to ``data``, i.e. a regulariser rather than the objective.
DEFAULT_WEIGHTS: dict[str, float] = {
    "data": 1.0,
    "support": 1.0,
    "band": 1.0,
    "exposure": 1.0,
    "smooth": 0.05,
}

_TERMS = ("data", "support", "band", "exposure", "smooth")
_EPS = 1e-8


def _unit(frame: torch.Tensor) -> torch.Tensor:
    """Normalise a ``(B, 1, H, W)`` frame to unit sum per sample."""
    return frame / frame.sum(dim=(1, 2, 3), keepdim=True).clamp_min(_EPS)


def _array(value: Any) -> np.ndarray:
    """``value`` as a host numpy array, whatever device/dtype it lives on."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _projection(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares projector onto the rows of ``basis``.

    The rows are Zernike modes sampled on a grid, so they are only
    *approximately* orthonormal there; the Gram pseudo-inverse is used rather
    than a plain dot product so the projection is exact (and idempotent) for the
    sampled basis whatever its normalisation.

    Args:
        basis: ``(K, grid, grid)`` modes, zero outside the pupil.

    Returns:
        ``(flat, pinv, idx)``: ``(K, N)`` flattened modes, ``(K, Np)`` projector
        rows and the ``(Np,)`` pupil-pixel indices.
    """
    flat = basis.reshape(basis.shape[0], -1).to(torch.float64)
    idx = (flat.abs().sum(dim=0) > 0).nonzero().squeeze(1)
    pupil = flat[:, idx]
    pinv = torch.linalg.pinv(pupil @ pupil.T) @ pupil
    return flat.to(torch.float32), pinv.to(torch.float32), idx


def _project(
    delta: torch.Tensor,
    flat: torch.Tensor,
    pinv: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Band-limited part of ``delta`` (the Zernike projection)."""
    shape = delta.shape
    values = delta.reshape(shape[0], -1)
    coeffs = values[:, idx.to(values.device)] @ pinv.to(values.device).T
    return (coeffs @ flat.to(values.device)).reshape(shape)


@lru_cache(maxsize=8)
def _cached_mask_basis(
    grid: int, radius: float, n_modes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zernike Noll ``4 .. n_modes`` basis for a pupil disc of ``radius`` px."""
    from ao_shaping.utils.wavefront.zernike_calc import (
        ZernikeGenerator,
        calc_n_zernike_terms,
    )

    order = 1
    while calc_n_zernike_terms(order) < n_modes:
        order += 1
    generator = ZernikeGenerator(resolution=(grid, grid), radius=radius, n_orders=order)
    nk = calc_n_zernike_terms(order)
    basis = np.zeros((n_modes - 3, grid, grid), dtype=np.float64)
    for index in range(3, n_modes):
        coeffs = np.zeros(nk, dtype=np.float64)
        coeffs[index] = 1.0
        basis[index - 3] = np.nan_to_num(
            generator.generate_noll(coeffs), nan=0.0, posinf=0.0, neginf=0.0
        )
    return _projection(torch.as_tensor(basis))


def zernike_band_basis(
    pupil: torch.Tensor | np.ndarray, n_modes: int = 20
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Band basis for a bare pupil mask, for callers without a bench.

    Args:
        pupil: ``(grid, grid)`` or ``(1, 1, grid, grid)`` disc mask.
        n_modes: Highest Noll index of the band (the band is Noll ``4..n_modes``).

    Returns:
        ``(flat, pinv, idx)`` as :func:`_projection`.
    """
    mask = np.asarray(pupil.detach().cpu() if torch.is_tensor(pupil) else pupil)
    mask = np.squeeze(mask) > 0
    grid = int(mask.shape[-1])
    radius = float(np.sqrt(mask.sum() / math.pi))
    return _cached_mask_basis(grid, round(radius, 3), int(n_modes))


class ResidualPhase(torch.nn.Module):
    """A small MLP over the pupil grid, the model of the residual phase.

    The network is deliberately **not** a Zernike-coefficient head: a head could
    not produce the above-band content the ``band`` term exists to penalise, so
    that term would be dead code.  The field is an over-parameterised smooth map
    (coordinate + fixed Fourier features -> 2 hidden ``tanh`` layers -> 1
    channel) and the physics terms decide which of it survives.

    Attributes:
        grid: Pupil grid side.
        basis: ``(K, grid, grid)`` Noll ``4..n_modes`` modes of the fitted band.
    """

    def __init__(
        self,
        basis: torch.Tensor | np.ndarray,
        *,
        n_freqs: int = 6,
        hidden: int = 48,
        init_scale: float = 0.02,
    ) -> None:
        """Build the head.

        Args:
            basis: ``(K, grid, grid)`` band modes (``Sim2f.zernike_basis(20)[3:]``).
            n_freqs: Highest fixed Fourier feature, in cycles per pupil diameter.
            hidden: Hidden width.
            init_scale: Standard deviation of the initial phase field, radians.
        """
        super().__init__()
        tensor = basis if torch.is_tensor(basis) else torch.as_tensor(np.asarray(basis))
        tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
        if tensor.ndim != 3 or tensor.shape[0] < 1:
            raise ValueError(
                f"basis must be (K, grid, grid), got {tuple(tensor.shape)}"
            )
        self.grid = int(tensor.shape[-1])
        flat, pinv, idx = _projection(tensor)
        self.register_buffer("_basis", flat)
        self.register_buffer("_pinv", pinv)
        self.register_buffer("_idx", idx)

        coords = torch.linspace(-1.0, 1.0, self.grid)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        features = [xx.reshape(-1), yy.reshape(-1)]
        for k in range(1, int(n_freqs) + 1):
            features.extend(
                [
                    torch.sin(math.pi * k * xx).reshape(-1),
                    torch.cos(math.pi * k * xx).reshape(-1),
                    torch.sin(math.pi * k * yy).reshape(-1),
                    torch.cos(math.pi * k * yy).reshape(-1),
                ]
            )
        self.register_buffer("_features", torch.stack(features, dim=-1)[None])

        self.net = torch.nn.Sequential(
            torch.nn.Linear(len(features), hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        last = self.net[-1]
        torch.nn.init.normal_(last.weight, std=init_scale / math.sqrt(hidden))
        torch.nn.init.zeros_(last.bias)

    @property
    def basis(self) -> torch.Tensor:
        """``(K, grid, grid)`` band modes."""
        return self._basis.reshape(-1, self.grid, self.grid)

    def forward(self, batch: int | None = None) -> torch.Tensor:
        """Residual phase on the pupil grid.

        Args:
            batch: Batch size to broadcast to; ``None``/``1`` gives ``(1, grid, grid)``.

        Returns:
            ``(B, grid, grid)`` radians (one shared residual replicated over ``B``).
        """
        raw = self.net(self._features).reshape(1, self.grid, self.grid)
        size = 1 if batch is None else int(batch)
        return raw if size == 1 else raw.expand(size, self.grid, self.grid)

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """Least-squares projection of ``delta`` onto the Noll ``4..n_modes`` band."""
        return _project(delta, self._basis, self._pinv, self._idx)

    def coefficients(self, batch: int | None = None) -> torch.Tensor:
        """Noll ``4..n_modes`` coefficients of the current residual, ``(B, K)`` radians."""
        values = self.forward(batch).reshape(-1, self.grid * self.grid)
        return values[:, self._idx] @ self._pinv.T


def _residual_map(
    delta: ResidualPhase | torch.Tensor,
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Residual as ``(B, grid, grid)`` for the current batch."""
    if isinstance(delta, ResidualPhase):
        return delta.forward(batch).to(device=device, dtype=dtype)
    value = torch.as_tensor(delta, device=device, dtype=dtype)
    if value.ndim == 2:
        value = value[None]
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1)
    return value


def _band_operator(
    delta: ResidualPhase | torch.Tensor, pupil: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projection operator for ``delta``: the module's own, or one built from ``pupil``."""
    if isinstance(delta, ResidualPhase):
        return delta._basis, delta._pinv, delta._idx
    return zernike_band_basis(pupil)


def pinn_terms(
    pred_frames: Any,
    meas_frames: Any,
    *,
    delta: ResidualPhase | torch.Tensor,
    illumination: torch.Tensor | np.ndarray,
    pupil: torch.Tensor | np.ndarray,
    weights: Mapping[str, float] | None = None,
    exposure_pairs: Sequence[tuple[int, int, float]] | None = None,
) -> dict[str, torch.Tensor]:
    """Every term of the physics-informed objective, as named scalars.

    Args:
        pred_frames: Model prediction on the window, ``(B, 1, h, w)`` (any of the
            shapes :func:`ml.sim2f.losses.as_frame` accepts).
        meas_frames: Measured frames on the same window, same shape.
        delta: The residual -- a :class:`ResidualPhase`, or a ``(grid, grid)`` /
            ``(B, grid, grid)`` tensor in radians.
        illumination: ``(1, 1, grid, grid)`` entrance amplitude; its square
            weights the phase-energy terms (a residual where the beam is dark is
            unobservable and irrelevant).
        pupil: ``(1, 1, grid, grid)`` or ``(grid, grid)`` addressed-pupil mask.
        weights: Term weights, defaults to :data:`DEFAULT_WEIGHTS`.
        exposure_pairs: ``(i, j, e_j / e_i)`` index triples into the batch; frames
            ``i`` and ``j`` are the same scene at two exposures.

    Returns:
        ``{"data", "support", "band", "exposure", "smooth", "total"}``, all
        scalar tensors so they can be logged and differentiated.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    pred = as_frame(pred_frames)
    meas = as_frame(meas_frames).to(dtype=pred.dtype, device=pred.device)
    if pred.shape != meas.shape:
        raise ValueError(
            f"prediction {tuple(pred.shape)} and measurement {tuple(meas.shape)} differ"
        )
    batch = int(pred.shape[0])
    device, dtype = pred.device, pred.dtype

    values = _residual_map(delta, batch, device=device, dtype=dtype)
    grid = int(values.shape[-1])
    mask = torch.as_tensor(_array(pupil), device=device, dtype=dtype).reshape(
        1, 1, grid, grid
    )
    if float(mask.max()) > 1.0:
        mask = (mask > 0).to(dtype)
    light = torch.as_tensor(_array(illumination), device=device, dtype=dtype).reshape(
        1, 1, grid, grid
    )
    weight_map = light.pow(2)

    data = (
        (_unit(prepare_frame(pred)) - _unit(prepare_frame(meas)))
        .abs()
        .flatten(1)
        .sum(dim=1)
        .mean()
    )

    # Absolute energies (rad^2), see the module docstring: a ratio here would be
    # scale free and would leave |delta| unbounded.
    support = ((1.0 - mask) * values.pow(2)).mean(dim=(1, 2)).mean()

    flat, pinv, idx = _band_operator(delta, mask)
    above_band = (values - _project(values, flat, pinv, idx)) * mask
    band = (weight_map * above_band.pow(2)).mean()

    if exposure_pairs:
        errors = []
        for i, j, ratio in exposure_pairs:
            predicted = pred[j].sum() / pred[i].sum().clamp_min(_EPS)
            measured = meas[j].sum() / meas[i].sum().clamp_min(_EPS)
            errors.append(((predicted - measured) / measured.clamp_min(_EPS)).pow(2))
        exposure = torch.stack(errors).mean()
    else:
        exposure = pred.sum() * 0.0

    smooth = values.diff(dim=-1).abs().mean() + values.diff(dim=-2).abs().mean()

    terms = {
        "data": data,
        "support": support,
        "band": band,
        "exposure": exposure,
        "smooth": smooth,
    }
    total = torch.zeros((), device=device, dtype=dtype)
    for name in _TERMS:
        total = total + float(weights.get(name, 0.0)) * terms[name]
    return {**terms, "total": total}


@dataclass(frozen=True)
class ResidualFit:
    """One fitted residual, with the evidence for it.

    Attributes:
        delta: ``(grid, grid)`` residual phase, radians.
        coeffs: ``(K,)`` Noll ``4..n_modes`` coefficients of ``delta``, radians.
        history: Term name -> value at every epoch (``data``, ``support``,
            ``band``, ``exposure``, ``smooth``, ``total``).
        l1_start: Windowed relative L1 at the initial residual.
        l1_end: The same after the last step.
        weights: Term weights actually used (zeros show the ablation).
        ok: Whether the fit produced a finite loss throughout.
        reason: Why not, when ``ok=False``.
    """

    delta: np.ndarray
    coeffs: np.ndarray
    history: dict[str, list[float]] = field(default_factory=dict)
    l1_start: float = float("nan")
    l1_end: float = float("nan")
    weights: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    reason: str | None = None


def fit_residual(
    device_sim: Any,
    model_sim: Any,
    phases: torch.Tensor | np.ndarray,
    *,
    window: Any,
    epochs: int = 200,
    lr: float = 0.003,
    weights: Mapping[str, float] | None = None,
    seed: int = 0,
    physics: bool = True,
    aberration: torch.Tensor | np.ndarray | None = None,
    illumination: torch.Tensor | np.ndarray | None = None,
    exposures: tuple[float, float] | None = None,
    n_modes: int = 20,
) -> ResidualFit:
    """Fit a residual phase from window pixels only.

    ``prediction = model_sim.forward(phase + delta, aberration=..., illumination=...)``
    is compared to ``window.crop(device_sim.forward(phase))``; nothing outside the
    window enters the loss, so the full-frame prediction is a pure extrapolation.

    Args:
        device_sim: The bench being measured (carries the unmodelled errors).
        model_sim: The ideal/calibrated bench the residual is added to.
        phases: ``(N, grid, grid)`` commanded phases, radians.
        window: :class:`~ml.sim2f.window.CameraWindow` on the device frame.
        epochs: Adam steps.
        lr: Adam learning rate on the head's parameters.
        weights: Term weights; defaults to :data:`DEFAULT_WEIGHTS`.
        seed: Torch seed for the head initialisation.
        physics: ``False`` zeroes ``support``/``band``/``exposure`` (``data`` and
            the ``smooth`` regulariser remain).
        aberration: Base aberration map, ``(1, grid, grid)`` radians (stage 1).
        illumination: Entrance amplitude; defaults to the model's nominal beam.
        exposures: ``(e1, e2)`` -- capture every training phase at both, for the
            detector-linearity term.  ``None`` disables that term.
        n_modes: Highest Noll index of the fitted band.

    Returns:
        The :class:`ResidualFit`.

    Raises:
        ValueError: If ``window`` does not fit the camera, or ``phases`` is empty.
    """
    grid = int(model_sim.grid)
    ccd = int(model_sim.ccd_px)
    if int(window.width) > ccd or int(window.height) > ccd:
        raise ValueError(f"window {tuple(window.size)} does not fit the {ccd}px camera")
    phase = torch.as_tensor(
        _array(phases), device=model_sim.device, dtype=model_sim.dtype
    )
    if phase.ndim == 2:
        phase = phase[None]
    if phase.ndim != 3 or phase.shape[0] < 1:
        raise ValueError(f"phases must be (N, grid, grid), got {tuple(phase.shape)}")

    resolved = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if not physics:
        for name in ("support", "band", "exposure"):
            resolved[name] = 0.0

    with torch.no_grad():
        measured = window.crop(device_sim.forward(phase))
        pairs: list[tuple[int, int, float]] | None = None
        if exposures is not None:
            e1, e2 = float(exposures[0]), float(exposures[1])
            low = window.crop(device_sim.forward(phase, exposure_ms=e1))
            high = window.crop(device_sim.forward(phase, exposure_ms=e2))
            measured = torch.cat([low, high], dim=0)
            pairs = [
                (i, i + int(phase.shape[0]), e2 / e1)
                for i in range(int(phase.shape[0]))
            ]

    torch.manual_seed(int(seed))
    model = ResidualPhase(model_sim.zernike_basis(n_modes)[3:])
    model.to(device=model_sim.device, dtype=model_sim.dtype)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr))

    light = model_sim.nominal_illumination if illumination is None else illumination
    pupil = model_sim.pupil_mask
    history: dict[str, list[float]] = {name: [] for name in (*_TERMS, "total")}
    ok, reason = True, None

    for _ in range(int(epochs)):
        optimiser.zero_grad(set_to_none=True)
        delta = model.forward(int(phase.shape[0]))
        low = model_sim.forward(
            phase + delta,
            aberration=aberration,
            illumination=light,
            exposure_ms=None if exposures is None else float(exposures[0]),
        )
        predicted = low
        if pairs is not None:
            high = model_sim.forward(
                phase + delta,
                aberration=aberration,
                illumination=light,
                exposure_ms=float(exposures[1]),  # type: ignore[index]
            )
            predicted = torch.cat([low, high], dim=0)
        terms = pinn_terms(
            window.crop(predicted),
            measured,
            delta=model,
            illumination=light,
            pupil=pupil,
            weights=resolved,
            exposure_pairs=pairs,
        )
        loss = terms["total"]
        if not bool(torch.isfinite(loss)):
            ok, reason = False, "non-finite loss"
            break
        loss.backward()
        optimiser.step()
        for name, value in terms.items():
            history[name].append(float(value.detach()))

    with torch.no_grad():
        delta_map = model.forward()[0].detach().cpu().numpy().astype(np.float64)
        coeffs = model.coefficients()[0].detach().cpu().numpy().astype(np.float64)
    start = history["data"][0] if history["data"] else float("nan")
    end = history["data"][-1] if history["data"] else float("nan")
    return ResidualFit(
        delta=delta_map,
        coeffs=coeffs,
        history=history,
        l1_start=start,
        l1_end=end,
        weights=resolved,
        ok=ok,
        reason=reason,
    )
