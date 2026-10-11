"""Forward physics bank for inverse beam shaping.

The inverse-shaping objective needs to round-trip a *predicted* SLM phase back
through the bench to a camera image (the ``L_phy`` term).  That forward map is
already implemented by :class:`ml.zernike.models.ZernikeAmpModel`; this module
does **not** reimplement it.  It only assembles one calibrated forward model per
supported camera FOV bucket and dispatches a batch of phases to the right model.

Why one model per FOV bucket
----------------------------
The far field is only pixel-comparable to the corpus images when the model's
Fourier padding matches the camera window that produced them.  ``fov_px`` varies
per family (64 / 248 / 320 px windows, plus a 1944 px full-sensor family for which
no padding is calibrated), so a single global padding cannot serve all of them.
:func:`ml.zernike.models.recommended_padding` is the *single* source of that
padding -- no literal padding appears here.

Coordinate convention
---------------------
A phase is a *unit phasor* ``exp(i*phi)``.  ``ForwardBank.forward`` accepts two
input forms:

* ``(B, 2, g, g)`` -- ``(cos, sin)`` real/imag channels (what the network emits
  and what :func:`ml.shaping.models.unit_phasor` produces), or
* ``(B, g, g)`` -- a real angle in radians (the raw ``phi`` form used by
  :func:`ml.zernike.inverse_design.model_far_field`).

Both are masked by the inscribed aperture disc before propagation, matching the
bench convention that the SLM only modulates inside the pupil.

Pure ``torch`` for supported paths; importing this module needs no corpus, no
pickles and no hardware.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ml.shaping.config import UNKNOWN_FOV
from ml.zernike.inverse_design import inscribed_mask
from ml.zernike.models import (
    NO_VALID_PADDING_FOV,
    PADDING_BY_FOV_PX,
    ZernikeAmpModel,
    recommended_padding,
)

__all__ = ["ForwardBank"]

#: Default Zernike order. Mirrors ``ZernikeAmpConfig.n_max`` so the bank's
#: coefficients are interchangeable with a checkpoint trained at that order.
DEFAULT_N_MAX = 4


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, else CPU.

    Args:
        device: ``"auto"``, a device string (``"cpu"``, ``"cuda:1"``) or a
            :class:`torch.device`.

    Returns:
        The concrete :class:`torch.device` the bank lives on.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ForwardBank:
    """Dispatches a batch of phases to a per-FOV calibrated forward model.

    One :class:`ZernikeAmpModel` is built for every FOV bucket in
    :data:`ml.zernike.models.PADDING_BY_FOV_PX`, each with that bucket's
    recommended Fourier padding.  Samples whose ``fov_px`` has no calibrated
    padding (``None``, :data:`ml.shaping.config.UNKNOWN_FOV`, or a member of
    :data:`ml.zernike.models.NO_VALID_PADDING_FOV`) are reported as *excluded*
    and their prediction is left at zero rather than silently mis-scaled.

    Attributes:
        grid: Side length ``g`` of the pupil/camera grid.
        n_max: Zernike order used by every bucket model.
        device: Device the bucket models and forward pass live on.

    Note:
        Model parameters are frozen (``requires_grad_(False)``), but gradients
        still flow to the *input* phase -- the bank is differentiable w.r.t. its
        argument, which is what ``L_phy`` needs.
    """

    def __init__(
        self,
        grid: int,
        *,
        device: str | torch.device = "auto",
        n_max: int = DEFAULT_N_MAX,
        forward_checkpoint: str | Path | None = None,
    ) -> None:
        """Build one forward model per supported FOV bucket.

        Args:
            grid: Pupil/camera grid side length. Must be positive.
            device: ``"auto"`` resolves to CUDA when available, else CPU.
            n_max: Zernike order for every bucket model.
            forward_checkpoint: Optional path to a trained forward-model
                checkpoint (see
                :func:`ml.zernike.amp_checkpoint.load_trained_forward_model`).
                When given, its Zernike coefficients are copied into every bucket
                model so the ``L_phy`` term reflects the *bench-measured* forward
                optics rather than a pure Fraunhofer model.  A checkpoint whose
                ``n_max`` differs, or whose stored padding disagrees with a
                bucket's, is rejected / warned about respectively.

        Raises:
            ValueError: If ``grid`` is not positive or the checkpoint's ``n_max``
                disagrees with ``n_max``.
        """
        if int(grid) < 1:
            raise ValueError(f"grid must be positive, got {grid!r}")

        self.grid: int = int(grid)
        self.n_max: int = int(n_max)
        self.device: torch.device = _resolve_device(device)

        # The supported FOV domain is the set of buckets PADDING_BY_FOV_PX knows
        # how to calibrate. Using its keys (the domain) is not a hard-coded
        # padding value -- recommended_padding() remains the only padding source.
        self._models: dict[int, ZernikeAmpModel] = {}
        for fov in sorted(PADDING_BY_FOV_PX):
            padding = recommended_padding(fov)
            model = ZernikeAmpModel(
                n_max=self.n_max,
                grid=self.grid,
                far_field_padding=padding,
                observable="intensity",
                normalization="peak",
                center_crop=True,
                attention=False,
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._models[int(fov)] = model.to(self.device)
        logger.debug(
            "ForwardBank ready: grid={} n_max={} device={} buckets={}",
            self.grid,
            self.n_max,
            self.device,
            sorted(self._models),
        )

        self._pad_by_fov: dict[int, int] = {
            fov: recommended_padding(fov) for fov in self._models
        }

        if forward_checkpoint is not None:
            self._load_checkpoint(forward_checkpoint)

    # ------------------------------------------------------------------
    # Calibration queries
    # ------------------------------------------------------------------
    def supports(self, fov: Any) -> bool:
        """Report whether ``fov`` has a calibrated forward model here.

        Args:
            fov: A per-sample FOV in pixels (any ``int``-like value).

        Returns:
            ``True`` only when ``fov`` maps to a bucket with a valid padding --
            i.e. it is a known FOV that is neither the unknown sentinel nor a
            member of :data:`ml.zernike.models.NO_VALID_PADDING_FOV`.
        """
        if fov is None:
            return False
        try:
            value = int(fov)
        except (TypeError, ValueError):
            return False
        if value == UNKNOWN_FOV or value in NO_VALID_PADDING_FOV:
            return False
        return value in self._models

    def padding_for(self, fov: Any) -> int:
        """Return the Fourier padding used for ``fov``.

        Args:
            fov: A FOV in pixels, or ``None``.

        Returns:
            The bucket padding for a supported FOV; otherwise the calibrated
            default returned by :func:`ml.zernike.models.recommended_padding`.
        """
        return recommended_padding(fov)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        phase: torch.Tensor,
        fov: int | Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate a batch of phases to camera-space intensity.

        Args:
            phase: Either ``(B, 2, g, g)`` ``(cos, sin)`` unit-phasor channels,
                or ``(B, g, g)`` real phase angles in radians (``g == grid``).
            fov: Per-sample FOV in pixels -- a scalar applied to the whole batch,
                or a length-``B`` sequence / tensor. Rows without a calibrated
                bucket are excluded.

        Returns:
            ``(pred, excluded)`` where ``pred`` is ``(B, 1, g, g)`` peak-normalised
            intensity (zero for excluded rows) and ``excluded`` is a ``(B,)`` bool
            tensor marking the dropped rows. Both live on :attr:`device`.

        Raises:
            TypeError: If ``phase`` is not a :class:`torch.Tensor`.
            ValueError: If ``phase`` has an unsupported rank/shape, is not
                floating point, or ``fov`` cannot be broadcast to the batch size.
        """
        if not isinstance(phase, torch.Tensor):
            raise TypeError(f"phase must be a torch.Tensor, got {type(phase)!r}")

        cos, sin = self._as_phasor(phase)
        batch = cos.shape[0]

        if cos.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"phase spatial size {tuple(cos.shape[-2:])} != grid "
                f"{(self.grid, self.grid)}"
            )

        fovs = self._fov_list(fov, batch)

        cos = cos.to(device=self.device, dtype=torch.float32)
        sin = sin.to(device=self.device, dtype=torch.float32)
        mask = inscribed_mask(self.grid).to(device=self.device, dtype=torch.float32)
        cos = cos * mask
        sin = sin * mask

        pred = torch.zeros(
            (batch, 1, self.grid, self.grid), dtype=torch.float32, device=self.device
        )
        excluded = torch.zeros(batch, dtype=torch.bool, device=self.device)

        by_fov: dict[int, list[int]] = {}
        for index, value in enumerate(fovs):
            if self.supports(value):
                by_fov.setdefault(value, []).append(index)
            else:
                excluded[index] = True

        for value, indices in by_fov.items():
            selection = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            out = self._models[value](
                cos.index_select(0, selection), sin.index_select(0, selection)
            )
            pred.index_copy_(0, selection, out)

        if bool(excluded.any()):
            logger.debug(
                "ForwardBank excluded {}/{} samples with uncalibrated fov",
                int(excluded.sum()),
                batch,
            )
        return pred, excluded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _as_phasor(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``phase`` into ``(cos, sin)`` ``(B, 1, g, g)`` channels.

        Args:
            phase: ``(B, 2, g, g)`` phasor channels or ``(B, g, g)`` angles.

        Returns:
            The real and imaginary channels, each ``(B, 1, g, g)``.

        Raises:
            ValueError: If ``phase`` is neither ``(B, 2, g, g)`` nor ``(B, g, g)``.
        """
        if phase.dim() == 4 and phase.shape[1] == 2:
            return phase[:, 0:1], phase[:, 1:2]
        if phase.dim() == 3:
            # Real angle -> unit phasor, identical to model_far_field's
            # torch.polar(ones, phi) convention (never arctan2 / torch.angle).
            unit = torch.polar(torch.ones_like(phase), phase)
            return unit.real.unsqueeze(1), unit.imag.unsqueeze(1)
        raise ValueError(
            "phase must be (B, 2, g, g) phasor channels or (B, g, g) angles, "
            f"got shape {tuple(phase.shape)}"
        )

    def _fov_list(
        self, fov: int | Sequence[int] | torch.Tensor, batch: int
    ) -> list[int]:
        """Normalise ``fov`` to a length-``batch`` list of ints.

        Args:
            fov: Scalar, sequence or tensor of per-sample FOV values.
            batch: The batch size the result must match.

        Returns:
            A list of ``batch`` integer FOV values.

        Raises:
            TypeError: If ``fov`` is not an int, sequence or tensor.
            ValueError: If the resolved length does not equal ``batch``.
        """
        if isinstance(fov, torch.Tensor):
            values = [int(v) for v in fov.detach().reshape(-1).tolist()]
        elif isinstance(fov, Sequence) and not isinstance(fov, (str, bytes)):
            values = [int(v) for v in fov]
        elif isinstance(fov, (int,)):
            values = [int(fov)] * batch
        else:
            raise TypeError(f"unsupported fov type: {type(fov)!r}")

        if len(values) == 1 and batch != 1:
            values = values * batch
        if len(values) != batch:
            raise ValueError(
                f"fov length {len(values)} does not match batch size {batch}"
            )
        return values

    def _load_checkpoint(self, path: str | Path) -> None:
        """Copy trained Zernike coefficients into every bucket model.

        Args:
            path: Forward-model checkpoint path.

        Raises:
            ValueError: If the checkpoint's ``n_max`` differs from the bank's,
                since the coefficient vectors would not be interchangeable.
        """
        from ml.zernike.amp_checkpoint import load_trained_forward_model

        trained = load_trained_forward_model(path)
        if int(trained.n_max) != self.n_max:
            raise ValueError(
                f"checkpoint n_max={trained.n_max} != bank n_max={self.n_max}"
            )

        coefficients = trained.model.coefficients.detach().to(self.device)
        for fov, model in self._models.items():
            with torch.no_grad():
                model.coefficients.copy_(coefficients)
            default_padding = recommended_padding(fov)
            if int(trained.far_field_padding) != int(default_padding):
                logger.warning(
                    "checkpoint padding {} != bucket {} padding {}; keeping bucket "
                    "calibration",
                    trained.far_field_padding,
                    fov,
                    default_padding,
                )
        logger.info("ForwardBank loaded trained coefficients from {}", path)
