"""CCD-focal-plane Zernike modal probing (HMWS five-point estimator).

Leaf ``utils/image`` layer, same discipline as :mod:`frame_analysis`: pure
numpy, operates on already-materialised 2D intensity frames, constructs no
devices and touches no SLM / CCD / laser hardware.

Theory (Pathak & Jomy 2026, APL 128, 191102; ds-AO):
    For each Zernike mode we acquire **five** focal-plane images at the
    detector (the CCD sits in the back focal plane of the 2f lens):

    * ``a``  -- no added probe bias       (I_a)
    * ``m``  -- added ``-b`` rad          (I_-)
    * ``p``  -- added ``+b`` rad          (I_+)
    * ``mm`` -- added ``-2b`` rad         (I_--)
    * ``pp`` -- added ``+2b`` rad         (I_++)

    For a small on-axis intensity that is well approximated by a quadratic in
    the coefficient difference, the signed mode coefficient ``a`` (radians) is
    recovered by the five-point formula

        a = 0.83 * b * (I_+ + I_++ - I_- - I_--) / (I_+ + I_++ + I_- + I_-- - 4*I_a)

    with ``0.83`` the paper's empirical calibration constant.  The five-point
    combination cancels the quadratic (as well as the ``I_a`` offset) to first
    order, so a moderate ``b`` works even when a single-sided two-point ratio
    would be pulled by the curvature.

All Noll <-> (n, m) conversions are delegated to the canonical
:mod:`ao_shaping.utils.wavefront.zernike_calc`, never re-derived here
(zernike_usage red line 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

#: Paper's empirical calibration constant for the HMWS five-point formula.
CALIBRATION_CONSTANT = 0.83

#: The five probe-frame keys, in canonical order.
FRAME_ORDER: tuple[str, ...] = ("a", "m", "p", "mm", "pp")

#: Internal shorthand for the ordered keys.
_FRAME_KEYS = FRAME_ORDER


@dataclass(frozen=True)
class FivePointIntensities:
    """Five scalar focal-plane intensities for one mode's HMWS probe.

    Attributes:
        ia:   I_a    (no probe bias).
        im:   I_-    (probe ``-b``).
        ip:   I_+    (probe ``+b``).
        imm:  I_--   (probe ``-2b``).
        ipp:  I_++   (probe ``+2b``).
    """

    ia: float
    im: float
    ip: float
    imm: float
    ipp: float

    @classmethod
    def from_mapping(cls, frames: Mapping[str, float]) -> "FivePointIntensities":
        """Build from a ``{"a": ..., "m": ..., "p": ..., "mm": ..., "pp": ...}`` dict."""
        missing = [k for k in _FRAME_KEYS if k not in frames]
        if missing:
            raise ValueError(
                f"missing probe frames {missing!r}; need keys {list(_FRAME_KEYS)}"
            )
        return cls(
            ia=frames["a"],
            im=frames["m"],
            ip=frames["p"],
            imm=frames["mm"],
            ipp=frames["pp"],
        )


def probe_peak_intensity(frame: np.ndarray) -> float:
    """Focal-plane peak intensity of one frame (full-array max).

    The CCD is at the 2f back focal plane, so the bright core of the PSF is
    the relevant response to a modal probe; ``max`` over the whole frame is the
    paper's on-axis peak proxy and is robust to small spot wander.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("frame is empty")
    return float(arr.max())


def five_point_estimate(intensities: FivePointIntensities, bias_rad: float) -> float:
    """Return the signed mode coefficient ``a`` in radians (the paper formula).

    Args:
        intensities: The five focal-plane intensities for one mode.
        bias_rad:    Probe bias ``b`` in radians (``> 0``; sign is carried by the
            intensity ordering ``+``/``-``, so ``b`` itself enters as a positive
            scale).

    Returns:
        Recovered coefficient in radians.  NaN when the denominator is
        degenerate (``I_+ + I_++ + I_- + I_-- - 4 I_a == 0``), which signals a
        mode whose probe leaves the peak intensity unchanged (or a broken
        probe).
    """
    if bias_rad <= 0:
        raise ValueError(f"bias_rad must be > 0, got {bias_rad}")
    num = intensities.ip + intensities.ipp - intensities.im - intensities.imm
    den = (
        intensities.ip
        + intensities.ipp
        + intensities.im
        + intensities.imm
        - 4.0 * intensities.ia
    )
    if den == 0.0:
        return float("nan")
    return float(CALIBRATION_CONSTANT * bias_rad * num / den)


@dataclass(frozen=True)
class ProbeResult:
    """Estimated coefficient of one Zernike mode from its five probe frames."""

    noll_index: int
    coefficient_rad: float  # the paper's ``a``, signed, radians
    contrast: float  # ``I_+ + I_++ - I_- - I_--``  (crosstalk/diagnostic)
    valid: bool  # denominator non-degenerate (not NaN, finite)

    @property
    def is_valid(self) -> bool:
        """True when the five-point division was non-degenerate."""
        return self.valid


_nm2noll_converter = None


def _nm_to_noll(n: int, m: int) -> int:
    """Convert ``(n, m)`` to 1-based Noll index via a cached canonical generator."""
    global _nm2noll_converter
    if _nm2noll_converter is None:
        from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

        _nm2noll_converter = ZernikeGenerator(resolution=(2, 2))
    return int(_nm2noll_converter.nm_to_noll(n, m))


def _coerce_index(index: int | tuple[int, int]) -> int:
    """Return a 1-based Noll index from a Noll int or an ``(n, m)`` pair."""
    if isinstance(index, tuple):
        n, m = index
        return _nm_to_noll(int(n), int(m))
    if not isinstance(index, int):
        raise TypeError(f"index must be int or (n, m) tuple, got {type(index)!r}")
    if index < 1:
        raise ValueError(f"Noll index must be >= 1, got {index}")
    return index


def estimate_zernike_probe(
    frames: Mapping[str, np.ndarray] | list[Mapping[str, np.ndarray]],
    indices: list[int | tuple[int, int]],
    bias_rad: float,
) -> list[ProbeResult]:
    """Estimate per-mode Zernike coefficients from per-mode five probe frames.

    Each entry of ``indices`` drives **its own** five-frame probe (the paper's
    HMWS is per-mode biased), so ``frames`` holds the five focal-plane intensity
    maps for the probed mode(s):

    * dict form: ``{"a": I_a, "m": I_-, "p": I_+, "mm": I_--, "pp": I_++}``
      for a **single** mode, giving one :class:`ProbeResult` (requires
      ``len(indices) == 1``).
    * list form: one such dict per index, giving one result per index.

    Args:
        frames: Five focal-plane intensity frames for the probed mode(s).
        indices: Noll indices (int ``>= 1``) or ``(n, m)`` pairs, in the order
            results should be returned.
        bias_rad: Probe bias ``b`` in radians (positive).

    Returns:
        One :class:`ProbeResult` per entry of ``indices``, aligned with the
        input ``indices`` order.
    """
    n_index = len(indices)
    if n_index == 0:
        return []

    if isinstance(frames, Mapping):
        if n_index != 1:
            raise ValueError(
                "a single Mapping holds five frames for ONE mode; provide "
                f"{n_index} groups (one per index) to estimate multiple modes"
            )
        groups: list[Mapping[str, np.ndarray]] = [frames]
    else:
        groups = list(frames)

    if len(groups) != n_index:
        raise ValueError(
            f"expected {n_index} probe groups for {n_index} indices, got {len(groups)}"
        )

    results: list[ProbeResult] = []
    for idx, group in zip(indices, groups):
        noll = _coerce_index(idx)
        if not isinstance(group, Mapping):
            raise TypeError("each probe group must be a Mapping of five frames")
        peak: dict[str, float] = {}
        for k in _FRAME_KEYS:
            peak[k] = probe_peak_intensity(group[k])
        est = five_point_estimate(
            FivePointIntensities(
                ia=peak["a"],
                im=peak["m"],
                ip=peak["p"],
                imm=peak["mm"],
                ipp=peak["pp"],
            ),
            bias_rad,
        )
        results.append(
            ProbeResult(
                noll_index=noll,
                coefficient_rad=est,
                contrast=peak["p"] + peak["pp"] - peak["m"] - peak["mm"],
                valid=bool(np.isfinite(est)),
            )
        )
    return results
