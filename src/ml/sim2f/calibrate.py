"""Fit the simulated 2f bench to the device's *own* measurements.

What this stage is
==================
:mod:`ml.sim2f.fit` estimates the aberration for a bench whose geometry is
already trusted; this module estimates the bench *state* that fit would
otherwise quietly absorb.  The two halves are complementary: left free, Zernike
coefficients explain a wrong beam width or focal magnification as defocus and
coma, the residual drops, the coefficients are wrong, and the next loop starts
from a bench that does not exist.  Geometry is therefore fitted first from
paired samples recorded by :mod:`ml.sim2f.bench`.  Everything fitted is a
parameter of :class:`ml.sim2f.optics.Sim2f`, so the fit is plain autograd
through the one propagator (package rule 1).

What is fitted, and what is deliberately not
============================================
* **Beam** -- width along two axes, rotation and centre offset, as an
  anisotropic Gaussian amplitude through the same ``illumination=`` hook.
* **Camera mapping** -- magnification and shift on the *predicted frame*:
  ``sampling_ratio`` is known only approximately, and a reported zoom is more
  useful than dumping that error into Zernike terms.  They stay in
  :attr:`CalibrationResult.camera`, never the physical config.  The post-warp
  detector MTF lives in :attr:`CalibrationResult.measurement` and is fitted
  second with geometry frozen; background is structural, off by default.  The
  coherent zero-order ``eta`` (:attr:`CalibrationResult.zero_order`) is fitted
  last, alone, bounded -- it trades against the aberration, so it is never free
  at the same time.
* **Aberration** -- Zernike coefficients Noll ``4 .. zernike_n_max``.  Noll 2/3
  (tip/tilt) are frozen: a pupil tilt *is* a shift of the far field, so moving
  them would make the state unidentifiable against ``fit_center``.
* **Gain** -- projected out in closed form, as :func:`ml.sim2f.fit._residual`
  does: absolute brightness is exposure x laser power x PRNU, not optical path.

Sample-side discipline
======================
A quarter of the samples (by seed) are held out and never differentiated
against.  :attr:`CalibrationResult.ok` is the generalisation gate: the calibrated
bench must predict them at least as well as the nominal bench did, else the
returned state is the nominal one -- an unusable estimate reads as "no
correction", never as a partially converged one.

Data source
===========
:func:`load_bench_session` reads a :mod:`ml.sim2f.bench` session
(``session.json`` + ``process_data/`` sidecars).  One recorded sample
contributes four pairs -- A1/B1/B2/A2 of the ABBA protocol.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import measurement
from .config import CalibrateConfig, OpticsErrors, Sim2fConfig
from .losses import prepare_frame
from .measurement import camera_affine as _camera_affine

__all__ = [
    "BenchSample",
    "BenchSession",
    "CalibratedBench",
    "CalibrationResult",
    "FrameScore",
    "calibrate_bench",
    "load_bench_session",
    "load_calibration",
    "save_calibration",
]

logger = logging.getLogger(__name__)

#: Guard for the gain denominators below.
_EPS = 1e-12

#: The ABBA conditions :func:`ml.sim2f.bench._record_row` writes per sample.
_CONDITIONS: tuple[str, ...] = ("a1", "b1", "b2", "a2")

#: Which recorded rows are "the flat command" and which are "the network's".
_CONDITION_GROUPS: dict[str, tuple[str, ...]] = {
    "flat": ("a1", "a2"),
    "net": ("b1", "b2"),
}

#: Bounds the fit is projected onto after every step, so a divergent Adam step
#: cannot leave the physically meaningful region and take the fit with it.  Wide
#: on purpose: these are plausibility limits, not regularisers, and a fit that
#: presses against one is a finding to report, not a silent clamp to hide.
_MAX_LOG_WIDTH = 0.7  # exp(+-0.7) ~ a factor of 2 either way on the waist
_MAX_ANGLE_RAD = math.pi
_MAX_SHIFT_PX = 64.0
_MAX_LOG_ZOOM = 0.6  # exp(0.6) ~ 1.8: the hw session's camera is ~1.6x the model
_MAX_COEFF_RAD = 3.0
#: Bound of the coherent zero-order fraction fitted in the last stage, and the
#: raw logit clamp that keeps ``sigmoid`` off its flat tails.
_ETA_MAX = 0.5
_ETA_LOGIT_MAX = 12.0

#: Noll indices frozen when the aberration is fitted alongside the camera shift.
_FROZEN_NOLL: tuple[int, ...] = (2, 3)

#: Layout of the flat parameter vector the Adam step differentiates.  Named
#: slots, because an index typed as a literal is how a fit ends up moving the
#: wrong knob while reporting a plausible number.
#: ``[0:5]`` beam (log width x, log width y, angle, centre x, centre y);
#: ``[5]`` log zoom; ``[6:8]`` camera shift; ``[8:]`` Zernike radians.
_LOG_ZOOM = 5
_SHIFT_X = 6
_SHIFT_Y = 7
_COEFF_START = 8


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchSample:
    """One measured (commanded phase, measured frame) pair.

    Attributes:
        phase: ``(grid, grid)`` float32 radians -- the command the panel got,
            in the model's own convention (zero is the flat command).
        frame: ``(ccd_px, ccd_px)`` float32 measured counts.
        label: Human-readable origin, e.g. ``"s47/b1"``.
    """

    phase: np.ndarray
    frame: np.ndarray
    label: str


@dataclass(frozen=True)
class BenchSession:
    """A recorded session, ready to fit against.

    Attributes:
        root: Directory the session was read from.
        samples: The (command, measurement) pairs, in recording order.
        sim_config: The bench geometry the session was recorded on, rebuilt
            from ``session.json``.
        backend: ``"hw"`` for a hardware session, ``"sim"`` for a twin run.
        note: Free-form note carried from the session, for the report.
    """

    root: Path
    samples: tuple[BenchSample, ...]
    sim_config: Sim2fConfig
    backend: str
    note: str = ""

    def phases(self) -> np.ndarray:
        """Stack every sample's command, ``(n, grid, grid)`` float32."""
        return np.stack([sample.phase for sample in self.samples]).astype(np.float32)

    def frames(self) -> np.ndarray:
        """Stack every sample's measurement, ``(n, ccd_px, ccd_px)`` float32."""
        return np.stack([sample.frame for sample in self.samples]).astype(np.float32)


def _error_value(name: str, value: object) -> object:
    """Coerce one recorded error field back into its config type.

    ``session.json`` is JSON, so the Zernike pairs come back as *lists* while
    :class:`~ml.sim2f.config.OpticsErrors` requires hashable tuples (a frozen
    dataclass holding a list is not hashable).  Everything else round-trips as is.
    """
    if name in ("slm_zernike", "lens_zernike") and isinstance(value, (list, tuple)):
        return tuple((int(noll), float(rad)) for noll, rad in value)
    return value


def _sim_config_from_bench(bench: dict[str, object]) -> Sim2fConfig:
    """Rebuild the session's :class:`Sim2fConfig` from its ``bench`` block.

    The block is :meth:`~ml.sim2f.config.Sim2fConfig.to_dict`-shaped, so fields
    are taken by name and a key the config does not carry (``sampling_ratio``
    and friends are derived) is skipped rather than guessed at.  Raises
    ``ValueError`` if the block lacks the geometry keys the config needs.
    """
    names = {item.name for item in dataclasses.fields(Sim2fConfig)}
    error_names = {item.name for item in dataclasses.fields(OpticsErrors)}
    payload: dict[str, object] = {}
    for name, value in bench.items():
        if name in names:
            payload[name] = value

    # ``fft_length`` is a derived property of the config, while the session
    # records it; back out the padding it was recorded with so the rebuild is
    # the same bench, not a bench with the same numbers in different slots.
    grid = int(payload.get("grid", 0))
    if "far_field_padding" not in payload and grid > 0 and "fft_length" in bench:
        payload["far_field_padding"] = int(bench["fft_length"]) / grid

    errors = bench.get("errors")
    if isinstance(errors, dict):
        payload["errors"] = OpticsErrors(
            **{k: _error_value(k, v) for k, v in errors.items() if k in error_names}
        )
    for name in ("grid", "ccd_px", "pupil_fill", "focal_length_m"):
        if not payload.get(name):
            raise ValueError(f"session bench block lacks {name!r}: {sorted(payload)}")
    return Sim2fConfig(**payload)  # type: ignore[arg-type]


def load_bench_session(
    root: str | Path,
    *,
    groups: tuple[str, ...] = ("flat", "net"),
    max_samples: int | None = None,
) -> BenchSession:
    """Read a recorded session directory into (command, measurement) pairs.

    ``root`` holds ``session.json`` and ``process_data/``.  ``groups`` selects
    the recorded conditions (``"flat"`` = the A pair, ``"net"`` = the B pair);
    ``max_samples`` keeps only the first n by recording order.  Raises
    ``FileNotFoundError`` when the recording is missing and ``ValueError`` when
    no row survives ``groups`` or an array does not match the session geometry.
    """
    directory = Path(root)
    session_path = directory / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"no session.json under {directory}")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    bench = dict(payload.get("bench") or {})
    if not bench:
        raise ValueError(f"{session_path} has no bench block")

    wanted = sorted({name for group in groups for name in _CONDITION_GROUPS[group]})
    table_path = directory / "process_data" / "bench_session.csv"
    sidecars = directory / "process_data" / "bench_session_sidecars"
    if not table_path.is_file() or not sidecars.is_dir():
        raise FileNotFoundError(f"{directory} has no process_data recording")

    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    samples: list[BenchSample] = []
    kept: set[int] = set()
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in wanted:
            continue
        sample_id = int(row.get("index", -1))
        if max_samples is not None and sample_id not in kept and len(kept) >= int(max_samples):
            continue
        index = int(row["_id"])
        phase = np.load(sidecars / f"bench_session__c_{index:04d}.npy")
        frame = np.load(sidecars / f"bench_session__img_{index:04d}.npy")
        if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
            raise ValueError(f"row {index}: command phase has shape {phase.shape}")
        if frame.ndim != 2:
            raise ValueError(f"row {index}: frame has shape {frame.shape}")
        kept.add(sample_id)
        samples.append(
            BenchSample(
                phase=np.asarray(phase, dtype=np.float32),
                frame=np.asarray(frame, dtype=np.float32),
                label=f"s{row.get('index', index)}/{condition}",
            )
        )

    if not samples:
        raise ValueError(f"no rows matched groups {groups!r} in {table_path}")
    return BenchSession(
        root=directory,
        samples=tuple(samples),
        sim_config=_sim_config_from_bench(bench),
        backend=str(payload.get("backend", "?")),
        note=str(payload.get("notes", ""))[:400],
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameScore:
    """How well a bench state predicts one set of frames.

    Attributes:
        l1: Mean relative L1 of the gain-matched prediction against the
            measurement (lower is better; 0 is a perfect prediction).
        ncc: Mean normalised cross-correlation (higher is better; 1 is perfect).
        gain: Mean of the per-frame gains that were projected out.
    """

    l1: float
    ncc: float
    gain: float


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted bench state, with the evidence for it.

    ``ok=False`` is a *documented* failure: :attr:`reason` names the gate, and
    :attr:`aberration` / :attr:`illumination` return the nominal bench.  The
    trajectory stays in :attr:`history` and :attr:`score` so it is not silent.

    Attributes:
        ok: Whether the fit generalised (passes the holdout gate).
        reason: Why not (``None`` when ``ok``).
        sim_config: The unchanged bench geometry (fitted knobs are runtime state).
        aberration: ``(grid, grid)`` float32 residual radians; zeros on failure.
        illumination: ``(grid, grid)`` float32 amplitude, peak-normalised to 1.
        beam: Fitted widths, model-pixel widths, angle and centre.
        camera: Fitted camera nuisance state -- ``zoom`` and ``shift_px``.
        measurement: Fitted :class:`ml.sim2f.measurement.MeasurementTransfer`;
            identity on failure or when not fitted.
        zero_order: Fitted coherent zero-order state (``{"eta": ...}``); empty
            when not fitted, so the wrapped bench's nominal eta applies.
        free: Names of the parameters the fit was allowed to move.
        n_train: Samples differentiated against.
        n_holdout: Samples held out of the fit.
        score: Fit quality after the fit, per split.
        score_start: The same, at the nominal state.
        history: Objective after each Adam step.
        iterations: Adam steps taken.
        config: The configuration the fit ran under.
    """

    ok: bool
    reason: str | None
    sim_config: Sim2fConfig
    aberration: np.ndarray
    illumination: np.ndarray
    beam: dict[str, float] = field(default_factory=dict)
    camera: dict[str, float] = field(default_factory=dict)
    measurement: dict[str, float] = field(default_factory=dict)
    zero_order: dict[str, float] = field(default_factory=dict)
    free: tuple[str, ...] = ()
    n_train: int = 0
    n_holdout: int = 0
    score: dict[str, FrameScore] = field(default_factory=dict)
    score_start: dict[str, FrameScore] = field(default_factory=dict)
    history: tuple[float, ...] = ()
    iterations: int = 0
    config: CalibrateConfig = CalibrateConfig()

    def to_dict(self) -> dict[str, object]:
        """JSON-able digest (the arrays are saved separately, see :func:`save_calibration`)."""
        return {
            "ok": self.ok,
            "reason": self.reason,
            "beam": dict(self.beam),
            "camera": dict(self.camera),
            "measurement": dict(self.measurement),
            "zero_order": dict(self.zero_order),
            "free": list(self.free),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "score": {name: dataclasses.asdict(score) for name, score in self.score.items()},
            "score_start": {
                name: dataclasses.asdict(score) for name, score in self.score_start.items()
            },
            "history": list(self.history),
            "iterations": self.iterations,
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _beam_amplitude(
    sim: object,
    log_width_x: torch.Tensor,
    log_width_y: torch.Tensor,
    angle: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic Gaussian entrance amplitude, ``(1, 1, grid, grid)``, peak 1.

    The anisotropic case is the point: :meth:`Sim2f.illumination` only offers an
    isotropic waist, and an elliptical pupil beam is a mismatch an aberration
    fit would otherwise spend coma on.  The two log widths are along the beam's
    own axes, rotated by ``angle`` radians; centres are model pixels off axis.
    """
    grid = int(sim.grid)  # type: ignore[attr-defined]
    nominal = float(sim.cfg.beam_w0_model_px)  # type: ignore[attr-defined]
    w_x = nominal * torch.exp(log_width_x)
    w_y = nominal * torch.exp(log_width_y)
    # The bench's own coordinate grid (``optics.Sim2f.__init__``): model pixels,
    # origin at ``(grid - 1) / 2``.  Sharing it is what makes the zero state the
    # nominal bench exactly.
    axis = torch.arange(grid, dtype=log_width_x.dtype, device=log_width_x.device) - (grid - 1) / 2.0
    x_px = axis.view(1, grid)
    y_px = axis.view(grid, 1)
    dx = x_px - center_x
    dy = y_px - center_y
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    u = (dx * cos_a + dy * sin_a) / w_x
    v = (-dx * sin_a + dy * cos_a) / w_y
    amplitude = torch.exp(-0.5 * (u * u + v * v))
    return amplitude.view(1, 1, grid, grid).contiguous()


def _scores(
    predicted: torch.Tensor, measured: torch.Tensor, *, fit_gain: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a batch of predictions against the matching measurements.

    Both sides are :func:`~ml.sim2f.losses.prepare_frame`d first: a measured
    frame carries a dark pedestal, so a raw residual measures the pedestal as
    much as the light.  ``fit_gain`` projects the per-frame gain out in closed
    form; ``False`` fixes both sides to unit sum (only meaningful when the
    exposure is known to match).  Returns ``(l1, ncc, gain)``, each ``(n,)``.
    """
    pred = prepare_frame(predicted)
    meas = prepare_frame(measured)
    if fit_gain:
        gain = (pred * meas).sum(dim=(-2, -1)) / pred.pow(2).sum(dim=(-2, -1)).clamp_min(_EPS)
    else:
        gain = meas.sum(dim=(-2, -1)) / pred.sum(dim=(-2, -1)).clamp_min(_EPS)
    difference = pred * gain.unsqueeze(-1).unsqueeze(-1) - meas
    l1 = difference.abs().sum(dim=(-2, -1)) / meas.abs().sum(dim=(-2, -1)).clamp_min(_EPS)

    pred_c = pred - pred.mean(dim=(-2, -1), keepdim=True)
    meas_c = meas - meas.mean(dim=(-2, -1), keepdim=True)
    num = (pred_c * meas_c).sum(dim=(-2, -1))
    den = (
        pred_c.pow(2).sum(dim=(-2, -1)) * meas_c.pow(2).sum(dim=(-2, -1))
    ).clamp_min(_EPS).sqrt()
    return l1, num / den, gain


def _mean_score(
    predicted: torch.Tensor, measured: torch.Tensor, *, fit_gain: bool
) -> FrameScore:
    """Score one split and reduce it to a :class:`FrameScore`.

    Args:
        predicted: ``(n, 1, ccd_px, ccd_px)`` model intensity.
        measured: ``(n, 1, ccd_px, ccd_px)`` measured counts.
        fit_gain: Passed through to :func:`_scores`.

    Returns:
        The split's mean score.
    """
    with torch.no_grad():
        l1, ncc, gain = _scores(predicted, measured, fit_gain=fit_gain)
    return FrameScore(
        l1=float(l1.mean()),
        ncc=float(ncc.mean()),
        gain=float(gain.mean()),
    )


def _eta_from_raw(value: torch.Tensor) -> torch.Tensor:
    """Bounded coherent zero-order fraction, ``_ETA_MAX * sigmoid(value)``."""
    return _ETA_MAX * torch.sigmoid(value)


def _eta_raw(eta: float) -> float:
    """Inverse of :func:`_eta_from_raw`, for the initial value."""
    ratio = min(max(float(eta) / _ETA_MAX, 1e-4), 1.0 - 1e-4)
    return math.log(ratio / (1.0 - ratio))


def _predict(
    sim: object,
    phases: torch.Tensor,
    raw: torch.Tensor,
    n_coeffs: int,
    n_measure: int = 0,
    eta: torch.Tensor | None = None,
) -> torch.Tensor:
    """One bench state applied to every command: the whole forward model.

    ``raw`` is the flat parameter vector of :func:`calibrate_bench`;
    ``n_measure`` counts its trailing detector-MTF parameters (0 identity, 1
    PSF sigma, 2 sigma + explicitly enabled pedestal) and ``eta`` is the
    optional differentiable coherent zero-order override.  Returns
    ``(n, 1, ccd_px, ccd_px)`` camera-warped intensity.
    """
    illumination = _beam_amplitude(sim, raw[0], raw[1], raw[2], raw[3], raw[4])
    aberration = None
    if n_coeffs > 0:
        aberration = sim.aberration(  # type: ignore[attr-defined]
            raw[_COEFF_START : _COEFF_START + n_coeffs].view(1, n_coeffs)
        )
    intensity = sim.forward(  # type: ignore[attr-defined]
        phases,
        aberration=aberration,
        illumination=illumination,
        detector=False,
        zero_order_eta=eta,
    )
    frame = _camera_affine(intensity, raw[_SHIFT_X], raw[_SHIFT_Y], raw[_LOG_ZOOM])
    if n_measure:
        start = _COEFF_START + n_coeffs
        background = raw[start + 1] if n_measure > 1 else 0.0
        frame = measurement.apply_transfer(frame, raw[start], background)
    return frame


def _moments(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First and second moments of a batch of prepared frames.

    Args:
        frame: ``(n, 1, h, w)`` non-negative intensity.

    Returns:
        ``(center_x, center_y, rms_radius)``, each ``(n,)``; the radius is the
        root of the mean of the per-axis variances.
    """
    total = frame.sum(dim=(-2, -1)).clamp_min(_EPS)
    height = int(frame.shape[-2])
    width = int(frame.shape[-1])
    ys = torch.arange(height, dtype=frame.dtype, device=frame.device).view(height, 1)
    xs = torch.arange(width, dtype=frame.dtype, device=frame.device).view(1, width)
    weight = frame[:, 0] / total.unsqueeze(-1).unsqueeze(-1)
    cx = (weight * xs).sum(dim=(-2, -1))
    cy = (weight * ys).sum(dim=(-2, -1))
    var_x = (weight * (xs - cx.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-2, -1))
    var_y = (weight * (ys - cy.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-2, -1))
    return cx, cy, torch.sqrt(0.5 * (var_x + var_y))


def _init_camera(
    sim: object,
    phases: torch.Tensor,
    measured: torch.Tensor,
    raw: torch.Tensor,
    n_coeffs: int,
) -> None:
    """Seed the camera zoom/shift from the image moments of the calmest rows.

    A gradient fit over zoom+shift is badly conditioned from zero (the warp has
    to travel ~40 px before any correlation gradient exists), while the moments
    of the measured and predicted frames give both quantities in closed form.
    The rows used are the ones whose command is closest to flat -- the flat
    frame is the one case where the *model* spot is trusted to be centred.

    Args:
        sim: The :class:`~ml.sim2f.optics.Sim2f` bench.
        phases: ``(n, grid, grid)`` commanded radians.
        measured: ``(n, 1, ccd_px, ccd_px)`` measured counts.
        raw: The flat parameter vector; its zoom/shift slots are set in place.
        n_coeffs: Coefficient block length in ``raw``.
    """
    rms = phases.flatten(1).pow(2).mean(dim=1)
    keep = torch.argsort(rms)[: max(4, phases.shape[0] // 8)]
    with torch.no_grad():
        prediction = prepare_frame(_predict(sim, phases[keep], raw, n_coeffs))
        reference = prepare_frame(measured[keep])
        mx, my, mr = _moments(reference)
        px, py, pr = _moments(prediction)
        zoom = (mr / pr.clamp_min(_EPS)).clamp(0.2, math.exp(_MAX_LOG_ZOOM)).mean()
        # The measured content sits where the model's content, scaled by the
        # zoom about the frame centre, lands: shift = m - c - zoom * (p - c).
        cx_frame = torch.as_tensor((measured.shape[-1] - 1) / 2.0, dtype=raw.dtype, device=raw.device)
        cy_frame = torch.as_tensor((measured.shape[-2] - 1) / 2.0, dtype=raw.dtype, device=raw.device)
        shift_x = ((mx - cx_frame) - zoom * (px - cx_frame)).mean()
        shift_y = ((my - cy_frame) - zoom * (py - cy_frame)).mean()
        raw[_LOG_ZOOM] = torch.log(zoom)
        raw[_SHIFT_X] = shift_x.clamp(-_MAX_SHIFT_PX, _MAX_SHIFT_PX)
        raw[_SHIFT_Y] = shift_y.clamp(-_MAX_SHIFT_PX, _MAX_SHIFT_PX)


def calibrate_bench(
    sim: object,
    session: BenchSession,
    config: CalibrateConfig | None = None,
) -> CalibrationResult:
    """Fit the bench state to the session's measured frames.

    The fit minimises ``w_l1 * relative L1 + w_ncc * (1 - NCC)`` over the
    training samples, through :meth:`Sim2f.forward` with the detector off: noise
    and quantisation are properties of one measurement, not of the optical path.

    When ``fit_measurement`` is enabled, a second Adam stage then fits detector
    MTF sigma alone with the geometry and aberration frozen; the same training
    objective and holdout gate judge the result.  Background fitting is a
    separate, default-off experiment because it is degenerate with gain.

    Args:
        sim: The :class:`~ml.sim2f.optics.Sim2f` bench to calibrate.  Its config
            is the starting point; the fitted knobs are returned as runtime state.
        session: The recorded (command, measurement) pairs.
        config: Fit contract; the defaults when ``None``.

    Returns:
        The :class:`CalibrationResult`.

    Raises:
        ValueError: If a sample does not match the bench's grids, or the session
            is too small to split.
    """
    cfg = config or CalibrateConfig()
    grid = int(sim.grid)  # type: ignore[attr-defined]
    ccd_px = int(sim.ccd_px)  # type: ignore[attr-defined]
    phases_np = session.phases()
    frames_np = session.frames()
    if phases_np.shape[1:] != (grid, grid):
        raise ValueError(
            f"session phases are {phases_np.shape[1:]}, bench grid is {(grid, grid)}"
        )
    if frames_np.shape[1:] != (ccd_px, ccd_px):
        raise ValueError(
            f"session frames are {frames_np.shape[1:]}, bench ccd_px is {(ccd_px, ccd_px)}"
        )

    device = torch.device(sim.device)  # type: ignore[attr-defined]
    dtype = sim.dtype  # type: ignore[attr-defined]
    phases = torch.as_tensor(phases_np, dtype=dtype, device=device)
    measured = torch.as_tensor(frames_np, dtype=dtype, device=device).unsqueeze(1)

    count = int(phases.shape[0])
    n_holdout = int(round(float(cfg.holdout_frac) * count))
    n_holdout = max(0, min(n_holdout, count - 1))
    order = np.random.default_rng(int(cfg.seed)).permutation(count)
    holdout = np.sort(order[:n_holdout])
    train = np.sort(order[n_holdout:])
    if train.size == 0:
        raise ValueError(f"session has {count} samples; nothing left to fit on")

    # -- vector: [beam(5), zoom, shift(2), coeffs..., measure(0|1|2), eta(0|1)] --
    n_coeffs = int(cfg.zernike_n_max) - 1 if cfg.fit_zernike else 0  # Noll 2..n_max
    measure_start = _COEFF_START + n_coeffs
    n_measure = 0
    if cfg.fit_measurement:
        n_measure = 2 if cfg.fit_measurement_background else 1
    zero_start = measure_start + n_measure
    n_eta = 1 if cfg.fit_zero_order else 0
    raw = torch.zeros(
        zero_start + n_eta, dtype=dtype, device=device, requires_grad=True
    )
    free = torch.zeros_like(raw, dtype=torch.bool)
    names: list[str] = [
        "log_width_x",
        "log_width_y",
        "beam_angle_rad",
        "beam_center_x_px",
        "beam_center_y_px",
        "log_zoom",
        "camera_shift_x_px",
        "camera_shift_y_px",
    ]
    if cfg.fit_beam:
        free[0] = free[1] = free[2] = True
    if cfg.fit_center:
        free[3] = free[4] = True
    if cfg.fit_scale:
        free[_LOG_ZOOM] = True
        free[_SHIFT_X] = free[_SHIFT_Y] = True
    # Tip/tilt (Noll 2,3) stay frozen: they are the far-field shift, and the
    # shift is already a free camera parameter.
    if n_coeffs > 2:
        free[_COEFF_START + 2 : _COEFF_START + n_coeffs] = True
        names.extend(f"noll_{noll}" for noll in range(4, int(cfg.zernike_n_max) + 1))
    if n_measure:
        names.append("psf_sigma_px")
        if n_measure > 1:
            names.append("background_fraction")
    if n_eta:
        names.append("zero_order_eta")
    free_names = tuple(
        name for index, name in enumerate(names) if free[index] or index >= measure_start
    )

    nominal_eta = float(sim.cfg.errors.zero_order_eta)  # type: ignore[attr-defined]
    if n_eta:
        with torch.no_grad():
            raw[zero_start] = _eta_raw(nominal_eta or 0.05)

    _init_camera(sim, phases, measured, raw, n_coeffs)

    holdout_t = torch.as_tensor(holdout, dtype=torch.long, device=device)
    train_t = torch.as_tensor(train, dtype=torch.long, device=device)

    def _split_scores(prediction: torch.Tensor) -> dict[str, FrameScore]:
        return {
            "train": _mean_score(prediction[train_t], measured[train_t], fit_gain=cfg.fit_gain),
            "holdout": _mean_score(
                prediction[holdout_t], measured[holdout_t], fit_gain=cfg.fit_gain
            ),
        }

    def _objective(rows: torch.Tensor, prediction: torch.Tensor, active: int) -> torch.Tensor:
        l1, ncc, gain = _scores(prediction[rows], measured[rows], fit_gain=cfg.fit_gain)
        per_frame = float(cfg.w_l1) * l1 + float(cfg.w_ncc) * (1.0 - ncc)
        loss = per_frame.mean()
        # Only the explicitly enabled pedestal fit sees the raw-domain term.
        if active == 1 and n_measure > 1:
            loss = loss + measurement.PEDESTAL_WEIGHT * measurement.pedestal_mismatch(
                prediction[rows], measured[rows], gain
            ).mean()
        return loss

    with torch.no_grad():
        eta = _eta_from_raw(raw[zero_start]) if n_eta else None
        start_prediction = _predict(sim, phases, raw, n_coeffs, n_measure, eta=eta)
    score_start = _split_scores(start_prediction)
    history: list[float] = []
    logger.info(
        "calibrate: %d samples (%d train / %d holdout), free %s",
        count,
        int(train.size),
        int(holdout.size),
        ", ".join(free_names) or "(none)",
    )
    logger.info(
        "calibrate: nominal train l1 %.4f ncc %.4f | holdout l1 %.4f ncc %.4f",
        score_start["train"].l1,
        score_start["train"].ncc,
        score_start["holdout"].l1,
        score_start["holdout"].ncc,
    )

    steps = int(cfg.iters)
    lower = torch.tensor(
        [-_MAX_LOG_WIDTH, -_MAX_LOG_WIDTH, -_MAX_ANGLE_RAD, -_MAX_SHIFT_PX,
         -_MAX_SHIFT_PX, -_MAX_LOG_ZOOM, -_MAX_SHIFT_PX, -_MAX_SHIFT_PX]
        + [-_MAX_COEFF_RAD] * n_coeffs + [0.0] * n_measure + [-_ETA_LOGIT_MAX] * n_eta,
        dtype=dtype, device=device,
    )
    upper = torch.tensor(
        [_MAX_LOG_WIDTH, _MAX_LOG_WIDTH, _MAX_ANGLE_RAD, _MAX_SHIFT_PX,
         _MAX_SHIFT_PX, _MAX_LOG_ZOOM, _MAX_SHIFT_PX, _MAX_SHIFT_PX]
        + [_MAX_COEFF_RAD] * n_coeffs
        + [measurement.MAX_PSF_SIGMA_PX, measurement.MAX_BACKGROUND_FRACTION][:n_measure]
        + [_ETA_LOGIT_MAX] * n_eta,
        dtype=dtype, device=device,
    )
    stages = [0] + ([1] if n_measure else []) + ([2] if n_eta else [])
    # Geometry, then detector MTF, then eta: sigma, beam width and camera zoom all
    # change the measured spot size, and eta trades against the fitted aberration.
    for stage in stages:
        if stage:
            free[:] = False
            if stage == 1:
                free[measure_start : measure_start + n_measure] = True
                with torch.no_grad():
                    raw[measure_start] = measurement.INIT_PSF_SIGMA_PX
            else:
                free[zero_start] = True
        optimiser = torch.optim.Adam([raw], lr=float(cfg.lr))
        for step in range(steps):
            decay = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, steps - 1)))
            for group in optimiser.param_groups:
                group["lr"] = float(cfg.lr) * decay
            eta = _eta_from_raw(raw[zero_start]) if n_eta else None
            prediction = _predict(sim, phases, raw, n_coeffs, n_measure, eta=eta)
            loss = _objective(train_t, prediction, stage)
            history.append(float(loss.detach()))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if raw.grad is not None:
                raw.grad[~free] = 0.0
            optimiser.step()
            with torch.no_grad():
                raw.clamp_(min=lower, max=upper)

    with torch.no_grad():
        eta = _eta_from_raw(raw[zero_start]) if n_eta else None
        final_prediction = _predict(sim, phases, raw, n_coeffs, n_measure, eta=eta)
    score = _split_scores(final_prediction)

    # -- the generalisation gate ---------------------------------------------
    ok = True
    reason: str | None = None
    if n_holdout > 0 and score["holdout"].l1 > score_start["holdout"].l1:
        ok = False
        reason = (
            f"holdout l1 {score['holdout'].l1:.4f} worse than nominal "
            f"{score_start['holdout'].l1:.4f}: the fit did not generalise"
        )
        logger.warning("calibrate rejected: %s", reason)

    if ok:
        params = raw.detach()
    else:
        params = torch.zeros_like(raw.detach())

    illumination_np = (
        _beam_amplitude(sim, params[0], params[1], params[2], params[3], params[4])
        .detach()[0, 0]
        .to("cpu")
        .numpy()
        .astype(np.float32)
    )
    aberration_np = (
        (
            sim.aberration(  # type: ignore[attr-defined]
                params[_COEFF_START : _COEFF_START + n_coeffs].view(1, n_coeffs)
            )[0]
            if n_coeffs > 0
            else torch.zeros((grid, grid), dtype=dtype, device=device)
        )
        .detach()
        .to("cpu")
        .numpy()
        .astype(np.float32)
    )

    nominal_w0 = float(sim.cfg.beam_w0_model_px)  # type: ignore[attr-defined]
    beam = {
        "width_scale_x": float(torch.exp(params[0])),
        "width_scale_y": float(torch.exp(params[1])),
        "w0_x_model_px": nominal_w0 * float(torch.exp(params[0])),
        "w0_y_model_px": nominal_w0 * float(torch.exp(params[1])),
        "angle_rad": float(params[2]),
        "center_x_px": float(params[3]),
        "center_y_px": float(params[4]),
    }
    camera = {
        "zoom": float(torch.exp(params[_LOG_ZOOM])),
        "shift_x_px": float(params[_SHIFT_X]),
        "shift_y_px": float(params[_SHIFT_Y]),
    }
    transfer = measurement.MeasurementTransfer(
        psf_sigma_px=float(params[measure_start]) if n_measure else 0.0,
        background_fraction=float(params[measure_start + 1]) if n_measure > 1 else 0.0,
    )
    zero_order = {"eta": nominal_eta} if cfg.fit_zero_order else {}
    if ok and n_eta:
        zero_order["eta"] = float(_eta_from_raw(params[zero_start]))
    return CalibrationResult(
        ok=ok,
        reason=reason,
        sim_config=session.sim_config,
        aberration=aberration_np,
        illumination=illumination_np,
        beam=beam,
        camera=camera,
        measurement=transfer.to_dict(),
        zero_order=zero_order,
        free=free_names,
        n_train=int(train.size),
        n_holdout=int(holdout.size),
        score=score,
        score_start=score_start,
        history=tuple(history),
        iterations=len(history),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibration(directory: str | Path, result: CalibrationResult) -> tuple[Path, Path]:
    """Write a calibration as ``calibration.json`` + ``calibration.npz``.

    The split mirrors a :mod:`ml.sim2f.bench` session's own on-disk contract:
    scalars and strings in JSON, arrays in one npz, because a 128x128 float map
    has no business being printed into a text file.

    Args:
        directory: Destination directory (created if missing).
        result: The calibration to save.

    Returns:
        ``(json_path, npz_path)``.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "calibration.json"
    npz_path = target / "calibration.npz"
    payload = result.to_dict()
    payload["sim_config"] = result.sim_config.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(npz_path, aberration=result.aberration, illumination=result.illumination)
    return json_path, npz_path


def load_calibration(directory: str | Path) -> CalibrationResult:
    """Read a calibration written by :func:`save_calibration`.

    Args:
        directory: Directory holding ``calibration.json`` + ``calibration.npz``.

    Returns:
        The :class:`CalibrationResult`.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    target = Path(directory)
    json_path = target / "calibration.json"
    npz_path = target / "calibration.npz"
    if not json_path.is_file():
        raise FileNotFoundError(f"no calibration.json under {target}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"no calibration.npz under {target}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    sim_config = _sim_config_from_bench(payload["sim_config"])
    score = {
        name: FrameScore(**values) for name, values in payload.get("score", {}).items()
    }
    score_start = {
        name: FrameScore(**values)
        for name, values in payload.get("score_start", {}).items()
    }
    return CalibrationResult(
        ok=bool(payload["ok"]),
        reason=payload.get("reason"),
        sim_config=sim_config,
        aberration=np.asarray(arrays["aberration"], dtype=np.float32),
        illumination=np.asarray(arrays["illumination"], dtype=np.float32),
        beam=dict(payload.get("beam", {})),
        camera=dict(payload.get("camera", {})),
        measurement=measurement.MeasurementTransfer.from_dict(
            payload.get("measurement")
        ).to_dict(),
        zero_order=dict(payload.get("zero_order", {})),
        free=tuple(payload.get("free", ())),
        n_train=int(payload.get("n_train", 0)),
        n_holdout=int(payload.get("n_holdout", 0)),
        score=score,
        score_start=score_start,
        history=tuple(payload.get("history", ())),
        iterations=int(payload.get("iterations", 0)),
        config=CalibrateConfig(**payload.get("config", {})),
    )


# ---------------------------------------------------------------------------
# The bench protocol, fitted state applied
# ---------------------------------------------------------------------------


class CalibratedBench:
    """A :class:`~ml.sim2f.optics.Sim2f` with a fitted bench state applied.

    :func:`calibrate_bench` estimates the entrance beam, the static aberration
    and the camera zoom/shift from measured frames.  The downstream stages --
    :func:`ml.sim2f.fit.refine` and :class:`ml.sim2f.finetune.PhaseRefiner` --
    only know the *bench protocol* (``grid`` / ``ccd_px`` / ``dtype`` /
    ``device`` / ``aberration`` / ``forward``), so this wrapper injects the
    fitted state into every forward pass and presents the result as one bench.

    Which *role* the object plays is the caller's decision: built from a
    calibration it is the model of the device; built from the ground-truth
    state of a synthetic bench (or from the calibration, offline) it is the
    device surrogate.  An honest trial never lets the two be the same object.

    Caller-supplied ``aberration`` and ``illumination`` compose with the fitted
    state -- the aberration adds, the illumination multiplies -- which is how a
    refit's residual estimate reaches the forward pass without touching the
    stored calibration.

    The fitted illumination is an **absolute** amplitude (anisotropic Gaussian,
    peak 1, :func:`_beam_amplitude`), not a correction: :meth:`forward` hands it
    to the bench, which uses it *instead of* its nominal beam.  An all-ones map
    is therefore not "no beam" but a full square aperture -- a different pupil
    and a different PSF, which no aberration fit can absorb.  Use
    :attr:`~ml.sim2f.optics.Sim2f.nominal_illumination` for the nominal case.

    Attributes:
        calibration: The fitted state this bench applies.
    """

    def __init__(self, sim: object, calibration: CalibrationResult) -> None:
        """Wrap ``sim`` with ``calibration``.

        Args:
            sim: The :class:`~ml.sim2f.optics.Sim2f` bench.
            calibration: The fitted state (arrays on the model grid, camera
                zoom/shift in camera pixels).

        Raises:
            ValueError: If the calibration's maps do not match the bench grid.
        """
        grid = int(sim.grid)  # type: ignore[attr-defined]
        for name, array in (
            ("aberration", calibration.aberration),
            ("illumination", calibration.illumination),
        ):
            if tuple(array.shape) != (grid, grid):
                raise ValueError(
                    f"calibration {name} is {tuple(array.shape)}, "
                    f"bench grid is {(grid, grid)}"
                )
        dtype = sim.dtype  # type: ignore[attr-defined]
        device = torch.device(sim.device)  # type: ignore[attr-defined]
        camera = calibration.camera
        self._sim = sim
        self.calibration = calibration
        self._aberration = torch.as_tensor(
            np.asarray(calibration.aberration), dtype=dtype, device=device
        )
        self._illumination = torch.as_tensor(
            np.asarray(calibration.illumination), dtype=dtype, device=device
        ).view(1, 1, grid, grid)
        self._shift_x = torch.as_tensor(
            float(camera.get("shift_x_px", 0.0)), dtype=dtype, device=device
        )
        self._shift_y = torch.as_tensor(
            float(camera.get("shift_y_px", 0.0)), dtype=dtype, device=device
        )
        self._log_zoom = torch.log(
            torch.as_tensor(float(camera.get("zoom", 1.0)), dtype=dtype, device=device)
        )
        self._measurement = measurement.MeasurementTransfer.from_dict(
            calibration.measurement
        )
        eta = calibration.zero_order.get("eta")
        self._zero_order_eta = (
            None
            if eta is None
            else torch.as_tensor(float(eta), dtype=dtype, device=device)
        )

    # -- the bench protocol ---------------------------------------------------
    @property
    def grid(self) -> int:
        """Model grid, pixels."""
        return int(self._sim.grid)  # type: ignore[attr-defined]

    @property
    def ccd_px(self) -> int:
        """Camera window, pixels."""
        return int(self._sim.ccd_px)  # type: ignore[attr-defined]

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype of the wrapped bench."""
        return self._sim.dtype  # type: ignore[attr-defined]

    @property
    def device(self) -> torch.device:
        """Compute device of the wrapped bench."""
        return torch.device(self._sim.device)  # type: ignore[attr-defined]

    @property
    def cfg(self) -> Sim2fConfig:
        """The wrapped bench's geometry (unchanged by the calibration)."""
        return self._sim.cfg  # type: ignore[attr-defined]

    def aberration(self, coeffs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Zernike map of a coefficient vector, on the wrapped bench's basis."""
        return self._sim.aberration(coeffs)  # type: ignore[attr-defined]

    def forward(
        self,
        phase: torch.Tensor,
        *,
        aberration: torch.Tensor | np.ndarray | None = None,
        illumination: torch.Tensor | np.ndarray | None = None,
        detector: bool = False,
    ) -> torch.Tensor:
        """Fitted-state forward: beam, base aberration, camera warp, transfer.

        Args:
            phase: ``(n, grid, grid)`` commanded radians.
            aberration: Optional *additional* aberration phase (e.g. the
                refit's residual estimate); added to the fitted static map.
            illumination: Optional *additional* amplitude; multiplies the
                fitted beam.
            detector: Passed through to the wrapped bench's detector chain.

        Returns:
            ``(n, 1, ccd_px, ccd_px)`` intensity, camera-warped.
        """
        if aberration is None:
            total_aberration = self._aberration
        else:
            total_aberration = self._aberration + aberration
        if illumination is None:
            total_illumination = self._illumination
        else:
            total_illumination = self._illumination * illumination
        frame = self._sim.forward(  # type: ignore[attr-defined]
            phase,
            aberration=total_aberration,
            illumination=total_illumination,
            detector=detector,
            zero_order_eta=self._zero_order_eta,
        )
        frame = _camera_affine(frame, self._shift_x, self._shift_y, self._log_zoom)
        return self._measurement.apply(frame)
