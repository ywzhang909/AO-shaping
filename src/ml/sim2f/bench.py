"""Sim-to-real measurement of a trained U-Net phase on the 2f bench.

One question, one protocol
==========================
``runs/sim2f_bench/README.md`` is the **pre-registered** protocol this module
implements, and it is the reference for every constant and every verdict string
below.  It asks one thing: once the published U-Net phase
(``runs/sim2f_fair/train/model_best.pt``) is downloaded to the real Santec panel
and measured on the real CCD, does ``net`` still lose to ``flat`` -- the result
the closed-loop report measured 7/7 families on the *twin*
(``report/sim2f_shaping/closed_loop_20261025/report.md`` §5, whose §9.1 says in
writing that it is a matched-model result, not sim-to-real)?

Per sample the protocol measures exactly four frames, in this order::

    A1(flat) -> B1(net) -> B2(net) -> A2(flat)

The two ``A`` readings are the same condition, so their spread **is** the noise
floor of that sample; the order is fixed (not randomised) so thermal drift and
LCOS response cancel in A->A and B->B instead of turning into unestimable
variance.  ``ncc_flat`` / ``ncc_net`` / the ``A`` drift are combined into the
three-valued verdict of :func:`verdict`, and Gate 0 (:func:`gate0`) is evaluated
before any verdict is reported.

Why the panel embedding is ``radius_px = 450`` (900 px), and why NOT 675
=======================================================================
``Sim2f.field`` (``src/ml/sim2f/optics.py:674``) pads the ``(B, 128, 128)``
pupil phase straight to ``fft_n = 512`` and transforms it -- **there is no panel
resampling anywhere in the twin**.  ``region_span_px = 900``
(``src/ml/sim2f/config.py`` geometry block) therefore only exists to set the
pupil pitch the transform is interpreted at: ``900 px * 8 um / 128 = 56.25 um``
= 7.03 panel pixels per model sample
(``Sim2fConfig.pupil_pitch_m``).  The twin's 128 samples **are** a 900 px square
of the panel, so the hardware side must place the same 128-grid phase into a
900 px square -- hence :data:`PUPIL_RADIUS_PX` = 450.0, because
``pupil_side(450) == 900`` (``src/ao_shaping/utils/slm/phase_embed.py:17``).

The tempting alternative is to use ``Sim2fConfig.pupil_fill = 0.75`` and embed a
``0.75 * 900 = 675`` px disc.  That is **wrong and silently ruins the session**:
``pupil_fill`` is the support of the Zernike basis / the U-Net mask -- the disc
the network parks its phase on in the ``(1, 0)`` phasor channel -- and is
documented as "a modelling disc, not an aperture"
(``src/ml/sim2f/config.py``, ``Sim2fConfig`` docstring and
``Sim2fConfig.pupil_radius_px``).  The twin's entrance is the pure Gaussian of
``beam_w0_px`` with **no hard aperture**, so there is no 675 px stop to match.
Embedding at 450 puts the phase at the right physical scale; embedding at 337.5
(radius) would shrink it to ``(675/900)^2 = 0.56 x`` of the phase slope the
twin modelled, i.e. a different optical element entirely.

WHERE that square goes is the second half of the same mapping and the operator's
call: :func:`build_panel_phase` takes ``center_xy`` (panel pixels, CLI
``--pupil-center-xy``).  ``None`` = the panel centre, which is only right if the
**beam** is centred on the panel; if it is not, the beam sees a cropped, off-axis
piece of the phase map and the far field is *wrong*, not shifted.  The record goes
into ``session.json`` under ``pupil_placement`` (with ``pupil_center_is_default``),
and each row carries a non-gating ``zero_order_offset_px`` diagnostic (measured
0-order centroid minus the twin's predicted one) so a mis-centred beam is visible
in the artefacts.

What the 128 -> 900 upsample convention costs (measured, not assumed)
=====================================================================
``compose_pupil_dof`` resamples with
:func:`ao_shaping.utils.image.resample.upsample_bilinear_to_shape`
(``scipy.ndimage.zoom(order=1)`` + ``fit_exact``), which is **endpoint aligned**:
input sample 0 lands on output pixel 0 and input 127 on pixel 899.  The twin's DFT
convention puts sample ``i`` at pupil position ``i * p`` with ``p`` =
``Sim2fConfig.pupil_pitch_m`` = 56.25 um, i.e. the 128 samples span ``127 * p`` =
7143.75 um -- a **pixel-centre** grid, where sample ``i`` sits at the centre of
its ``900/128 = 7.03125`` px block.  Endpoint alignment instead puts sample ``i``
at ``i * 899/127`` = ``i * 7.07874`` px, so the embedded map has an effective pitch
of **899/127 = 7.07874 px per grid sample**, not 900/128 = 7.03125:

    pitch ratio (899/127) / (900/128) = **1.006755**  ->  the phase map is **stretched
    by +0.6755 %** *about the region centre* (it is rescaled, not translated: sample 0
    and sample 127 are pulled 3.0156 px in *opposite* directions, and the region
    centre is the fixed point).

Measured on the actual embed (``tests/.../test_bench.py``
``TestResampleConvention``; both axes identical to 1e-14; the ramp's resampled
value *is* the input coordinate, so the displacement is read off, not fitted):

======================================  ===========================
max |displacement|, panel px            **3.0156** (at the two edges, opposite signs)
mean |displacement|, panel px           **1.5095**
displacement at the region centre       < 0.05 (the fixed point)
effective pitch, panel px per sample    **7.07874** (= 899/127, vs 7.03125)
pupil pitch ratio                       **1.006755** -> **+0.6755 % stretch**
far-field consequence                   **-0.671 % scale** (= 1/1.006755):
                                        **0.161 CCD px** on the corpus' finest
                                        feature (24 CCD px)
======================================  ===========================

**This is a scale error, not a translation.**  Displacing a *shaping* phase map
does not translate the focal pattern -- it changes the map's local gradient
(``-delta * grad(phi)``), which is why the "pupil shift = focal shift" shortcut is
wrong here.  A pupil **pitch** ratio of 1.006755 scales the far field by its
reciprocal, 0.993291, i.e. -0.671 %, and ``SynthConfig.min_feature_px`` = 24 CCD px
is the finest feature the corpus contains, so the residual is
``0.00671 * 24 = 0.16`` CCD px.

The pixel-centre convention would remove the 0.6755 % entirely, but
``compose_pupil_dof`` is the repository's standard pupil embed -- the GS and
transfer lines use the same function -- so **the standard embed is kept** and the
residual is recorded here and in README §1.1 instead of forking a second embed.
If the resampler ever changes, these numbers must be re-measured (the test pins
them).

Where the numbers live
======================
* :func:`measure_sample` -- the A1/B1/B2/A2 quadruple and its sample row.
* :func:`gate0` -- the four Gate-0 checks of README §4.
* :func:`verdict` -- the README §3 criterion, in its three-valued vocabulary.
* :func:`run_session` -- the whole session plus the README §7 artefacts.

Every metric route goes through :func:`ml.sim2f.loop._score`, which is
:func:`ml.sim2f.evaluate._row` -- the **same** code the published simulation
report used, on prepared frames (median subtracted, clipped at 0, see
:func:`ml.sim2f.losses.prepare_frame`).  That is what makes ``--backend sim``
reproduce ``runs/sim2f_fair/loop/a_no_noise/eval_loop.json`` and therefore
proves the measurement chain before any hardware is touched.

Backends
========
:class:`SimInstrument` drives the calibrated twin and is bit-for-bit the
construction of ``loop.run_loop``'s ``a_no_noise`` arm: same bench
(:meth:`Sim2fConfig.calibrated`), same per-sample state
(``np.random.default_rng([seed, index])`` + :func:`ml.sim2f.optics.draw_aberration`),
same :class:`ml.sim2f.loop._Measurer`.  It is the dry run of README §6 step 0.

:class:`HwInstrument` drives the real Santec SLM (memory mode only, never
``video_mode=1``) and the real CCD.  **All** device imports are inside its
methods, so importing this module stays SDK-free (``drivers/AGENTS.md``: an
``import`` may not load a native SDK).
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ml.sim2f._device import _resolve_device
from ml.sim2f.config import AberrationConfig, Sim2fConfig, TeacherConfig
from ml.sim2f.loop import _check_geometry, _finite, _load_network, _Measurer, _score
from ml.sim2f.model import network_input
from ml.sim2f.optics import Sim2f, draw_aberration, write_aberration
from ml.sim2f.losses import from_phasor

__all__ = [
    "CONDITIONS",
    "HwInstrument",
    "Instrument",
    "LOOP_COUNT",
    "LOOP_POPULATION",
    "LOOP_SEED",
    "MAX_EXPOSURE_ENV",
    "MIN_DELTA",
    "MIN_WINS",
    "NET_DOES_NOT_RECOVER",
    "NET_RECOVERS",
    "INSUFFICIENT_POWER",
    "NOISE_FACTOR",
    "PANEL_SHAPE",
    "PUPIL_RADIUS_PX",
    "SETTLE_S",
    "SimInstrument",
    "build_panel_phase",
    "default_exposure_ladder",
    "format_gate0",
    "format_rows",
    "gate0",
    "measure_sample",
    "panel_corner",
    "pupil_center",
    "recorded_grayscale",
    "run_session",
    "select_indices",
    "verdict",
]

#: Pupil radius, **panel pixels**, that makes the 128 model samples cover exactly
#: ``Sim2fConfig.region_span_px``: ``pupil_side(450) = 900 = region_span_px``.
#: Not ``pupil_fill``-derived -- see the module docstring.
PUPIL_RADIUS_PX = 450.0

#: SLM-200 panel ``(height, width)``, in panel pixels.  Only a fallback: a live
#: :class:`HwInstrument` asks the device (``panel_shape_from_slm``).
PANEL_SHAPE: tuple[int, int] = (1200, 1920)

#: Measurement order (README §2).  Fixed, never randomised.
CONDITIONS: tuple[str, ...] = ("a1", "b1", "b2", "a2")

#: Frames averaged per capture (README §6).  ``skip_first`` stays on: the first
#: frame after a write / ROI change can be stale.
N_SAMPLE = 3

#: Seconds to wait after a panel write before reading (README §2: ``0.3``).
SETTLE_S = 0.3

#: README §3 noise-floor factor: a sample whose ``|ncc_net - ncc_flat|`` is below
#: this multiple of its own ``A1``/``A2`` spread carries no information.
NOISE_FACTOR = 3.0

#: README §3 criterion: at least this many **counted** samples must favour ``net``.
MIN_WINS = 6

#: README §3 criterion: and the mean ``ncc_net - ncc_flat`` must exceed this.
MIN_DELTA = 0.05

#: README §3 verdict vocabulary.  These three strings are the whole output
#: alphabet; nothing else may be written as a conclusion.
NET_RECOVERS = "net_recovers"
NET_DOES_NOT_RECOVER = "net_does_not_recover"
INSUFFICIENT_POWER = "insufficient_power"

#: ``--samples loop`` reproduces the published §5 sample set: the first 24 of
#: ``torch.randperm(64, seed=7)``.  Both numbers are protocol constants, not
#: knobs (verified against ``runs/sim2f_fair/loop/a_no_noise/eval_loop.json``).
LOOP_SEED = 7
LOOP_POPULATION = 64
LOOP_COUNT = 24

#: Environment variable holding the camera protection ceiling, ms.  The session
#: **refuses to open** without it (README §8.3: the 1064 nm bench reaches near
#: saturation around 0.02 ms, so an unguarded exposure can burn the sensor).
MAX_EXPOSURE_ENV = "AO_DAHENG_MAX_EXPOSURE_MS"

#: Search box (panel/camera pixels, per side, from the anchor) within which the
#: 0-order is re-located in a non-flat frame.  It must be much larger than the
#: G0-2 tolerance (2 px) so real drift is measurable, and small enough that a
#: shaped speckle -- which can be the frame's global maximum, 49 px from the
#: 0-order on a healthy twin sample -- cannot hijack the reading.
ZERO_ORDER_SEARCH_PX = 16.0

#: Fraction of full scale the exposure ladder aims the flat-frame peak at.  The
#: middle of Gate-0's ``[0.2, 0.8]`` band, so a small mis-estimate still lands
#: inside it.
EXPOSURE_TARGET_FRACTION = 0.5

#: Ladder steps the default (geometric) ladder is built from: ``cap/64 .. cap``.
EXPOSURE_LADDER_STEPS = 8

#: Artefact names, README §7.
SESSION_NAME = "session.json"
ROWS_NAME = "rows.csv"
FRAMES_NAME = "frames.npz"
FETCH_NAME = "fetch.png"
VERDICT_NAME = "VERDICT.md"


class BenchmarkError(RuntimeError):
    """A session that must not run (configuration or bench precondition)."""


# ----------------------------------------------------------------------
# Phase -> panel
# ----------------------------------------------------------------------
def build_panel_phase(
    phase_rad: Tensor | np.ndarray,
    panel_shape: tuple[int, int] = PANEL_SHAPE,
    radius_px: float = PUPIL_RADIUS_PX,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """Place a ``(grid, grid)`` pupil phase into a full SLM panel, in radians.

    The single correct mapping of README §1.1::

        (128,128) radians --compose_pupil_dof(base=zeros(1200,1920), radius=450)-->
        1920x1200 radians, phase exactly 0 outside the 900x900 pupil square

    ``compose_pupil_dof`` does the panel work: it bilinearly resamples the DOF to
    ``pupil_side(radius_px)`` (900 here) and pastes that into a square of a zero
    base.  The base is zeros -- "flat phase" outside the pupil region, which is
    what the twin's pad-to-``fft_n`` models.

    ``center_xy`` is where that square goes, and it is **not** cosmetic: the twin
    puts the illumination centre on the 128-grid centre and that grid maps to the
    *region* centre, so the illuminated pupil -- not the panel -- must be centred
    on the phase map.  A phase pasted at the panel centre while the beam sits
    elsewhere is applied to the wrong pixels: the beam sees a cropped, off-axis
    piece of the map and the far field is **wrong**, not merely shifted.  ``None``
    assumes the beam is centred on the panel (the 2f bench's alignment target, and
    the value every dry run uses).

    Args:
        phase_rad: ``(grid, grid)`` raw unwrapped radians (a leading singleton
            batch channel is squeezed).  Torch or numpy; the values are used
            as-is (no wrap: the only ``mod 2*pi`` in the repository lives in the
            SLM driver).
        panel_shape: ``(height, width)`` in panel pixels.
        radius_px: Pupil radius in panel pixels; defaults to
            :data:`PUPIL_RADIUS_PX` (450 -> a 900 px square, matching
            ``Sim2fConfig.region_span_px``).
        center_xy: Centre of the illuminated pupil square, panel pixels
            ``(x, y)``; ``None`` = the panel centre ``(width // 2, height // 2)``.

    Returns:
        ``(height, width)`` float64 radians, zero outside the pupil square.

    Raises:
        ValueError: If ``phase_rad`` is not a square 2-D map, or the pupil
            square does not fit on the panel at ``center_xy``.
    """
    from ao_shaping.utils.slm.phase_embed import compose_pupil_dof, pupil_side

    array = phase_rad.detach().cpu().numpy() if torch.is_tensor(phase_rad) else phase_rad
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            "build_panel_phase needs a square (grid, grid) phase map, got shape "
            f"{array.shape}"
        )
    side = pupil_side(radius_px)
    height, width = (int(panel_shape[0]), int(panel_shape[1]))
    if side > min(height, width):
        raise ValueError(
            f"pupil square {side} px does not fit a {height}x{width} panel; "
            "check radius_px / panel_shape"
        )
    x0, y0 = panel_corner(panel_shape, radius_px, center_xy)
    if x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
        raise ValueError(
            f"pupil square {side} px at centre {center_xy} runs off the "
            f"{height}x{width} panel (corner ({x0}, {y0})); check "
            "--pupil-center-xy / panel_shape"
        )
    base = np.zeros((height, width), dtype=np.float64)
    return compose_pupil_dof(
        base, array, (height, width), float(radius_px), center_xy=center_xy
    )


def recorded_grayscale(
    phase_rad: Tensor | np.ndarray,
    *,
    slm: Any | None = None,
    panel_shape: tuple[int, int] = PANEL_SHAPE,
    radius_px: float = PUPIL_RADIUS_PX,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """The uint16 panel grayscale that one ``(grid, grid)`` command phase produces.

    ``SimInstrument.sent_grayscale`` and ``HwInstrument.sent_grayscale`` record the
    same quantity, so they share this single implementation: embed the pupil
    command onto the panel with :func:`build_panel_phase` -- the same call the
    write path makes -- then convert radians to grayscale through
    :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`.

    The ``slm`` argument is the entire difference between the two instruments, and
    it is why the helper takes the device instead of hard-coding either path.  With
    a live device the conversion delegates to ``create_phase_from_array``: wavefront
    correction, gamma LUT, base-phase overlay, panel shift and the
    wavelength-dependent ``2*pi = _max_gray`` scaling all apply, so the returned
    array is bit-for-bit what the panel physically received.  With ``slm=None`` the
    same function falls back to its pure-math conversion (fold to ``[0, 2*pi)``,
    scale to ``max_grayscale`` default 1023), which is all a panel-less twin can
    record -- the twin's ``_phase`` column then has a hardware session's shape and
    reading order without claiming the device's scaling.

    Args:
        phase_rad: ``(grid, grid)`` command phase, radians.
        slm: Live SLM whose driver pipeline performs the conversion; ``None``
            (twin, offline analysis, unit tests) uses the built-in pure math.
        panel_shape: Panel ``(height, width)`` in pixels; this module's
            :data:`PANEL_SHAPE` by default.
        radius_px: Illuminated pupil radius; :data:`PUPIL_RADIUS_PX` by default.
        center_xy: Pupil centre ``(x, y)`` in panel pixels; ``None`` = panel centre.

    Returns:
        ``(panel_h, panel_w)`` uint16 grayscale.  Outside the pupil square the
        embedded phase is 0 rad, so on the ``slm=None`` path those pixels are
        grayscale 0; a live device still applies its correction/LUT/overlay there.

    Raises:
        ValueError: Propagated from :func:`build_panel_phase` when the phase map is
            not square or the pupil square does not fit the panel.
    """
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    panel = build_panel_phase(
        phase_rad,
        panel_shape=panel_shape,
        radius_px=radius_px,
        center_xy=center_xy,
    )
    return phase_to_slm_grayscale(panel, slm=slm)


def pupil_center(
    panel_shape: tuple[int, int] = PANEL_SHAPE,
    center_xy: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """The pupil centre actually used: ``center_xy`` or the panel centre.

    Args:
        panel_shape: ``(height, width)`` in panel pixels.
        center_xy: Requested pupil centre ``(x, y)``, or ``None``.

    Returns:
        ``(x, y)`` in panel pixels.
    """
    height, width = (int(panel_shape[0]), int(panel_shape[1]))
    if center_xy is None:
        return width // 2, height // 2
    return int(center_xy[0]), int(center_xy[1])


def panel_corner(
    panel_shape: tuple[int, int] = PANEL_SHAPE,
    radius_px: float = PUPIL_RADIUS_PX,
    center_xy: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Top-left ``(x0, y0)`` of the pupil square, in panel pixels.

    Mirrors the placement rule of :func:`ao_shaping.utils.slm.phase_embed.embed_pupil_square`
    (``x0 = cx - side // 2``) so callers can reason about the embedding without
    importing the pure numpy helper.

    Args:
        panel_shape: ``(height, width)`` in panel pixels.
        radius_px: Pupil radius in panel pixels.
        center_xy: Pupil centre ``(x, y)``; ``None`` = panel centre.

    Returns:
        ``(x0, y0)``.
    """
    side = int(round(2.0 * float(radius_px)))
    cx, cy = pupil_center(panel_shape, center_xy)
    return cx - side // 2, cy - side // 2


def _phase_array(phase_rad: Tensor | np.ndarray) -> np.ndarray:
    """Normalise a command phase to a float32 ``(grid, grid)`` numpy array."""
    array = phase_rad.detach().cpu().numpy() if torch.is_tensor(phase_rad) else phase_rad
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"a command phase must be (grid, grid), got {array.shape}")
    return array


# ----------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------
class Instrument(Protocol):
    """A bench the session can push a phase into and read a frame from.

    Implementations: :class:`SimInstrument` (calibrated twin) and
    :class:`HwInstrument` (Santec SLM + CCD).

    ``twin`` is deliberately part of the contract even though it is not one of
    the three members README names: Gate-0's G0-4 compares the *measured* shaped
    frame with the **twin's prediction** of the same command, so a gate cannot be
    evaluated without it.  For :class:`HwInstrument` the prediction comes from
    the calibrated twin with **no** injected bench state (the real state is
    unknown); for :class:`SimInstrument` it is the same bench *with* its drawn
    state, so G0-4 is trivially 1.0 there -- the gate exists for hardware and the
    dry run only exercises its plumbing.
    """

    def capture(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """Push one ``(grid, grid)`` radian phase and return one camera frame."""
        ...

    def twin(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """Return the twin's prediction for the same command (Gate 0-4)."""
        ...

    def close(self) -> None:
        """Release the devices (idempotent)."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable device metadata for ``session.json``."""
        ...


class SimInstrument:
    """The calibrated twin as an instrument -- the dry run of README §6 step 0.

    Construction is identical to ``loop.run_loop``'s ``a_no_noise`` arm: the same
    bench (:meth:`Sim2fConfig.calibrated`), one drawn aberration state per sample
    (:func:`ml.sim2f.optics.draw_aberration` + :func:`~ml.sim2f.optics.write_aberration`)
    and one :class:`ml.sim2f.loop._Measurer` for the camera read.  Reusing the
    measurer (rather than calling ``bench.forward`` here) is what keeps this class
    numerically identical to the published loop instead of merely similar.

    No panel embedding happens on this path: the twin's 128 grid **is** the
    900 px pupil region (``field`` pads 128 -> 512 with no resampling), so the
    phase goes in exactly as ``Sim2f.forward`` expects it.  Embedding it into a
    panel here would double-apply the geometry the twin already models.

    Attributes:
        bench: The :class:`~ml.sim2f.optics.Sim2f` instance.
    """

    def __init__(
        self,
        *,
        bench: Sim2f | None = None,
        aberration: Tensor | None = None,
        illumination: Tensor | None = None,
        detector: bool = False,
        seed: int = 0,
        device: str | torch.device = "auto",
    ) -> None:
        """Store the bench and one sample's state.

        Args:
            bench: Bench to measure on; defaults to the calibrated twin.
            aberration: ``(1, grid, grid)`` radians, from ``write_aberration``.
            illumination: ``(1, 1, grid, grid)`` amplitude, from
                ``write_aberration``.
            detector: Route frames through the detector model.  Off by default:
                the published ``a_no_noise`` arm was measured without it, and a
                quantised frame would make ``A1``/``A2`` differ by noise instead
                of being the same reading.
            seed: Base noise seed (only used when ``detector`` is on).
            device: Device for a bench built here.
        """
        self.bench = (
            Sim2f(Sim2fConfig.calibrated(), device=str(_resolve_device(device)))
            if bench is None
            else bench
        )
        self.detector = bool(detector)
        #: Accepted and recorded, but **ignored**: this path never builds a panel
        #: (the twin's grid *is* the illuminated 900 px region), so there is no
        #: placement to correct.  The attribute exists so
        #: :func:`calibrate_pupil_center` can drive every instrument through one
        #: interface -- on the twin the sweep then comes out flat, which is the
        #: documented "not decisive" case rather than an error.
        self.pupil_center_xy: tuple[int, int] | None = None
        self._measurer = _Measurer(
            self.bench, aberration, illumination, detector=self.detector, seed=int(seed)
        )

    def capture(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """Read one frame from the twin.

        Args:
            phase_rad: ``(grid, grid)`` radians (a singleton batch axis is
                accepted).

        Returns:
            ``(ccd_px, ccd_px)`` float32 model intensity (counts when the
            detector is on).
        """
        phase = _phase_array(phase_rad)
        tensor = torch.as_tensor(
            phase, dtype=self.bench.dtype, device=self.bench.device
        )[None]
        with torch.no_grad():
            frame = self._measurer(tensor)
        return frame[0, 0].detach().cpu().numpy()

    def twin(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """The twin's prediction: this bench, this state, this command."""
        return self.capture(phase_rad)

    def sent_grayscale(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """The panel grayscale the *equivalent* hardware command would send.

        There is no panel on this path (see the class docstring), so the value is
        :func:`recorded_grayscale` of the same command -- the panel embedding plus
        the pure-math radians-to-grayscale conversion (``slm=None``, 2*pi = 1023).
        It is recorded so a twin session's ``_phase`` column has the same shape
        and meaning as a hardware session's, which is what lets one analysis read
        both.
        """
        return recorded_grayscale(phase_rad)

    def close(self) -> None:
        """Nothing to release: the twin owns no device."""

    def describe(self) -> dict[str, Any]:
        """Bench metadata, JSON-safe."""
        return {
            "backend": "sim",
            "bench": self.bench.cfg.describe(),
            "detector": self.detector,
            "phase_input": (
                "(grid, grid) pupil radians -> Sim2f.forward; the twin's grid IS "
                "the 900 px pupil region, so no panel embedding on this path"
            ),
        }


class HwInstrument:
    """Santec SLM (memory mode) + CCD as an instrument.

    Acquisition, exactly as README §1.1/§6/§8 require:

    1. open the SLM in **memory mode** (``video_mode=0``); DVI mode can hang the
       controller until a physical power cycle, so it is never tried;
    2. open the camera and settle the exposure: an explicit ``exposure_ms`` is
       used as given, otherwise an **exposure ladder** is run on the flat phase at
       full frame (before the window exists) and the step whose peak is closest to
       ``0.5 * full_scale`` is chosen -- see :meth:`_resolve_exposure`.  The
       chosen value is then written back and read back, because the measured
       session must use exactly the exposure the ladder selected;
    3. grab a **native full frame**, locate the 0-order by ``argmax`` (it is not
       the frame centre) and ``reset_window`` a ``ccd_px`` square there --
       re-locating once inside the window, because ``reset_window`` clamps the
       offset and its returned centre is not necessarily the spot;
    4. per capture: ``build_panel_phase`` -> ``display_phase`` (rotating memory
       slot + settle) -> ``get_numpy_image(n_sample=3, skip_first=True)``.

    Every device import lives inside :meth:`__init__`, so importing this module
    loads no SDK.  The exposure ceiling :data:`MAX_EXPOSURE_ENV` is checked before
    any device is touched, and no ladder step may exceed it.
    """

    def __init__(
        self,
        *,
        exposure_ms: float | None = None,
        exposure_ladder: Sequence[float] | None = None,
        panel_shape: tuple[int, int] | None = None,
        pupil_center_xy: tuple[int, int] | None = None,
        ccd_px: int = 256,
        settle_s: float = SETTLE_S,
        cam_type: str = "daheng",
        cam_id: int = 0,
        slm_number: int = 1,
        slm_wavelength_nm: int = 0,
        radius_px: float = PUPIL_RADIUS_PX,
        n_sample: int = N_SAMPLE,
        device: str | torch.device = "cpu",
    ) -> None:
        """Open the bench.

        Args:
            exposure_ms: Camera exposure, ms.  An explicit value **always wins**
                and skips the ladder.
            exposure_ladder: Exposures to probe, ms, used only when
                ``exposure_ms`` is ``None``.  ``None`` = the default geometric
                ladder :func:`default_exposure_ladder` (``cap/64 .. cap``).
            panel_shape: ``(height, width)``; ``None`` asks the device.
            pupil_center_xy: Centre of the **illuminated pupil** on the panel,
                panel pixels ``(x, y)``; ``None`` assumes the beam is centred on
                the panel.  This is where the 900x900 phase square is pasted, so
                a wrong value puts the network's phase on the wrong pixels (a
                wrong far field, not a shifted one).
            ccd_px: Camera window side, camera pixels.
            settle_s: Wait after each panel write.
            cam_type: Camera backend name for ``create_camera``.
            cam_id: Camera index.
            slm_number: Santec SLM number.
            slm_wavelength_nm: ``0`` = use the wavelength programmed on the device.
            radius_px: Pupil radius in panel pixels (:data:`PUPIL_RADIUS_PX`).
            n_sample: Frames averaged per capture.
            device: Device for the twin used by :meth:`twin`.

        Raises:
            BenchmarkError: If the exposure ceiling is unset/too small, if the
                explicit exposure or any ladder step is invalid or above the
                ceiling, or if any device fails to open (the message names which).
        """
        cap = _exposure_cap_ms()
        exposure: float | None = None
        if exposure_ms is not None:
            exposure = float(exposure_ms)
            if not math.isfinite(exposure) or exposure <= 0.0:
                raise BenchmarkError(
                    f"exposure must be finite and > 0 ms, got {exposure!r}"
                )
            if exposure > cap:
                raise BenchmarkError(
                    f"requested exposure {exposure:g} ms exceeds {MAX_EXPOSURE_ENV}"
                    f"={cap:g} ms"
                )
            ladder = (exposure,)
        else:
            ladder = (
                tuple(float(value) for value in exposure_ladder)
                if exposure_ladder is not None
                else default_exposure_ladder(cap)
            )
            _validate_ladder(ladder, cap)

        # Device imports stay function-local: importing this module must not load
        # an SDK (drivers/AGENTS.md).  ``Santec``'s real signature is
        # ``(slm_number=1, use_120hz=False, wavelength=None,
        # video_mode=VideoMode.Memory, shift_x=None, shift_y=None,
        # correction_csv_path=None)`` -- verified against
        # ``drivers/slm/santec/driver.py``; direct construction with these three
        # keywords is what the bench tools do (``tools/slm/panel/phase_response.py``
        # and most ``scripts/slm/*``), and ``video_mode=0`` IS
        # ``VideoMode.Memory``.  ``Santec.from_params`` (the shaping runners' path
        # via ``drivers/slm/factory.open_slm``) is equivalent but needs a params
        # carrier; it is not used here because there is no carrier to reuse.
        from ao_shaping.drivers.ccd.common import create_camera
        from ao_shaping.drivers.slm.santec import Santec
        from ao_shaping.utils.slm.phase_embed import panel_shape_from_slm
        from ao_shaping.utils.slm.phase_display import display_phase

        self._display_phase = display_phase
        self._exposure_ms = exposure if exposure is not None else float(ladder[0])
        self._exposure_ladder = ladder
        self._exposure_source = "explicit" if exposure is not None else "ladder"
        self._exposure_trace: dict[str, Any] = {
            "source": self._exposure_source,
            "requested_ms": list(ladder),
            "applied_ms": [],
            "peak_counts": [],
            "peak_fraction": [],
            "full_scale": None,
            "target_fraction": EXPOSURE_TARGET_FRACTION,
            "chosen_ms": None,
            "chosen_peak_fraction": None,
        }
        if exposure is not None:
            # An explicit exposure is a decision already taken: record it, and leave
            # ``applied_ms`` empty so the trace cannot look like a ladder that ran.
            self._exposure_trace["chosen_ms"] = float(exposure)
        self._max_exposure_ms = cap
        self._ccd_px = int(ccd_px)
        self._settle_s = float(settle_s)
        self._radius_px = float(radius_px)
        self._n_sample = int(n_sample)
        self._cam_type = str(cam_type)
        self._cam_id = int(cam_id)
        self._closed = False
        self._full_scale = 255.0
        self._window: tuple[int, int] | None = None
        self._center: tuple[int, int] | None = None
        self._twin_bench: Sim2f | None = None
        self._twin_device = device
        self._pupil_center_xy = (
            None
            if pupil_center_xy is None
            else (int(pupil_center_xy[0]), int(pupil_center_xy[1]))
        )

        self._slm = Santec(
            slm_number=int(slm_number),
            wavelength=None if int(slm_wavelength_nm) == 0 else int(slm_wavelength_nm),
            video_mode=0,
        )
        try:
            self._slm.open()
            panel_res = getattr(self._slm, "Panel_Res", None)
            logger.info(
                "SLM opened (memory mode): serial={} panel={}",
                getattr(self._slm, "_serial_number", None),
                None if panel_res is None else tuple(panel_res),
            )
            self._panel_shape = (
                tuple(int(v) for v in panel_shape)
                if panel_shape is not None
                else panel_shape_from_slm(self._slm)
            )
            self._cam = create_camera(
                self._cam_type,
                cam_id=self._cam_id,
                exposure_time_ms=self._exposure_ms,
            )
            self._cam.open()
            self._cam.reset_exposure_time(self._exposure_ms)
        except Exception:
            self.close()
            raise
        try:
            if self._exposure_source == "ladder":
                self._resolve_exposure()
            self._locate_zero_order()
        except Exception:
            self.close()
            raise

    # -- exposure --------------------------------------------------------
    def _resolve_exposure(self) -> None:
        """Probe the exposure ladder on the flat phase and settle on one step.

        Runs on the **flat** phase at **full frame**, i.e. before
        :meth:`_locate_zero_order` windows the camera, so the peak it measures is
        the same light the session's own A frames will see.  The step whose flat
        peak is closest to ``target * full_scale`` is chosen; the choice is *not*
        filtered to the Gate-0 band -- an out-of-band best is reported and G0-1
        then fails with the trace attached, which is the honest outcome (silently
        substituting an in-band exposure would hide a light-level problem).

        Uses the driver base class's
        :meth:`ao_shaping.drivers.ccd.base.BaseCamera._acquire_exposure_ladder`,
        which discards the first frame of every step (a stale frame after an
        exposure change), reads back the **applied** exposure (the backend clamps
        out-of-range values) and restores the entry exposure.  The chosen value is
        then written and read back explicitly, so the camera is left at exactly
        the exposure the session uses.
        """
        from ao_shaping.drivers.ccd import common
        from ao_shaping.drivers.ccd.base import BaseCamera

        flat_panel = build_panel_phase(
            np.zeros((Sim2fConfig.calibrated().grid,) * 2, dtype=np.float32),
            self._panel_shape,
            self._radius_px,
            self._pupil_center_xy,
        )
        self._display_phase(self._slm, flat_panel, self._settle_s)
        # The acquisition lives in the camera base class -- use *its* implementation
        # rather than re-writing it here: it is the one place that knows each step
        # must drop its first frame (a stale frame after an exposure change), that
        # the backend may clamp the value (so the *applied* exposure is read back)
        # and that the entry exposure is restored in a finally block.  It only
        # touches the ABC surface (``reset_exposure_time`` / ``get_numpy_image``),
        # per its own docstring, so it is called on whatever camera object we hold.
        frames, applied, original = BaseCamera._acquire_exposure_ladder(
            self._cam, tuple(self._exposure_ladder), 1, restore_exposure=True
        )
        self._full_scale = float(common.full_scale(frames[0]))
        peaks = [float(np.max(frame)) for frame in frames]
        fractions = [peak / self._full_scale for peak in peaks]
        target = EXPOSURE_TARGET_FRACTION
        best = min(
            range(len(applied)),
            key=lambda index: abs(fractions[index] - target),
        )
        chosen = float(applied[best])
        common.set_camera_exposure_ms(self._cam, chosen)
        readback = float(common.get_camera_exposure_ms(self._cam))
        if abs(readback - chosen) > 1e-9:
            logger.warning(
                "camera reports {:.6f} ms after setting {:.6f} ms; the session will "
                "use the reported value",
                readback,
                chosen,
            )
        self._exposure_ms = readback
        self._exposure_trace.update(
            {
                "applied_ms": [float(value) for value in applied],
                "peak_counts": peaks,
                "peak_fraction": fractions,
                "full_scale": self._full_scale,
                "chosen_ms": self._exposure_ms,
                "chosen_peak_fraction": fractions[best],
                "restored_entry_ms": None if original is None else float(original),
            }
        )
        logger.info(
            "exposure ladder: {} step(s) -> {:.4f} ms (peak {:.1f}/{:.0f} = {:.1%} of "
            "full scale; target {:.0%})",
            len(applied),
            self._exposure_ms,
            peaks[best],
            self._full_scale,
            fractions[best],
            target,
        )

    @property
    def exposure_ms(self) -> float:
        """The exposure the session actually uses (explicit or ladder-chosen)."""
        return float(self._exposure_ms)

    @property
    def exposure_trace(self) -> dict[str, Any]:
        """The recorded exposure decision (JSON-safe)."""
        return dict(self._exposure_trace)

    # -- acquisition -----------------------------------------------------
    def _locate_zero_order(self) -> None:
        """Centre a ``ccd_px`` window on the 0-order, located by ``argmax``.

        The 0-order is the frame's global maximum in a 2f bench and is **not** at
        the geometric centre (measured 1441-1443, 705-706 against a centre of
        1344, 760), so geometry is never assumed.  ``reset_window`` clamps the
        offset and returns the window it really applied, so the spot is
        re-located inside the window and one corrective move is made when the
        peak is more than 4 px off-centre.
        """
        full = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        if np.issubdtype(full.dtype, np.integer):
            self._full_scale = float(np.iinfo(full.dtype).max)
        row, col = np.unravel_index(int(np.argmax(full)), full.shape)
        size = (self._ccd_px, self._ccd_px)
        window, center = self._cam.reset_window((int(col), int(row)), size)
        probe = np.asarray(self._cam.get_numpy_image(1, skip_first=True))
        peak_row, peak_col = np.unravel_index(int(np.argmax(probe)), probe.shape)
        half_r, half_c = probe.shape[0] // 2, probe.shape[1] // 2
        if abs(peak_row - half_r) > 4 or abs(peak_col - half_c) > 4:
            corrected = (
                int(center[0]) + (int(peak_col) - half_c),
                int(center[1]) + (int(peak_row) - half_r),
            )
            logger.warning(
                "0-order sits {}px off the window centre after reset_window; "
                "re-centring on {}",
                max(abs(peak_row - half_r), abs(peak_col - half_c)),
                corrected,
            )
            window, center = self._cam.reset_window(corrected, size)
        self._window = (int(window[0]), int(window[1]))
        self._center = (int(center[0]), int(center[1]))
        logger.info(
            "CCD window {}x{} centred on the 0-order at {} (full-frame peak was "
            "{},{}, full scale {})",
            self._window[0],
            self._window[1],
            self._center,
            row,
            col,
            self._full_scale,
        )

    def capture(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """Write the phase, settle, and read one averaged frame.

        Args:
            phase_rad: ``(grid, grid)`` radians.

        Returns:
            ``(ccd_px, ccd_px)`` float frame in detector counts.

        Raises:
            ValueError: If the frame is not the requested window -- a wrong ROI
                would silently invalidate every metric.
        """
        panel = build_panel_phase(
            phase_rad, self._panel_shape, self._radius_px, self._pupil_center_xy
        )
        self._display_phase(self._slm, panel, self._settle_s)
        frame = np.asarray(
            self._cam.get_numpy_image(self._n_sample, skip_first=True),
            dtype=np.float32,
        )
        expected = (self._ccd_px, self._ccd_px)
        if frame.shape != expected:
            raise ValueError(
                f"camera returned {frame.shape}, expected {expected}; the ROI changed "
                "mid-session"
            )
        return frame

    def twin(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """The calibrated twin's prediction of the same command.

        No bench state is injected: on hardware the true aberration is unknown,
        and G0-4 is a *loose absolute* geometry gate (>= 0.5 ncc) whose job is to
        catch a misplaced or wrongly scaled phase, not to judge twin accuracy.
        """
        if self._twin_bench is None:
            self._twin_bench = Sim2f(Sim2fConfig.calibrated(), device=str(self._twin_device))
        phase = _phase_array(phase_rad)
        tensor = torch.as_tensor(
            phase, dtype=self._twin_bench.dtype, device=self._twin_bench.device
        )[None]
        with torch.no_grad():
            frame = self._twin_bench.forward(tensor, detector=False)
        return frame[0, 0].detach().cpu().numpy()

    # -- pupil centre (auto-calibration) ---------------------------------
    @property
    def pupil_center_xy(self) -> tuple[int, int]:
        """Centre of the **illuminated pupil** currently used for embedding."""
        return pupil_center(self._panel_shape, self._pupil_center_xy)

    @pupil_center_xy.setter
    def pupil_center_xy(self, value: tuple[int, int]) -> None:
        """Move the 900 px phase square, refusing a placement that runs off.

        The placement is validated with the same rule :func:`build_panel_phase`
        applies, so a candidate from :func:`calibrate_pupil_center` cannot leave
        the instrument in a state whose next write raises.

        Args:
            value: New centre ``(x, y)``, panel pixels.

        Raises:
            ValueError: If the pupil square would not fit at ``value``.
        """
        candidate = (int(value[0]), int(value[1]))
        side = int(round(2.0 * self._radius_px))
        x0, y0 = panel_corner(self._panel_shape, self._radius_px, candidate)
        height, width = int(self._panel_shape[0]), int(self._panel_shape[1])
        if x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
            raise ValueError(
                f"pupil square {side} px at centre {candidate} runs off the "
                f"{height}x{width} panel (corner ({x0}, {y0}))"
            )
        self._pupil_center_xy = candidate

    def sent_grayscale(self, phase_rad: Tensor | np.ndarray) -> np.ndarray:
        """The display-ready uint16 panel this instrument's write path sends.

        Read this as "what the panel got", and note *why* it is exact rather than
        approximate.  :meth:`capture` calls
        :func:`ao_shaping.utils.slm.phase_display.display_phase`, whose whole body
        is ``slm.display_phase(phase_rad, memory_number=slot)``; the driver
        (``drivers/slm/santec/driver.py``) turns that into
        ``create_phase_from_array`` -> ``display_data``.  Everything the panel
        applies -- the 2*pi = device ``_max_gray`` scaling, the wavefront
        correction map, the gamma LUT, the grayscale base-phase overlay, the
        ``(shift_x, shift_y)`` translation and the final ``mod max_gray`` -- lives
        inside ``create_phase_from_array``.  ``display_data`` adds exactly one
        thing on top: a hard clamp to ``MAX_GRAYSCALE_VALUE`` (1023), which is a
        no-op because ``create_phase_from_array`` already returns
        ``mod _max_gray <= 1023``.  Its other jobs (slot rotation, the pixel-flip
        wait) do not change the numbers.

        So ``phase_to_slm_grayscale(panel, slm=self._slm)`` -- the call below, with
        the live device -- reproduces the written array bit-for-bit.  The
        alternative, ``get_displayed_phase()``, returns the *cached copy of the very
        same array* for the slot just written (its docstring says a slot with no
        cache "cannot be read back from the device"), so it would cost an SDK
        round-trip per row and tell us nothing new.

        Args:
            phase_rad: ``(grid, grid)`` command phase, radians.

        Returns:
            ``(panel_h, panel_w)`` uint16 grayscale.
        """
        return recorded_grayscale(
            phase_rad,
            slm=self._slm,
            panel_shape=self._panel_shape,
            radius_px=self._radius_px,
            center_xy=self._pupil_center_xy,
        )

    def close(self) -> None:
        """Close the camera and the SLM, independently and idempotently."""
        if self._closed:
            return
        self._closed = True
        for name in ("_cam", "_slm"):
            device = getattr(self, name, None)
            if device is None:
                continue
            try:
                device.close()
            except Exception as exc:  # a close failure must not mask the result
                logger.warning("closing {} failed: {}", name, exc)

    # -- metadata --------------------------------------------------------
    @property
    def full_scale(self) -> float:
        """Detector saturation value of the native frame (255 for 8-bit backends)."""
        return self._full_scale

    def describe(self) -> dict[str, Any]:
        """Device metadata, JSON-safe."""
        slm = getattr(self, "_slm", None)
        panel = getattr(slm, "Panel_Res", None)
        return {
            "backend": "hw",
            "slm": {
                "number": int(getattr(slm, "slm_number", -1)),
                "serial": getattr(slm, "_serial_number", None),
                "panel_res": None if panel is None else [int(v) for v in panel],
                "wavelength_nm": getattr(slm, "wavelength", None),
                "video_mode": int(getattr(slm, "video_mode", 0)),
            },
            "camera": {
                "type": self._cam_type,
                "id": self._cam_id,
                "exposure_ms": self._exposure_ms,
                "window": None if self._window is None else list(self._window),
                "center_xy": None if self._center is None else list(self._center),
                "n_sample": self._n_sample,
                "full_scale": self._full_scale,
            },
            "exposure": self.exposure_trace,
            "panel_shape": [int(v) for v in self._panel_shape],
            "panel_center_xy": [int(self._panel_shape[1] // 2), int(self._panel_shape[0] // 2)],
            "pupil_center_xy": list(
                pupil_center(self._panel_shape, self._pupil_center_xy)
            ),
            "pupil_center_is_default": self._pupil_center_xy is None,
            "pupil_corner_xy": list(
                panel_corner(self._panel_shape, self._radius_px, self._pupil_center_xy)
            ),
            "pupil_radius_px": self._radius_px,
            "pupil_side_px": int(round(2.0 * self._radius_px)),
            "ccd_px": self._ccd_px,
            "settle_s": self._settle_s,
            "max_exposure_ms": self._max_exposure_ms,
            "max_exposure_env": MAX_EXPOSURE_ENV,
        }


def default_exposure_ladder(cap_ms: float) -> tuple[float, ...]:
    """The geometric exposure ladder used when none is given: ``cap/64 .. cap``.

    Eight geometric steps, so the ratio is ``64 ** (1 / 7)`` = 1.812 and the last
    step is exactly the ceiling.  A ladder rather than a fixed number because the
    bench's light level moves between sessions (the repo has measured the near
    saturation exposure anywhere in the ``0.01-0.05 ms`` decade); guessing it
    wastes a session at G0-1, measuring it costs eight frames.

    Args:
        cap_ms: The protection ceiling (:data:`MAX_EXPOSURE_ENV`), ms.

    Returns:
        ``EXPOSURE_LADDER_STEPS`` exposures, ascending, the last equal to
        ``cap_ms``.

    Raises:
        BenchmarkError: If ``cap_ms`` is not a positive finite number.
    """
    cap = float(cap_ms)
    if not math.isfinite(cap) or cap <= 0.0:
        raise BenchmarkError(f"exposure ceiling must be > 0, got {cap_ms!r}")
    ratio = (64.0) ** (1.0 / (EXPOSURE_LADDER_STEPS - 1))
    steps = [cap / 64.0 * ratio**step for step in range(EXPOSURE_LADDER_STEPS - 1)]
    # The top step is the ceiling itself, exactly (the geometric product lands on
    # 0.8 - 1 ulp otherwise, and "the last step is the cap" is worth being literal).
    steps.append(cap)
    return tuple(steps)


def _validate_ladder(ladder: Sequence[float], cap_ms: float) -> None:
    """Require a usable ladder: at least two finite positive steps under the cap.

    Args:
        ladder: Candidate exposures, ms.
        cap_ms: The protection ceiling, ms.

    Raises:
        BenchmarkError: If the ladder is too short, not finite/positive, or one of
            its steps is above the ceiling (the ceiling is a sensor-protection
            limit, so probing past it is refused rather than clamped).
    """
    if len(ladder) < 2:
        raise BenchmarkError(
            "an exposure ladder needs at least two steps, got "
            f"{list(ladder)!r}; pass --exposure-ms for a single fixed exposure"
        )
    for index, value in enumerate(ladder):
        step = float(value)
        if not math.isfinite(step) or step <= 0.0:
            raise BenchmarkError(
                f"ladder step {index} must be finite and > 0 ms, got {value!r}"
            )
        if step > float(cap_ms):
            raise BenchmarkError(
                f"ladder step {index} ({step:g} ms) exceeds {MAX_EXPOSURE_ENV}"
                f"={float(cap_ms):g} ms"
            )


def _exposure_cap_ms() -> float:
    """Read the camera protection ceiling from the environment.

    Returns:
        The ceiling in ms.

    Raises:
        BenchmarkError: If the variable is unset, unparsable or ``<= 0``.  README
            §8.3 makes this a precondition of opening the bench at all, so the
            missing value is an error rather than a default.
    """
    # The literal is spelled out (not ``os.environ.get(MAX_EXPOSURE_ENV)``) on
    # purpose: the environment census in ``tests/ao_shaping/test_conventions.py``
    # is AST-based and only sees literals, and a knob it cannot see is a knob
    # nobody can find.  The name is documented in AGENTS.md §Environment Variables.
    raw = os.environ.get("AO_DAHENG_MAX_EXPOSURE_MS")
    if raw is None or not str(raw).strip():
        raise BenchmarkError(
            f"{MAX_EXPOSURE_ENV} is not set; set it to the session's exposure "
            "ceiling (ms) before running against hardware"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{MAX_EXPOSURE_ENV}={raw!r} is not a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkError(f"{MAX_EXPOSURE_ENV}={raw!r} must be finite and > 0")
    return value


# ----------------------------------------------------------------------
# One sample: A1 -> B1 -> B2 -> A2
# ----------------------------------------------------------------------
def measure_sample(
    instrument: Instrument,
    *,
    net_phase: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    support: Tensor | np.ndarray,
    teacher: TeacherConfig,
    index: int = 0,
    kind: str = "",
    noise_factor: float = NOISE_FACTOR,
) -> dict[str, Any]:
    """Run the A1/B1/B2/A2 quadruple on one sample and score it.

    Scoring is :func:`ml.sim2f.loop._score` (== ``evaluate._row``), i.e. the
    teacher's ``L_shape + w_ee * (1 - EE)`` plus the camera-space metrics on
    **prepared** frames -- the same code the published simulation table used.

    Args:
        instrument: Bench to measure on.
        net_phase: ``(grid, grid)`` network phase, radians.
        target: ``(1, 1, ccd_px, ccd_px)`` requested distribution.
        support: ``(1, 1, ccd_px, ccd_px)`` analytic support.
        teacher: Supplies ``w_ee`` and ``metric``.
        index: Cache index, for the row and the frame keys.
        kind: Target family name, for the row.
        noise_factor: Multiplier on the ``A1``/``A2`` spread that defines
            "indistinguishable" (README §3).

    Returns:
        ``{"index", "kind", "frames": {a1,b1,b2,a2}, "conditions": {a1,...},
        "ncc_flat", "ncc_net", "ncc_delta", "ncc_a_drift", "ncc_b_drift",
        "objective_flat", "objective_net", "objective_delta", "l1_flat",
        "l1_net", "ee_flat", "ee_net", "zero_order_offset_px",
        "indistinguishable"}``.
    """
    net = _phase_array(net_phase)
    flat = np.zeros_like(net)
    commands = {"a1": flat, "b1": net, "b2": net, "a2": flat}

    reference = torch.as_tensor(target, dtype=torch.float32)
    area = torch.as_tensor(support, dtype=torch.float32)
    device = reference.device

    frames: dict[str, np.ndarray] = {}
    conditions: dict[str, dict[str, float]] = {}
    for name in CONDITIONS:
        frame = np.asarray(instrument.capture(commands[name]), dtype=np.float32)
        frames[name] = frame
        tensor = torch.from_numpy(frame)[None, None].to(device=device)
        conditions[name] = _score(tensor, reference, area, teacher=teacher)

    def _mean(key: str, left: str, right: str) -> float:
        return 0.5 * (conditions[left][key] + conditions[right][key])

    ncc_flat = _mean("ncc", "a1", "a2")
    ncc_net = _mean("ncc", "b1", "b2")
    drift = abs(conditions["a1"]["ncc"] - conditions["a2"]["ncc"])
    row: dict[str, Any] = {
        "index": int(index),
        "kind": str(kind),
        "frames": frames,
        "conditions": conditions,
        "ncc_flat": float(ncc_flat),
        "ncc_net": float(ncc_net),
        "ncc_delta": float(ncc_net - ncc_flat),
        "ncc_a_drift": float(drift),
        # Diagnostic only -- the protocol's noise floor is the A drift
        # (README §3); the B drift is reported so a flaky panel write is visible.
        "ncc_b_drift": float(abs(conditions["b1"]["ncc"] - conditions["b2"]["ncc"])),
        "objective_flat": float(_mean("objective", "a1", "a2")),
        "objective_net": float(_mean("objective", "b1", "b2")),
        "objective_delta": float(
            _mean("objective", "b1", "b2") - _mean("objective", "a1", "a2")
        ),
        "l1_flat": float(_mean("l1", "a1", "a2")),
        "l1_net": float(_mean("l1", "b1", "b2")),
        "ee_flat": float(_mean("ee", "a1", "a2")),
        "ee_net": float(_mean("ee", "b1", "b2")),
        # Diagnostic, never a criterion (README §4 amendment): where the bench's
        # beam actually is versus where the twin's own flat prediction puts the
        # 0-order.  ``AberrationState.tilt_px`` is a *model* tilt, not the bench's,
        # so this reads ~0 on the sim backend and the real beam/twin
        # misalignment on hardware -- the operator reads it, nothing gates on it.
        "zero_order_offset_px": _zero_order_offset(instrument, frames["a1"], flat),
    }
    row["indistinguishable"] = _indistinguishable(row, noise_factor)
    return row


def _zero_order_offset(
    instrument: Instrument, measured_flat: np.ndarray, flat_phase: np.ndarray
) -> float:
    """Distance between the measured and the twin-predicted 0-order, pixels.

    Both centres are the ``argmax`` + 3x3 weighted centroid of a **flat** frame
    (the only condition where the 0-order is the frame maximum), so this is a
    like-for-like comparison of where the light lands.

    Args:
        instrument: Supplies the twin's prediction of the same flat command.
        measured_flat: ``(ccd_px, ccd_px)`` measured flat frame (A1).
        flat_phase: The zero phase used for A1/A2.

    Returns:
        The offset in camera pixels; ``nan`` when the prediction is unusable
        (a diagnostic that cannot be taken must not look like a good one).
    """
    try:
        predicted = np.asarray(instrument.twin(flat_phase), dtype=np.float32)
    except Exception as exc:  # a diagnostic must never fail a measurement
        logger.debug("twin prediction for the 0-order offset failed: {}", exc)
        return float("nan")
    if predicted.shape != np.asarray(measured_flat).shape or not np.any(predicted):
        return float("nan")
    measured = _zero_order_centroid(measured_flat)
    reference = _zero_order_centroid(predicted)
    return float(math.hypot(measured[0] - reference[0], measured[1] - reference[1]))


def _indistinguishable(row: dict[str, Any], noise_factor: float) -> bool:
    """Whether a sample's ``net`` vs ``flat`` difference is inside its noise floor.

    README §3: ``|ncc_net - ncc_flat| < noise_factor * |ncc_A1 - ncc_A2|``.
    Such a sample carries no information and is counted in neither the numerator
    nor the denominator of the win count.
    """
    return bool(abs(float(row["ncc_net"]) - float(row["ncc_flat"])) < float(
        noise_factor
    ) * abs(float(row["ncc_a_drift"])))


# ----------------------------------------------------------------------
# Gate 0
# ----------------------------------------------------------------------
def _zero_order_centroid(
    frame: np.ndarray,
    anchor: tuple[int, int] | None = None,
    search_px: float = ZERO_ORDER_SEARCH_PX,
) -> tuple[float, float]:
    """``argmax`` + 3x3 weighted centroid of the 0-order, in ``(row, col)``.

    Args:
        frame: 2-D frame.
        anchor: ``(row, col)`` of the 0-order.  ``None`` searches the whole frame
            (valid only on a flat frame, where the 0-order *is* the global max).
        search_px: Half-side of the search box around ``anchor``.

    Returns:
        The 0-order position in pixels.  Falls back to the peak pixel when the
        3x3 window carries no positive weight.
    """
    data = np.asarray(frame, dtype=np.float64)
    if anchor is None:
        row, col = np.unravel_index(int(np.argmax(data)), data.shape)
    else:
        radius = max(1, int(round(float(search_px))))
        r0 = max(0, int(anchor[0]) - radius)
        r1 = min(data.shape[0], int(anchor[0]) + radius + 1)
        c0 = max(0, int(anchor[1]) - radius)
        c1 = min(data.shape[1], int(anchor[1]) + radius + 1)
        found = np.unravel_index(int(np.argmax(data[r0:r1, c0:c1])), (r1 - r0, c1 - c0))
        row, col = r0 + int(found[0]), c0 + int(found[1])
    r0, r1 = max(0, row - 1), min(data.shape[0], row + 2)
    c0, c1 = max(0, col - 1), min(data.shape[1], col + 2)
    window = np.clip(data[r0:r1, c0:c1], 0.0, None)
    total = float(window.sum())
    if total <= 0.0:
        return float(row), float(col)
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    cols = np.arange(c0, c1, dtype=np.float64)[None, :]
    return float((rows * window).sum() / total), float((cols * window).sum() / total)


def _l2(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius 2-norm of ``left - right`` (``||.||_2`` of README §4 G0-3)."""
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.sum(difference * difference)))


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    """Mean-subtracted correlation of two frames, finite by construction."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator <= 1e-12:
        return 0.0
    return float((a * b).sum() / denominator)


def gate0(
    rows: list[dict[str, Any]],
    frames: dict[int, dict[str, np.ndarray]],
    *,
    full_scale: float | None = None,
    twin_frames: dict[int, np.ndarray] | None = None,
    exposure_ladder: dict[str, Any] | None = None,
    exposure_band: tuple[float, float] = (0.2, 0.8),
    saturation_fraction: float = 1e-3,
    centroid_tolerance_px: float = 2.0,
    phase_factor: float = 3.0,
    geometry_ncc: float = 0.5,
) -> dict[str, Any]:
    """The four Gate-0 checks of README §4, with their measured values.

    Failing any check means the session's main criterion must not be reported.
    The four checks and the constants are the protocol's; nothing is added.

    How each one is measured (the thresholds are untouched):

    * G0-1 -- the flat (A1/A2) peak as a fraction of ``full_scale``, and the
      fraction of pixels within 0.3% of full scale (a saturated pixel survives
      the 3-frame average the camera returns).
    * G0-2 -- ``argmax`` + 3x3 weighted centroid of the 0-order, compared
      **within a sample** (A1 against A2, the same condition measured twice), so
      the reading cannot be hijacked by the shaped frames; shaped excursions and
      the session-wide spread are reported as diagnostics only.  See the block
      comment below for the measured alternatives and why they false-trip.
    * G0-3 -- ``norm(mean(B1,B2) - mean(A1,A2)) / norm(A1 - A2)`` per sample.
    * G0-4 -- ``ncc(mean(B1,B2), twin prediction)`` per sample.

    Args:
        rows: Per-sample rows from :func:`measure_sample` (used for the counts).
        frames: ``{sample index: {a1, b1, b2, a2}}`` measured frames.
        full_scale: Detector saturation value.  ``None`` means the frames are
            digital model intensities (the ``sim`` backend), where an exposure
            gate is meaningless -- G0-1 then reports ``applicable: False`` and
            passes with that note, because the band it checks is a radiometry
            property of the real detector.
        twin_frames: ``{sample index: prediction}`` for G0-4.  Missing
            predictions make G0-4 **fail** (a gate that cannot be evaluated must
            not silently pass).
        exposure_ladder: The exposure decision from
            :meth:`HwInstrument._resolve_exposure` (or its ``explicit``
            counterpart).  Attached to G0-1 so an out-of-band failure carries the
            whole trace -- which step was probed, what each one measured, which was
            chosen.  ``None`` on the sim backend (no camera, no ladder).
        exposure_band: G0-1 band, as a fraction of full scale.
        saturation_fraction: G0-1 saturated-pixel ceiling.
        centroid_tolerance_px: G0-2 drift ceiling, pixels.
        phase_factor: G0-3 required ratio over the A drift.
        geometry_ncc: G0-4 ncc floor.

    Returns:
        ``{"g0_1_exposure", "g0_2_zero_order", "g0_3_phase_on_panel",
        "g0_4_geometry", "pass"}`` -- each check carries its readings, a
        ``pass`` flag and a one-line ``detail``.
    """
    indices = [int(row["index"]) for row in rows if int(row["index"]) in frames]

    # -- G0-1 exposure ---------------------------------------------------
    if full_scale is None:
        exposure = {
            "pass": True,
            "applicable": False,
            "detail": "digital frame (no detector): the exposure band is not measurable",
            "peak_fraction": None,
            "saturated_fraction": None,
            "full_scale": None,
            "exposure_ladder": exposure_ladder,
        }
    else:
        scale = float(full_scale)
        flats = [
            np.asarray(frames[index][name], dtype=np.float64)
            for index in indices
            for name in ("a1", "a2")
        ]
        peaks = [float(frame.max()) / scale for frame in flats]
        # A saturated pixel is within 0.3% of the ceiling even after the 3-frame
        # average the camera returns (see N_SAMPLE).
        saturated = [float((frame >= 0.997 * scale).mean()) for frame in flats]
        low, high = exposure_band
        peak_min = min(peaks) if peaks else float("nan")
        peak_max = max(peaks) if peaks else float("nan")
        sat_max = max(saturated) if saturated else float("nan")
        ok = bool(
            peaks
            and low <= peak_min
            and peak_max <= high
            and sat_max < saturation_fraction
        )
        exposure = {
            "pass": ok,
            "applicable": True,
            "detail": (
                f"flat peak {peak_min:.3f}-{peak_max:.3f} of full scale "
                f"(band {low}-{high}), saturated {sat_max:.3%}"
                + (
                    ""
                    if exposure_ladder is None
                    else f"; exposure {exposure_ladder.get('chosen_ms')} ms via "
                    f"{exposure_ladder.get('source')}"
                )
            ),
            "peak_fraction": peak_max,
            "peak_fraction_min": peak_min,
            "saturated_fraction": sat_max,
            "full_scale": scale,
            # The whole decision rides along, pass or fail: an out-of-band best
            # needs the trace to be diagnosable ("which steps did it try?").
            "exposure_ladder": exposure_ladder,
        }

    # -- G0-2 zero-order stability ---------------------------------------
    # Measured **within a sample**: A1 and A2 are the same condition measured
    # twice, so their spread is the bench's drift and nothing else -- and it
    # cannot be hijacked by the shaping, because both frames are flat.  Both
    # alternatives were measured on a healthy 24-sample twin session and both
    # false-trip: per-frame ``argmax`` on the shaped frames reports 49.5 px, a
    # +-16 px local search reports 16.5 px, and a session-wide comparison of the
    # *flat* frames reports 38.1 px -- that last one is not drift at all but the
    # per-sample aberration state, which includes a deliberate tip/tilt (up to
    # 6 camera px, ``AberrationConfig.tilt_px``).  On hardware the state is
    # constant, so both diagnostics are still reported (``session_spread_px``) and
    # read as extra drift evidence there.
    anchor_positions: list[tuple[float, float]] = []
    drift = 0.0
    worst = None
    shaped_excursion = 0.0
    for index in indices:
        first = _zero_order_centroid(frames[index]["a1"])
        anchor = (int(round(first[0])), int(round(first[1])))
        second = _zero_order_centroid(frames[index]["a2"], anchor)
        spread = math.hypot(second[0] - first[0], second[1] - first[1])
        if spread > drift:
            drift = spread
            worst = index
        anchor_positions.append(first)
        for name in ("b1", "b2"):
            shifted = _zero_order_centroid(frames[index][name], anchor)
            shaped_excursion = max(
                shaped_excursion,
                math.hypot(shifted[0] - first[0], shifted[1] - first[1]),
            )
    session_spread = 0.0
    for left in range(len(anchor_positions)):
        for right in range(left + 1, len(anchor_positions)):
            session_spread = max(
                session_spread,
                math.hypot(
                    anchor_positions[left][0] - anchor_positions[right][0],
                    anchor_positions[left][1] - anchor_positions[right][1],
                ),
            )
    zero_order = {
        "pass": bool(drift < centroid_tolerance_px),
        "detail": (
            f"max A1-vs-A2 0-order drift {drift:.2f} px (limit "
            f"{centroid_tolerance_px:g} px)"
            + ("" if worst is None else f", worst sample {worst}")
            + f"; session spread {session_spread:.2f} px and shaped-frame peak "
            f"excursion {shaped_excursion:.2f} px (diagnostics, not gated)"
        ),
        "max_drift_px": drift,
        "tolerance_px": float(centroid_tolerance_px),
        "session_spread_px": session_spread,
        "shaped_peak_excursion_px": shaped_excursion,
        "anchor": (
            "argmax + 3x3 weighted centroid of each sample's first flat frame "
            "(A1); A2 read in a "
            f"{ZERO_ORDER_SEARCH_PX:g} px local search around it. Shaped frames "
            "and the session-wide spread are reported only."
        ),
    }

    # -- G0-3 phase reaches the panel ------------------------------------
    ratios: dict[int, float] = {}
    for index in indices:
        block = frames[index]
        flat_mean = 0.5 * (
            np.asarray(block["a1"], dtype=np.float64)
            + np.asarray(block["a2"], dtype=np.float64)
        )
        net_mean = 0.5 * (
            np.asarray(block["b1"], dtype=np.float64)
            + np.asarray(block["b2"], dtype=np.float64)
        )
        numerator = _l2(net_mean, flat_mean)
        denominator = _l2(
            np.asarray(block["a1"], dtype=np.float64),
            np.asarray(block["a2"], dtype=np.float64),
        )
        if denominator <= 1e-12:
            ratios[int(index)] = float("inf") if numerator > 0.0 else 0.0
        else:
            ratios[int(index)] = numerator / denominator
    finite = [value for value in ratios.values() if math.isfinite(value)]
    min_ratio = min(finite) if finite else float("inf")
    phase = {
        "pass": bool(ratios and all(value > phase_factor for value in ratios.values())),
        "detail": (
            f"min norm(B-A)/norm(A1-A2) = {min_ratio:.2f} (needs > {phase_factor:g}); "
            f"{sum(1 for value in ratios.values() if value > phase_factor)}"
            f"/{len(ratios)} samples pass"
        ),
        "min_ratio": min_ratio,
        "factor": float(phase_factor),
        "ratios": {int(index): float(value) for index, value in ratios.items()},
    }

    # -- G0-4 geometry identifiable --------------------------------------
    if twin_frames is None:
        geometry = {
            "pass": False,
            "detail": "no twin prediction supplied, so G0-4 cannot be evaluated",
            "min_ncc": None,
            "ncc": {},
        }
    else:
        values: dict[int, float] = {}
        for index in indices:
            block = frames[index]
            measured = 0.5 * (
                np.asarray(block["b1"], dtype=np.float64)
                + np.asarray(block["b2"], dtype=np.float64)
            )
            prediction = twin_frames.get(index)
            if prediction is None:
                values[int(index)] = float("-inf")
                continue
            values[int(index)] = _ncc(measured, prediction)
        worst_ncc = min(values.values()) if values else float("-inf")
        geometry = {
            "pass": bool(values and worst_ncc >= geometry_ncc),
            "detail": (
                f"min ncc(measured net, twin net) = {worst_ncc:.3f} "
                f"(floor {geometry_ncc:g})"
            ),
            "min_ncc": worst_ncc,
            "floor": float(geometry_ncc),
            "ncc": {int(index): float(value) for index, value in values.items()},
        }

    return {
        "g0_1_exposure": exposure,
        "g0_2_zero_order": zero_order,
        "g0_3_phase_on_panel": phase,
        "g0_4_geometry": geometry,
        "pass": bool(
            exposure["pass"]
            and zero_order["pass"]
            and phase["pass"]
            and geometry["pass"]
        ),
    }


# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------
def verdict(
    rows: list[dict[str, Any]],
    *,
    min_wins: int = MIN_WINS,
    min_delta: float = MIN_DELTA,
    noise_factor: float = NOISE_FACTOR,
) -> str:
    """Apply README §3 and return one of the three verdict strings.

    The rule, in order:

    1. drop every sample where ``|ncc_net - ncc_flat| < noise_factor * |A1-A2|``
       (it is indistinguishable at this sample's own noise floor);
    2. if fewer than ``min_wins`` samples survive, there is **no power** to answer
       the question -> :data:`INSUFFICIENT_POWER` (not "does not recover": the
       two are different claims and the protocol keeps them apart);
    3. otherwise ``net`` recovers iff at least ``min_wins`` counted samples have
       ``ncc_net > ncc_flat`` **and** the counted mean of ``ncc_net - ncc_flat``
       exceeds ``min_delta``.

    The noise flag is recomputed here from the reported columns rather than taken
    from ``row["indistinguishable"]``, so the constant can be swept over an
    already-measured session without re-measuring; both paths call
    :func:`_indistinguishable`.

    Args:
        rows: Per-sample rows from :func:`measure_sample`.
        min_wins: Minimum counted samples favouring ``net`` (README: 6 of 8).
        min_delta: Minimum counted mean ``ncc_net - ncc_flat``.
        noise_factor: Noise-floor multiplier.

    Returns:
        :data:`NET_RECOVERS`, :data:`NET_DOES_NOT_RECOVER` or
        :data:`INSUFFICIENT_POWER`.
    """
    counted = [row for row in rows if not _indistinguishable(row, noise_factor)]
    if len(counted) < int(min_wins):
        return INSUFFICIENT_POWER
    wins = sum(1 for row in counted if float(row["ncc_net"]) > float(row["ncc_flat"]))
    mean_delta = sum(
        float(row["ncc_net"]) - float(row["ncc_flat"]) for row in counted
    ) / len(counted)
    if wins >= int(min_wins) and mean_delta > float(min_delta):
        return NET_RECOVERS
    return NET_DOES_NOT_RECOVER


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------
def select_indices(
    corpus: Any, samples: str = "test", limit: int | None = None
) -> list[int]:
    """Resolve ``--samples`` to cache indices.

    Args:
        corpus: The loaded :class:`ml.sim2f.train.Corpus`.
        samples: ``"test"`` (the corpus' test split) or ``"loop"`` (the published
            §5 set: the first :data:`LOOP_COUNT` of
            ``torch.randperm(LOOP_POPULATION, seed=LOOP_SEED)``).
        limit: Keep only the first ``limit`` indices; ``None`` keeps all.

    Returns:
        Cache indices, in measurement order.

    Raises:
        ValueError: If ``samples`` is unknown, or the corpus is too small for the
            ``loop`` set.
    """
    name = str(samples).lower()
    if name == "test":
        order = [int(value) for value in corpus.indices("test").tolist()]
    elif name == "loop":
        if corpus.n_samples < LOOP_POPULATION:
            raise ValueError(
                f"--samples loop needs the {LOOP_POPULATION}-pair corpus the published "
                f"§5 set was drawn from; this cache holds {corpus.n_samples}"
            )
        generator = torch.Generator().manual_seed(LOOP_SEED)
        order = torch.randperm(LOOP_POPULATION, generator=generator)[:LOOP_COUNT].tolist()
        order = [int(value) for value in order]
    else:
        raise ValueError(f"unknown sample set {samples!r}; expected 'test' or 'loop'")
    if limit is not None:
        order = order[: max(0, int(limit))]
    if not order:
        raise ValueError("the sample selection is empty")
    return order


def run_session(
    *,
    checkpoint: str | Path,
    cache: str | Path,
    samples: str = "test",
    limit: int | None = None,
    out_dir: str | Path = "runs/sim2f_bench/session",
    backend: str = "sim",
    exposure_ms: float | None = None,
    exposure_ladder: Sequence[float] | None = None,
    settle_s: float = SETTLE_S,
    ccd_px: int = 256,
    save_frames: bool = True,
    cam_type: str = "daheng",
    cam_id: int = 0,
    panel_shape: tuple[int, int] | None = None,
    pupil_center_xy: tuple[int, int] | None = None,
    device: str | torch.device = "auto",
    seed: int = LOOP_SEED,
    aberration_cfg: AberrationConfig | None = None,
) -> dict[str, Any]:
    """Measure one session and write the README §7 artefacts.

    Args:
        checkpoint: ``model_best.pt`` -- the frozen network (README §1).
        cache: Corpus directory (``pairs.npz``).
        samples: ``"test"`` or ``"loop"`` (see :func:`select_indices`).
        limit: Optional cap on the number of samples.
        out_dir: Artefact directory.
        backend: ``"sim"`` (calibrated twin as the instrument: the dry run) or
            ``"hw"`` (Santec SLM + CCD).
        exposure_ms: Camera exposure, ms.  An explicit value always wins and skips
            the ladder.  ``None`` (hardware) selects the exposure by measuring an
            **exposure ladder** on the flat phase -- see
            :class:`HwInstrument` and :func:`default_exposure_ladder`.
        exposure_ladder: Ladder steps to probe when ``exposure_ms`` is ``None``;
            ``None`` = the default geometric ladder ``cap/64 .. cap``.
        settle_s: Wait after each panel write.
        ccd_px: Camera window side.
        save_frames: Write ``frames.npz``.
        cam_type: Camera backend name (hardware only).
        cam_id: Camera index (hardware only).
        panel_shape: ``(height, width)``; ``None`` asks the device / uses the
            SLM-200 default.
        pupil_center_xy: Centre of the **illuminated pupil** on the panel, panel
            pixels ``(x, y)``; ``None`` assumes the beam is centred on the panel.
            Affects the hardware panel embedding only (the twin needs none).
        device: Torch device for the network and the twin.
        seed: Master seed.  ``7`` reproduces the published sample set.
        aberration_cfg: Distribution the ``sim`` states are drawn from; defaults
            to :class:`~ml.sim2f.config.AberrationConfig` (what the loop used).

    Returns:
        The session payload (also written to ``session.json``).

    Raises:
        ValueError: Unknown backend, empty selection, or a bench that disagrees
            with the corpus geometry.
        FileNotFoundError: Missing checkpoint or corpus.
        BenchmarkError: Hardware precondition not met (see
            :class:`HwInstrument`).
    """
    from ml.sim2f.train import _apply_determinism, load_corpus, seed_everything

    name = str(backend).lower()
    if name not in ("sim", "hw"):
        raise ValueError(f"unknown backend {backend!r}; expected 'sim' or 'hw'")

    resolved = _resolve_device(device)
    # Same preamble as ``loop.run_loop``: without ``_apply_determinism`` cuDNN may
    # pick a different convolution algorithm, and the U-Net output then differs in
    # its last bits -- enough to move ``net``'s metrics by ~1e-8 and break the
    # bit-exact reproduction of the published table.
    seed_everything(int(seed))
    _apply_determinism(True)
    corpus = load_corpus(cache)
    model, context, label_cfg, _payload = _load_network(checkpoint, resolved)
    bench_cfg = Sim2fConfig.calibrated()
    _check_geometry(label_cfg, bench_cfg)
    bench = Sim2f(bench_cfg, device=str(resolved))
    states = AberrationConfig() if aberration_cfg is None else aberration_cfg
    teacher = corpus.teacher_cfg
    order = select_indices(corpus, samples, limit)
    kinds = corpus.kinds()

    logger.info(
        "bench session: backend={} samples={} n={} checkpoint={} bench(exposure={} ms) "
        "ccd_px={} panel={} radius={} px settle={} s",
        name,
        samples,
        len(order),
        Path(checkpoint).name,
        bench_cfg.exposure_ms,
        ccd_px,
        panel_shape if panel_shape is not None else "device",
        PUPIL_RADIUS_PX,
        settle_s,
    )

    session_instrument: Instrument | None = None
    if name == "hw":
        session_instrument = HwInstrument(
            exposure_ms=exposure_ms,
            exposure_ladder=exposure_ladder,
            panel_shape=panel_shape,
            pupil_center_xy=pupil_center_xy,
            ccd_px=ccd_px,
            settle_s=settle_s,
            cam_type=cam_type,
            cam_id=cam_id,
            device=resolved,
        )

    rows: list[dict[str, Any]] = []
    frames: dict[int, dict[str, np.ndarray]] = {}
    twin_frames: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    instruments: list[Instrument] = []
    describe_instrument: Instrument | None = session_instrument
    if session_instrument is not None:
        instruments.append(session_instrument)

    try:
        for index in order:
            target = corpus.target[index : index + 1].to(
                device=resolved, dtype=torch.float32
            )
            support = corpus.support[index : index + 1].to(
                device=resolved, dtype=torch.float32
            )
            with torch.no_grad():
                net_phase = from_phasor(model(network_input(target, context)))

            instrument: Instrument
            if name == "sim":
                rng = np.random.default_rng([int(seed), int(index)])
                state = draw_aberration(bench, states, rng)
                aberration, illumination = write_aberration(bench, state)
                instrument = SimInstrument(
                    bench=bench,
                    aberration=aberration,
                    illumination=illumination,
                    detector=False,
                    seed=int(seed),
                )
                instruments.append(instrument)
                describe_instrument = instrument
            else:
                if session_instrument is None:
                    raise BenchmarkError("hardware backend opened no instrument")
                instrument = session_instrument

            row = measure_sample(
                instrument,
                net_phase=net_phase,
                target=target,
                support=support,
                teacher=teacher,
                index=index,
                kind=kinds[index],
            )
            rows.append(row)
            frames[int(index)] = row["frames"]
            targets[int(index)] = target[0, 0].detach().cpu().numpy()
            twin_frames[int(index)] = instrument.twin(net_phase)
            logger.info(
                "sample {} kind={} ncc flat={:.4f} net={:.4f} drift={:.4f} "
                "objective flat={:.4f} net={:.4f}{}",
                index,
                kinds[index],
                row["ncc_flat"],
                row["ncc_net"],
                row["ncc_a_drift"],
                row["objective_flat"],
                row["objective_net"],
                " [indistinguishable]" if row["indistinguishable"] else "",
            )
    finally:
        for instrument in instruments:
            instrument.close()

    describe: dict[str, Any] = (
        describe_instrument.describe()
        if describe_instrument is not None
        else {
            "backend": "sim",
            "bench": bench_cfg.describe(),
            "phase_input": (
                "(grid, grid) pupil radians -> Sim2f.forward; the twin's grid IS "
                "the 900 px pupil region, so no panel embedding on this path"
            ),
            "note": (
                "G0-1 is not applicable to digital frames (see gate0), so a sim "
                "session can only fail G0-2/G0-3/G0-4"
            ),
        }
    )
    full_scale = (
        getattr(session_instrument, "full_scale", None)
        if session_instrument is not None
        else None
    )
    exposure_trace: dict[str, Any] = (
        session_instrument.exposure_trace
        if session_instrument is not None
        else {
            "source": "not applicable (sim backend: no camera, no detector)",
            "chosen_ms": None,
        }
    )
    gate = gate0(
        rows,
        frames,
        full_scale=full_scale,
        twin_frames=twin_frames,
        exposure_ladder=exposure_trace,
    )
    conclusion = verdict(rows) if gate["pass"] else None
    criterion = _criterion(rows)

    aggregate = {
        "n": len(rows),
        "flat": _aggregate(rows, "flat"),
        "net": _aggregate(rows, "net"),
        **{key: value for key, value in criterion.items() if key != "rows"},
    }

    payload = _finite(
        {
            "schema": 1,
            "protocol": "runs/sim2f_bench/README.md",
            "checkpoint": str(Path(checkpoint)),
            "cache": str(Path(cache)),
            "backend": name,
            "samples": str(samples),
            "indices": [int(index) for index in order],
            "seed": int(seed),
            "exposure_ms": (
                session_instrument.exposure_ms
                if session_instrument is not None
                else (None if exposure_ms is None else float(exposure_ms))
            ),
            "exposure_ladder": exposure_trace,
            "settle_s": float(settle_s),
            "ccd_px": int(ccd_px),
            "panel_shape": None if panel_shape is None else [int(v) for v in panel_shape],
            "pupil_radius_px": PUPIL_RADIUS_PX,
            "pupil_side_px": int(round(2.0 * PUPIL_RADIUS_PX)),
            "pupil_placement": _pupil_placement(
                name, describe, panel_shape, pupil_center_xy
            ),
            "n_sample": N_SAMPLE,
            "bench": bench_cfg.to_dict(),
            "teacher": teacher.to_dict(),
            "constants": {
                "noise_factor": float(NOISE_FACTOR),
                "min_wins": int(MIN_WINS),
                "min_delta": float(MIN_DELTA),
                "conditions": list(CONDITIONS),
                "loop_seed": int(LOOP_SEED),
                "loop_population": int(LOOP_POPULATION),
                "loop_count": int(LOOP_COUNT),
            },
            "devices": describe,
            "gate0": gate,
            "criterion": criterion,
            "aggregate": aggregate,
            "verdict": conclusion,
            "rows": [
                {key: value for key, value in row.items() if key != "frames"}
                for row in rows
            ],
            "notes": (
                "A1(flat) -> B1(net) -> B2(net) -> A2(flat), fixed order, one bench "
                "state per sample. ncc_flat / ncc_net are the means of the two A / two "
                "B readings; ncc_a_drift is |ncc_A1 - ncc_A2|, this sample's noise "
                "floor. Scored with ml.sim2f.loop._score on prepared frames (the "
                "published report's own metric code). Gate 0 must pass before the "
                "criterion is reported; a failed gate writes no verdict string."
            ),
        }
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / SESSION_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_rows(out / ROWS_NAME, rows)
    if save_frames:
        _write_frames(out / FRAMES_NAME, frames, targets)
    _render_fetch(out / FETCH_NAME, rows, targets)
    _write_verdict(out / VERDICT_NAME, payload)
    logger.info(
        "session artefacts -> {} ({}, {}, {}, {}, {})",
        out,
        SESSION_NAME,
        ROWS_NAME,
        FRAMES_NAME if save_frames else "(frames skipped)",
        FETCH_NAME,
        VERDICT_NAME,
    )
    return payload


def _pupil_placement(
    backend: str,
    describe: dict[str, Any],
    panel_shape: tuple[int, int] | None,
    pupil_center_xy: tuple[int, int] | None,
) -> dict[str, Any]:
    """Record where the 900 px pupil square went, and whether it was defaulted.

    The operator needs this in the JSON because a wrong centre is not a shifted
    result -- it is a *wrong* one (the beam sees a cropped, off-axis piece of the
    phase map), and it cannot be reconstructed from the artefacts afterwards.  The
    ``sim`` backend never embeds into a panel (the twin's grid *is* the region), so
    its record says so instead of implying a placement that did not happen.

    Args:
        backend: ``"sim"`` or ``"hw"``.
        describe: The instrument's ``describe()`` (carries the real panel).
        panel_shape: Requested panel shape, or ``None`` (ask the device).
        pupil_center_xy: Requested pupil centre, or ``None`` (panel centre).

    Returns:
        A JSON-safe mapping with ``panel_shape``, ``panel_center_xy``,
        ``pupil_center_xy``, ``pupil_center_is_default``, ``pupil_corner_xy`` and
        ``applies_to``.
    """
    shape = panel_shape
    if shape is None:
        reported = describe.get("panel_shape")
        shape = PANEL_SHAPE if not reported else (int(reported[0]), int(reported[1]))
    shape = (int(shape[0]), int(shape[1]))
    center = pupil_center(shape, pupil_center_xy)
    return {
        "applies_to": (
            "panel embedding (hw backend only)"
            if backend == "hw"
            else "not used: the sim backend feeds the twin's grid directly"
        ),
        "panel_shape": [shape[0], shape[1]],
        "panel_center_xy": [shape[1] // 2, shape[0] // 2],
        "pupil_center_xy": [center[0], center[1]],
        "pupil_center_is_default": pupil_center_xy is None,
        "pupil_corner_xy": list(
            panel_corner(shape, PUPIL_RADIUS_PX, pupil_center_xy)
        ),
        "pupil_radius_px": float(PUPIL_RADIUS_PX),
    }


def _criterion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The README §3 numbers, as reported alongside the verdict."""
    counted = [row for row in rows if not _indistinguishable(row, NOISE_FACTOR)]
    wins = sum(1 for row in counted if float(row["ncc_net"]) > float(row["ncc_flat"]))
    mean_delta = (
        sum(float(row["ncc_net"]) - float(row["ncc_flat"]) for row in counted)
        / len(counted)
        if counted
        else float("nan")
    )
    return {
        "counted": len(counted),
        "excluded": len(rows) - len(counted),
        "wins": int(wins),
        "counted_mean_delta": float(mean_delta) if counted else None,
        "min_wins": int(MIN_WINS),
        "min_delta": float(MIN_DELTA),
        "noise_factor": float(NOISE_FACTOR),
        "indistinguishable": [
            int(row["index"]) for row in rows if row["indistinguishable"]
        ],
    }


def _aggregate(rows: list[dict[str, Any]], condition: str) -> dict[str, float]:
    """Column means of one condition (``flat`` = A, ``net`` = B)."""
    keys = ("objective", "l1", "ee", "ncc")
    values: dict[str, float] = {}
    for key in keys:
        column = f"{key}_{condition}"
        values[key] = float(sum(float(row[column]) for row in rows) / len(rows))
    return values


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-sample table (README §7: one row per sample)."""
    columns = (
        "index",
        "kind",
        "ncc_flat",
        "ncc_net",
        "ncc_delta",
        "ncc_a_drift",
        "ncc_b_drift",
        "objective_flat",
        "objective_net",
        "objective_delta",
        "l1_flat",
        "l1_net",
        "ee_flat",
        "ee_net",
        "zero_order_offset_px",
        "indistinguishable",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            cells: list[Any] = []
            for column in columns:
                value = row[column]
                if isinstance(value, bool):
                    cells.append("1" if value else "0")
                elif isinstance(value, float):
                    cells.append(f"{value:.10g}")
                else:
                    cells.append(value)
            writer.writerow(cells)


def _write_frames(
    path: Path,
    frames: dict[int, dict[str, np.ndarray]],
    targets: dict[int, np.ndarray],
) -> None:
    """Write the four measured frames per sample (+ its target) as float32."""
    arrays: dict[str, np.ndarray] = {}
    for index, block in frames.items():
        for name in CONDITIONS:
            arrays[f"s{index}_{name}"] = np.asarray(block[name], dtype=np.float32)
        if index in targets:
            arrays[f"s{index}_target"] = np.asarray(targets[index], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _render_fetch(
    path: Path, rows: list[dict[str, Any]], targets: dict[int, np.ndarray]
) -> None:
    """Render A1/B1/B2/A2 + target as one grid (``fetch.png``, README §7).

    Up to :data:`FETCH_ROWS` samples are drawn (one row each); a longer session is
    truncated with that stated in the title, because 24 full-size rows would be a
    multi-metre figure.  The table and the ``npz`` keep every sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = rows[:FETCH_ROWS]
    if not shown:
        return
    names = (*CONDITIONS, "target")
    figure, axes = plt.subplots(
        len(shown), len(names), figsize=(2.6 * len(names), 2.6 * len(shown)), squeeze=False
    )
    for position, row in enumerate(shown):
        index = int(row["index"])
        panels = dict(row["frames"])
        if index in targets:
            panels["target"] = targets[index]
        for column, label in enumerate(names):
            axis = axes[position][column]
            image = panels.get(label)
            if image is None:
                axis.set_axis_off()
                continue
            axis.imshow(image, cmap="inferno")
            axis.set_xticks([])
            axis.set_yticks([])
            if position == 0:
                axis.set_title(label)
        axes[position][0].set_ylabel(
            f"#{index} {row['kind']}\nflat {row['ncc_flat']:.3f}\n"
            f"net {row['ncc_net']:.3f}",
            fontsize=7,
        )
    title = "2f bench: A1 / B1 / B2 / A2 and target"
    if len(rows) > len(shown):
        title += f" (first {len(shown)} of {len(rows)} samples)"
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


#: Rows drawn into ``fetch.png``; the table/npz carry every sample.
FETCH_ROWS = 8


def _write_verdict(path: Path, payload: dict[str, Any]) -> None:
    """Write ``VERDICT.md``: criterion values and the verdict string, nothing else.

    README §7 forbids causal explanation here -- attribution belongs to the later
    analysis, not to the script.  When Gate 0 failed no verdict string is written
    at all (the three-value vocabulary describes *measured* sessions; a gate
    failure means the criterion was never reported, and calling that
    ``insufficient_power`` would conflate "not measured" with "measured but
    underpowered").
    """
    criterion = payload["criterion"]
    gate = payload["gate0"]
    verdict_value = payload["verdict"]
    lines = [
        "# 2f 实验系统实测 — 判据与结论",
        "",
        f"- 协议: `{payload['protocol']}`",
        f"- backend: `{payload['backend']}`",
        f"- 样本: `{payload['samples']}` × {payload['aggregate']['n']}",
        f"- checkpoint: `{payload['checkpoint']}`",
        "",
        "## Gate 0",
        "",
        "| 门 | 读数 | 结果 |",
        "|---|---|---|",
    ]
    for key, label in (
        ("g0_1_exposure", "G0-1 曝光"),
        ("g0_2_zero_order", "G0-2 零级稳定"),
        ("g0_3_phase_on_panel", "G0-3 相位到达面板"),
        ("g0_4_geometry", "G0-4 几何可辨识"),
    ):
        entry = gate[key]
        lines.append(
            f"| {label} | {entry['detail']} | {'通过' if entry['pass'] else '**未通过**'} |"
        )
    lines += [
        "",
        "## 判据 (README §3)",
        "",
        "| 项 | 值 | 阈 |",
        "|---|---|---|",
        f"| 计入样本 | {criterion['counted']} (剔除 {criterion['excluded']}) | ≥ {criterion['min_wins']} |",
        f"| `ncc_net > ncc_flat` 的样本数 | {criterion['wins']} | ≥ {criterion['min_wins']} |",
        "| `mean(ncc_net − ncc_flat)` | "
        + (
            "—"
            if criterion["counted_mean_delta"] is None
            else f"{criterion['counted_mean_delta']:+.4f}"
        )
        + f" | > {criterion['min_delta']:.2f} |",
        "",
        f"- 噪底系数: `{criterion['noise_factor']:g}` × `|ncc_A1 − ncc_A2|`",
        "- 判据与阈值取自预注册协议, 本期未改。",
        "",
        "## 结论",
        "",
    ]
    if verdict_value is None:
        failed = [
            label
            for key, label in (
                ("g0_1_exposure", "G0-1"),
                ("g0_2_zero_order", "G0-2"),
                ("g0_3_phase_on_panel", "G0-3"),
                ("g0_4_geometry", "G0-4"),
            )
            if not gate[key]["pass"]
        ]
        lines.append(f"主判据未报 (Gate 0 未通过: {', '.join(failed)})。")
    else:
        lines.append(f"`{verdict_value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Presentation (used by runs/sim2f_bench/run_bench.py)
# ----------------------------------------------------------------------
def format_gate0(gate: dict[str, Any]) -> str:
    """Render the Gate-0 table as fixed-width text."""
    lines = [
        f"{'gate':<22}{'result':>8}  reading",
        "-" * 78,
    ]
    for key, label in (
        ("g0_1_exposure", "G0-1 exposure"),
        ("g0_2_zero_order", "G0-2 zero-order"),
        ("g0_3_phase_on_panel", "G0-3 phase on panel"),
        ("g0_4_geometry", "G0-4 geometry"),
    ):
        entry = gate[key]
        lines.append(
            f"{label:<22}{'PASS' if entry['pass'] else 'FAIL':>8}  {entry['detail']}"
        )
    lines.append("-" * 78)
    lines.append(f"{'Gate 0':<22}{'PASS' if gate['pass'] else 'FAIL':>8}")
    return "\n".join(lines)


def format_rows(rows: list[dict[str, Any]]) -> str:
    """Render the per-sample table as fixed-width text."""
    header = (
        f"{'idx':>4} {'kind':<9} {'ncc_flat':>9} {'ncc_net':>9} {'delta':>8} "
        f"{'a_drift':>8} {'obj_flat':>9} {'obj_net':>9} {'z0_off':>7}  flag"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        offset = row.get("zero_order_offset_px")
        cell = "-" if offset is None or not math.isfinite(float(offset)) else f"{float(offset):>7.2f}"
        lines.append(
            f"{int(row['index']):>4} {str(row['kind']):<9} {row['ncc_flat']:>9.4f} "
            f"{row['ncc_net']:>9.4f} {row['ncc_delta']:>+8.4f} "
            f"{row['ncc_a_drift']:>8.4f} {row['objective_flat']:>9.4f} "
            f"{row['objective_net']:>9.4f} {cell}  "
            f"{'indistinguishable' if row['indistinguishable'] else ''}"
        )
    return "\n".join(lines)
