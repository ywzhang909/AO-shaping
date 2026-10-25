"""Parity of :class:`ml.sim2f.optics.Sim2f` against OOPAO's independent propagator.

Why this file exists
====================
``src/ml/sim2f/optics.py`` is the single differentiable 2f forward model that
every evaluation arm of ``report/shape_trials/`` shares: the teacher's labels,
the physics loss, the aberration fit and the closed loop all call
:class:`Sim2f`.  The committed report's decisive conclusion -- that the
**camera-defocus** term, not the network, flips the ranking -- therefore rests
on one implementation, and a bug inside it is invisible *from inside*: every arm
inherits the same wrong image, so the ranking stays self-consistent and no
internal metric can notice.  OOPAO is the only independent propagator available
here, and this module pins the two against each other.  It is the same
discipline as ``test_twin_parity.py`` (which pins us against
``ao_shaping.drivers.sim.bench_twin``); that one cannot close this hole because
``bench_twin`` and ``optics`` share a lineage, whereas OOPAO does not.

OOPAO is reached only through the sanctioned shim
=================================================
``ao_shaping.drivers.sim._oopao_compat`` is the **only** way to touch OOPAO in
this repository: upstream ``OOPAO/__init__.py`` is broken on numpy 2.x /
Windows (it scans ``sys.path`` for its own install directory and imports a
``tools`` module that no longer ships), so a direct ``import OOPAO`` fails.  The
module therefore loads ``Telescope`` from the shim and probes availability with
``ao_shaping.drivers.sim.oopao_backend._oopao_available()``.

Absolute photometry is deliberately not compared
-----------------------------------------------
The repository measured that the numpy and OOPAO simulation branches are not
equivalent (``report/oopao_impact/``), so absolute Strehl / FWHM / PIB / peak /
total energy must never be compared across them.  This file compares only
(a) **shape** similarity and (b) the **relative change** between an aberrated
and an unaberrated frame.  Both are invariant to a backend's absolute scale --
and indeed the two totals here are *identified* only because both models are
handed the same sampling; nothing in the assertions depends on that.

The shared sub-problem
======================
Both models are handed exactly the same pupil-plane problem, so a failure can
only be the propagator's:

* a **hard circular pupil** -- ``Sim2f.pupil_mask`` (the modelling disc of
  ``pupil_fill * grid / 2 = 48`` samples) is passed to OOPAO as its ``pupil``
  *and* to :meth:`Sim2f.field` as the ``illumination``, so **neither side
  apodizes** (the package's Gaussian entrance beam is overridden).  The OOPAO
  pupil is asserted sample-for-sample identical to it.
* the **same phase map in raw radians**, built from a few low-order Noll
  coefficients on the package's own basis (:meth:`Sim2f.aberration`), so the
  Zernike normalisation is shared and cannot be the variable under test.
* the **same pupil pitch**: OOPAO's ``pixelSize = diameter / resolution`` is set
  to ``Sim2fConfig.pupil_pitch_m`` (56.25 um) in a 128-sample grid, zero-padded
  by ``zeroPaddingFactor = 4`` to the same 512 samples ``Sim2f.fft_length`` uses.

Not covered -- deliberately, not by oversight: the metric camera-defocus
distance (``errors.defocus_m``), 0-order coherent mixing
(``errors.zero_order_eta``), SLM LUT quantisation (``lut_gamma`` /
``gray_levels``), the detector chain (read noise, dark/PRNU maps, 8-bit
quantisation), the camera resampling ratio ``2.099`` (``Sim2f.resample`` is
bypassed -- this file compares the model's own far field), and the Gaussian beam
apodization.  Each is either a *device* term OOPAO has no counterpart for (they
are pinned inside this package by ``test_twin_parity.py`` and the fitted
``bench_twin`` round) or a camera-side convention that does not touch the
propagator.

Residual convention gap: exactly half a sample, per axis
=======================================================
The two grids coincide in **pitch**: OOPAO's angular PSF scale is
``wavelength / (diameter * zeroPaddingFactor)`` = 3.6944e-5 rad and this
package's is ``far_field_pitch_m / focal_length_m`` = 3.6944e-5 rad
(``test_the_two_grids_have_the_same_pitch`` asserts it).  They differ only in
their **sampling origin**: ``Telescope.PropagateField`` multiplies the padded
pupil by ``exp(-i*pi*(N+1)/N*(x+y))`` for its odd/even crop arithmetic, which
leaves the optical axis *between* samples 255 and 256 of a 512 grid, while the
repository's centred-FFT convention puts it *on* sample 256.  Measured centroid
difference: ``-0.5`` samples per axis (exact on the symmetric flat frame, within
5.8e-3 on the aberrated ones) for every phase below.

That is a translation of the sampling grid, not physics, so each metric is
reported twice and both are asserted:

* **raw** -- the honest cross-implementation number with the half-sample still
  in it, i.e. what a user eyeballing the two packages sees;
* **aligned** -- after translating OOPAO's frame by *exactly* half a sample per
  axis (a band-limited shift whose amount is the constant ``_OFFSET_PX``,
  written down once here and never fitted per case).  The aberrated frame *and*
  its flat reference go through the same shift, so nothing is cherry-picked.
  If the propagators differed in anything but their origin this could not be
  ~1e-15.

Note what the raw column costs on purpose: a half-sample origin makes the
sampled peak of the sharpest pattern (flat, or tip/tilt) sit 4.26% low on
OOPAO's grid, which is why the raw L1 is as large as 0.27 there.  Those two
contacts are the *only* reason the raw numbers are not already exact.

Measured (float64, grid = 128, fft_n = 512, support = centred 128x128,
frames peak-normalised)
====================================================================================
phase (Noll: rad)         raw NCC / raw L1      aligned NCC / aligned rel-L2
=======================   ===================   =============================
flat                      0.9739 / 0.267        1.000000 / 3.8e-15
defocus   {4: 1.2}        0.9931 / 0.114        1.000000 / 4.6e-15
astig     {5: 1.0}        0.9889 / 0.170        1.000000 / 3.0e-15
coma      {7: 1.0}        0.9768 / 0.246        1.000000 / 5.0e-15
mixture   (5 modes)       0.9947 / 0.111        1.000000 / 6.2e-15
tip/tilt  {2,3: 0.8,-0.5} 0.9739 / 0.273        1.000000 / 3.7e-15
=======================   ===================   =============================

Relative change (the number the report actually relies on) on the same support,
defined as ``I_ab / peak(I_flat)`` -- this package's own flat-referenced
photometry convention (``counts_reference = "flat"``), i.e. how much the
aberration moves the frame *and* how much it dims the peak:

=====================================  ===========================  =======================
metric                                 raw                          aligned
=====================================  ===========================  =======================
NCC(delta_mine, delta_oopao)           0.9739 .. 0.9947              1.000000
normalised L1                          0.111 .. 0.267                < 8e-15
peak-change ratio                      0.9503 .. 0.99957              1.000000000
=====================================  ===========================  =======================

The 4.97% low end of the raw peak-change ratio is again OOPAO's flat peak being
sampled between samples (measured 0.95744 of ours); it is sampling of a
translation, not a photometric disagreement, and aligned it is 1 to 1e-15.
"""

from __future__ import annotations

import contextlib
import functools
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import Sim2fConfig
from ml.sim2f.optics import Sim2f

# ---------------------------------------------------------------------------
# Pre-registered tolerances.
#
# Each constant is the *pre-registered* budget, kept next to the measurement it
# guards so that loosening one means editing a number that sits beside its own
# justification.  ``test_the_tolerances_are_discriminative`` proves they are not
# decoration: it perturbs the OOPAO side the way a real bug would and requires
# every perturbation to fall OUTSIDE the aligned budget.
# ---------------------------------------------------------------------------

#: Zero-padding factor handed to ``PropagateField``.  It is ``Sim2fConfig``'s
#: fitted angular scale (``bench_twin``'s own constructor default of 10 does not
#: reproduce the measured spot), and ``4 * 128 = 512`` is exactly
#: ``Sim2fConfig.fft_length``, so the two PSFs land on one grid with no resampling.
ZERO_PADDING_FACTOR = 4

#: Centred support the metrics are computed on, in model samples.  128 samples
#: = 591 um, i.e. the reference ``ccd_px = 256`` camera window (563 um at
#: ``sampling_ratio = 2.099``) rounded to a power of two: the region the report's
#: arms are actually scored on, so the comparison is not dominated by the
#: hard pupil's high-order Airy rings outside it.
SUPPORT = 128

#: Raw (no convention compensation) shape agreement.  Measured worst case
#: 0.9739 NCC / 0.273 L1, both on the sharpest patterns (flat, tip/tilt): the
#: half-sample origin alone costs 2.6 NCC points there, and that cost is not
#: removable without interpolating, so the budget must sit under it.  0.97 is
#: still far above what a real modelling difference gives -- a Gaussian-vs-hard
#: apodization mismatch measures 0.908 raw here, and a 2x phase-amplitude error
#: 0.53 once aligned.
_RAW_NCC_MIN = 0.97
_RAW_L1_MAX = 0.30

#: Aligned (half-sample origin removed) agreement.  Measured 1e-15, i.e. machine
#: precision, so these are a numerically-precision contract with nine orders of
#: margin -- and 1e-4 still leaves three orders of headroom for a float32 OOPAO
#: build while catching a 2% pupil-radius error (L1 0.076) and a 10% phase error
#: (NCC 0.981).
_ALIGNED_NCC_MIN = 0.999
_ALIGNED_L1_MAX = 1e-4
_ALIGNED_REL_L2_MAX = 1e-4

#: The convention offset: half a sample, per axis, OOPAO's origin minus ours.
#: Measured exactly ``-0.5`` on the flat frame and within 5.8e-3 on aberrated
#: ones (a centroid is only defined to the discretisation of a half-sample
#: shear).  The values are deterministic -- fixed phases, float64 FFT -- so
#: there is no sampling variance to absorb, and 1e-2 still distinguishes this
#: unambiguously from 0 or 1 sample.
_OFFSET_PX = 0.5
_OFFSET_TOL_PX = 1e-2
_OFFSET_TOL_FLAT_PX = 1e-6

#: Relative-change agreement (aberrated frame referenced to its own flat peak).
#: Raw NCC worst 0.9739; raw peak-change ratio spans 0.9503..0.99957 because a
#: half-sample-offset pattern's sampled peak is up to 4.97% low, so a +-6% window
#: is the honest uncompensated bound.  Aligned both are exact.
_DELTA_RAW_NCC_MIN = 0.97
_DELTA_RAW_PEAK_RATIO = (0.94, 1.06)
_DELTA_ALIGNED_L1_MAX = 1e-4
_DELTA_ALIGNED_PEAK_RATIO_TOL = 1e-4

#: The phase states under test: ``{Noll index: radians}``, low order only --
#: defocus, astigmatism, coma, a five-mode mixture, and tip/tilt (which exercises
#: the transform's translation convention, the sharpest test of the half-sample
#: story).  Amplitudes are drawn from the range the report's own aberration
#: state uses (``AberrationConfig.rms_rad = 0.05 .. 0.45`` rad).
ABERRATIONS: dict[str, dict[int, float]] = {
    "flat": {},
    "defocus": {4: 1.2},
    "astigmatism": {5: 1.0},
    "coma": {7: 1.0},
    "mixture": {4: 0.9, 5: 0.7, 7: 0.5, 8: -0.4, 11: 0.3},
    "tip_tilt": {2: 0.8, 3: -0.5},
}
_IDS = list(ABERRATIONS)
_ABERRATED_IDS = [name for name in _IDS if name != "flat"]


def _oopao_unavailable_reason() -> str | None:
    """Why OOPAO cannot be used here, or ``None`` when it can.

    The probe mirrors what the test needs: the backend's availability flag (so a
    silent numpy fallback cannot masquerade as a pass) *and* the shadow package
    actually loading ``Telescope``.

    Returns:
        A skip reason naming what failed, or ``None``.
    """
    try:
        from ao_shaping.drivers.sim import oopao_backend
    except Exception as exc:
        return f"ao_shaping.drivers.sim.oopao_backend could not be imported: {exc!r}"
    if not oopao_backend._oopao_available():
        return (
            "OOPAO is not importable (ao_shaping.drivers.sim.oopao_backend."
            "_oopao_available() is False); install it with "
            "`uv pip install --no-deps -e libs/OOPAO`"
        )
    try:
        from ao_shaping.drivers.sim._oopao_compat import Telescope
    except Exception as exc:
        return f"the OOPAO shadow package failed to load Telescope: {exc!r}"
    del Telescope
    return None


_SKIP_REASON = _oopao_unavailable_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


class _Bench:
    """One hard pupil and one phase map, propagated by both implementations.

    The pupil disc is :attr:`Sim2f.pupil_mask`, so the amplitude passed to
    OOPAO and the amplitude passed to :meth:`Sim2f.field` are the same array --
    neither side gets an apodization the other does not have.
    """

    def __init__(self) -> None:
        """Build the shared problem and OOPAO's telescope."""
        from ao_shaping.drivers.sim._oopao_compat import Telescope

        self.cfg = Sim2fConfig()
        self.sim = Sim2f(self.cfg, device="cpu", dtype=torch.float64)
        self.disc = self.sim.pupil_mask[0, 0].numpy().astype(np.float64)
        self.illumination = torch.as_tensor(
            self.disc, dtype=torch.float64
        )[None, None]
        # OOPAO's declared diameter is the *array* span (grid * pupil_pitch =
        # 7.2 mm), not the illuminated disc (5.4 mm): its focal sampling is
        # ``wavelength / (diameter * zeroPaddingFactor)``, so the declared
        # diameter is what must make the PSF pitch equal
        # ``lambda * f / (fft_n * pupil_pitch)``.  ``pixelSize = diameter /
        # resolution`` then equals ``pupil_pitch_m`` and the two pupil grids are
        # sample-for-sample identical.  Declaring the disc diameter instead would
        # mis-scale the focal plane by grid / (2 * pupil_radius) = 4/3.
        with contextlib.redirect_stdout(io.StringIO()):
            self.telescope = Telescope(
                resolution=self.cfg.grid,
                diameter=self.cfg.grid * self.cfg.pupil_pitch_m,
                fov=0.0,
                samplingTime=0.001,
                pupil=self.disc.copy(),
            )

    def phase(self, modes: dict[int, float]) -> np.ndarray:
        """Pupil phase in radians for a ``{Noll index: radians}`` mapping.

        Args:
            modes: Noll indices ``2 .. 11`` mapped to radians (empty = flat).

        Returns:
            ``(grid, grid)`` float64 phase, zero outside the pupil disc.
        """
        coefficients = np.zeros(10, dtype=np.float64)  # Noll 2 .. 11
        for noll, value in modes.items():
            coefficients[noll - 2] = value
        return self.sim.aberration(coefficients[None, :])[0].numpy().astype(
            np.float64
        )

    def oopao_psf(
        self, phase: np.ndarray, pupil: np.ndarray | None = None
    ) -> np.ndarray:
        """OOPAO's far field for the shared pupil and phase.

        Args:
            phase: ``(grid, grid)`` radians.
            pupil: Optional amplitude override (used to perturb the aperture in
                the discriminative check).

        Returns:
            ``(fft_n, fft_n)`` float64 intensity.
        """
        amplitude = self.disc.copy() if pupil is None else np.asarray(pupil, float)
        self.telescope.PropagateField(
            amplitude,
            np.asarray(phase, dtype=np.float64),
            ZERO_PADDING_FACTOR,
            self.cfg.fft_length,
        )
        # ``tel.PSF`` is the telescope's live buffer: copy, so a later call
        # cannot rewrite a frame a test still holds.
        return np.array(self.telescope.PSF, dtype=np.float64, copy=True)

    def mine_psf(self, phase: np.ndarray) -> np.ndarray:
        """This package's far field, on the same pupil and phase.

        ``field`` (not ``forward``) on purpose: it is the model's own far field,
        so the camera-resampling convention is out of scope here.

        Args:
            phase: ``(grid, grid)`` radians.

        Returns:
            ``(fft_n, fft_n)`` float64 intensity.
        """
        field = self.sim.field(phase[None], illumination=self.illumination)
        return field.abs().square()[0, 0].numpy()


@functools.cache
def _bench() -> _Bench:
    """The session's shared bench (OOPAO's constructor prints a banner)."""
    return _Bench()


# ---------------------------------------------------------------------------
# Metrics.  Every one of them is scale-free except the peak ratio, which compares
# one implementation's *own* aberrated/flat ratio with the other's -- never an
# absolute level (see the module docstring).
# ---------------------------------------------------------------------------
def _support(image: np.ndarray) -> np.ndarray:
    """Centred ``SUPPORT x SUPPORT`` crop of an ``(fft_n, fft_n)`` frame."""
    low = image.shape[0] // 2 - SUPPORT // 2
    return image[low : low + SUPPORT, low : low + SUPPORT]


def _peak(image: np.ndarray) -> np.ndarray:
    """Peak-normalised copy (the only like-for-like form between backends)."""
    return image / float(image.max())


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    """Normalised cross-correlation of the de-meaned images."""
    a = left - left.mean()
    b = right - right.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _normalised_l1(measured: np.ndarray, reference: np.ndarray) -> float:
    """``sum|measured - reference| / sum|reference|``."""
    difference = np.abs(
        np.asarray(measured, np.float64) - np.asarray(reference, np.float64)
    )
    return float(difference.sum() / np.abs(reference).sum())


def _relative_l2(measured: np.ndarray, reference: np.ndarray) -> float:
    """``||measured - reference||_2 / ||reference||_2``."""
    difference = np.asarray(measured, np.float64) - np.asarray(
        reference, np.float64
    )
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """First-moment centre ``(x, y)`` of an intensity frame, in samples."""
    total = float(image.sum())
    coordinate = np.arange(image.shape[0], dtype=np.float64)
    return (
        float((image.sum(axis=0) * coordinate).sum() / total),
        float((image.sum(axis=1) * coordinate).sum() / total),
    )


def _translate(image: np.ndarray, offset: float = _OFFSET_PX) -> np.ndarray:
    """Translate a band-limited image by ``offset`` samples in both axes.

    Implemented as a centred-FFT phase ramp, so it is a genuine interpolation of
    a sampled field rather than a roll: exact (`~1e-15`) for the band-limited
    frames compared here, which is what makes "the residual gap is *only* a
    translation" a checkable claim instead of an assumption.  ``offset`` defaults
    to the measured convention constant ``_OFFSET_PX`` and is never fitted.

    Args:
        image: ``(n, n)`` intensity frame.
        offset: Translation in samples (positive moves towards higher indices).

    Returns:
        The translated frame, real float64.
    """
    size = int(image.shape[0])
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    frequency = np.arange(size, dtype=np.float64) - size / 2.0
    ramp = np.exp(
        -2j * np.pi * float(offset) * (frequency[:, None] + frequency[None, :]) / size
    )
    shifted = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * ramp)))
    return np.real(shifted)


# ---------------------------------------------------------------------------
# The grids
# ---------------------------------------------------------------------------
def test_the_two_grids_have_the_same_pitch() -> None:
    """Same pupil pitch and same focal sampling: the PSFs share one grid."""
    bench = _bench()
    cfg = bench.cfg
    assert cfg.far_field_padding == ZERO_PADDING_FACTOR
    assert bench.telescope.resolution == cfg.grid
    assert bench.telescope.pixelSize == pytest.approx(cfg.pupil_pitch_m, rel=1e-12)
    assert bench.telescope.D == pytest.approx(
        cfg.grid * cfg.pupil_pitch_m, rel=1e-12
    )

    oopao_angle_per_sample = cfg.wavelength_m / (
        bench.telescope.D * ZERO_PADDING_FACTOR
    )
    mine_angle_per_sample = cfg.far_field_pitch_m / cfg.focal_length_m
    assert oopao_angle_per_sample == pytest.approx(
        mine_angle_per_sample, rel=1e-12
    )

    psf = bench.oopao_psf(bench.phase({}))
    assert psf.shape == (cfg.fft_length, cfg.fft_length)
    assert not np.iscomplexobj(psf)
    assert float(psf.min()) >= 0.0


def test_the_two_pupils_are_the_same_aperture() -> None:
    """OOPAO's pupil is this package's disc, sample for sample."""
    bench = _bench()
    pupil = np.asarray(bench.telescope.pupil, dtype=np.float64)
    assert pupil.shape == bench.disc.shape
    assert float(np.abs(pupil - bench.disc).max()) == 0.0
    assert float(pupil.sum()) == float(bench.disc.sum())


# ---------------------------------------------------------------------------
# (a) shape similarity, and (b) the convention gap that costs it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("modes", ABERRATIONS.values(), ids=_IDS)
def test_far_field_matches_on_the_pixel_grid(modes: dict[int, float]) -> None:
    """Raw agreement, half-sample origin included: NCC >= 0.97, L1 <= 0.30."""
    bench = _bench()
    phase = bench.phase(modes)
    mine = _peak(_support(bench.mine_psf(phase)))
    theirs = _peak(_support(bench.oopao_psf(phase)))
    assert _ncc(mine, theirs) >= _RAW_NCC_MIN
    assert _normalised_l1(mine, theirs) <= _RAW_L1_MAX


@pytest.mark.parametrize("modes", ABERRATIONS.values(), ids=_IDS)
def test_the_residual_is_exactly_a_half_sample_origin(
    modes: dict[int, float],
) -> None:
    """The whole residual is a 0.5-sample translation, not physics.

    Removing it (a fixed, non-fitted constant -- see ``_translate``) makes the
    two images agree to machine precision, which also rules out the failure mode
    that a looser test would let through: a real scale, apodization or sign error
    cannot be undone by any translation.
    """
    bench = _bench()
    phase = bench.phase(modes)
    mine_psf = bench.mine_psf(phase)
    theirs = bench.oopao_psf(phase)

    mine_centre = _centroid(mine_psf)
    their_centre = _centroid(theirs)
    for axis, mine, their in zip("xy", mine_centre, their_centre, strict=True):
        assert their - mine == pytest.approx(
            -_OFFSET_PX, abs=_OFFSET_TOL_PX
        ), f"{axis} centroid offset"

    mine = _peak(_support(mine_psf))
    aligned = _peak(_support(_translate(theirs)))
    assert _ncc(mine, aligned) >= _ALIGNED_NCC_MIN
    assert _normalised_l1(mine, aligned) <= _ALIGNED_L1_MAX
    assert _relative_l2(mine, aligned) <= _ALIGNED_REL_L2_MAX


def test_the_flat_frame_offset_is_exact() -> None:
    """On a symmetric frame the convention offset is exactly 0.5 samples."""
    bench = _bench()
    flat = np.zeros((bench.cfg.grid, bench.cfg.grid), dtype=np.float64)
    mine = _centroid(bench.mine_psf(flat))
    theirs = _centroid(bench.oopao_psf(flat))
    for axis, mine_c, their_c in zip("xy", mine, theirs, strict=True):
        assert their_c - mine_c == pytest.approx(
            -_OFFSET_PX, abs=_OFFSET_TOL_FLAT_PX
        ), f"{axis} centroid offset"


# ---------------------------------------------------------------------------
# (c) the relative change -- what the report's conclusion actually rests on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "modes",
    [ABERRATIONS[name] for name in _ABERRATED_IDS],
    ids=_ABERRATED_IDS,
)
def test_relative_change_matches(modes: dict[int, float]) -> None:
    """The aberration's effect on the flat frame agrees, raw and aligned.

    ``delta = I / peak(I_flat)`` is this package's flat-referenced photometry
    convention, so the comparison covers both "where does the light go" and "how
    much does the peak drop" -- the two things the report's ranking depends on.
    """
    bench = _bench()
    flat = np.zeros((bench.cfg.grid, bench.cfg.grid), dtype=np.float64)
    flat_mine = bench.mine_psf(flat)
    flat_theirs = bench.oopao_psf(flat)
    phase = bench.phase(modes)
    mine = bench.mine_psf(phase)
    theirs = bench.oopao_psf(phase)

    delta_mine = _support(mine / float(flat_mine.max()))
    delta_theirs = _support(theirs / float(flat_theirs.max()))
    assert _ncc(delta_mine, delta_theirs) >= _DELTA_RAW_NCC_MIN
    raw_ratio = float(delta_mine.max() / delta_theirs.max())
    assert _DELTA_RAW_PEAK_RATIO[0] <= raw_ratio <= _DELTA_RAW_PEAK_RATIO[1]

    # Aligned: both the aberrated frame and its flat reference go through the
    # same translation, so the reference peak is sampled at the same origin.
    aligned_flat = _translate(flat_theirs)
    delta_aligned = _support(_translate(theirs) / float(aligned_flat.max()))
    assert _normalised_l1(delta_mine, delta_aligned) <= _DELTA_ALIGNED_L1_MAX
    aligned_ratio = float(delta_mine.max() / delta_aligned.max())
    assert aligned_ratio == pytest.approx(1.0, abs=_DELTA_ALIGNED_PEAK_RATIO_TOL)


# ---------------------------------------------------------------------------
# The tolerances above are tight enough to fail on a real bug
# ---------------------------------------------------------------------------
def test_the_tolerances_are_discriminative() -> None:
    """Perturbations a real bug would resemble all fall outside the budget.

    Three independent distortions of the *OOPAO* side only, each of the kind a
    genuine defect in either propagator would produce.  All must land below
    ``_ALIGNED_NCC_MIN`` -- otherwise the aligned thresholds would be decoration
    rather than a check.
    """
    bench = _bench()
    cfg = bench.cfg
    flat = np.zeros((cfg.grid, cfg.grid), dtype=np.float64)
    mine_flat = _peak(_support(bench.mine_psf(flat)))

    # (1) Aperture 4% too large: any pupil-radius / pitch error looks like this.
    axis_px = np.arange(cfg.grid, dtype=np.float64) - (cfg.grid - 1) / 2.0
    radius = float(cfg.pupil_radius_px)
    wider = (
        np.hypot(axis_px[None, :], axis_px[:, None]) <= radius * 1.04
    ).astype(np.float64)
    widened = _peak(_support(_translate(bench.oopao_psf(flat, pupil=wider))))
    assert _ncc(mine_flat, widened) < _ALIGNED_NCC_MIN

    # (2) Phase amplitude 10% too strong: a units or scaling error looks like
    # this, and it is the defect class this repository has actually shipped
    # twice in the WFS->SLM chain (2*pi and 0.532 um factors).
    defocus = {4: 1.2}
    mine_defocus = _peak(_support(bench.mine_psf(bench.phase(defocus))))
    overstated = _peak(
        _support(_translate(bench.oopao_psf(bench.phase({4: 1.2 * 1.1}))))
    )
    assert _ncc(mine_defocus, overstated) < _ALIGNED_NCC_MIN

    # (3) The alignment itself off by a quarter sample: proves the 0.5 is a
    # measured constant, not a knob that would absorb whatever the residual is.
    quarter_shifted = _peak(
        _support(_translate(bench.oopao_psf(bench.phase(defocus)), 0.25))
    )
    assert _ncc(mine_defocus, quarter_shifted) < _ALIGNED_NCC_MIN
