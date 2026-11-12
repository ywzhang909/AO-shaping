"""Pure-math SLM gray-to-phase LUT calibration utilities.

This module contains only pure NumPy math for building and inverting
gray-to-phase lookup tables for a spatial light modulator (SLM). It is a
leaf module in ``utils/`` and must not import any hardware drivers or
higher-level packages.

The physical model used for the depth-scan inversion is the blazed-grating
efficiency::

    eta(D) = sinc^2((D - 2*pi) / (2*pi))

with ``D = phi(g)`` the phase depth in gray, peaking at ``D = 2*pi``.

The checkerboard model used by :func:`invert_checkerboard_scan` is instead the
zero-order retained fraction of an alternating ``0`` / ``delta`` pattern::

    r(delta) = cos^2(delta / 2)

Beyond the math, this module owns the **serial-bound LUT archive** and its
**trust gate** (:data:`LUT_CALIB_DIR`, :func:`save_serial_lut`,
:func:`require_validated_lut`). A calibrated LUT is stored under
``data/calib/lut/<slm_serial>/`` next to a ``meta.json`` sidecar whose ``status``
is either ``provisional`` (measured, not trusted) or ``validated`` (explicitly
verified). The production display path refuses everything that is not
``validated`` -- including a directory with no sidecar at all -- so an
unverified inversion can never reach the panel by being silently picked up.
This mirrors the one-time dark-field calibration convention in
``scripts/slm/capture_dark_flat_baselines.py`` (serial-keyed cache, atomic
tmp+replace writes, no cross-device reuse).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np


def blaze_ramp_gray(period: int, peak_gray: int, width: int) -> np.ndarray:
    """1D blaze gray ramp, shape (1, width), phase varies along x (vertical stripes).

    ``gray(x) = round(frac * peak_gray)`` with ``frac = (x mod period)/period``
    for ``x in [0, width)``.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        width: Number of pixels along x.

    Returns:
        Float array of shape (1, width) with the blaze gray ramp.
    """
    x = np.arange(width)
    frac = (x % period) / period
    return np.round(frac * peak_gray).reshape(1, width)


def depth_pattern(period: int, peak_gray: int, height: int, width: int) -> np.ndarray:
    """Depth-scan TEST half-pattern (full screen).

    Tiles ``blaze_ramp_gray(period, peak_gray, width)`` to ``(height, width)``,
    dtype uint16. ``peak_gray`` is in ``[0, grayscale_max]``.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        height: Number of rows.
        width: Number of columns.

    Returns:
        uint16 array of shape (height, width).
    """
    ramp = blaze_ramp_gray(period, peak_gray, width)
    return np.tile(ramp, (height, 1)).astype(np.uint16)


def offset_pattern(
    period: int,
    peak_gray: int,
    gray_offset: int,
    bits: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Offset-scan TEST half-pattern.

    ``mod(blaze_ramp_gray(period, peak_gray, width) + gray_offset, 2**bits)``,
    tiled to ``(height, width)``, dtype uint16.

    Args:
        period: Number of pixels per blaze period.
        peak_gray: Peak grayscale value of the ramp.
        gray_offset: Grayscale offset added before the modulo wrap.
        bits: Bit depth of the grayscale range (modulus is ``2**bits``).
        height: Number of rows.
        width: Number of columns.

    Returns:
        uint16 array of shape (height, width).
    """
    ramp = blaze_ramp_gray(period, peak_gray, width)
    shifted = np.mod(ramp + gray_offset, 2**bits)
    return np.tile(shifted, (height, 1)).astype(np.uint16)


def random_phase_pattern(seed: int, panel_shape: tuple[int, int]) -> np.ndarray:
    """Uniform random phase screen on ``[0, 2*pi)``, tiled over the panel.

    Unlike the checkerboard and blaze patterns, the zero order of a random phase
    screen is ``|mean(exp(i*phi))|``, so it is insensitive to the *gain* of the
    gray-to-phase response: even when the SLM saturates below the designed depth
    (measured gain ~0.5-0.8, 2026-11-11), a screen covering a large enough arc of
    the unit circle still empties the zero order. That is what makes it the
    reference blocker for knife-edge probes.

    Args:
        seed: Seed for ``numpy.random.default_rng`` (byte-identical repeats).
        panel_shape: ``(height, width)`` panel resolution.

    Returns:
        float32 array of shape ``panel_shape``, units radians.
    """
    rng = np.random.default_rng(seed)
    return (rng.random(panel_shape) * 2.0 * np.pi).astype(np.float32)


def stack_halves(ref: np.ndarray, test: np.ndarray, axis: int = 0) -> np.ndarray:
    """Vertical stack (axis=0, ref on top, test below).

    Each half is ``(H/2, W)``; output is ``(H, W)``. Validates that all
    dimensions except the stacking axis match.

    Args:
        ref: Top half array.
        test: Bottom half array.
        axis: Axis along which to stack (default 0).

    Returns:
        Stacked array.

    Raises:
        ValueError: If shapes are incompatible for stacking along ``axis``.
    """
    ref = np.asarray(ref)
    test = np.asarray(test)
    if ref.ndim != test.ndim:
        raise ValueError(
            f"ref and test must have the same number of dims, "
            f"got {ref.ndim} and {test.ndim}"
        )
    for i in range(ref.ndim):
        if i != axis and ref.shape[i] != test.shape[i]:
            raise ValueError(
                f"shapes {ref.shape} and {test.shape} incompatible for "
                f"stacking along axis {axis}"
            )
    return np.concatenate([ref, test], axis=axis)


def normalize_efficiency(eta: np.ndarray) -> np.ndarray:
    """Normalize efficiency by its maximum.

    Returns ``eta / max(eta)``; if ``max(eta) <= 0`` returns zeros. Guards
    against division by zero.

    Args:
        eta: Efficiency array.

    Returns:
        Normalized efficiency array (same shape).
    """
    eta = np.asarray(eta, dtype=float)
    m = np.max(eta)
    if m <= 0:
        return np.zeros_like(eta)
    return eta / m


def sinc_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of ``sinc(x) = sin(pi*x)/(pi*x)`` restricted to branch ``x in [-1, 0]``.

    Returns non-positive values. Vectorized numerical root-finding via
    bisection on the fixed interval ``[-1, 0]`` (pure NumPy, no scipy).

    Edge cases: ``y=1 -> 0``; ``y<=0 -> -1``; ``y>1 -> ValueError``;
    ``NaN -> NaN``.

    Args:
        y: Array of values in ``(0, 1]``.

    Returns:
        Array of ``x in [-1, 0]`` with ``sinc(x) = y``.

    Raises:
        ValueError: If any ``y > 1``.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    nan_mask = np.isnan(y)
    out[nan_mask] = np.nan
    finite = ~nan_mask
    if np.any(y[finite] > 1.0):
        raise ValueError("sinc_inv: y must be in (0, 1]")
    le0 = finite & (y <= 0.0)
    out[le0] = -1.0
    eq1 = finite & (y == 1.0)
    out[eq1] = 0.0
    valid = finite & (y > 0.0) & (y < 1.0)
    if np.any(valid):
        ys = y[valid]
        lo = np.full_like(ys, -1.0)
        hi = np.zeros_like(ys)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            smid = np.sinc(mid)
            lo = np.where(smid < ys, mid, lo)
            hi = np.where(smid < ys, hi, mid)
        out[valid] = 0.5 * (lo + hi)
    return out


def invert_depth_scan(eta: np.ndarray) -> np.ndarray:
    """Oracle-verified depth-scan inversion.

    ``eta`` is the measured efficiency array (parallel to the gray scan
    ``g_values``). Let ``p = argmax(eta)``, ``etap = normalize_efficiency(eta)``,
    ``u = sinc_inv(sqrt(etap))`` (in ``[-1, 0]``).

    - ``i < p`` (rising flank, phi < 2pi): ``phi[i] = 2*pi*(1 + u[i])``
    - ``i == p`` (peak, phi == 2pi): ``phi[p] = 2*pi``
    - ``i > p`` (falling flank, phi > 2pi): ``phi[i] = 2*pi*(1 - u[i])``

    The final result is clipped to ``[0, 2*pi*1.04]``.

    Args:
        eta: Measured efficiency array.

    Returns:
        float64 array of recovered phase in radians (same shape as ``eta``).
    """
    eta = np.asarray(eta, dtype=float)
    etap = normalize_efficiency(eta)
    p = int(np.argmax(eta))
    u = sinc_inv(np.sqrt(etap))
    phi = np.empty_like(eta, dtype=float)
    phi[:p] = 2 * np.pi * (1.0 + u[:p])
    phi[p] = 2 * np.pi
    phi[p + 1 :] = 2 * np.pi * (1.0 - u[p + 1 :])
    phi = np.clip(phi, 0.0, 2 * np.pi * 1.04)
    return phi


def invert_offset_scan(
    eta: np.ndarray, g_values: np.ndarray, gray_for_2pi: int
) -> np.ndarray:
    """User-spec offset-scan inversion (kept for method parity).

    ``etap = normalize_efficiency(eta)``; ``dphi = 2*arcsin(sqrt(etap))``;
    ``phi_raw = 2*pi*(g/gray_for_2pi) + dphi`` where ``g = g_values``.
    Then monotonicity is enforced: ``phi = np.maximum.accumulate(phi_raw)``.

    For ``g_values > gray_for_2pi`` the gray is duplicated to ``gray_for_2pi``
    and the phase is clipped at ``2*pi`` there (phase saturates).

    Args:
        eta: Measured efficiency array.
        g_values: Gray values parallel to ``eta``.
        gray_for_2pi: Gray value corresponding to a ``2*pi`` phase depth.

    Returns:
        float64 array of recovered phase in radians (same shape as ``eta``).
    """
    etap = normalize_efficiency(eta)
    dphi = 2 * np.arcsin(np.sqrt(etap))
    g = np.asarray(g_values, dtype=float)
    g_eff = np.minimum(g, gray_for_2pi)
    phi_raw = 2 * np.pi * (g_eff / gray_for_2pi) + dphi
    phi = np.maximum.accumulate(phi_raw)
    mask = g > gray_for_2pi
    phi = np.where(mask, np.minimum(phi, 2 * np.pi), phi)
    return phi


def invert_checkerboard_scan(retained_fraction: np.ndarray) -> np.ndarray:
    """Two-branch inversion of a checkerboard alternating-phase scan.

    A checkerboard alternating phase ``0`` and ``delta`` keeps an ideal
    zero-order retained fraction::

        r = cos^2(delta / 2)

    Inverting it gives the principal value ``2 * arccos(sqrt(r))``, which lives
    in ``[0, pi]`` and cannot tell ``delta`` from ``2*pi - delta``. The scan's
    interior minimum (``r ~ 0``, i.e. the ``pi`` crossing) is the branch point:
    points up to it keep the principal value, and the higher-gray branch beyond
    it is unwrapped as ``2*pi - principal``, so the returned phase spans
    ``[0, 2*pi]``.

    Measured ratios drift, so values outside ``[0, 1]`` are clipped to the
    physical range; the scan is never normalized by its maximum. Measurement
    noise can make the phase locally non-monotonic, so each branch is made
    nondecreasing on its own with a cumulative maximum. That keeps the branch
    point at exactly ``pi`` (the lower branch never exceeds ``pi``, the upper
    branch never falls below it) while still returning a phase array directly
    usable by :func:`build_inverse_lut`. Where noise clips the ratio at the flat
    ends of the response the result can hold equal neighbours; a caller that
    needs a strictly increasing phase must add its own tie-break.

    Args:
        retained_fraction: 1-D array of measured zero-order retained fractions,
            ordered like the gray scan they were taken at. Needs at least 3
            points, and its minimum must lie strictly inside the scan.

    Returns:
        float64 array of recovered phase in radians, same shape as
        ``retained_fraction``, nondecreasing and within ``[0, 2*pi]``.

    Raises:
        ValueError: If the input is not a finite 1-D array with at least 3
            points, or its minimum sits at an endpoint (the full two-branch
            ``[0, 2*pi]`` range is then not identifiable).
    """
    r = np.asarray(retained_fraction, dtype=float)
    if r.ndim != 1:
        raise ValueError(
            f"retained_fraction must be a 1-D scan, got {r.ndim}-D shape {r.shape}"
        )
    if r.size < 3:
        raise ValueError(f"retained_fraction needs at least 3 points, got {r.size}")
    if not np.all(np.isfinite(r)):
        raise ValueError("retained_fraction must contain only finite values")

    r_phys = np.clip(r, 0.0, 1.0)
    branch = int(np.argmin(r_phys))
    if branch == 0 or branch == r.size - 1:
        raise ValueError(
            "retained_fraction minimum must be an interior point of the scan "
            f"(found index {branch} of {r.size}); without the pi crossing the "
            "two-branch [0, 2*pi] range is not identifiable"
        )

    principal = 2.0 * np.arccos(np.sqrt(r_phys))
    lower = principal[: branch + 1].copy()
    lower[-1] = np.pi
    upper = 2.0 * np.pi - principal[branch + 1 :]
    phi = np.concatenate(
        [np.maximum.accumulate(lower), np.maximum.accumulate(upper)]
    )
    return np.clip(phi, 0.0, 2.0 * np.pi)


def build_inverse_lut(
    phi: np.ndarray, g_values: np.ndarray, n: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Build an inverse LUT (phase -> gray).

    Uses a uniform target grid ``t`` in ``[0, 2*pi]`` (``n`` points) and sets
    ``gray_lut[i] = g_values[argmin(|phi - t[i]|)]`` (first occurrence on ties).

    Args:
        phi: Measured phase array (must be sorted ascending).
        g_values: Gray values parallel to ``phi``.
        n: Number of points in the target grid.

    Returns:
        Tuple ``(t_grid, gray_lut)`` where ``t_grid`` is the float target grid
        of size ``n`` and ``gray_lut`` is a uint16 index array of size ``n``.

    Raises:
        ValueError: If ``phi`` is not sorted ascending.
    """
    phi = np.asarray(phi, dtype=float)
    g_values = np.asarray(g_values)
    if np.any(np.diff(phi) < 0):
        raise ValueError("phi must be sorted ascending")
    t = np.linspace(0.0, 2 * np.pi, n)
    idx = np.argmin(np.abs(phi[:, None] - t[None, :]), axis=0)
    gray_lut = g_values[idx].astype(np.uint16)
    return t, gray_lut


def save_lut(
    dir_path: Path,
    g_values: np.ndarray,
    phi: np.ndarray,
    inverse_gray: np.ndarray,
    meta: dict | None = None,
) -> Path:
    """Write the LUT files into ``dir_path``.

    Writes three files:
    - ``lut_forward.csv``: header ``g,phi``, g/phi float columns.
    - ``lut_inverse.csv``: header ``target_phi,gray``, inverse table.
    - ``lut.npz``: arrays ``g_values``, ``phi``, ``inverse_gray`` plus ``meta``.

    Args:
        dir_path: Directory to write into (created if missing).
        g_values: Gray values (measured gray -> phase response).
        phi: Measured phase values (radians).
        inverse_gray: Inverse table gray values.
        meta: Optional metadata dict.

    Returns:
        The ``dir_path``.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    g_values = np.asarray(g_values, dtype=float)
    phi = np.asarray(phi, dtype=float)
    inverse_gray = np.asarray(inverse_gray)

    np.savetxt(
        dir_path / "lut_forward.csv",
        np.column_stack([g_values, phi]),
        delimiter=",",
        header="g,phi",
        comments="",
    )
    target_phi = np.linspace(0.0, 2 * np.pi, inverse_gray.size)
    np.savetxt(
        dir_path / "lut_inverse.csv",
        np.column_stack([target_phi, inverse_gray]),
        delimiter=",",
        header="target_phi,gray",
        comments="",
    )
    np.savez_compressed(
        dir_path / "lut.npz",
        g_values=g_values,
        phi=phi,
        inverse_gray=inverse_gray,
        meta=np.asarray(meta if meta is not None else {}, dtype=object),
        allow_pickle=True,
    )
    return dir_path


@dataclass
class LUTData:
    """Container for a loaded LUT.

    Attributes:
        g_values: Gray values (measured gray -> phase response).
        phi: Measured phase values (radians).
        inverse_gray: Inverse table gray values.
        meta: Metadata dict.
    """

    g_values: np.ndarray
    phi: np.ndarray
    inverse_gray: np.ndarray
    meta: dict = field(default_factory=dict)


def load_lut(dir_path: Path) -> LUTData:
    """Load a LUT from ``dir_path``.

    Loads ``lut.npz`` if present, otherwise falls back to the CSVs.

    Args:
        dir_path: Directory containing the LUT files.

    Returns:
        A :class:`LUTData` with the loaded arrays and metadata.

    Raises:
        FileNotFoundError: If neither the npz nor the CSVs are present.
    """
    dir_path = Path(dir_path)
    npz = dir_path / "lut.npz"
    if npz.exists():
        data = np.load(npz, allow_pickle=True)
        meta = data["meta"].item() if "meta" in data else {}
        return LUTData(
            g_values=data["g_values"],
            phi=data["phi"],
            inverse_gray=data["inverse_gray"],
            meta=meta,
        )
    fwd = dir_path / "lut_forward.csv"
    inv = dir_path / "lut_inverse.csv"
    if not fwd.exists() or not inv.exists():
        raise FileNotFoundError(f"LUT files not found in {dir_path}")
    gphi = np.loadtxt(fwd, delimiter=",", skiprows=1)
    tphi = np.loadtxt(inv, delimiter=",", skiprows=1)
    return LUTData(
        g_values=gphi[:, 0],
        phi=gphi[:, 1],
        inverse_gray=tphi[:, 1].astype(np.uint16),
        meta={},
    )


# ---------------------------------------------------------------------------
# Serial-bound LUT archive + production trust gate
# ---------------------------------------------------------------------------

#: Persistent LUT archive, one directory per SLM serial number. CWD-relative,
#: exactly like ``capture_dark_flat_baselines.DARK_CALIB_DIR``; tests and tools
#: that need a different location pass ``root_dir=``.
LUT_CALIB_DIR = Path("data/calib/lut")

#: Trust sidecar inside a LUT directory. **No sidecar means no trust.**
LUT_META_FILENAME = "meta.json"

#: ``provisional`` -- measured but not verified, the production path must refuse
#: it. ``validated`` -- explicitly verified, loadable. No third state exists.
LUT_STATUS_PROVISIONAL: str = "provisional"
LUT_STATUS_VALIDATED: str = "validated"

LutStatus = Literal["provisional", "validated"]

#: The only two status strings the archive accepts, in trust order.
LUT_STATUSES: tuple[str, ...] = (LUT_STATUS_PROVISIONAL, LUT_STATUS_VALIDATED)

#: Bump when the required ``meta.json`` keys change.
LUT_META_SCHEMA_VERSION = 1

#: Keys every readable ``meta.json`` must carry. Deliberately strict: a sidecar
#: that cannot say what device, when, or how good the inversion was is not
#: evidence, and the gate would be theatre if it accepted one.
LUT_META_REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "status",
    "serial",
    "created_utc",
    "source",
    "n_points",
    "inverse_length",
    "quality",
    "notes",
)

#: A device serial becomes a directory name, so it has to survive being one.
_SERIAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class LutSerialError(ValueError):
    """The SLM serial cannot name a calibration directory.

    Raised for an empty (device never opened / identity read failed),
    whitespace-only, or path-unsafe serial, so that nothing is ever written to
    ``data/calib/lut//`` and silently shared by every device.
    """


class LutTrustError(ValueError):
    """This LUT directory is not trustworthy for the production display path.

    Raised by :func:`require_validated_lut` for a missing or unreadable
    ``meta.json``, an unknown status, a serial/directory mismatch, or a
    ``provisional`` LUT. Never downgraded to a warning by callers that load a
    LUT into a live SLM: the point is that an unverified inversion cannot reach
    the panel just because nobody looked.
    """


def _utc_now_iso() -> str:
    """Second-resolution UTC stamp, the same shape as the dark cache's."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, write_fn: Callable[[Any], None]) -> None:
    """Write ``path`` through a same-directory ``.tmp`` file, then ``replace``.

    Mirrors ``capture_dark_flat_baselines._write_dark_calibration``: a reader
    (or a crash mid-write) never sees a half-written table or sidecar, and a
    failed write leaves the previous content intact and no ``.tmp`` behind.

    Args:
        path: Destination file. Its parent directory is created.
        write_fn: Called with the open binary ``.tmp`` handle.

    Raises:
        Whatever ``write_fn`` or the filesystem raises, unchanged: the ``.tmp`` is
        removed on every failure path (no blind ``except`` needed, and nothing is
        swallowed), so a crash mid-write leaves neither a half-written file nor a
        stray temp file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    published = False
    try:
        with open(tmp, "wb") as handle:
            write_fn(handle)
        tmp.replace(path)
        published = True
    finally:
        # 失败原因（磁盘满、序列化崩、权限……）一律原样上抛；这里只用 finally
        # 保证"任何异常路径都不留 .tmp"，因此不需要一个宽接的 except。
        if not published:
            tmp.unlink(missing_ok=True)


def normalize_slm_serial(slm_serial: str | None) -> str:
    """Validate an SLM serial for use as an archive directory name.

    Args:
        slm_serial: Serial as reported by the device (``Santec.serial_number``).

    Returns:
        The serial, stripped.

    Raises:
        LutSerialError: If it is ``None``, empty/whitespace-only, or contains
            anything that is not ``[A-Za-z0-9._-]`` (path separators, ``..``,
            control characters included). An empty serial means the identity
            was never read -- refusing to write is the only safe answer.
    """
    if slm_serial is None:
        raise LutSerialError(
            "SLM 序列号为 None：设备未 open() 或控制卡身份读取失败，"
            "拒绝写入 LUT 归档（否则所有设备会共用同一个目录）"
        )
    text = str(slm_serial).strip()
    if not text:
        raise LutSerialError(
            "SLM 序列号为空字符串：设备未 open() 或控制卡身份读取失败，"
            "拒绝写入 LUT 归档（否则所有设备会共用同一个目录）"
        )
    if not _SERIAL_RE.match(text):
        raise LutSerialError(
            f"SLM 序列号不能用作目录名: {slm_serial!r}"
            f"（允许 [A-Za-z0-9._-]，首字符必须是字母或数字，长度 1..64）"
        )
    return text


def lut_dir_for_serial(slm_serial: str | None, *, root_dir: Path | None = None) -> Path:
    """Return ``<root>/data/calib/lut/<slm_serial>`` for one SLM.

    Args:
        slm_serial: Device serial; validated by :func:`normalize_slm_serial`.
        root_dir: Archive root. ``None`` keeps the repo convention of a
            CWD-relative ``data/calib/lut``.

    Returns:
        The serial-bound directory (not created here).

    Raises:
        LutSerialError: If the serial is unusable.
    """
    serial = normalize_slm_serial(slm_serial)
    base = Path(root_dir) if root_dir is not None else Path()
    return base / LUT_CALIB_DIR / serial


def build_lut_meta(
    *,
    serial: str | None,
    source: str,
    g_values: np.ndarray,
    phi: np.ndarray,
    inverse_gray: np.ndarray,
    status: str = LUT_STATUS_PROVISIONAL,
    wavelength_nm: float | None = None,
    gray_for_2pi: int | None = None,
    valley: float,
    quality: Mapping[str, Any] | None = None,
    notes: str = "",
    created_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``meta.json`` payload for one calibration run.

    Args:
        serial: SLM serial the LUT is bound to (validated here).
        source: Producing script/command, e.g.
            ``"python scripts/slm/calib_slm_lut_checkerboard.py --out ..."``.
        g_values: Measured gray points.
        phi: Recovered phase in radians, same length as ``g_values``.
        inverse_gray: The table the driver consumes (``uint16``).
        status: ``"provisional"`` (default) or ``"validated"``. Validation is a
        measurement claim, so callers that did not verify should leave the
        default and use :func:`mark_lut_validated` later.
        wavelength_nm: Working wavelength; give this or ``gray_for_2pi``.
        gray_for_2pi: Nominal 2-pi gray at the working wavelength.
        valley: The scan's interior minimum -- the quality number that decides
            whether the inversion is trustworthy (for the checkerboard method
            it is the smallest zero-order retained fraction; ideal is 0.0).
        quality: Extra per-method quality metrics (merged into ``quality``).
        notes: Free text; what was measured and why it is or is not trusted.
        created_utc: Override the timestamp (reproducible writes/tests).
        extra: Non-quality provenance merged in as-is (exposure, cells, ...).

    Returns:
        A JSON-serialisable dict with the required keys filled in.

    Raises:
        LutSerialError: If the serial is unusable.
        ValueError: If ``status`` is not one of :data:`LUT_STATUSES``, both
            ``wavelength_nm`` and ``gray_for_2pi`` are missing, ``phi`` and
            ``g_values`` disagree in length, or a valley/quality value is not a
            finite number.
    """
    if status not in LUT_STATUSES:
        raise ValueError(
            f"LUT status 非法: {status!r}，只允许 {LUT_STATUSES}（其他取值没有语义，"
            "加载侧一律拒绝）"
        )
    if wavelength_nm is None and gray_for_2pi is None:
        raise ValueError(
            "meta 必须记录波长或名义 2π 灰度至少一项"
            "（相位→灰度映射随波长变化，缺了它无法判断能否复用）"
        )
    g_values = np.asarray(g_values)
    phi = np.asarray(phi)
    inverse_gray = np.asarray(inverse_gray)
    if g_values.ndim != 1 or phi.ndim != 1 or g_values.shape != phi.shape:
        raise ValueError(
            f"g_values 与 phi 必须是一维等长数组，收到 "
            f"g_values={g_values.shape} phi={phi.shape}"
        )
    if inverse_gray.ndim != 1:
        raise ValueError(
            f"inverse_gray 必须是一维数组，收到 shape={inverse_gray.shape}"
        )

    merged: dict[str, Any] = {}
    if quality:
        for key, value in quality.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"quality[{key!r}] 必须是数值，收到 {value!r}")
            merged[str(key)] = float(value)
    if not isinstance(valley, (int, float)) or isinstance(valley, bool):
        raise ValueError(f"valley 必须是数值（谷值），收到 {valley!r}")
    if not np.isfinite(float(valley)):
        raise ValueError(f"valley 必须是有限数值，收到 {valley!r}")
    merged["valley"] = float(valley)

    meta: dict[str, Any] = {
        "schema": LUT_META_SCHEMA_VERSION,
        "status": status,
        "serial": normalize_slm_serial(serial),
        "created_utc": created_utc or _utc_now_iso(),
        "source": str(source),
        "wavelength_nm": None if wavelength_nm is None else float(wavelength_nm),
        "gray_for_2pi": None if gray_for_2pi is None else int(gray_for_2pi),
        "n_points": int(g_values.size),
        "inverse_length": int(inverse_gray.size),
        "quality": merged,
        "notes": str(notes),
    }
    if extra:
        meta["extra"] = dict(extra)
    return meta


def write_lut_meta(dir_path: Path, meta: Mapping[str, Any]) -> Path:
    """Atomically write ``meta.json`` into ``dir_path`` after validating it.

    Args:
        dir_path: LUT directory (named by the serial the meta is bound to).
        meta: Payload from :func:`build_lut_meta`.

    Returns:
        The sidecar path.

    Raises:
        ValueError: If the payload fails :func:`_check_meta_shape` (required
            keys, status, serial/directory binding, finite valley).
    """
    payload = dict(meta)
    _check_meta_shape(payload, lut_dir=Path(dir_path))
    _atomic_write(
        Path(dir_path) / LUT_META_FILENAME,
        lambda handle: handle.write(
            (
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        ),
    )
    return Path(dir_path) / LUT_META_FILENAME


def _check_meta_shape(meta: Mapping[str, Any], *, lut_dir: Path) -> dict[str, Any]:
    """Shared structural validation of a ``meta.json`` payload."""
    missing = [key for key in LUT_META_REQUIRED_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{LUT_META_FILENAME} 缺必填字段: {missing} ({lut_dir})")
    status = meta["status"]
    if status not in LUT_STATUSES:
        raise ValueError(
            f"{LUT_META_FILENAME} status 非法: {status!r}，只允许 {LUT_STATUSES}"
        )
    serial = normalize_slm_serial(meta["serial"])
    if serial != lut_dir.name:
        raise ValueError(
            f"{LUT_META_FILENAME} 序列号与目录不匹配: meta={serial!r} "
            f"目录={lut_dir.name!r}（LUT 随面板绑定，不允许把 sidecar 复制到别台设备）"
        )
    quality = meta["quality"]
    if not isinstance(quality, dict) or "valley" not in quality:
        raise ValueError(
            f"{LUT_META_FILENAME} 的 quality 必须记录 valley（反演谷底）: {quality!r}"
        )
    if not isinstance(quality["valley"], (int, float)) or isinstance(
        quality["valley"], bool
    ):
        raise ValueError(f"quality.valley 必须是数值，收到 {quality['valley']!r}")
    if meta["wavelength_nm"] is None and meta["gray_for_2pi"] is None:
        raise ValueError(
            f"{LUT_META_FILENAME} 必须记录 wavelength_nm 或 gray_for_2pi 至少一项"
        )
    return dict(meta)


def read_lut_meta(lut_dir: Path) -> dict[str, Any]:
    """Read and structurally validate a LUT directory's ``meta.json``.

    Args:
        lut_dir: LUT directory, named by the SLM serial it belongs to.

    Returns:
        The validated metadata dict.

    Raises:
        LutTrustError: If the sidecar is absent, unreadable, incomplete, has an
            unknown status, or does not match the directory's serial. A missing
            sidecar is treated as **untrusted**, never as "old format, let it
            through".
    """
    lut_dir = Path(lut_dir)
    meta_path = lut_dir / LUT_META_FILENAME
    if not lut_dir.is_dir():
        raise LutTrustError(f"LUT 目录不存在: {lut_dir}")
    if not meta_path.is_file():
        raise LutTrustError(
            f"LUT 目录缺少 {LUT_META_FILENAME}: {lut_dir}"
            "（没有元数据就无法证明它是哪台设备、哪次测量、质量如何，"
            "因此一律视为不可信，不做向后兼容放行）"
        )
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LutTrustError(
            f"{LUT_META_FILENAME} 无法解析: {meta_path} ({exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise LutTrustError(
            f"{LUT_META_FILENAME} 顶层必须是对象，收到 {type(raw).__name__}"
        )
    try:
        return _check_meta_shape(raw, lut_dir=lut_dir)
    except ValueError as exc:
        raise LutTrustError(str(exc)) from exc


def require_validated_lut(lut_dir: Path) -> dict[str, Any]:
    """Trust gate for the production display path: only ``validated`` loads.

    Args:
        lut_dir: LUT directory handed to ``Santec.load_lut``.

    Returns:
        The metadata of a ``validated`` LUT (for logging provenance).

    Raises:
        LutTrustError: If the sidecar is missing/invalid, or the status is
            ``provisional`` -- a measured-but-unverified inversion. The message
            names the status and the recorded notes so the operator can see
            *why* it was refused rather than hunting for a silent no-op.
    """
    meta = read_lut_meta(lut_dir)
    status = meta["status"]
    if status != LUT_STATUS_VALIDATED:
        raise LutTrustError(
            f"LUT status={status!r}（目录 {lut_dir}，序列号 {meta['serial']}）："
            f"只有 {LUT_STATUS_VALIDATED!r} 的 LUT 才允许进入生产显示路径；"
            f"当前记录的判定依据见 meta.notes={meta['notes']!r}"
        )
    return meta


def write_lut_calibration(
    dir_path: Path,
    g_values: np.ndarray,
    phi: np.ndarray,
    inverse_gray: np.ndarray,
    meta: Mapping[str, Any],
) -> Path:
    """Atomically write a serial-bound LUT directory (arrays **and** sidecar).

    Same file names and field names as :func:`save_lut` (``lut.npz`` with
    ``g_values`` / ``phi`` / ``inverse_gray`` / ``meta``, plus the two CSVs), so
    the production loader contract is unchanged; on top of that it emits
    ``meta.json`` and writes every file through tmp+replace.

    Args:
        dir_path: Target directory, normally from :func:`lut_dir_for_serial`.
        g_values: Measured gray points.
        phi: Recovered phase in radians.
        inverse_gray: Phase -> gray table the driver consumes.
        meta: Payload from :func:`build_lut_meta`; validated before writing.

    Returns:
        ``dir_path``.

    Raises:
        ValueError: If the metadata fails validation (bad status, serial that
            does not match ``dir_path``, missing quality). Nothing is written in
        that case.
    """
    dir_path = Path(dir_path)
    write_lut_meta(dir_path, meta)
    g_values = np.asarray(g_values, dtype=float)
    phi = np.asarray(phi, dtype=float)
    inverse_gray = np.asarray(inverse_gray, dtype=np.uint16)

    _atomic_write(
        dir_path / "lut_forward.csv",
        lambda handle: np.savetxt(
            handle,
            np.column_stack([g_values, phi]),
            delimiter=",",
            header="g,phi",
            comments="",
        ),
    )
    target_phi = np.linspace(0.0, 2 * np.pi, inverse_gray.size)
    _atomic_write(
        dir_path / "lut_inverse.csv",
        lambda handle: np.savetxt(
            handle,
            np.column_stack([target_phi, inverse_gray.astype(float)]),
            delimiter=",",
            header="target_phi,gray",
            comments="",
        ),
    )
    _atomic_write(
        dir_path / "lut.npz",
        lambda handle: np.savez_compressed(
            handle,
            g_values=g_values,
            phi=phi,
            inverse_gray=inverse_gray,
            meta=np.asarray(dict(meta), dtype=object),
            allow_pickle=True,
        ),
    )
    return dir_path


def save_serial_lut(
    *,
    slm_serial: str | None,
    g_values: np.ndarray,
    phi: np.ndarray,
    inverse_gray: np.ndarray,
    source: str,
    status: str = LUT_STATUS_PROVISIONAL,
    valley: float,
    wavelength_nm: float | None = None,
    gray_for_2pi: int | None = None,
    quality: Mapping[str, Any] | None = None,
    notes: str = "",
    created_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
    root_dir: Path | None = None,
) -> Path:
    """Persist one calibration's LUT under its SLM serial directory.

    This is the archive entry point: it refuses to write without a real serial,
    defaults to ``provisional`` (so a measurement cannot become production
    truth by accident), and writes atomically.

    Args:
        slm_serial: SLM serial the LUT is bound to.
        g_values: Measured gray points.
        phi: Recovered phase in radians.
        inverse_gray: Phase -> gray table.
        source: Producing script/command line.
        status: ``provisional`` (default) or ``validated``.
        valley: Key quality number (see :func:`build_lut_meta`).
        wavelength_nm: Working wavelength.
        gray_for_2pi: Nominal 2-pi gray.
        quality: Additional quality metrics.
        notes: Why this LUT is / is not trusted.
        created_utc: Optional timestamp override.
        extra: Additional provenance fields.
        root_dir: Archive root; ``None`` uses CWD-relative ``data/calib/lut``.

    Returns:
        The serial-bound directory that was written.

    Raises:
        LutSerialError: If ``slm_serial`` is empty/unsafe (nothing is written).
        ValueError: If the metadata is inconsistent.
    """
    lut_dir = lut_dir_for_serial(slm_serial, root_dir=root_dir)
    meta = build_lut_meta(
        serial=lut_dir.name,
        source=source,
        g_values=g_values,
        phi=phi,
        inverse_gray=inverse_gray,
        status=status,
        wavelength_nm=wavelength_nm,
        gray_for_2pi=gray_for_2pi,
        valley=valley,
        quality=quality,
        notes=notes,
        created_utc=created_utc,
        extra=extra,
    )
    return write_lut_calibration(lut_dir, g_values, phi, inverse_gray, meta)


def mark_lut_validated(
    lut_dir: Path,
    *,
    verified_by: str,
    notes: str,
    verified_utc: str | None = None,
) -> dict[str, Any]:
    """Promote a LUT from ``provisional`` to ``validated``, explicitly.

    Verification is a claim about a measurement, so it has to name who/what did
    it and why the numbers are now believed. Read-modify-write keeps the
    original provenance and rewrites the sidecar atomically.

    Args:
        lut_dir: The serial-bound LUT directory.
        verified_by: Person / procedure / gate that verified it.
        notes: What was verified (and against what evidence).
        verified_utc: Optional timestamp override.

    Returns:
        The new metadata.

    Raises:
        LutTrustError: If the existing sidecar is missing or invalid -- there is
            no way to validate an unknown LUT.
        ValueError: If ``verified_by`` or ``notes`` is empty.
    """
    lut_dir = Path(lut_dir)
    if not str(verified_by).strip():
        raise ValueError("verified_by 不能为空：验证必须署名（人或判据流程）")
    if not str(notes).strip():
        raise ValueError("notes 不能为空：必须写明验证依据")
    meta = read_lut_meta(lut_dir)
    meta["status"] = LUT_STATUS_VALIDATED
    meta["verified_utc"] = verified_utc or _utc_now_iso()
    meta["verified_by"] = str(verified_by).strip()
    meta["notes"] = (
        f"{meta['notes']} | 验证: {str(verified_by).strip()}: {str(notes).strip()}".strip()
    )
    write_lut_meta(lut_dir, meta)
    return meta
