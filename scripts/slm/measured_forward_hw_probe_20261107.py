"""Hardware acquisition probe for issue #237: measured forward, stray light, LUT, registration.

This script is the *on-bench* companion to the offline analysis already archived for
issue #237 (``registration_gain_probe_20261107.py``).  It owns one thing: a single,
serialised, fully-recorded acquisition plan that produces the four data products the
offline models are still missing.

Stages (fixed order, one device owner at a time)
------------------------------------------------
``dark_field``
    Laser **off**, SLM displaying nothing but a blank frame, multi-exposure bracket,
    ``--frames-per-level`` raw frames per level.  Gives the read-out/dark floor that
    every normalised prediction is compared against.
``stray_light_blank``
    Laser **on**, SLM at zero phase (flat panel), multi-exposure bracket.  Any signal
    here is unmodulated leakage / stray light, i.e. the floor the forward model cannot
    explain.
``lut_bracket``
    Per branch (U / C) the archived ``reference_start`` panel phase is held *fixed*
    while the exposure is stepped through a multiplicative bracket
    (``--exposure-factors``, >= 4 levels), repeated ``--frames-per-level`` times.
    Rows are written as ``{branch, exposure_ms, frame, panel_phase_hash}`` so the
    CCD-response LUT can be fitted straight off ``manifest.json``.
``registration``
    One frame per branch at the same ``reference_start`` phase, then the *same*
    estimator the offline probe used: +-4 px integer Pearson grid on the frozen M3
    prediction plus a separable three-point parabola for subpixel.  Results are
    compared against the archived pooled values and the deviation is reported.

Device discipline (仓库 AGENTS.md 必读项)
-----------------------------------------
* **Without ``--execute`` this script touches no hardware at all.**  It only reads
  archived JSON/NPZ from disk, prints the plan, and writes the plan artefacts.  No
  ``ao_shaping.drivers`` module is imported -- the run config records the check
  result, not an assumption.
* **With ``--execute``** the driver imports are lazy, the cross-process device lock
  is held for the whole acquisition, devices are opened serially (never a second
  process on the same device), and acquire/release plus open/close are logged.
* There is **no laser/shutter driver in this repository**, so the laser-off stage is
  operator-gated: the script stops, asks, and only then captures.  It never claims to
  have switched a laser it cannot address.
* A Santec controller that answers ``ReadSU = -1000`` is wedged.  That is treated as
  fatal: acquisition stops immediately and the operator is told to power-cycle the
  controller (no retry, no further frames).

Outputs (all under ``--out``, default
``data/debug/slm_transfer/measured_forward_hw_probe_20261107/``)
-------------------------------------------------------------------------------
``manifest.json``  one row per captured frame (dry-run: one row per *planned* shot)
``run_config.json`` resolved settings, the plan, device statement, provenance
``results.json``   stage summaries, LUT rows, registration estimates vs offline
``REPORT.md``      human readout, including the device-occupation statement
``run.log``        timestamped log of every step and every acquire/release
``frames/``        executed runs only: ``sample_XXXX.npz`` with ``raw_frames``/
                   ``frame``/``panel_phase`` in the archived dataset schema
``phases/``        executed runs and the dry-run plan: the reference panel phases

Determinism: CPU only, fixed grid, fixed seeds are not needed (no stochastic fitting).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --------------------------------------------------------------------------- #
# archived inputs (read-only; this script never writes outside ``--out``)
# --------------------------------------------------------------------------- #
BASE_DIR = ROOT / "data/debug/slm_transfer"
R2_DIR = BASE_DIR / "defocus_forward_20261107_focus_r2"
FIT_PATH = BASE_DIR / "defocus_forward_20261107_focus_r2_fit/forward_fit.json"
OFFLINE_PROBE_RESULTS = BASE_DIR / "registration_gain_probe_20261107/results.json"
TRANSFER_SCRIPT = ROOT / "scripts/slm/gated_recurrent_transfer_20261107.py"
DEFAULT_OUT = BASE_DIR / "measured_forward_hw_probe_20261107"

MANIFEST_NAME = "manifest.json"
RUN_CONFIG_NAME = "run_config.json"
RESULTS_NAME = "results.json"
REPORT_NAME = "REPORT.md"
LOG_NAME = "run.log"

# --------------------------------------------------------------------------- #
# fixed method constants (mirrored from the offline probe -- do not drift)
# --------------------------------------------------------------------------- #
BRANCHES = ("U", "C")
PHASE_SIGN = -1.0  # the sign the archived fit was trained/evaluated with
COARSE_HALF = 4  # +-4 px integer grid, as in registration_gain_probe
COARSE_EXPAND = 4  # expand only while the optimum sits on the border ...
COARSE_MAX = 12  # ... and never past +-12 px
SHIFT_ORDER = 1  # bilinear, no prefilter
CROP_STEP = 2
CROP_HALF = 120
GRID = 128
DARK_EXPOSURES_MS = (0.2, 1.0, 5.0)
BLANK_EXPOSURES_MS = (0.2, 1.0, 5.0)
# the two floor brackets deliberately share levels: stray light is only computable as
# ``blank - dark`` at a *matched* exposure, see ``floor_summary``
EXPOSURE_FACTORS = (0.5, 1.0, 2.0, 4.0)
FRAMES_PER_LEVEL = 2
REFERENCE_LABEL = "reference_start"
SLM_CONTROLLER_HANG_CODE = -1000
DISPLAY_WAIT_BLANK_S = 0.5
DISPLAY_WAIT_PHASE_S = 0.8
SATURATION_COUNTS = 255.0

DEVICE_STATEMENT_OFFLINE = (
    "本次为 dry-run：未占用任何设备 —— 未打开 SLM、未打开相机、未启动或停止激光、"
    "未向任何面板下发相位；未 import 任何 ``ao_shaping.drivers`` 模块（该结论由运行时"
    "对 ``sys.modules`` 的实际检查记录在 run_config.json 的 drivers_import 字段）。"
    "全部数字来自磁盘存档 JSON/NPZ。"
)
DEVICE_STATEMENT_EXECUTE = (
    "本次为 --execute：独占 SLM #{slm_number} 与相机 {cam_type}#{cam_id}；"
    "跨进程设备锁在整段采集期间持有，打开/关闭与加锁/放锁逐条写入 run.log；"
    "激光状态由操作员确认（仓库无激光/快门驱动，脚本不假装自动关闭激光）。"
)
HANG_HINT = (
    "SLM 控制器 ReadSU 返回 -1000（SLM_OTHER_ERROR）：控制器已挂起，面板不再应答。"
    "继续采集只会得到未调制的光。处置：**物理断电重启 SLM 控制器**后再重跑；"
    "强杀本进程不会走 finally close，控制器可能残留异常态。"
)


# --------------------------------------------------------------------------- #
# small utilities: time, hashing, json, logging
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _panel_fingerprint(phase: np.ndarray) -> dict[str, Any]:
    """Identify exactly what the SLM is being told to display.

    The hash is over the float32 radians actually handed to
    ``Santec.create_phase_from_array`` -- i.e. the display content, not a filename.
    """
    arr = np.ascontiguousarray(np.asarray(phase, dtype=np.float32))
    return {
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shape": [int(v) for v in arr.shape],
        "dtype": "float32_radians",
        "nbytes": int(arr.nbytes),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def _parse_floats(text: str) -> tuple[float, ...]:
    """Parse a comma-separated positive-number list from the command line."""
    out = [float(part) for part in str(text).split(",") if part.strip()]
    if not out:
        raise ValueError(f"empty number list: {text!r}")
    if any(value <= 0 for value in out):
        raise ValueError(f"numbers must be positive: {text!r}")
    return tuple(out)


class RunLog:
    """Timestamped log that is mirrored to ``run.log`` and stderr.

    ``run.log`` is opened in append mode and flushed on every line, so an aborted
    run still carries the trace that explains it.
    """

    def __init__(self, path: Path, *, quiet: bool = False):
        self.path = path
        self.quiet = quiet
        self.lines: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._handle = self.path.open("a", encoding="utf-8")

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def log(self, message: str, level: str = "INFO") -> None:
        line = f"{_utc_now()} [{level}] {message}"
        self.lines.append(line)
        self._handle.write(line + "\n")
        self._handle.flush()
        if not self.quiet:
            print(line, file=sys.stderr)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


# --------------------------------------------------------------------------- #
# plan data model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Shot:
    """One planned (or executed) capture."""

    index: int
    stage: str
    branch: str
    label: str
    phase_source: str
    laser: str
    exposure_ms: float
    frame: int
    note: str = ""

    def key(self) -> str:
        return f"{self.stage}/{self.branch}/{self.label}/f{self.frame}"


@dataclass(slots=True)
class StageSpec:
    """Order and operator requirements of the four stages."""

    name: str
    purpose: str
    laser: str
    needs_laser_off: bool
    phase_kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "purpose": self.purpose,
            "laser": self.laser,
            "operator_gate_laser_off": self.needs_laser_off,
            "slm_display": self.phase_kind,
        }


STAGE_SPECS = (
    StageSpec(
        name="dark_field",
        purpose="read-out / dark floor of the camera at several exposures",
        laser="off",
        needs_laser_off=True,
        phase_kind="blank (zero phase, panel held flat)",
    ),
    StageSpec(
        name="stray_light_blank",
        purpose="unmodulated leakage and stray light at zero phase, several exposures",
        laser="on",
        needs_laser_off=False,
        phase_kind="blank (zero phase, flat panel)",
    ),
    StageSpec(
        name="lut_bracket",
        purpose="CCD response LUT: fixed panel phase, multiplicative exposure bracket",
        laser="on",
        needs_laser_off=False,
        phase_kind="archived reference_start phase (branch specific)",
    ),
    StageSpec(
        name="registration",
        purpose="global subpixel registration of one live frame per branch vs the "
        "frozen M3 prediction",
        laser="on",
        needs_laser_off=False,
        phase_kind="archived reference_start phase (branch specific)",
    ),
)


@dataclass(slots=True)
class Settings:
    """Everything the acquisition needs, resolved before any device is touched."""

    out: Path
    slm_number: int
    cam_type: str
    cam_id: int
    wavelength: float
    dark_exposures_ms: tuple[float, ...]
    blank_exposures_ms: tuple[float, ...]
    exposure_factors: tuple[float, ...]
    frames_per_level: int
    registration_exposure_ms: float
    wait_until_idle_s: float
    execute: bool
    assume_laser_off: bool
    skip_selftest: bool
    quiet: bool = False
    print_plan: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": str(self.out),
            "slm_number": self.slm_number,
            "cam_type": self.cam_type,
            "cam_id": self.cam_id,
            "wavelength_nm": self.wavelength,
            "dark_exposures_ms": list(self.dark_exposures_ms),
            "blank_exposures_ms": list(self.blank_exposures_ms),
            "lut_exposure_factors": list(self.exposure_factors),
            "frames_per_level": self.frames_per_level,
            "registration_exposure_ms": self.registration_exposure_ms,
            "wait_until_idle_s": self.wait_until_idle_s,
            "execute": self.execute,
            "assume_laser_off": self.assume_laser_off,
            "skip_selftest": self.skip_selftest,
            "quiet": self.quiet,
            "print_plan": self.print_plan,
            "phase_sign": PHASE_SIGN,
            "registration_method": {
                "integer_grid_half_px": COARSE_HALF,
                "expand_px": COARSE_EXPAND,
                "max_px": COARSE_MAX,
                "interpolation_order": SHIFT_ORDER,
                "subpixel": "separable three-point parabola on the integer grid",
                "score": "full-crop Pearson on the exposure-normalised image",
            },
        }


def _blank_phase(shape: tuple[int, int]) -> np.ndarray:
    """Zero-phase panel content (radians) as the capture scripts display it."""
    return np.zeros(shape, dtype=np.float32)


# --------------------------------------------------------------------------- #
# lazy module loading (torch + frozen forward; never needed for the plan alone)
# --------------------------------------------------------------------------- #
def _load_module(name: str, path: Path) -> Any:
    """Load a script-in-tree as a module (the offline probe's own pattern)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROBE: Any = None


def load_offline_probe() -> Any:
    """Load ``registration_gain_probe_20261107`` and through it the frozen forward.

    Kept lazy and side-effect free: module import only binds constants and loads the
    transfer helpers, it reads no dataset and opens no device.  Importing it is also
    how this script reuses the *exact* registration estimator (grid, tie-breaking,
    border expansion, parabola clip) instead of paraphrasing it.
    """
    global _PROBE
    if _PROBE is None:
        if not OFFLINE_PROBE_RESULTS.parent.is_dir():
            raise FileNotFoundError(f"offline probe archive missing: {OFFLINE_PROBE_RESULTS}")
        _PROBE = _load_module(
            "_measured_forward_registration_probe",
            ROOT / "scripts/slm/registration_gain_probe_20261107.py",
        )
    return _PROBE


def drivers_import_status() -> dict[str, Any]:
    """What the process actually loaded -- a check, not a claim."""
    loaded = sorted(
        name
        for name in sys.modules
        if name == "ao_shaping.drivers" or name.startswith("ao_shaping.drivers.")
    )
    return {
        "drivers_modules_loaded": len(loaded),
        "modules": loaded,
        "device_free": not loaded,
    }


# --------------------------------------------------------------------------- #
# archived reference phases (the content the panel will be told to display)
# --------------------------------------------------------------------------- #
def read_source_rows() -> list[dict[str, Any]]:
    """Read the r2 dataset manifest (read-only)."""
    return list(_read_json(R2_DIR / MANIFEST_NAME))


def resolve_reference_phases(
    settings: Settings, rows: list[dict[str, Any]], log: RunLog
) -> dict[str, dict[str, Any]]:
    """Pick the ``reference_start`` row per branch and fingerprint its panel phase.

    Returns one record per branch with the archived exposure (the anchor of the LUT
    bracket), the archived spot metadata (so the live acquisition can be compared to
    the training condition), and the hash of the exact array that will be displayed.
    """
    refs: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        match = [
            row for row in rows
            if str(row.get("branch")) == branch and str(row.get("label")) == REFERENCE_LABEL
        ]
        if not match:
            raise KeyError(f"no {REFERENCE_LABEL!r} row for branch {branch} in {R2_DIR}")
        if len(match) > 1:
            log.log(f"branch {branch}: {len(match)} reference_start rows, using index 0", "WARN")
        row = match[0]
        npz_path = R2_DIR / str(row["file"])
        with np.load(npz_path) as archive:
            panel_phase = np.asarray(archive["panel_phase"], dtype=np.float32)
            model_phase_shape = [int(v) for v in np.asarray(archive["model_phase"]).shape]
        fingerprint = _panel_fingerprint(panel_phase)
        is_zero = bool(np.all(panel_phase == 0.0))
        saved: dict[str, Any] = {"written": False}
        if not settings.execute:
            # dry-run: still pin the content to disk so the hash is auditable
            dest = settings.out / "phases" / f"{branch}_{REFERENCE_LABEL}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, panel_phase=panel_phase)
            saved = {"written": True, "path": str(dest.relative_to(settings.out))}
        spot = row.get("spot") or {}
        refs[branch] = {
            "branch": branch,
            "file": str(row["file"]),
            "label": REFERENCE_LABEL,
            "npz": str(npz_path.relative_to(ROOT)),
            "exposure_ms": float(row["exposure_ms"]),
            "exposure_status": str(row.get("exposure_status", "")),
            "utc_start": str(row.get("utc_start", "")),
            "panel_shape": fingerprint["shape"],
            "panel_dtype": "float32_radians",
            "model_phase_shape": model_phase_shape,
            "panel_is_zero": is_zero,
            "panel_phase_hash": fingerprint["sha256"],
            "panel_phase_stats": {
                "min": fingerprint["min"],
                "max": fingerprint["max"],
                "mean": fingerprint["mean"],
            },
            "fingerprint": fingerprint,
            "archived_spot": {
                "peak": spot.get("peak"),
                "fwhm_px": spot.get("fwhm_px"),
                "centroid_x": spot.get("centroid_x"),
                "centroid_y": spot.get("centroid_y"),
                "hollowness": spot.get("hollowness"),
            },
            "saved_phase": saved,
        }
        log.log(
            f"reference phase {branch}: {row['file']} exposure={row['exposure_ms']}ms "
            f"shape={fingerprint['shape']} zero={is_zero} "
            f"sha256={fingerprint['sha256'][:16]}…"
        )
    return refs


def read_offline_registration() -> dict[str, Any]:
    """Archived pooled registration the live estimate is compared against."""
    payload = _read_json(OFFLINE_PROBE_RESULTS)
    by_sign = payload["by_sign"]
    # the archive keys signs as "-1" (int form), so do not use str(float) here
    key = str(int(PHASE_SIGN)) if float(PHASE_SIGN).is_integer() else str(PHASE_SIGN)
    if key not in by_sign:
        raise KeyError(f"no by_sign[{key!r}] in {OFFLINE_PROBE_RESULTS}; keys={sorted(by_sign)}")
    block = by_sign[key]["registration"]
    return {
        "source": str(OFFLINE_PROBE_RESULTS.relative_to(ROOT)),
        "phase_sign": PHASE_SIGN,
        "scope": block.get("scope"),
        "method": block.get("method"),
        "integer_dy": block["integer_dy"],
        "integer_dx": block["integer_dx"],
        "subpixel_dy": block["subpixel_dy"],
        "subpixel_dx": block["subpixel_dx"],
        "integer_peak_score": block["integer_peak_score"],
    }


def crop_grid(probe: Any, forward_and_offset: Any, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixel index grids that turn a full CCD frame into the model-space crop.

    Identical to the offline loader: ``camera_xy = offset + center`` sampled with
    ``CROP_STEP`` over ``+/-CROP_HALF`` px around the fit's camera crop centre.
    """
    _, offset = forward_and_offset
    center = np.asarray(fit["camera_crop_center"], dtype=np.float64)
    cam = offset + center
    return cam[..., 1].astype(int), cam[..., 0].astype(int)


# --------------------------------------------------------------------------- #
# the acquisition plan
# --------------------------------------------------------------------------- #
def _frozen_panel_shapes(refs: dict[str, dict[str, Any]]) -> tuple[int, int]:
    shapes = {tuple(int(v) for v in refs[b]["panel_shape"]) for b in refs}
    if len(shapes) != 1:
        raise ValueError(f"reference branches disagree on panel shape: {shapes}")
    return next(iter(shapes))


def build_shots(settings: Settings, refs: dict[str, dict[str, Any]]) -> list[Shot]:
    """Flatten the four stages into the exact serial capture order."""
    shots: list[Shot] = []
    repeats = max(1, int(settings.frames_per_level))

    def add(**kwargs: Any) -> None:
        shots.append(Shot(index=len(shots), **kwargs))

    for exposure in settings.dark_exposures_ms:
        for frame in range(1, repeats + 1):
            add(stage="dark_field", branch="-", label="dark", phase_source="blank",
                laser="off", exposure_ms=float(exposure), frame=frame,
                note="laser off, panel blank; dark + read-out floor of this exposure level")

    for exposure in settings.blank_exposures_ms:
        for frame in range(1, repeats + 1):
            add(stage="stray_light_blank", branch="-", label="slm_blank", phase_source="blank",
                laser="on", exposure_ms=float(exposure), frame=frame,
                note="zero phase (flat panel) with laser on; anything above the dark "
                     "floor here is unmodulated leakage / stray light")

    for branch in BRANCHES:
        anchor = float(refs[branch]["exposure_ms"])
        for factor in settings.exposure_factors:
            for frame in range(1, repeats + 1):
                add(stage="lut_bracket", branch=branch, label=f"lut_x{factor:g}",
                    phase_source=f"{branch}:{REFERENCE_LABEL}", laser="on",
                    exposure_ms=anchor * float(factor), frame=frame,
                    note=f"phase held fixed; exposure = {factor:g} x archived anchor "
                         f"{anchor:g} ms, requested value only -- the device readback "
                         "in the manifest row is authoritative")

    for branch in BRANCHES:
        anchor = float(settings.registration_exposure_ms or refs[branch]["exposure_ms"])
        add(stage="registration", branch=branch, label="registration",
            phase_source=f"{branch}:{REFERENCE_LABEL}", laser="on", exposure_ms=anchor,
            frame=1,
            note="single frame at the branch reference phase, fed to the same "
                 "registration estimator as the offline probe")

    return shots


def planned_phase_fingerprints(
    settings: Settings, refs: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Display content per ``phase_source`` string, for the planned manifest rows."""
    shape = _frozen_panel_shapes(refs)
    out: dict[str, dict[str, Any]] = {"blank": _panel_fingerprint(_blank_phase(shape))}
    for branch in BRANCHES:
        out[f"{branch}:{REFERENCE_LABEL}"] = refs[branch]["fingerprint"]
    return out


def build_plan(
    settings: Settings,
    refs: dict[str, dict[str, Any]],
    offline: dict[str, Any],
    shots: list[Shot],
) -> dict[str, Any]:
    """The full plan document: what will be displayed, captured, and recorded."""
    fingerprints = planned_phase_fingerprints(settings, refs)
    stage_summaries: list[dict[str, Any]] = []
    for spec in STAGE_SPECS:
        stage_shots = [s for s in shots if s.stage == spec.name]
        levels = sorted({round(s.exposure_ms, 6) for s in stage_shots})
        stage_summaries.append({
            **spec.as_dict(),
            "n_levels": len(levels),
            "levels_exposure_ms": levels,
            "frames_per_level": settings.frames_per_level,
            "n_frames": len(stage_shots),
            "branches": sorted({s.branch for s in stage_shots}),
        })
    return {
        "generated_utc": _utc_now(),
        "mode": "execute" if settings.execute else "dry_run",
        "device_statement": (
            DEVICE_STATEMENT_EXECUTE.format(
                slm_number=settings.slm_number,
                cam_type=settings.cam_type,
                cam_id=settings.cam_id,
            ) if settings.execute else DEVICE_STATEMENT_OFFLINE
        ),
        "stages": stage_summaries,
        "capture_order": [dataclasses.asdict(shot) for shot in shots],
        "total_frames": len(shots),
        "display_content": {
            key: {
                "panel_phase_hash": value["sha256"],
                "shape": value["shape"],
                "min": value["min"],
                "max": value["max"],
                "mean": value["mean"],
            }
            for key, value in fingerprints.items()
        },
        "reference_phases": {
            branch: {k: v for k, v in record.items() if k != "fingerprint"}
            for branch, record in refs.items()
        },
        "exposure_policy": {
            "anchor": "archived reference_start exposure per branch (multiplicative bracket)",
            "requested_vs_effective": (
                "planned exposure values are *requests*; every manifest row stores the "
                "device readback from get_camera_exposure_ms as exposure_ms, and the "
                "request separately"
            ),
            "range_check": "levels outside the camera's read-back exposure range are "
                           "clamped by the driver and recorded as clamped",
            "saturation_guard": "any frame whose max reaches 255 counts is recorded "
                                "saturated; LUT fitting must drop saturated levels",
            "settle": "exposure changes go through set_camera_exposure_ms, which on "
                      "Daheng consumes the stale frames the SDK leaves in the pipeline "
                      "(driver invariant), so the first frame of a level is that level",
        },
        "laser_policy": {
            "driver_in_repo": False,
            "handling": "operator gate: the script stops and asks before the dark "
                        "stage and again before resuming the laser",
            "non_interactive": (
                "without a TTY and without --assume-laser-off the dark_field stage is "
                "skipped and recorded as skipped, never faked"
            ),
        },
        "registration": {
            "method": offline["method"],
            "phase_sign": PHASE_SIGN,
            "model": "frozen M3_illumination forward from "
                     f"{FIT_PATH.relative_to(ROOT)}",
            "scoring_image": "exposure-normalised crop (counts/ms), same convention as "
                             "the archived dataset loader",
            "offline_reference": offline,
            "note": (
                "a single live frame is compared numerically against pooled training-set "
                "values; that is a sanity comparison, not a statistical equivalence claim"
            ),
        },
        "operator_actions": [
            "confirm the laser is off and safe, then acknowledge the dark-field stage",
            "confirm the laser is back on (shutter/interlock as usual), then acknowledge",
            "keep the SLM controller powered and unoccupied for the whole run; the "
            "device lock is held until capture finishes",
            "if the log reports ReadSU=-1000, stop: power-cycle the SLM controller "
            "before any retry",
        ],
    }


# --------------------------------------------------------------------------- #
# model side: frozen forward, prediction, crop, registration estimator
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ScoreFrame:
    """The two fields ``estimate_global_shift`` actually reads."""

    key: str
    image: np.ndarray


@dataclass(slots=True)
class ModelSide:
    """Frozen forward + its sampling grids, built once per process."""

    probe: Any
    fit: dict
    forward: Any
    offset: np.ndarray
    cam_y: np.ndarray
    cam_x: np.ndarray
    camera_crop_center: list[float]

    @property
    def crop_shape(self) -> list[int]:
        return [int(self.cam_y.shape[0]), int(self.cam_y.shape[1])]


def build_model_side(log: RunLog) -> ModelSide:
    probe = load_offline_probe()
    fit = probe._TRANSFER.load_fit()
    forward, offset = probe._TRANSFER.make_forward(fit, PHASE_SIGN)
    cam_y, cam_x = crop_grid(probe, (forward, offset), fit)
    log.log(
        f"frozen M3 forward loaded: sign={PHASE_SIGN} "
        f"crop={int(cam_y.shape[0])}x{int(cam_y.shape[1])} step={CROP_STEP} "
        f"center={list(np.asarray(fit['camera_crop_center'], dtype=float))}"
    )
    return ModelSide(
        probe=probe,
        fit=fit,
        forward=forward,
        offset=offset,
        cam_y=cam_y,
        cam_x=cam_x,
        camera_crop_center=[float(v) for v in np.asarray(fit["camera_crop_center"], dtype=float)],
    )


def predict_phase(model: ModelSide, panel_phase: np.ndarray) -> np.ndarray:
    """Frozen-forward prediction for a full panel phase (model-space crop)."""
    transfer = model.probe._TRANSFER
    grid = np.asarray(panel_phase, dtype=np.float32)
    if grid.shape == (GRID, GRID):
        phase = grid
    else:
        phase = transfer.resample_panel(grid)
    return np.asarray(model.probe._predict(model.forward, phase), dtype=np.float32)


def crop_frame(model: ModelSide, frame: np.ndarray) -> np.ndarray:
    """Sample a full CCD frame onto the model-space crop grid."""
    return np.asarray(frame, dtype=np.float64)[model.cam_y, model.cam_x].astype(np.float32)


def archived_crop(
    model: ModelSide, npz_path: Path, exposure_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean_uint8_frame, exposure_normalised_crop)`` from an archived npz."""
    with np.load(npz_path) as archive:
        raw = np.asarray(archive["raw_frames"], dtype=np.float64)
    mean_frame = raw.mean(axis=0)
    crop = mean_frame[model.cam_y, model.cam_x] / float(exposure_ms)
    return mean_frame.astype(np.uint8), crop.astype(np.float32)


def estimate_registration(
    model: ModelSide, pairs: list[tuple[str, np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Run the offline probe's estimator unchanged on ``(name, pred, image)`` triples.

    ``pairs`` with more than one entry is scored as a single global translation over
    all of them, exactly as the pooled offline estimate was.
    """
    frames = [ScoreFrame(key=name, image=image) for name, _pred, image in pairs]
    preds = {name: pred for name, pred, _image in pairs}
    out = model.probe.estimate_global_shift(frames, preds)
    out["n_frames"] = len(pairs)
    out["frame_keys"] = [name for name, _p, _i in pairs]
    return out


def registration_comparison(
    estimate: dict[str, Any], offline: dict[str, Any]
) -> dict[str, Any]:
    """Deviation of a live/pooled estimate from the archived pooled values."""
    return {
        "integer_dy": estimate["integer_dy"],
        "integer_dx": estimate["integer_dx"],
        "subpixel_dy": estimate["subpixel_dy"],
        "subpixel_dx": estimate["subpixel_dx"],
        "integer_peak_score": estimate["integer_peak_score"],
        "offline_integer": [offline["integer_dy"], offline["integer_dx"]],
        "offline_subpixel": [offline["subpixel_dy"], offline["subpixel_dx"]],
        "delta_subpixel_dy": float(estimate["subpixel_dy"] - offline["subpixel_dy"]),
        "delta_subpixel_dx": float(estimate["subpixel_dx"] - offline["subpixel_dx"]),
        "delta_integer_dy": int(estimate["integer_dy"] - offline["integer_dy"]),
        "delta_integer_dx": int(estimate["integer_dx"] - offline["integer_dx"]),
        "delta_peak_score": float(estimate["integer_peak_score"] - offline["integer_peak_score"]),
        "search_grid_points": estimate.get("search_grid_points"),
        "search_half_px": estimate.get("search_half_px"),
    }


def offline_selftest(
    settings: Settings, refs: dict[str, dict[str, Any]], offline: dict[str, Any], log: RunLog
) -> dict[str, Any]:
    """Validate the estimator that ``--execute`` will use, on archived frames.

    This is the device-free half of the registration work: the live path scores one
    captured frame the same way, so any systematic offset between live and archived
    numbers is attributable to the frame, not to two different estimators.
    """
    model = build_model_side(log)
    per_branch: dict[str, Any] = {}
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for branch in BRANCHES:
        record = refs[branch]
        npz_path = ROOT / record["npz"]
        with np.load(npz_path) as archive:
            panel_phase = np.asarray(archive["panel_phase"], dtype=np.float32)
        pred = predict_phase(model, panel_phase)
        _mean, image = archived_crop(model, npz_path, record["exposure_ms"])
        pairs.append((f"{branch}_archived", pred, image))
        estimate = estimate_registration(model, [(f"{branch}_archived", pred, image)])
        per_branch[branch] = {
            "npz": record["npz"],
            "panel_phase_hash": record["panel_phase_hash"],
            "exposure_ms": record["exposure_ms"],
            "prediction_shape": [int(pred.shape[0]), int(pred.shape[1])],
            "prediction_stats": {
                "min": float(pred.min()),
                "max": float(pred.max()),
                "mean": float(pred.mean()),
            },
            "image_stats": {
                "min": float(image.min()),
                "max": float(image.max()),
                "mean": float(image.mean()),
            },
            "pearson_zero_shift": float(model.probe._pearson(pred, image)),
            "estimate": estimate,
            "comparison": registration_comparison(estimate, offline),
        }
        log.log(
            f"selftest {branch}: integer=({estimate['integer_dy']},{estimate['integer_dx']}) "
            f"subpixel=({estimate['subpixel_dy']:+.3f},{estimate['subpixel_dx']:+.3f}) "
            f"peak={estimate['integer_peak_score']:.4f}"
        )
    pooled = estimate_registration(model, pairs)
    pooled_comparison = registration_comparison(pooled, offline)
    log.log(
        f"selftest pooled(2): integer=({pooled['integer_dy']},{pooled['integer_dx']}) "
        f"subpixel=({pooled['subpixel_dy']:+.3f},{pooled['subpixel_dx']:+.3f}) "
        f"peak={pooled['integer_peak_score']:.4f}"
    )
    return {
        "status": "ok",
        "scope": "the two archived reference_start frames (U, C), same estimator as --execute",
        "model": {
            "fit": str(FIT_PATH.relative_to(ROOT)),
            "phase_sign": PHASE_SIGN,
            "crop_shape": model.crop_shape,
            "camera_crop_center": model.camera_crop_center,
            "grid": GRID,
            "crop_step": CROP_STEP,
            "crop_half": CROP_HALF,
        },
        "archived_reference": offline,
        "per_branch": per_branch,
        "comparison_rows": {
            **{branch: per_branch[branch]["comparison"] for branch in BRANCHES},
            "pooled 2 frames": pooled_comparison,
        },
        "pooled_two_frame": {
            "estimate": pooled,
            "comparison": pooled_comparison,
            "caveat": "2 frames, not the 60-frame pooled training split; agreement is "
                      "directional, not a reproduction of the archived value",
        },
    }


# --------------------------------------------------------------------------- #
# analysis over manifest rows (works for planned rows and for measured rows)
# --------------------------------------------------------------------------- #
def _frame_stats(frame: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(frame)
    peak = float(np.max(arr)) if arr.size else None
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()) if arr.size else None,
        "max": peak,
        "min": float(arr.min()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "saturated": bool(peak is not None and peak >= SATURATION_COUNTS),
    }


def planned_row(
    shot: Shot, fingerprints: dict[str, dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    """A dry-run manifest row: the same schema as a measured row, nothing invented."""
    return {
        "index": shot.index,
        "file": None,
        "status": "planned",
        "stage": shot.stage,
        "branch": shot.branch,
        "label": shot.label,
        "phase_source": shot.phase_source,
        "laser": shot.laser,
        "frame": shot.frame,
        "exposure_ms": float(shot.exposure_ms),
        "exposure_requested_ms": float(shot.exposure_ms),
        "exposure_source": "planned (not read back from the device)",
        "exposure_clamped": None,
        "exposure_status": "planned",
        "utc_start": None,
        "monotonic_start_ns": None,
        "monotonic_end_ns": None,
        "display_wait_s": (
            DISPLAY_WAIT_PHASE_S if shot.phase_source != "blank" else DISPLAY_WAIT_BLANK_S
        ),
        "panel_phase": fingerprints[shot.phase_source],
        "frame_stats": None,
        "crop_stats": None,
        "spot": None,
        "note": shot.note,
    }


def lut_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows ready for LUT fitting: ``{branch, exposure_ms, frame, panel_phase_hash}``.

    One row per captured (or planned) frame, in the archived dataset's units so the
    fit can read ``manifest.json`` directly.  ``mean_counts_ms`` is the exposure
    normalised crop mean, which is the quantity the forward model is trained on.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("stage") != "lut_bracket":
            continue
        stats = row.get("frame_stats") or {}
        crop = row.get("crop_stats") or {}
        exposure = float(row["exposure_ms"])
        mean_counts = crop.get("mean_counts", stats.get("mean"))
        per_ms = crop.get("mean_counts_ms")
        out.append({
            "branch": row["branch"],
            "exposure_ms": exposure,
            "exposure_readback_ms": exposure,
            "exposure_requested_ms": row.get("exposure_requested_ms"),
            "exposure_status": row.get("exposure_status"),
            "frame": row["frame"],
            "panel_phase_hash": row["panel_phase"]["sha256"],
            "panel_phase_shape": row["panel_phase"]["shape"],
            "label": row["label"],
            "status": row.get("status"),
            "mean_counts": mean_counts,
            "mean_counts_per_ms": (
                per_ms if per_ms is not None
                else (None if mean_counts is None or exposure <= 0 else mean_counts / exposure)
            ),
            "crop_peak_counts": crop.get("max_counts"),
            "peak_counts": stats.get("max"),
            "saturated": stats.get("saturated"),
            "file": row.get("file"),
            "utc_start": row.get("utc_start"),
        })
    return out


def lut_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Linearity read-out of the bracket: per-level means, log-log slope, fit recipe."""
    per_branch: dict[str, Any] = {}
    for branch in BRANCHES:
        selected = [row for row in lut_rows(rows) if row["branch"] == branch]
        levels: dict[float, dict[str, Any]] = {}
        for row in selected:
            bucket = levels.setdefault(row["exposure_ms"], {
                "exposure_ms": row["exposure_ms"],
                "panel_phase_hash": row["panel_phase_hash"],
                "frames": [],
                "phase_hashes": [],
                "saturated": [],
            })
            bucket["frames"].append(row["frame"])
            bucket["phase_hashes"].append(row["panel_phase_hash"])
            bucket["saturated"].append(bool(row["saturated"]))
            if row["mean_counts"] is not None:
                bucket.setdefault("means", []).append(float(row["mean_counts"]))
        table: list[dict[str, Any]] = []
        for exposure in sorted(levels):
            bucket = levels[exposure]
            means = bucket.get("means", [])
            panel_hashes = sorted(set(bucket["phase_hashes"]))
            table.append({
                "exposure_ms": exposure,
                "n_frames": len(bucket["frames"]),
                "frames": bucket["frames"],
                "phase_constant": len(panel_hashes) == 1,
                "panel_phase_hash": panel_hashes[0] if panel_hashes else None,
                "mean_counts": float(np.mean(means)) if means else None,
                "std_counts": float(np.std(means)) if len(means) > 1 else None,
                "any_saturated": any(bucket["saturated"]),
            })
        usable = [t for t in table if t["mean_counts"] is not None and not t["any_saturated"]]
        slope = correlation = None
        if len(usable) >= 2:
            x = np.log(np.asarray([t["exposure_ms"] for t in usable], dtype=float))
            y = np.log(np.asarray([t["mean_counts"] for t in usable], dtype=float))
            slope = float(np.polyfit(x, y, 1)[0])
            correlation = float(np.corrcoef(x, y)[0, 1]) if len(usable) > 1 else None
        per_branch[branch] = {
            "levels": table,
            "n_levels": len(table),
            "n_usable_levels": len(usable),
            "saturated_levels": [t["exposure_ms"] for t in table if t["any_saturated"]],
            "phase_constant_across_bracket": all(t["phase_constant"] for t in table),
            "loglog_slope": slope,
            "loglog_pearson": correlation,
            "interpretation": (
                "log-log slope 1.0 == linear response; <1 saturates, >1 suggests "
                "background-dominated low levels (subtract the dark floor first)"
            ),
        }
    return {
        "rows": lut_rows(rows),
        "per_branch": per_branch,
        "fit_recipe": (
            "fit response(exposure) per branch on mean_counts (or the crop image) with "
            "panel_phase_hash held constant; drop saturated levels; the dark stage rows "
            "give the offset to subtract before fitting"
        ),
    }


def floor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Dark floor vs zero-phase leakage, matched by exposure level."""
    def collect(stage: str) -> dict[float, list[dict[str, Any]]]:
        out: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("stage") != stage:
                continue
            out.setdefault(float(row["exposure_ms"]), []).append(row)
        return out

    dark, blank = collect("dark_field"), collect("stray_light_blank")
    table: list[dict[str, Any]] = []
    for exposure in sorted(set(dark) | set(blank)):
        d_rows, b_rows = dark.get(exposure, []), blank.get(exposure, [])
        d_stats = [r["frame_stats"] for r in d_rows if r.get("frame_stats")]
        b_stats = [r["frame_stats"] for r in b_rows if r.get("frame_stats")]
        dark_mean = float(np.mean([s["mean"] for s in d_stats])) if d_stats else None
        blank_mean = float(np.mean([s["mean"] for s in b_stats])) if b_stats else None
        table.append({
            "exposure_ms": exposure,
            "dark_frames": len(d_rows),
            "blank_frames": len(b_rows),
            "dark_mean_counts": dark_mean,
            "dark_peak_counts": max([s["max"] for s in d_stats], default=None),
            "blank_mean_counts": blank_mean,
            "blank_peak_counts": max([s["max"] for s in b_stats], default=None),
            "stray_mean_counts": (
                None if dark_mean is None or blank_mean is None else blank_mean - dark_mean
            ),
            "matched": bool(d_rows) and bool(b_rows),
        })
    return {
        "levels": table,
        "note": (
            "stray light is blank(zero phase, laser on) minus dark(laser off) at the "
            "same exposure; levels without both brackets are reported, not subtracted"
        ),
    }


def stage_status_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("stage"))] = counts.get(str(row.get("stage")), 0) + 1
    return {
        "n_rows": len(rows),
        "frames_per_stage": counts,
        "n_saturated": sum(
            1 for row in rows
            if (row.get("frame_stats") or {}).get("saturated")
        ),
    }


# --------------------------------------------------------------------------- #
# REPORT.md
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("—" if cell is None else str(cell) for cell in row) + " |")
    out.append("")
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _short(digest: Any) -> str:
    return f"`{str(digest)[:12]}`" if digest else "—"


def render_report(
    settings: Settings,
    plan: dict[str, Any],
    results: dict[str, Any],
    rows: list[dict[str, Any]],
    files: list[str],
) -> str:
    measured = settings.execute
    lines: list[str] = []
    add = lines.append

    add("# Measured-forward hardware probe — issue #237 (`20261107`)")
    add("")
    add(f"- 模式：**{'--execute 上机采集' if measured else 'device-free dry-run（仅计划 + 离线自检）'}**")
    add(f"- 生成时间：{results['generated_utc']}")
    add(f"- 输出目录：`{settings.out}`")
    add("")
    add("## 1. 设备占用声明")
    add("")
    add(f"> {results['device_statement']}")
    add("")
    drivers = results["drivers_import"]
    add(f"- 运行时检查 `sys.modules`：`ao_shaping.drivers*` 已加载模块数 = "
        f"**{drivers['drivers_modules_loaded']}**，device-free = **{drivers['device_free']}**"
        + (f"（{', '.join(drivers['modules'])}）" if drivers["modules"] else ""))
    if measured:
        device = results.get("device") or {}
        add(f"- 设备锁：acquire `{device.get('lock', {}).get('acquired_utc', '—')}` → "
            f"release `{device.get('lock', {}).get('released_utc', '—')}`，"
            f"持锁 `{_fmt(device.get('lock', {}).get('held_s'), 1)} s`，"
            f"释放成功 = `{device.get('lock', {}).get('released', '—')}`")
        add(f"- 打开 / 关闭：SLM #{settings.slm_number} "
            f"`{device.get('slm', {}).get('opened_utc', '—')}` → "
            f"`{device.get('slm', {}).get('closed_utc', '—')}`"
            f"（正常关闭 = `{device.get('slm', {}).get('closed_cleanly', '—')}）`；"
            f"相机 `{settings.cam_type}#{settings.cam_id}` "
            f"`{device.get('camera', {}).get('opened_utc', '—')}` → "
            f"`{device.get('camera', {}).get('closed_utc', '—')}`"
            f"（正常关闭 = `{device.get('camera', {}).get('closed_cleanly', '—')}）`")
        add("- 激光：无驱动可寻址，状态由操作员在 gate 处确认，逐条记录于 "
            "`results.json → device.laser_gates`。")
    add("")

    add("## 2. 输入来源（只读）")
    add("")
    lines.extend(_md_table(
        ["用途", "路径", "指纹 / 值"],
        [
            ["上机数据集 manifest", f"`{R2_DIR.relative_to(ROOT)}/{MANIFEST_NAME}`",
             f"{results['sources']['r2_rows']} 行"],
            ["冻结 forward 拟合", f"`{FIT_PATH.relative_to(ROOT)}`",
             f"phase_sign={PHASE_SIGN}，crop `{results['sources']['crop_shape']}`，"
             f"center {results['sources']['camera_crop_center']}"],
            ["离线配准参考", f"`{OFFLINE_PROBE_RESULTS.relative_to(ROOT)}`",
             f"integer (dy,dx)=({plan['registration']['offline_reference']['integer_dy']},"
             f"{plan['registration']['offline_reference']['integer_dx']})，"
             f"subpixel ({_fmt(plan['registration']['offline_reference']['subpixel_dy'], 3)},"
             f"{_fmt(plan['registration']['offline_reference']['subpixel_dx'], 3)})"],
        ],
    ))
    add("参考相位（将要下发到面板的内容）：")
    add("")
    lines.extend(_md_table(
        ["branch", "npz", "归档曝光 ms", "panel 形状", "相位范围 rad", "panel_phase_hash"],
        [
            [branch, f"`{rec['file']}`", _fmt(rec["exposure_ms"], 3),
             "×".join(str(v) for v in rec["panel_shape"]),
             f"[{_fmt(rec['panel_phase_stats']['min'], 3)}, "
             f"{_fmt(rec['panel_phase_stats']['max'], 3)}]",
             _short(rec["panel_phase_hash"])]
            for branch, rec in plan["reference_phases"].items()
        ],
    ))
    add("相位内容按 float32 弧度数组字节做 SHA-256；空场（blank）为同形状零数组，"
        "其哈希见 `run_config.json → plan.display_content.blank`。")
    add("")

    add("## 3. 采集计划（串行、单设备独占）")
    add("")
    lines.extend(_md_table(
        ["stage", "目的", "激光", "SLM 显示", "曝光档", "每档帧", "帧数"],
        [
            [f"`{stage['stage']}`", stage["purpose"], stage["laser"], stage["slm_display"],
             stage["n_levels"], stage["frames_per_level"], stage["n_frames"]]
            for stage in plan["stages"]
        ] + [["**合计**", "", "", "", "", "", plan["total_frames"]]],
    ))
    add(f"- 曝光口径：{plan['exposure_policy']['requested_vs_effective']}")
    add(f"- 档位切换：{plan['exposure_policy']['settle']}")
    add(f"- 饱和：{plan['exposure_policy']['saturation_guard']}")
    add(f"- 激光：{plan['laser_policy']['handling']}（仓库内无激光/快门驱动 = "
        f"`{plan['laser_policy']['driver_in_repo']}`）；{plan['laser_policy']['non_interactive']}。")
    add("")

    add("## 4. CCD 响应 LUT 记录")
    add("")
    lut = results["lut"]
    lines.extend(_md_table(
        ["branch", "exposure_ms", "frame", "panel_phase_hash", "mean counts",
         "peak counts", "状态"],
        [
            [row["branch"], _fmt(row["exposure_ms"], 4), row["frame"],
             _short(row["panel_phase_hash"]), _fmt(row["mean_counts"], 3),
             _fmt(row["peak_counts"], 1), row["status"]]
            for row in lut["rows"]
        ],
    ))
    add("拟合口径：" + lut["fit_recipe"])
    add("")
    lines.extend(_md_table(
        ["branch", "档数", "可用档", "相位恒定", "log-log 斜率", "log-log Pearson", "饱和档"],
        [
            [branch, block["n_levels"], block["n_usable_levels"],
             block["phase_constant_across_bracket"], _fmt(block["loglog_slope"], 3),
             _fmt(block["loglog_pearson"], 4),
             ", ".join(_fmt(v, 3) for v in block["saturated_levels"]) or "无"]
            for branch, block in lut["per_branch"].items()
        ],
    ))
    if not measured:
        add("dry-run 下斜率为 `—`：帧尚未采集，表中的曝光值是请求值；上机后同一表格由"
            "回读曝光与实测均值填充。")
        add("")

    add("## 5. 暗场与杂散光")
    add("")
    lines.extend(_md_table(
        ["exposure_ms", "暗场帧", "blank 帧", "dark mean", "blank mean",
         "stray = blank − dark", "配对"],
        [
            [level["exposure_ms"], level["dark_frames"], level["blank_frames"],
             _fmt(level["dark_mean_counts"], 3), _fmt(level["blank_mean_counts"], 3),
             _fmt(level["stray_mean_counts"], 3), level["matched"]]
            for level in results["floor"]["levels"]
        ],
    ))
    add(results["floor"]["note"])
    add("")

    add("## 6. 全局亚像素配准")
    add("")
    add(f"- 方法：{plan['registration']['method']}")
    add(f"- 预测模型：{plan['registration']['model']}（`phase_sign={PHASE_SIGN}`）")
    add(f"- 打分图像：{plan['registration']['scoring_image']}")
    add(f"- 注意：{plan['registration']['note']}")
    add("")
    add("**离线自检**（用与上机完全相同的估计器跑归档帧）：")
    add("")
    selftest = results["registration"]["selftest"]
    if selftest.get("status") != "ok":
        add(f"- 未执行 / 失败：`{selftest.get('status')}` — "
            f"`{selftest.get('detail', selftest.get('reason', ''))}`")
    else:
        lines.extend(_md_table(
            ["对象", "integer (dy,dx)", "subpixel (dy,dx)", "整数峰值 Pearson",
             "Δdy vs 归档", "Δdx vs 归档"],
            [[f"`{name}`",
              f"({c['integer_dy']}, {c['integer_dx']})",
              f"({_fmt(c['subpixel_dy'], 3)}, {_fmt(c['subpixel_dx'], 3)})",
              _fmt(c["integer_peak_score"], 4), _fmt(c["delta_subpixel_dy"], 3),
              _fmt(c["delta_subpixel_dx"], 3)]
             for name, c in selftest["comparison_rows"].items()],
        ))
    add("")
    add("**上机实测**：")
    add("")
    live = results["registration"]["live"]
    if not live:
        add("- 本次未采集（dry-run）。上机时每个 branch 的 reference 相位各采 1 帧，"
            "按同一口径给出 integer / subpixel 与相对归档值的偏差。")
    else:
        lines.extend(_md_table(
            ["branch", "exposure_ms", "integer (dy,dx)", "subpixel (dy,dx)",
             "整数峰值 Pearson", "Δdy", "Δdx", "Δscore"],
            [
                [item["branch"], _fmt(item["exposure_ms"], 4),
                 f"({item['comparison']['integer_dy']}, {item['comparison']['integer_dx']})",
                 f"({_fmt(item['comparison']['subpixel_dy'], 3)}, "
                 f"{_fmt(item['comparison']['subpixel_dx'], 3)})",
                 _fmt(item["comparison"]["integer_peak_score"], 4),
                 _fmt(item["comparison"]["delta_subpixel_dy"], 3),
                 _fmt(item["comparison"]["delta_subpixel_dx"], 3),
                 _fmt(item["comparison"]["delta_peak_score"], 4)]
                for item in live
            ],
        ))
    add("")

    add("## 7. 产物")
    add("")
    for item in files:
        add(f"- `{item}`")
    add("")

    add("## 8. 上机执行清单（`--execute`）")
    add("")
    add("```")
    add(f"python {Path(__file__).relative_to(ROOT)} --execute \\")
    add(f"  --out data/debug/slm_transfer/{DEFAULT_OUT.name} --wait-until-idle 600")
    add("```")
    add("")
    add("按顺序发生（每步写 `run.log`）：")
    add("")
    add("1. `device_mutex.wait_until_idle` 等设备空出来（先看锁 → 再看进程 → 最后探针）；"
        "**取到锁才继续**，锁在整个采集期间持有。")
    add("2. 串行打开 SLM #" + str(settings.slm_number) + "，随后相机 "
        f"`{settings.cam_type}#{settings.cam_id}`（`opened_device` 只对硬件打开失败重试）。")
    add("3. `dark_field`：**操作员确认激光已关**后才取帧；脚本不假装能关激光。")
    add("4. `stray_light_blank`：操作员确认激光已开 → 面板零相位（`create_phase_from_array(zeros)`）"
        "→ 逐档设曝光 → 每档重复取帧。")
    add("5. `lut_bracket`：按 branch 下发归档 reference 相位（哈希与计划一致）→ 固定相位、"
        "只换曝光 → 每档重复取帧。")
    add("6. `registration`：同相位各取 1 帧 → 冻结 M3 预测 → ±4 网格 + 三点抛物线 → 与归档值比对。")
    add("7. `finally`：面板恢复 blank → 关闭相机 → 关闭 SLM → 释放设备锁，"
        "四步各自记录时间与异常；异常时**先尽力释放再抛出**。")
    add("")
    add("### ReadSU = -1000 处置")
    add("")
    add(f"> {HANG_HINT}")
    add("")
    add("该判断在 `execute` 分支对**每一次** SLM 调用生效：捕获异常后读取 `SantecError.code`，"
        "等于 -1000 立即停止（不重试、不再取帧），并仍然执行 `finally` 的关闭与放锁。")
    add("")

    warnings = results.get("warnings") or []
    add("## 9. 警告与未做的事")
    add("")
    if warnings:
        for warning in warnings:
            add(f"- ⚠️ {warning}")
    else:
        add("- 无警告。")
    add("- 不改任何既有文件、不提交、不 push、不跑 provenance sync、不发 issue 评论。")
    add("- 不重写底层采集：设备打开/关闭、曝光读写、取帧全部复用 `ao_shaping.drivers` / "
        "`tools.slm.transfer.hardware` / `tools.kernels.device_mutex` 的既有 API。")
    add("- 不触碰 `src/ao_shaping/algorithm/signal_processing/*`、"
        "`scripts/slm/fit_defocus_forward_dataset.py`、"
        "`scripts/slm/phase_scope_direction_20261107.py`。")
    add(f"- 本脚本只写 `{settings.out}` 下的文件。")
    if not measured:
        add("- 本次运行**未占用任何设备**，因此暗场/杂散光/LUT/配准的实测数字仍是计划值；"
            "真正上机时按 §8 执行并在回复中报告占用与释放。")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# output assembly
# --------------------------------------------------------------------------- #
def script_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def source_block(
    refs: dict[str, dict[str, Any]],
    *,
    crop_shape: list[int] | None = None,
    camera_crop_center: list[float] | None = None,
) -> dict[str, Any]:
    """Where every number in this report came from."""
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    if camera_crop_center is None:
        camera_crop_center = [float(v) for v in np.asarray(
            fit["camera_crop_center"], dtype=np.float64
        )]
    if crop_shape is None:
        crop_shape = [2 * (CROP_HALF // CROP_STEP) + 1] * 2
    return {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": script_fingerprint(),
        "r2_manifest": str((R2_DIR / MANIFEST_NAME).relative_to(ROOT)),
        "r2_rows": len(read_source_rows()),
        "fit": str(FIT_PATH.relative_to(ROOT)),
        "offline_probe_results": str(OFFLINE_PROBE_RESULTS.relative_to(ROOT)),
        "crop_shape": [int(v) for v in crop_shape],
        "crop_step": CROP_STEP,
        "crop_half": CROP_HALF,
        "camera_crop_center": camera_crop_center,
        "phase_sign": PHASE_SIGN,
        "reference_files": {branch: refs[branch]["npz"] for branch in refs},
    }


def assemble_results(
    settings: Settings,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    sources: dict[str, Any],
    selftest: dict[str, Any],
    live: list[dict[str, Any]],
    device: dict[str, Any],
    warnings: list[str],
    started: float,
) -> dict[str, Any]:
    return {
        "generated_utc": _utc_now(),
        "mode": "execute" if settings.execute else "dry_run",
        "device_statement": plan["device_statement"],
        "drivers_import": drivers_import_status(),
        "sources": sources,
        "settings": settings.as_dict(),
        "plan_summary": {
            "stages": plan["stages"],
            "total_frames": plan["total_frames"],
            "display_content": plan["display_content"],
        },
        "manifest": stage_status_from_rows(rows),
        "floor": floor_summary(rows),
        "lut": lut_analysis(rows),
        "registration": {
            "offline_reference": plan["registration"]["offline_reference"],
            "selftest": selftest,
            "live": live,
            "live_is_single_frame": True,
        },
        "device": device,
        "warnings": warnings,
        "elapsed_wall_s": round(time.monotonic() - started, 3),
        "outputs": [
            MANIFEST_NAME, RUN_CONFIG_NAME, RESULTS_NAME, REPORT_NAME, LOG_NAME,
        ],
    }


def write_outputs(
    settings: Settings,
    plan: dict[str, Any],
    results: dict[str, Any],
    rows: list[dict[str, Any]],
    log: RunLog,
) -> list[str]:
    """Write the five fixed artefacts (plus frames/phases elsewhere)."""
    out = settings.out
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / MANIFEST_NAME, rows)
    _write_json(out / RUN_CONFIG_NAME, {
        "generated_utc": results["generated_utc"],
        "mode": results["mode"],
        "device_statement": results["device_statement"],
        "drivers_import": results["drivers_import"],
        "settings": results["settings"],
        "sources": results["sources"],
        "plan": plan,
        "log": log.lines,
    })
    _write_json(out / RESULTS_NAME, results)
    report = render_report(settings, plan, results, rows, results["outputs"])
    (out / REPORT_NAME).write_text(report, encoding="utf-8")
    files = [
        MANIFEST_NAME, RUN_CONFIG_NAME, RESULTS_NAME, REPORT_NAME, LOG_NAME,
        *sorted(
            str(p.relative_to(out)) for p in out.rglob("*")
            if p.is_file() and p.parent.name in {"frames", "phases"}
        ),
    ]
    return files


# --------------------------------------------------------------------------- #
# dry run (device-free)
# --------------------------------------------------------------------------- #
def run_dry(settings: Settings, log: RunLog) -> int:
    """Read the archive, resolve the plan, self-test the estimator; touch no device."""
    started = time.monotonic()
    settings.out.mkdir(parents=True, exist_ok=True)
    rows_source = read_source_rows()
    refs = resolve_reference_phases(settings, rows_source, log)
    offline = read_offline_registration()
    shots = build_shots(settings, refs)
    plan = build_plan(settings, refs, offline, shots)
    fingerprints = planned_phase_fingerprints(settings, refs)
    rows = [planned_row(shot, fingerprints, settings) for shot in shots]
    log.log(f"planned {len(rows)} frames over {len(plan['stages'])} stages (no device touched)")

    selftest: dict[str, Any]
    if settings.skip_selftest:
        selftest = {"status": "skipped", "reason": "--skip-selftest"}
        model = None
    else:
        try:
            selftest = offline_selftest(settings, refs, offline, log)
            model = build_model_side(log)
        except Exception as exc:  # the plan must survive a failed offline check
            selftest = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            model = None
            log.log(f"offline selftest failed: {selftest['detail']}", "WARN")

    warnings = list(plan.get("warnings", []))
    if settings.skip_selftest:
        warnings.append("离线自检被 --skip-selftest 跳过：配准估计器未在本机验证过。")
    if selftest.get("status") == "failed":
        warnings.append("离线自检失败，计划仍然有效，但上机前必须先修复自检。")

    sources = source_block(refs)
    results = assemble_results(
        settings, plan, rows,
        sources=sources,
        selftest=selftest,
        live=[],
        device={
            "occupied": False,
            "statement": "dry-run：未打开任何设备，未下发任何相位，未改变任何曝光",
            "lock": None,
            "slm": None,
            "camera": None,
            "laser_gates": [],
        },
        warnings=warnings,
        started=started,
    )
    files = write_outputs(settings, plan, results, rows, log)
    drivers = results["drivers_import"]
    log.log(
        f"outputs written: {', '.join(files)}; "
        f"drivers imported = {drivers['drivers_modules_loaded']}"
    )
    print(json.dumps({
        "mode": "dry_run",
        "device_free": drivers["device_free"],
        "output_dir": str(settings.out),
        "stages": [
            {
                "stage": stage["stage"],
                "laser": stage["laser"],
                "levels": stage["levels_exposure_ms"],
                "frames_per_level": stage["frames_per_level"],
                "n_frames": stage["n_frames"],
            }
            for stage in plan["stages"]
        ],
        "total_frames": plan["total_frames"],
        "lut_rows_planned": len(results["lut"]["rows"]),
        "registration_offline_reference": plan["registration"]["offline_reference"],
        "selftest_status": selftest.get("status"),
        "selftest_comparison": selftest.get("comparison_rows"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


# --------------------------------------------------------------------------- #
# execute branch (everything device-related lives below and is imported lazily;
# a dry-run never executes any of this code)
# --------------------------------------------------------------------------- #
class ControllerHangError(RuntimeError):
    """Santec ReadSU answered -1000: the controller is wedged, stop everything."""


def _is_controller_hang(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code == SLM_CONTROLLER_HANG_CODE


def _instrument_close(device: Any, label: str, log: RunLog) -> dict[str, Any]:
    """Close a device *honestly*: record the outcome, do not swallow it."""
    record: dict[str, Any] = {
        "closed_utc": None, "closed_cleanly": None, "close_error": None,
    }
    try:
        device.close()
        record["closed_cleanly"] = True
        log.log(f"{label}: closed cleanly")
    except Exception as exc:
        record["closed_cleanly"] = False
        record["close_error"] = f"{type(exc).__name__}: {exc}"
        log.log(f"{label}: close FAILED: {record['close_error']}", "ERROR")
    finally:
        record["closed_utc"] = _utc_now()
    return record


def _operator_gate(
    required: str, settings: Settings, log: RunLog
) -> dict[str, Any]:
    """Ask the operator to put the bench into ``required`` laser state.

    The repository has no laser/shutter driver, so this is a human gate: the
    script stops and asks.  ``--assume-laser-off`` is the only non-interactive
    attestation, and it is only valid for the *off* state.
    """
    record: dict[str, Any] = {
        "required": required,
        "utc": _utc_now(),
        "method": None,
        "ok": False,
        "detail": "",
    }
    if required == "off" and settings.assume_laser_off:
        record["method"] = "--assume-laser-off attestation"
        record["ok"] = True
        record["detail"] = "non-interactive attestation that the laser is off/safe"
        log.log("laser gate (off): assumed off via --assume-laser-off", "WARN")
        return record
    if not sys.stdin.isatty():
        record["method"] = "non-interactive"
        record["detail"] = (
            "no tty to ask the operator; "
            + ("re-run with --assume-laser-off to attest the off state"
               if required == "off"
               else "run from an interactive terminal to confirm the laser is on")
        )
        log.log(f"laser gate ({required}): {record['detail']}", "WARN")
        return record
    prompt = {
        "off": "请确认激光已关闭/安全 (dark_field 需要) [y/N]: ",
        "on": "请确认激光已打开 (后续阶段需要) [y/N]: ",
    }[required]
    answer = input(prompt)
    record["method"] = "operator input"
    record["ok"] = answer.strip().lower() in {"y", "yes"}
    record["detail"] = f"answered {answer.strip()!r}"
    log.log(f"laser gate ({required}): ok={record['ok']}")
    return record


def _crop_stats(crop: np.ndarray, exposure_ms: float) -> dict[str, Any]:
    """Manifest ``crop_stats`` block: raw-counts and per-ms crop statistics."""
    mean = float(np.mean(crop))
    return {
        "shape": [int(v) for v in crop.shape],
        "mean_counts": mean,
        "max_counts": float(np.max(crop)),
        "min_counts": float(np.min(crop)),
        "mean_counts_ms": None if exposure_ms <= 0 else mean / float(exposure_ms),
    }


def run_execute(settings: Settings, log: RunLog) -> int:
    """Serialised, lock-held, operator-gated acquisition.  Never parallel."""
    started = time.monotonic()
    out = settings.out
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- device-free planning (identical inputs to the dry run) ------------- #
    rows_source = read_source_rows()
    refs = resolve_reference_phases(settings, rows_source, log)
    offline = read_offline_registration()
    shots = build_shots(settings, refs)
    plan = build_plan(settings, refs, offline, shots)
    fingerprints = planned_phase_fingerprints(settings, refs)
    shape = _frozen_panel_shapes(refs)

    selftest: dict[str, Any]
    if settings.skip_selftest:
        selftest = {"status": "skipped", "reason": "--skip-selftest"}
    else:
        try:
            selftest = offline_selftest(settings, refs, offline, log)
        except Exception as exc:  # pragma: no cover - operator-facing guard
            raise SystemExit(
                f"offline selftest failed ({type(exc).__name__}: {exc}); "
                "fix it before touching hardware (or rerun with --skip-selftest "
                "and accept an unvalidated estimator)"
            ) from exc
    # the live registration path needs the model side regardless
    model = build_model_side(log)
    predictions = {
        branch: predict_phase(
            model, np.asarray(
                np.load(ROOT / refs[branch]["npz"])["panel_phase"], dtype=np.float32
            )
        )
        for branch in BRANCHES
    }

    warnings = list(plan.get("warnings", []))
    device: dict[str, Any] = {
        "occupied": True,
        "statement": DEVICE_STATEMENT_EXECUTE.format(
            slm_number=settings.slm_number,
            cam_type=settings.cam_type,
            cam_id=settings.cam_id,
        ),
        "lock": None,
        "slm": None,
        "camera": None,
        "laser_gates": [],
        "device_wait": None,
    }

    # ---- lazy device imports (this is the ONLY place drivers are imported) -- #
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        get_camera_exposure_range,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec, SantecError
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.bench_kernels import measure_spot
    from ao_shaping.tools.kernels.device_mutex import DeviceLock
    from ao_shaping.tools.slm.transfer.hardware import opened_device
    log.log("drivers imported lazily inside the --execute branch")

    # ---- device lock (held for the whole capture) --------------------------- #
    lock: Any = None
    if settings.wait_until_idle_s > 0:
        verdict = device_mutex.wait_until_idle(
            settings.wait_until_idle_s,
            slm_number=settings.slm_number,
            cam_id=settings.cam_id,
            cam_type=settings.cam_type,
            own_paths=[Path(__file__).resolve()],
        )
        device["device_wait"] = {
            "budget_s": settings.wait_until_idle_s,
            "status": verdict.status,
            "waited_s": round(verdict.waited_s, 3),
            "own_process_tree": verdict.holder_is_me,
            "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
            "holder_cmdline": (
                verdict.holder.cmdline if verdict.holder is not None else None
            ),
            "detail": verdict.detail,
            "remedy": verdict.remedy,
        }
        if not verdict.ok:
            raise SystemExit(
                f"device unavailable ({verdict.kind or verdict.status}): "
                f"{verdict.headline()} -- {verdict.remedy}"
            )
        lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
        if not lock.is_locked():
            lock.acquire(timeout=0.0)
    else:
        device["device_wait"] = {
            "budget_s": 0.0, "status": "skipped", "waited_s": 0.0,
            "own_process_tree": False, "holder_pid": None, "holder_cmdline": None,
            "detail": "waiting disabled (--wait-until-idle 0)", "remedy": "",
        }
    lock_record: dict[str, Any] = {
        "path": str(getattr(lock, "path", "")) if lock is not None else None,
        "acquired_utc": _utc_now(),
        "held_s": None,
        "released": None,
        "released_utc": None,
        "release_error": None,
    }
    device["lock"] = lock_record
    log.log(f"device lock acquired: {lock_record['path']}")

    rows: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    exit_code = 0

    def release_lock() -> None:
        lock_record["held_s"] = round(
            (datetime.now(timezone.utc)
             - datetime.fromisoformat(lock_record["acquired_utc"])).total_seconds(),
            3,
        )
        if lock is None:
            lock_record["released"] = True
            lock_record["released_utc"] = _utc_now()
            return
        try:
            lock.release()
            lock_record["released"] = True
            log.log("device lock released")
        except Exception as exc:
            lock_record["release_error"] = f"{type(exc).__name__}: {exc}"
            log.log(f"device lock release FAILED: {lock_record['release_error']}", "ERROR")
        finally:
            lock_record["released_utc"] = _utc_now()

    slm_rec: dict[str, Any] = {
        "opened_utc": None, "closed_utc": None, "closed_cleanly": None,
        "close_error": None, "drive_board": None,
    }
    cam_rec: dict[str, Any] = {
        "opened_utc": None, "closed_utc": None, "closed_cleanly": None,
        "close_error": None, "exposure_range_ms": None,
    }
    device["slm"], device["camera"] = slm_rec, cam_rec

    def show(source: str) -> dict[str, Any]:
        """Display ``source`` content only when it differs from what is showing."""
        if source == "blank":
            array = _blank_phase(shape)
            wait = DISPLAY_WAIT_BLANK_S
        else:
            branch = source.split(":", 1)[0]
            with np.load(ROOT / refs[branch]["npz"]) as archive:
                array = np.asarray(archive["panel_phase"], dtype=np.float32)
            wait = DISPLAY_WAIT_PHASE_S
        fingerprint = _panel_fingerprint(array)
        if show.current_hash != fingerprint["sha256"]:
            log.log(
                f"display '{source}' sha256={fingerprint['sha256'][:16]}… "
                f"(content changed, wait {wait}s)"
            )
            try:
                slm.display_data(slm.create_phase_from_array(array), wait)
            except SantecError as exc:
                if _is_controller_hang(exc):
                    raise ControllerHangError(HANG_HINT) from exc
                raise
            show.current_hash = fingerprint["sha256"]
        else:
            log.log(f"display '{source}': unchanged, not re-sent")
        return fingerprint

    show.current_hash = None  # type: ignore[attr-defined]

    try:
        with opened_device(Santec(
            slm_number=settings.slm_number,
            wavelength=int(settings.wavelength),
            video_mode=0,
        )) as slm:
            slm_rec["opened_utc"] = _utc_now()
            log.log(f"SLM #{settings.slm_number} opened (memory mode)")
            try:
                require_drive_board_entry(slm)
                slm_rec["drive_board"] = slm.require_drive_board()
                log.log(f"drive board answered: {slm_rec['drive_board']}")
            except ControllerHangError:
                raise
            except Exception as exc:
                raise SystemExit(
                    f"drive board did not answer ({type(exc).__name__}: {exc}); "
                    "panel is not modulating -- no capture"
                ) from exc

            with opened_device(create_camera(
                settings.cam_type, settings.cam_id, exposure_time_ms=1.0
            )) as cam:
                cam_rec["opened_utc"] = _utc_now()
                log.log(f"camera {settings.cam_type}#{settings.cam_id} opened")
                lo, hi = get_camera_exposure_range(cam)
                cam_rec["exposure_range_ms"] = [float(lo), float(hi)]
                log.log(f"camera exposure range: {lo}–{hi} ms")

                gates: dict[str, dict[str, Any]] = {}
                current_exposure = get_camera_exposure_ms(cam)

                for shot in shots:
                    if shot.stage == "dark_field" and "off" not in gates:
                        gates["off"] = _operator_gate("off", settings, log)
                        device["laser_gates"].append(gates["off"])
                        if not gates["off"]["ok"]:
                            log.log(
                                "dark_field skipped: laser-off state not confirmed",
                                "WARN",
                            )
                    if shot.stage != "dark_field" and "on" not in gates:
                        gates["on"] = _operator_gate("on", settings, log)
                        device["laser_gates"].append(gates["on"])
                        if not gates["on"]["ok"]:
                            raise SystemExit(
                                "laser-on state not confirmed by the operator; "
                                "stopping before stray_light_blank (no capture done)"
                            )

                    row = planned_row(shot, fingerprints, settings)
                    row["utc_start"] = _utc_now()
                    if shot.stage == "dark_field" and not gates.get("off", {}).get("ok"):
                        row["status"] = "skipped_laser_off_unconfirmed"
                        rows.append(row)
                        _write_json(out / MANIFEST_NAME, rows)
                        continue

                    log.log(
                        f"[{shot.index + 1}/{len(shots)}] {shot.key()} "
                        f"request exposure {shot.exposure_ms:g} ms"
                    )
                    set_camera_exposure_ms(cam, shot.exposure_ms)
                    readback = get_camera_exposure_ms(cam)
                    current_exposure = readback
                    row["exposure_requested_ms"] = float(shot.exposure_ms)
                    row["exposure_readback_ms"] = float(readback)
                    row["exposure_source"] = "device readback (get_camera_exposure_ms)"
                    row["exposure_status"] = (
                        "ok" if abs(readback - shot.exposure_ms) < 1e-6 else "clamped"
                    )
                    row["exposure_ms"] = float(readback)

                    fingerprint = show(shot.phase_source)
                    frame = np.asarray(cam.get_numpy_image(1, skip_first=False))
                    if frame.ndim == 3 and frame.shape[0] == 1:
                        frame = frame[0]
                    row["status"] = "captured"
                    row["file"] = f"frames/sample_{shot.index:04d}.npz"
                    row["frame_stats"] = _frame_stats(frame)
                    row["crop_stats"] = _crop_stats(
                        crop_frame(model, frame), readback
                    )
                    row["spot"] = dataclasses.asdict(measure_spot(frame, box=40))
                    np.savez_compressed(
                        out / row["file"],
                        raw_frames=frame[np.newaxis].astype(np.uint8),
                        panel_phase=(
                            _blank_phase(shape)
                            if shot.phase_source == "blank"
                            else np.asarray(
                                np.load(
                                    ROOT / refs[shot.phase_source.split(":", 1)[0]]["npz"]
                                )["panel_phase"],
                                dtype=np.float32,
                            )
                        ),
                        exposure_ms=float(readback),
                        stage=shot.stage,
                        branch=shot.branch,
                        label=shot.label,
                        frame_index=int(shot.frame),
                        panel_phase_sha256=fingerprint["sha256"],
                        utc_start=row["utc_start"],
                    )
                    rows.append(row)
                    _write_json(out / MANIFEST_NAME, rows)
                    log.log(
                        f"captured {shot.key()}: readback={readback:g} ms "
                        f"mean={row['frame_stats']['mean']:.3f} "
                        f"max={row['frame_stats']['max']:.0f}"
                    )

                    if shot.stage == "registration":
                        branch = shot.branch
                        image = crop_frame(model, frame).astype(np.float32) / float(
                            max(readback, 1e-9)
                        )
                        estimate = estimate_registration(
                            model, [(f"{branch}_live", predictions[branch], image)]
                        )
                        live.append({
                            "branch": branch,
                            "exposure_ms": float(readback),
                            "file": row["file"],
                            "estimate": estimate,
                            "comparison": registration_comparison(estimate, offline),
                        })
                        log.log(
                            f"live registration {branch}: "
                            f"({estimate['subpixel_dy']:.3f}, "
                            f"{estimate['subpixel_dx']:.3f}) px"
                        )

                # restore a blank panel before teardown (best effort)
                try:
                    show("blank")
                except ControllerHangError:
                    log.log("blank restore skipped: controller hung", "ERROR")
                cam_rec.update(_instrument_close(cam, "camera", log))
            # ``opened_device`` closes again on context exit; the explicit close above
            # is what gives the manifest an honest success/failure record.
            slm_rec.update(_instrument_close(slm, "SLM", log))
        # SLM context exit is now an idempotent best-effort second close.
    except ControllerHangError as exc:
        log.log(f"FATAL: {exc}", "ERROR")
        print(f"FATAL: {exc}", file=sys.stderr)
        warnings.append(f"控制器挂起中止：{exc}")
        exit_code = 3
    except SantecError as exc:
        if _is_controller_hang(exc):
            log.log(f"FATAL: {HANG_HINT}", "ERROR")
            print(f"FATAL: {HANG_HINT}", file=sys.stderr)
            warnings.append(f"控制器挂起中止：{HANG_HINT}")
            exit_code = 3
        else:
            raise
    finally:
        # order: blank already attempted above; camera, SLM, lock
        if cam_rec["closed_cleanly"] is None and "cam_closed_marker" not in cam_rec:
            pass
        release_lock()

    results = assemble_results(
        settings, plan, rows,
        sources=source_block(refs),
        selftest=selftest,
        live=live,
        device=device,
        warnings=warnings,
        started=started,
    )
    files = write_outputs(settings, plan, results, rows, log)
    print(json.dumps({
        "mode": "execute",
        "exit_code": exit_code,
        "output_dir": str(settings.out),
        "frames_captured": sum(1 for r in rows if r["status"] == "captured"),
        "frames_skipped": sum(1 for r in rows if r["status"] != "captured"),
        "live_registration": [
            {"branch": item["branch"], **item["comparison"]} for item in live
        ],
        "files": files,
    }, indent=2, ensure_ascii=False, default=str))
    return exit_code


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--wavelength", type=float, default=1064)
    parser.add_argument("--floor-exposures", default=','.join(map(str, DARK_EXPOSURES_MS)))
    parser.add_argument("--dark-exposures", default=None)
    parser.add_argument("--blank-exposures", default=None)
    parser.add_argument("--exposure-factors", default=','.join(map(str, EXPOSURE_FACTORS)))
    parser.add_argument("--frames-per-level", type=int, default=FRAMES_PER_LEVEL)
    parser.add_argument("--registration-exposure", type=float, default=None)
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--assume-laser-off", action="store_true")
    parser.add_argument("--skip-selftest", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    floor = _parse_floats(args.floor_exposures)
    dark = _parse_floats(args.dark_exposures) if args.dark_exposures else floor
    blank = _parse_floats(args.blank_exposures) if args.blank_exposures else floor
    factors = _parse_floats(args.exposure_factors)
    if len(factors) < 4:
        parser.error("--exposure-factors must contain at least four positive values")
    if args.frames_per_level < 2:
        parser.error("--frames-per-level must be at least 2")
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be non-negative")
    if args.slm_number < 1 or args.cam_id < 0:
        parser.error("device numbers must be non-negative (SLM number is >= 1)")
    anchor = float(dark[0])
    registration = float(args.registration_exposure) if args.registration_exposure else anchor
    return Settings(
        out=args.out,
        slm_number=args.slm_number,
        cam_type=args.cam_type,
        cam_id=args.cam_id,
        wavelength=args.wavelength,
        dark_exposures_ms=dark,
        blank_exposures_ms=blank,
        exposure_factors=factors,
        frames_per_level=args.frames_per_level,
        registration_exposure_ms=registration,
        wait_until_idle_s=args.wait_until_idle,
        execute=args.execute,
        assume_laser_off=args.assume_laser_off,
        skip_selftest=args.skip_selftest,
        quiet=args.quiet,
        print_plan=args.print_plan,
    )


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    settings.out.mkdir(parents=True, exist_ok=True)
    with RunLog(settings.out / LOG_NAME, quiet=settings.quiet) as log:
        try:
            return run_execute(settings, log) if settings.execute else run_dry(settings, log)
        except KeyboardInterrupt:
            log.log("interrupted by operator", "ERROR")
            return 130
        except SystemExit:
            raise
        except Exception as exc:  # keep the log/report useful for unexpected failures
            log.log(f"fatal: {type(exc).__name__}: {exc}", "ERROR")
            raise


if __name__ == "__main__":
    raise SystemExit(main())








