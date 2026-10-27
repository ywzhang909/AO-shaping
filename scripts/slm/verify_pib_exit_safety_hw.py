"""Hardware acceptance for issue #145 (H-28): does a dead shaping run leave the SLM safe?

``slm_zernike_pib`` wraps its whole search in ``try: ... finally:
_apply_best_on_exit()`` (commit ``3070f791``), and that cleanup falls back to
``set_grayscale(0)`` if the best-phase restore itself fails. The offline suite
(`tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_hardware_safety.py`)
proves the control flow against the digital twin. It cannot prove what the
**real Santec panel** displays, because the panel's behaviour is dominated by two
piece of firmware that a fake driver does not have:

* ``display_data(gray, memory_number=slot)`` is a **no-op when ``slot`` is
  already displayed** -- the LCOS does not refresh. A cleanup that "wrote the
  best phase" can therefore write into the void.
* ``set_grayscale(0)`` switches the panel from memory mode to direct grayscale
  mode; whether that actually lands as flat is a device fact.

So this script runs the four scenarios from the issue on the real bench and
then asks the **device itself** what it is displaying, two independent ways:

1. Firmware readback, taken *before* the panel is touched:
   ``get_displayed_memory_number()`` (which slot the LCOS shows) or, in direct
   grayscale mode, ``get_current_grayscale()``. A memory slot's content is
   exactly the gray array this script recorded when the optimizer wrote it, so
   the slot number identifies the displayed phase without any inference.
2. Optics: capture the live far field, then re-render each candidate phase
   (flat gray-0, the cleanup's exit phase, the last search iterate) and compare
   frames. This is what catches a "slot says X but the LCOS never refreshed"
   no-op, and it is measured against the repeat-frame noise floor.

Scenarios (each gets its own optimizer call, so nothing leaks between them):

==================  ==========================================================
A  ctrl-c spgd      ``KeyboardInterrupt`` delivered asynchronously through
                    ``_thread.interrupt_main()`` from a timer thread -- the same
                    mechanism a console Ctrl-C uses, so it lands at an arbitrary
                    bytecode boundary rather than inside a chosen call.
B  camera dropout   ``CameraError`` raised from the real camera driver's
                    ``get_numpy_image`` on the Nth capture, mid-search. The
                    device stays open so the teardown path is the production
                    one.
C  rerun            a normal short run immediately after B: epoch 0 must start
                    from a sane baseline (``pib`` at a normal level, not the
                    ``mean_b ~= 0.01`` symptom of a mis-aimed target box).
D  ctrl-c heuristic the same async Ctrl-C on the ``ga`` branch, which shares the
                    ``finally`` but not the search body.
E  ctrl-c demanding the same as A with a small target box (``--target-size 16``)
                    so the search genuinely improves on flat; that forces the
                    *best-phase* exit branch instead of the flat branch, and it
                    is the branch the issue is actually about.
==================  ==========================================================

The SLM is opened **once** by this script and handed to the optimizer through a
keep-open proxy, because ``Santec.close()`` drops the panel: verifying "what was
left on the panel" after the driver was told to close it would measure nothing.

Everything is written under ``report/slm/pib_exit_safety/artifacts/run_<stamp>/``
(``summary.json`` is the contract consumed by
``scripts/slm/generate_pib_exit_safety_report.py``; ``frames/`` holds the crops).

Usage (hardware; SLM in memory mode, nothing else holding the USB handle)::

    python scripts/slm/verify_pib_exit_safety_hw.py
    python scripts/slm/verify_pib_exit_safety_hw.py --only A B
    python scripts/slm/verify_pib_exit_safety_hw.py --dry-run   # config only
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import _thread
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

from ao_shaping.drivers.ccd.common import (
    auto_exposure,
    capture_with_exposure,
    create_camera,
    get_camera_exposure_ms,
)
from ao_shaping.drivers.ccd.base import CameraError
from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.optimizer.wfless.slm_zernike_primitives import SlotRotator
from ao_shaping.optimizer.wfless import slm_zernike_pib as engine
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)

ARTIFACT_ROOT = ROOT / "report" / "slm" / "pib_exit_safety" / "artifacts"

#: Crop side (px) used for every optical comparison. The 0-order of a flat
#: phase is ~50 px wide at this bench's exposure; 700 px keeps the whole envelope
#: of a shaped phase (which spreads energy outward) inside the comparison.
CROP_SIDE = 700
#: Frames averaged per verification capture.
VERIFY_FRAMES = 3
#: Settle time after a panel write before the verification frame is read.
SETTLE_S = 0.35


#: Predicates that decide PASS/FAIL. A key listed here must be a ``bool``; a key
#: whose value is ``None`` is "not applicable to this scenario" and is skipped
#: (a normal-exit run has no cleanup duration to judge).
VERDICT_KEYS: tuple[str, ...] = (
    "failure_reached_caller",
    "original_exception_survived_cleanup",
    "run_completed",
    "firmware_is_named_state",
    "exit_used_a_fresh_slot",
    "optical_is_defined_state",
    "cleanup_under_5s",
    "energy_loss_violations_recorded",
    "epoch0_is_sane",
)


@dataclass
class Scenario:
    """One optimizer run and the failure it is asked to die from."""

    key: str
    name: str
    algorithm: str
    epochs: int
    description: str = ""
    pop_size: int | None = None
    target_size: float = 64.0
    #: Fire the async Ctrl-C once this many memory writes have happened.
    interrupt_at_write: int | None = None
    #: Raise CameraError from the Nth ``get_numpy_image`` call.
    dropout_at_capture: int | None = None

    @property
    def injects(self) -> bool:
        return self.interrupt_at_write is not None or self.dropout_at_capture is not None


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="A",
        name="A_ctrl_c_spgd",
        algorithm="spgd",
        epochs=200,
        interrupt_at_write=30,
        description="async KeyboardInterrupt mid-SPGD",
    ),
    Scenario(
        key="B",
        name="B_camera_dropout_spgd",
        algorithm="spgd",
        epochs=200,
        dropout_at_capture=60,
        description="camera get_numpy_image raises CameraError mid-SPGD",
    ),
    Scenario(
        key="C",
        name="C_rerun_after_crash",
        algorithm="spgd",
        epochs=10,
        description="normal short run right after B",
    ),
    Scenario(
        key="D",
        name="D_ctrl_c_heuristic_ga",
        algorithm="ga",
        epochs=200,
        pop_size=6,
        interrupt_at_write=50,
        description="async KeyboardInterrupt mid-GA (heuristic branch)",
    ),
    Scenario(
        key="E",
        name="E_ctrl_c_spgd_demanding",
        algorithm="spgd",
        epochs=200,
        target_size=16.0,
        interrupt_at_write=40,
        description="async KeyboardInterrupt mid-SPGD on a 16 px box (forces best != flat)",
    ),
)


class _KeepOpen:
    """Context-manager proxy that hands out an already-open SLM and never closes it.

    The optimizer treats ``Santec.from_params(...)`` as a context manager and
    closes it on the way out. Closing drops the panel, so the post-mortem would
    have nothing to look at; this proxy keeps the handle (and therefore the
    displayed phase) alive past the crash.
    """

    def __init__(self, slm: Santec) -> None:
        self._slm = slm

    def __enter__(self) -> Santec:
        return self._slm

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._slm, name)


@dataclass
class Bench:
    """The one SLM handle plus every write the optimizer performed through it."""

    slm: Santec
    writes: list[dict[str, Any]] = field(default_factory=list)
    recorders: list[Any] = field(default_factory=list)
    exposure_ms: float = 0.6
    camera_shape: tuple[int, int] = (0, 0)
    target_peak: float = 160.0
    #: Set by the injection; ``None`` means the trigger never fired.
    inject_t: float | None = None
    inject_iso: str | None = None
    inject_detail: str = ""
    interrupt_at_write: int | None = None
    _armed: bool = False
    _timer: threading.Timer | None = None
    _won: dict[str, Any] = field(default_factory=dict)

    # -- write recording ---------------------------------------------------
    def record_memory_write(self, slot: int, gray: np.ndarray) -> None:
        arr = np.asarray(gray)
        self.writes.append(
            {
                "i": len(self.writes),
                "kind": "memory",
                "slot": int(slot),
                "t": time.monotonic(),
                "shape": list(arr.shape),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "gray": arr,
            }
        )
        if (
            self.interrupt_at_write is not None
            and len(self.writes) >= self.interrupt_at_write
        ):
            self._arm_interrupt()

    def record_grayscale_write(self, gray: int) -> None:
        self.writes.append(
            {
                "i": len(self.writes),
                "kind": "grayscale",
                "slot": None,
                "t": time.monotonic(),
                "gray_value": int(gray),
            }
        )

    # -- injections --------------------------------------------------------
    def _arm_interrupt(self) -> None:
        """Deliver the Ctrl-C from a timer thread, not from the write itself.

        ``_thread.interrupt_main()`` sets the interpreter's interrupt flag, which
        raises ``KeyboardInterrupt`` in the main thread at the next bytecode
        boundary -- the same path a console Ctrl-C takes. The small delay keeps
        the raise out of the write call stack, so it lands *between* operations
        instead of at a spot the harness chose.
        """
        if self._armed:
            return
        self._armed = True
        self._timer = threading.Timer(0.05, self._fire_interrupt)
        self._timer.daemon = True
        self._timer.start()

    def _fire_interrupt(self) -> None:
        if self.inject_t is None:
            self.inject_t = time.monotonic()
            self.inject_iso = _utc_now()
            self.inject_detail = (
                f"_thread.interrupt_main from a timer at memory write "
                f"#{len(self.writes)}"
            )
        _thread.interrupt_main()

    def fire_dropout(self, capture_index: int) -> None:
        if self.inject_t is None:
            self.inject_t = time.monotonic()
            self.inject_iso = _utc_now()
            self.inject_detail = (
                f"CameraError from get_numpy_image at capture #{capture_index}"
            )

    def cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    # -- queries -----------------------------------------------------------
    def writes_before_inject(self) -> list[dict[str, Any]]:
        cut = self.inject_t if self.inject_t is not None else float("inf")
        return [w for w in self.writes if w["t"] <= cut]

    def writes_after_inject(self) -> list[dict[str, Any]]:
        cut = self.inject_t if self.inject_t is not None else float("inf")
        return [w for w in self.writes if w["t"] > cut]

    def exit_write(self) -> dict[str, Any] | None:
        """The last thing the cleanup put on the panel (``None`` if it never wrote)."""
        after = self.writes_after_inject()
        if after:
            return after[-1]
        # A run that was never interrupted exits through the same cleanup, so its
        # final write IS the exit state.
        if self.inject_t is None and self.writes:
            return self.writes[-1]
        return None

    def last_search_write(self) -> dict[str, Any] | None:
        before = self.writes_before_inject()
        memory = [w for w in before if w["kind"] == "memory"]
        return memory[-1] if memory else None


def _json_default(value: Any) -> Any:
    """Make numpy scalars/bools/arrays safe for ``json.dumps``."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@contextmanager
def patched_engine(bench: Bench, scenario: Scenario):
    """Route the optimizer's device writes through this script's recorder."""
    real_display = engine.display_rotating
    real_create_camera = engine.create_camera
    real_recorder = engine.Recorder
    real_santec = engine.Santec
    real_set_gray = bench.slm.set_grayscale

    def display(slm_obj, gray, rotator):
        slot = real_display(slm_obj, gray, rotator)
        bench.record_memory_write(slot, gray)
        return slot

    def set_gray(gs):
        bench.record_grayscale_write(gs)
        return real_set_gray(gs)

    def create_camera_wrapped(params):
        cam = real_create_camera(params)
        if scenario.dropout_at_capture is not None:
            state = {"n": 0}
            original_get = cam.get_numpy_image

            def get_numpy_image(*args: Any, **kwargs: Any):
                state["n"] += 1
                if state["n"] >= scenario.dropout_at_capture:
                    bench.fire_dropout(state["n"])
                    raise CameraError(
                        f"injected camera dropout at capture #{state['n']}"
                    )
                return original_get(*args, **kwargs)

            cam.get_numpy_image = get_numpy_image
        return cam

    class TrackedRecorder(real_recorder):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bench.recorders.append(self)

    class SlmShim:
        @classmethod
        def from_params(cls, params, **overrides):
            return _KeepOpen(bench.slm)

    bench.slm.set_grayscale = set_gray
    engine.display_rotating = display
    engine.create_camera = create_camera_wrapped
    engine.Recorder = TrackedRecorder
    engine.Santec = SlmShim
    try:
        yield
    finally:
        engine.display_rotating = real_display
        engine.create_camera = real_create_camera
        engine.Recorder = real_recorder
        engine.Santec = real_santec
        bench.slm.set_grayscale = real_set_gray


def build_config(scenario: Scenario, exposure_ms: float):
    """Config for one scenario.

    ``fold_ratio=0`` and ``noise_gate_k=-1`` switch off the two *search* quality
    gates. They decide whether an epoch's gradient update is committed and are
    orthogonal to the exit path under test, but on this bench they reject every
    epoch (the flat far field is speckled, so per-epoch peak scatters 2-4x and
    every frame looks like a brightness fold), which would leave the search with
    no iterates and the cleanup with nothing to restore. With them off the search
    really moves the panel, which is the precondition scenario E needs.
    """
    return engine.SlmZernikePibConfig(
        center="shape",
        epochs=scenario.epochs,
        algorithm=scenario.algorithm,
        pop_size=scenario.pop_size,
        fold_ratio=0.0,
        noise_gate_k=-1.0,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name="pib", target_shape=None),
            cam_type="daheng",
            cam_id=0,
            cam_size=250,
            exposure_time_ms=float(exposure_ms),
            target_size=scenario.target_size,
        ),
        slm=SlmParamsPib(n_max=7, slm_number=1),
        show=False,
    )


# --------------------------------------------------------------------------
# Verification of the panel state
# --------------------------------------------------------------------------
def _open_camera(exposure_ms: float):
    """Open the bench camera, retrying the 'device still closing' window."""
    last: BaseException | None = None
    for attempt in range(4):
        try:
            cam = create_camera("daheng", 0, exposure_time_ms=exposure_ms)
            cam.open()
            return cam
        except Exception as exc:
            last = exc
            time.sleep(0.6 + 0.4 * attempt)
    raise RuntimeError(f"could not reopen the camera for verification: {last}")


def _capture(cam, exposure_ms: float) -> np.ndarray:
    cam.reset_exposure_time(exposure_ms)
    frame = capture_with_exposure(
        cam, exposure_ms, 0.0, n_sample=VERIFY_FRAMES
    )
    return np.asarray(frame, dtype=np.float64)


def _default_crop(frame: np.ndarray, center: tuple[int, int] | None) -> tuple[int, int]:
    """Top-left of a CROP_SIDE x CROP_SIDE box, centred on the 0-order."""
    h, w = frame.shape
    if center is None:
        center = tuple(int(v) for v in np.unravel_index(int(np.argmax(frame)), frame.shape))
    cy, cx = int(center[0]), int(center[1])
    half = CROP_SIDE // 2
    y0 = max(0, min(h - CROP_SIDE, cy - half))
    x0 = max(0, min(w - CROP_SIDE, cx - half))
    return y0, x0


def _crop(frame: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    y0, x0 = origin
    return frame[y0 : y0 + CROP_SIDE, x0 : x0 + CROP_SIDE]


def _normalise(crop: np.ndarray) -> np.ndarray:
    a = np.asarray(crop, dtype=np.float64)
    a = np.clip(a - float(np.median(a)), 0.0, None)
    total = float(a.sum())
    return a / total if total > 0 else a


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _normalise(a).ravel(), _normalise(b).ravel()
    sx, sy = float(x.std()), float(y.std())
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _firmware_state(slm: Santec) -> dict[str, Any]:
    """What the controller says it is displaying, read without changing it."""
    state: dict[str, Any] = {
        "displayed_slot": None,
        "displayed_slot_error": None,
        "current_grayscale": None,
        "current_grayscale_error": None,
    }
    try:
        state["displayed_slot"] = slm.get_displayed_memory_number()
    except Exception as exc:
        state["displayed_slot_error"] = f"{type(exc).__name__}: {exc}"
    try:
        state["current_grayscale"] = int(slm.get_current_grayscale())
    except Exception as exc:
        state["current_grayscale_error"] = f"{type(exc).__name__}: {exc}"
    return state


def verify_panel(bench: Bench, scenario: Scenario, outdir: Path) -> dict[str, Any]:
    """Capture the live panel, then each candidate state, and compare them."""
    slm = bench.slm
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cam = _open_camera(bench.exposure_ms)
    try:
        # 1. The state the crashed run left behind -- read before touching anything.
        live = _capture(cam, bench.exposure_ms)
        firmware = _firmware_state(slm)
        settle_exposure = float(get_camera_exposure_ms(cam))

        exit_write = bench.exit_write()
        last_search = bench.last_search_write()

        # 2. Reference states, rendered exactly the way the optimizer rendered them.
        slm.set_grayscale(0)
        time.sleep(SETTLE_S)
        flat = _capture(cam, settle_exposure)

        rotor = SlotRotator()
        exit_frame = None
        exit_slot = None
        if exit_write is not None and exit_write["kind"] == "memory":
            exit_slot = engine.display_rotating(slm, exit_write["gray"], rotor)
            time.sleep(SETTLE_S)
            exit_frame = _capture(cam, settle_exposure)

        iterate_frame = None
        if last_search is not None:
            engine.display_rotating(slm, last_search["gray"], rotor)
            time.sleep(SETTLE_S)
            iterate_frame = _capture(cam, settle_exposure)

        # 3. Repeat-frame noise floor: two captures of the SAME flat state.
        slm.set_grayscale(0)
        time.sleep(SETTLE_S)
        flat_repeat = _capture(cam, settle_exposure)

        origin = _default_crop(flat, None)
        crops = {
            "live": _crop(live, origin),
            "flat": _crop(flat, origin),
            "flat_repeat": _crop(flat_repeat, origin),
        }
        if exit_frame is not None:
            crops["exit"] = _crop(exit_frame, origin)
        if iterate_frame is not None:
            crops["last_iterate"] = _crop(iterate_frame, origin)

        for name, crop in crops.items():
            np.savez_compressed(frames_dir / f"{scenario.name}_{name}.npz", frame=crop)

        corr = {
            "live_vs_flat": _corr(crops["live"], crops["flat"]),
            "flat_repeat": _corr(crops["flat"], crops["flat_repeat"]),
        }
        if "exit" in crops:
            corr["live_vs_exit"] = _corr(crops["live"], crops["exit"])
        if "last_iterate" in crops:
            corr["live_vs_last_iterate"] = _corr(crops["live"], crops["last_iterate"])

        peaks = {
            name: float(crop.max()) for name, crop in crops.items()
        }
        return {
            "exposure_ms": settle_exposure,
            "crop_origin_yx": [int(origin[0]), int(origin[1])],
            "crop_side": CROP_SIDE,
            "firmware": firmware,
            "exit_slot_written_by_harness": exit_slot,
            "corr": corr,
            "peaks": peaks,
            "frame_files": sorted(p.name for p in frames_dir.glob(f"{scenario.name}_*.npz")),
        }
    finally:
        try:
            slm.set_grayscale(0)
        except Exception as exc:
            logger.error("could not return the panel to flat: {}", exc)
        try:
            cam.close()
        except Exception as exc:
            logger.warning("verification camera close failed: {}", exc)


def _checks(
    scenario: Scenario,
    outcome: str,
    exc_type: str | None,
    exc_message: str | None,
    firmware: dict[str, Any],
    exit_write: dict[str, Any] | None,
    writes: list[dict[str, Any]],
    corr: dict[str, float],
    cleanup_s: float | None,
    energy_loss_violations_present: bool,
    row0: dict[str, Any] | None,
) -> dict[str, Any]:
    """The pass/fail predicates, computed from recorded evidence only.

    Keys listed in :data:`VERDICT_KEYS` must be ``bool``; informational values
    live under keys that are not in it, so a number can never be mistaken for a
    threshold that was cleared.
    """
    checks: dict[str, Any] = {}

    if scenario.injects:
        expected = (
            "KeyboardInterrupt"
            if scenario.interrupt_at_write is not None
            else "CameraError"
        )
        checks["failure_reached_caller"] = bool(outcome == "raised")
        checks["original_exception_survived_cleanup"] = bool(exc_type == expected)
        checks["exception_message"] = exc_message
    else:
        checks["run_completed"] = bool(outcome == "returned")

    # --- firmware: is the panel on a state this script can name? -------------
    slot = firmware.get("displayed_slot")
    gray = firmware.get("current_grayscale")
    if exit_write is None:
        checks["firmware_mode"] = "unknown"
        checks["firmware_is_named_state"] = False
    elif exit_write["kind"] == "grayscale":
        checks["firmware_mode"] = "grayscale"
        checks["expected_grayscale"] = 0
        checks["firmware_is_named_state"] = bool(gray == 0)
        checks["firmware_left_memory_mode"] = slot is None
    else:
        checks["firmware_mode"] = "memory"
        checks["expected_slot"] = int(exit_write["slot"])
        checks["firmware_is_named_state"] = bool(
            slot is not None and int(slot) == int(exit_write["slot"])
        )

    # --- the write must not have landed in the already-displayed slot --------
    # display_data() to the slot the LCOS is already showing is a firmware
    # no-op, so a cleanup that ignores rotation "writes" the best phase and
    # leaves the previous one visible. Rotation leaves a trace in the log.
    if exit_write is not None and exit_write["kind"] == "memory":
        prior_memory = [
            w
            for w in writes
            if w["kind"] == "memory" and int(w["i"]) < int(exit_write["i"])
        ]
        previous = prior_memory[-1] if prior_memory else None
        checks["exit_previous_slot"] = None if previous is None else int(previous["slot"])
        checks["exit_used_a_fresh_slot"] = not (
            previous is not None and int(previous["slot"]) == int(exit_write["slot"])
        )

    # Optics: the live frame must match a state the optimizer actually wrote.
    # Firmware (above) is the decisive identifier; optics guards the one thing it
    # cannot see -- a write into an already-displayed slot is a firmware no-op, so
    # the slot readback can name a phase the LCOS never refreshed. Here the check
    # is "the live frame is one of the written states, at the repeat-frame noise
    # floor"; preferring the exit phase over the last iterate is reported but not
    # gated, because on a speckled far field the two can be optically identical.
    floor = corr.get("flat_repeat")
    candidate_corrs = {
        name: corr.get(key)
        for name, key in (
            ("flat", "live_vs_flat"),
            ("exit", "live_vs_exit"),
            ("last_iterate", "live_vs_last_iterate"),
        )
        if corr.get(key) is not None
    }
    finite = {k: v for k, v in candidate_corrs.items() if np.isfinite(v)}
    best_name = max(finite, key=finite.get) if finite else None
    checks["optical_class"] = best_name
    checks["optical_corrs"] = candidate_corrs
    checks["optical_noise_floor"] = floor
    checks["optical_is_defined_state"] = bool(
        best_name is not None
        and floor is not None
        and np.isfinite(floor)
        and finite[best_name] >= floor - 0.01
    )
    checks["optical_prefers_exit"] = bool(
        exit_write is not None
        and exit_write["kind"] == "memory"
        and best_name == "exit"
    )

    checks["cleanup_under_5s"] = (
        None if cleanup_s is None else bool(cleanup_s <= 5.0)
    )
    checks["cleanup_s"] = cleanup_s
    checks["energy_loss_violations_recorded"] = bool(energy_loss_violations_present)

    if row0 is not None:
        pib0 = row0.get("pib")
        peak0 = row0.get("max_brt")
        checks["epoch0_pib"] = pib0
        checks["epoch0_max_brightness"] = peak0
        # The failure mode the issue names is a target box that is not on the
        # beam: the metric collapses (mean_b ~ 0.01) rather than merely being
        # low. This bench's flat far field is speckled, so pib ~ 0.16 is its
        # normal level -- the floor below is deliberately far under it.
        checks["epoch0_is_sane"] = bool(
            pib0 is not None
            and np.isfinite(pib0)
            and pib0 > 0.05
            and peak0 is not None
            and peak0 > 20.0
        )

    return checks


def run_scenario(
    bench: Bench, scenario: Scenario, outdir: Path, watchdog_s: float
) -> dict[str, Any]:
    logger.info("=== {}: {} ===", scenario.key, scenario.description)
    bench.writes.clear()
    bench.recorders.clear()
    bench.inject_t = None
    bench.inject_iso = None
    bench.inject_detail = ""
    bench.interrupt_at_write = scenario.interrupt_at_write
    bench._armed = False
    bench._timer = None

    watchdog: threading.Timer | None = None
    if scenario.injects:
        watchdog = threading.Timer(watchdog_s, bench._fire_interrupt)
        watchdog.daemon = True
        watchdog.start()

    config = build_config(scenario, bench.exposure_ms)
    outcome, exc = "returned", None
    t_start = time.monotonic()
    t_stop = t_start
    try:
        with patched_engine(bench, scenario):
            try:
                engine.optimize_slm_zernike_pib(config)
            except BaseException as caught:
                outcome, exc = "raised", caught
            finally:
                t_stop = time.monotonic()
    finally:
        bench.cancel_timer()
        if watchdog is not None:
            watchdog.cancel()

    cleanup_s = (
        t_stop - bench.inject_t if bench.inject_t is not None else None
    )
    logger.info(
        "{}: outcome={} exc={} cleanup_to_return={}",
        scenario.key,
        outcome,
        type(exc).__name__ if exc is not None else None,
        f"{cleanup_s:.2f}s" if cleanup_s is not None else "n/a",
    )

    recorder = bench.recorders[-1] if bench.recorders else None
    history = getattr(recorder, "history", None) or []
    row0 = history[0] if history else None
    row0_summary = None
    if isinstance(row0, dict):
        row0_summary = {}
        for key in ("J", "pib", "rms_out", "_max_r", "max_brt", "r", "_epoch"):
            value = row0.get(key)
            row0_summary[key] = (
                float(value) if isinstance(value, (int, float, np.floating)) else None
            )
    exit_write = bench.exit_write()
    writes_before = bench.writes_before_inject()
    writes_after = bench.writes_after_inject()
    evidence = verify_panel(bench, scenario, outdir)
    checks = _checks(
        scenario,
        outcome,
        type(exc).__name__ if exc is not None else None,
        str(exc) if exc is not None else None,
        evidence["firmware"],
        exit_write,
        bench.writes,
        evidence["corr"],
        cleanup_s,
        bool(
            recorder is not None
            and isinstance(getattr(recorder, "energy_loss_violations", None), int)
        ),
        row0_summary,
    )

    # ``None`` means "not applicable to this scenario" and is skipped, so a
    # normal-exit run is not failed for lacking a cleanup duration.
    verdict_keys = [
        k for k in VERDICT_KEYS if k in checks and checks[k] is not None
    ]
    failed_keys = [k for k in verdict_keys if not checks[k]]
    passed = not failed_keys
    optical_class = checks.get("optical_class")

    record = {
        "key": scenario.key,
        "name": scenario.name,
        "description": scenario.description,
        "config": {
            "algorithm": scenario.algorithm,
            "epochs": scenario.epochs,
            "pop_size": scenario.pop_size,
            "target_size": scenario.target_size,
            "center": "shape",
            "n_max": 7,
            "exposure_ms": bench.exposure_ms,
            "fold_ratio": 0.0,
            "noise_gate_k": -1.0,
        },
        "injected": {
            "kind": (
                "KeyboardInterrupt"
                if scenario.interrupt_at_write is not None
                else ("CameraError" if scenario.dropout_at_capture is not None else None)
            ),
            "detail": bench.inject_detail,
            "t_iso": bench.inject_iso,
            "fired": bench.inject_t is not None,
        },
        "outcome": {
            "result": outcome,
            "exception_type": type(exc).__name__ if exc is not None else None,
            "exception_message": str(exc) if exc is not None else None,
            "wall_s": t_stop - t_start,
            "cleanup_s": cleanup_s,
        },
        "writes": {
            "n_total": len(bench.writes),
            "n_before_inject": len(writes_before),
            "n_after_inject": len(writes_after),
            "before": [
                {
                    "i": w["i"],
                    "kind": w["kind"],
                    "slot": w["slot"],
                    "min": w.get("min"),
                    "max": w.get("max"),
                    "gray_value": w.get("gray_value"),
                }
                for w in writes_before[-6:]
            ],
            "after": [
                {
                    "i": w["i"],
                    "kind": w["kind"],
                    "slot": w["slot"],
                    "min": w.get("min"),
                    "max": w.get("max"),
                    "gray_value": w.get("gray_value"),
                }
                for w in writes_after
            ],
        },
        "exit_branch": (
            "no_write"
            if exit_write is None
            else (
                "flat_fallback"
                if exit_write["kind"] == "grayscale"
                else "best_phase"
            )
        ),
        "panel": evidence,
        "recorder": {
            "rows": len(history),
            "energy_loss_violations": getattr(recorder, "energy_loss_violations", None),
            "epoch0": row0_summary,
        },
        "checks": checks,
        "verdict_keys": verdict_keys,
        "failed_keys": failed_keys,
        "verdict": "pass" if passed else "fail",
    }
    logger.info(
        "{}: verdict={} failed={} optical={} firmware slot={} gray={}",
        scenario.key,
        record["verdict"],
        failed_keys or "-",
        optical_class,
        evidence["firmware"].get("displayed_slot"),
        evidence["firmware"].get("current_grayscale"),
    )
    return record


def _write_summary(outdir: Path, summary: dict[str, Any]) -> None:
    """Persist the summary after every scenario, so a crash still leaves evidence."""
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def calibrate_exposure(bench: Bench, target_peak: float) -> None:
    """Fix one verification exposure from a flat panel -- shared by every scenario."""
    bench.slm.set_grayscale(0)
    time.sleep(SETTLE_S)
    cam = _open_camera(0.6)
    try:
        frame = auto_exposure(cam, target_peak, n_sample=1)
        bench.exposure_ms = float(get_camera_exposure_ms(cam))
        bench.camera_shape = tuple(int(v) for v in frame.shape)  # type: ignore[assignment]
        logger.info(
            "verification exposure fixed at {:.3f} ms (peak {:.0f} counts, frame {})",
            bench.exposure_ms,
            float(np.max(frame)),
            bench.camera_shape,
        )
    finally:
        cam.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Run only these scenario keys (A B C D E); commas are accepted too.",
    )
    parser.add_argument(
        "--watchdog-s",
        type=float,
        default=150.0,
        help="Hard cap per injected scenario before a watchdog Ctrl-C is fired.",
    )
    parser.add_argument("--target-peak", type=float, default=160.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scenario plan and exit without touching hardware.",
    )
    args = parser.parse_args(argv)

    wanted: set[str] | None = None
    if args.only is not None:
        wanted = {
            key.strip()
            for item in args.only
            for key in str(item).split(",")
            if key.strip()
        }
        known = {s.key for s in DEFAULT_SCENARIOS}
        unknown = sorted(wanted - known)
        if unknown:
            parser.error(f"unknown scenario key(s): {', '.join(unknown)}")

    scenarios = [
        s for s in DEFAULT_SCENARIOS if wanted is None or s.key in wanted
    ]
    if not scenarios:
        parser.error("no scenario matched --only")

    if args.dry_run:
        for s in scenarios:
            logger.info(
                f"{s.key}: {s.name} algorithm={s.algorithm} epochs={s.epochs} "
                f"target_size={s.target_size} interrupt@{s.interrupt_at_write} "
                f"dropout@{s.dropout_at_capture}"
            )
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ARTIFACT_ROOT / f"run_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "issue": 145,
        "title": "H-28 R-8 try/finally hardware safety exit",
        "started_iso": _utc_now(),
        "host": sys.platform,
        "scenarios": [],
    }

    logger.info("opening SLM #1 (memory mode) for the whole session ...")
    bench = Bench(slm=Santec(slm_number=1, video_mode=0))
    bench.slm.open()
    try:
        wavelength_nm, max_gray_2pi = bench.slm.get_wavelength_info()
        summary["slm"] = {
            "slm_number": 1,
            "video_mode": 0,
            "wavelength_nm": int(wavelength_nm),
            "max_gray_for_2pi": int(max_gray_2pi),
        }
        try:
            summary["baseline_slot"] = bench.slm.get_displayed_memory_number()
        except Exception as exc:
            summary["baseline_slot"] = None
            summary["baseline_slot_error"] = f"{type(exc).__name__}: {exc}"
        calibrate_exposure(bench, args.target_peak)
        summary["camera"] = {
            "type": "daheng",
            "cam_id": 0,
            "frame_shape": list(bench.camera_shape),
            "verification_exposure_ms": bench.exposure_ms,
        }

        for scenario in scenarios:
            try:
                record = run_scenario(bench, scenario, outdir, args.watchdog_s)
            except BaseException as exc:
                logger.exception("{}: harness error, scenario abandoned", scenario.key)
                try:  # leave something safe behind before moving on
                    bench.slm.set_grayscale(0)
                except Exception:
                    pass
                record = {
                    "key": scenario.key,
                    "name": scenario.name,
                    "description": scenario.description,
                    "config": {
                        "algorithm": scenario.algorithm,
                        "epochs": scenario.epochs,
                        "pop_size": scenario.pop_size,
                        "target_size": scenario.target_size,
                    },
                    "injected": {"kind": None, "detail": "", "t_iso": None, "fired": False},
                    "outcome": {
                        "result": "harness_error",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "wall_s": None,
                        "cleanup_s": None,
                    },
                    "writes": {
                        "n_total": len(bench.writes),
                        "n_before_inject": 0,
                        "n_after_inject": 0,
                        "before": [],
                        "after": [],
                    },
                    "exit_branch": "unknown",
                    "panel": {"firmware": {}, "corr": {}, "peaks": {}, "frame_files": []},
                    "recorder": {"rows": 0, "energy_loss_violations": None, "epoch0": None},
                    "checks": {},
                    "verdict_keys": [],
                    "failed_keys": ["harness_error"],
                    "verdict": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            summary["scenarios"].append(record)
            _write_summary(outdir, summary)
    finally:
        try:
            bench.slm.set_grayscale(0)
            logger.info("SLM returned to flat gray-0; closing")
        except Exception as exc:
            logger.error("failed to leave the panel at flat: {}", exc)
        bench.slm.close()

    summary["finished_iso"] = _utc_now()
    _write_summary(outdir, summary)

    logger.info()
    logger.info(f"artifacts: {outdir}")
    logger.info(f"{'':2} {'scenario':<30} {'verdict':<6} {'exit branch':<14} "
          f"{'slot':>5} {'gray':>5} {'corr flat':>9} {'corr exit':>9}  failed")
    for rec in summary["scenarios"]:
        panel = rec["panel"]
        firmware = panel.get("firmware") or {}
        corr = panel.get("corr") or {}

        def fmt(value):
            return "-" if value is None else f"{value:.4f}"

        logger.info(
            f"{rec['key']:>2} {rec['name']:<30} {rec['verdict']:<6} "
            f"{rec['exit_branch']:<14} {str(firmware.get('displayed_slot')):>5} "
            f"{str(firmware.get('current_grayscale')):>5} "
            f"{fmt(corr.get('live_vs_flat')):>9} {fmt(corr.get('live_vs_exit')):>9}  "
            f"{','.join(rec.get('failed_keys') or []) or '-'}"
        )
    return 0 if all(r["verdict"] == "pass" for r in summary["scenarios"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
