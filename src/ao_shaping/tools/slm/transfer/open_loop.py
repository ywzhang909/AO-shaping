"""Open-loop SLM display for a session that must not open the camera.

``ccd_mode="none"`` keeps the model-based synthesis but drops every CCD stage.
The synthesized phase of each shape is then put on the SLM, one after another,
with a short dwell: an operator can look at the far field while the camera and
its SDK stay closed. This module owns only that display step, separated from
``workflow`` so the device-free path stays testable with an injected factory.

Device discipline (``docs/dev/hardware_discipline.md``): the SLM is opened
exactly once, every open/close is logged, the bench :class:`DeviceLock` is taken
before the handle and released in ``finally``, and the two never run back to
back.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from .capture import clock, to_panel

#: Phase that ``workflow._offline_synthesize`` writes into ``<shape>_offline_seed.npz``.
PHASE_KEY = "optimized_phase"
#: Seconds each phase stays on the panel before the next one is written.
DEFAULT_DWELL_S = 2.0
#: Source of the panel phase handed to ``display_data``.
WAVELENGTH_NM = 1064


def _default_slm_factory(*, slm_number: int, wavelength_nm: int = WAVELENGTH_NM):
    """Construct (but do not open) the Santec SLM used by the measured stages."""
    from ao_shaping.drivers.slm.santec import Santec

    return Santec(slm_number=slm_number, wavelength=wavelength_nm, video_mode=0)


def _default_lock():
    """Cross-process bench lock; tests replace this hook with a fake."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    return DeviceLock.default()


def _load_phase(source: Path, shape: str) -> tuple[np.ndarray, Path]:
    """Read the final synthesized phase of ``shape`` (radians, 2-D, finite)."""
    path = Path(source) / f"{shape}_offline_seed.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"open-loop display needs {path} (written by the synthesis stage)"
        )
    with np.load(path) as payload:
        phase = np.asarray(payload[PHASE_KEY], dtype=np.float32)
    if phase.ndim != 2:
        raise ValueError(f"{path}: {PHASE_KEY} must be 2-D, got shape {phase.shape}")
    if not np.isfinite(phase).all():
        raise ValueError(f"{path}: {PHASE_KEY} contains non-finite values")
    return phase, path


def display_phases(
    shapes,
    *,
    source: Path,
    out_path: Path,
    dwell_s: float = DEFAULT_DWELL_S,
    slm_number: int = 1,
    wavelength_nm: int = WAVELENGTH_NM,
    slm_factory=None,
    lock=None,
    sleep=None,
) -> dict:
    """Show each shape's final phase on the SLM, without a camera.

    Args:
        shapes: Shape names, in display order.
        source: Directory holding ``<shape>_offline_seed.npz``.
        out_path: Destination of ``open_loop_display.json``.
        dwell_s: Seconds each phase stays on the panel (must be > 0).
        slm_number: Santec SLM number.
        wavelength_nm: Programming wavelength recorded with the run.
        slm_factory: Injected constructor for tests; ``None`` opens the real
            :class:`~ao_shaping.drivers.slm.santec.Santec`.
        lock: Injected :class:`DeviceLock`; ``None`` takes the bench lock.
        sleep: Injected sleep for tests; ``None`` uses :func:`time.sleep`.

    Returns:
        The payload written to ``out_path``.

    Raises:
        ValueError: a phase file is malformed or ``dwell_s`` is not positive.
        FileNotFoundError: the synthesis stage did not write a shape's phase.
        RuntimeError: the bench lock is held by another process.
    """
    dwell = float(dwell_s)
    if not np.isfinite(dwell) or dwell <= 0:
        raise ValueError(f"dwell_s must be a positive finite number, got {dwell_s!r}")

    sleeper = time.sleep if sleep is None else sleep
    factory = _default_slm_factory if slm_factory is None else slm_factory
    bench_lock = _default_lock() if lock is None else lock
    if not bench_lock.acquire(timeout=1800):
        raise RuntimeError("open-loop display could not take the bench device lock")

    slm = factory(slm_number=slm_number, wavelength_nm=wavelength_nm)
    entries: list[dict] = []
    try:
        logger.info("open-loop: 打开 SLM #{} (不开相机)", slm_number)
        slm.open()
        logger.info("open-loop: SLM #{} 已打开", slm_number)
        # Duck-typed drive-board admission guard (#186): a dead drive board
        # leaves the panel unmodulated, so the operator would see a blank frame.
        from ao_shaping.drivers.slm.factory import require_drive_board_entry

        require_drive_board_entry(slm)
        for shape in shapes:
            phase, path = _load_phase(Path(source), str(shape))
            slm.display_data(slm.create_phase_from_array(to_panel(phase)))
            entry = dict(
                shape=str(shape),
                phase_file=str(path),
                dwell_s=dwell,
                phase_rms=float(np.sqrt(np.mean(np.square(phase, dtype=np.float64)))),
                **clock(),
            )
            entries.append(entry)
            logger.info("open-loop: 上屏 {}，驻留 {:g} s", shape, dwell)
            sleeper(dwell)
    finally:
        try:
            slm.close()
            logger.info("open-loop: SLM #{} 已关闭", slm_number)
        except Exception as exc:  # best-effort close, never mask the real error
            logger.warning(
                "open-loop: 关闭 SLM #{} 失败 ({}): {}",
                slm_number,
                type(exc).__name__,
                exc,
            )
        finally:
            bench_lock.release()

    payload = dict(
        mode="open_loop",
        ccd_mode="none",
        acceptance="open_loop",
        slm_number=int(slm_number),
        wavelength_nm=int(wavelength_nm),
        dwell_s=dwell,
        shapes=[str(shape) for shape in shapes],
        entries=entries,
        **clock(),
    )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def write_summary(display_path: Path, dest: Path) -> dict:
    """Record the open-loop acceptance verdict for the ``summary`` stage.

    ``ccd_mode="none"`` never creates an ``anchored/`` directory, so the generic
    ``summary.summarize`` (which reads the anchored blocks' ``config.json`` and
    ``manifest.json``) cannot run. This writes a compact ``summary.json`` from
    the display payload instead, keeping the ``summary`` stage present in the
    plan and the journal.
    """
    display = json.loads(Path(display_path).read_text(encoding="utf-8"))
    summary = dict(
        run=str(Path(display_path).parent),
        ccd_mode="none",
        acceptance="open_loop",
        mode=display.get("mode"),
        slm_number=display.get("slm_number"),
        wavelength_nm=display.get("wavelength_nm"),
        dwell_s=display.get("dwell_s"),
        shapes=display.get("shapes"),
        entries=display.get("entries"),
    )
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
