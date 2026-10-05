"""Is the camera really in manual exposure, and how much does it drift?

Why this is a bench tool and not a one-off
-----------------------------------------
Every quantitative number on this bench -- speckle correlation, sweep
calibration, shaping metrics -- is a ratio or a fitted slope, so a drifting
exposure corrupts all of them silently. On this bench the camera turned out to be
fine (``ExposureAuto`` off, ``ExposureTime`` reading back exactly, 4.4 % peak drift
over 12 grabs), which is worth *knowing* rather than assuming: the same apparent
symptom has two very different causes.

The two causes that actually produce "the bench got 4x dimmer between runs"
-------------------------------------------------------------------------
1. **Camera gain drift.** What this tool rules out.
2. **The SLM retains the last displayed pattern.** A "flat" reference captured
   before any write is the previous run's speckle, not a flat field. This is the
   one that bit us: the same 3 ms setting measured 100 counts in one run and 23 in
   the next. The driver is fine; the *reference* was wrong. Use
   :func:`ao_shaping.tools.slm.bench_kernels.measure_flat_reference`, which
   displays flat first.

What it reports
---------------
* the camera's ``ExposureAuto`` / ``ExposureTime`` / gain state as opened, and
  again after forcing manual -- ``_init_exposure`` logs a warning when it fails to
  disable auto exposure, and a saved CCD config can switch it back on afterwards;
* the exposure readback, so a silently clamped value is visible;
* peak and total-sum drift over a series of grabs at a fixed setting.

Usage
-----
::

    python -m ao_shaping.tools.slm.slm_exposure_check --exposure-ms 3.0
    python -m ao_shaping.tools.slm.slm_exposure_check --grab-delay-s 1.0 --n-grabs 30
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated

import click
import numpy as np
from loguru import logger

from ao_shaping.utils.cli.params import option, with_params


@dataclass
class ExposureCheckParams:
    """CLI surface of the exposure-check probe.

    Field order *is* the ``--help`` order, and the whole five-option surface is
    frozen by ``tests/ao_shaping/runners/_cli_help_golden.json``.

    This probe is camera-only, so neither shared group is a fit:
    ``SlmBenchParams`` would add ``--slm-number``/``--slm-wavelength`` that this
    command does not own, and ``SlmAcquireParams`` adds
    ``--discard``/``--settle-s``/``--stable-tol``/``--max-wait-s``. Both also type
    ``--cam-type`` as a ``click.Choice``, which renders ``[daheng|miicam]``
    instead of the frozen ``TEXT``.
    """

    cam_type: Annotated[
        str, option("--cam-type", help="相机类型 (daheng/miicam, 默认 daheng)")
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="相机 ID (默认 0)")] = 0
    exposure_ms: Annotated[
        float, option("--exposure-ms", help="相机曝光 ms (默认 3.0)")
    ] = 3.0
    n_grabs: Annotated[
        int, option("--n-grabs", help="连续采集次数 (默认 12)")
    ] = 12
    grab_delay_s: Annotated[
        float, option("--grab-delay-s", help="采集间隔 s (默认 0.4)")
    ] = 0.4


@click.command()
@with_params(ExposureCheckParams, kw_name="params")
def main(params: ExposureCheckParams) -> None:
    """检查相机自动曝光状态与固定设置下的亮度漂移。"""
    from ao_shaping.drivers.ccd.common import create_camera

    # Local aliases keep the measurement body below verbatim.
    cam_type = params.cam_type
    cam_id = params.cam_id
    exposure_ms = params.exposure_ms
    n_grabs = params.n_grabs
    grab_delay_s = params.grab_delay_s

    with create_camera(cam_type, cam_id, exposure_time_ms=exposure_ms) as cam:
        # The GenICam feature tree is Daheng-specific and is not on the base
        # camera interface, so reach it defensively rather than pretending every
        # backend has it.
        genicam = getattr(cam, "cam", None)

        logger.info("=== state as opened ===")
        _log_state(cam, "as opened")

        logger.info("")
        logger.info("=== forcing manual ===")
        for label, fn in (
            ("enable_auto_exposure(False)", lambda: cam.enable_auto_exposure(False)),
            ("ExposureAuto=OFF", lambda: _genicam_set(genicam, "ExposureAuto", 0)),
            ("Gain=0", lambda: _genicam_set(genicam, "Gain", 0.0)),
        ):
            try:
                logger.info("  {:<30} -> {!r}", label, fn())
            except (AttributeError, RuntimeError, OSError, ValueError) as exc:
                logger.warning("  {:<30} -> FAILED {}: {}", label, type(exc).__name__, exc)
        _log_state(cam, "after forcing manual")

        try:
            logger.info(
                "ExposureTime.get() = {} us (requested {:.0f} us)",
                _genicam_get(genicam, "ExposureTime"),
                exposure_ms * 1000.0,
            )
        except (AttributeError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("exposure readback failed {}: {}", type(exc).__name__, exc)

        logger.info("")
        logger.info("=== drift at a fixed setting ({} grabs) ===", n_grabs)
        peaks: list[float] = []
        sums: list[float] = []
        for i in range(int(n_grabs)):
            img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
            peaks.append(float(img.max()))
            sums.append(float(img.sum()))
            time.sleep(float(grab_delay_s))
            logger.info("   {:>2}: peak={:>6.0f} sum={:>10.0f}", i + 1, peaks[-1], sums[-1])

    p = np.array(peaks)
    s = np.array(sums)
    peak_spread = (p.max() - p.min()) / max(p.mean(), 1e-9)
    sum_spread = (s.max() - s.min()) / max(s.mean(), 1e-9)
    logger.info("")
    logger.info(
        "peak: min={:.0f} max={:.0f} spread={:.1%} of mean", p.min(), p.max(), peak_spread
    )
    logger.info(
        "sum : min={:.0f} max={:.0f} spread={:.1%} of mean", s.min(), s.max(), sum_spread
    )
    if peak_spread > 0.15:
        logger.error(
            "peak drift is {:.1%} -- too large to ignore. Check the laser output "
            "first (it drifts ~15% day to day here), then re-check that flat is "
            "*displayed* before each reference.", peak_spread,
        )
    else:
        logger.info(
            "drift is small, so the camera is not the noise source. If brightness "
            "still differs between runs, the SLM is holding the previous run's "
            "pattern -- display flat before reading a reference."
        )


def _genicam_set(genicam, feature: str, value) -> None:
    """Set a GenICam feature, with a clear error when the backend lacks it."""
    if genicam is None:
        raise AttributeError(
            f"this camera backend has no GenICam feature tree (cannot set {feature})"
        )
    getattr(genicam, feature).set(value)


def _genicam_get(genicam, feature: str):
    """Read a GenICam feature, with a clear error when the backend lacks it."""
    if genicam is None:
        raise AttributeError(
            f"this camera backend has no GenICam feature tree (cannot read {feature})"
        )
    return getattr(genicam, feature).get()


def _log_state(cam, tag: str) -> None:
    try:
        logger.info("  [{}] {}", tag, cam.get_auto_exposure_state())
    except (AttributeError, NotImplementedError, RuntimeError, OSError) as exc:
        logger.warning("  [{}] state unavailable {}: {}", tag, type(exc).__name__, exc)


if __name__ == "__main__":
    main()
