"""Bench device factories for the SLM + CCD shaping optimizers.

The wavefront-sensorless shaping optimizers open their devices *inside*
the optimizer call (the SDK handles are not re-entrant, so a device is
never reused across runs). This module is the single home for that
construction:

- :func:`open_slm` builds the real Santec panel, or the simulated
  panel for ``cam_type="sim"``.
- :func:`open_camera` builds the camera through the ``ccd`` registry
  (the ``sim`` backend must already be registered by the caller).
- :func:`sync_device_wavelength` adopts the wavelength the SLM reports
  into the optimizer config, so the GS propagation sees the wavelength
  the panel is really programmed for.

:func:`open_camera` lives beside the SLM factory rather than in
``ccd/`` because the two devices are always opened as one bench unit:
every consumer constructs both, back to back, from the same parameter
object.

The parameter object is duck-typed (:class:`BenchDeviceParams`) so this
module has no dependency on the optimizer layer -- drivers must not
import the optimizers that consume them. Backend imports stay
function-local, preserving the drivers' lazy loading contract
(``import ao_shaping`` loads no native SDK).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

__all__ = [
    "BenchDeviceParams",
    "DEFAULT_WAVELENGTH_NM",
    "open_slm",
    "open_camera",
    "sync_device_wavelength",
]

#: Wavelength adopted when the config did not pin one and the SLM
#: reports nothing (the simulated panel has no ``wavelength``
#: attribute). The lab has more than one SLM (see the serial-number
#: conflict in drivers/AGENTS.md), so the device is always asked
#: first and this is only the fallback.
DEFAULT_WAVELENGTH_NM = 1064.0


class BenchDeviceParams(Protocol):
    """Device-selection fields shared by the shaping optimizer configs.

    Satisfied structurally by ``SlmGsRefineConfig`` (and the runner-side
    params dataclass mirroring it), so the factories need no import from
    the optimizer layer.
    """

    cam_type: str
    cam_id: int
    exposure_time_ms: float
    cam_size: int
    slm_number: int
    slm_wavelength: int


def sync_device_wavelength(config: BenchDeviceParams, slm: Any) -> None:
    """Adopt the SLM's actual wavelength when the config did not pin one.

    Runs right after the SLM opens and before any phase math, so the GS
    propagation and the focal-length derivation both see the wavelength the
    panel is really programmed for. The simulated SLM has no wavelength
    attribute, in which case the bench fallback stays in place.
    """
    if config.slm_wavelength > 0:
        return
    reported = getattr(slm, "wavelength", None)
    if not isinstance(reported, int | float) or reported <= 0:
        # The simulated panel reports nothing. Adopt the bench fallback so no
        # downstream phase math ever sees a zero wavelength -- but say so,
        # because the caller asked the device and got nothing.
        config.slm_wavelength = int(DEFAULT_WAVELENGTH_NM)
        logger.info(
            "SLM 未报告工作波长(仿真面板?), 几何推导沿用台架fallback {}nm; "
            "如实际不同请显式传 --slm-wavelength",
            DEFAULT_WAVELENGTH_NM,
        )
        return
    config.slm_wavelength = int(reported)
    logger.info("波长改用 SLM 实际报告值 {}nm (原默认 0 = 询问设备)", reported)
    # The derived focal length needs no recompute: it is referred through
    # the tilt-shift scale's reference wavelength, so it is
    # wavelength-invariant (which a lens focal length must be). What the
    # device's wavelength DOES change is the GS propagation, from here on.


def open_slm(params: BenchDeviceParams) -> Any:
    """Build the SLM: real Santec, or the simulated panel for ``cam_type=sim``."""
    if params.cam_type == "sim":
        from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib

        return SimSLMPib()
    from ao_shaping.drivers.slm.santec import Santec

    # ``Santec.from_params`` reads ``slm_number`` / ``slm_wavelength`` off
    # the params object and maps a 0 wavelength to ``None`` ("read the
    # wavelength configured on the device") -- exactly the "ask the device"
    # semantics :func:`sync_device_wavelength` relies on, so no adapter
    # carrier is needed here.
    return Santec.from_params(params)


def open_camera(params: BenchDeviceParams) -> Any:
    """Build the camera and window it to ``cam_size``.

    ``create_camera`` forwards only the keys in the backend's ``accepted_kwargs``,
    and Daheng's allow-list does **not** include ``cam_size`` -- so the window has
    to be set explicitly with ``reset_window``.
    """
    from ao_shaping.drivers.ccd.common import create_camera

    if params.cam_type == "sim":
        return create_camera(
            "sim",
            cam_id=params.cam_id,
            exposure_time_ms=params.exposure_time_ms,
            cam_size=params.cam_size,
        )

    return create_camera(
        params.cam_type,
        cam_id=params.cam_id,
        exposure_time_ms=params.exposure_time_ms,
    )
