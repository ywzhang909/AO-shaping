"""SLM calibration package (geometry + shift + legacy LUT calibrators).

The single-file toolkit was split into a package (2026-10-12) alongside the
``tools/`` reorganisation. The calibration code itself lives in
``calibration.py``; this package re-exports its public API **lazily** so that
``from ao_shaping.tools.slm.calibration import SantecCalibrator`` keeps working
while ``import ao_shaping.tools.slm.calibration.train_data_collect`` stays cheap
— ``calibration.py`` imports the Daheng / MiiCam / Santec drivers at module
scope, and an eager re-export here would drag a native SDK into every
``train_data_collect`` import (guarded by
``test_slm_train_data_collect.TestNoHardware``).

The canonical gray↔phase LUT is **not** here: it lives in
``tools/slm/calibration/lut_runner.py`` (``slm-lut``) + ``utils/slm/slm_lut``,
consumed by ``Santec.load_lut``. ``SLMLUTCalibrator`` is a deprecated
self-reference-interferometry reference implementation kept for study/unit tests.

CLI (see ``__main__``):
    python -m ao_shaping.tools.slm.calibration [OPTIONS]        # geometry calibration
    python -m ao_shaping.tools.slm.calibration shift [OPTIONS]  # defocus shift calibration
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AutoExposureController",
    "CalibrationMethod",
    "CalibrationResult",
    "CameraProtocol",
    "CameraWithExposureProtocol",
    "DiffractionEfficiencyCalibrator",
    "GeometryCalibParams",
    "InterferometerCalibrator",
    "SantecCalibrator",
    "ShiftCalibParams",
    "SLMCalibratorBase",
    "SLMCCDCalibrator",
    "SLMLUTCalibrator",
    "SLMProtocol",
    "calibrate_santec",
    "create_calibration_curve",
    "diagnose_radius",
    "main",
    "main_shift_calib",
    "plot_calibration_result",
]

_IMPL = "ao_shaping.tools.slm.calibration.calibration"


def __getattr__(name: str) -> Any:
    """Lazily resolve the public API from ``calibration.calibration``."""
    if name in __all__:
        module = importlib.import_module(_IMPL)
        value = getattr(module, name)
        globals()[name] = value  # cache so subsequent access is a plain lookup
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
