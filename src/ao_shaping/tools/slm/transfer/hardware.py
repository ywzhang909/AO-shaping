"""Explicit hardware-runtime setup; nothing is loaded at module import."""
from __future__ import annotations


def prepare_camera_sdk(camera):
    """Expose the bundled Daheng binding only when a hardware action is requested."""
    if camera != "daheng":
        return
    import importlib.util
    import os
    from pathlib import Path
    import sys

    if importlib.util.find_spec("gxipy") is None:
        libs = Path(os.environ.get("AO_SHAPING_SDK_PATH", Path(__file__).resolve().parents[5] / "libs"))
        if not (libs / "gxipy/__init__.py").is_file():
            raise ImportError("gxipy unavailable; install the vendor binding or set AO_SHAPING_SDK_PATH to its parent directory")
        sys.path.insert(0, str(libs))


def enumerate_camera():
    """Read camera discovery without opening a stream or loading an SLM phase."""
    from .capture import clock

    prepare_camera_sdk("daheng")
    import gxipy as gx
    count, devices = gx.DeviceManager().update_device_list()
    return dict(camera="daheng", count=count, devices=devices, **clock())
