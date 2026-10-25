"""波前传感器驱动 (Thorlabs WFS 与仿真传感器)。"""
from __future__ import annotations

from ao_shaping.drivers.wfs._registry import create_wfs, resolve_wfs
from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes, ThorlabWFS, WfsError

__all__ = ["BaseWFS", "MlaRes", "ThorlabWFS", "WfsError", "create_wfs", "resolve_wfs"]
