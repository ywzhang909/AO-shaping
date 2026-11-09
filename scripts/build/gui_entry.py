"""Nuitka standalone launcher for the AO-Shaping Qt console.

Configures runtime paths (vendor DLLs, SDK locations, config directories)
*before* the GUI package is imported, then hands control to
``ao_shaping.gui.ccd.beam_quality.shell.main``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _runtime_root() -> Path:
    """Return the directory that holds the bundled resources.

    Frozen (Nuitka standalone): the directory containing the EXE.
    Source checkout: the repository root (this file lives in scripts/build/).
    """
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _configure_runtime(root: Path) -> None:
    slm_dll_dir = root / "libs" / "santec_slm"
    daheng_dll_dir = root / "libs" / "daheng"
    miicam_sdk_dir = root / "ao_shaping" / "drivers" / "ccd" / "_miicam_sdk"
    data_dir = root / "data"

    # Native DLL search locations (Windows does not search these by default).
    # ``WinDLL(..., winmode=0)`` (Daheng wrappers) uses the legacy search order,
    # which ignores add_dll_directory, so PATH is extended as well.
    dll_dirs = [d for d in (slm_dll_dir, daheng_dll_dir, miicam_sdk_dir) if d.is_dir()]
    for dll_dir in dll_dirs:
        os.add_dll_directory(str(dll_dir))
    os.environ["PATH"] = os.pathsep.join([*map(str, dll_dirs), os.environ.get("PATH", "")])

    os.environ.setdefault("AO_SHAPING_SLM_DLL_DIR", str(slm_dll_dir))
    os.environ.setdefault("AO_SHAPING_SDK_PATH", str(root / "libs"))
    os.environ.setdefault("MIICAM_SDK_PATH", str(miicam_sdk_dir))
    os.environ.setdefault("CCD_CONFIG_DIR", str(data_dir / "ccd_configs"))
    os.environ.setdefault("SLM_CONFIG_DIR", str(data_dir / "slm_configs"))
    os.environ.setdefault("DM_CONFIG_DIR", str(data_dir / "dm_configs"))
    os.environ.setdefault("MICRO_DM_CONFIG_DIR", str(data_dir / "micro_dm_configs"))


def _fix_root_dir(root: Path) -> None:
    """Point ``ao_shaping.utils.io.file.ROOT_DIR`` at the bundle root.

    In standalone builds ``Path(__file__)`` resolves one level above the
    ``.dist`` folder, so ROOT_DIR is wrong for icon and default-config lookups.
    Patching it here works for modules that read ROOT_DIR at call time.
    """
    import ao_shaping.utils.io.file as file_io

    file_io.ROOT_DIR = root


def main() -> int:
    root = _runtime_root()
    _configure_runtime(root)
    _fix_root_dir(root)

    from ao_shaping.gui.ccd.beam_quality.shell import main as shell_main

    return shell_main()


if __name__ == "__main__":
    raise SystemExit(main())
