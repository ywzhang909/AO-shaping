"""The ``sim`` camera type must be available without a manual registration call.

Reproduced 2026-10-01 through the real CLI::

    python src/ao_shaping/main.py slm-pib spgd --cam_type sim -e 3
    ...
    ValueError: Unknown camera type: 'sim'.
             Available: ['daheng', 'ffmpeg', 'image_folder', 'miicam']

``--cam_type sim`` is an advertised choice on the ``slm-pib`` / ``slm-gsnet``
families and the README documents ``slm-pib spgd --cam_type sim`` as a working
command, yet the only registration call sites were
``runners/slm/gsnet_runner.py`` (inside its ``_maybe_sim_patch``) and the
``scripts/slm_pib_sim_run.py`` harness — so the plain CLI path could never work.

The fix makes the invariant explicit: importing the simulated optical system
registers the simulated camera that reads its far field. Anyone who has the sim
SLM/System necessarily has the sim CCD.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tests.ao_shaping.drivers._subprocess_env import subprocess_env_with_pythonpath

_ROOT = Path(__file__).resolve().parents[4]


def _run(snippet: str) -> subprocess.CompletedProcess[str]:
    """Execute ``snippet`` in a fresh interpreter rooted at the repo.

    The env comes from the shared helper. This used to hardcode a POSIX literal
    (``PATH=/usr/bin:/bin``, ``HOME=/tmp``) and point ``_ROOT`` at ``tests/``: on
    Windows, stripping ``SYSTEMROOT`` killed the child with ``WinError 10106`` before
    any assertion ran, and the colon-separated ``"src:libs"`` ``PYTHONPATH`` was inert,
    so imports resolved only because the package is pip-installed.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=subprocess_env_with_pythonpath(root=str(_ROOT), absolute=True),
        timeout=180,
    )


class TestSimCameraAutoRegistration:
    def test_importing_the_sim_system_registers_the_sim_camera(self):
        result = _run(
            """
            from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem
            from ao_shaping.drivers.ccd.common import list_camera_types
            print("CAMERAS", sorted(list_camera_types()))
            assert "sim" in list_camera_types(), list_camera_types()
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "'sim'" in result.stdout and "OK" in result.stdout

    def test_create_camera_sim_needs_no_manual_registration(self):
        result = _run(
            """
            from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem
            from ao_shaping.drivers.ccd.common import create_camera
            cam = create_camera("sim", cam_id=0, exposure_time_ms=80.0)
            print("CAM", type(cam).__name__)
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "CAM SimPibCCD" in result.stdout

    def test_importing_drivers_sim_registers_it_too(self):
        """``import ao_shaping.drivers.sim`` is the coarse entry point."""
        result = _run(
            """
            import ao_shaping.drivers.sim  # noqa: F401
            from ao_shaping.drivers.ccd.common import list_camera_types
            assert "sim" in list_camera_types(), list_camera_types()
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK" in result.stdout

    def test_registration_is_idempotent(self):
        """Re-registering must not raise or duplicate the entry."""
        result = _run(
            """
            from ao_shaping.drivers.sim.slm_pib_sim import register_sim_camera
            from ao_shaping.drivers.ccd.common import list_camera_types
            before = sorted(list_camera_types())
            register_sim_camera()
            register_sim_camera()
            after = sorted(list_camera_types())
            assert before == after, (before, after)
            print("OK", after.count("sim"))
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK 1" in result.stdout

    def test_hardware_backends_still_register(self):
        """Lazy registration must not drop or shadow the real backends."""
        result = _run(
            """
            from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem
            from ao_shaping.drivers.ccd.common import list_camera_types
            cams = set(list_camera_types())
            missing = {"daheng", "miicam", "sim"} - cams
            assert not missing, f"missing backends: {missing} (have {sorted(cams)})"
            print("OK", sorted(cams))
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK" in result.stdout
