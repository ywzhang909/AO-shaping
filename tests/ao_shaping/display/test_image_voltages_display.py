"""Tests for the canonical ``ImageVoltagesDisplay``.

``ao_shaping.display`` owns the single implementation; the legacy
``ao_shaping.utils[.image].display`` paths re-export it, so both must resolve to
the same class. Runs headless via the SDL dummy video driver.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest


def test_single_implementation_across_import_paths():
    import ao_shaping.display as display_pkg
    import ao_shaping.utils.image.display as utils_display
    from ao_shaping.utils import ImageVoltagesDisplay as utils_pkg_cls

    assert display_pkg.ImageVoltagesDisplay is utils_display.ImageVoltagesDisplay
    assert utils_pkg_cls is display_pkg.ImageVoltagesDisplay
    assert utils_display.VOLT_HEIGHT == display_pkg.windows.VOLT_HEIGHT


class TestImageVoltagesDisplay:
    def test_seven_arg_render_and_close(self):
        pytest.importorskip("pygame")
        from ao_shaping.display import ImageVoltagesDisplay

        disp = ImageVoltagesDisplay((40, 30), volt_height=20)
        disp.init_window()
        ok = disp.render(
            np.random.randint(0, 255, (30, 40), dtype=np.uint8),
            np.linspace(-300, 500, 8),
            -300,
            500,
            (20, 15),
            5,
            "epoch 1",
        )
        assert ok is True
        disp.close()

    def test_context_manager_and_robust_inputs(self):
        pytest.importorskip("pygame")
        from ao_shaping.display import ImageVoltagesDisplay

        with ImageVoltagesDisplay((24, 24), volt_height=10) as disp:
            # float image + a NaN voltage must not raise.
            volts = np.array([np.nan, 0.5, -0.25, 1.0])
            assert (
                disp.render(np.random.rand(24, 24), volts, -1.0, 1.0, (12, 12), 3.0)
                is True
            )

    def test_dm_optimizer_call_pattern(self):
        """Mirror optimize_pib's exact positional call."""
        pytest.importorskip("pygame")
        from ao_shaping.display import ImageVoltagesDisplay

        window = ImageVoltagesDisplay((32, 32))
        window.init_window()
        try:
            assert window.render(
                np.zeros((32, 32), dtype=np.uint8),
                np.zeros(64),
                -20.0,
                120.0,
                (16, 16),
                4,
                "0",
            )
        finally:
            window.close()
