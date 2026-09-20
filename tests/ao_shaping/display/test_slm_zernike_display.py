"""Tests for the pygame rendering now hosted in ``ao_shaping.display``.

Runs headless by forcing the SDL dummy video/audio drivers (set before pygame is
initialised). Covers the display-only normaliser, the dtype-robust image frames
and the :class:`SlmZernikeDisplay` live view.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from ao_shaping.display import (
    AutoDisplay,
    DisplayClosedError,
    EpochCurveFrame,
    FrameInfo,
    Image2DFrame,
    Image2DWithBucketFrame,
    SlmZernikeDisplay,
    to_display_uint8,
)
from ao_shaping.display.frames import Title_height


class TestToDisplayUint8:
    def test_uint8_passthrough(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = to_display_uint8(img)
        assert out.dtype == np.uint8
        assert np.array_equal(out, img)

    def test_float_phase_scaled_by_peak(self):
        phase = np.linspace(0.0, 2 * np.pi, 25).reshape(5, 5)
        out = to_display_uint8(phase)
        assert out.dtype == np.uint8
        assert out.max() == 255
        assert out.min() == 0

    def test_uint16_scaled(self):
        img = np.full((4, 4), 4096, dtype=np.uint16)
        out = to_display_uint8(img)
        assert out.dtype == np.uint8
        assert out.max() == 255

    def test_nan_and_inf_are_zero(self):
        img = np.array([[np.nan, np.inf, -np.inf, 2.0]])
        out = to_display_uint8(img)
        assert out.dtype == np.uint8
        assert out[0, 0] == 0
        assert out[0, 1] == 0
        assert out[0, 2] == 0
        assert out[0, 3] == 255

    def test_all_zero_is_safe(self):
        out = to_display_uint8(np.zeros((3, 3)))
        assert out.dtype == np.uint8
        assert out.max() == 0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            to_display_uint8(np.zeros((2, 2, 3)))


class TestFramesAcceptAnyDtype:
    """Frame inputs are normalised inside the display layer, not by callers."""

    def test_image2d_renders_float_phase(self):
        pygame = pytest.importorskip("pygame")
        pygame.init()
        window = pygame.display.set_mode((80, 80))
        frame = Image2DFrame(window, (0, 0), (80, 80), title="phase")
        frame.render(np.random.rand(20, 20))  # float radian-like input
        pygame.quit()

    def test_bucket_frame_scales_source_coords(self):
        pygame = pytest.importorskip("pygame")
        pygame.init()
        window = pygame.display.set_mode((80, 80))
        frame = Image2DWithBucketFrame(window, (0, 0), (80, 80), title="ccd")
        # center / r are in source-image coordinates; the frame scales them.
        frame.render(np.full((40, 40), 100, dtype=np.uint8), (20, 20), 5)
        pygame.quit()


class TestSlmZernikeDisplay:
    def test_context_manager_and_update(self):
        pytest.importorskip("pygame")
        with SlmZernikeDisplay(frame_size=(64, 64), display_size=(160, 160)) as disp:
            ok = disp.update(
                img=np.random.randint(0, 255, (32, 32), dtype=np.uint8),
                phase=np.random.rand(60, 60) * 2 * np.pi,
                coeffs=np.linspace(-1.0, 1.0, 15),
                center=(16, 16),
                r_bucket=6.0,
                info="epoch 1",
                value=0.5,
                epoch=1,
                total_epochs=10,
            )
            assert ok is True
            assert disp.closed is False
        assert disp.closed is True

    def test_curve_panel_records_metric(self):
        pytest.importorskip("pygame")
        with SlmZernikeDisplay(frame_size=(64, 64), display_size=(160, 160)) as disp:
            for epoch, value in ((1, 0.1), (2, 0.3)):
                disp.update(
                    np.zeros((8, 8), dtype=np.uint8),
                    np.zeros((8, 8)),
                    np.zeros(3),
                    (4, 4),
                    2.0,
                    value=value,
                    epoch=epoch,
                    total_epochs=4,
                )
            assert disp._frames["curve"].data == {
                "epoch": [1.0, 2.0],
                "value": [0.1, 0.3],
            }

    def test_update_after_close_returns_false(self):
        pytest.importorskip("pygame")
        disp = SlmZernikeDisplay(frame_size=(64, 64), display_size=(160, 160))
        disp.init_window()
        disp.close()

        assert (
            disp.update(
                np.zeros((8, 8), dtype=np.uint8),
                np.zeros((8, 8)),
                np.zeros(3),
                (4, 4),
                2.0,
                "",
            )
            is False
        )

    def test_close_without_init_is_safe(self):
        pytest.importorskip("pygame")
        disp = SlmZernikeDisplay()
        disp.close()  # never initialised -> must not raise
        assert disp.closed is True

    def test_display_closed_error_is_runtime_error(self):
        assert issubclass(DisplayClosedError, RuntimeError)

    def test_default_layout_is_2x2(self):
        pytest.importorskip("pygame")
        frame_size, margin = (100, 100), 10
        with SlmZernikeDisplay(
            frame_size=frame_size, display_size=(240, 240), margin=margin
        ) as disp:
            assert (disp.n_cols, disp.n_rows) == (2, 2)
            assert len(disp._frames) == 4
            cells = {
                (
                    int(round((f.top - Title_height) / (frame_size[1] + margin))),
                    int(round(f.left / (frame_size[0] + margin))),
                )
                for f in disp._frames.values()
            }
            assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}


class TestAutoDisplayGridOverride:
    def test_grid_override_pins_layout(self):
        pytest.importorskip("pygame")
        disp = AutoDisplay(
            [FrameInfo("a", "A", "Image2DFrame", {})],
            frame_size=(100, 100),
            display_size=(400, 100),
            grid=(1, 1),
        )
        disp.init_window()
        assert (disp.n_cols, disp.n_rows) == (1, 1)
        disp.close()

    def test_grid_too_small_raises(self):
        pytest.importorskip("pygame")
        disp = AutoDisplay(
            [FrameInfo(n, n, "Image2DFrame", {}) for n in ("a", "b", "c", "d", "e")],
            frame_size=(50, 50),
            display_size=(200, 200),
            grid=(2, 2),
        )
        with pytest.raises(ValueError, match="cannot hold"):
            disp.init_window()


def test_frame_classes_used_by_display_exist():
    from ao_shaping.display import frames as frames_module

    for name in (
        "Image2DFrame",
        "Image2DWithBucketFrame",
        "VoltageFrame",
        "TextFrame",
        "EpochCurveFrame",
    ):
        assert hasattr(frames_module, name)


class TestEpochCurveFrame:
    @staticmethod
    def _frame(pygame, size=(100, 100)):
        window = pygame.display.set_mode((size[0] + 20, size[1] + 20))
        return EpochCurveFrame(
            window=window, render_pos=(0, 0), frame_size=size, title="curve"
        )

    def test_x_offset_is_fixed_by_total_epochs(self):
        pygame = pytest.importorskip("pygame")
        pygame.init()
        frame = self._frame(pygame)
        x0 = frame.left
        x1 = frame.left + frame.width - 1

        assert frame._x_of_epoch(0, 100, 10) == x0
        assert frame._x_of_epoch(100, 100, 10) == x1
        assert abs(frame._x_of_epoch(50, 100, 10) - (x0 + (x1 - x0) // 2)) <= 1
        pygame.quit()

    def test_x_offset_falls_back_to_even_spacing(self):
        pygame = pytest.importorskip("pygame")
        pygame.init()
        frame = self._frame(pygame)
        x0 = frame.left
        x1 = frame.left + frame.width - 1

        assert frame._x_of_epoch(0, None, 5) == x0
        assert frame._x_of_epoch(4, None, 5) == x1
        pygame.quit()

    def test_records_rerenders_and_resets(self):
        pygame = pytest.importorskip("pygame")
        pygame.init()
        frame = self._frame(pygame)

        frame.render(value=0.2, epoch=1, total_epochs=10, label="pib 0.20")
        frame.render(value=0.5, epoch=2, total_epochs=10, label="pib 0.50")
        assert frame.data == {"epoch": [1.0, 2.0], "value": [0.2, 0.5]}

        # Same epoch replaces the point instead of duplicating it.
        frame.render(value=0.6, epoch=2, total_epochs=10)
        assert frame.data == {"epoch": [1.0, 2.0], "value": [0.2, 0.6]}

        # Redraw without a value keeps history.
        frame.render()
        assert frame.data == {"epoch": [1.0, 2.0], "value": [0.2, 0.6]}

        frame.close()
        assert frame.data == {"epoch": [], "value": []}
        pygame.quit()
