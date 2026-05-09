import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Test parameters from task specification
N_WFS_TERMS = 66
N_SLM_TERMS = 55


def _pygame_font_available():
    """Check if pygame.font submodule is available (not present in headless installs)."""
    try:
        import pygame
        pygame.font.SysFont
        return True
    except (NotImplementedError, AttributeError, ModuleNotFoundError):
        return False


skip_no_pygame_font = pytest.mark.skipif(
    not _pygame_font_available(),
    reason="pygame.font not available in headless environment"
)


class TestZernikeCalibrationDisplayImport:
    def test_import(self):
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        assert ZernikeCalibrationDisplay is not None


class TestZernikeCalibrationDisplayInstantiation:
    def test_instantiation(self):
        from ao_shaping.utils.display import ZernikeCalibrationDisplay

        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )

        assert display.n_wfs_terms == N_WFS_TERMS
        assert display.n_slm_terms == N_SLM_TERMS


class TestZernikeCalibrationDisplayInitWindow:
    @skip_no_pygame_font
    def test_init_window_skip_without_pygame(self):
        pygame = pytest.importorskip("pygame")

        from ao_shaping.utils.display import ZernikeCalibrationDisplay

        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )

        with patch.object(pygame, 'init') as mock_init, \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont:

            mock_display.set_mode.return_value = MagicMock()
            mock_sysfont.return_value = MagicMock()

            display.init_window()

            mock_init.assert_called_once()


class TestZernikeCalibrationDisplayUpdate:
    @skip_no_pygame_font
    def test_update_renders(self):
        pygame = pytest.importorskip("pygame")

        from ao_shaping.utils.display import ZernikeCalibrationDisplay

        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )

        np.random.seed(42)
        response_col = np.random.rand(N_WFS_TERMS)
        variance_col = np.random.rand(N_WFS_TERMS) * 0.01

        mode_index = 0
        mode_name = "Zernike(4,2)"
        current_cycle = 0
        total_cycles = 1
        mean_variance = np.mean(variance_col)

        with patch.object(pygame, 'init'), \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont, \
             patch.object(pygame.event, 'get') as mock_event_get, \
             patch.object(pygame.time, 'Clock') as mock_clock, \
             patch.object(pygame, 'draw') as mock_draw:

            mock_window = MagicMock()
            mock_display.set_mode.return_value = mock_window

            mock_font = MagicMock()
            mock_title_font = MagicMock()
            mock_sysfont.side_effect = [mock_title_font, mock_font]

            mock_event_get.return_value = []
            mock_clock.return_value = MagicMock()

            mock_draw.rect.return_value = MagicMock()

            display.init_window()

            mock_window.fill = MagicMock()
            mock_window.blit = MagicMock()

            result = display.update(
                mode_index=mode_index,
                mode_name=mode_name,
                response_col=response_col,
                variance_col=variance_col,
                current_cycle=current_cycle,
                total_cycles=total_cycles,
                mean_variance=mean_variance
            )

            assert result is True


class TestZernikeCalibrationDisplayClose:
    def test_close(self):
        pygame = pytest.importorskip("pygame")

        from ao_shaping.utils.display import ZernikeCalibrationDisplay

        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )

        with patch.object(pygame, 'quit') as mock_quit:
            display.close()
            mock_quit.assert_called_once()


class TestZernikeCalibrationDisplayContextManager:
    @skip_no_pygame_font
    def test_context_manager(self):
        pygame = pytest.importorskip("pygame")

        from ao_shaping.utils.display import ZernikeCalibrationDisplay

        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )

        with patch.object(pygame, 'init'), \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont:

            mock_display.set_mode.return_value = MagicMock()
            mock_sysfont.return_value = MagicMock()

            with display as disp:
                assert disp is not None
                assert disp.n_wfs_terms == N_WFS_TERMS
                assert disp.n_slm_terms == N_SLM_TERMS
