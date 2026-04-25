import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Test parameters from task specification
N_WFS_TERMS = 66
N_SLM_TERMS = 55


class TestZernikeCalibrationDisplayImport:
    def test_import(self):
        """Verify ZernikeCalibrationDisplay can be imported."""
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        assert ZernikeCalibrationDisplay is not None


class TestZernikeCalibrationDisplayInstantiation:
    def test_instantiation(self):
        """Verify class can be instantiated with n_wfs_terms=66, n_slm_terms=55."""
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        
        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )
        
        assert display.n_wfs_terms == N_WFS_TERMS
        assert display.n_slm_terms == N_SLM_TERMS


class TestZernikeCalibrationDisplayInitWindow:
    def test_init_window_skip_without_pygame(self):
        """Use pytest.importorskip('pygame') - skip if pygame not available."""
        pygame = pytest.importorskip("pygame")
        
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        
        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )
        
        # Mock pygame.init and pygame.display.set_mode to avoid actually opening window
        with patch.object(pygame, 'init') as mock_init, \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont:
            
            mock_display.set_mode.return_value = MagicMock()
            mock_sysfont.return_value = MagicMock()
            
            display.init_window()
            
            mock_init.assert_called_once()


class TestZernikeCalibrationDisplayUpdate:
    def test_update_renders(self):
        """With mock data (np.random.rand(66), np.random.rand(66)*0.01)."""
        pygame = pytest.importorskip("pygame")
        
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        
        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )
        
        # Generate mock data as specified
        np.random.seed(42)
        response_col = np.random.rand(N_WFS_TERMS)
        variance_col = np.random.rand(N_WFS_TERMS) * 0.01
        
        mode_index = 0
        mode_name = "Zernike(4,2)"
        current_cycle = 0
        total_cycles = 1
        mean_variance = np.mean(variance_col)
        
        # Mock all pygame functionality
        with patch.object(pygame, 'init'), \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont, \
             patch.object(pygame.event, 'get') as mock_event_get, \
             patch.object(pygame.time, 'Clock') as mock_clock, \
             patch.object(pygame, 'draw') as mock_draw:
            
            # Setup mocks
            mock_window = MagicMock()
            mock_display.set_mode.return_value = mock_window
            
            mock_font = MagicMock()
            mock_title_font = MagicMock()
            mock_sysfont.side_effect = [mock_title_font, mock_font]
            
            mock_event_get.return_value = []
            mock_clock.return_value = MagicMock()
            
            # Make pygame.draw.rect work with mock window
            mock_draw.rect.return_value = MagicMock()
            
            display.init_window()
            
            # Mock window fill and other drawing methods
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
            
            # Verify update returns True (continue)
            assert result is True


class TestZernikeCalibrationDisplayClose:
    def test_close(self):
        """Verify pygame.quit is called."""
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
    def test_context_manager(self):
        """Use 'with ZernikeCalibrationDisplay(...) as display:'."""
        pygame = pytest.importorskip("pygame")
        
        from ao_shaping.utils.display import ZernikeCalibrationDisplay
        
        display = ZernikeCalibrationDisplay(
            n_wfs_terms=N_WFS_TERMS,
            n_slm_terms=N_SLM_TERMS
        )
        
        # Mock pygame.init to avoid actual initialization
        with patch.object(pygame, 'init'), \
             patch.object(pygame, 'display') as mock_display, \
             patch.object(pygame.font, 'SysFont') as mock_sysfont:
            
            mock_display.set_mode.return_value = MagicMock()
            mock_sysfont.return_value = MagicMock()
            
            with display as disp:
                assert disp is not None
                assert disp.n_wfs_terms == N_WFS_TERMS
                assert disp.n_slm_terms == N_SLM_TERMS