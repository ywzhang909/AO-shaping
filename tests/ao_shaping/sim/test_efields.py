#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for efields.py module
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import the module to test
from src.ao_shaping.sim.efields import (
    OptConfig,
    GlDim,
    FDTDIntervalConfig,
    FDTDInterval,
    OpticParameters,
    ElectronicField,
    phase_to_electronic_field,
    expande_electronic_field,
    interp_phase,
    trans_beam,
    calculate_efficiency,
    perform_fft,
    rescale_EF,
)


class TestOptConfig:
    """Test OptConfig class"""

    def test_optconfig_initialization(self):
        """Test OptConfig initialization"""
        config = OptConfig()
        assert hasattr(config, 'pol')
        assert hasattr(config, 'shift')
        assert config.pol == (1, 1)
        assert config.shift == (0, 0)


class TestGlDim:
    """Test GlDim class"""

    def test_gldim_initialization(self):
        """Test GlDim initialization"""
        dim = GlDim()
        assert hasattr(dim, 'x')
        assert hasattr(dim, 'y')
        assert dim.x == 1
        assert dim.y == 1


class TestFDTDIntervalConfig:
    """Test FDTDIntervalConfig class"""

    def test_fdtdintervalconfig_initialization(self):
        """Test FDTDIntervalConfig initialization"""
        config = FDTDIntervalConfig(x=10, y=20, z=30)
        assert config.x == 10
        assert config.y == 20
        assert config.z == 30

    def test_fdtdintervalconfig_post_init(self):
        """Test FDTDIntervalConfig post_init validation"""
        # Test with valid values
        config = FDTDIntervalConfig(x=1, y=1, z=1)
        assert config.x == 1
        assert config.y == 1
        assert config.z == 1

        # Test with values less than 1
        config = FDTDIntervalConfig(x=0, y=0, z=0)
        assert config.x == 1
        assert config.y == 1
        assert config.z == 1


class TestOpticParameters:
    """Test OpticParameters class"""

    def test_opticparameters_initialization(self):
        """Test OpticParameters initialization"""
        params = OpticParameters(
            _num_elements_x=10,
            _num_elements_y=20,
            period=1.5,
            wavelength=0.5,
            focal_length=5.0
        )
        assert params.num_elements_x == 10
        assert params.num_elements_y == 20
        assert params.period == 1.5
        assert params.wavelength == 0.5
        assert params.focal_length == 5.0

    def test_opticparameters_properties(self):
        """Test OpticParameters properties"""
        params = OpticParameters(
            _num_elements_x=10,
            _num_elements_y=20,
            period=1.5,
            wavelength=0.5,
            focal_length=5.0
        )
        assert params.k0 == 2.0
        assert params.Px == 15.0
        assert params.Py == 30.0
        assert params.R == 7.5  # Fixed: min(15, 30) / 2 = 7.5
        assert params.F == 0.3333333333333333  # Fixed: 5.0 / (2 * 7.5) = 0.333...
        assert params.NA == pytest.approx(0.8320502943378436)  # Fixed: np.sin(np.arctan(7.5 / 5.0))


class TestElectronicField:
    """Test ElectronicField class"""

    def test_electronicfield_initialization(self):
        """Test ElectronicField initialization"""
        Ex = np.array([[1, 2], [3, 4]])
        Ey = np.array([[5, 6], [7, 8]])
        Ez = np.array([[9, 10], [11, 12]])
        
        efield = ElectronicField(Ex, Ey, Ez)
        assert np.array_equal(efield.Ex, Ex)
        assert np.array_equal(efield.Ey, Ey)
        assert np.array_equal(efield.Ez, Ez)

    def test_electronicfield_G_property(self):
        """Test ElectronicField G property"""
        Ex = np.array([[1, 2], [3, 4]])
        Ey = np.array([[5, 6], [7, 8]])
        Ez = np.array([[9, 10], [11, 12]])
        
        efield = ElectronicField(Ex, Ey, Ez)
        G = efield.G
        
        expected_G = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        expected_G[expected_G < 0] = 0
        
        assert np.array_equal(G, expected_G)


class TestPhaseToElectronicField:
    """Test phase_to_electronic_field function"""

    def test_phase_to_electronic_field(self):
        """Test phase_to_electronic_field function"""
        phase = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]])
        x = 2.0
        y = 2.0
        
        efield = phase_to_electronic_field(phase, x, y)
        
        assert hasattr(efield, 'Ex')
        assert hasattr(efield, 'Ey')
        assert hasattr(efield, 'Ez')
        assert efield.Ex.shape == phase.shape
        assert efield.Ey.shape == phase.shape
        assert efield.Ez.shape == phase.shape


class TestExpandeElectronicField:
    """Test expande_electronic_field function"""

    def test_expande_electronic_field(self):
        """Test expande_electronic_field function"""
        init_phase = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]])
        Px = 2.0
        Py = 2.0
        fdtd_x = np.array([-1, 0, 1])
        fdtd_y = np.array([-1, 0, 1])
        
        efield = expande_electronic_field(init_phase, Px, Py, fdtd_x, fdtd_y)
        
        assert hasattr(efield, 'Ex')
        assert hasattr(efield, 'Ey')
        assert hasattr(efield, 'Ez')


class TestInterpolatePhase:
    """Test interp_phase function"""

    def test_interp_phase(self):
        """Test interp_phase function"""
        num_structure_x = 10
        num_structure_y = 10
        Px = 2.0
        Py = 2.0
        ws = [0.5, 1.0, 1.5]
        ps = [0, 90, 180]
        f = 5.0
        lambda_ = 0.5
        
        P_a, dist_phase = interp_phase(num_structure_x, num_structure_y, Px, Py, ws, ps, f, lambda_)
        
        assert P_a.shape == (num_structure_y, num_structure_x)
        assert dist_phase.shape == (num_structure_y, num_structure_x)




class TestTransBeam:
    """Test trans_beam function"""

    def test_trans_beam(self):
        """Test trans_beam function"""
        init_phase = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]])
        params = OpticParameters(
            _num_elements_x=10,
            _num_elements_y=20,
            period=1.5,
            wavelength=0.5,
            focal_length=5.0
        )
        mon_dist = 1.0
        
        # Mock the FDTDInterval to avoid AttributeError
        with patch('src.ao_shaping.sim.efields.FDTDInterval') as mock_fdtd_interval:
            mock_fdtd_interval.x = 1
            mock_fdtd_interval.y = 1
            
            with patch('src.ao_shaping.sim.efields.expande_electronic_field') as mock_expand, \
                 patch('src.ao_shaping.sim.efields.perform_fft') as mock_fft, \
                 patch('src.ao_shaping.sim.efields.calculate_efficiency') as mock_efficiency:
                
                # Setup mocks
                mock_expand.return_value = ElectronicField(
                    np.array([[1, 2], [3, 4]]),
                    np.array([[5, 6], [7, 8]]),
                    np.array([[9, 10], [11, 12]])
                )
                mock_fft.return_value = (
                    ElectronicField(
                        np.array([[1, 2], [3, 4]]),
                        np.array([[5, 6], [7, 8]]),
                        np.array([[9, 10], [11, 12]])
                    ),
                    np.array([[-1, 0, 1], [-1, 0, 1]]),
                    np.array([[-1, 0, 1], [-1, 0, 1]])
                )
                mock_efficiency.return_value = (0.8, (0.5, 0.5))
                
                Eo, efficiency = trans_beam(init_phase, params, mon_dist)
                
                # Verify function calls
                mock_expand.assert_called_once()
                mock_fft.assert_called_once()
                mock_efficiency.assert_called_once()
                
                assert isinstance(Eo, ElectronicField)
                assert efficiency == 0.8


class TestCalculateEfficiency:
    """Test calculate_efficiency function"""

    def test_calculate_efficiency(self):
        """Test calculate_efficiency function"""
        G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        x = np.array([-1, 0, 1])
        y = np.array([-1, 0, 1])
        R = 1.0
        
        efficiency, (fl, fr) = calculate_efficiency(G, x, y, R)
        
        assert isinstance(efficiency, float)
        assert isinstance(fl, (float, np.ndarray))
        assert isinstance(fr, (float, np.ndarray))


class TestPerformFFT:
    """Test perform_fft function"""

    def test_perform_fft(self):
        """Test perform_fft function"""
        efield = ElectronicField(
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        )
        x_FDTD = np.array([-1, 0, 1])
        y_FDTD = np.array([-1, 0, 1])
        k0 = 2.0
        focal_length = 5.0
        mon_dist = 1.0
        
        result = perform_fft(efield, x_FDTD, y_FDTD, k0, focal_length, mon_dist)
        
        assert len(result) == 3
        assert isinstance(result[0], ElectronicField)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)


class TestRescaleEF:
    """Test rescale_EF function"""

    def test_rescale_ef(self):
        """Test rescale_EF function"""
        efield = ElectronicField(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        )
        jmX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        jmY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        
        result = rescale_EF(efield, jmX, jmY)
        
        assert len(result) == 3
        assert isinstance(result[0], ElectronicField)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)