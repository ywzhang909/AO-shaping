import numpy as np
from ao_shaping.utils.zernike import (
    factorial_jit,
    zernike_radial,
    compute_zernike,
    ZernikeGenerator
)


class TestFactorialJit:
    """Test cases for factorial_jit function."""

    def test_factorial_zero(self):
        """Test factorial of 0."""
        assert factorial_jit(0) == 1

    def test_factorial_one(self):
        """Test factorial of 1."""
        assert factorial_jit(1) == 1

    def test_factorial_positive(self):
        """Test factorial of positive integers."""
        assert factorial_jit(5) == 120
        assert factorial_jit(3) == 6


class TestZernikeRadial:
    """Test cases for zernike_radial function."""

    def test_zernike_radial_basic(self):
        """Test basic functionality of zernike_radial."""
        # Create a simple radial grid
        rho = np.array([0.0, 0.5, 1.0])
        
        # Test with n=0, m=0 (piston term)
        result = zernike_radial(0, 0, rho)
        expected = np.ones_like(rho)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zernike_radial_with_different_nm(self):
        """Test zernike_radial with different n,m values."""
        rho = np.linspace(0, 1, 5)
        
        # Test with n=2, m=0
        result = zernike_radial(2, 0, rho)
        # For n=2, m=0: R_2^0(rho) = 2*rho^2 - 1
        expected = 2 * rho**2 - 1
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeZernike:
    """Test cases for compute_zernike function."""

    def test_compute_zernike_m_positive(self):
        """Test compute_zernike with positive m."""
        rho = np.array([0.5])
        theta = np.array([np.pi/4])  # 45 degrees
        
        # Test with n=1, m=1
        result = compute_zernike(1, 1, rho, theta)
        # For n=1, m=1: Z_1^1 = R_1^1 * cos(theta) = rho * cos(theta)
        expected = rho * np.cos(theta)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_zernike_m_negative(self):
        """Test compute_zernike with negative m."""
        rho = np.array([0.5])
        theta = np.array([np.pi/4])  # 45 degrees
        
        # Test with n=1, m=-1
        result = compute_zernike(1, -1, rho, theta)
        # For n=1, m=-1: Z_1^{-1} = R_1^1 * sin(theta) = rho * sin(theta)
        expected = rho * np.sin(theta)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_zernike_m_zero(self):
        """Test compute_zernike with m=0."""
        rho = np.array([0.5])
        theta = np.array([np.pi/4])  # 45 degrees
        
        # Test with n=2, m=0 (focus term)
        result = compute_zernike(2, 0, rho, theta)
        # For n=2, m=0: Z_2^0 = R_2^0 = 2*rho^2 - 1
        expected = 2 * rho**2 - 1
        np.testing.assert_array_almost_equal(result, expected)


class TestZernikeGenerator:
    """Test cases for ZernikeGenerator class."""

    def test_init_and_precompute(self):
        """Test initialization and precomputation of Zernike modes."""
        # Create test data
        indices = [(0, 0), (1, 1), (1, -1)]
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        mask = R <= 1  # Unit circle mask
        
        # Create generator
        generator = ZernikeGenerator(indices, R, Theta, mask)
        
        # Check attributes
        assert generator.zernike_indices == indices
        assert generator.num_zernike == len(indices)
        assert generator.zernike_modes.shape == (len(indices), 10, 10)
        assert generator.mask is mask

    def test_generate_wavefront(self):
        """Test wavefront generation."""
        # Create test data
        indices = [(0, 0)]  # Just piston term
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        mask = R <= 1  # Unit circle mask
        
        # Create generator
        generator = ZernikeGenerator(indices, R, Theta, mask)
        
        # Generate wavefront with coefficient 1.0
        coefficients = np.array([1.0])
        wavefront = generator.generate_wavefront(coefficients)
        
        # Check shape and values
        assert wavefront.shape == (5, 5)
        # Piston term should be constant (1.0) within mask
        masked_wavefront = np.where(mask, wavefront, 0)
        # Values within mask should be approximately 1.0
        np.testing.assert_array_almost_equal(
            masked_wavefront[mask], 
            np.ones(np.count_nonzero(mask))
        )

    def test_fit_wavefront(self):
        """Test fitting a wavefront to Zernike coefficients."""
        # Create test data
        indices = [(0, 0), (1, 1), (2, 0)]  # Piston, tilt, focus
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        mask = R <= 1  # Unit circle mask
        
        # Create generator
        generator = ZernikeGenerator(indices, R, Theta, mask)
        
        # Create a test wavefront with known coefficients
        true_coefficients = np.array([1.0, 0.5, -0.3])  # piston, tilt, focus
        true_wavefront = generator.generate_wavefront(true_coefficients)
        
        # Add some noise to make it more realistic
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 1e-10, true_wavefront.shape)
        noisy_wavefront = true_wavefront + noise
        
        # Fit the wavefront
        fitted_coefficients = generator.fit_wavefront(noisy_wavefront)
        
        # Check that the fitted coefficients are close to the true ones
        np.testing.assert_array_almost_equal(fitted_coefficients, true_coefficients, decimal=8)
        
    def test_fit_wavefront_simple_case(self):
        """Test fitting with a simple case: pure piston term."""
        # Create test data
        indices = [(0, 0)]  # Just piston
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        mask = R <= 1  # Unit circle mask
        
        # Create generator
        generator = ZernikeGenerator(indices, R, Theta, mask)
        
        # Create a test wavefront: constant value
        true_coefficients = np.array([2.5])
        true_wavefront = generator.generate_wavefront(true_coefficients)
        
        # Fit the wavefront
        fitted_coefficients = generator.fit_wavefront(true_wavefront)
        
        # Check that the fitted coefficients are close to the true ones
        np.testing.assert_array_almost_equal(fitted_coefficients, true_coefficients)