"""Adam optimizer tests"""

import numpy as np
import pytest

from ao_shaping.algorithm.adam import Adam, SGD


class TestAdam:
    """Test Adam optimizer"""

    def test_adam_initialization(self):
        """Test Adam optimizer initialization"""
        optimizer = Adam(dim=10, lr=0.1)
        assert optimizer.dim == 10
        assert optimizer.lr == 0.1
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.99

    def test_adam_update(self):
        """Test Adam optimizer update"""
        optimizer = Adam(dim=5, lr=0.1)
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        update = optimizer.update(grad)

        assert update.shape == (5,)
        assert update.dtype == np.float32

    def test_adam_convergence(self):
        """Test Adam optimizer convergence on simple quadratic"""
        optimizer = Adam(dim=2, lr=1.0)  # Higher learning rate for convergence

        # Minimize f(x, y) = x^2 + y^2
        # Gradient: [2x, 2y]
        losses = []
        position = np.array([10.0, 10.0], dtype=np.float32)

        for _ in range(200):
            grad = 2 * position
            position = position - optimizer.update(grad)
            losses.append(np.sum(position**2))

        # Adam should converge, but the exact threshold depends on lr
        assert losses[-1] < losses[0]  # Should at least decrease
        assert losses[-1] < 100  # Should significantly reduce from initial 200


class TestSGD:
    """Test SGD optimizer"""

    def test_sgd_initialization(self):
        """Test SGD initialization"""
        optimizer = SGD(dim=10, lr=0.1)
        assert optimizer.dim == 10
        assert optimizer.lr == 0.1

    def test_sgd_update(self):
        """Test SGD update"""
        optimizer = SGD(dim=5, lr=0.1)
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        update = optimizer.update(grad)

        assert update.shape == (5,)
        assert np.allclose(update, 0.1 * grad)
