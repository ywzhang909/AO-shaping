"""Pytest configuration and shared fixtures for AO-Shaping tests."""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def dm_n_actuators():
    """Number of DM actuators."""
    return 64


@pytest.fixture
def init_voltages(dm_n_actuators):
    """Initial zero voltages for DM."""
    return [0.0] * dm_n_actuators


@pytest.fixture
def dm_unit_mask_all(dm_n_actuators):
    """DM unit mask with all actuators enabled except disabled ones."""
    mask = np.ones(dm_n_actuators, dtype=bool)
    mask[0] = False
    return mask


@pytest.fixture
def dm_unit_mask_inner():
    """DM unit mask with only inner actuators enabled."""
    mask = np.ones(64, dtype=bool)
    mask[0] = False
    mask[21:] = False
    return mask


@pytest.fixture
def dm_unit_mask_outer():
    """DM unit mask with only outer actuators enabled."""
    mask = np.ones(64, dtype=bool)
    mask[0] = False
    mask[:39] = False
    return mask


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(200, 200).astype(np.float32)


@pytest.fixture
def sample_wavefront():
    """Create a sample wavefront for testing."""
    return np.random.rand(64, 64).astype(np.float32)