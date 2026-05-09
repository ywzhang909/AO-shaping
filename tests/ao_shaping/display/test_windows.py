"""Tests for ao_shaping.display.windows — display window classes.

Note: These tests require a graphical display (pygame). They are skipped in headless/CI environments.
"""
import pytest

pytestmark = pytest.mark.skip(reason="Requires graphical display (pygame)")