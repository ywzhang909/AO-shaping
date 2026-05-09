"""Tests for ao_shaping.display.frames — registry and registration logic."""

import pytest

from ao_shaping.display.frames import (
    BaseFrame,
    get_frame,
    get_frame_names,
    register_frame,
)


class TestFrameRegistry:
    def test_builtin_frames_registered(self):
        names = get_frame_names()
        assert "Image2D" in names
        assert "Image2DWithBucket" in names
        assert "Voltage" in names
        assert "Log" in names
        assert "Text" in names

    def test_get_frame_returns_class(self):
        cls = get_frame("Image2D")
        assert cls is not None
        assert issubclass(cls, BaseFrame)

    def test_get_frame_voltage(self):
        cls = get_frame("Voltage")
        assert issubclass(cls, BaseFrame)

    def test_get_frame_log(self):
        cls = get_frame("Log")
        assert issubclass(cls, BaseFrame)

    def test_get_frame_text(self):
        cls = get_frame("Text")
        assert issubclass(cls, BaseFrame)

    def test_get_frame_nonexistent_raises(self):
        with pytest.raises(KeyError):
            get_frame("NonExistentFrame")

    def test_register_duplicate_name_raises(self):
        with pytest.raises(AssertionError, match="already registered"):
            @register_frame("Image2D")
            class DuplicateFrame(BaseFrame):
                pass
