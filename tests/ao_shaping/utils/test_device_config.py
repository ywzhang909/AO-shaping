"""Tests for the generic device configuration module (device_config.py).

Tests the ConfigHandler, DeviceParam, and param() utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.utils.device_config import ConfigHandler, DeviceParam, param


# ── Test fixtures ────────────────────────────────────────


@dataclass
class DummyParams(DeviceParam):
    """Simple params for testing."""
    name: str = param(default="default_name")
    count: int = param(default=42, cast=int)
    enabled: bool = param(default=False, cast=bool, attr="_is_enabled")
    value: float | None = param(default=None, cast=float)


class DummyDevice:
    """Minimal device simulation."""
    def __init__(self):
        self.name: str = ""
        self.count: int = 0
        self._is_enabled: bool = False
        self.value: float | None = None


# ── Tests ────────────────────────────────────────────────


class TestDeviceParam:
    """DeviceParam and param() marker basics."""

    def test_field_meta_defaults(self):
        """Verify field metadata defaults match field name."""
        assert DummyParams._field_meta(DummyParams.__dataclass_fields__["name"]) == {
            "config_key": "name",
            "cast": None,
            "attr": "name",
        }

    def test_field_meta_custom_attr(self):
        """Verify custom attr is correctly stored."""
        meta = DummyParams._field_meta(DummyParams.__dataclass_fields__["enabled"])
        assert meta["attr"] == "_is_enabled"

    def test_field_meta_custom_cast(self):
        """Verify custom cast is correctly stored."""
        meta = DummyParams._field_meta(DummyParams.__dataclass_fields__["count"])
        assert meta["cast"] is int
        assert meta["config_key"] == "count"


class TestConfigHandlerResolve:
    """ConfigHandler.resolve() and resolve_from_config()."""

    def setup_method(self):
        self.handler: ConfigHandler[DummyParams] = ConfigHandler(
            Path("/tmp/_test_configs"), "dummy", DummyParams
        )

    def test_defaults_empty_config(self):
        """With empty config and no init_values, use dataclass defaults."""
        params = self.handler.resolve_from_config({})
        assert params.name == "default_name"
        assert params.count == 42
        assert params.enabled is False
        assert params.value is None

    def test_config_overrides_defaults(self):
        """Config values override dataclass defaults."""
        params = self.handler.resolve_from_config(
            {"name": "from_config", "count": "99", "enabled": "true"}
        )
        assert params.name == "from_config"
        assert params.count == 99  # cast=int applied
        assert params.enabled is True  # cast=bool applied ("true" → True)

    def test_init_overrides_config(self):
        """Init_values have highest priority."""
        params = self.handler.resolve_from_config(
            {"name": "config_name", "count": 1},
            init_values={"name": "init_name", "count": 999},
        )
        assert params.name == "init_name"  # init > config
        assert params.count == 999

    def test_init_none_does_not_override(self):
        """None init_values should not override non-None config."""
        params = self.handler.resolve_from_config(
            {"count": 50},
            init_values={"count": None},
        )
        assert params.count == 50

    def test_missing_key_falls_to_default(self):
        """Missing config key uses default even if others present."""
        params = self.handler.resolve_from_config({"name": "test"})
        assert params.name == "test"
        assert params.count == 42  # default

    def test_bad_cast_returns_default(self):
        """When cast fails, field falls back to default."""
        params = self.handler.resolve_from_config(
            {"count": "not_a_number"}
        )
        # cast=int("not_a_number") raises ValueError → fallback to default
        assert params.count == 42


class TestConfigHandlerApply:
    """ConfigHandler.apply() and apply_from_config()."""

    def setup_method(self):
        self.handler: ConfigHandler[DummyParams] = ConfigHandler(
            Path("/tmp/_test_configs"), "dummy", DummyParams
        )
        self.device = DummyDevice()

    def test_apply_sets_attributes(self):
        """apply_from_config sets instance attributes correctly."""
        self.handler.apply_from_config(
            self.device,
            {"name": "test_dev", "count": 7, "enabled": True, "value": 3.14},
        )
        assert self.device.name == "test_dev"
        assert self.device.count == 7
        assert self.device._is_enabled is True  # custom attr
        assert self.device.value == 3.14

    def test_apply_custom_attr_mapping(self):
        """Fields with custom attr are set on the correct attribute."""
        self.handler.apply_from_config(
            self.device,
            {"enabled": True},
        )
        assert self.device._is_enabled is True

    def test_apply_returns_params(self):
        """apply_from_config returns the params object."""
        params = self.handler.apply_from_config(
            self.device,
            {"name": "return_test"},
            init_values={"count": 100},
        )
        assert isinstance(params, DummyParams)
        assert params.name == "return_test"
        assert params.count == 100


class TestConfigHandlerCollect:
    """ConfigHandler.collect() builds config dict from instance attrs."""

    def setup_method(self):
        self.handler: ConfigHandler[DummyParams] = ConfigHandler(
            Path("/tmp/_test_configs"), "dummy", DummyParams
        )
        self.device = DummyDevice()

    def test_collect_defaults(self):
        """Collect reads instance attrs back."""
        self.handler.apply_from_config(
            self.device,
            {"name": "dev1", "count": 55, "enabled": True, "value": 1.5},
        )
        collected = self.handler.collect(self.device)
        assert collected == {
            "name": "dev1",
            "count": 55,
            "enabled": self.device._is_enabled,  # collected from _is_enabled attr
            "value": 1.5,
        }

    def test_collect_custom_attr_key(self):
        """Custom attr maps to config_key correctly in collect."""
        self.device._is_enabled = True
        collected = self.handler.collect(self.device)
        assert collected["enabled"] is True  # config_key="enabled", attr="_is_enabled"
