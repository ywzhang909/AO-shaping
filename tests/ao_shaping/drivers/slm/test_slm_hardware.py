"""Hardware tests for Santec SLM-200 initialization.

Tests the ``open()`` initialization flow with a physically connected Santec SLM-200.
Config files are isolated per test via ``monkeypatch`` on ``_SLM_CONFIG_DIR``
pointing to a per-test ``tmp_path``, so tests never pollute the production
config directory and never interfere with each other.

Run with:
    pytest tests/ao_shaping/drivers/slm/test_slm_hardware.py -v --hardware
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware

import ao_shaping.drivers.slm.santec_slm200 as slm_module
from ao_shaping.drivers.slm.santec_slm200 import (
    SantecSLM200,
    VideoMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slm_config_dir(monkeypatch, tmp_path):
    """Redirect ``_SLM_CONFIG_DIR`` to a per-test temp directory.

    Every test gets a clean config directory, so config-file state is never
    carried over between tests.
    """
    config_dir = tmp_path / "slm_configs"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(slm_module, "_SLM_CONFIG_DIR", config_dir)
    return config_dir


# ---------------------------------------------------------------------------
# Scenario 1 — Default initialization (no config file, all defaults)
# ---------------------------------------------------------------------------


class TestDefaultInit:
    """Open/close with all default parameters and no pre-existing config."""

    def test_open_close_default(self, slm_config_dir):
        """All defaults: wavelength read from device; shifts = 0; 120Hz off."""
        with SantecSLM200() as slm:
            assert slm.is_open
            assert slm.slm_number == 1
            assert slm.wavelength is not None, (
                "wavelength should be populated from device when init wl=None"
            )
            assert isinstance(slm.wavelength, int)
            assert not slm._use_120hz
            assert slm.flags == 0
            assert slm._shift_x == 0
            assert slm._shift_y == 0
        assert not slm.is_open

    def test_wavelength_matches_device_reading(self, slm_config_dir):
        """Default-init wavelength equals the device's current wavelength."""
        with SantecSLM200() as slm:
            device_wl, device_mg = slm.get_wavelength_info()
            assert slm.wavelength == device_wl, (
                f"open() wl ({slm.wavelength}) != device wl ({device_wl})"
            )

    def test_config_saved_on_close(self, slm_config_dir):
        """Default parameters are persisted to a JSON config file on close."""
        with SantecSLM200() as slm:
            serial = slm._serial_number
            assert serial is not None, "serial number should be readable"

        config_file = slm_config_dir / "slm" / f"{serial}.json"
        assert config_file.exists(), f"config file missing: {config_file}"

        with open(config_file) as f:
            config = json.load(f)

        assert config["serial_number"] == serial
        assert isinstance(config["wavelength"], int)
        assert config["use_120hz"] is False
        assert config["shift_x"] == 0
        assert config["shift_y"] == 0


# ---------------------------------------------------------------------------
# Scenario 2 — Explicit __init__ parameters (no config file)
# ---------------------------------------------------------------------------


class TestExplicitInit:
    """SLM opened with explicit init parameters, no pre-existing config."""

    def test_all_params(self, slm_config_dir):
        """All init parameters provided explicitly."""
        with SantecSLM200(
            slm_number=1,
            wavelength=1064,
            use_120hz=True,
            shift_x=100,
            shift_y=-50,
            video_mode=VideoMode.Memory,
        ) as slm:
            assert slm.wavelength == 1064
            assert slm._use_120hz is True
            assert slm.flags == 1
            assert slm._shift_x == 100
            assert slm._shift_y == -50
            assert slm.video_mode == 0

    def test_wavelength_only(self, slm_config_dir):
        """Only wavelength provided; shifts and 120Hz get defaults."""
        with SantecSLM200(wavelength=1064) as slm:
            assert slm.wavelength == 1064
            assert slm._use_120hz is False
            assert slm.flags == 0
            assert slm._shift_x == 0
            assert slm._shift_y == 0

    def test_shifts_only(self, slm_config_dir):
        """Only shift parameters provided; wavelength read from device."""
        with SantecSLM200(shift_x=50, shift_y=-30) as slm:
            assert slm._shift_x == 50
            assert slm._shift_y == -30
            assert slm.wavelength is not None  # read from device

    def test_120hz_true(self, slm_config_dir):
        """``use_120hz=True`` sets ``flags`` to ``FLAGS_RATE120`` (1)."""
        with SantecSLM200(use_120hz=True) as slm:
            assert slm._use_120hz is True
            assert slm.flags == 1

    def test_120hz_false(self, slm_config_dir):
        """``use_120hz=False`` keeps ``flags`` at 0."""
        with SantecSLM200(use_120hz=False) as slm:
            assert slm._use_120hz is False
            assert slm.flags == 0

    def test_video_mode_memory(self, slm_config_dir):
        """``VideoMode.Memory`` is stored as integer 0."""
        with SantecSLM200(video_mode=VideoMode.Memory) as slm:
            assert slm.video_mode == 0

    def test_video_mode_dvi(self, slm_config_dir):
        """``VideoMode.DVI`` is stored as integer 1."""
        with SantecSLM200(video_mode=VideoMode.DVI) as slm:
            assert slm.video_mode == 1

    def test_video_mode_raw_int(self, slm_config_dir):
        """Raw integers for video_mode are accepted."""
        with SantecSLM200(video_mode=0) as slm:
            assert slm.video_mode == 0
        with SantecSLM200(video_mode=1) as slm:
            assert slm.video_mode == 1


# ---------------------------------------------------------------------------
# Scenario 3 — Config file exists (config values take priority)
# ---------------------------------------------------------------------------


class TestConfigPriority:
    """When a config file exists for the device serial, its values override
    ``__init__`` parameters."""

    @pytest.fixture
    def saved_serial(self, slm_config_dir):
        """Open then close the SLM with known parameters to create a config file.

        Returns the device serial number so tests can verify the config path.
        The config on disk contains: wavelength=1064, use_120hz=True,
        shift_x=50, shift_y=30.
        """
        with SantecSLM200(
            wavelength=1064,
            use_120hz=True,
            shift_x=50,
            shift_y=30,
        ) as slm:
            serial = slm._serial_number
        return serial

    def test_config_overrides_init_params(self, slm_config_dir, saved_serial):
        """Config file values are used even when ``__init__`` passes different values."""
        with SantecSLM200(
            wavelength=532,   # should be ignored — config exists
            shift_x=999,      # should be ignored
        ) as slm:
            assert slm._serial_number == saved_serial
            assert slm.wavelength == 1064, "config value should win"
            assert slm._use_120hz is True, "config value should win"
            assert slm.flags == 1
            assert slm._shift_x == 50, "config value should win"
            assert slm._shift_y == 30, "config value should win"

    def test_config_partial_missing_keys(self, slm_config_dir, saved_serial):
        """A config missing some keys fills defaults from ``_PARAM_SPEC``."""
        config_file = slm_config_dir / "slm" / f"{saved_serial}.json"
        with open(config_file, "w") as f:
            json.dump({"wavelength": 532}, f)

        with SantecSLM200() as slm:
            assert slm.wavelength == 532
            # Missing keys get _PARAM_SPEC defaults
            assert slm._shift_x == 0, "missing key should default to 0"
            assert slm._shift_y == 0, "missing key should default to 0"
            assert slm._use_120hz is False, "missing key should default to False"

    def test_reopen_loads_saved_config(self, slm_config_dir):
        """After save-on-close, a fresh open loads the saved values."""
        with SantecSLM200(wavelength=800, shift_x=42) as slm:
            serial = slm._serial_number

        config_file = slm_config_dir / "slm" / f"{serial}.json"
        assert config_file.exists()

        # Reopen with different params — saved config should win
        with SantecSLM200(wavelength=633) as slm:
            assert slm.wavelength == 800, "saved config should override init"
            assert slm._shift_x == 42

    def test_manual_config_edit(self, slm_config_dir, saved_serial):
        """Manually editing the config file is reflected on the next open."""
        config_file = slm_config_dir / "slm" / f"{saved_serial}.json"
        with open(config_file) as f:
            config = json.load(f)
        config["wavelength"] = 780
        config["shift_x"] = 99
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        with SantecSLM200() as slm:
            assert slm.wavelength == 780
            assert slm._shift_x == 99
            assert slm._shift_y == 30  # unchanged from original config


# ---------------------------------------------------------------------------
# Scenario 4 — Wavelength behavior during open()
# ---------------------------------------------------------------------------


class TestWavelengthOnOpen:
    """How ``open()`` resolves the wavelength:

    * If init wavelength is ``None`` → read from device.
    * If init wavelength matches device value → skip ``set_wavelength``.
    * If init wavelength differs → call ``set_wavelength(…, save_to_device=True)``.
    """

    def test_wavelength_none(self, slm_config_dir):
        """``wavelength=None`` reads from device via ``get_wavelength_info()``."""
        with SantecSLM200(wavelength=None) as slm:
            device_wl, _ = slm.get_wavelength_info()
            assert slm.wavelength == device_wl

    def test_explicit_wavelength_set(self, slm_config_dir):
        """Opening with an explicit wavelength sets it correctly on the device."""
        with SantecSLM200(wavelength=1064) as slm:
            current_wl, _ = slm.get_wavelength_info()
            assert slm.wavelength == current_wl


# ---------------------------------------------------------------------------
# Scenario 5 — Config persistence lifecycle
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    """Config file save/load/verify across open/close cycles."""

    def test_config_file_contents(self, slm_config_dir):
        """All relevant fields are saved to the config JSON."""
        with SantecSLM200(
            wavelength=1064,
            use_120hz=True,
            shift_x=10,
            shift_y=-10,
        ) as slm:
            serial = slm._serial_number

        config_file = slm_config_dir / "slm" / f"{serial}.json"
        assert config_file.exists()
        with open(config_file) as f:
            config = json.load(f)

        assert config["serial_number"] == serial
        assert config["wavelength"] == 1064
        assert config["use_120hz"] is True
        assert config["shift_x"] == 10
        assert config["shift_y"] == -10
        assert isinstance(config["max_gray"], int)
        assert "video_mode" in config

    def test_config_not_saved_without_serial(self, slm_config_dir, monkeypatch):
        """When serial number cannot be read, config save is skipped (no crash)."""
        with SantecSLM200() as slm:
            monkeypatch.setattr(slm, "_serial_number", None)
            slm.save_config()  # should log warning, no error

    def test_open_close_once(self, slm_config_dir):
        """A single open/close cycle works correctly."""
        with SantecSLM200() as slm:
            assert slm.is_open
            slm.get_wavelength_info()

    def test_config_dir_starts_clean(self, slm_config_dir):
        """Each test's config dir is initially empty."""
        config_files = list(slm_config_dir.rglob("*.json"))
        assert config_files == [], (
            f"expected empty config dir, found: {config_files}"
        )


# ---------------------------------------------------------------------------
# Scenario 6 — Serial number
# ---------------------------------------------------------------------------


class TestSerialNumber:
    """SLM serial-number reading and consistency."""

    def test_get_serial_number_returns_string(self, slm_config_dir):
        """``get_serial_number()`` returns a non-empty string."""
        with SantecSLM200() as slm:
            serial = slm.get_serial_number()
            assert serial is not None
            assert isinstance(serial, str)
            assert len(serial) > 0

    def test_serial_consistent_across_opens(self, slm_config_dir):
        """Serial number is the same when read twice in a single session."""
        with SantecSLM200() as slm:
            serial_a = slm._serial_number
            serial_b = slm.get_serial_number()
            assert serial_a == serial_b
            assert len({serial_a, serial_b}) == 1

    def test_serial_double_read(self, slm_config_dir):
        """Two reads within the same session return the same serial."""
        with SantecSLM200() as slm:
            s1 = slm.get_serial_number()
            s2 = slm.get_serial_number()
            assert s1 == s2

    def test_serial_used_as_config_filename(self, slm_config_dir):
        """Config file is named ``<serial>.json``."""
        with SantecSLM200() as slm:
            serial = slm._serial_number
        config_file = slm_config_dir / "slm" / f"{serial}.json"
        assert config_file.exists()
        assert config_file.stem == serial


# ---------------------------------------------------------------------------
# Scenario 7 — Open/close lifecycle edge cases
# ---------------------------------------------------------------------------


class TestOpenCloseLifecycle:
    """Robustness of open/close against unusual calling patterns."""

    def test_double_open(self, slm_config_dir):
        """Opening an already-open SLM is safe (logs warning, returns)."""
        slm = SantecSLM200()
        slm.open()
        assert slm.is_open
        slm.open()  # should log "already open" and return
        assert slm.is_open
        slm.close()

    def test_close_not_open(self, slm_config_dir):
        """Closing an SLM that was never opened is a no-op."""
        slm = SantecSLM200()
        slm.close()  # must not raise
        assert not slm.is_open

    def test_double_close(self, slm_config_dir):
        """Closing an already-closed SLM is a no-op."""
        slm = SantecSLM200()
        slm.open()
        slm.close()
        slm.close()  # must not raise
        assert not slm.is_open

    def test_context_manager(self, slm_config_dir):
        """Context manager opens and closes cleanly."""
        with SantecSLM200() as slm:
            assert slm.is_open
        assert not slm.is_open

    def test_config_saved_on_context_exit(self, slm_config_dir):
        """Config is saved when the context manager's ``__exit__`` calls ``close()``."""
        with SantecSLM200(wavelength=532) as slm:
            serial = slm._serial_number
        config_file = slm_config_dir / "slm" / f"{serial}.json"
        assert config_file.exists()


# ---------------------------------------------------------------------------
# Scenario 8 — Runtime shift setting
# ---------------------------------------------------------------------------


class TestShiftRuntime:
    """``set_shift()`` modifies the device shift after ``open()``."""

    def test_set_shift_positive(self, slm_config_dir):
        """Positive shifts are stored correctly."""
        with SantecSLM200() as slm:
            slm.set_shift(shift_x=50, shift_y=-30)
            assert slm.shift_x == 50
            assert slm.shift_y == -30

    def test_set_shift_zero(self, slm_config_dir):
        """``set_shift(0, 0)`` works."""
        with SantecSLM200() as slm:
            slm.set_shift(0, 0)
            assert slm.shift_x == 0
            assert slm.shift_y == 0

    def test_shift_saved_in_config(self, slm_config_dir):
        """Shift values set via ``set_shift()`` are persisted on close."""
        with SantecSLM200() as slm:
            slm.set_shift(shift_x=20, shift_y=-10)
            serial = slm._serial_number

        config_file = slm_config_dir / "slm" / f"{serial}.json"
        with open(config_file) as f:
            config = json.load(f)
        assert config["shift_x"] == 20
        assert config["shift_y"] == -10

    def test_shift_properties_readonly(self):
        """``shift_x`` / ``shift_y`` are read-only properties (no public setter)."""
        slm = SantecSLM200()
        assert isinstance(type(slm).shift_x, property)
        assert isinstance(type(slm).shift_y, property)


# ---------------------------------------------------------------------------
# Scenario 9 — Init-time validation (no hardware needed)
# ---------------------------------------------------------------------------


class TestInitValidation:
    """Constructor validation that does not require an open device."""

    def test_default_slm_number(self):
        """Default SLM number is 1."""
        slm = SantecSLM200()
        assert slm.slm_number == 1

    def test_custom_slm_number(self):
        """Non-default SLM number is stored."""
        slm = SantecSLM200(slm_number=2)
        assert slm.slm_number == 2

    def test_default_wavelength_is_none(self):
        """Default ``__init__`` wavelength is ``None`` before ``open()``."""
        slm = SantecSLM200()
        assert slm.wavelength is None

    def test_repr_closed(self):
        """``repr()`` shows closed state before open."""
        slm = SantecSLM200()
        r = repr(slm)
        assert "未连接" in r

    def test_repr_open(self, slm_config_dir):
        """``repr()`` shows connected state after open."""
        with SantecSLM200() as slm:
            r = repr(slm)
            assert "已连接" in r

    def test_repr_after_close(self, slm_config_dir):
        """``repr()`` shows closed state after close()."""
        slm = SantecSLM200()
        slm.open()
        slm.close()
        r = repr(slm)
        assert "未连接" in r
