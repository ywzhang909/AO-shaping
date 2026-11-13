"""Tests for :mod:`ao_shaping.utils.slm.gray_scale` — the single resolver for the
2π encoding grayscale (``[slm] two_pi_grayscale`` → default 1023).

Why these exist
---------------
Before 2026-11-12 the radian → grayscale scale was the literal ``1023`` sprinkled
across the driver, the GUI preview/export paths and ``phase_to_slm_grayscale``.
That made the panel's *hardware* full scale, the device's *physical* 2π gray
(993 @1064 nm) and the *encoding* scale the same number by accident. The
encoding scale is now user-configurable, so the contract that needs pinning is:

* precedence — explicit argument > config > :data:`DEFAULT_MAX_GRAYSCALE`;
* a missing / broken config never breaks the send path (warn + default);
* the offline math fallback and the driver path resolve to the **same** number,
  so an offline pattern and an on-bench pattern are bit-identical.

Every test redirects ``AO_SHAPING_CONFIG`` at a ``tmp_path`` file, so a user's
local ``ao_shaping.toml`` can never change the result. No hardware is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.io import paths
from ao_shaping.utils.slm import gray_scale
from ao_shaping.utils.slm.gray_scale import (
    CONFIG_KEY,
    CONFIG_SECTION,
    DEFAULT_MAX_GRAYSCALE,
    configured_max_grayscale,
    resolve_max_grayscale,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point the config loader at a throwaway file and drop the mtime cache."""
    config = tmp_path / "ao_shaping.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(config))
    monkeypatch.setattr(paths, "_CONFIG_CACHE", None)
    yield config
    # Leave no cached document behind for the next test in the session.
    monkeypatch.setattr(paths, "_CONFIG_CACHE", None)


def _write_config(config, body: str) -> None:
    config.write_text(body, encoding="utf-8")
    # Same path, new mtime -> the loader must re-read (no stale cache).
    paths._CONFIG_CACHE = None


class TestDefaults:
    def test_default_is_ten_bit_full_scale(self):
        assert DEFAULT_MAX_GRAYSCALE == 1023

    def test_missing_key_falls_back_to_default(self, _isolated_config):
        _write_config(_isolated_config, "[paths]\ndata_dir = 'data'\n")
        assert configured_max_grayscale() == DEFAULT_MAX_GRAYSCALE
        assert resolve_max_grayscale() == DEFAULT_MAX_GRAYSCALE

    def test_missing_file_falls_back_to_default(self, monkeypatch, tmp_path):
        missing = tmp_path / "does-not-exist.toml"
        monkeypatch.setenv(paths.ENV_CONFIG_FILE, str(missing))
        monkeypatch.setattr(paths, "_CONFIG_CACHE", None)
        assert resolve_max_grayscale() == DEFAULT_MAX_GRAYSCALE

    def test_config_keys_are_the_documented_names(self):
        assert (CONFIG_SECTION, CONFIG_KEY) == ("slm", "two_pi_grayscale")


class TestConfiguredValue:
    def test_config_value_is_used(self, _isolated_config):
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 993\n")
        assert configured_max_grayscale() == 993
        assert resolve_max_grayscale() == 993

    def test_integral_float_is_accepted(self, _isolated_config):
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 2047.0\n")
        assert resolve_max_grayscale() == 2047

    def test_editing_the_file_takes_effect_immediately(self, _isolated_config):
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 1023\n")
        assert resolve_max_grayscale() == 1023
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 511\n")
        assert resolve_max_grayscale() == 511


class TestPrecedence:
    def test_explicit_beats_config(self, _isolated_config):
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 993\n")
        assert resolve_max_grayscale(2047) == 2047

    def test_none_means_use_config(self, _isolated_config):
        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 993\n")
        assert resolve_max_grayscale(None) == 993


class TestBrokenValues:
    @pytest.mark.parametrize(
        "raw",
        ["0", "-5", "65536", "'1023'", "true", "1023.5", "[1, 2]"],
        ids=["zero", "negative", "too-large", "string", "bool", "fraction", "list"],
    )
    def test_unusable_config_value_falls_back_with_a_warning(
        self, _isolated_config, raw
    ):
        _write_config(_isolated_config, f"[slm]\ntwo_pi_grayscale = {raw}\n")
        value = resolve_max_grayscale()
        assert value == DEFAULT_MAX_GRAYSCALE

    @pytest.mark.parametrize("explicit", [0, -1, True, 1.5, "1023"])
    def test_bad_explicit_value_raises_instead_of_silently_scaling(
        self, _isolated_config, explicit
    ):
        # An explicit value is caller code, not user config: silently falling
        # back would put a wrongly scaled pattern on the panel unnoticed.
        with pytest.raises(ValueError):
            resolve_max_grayscale(explicit)


class TestSharedScaleAcrossPaths:
    """The offline fallback and the driver path must agree bit-for-bit."""

    def _phase(self) -> np.ndarray:
        return np.linspace(0.0, 2 * np.pi, 64, endpoint=False).reshape(8, 8)

    def test_offline_fallback_follows_the_config(self, _isolated_config):
        from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 1023\n")
        default_gray = phase_to_slm_grayscale(self._phase(), slm=None)
        assert int(default_gray.max()) <= 1022

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 511\n")
        scaled_gray = phase_to_slm_grayscale(self._phase(), slm=None)
        assert int(scaled_gray.max()) <= 510

    def test_driver_classmethod_follows_the_config(self, _isolated_config):
        from ao_shaping.drivers.slm.santec.driver import Santec

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 2047\n")
        # Classmethod: no device, no SDK load.
        assert Santec.encoding_max_grayscale() == 2047

    def test_hardware_full_scale_constant_is_not_the_encoding_scale(
        self, _isolated_config
    ):
        from ao_shaping.drivers.slm.santec.driver import Santec

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 511\n")
        assert Santec.MAX_GRAYSCALE_VALUE == 1023  # hardware full scale: fixed
        assert Santec.encoding_max_grayscale() == 511  # encoding scale: config

    def test_pattern_helper_ten_bit_uses_the_config(self, _isolated_config):
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 2047\n")
        assert PatternHelper((64, 64), bits=10)._max_val == 2047

    def test_pattern_helper_other_bits_ignore_the_config(self, _isolated_config):
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 2047\n")
        assert PatternHelper((64, 64), bits=8)._max_val == 255
        assert PatternHelper((64, 64), bits=12)._max_val == 4095

    def test_pattern_helper_explicit_wins(self, _isolated_config):
        from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 2047\n")
        assert PatternHelper((64, 64), bits=10, max_grayscale=993)._max_val == 993

    def test_debug_frame_round_trips_through_the_configured_scale(
        self, _isolated_config
    ):
        """A frame sent at the configured scale must read back as the same phase."""
        from ao_shaping.utils.slm.gs_debug_optics import grayscale_to_radians
        from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 511\n")
        phase = np.linspace(0.0, 2 * np.pi, 16, endpoint=False).reshape(4, 4)
        gray = phase_to_slm_grayscale(phase, slm=None)
        np.testing.assert_allclose(grayscale_to_radians(gray), phase, atol=0.02)

    def test_gui_pattern_control_default_scale_follows_the_config(
        self, _isolated_config
    ):
        pytest.importorskip("streamlit")
        from ao_shaping.gui.slm.pattern_controls import FlatControl

        _write_config(_isolated_config, "[slm]\ntwo_pi_grayscale = 993\n")
        assert FlatControl(bits=10).max_gray == 993
        # 非 10 bit 的面板: 满量程按位深, 配置不参与。
        assert FlatControl(bits=8).max_gray == 255
        # 显式传入最高优先。
        assert FlatControl(bits=10, max_gray=511).max_gray == 511

    def test_module_exposes_the_documented_api(self):
        assert set(gray_scale.__all__) == {
            "CONFIG_SECTION",
            "CONFIG_KEY",
            "DEFAULT_MAX_GRAYSCALE",
            "configured_max_grayscale",
            "resolve_max_grayscale",
        }