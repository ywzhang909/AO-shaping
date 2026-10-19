from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from ao_shaping.drivers.slm.santec.constants import VideoMode
from ao_shaping.drivers.slm.santec.driver import Santec


def _constructor_defaults() -> dict[str, Any]:
    signature = inspect.signature(Santec.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _capture_constructor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_init(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(Santec, "__init__", fake_init)
    return calls


def _assert_factory_matches(
    monkeypatch: pytest.MonkeyPatch,
    params: object,
    expected: dict[str, Any],
    **overrides: Any,
) -> None:
    calls = _capture_constructor(monkeypatch)
    expected_instance = Santec(**expected)
    actual_instance = Santec.from_params(params, **overrides)

    assert calls == [expected, expected]
    assert type(actual_instance) is Santec
    assert type(expected_instance) is Santec


def test_santec_from_params_maps_slm_wavelength_to_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = SimpleNamespace(
        slm_number=2,
        use_120hz=True,
        slm_wavelength=1064,
        video_mode=VideoMode.Memory,
        shift_x=7,
        shift_y=-9,
        correction_csv_path="correction.csv",
    )
    expected = {
        "slm_number": 2,
        "use_120hz": True,
        "wavelength": 1064,
        "video_mode": VideoMode.Memory,
        "shift_x": 7,
        "shift_y": -9,
        "correction_csv_path": "correction.csv",
    }

    _assert_factory_matches(monkeypatch, params, expected)


def test_santec_from_params_accepts_current_wavelength_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _constructor_defaults() | {"wavelength": 532}
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(wavelength=532),
        expected,
    )


def test_santec_from_params_zero_wavelength_uses_device_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(slm_wavelength=0),
        _constructor_defaults(),
    )


def test_santec_from_params_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_factory_matches(monkeypatch, SimpleNamespace(), _constructor_defaults())


def test_santec_from_params_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _constructor_defaults() | {
        "slm_number": 3,
        "wavelength": 633,
    }
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(slm_number=1, slm_wavelength=532),
        expected,
        slm_number=3,
        wavelength=633,
    )


class TestLoadedConfig:
    """``open()`` caches the per-serial config it reads; callers must not re-read.

    ``Santec.open()`` reads the matching JSON file to apply wavelength / shift /
    refresh defaults.  The GUI then called ``load_config()`` a second time to
    show the same values in the sidebar, re-reading and re-parsing one file per
    connect.  ``loaded_config`` is the snapshot ``open()`` already holds.
    """

    def test_unopened_handle_reports_none(self) -> None:
        pytest.importorskip("ao_shaping.drivers.slm.santec._slm_win")
        assert Santec(slm_number=1).loaded_config is None

    def test_open_stores_what_it_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("ao_shaping.drivers.slm.santec._slm_win")

        config = {"wavelength": 1064, "shift_x": 3}
        reads: list[str] = []

        class _Sdk:
            def SLM_Ctrl_Close(self, *_args: Any) -> int:
                return 0

            def SLM_Ctrl_Open(self, *_args: Any) -> int:
                return 0

        slm = Santec(slm_number=1)
        monkeypatch.setattr(slm, "_slm", _Sdk())
        monkeypatch.setattr(slm, "_wait_for_ready", lambda: None)
        monkeypatch.setattr(slm, "get_serial_number", lambda *a, **k: "TEST-SERIAL")
        monkeypatch.setattr(slm, "_apply_config_params", lambda _cfg: None)
        monkeypatch.setattr(slm, "_load_correction", lambda _cfg: None)
        monkeypatch.setattr(slm, "_setup_wavelength", lambda: None)
        monkeypatch.setattr(slm, "_set_memory_mode", lambda *_a, **_k: None)

        def fake_load_config() -> dict[str, Any]:
            reads.append("disk")
            return dict(config)

        monkeypatch.setattr(slm, "load_config", fake_load_config)

        slm.open()

        # Read exactly once, and the snapshot is the object that was read.
        assert reads == ["disk"]
        assert slm.loaded_config == config
