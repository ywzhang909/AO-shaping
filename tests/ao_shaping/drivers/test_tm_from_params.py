from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from ao_shaping.drivers.tm.serial_port_fsm import SerialPortFSM


def _constructor_defaults() -> dict[str, Any]:
    signature = inspect.signature(SerialPortFSM.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _capture_constructor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_init(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(SerialPortFSM, "__init__", fake_init)
    return calls


def _assert_factory_matches(
    monkeypatch: pytest.MonkeyPatch,
    params: object,
    expected: dict[str, Any],
    **overrides: Any,
) -> None:
    calls = _capture_constructor(monkeypatch)
    expected_instance = SerialPortFSM(**expected)
    actual_instance = SerialPortFSM.from_params(params, **overrides)

    assert calls == [expected, expected]
    assert type(actual_instance) is SerialPortFSM
    assert type(expected_instance) is SerialPortFSM


def test_serial_fsm_from_params_matches_explicit_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = SimpleNamespace(port="COM7", baud=115200)
    _assert_factory_matches(monkeypatch, params, vars(params))


def test_serial_fsm_from_params_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_factory_matches(monkeypatch, SimpleNamespace(), _constructor_defaults())


def test_serial_fsm_from_params_overrides_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _constructor_defaults() | {"port": "COM9", "baud": 9600}
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(port="COM7", baud=115200),
        expected,
        port="COM9",
        baud=9600,
    )
