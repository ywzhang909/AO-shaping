from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from ao_shaping.drivers.dm.NLight import NLight


def _constructor_defaults() -> dict[str, Any]:
    signature = inspect.signature(NLight.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _capture_constructor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_init(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(NLight, "__init__", fake_init)
    return calls


def _assert_factory_matches(
    monkeypatch: pytest.MonkeyPatch,
    params: object,
    expected: dict[str, Any],
    **overrides: Any,
) -> None:
    calls = _capture_constructor(monkeypatch)
    nlight_cls: Any = NLight
    expected_instance = NLight(**expected)
    actual_instance = nlight_cls.from_params(params, **overrides)

    assert calls == [expected, expected]
    assert type(actual_instance) is NLight
    assert type(expected_instance) is NLight


def test_nlight_from_params_matches_explicit_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = SimpleNamespace(
        max_iter_diff=12,
        max_neibor_diff=175.5,
        keep_when_exit=False,
        safety_mode=False,
    )
    _assert_factory_matches(monkeypatch, params, vars(params))


def test_nlight_from_params_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_factory_matches(monkeypatch, SimpleNamespace(), _constructor_defaults())


def test_nlight_from_params_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _constructor_defaults() | {
        "max_iter_diff": 30,
        "safety_mode": True,
    }
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(max_iter_diff=10, safety_mode=False),
        expected,
        max_iter_diff=30,
        safety_mode=True,
    )
