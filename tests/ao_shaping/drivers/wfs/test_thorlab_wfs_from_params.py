from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from ao_shaping.drivers.wfs.thorlab_wfs import MlaRes, ThorlabWFS


def _constructor_defaults() -> dict[str, Any]:
    signature = inspect.signature(ThorlabWFS.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _capture_constructor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_init(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(ThorlabWFS, "__init__", fake_init)
    return calls


def _assert_factory_matches(
    monkeypatch: pytest.MonkeyPatch,
    params: object,
    expected: dict[str, Any],
    **overrides: Any,
) -> None:
    calls = _capture_constructor(monkeypatch)
    expected_instance = ThorlabWFS(**expected)
    actual_instance = ThorlabWFS.from_params(params, **overrides)

    assert calls == [expected, expected]
    assert type(actual_instance) is ThorlabWFS
    assert type(expected_instance) is ThorlabWFS


def test_wfs_from_params_matches_explicit_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = SimpleNamespace(
        mla_index=MlaRes.Res768,
        exposure_time=6.5,
        high_speed=True,
        use_custom_ref=False,
        pupil_diameter=4.6,
        pupil_center=(12.0, -8.0),
        stable_sample_enable=True,
        stable_sample_n=7,
        stable_variance_threshold=0.25,
        stable_max_attempts=12,
        device_id="wfs-1",
    )

    _assert_factory_matches(monkeypatch, params, vars(params))


def test_wfs_from_params_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_factory_matches(monkeypatch, SimpleNamespace(), _constructor_defaults())


def test_wfs_from_params_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _constructor_defaults() | {
        "mla_index": MlaRes.Res512,
        "device_id": "wfs-override",
    }
    _assert_factory_matches(
        monkeypatch,
        SimpleNamespace(mla_index=MlaRes.Res768, device_id="wfs-param"),
        expected,
        mla_index=MlaRes.Res512,
        device_id="wfs-override",
    )
