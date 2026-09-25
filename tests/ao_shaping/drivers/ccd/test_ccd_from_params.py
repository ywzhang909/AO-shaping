from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from ao_shaping.drivers.ccd import common
from ao_shaping.drivers.ccd.common import CameraSpec
from ao_shaping.drivers.ccd.daheng.driver import DahengCamera
from ao_shaping.drivers.ccd.miicam.driver import MIICamera


def _constructor_defaults(driver_cls: type[Any]) -> dict[str, Any]:
    signature = inspect.signature(driver_cls.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _capture_constructor(
    monkeypatch: pytest.MonkeyPatch, driver_cls: type[Any]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_init(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(driver_cls, "__init__", fake_init)
    return calls


def _assert_factory_matches(
    monkeypatch: pytest.MonkeyPatch,
    driver_cls: type[Any],
    params: object,
    expected: dict[str, Any],
    **overrides: Any,
) -> None:
    calls = _capture_constructor(monkeypatch, driver_cls)
    expected_instance = driver_cls(**expected)
    actual_instance = driver_cls.from_params(params, **overrides)

    assert calls == [expected, expected]
    assert type(actual_instance) is driver_cls
    assert type(expected_instance) is driver_cls


@pytest.mark.parametrize(
    ("driver_cls", "params"),
    [
        (
            DahengCamera,
            SimpleNamespace(
                cam_id=2,
                exposure_time_ms=12.5,
                skip_sampling=True,
                bit_depth=16,
            ),
        ),
        (
            MIICamera,
            SimpleNamespace(
                cam_id=3,
                exposure_time_ms=4.5,
                skip_sampling=False,
                bit_depth=12,
                capture_mode="callback",
            ),
        ),
    ],
)
def test_camera_from_params_matches_explicit_constructor(
    monkeypatch: pytest.MonkeyPatch,
    driver_cls: type[Any],
    params: SimpleNamespace,
) -> None:
    _assert_factory_matches(monkeypatch, driver_cls, params, vars(params))


@pytest.mark.parametrize("driver_cls", [DahengCamera, MIICamera])
def test_camera_from_params_uses_constructor_defaults(
    monkeypatch: pytest.MonkeyPatch, driver_cls: type[Any]
) -> None:
    expected = _constructor_defaults(driver_cls)
    _assert_factory_matches(monkeypatch, driver_cls, SimpleNamespace(), expected)


@pytest.mark.parametrize(
    ("driver_cls", "override"),
    [(DahengCamera, {"cam_id": 9}), (MIICamera, {"capture_mode": "callback"})],
)
def test_camera_from_params_overrides_win(
    monkeypatch: pytest.MonkeyPatch,
    driver_cls: type[Any],
    override: dict[str, Any],
) -> None:
    expected = _constructor_defaults(driver_cls) | override
    _assert_factory_matches(
        monkeypatch,
        driver_cls,
        SimpleNamespace(),
        expected,
        **override,
    )


class _StubCamera:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_camera_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CameraSpec(
        name="stub",
        target="tests.ao_shaping.drivers.ccd.test_from_params:_StubCamera",
        accepted_kwargs=frozenset({"cam_id", "exposure_time_ms", "skip_sampling"}),
    )
    monkeypatch.setattr(common, "CAMERA_TYPES", {"stub": spec})
    monkeypatch.setattr(common, "_resolve_camera_class", lambda target: _StubCamera)


def test_create_camera_param_object_matches_string_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_registry(monkeypatch)
    params: Any = SimpleNamespace(
        cam_type="StUb",
        cam_id=3,
        exposure_time_ms=5.0,
        skip_sampling=True,
    )
    from_string: Any = common.create_camera(
        "StUb",
        cam_id=3,
        exposure_time_ms=5.0,
        skip_sampling=True,
    )
    from_params: Any = common.create_camera(params)

    assert type(from_params) is _StubCamera
    assert from_params.kwargs == from_string.kwargs


def test_create_camera_param_object_falls_back_without_optional_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_registry(monkeypatch)
    params: Any = SimpleNamespace(cam_type="stub")

    camera: Any = common.create_camera(
        params,
        cam_id=6,
        exposure_time_ms=3.25,
    )

    assert isinstance(camera, _StubCamera)
    assert camera.kwargs == {"cam_id": 6, "exposure_time_ms": 3.25}
