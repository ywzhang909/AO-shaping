"""Contract tests for driver lazy loading.

Two distinct laziness properties are pinned here, because they are separate
defects:

**Package level.** ``import ao_shaping.drivers.<family>`` must not pull in a
native SDK. This was already true for the camera family (``test_lazy_backend_
imports.py``) but not for the others: ``drivers/__init__.py`` eagerly imported
``NLight`` and ``ThorlabWFS``, and the PEP 562 ``__getattr__`` body was
duplicated verbatim in ``drivers/__init__.py`` and ``drivers/ccd/__init__.py``.

**Construction level.** Constructing a driver must not load its SDK either.
``ThorlabWFS.__init__`` called ``load_dll()`` directly, so merely *building* the
object raised ``OSError`` on any machine without the vendor DLL — even though
``load_dll()`` was already written as a properly deferred function. That makes
the driver impossible to construct, register or introspect offline, and it is
the same class of problem that makes a silent hardware/sim swap unsafe.

The SDK must still be loaded when hardware is actually used, i.e. on ``open()``.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ao_shaping.drivers.device_base import Device

_ROOT = Path(__file__).resolve().parents[3]


def _run(snippet: str) -> subprocess.CompletedProcess[str]:
    """Execute ``snippet`` in a clean interpreter rooted at the repo."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env={"PYTHONPATH": "src:libs", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        timeout=180,
    )


class TestBaseClassSdkLaziness:
    """``Device`` must expose a declarative, once-resolved SDK hook."""

    def test_device_declares_the_hook(self):
        assert hasattr(Device, "_ensure_sdk"), "Device must offer _ensure_sdk()"

    def test_sdk_is_not_resolved_by_construction(self):
        """A driver subclass with an SDK loader must construct without invoking it."""
        loaded: list[int] = []

        class _Fake(Device):
            @staticmethod
            def _load_sdk() -> str:
                loaded.append(1)
                return "sdk-handle"

            def open(self) -> None:
                self._ensure_sdk()

            def close(self) -> None:
                pass

            def is_connected(self) -> bool:
                return False

            def get_hardware_info(self) -> dict:
                return {}

        obj = _Fake()
        assert loaded == [], "constructing a driver must not load its SDK"
        assert obj._ensure_sdk() == "sdk-handle"
        assert loaded == [1], "the loader runs exactly once"

    def test_sdk_handle_is_cached(self):
        calls: list[int] = []

        class _Fake(Device):
            @staticmethod
            def _load_sdk() -> int:
                calls.append(1)
                return len(calls)

            def open(self) -> None:
                pass

            def close(self) -> None:
                pass

            def is_connected(self) -> bool:
                return False

            def get_hardware_info(self) -> dict:
                return {}

        obj = _Fake()
        first = obj._ensure_sdk()
        assert obj._ensure_sdk() is first, "the handle must be cached, not reloaded"
        assert len(calls) == 1

    def test_driver_without_sdk_is_supported(self):
        """Simulated devices have no SDK; the hook must degrade to None."""

        class _NoSdk(Device):
            def open(self) -> None:
                pass

            def close(self) -> None:
                pass

            def is_connected(self) -> bool:
                return True

            def get_hardware_info(self) -> dict:
                return {}

        assert _NoSdk()._ensure_sdk() is None


class TestThorlabWfsConstructionIsSdkFree:
    """The real regression: constructing ``ThorlabWFS`` used to raise OSError."""

    def test_constructs_without_the_vendor_dll(self):
        result = _run(
            """
            from ao_shaping.drivers.wfs import ThorlabWFS
            wfs = ThorlabWFS()
            print("CONSTRUCTED", type(wfs).__name__)
            """
        )
        assert result.returncode == 0, (
            "constructing ThorlabWFS must not require the vendor DLL:\n"
            f"{result.stderr[-2000:]}"
        )
        assert "CONSTRUCTED ThorlabWFS" in result.stdout

    def test_sdk_load_is_deferred_to_open(self):
        """``open()`` is where the SDK must be demanded — and it must still fail."""
        result = _run(
            """
            from ao_shaping.drivers.wfs import ThorlabWFS
            wfs = ThorlabWFS()
            try:
                wfs.open()
            except Exception as exc:
                print("OPEN_RAISED", type(exc).__name__)
            else:
                print("OPEN_SUCCEEDED")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OPEN_RAISED" in result.stdout, (
            "open() must still demand the SDK; a silent success would mean the "
            "hook is not wired"
        )

    def test_constructor_no_longer_calls_the_dll_loader(self):
        """Guard the specific line that used to do the eager load."""
        from ao_shaping.drivers.wfs import thorlab_wfs

        src = inspect.getsource(thorlab_wfs.ThorlabWFS.__init__)
        assert "load_dll()" not in src, (
            "ThorlabWFS.__init__ must not call load_dll(); resolution belongs in "
            "the lazy SDK hook"
        )


class TestNoDuplicatedLazyHelper:
    """The PEP 562 body must exist once, not copy-pasted per package."""

    def test_shared_helper_module_exists(self):
        from ao_shaping.drivers import _lazy

        assert hasattr(_lazy, "install_lazy_attrs"), (
            "the PEP 562 __getattr__ logic must be shared, not duplicated"
        )

    def test_driver_packages_delegate_to_the_helper(self):
        """No driver package should hand-roll its own ``_LAZY_BACKENDS`` map."""
        offenders = []
        for init in sorted((_ROOT / "src/ao_shaping/drivers").rglob("__init__.py")):
            text = init.read_text(encoding="utf-8")
            if "_LAZY_BACKENDS" in text and "install_lazy_attrs" not in text:
                offenders.append(str(init.relative_to(_ROOT)))
        assert not offenders, f"hand-rolled lazy maps duplicated in: {offenders}"

    def test_wfs_package_still_exposes_its_public_names(self):
        result = _run(
            """
            from ao_shaping.drivers.wfs import BaseWFS, ThorlabWFS, MlaRes
            from ao_shaping.drivers import ThorlabWFS as T2
            print("OK", BaseWFS.__name__, ThorlabWFS.__name__, MlaRes.__name__)
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK BaseWFS ThorlabWFS MlaRes" in result.stdout
