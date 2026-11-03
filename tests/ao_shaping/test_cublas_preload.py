"""``_cublas_preload`` -- the guards, which pair it pins, and where it is called from.

The bug it closes: on Windows, ``import cupy`` loads ``cublasLt64_13.dll`` and
``cublas64_13.dll`` *by bare name*, so a CUDA toolkit on ``PATH`` takes them over.
That is harmless while both names land on the same build, but a **mixed** pair --
cuBLASLt from one build, cublas from another -- faults with ``0xC0000005`` inside
cuBLASLt, with no Python exception, no traceback and no journal entry (2026-11-03
transfer fit, ``data/debug/slm_transfer/20261103_213400``). The fix is to pin both
names to one self-consistent build, and the only self-consistent set is the
toolkit's (torch ships no cuSOLVER, cudart or cuFFT).

These tests stay on the decision and on the call sites: mapping a real DLL and
importing torch are both out of scope here on purpose. One test reads a real file
version, and skips when torch or Windows is absent.
"""

from __future__ import annotations

import logging
import sys
from importlib import util as importlib_util
from pathlib import Path

import pytest

from ao_shaping import _cublas_preload
from ao_shaping._cublas_preload import pin_cublas

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_VERSION = (6, 14, 11, 1300)
TOOLKIT_VERSION = (6, 14, 11, 1340)


@pytest.fixture(autouse=True)
def _fresh_process_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts as a fresh process: the latch is module-level on purpose."""
    monkeypatch.setattr(_cublas_preload, "_done", False)
    monkeypatch.delenv("AO_SKIP_CUBLAS_PRELOAD", raising=False)


def _fake_torch(tmp_path: Path, *dll_names: str) -> Path:
    """A stand-in torch install whose ``lib/`` holds the named files."""
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (tmp_path / "torch" / "__init__.py").write_text("", encoding="utf-8")
    for name in dll_names:
        (lib / name).write_bytes(b"not a real PE")
    return tmp_path / "torch" / "__init__.py"


def _fake_dir(tmp_path: Path, name: str, *dll_names: str) -> Path:
    """A stand-in ``PATH`` entry holding the named files."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    for dll_name in dll_names:
        (directory / dll_name).write_bytes(b"not a real PE")
    return directory


class _SpyLoader:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def __call__(self, path: str) -> object:
        self.loaded.append(path)
        return object()


class _FailingLoader:
    def __call__(self, path: str) -> object:
        raise OSError(126, "The specified module could not be found", path)


def test_non_windows_is_a_no_op() -> None:
    spy = _SpyLoader()
    assert pin_cublas(platform="linux", loader=spy) is None
    assert spy.loaded == []


def test_skip_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    monkeypatch.setenv("AO_SKIP_CUBLAS_PRELOAD", "1")
    spy = _SpyLoader()
    assert (
        pin_cublas(platform="win32", loader=spy, torch_origin=str(origin))
        is None
    )
    assert spy.loaded == []


def test_missing_torch_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cublas_preload, "_torch_origin", lambda: None)
    spy = _SpyLoader()
    assert pin_cublas(platform="win32", loader=spy) is None
    assert spy.loaded == []


def test_torch_without_a_bundled_cublaslt_is_a_no_op(tmp_path: Path) -> None:
    origin = _fake_torch(tmp_path, "cublas64_13.dll")
    spy = _SpyLoader()
    assert pin_cublas(platform="win32", loader=spy, torch_origin=str(origin)) is None
    assert spy.loaded == []


def test_no_same_major_on_path_pins_the_torch_pair(tmp_path: Path) -> None:
    """A toolkit on a different CUDA major is not a conflict: the bare names differ."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    other = _fake_dir(tmp_path, "cuda12", "cublasLt64_12.dll", "cublas64_12.dll")
    spy = _SpyLoader()

    loaded = pin_cublas(
        platform="win32",
        loader=spy,
        torch_origin=str(origin),
        search_path=[str(other), ""],
    )

    lib = tmp_path / "torch" / "lib"
    assert loaded == [lib / "cublasLt64_13.dll", lib / "cublas64_13.dll"]
    assert spy.loaded == [str(p) for p in loaded]


def test_newer_toolkit_wins_and_is_loaded_lt_first(tmp_path: Path) -> None:
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    toolkit = _fake_dir(tmp_path, "toolkit", "cublasLt64_13.dll", "cublas64_13.dll")
    spy = _SpyLoader()

    loaded = pin_cublas(
        platform="win32",
        loader=spy,
        torch_origin=str(origin),
        search_path=[str(tmp_path / "missing"), str(toolkit)],
        file_version=lambda path: (
            TOOLKIT_VERSION if path.parent == toolkit else TORCH_VERSION
        ),
    )

    # Full paths, cuBLASLt before cublas -- the full path is the whole point, a
    # bare name would re-run the PATH search that got us here.
    assert loaded == [toolkit / "cublasLt64_13.dll", toolkit / "cublas64_13.dll"]
    assert spy.loaded == [
        str(toolkit / "cublasLt64_13.dll"),
        str(toolkit / "cublas64_13.dll"),
    ]


def test_older_toolkit_loses_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A toolkit older than torch's build must not take over: torch would regress."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    toolkit = _fake_dir(tmp_path, "toolkit", "cublasLt64_13.dll", "cublas64_13.dll")
    spy = _SpyLoader()
    caplog.set_level(logging.WARNING, logger="ao_shaping._cublas_preload")

    loaded = pin_cublas(
        platform="win32",
        loader=spy,
        torch_origin=str(origin),
        search_path=[str(toolkit)],
        file_version=lambda path: (
            (6, 13, 0, 2) if path.parent == toolkit else TORCH_VERSION
        ),
    )

    lib = tmp_path / "torch" / "lib"
    assert loaded == [lib / "cublasLt64_13.dll", lib / "cublas64_13.dll"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "older" in warnings[0].getMessage()
    assert "6.13.0.2" in warnings[0].getMessage()


def test_unreadable_version_prefers_the_toolkit(tmp_path: Path) -> None:
    """No version is no evidence of regression, so the ordering that never crashed wins."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    toolkit = _fake_dir(tmp_path, "toolkit", "cublasLt64_13.dll", "cublas64_13.dll")
    spy = _SpyLoader()

    loaded = pin_cublas(
        platform="win32",
        loader=spy,
        torch_origin=str(origin),
        search_path=[str(toolkit)],
        file_version=lambda path: None,
    )

    assert loaded == [toolkit / "cublasLt64_13.dll", toolkit / "cublas64_13.dll"]


def test_second_call_does_not_load_again(tmp_path: Path) -> None:
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    first = _SpyLoader()
    second = _SpyLoader()
    kwargs = {"platform": "win32", "torch_origin": str(origin), "search_path": []}

    assert pin_cublas(loader=first, **kwargs) is not None
    assert pin_cublas(loader=second, **kwargs) is None
    assert len(first.loaded) == 2
    assert second.loaded == []


def test_loader_failure_is_swallowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No driver / a truncated install must not make ``import cupy`` fail."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    caplog.set_level(logging.DEBUG, logger="ao_shaping._cublas_preload")

    assert (
        pin_cublas(
            platform="win32",
            loader=_FailingLoader(),
            torch_origin=str(origin),
            search_path=[],
        )
        is None
    )
    records = [r for r in caplog.records if r.name == "ao_shaping._cublas_preload"]
    assert any(
        "pinning" in r.getMessage() and "could not be found" in r.getMessage()
        for r in records
    )
    assert all(r.levelno < logging.WARNING for r in records)


def test_version_probe_failure_is_swallowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll", "cublas64_13.dll")
    toolkit = _fake_dir(tmp_path, "toolkit", "cublasLt64_13.dll", "cublas64_13.dll")
    caplog.set_level(logging.DEBUG, logger="ao_shaping._cublas_preload")
    spy = _SpyLoader()

    def explode(path: Path) -> tuple[int, int, int, int] | None:
        raise OSError("version.dll is unhappy")

    assert (
        pin_cublas(
            platform="win32",
            loader=spy,
            torch_origin=str(origin),
            search_path=[str(toolkit)],
            file_version=explode,
        )
        is None
    )
    assert spy.loaded == []
    assert any("pinning failed" in record.getMessage() for record in caplog.records)


def test_prefers_the_newest_cuda_major(tmp_path: Path) -> None:
    """torch ships one copy per CUDA major; the newest is the one torch 2.x+ links."""
    origin = _fake_torch(
        tmp_path,
        "cublasLt64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_13.dll",
        "cublas64_13.dll",
    )
    spy = _SpyLoader()
    loaded = pin_cublas(
        platform="win32", loader=spy, torch_origin=str(origin), search_path=[]
    )
    assert loaded is not None
    assert [path.name for path in loaded] == ["cublasLt64_13.dll", "cublas64_13.dll"]


def test_file_version_reads_a_real_dll() -> None:
    """The parser against the DLL this bench actually loads, not a stub."""
    if sys.platform != "win32":
        pytest.skip("file versions come from version.dll")
    try:
        spec = importlib_util.find_spec("torch")
    except (ImportError, ValueError):  # pragma: no cover - broken torch install
        spec = None
    if spec is None or not spec.origin:  # pragma: no cover - torch not installed
        pytest.skip("torch is not installed in this environment")
    dll = Path(spec.origin).parent / "lib" / "cublasLt64_13.dll"
    if not dll.is_file():  # pragma: no cover - a non-cu130 torch build
        pytest.skip(f"{dll} does not exist")
    assert _cublas_preload._file_version(dll) == TORCH_VERSION


def test_torch_origin_prefers_an_already_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded torch is read from ``sys.modules``: re-finding it could import it."""
    origin = Path("/nowhere/torch/__init__.py")

    class _Loaded:
        __file__ = str(origin)

    monkeypatch.setitem(sys.modules, "torch", _Loaded())
    assert _cublas_preload._torch_origin() == str(origin)


def test_torch_origin_falls_back_to_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    seen: list[str] = []

    def fake_find_spec(name: str) -> object:
        seen.append(name)
        return type("Spec", (), {"origin": "/found/torch/__init__.py"})()

    monkeypatch.setattr(_cublas_preload._importlib_util, "find_spec", fake_find_spec)
    assert _cublas_preload._torch_origin() == "/found/torch/__init__.py"
    assert seen == ["torch"]


def test_partially_initialised_torch_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """``find_spec`` raises ``ValueError`` for a module whose ``__spec__`` is None."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def boom(name: str) -> object:
        raise ValueError(f"{name}.__spec__ is None")

    monkeypatch.setattr(_cublas_preload._importlib_util, "find_spec", boom)
    assert _cublas_preload._torch_origin() is None


#: The modules that hand this process a CuPy CUDA runtime. CuPy's import is what
#: binds the bare names, so the pin has to be textually earlier in these files, not
#: merely "somewhere earlier at runtime".
_CUPY_ENTRY_POINTS = (
    ("src/ao_shaping/utils/image/spots_calc.py", "import cupy"),
    ("src/ao_shaping/drivers/sim/_oopao_compat.py", "_load_submodule(name)"),
)


@pytest.mark.parametrize(("relative_path", "hazard"), _CUPY_ENTRY_POINTS)
def test_pin_precedes_the_hazard_in_every_cupy_entry_point(
    relative_path: str, hazard: str
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    call = text.find("pin_cublas()")
    assert call != -1, f"{relative_path} no longer pins the cuBLAS names"
    assert call < text.find(hazard), (
        f"{relative_path}: the pin must come before {hazard!r}, or CuPy binds the "
        "bare cuBLAS names to whatever PATH exposes and a mixed pair crashes torch"
    )
