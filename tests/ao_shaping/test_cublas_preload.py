"""``_cublas_preload`` -- the guards, the file it picks, and where it is called from.

The bug it closes: on Windows, ``import cupy`` loads ``cublasLt64_13.dll`` *by bare
name*, so a CUDA toolkit on ``PATH`` wins over the copy in ``torch/lib``; torch's
``cublas64_13.dll`` then calls a cuBLASLt it was not built against and the process
dies with ``0xc0000005`` inside cuBLASLt. No Python exception, no traceback, no
journal entry (2026-11-03 transfer fit, ``data/debug/slm_transfer/20261103_213400``).

These tests stay on the decision and on the call sites that make the fix hold:
mapping a 456 MB DLL, and loading torch, are both out of scope here on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ao_shaping import _cublas_preload
from ao_shaping._cublas_preload import preload_torch_cublaslt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _fake_torch(tmp_path: Path, *dll_names: str) -> Path:
    """A stand-in torch install whose ``lib/`` holds the named files."""
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (tmp_path / "torch" / "__init__.py").write_text("", encoding="utf-8")
    for name in dll_names:
        (lib / name).write_bytes(b"not a real PE")
    return tmp_path / "torch" / "__init__.py"


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
    assert preload_torch_cublaslt(platform="linux", loader=spy) is None
    assert spy.loaded == []


def test_skip_env_var_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll")
    monkeypatch.setenv("AO_SKIP_CUBLAS_PRELOAD", "1")
    spy = _SpyLoader()
    assert preload_torch_cublaslt(platform="win32", loader=spy, torch_origin=str(origin)) is None
    assert spy.loaded == []


def test_missing_torch_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AO_SKIP_CUBLAS_PRELOAD", raising=False)
    monkeypatch.setattr(_cublas_preload, "_torch_origin", lambda: None)
    spy = _SpyLoader()
    assert preload_torch_cublaslt(platform="win32", loader=spy) is None
    assert spy.loaded == []


def test_torch_without_a_bundled_cublaslt_is_a_no_op(tmp_path: Path) -> None:
    origin = _fake_torch(tmp_path, "cublas64_13.dll")
    spy = _SpyLoader()
    assert preload_torch_cublaslt(platform="win32", loader=spy, torch_origin=str(origin)) is None
    assert spy.loaded == []


def test_prefers_the_newest_cuda_major(tmp_path: Path) -> None:
    """torch ships one copy per CUDA major; the newest is the one torch 2.x+ links."""
    origin = _fake_torch(tmp_path, "cublasLt64_12.dll", "cublasLt64_13.dll")
    spy = _SpyLoader()
    picked = preload_torch_cublaslt(platform="win32", loader=spy, torch_origin=str(origin))
    assert picked is not None
    assert picked.name == "cublasLt64_13.dll"


def test_loads_by_full_path(tmp_path: Path) -> None:
    """A bare name would re-run the PATH search -- the whole point is the full path."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll")
    spy = _SpyLoader()
    picked = preload_torch_cublaslt(platform="win32", loader=spy, torch_origin=str(origin))
    assert picked is not None
    assert spy.loaded == [str(picked)]
    assert Path(spy.loaded[0]).is_absolute()


def test_load_failure_does_not_propagate(tmp_path: Path) -> None:
    """No driver / a truncated install must not make ``import ao_shaping`` fail."""
    origin = _fake_torch(tmp_path, "cublasLt64_13.dll")
    assert preload_torch_cublaslt(
        platform="win32", loader=_FailingLoader(), torch_origin=str(origin)
    ) is None


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
#: loads the bare-name cuBLASLt, so the preload has to be textually earlier in
#: these files, not merely "somewhere earlier at runtime".
_CUPPY_ENTRY_POINTS = (
    ("src/ao_shaping/utils/image/spots_calc.py", "import cupy"),
    ("src/ao_shaping/drivers/sim/_oopao_compat.py", "_load_submodule(name)"),
)


@pytest.mark.parametrize(("relative_path", "hazard"), _CUPPY_ENTRY_POINTS)
def test_preload_precedes_the_hazard_in_every_cupy_entry_point(
    relative_path: str, hazard: str
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    call = text.find("preload_torch_cublaslt()")
    assert call != -1, f"{relative_path} no longer runs the cuBLASLt preload"
    assert call < text.find(hazard), (
        f"{relative_path}: the preload must come before {hazard!r}, or CuPy binds "
        "cuBLASLt to whichever CUDA toolkit PATH exposes and torch crashes inside it"
    )
