"""生成后端内核性能对比报告（NumPy / Numba / CuPy 逐函数耗时）。

仓库约定：markdown 报告生成一律放 ``scripts/``（命名 ``generate_*_report.py``），
产物写入 ``report/<topic>/``；脚本离线可跑。

本脚本是原 ``tests/ao_shaping/utils/test_spots_calc.py::TestPerformance::
test_performance_comparison`` 的**逐字移植**。原实现是一个零断言、只打印与写文件
的报告生成器，被挂在测试用例里：它写回仓库，于是在 ``run_pytest.sh`` 的只读挂载下
必然以 ``OSError[Errno 30]`` 失败（issue #162），而注册在 provenance 里的复现命令
``pytest ... -k benchmark`` 实际匹配 **0 个测试**。计时口径、采样数、表头、CuPy
不可用原因的记录方式全部保持原样，因此新旧报告的数字可直接对比。

用法：
    python scripts/generate_backend_kernels_report.py
"""

from __future__ import annotations

import sys
from loguru import logger
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap (repo rule): top-level script -> parents[1]
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))
# The repo root itself: `scripts._common.provenance` is imported as a package
# path, and `sys.path[0]` is this file's own directory when the script is run
# as `python scripts/<name>.py` -- so the root has to be added explicitly.
sys.path.insert(0, str(ROOT))

from ao_shaping.utils.image.spots_calc import (
    calculate_sharpness,
    calculate_sharpness_numba,
    center_of_brightness,
    center_of_brightness_numba,
    center_of_mass_numba,
    centroid,
    crop,
    crop_numba,
)

# ---------------------------------------------------------------------------
# Optional CuPy backend (same import-time probing as the test module)
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
    CUPY_UNAVAILABLE_REASON = None
except ImportError as exc:
    cp = None
    CUPY_AVAILABLE = False
    CUPY_UNAVAILABLE_REASON = (
        f"CuPy is not installed in the active Python environment ({exc})"
    )
except Exception as exc:  # mirror the test's broad guard
    cp = None
    CUPY_AVAILABLE = False
    CUPY_UNAVAILABLE_REASON = f"CuPy initialization failed ({type(exc).__name__}: {exc})"
else:
    if not CUPY_AVAILABLE:
        CUPY_UNAVAILABLE_REASON = "No CUDA-capable GPU detected by CuPy"

#: CuPy kernels by report row name. Populated only when CuPy is importable and a
#: GPU is visible, so a missing entry means exactly what the original in-line
#: ``if CUPY_AVAILABLE: ... else: 'N/A'`` recorded. Names are never referenced
#: outside this guard, which is why the port cannot raise ``NameError`` on a
#: GPU-less host.
_CUPY_KERNELS: dict[str, object] = {}
if CUPY_AVAILABLE:
    from ao_shaping.utils.image.spots_calc import (
        calculate_sharpness_cupy,
        center_of_brightness_cupy,
        center_of_mass_cupy,
        crop_cupy,
    )

    _CUPY_KERNELS = {
        "calculate_sharpness": calculate_sharpness_cupy,
        "crop": crop_cupy,
        "center_of_mass": center_of_mass_cupy,
        "center_of_brightness": center_of_brightness_cupy,
    }

#: Repo-root-relative artifact path; also the provenance registry key.
REPORT_KEY = "report/benchmarks/backend_kernels_report/backend_kernels_performance.md"

#: Benchmark functions in the order the original table rows appear.
ROWS = (
    "calculate_sharpness",
    "crop",
    "center_of_mass",
    "center_of_brightness",
)


def _time_mean(func, *args, n: int = 100) -> float:
    """Mean wall time over ``n`` calls (same sampler the original used)."""
    times = []
    for _ in range(n):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))


def _bench_pair(
    results: dict[str, object],
    name: str,
    numpy_func,
    numba_func,
    *args,
) -> None:
    """Warm up NumPy and Numba, then record the mean time of each."""
    numpy_func(*args)
    results[f"{name}_numpy"] = _time_mean(numpy_func, *args)
    numba_func(*args)
    results[f"{name}_numba"] = _time_mean(numba_func, *args)


def _bench_cupy(
    results: dict[str, object],
    cupy_reasons: dict[str, str],
    name: str,
    *args,
) -> None:
    """Record the CuPy timing, or the reason it is unavailable.

    ``'N/A'`` and the reason strings are byte-for-byte what the original
    recorded. Device conversion is applied only to array arguments, matching the
    original's ``cp.asarray(img)`` while leaving scalars such as ``crop``'s
    ``sample_pix`` untouched. The original warmed up only the NumPy/Numba paths,
    so no warm-up is done here either.
    """
    kernel = _CUPY_KERNELS.get(name)
    if kernel is None:
        results[f"{name}_cupy"] = "N/A"
        cupy_reasons[name] = CUPY_UNAVAILABLE_REASON
        return
    try:
        device_args = tuple(
            cp.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args
        )
        results[f"{name}_cupy"] = _time_mean(kernel, *device_args)
    except Exception as exc:  # record, never abort the report
        results[f"{name}_cupy"] = "N/A"
        cupy_reasons[name] = f"{type(exc).__name__}: {exc}"


def run_benchmark() -> tuple[dict[str, object], dict[str, str]]:
    """Benchmark the four kernels across the available backends."""
    img = np.random.rand(100, 100).astype(np.float32)
    xv, yv = np.meshgrid(np.arange(100), np.arange(100))

    results: dict[str, object] = {}
    cupy_reasons: dict[str, str] = {}

    _bench_pair(results, "calculate_sharpness", calculate_sharpness, calculate_sharpness_numba, img)
    _bench_cupy(results, cupy_reasons, "calculate_sharpness", img)

    _bench_pair(results, "crop", crop, crop_numba, img, 10)
    _bench_cupy(results, cupy_reasons, "crop", img, 10)

    # Not _bench_pair: centroid takes the image alone, center_of_mass_numba also
    # needs the coordinate grids -- no single shared arg list can serve both.
    centroid(img)
    results["center_of_mass_numpy"] = _time_mean(centroid, img)
    center_of_mass_numba(img, xv, yv)
    results["center_of_mass_numba"] = _time_mean(center_of_mass_numba, img, xv, yv)
    _bench_cupy(results, cupy_reasons, "center_of_mass", img, xv, yv)

    _bench_pair(
        results, "center_of_brightness", center_of_brightness, center_of_brightness_numba, img
    )
    _bench_cupy(results, cupy_reasons, "center_of_brightness", img)

    return results, cupy_reasons


def render_markdown(results: dict[str, object], cupy_reasons: dict[str, str]) -> str:
    """Render the comparison table; header and wording identical to the original."""
    markdown = "# Performance Comparison\n\n"
    markdown += "| Function | NumPy (s) | Numba (s) | CuPy (s) |\n"
    markdown += "|----------|-----------|-----------|----------|\n"

    for func in ROWS:
        numpy_time = results.get(f"{func}_numpy", "N/A")
        numba_time = results.get(f"{func}_numba", "N/A")
        cupy_time = results.get(f"{func}_cupy", "N/A")
        markdown += f"| {func} | {numpy_time} | {numba_time} | {cupy_time} |\n"

    markdown += "\n### CuPy availability\n\n"
    na_reasons = [
        (func, cupy_reasons.get(func, "CuPy benchmark unavailable"))
        for func in ROWS
        if results.get(f"{func}_cupy") == "N/A"
    ]
    if na_reasons:
        for func, reason in na_reasons:
            markdown += f"- `{func}`: {reason}\n"
    else:
        markdown += "- No CuPy benchmark returned N/A.\n"

    return markdown


def main() -> int:
    results, cupy_reasons = run_benchmark()
    markdown = render_markdown(results, cupy_reasons)
    logger.info(markdown)

    # Stamp the provenance header at write time: this script overwrites its own
    # report, so a header left only on disk would be deleted by the next run.
    from scripts._common.provenance import insert_header

    out_path = ROOT / REPORT_KEY
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(insert_header(markdown, REPORT_KEY), encoding="utf-8")
    logger.info(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
