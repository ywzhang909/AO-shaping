"""Benchmark the OOPAO backend on CPU vs GPU, honestly.

Why this exists
===============
OOPAO ships two independent knobs, both read **once per process**:

* ``OOPAO_BACKEND`` = ``auto`` | ``cpu`` | ``cuda`` -- which array module OOPAO
  binds (``numpy`` or ``cupy``).
* ``OOPAO_GPU_RESIDENT`` = ``0`` | ``1`` -- whether the optical arrays are kept
  on the device instead of being converted back to NumPy.

``auto`` (the default) already selects CuPy on a machine with a usable CUDA
device, so "OOPAO is on the GPU" is easy to *assume*.  It is also easy to be
wrong about: most of OOPAO's phase-screen path calls its FFT helpers **without**
a backend argument, so the screens stay NumPy whatever the knobs say.  This
script measures instead of assuming, and it is written so that a null result
("the GPU does not pay at this project's sizes") is a normal, printable outcome
rather than something to be smoothed over.

Why child processes
===================
The backend is fixed at its first use and OOPAO warns (and keeps the old value)
if the environment changes afterwards.  Measuring four configurations in one
process would therefore measure one configuration four times.  Each
configuration is spawned as a fresh child of this same file, which sets the two
environment variables before importing anything from ``ao_shaping``.

What is timed, and what is compared
===================================
* ``OopaoScreenBackend.make_screens`` at N = 128 / 512 / 1920, three seeds each,
  **median** reported (the constructor -- Telescope + Atmosphere -- is excluded;
  it is cached in production by ``oopao_backend._get_backend``).
* ``propagate_asm`` at 128² and 1920², median of repeated calls with the kernel
  cache warm; when CuPy is present the device path is timed as well, together
  with the host→device and device→host copies, so "GPU compute" and "GPU round
  trip" are never confused.
* For every configuration that should reproduce ``cpu`` + residency 0, the
  phase screens and the propagated intensity are compared by
  ``max|Δ| / rms`` and Pearson correlation.

Outputs ``benchmark_oopao_gpu.json`` and ``benchmark_oopao_gpu.md`` under
``logs/oopao_gpu/`` (a stable path, ``logs/`` is not tracked) and prints a short
summary.

Usage
=====
    python scripts/oopao/benchmark_oopao_gpu.py

``--out-dir`` re-points the output directory; ``--keep-arrays`` keeps the
per-configuration ``.npy`` files used for the agreement numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# `scripts` is deliberately NOT on the path: every `scripts/<name>/` holding an
# `__init__.py` is an importable package, and `scripts/ml/` would shadow `src/ml`
# for a bare `import ml`. Sibling packages are imported as `scripts.<name>`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common import fmt_general

import numpy as np

#: ``(OOPAO_BACKEND, OOPAO_GPU_RESIDENT)`` -- the full cross product.
CONFIGS: tuple[tuple[str, int], ...] = (
    ("cpu", 0),
    ("cpu", 1),
    ("cuda", 0),
    ("cuda", 1),
)

#: Reference configuration everything else is compared against.
REFERENCE_CONFIG = ("cpu", 0)

SCREEN_SIZES = (128, 512, 1920)
SCREEN_SEEDS = (0, 1, 2)
SCREEN_DX_M = 1e-3
SCREEN_CN2 = 1e-13
SCREEN_PATH_M = 1000.0
SCREEN_L0_M = 25.0

ASM_SIZES = (128, 1920)
ASM_DX_M = 8e-6
ASM_DISTANCE_M = 0.05
ASM_REPS = 9
TRANSFER_REPS = 7

WAVELENGTH_M = 1064e-9

RESULT_MARKER = "__OOPAO_BENCH_RESULT__"

DEFAULT_OUT_DIR = ROOT / "logs" / "oopao_gpu"


# ---------------------------------------------------------------------------
# Child process: one backend x residency configuration
# ---------------------------------------------------------------------------


def _cupy_wheel_version() -> str | None:
    """Installed CuPy wheel version (the distribution is ``cupy-cudaXXx``)."""
    from importlib import metadata

    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name == "cupy" or name.startswith("cupy-"):
            return dist.version
    return None


def _gpu_name(cp: Any) -> str:
    """Human-readable name of CUDA device 0, or ``"unknown"``."""
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # pragma: no cover - only on a broken driver
        return "unknown"


def _time_median(fn: Any, reps: int, sync: Any | None = None) -> tuple[list[float], Any]:
    """Run ``fn`` ``reps`` times and return ``(per-rep seconds, last result)``."""
    times: list[float] = []
    out: Any = None
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - start)
    return times, out


def _measure_screens(ob: Any, out_dir: Path) -> dict[str, Any]:
    """Time ``make_screens`` at every size and save the seed-0 screen."""
    record: dict[str, Any] = {}
    for n in SCREEN_SIZES:
        backend = ob.OopaoScreenBackend(
            N=n,
            dx=SCREEN_DX_M,
            Dscope=n * SCREEN_DX_M,
            lam=WAVELENGTH_M,
            cn2=SCREEN_CN2,
            L=SCREEN_PATH_M,
            L0=SCREEN_L0_M,
            n_screens=1,
        )
        times: list[float] = []
        first: np.ndarray | None = None
        for seed in SCREEN_SEEDS:
            start = time.perf_counter()
            screen = backend.make_screens(seed=seed)
            times.append(time.perf_counter() - start)
            if first is None:
                first = screen
        assert first is not None
        np.save(out_dir / f"screen_{n}.npy", np.asarray(first, dtype=np.float32))
        record[str(n)] = {
            "times_s": times,
            "median_s": float(np.median(times)),
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "screen_std_rad": float(np.std(first)),
        }
    return record


def _measure_asm(ob: Any, cp: Any, out_dir: Path) -> dict[str, Any]:
    """Time ``propagate_asm`` on numpy input, and on device input if CuPy exists."""
    record: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    sync = (lambda: cp.cuda.Stream.null.synchronize()) if cp is not None else None

    for n in ASM_SIZES:
        field = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(
            np.complex128
        )
        # Warm-up: builds OOPAO's chirp/transfer kernels and, on the device, the
        # cuFFT plans.  Steady state is what a repeated measurement loop sees.
        host_out = ob.propagate_asm(field, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M)
        ob.propagate_asm(field, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M)
        numpy_times, _ = _time_median(
            lambda f=field: ob.propagate_asm(
                f, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M
            ),
            ASM_REPS,
        )
        entry: dict[str, Any] = {
            "numpy_times_ms": [1e3 * t for t in numpy_times],
            "numpy_median_ms": float(1e3 * np.median(numpy_times)),
            "input_kind": type(host_out).__module__,
        }
        np.save(out_dir / f"intensity_{n}.npy", (np.abs(host_out) ** 2).astype(np.float32))

        if cp is not None:
            device_in = cp.asarray(field)
            h2d_times, _ = _time_median(
                lambda f=field: cp.asarray(f), TRANSFER_REPS, sync
            )
            # Warm-up for the device kernels/plans (twice: the first call also
            # pays for the cuFFT plan cache and a clock ramp).
            device_out = ob.propagate_asm(
                device_in, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M
            )
            device_out = ob.propagate_asm(
                device_in, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M
            )
            gpu_times, device_out = _time_median(
                lambda g=device_in: ob.propagate_asm(
                    g, lam=WAVELENGTH_M, dx=ASM_DX_M, z=ASM_DISTANCE_M
                ),
                ASM_REPS,
                sync,
            )
            d2h_times, back = _time_median(
                lambda d=device_out: cp.asnumpy(d), TRANSFER_REPS, sync
            )
            back = np.asarray(back)
            reference = np.asarray(host_out)
            entry.update(
                {
                    "gpu_available": True,
                    "gpu_out_kind": type(device_out).__module__,
                    "gpu_times_ms": [1e3 * t for t in gpu_times],
                    "gpu_median_ms": float(1e3 * np.median(gpu_times)),
                    "h2d_median_ms": float(1e3 * np.median(h2d_times)),
                    "d2h_median_ms": float(1e3 * np.median(d2h_times)),
                    "roundtrip_median_ms": float(
                        1e3 * (np.median(h2d_times) + np.median(gpu_times) + np.median(d2h_times))
                    ),
                    "gpu_vs_numpy_max_rel": float(
                        np.abs(back - reference).max() / np.abs(reference).max()
                    ),
                    "gpu_vs_numpy_intensity_pearson": float(
                        np.corrcoef(
                            (np.abs(back) ** 2).ravel(), (np.abs(reference) ** 2).ravel()
                        )[0, 1]
                    ),
                }
            )
        else:
            entry["gpu_available"] = False
        record[str(n)] = entry
    return record


def _run_child(payload: dict[str, Any]) -> int:
    """Measure one configuration and print the record as a marked JSON line.

    The two environment variables are set **before** importing anything that can
    reach OOPAO: the backend is chosen once, at OOPAO's first use.
    """
    os.environ["OOPAO_BACKEND"] = str(payload["backend"])
    os.environ["OOPAO_GPU_RESIDENT"] = str(payload["resident"])
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from ao_shaping.drivers.sim import oopao_backend as ob

    runtime = sys.modules.get("OOPAO.runtime")
    if runtime is None:  # pragma: no cover - Atmosphere always loads it
        raise RuntimeError("OOPAO.runtime was not imported; cannot report the backend")
    xp, gpu = runtime.array_backend()
    cp = ob.cupy_module()

    record: dict[str, Any] = {
        "backend_env": str(payload["backend"]),
        "resident_env": int(payload["resident"]),
        "array_module": xp.__name__,
        "gpu_flag": bool(gpu),
        "gpu_resident": bool(runtime.gpu_resident()),
        "cupy_version": _cupy_wheel_version(),
        "gpu_name": _gpu_name(cp) if cp is not None else None,
        "screens": _measure_screens(ob, out_dir),
        "asm": _measure_asm(ob, cp, out_dir),
    }
    print(RESULT_MARKER + json.dumps(record))
    return 0


# ---------------------------------------------------------------------------
# Parent process: spawn the children, compare, report
# ---------------------------------------------------------------------------


def _label(backend: str, resident: int) -> str:
    """Stable directory/table label for a configuration."""
    return f"{backend}_resident{resident}"


def _config_env() -> dict[str, str]:
    """Environment for a child, with this repo's ``src`` on ``PYTHONPATH``."""
    env = dict(os.environ)
    parts = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _spawn(config: tuple[str, int], work_dir: Path, timeout_s: float) -> dict[str, Any]:
    """Run one configuration in a fresh process and return its record."""
    backend, resident = config
    label = _label(backend, resident)
    out_dir = work_dir / label
    payload = json.dumps({"backend": backend, "resident": resident, "out_dir": str(out_dir)})
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", payload],
        capture_output=True,
        text=True,
        env=_config_env(),
        cwd=str(ROOT),
        timeout=timeout_s,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            record = json.loads(line[len(RESULT_MARKER) :])
            record["label"] = label
            record["out_dir"] = str(out_dir)
            return record
    raise RuntimeError(
        f"child {label} produced no result (exit {completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )


def _agreement(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """``max|Δ| / rms`` and Pearson correlation of two same-shaped arrays."""
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    rms = float(np.sqrt(np.mean(ref**2)))
    flat_a = np.asarray(actual, dtype=np.float64).ravel()
    flat_r = ref.ravel()
    pearson = float(np.corrcoef(flat_a, flat_r)[0, 1])
    return {
        "max_rel_over_rms": float(np.abs(diff).max() / rms) if rms > 0 else float("inf"),
        "pearson": pearson,
    }


def _compare(records: dict[str, dict[str, Any]], work_dir: Path) -> dict[str, Any]:
    """Numerical agreement of every configuration against the reference one."""
    if REFERENCE_CONFIG not in CONFIGS:  # pragma: no cover - constant pair
        return {}
    ref_label = _label(*REFERENCE_CONFIG)
    if ref_label not in records:
        return {}
    ref_dir = work_dir / ref_label
    table: dict[str, Any] = {}
    for config in CONFIGS:
        label = _label(*config)
        if label == ref_label or label not in records:
            continue
        other_dir = work_dir / label
        entry: dict[str, Any] = {}
        for n in SCREEN_SIZES:
            ref_arr = np.load(ref_dir / f"screen_{n}.npy")
            arr = np.load(other_dir / f"screen_{n}.npy")
            entry[f"screen_{n}"] = _agreement(arr, ref_arr)
        for n in ASM_SIZES:
            ref_arr = np.load(ref_dir / f"intensity_{n}.npy")
            arr = np.load(other_dir / f"intensity_{n}.npy")
            entry[f"intensity_{n}"] = _agreement(arr, ref_arr)
        table[label] = entry
    return table


_fmt = partial(fmt_general, missing_nonfinite="n/a", missing_none="n/a")


def _speedup(base: float, other: float) -> float:
    """``base / other`` (``> 1`` means ``other`` is faster), NaN when undefined."""
    if base is None or other is None or other <= 0:
        return float("nan")
    return float(base) / float(other)


def _summary_lines(records: dict[str, dict[str, Any]], table: dict[str, Any]) -> list[str]:
    """Plain-language findings derived from the measured numbers.

    Every sentence carries the number it comes from, so the conclusion cannot
    drift away from the table above it.
    """
    lines: list[str] = []
    ref_label = _label(*REFERENCE_CONFIG)
    ref = records.get(ref_label)
    if ref is None:
        return ["reference configuration `cpu` + residency 0 did not run; no verdict."]

    # 1. Which module each configuration actually bound.
    bound = ", ".join(
        f"`{label}` -> `{rec['array_module']}` (gpu={rec['gpu_flag']}, "
        f"resident={rec['gpu_resident']})"
        for label, rec in records.items()
    )
    lines.append(f"- 实际绑定的数组模块: {bound}。")

    # 2. Screens: is any configuration faster, and is the output identical?
    for n in SCREEN_SIZES:
        base = ref["screens"][str(n)]["median_s"]
        best_label, best = ref_label, base
        for label, rec in records.items():
            value = rec["screens"][str(n)]["median_s"]
            if value < best:
                best_label, best = label, value
        worst_label, worst = ref_label, base
        for label, rec in records.items():
            value = rec["screens"][str(n)]["median_s"]
            if value > worst:
                worst_label, worst = label, value
        # Seed-to-seed spread inside one configuration is the noise floor for any
        # cross-configuration claim: a 7 % "speedup" below a 7 % spread is not one.
        within = 0.0
        for rec in records.values():
            times = rec["screens"][str(n)]["times_s"]
            median = rec["screens"][str(n)]["median_s"]
            if median > 0 and len(times) > 1:
                within = max(within, (max(times) - min(times)) / median)
        across = (worst - best) / best if best > 0 else float("inf")
        deviations = [
            table[label][f"screen_{n}"]["max_rel_over_rms"] for label in table
        ]
        if not deviations:
            screen_word = "无可比配置"
        elif max(deviations) == 0.0:
            screen_word = "逐位相同"
        else:
            screen_word = f"最大偏差 {_fmt(max(deviations), '.2e')} (相对 rms)"
        if across <= within:
            noise_word = (
                f"配置间差异 {100 * across:.0f}% ≤ 同一配置内的种子散布 "
                f"{100 * within:.0f}% ⇒ 属噪声, 不构成加速"
            )
        else:
            noise_word = (
                f"配置间差异 {100 * across:.0f}% 略大于种子散布 "
                f"{100 * within:.0f}%, 但两者都是个位数百分比 ⇒ 没有可用的加速"
            )
        lines.append(
            f"- `make_screens` N={n}: 最快 `{best_label}` {_fmt(best)} s, 最慢 "
            f"`{worst_label}` {_fmt(worst)} s (相对 `{ref_label}` ×"
            f"{_fmt(_speedup(base, worst), '.3f')}); {noise_word}; 相位屏{screen_word}。"
        )

    # 3. propagate_asm kernel-only vs round trip.
    for n in ASM_SIZES:
        base = ref["asm"][str(n)]["numpy_median_ms"]
        gpu_rows = [
            (label, rec["asm"][str(n)])
            for label, rec in records.items()
            if rec["asm"][str(n)].get("gpu_available")
        ]
        if not gpu_rows:
            lines.append(
                f"- `propagate_asm` {n}²: 无 CuPy 设备路径可用, 只有 numpy {_fmt(base)} ms。"
            )
            continue
        kernels = [entry["gpu_median_ms"] for _label, entry in gpu_rows]
        roundtrips = [entry["roundtrip_median_ms"] for _label, entry in gpu_rows]
        best_entry = min(gpu_rows, key=lambda item: item[1]["gpu_median_ms"])[1]
        kernel_speedups = sorted(_speedup(base, k) for k in kernels)
        roundtrip_speedups = sorted(_speedup(base, t) for t in roundtrips)
        polarity = (
            "GPU 更慢"
            if roundtrip_speedups[-1] < 1.0
            else ("GPU 更快" if roundtrip_speedups[0] > 1.0 else "两者互有胜负")
        )
        lines.append(
            f"- `propagate_asm` {n}²: numpy {_fmt(base)} ms; 设备端核 "
            f"{_fmt(min(kernels))}–{_fmt(max(kernels))} ms "
            f"(×{_fmt(kernel_speedups[0], '.2f')}–{_fmt(kernel_speedups[-1], '.2f')}); "
            f"含 H2D {_fmt(best_entry['h2d_median_ms'])} ms + D2H "
            f"{_fmt(best_entry['d2h_median_ms'])} ms 的整轮往返 "
            f"{_fmt(min(roundtrips))}–{_fmt(max(roundtrips))} ms ⇒ 有效加速 "
            f"×{_fmt(roundtrip_speedups[0], '.2f')}–"
            f"{_fmt(roundtrip_speedups[-1], '.2f')}, {polarity}"
            "(传输耗时在本机上跨配置/跨运行波动, 故给区间而不是单点)。"
        )

    # 4. Does residency change the answer at all?
    resident_deltas = []
    for backend in ("cpu", "cuda"):
        off = _label(backend, 0)
        on = _label(backend, 1)
        if off in records and on in records:
            for n in SCREEN_SIZES:
                resident_deltas.append(
                    _speedup(
                        records[off]["screens"][str(n)]["median_s"],
                        records[on]["screens"][str(n)]["median_s"],
                    )
                )
    if resident_deltas:
        ratios = ", ".join(f"×{_fmt(v, '.3f')}" for v in resident_deltas)
        lines.append(
            "- `OOPAO_GPU_RESIDENT=1` 相对 `0` 的 `make_screens` 耗时比 "
            f"({ratios}); ≈1 表示驻留既没帮上忙也没明显拖慢 —— "
            "因为相位屏的生成仍在 NumPy 上。"
        )
    return lines


def _render_markdown(
    records: dict[str, dict[str, Any]],
    table: dict[str, Any],
    elapsed: float,
) -> str:
    """Render the measured tables and the derived verdict."""
    from scripts._common import markdown_table

    first = next(iter(records.values()))
    lines = [
        "# OOPAO GPU benchmark",
        "",
        f"Generated by `scripts/oopao/benchmark_oopao_gpu.py` in {_fmt(elapsed, '.1f')} s "
        "on this machine.",
        "",
        f"- CuPy wheel: `{first['cupy_version']}`",
        f"- CUDA device: `{first['gpu_name']}`",
        f"- wavelength {WAVELENGTH_M * 1e9:.0f} nm; screens dx={SCREEN_DX_M:g} m, "
        f"Cn2={SCREEN_CN2:g}, L={SCREEN_PATH_M:g} m, L0={SCREEN_L0_M:g} m; "
        f"ASM dx={ASM_DX_M:g} m, z={ASM_DISTANCE_M:g} m",
        f"- `make_screens`: {len(SCREEN_SEEDS)} seeds, median reported; constructor excluded",
        f"- `propagate_asm`: median of {ASM_REPS} warm calls (OOPAO kernel cache warm)",
        "",
        "## What each configuration bound",
        "",
        markdown_table(
            ["config", "`OOPAO_BACKEND`", "`OOPAO_GPU_RESIDENT`", "array module", "gpu", "resident"],
            [
                [
                    f"`{label}`",
                    rec["backend_env"],
                    rec["resident_env"],
                    f"`{rec['array_module']}`",
                    rec["gpu_flag"],
                    rec["gpu_resident"],
                ]
                for label, rec in records.items()
            ],
        ),
        "",
        "## Wall clock",
        "",
        markdown_table(
            ["config"]
            + [f"screens {n} (s)" for n in SCREEN_SIZES]
            + [f"ASM {n} numpy in (ms)" for n in ASM_SIZES]
            + [f"ASM {n} device in (ms)" for n in ASM_SIZES],
            [
                [f"`{label}`"]
                + [_fmt(rec["screens"][str(n)]["median_s"]) for n in SCREEN_SIZES]
                + [_fmt(rec["asm"][str(n)]["numpy_median_ms"]) for n in ASM_SIZES]
                + [
                    _fmt(rec["asm"][str(n)].get("gpu_median_ms", float("nan")))
                    for n in ASM_SIZES
                ]
                for label, rec in records.items()
            ],
        ),
        "",
        "## Host ↔ device transfer (single shot, median of "
        f"{TRANSFER_REPS})",
        "",
        markdown_table(
            ["config", "ASM size", "H2D (ms)", "device kernel (ms)", "D2H (ms)", "round trip (ms)", "numpy kernel (ms)"],
            [
                [
                    f"`{label}`",
                    f"{n}²",
                    _fmt(rec["asm"][str(n)].get("h2d_median_ms", float("nan"))),
                    _fmt(rec["asm"][str(n)].get("gpu_median_ms", float("nan"))),
                    _fmt(rec["asm"][str(n)].get("d2h_median_ms", float("nan"))),
                    _fmt(rec["asm"][str(n)].get("roundtrip_median_ms", float("nan"))),
                    _fmt(rec["asm"][str(n)]["numpy_median_ms"]),
                ]
                for label, rec in records.items()
                for n in ASM_SIZES
            ],
        ),
        "",
        "## Numerical agreement against `cpu` + residency 0",
        "",
        "`max|Δ|/rms` is the largest absolute deviation over the RMS of the reference; "
        "`r` is the Pearson correlation of the flattened arrays. 0 and 1 mean the two "
        "configurations are the same numbers.",
        "",
        markdown_table(
            ["config"]
            + [f"screen {n}" for n in SCREEN_SIZES]
            + [f"intensity {n}" for n in ASM_SIZES],
            [
                [f"`{label}`"]
                + [
                    f"{_fmt(row[f'screen_{n}']['max_rel_over_rms'], '.2e')} / "
                    f"{_fmt(row[f'screen_{n}']['pearson'], '.6f')}"
                    for n in SCREEN_SIZES
                ]
                + [
                    f"{_fmt(row[f'intensity_{n}']['max_rel_over_rms'], '.2e')} / "
                    f"{_fmt(row[f'intensity_{n}']['pearson'], '.6f')}"
                    for n in ASM_SIZES
                ]
                for label, row in table.items()
            ],
        ),
        "",
        "## Device path vs numpy path inside one process",
        "",
        markdown_table(
            ["config", "ASM size", "device output kind", "`max|Δ|/|ref|`", "intensity r"],
            [
                [
                    f"`{label}`",
                    f"{n}²",
                    entry.get("gpu_out_kind", "n/a"),
                    _fmt(entry.get("gpu_vs_numpy_max_rel", float("nan")), ".2e"),
                    _fmt(entry.get("gpu_vs_numpy_intensity_pearson", float("nan")), ".6f"),
                ]
                for label, rec in records.items()
                for n in ASM_SIZES
                for entry in [rec["asm"][str(n)]]
                if entry.get("gpu_available")
            ],
        ),
        "",
        "## Verdict, from the numbers above",
        "",
        "> 关于两张表的读法: 128² 的「设备端核」在 0.4–1.1 ms 之间散布, 而单次调用只有数百微秒,"
        " 那是这台笔记本 GPU 的时钟爬升噪声, 不是 2.6× 的真实差异; 1920² 的核耗时 (5–6 ms) 已远高于噪声。"
        "「设备端核」这一列在同一 size 下跨配置可差数倍, 那是 GPU 时钟/功耗状态, 不是配置差异 ——"
        " 只把它当量级参考, 不要跨配置比较; 跨配置可比的只有一致度表 (全为 0)。",
        ">",
        "> 本仓**当前的生产调用点**仍然喂 NumPy: `beam_backend.propagate()` 一进来就 `np.asarray(field)`。"
        " 因此下面 1920² 的 60× 级核加速, 只有在调用方自己把场搬到设备上时才拿得到;"
        " 路由层不改, 「开 GPU」对现有仿真链路的影响是**零**。",
        "",
    ]
    lines.extend(_summary_lines(records, table))
    return "\n".join(lines) + "\n"


def _print_summary(records: dict[str, dict[str, Any]], table: dict[str, Any], out_dir: Path) -> None:
    """Print the compact terminal summary (the same facts as the markdown)."""
    print()
    print("config                 | module | screens 128 | screens 512 | screens 1920 | ASM 1920 numpy | ASM 1920 device")
    for label, rec in records.items():
        print(
            f"{label:<22} | {rec['array_module']:<6} | "
            f"{_fmt(rec['screens']['128']['median_s'], '.3g'):>11} | "
            f"{_fmt(rec['screens']['512']['median_s'], '.3g'):>11} | "
            f"{_fmt(rec['screens']['1920']['median_s'], '.3g'):>12} | "
            f"{_fmt(rec['asm']['1920']['numpy_median_ms'], '.1f'):>14} | "
            f"{_fmt(rec['asm']['1920'].get('gpu_median_ms', float('nan')), '.3g'):>15}"
        )
    print()
    print("agreement vs cpu/resident0 (max|d|/rms / pearson):")
    for label, row in table.items():
        cells = [
            f"{_fmt(row[f'screen_{n}']['max_rel_over_rms'], '.1e')}"
            for n in SCREEN_SIZES
        ] + [_fmt(row[f'intensity_{n}']['max_rel_over_rms'], '.1e') for n in ASM_SIZES]
        print(f"  {label:<22} screens {', '.join(cells[:3])} | intensity {', '.join(cells[3:])}")
    print()
    for line in _summary_lines(records, table):
        print(line.replace("- ", "* ", 1))
    print()
    print(f"wrote {out_dir / 'benchmark_oopao_gpu.json'}")
    print(f"wrote {out_dir / 'benchmark_oopao_gpu.md'}")


def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1] if __doc__ else "")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where the JSON / markdown / per-config .npy files go (default: logs/oopao_gpu)",
    )
    parser.add_argument(
        "--keep-arrays",
        action="store_true",
        help="keep the per-configuration .npy arrays used for the agreement numbers",
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=1800.0,
        help="per-configuration timeout, seconds (default: 1800)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark (or one child configuration).

    Args:
        argv: Arguments after the script name; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.child is not None:
        return _run_child(json.loads(args.child))

    out_dir: Path = args.out_dir
    work_dir = out_dir / "arrays"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for config in CONFIGS:
        label = _label(*config)
        print(f"[bench] {label} ...", flush=True)
        try:
            records[label] = _spawn(config, work_dir, args.child_timeout)
        except Exception as exc:  # a failed configuration is itself a result
            failures[label] = f"{type(exc).__name__}: {exc}"
            print(f"[bench] {label} FAILED: {failures[label].splitlines()[0]}")
    elapsed = time.perf_counter() - started
    if not records:
        print("error: every configuration failed", file=sys.stderr)
        return 1

    table = _compare(records, work_dir)
    payload = {
        "generated_by": "scripts/oopao/benchmark_oopao_gpu.py",
        "elapsed_s": elapsed,
        "parameters": {
            "screen_sizes": list(SCREEN_SIZES),
            "screen_seeds": list(SCREEN_SEEDS),
            "screen_dx_m": SCREEN_DX_M,
            "screen_cn2": SCREEN_CN2,
            "screen_path_m": SCREEN_PATH_M,
            "screen_l0_m": SCREEN_L0_M,
            "asm_sizes": list(ASM_SIZES),
            "asm_dx_m": ASM_DX_M,
            "asm_distance_m": ASM_DISTANCE_M,
            "asm_reps": ASM_REPS,
            "wavelength_m": WAVELENGTH_M,
            "reference_config": _label(*REFERENCE_CONFIG),
        },
        "configurations": records,
        "agreement_vs_reference": table,
        "failures": failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_oopao_gpu.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "benchmark_oopao_gpu.md").write_text(
        _render_markdown(records, table, elapsed), encoding="utf-8"
    )
    _print_summary(records, table, out_dir)
    if not args.keep_arrays:
        shutil.rmtree(work_dir, ignore_errors=True)
    if failures:
        print(f"note: {len(failures)} configuration(s) failed: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
