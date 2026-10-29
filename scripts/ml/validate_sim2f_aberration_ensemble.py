"""Validate the shipped sim2f aberration model against OOPAO at high N.

What is being validated
=======================
``ml.sim2f`` draws high-order Zernike coefficients from a von Karman covariance
asset, ``src/ml/sim2f/assets/zernike_cov_kolmogorov_n21.npz`` (Noll 4..22, at the
bench geometry: ``grid=128``, ``pupil_radius_px=48``).  That asset is **analytic**
-- OOPAO's closed-form von Karman phase covariance, projected onto the model's own
Zernike basis by ``scripts/ml/build_sim2f_aberration_covariance.py``.  Nothing in
the repository has ever checked it against the *stochastic* thing it claims to
describe, i.e. an actual ensemble of OOPAO phase screens.

This script does that check, twice:

1. **The generator.**  Draw ``N`` OOPAO phase screens (``Atmosphere.
   generateNewPhaseScreen``, the same generator ``OopaoScreenBackend`` uses),
   project each onto the canonical basis (``Sim2f.zernike_basis`` ->
   ``ao_shaping.utils.wavefront.zernike_calc``; no second Zernike
   implementation), and accumulate the empirical coefficient covariance.
2. **The shipped sampler.**  Draw ``M`` coefficient vectors from
   ``ml.sim2f.optics.draw_aberration(..., aberration_model="kolmogorov")`` and
   compare *their* sample covariance with the same asset.

How the comparison is scored -- pre-registered
==============================================
Written down **before** the numbers are looked at, and printed first so it cannot
be retrofitted:

* Every entry is compared on the correlation scale,
  ``e_ij = |S_emp,ij - S_ref,ij| / sqrt(S_ref,ii * S_ref,jj)``, which is
  dimensionless and stable where a raw relative error divides by ~0.
* The sampling-noise floor of a covariance estimated from ``n`` samples is
  ``sqrt(2/n)`` per entry (the standard error of a second moment).  With
  ``n = 2000`` that is 0.032.
* **Accept = "consistent within sampling noise"** iff the RMS of ``e`` over all
  ``K*K`` entries is ``<= sqrt(2/n)``.  Anything else is reported as
  **inconsistent**, with the ratio to the floor.  The phrasing "matches" is
  reserved for the former; a per-entry scatter at the floor is not a match, it is
  agreement to within the noise of the estimator.
* The per-mode variance ladder is reported separately (ratio ``emp/ref`` per
  Noll mode) because a global scale error can hide inside a good off-diagonal
  correlation, and vice versa.

For the *sampler* arm there is a known, documented deviation that this script
must not paper over: ``draw_aberration`` rescales every draw to a uniform
``rms_rad``, and an anisotropic ensemble squeezed onto a sphere loses anisotropy
(Bingham).  ``src/ml/sim2f/optics.py`` measures that as ~0.85x on the low orders
and ~1.27x on the highest mode.  So the sampler's *diagonal* is expected to fail
the noise-floor test on purpose; the script reports the realised ratios next to
those documented factors instead of excusing them.

Outputs ``validate_sim2f_aberration_ensemble.json`` and ``.md`` under
``logs/oopao_gpu/`` (stable path, ``logs/`` is untracked) and prints a short
summary including the backend OOPAO actually bound and the wall clock.

Usage
=====
    python scripts/ml/validate_sim2f_aberration_ensemble.py --n-samples 2000
"""

from __future__ import annotations

import argparse
from loguru import logger
import contextlib
import io
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Order matters: `src` before `scripts`, and `scripts` is deliberately NOT on the
# path at all. `scripts/ml/` is an importable package (`scripts/__init__.py` exists),
# so putting `scripts` on `sys.path` makes a bare `import ml` resolve to
# `scripts/ml` instead of `src/ml` -- the documented subpackage-shadowing trap.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ml.sim2f.config import AberrationConfig
from ml.sim2f.optics import DEFAULT_COVARIANCE_PATH, draw_aberration
from scripts._common import fmt_general
from scripts.ml.build_sim2f_aberration_covariance import (
    _build_bench,
    _grid_span_m,
    _pupil_geometry,
    _roundtrip_error,
    _shim_atmosphere,
)

#: Default output directory (untracked; ``logs/`` is gitignored).
DEFAULT_OUT_DIR = ROOT / "logs" / "oopao_gpu"

#: Printed before any number is computed.  Do not move it below the measurements.
PRE_REGISTRATION = """\
预登记判据 (在看数字之前写下, 之后不得修改):
  1. 逐条误差 e_ij = |S_emp,ij - S_ref,ij| / sqrt(S_ref,ii * S_ref,jj)  —— 相关尺度, 无量纲。
  2. 采样噪声地板 = sqrt(2/n) (二阶矩估计的标准误)。
  3. 判定 "在采样噪声内一致" 当且仅当 RMS(e) <= sqrt(2/n);
     否则记为 "不一致", 并给出相对地板的倍数。只有前者可以说 "一致",
     逐条散布贴着地板时只能说 "在估计器噪声内吻合", 不能说 "匹配"。
  4. 对角线 (逐模方差) 单独报告 emp/ref 比值, 因为整体尺度误差可以藏在一个
     很好的非对角相关里, 反之亦然。
  5. 变体 2 (出厂采样器 `draw_aberration`) 的**绝对幅度**由 `rms_rad` 旋钮 (0.05-0.45 rad)
     决定 —— 那是有意的难度旋钮 (r0=0.15 m 在 5.4 mm 瞳面上几乎无像差, 资产本身的
     rad² 只有 ~1e-3), 不是待验证的断言。故变体 2 先按其平均对角把幅度对齐, 只验证
     **形状** (逐模相对方差与非对角相关), 并把幅度系数单独报出来。变体 1 不做此对齐。
  6. 该判据对 n 敏感: 任何固定大小的**系统**残差最终都会超过随 n 收缩的地板。
     因此 "在噪声内" 只意味着 "在当前 n 下不可分辨", 不意味着残差为零; 报告一并给出
     中位误差、超地板条目的占比与误差最集中的模式, 以便区分 "整体尺度错" 与
     "少数模式错"。
"""


# ---------------------------------------------------------------------------
# Reference asset and bench
# ---------------------------------------------------------------------------


def _reference() -> dict[str, Any]:
    """The committed covariance plus the geometry it was built for.

    Geometry comes from the provenance JSON rather than from a hard-coded default,
    so the bench used here is the one the asset describes (and a mismatch is
    visible instead of silently compared against the wrong pupil).

    Returns:
        ``{"covariance", "noll", "grid", "pupil_radius_px", "pupil_pitch_m",
        "aperture_m", "n_modes", "wavelength_m", "asset"}``.
    """
    provenance_path = DEFAULT_COVARIANCE_PATH.with_suffix(".json")
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    with np.load(DEFAULT_COVARIANCE_PATH) as data:
        covariance = np.asarray(data["covariance"], dtype=np.float64)
        noll = np.asarray(data["noll"], dtype=np.int64)
        wavelength_m = float(data["wavelength_m"])
    return {
        "covariance": covariance,
        "noll": noll,
        "grid": int(record["sim2f"]["grid"]),
        "pupil_radius_px": float(record["sim2f"]["pupil_radius_px"]),
        "pupil_pitch_m": float(record["sim2f"]["pupil_pitch_m"]),
        "aperture_m": float(record["sim2f"]["aperture_m"]),
        "n_modes": int(record["sim2f"]["n_modes"]),
        "wavelength_m": wavelength_m,
        "asset": record["asset"],
        "sha256": record["covariance_sha256"],
        "turbulence": record["turbulence"],
    }


def _bench(ref: dict[str, Any]) -> Any:
    """The ``Sim2f`` bench at the geometry the asset was built for.

    Reuses the builder's own :func:`_build_bench` so the pupil, the basis and the
    projection matrix are *the same objects* the asset came from; a second copy of
    that geometry is exactly how a validation starts validating the wrong thing.
    """
    args = argparse.Namespace(
        grid=ref["grid"],
        pupil_radius_px=ref["pupil_radius_px"],
        pupil_pitch_m=ref["pupil_pitch_m"],
    )
    return _build_bench(args)


def _backend_record() -> dict[str, Any]:
    """The array backend OOPAO actually bound in this process (see the builder)."""
    record: dict[str, Any] = {
        "oopao_backend_env": os.environ.get("OOPAO_BACKEND", "<unset = auto>"),
        "oopao_gpu_resident_env": os.environ.get("OOPAO_GPU_RESIDENT", "<unset = 0>"),
        "array_module": "unknown",
        "gpu": None,
        "gpu_resident": None,
    }
    runtime = sys.modules.get("OOPAO.runtime")
    if runtime is None:  # pragma: no cover - _oopao_compat imports it
        return record
    xp, gpu = runtime.array_backend()
    record["array_module"] = xp.__name__
    record["gpu"] = bool(gpu)
    record["gpu_resident"] = bool(runtime.gpu_resident())
    return record


# ---------------------------------------------------------------------------
# Two estimators of the same thing
# ---------------------------------------------------------------------------


def oopao_ensemble(
    sim: Any, ref: dict[str, Any], *, n_samples: int, seed: int
) -> np.ndarray:
    """Coefficient covariance of ``n_samples`` OOPAO phase screens, ``rad**2``.

    Screen ``i`` uses ``seed + i`` (the same convention as the builder's sampled
    estimator), is cropped from OOPAO's ``N+4`` layer to ``N``, projected with the
    model's own least-squares projector, and has its mean removed -- i.e. piston
    is removed twice over, exactly as the analytic arm does.

    The phase itself is fetched through ``_oopao_compat.as_numpy``: with
    ``OOPAO_GPU_RESIDENT=1`` the layer array can live on the device, and
    ``np.asarray`` on a CuPy array raises.

    Args:
        sim: The bench (defines support and basis).
        ref: The loaded asset record (polarity of the wavelength factor only).
        n_samples: Number of independent screens.
        seed: Seed of the first screen.

    Returns:
        ``(K, K)`` covariance at the asset's wavelength, ``rad**2``.
    """
    from ao_shaping.drivers.sim._oopao_compat import as_numpy

    n_modes = int(ref["n_modes"])
    _coords, basis, _noll, support = _pupil_geometry(sim, n_modes)
    atm = _shim_atmosphere(
        # The grid span, NOT the disc diameter: OOPAO's layer pitch is
        # ``aperture/grid`` and the screens are indexed with the model's grid, so
        # the disc diameter would sample them 4/3 finer and halve every variance.
        grid=int(ref["grid"]),
        aperture_m=_grid_span_m(sim),
        r0_m=float(ref["turbulence"]["r0_m_at_500nm"]),
        l0_m=float(ref["turbulence"]["L0_m"]),
    )
    projector = np.linalg.pinv(basis)

    with contextlib.redirect_stdout(io.StringIO()):
        atm.initializeAtmosphere(atm.telescope, compute_covariance=False)

    acc = np.zeros((basis.shape[1], basis.shape[1]), dtype=np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(int(n_samples)):
            atm.generateNewPhaseScreen(seed=int(seed) + i)
            phase = as_numpy(atm.layer_1.OPD)[2:-2, 2:-2]
            values = np.asarray(phase, dtype=np.float64)[support]
            coeffs = projector @ (values - values.mean())
            acc += np.outer(coeffs, coeffs)
    sigma = acc / float(n_samples) * (
        float(ref["turbulence"]["oopao_reference_wavelength_m"])
        / float(ref["wavelength_m"])
    ) ** 2
    return 0.5 * (sigma + sigma.T)


def sampler_ensemble(
    sim: Any, ref: dict[str, Any], *, n_draws: int, seed: int
) -> np.ndarray:
    """Coefficient covariance of ``n_draws`` draws from the shipped sampler.

    Args:
        sim: The bench.
        ref: The loaded asset record (``n_modes`` only).
        n_draws: Number of independent draws.
        seed: RNG seed.

    Returns:
        ``(K, K)`` sample covariance, ``rad**2``.
    """
    cfg = AberrationConfig(
        n_modes=int(ref["n_modes"]),
        aberration_model="kolmogorov",
        covariance_path=str(DEFAULT_COVARIANCE_PATH),
    )
    rng = np.random.default_rng(int(seed))
    k = int(ref["n_modes"]) - 2
    acc = np.zeros((k, k), dtype=np.float64)
    for _ in range(int(n_draws)):
        state = draw_aberration(sim, cfg, rng)
        high = np.asarray(state.coeffs[2:], dtype=np.float64)
        acc += np.outer(high, high)
    sigma = acc / float(n_draws)
    return 0.5 * (sigma + sigma.T)


# ---------------------------------------------------------------------------
# Scoring (pre-registered)
# ---------------------------------------------------------------------------


def compare(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    n_samples: int,
    match_amplitude: bool = False,
) -> dict[str, Any]:
    """Score one covariance estimate against the asset.

    Args:
        estimate: ``(K, K)`` sample covariance.
        reference: ``(K, K)`` asset covariance.
        n_samples: Samples behind ``estimate``; sets the noise floor.
        match_amplitude: Divide ``estimate`` by its overall amplitude ratio first,
            so only the **shape** is scored. Arm 2 needs this: ``draw_aberration``
            renormalises every draw to ``rms_rad`` (0.05-0.45 rad), which is a
            deliberate difficulty knob and therefore ~1e4x the asset's absolute
            rad**2 (r0=0.15 m over a 5.4 mm disc is almost aberration-free). The
            amplitude factor is reported either way, so nothing is hidden.

    Returns:
        Dict with the diagonal ladder, the entry-error distribution, the noise
        floor and the pre-registered verdict.
    """
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    diagonal_ref = np.diag(ref)
    amplitude_scale = float(np.mean(np.diag(est)) / np.mean(diagonal_ref))
    if match_amplitude:
        est = est / amplitude_scale

    diag_ref = np.diag(ref)
    diag_est = np.diag(est)
    scale = np.sqrt(np.outer(diag_ref, diag_ref))
    error = np.abs(est - ref) / scale

    corr_ref = ref / scale
    corr_est = est / scale
    floor = float(np.sqrt(2.0 / float(n_samples)))
    rms_error = float(np.sqrt(np.mean(error**2)))

    off = ~np.eye(ref.shape[0], dtype=bool)
    squared = (error**2).ravel()
    order = np.argsort(squared)[::-1]
    total_squared = float(squared.sum()) if squared.sum() > 0 else 1.0
    return {
        "n_samples": int(n_samples),
        "noise_floor": floor,
        "match_amplitude": bool(match_amplitude),
        "amplitude_scale": amplitude_scale,
        "amplitude_within_floor": bool(abs(amplitude_scale - 1.0) <= floor),
        "entry_error_rms": rms_error,
        "entry_error_median": float(np.median(error)),
        "entry_error_p95": float(np.percentile(error, 95)),
        "entry_error_max": float(error.max()),
        "entry_error_max_at": [int(v) for v in np.unravel_index(np.argmax(error), error.shape)],
        "error_share_top1": float(squared[order[0]] / total_squared),
        "error_share_top2": float((squared[order[0]] + squared[order[1]]) / total_squared),
        "fraction_within_floor": float(np.mean(error <= floor)),
        "rms_over_floor": rms_error / floor,
        "verdict": "consistent within sampling noise" if rms_error <= floor else "inconsistent",
        "diagonal_ratio": (diag_est / diag_ref).tolist(),
        "diagonal_rel_error": (np.abs(diag_est - diag_ref) / diag_ref).tolist(),
        "diagonal_within_floor": [
            bool(v <= floor) for v in np.abs(diag_est - diag_ref) / diag_ref
        ],
        "offdiag_corr_rms_error": float(np.sqrt(np.mean((corr_est[off] - corr_ref[off]) ** 2))),
        "corr_ref": corr_ref,
        "corr_est": corr_est,
        "diagonal_ref": diag_ref,
        "diagonal_est": diag_est,
    }


def _worst_rows(
    result: dict[str, Any], noll: np.ndarray, count: int = 5
) -> list[list[str]]:
    """The ``count`` modes farthest from the asset, as markdown rows.

    A residual that is a *few* modes is a different finding from one that is a
    global scale error, and the entry-error RMS alone cannot tell them apart -- so
    the modes carrying the deviation are always printed next to the verdict.
    """
    ratios = np.asarray(result["diagonal_ratio"], dtype=np.float64)
    rel = np.asarray(result["diagonal_rel_error"], dtype=np.float64)
    order = np.argsort(-rel)[:count]
    return [
        [
            str(int(noll[index])),
            _fmt(result["diagonal_ref"][index], ".4e"),
            _fmt(result["diagonal_est"][index], ".4e"),
            _fmt(ratios[index], ".3f"),
            _fmt(rel[index], ".3f"),
            "yes" if result["diagonal_within_floor"][index] else "**no**",
        ]
        for index in order
    ]


def _strongest_pairs(result: dict[str, Any], noll: np.ndarray, count: int = 12) -> list[dict[str, Any]]:
    """The ``count`` strongest reference off-diagonal correlations, with both sides.

    Args:
        result: Output of :func:`compare`.
        noll: Noll indices per mode.
        count: How many pairs.

    Returns:
        List of ``{"noll_i", "noll_j", "corr_ref", "corr_est"}`` records.
    """
    corr_ref = result["corr_ref"]
    corr_est = result["corr_est"]
    k = corr_ref.shape[0]
    pairs = [
        (abs(corr_ref[i, j]), i, j)
        for i in range(k)
        for j in range(i + 1, k)
    ]
    pairs.sort(reverse=True)
    return [
        {
            "noll_i": int(noll[i]),
            "noll_j": int(noll[j]),
            "corr_ref": float(corr_ref[i, j]),
            "corr_est": float(corr_est[i, j]),
        }
        for _abs, i, j in pairs[:count]
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_fmt = partial(fmt_general, missing_nonfinite="n/a")


def _ladder_rows(result: dict[str, Any], noll: np.ndarray) -> list[list[str]]:
    """Markdown rows of the per-mode variance ladder."""
    rows = []
    for index, mode in enumerate(noll):
        rows.append(
            [
                str(int(mode)),
                _fmt(result["diagonal_ref"][index], ".4e"),
                _fmt(result["diagonal_est"][index], ".4e"),
                _fmt(result["diagonal_ratio"][index], ".3f"),
                _fmt(result["diagonal_rel_error"][index], ".3f"),
                "yes" if result["diagonal_within_floor"][index] else "**no**",
            ]
        )
    return rows


def _render_markdown(
    ref: dict[str, Any],
    backend: dict[str, Any],
    oopao_result: dict[str, Any],
    oopao_pairs: list[dict[str, Any]],
    sampler_result: dict[str, Any],
    sampler_pairs: list[dict[str, Any]],
    timings: dict[str, float],
    config: dict[str, Any],
) -> str:
    """Render the validation report."""
    from scripts._common import markdown_table

    noll = ref["noll"]
    lines = [
        "# sim2f aberration model vs OOPAO ensemble",
        "",
        "Generated by `scripts/ml/validate_sim2f_aberration_ensemble.py`.",
        "",
        "## Pre-registered acceptance wording (written before the numbers)",
        "",
        "```",
        PRE_REGISTRATION.rstrip(),
        "```",
        "",
        "## What was validated",
        "",
        f"- Asset: `src/ml/sim2f/assets/{ref['asset']}` (sha256 `{ref['sha256'][:16]}…`), "
        f"K={noll.size} modes, Noll {int(noll[0])}..{int(noll[-1])}, "
        f"grid={ref['grid']}, pupil_radius_px={ref['pupil_radius_px']:g}, "
        f"pupil_pitch={ref['pupil_pitch_m']:g} m",
        f"- Turbulence: r0={ref['turbulence']['r0_m_at_500nm']} m at 500 nm, "
        f"L0={ref['turbulence']['L0_m']} m; covariance expressed at "
        f"{1e9 * ref['wavelength_m']:.0f} nm (global scalar (500/λ)²)",
        f"- OOPAO screens: {config['n_samples']} draws, seeds "
        f"{config['seed']}..{config['seed'] + config['n_samples'] - 1}, on a pupil of "
        f"diameter {config['oopao_pupil_diameter_m'] * 1e3:.4g} mm = grid × pitch (so "
        "the screen pixel pitch equals the model's; using the disc diameter instead "
        "samples them 4/3 finer and lowers every variance by 0.62×)",
        f"- Sampler draws: {config['n_draws']} (`draw_aberration`, "
        "`aberration_model=\"kolmogorov\"`), seed "
        f"{config['seed']}",
        "",
        "## Backend actually used and wall clock",
        "",
        markdown_table(
            ["field", "value"],
            [
                ["array module", f"`{backend['array_module']}`"],
                ["gpu", str(backend["gpu"])],
                ["`OOPAO_GPU_RESIDENT`", str(backend["gpu_resident"])],
                ["`OOPAO_BACKEND`", str(backend["oopao_backend_env"])],
                ["OOPAO screens (s)", _fmt(timings["oopao_s"], ".2f")],
                ["sampler draws (s)", _fmt(timings["sampler_s"], ".2f")],
                ["total (s)", _fmt(timings["total_s"], ".2f")],
            ],
        ),
        "",
        "## Arm 1 — OOPAO phase-screen ensemble vs the analytic asset",
        "",
        f"**Verdict: {oopao_result['verdict']}** — entry-error RMS "
        f"{_fmt(oopao_result['entry_error_rms'], '.4f')} vs noise floor "
        f"sqrt(2/{oopao_result['n_samples']}) = {_fmt(oopao_result['noise_floor'], '.4f')} "
        f"({_fmt(oopao_result['rms_over_floor'], '.2f')}× the floor); "
        f"{100 * oopao_result['fraction_within_floor']:.0f}% of entries inside the floor; "
        f"amplitude {_fmt(oopao_result['amplitude_scale'], '.4f')} "
        f"({'within' if oopao_result['amplitude_within_floor'] else 'outside'} the floor).",
        "",
        "Entry error (correlation scale): median "
        f"{_fmt(oopao_result['entry_error_median'], '.4f')}, p95 "
        f"{_fmt(oopao_result['entry_error_p95'], '.4f')}, max "
        f"{_fmt(oopao_result['entry_error_max'], '.4f')} at mode index "
        f"{tuple(oopao_result['entry_error_max_at'])}; off-diagonal correlation RMS error "
        f"{_fmt(oopao_result['offdiag_corr_rms_error'], '.4f')}. "
        f"The two largest entries carry {100 * oopao_result['error_share_top2']:.0f}% of "
        "the squared error, so read the worst-modes table before calling this a "
        "global mismatch.",
        "",
        "Per-mode variance ladder (rad²):",
        "",
        markdown_table(
            ["Noll", "asset", "empirical", "emp/asset", "rel. error", "within floor"],
            _ladder_rows(oopao_result, noll),
        ),
        "",
        "Worst modes (largest per-mode relative variance error):",
        "",
        markdown_table(
            ["Noll", "asset", "empirical", "emp/asset", "rel. error", "within floor"],
            _worst_rows(oopao_result, noll),
        ),
        "",
        "Strongest reference off-diagonal correlations:",
        "",
        markdown_table(
            ["Noll i", "Noll j", "asset ρ", "empirical ρ", "Δ"],
            [
                [
                    str(pair["noll_i"]),
                    str(pair["noll_j"]),
                    _fmt(pair["corr_ref"], "+.3f"),
                    _fmt(pair["corr_est"], "+.3f"),
                    _fmt(pair["corr_est"] - pair["corr_ref"], "+.3f"),
                ]
                for pair in oopao_pairs
            ],
        ),
        "",
        "## Arm 2 — the shipped sampler `draw_aberration` vs the same asset",
        "",
        f"`draw_aberration` renormalises every draw to `rms_rad`, which multiplies the "
        f"whole covariance by a design factor: measured mean-diagonal ratio "
        f"**{_fmt(sampler_result['amplitude_scale'], '.4g')}×**. The table below has "
        "that factor divided out (as pre-registered), so what is scored is the "
        "**shape**; the factor itself is the difficulty knob, not a claim.",
        "",
        f"**Verdict (shape): {sampler_result['verdict']}** — entry-error RMS "
        f"{_fmt(sampler_result['entry_error_rms'], '.4f')} vs noise floor "
        f"sqrt(2/{sampler_result['n_samples']}) = {_fmt(sampler_result['noise_floor'], '.4f')} "
        f"({_fmt(sampler_result['rms_over_floor'], '.2f')}× the floor); "
        f"{100 * sampler_result['fraction_within_floor']:.0f}% of entries inside the floor.",
        "",
        "Off-diagonal correlation RMS error "
        f"{_fmt(sampler_result['offdiag_corr_rms_error'], '.4f')}. The two largest "
        f"entries carry {100 * sampler_result['error_share_top2']:.0f}% of the squared error.",
        "",
        "Per-mode variance ladder, amplitude matched (rad²):",
        "",
        markdown_table(
            ["Noll", "asset", "sampled ÷ knob", "ratio", "rel. error", "within floor"],
            _ladder_rows(sampler_result, noll),
        ),
        "",
        "Worst modes (largest per-mode relative variance error):",
        "",
        markdown_table(
            ["Noll", "asset", "sampled ÷ knob", "ratio", "rel. error", "within floor"],
            _worst_rows(sampler_result, noll),
        ),
        "",
        "The sampler rescales every draw to a uniform `rms_rad`, which squeezes an "
        "anisotropic ensemble toward isotropy (Bingham). `src/ml/sim2f/optics.py` "
        "documents the cost as ~0.85× on the low orders and ~1.27× on the highest "
        "mode; the realised ratios above are the measurement of exactly that, and "
        "are the reason a diagonal deviation here is **expected**, not a surprise.",
        "",
        "Strongest reference off-diagonal correlations:",
        "",
        markdown_table(
            ["Noll i", "Noll j", "asset ρ", "sampled ρ", "Δ"],
            [
                [
                    str(pair["noll_i"]),
                    str(pair["noll_j"]),
                    _fmt(pair["corr_ref"], "+.3f"),
                    _fmt(pair["corr_est"], "+.3f"),
                    _fmt(pair["corr_est"] - pair["corr_ref"], "+.3f"),
                ]
                for pair in sampler_pairs
            ],
        ),
        "",
        "## How to read the two arms",
        "",
        "- Arm 1 asks whether the **analytic** asset describes the **stochastic** "
        "generator it is derived from. It is the load-bearing check: if it fails, "
        "the asset is wrong.",
        "- Arm 2 asks whether the **shipped sampler** reproduces the asset. A diagonal "
        "deviation there is the documented RMS-rescale artefact, and the correlation "
        "structure is the part that survives; both are reported above.",
        "- Neither arm compares simulator *outputs* (images, Strehl). This is the "
        "coefficient-distribution contract only -- see "
        "`tests/ao_shaping/ml/sim2f/test_oopao_parity.py` for the forward-model parity.",
        "",
    ]
    return "\n".join(lines)


def _print_summary(
    backend: dict[str, Any],
    oopao_result: dict[str, Any],
    sampler_result: dict[str, Any],
    timings: dict[str, float],
    out_dir: Path,
) -> None:
    """Print the compact terminal summary."""
    logger.info()
    logger.info(f"backend: array module {backend['array_module']} (gpu={backend['gpu']}, "
          f"gpu_resident={backend['gpu_resident']}, OOPAO_BACKEND={backend['oopao_backend_env']})")
    logger.info(f"wall clock: OOPAO screens {timings['oopao_s']:.2f} s, "
          f"sampler draws {timings['sampler_s']:.2f} s, total {timings['total_s']:.2f} s")
    for name, result in (("OOPAO ensemble", oopao_result), ("shipped sampler", sampler_result)):
        logger.info()
        logger.info(f"[{name}] n={result['n_samples']}  floor sqrt(2/n)={result['noise_floor']:.4f}")
        logger.info(f"  amplitude (mean diag emp/ref): {result['amplitude_scale']:.4g}"
              f"{'  [divided out]' if result['match_amplitude'] else ''}")
        logger.info(f"  entry error: rms {result['entry_error_rms']:.4f} "
              f"({result['rms_over_floor']:.2f}x floor), median {result['entry_error_median']:.4f}, "
              f"p95 {result['entry_error_p95']:.4f}, max {result['entry_error_max']:.4f}")
        logger.info(f"  entries within floor: {100 * result['fraction_within_floor']:.0f}%")
        logger.info(f"  off-diagonal correlation rms error: {result['offdiag_corr_rms_error']:.4f}")
        ratios = np.asarray(result["diagonal_ratio"], dtype=np.float64)
        logger.info(f"  diagonal emp/ref: min {ratios.min():.3f}, median {np.median(ratios):.3f}, "
              f"max {ratios.max():.3f}")
        logger.info(f"  verdict: {result['verdict']}")
    logger.info()
    logger.info(f"wrote {out_dir / 'validate_sim2f_aberration_ensemble.json'}")
    logger.info(f"wrote {out_dir / 'validate_sim2f_aberration_ensemble.md'}")


def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="OOPAO phase screens to draw (default: 2000)",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=0,
        help="sampler draws for arm 2 (0 = same as --n-samples)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the first screen (default: 0)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default: logs/oopao_gpu)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both validation arms and write the report.

    Args:
        argv: Arguments after the script name; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    n_draws = int(args.n_draws) if args.n_draws else int(args.n_samples)
    if int(args.n_samples) < 10 or n_draws < 10:
        logger.info("error: --n-samples and --n-draws must be >= 10")
        return 2

    logger.info(PRE_REGISTRATION)

    ref = _reference()
    logger.info(
        f"asset: {ref['asset']}  K={ref['noll'].size}  Noll "
        f"{int(ref['noll'][0])}..{int(ref['noll'][-1])}  grid={ref['grid']}"
    )
    backend = _backend_record()
    logger.info(
        f"backend: array module {backend['array_module']} (gpu={backend['gpu']}, "
        f"gpu_resident={backend['gpu_resident']})"
    )

    started = time.perf_counter()
    sim = _bench(ref)
    # A least-squares round trip on the sampled basis: proves the basis and the
    # support describe the same pupil before any statistics are trusted.
    _coords, basis, _noll, _support = _pupil_geometry(sim, int(ref["n_modes"]))
    roundtrip = _roundtrip_error(basis)
    logger.info(f"basis round trip max|Δa| = {roundtrip:.3e} rad")
    pupil_diameter_m = _grid_span_m(sim)
    logger.info(
        f"OOPAO pupil: diameter={pupil_diameter_m * 1e3:.4g} mm "
        f"(= grid {ref['grid']} × {ref['pupil_pitch_m'] * 1e6:.4g} µm) -> "
        f"screen pixel pitch {pupil_diameter_m / ref['grid'] * 1e6:.4g} µm, "
        "matching the model grid"
    )
    arm_start = time.perf_counter()
    oopao_sigma = oopao_ensemble(sim, ref, n_samples=int(args.n_samples), seed=int(args.seed))
    oopao_seconds = time.perf_counter() - arm_start

    arm_start = time.perf_counter()
    sampler_sigma = sampler_ensemble(sim, ref, n_draws=n_draws, seed=int(args.seed))
    sampler_seconds = time.perf_counter() - arm_start

    oopao_result = compare(oopao_sigma, ref["covariance"], n_samples=int(args.n_samples))
    sampler_result = compare(
        sampler_sigma, ref["covariance"], n_samples=n_draws, match_amplitude=True
    )
    timings = {
        "oopao_s": oopao_seconds,
        "sampler_s": sampler_seconds,
        "total_s": time.perf_counter() - started,
    }

    payload = {
        "generated_by": "scripts/ml/validate_sim2f_aberration_ensemble.py",
        "asset": {
            "path": str(DEFAULT_COVARIANCE_PATH.relative_to(ROOT)),
            "sha256": ref["sha256"],
            "noll": [int(v) for v in ref["noll"]],
            "grid": ref["grid"],
            "pupil_radius_px": ref["pupil_radius_px"],
            "pupil_pitch_m": ref["pupil_pitch_m"],
            "n_modes": ref["n_modes"],
            "wavelength_m": ref["wavelength_m"],
            "turbulence": ref["turbulence"],
        },
        "pre_registration": PRE_REGISTRATION.splitlines(),
        "backend": backend,
        "config": {
            "n_samples": int(args.n_samples),
            "n_draws": n_draws,
            "seed": int(args.seed),
            "oopao_pupil_diameter_m": pupil_diameter_m,
        },
        "timings_s": timings,
        "basis_roundtrip_max_abs_err": roundtrip,
        "arms": {
            "oopao_ensemble": _jsonable(oopao_result),
            "shipped_sampler": _jsonable(sampler_result),
        },
        "oopao_diagonal_ratio_mean": float(np.mean(oopao_result["diagonal_ratio"])),
        "sampler_diagonal_ratio_mean": float(np.mean(sampler_result["diagonal_ratio"])),
    }
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validate_sim2f_aberration_ensemble.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "validate_sim2f_aberration_ensemble.md").write_text(
        _render_markdown(
            ref,
            backend,
            oopao_result,
            _strongest_pairs(oopao_result, ref["noll"]),
            sampler_result,
            _strongest_pairs(sampler_result, ref["noll"]),
            timings,
            payload["config"],
        ),
        encoding="utf-8",
    )
    _print_summary(backend, oopao_result, sampler_result, timings, out_dir)
    return 0


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``(K, K)`` correlation matrices and stringify numpy arrays.

    The matrices are dropped because the markdown already tabulates them (and they
    would bloat the JSON); the remaining numpy arrays are the diagonal vectors,
    which stay as plain lists for a diffable, dependency-free record.
    """
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key in ("corr_ref", "corr_est"):
            continue
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
