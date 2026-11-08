"""issue #237：训练帧标定实测 PSF（生产 sigma_ccd_px 路径）与加性二次背景，held-out 离线评估。

全程离线：不 import ``ao_shaping.drivers``，不打开 SLM / 相机 / 激光；只读磁盘存档。

口径（预先固定，见 results.json ``protocol``）：

* 前向：沿用冻结模型生产路径，``dataclasses.replace(params, sigma_ccd_px=s)`` 后重建
  ``FrozenDefocusForward``（零填充 conv2d，σ_array = σ_ccd_px / 2，25×25 核）。
  只替换 PSF，不改 gain / background / 拟合参数。
* PSF 候选网格 ``sigma_ccd_px ∈ {0,1,1.5,2,3,4}``（相机像素）；每个相位符号单独
  在**训练帧**（r2 train 18 + ladder pass1 42）上按合并 full-frame Pearson 选择，
  不用 rel-L2 选择（避免模糊过度平滑时的 MSE 陷阱）。
* 背景：二次多项式 6 系数（1,x,y,x²,xy,y²，坐标归一到 [-1,1]），在训练帧残差
  ``image - pred`` 上按域（r2 / ladder）无约束最小二乘拟合，加性施加；不做非负截断。
* 评估集：ladder pass2（42，held-out）、r2 usable（36，含训练帧，仅作重代入诊断）、
  r2 held-out（18）。核心掩膜 = 评估集均值图 ≥ 0.1·max（同基准口径）。
* 辅助：``compute_forward_metrics`` 的 ``relative_l2_affine_diagnostic`` 与
  ``cv_abs_error``（核心掩膜，逐帧均值）；方向一致性复用 ``direction_pairs/direction_metric``。
* 相位符号：-1 为主，+1 仅作参考。
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TRANSFER_PATH = ROOT / "scripts/slm/gated_recurrent_transfer_20261107.py"
_SPEC = importlib.util.spec_from_file_location("gated_recurrent_transfer_20261107", _TRANSFER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载 {_TRANSFER_PATH}")
_TRANSFER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRANSFER
_SPEC.loader.exec_module(_TRANSFER)

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)
from ao_shaping.utils.image.physical_model_metrics import compute_forward_metrics

SEED = 20261107
OUT_DIR = ROOT / "data/debug/slm_transfer/measured_psf_probe_20261107"
SIGMA_GRID = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0)  # camera px, predeclared
SIGNS = (-1.0, 1.0)
CORE_LEVEL = 0.1
DEVICE_STATEMENT = (
    "全程离线；未占用、未打开、未初始化任何 SLM / CCD 相机 / 激光设备；无设备需要释放或关闭。"
    "只读磁盘历史 NPZ/JSON；CPU、固定种子 20261107。"
)
FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))  # camera px

_LOG_LINES: list[str] = []


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _tag_frames(fit: dict, offset: np.ndarray, center: np.ndarray) -> tuple[list, list, list]:
    """Return (r2_usable, ladder_pass1_train, ladder_pass2_heldout) with prefixed keys."""
    r2 = _TRANSFER.usable(_TRANSFER.load_frames(_TRANSFER.R2_DIR, None, offset, center))
    for f in r2:
        f.key = f"r2/{f.key}"
    pass1: list = []
    pass2: list = []
    for tag, directory in _TRANSFER.LADDER_DIRS.items():
        for pass_name in ("pass1", "pass2"):
            frames = _TRANSFER.usable(_TRANSFER.load_frames(directory, pass_name, offset, center))
            kept = [f for f in frames if f.split == "train"]
            for f in kept:
                f.key = f"{tag}/{pass_name}/{f.key}"
                f.split = "train" if pass_name == "pass1" else "test"
            (pass1 if pass_name == "pass1" else pass2).extend(kept)
    return r2, pass1, pass2


def _forward_with_sigma(base: Any, sigma: float) -> FrozenDefocusForward:
    if sigma == 0.0:
        return base
    params = dataclasses.replace(base.params, sigma_ccd_px=float(sigma))
    return FrozenDefocusForward(params, np.asarray(base.camera_xy, dtype=np.float64), device=str(base._device))


def _predict_all(forward: FrozenDefocusForward, frames: list) -> dict[str, np.ndarray]:
    return {f.key: _TRANSFER.predict_frame(forward, f.phase) for f in frames}


# --------------------------------------------------------------------------- #
# background
# --------------------------------------------------------------------------- #
def _bg_basis(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    x = (xx - (shape[1] - 1) / 2.0) / ((shape[1] - 1) / 2.0)
    y = (yy - (shape[0] - 1) / 2.0) / ((shape[0] - 1) / 2.0)
    cols = [np.ones_like(x), x, y, x * x, x * y, y * y]
    return np.stack([c.ravel() for c in cols], axis=1)


def _fit_background(frames: list, preds: dict[str, np.ndarray]) -> np.ndarray:
    basis = _bg_basis(frames[0].image.shape)
    resid = np.concatenate([(f.image - preds[f.key]).ravel() for f in frames])
    design = np.concatenate([basis for _ in frames], axis=0)
    coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
    return coef


def _apply_background(pred: np.ndarray, coef: np.ndarray) -> np.ndarray:
    basis = _bg_basis(pred.shape)
    return pred + (basis @ coef).reshape(pred.shape)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _core(frames: list) -> np.ndarray:
    mean_image = np.mean([f.image for f in frames], axis=0)
    return mean_image >= CORE_LEVEL * float(np.max(mean_image))


def _rel_l2_core(pred: np.ndarray, image: np.ndarray, core: np.ndarray) -> float:
    return float(np.linalg.norm(pred[core] - image[core]) / np.linalg.norm(image[core]))


def _eval_row(frames: list, preds: dict[str, np.ndarray]) -> dict[str, float]:
    core = _core(frames)
    full_p, core_p, full_l2, core_l2, aff, cv = [], [], [], [], [], []
    for f in frames:
        p = preds[f.key]
        full_p.append(_TRANSFER.pearson(p, f.image))
        core_p.append(_TRANSFER.pearson(p[core], f.image[core]))
        full_l2.append(_TRANSFER.rel_l2(p, f.image))
        core_l2.append(_rel_l2_core(p, f.image, core))
        aux = compute_forward_metrics(p, f.image, core)
        if aux.get("relative_l2_affine_diagnostic") is not None:
            aff.append(aux["relative_l2_affine_diagnostic"])
        if aux.get("cv_abs_error") is not None:
            cv.append(aux["cv_abs_error"])
    return {
        "n": len(frames),
        "core_pixels": int(core.sum()),
        "full_pearson": float(np.mean(full_p)),
        "core_pearson": float(np.mean(core_p)),
        "full_rel_l2": float(np.mean(full_l2)),
        "core_rel_l2": float(np.mean(core_l2)),
        "rel_l2_affine_diagnostic": float(np.mean(aff)) if aff else None,
        "cv_abs_error": float(np.mean(cv)) if cv else None,
    }


def _spot_fwhm(image: np.ndarray) -> tuple[float, float]:
    """Row/col half-max widths through the peak, in camera px (grid step = 2 camera px)."""
    img = np.asarray(image, dtype=np.float64) - float(np.median(image))
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)

    def width(v: np.ndarray) -> float:
        idx = np.where(v >= 0.5 * v.max())[0]
        return float((idx[-1] - idx[0] + 1) * 2.0) if idx.size else 0.0

    return width(img[iy]), width(img[:, ix])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.perf_counter()
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("device statement: " + DEVICE_STATEMENT)

    fit = _TRANSFER.load_fit()
    center = np.asarray(fit["camera_crop_center"], dtype=np.float64)
    results: dict[str, Any] = {
        "device_statement": DEVICE_STATEMENT,
        "seed": SEED,
        "git_head": _git("rev-parse", "HEAD"),
        "core_module_sha256": hashlib.sha256(
            (ROOT / "src/ao_shaping/algorithm/signal_processing/differentiable_defocus_uniformity.py").read_bytes()
        ).hexdigest(),
        "protocol": {
            "sigma_grid_ccd_px": list(SIGMA_GRID),
            "sigma_selection": "per sign, max pooled train full-frame Pearson (r2 train 18 + ladder pass1 42)",
            "background": "quadratic 6-term (1,x,y,x2,xy,y2) LSQ on train residuals, per domain, additive, no clipping",
            "core_mask": f"eval-set mean image >= {CORE_LEVEL}*max",
            "eval_sets": ["ladder_pass2 (42, held-out)", "r2_usable (36, contains train, re-substitution diagnostic)",
                          "r2_heldout (18)"],
            "phase_sign": {"primary": -1.0, "reference": 1.0},
        },
        "signs": {},
    }

    offset = np.asarray(_TRANSFER.make_forward(fit, -1.0)[1], dtype=np.float64)
    r2_all, lad_pass1, lad_pass2 = _tag_frames(fit, offset, center)
    r2_train = [f for f in r2_all if f.split == "train"]
    r2_test = [f for f in r2_all if f.split == "test"]
    log(f"frames: r2 usable {len(r2_all)} (train {len(r2_train)} / test {len(r2_test)}), "
        f"ladder pass1 train {len(lad_pass1)}, ladder pass2 held-out {len(lad_pass2)}")
    if (len(r2_all), len(r2_train), len(r2_test), len(lad_pass1), len(lad_pass2)) != (36, 18, 18, 42, 42):
        raise RuntimeError("frame counts differ from the baseline protocol; abort")

    train_frames = r2_train + lad_pass1
    ladder_frames = lad_pass1 + lad_pass2
    all_frames = r2_all + lad_pass1 + lad_pass2
    ladder_keys = {f.key for f in ladder_frames}
    pass_groups = []
    for tag in _TRANSFER.LADDER_DIRS:
        for pass_name in ("pass1", "pass2"):
            seq = sorted([f for f in ladder_frames if f.key.startswith(f"{tag}/{pass_name}/")], key=lambda f: f.ns)
            pass_groups.append(seq)

    for sign in SIGNS:
        tag_sign = f"{sign:+.0f}"
        log(f"=== phase_sign {tag_sign} ===")
        fit_forward, _ = _TRANSFER.make_forward(fit, sign)
        preds_by_sigma: dict[float, dict[str, np.ndarray]] = {}
        train_scores: dict[float, float] = {}
        for sigma in SIGMA_GRID:
            forward = _forward_with_sigma(fit_forward, sigma)
            t0 = time.perf_counter()
            preds = _predict_all(forward, all_frames)
            preds_by_sigma[sigma] = preds
            train_scores[sigma] = float(np.mean([_TRANSFER.pearson(preds[f.key], f.image) for f in train_frames]))
            log(f"sign {tag_sign} sigma {sigma:4.1f}: train full-Pearson {train_scores[sigma]:.6f} "
                f"({time.perf_counter() - t0:.1f}s)")

        best_sigma = max(SIGMA_GRID, key=lambda s: (train_scores[s], -s))
        base_preds = preds_by_sigma[0.0]
        psf_preds = preds_by_sigma[best_sigma]

        # background per domain, fitted on train residuals of the chosen-sigma and sigma-0 predictions
        bg0 = {
            "r2": _fit_background(r2_train, base_preds),
            "ladder": _fit_background(lad_pass1, base_preds),
        }
        bgs = {
            "r2": _fit_background(r2_train, psf_preds),
            "ladder": _fit_background(lad_pass1, psf_preds),
        }

        def variant_preds(
            kind: str,
            pred_base: dict[str, np.ndarray],
            pred_psf: dict[str, np.ndarray],
            bg_base: dict[str, np.ndarray],
            bg_psf: dict[str, np.ndarray],
        ) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for f in all_frames:
                domain = "ladder" if f.key in ladder_keys else "r2"
                if kind == "baseline":
                    out[f.key] = pred_base[f.key]
                elif kind == "psf":
                    out[f.key] = pred_psf[f.key]
                elif kind == "bg":
                    out[f.key] = _apply_background(pred_base[f.key], bg_base[domain])
                elif kind == "psf_bg":
                    out[f.key] = _apply_background(pred_psf[f.key], bg_psf[domain])
                else:
                    raise ValueError(kind)
            return out

        variants = ("baseline", "psf", "bg", "psf_bg")
        var_preds = {v: variant_preds(v, base_preds, psf_preds, bg0, bgs) for v in variants}
        eval_sets = {
            "ladder_pass2": lad_pass2,
            "r2_usable": r2_all,
            "r2_heldout": r2_test,
        }
        rows = {
            v: {name: _eval_row(frames, var_preds[v]) for name, frames in eval_sets.items()}
            for v in variants
        }

        # direction consistency over all ladder pass sequences (42 pairs expected)
        dir_pairs = _TRANSFER.direction_pairs(pass_groups)
        direction = {
            v: _TRANSFER.direction_metric({f.key: var_preds[v][f.key] for f in ladder_frames}, dir_pairs)
            for v in variants
        }

        # spot widths: measured (camera px, row/col half-max through peak) vs predicted of the chosen model
        width = {}
        for name, frames in (("r2_train", r2_train), ("r2_heldout", r2_test), ("ladder_pass1", lad_pass1),
                             ("ladder_pass2", lad_pass2)):
            meas = np.mean([_spot_fwhm(f.image) for f in frames], axis=0)
            pred_base = np.mean([_spot_fwhm(base_preds[f.key]) for f in frames], axis=0)
            pred_psf = np.mean([_spot_fwhm(psf_preds[f.key]) for f in frames], axis=0)
            width[name] = {
                "measured_row_col_px": [float(meas[0]), float(meas[1])],
                "model_sigma0_row_col_px": [float(pred_base[0]), float(pred_base[1])],
                f"model_sigma{best_sigma:g}_row_col_px": [float(pred_psf[0]), float(pred_psf[1])],
            }
        manifest_spots = {}
        rows_manifest = json.loads((_TRANSFER.R2_DIR / "manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "test", "calibration"):
            vals = [r["spot"]["fwhm_px"] for r in rows_manifest
                    if r.get("split") == split and isinstance(r.get("spot"), dict) and "fwhm_px" in r["spot"]]
            manifest_spots[f"r2_{split}"] = {"n": len(vals), "mean_fwhm_px": float(np.mean(vals)) if vals else None}
        cal_refs = [r["spot"]["fwhm_px"] for r in rows_manifest
                    if r.get("split") == "calibration" and str(r.get("label", "")).startswith("reference")
                    and isinstance(r.get("spot"), dict)]
        manifest_spots["r2_calibration_reference_only"] = {
            "n": len(cal_refs), "mean_fwhm_px": float(np.mean(cal_refs)) if cal_refs else None}

        results["signs"][tag_sign] = {
            "sigma_selected_ccd_px": best_sigma,
            "sigma_selected_fwhm_px": best_sigma * FWHM_PER_SIGMA,
            "train_full_pearson_curve": {f"{s:g}": v for s, v in train_scores.items()},
            "background_coefficients": {
                "order": ["1", "x", "y", "x2", "xy", "y2"],
                "analytic_sigma0": {k: v.tolist() for k, v in bg0.items()},
                "psf_selected": {k: v.tolist() for k, v in bgs.items()},
            },
            "rows": rows,
            "deltas_vs_baseline": {
                v: {name: {m: (rows[v][name][m] - rows["baseline"][name][m])
                           if rows[v][name][m] is not None and rows["baseline"][name][m] is not None else None
                           for m in ("full_pearson", "core_pearson", "full_rel_l2", "core_rel_l2",
                                     "rel_l2_affine_diagnostic", "cv_abs_error")}
                    for name in eval_sets}
                for v in variants if v != "baseline"
            },
            "direction": direction,
            "spot_width": {"model_and_measured_frames": width, "manifest_spot_fwhm": manifest_spots},
        }
        log(f"sign {tag_sign}: selected sigma {best_sigma:g} ccd px; ladder_pass2 core Pearson baseline "
            f"{rows['baseline']['ladder_pass2']['core_pearson']:.6f}; direction mean baseline "
            f"{direction['baseline']['mean']:.4f} -> psf_bg {direction['psf_bg']['mean']:.4f}")

        # baseline reproduction gate for the primary sign
        if tag_sign == "-1":
            expected = {
                ("ladder_pass2", "full_pearson"): 0.840117,
                ("ladder_pass2", "core_pearson"): 0.693776,
                ("ladder_pass2", "full_rel_l2"): 0.530658,
                ("r2_usable", "full_pearson"): 0.647344,
                ("r2_usable", "core_pearson"): 0.424922,
                ("r2_usable", "full_rel_l2"): 0.784888,
                ("r2_heldout", "full_pearson"): 0.655722,
                ("r2_heldout", "core_pearson"): 0.498610,
                ("r2_heldout", "full_rel_l2"): 0.818238,
            }
            for (name, metric), value in expected.items():
                got = rows["baseline"][name][metric]
                if abs(got - value) > 2e-6:
                    raise RuntimeError(f"baseline reproduction mismatch {name}/{metric}: {got} vs {value}")
            log("baseline reproduction gate: PASSED (9/9 values match the archived baseline)")

    # ---- verdict on the primary sign, held-out r2 ----
    primary = results["signs"]["-1"]
    r2h_base = primary["rows"]["baseline"]["r2_heldout"]
    r2h_psf_bg = primary["rows"]["psf_bg"]["r2_heldout"]
    improves = (r2h_psf_bg["core_pearson"] > r2h_base["core_pearson"]
                and r2h_psf_bg["core_rel_l2"] < r2h_base["core_rel_l2"])
    results["verdict"] = {
        "criterion": "phase_sign -1, r2 held-out: psf_bg core Pearson up AND core rel-L2 down vs baseline",
        "core_pearson_delta": r2h_psf_bg["core_pearson"] - r2h_base["core_pearson"],
        "core_rel_l2_delta": r2h_psf_bg["core_rel_l2"] - r2h_base["core_rel_l2"],
        "full_pearson_delta": r2h_psf_bg["full_pearson"] - r2h_base["full_pearson"],
        "full_rel_l2_delta": r2h_psf_bg["full_rel_l2"] - r2h_base["full_rel_l2"],
        "both_core_metrics_improve": bool(improves),
        "proceed_to_237_implementation": bool(improves),
        "text": "可进入 #237 实现阶段" if improves else "无改善",
    }
    results["not_measured"] = {
        "d_vs_log_width_correlation": "未测：单一冻结检查点，本探针不重拟合 defocus/log_width，无可用的 d-log_width 配对样本；未声称通过 <0.7 判据。",
    }
    results["runtime_s"] = time.perf_counter() - t_start
    results["log"] = _LOG_LINES

    (OUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "REPORT.md").write_text(_render_report(results), encoding="utf-8")
    (OUT_DIR / "run.log").write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    log(f"wrote {OUT_DIR}")


def _render_report(r: dict[str, Any]) -> str:
    lines = ["# 实测 PSF 与加性背景离线探针（issue #237）", "", "## 设备声明", "", r["device_statement"], "",
             f"git HEAD：`{r['git_head']}`；核心模块 sha256：`{r['core_module_sha256']}`；种子 {r['seed']}。", "",
             "## 协议（预先固定）", ""]
    for k, v in r["protocol"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## PSF 选择与宽度", "",
              "| sign | 选中 sigma (ccd px) | 等效 FWHM (px) | 背景系数 (psf, r2) | 背景系数 (psf, ladder) |",
              "|---:|---:|---:|---|---|"]
    for sign, s in r["signs"].items():
        lines.append(f"| {sign} | {s['sigma_selected_ccd_px']:g} | {s['sigma_selected_fwhm_px']:.2f} | "
                     f"{[round(c, 3) for c in s['background_coefficients']['psf_selected']['r2']]} | "
                     f"{[round(c, 3) for c in s['background_coefficients']['psf_selected']['ladder']]} |")
    lines += ["", "训练集全帧 Pearson 随 sigma 曲线（选择依据）：", ""]
    for sign, s in r["signs"].items():
        curve = ", ".join(f"{k}: {v:.6f}" for k, v in s["train_full_pearson_curve"].items())
        lines.append(f"- sign {sign}: {curve}")
    lines += ["", "测得光斑宽度（camera px，行/列半高全宽，sign 与 sigma 无关的测量项）：", ""]
    for sign, s in r["signs"].items():
        for name, w in s["spot_width"]["model_and_measured_frames"].items():
            lines.append(f"- sign {sign} {name}: " + ", ".join(f"{k}={[round(x, 2) for x in v]}" for k, v in w.items()))
        for name, m in s["spot_width"]["manifest_spot_fwhm"].items():
            lines.append(f"- sign {sign} manifest {name}: n={m['n']}, mean fwhm_px={m['mean_fwhm_px']}")
    lines += ["", "历史口径对照：约 14 px（实测）与 21 px（模型）未在本探针中复现为同一口径；见上表。", ""]
    for sign, s in r["signs"].items():
        lines += [f"## sign {sign}", "", "| 变体 | 评估集 | full Pearson | core Pearson | full rel-L2 | core rel-L2 | rel-L2 affine(diag) | cv_abs_error |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for v, per in s["rows"].items():
            for name, m in per.items():
                def f(x: Any) -> str:
                    return "n/a" if x is None else f"{x:.6f}"
                lines.append(f"| {v} | {name} (n={m['n']}) | {f(m['full_pearson'])} | {f(m['core_pearson'])} | "
                             f"{f(m['full_rel_l2'])} | {f(m['core_rel_l2'])} | {f(m['rel_l2_affine_diagnostic'])} | "
                             f"{f(m['cv_abs_error'])} |")
        lines += ["", "增量（变体 - baseline；rel-L2 负值为改善，Pearson 正值为改善）：", ""]
        for v, per in s["deltas_vs_baseline"].items():
            for name, d in per.items():
                lines.append(f"- {v} / {name}: " + ", ".join(
                    f"{k}={'n/a' if x is None else f'{x:+.6f}'}" for k, x in d.items()))
        lines += ["", "方向一致性（ladder 符号对，均值）：", ""]
        for v, d in s["direction"].items():
            lines.append(f"- {v}: mean {d['mean']:.4f}, n={d['n']}")
        lines.append("")
    v = r["verdict"]
    lines += ["## 结论", "", f"- 判据：{v['criterion']}",
              f"- r2 held-out core Pearson Δ = {v['core_pearson_delta']:+.6f}；core rel-L2 Δ = {v['core_rel_l2_delta']:+.6f}",
              f"- r2 held-out full Pearson Δ = {v['full_pearson_delta']:+.6f}；full rel-L2 Δ = {v['full_rel_l2_delta']:+.6f}",
              f"- 结论：**{v['text']}**", "",
              "## 未测项", "", f"- d vs log_width：{r['not_measured']['d_vs_log_width_correlation']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
