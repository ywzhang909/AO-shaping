"""Offline verification for the ``--export-correction`` gray-offset CSV (2026-09-16 contract).

Contract under test: the exported correction CSV **must not** bake the SLM panel
shift, and the full-scale 2π must equal the device constant
``get_max_grayscale()`` (1023) — NOT a wavelength-dependent ``two_pi_gray``
(e.g. 998 at 532 nm, which would scale the correction 1.025× when the official
software converts back with 1023).

The production pipeline is recomputed line-by-line identically to
``slm_zernike_response.export_driver_correction``:

    h5/device_config + w json → c = -pinv(M) @ w (w[0]=0)
    → make_phase ({nm: c·2π}) → WavefrontCorrection.correction_gray_offsets(max_grayscale=1023)

and compared **byte-for-byte** against the exported CSV. Also verifies loader
consistency (np.loadtxt / ``Santec.load_gray_from_csv`` /
``WavefrontCorrection.load_gray_from_csv``) and the ``map_error`` additive
semantics (``displayed = mod(base + corr, 1024)``).

Usage (fully offline, no hardware):
    python scripts/verify_correction_csv.py
    python scripts/verify_correction_csv.py --h5 <matrix.h5> --w <w_before.json> --csv <corr.csv>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LIBS = ROOT / "libs"
for p in (SRC, LIBS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ao_shaping.drivers.slm.santec import (  # noqa: E402
    Santec,
    WavefrontCorrection,
    get_max_grayscale,
)
from ao_shaping.optimizer.wf.zernike_response_matrix import (  # noqa: E402
    load_zernike_response_matrix,
)
from ao_shaping.tools.slm.slm_zernike_common import (  # noqa: E402
    DLL_ZERNIKE_ORDER,
    PANEL_H,
    PANEL_W,
    make_phase,
)
from ao_shaping.utils.pattern_helper import PatternHelper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--h5",
        type=Path,
        default=ROOT / "data/zernike_response_matrix/zm_recal_532_20260916.h5",
        help="Zernike response matrix h5 (default: latest recal matrix)",
    )
    p.add_argument(
        "--w",
        "--w-file",
        dest="w_file",
        type=Path,
        default=ROOT / "data/zernike_correction/_w_before_66.json",
        help="WFS wavefront JSON (66-length, unit λ; or dict with w/w_before key)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT
        / "data/zernike_response_matrix/zm_recal_532_20260916_correction_gray.csv",
        help="Exported correction gray CSV to verify (default: h5-sidecar deliverable)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    h5, wj, csv = args.h5, args.w_file, args.csv
    side = csv.with_suffix(".json")

    for f in (h5, wj, csv, side):
        if not f.is_file():
            raise SystemExit(f"[FAIL] 缺少输入文件: {f}")

    MG = get_max_grayscale()
    assert MG == 1023, f"期望设备常量 1023, 得到 {MG}"

    # ---------- 0) sidecar JSON ----------
    side_data = json.loads(side.read_text(encoding="utf-8"))
    assert side_data["max_gray"] == 1023, side_data
    assert side_data["shift_included"] is False, side_data
    assert "不含 shift" in side_data["gray_encoding"], side_data
    assert side_data["shift_note"], side_data
    assert "two_pi_gray" not in side_data["consumption"], side_data
    print("[0] 侧车 JSON: max_gray=1023, shift_included=False, 不含 shift ✔")

    # ---------- 1) CSV structure / value range ----------
    csv_data = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=range(1, PANEL_W + 1))
    assert csv_data.shape == (PANEL_H, PANEL_W), csv_data.shape
    assert csv_data.min() >= 0 and csv_data.max() <= 1023, (csv_data.min(), csv_data.max())
    print(f"[1] CSV {csv_data.shape}, 值域 [{csv_data.min()}, {csv_data.max()}] ⊂ 0..1023 ✔")

    # ---------- 2) loaders byte-identical ----------
    g_drv = Santec.load_gray_from_csv(str(csv))
    g_wc = WavefrontCorrection.load_gray_from_csv(str(csv))
    assert np.array_equal(g_drv, csv_data), "Santec.load_gray_from_csv ≠ CSV"
    assert np.array_equal(g_wc, csv_data), "WavefrontCorrection.load_gray_from_csv ≠ CSV"
    print("[2] Santec / WavefrontCorrection.load_gray_from_csv 与 CSV 逐字节一致 ✔")

    # ---------- 3) recompute production pipeline → proves no baked shift ----------
    r = load_zernike_response_matrix(h5)
    dc = dict(r.device_config or {})
    ids = dc.get("slm_mode_ids_dll") or list(range(2, r.matrix.shape[1] + 2))
    nm_list = dc.get("slm_mode_nm") or [list(DLL_ZERNIKE_ORDER[i - 1]) for i in ids]
    radius = float(dc.get("zernike_radius_px") or 300.0)
    w_raw = json.loads(wj.read_text(encoding="utf-8"))
    if isinstance(w_raw, dict):
        w_raw = w_raw.get("w") or w_raw.get("w_before") or w_raw.get("wavefront")
    w = np.asarray(w_raw, dtype=float).reshape(-1)
    if w.size == 67:
        w = w[1:]
    assert w.size == r.pinv_matrix.shape[1], (w.size, r.pinv_matrix.shape[1])
    w = w.copy()
    w[0] = 0.0
    c_waves = -r.pinv_matrix @ w
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))
    coeff = {
        (int(nm[0]), int(nm[1])): float(c_waves[i]) * 2.0 * np.pi
        for i, nm in enumerate(nm_list)
    }
    phase_rad = make_phase(
        ph, coeff, radius=radius, n_max=int(getattr(r, "n_max", None) or 4)
    )
    assert "shift" not in str(make_phase.__doc__ or ""), "make_phase 不应有 shift 语义"
    g_re = WavefrontCorrection.correction_gray_offsets(phase_rad, max_grayscale=MG)
    assert np.array_equal(g_re, csv_data), "复算管线 ≠ 导出 CSV → 导出烘焙了额外变换?"
    print("[3] 复算管线 (make_phase 无 shift) 与导出逐字节一致 → 不烘焙 shift ✔")

    # ---------- 4) map_error additive semantics (displayed = mod(base + corr, 1024)) ----------
    wc_id = WavefrontCorrection(str(csv), calc_fn=lambda raw: raw.astype(np.float64))
    wc_id.load_csv()
    wc_id.calc(panel_resolution=(PANEL_W, PANEL_H), max_grayscale=1023)
    zero = np.zeros_like(csv_data)
    disp = wc_id.map_error(zero, max_grayscale=1023)
    assert np.array_equal(disp, csv_data), "map_error(0 + corr) ≠ corr"
    # mod wrap check: base=500, corr=999 → 499 (500+999=1499 mod 1024)
    half = np.full_like(csv_data, 500)
    disp2 = wc_id.map_error(half, max_grayscale=1023)
    expect = np.mod(500 + csv_data.astype(np.int64), 1024)
    assert np.array_equal(disp2, expect), "map_error mod 环绕不一致"
    print("[4] map_error 加法语义 + mod 环绕 (base=500) 一致 ✔")

    # ---------- 5) quantization error bound (circular distance: wrap pixel 1024≡0) ----------
    scaled = np.mod(phase_rad / (2 * np.pi) * MG, MG + 1)
    cir_err = (
        np.mod(csv_data.astype(np.float64) - scaled + (MG + 1) / 2, MG + 1)
        - (MG + 1) / 2
    )
    assert np.abs(cir_err).max() <= 0.5 + 1e-9, np.abs(cir_err).max()
    print(
        f"[5] 量化误差上界 ≤ 0.5 级 (圆周距离, max={np.abs(cir_err).max():.4f}) ✔"
    )

    print("\nALL PASS: 满量程 1023 + 不烘焙 shift + 官方加载器逐字节一致")


if __name__ == "__main__":
    main()