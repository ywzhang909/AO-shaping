"""Generate the illustrated Zernike → WFS readout distribution report.

Loads a series of SLM Zernike patterns (different modes / radii / amplitudes) at
the calibrated SLM shift and captures, per case:

  - **SLM phase**: left = radian source phase, right = **actual displayed
    grayscale pattern** (mod 2π + shift applied)
  - **WFS readout**: spots image / read phase map (wavefront) / Zernike
    coefficient distribution / key metrics

and writes an illustrated markdown report (``report.md`` + ``phase/`` + ``wfs/``)
to ``docs/slm/zernike_wfs_report/`` by default.

The WFS reference plane is created from a **flat SLM phase**, so every readout
is the *increment relative to flat* and excludes static system aberration.

Usage:
    $env:PYTHONPATH = "src"
    python scripts/generate_zernike_wfs_report.py
    python scripts/generate_zernike_wfs_report.py -o docs/slm/zernike_wfs_report

Hardware: Santec SLM-200 (memory mode) + Thorlabs WFS. WFS exposure is capped at
7 ms (repo constraint).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Matplotlib must be configured to Agg BEFORE importing pyplot (repo-wide rule)
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ao_shaping.drivers.slm import Santec  # noqa: E402
from ao_shaping.drivers.wfs import ThorlabWFS  # noqa: E402
from ao_shaping.tools.slm.slm_zernike_common import (  # noqa: E402
    DEFAULT_EXPOSURE_MS,
    DLL_ZERNIKE_NAMES,
    MAX_EXPOSURE_MS,
    PANEL_H,
    PANEL_W,
    SETTLE_REDUNDANCY_S,
    WFS_ZERNIKE_ORDER,
    flat_gray,
    make_phase,
    measure_zernike,
    show_phase,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper  # noqa: E402

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo-wide)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_OUTPUT = ROOT / "docs" / "slm" / "zernike_wfs_report"

# (label, mode (n,m), radius_px, amplitude_rad); mode=None → flat phase
CASES: list[tuple[str, tuple[int, int] | None, float | None, float | None]] = [
    ("flat", None, None, None),
    ("tilt_n1m1", (1, 1), 600.0, 10.0),
    ("defocus_n2m0", (2, 0), 600.0, 10.0),
    ("astig_n2m2", (2, 2), 600.0, 10.0),
    ("coma_n3m1", (3, 1), 600.0, 10.0),
    ("spherical_n4m0", (4, 0), 600.0, 10.0),
    ("defocus_R300", (2, 0), 300.0, 10.0),
    ("defocus_R900", (2, 0), 900.0, 10.0),
    ("defocus_A2", (2, 0), 600.0, 2.0),
    ("defocus_A20", (2, 0), 600.0, 20.0),
]


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def phase_figure(rec: dict, out_png: Path, max_gray: int,
                 shift: tuple[int, int]) -> None:
    """SLM phase figure: radian source | actual displayed grayscale pattern."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    phase, gray = rec.get("phase"), rec.get("gray")

    if phase is not None:
        im = ax1.imshow(phase, cmap="twilight", origin="upper")
        fig.colorbar(im, ax=ax1, shrink=0.85, label="phase (rad)")
        ax1.set_title(f"SLM phase source (rad) — {rec['label']}")
    else:
        ax1.text(0.5, 0.5, "flat phase (gray=0)", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=13)
        ax1.set_title(f"SLM phase source — {rec['label']} (flat)")
    ax1.set_xlabel("SLM x (px)")
    ax1.set_ylabel("SLM y (px)")

    if gray is not None:
        im = ax2.imshow(gray, cmap="gray", origin="upper", vmin=0, vmax=max_gray)
        fig.colorbar(im, ax=ax2, shrink=0.85, label=f"gray (0..{max_gray})")
        ax2.set_title(f"SLM displayed pattern (gray, mod 2pi + shift "
                      f"{shift}) — {rec['label']}")
    else:
        ax2.text(0.5, 0.5, "flat", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=13)
        ax2.set_title(f"SLM displayed pattern — {rec['label']} (flat)")
    ax2.set_xlabel("SLM x (px)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def wfs_figure(rec: dict, out_png: Path) -> None:
    """WFS readout figure: spots | read phase map / Zernike distribution | metrics."""
    spots, wf, z_um = rec["spots"], rec["wf"], rec.get("z_um")
    st = rec["stats"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    im = ax.imshow(spots, cmap="gray")
    fig.colorbar(im, ax=ax, shrink=0.85, label="8-bit")
    ax.set_title(f"WFS spots image — {rec['label']}")

    ax = axes[0, 1]
    im = ax.imshow(np.where(np.isfinite(wf), wf, np.nan), cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.85, label="wavefront (waves)")
    ax.set_title(f"WFS read phase map — RMS={st.get('rms', float('nan')):.4f}λ, "
                 f"PV={st.get('diff', float('nan')):.4f}λ")

    ax = axes[1, 0]
    if z_um is not None:
        z = np.asarray(z_um, dtype=float) / 0.532
        n = min(len(z), 16)
        idx = np.arange(1, n)
        vals = z[1:n]
        mx = float(np.max(np.abs(vals))) if len(vals) else 0.0
        ax.bar(idx, vals,
               color=["tab:red" if abs(v) == mx else "tab:blue" for v in vals])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{i}\n{DLL_ZERNIKE_NAMES.get(int(i), '')}"
                            for i in idx], fontsize=7)
        ax.set_ylabel("coefficient (λ)")
        ax.set_title("WFS Zernike readout (DLL order, λ)")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no zernike", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("WFS Zernike readout")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [rec["label"], ""]
    mode = rec.get("mode")
    lines.append(f"mode (n,m) : {'—' if mode is None else tuple(mode)}")
    lines.append(f"radius     : "
                 f"{'—' if rec['radius'] is None else str(int(rec['radius'])) + ' px'}")
    lines.append(f"amplitude  : "
                 f"{'—' if rec['amplitude'] is None else str(rec['amplitude']) + ' rad'}")
    pv = rec["phase_pv_rad"]
    lines.append(f"phase PV   : {'—' if pv is None else f'{pv:.1f} rad'}")
    lines.append("")
    lines.append(f"WFS RMS    : {st.get('rms', float('nan')):.4f} lambda")
    lines.append(f"WFS PV     : {st.get('diff', float('nan')):.4f} lambda")
    if z_um is not None:
        z = np.asarray(z_um, dtype=float) / 0.532
        lines += ["", f"tiltA[2]   : {z[2]:+.4f} lambda",
                  f"tiltB[3]   : {z[3]:+.4f} lambda",
                  f"defocus[5] : {z[5]:+.4f} lambda",
                  f"|z [2..6]| : {np.linalg.norm(z[2:7]):.4f} lambda", "",
                  "top |coeff|:"]
        for k in np.argsort(np.abs(z[1:16]))[::-1][:5] + 1:
            lines.append(f"  [{int(k):<2d}] {DLL_ZERNIKE_NAMES.get(int(k), ''):<14s} "
                         f"{z[k]:+.4f}")
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=11, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def capture(wfs: ThorlabWFS) -> dict:
    wfs.take_image(n_sample=1, dynamicNoiseCut=True)
    spots = wfs.get_spotfiled_image()
    wf, stats = wfs.get_wavefront(cancel_tile=False)
    z = measure_zernike(wfs, n_avg=1, order=WFS_ZERNIKE_ORDER)
    return {"spots": spots, "wf": wf, "stats": stats, "z_um": z}


def acquire(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper, out: Path,
            shift: tuple[int, int], settle_extra_s: float,
            max_gray: int) -> list[dict]:
    """Run every case and write both figures; returns the record list."""
    phase_dir, wfs_dir = out / "phase", out / "wfs"
    phase_dir.mkdir(parents=True, exist_ok=True)
    wfs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for i, (label, mode, radius, amp) in enumerate(CASES):
        try:
            if mode is None:
                slm.display_data(flat_gray(), wait_time_s=0.6)
                phase, gray = None, None
                time.sleep(0.4)
            else:
                assert amp is not None and radius is not None
                phase = make_phase(ph, {mode: float(amp)}, float(radius), n_max=6)
                show_phase(slm, phase, settle_extra_s)
                gray = slm.create_phase_from_array(phase)   # pure fn, no write
            cap = capture(wfs)
            rec = {"idx": i, "label": label,
                   "mode": None if mode is None else list(mode),
                   "radius": radius, "amplitude": amp,
                   "phase": phase, "gray": gray,
                   "phase_pv_rad": None if phase is None
                   else float(phase.max() - phase.min()), **cap}
            records.append(rec)
            phase_figure(rec, phase_dir / f"{i:02d}_{label}.png", max_gray, shift)
            wfs_figure(rec, wfs_dir / f"{i:02d}_{label}.png")
            z = (np.asarray(cap["z_um"], dtype=float) / 0.532
                 if cap["z_um"] is not None else None)
            msg = (f"RMS={cap['stats']['rms']:.4f}λ [2]={z[2]:+.4f} [3]={z[3]:+.4f} "
                   f"[5]={z[5]:+.4f} |z[2..6]|={np.linalg.norm(z[2:7]):.4f}λ"
                   if z is not None else f"RMS={cap['stats']['rms']:.4f}λ")
            print(f"[{i:02d}] {label:16s} {msg}")
        except Exception as e:  # noqa: BLE001 - keep going per case
            print(f"[WARN] case {label} failed: {type(e).__name__}: {e}")
    return records


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def write_markdown(out: Path, meta: dict, records: list[dict]) -> None:
    def f(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    summary = []
    for rec in records:
        z = (np.asarray(rec["z_um"], dtype=float) / 0.532
             if rec["z_um"] is not None else None)
        summary.append({
            "idx": rec["idx"], "label": rec["label"], "mode": rec["mode"],
            "radius": rec["radius"], "amplitude": rec["amplitude"],
            "phase_pv_rad": rec["phase_pv_rad"],
            "rms": float(rec["stats"].get("rms", float("nan"))),
            "pv": float(rec["stats"].get("diff", float("nan"))),
            "tilt_a": None if z is None else float(z[2]),
            "tilt_b": None if z is None else float(z[3]),
            "defocus": None if z is None else float(z[5]),
            "z_norm_2_6": None if z is None else float(np.linalg.norm(z[2:7])),
            "z_all": None if z is None else [float(v) for v in z[:16]],
        })

    (out / "data.json").write_text(
        json.dumps({"meta": meta, "cases": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    md: list[str] = []
    md.append("# Zernike 相位 → WFS 读数分布报告\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**设备**: SLM #{meta.get('slm_serial')} "
              f"({meta.get('wavelength_nm')}nm, 2π={meta.get('slm_2pi_gray')} gray) + "
              f"WFS {meta.get('wfs_serial')}  ")
    md.append(f"**SLM shift**: `{meta.get('shift')}`  ")
    md.append(f"**WFS 曝光**: {meta.get('wfs_exposure_ms', float('nan')):.3f} ms (≤ 7ms)  ")
    md.append("**WFS pupil**: center=(%.3f, %.3f) mm, diameter=(%.3f, %.3f) mm\n" %
              tuple(meta.get("pupil_center_mm", [float("nan")] * 2)
                    + meta.get("pupil_diameter_mm", [float("nan")] * 2)))
    md.append("## 1. 说明\n")
    md.append("- **SLM 相位**由 `PatternHelper.generate_zernike_polynomial()` 生成; "
              "**上屏图案**为 `create_phase_from_array()` 输出 (弧度→灰度 + mod 2π + 平移)。")
    md.append(f"- **WFS 读数** `get_zernike(order={WFS_ZERNIKE_ORDER})` → 67 长数组 "
              "(`z_um[i]`, index 0 未用), 单位 µm → λ (÷0.532)。"
              "`orders` 有效值 0=auto 或 2..10。")
    md.append("- **⚠️ 索引为 DLL 顺序 m 枚举 (非标准 Noll)**: "
              "`[5](2,0)defocus` `[9](3,1)coma` `[13](4,0)spherical`; "
              "手册佐证 `roCMm` \"derived from Zernike coefficient Z[5]\"。")
    md.append("- **参考平面**: 纯平相位用户参考 → 读数反映**相对纯平的增量**。\n")
    md.append("## 2. 汇总表\n")
    md.append("| # | 用例 | 模式 (n,m) | R(px) | A(rad) | 相位 PV(rad) | WFS RMS(λ) | "
              "WFS PV(λ) | [2](λ) | [3](λ) | [5]defocus(λ) | ‖z[2-6]‖(λ) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        mode = "—" if r["mode"] is None else f"({r['mode'][0]},{r['mode'][1]})"
        md.append(f"| {r['idx']} | `{r['label']}` | {mode} | {f(r['radius'], 0)} | "
                  f"{f(r['amplitude'], 1)} | {f(r['phase_pv_rad'], 1)} | {f(r['rms'])} | "
                  f"{f(r['pv'])} | {f(r['tilt_a'])} | {f(r['tilt_b'])} | "
                  f"{f(r['defocus'])} | {f(r['z_norm_2_6'])} |")
    md.append("")
    md.append("## 3. 各用例详情\n")
    for r in summary:
        mode = "—" if r["mode"] is None else f"({r['mode'][0]},{r['mode'][1]})"
        md.append(f"### 3.{r['idx']} `{r['label']}`\n")
        md.append(f"- 模式 **{mode}**, 尺寸 **{f(r['radius'], 0)} px**, "
                  f"幅度 **{f(r['amplitude'], 1)} rad**  ")
        md.append(f"- 相位 PV {f(r['phase_pv_rad'], 1)} rad; "
                  f"WFS RMS/PV {f(r['rms'])}λ / {f(r['pv'])}λ  ")
        md.append(f"- Zernike: [2]={f(r['tilt_a'])}, [3]={f(r['tilt_b'])}, "
                  f"[5]defocus={f(r['defocus'])}, ‖z[2-6]‖={f(r['z_norm_2_6'])}  \n")
        md.append("**① SLM 相位图** (左: 弧度源; 右: 实际上屏灰度):\n")
        md.append(f"![{r['label']} slm phase](phase/{r['idx']:02d}_{r['label']}.png)\n")
        md.append("**② WFS 读取图** (spots / 读取相位图 / Zernike 分布 / 指标):\n")
        md.append(f"![{r['label']} wfs](wfs/{r['idx']:02d}_{r['label']}.png)\n")
        if r["z_all"] is not None:
            z_all = r["z_all"]
            md.append("**Zernike 主导项 (|系数| 前 5)**:\n")
            md.append("| 索引 | 名称 (n,m) | 系数 (λ) |")
            md.append("|---|---|---|")
            for k in np.argsort(np.abs(np.asarray(z_all[1:])))[::-1][:5] + 1:
                md.append(f"| {k} | {DLL_ZERNIKE_NAMES.get(int(k), '')} | "
                          f"{z_all[k]:+.4f} |")
            md.append("")

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slm-number", type=int, default=1)
    ap.add_argument("--slm-wavelength", type=int, default=532)
    ap.add_argument("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    ap.add_argument("--shift-x", type=int, default=None,
                    help="SLM shift_x (default: read from device config)")
    ap.add_argument("--shift-y", type=int, default=None)
    ap.add_argument("--settle-extra-s", type=float, default=SETTLE_REDUNDANCY_S)
    ap.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    if args.wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise SystemExit(f"WFS exposure {args.wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[Zernike → WFS 分布报告] 带图 md")
    print("=" * 72)

    slm = Santec(slm_number=args.slm_number, wavelength=args.slm_wavelength,
                 video_mode=0)
    wfs = ThorlabWFS(exposure_time=args.wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))
    meta: dict = {}
    records: list[dict] = []

    try:
        slm.open()
        wfs.open()
        exp = float(wfs.exposure_time)
        assert exp <= MAX_EXPOSURE_MS
        wl, max_gray = slm.get_wavelength_info()
        sx = slm.shift_x if args.shift_x is None else args.shift_x
        sy = slm.shift_y if args.shift_y is None else args.shift_y

        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        slm.set_shift(sx, sy)
        slm.display_data(flat_gray(), wait_time_s=0.6)
        time.sleep(0.4)
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        ok_ref = wfs.create_default_user_ref()
        wfs.set_ref_plane(custom=True)
        print(f"[OK] SLM #{slm._serial_number} shift=({sx},{sy}); "
              f"WFS {wfs.serial_num} exp={exp:.3f}ms; "
              f"flat ref create={ok_ref} use_custom_ref={wfs.use_custom_ref}")

        meta = {"slm_serial": slm._serial_number, "wfs_serial": wfs.serial_num,
                "wavelength_nm": args.slm_wavelength, "slm_2pi_gray": max_gray,
                "wfs_exposure_ms": exp, "shift": [sx, sy],
                "pupil_center_mm": [cx, cy], "pupil_diameter_mm": [dx, dy],
                "zernike_order": WFS_ZERNIKE_ORDER,
                "reference": "custom user ref @ flat phase (shift applied)"}

        records = acquire(slm, wfs, ph, out, (sx, sy), args.settle_extra_s, max_gray)
        slm.display_data(flat_gray(), wait_time_s=0.5)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {type(e).__name__}: {e}")
        return 1
    finally:
        for dev in (wfs, slm):
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass

    write_markdown(out, meta, records)
    print(f"\n[OK] 报告: {out / 'report.md'}")
    print(f"[OK] 图: {out / 'phase'} | {out / 'wfs'} ({len(records)} 用例)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
