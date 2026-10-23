# -*- coding: utf-8 -*-
"""Generate PPT figures for the shaping research report (2026-10-23).
All numbers come from on-disk experiment artifacts; no new measurements.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# ---------- CJK font ----------
for cand in ["Microsoft YaHei", "SimHei", "MS Gothic", "DejaVu Sans"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
        print("font:", cand)
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "D:/Projects/TIFO/AO-shaping/report/shape_trials/figures"
RUN = "D:/Projects/TIFO/AO-shaping/data/debug/differentiable_opt/20261022_213000"
FIGW, FIGH, DPI = 13.66, 6.0, 150  # aspect 2.277 == pptx image box 10.933x4.8 in

# ---------- 1. focus sweep ----------
rows = list(csv.DictReader(open(f"{RUN}/focus_sweep.csv", encoding="utf-8-sig")))
d = [float(r["diopter_m1"]) for r in rows]
fwhm = [float(r["fwhm_px"]) for r in rows]
ppm = [float(r["peak_per_ms"]) for r in rows]
fig, ax = plt.subplots(figsize=(FIGW, FIGH), dpi=DPI)
ax.plot(d, fwhm, "o-", color="#1f4e79", lw=2, ms=5, label="光斑 FWHM (CCD px)")
ax.axhline(40, color="#b22222", ls="--", lw=1.8, label="在焦门限 40 px")
bi = int(np.argmin(fwhm))
ax.plot([d[bi]], [fwhm[bi]], "*", color="#2e7d32", ms=22, zorder=5,
        label=f"选中点 +{d[bi]:.3f} m$^{{-1}}$ (f=1637.6 mm), FWHM {fwhm[bi]:.2f} px")
ax.annotate(f"FWHM {fwhm[bi]:.2f} px\n峰值/曝光 {ppm[bi]:.1f} counts/ms",
            xy=(d[bi], fwhm[bi]), xytext=(d[bi] + 0.9, fwhm[bi] + 55),
            fontsize=13, color="#2e7d32",
            arrowprops=dict(arrowstyle="->", color="#2e7d32"))
ax.set_xlabel("屈光度 1/f / m$^{-1}$", fontsize=14)
ax.set_ylabel("光斑 FWHM / CCD px", fontsize=14)
ax.set_title("在焦判定：焦面扫描（2026-10-22 可微运行，测量前门禁）", fontsize=16)
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(d, ppm, "s--", color="#888888", lw=1.2, ms=4, label="峰值/曝光 (counts/ms)")
ax2.set_ylabel("峰值/曝光 / (counts/ms)", fontsize=13, color="#555555")
ax2.tick_params(axis="y", labelcolor="#555555")
ax2.set_yscale("log")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/ppt_fig_focus_sweep.png")
plt.close(fig)
print("focus sweep ok, best diopter", d[bi], "fwhm", fwhm[bi], "ppm", ppm[bi])

# ---------- 2. loss history (model) ----------
hist = list(csv.DictReader(open(f"{RUN}/history.csv", encoding="utf-8-sig")))
it = [int(r["iteration"]) for r in hist]
loss = [float(r["loss"]) for r in hist]
prms = [float(r["phase_rms_rad"]) for r in hist]
floor = 31.0 ** 2 / 1024.0 ** 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, FIGH), dpi=DPI)
a1.semilogy(it, loss, "-", color="#1f4e79", lw=2)
a1.axhline(floor, color="#b22222", ls="--", lw=1.8,
           label=f"摊平下界 31²/1024² = {floor:.2e}")
a1.annotate(f"第 1 步: {loss[0]:.2e}", xy=(1, loss[0]), xytext=(25, loss[0] * 0.82),
            fontsize=12.5, arrowprops=dict(arrowstyle="->", color="#333333"))
a1.annotate(f"第 200 步: {loss[-1]:.2e}", xy=(200, loss[-1]), xytext=(95, loss[-1] * 2.2),
            fontsize=12.5, arrowprops=dict(arrowstyle="->", color="#333333"))
a1.set_xlabel("迭代", fontsize=13)
a1.set_ylabel("损失（峰值归一化 MSE，对数轴）", fontsize=13)
a1.set_title("纯模型 200 次 Adam：一步掉到摊平下界", fontsize=15)
a1.grid(alpha=0.3, which="both")
a1.legend(fontsize=11.5, loc="center right")
a2.plot(it, prms, "-", color="#2e7d32", lw=2)
a2.set_ylim(3.6290, 3.6300)
a2.set_xlabel("迭代", fontsize=13)
a2.set_ylabel("相位 RMS / rad", fontsize=13)
a2.set_title("相位 RMS 全程 3.63 rad（≈均匀随机相位，几乎不变）", fontsize=15)
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ppt_fig_loss_history.png")
plt.close(fig)
print("loss history ok", loss[0], loss[-1], "floor", floor)

# ---------- 3. CCD A/B (measured frames) ----------
flat = np.load(f"{RUN}/ccd_flat.npz")["frame"].astype(np.float64)
opt = np.load(f"{RUN}/ccd_optimized.npz")["frame"].astype(np.float64)


def boxstats(f, box=60):
    iy, ix = np.unravel_index(np.argmax(f), f.shape)
    r = 100
    ys, ye = max(0, iy - r), min(f.shape[0], iy + r + 1)
    xs, xe = max(0, ix - r), min(f.shape[1], ix + r + 1)
    win = f[ys:ye, xs:xe]
    yy, xx = np.mgrid[ys:ye, xs:xe]
    s = win.sum()
    cy, cx = int(round((win * yy).sum() / s)), int(round((win * xx).sum() / s))
    h = box // 2
    b = f[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
    return ix, iy, cx, cy, b.mean(), b.std() / (b.mean() + 1e-12), b.sum() / f.sum()


fx_, fy_, fcx, fcy, fmu, fcv, fee = boxstats(flat)
ox_, oy_, ocx, ocy, omu, ocv, oee = boxstats(opt)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, FIGH), dpi=DPI)
for ax, fr, ttl, (px, py, cx, cy, mu, cv, ee), vmax in [
    (a1, flat, "平场参考（2 ms，零相位+焦面补偿）", (fx_, fy_, fcx, fcy, fmu, fcv, fee), np.percentile(flat, 99.95)),
    (a2, opt, "“优化后”（72 ms，模型相位+焦面补偿）", (ox_, oy_, ocx, ocy, omu, ocv, oee), np.percentile(opt, 99.95)),
]:
    im = ax.imshow(fr, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
    ax.add_patch(plt.Rectangle((cx - 30, cy - 30), 60, 60, fill=False, ec="#00e5ff", lw=1.6))
    ax.plot(px, py, "c+", ms=14, mew=2)
    ax.set_title(ttl, fontsize=14.5)
    ax.set_xlabel("CCD x / px", fontsize=12)
    ax.set_ylabel("CCD y / px", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("counts", fontsize=11)
    ax.text(0.02, 0.02,
            f"峰值 {fr.max():.1f} counts（{fr.max()/ (2 if '平场' in ttl else 72):.1f} counts/ms）\n"
            f"框内(60 px)均值 {mu:.2f}，CV {cv:.2f}\n框内能量占比 {ee*100:.2f}%",
            transform=ax.transAxes, fontsize=11.5, color="white", va="bottom",
            bbox=dict(fc="black", alpha=0.55, pad=4))
fig.suptitle("CCD 实测对比：同一可微运行（2026-10-22，Santec SLM #1 + Daheng CCD）", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/ppt_fig_ccd_ab.png")
plt.close(fig)
print("ccd ab ok: flat peak", flat.max(), "cv", fcv, "| opt peak", opt.max(), "cv", ocv,
      "| offset", ocx - fcx, ocy - fcy)

# ---------- 4. pad existing figures to 2.277 aspect ----------
def pad(src, dst, bg=(255, 255, 255)):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    target = 10.933 / 4.8
    if w / h > target:
        nh = int(round(w / target))
        canvas = Image.new("RGB", (w, nh), bg)
        canvas.paste(im, (0, (nh - h) // 2))
    else:
        nw = int(round(h * target))
        canvas = Image.new("RGB", (nw, h), bg)
        canvas.paste(im, ((nw - w) // 2, 0))
    canvas.save(dst)
    print("padded", dst, canvas.size)


pad(f"{OUT}/defocus_hw_vs_model_20261022.png", f"{OUT}/ppt_fig_defocus_hw_vs_model.png")
pad(f"{OUT}/gs_issue_live_farfield_20261015_205654.png", f"{OUT}/ppt_fig_gs_farfield.png")
pad(f"{OUT}/mil_20261021_172309_overview.png", f"{OUT}/ppt_fig_mil_overview.png")
print("ALL DONE")
