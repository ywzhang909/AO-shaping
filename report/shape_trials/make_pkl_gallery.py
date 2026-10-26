# -*- coding: utf-8 -*-
"""
Visualize pkl contents under data/debug/physical_forward_defocus/.

For each of the 6 continuation directories referenced in
physical_forward_sampling_20261022_r2.md (the "接续" table):
  - sampling_audit_20261022_r1      -> diagnostic_iterations.pkl
  - response_bias_20261022_r2       -> capture_records.pkl
  - radiometry_audit_20261022_r2    -> diagnostic_iterations.pkl
  - response_repeat_20261022_r3     -> capture_records.pkl
  - uniformity_refit_20261022_r1    -> iterations.pkl (151 snapshots)
  - fixed_phase_repeat_20261022_r4  -> capture_records.pkl

We render:
  * SLM panel total phase (unwrapped, rad)  -> "SLM相位"
  * wrapped phase (mod 2pi, rad)            -> "相位"
  * CCD far-field spot (mean raw frame)     -> "CCD远场光斑"
For pkls that store multiple iteration steps we also draw metric line charts.
"""
import os
import pickle
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = "D:/Projects/TIFO/AO-shaping/data/debug/physical_forward_defocus"
OUT = "D:/Projects/TIFO/AO-shaping/report/shape_trials/pkl_visualizations"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 10,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def save(fig, path):
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def wrap(phase):
    return np.mod(phase, 2 * np.pi)


def render_phase_composite(
    out_path, phase, ccd, title_extra="", phase_crop=None, ccd_crop=None, predicted=None
):
    """3 or 4 panel composite: SLM phase unwrapped / wrapped / CCD spot [/ predicted]."""
    if phase_crop is not None:
        phase = phase[phase_crop[0] : phase_crop[1], phase_crop[2] : phase_crop[3]]
    if ccd_crop is not None:
        ccd = ccd[ccd_crop[0] : ccd_crop[1], ccd_crop[2] : ccd_crop[3]]
    ncol = 4 if predicted is not None else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.0))
    # SLM phase unwrapped
    pmax = np.nanpercentile(np.abs(phase), 99) if phase.size else 1
    if pmax <= 0:
        pmax = 1
    im0 = axes[0].imshow(phase, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-pmax, vmax=pmax))
    axes[0].set_title("SLM phase\n(unwrapped, rad)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # wrapped
    w = wrap(phase)
    im1 = axes[1].imshow(w, cmap="twilight", vmin=0, vmax=2 * np.pi)
    axes[1].set_title("Phase\n(wrapped mod 2pi, rad)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # CCD spot (log scaled)
    ccd_pos = np.clip(ccd.astype(np.float64), 0, None)
    if ccd_pos.max() > 0:
        im2 = axes[2].imshow(np.log1p(ccd_pos), cmap="inferno")
    else:
        im2 = axes[2].imshow(ccd_pos, cmap="inferno")
    axes[2].set_title("CCD far-field spot\n(log1p of mean frame)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    if predicted is not None:
        if ccd_crop is not None:
            predicted = predicted[ccd_crop[0] : ccd_crop[1], ccd_crop[2] : ccd_crop[3]]
        pp = np.clip(predicted.astype(np.float64), 0, None)
        if pp.max() > 0:
            im3 = axes[3].imshow(np.log1p(pp), cmap="inferno")
        else:
            im3 = axes[3].imshow(pp, cmap="inferno")
        axes[3].set_title("Model prediction\n(log1p)")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title_extra, fontsize=10)
    return save(fig, out_path)


def crop_box_from_centroid(cx, cy, half=350, H=1944, W=2592):
    x0 = max(0, int(cx) - half)
    x1 = min(W, int(cx) + half)
    y0 = max(0, int(cy) - half)
    y1 = min(H, int(cy) + half)
    return (y0, y1, x0, x1)


# ----------------------------------------------------------------------------
# GROUP A: capture_records.pkl directories (phase + CCD spot images)
# ----------------------------------------------------------------------------
cap_specs = {
    "fixed_phase_repeat_20261022_r4": "U/C 同相位同曝光重复参考（区分测量重复性与模型误差）",
    "response_bias_20261022_r2": "多曝光逐帧响应偏置补测（含超时组）",
    "response_repeat_20261022_r3": "U 分支固定曝光重复采集（短时响应稳定性）",
}

gal_capture = {}
for dname, desc in cap_specs.items():
    dpath = os.path.join(BASE, dname)
    recs = load_pkl(os.path.join(dpath, "capture_records.pkl"))
    # choose representative: first U and first C (by branch)
    chosen = {}
    for rec in recs:
        b = rec.get("branch")
        if b not in chosen:
            chosen[b] = rec
    imgs = []
    for b, rec in chosen.items():
        npz_path = os.path.join(dpath, rec["file"])
        d = np.load(npz_path, allow_pickle=True)
        panel_phase = d["panel_phase"].astype(np.float64)
        frames = d["raw_frames"].astype(np.float64)
        mean_frame = frames.mean(axis=0)
        # CCD crop around settled centroid if available
        settle = rec.get("settled", {}) or {}
        cx = settle.get("centroid_x", 1424)
        cy = settle.get("centroid_y", 972)
        # panel crop active region (960,600) r450
        pcrop = (150, 1050, 510, 1410)
        ccd_crop = crop_box_from_centroid(cx, cy, half=350)
        stem = f"{dname}__{b}_frac{rec.get('fraction')}_exp{rec.get('exposure_ms')}ms"
        outp = os.path.join(OUT, dname)
        os.makedirs(outp, exist_ok=True)
        title = (
            f"{dname}  branch={b}  fraction={rec.get('fraction')}  "
            f"exposure={rec.get('exposure_ms')} ms  peak={rec.get('peak')}"
        )
        p = render_phase_composite(
            os.path.join(outp, stem + ".png"),
            panel_phase,
            mean_frame,
            title,
            phase_crop=pcrop,
            ccd_crop=ccd_crop,
        )
        imgs.append((b, rec, os.path.relpath(p, OUT)))
    gal_capture[dname] = (desc, imgs)
    print(f"[capture] {dname}: {len(recs)} records -> {len(imgs)} composites")

# ----------------------------------------------------------------------------
# GROUP B1: sampling_audit diagnostic_iterations.pkl  (8 blur steps)
# ----------------------------------------------------------------------------
sa = load_pkl(
    os.path.join(BASE, "sampling_audit_20261022_r1/diagnostic_iterations.pkl")
)
sigmas = [r["sigma_ccd_px"] for r in sa]
tscore = [r["training_score"] for r in sa]
gains = [r["gain"] for r in sa]
backs = [r["background"] for r in sa]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(sigmas, tscore, "o-", color="#1f77b4")
ax1.set_xlabel("Gaussian blur sigma (CCD px)")
ax1.set_ylabel("Training score")
ax1.set_title("sampling_audit: training score vs effective blur")
ax2.plot(sigmas, gains, "s-", color="#2ca02c", label="gain")
ax2.plot(sigmas, backs, "^--", color="#d62728", label="background")
ax2.set_xlabel("Gaussian blur sigma (CCD px)")
ax2.set_ylabel("fitted value")
ax2.set_title("sampling_audit: gain & background vs blur")
ax2.legend()
fig.tight_layout()
sa_metrics_png = os.path.join(OUT, "sampling_audit_20261022_r1", "metrics.png")
os.makedirs(os.path.dirname(sa_metrics_png), exist_ok=True)
save(fig, sa_metrics_png)
print(f"[diag] sampling_audit: {len(sa)} steps -> {sa_metrics_png}")

# sampling_audit measured vs predicted montage (one stage, blur_04) bonus
blur_dirs = sorted(glob.glob(os.path.join(BASE, "sampling_audit_20261022_r1/blur_*")))
if blur_dirs:
    arr = np.load(os.path.join(blur_dirs[4], "arrays.npz"), allow_pickle=True)
    meas = arr["measured"]
    pred = arr["predicted"]
    n = min(6, meas.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    for i in range(n):
        m = np.clip(meas[i], 0, None)
        p = np.clip(pred[i], 0, None)
        axes[0, i].imshow(np.log1p(m), cmap="inferno")
        axes[0, i].set_title(f"meas {i}")
        axes[1, i].imshow(np.log1p(p), cmap="inferno")
        axes[1, i].set_title(f"pred {i}")
        for a in axes[:, i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("sampling_audit blur_04: measured (top) vs predicted (bottom)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sa_montage_png = os.path.join(OUT, "sampling_audit_20261022_r1", "meas_pred_montage.png")
    save(fig, sa_montage_png)
    print(f"[diag] sampling_audit montage -> {sa_montage_png}")

# ----------------------------------------------------------------------------
# GROUP B2: radiometry_audit diagnostic_iterations.pkl (14 stages)
# ----------------------------------------------------------------------------
ra = load_pkl(
    os.path.join(BASE, "radiometry_audit_20261022_r2/diagnostic_iterations.pkl")
)


def sigma_from_stage(s):
    # e.g. "single_sigma_2" or "mixture_sigma_3"
    return float(s.split("_")[-1])


single_sig, single_score, single_coef = [], [], []
mix_sig, mix_score, mix_coef = [], [], []
for r in ra:
    s = float(sigma_from_stage(r["stage"]))
    sc = r["training_score"]
    co = r["coefficients"]
    if r["stage"].startswith("single"):
        single_sig.append(s)
        single_score.append(sc)
        single_coef.append(co)
    else:
        mix_sig.append(s)
        mix_score.append(sc)
        mix_coef.append(co)
order_s = np.argsort(single_sig)
order_m = np.argsort(mix_sig)
single_sig = [single_sig[i] for i in order_s]
single_score = [single_score[i] for i in order_s]
single_coef = [single_coef[i] for i in order_s]
mix_sig = [mix_sig[i] for i in order_m]
mix_score = [mix_score[i] for i in order_m]
mix_coef = [mix_coef[i] for i in order_m]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(single_sig, single_score, "o-", color="#1f77b4", label="single-component")
ax1.plot(mix_sig, mix_score, "s--", color="#ff7f0e", label="mixture (unmodulated template)")
ax1.set_xlabel("Gaussian blur sigma (CCD px)")
ax1.set_ylabel("Training score")
ax1.set_title("radiometry_audit: training score vs blur")
ax1.legend()
# mixture coefficients breakdown (signal, template, constant)
mix_coef = np.array(mix_coef)  # shape (n,3) typically [main, template, const]
if mix_coef.ndim == 2 and mix_coef.shape[1] >= 3:
    ax2.plot(mix_sig, mix_coef[:, 0], "o-", label="main signal coef")
    ax2.plot(mix_sig, mix_coef[:, 1], "s--", label="template coef")
    ax2.plot(mix_sig, mix_coef[:, 2], "^:", label="constant coef")
    ax2.set_ylabel("coefficient")
    ax2.legend()
else:
    ax2.plot(mix_sig, [c[0] for c in mix_coef], "o-")
    ax2.set_ylabel("first coefficient")
ax2.set_xlabel("Gaussian blur sigma (CCD px)")
ax2.set_title("radiometry_audit: mixture coefficients vs blur")
fig.tight_layout()
ra_metrics_png = os.path.join(OUT, "radiometry_audit_20261022_r2", "metrics.png")
os.makedirs(os.path.dirname(ra_metrics_png), exist_ok=True)
save(fig, ra_metrics_png)
print(f"[diag] radiometry_audit: {len(ra)} stages -> {ra_metrics_png}")

# ----------------------------------------------------------------------------
# GROUP C: uniformity_refit iterations.pkl (151 snapshots) + phase/CCD images
# ----------------------------------------------------------------------------
it = load_pkl(os.path.join(BASE, "uniformity_refit_20261022_r1/iterations.pkl"))
iters = [r["iteration"] for r in it]
loss = [r["loss"] for r in it]
nmse = [r["nmse"] for r in it]
cvmse = [r["cv_mse"] for r in it]


def get_param(key, sub=None):
    out = []
    for r in it:
        p = r["parameters"]
        if sub is None:
            out.append(p[key])
        else:
            out.append(p[key][sub])
    return out


defocus = get_param("defocus_diopter")
sigma = get_param("sigma_ccd_px")
gain = get_param("gain")
back = get_param("background")
amp_cx = get_param("amplitude_center", 0)
amp_cy = get_param("amplitude_center", 1)
amp_wx = get_param("amplitude_width", 0)
amp_wy = get_param("amplitude_width", 1)
resid = [
    np.sqrt(
        np.mean(
            np.square(np.array(list(r["parameters"]["residual_noll_radians"].values())))
        )
    )
    for r in it
]

# fig 1: loss / nmse / cvmse
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2))
axa.plot(iters, loss, color="#1f77b4", label="loss")
axa.plot(iters, nmse, color="#2ca02c", label="nmse")
axa.set_yscale("log")
axa.set_xlabel("iteration")
axa.set_ylabel("loss / nmse (log)")
axa.set_title("uniformity_refit: joint objective")
axa.legend()
axb.plot(iters, cvmse, color="#d62728", label="cv_mse")
axb.set_yscale("log")
axb.set_xlabel("iteration")
axb.set_ylabel("cv_mse (log)")
axb.set_title("uniformity_refit: CV squared error")
fig.tight_layout()
ur_loss_png = os.path.join(OUT, "uniformity_refit_20261022_r1", "metrics_loss.png")
os.makedirs(os.path.dirname(ur_loss_png), exist_ok=True)
save(fig, ur_loss_png)

# fig 2: parameter trajectories
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes[0, 0].plot(iters, defocus, color="#1f77b4")
axes[0, 0].set_title("defocus (diopter)")
axes[0, 1].plot(iters, sigma, color="#ff7f0e")
axes[0, 1].set_title("effective blur sigma (CCD px)")
axes[0, 2].plot(iters, gain, color="#2ca02c")
axes[0, 2].set_title("gain")
axes[1, 0].plot(iters, back, color="#d62728")
axes[1, 0].set_title("background")
axes[1, 1].plot(iters, resid, color="#9467bd")
axes[1, 1].set_title("residual Noll RMS (rad)")
axes[1, 2].plot(iters, amp_wx, color="#8c564b", label="width x")
axes[1, 2].plot(iters, amp_wy, color="#17becf", label="width y")
axes[1, 2].set_title("illumination amplitude width")
axes[1, 2].legend()
for ax in axes.ravel():
    ax.set_xlabel("iteration")
fig.suptitle("uniformity_refit: fitted parameter trajectories")
fig.tight_layout(rect=(0, 0, 1, 0.96))
ur_param_png = os.path.join(OUT, "uniformity_refit_20261022_r1", "metrics_params.png")
save(fig, ur_param_png)
print(f"[iter] uniformity_refit: {len(it)} snapshots -> loss & param charts")

# phase / CCD images for a couple of representative probes
inp = np.load(
    os.path.join(BASE, "uniformity_refit_20261022_r1/inputs.npz"), allow_pickle=True
)
pred = np.load(
    os.path.join(BASE, "uniformity_refit_20261022_r1/prediction.npz"), allow_pickle=True
)
phases = inp["phases"]
measured = inp["measured"]
predicted = pred["predicted"]
nprobe = phases.shape[0]
# pick probe with largest measured peak and one with smallest, for contrast
peaks = measured.reshape(nprobe, -1).max(axis=1)
i_hi = int(np.argmax(peaks))
i_lo = int(np.argmin(peaks))
ur_imgs = []
for tag, idx in [("peak_max", i_hi), ("peak_min", i_lo)]:
    ph = phases[idx]
    me = np.clip(measured[idx], 0, None)
    pr = predicted[idx]
    outp = os.path.join(OUT, "uniformity_refit_20261022_r1")
    os.makedirs(outp, exist_ok=True)
    title = f"uniformity_refit probe[{idx}]  measured peak={peaks[idx]:.2f}"
    p = render_phase_composite(
        os.path.join(outp, f"probe_{tag}.png"), ph, me, title, predicted=pr
    )
    ur_imgs.append((tag, idx, os.path.relpath(p, OUT)))
print(f"[iter] uniformity_refit probe images: {ur_imgs}")

# ----------------------------------------------------------------------------
# Build HTML gallery
# ----------------------------------------------------------------------------


def rel(p):
    return os.path.relpath(p, OUT).replace("\\", "/")


html = []
html.append(
    """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>physical_forward_defocus · PKL 可视化画廊</title>
<style>
 body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#1f2937;}
 header{background:#0f172a;color:#fff;padding:22px 28px;}
 header h1{margin:0;font-size:20px;} header p{margin:6px 0 0;opacity:.8;font-size:13px;}
 main{max-width:1180px;margin:0 auto;padding:24px;}
 section{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:22px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
 h2{font-size:17px;margin:0 0 4px;color:#0f172a;border-left:4px solid #2563eb;padding-left:10px;}
 .desc{color:#6b7280;font-size:13px;margin:0 0 14px;}
 .cap{font-size:12px;color:#6b7280;margin:4px 0 12px;}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;}
 .card img{width:100%;display:block;}
 .card .meta{padding:8px 10px;font-size:12px;color:#374151;}
 .row{margin-bottom:18px;}
 .chart img{width:100%;border:1px solid #e5e7eb;border-radius:8px;}
</style></head><body>
<header><h1>相机主动离焦 · 物理预测模型 PKL 可视化画廊</h1>
<p>数据根目录：<code>data/debug/physical_forward_defocus/</code> · 对应报告 <code>physical_forward_defocus_live_20261022.md</code> 与 <code>physical_forward_sampling_20261022_r2.md</code> 的"接续"目录表。</p></header>
<main>"""
)

# capture sections
cap_notes = {
    "fixed_phase_repeat_20261022_r4": "U 分支为平相位基线（panel_phase≈0 属预期），C 分支为平相位 + 菲涅尔聚焦补偿；两者同曝光重复采集。",
}
for dname, (desc, imgs) in gal_capture.items():
    html.append(f'<section><h2>{dname}</h2><p class="desc">{desc}</p>')
    if dname in cap_notes:
        html.append(f'<p class="cap">说明：{cap_notes[dname]}</p>')
    html.append(
        '<p class="cap">每个卡片为「SLM 面板总相位（展开, rad）｜相位包裹图（mod 2π）｜CCD 远场光斑（平均帧 log 缩放）」三联图。</p>'
    )
    html.append('<div class="grid">')
    for b, rec, rp in imgs:
        meta = (
            f'branch={b} · fraction={rec.get("fraction")} · '
            f'exposure={rec.get("exposure_ms")} ms · peak={rec.get("peak")} counts'
        )
        html.append(
            f'<div class="card"><img src="{rel(os.path.join(OUT, rp))}">'
            f'<div class="meta">{meta}</div></div>'
        )
    html.append('</div></section>')

# sampling_audit
html.append(
    '<section><h2>sampling_audit_20261022_r1</h2>'
    '<p class="desc">传播采样 / 像素积分 / 有效展宽诊断（8 个模糊步）。'
    'diagnostic_iterations.pkl 记录每一步的训练得分、增益与背景。</p>'
)
html.append('<div class="row chart"><img src="%s"></div>' % rel(sa_metrics_png))
if "sa_montage_png" in dir():
    html.append('<div class="row chart"><img src="%s"></div>' % rel(sa_montage_png))
html.append('</section>')

# radiometry_audit
html.append(
    '<section><h2>radiometry_audit_20261022_r2</h2>'
    '<p class="desc">逐像素仿射截距与展宽/模板对照（14 个阶段：单分量展宽 + 非相干混合模板）。'
    'diagnostic_iterations.pkl 记录训练得分与混合系数。</p>'
)
html.append('<div class="row chart"><img src="%s"></div>' % rel(ra_metrics_png))
html.append('</section>')

# uniformity_refit
html.append(
    '<section><h2>uniformity_refit_20261022_r1</h2>'
    '<p class="desc">联合强度 / CV 目标的物理预测模型微调，共 151 个迭代快照（含初值）。'
    'iterations.pkl 记录 loss / nmse / cv_mse 与全部物理参数。</p>'
)
html.append('<div class="row chart"><img src="%s"></div>' % rel(ur_loss_png))
html.append('<div class="row chart"><img src="%s"></div>' % rel(ur_param_png))
html.append(
    '<p class="cap">下方为代表性探针的「SLM 相位（展开）｜相位包裹｜CCD 实测 ｜模型预测」四联图（log 缩放）。</p>'
)
html.append('<div class="grid">')
for tag, idx, rp in ur_imgs:
    html.append(
        f'<div class="card"><img src="{rel(os.path.join(OUT, rp))}">'
        f'<div class="meta">probe[{idx}] ({tag})</div></div>'
    )
html.append('</div></section>')

html.append("</main></body></html>")

with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
    f.write("\n".join(html))
print("HTML written:", os.path.join(OUT, "index.html"))
print("DONE")
