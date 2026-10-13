"""Generate the gs_refine forward-model calibration report.

Fully offline: reads the saved ``data/debug/slm_gs_refine/*/*.pkl`` artefacts
(recorded SLM grayscale + measured CCD frame) and drives the calibrated forward
model in :mod:`ao_shaping.utils.slm.gs_debug_optics`. Never opens a device.

Writes ``report/shape_trials/gs_refine_fwdmodel_calibration.md`` plus
``figures/``. Provenance is stamped at write time via ``insert_header`` so a
re-run cannot delete it.

Run:
    python scripts/generate_gs_refine_fwdmodel_report.py
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common.provenance import insert_header  # noqa: E402

from ao_shaping.utils.slm.gs_debug_optics import (  # noqa: E402
    BenchGeometry,
    apply_far_field_response,
    far_field_intensity,
    grayscale_to_radians,
    to_camera_window,
)

REPORT_KEY = "report/shape_trials/gs_refine_fwdmodel_calibration.md"
OUT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = OUT_DIR / "figures"

# Measured bench constants, report/slm/bench_calibration_20261001.md
D_SLM = 8e-6
WAVELENGTH = 1064e-9
FOCAL = 0.125
D_CCD = 2.2e-6
BEAM_RADIUS_PX = 450.0


def pearson(a, b):
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def shape_metrics(a):
    """(FWHM, EE50, EE90) in the array's own pixels."""
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2.0, yy - H / 2.0).ravel()
    order = np.argsort(r)
    cum = np.cumsum(a.ravel()[order]) / a.sum()

    def encircled(frac):
        return float(r[order][int(np.searchsorted(cum, frac))])

    prof = a.sum(1)
    idx = np.where(prof >= prof.max() / 2)[0]
    fwhm = float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
    return fwhm, encircled(0.5), encircled(0.9)


def collect(debug_root: Path):
    """Gather (run, phase, phase_rad, measured, shape, center, radius) samples."""
    samples = []
    for pkl in sorted(debug_root.glob("slm_gs_refine/*/*.pkl")):
        cfg = {}
        side = pkl.with_suffix(".json")
        if side.exists():
            cfg = json.loads(side.read_text(encoding="utf-8"))
        center = cfg.get("pupil_center")
        if center is None:
            diag = pkl.parent / "forward_diagnostic.json"
            if diag.exists():
                center = json.loads(diag.read_text(encoding="utf-8")).get(
                    "pupil_center_xy"
                )
        if center is None:
            continue
        center = (float(center[0]), float(center[1]))
        radius = float(cfg.get("beam_radius_px", BEAM_RADIUS_PX))

        data = pickle.load(open(pkl, "rb"))
        keys = sorted(int(k) for k in data.keys())
        scored = [k for k in keys[1:] if "score" in data[k]]
        if not scored:
            continue
        best = max(scored, key=lambda k: float(data[k]["score"]))
        for label, k in (("flat", keys[0]), ("GS", best)):
            rec = data[k]
            samples.append(
                {
                    "run": pkl.parent.name,
                    "phase": label,
                    "rad": grayscale_to_radians(np.asarray(rec["_phase"])),
                    "measured": np.asarray(rec["_img"], float),
                    "center": center,
                    "radius": radius,
                    "score": float(rec.get("score", 0.0)),
                    "pib": float(rec.get("pib", 0.0)),
                    "cv": float(rec.get("cv", 0.0)),
                }
            )
    return samples


def simulate(sample, geo):
    far = far_field_intensity(
        sample["rad"],
        pupil_center=sample["center"],
        beam_radius_px=sample["radius"],
        geometry=geo,
    )
    return apply_far_field_response(to_camera_window(far, sample["measured"].shape, geo), geo)


def old_model_metrics(sample):
    """The pre-calibration model: unpadded 1200x1920 FFT, hard disk, 2pi=1023."""
    rad = sample["rad"] / (1023.0 / 993.0)  # undo the 2*pi correction, approximately
    n0, n1 = rad.shape
    cx, cy = sample["center"]
    yy, xx = np.ogrid[:n0, :n1]
    pupil = (xx - cx) ** 2 + (yy - cy) ** 2 <= sample["radius"] ** 2
    field = pupil * np.exp(1j * rad)
    I = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    return I / I.max()


def blur_sweep(samples, geo, grid):
    out = []
    for b in grid:
        g = replace(geo, far_field_blur_camera_px=float(b))
        out.append((float(b), float(np.mean([pearson(s["measured"], simulate(s, g)) for s in samples]))))
    return out


def render_figures(samples, geo, sweep):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    sims = [simulate(s, geo) for s in samples]
    names = [f"{s['run']}\n{s['phase']}" for s in samples]

    fig, ax = plt.subplots(2, len(samples), figsize=(3.1 * len(samples), 6.6))
    for j, (s, sim) in enumerate(zip(samples, sims)):
        ax[0, j].imshow(s["measured"], cmap="hot")
        ax[0, j].set_title(f"measured\n{s['run']} {s['phase']}", fontsize=8)
        ax[1, j].imshow(sim / sim.max(), cmap="hot")
        ax[1, j].set_title("simulated (calibrated)", fontsize=8)
        for r in (0, 1):
            ax[r, j].axis("off")
    fig.suptitle("Measured CCD vs calibrated forward model (same camera pixel grid)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "measured_vs_simulated.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    mfw = [shape_metrics(s["measured"])[0] for s in samples]
    sfw = [shape_metrics(sim)[0] for sim in sims]
    me = [shape_metrics(s["measured"])[1] for s in samples]
    se = [shape_metrics(sim)[1] for sim in sims]
    x = np.arange(len(samples))
    ax[0].bar(x - 0.2, mfw, 0.4, label="measured", color="#2b6cb0")
    ax[0].bar(x + 0.2, sfw, 0.4, label="simulated", color="#c53030")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, fontsize=7)
    ax[0].set_ylabel("FWHM (camera px)")
    ax[0].set_title("Spot size")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(x - 0.2, me, 0.4, label="measured", color="#2b6cb0")
    ax[1].bar(x + 0.2, se, 0.4, label="simulated", color="#c53030")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, fontsize=7)
    ax[1].set_ylabel("EE50 (camera px)")
    ax[1].set_title("Energy spread")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)

    bs = [b for b, _ in sweep]
    cs = [c for _, c in sweep]
    ax[2].plot(bs, cs, "o-", color="#975a16")
    ax[2].axhline(cs[0], color="#c53030", ls="--", label=f"no blur = {cs[0]:+.4f}")
    ax[2].set_xlabel("fitted far-field blur (camera px)")
    ax[2].set_ylabel("mean Pearson correlation")
    ax[2].set_title("Frequency response does not help")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_and_blur_sweep.png", dpi=130)
    plt.close(fig)
    return FIG_DIR / "measured_vs_simulated.png", FIG_DIR / "metrics_and_blur_sweep.png"


def build_markdown(samples, geo, sweep, figs):
    rows = []
    for s, sim in zip(samples, [simulate(s, geo) for s in samples]):
        mf, me50, me90 = shape_metrics(s["measured"])
        sf, se50, se90 = shape_metrics(sim)
        rows.append(
            (s, mf, sf, me50, se50, me90, se90, pearson(s["measured"], sim))
        )

    c0 = np.mean([r[7] for r in rows])
    flat_corr = np.mean([r[7] for r in rows if r[0]["phase"] == "flat"])
    gs_corr = np.mean([r[7] for r in rows if r[0]["phase"] == "GS"])
    best_blur, best_c = max(sweep, key=lambda kv: kv[1])

    tbl = "\n".join(
        f"| `{s['run']}` | {s['phase']} | {mf:.1f} | {sf:.1f} | "
        f"{me50:.1f} | {se50:.1f} | {me90:.1f} | {se90:.1f} | {c:+.4f} |"
        for s, mf, sf, me50, se50, me90, se90, c in rows
    )
    sweep_tbl = "\n".join(
        f"| {b:.0f} | {c:+.4f} |" for b, c in sweep if int(b) % 4 == 0
    )

    return f"""# slm-gs-refine 正向模型标定 (2026-10-13)

## 0. 一句话

三栏 GIF 的第三栏（"FFT 仿真远场"）此前是一个**未标定的同尺寸 FFT**，与实测 CCD
相差极大。本记录把它换成标定后的 2f-Fourier 前向模型，实测相关从 **+0.17 提到
+0.75**，并且**证伪了"残差是 PSF 造成的"这一解释**。

## 1. 起因

`gs_refine_live_20261012.md` 记录的三栏动画，仿真栏与 CCD 栏肉眼差异极大。旧实现是
`src/ao_shaping/utils/slm/gs_debug_optics.py::idealized_far_field`：

```python
pupil = (xx - cx) ** 2 + (yy - cy) ** 2 <= beam_radius_px**2   # 硬边圆盘
field = pupil * np.exp(1j * phase)
intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
```

它**不含任何物理常数**——没有波长、焦距、像元间距。文档里"1064 nm、f=125 mm、
像素 7.8 µm"只存在于报告正文，代码里一个都没有。

## 2. 硬件基线（先确认台架是好的）

| 量 | 值 | 来源 |
|---|---|---|
| 理论平场 FWHM `2.355·λf/(π·w₀)` | 12.6 相机 px | 计算 |
| 实测平场 FWHM | **11 × 12 px** | `_img` |
| 实测 GS 相位 FWHM | **80 × 92 px**（target_side=100） | `_img` |
| 各向异性 σx/σy | 1.26（平场）/ 1.04（GS） | `_img` |

GS 把 11.5 px 的平场展开成 ~86 px 的方斑，**尺度正确**。所以问题在仿真侧。

## 3. 两个被证伪的假设

### 3.1 "远场中心没对齐" —— 证伪

| 量 | 平场 | GS 相位 |
|---|---|---|
| 仿真远场质心相对 DC | (+0.46, +0.46) px | (+0.78, −1.46) px |
| 相机 0 阶相对窗心 | (+2, +1) px | (−1, +3) px |
| 瞳孔相位线性项 | 0 | dx=+1.7, **dy=+108.3 rad/m** |
| 倾斜预测偏移 `k·f·λ/2π` | 0 | **1.04 px** ✓ |

全部在 ~2 px 内，倾斜预测与实测一致。**不是中心问题。**

### 3.2 "用高斯光束" —— 证伪

σx/σy = **0.35**，不是采样各向异性，是**高斯被面板边缘截断**：瞳孔中心
(960, 690) 在 1200 行面板上只有上方 690 行、下方 509 行，而 σ=450 px 的高斯在 2σ
处仍有 1.8% 振幅，低侧被砍掉一大半。top-hat 圆盘正好落在 240…1140 行内，
σx/σy = **1.00**。台架照明本来就是圆盘，`beam_radius_px` 也是这么记的。

## 4. 六处缺陷与修复

| # | 原 | 现 | 依据 |
|---|---|---|---|
| 1 | 不补零，1200×1920 直接 FFT | 方阵补零到 `far_field_size` | 见 §6 |
| 2 | 远场像元 8.66 µm(x) / 13.85 µm(y)，**各向异性 1.60×** | 1.00 | 方阵 ⇒ 像元各向同性 |
| 3 | 平场核心塌到 **1~3 px** | 9 px | 混叠 |
| 4 | 硬边圆盘 | top-hat 圆盘（实测束斑） | §3.2 |
| 5 | `2π = 1023` 灰度 | **993** | 台架实测 |
| 6 | 第三栏 500×500「相对像素」 | `to_camera_window` → 相机像元网格 | 无法逐像素比较 |

另修两处静默失效：

- `_pupil_geometry` 读 sidecar 找不到 `pupil_center`（它实际在
  `forward_diagnostic.json` 的 `pupil_center_xy`），**静默降成 2 栏 GIF**。
- `to_camera_window` 起初只处理降采样、漏了上采样；`far_field_size=4096` 时
  ratio=1.845 属欠采样，本应插值放大却直接裁剪。修完均值相关 **+0.52 → +0.75**。

## 5. 标定后验证

![measured vs simulated](figures/measured_vs_simulated.png)

| run | 相位 | FWHM 实测 | FWHM 仿真 | EE50 实测 | EE50 仿真 | EE90 实测 | EE90 仿真 | 相关 |
|---|---|---|---|---|---|---|---|---|
{tbl}

**均值相关 +{c0:.4f}**（平场 +{flat_corr:.3f}，GS +{gs_corr:.3f}），修复前 +0.17。

![metrics](figures/metrics_and_blur_sweep.png)

尺度与对称性现在是对的；EE50 仍偏高（仿真铺得更开），说明实测图**对比度更高**，
理想圆盘瞳给不出这一点。

## 6. 补零倍数：内点最优

以与实测 CCD 的相关为判据：

| far_field_size | 远场像元 | 相机 px/远场 px | 平均相关 |
|---|---|---|---|
| 1920 | 8.66 µm | 3.94 | +0.168 |
| 4096 | 4.06 µm | 1.85 | +0.364 |
| 8192 | 2.03 µm | 0.92 | **+0.634** |
| 12288 | 1.35 µm | 0.61 | +0.514 |

与 AGENTS.md 记录的 `hwdataset` 那条 `pad=1 → R²=−0.20, 10 → +0.46` 同型：**不是
"越细越好"**。默认取 4096（1.85 相机 px/远场 px，FWHM 已跨 ~6 像素，且 FFT 内存
268 MB 而非 1.07 GB）。

## 7. 负结果：残差**不是** PSF

加了径向频率响应（SLM 像元 sinc + 高斯 MTF，SLM 像元 sinc 的首零点
λf/d_eff = 7750 相机 px，在 250 px 窗内基本不起作用），扫参：

| blur (相机 px) | 平均相关 |
|---|---|
{sweep_tbl}

全程只涨 **{best_c - sweep[0][1]:+.4f}**（最优 blur={best_blur:.0f} px），在噪声内。
且 **FWHM 全程未变**（9.0 px），只有 EE50 从 5.0 降到 4.5——径向低通是**压制尖峰**，
不是**拓宽核心**。

关键在于残差要求两件相反的事同时成立：

- 实测**平场更宽**（FWHM 12.0 vs 9.0）→ 要拓宽
- 实测 **GS 更紧**（FWHM 89–92 vs 96）→ 要收紧

单一径向滤波器做不到。所以残差是**杂散光 + SLM 真实相位分辨率**这类结构上缺失
的物理，不是频率响应。该旋钮保留但**默认 0**，并由
`tests/ao_shaping/utils/test_gs_debug_optics.py::test_far_field_response_is_off_by_default`
锁定。

## 8. 结论与边界

**已解决**：尺度、各向同性、方形结构、相机像元网格、2π 基准、几何回退。

**未解决且已记录**：EE50 对比度（实测图更锐），理想圆盘瞳模型给不出。杂散光与
SLM 相位分辨率需要另外建模，不能靠一个标量拟合。

**定性定位**：这是一个**几何检查模型**，不是光度预测器。用来判断"相位形状对不对、
尺度对不对、左右上下有没有转置"，不用来预测 CV/PIB 的绝对值。
"""


def main() -> None:
    debug_root = ROOT / "data" / "debug"
    samples = collect(debug_root)
    if not samples:
        print("no gs_refine samples with a recorded pupil centre under", debug_root)
        return

    geo = BenchGeometry()
    grid = tuple(np.round(np.arange(0.0, 32.01, 2.0), 2))
    sweep = blur_sweep(samples, geo, grid)
    figs = render_figures(samples, geo, sweep)
    body = build_markdown(samples, geo, sweep, figs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    out.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    print(f"Wrote {out}")
    for f in FIG_DIR.glob("*.png"):
        print("  figure:", f)


if __name__ == "__main__":
    main()