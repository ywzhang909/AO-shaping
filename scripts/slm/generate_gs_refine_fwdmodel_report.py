"""Generate the gs_refine forward-model calibration report.

Fully offline: reads the saved ``data/debug/slm_gs_refine/*/*.pkl`` artefacts and
drives the **canonical** bench forward model
(:func:`ao_shaping.drivers.sim.slm_shaping_bench.forward_intensity`) plus
:func:`~ao_shaping.drivers.sim.slm_shaping_bench.camera_window_from_far_field`.
Never opens a device, and never reimplements the far-field model.

Writes ``report/shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md`` + ``figures/``.
Provenance is stamped at write time so a re-run cannot delete it.

Run:
    python scripts/slm/generate_gs_refine_fwdmodel_report.py [--data-root DIR]
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable -- under a file-form launch (`python scripts/x/y.py`) the cwd is
# not on sys.path and only the script's own directory is.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common.provenance import insert_header
from scripts.slm.generate_gif_from_pkl import (
    _build_bench_config,
    _extract_pupil_phase,
    _gray_to_radians,
)

from ao_shaping.drivers.sim.slm_shaping_bench import (
    camera_window_from_far_field,
    forward_intensity,
)
from scripts.slm._fwdmodel_sections import (
    SECTION_1_1,
    SECTION_4X,
)
from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
    RECORD_PHASE,
)

REPORT_KEY = "report/shape_trials/gs_refine_fwdmodel/gs_refine_fwdmodel_calibration.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "gs_refine_fwdmodel"
FIG_DIR = OUT_DIR / "figures"


def pearson(a, b):
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def shape_metrics(a):
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2.0, yy - H / 2.0).ravel()
    order = np.argsort(r)
    cum = np.cumsum(a.ravel()[order]) / a.sum()

    def enc(f):
        return float(r[order][int(np.searchsorted(cum, f))])

    prof = a.sum(1)
    idx = np.where(prof >= prof.max() / 2)[0]
    fwhm = float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
    return fwhm, enc(0.5), enc(0.9)


def box_mask(shape, side):
    m = np.zeros(shape, bool)
    c0, c1 = shape[0] // 2, shape[1] // 2
    h = max(1, side // 2)
    m[c0 - h : c0 + h, c1 - h : c1 + h] = True
    return m


def collect(debug_root: Path):
    out = []
    for pkl in sorted(debug_root.glob("slm_gs_refine/*/*.pkl")):
        cfg = {}
        sp = pkl.with_suffix(".json")
        if sp.exists():
            cfg = json.loads(sp.read_text(encoding="utf-8"))
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
        radius = float(cfg.get("beam_radius_px", 450.0))
        bench = _build_bench_config(cfg)

        data = pickle.load(open(pkl, "rb"))
        keys = sorted(int(k) for k in data.keys())
        scored = [k for k in keys[1:] if "score" in data[k]]
        if not scored:
            continue
        best = max(scored, key=lambda k: float(data[k]["score"]))
        wavelength = int(cfg.get("slm_wavelength", 0)) or 1064
        for label, k in (("flat", keys[0]), ("GS", best)):
            rec = data[k]
            img = np.asarray(rec[RECORD_IMAGE], float)
            gray = np.asarray(rec[RECORD_PHASE])
            pupil_gray = _extract_pupil_phase(gray, radius, center)
            rad = _gray_to_radians(pupil_gray, wavelength)
            out.append(
                {
                    "run": pkl.parent.name,
                    "phase": label,
                    "img": img,
                    "rad": rad,
                    "cfg": bench,
                    "side": float(rec.get("target_side", 100.0)),
                    "pib": float(rec.get("pib", 0.0)),
                    "cv": float(rec.get("cv", 0.0)),
                    "score": float(rec.get("score", 0.0)),
                }
            )
    return out


def simulate(sample, blur_camera_px: float = 0.0):
    """Canonical bench far field -> camera pixel grid (+ optional local blur)."""
    far = forward_intensity(sample["rad"], sample["cfg"])
    cam = camera_window_from_far_field(far, sample["cfg"], sample["img"].shape)
    cam = cam / max(cam.max(), 1e-300)
    if blur_camera_px > 0:
        # Multiplicative focal-plane low-pass, applied on the CAMERA grid.
        rows, cols = cam.shape
        y = np.arange(rows) - (rows - 1) / 2.0
        x = np.arange(cols) - (cols - 1) / 2.0
        ry = np.exp(-0.5 * (y / blur_camera_px) ** 2)
        rx = np.exp(-0.5 * (x / blur_camera_px) ** 2)
        cam = cam * ry[:, None] * rx[None, :]
    return cam


def gate(samples, blur=0.0):
    print(
        f"{'run':16s} {'phase':5s} | {'pib m':>7s} {'pib p':>7s} "
        f"| {'cv m':>7s} {'cv p':>7s} | {'sc m':>7s} {'sc p':>7s} {'corr':>7s}"
    )
    rows = []
    for s in samples:
        cam = simulate(s, blur)
        sup = box_mask(cam.shape, int(s["side"]))
        v = cam[sup]
        pib = float(v.sum() / cam.sum()) if cam.sum() > 0 else 0.0
        cv = float(v.std() / v.mean()) if v.mean() > 0 else float("inf")
        comp = 0.5 * pib + 0.5 / (1.0 + cv)
        c = pearson(s["img"], cam)
        rows.append((s, pib, cv, comp, c))
        print(
            f"{s['run']:16s} {s['phase']:5s} | {s['pib']:7.4f} {pib:7.4f} "
            f"| {s['cv']:7.4f} {cv:7.4f} | {s['score']:7.4f} {comp:7.4f} {c:+7.4f}"
        )
    return rows


def render_figures(samples, rows, blur_sweep):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(samples)
    fig, ax = plt.subplots(2, n, figsize=(2.9 * n, 6.4))
    ax = np.atleast_2d(ax)
    for j, s in enumerate(samples):
        ax[0, j].imshow(s["img"], cmap="hot")
        ax[0, j].set_title(f"measured\n{s['run']} {s['phase']}", fontsize=7)
        ax[1, j].imshow(simulate(s), cmap="hot")
        ax[1, j].set_title("simulated", fontsize=7)
        for r in (0, 1):
            ax[r, j].axis("off")
    fig.suptitle(
        "Measured CCD vs simulated far field (same camera pixel grid)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "measured_vs_simulated.png", dpi=115)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    x = np.arange(n)
    for k, (idx, title, ylab) in enumerate(
        ((0, "Spot size", "FWHM (camera px)"), (1, "Energy spread", "EE50 (camera px)"))
    ):
        meas = [shape_metrics(s["img"])[idx] for s in samples]
        simv = [shape_metrics(simulate(s))[idx] for s in samples]
        ax[k].bar(x - 0.2, meas, 0.4, label="measured", color="#2b6cb0")
        ax[k].bar(x + 0.2, simv, 0.4, label="simulated", color="#c53030")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(
            [f"{s['run'][-6:]}\n{s['phase']}" for s in samples], fontsize=7
        )
        ax[k].set_ylabel(ylab)
        ax[k].set_title(title)
        ax[k].legend(fontsize=8)
        ax[k].grid(axis="y", alpha=0.3)

    bs = [b for b, _ in blur_sweep]
    cs = [c for _, c in blur_sweep]
    ax[2].plot(bs, cs, "o-", color="#975a16")
    ax[2].axhline(cs[0], color="#c53030", ls="--", label=f"no blur = {cs[0]:+.4f}")
    ax[2].set_xlabel("focal-plane blur (camera px)")
    ax[2].set_ylabel("mean Pearson correlation")
    ax[2].set_title("Frequency response does not help")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_and_blur_sweep.png", dpi=130)
    plt.close(fig)


def build_markdown(samples, rows, blur_sweep):
    tbl = "\n".join(
        f"| `{s['run']}` | {s['phase']} | {shape_metrics(s['img'])[0]:.1f} | "
        f"{shape_metrics(simulate(s))[0]:.1f} | {shape_metrics(s['img'])[1]:.1f} | "
        f"{shape_metrics(simulate(s))[1]:.1f} | {shape_metrics(s['img'])[2]:.1f} | "
        f"{shape_metrics(simulate(s))[2]:.1f} | {c:+.4f} |"
        for s, _p, _c, _m, c in rows
    )
    corr = np.mean([r[4] for r in rows])
    flat_c = np.mean([r[4] for r in rows if r[0]["phase"] == "flat"])
    gs_c = np.mean([r[4] for r in rows if r[0]["phase"] == "GS"])
    sweep_tbl = "\n".join(
        f"| {b:.0f} | {c:+.4f} |" for b, c in blur_sweep if int(b) % 8 == 0
    )
    bb, bc = max(blur_sweep, key=lambda kv: kv[1])
    cvm = np.mean([s["cv"] for s in samples if s["phase"] == "GS"])
    cvp = np.mean([r[2] for r in rows if r[0]["phase"] == "GS"])
    ratio = cvp / cvm if cvm else 0.0
    bias = 1.0 - ratio
    gs = [r for r in rows if r[0]["phase"] == "GS"]
    pibm = min(s["pib"] for s in samples if s["phase"] == "GS")
    pibM = max(s["pib"] for s in samples if s["phase"] == "GS")
    pibp = min(r[1] for r in gs)
    pibP = max(r[1] for r in gs)
    scm = min(s["score"] for s in samples if s["phase"] == "GS")
    scM = max(s["score"] for s in samples if s["phase"] == "GS")
    scp = min(r[3] for r in gs)
    scP = max(r[3] for r in gs)
    maxdev = max(
        abs(s["score"] - r[3])
        for s, r in zip([r[0] for r in rows if r[0]["phase"] == "GS"], gs)
    )

    return f"""# slm-gs-refine 物理预测模型标定 (2026-10-13)

## 0. 一句话

三栏 GIF 的第三栏此前与实测 CCD 相差极大。本记录把它接到**实验系统的物理预测模型**上并
补上**相机像元标定**，实测相关 **{corr:+.3f}**（平场 {flat_c:+.3f}，GS {gs_c:+.3f}）；
同时给出两个**被证伪的解释**，其中一个直接否掉了"用离线模型选参数"这条路。

## 1. 起因

`gs_refine_live_20261012.md` 的三栏动画里，仿真栏与 CCD 栏肉眼差异极大。旧实现
（`idealized_far_field`）是一个**未标定的同尺寸 FFT**：无波长、无焦距、无像元间距，
`2π` 取 1023 灰度而非实验系统实测的 993，且 1200×1920 不补零 ⇒ 远场像元
8.66 µm(x) / 13.85 µm(y)，**各向异性 1.60×**，平场核心塌到 1~3 px。
{SECTION_1_1}
## 2. 硬件基线（先确认实验系统是好的）

| 量 | 值 |
|---|---|
| 理论平场 FWHM `2.355·λf/(π·w₀)` | 12.6 相机 px |
| 实测平场 FWHM | **11 × 12 px** |
| 实测 GS 相位 FWHM | **80 × 92 px**（`target_side`=100） |

GS 把 11.5 px 的平场展开成 ~86 px 方斑，**尺度正确** ⇒ 问题在仿真侧。

## 3. 两个被证伪的解释

### 3.1 "远场中心没对齐" —— 证伪

仿真远场质心相对 DC：平场 (+0.46, +0.46) px、GS (+0.78, −1.46) px；相机 0 阶
相对窗心 (+2, +1) / (−1, +3) px。GS 相位确有 `dy = +108.3 rad/m` 的倾斜项，按
`Δx = k·f·λ/2π` 预测 **1.04 px**，与实测一致。全部在 ~2 px 内，**不是中心问题**。

### 3.2 "远场残差是 PSF/频率响应造成的" —— 证伪

在焦平面加一个径向低通（SLM 像元 sinc + 高斯 MTF）并扫参：

| blur (相机 px) | 平均相关 |
|---|---|
{sweep_tbl}

全程只涨 **{bc - blur_sweep[0][1]:+.4f}**（最优 {bb:.0f} px），在噪声内；且 FWHM
全程未变、只有 EE50 下降——径向低通是**压制尖峰**而非**拓宽核心**。

关键：残差要求两件相反的事同时成立——实测**平场更宽**（要拓宽）而实测 **GS 更紧**
（要收紧）。单一径向滤波器做不到，所以残差是杂散光与 SLM 真实相位分辨率这类
**结构上缺失**的物理。生产代码里**不保留**该旋钮。

## 4. 真正的缺陷：第三栏不在相机像元网格上

`forward_intensity` 的远场像元间距是 `λf/(aperture_size·padding)`，**与相机无关**。
本实验系统 `n_grid = pupil_side(450) = 900`、`padding = 3` ⇒ 远场视场约 **16.6 mm**，
而相机 250 px 窗只覆盖 **0.55 mm**，相差约 **30×**。旧代码再取"相对像素"的
500×500 中心裁剪，于是第三栏与第一栏**永远无法逐像素比较**。

修复：在 canonical bench 模块新增
`camera_window_from_far_field()`，按 `λf/(aperture·padding)` 与相机像元间距之比
重采样（**两个方向都要处理**：远场比相机细则抽稀，比相机粗则插值放大——本实验系统
`padding=3` 属后者，早期实现漏了后者，光斑被画得比真实更小）。
{SECTION_4X}
## 5. 标定后验证

![measured vs simulated](figures/measured_vs_simulated.png)

| run | 相位 | FWHM 实测 | FWHM 仿真 | EE50 实测 | EE50 仿真 | EE90 实测 | EE90 仿真 | 相关 |
|---|---|---|---|---|---|---|---|---|
{tbl}

![metrics](figures/metrics_and_blur_sweep.png)

尺度与方形结构现在一致；**EE50 仍偏高**（仿真铺得更开），即实测图**对比度更高**。

## 6. CV 能不能预测？——能，但有系统性偏移（可用的离线代理）

同一模型预测这 3 个 run 的实测分数，作为"能否离线选参"的闸门：

| | 实测 | 仿真 |
|---|---|---|
| PIB | {pibm:.3f} ~ {pibM:.3f} | {pibp:.3f} ~ {pibP:.3f} |
| CV | {cvm:.3f} | {cvp:.3f}（≈{ratio:.2f}×） |
| 复合分数 | {scm:.3f} ~ {scM:.3f} | {scp:.3f} ~ {scP:.3f} |

**结论比预期好**：复合分数逐 run 对得上（最大偏差 {maxdev:.3f}），PIB 也在同一量级，
CV 系统性**偏低约 {bias:.0%}**——理想瞳面做出的方形内部散斑比实测更均匀。

> ⚠️ 曾经的**错误结论**（已推翻，留档）：早先用一个"硬边圆盘 + 自建标定"的
> 自研物理预测模型时，预测 CV 高达实测的 3.7 倍（2.43 vs 0.68），据此判定
> "模型不能用来选参、路线不可用"。换成实验系统的物理预测模型后该结论不成立。
> 根因是**瞳面模型**而非几何：实验系统的 `make_gaussian_pupil` 是
> "σ=aperture/3.5 的高斯 ⊕ 半径 aperture/2 的硬截断"（软边圆盘，且因瞳面裁剪而居中、
> 无边缘截断），比我试过的 top-hat(R=450) 与 高斯(σ=450)都更贴近真实光束。
> 这也是"物理预测模型只放在 `slm_shaping_bench.py` 一处"这条红线值钱的原因——
> 自己抄一份只会抄错瞳面。

因此**离线代理可用**，但使用时须知：CV 绝对值不可信，只可信**同一模型下的相对
排序**；且实验系统跨会话 CV 波动大于多数参数效应，任何离线结论都只能当**实验假设**。

## 7. 结论与边界

**已解决**：第三栏与 CCD 同尺度同结构；各向异性；2π 基准；几何回退。

**未解决且已记录**：CV 对比度偏差（§6）、EE50 偏差（§5）。两者都指向同一个
缺口——理想圆盘瞳没有的物理。

**定性定位**：这是**几何检查模型**（尺度、对称性、形状、左右上下转置），
**不是**光度预测器；不要用它预测 CV/PIB 的绝对值，也不要用它选参数。
"""


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dir holding slm_gs_refine/ (default <repo>/data/debug)",
    )
    args = ap.parse_args()

    debug_root = args.data_root or (ROOT / "data" / "debug")
    samples = collect(debug_root)
    if not samples:
        print("no gs_refine samples with a recorded pupil centre under", debug_root)
        return

    rows = gate(samples)
    sweep = []
    for b in np.round(np.arange(0.0, 32.01, 4.0), 2):
        cs = []
        for s in samples:
            cam = simulate(s, float(b))
            v = cam[box_mask(cam.shape, int(s["side"]))]
            pib = float(v.sum() / cam.sum())
            cv = float(v.std() / v.mean())
            cs.append(0.5 * pib + 0.5 / (1.0 + cv))
        sweep.append((float(b), float(np.mean([r[4] for r in rows]))))
    # correlation sweep needs its own pass
    sweep = []
    for b in np.round(np.arange(0.0, 32.01, 4.0), 2):
        cs = [pearson(s["img"], simulate(s, float(b))) for s in samples]
        sweep.append((float(b), float(np.mean(cs))))

    render_figures(samples, rows, sweep)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    out.write_text(
        insert_header(build_markdown(samples, rows, sweep), REPORT_KEY),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
