"""Figures for the forward-model search: pred vs true, per arm, with the residual.

Two figures, because they answer different questions.

``pred_vs_true.png``
    A per-sample grid: for several held-out frames, the measured frame, each arm's
    prediction, and the residual. This is the qualitative check -- a metric can be averaged
    over a split and still hide that one objective is badly wrong, and this project has been
    bitten by exactly that (§6: the family is four different optimisation objectives).

``arm_summary.png``
    The paired delta per arm with its standard error, and the count of seeds where the arm
    beat the incumbent. The seed count is on the plot on purpose: with a between-seed spread
    of ~0.075 in R2, a mean alone would let a coin flip look like a result.

Every panel is peak-normalised with a SHARED colour scale, because per-panel scaling makes a
bad prediction look identical to a good one -- the exact failure that made the first version
of the forward figure meaningless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from compare_unet_baseline import _peak_normalise, _select_records, _inputs  # noqa: E402
from forward_search import (  # noqa: E402
    APERTURE_R,
    BEAM_W0,
    Arm,
    GRID,
    N_MAX,
    PADDING,
    fwd,
    train_arm,
)
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import augment as aug  # noqa: E402
from ml.zernike.metrics import batch_image_metrics  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

FIG_DIR = ROOT / "report" / "loss_defects" / "figures"
CJK = {"Microsoft YaHei", "SimHei", "DejaVu Sans"}
plt.rcParams["font.sans-serif"] = [f for f in ("Microsoft YaHei", "SimHei", "DejaVu Sans")]
plt.rcParams["axes.unicode_minus"] = False

# The arms worth showing side by side: the incumbent, the best loss arm, the best
# augmentation arm, and the architecture arm with the best SSIM.
SHOW = ["A0 mse (incumbent)", "A3 +ellipse 1.0", "C4 +strong 25%", "A5 unet"]
SAMPLES = [0, 1, 2, 3]


def norm(img: np.ndarray) -> np.ndarray:
    peak = float(img.max())
    return img / peak if peak > 0 else img


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=60, lr=0.02,
        batch_size=64, max_train=512, max_val=128, beam_samples=16, seed=0,
        use_wandb=False, save_checkpoint=False,
    )
    torch.manual_seed(0)
    tr, va = _select_records(dataset, cfg)
    t = collect_split(dataset, tr, device)
    xv = _inputs(collect_split(dataset, va, device))
    yv = _peak_normalise(collect_split(dataset, va, device)["target"].clone())
    x = torch.cat([t["phase_cos"], t["phase_sin"]], dim=1)
    y = _peak_normalise(t["target"].clone())

    specs = {
        "A0 mse (incumbent)": Arm("A0"),
        "A3 +ellipse 1.0": Arm("A3", weights=_ellipse(1.0)),
        "C4 +strong 25%": Arm("C4", strong_frac=0.25),
        "A5 unet": Arm("A5", kind="unet"),
    }
    preds: dict[str, torch.Tensor] = {}
    for name, arm in specs.items():
        strong = None
        if arm.strong_frac > 0:
            strong = aug.strong_phase_pairs(
                300, grid=GRID, beam_w0=BEAM_W0, far_field_padding=PADDING,
                aperture_radius=APERTURE_R, n_max=N_MAX, seed=99,
            )
        model = train_arm(arm, x, y, 0, device, strong)
        with torch.no_grad():
            preds[name] = torch.cat([fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])

    # ---------------------------------------------------------------- pred vs true grid
    cols = 1 + 2 * len(SHOW)
    fig, axes = plt.subplots(len(SAMPLES), cols, figsize=(3.0 * cols, 3.1 * len(SAMPLES)))
    for r, si in enumerate(SAMPLES):
        true = norm(yv[si, 0].cpu().numpy())
        vmax = float(max(true.max(), *(norm(preds[n][si, 0].cpu().numpy()).max() for n in SHOW)))
        ax = axes[r, 0]
        im = ax.imshow(true, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title("实测 true（CCD）" if r == 0 else "", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_ylabel(f"样本 #{si}", fontsize=9)
        for c, name in enumerate(SHOW):
            p = norm(preds[name][si, 0].cpu().numpy())
            ax = axes[r, 1 + 2 * c]
            ax.imshow(p, cmap="inferno", vmin=0, vmax=vmax)
            if r == 0:
                ax.set_title(f"预测 pred\n{name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax = axes[r, 2 + 2 * c]
            imd = ax.imshow(p - true, cmap="coolwarm", vmin=-0.5, vmax=0.5)
            if r == 0:
                ax.set_title("残差 pred − true", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(
        "pred vs true —— 每个 arm 的预测与实测远场（同一色标，逐样本对比）\n"
        "色标统一：各面板若各自归一化，坏预测会看起来和好预测一样",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "forward_search_pred_vs_true.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'forward_search_pred_vs_true.png'}")

    # ---------------------------------------------------------------- arm summary
    data = json.loads((ROOT / "report" / "loss_defects" / "forward_search.json").read_text(encoding="utf-8"))
    table = data["summary"]
    names = [s["arm"] for s in table]
    d = np.array([s["d_r2"] for s in table])
    se = np.array([s["d_r2_se"] for s in table])
    pos = np.array([s["pos"] for s in table])
    colours = ["#444444" if s["d_r2"] == 0 else ("#2e7d32" if s["pos"] == 3 else "#c62828")
               for s in table]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    ypos = np.arange(len(names))
    ax.barh(ypos, d, xerr=se, color=colours, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ΔR² 相对基线（配对，同 seed）")
    ax.set_title("各 arm 对正向 R² 的贡献\n绿=3/3 seed 全部更好，红=不一致", fontsize=11)
    for i, (dd, ss, pp) in enumerate(zip(d, se, pos)):
        ax.text(dd + np.sign(dd) * (ss + 0.004), i, f"{dd:+.4f} ({pp}/3)",
                va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax2.barh(ypos, [s["ssim"] for s in table],
             xerr=[np.std([r["ssim"] for r in data["rows"] if r["arm"] == s["arm"]], ddof=1) / np.sqrt(3)
                   for s in table],
             color="#1565c0", capsize=3, alpha=0.8)
    ax2.set_yticks(ypos); ax2.set_yticklabels([])
    ax2.set_xlabel("val SSIM（越高越好）")
    ax2.set_title("结构相似度", fontsize=11)
    ax2.grid(alpha=0.3, axis="x")
    fig.suptitle(
        "正向模型搜索：没有一个 arm 显著提升 R²\n"
        "误差棒 = 配对差值的标准误；括号内 = 3 个 seed 中胜过基线的个数",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forward_search_summary.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'forward_search_summary.png'}")


def _ellipse(w: float):
    from ml.zernike.losses import LossConfig

    return LossConfig(w_mse=1.0, w_ellipse=w)


if __name__ == "__main__":
    main()