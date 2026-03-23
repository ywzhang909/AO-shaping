from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem
from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv
from ao_shaping.optimizer.wfless.sim_spgd import (
    optimize_ga,
    optimize_pso,
    optimize_sa,
    optimize_spgd,
    optimize_spgd_zernike,
)


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"


def make_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = LOG_ROOT / f"sim_visual_report_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_config(cn2: float, n_grid: int = 64) -> AOConfig:
    return AOConfig(
        N=n_grid,
        L=0.04,
        Cn2=cn2,
        L0=20.0,
        l0=1e-3,
        dm_actuators=4,
        subapertures=4,
        pixel_scale=0.5,
        propagation_distance=1000.0,
    )


def run_turbulence_scan(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    cn2_values = [0.0, 1e-15, 1e-14, 5e-14, 1e-13]

    fig, axes = plt.subplots(len(cn2_values), 3, figsize=(12, 3 * len(cn2_values)))
    if len(cn2_values) == 1:
        axes = np.array([axes])

    for row_idx, cn2 in enumerate(cn2_values):
        np.random.seed(100 + row_idx)
        ao = TraditionalAOSystem(make_config(cn2))
        state = ao.reset()

        image = state["image"].astype(float)
        phase = ao.turbulence.phase_screen
        center = np.array(image.shape) // 2
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= 4**2
        bucket_ratio = float(np.sum(image[mask]) / np.sum(image))

        rows.append(
            {
                "cn2": cn2,
                "phase_std": float(np.std(phase)),
                "phase_rms": float(state["phase_rms"]),
                "strehl": float(state["strehl"]),
                "power": float(state["power"]),
                "peak_ratio": float(np.max(image) / np.sum(image)),
                "bucket_ratio_r4": bucket_ratio,
            }
        )

        ax0, ax1, ax2 = axes[row_idx]
        im0 = ax0.imshow(phase, cmap="coolwarm")
        ax0.set_title(f"Phase Screen\nCn2={cn2:.1e}")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(image, cmap="inferno")
        ax1.set_title(f"Focal Image\nStrehl={state['strehl']:.3f}")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        slopes = state["slopes"]
        half = slopes.size // 2
        slope_map = np.stack(
            [
                slopes[:half].reshape(ao.config.subapertures, ao.config.subapertures),
                slopes[half:].reshape(ao.config.subapertures, ao.config.subapertures),
            ],
            axis=-1,
        )
        mag = np.linalg.norm(slope_map, axis=-1)
        im2 = ax2.imshow(mag, cmap="viridis")
        ax2.set_title(f"Slope Magnitude\nphase_std={np.std(phase):.3f}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_dir / "turbulence_scan_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    metrics = [
        ("phase_std", "Phase STD"),
        ("phase_rms", "Phase RMS"),
        ("strehl", "Strehl"),
        ("bucket_ratio_r4", "Bucket Ratio r=4"),
    ]
    x = np.arange(len(df))
    labels = [f"{v:.0e}" if v else "0" for v in df["cn2"]]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(x, df[col], marker="o")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Cn2")
    fig.tight_layout()
    fig.savefig(output_dir / "turbulence_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return df


def _history_df(recorder) -> pd.DataFrame:
    rows = []
    for row in recorder.history:
        rows.append(
            {
                "epoch": row.get("_epoch", 0),
                "pib": float(row.get("pib", np.nan)),
                "J": float(row.get("J", np.nan)),
                "strehl": float(row.get("strehl", np.nan)),
                "gamma": float(row.get("gamma", np.nan)),
                "delta": float(row.get("delta", np.nan)),
                "r": float(row.get("r", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def run_optimizer_suite(output_dir: Path) -> pd.DataFrame:
    configs = {
        "spgd_adamod": dict(
            fn=optimize_spgd,
            kwargs=dict(
                epochs=60,
                r_bucket=15,
                n_grid=64,
                Cn2=1e-14,
                seed=42,
                delta=0.03,
                gamma=1e-2,
                optimizer_type="adamod",
                use_momentum=False,
            ),
        ),
        "spgd_zernike_adamod": dict(
            fn=optimize_spgd_zernike,
            kwargs=dict(
                epochs=60,
                n_max=6,
                r_bucket=15,
                n_grid=64,
                Cn2=1e-14,
                seed=42,
                delta=0.01,
                gamma=1e-2,
                optimizer_type="adamod",
                use_momentum=False,
            ),
        ),
        "pso": dict(
            fn=optimize_pso,
            kwargs=dict(epochs=30, n_particles=15, r_bucket=15, n_grid=64, Cn2=1e-14, seed=42),
        ),
        "ga": dict(
            fn=optimize_ga,
            kwargs=dict(epochs=30, population_size=15, r_bucket=15, n_grid=64, Cn2=1e-14, seed=42),
        ),
        "sa": dict(
            fn=optimize_sa,
            kwargs=dict(
                epochs=30,
                r_bucket=15,
                n_grid=64,
                Cn2=1e-14,
                seed=42,
                T_init=100.0,
                cooling_rate=0.95,
            ),
        ),
    }

    summaries: list[dict[str, float | str]] = []
    histories: dict[str, pd.DataFrame] = {}
    for name, spec in configs.items():
        recorder = spec["fn"](**spec["kwargs"])
        df = _history_df(recorder)
        histories[name] = df
        init_pib = float(df.iloc[0]["pib"])
        final_pib = float(df.iloc[-1]["pib"])
        best_pib = float(df["pib"].max())
        best_epoch = int(df.loc[df["pib"].idxmax(), "epoch"])
        summaries.append(
            {
                "optimizer": name,
                "initial_pib": init_pib,
                "final_pib": final_pib,
                "best_pib": best_pib,
                "best_epoch": best_epoch,
                "final_ratio": final_pib / max(init_pib, 1e-12),
                "best_ratio": best_pib / max(init_pib, 1e-12),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values("best_ratio", ascending=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, (name, df) in zip(axes, histories.items()):
        ax.plot(df["epoch"], df["pib"], label="PIB", color="tab:blue")
        if df["strehl"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(df["epoch"], df["strehl"], label="Strehl", color="tab:orange", alpha=0.7)
            ax2.set_ylabel("Strehl")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PIB")
        ax.grid(alpha=0.3)
    for idx in range(len(histories), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "optimizer_histories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    ax.bar(x - 0.2, summary_df["final_ratio"], width=0.4, label="Final / Init")
    ax.bar(x + 0.2, summary_df["best_ratio"], width=0.4, label="Best / Init")
    ax.set_xticks(x, summary_df["optimizer"], rotation=20)
    ax.set_ylabel("PIB Ratio")
    ax.set_title("Optimizer Outcome Ratios")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimizer_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return summary_df


def run_rl_rollout(output_dir: Path) -> pd.DataFrame:
    env = SimTurbulenceAOEnv(
        n_grid=32,
        n_actuators=4,
        n_subapertures=4,
        max_steps=20,
        cn2=1e-14,
        pib_radius=4,
    )
    obs, info = env.reset(seed=123)
    rows = [
        {
            "step": 0,
            "reward": 0.0,
            "strehl": float(info["strehl"]),
            "rms": float(info["rms"]),
            "pib": float(info["pib"]),
            "best_pib": float(info["best_pib"]),
            "actuation_rms": float(obs["metrics"][4]),
        }
    ]
    frames = [obs["ccd"][-1]]
    slope_frames = [obs["hartmann_slopes"][-1].reshape(2, 4, 4)]

    for step in range(1, 21):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if step % 2 == 0:
            action += 0.01
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": step,
                "reward": float(reward),
                "strehl": float(info["strehl"]),
                "rms": float(info["rms"]),
                "pib": float(info["pib"]),
                "best_pib": float(info["best_pib"]),
                "actuation_rms": float(obs["metrics"][4]),
            }
        )
        frames.append(obs["ccd"][-1])
        slope_frames.append(obs["hartmann_slopes"][-1].reshape(2, 4, 4))
        if terminated or truncated:
            break

    env.close()
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].plot(df["step"], df["reward"], marker="o")
    axes[0].set_title("RL Reward")
    axes[1].plot(df["step"], df["pib"], marker="o")
    axes[1].plot(df["step"], df["best_pib"], linestyle="--", label="best_pib")
    axes[1].set_title("PIB / Best PIB")
    axes[1].legend()
    axes[2].plot(df["step"], df["strehl"], marker="o")
    axes[2].set_title("Strehl")
    axes[3].plot(df["step"], df["rms"], marker="o")
    axes[3].set_title("Phase RMS")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rl_rollout_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample_steps = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    fig, axes = plt.subplots(2, len(sample_steps), figsize=(3 * len(sample_steps), 6))
    for col, idx in enumerate(sample_steps):
        axes[0, col].imshow(frames[idx], cmap="inferno")
        axes[0, col].set_title(f"CCD step {idx}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        slope_mag = np.linalg.norm(np.moveaxis(slope_frames[idx], 0, -1), axis=-1)
        axes[1, col].imshow(slope_mag, cmap="viridis")
        axes[1, col].set_title(f"Slope | step {idx}")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "rl_rollout_frames.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return df


def write_report(
    output_dir: Path,
    turbulence_df: pd.DataFrame,
    optimizer_df: pd.DataFrame,
    rl_df: pd.DataFrame,
) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "turbulence_cases": turbulence_df.to_dict(orient="records"),
        "optimizer_summary": optimizer_df.to_dict(orient="records"),
        "rl_rollout_summary": {
            "steps": int(len(rl_df) - 1),
            "initial_pib": float(rl_df.iloc[0]["pib"]),
            "final_pib": float(rl_df.iloc[-1]["pib"]),
            "best_pib": float(rl_df["best_pib"].max()),
            "final_strehl": float(rl_df.iloc[-1]["strehl"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Simulation Visual Report",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Files",
        "- `turbulence_scan_grid.png`: 湍流相位屏、焦平面图像、斜率幅值并排对比",
        "- `turbulence_metrics.png`: Cn2 扫描下 phase/std、RMS、Strehl 趋势",
        "- `optimizer_histories.png`: SPGD、Zernike-SPGD、PSO、GA、SA 收敛历史",
        "- `optimizer_summary.png`: 各优化器 final/init 与 best/init PIB 比值",
        "- `rl_rollout_metrics.png`: SimTurbulenceAOEnv rollout 奖励与指标曲线",
        "- `rl_rollout_frames.png`: RL rollout 中 CCD 与 slope 快照",
        "- `turbulence_summary.csv`, `optimizer_summary.csv`, `rl_rollout.csv`, `summary.json`",
        "",
        "## Notes",
        "- SPGD 类可视化使用迁移后的 beam-project 仿真后端。",
        "- Zernike/SPGD 采用当前稳定参数；PSO/GA/SA 主要展示搜索轨迹，不强行要求最终值优于初始值。",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    output_dir = make_output_dir()
    turbulence_df = run_turbulence_scan(output_dir)
    optimizer_df = run_optimizer_suite(output_dir)
    rl_df = run_rl_rollout(output_dir)

    turbulence_df.to_csv(output_dir / "turbulence_summary.csv", index=False)
    optimizer_df.to_csv(output_dir / "optimizer_summary.csv", index=False)
    rl_df.to_csv(output_dir / "rl_rollout.csv", index=False)
    write_report(output_dir, turbulence_df, optimizer_df, rl_df)
    print(output_dir)


if __name__ == "__main__":
    main()
