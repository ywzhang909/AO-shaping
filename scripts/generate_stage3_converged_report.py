from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_long_sac_experiments import (  # noqa: E402
    _format_command,
    _load_eval,
    _plot_rollout,
    _plot_training,
    _rollout,
    _summarize_rollout,
    STATIC_ARGS,
)


def main() -> None:
    static_log = ROOT / "logs" / "static_long_20260323_172155"
    static_model = ROOT / "models" / "static_long_20260323_172155"
    curriculum = [
        ("stage1_easy", ROOT / "logs" / "stage1_easy_sweep_20260323_162629", ROOT / "models" / "stage1_easy_sweep_20260323_162629"),
        ("stage2_medium", ROOT / "logs" / "stage2_medium_sweep_20260323_162629", ROOT / "models" / "stage2_medium_sweep_20260323_162629"),
        (
            "stage3_target",
            ROOT / "logs" / "stage3_long_horizon_20260323_162629",
            ROOT / "models" / "stage3_long_horizon_20260323_162629",
        ),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "logs" / f"sac_stage3_converged_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    curriculum_rows: list[dict[str, object]] = []
    for stage_name, log_dir, model_dir in curriculum:
        config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
        curriculum_rows.append(
            {
                "stage": stage_name,
                "log_dir": str(log_dir),
                "model_dir": str(model_dir),
                **config,
                **summary,
            }
        )

    pd.DataFrame(curriculum_rows).to_csv(report_dir / "turbulence_curriculum_summary.csv", index=False)
    with (report_dir / "turbulence_curriculum_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(curriculum_rows, fh, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    stage_names = [row["stage"] for row in curriculum_rows]
    axes[0].bar(stage_names, [float(row["mean_reward"]) for row in curriculum_rows])
    axes[0].set_ylabel("Mean Reward")
    axes[1].bar(stage_names, [float(row["mean_final_strehl"]) for row in curriculum_rows])
    axes[1].set_ylabel("Final Strehl")
    axes[2].bar(stage_names, [float(row["mean_best_pib"]) / 1e6 for row in curriculum_rows])
    axes[2].set_ylabel("Best PIB (x1e6)")
    axes[2].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "turbulence_curriculum_stage_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in curriculum_rows:
        _, rewards = _load_eval(Path(str(row["log_dir"])))
        ax.plot(rewards, marker="o", label=row["stage"])
    ax.set_title("Turbulence Curriculum Eval Reward")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_xlabel("Eval Index")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "turbulence_curriculum_eval_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    turb_log = Path(str(curriculum_rows[-1]["log_dir"]))
    turb_model = Path(str(curriculum_rows[-1]["model_dir"]))
    static_summary = json.loads((static_log / "summary.json").read_text(encoding="utf-8"))
    turb_summary = json.loads((turb_log / "summary.json").read_text(encoding="utf-8"))

    static_conv = _plot_training(
        static_log,
        report_dir,
        "static_long",
        mode="static",
        mean_final_strehl=float(static_summary["mean_final_strehl"]),
    )
    turb_conv = _plot_training(
        turb_log,
        report_dir,
        "turbulence_long",
        mode="turbulence",
        mean_final_strehl=float(turb_summary["mean_final_strehl"]),
    )

    static_rollout = _rollout(static_log)
    turb_rollout = _rollout(turb_log)
    static_rollout_df = _plot_rollout(static_rollout, report_dir, "static_long")
    turb_rollout_df = _plot_rollout(turb_rollout, report_dir, "turbulence_long")
    static_rollout_summary = _summarize_rollout(static_rollout_df)
    turb_rollout_summary = _summarize_rollout(turb_rollout_df)

    combined = {
        "static": {
            **static_summary,
            **static_conv,
            **static_rollout_summary,
            "log_dir": str(static_log),
            "model_dir": str(static_model),
            "converged": bool(static_conv["is_plateau"]),
        },
        "turbulence": {
            **turb_summary,
            **turb_conv,
            **turb_rollout_summary,
            "log_dir": str(turb_log),
            "model_dir": str(turb_model),
            "converged": bool(turb_conv["is_plateau"]),
        },
        "turbulence_curriculum": {
            "stages": curriculum_rows,
            "final_stage": curriculum_rows[-1]["stage"],
        },
        "commands": {
            "static": "python scripts/run_long_sac_experiments.py",
            "direct_static": _format_command("src/ao_shaping/optimizer/rl/sac_train.py", STATIC_ARGS, static_log, static_model),
            "curriculum": "python scripts/run_curriculum_mamba_turbulence.py",
            "stage3_sweep": "python scripts/sweep_stage3_target.py",
        },
    }

    with (report_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)

    pd.DataFrame(
        [
            {"task": "static", **combined["static"]},
            {"task": "turbulence", **combined["turbulence"]},
        ]
    ).to_csv(report_dir / "summary.csv", index=False)

    report_lines = [
        "# SAC Stage3 Converged Report",
        "",
        f"- Report dir: `{report_dir}`",
        f"- Static converged: `{combined['static']['converged']}`",
        f"- Turbulence converged: `{combined['turbulence']['converged']}`",
        "",
        "## Turbulence",
        f"- curriculum_final_stage: `{combined['turbulence_curriculum']['final_stage']}`",
        f"- mean_reward: `{combined['turbulence']['mean_reward']:.3f}`",
        f"- mean_final_strehl: `{combined['turbulence']['mean_final_strehl']:.3f}`",
        f"- mean_best_pib: `{combined['turbulence']['mean_best_pib']:.1f}`",
        f"- rollout_final_strehl_mean: `{combined['turbulence']['rollout_final_strehl_mean']:.3f}`",
        f"- late_reward_mean: `{combined['turbulence']['late_reward_mean']:.3f}`",
        f"- late_reward_std: `{combined['turbulence']['late_reward_std']:.3f}`",
        f"- late_reward_slope: `{combined['turbulence']['late_reward_slope']:.3f}`",
        "",
        "## One-Click",
        f"`{combined['commands']['stage3_sweep']}`",
        f"`{combined['commands']['curriculum']}`",
    ]
    (report_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(report_dir)


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "src;.venv\\Lib\\site-packages"
    main()
