from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm as tqdm_lib

from ao_shaping.optimizer.wfless import pib as pib_module


@dataclass
class SimLandscape:
    """Synthetic non-convex PIB landscape with local/global optima."""

    center: tuple[int, int] = (48, 48)
    shape: tuple[int, int] = (96, 96)
    global_center: np.ndarray = field(
        default_factory=lambda: np.array([-4.8, -4.2, -4.5, -4.0], dtype=np.float64)
    )
    local_center: np.ndarray = field(
        default_factory=lambda: np.array([2.5, 3.2, 2.2, 2.8], dtype=np.float64)
    )
    global_width: float = 2.6
    local_width: float = 1.45
    local_scale: float = 0.90
    base_sigma: float = 8.6
    peak_sigma_gain: float = 4.8

    def score(self, voltages: np.ndarray) -> tuple[float, float, float]:
        active = np.asarray(voltages, dtype=np.float64)[:4]
        global_score = float(
            np.exp(
                -np.sum((active - self.global_center) ** 2) / (2 * self.global_width**2)
            )
        )
        local_score = float(
            self.local_scale
            * np.exp(
                -np.sum((active - self.local_center) ** 2) / (2 * self.local_width**2)
            )
        )
        return max(global_score, local_score), global_score, local_score

    def render(self, voltages: np.ndarray) -> np.ndarray:
        score, global_score, _ = self.score(voltages)
        sigma = self.base_sigma - self.peak_sigma_gain * score
        sigma = float(np.clip(sigma, 2.2, 10.0))
        amplitude = 180.0 + 70.0 * score + 18.0 * global_score

        yy, xx = np.indices(self.shape, dtype=np.float64)
        cx, cy = self.center
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        img = amplitude * np.exp(-rr2 / (2.0 * sigma**2))
        img += 3.0
        return img.astype(np.float64)


LANDSCAPE = SimLandscape()
ACTIVE_MASK = np.array([True, True, True, True, False, False, False, False])


class SimDM:
    """Minimal DM shim compatible with optimize_pib()."""

    current_voltages = np.zeros(8, dtype=np.float64)

    def __init__(
        self,
        keep_when_exit: bool = True,
        max_neibor_diff: float = 400,
        max_voltage: float | None = None,
        min_voltage: float | None = None,
    ) -> None:
        self.DM_Num = 8
        self.default_dm_unit_mask = ACTIVE_MASK.copy()
        self.max_neibor_diff = max_neibor_diff
        self.min_voltage = -12.0 if min_voltage is None else float(min_voltage)
        self.max_voltage = 12.0 if max_voltage is None else float(max_voltage)
        self.V_Min = self.min_voltage
        self.V_Max = self.max_voltage
        self.keep_when_exit = keep_when_exit

    def __enter__(self) -> "SimDM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.0) -> np.ndarray:
        clipped = np.clip(
            np.asarray(vs, dtype=np.float64), self.min_voltage, self.max_voltage
        )
        self.current_voltages = clipped.copy()
        SimDM.current_voltages = clipped.copy()
        return clipped

    def check_dm_unit_grad_safe(self, vs: np.ndarray) -> bool:
        active = np.asarray(vs, dtype=np.float64)[:4]
        return bool(np.all(np.abs(np.diff(active)) <= self.max_neibor_diff))


class SimCamera:
    """Minimal camera shim compatible with optimize_pib()."""

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 80.0,
        skip_sampling: bool = False,
    ) -> None:
        self.cam_id = cam_id
        self.exposure_time = exposure_time_ms

    def __enter__(self) -> "SimCamera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def autoset_exposure_time_ms(
        self,
        target_max_brightness: float,
        threshold: int = 5,
        twice_valid: bool = True,
    ) -> np.ndarray:
        return self.get_numpy_image(1)

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return LANDSCAPE.render(SimDM.current_voltages)

    def reset_window(self, center: tuple[int, int], img_size: tuple[int, int]):
        return img_size, center


SIM_CASES: list[tuple[str, dict[str, object]]] = [
    ("baseline_adamod", {"optimizer_type": "adamod", "enable_adaptive_search": False}),
    (
        "adamod_search_small",
        {
            "optimizer_type": "adamod",
            "enable_adaptive_search": True,
            "search_interval": 30,
            "search_warmup": 60,
            "search_patience": 30,
            "search_samples": 8,
            "search_radius": 2.5,
            "tabu_memory_size": 48,
            "lr": 0.45,
        },
    ),
    (
        "adamod_search_medium",
        {
            "optimizer_type": "adamod",
            "enable_adaptive_search": True,
            "search_interval": 25,
            "search_warmup": 50,
            "search_patience": 20,
            "search_samples": 14,
            "search_radius": 4.0,
            "tabu_memory_size": 96,
            "lr": 0.40,
        },
    ),
    (
        "adam_search_medium",
        {
            "optimizer_type": "adam",
            "enable_adaptive_search": True,
            "search_interval": 25,
            "search_warmup": 50,
            "search_patience": 20,
            "search_samples": 14,
            "search_radius": 4.0,
            "tabu_memory_size": 96,
            "lr": 0.32,
        },
    ),
    (
        "adamod_search_stable",
        {
            "optimizer_type": "adamod",
            "enable_adaptive_search": True,
            "search_interval": 20,
            "search_warmup": 40,
            "search_patience": 15,
            "search_samples": 20,
            "search_radius": 5.5,
            "search_max_radius": 8.0,
            "tabu_memory_size": 128,
            "lr": 0.28,
            "epochs": 180,
        },
    ),
]


@contextmanager
def patched_pib_simulation():
    """Temporarily replace physical devices/tqdm with deterministic simulation."""
    original_camera = pib_module.CameraStreamManager
    original_dm = pib_module.NlightDM
    original_tqdm = pib_module.tqdm.tqdm
    SimDM.current_voltages = np.zeros(8, dtype=np.float64)
    pib_module.CameraStreamManager = SimCamera
    pib_module.NlightDM = SimDM
    pib_module.tqdm.tqdm = lambda *args, **kwargs: original_tqdm(
        *args, **({"disable": True} | kwargs)
    )
    try:
        yield
    finally:
        pib_module.CameraStreamManager = original_camera
        pib_module.NlightDM = original_dm
        pib_module.tqdm.tqdm = original_tqdm


def run_case(name: str, **kwargs):
    """Run optimize_pib() for one deterministic simulation setting."""
    epochs = int(kwargs.pop("epochs", 220))
    lr = float(kwargs.pop("lr", 0.55))
    delta = float(kwargs.pop("delta", 1.2))
    recorder = pib_module.optimize_pib(
        center=LANDSCAPE.center,
        epochs=epochs,
        r_bucket=10,
        delta=delta,
        lr=lr,
        exposure_time_ms=20,
        shrink_iter=0,
        cam_id=0,
        show=False,
        init_v=np.zeros(8, dtype=np.float64),
        cam_size=LANDSCAPE.shape[0],
        target_max_brightness=0,
        dm_unit_mask=ACTIVE_MASK.copy(),
        dm_neibor_diff=100,
        dm_max_voltage=12,
        dm_min_voltage=-12,
        random_seed=42,
        **kwargs,
    )
    df = recorder.dataframe.copy()
    df["global_score"] = df["_v"].apply(
        lambda voltages: LANDSCAPE.score(np.asarray(voltages))[1]
    )
    df["local_score"] = df["_v"].apply(
        lambda voltages: LANDSCAPE.score(np.asarray(voltages))[2]
    )
    df["epoch_index"] = np.arange(len(df))
    best_iter, (_, best_pib) = recorder.get_best_iter()
    final = recorder.last
    _, global_score, local_score = LANDSCAPE.score(
        np.asarray(final["_v"], dtype=np.float64)
    )
    summary = {
        "case": name,
        "final_pib": float(final["pib"]),
        "best_pib": float(best_pib),
        "final_J": float(final["J"]),
        "search_accepts": int(
            sum(1 for item in recorder.history if item.get("search_accept"))
        ),
        "tabu_peak": int(max(item.get("tabu_size", 0) for item in recorder.history)),
        "search_radius_final": float(final.get("search_radius", np.nan)),
        "global_score": global_score,
        "local_score": local_score,
        "used_search": bool(kwargs.get("enable_adaptive_search", False)),
        "best_epoch": int(df["pib"].astype(float).idxmax()),
    }
    return recorder, df, summary


def run_suite(cases: list[tuple[str, dict[str, object]]] | None = None):
    """Run the full simulation benchmark suite and return raw outputs."""
    suite_cases = cases or SIM_CASES
    rows: list[dict[str, object]] = []
    recorders: dict[str, object] = {}
    histories: dict[str, pd.DataFrame] = {}
    with patched_pib_simulation():
        for name, config in suite_cases:
            recorder, history_df, summary = run_case(name, **dict(config))
            rows.append(summary)
            recorders[name] = recorder
            histories[name] = history_df
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("best_pib", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df, histories, recorders


def _find_stage_row(history_df: pd.DataFrame, stage: str) -> pd.Series:
    search_accept = pd.Series(history_df["search_accept"], dtype="boolean").fillna(
        False
    )
    if stage == "init":
        return history_df.iloc[0]
    if stage == "best":
        return history_df.iloc[int(history_df["pib"].astype(float).idxmax())]
    if stage == "final":
        return history_df.iloc[-1]
    if stage == "search_accept" and search_accept.any():
        accepted_idx = history_df.index[search_accept][0]
        return history_df.loc[accepted_idx]
    return history_df.iloc[-1]


def save_visualizations(
    summary_df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist CSV and diagnostic plots for the simulation suite."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_csv = output_path / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    artifacts = {"summary_csv": summary_csv}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = summary_df.sort_values("final_pib", ascending=False)
    x = np.arange(len(plot_df))
    axes[0].bar(x - 0.18, plot_df["final_pib"], width=0.36, label="Final PIB")
    axes[0].bar(x + 0.18, plot_df["best_pib"], width=0.36, label="Best PIB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["case"], rotation=25, ha="right")
    axes[0].set_title("PIB Summary by Case")
    axes[0].set_ylabel("PIB")
    axes[0].legend()

    axes[1].bar(x - 0.18, plot_df["global_score"], width=0.36, label="Global Score")
    axes[1].bar(x + 0.18, plot_df["local_score"], width=0.36, label="Local Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["case"], rotation=25, ha="right")
    axes[1].set_title("Final Basin Alignment")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    fig.tight_layout()
    summary_png = output_path / "summary_metrics.png"
    fig.savefig(summary_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    artifacts["summary_png"] = summary_png

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for case_name, history_df in histories.items():
        pib_values = history_df["pib"].astype(float).to_numpy()
        axes[0].plot(pib_values, label=case_name, linewidth=1.8)
        axes[1].plot(np.maximum.accumulate(pib_values), label=case_name, linewidth=1.8)
    axes[0].set_title("PIB Iteration Curves")
    axes[0].set_ylabel("PIB")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_title("Best-So-Far PIB")
    axes[1].set_ylabel("Best PIB")
    axes[1].set_xlabel("Recorder Step")
    fig.tight_layout()
    curves_png = output_path / "pib_curves.png"
    fig.savefig(curves_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    artifacts["curves_png"] = curves_png

    best_case = summary_df.sort_values("best_pib", ascending=False).iloc[0]["case"]
    baseline_case = "baseline_adamod"
    best_history = histories[best_case]
    baseline_history = histories[baseline_case]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, stage in enumerate(["init", "best", "final"]):
        baseline_row = _find_stage_row(baseline_history, stage)
        best_row = _find_stage_row(
            best_history, stage if stage != "best" else "search_accept"
        )
        axes[0, col].imshow(baseline_row["_img"], cmap="gray")
        axes[0, col].set_title(
            f"Baseline {stage}\nPIB={float(baseline_row['pib']):.3f}"
        )
        axes[0, col].axis("off")
        axes[1, col].imshow(best_row["_img"], cmap="gray")
        axes[1, col].set_title(f"{best_case} {stage}\nPIB={float(best_row['pib']):.3f}")
        axes[1, col].axis("off")
    fig.tight_layout()
    spots_png = output_path / "spot_stages.png"
    fig.savefig(spots_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    artifacts["spots_png"] = spots_png

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pib_values = best_history["pib"].astype(float).to_numpy()
    axes[0, 0].plot(pib_values, label="PIB", color="tab:blue")
    axes[0, 0].plot(
        np.maximum.accumulate(pib_values), label="Best PIB", color="tab:orange"
    )
    accepted_mask = pd.Series(best_history["search_accept"], dtype="boolean").fillna(
        False
    )
    accepted = best_history.index[accepted_mask]
    if len(accepted):
        axes[0, 0].scatter(
            accepted, pib_values[accepted], color="red", s=25, label="Search Accept"
        )
    axes[0, 0].set_title(f"{best_case} PIB Trace")
    axes[0, 0].legend()

    axes[0, 1].plot(best_history["search_radius"].astype(float), label="Search Radius")
    axes[0, 1].plot(best_history["tabu_size"].astype(float), label="Tabu Size")
    axes[0, 1].set_title("Search Diagnostics")
    axes[0, 1].legend()

    voltages = np.vstack(
        best_history["_v"].apply(lambda value: np.asarray(value, dtype=np.float64))
    )
    im = axes[1, 0].imshow(voltages.T, aspect="auto")
    axes[1, 0].set_title("Voltage History")
    axes[1, 0].set_xlabel("Recorder Step")
    axes[1, 0].set_ylabel("Unit")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(best_history["global_score"].astype(float), label="Global Score")
    axes[1, 1].plot(best_history["local_score"].astype(float), label="Local Score")
    axes[1, 1].set_title("Basin Occupancy")
    axes[1, 1].legend()
    fig.tight_layout()
    diagnostics_png = output_path / "best_case_diagnostics.png"
    fig.savefig(diagnostics_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    artifacts["diagnostics_png"] = diagnostics_png

    return artifacts


def _format_metric_table(summary_df: pd.DataFrame) -> str:
    columns = [
        "case",
        "final_pib",
        "best_pib",
        "search_accepts",
        "tabu_peak",
        "global_score",
        "local_score",
    ]
    table_df = summary_df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in table_df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(value)
            elif column in {"search_accepts", "tabu_peak"}:
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.6f}")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_report(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    artifacts: dict[str, Path],
) -> dict[str, Path]:
    """Generate Markdown and HTML summary reports for the regression suite."""
    output_path = Path(output_dir)
    best_case = summary_df.sort_values("best_pib", ascending=False).iloc[0]
    baseline = summary_df.loc[summary_df["case"] == "baseline_adamod"].iloc[0]
    best_final_gain = float(best_case["final_pib"] - baseline["final_pib"])
    best_best_gain = float(best_case["best_pib"] - baseline["best_pib"])

    summary_table = _format_metric_table(summary_df)
    markdown = f"""# PIB Hybrid Search Regression Report

## Overview

- Cases evaluated: `{len(summary_df)}`
- Best case by peak PIB: `{best_case["case"]}`
- Final PIB gain over baseline: `{best_final_gain:.6f}`
- Best PIB gain over baseline: `{best_best_gain:.6f}`

## Metrics

{summary_table}

## Figures

### Summary Metrics
![summary_metrics](./{artifacts["summary_png"].name})

### PIB Curves
![pib_curves](./{artifacts["curves_png"].name})

### Spot Stages
![spot_stages](./{artifacts["spots_png"].name})

### Best-Case Diagnostics
![best_case_diagnostics](./{artifacts["diagnostics_png"].name})

## Files

- Summary CSV: `{artifacts["summary_csv"].name}`
- Summary report HTML: `report.html`
"""
    markdown_path = output_path / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    rows_html = []
    for _, row in summary_df.iterrows():
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{float(row['final_pib']):.6f}</td>"
            f"<td>{float(row['best_pib']):.6f}</td>"
            f"<td>{int(row['search_accepts'])}</td>"
            f"<td>{int(row['tabu_peak'])}</td>"
            f"<td>{float(row['global_score']):.6f}</td>"
            f"<td>{float(row['local_score']):.6f}</td>"
            "</tr>"
        )
    html_report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PIB Hybrid Search Regression Report</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 32px auto;
      max-width: 1200px;
      line-height: 1.5;
      color: #182028;
      background: linear-gradient(180deg, #f5f1e8 0%, #ffffff 220px);
      padding: 0 24px 40px;
    }}
    h1, h2 {{ color: #23374d; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #d8d0c0;
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 8px 20px rgba(35, 55, 77, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 28px;
      background: white;
    }}
    th, td {{
      border: 1px solid #d9dde3;
      padding: 8px 10px;
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef3f8; }}
    figure {{ margin: 24px 0; }}
    img {{
      width: 100%;
      border: 1px solid #d9dde3;
      border-radius: 8px;
      background: white;
    }}
    figcaption {{
      margin-top: 8px;
      color: #4e5d6c;
      font-size: 0.95rem;
    }}
    code {{
      background: #f1f4f8;
      padding: 2px 6px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>PIB Hybrid Search Regression Report</h1>
  <p>This report summarizes the simulation regression matrix for the PIB hybrid optimizer.</p>

  <div class="metrics">
    <div class="card"><strong>Cases</strong><br>{len(summary_df)}</div>
    <div class="card"><strong>Best Case</strong><br>{html.escape(str(best_case["case"]))}</div>
    <div class="card"><strong>Final PIB Gain</strong><br>{best_final_gain:.6f}</div>
    <div class="card"><strong>Peak PIB Gain</strong><br>{best_best_gain:.6f}</div>
  </div>

  <h2>Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Final PIB</th>
        <th>Best PIB</th>
        <th>Search Accepts</th>
        <th>Tabu Peak</th>
        <th>Global Score</th>
        <th>Local Score</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>

  <h2>Figures</h2>
  <figure>
    <img src="{artifacts["summary_png"].name}" alt="Summary metrics">
    <figcaption>Case-level final/best PIB and basin alignment.</figcaption>
  </figure>
  <figure>
    <img src="{artifacts["curves_png"].name}" alt="PIB curves">
    <figcaption>Iteration PIB curves and best-so-far traces across the regression matrix.</figcaption>
  </figure>
  <figure>
    <img src="{artifacts["spots_png"].name}" alt="Spot stages">
    <figcaption>Spot snapshots for baseline and best-performing search case at init, search/best, and final stages.</figcaption>
  </figure>
  <figure>
    <img src="{artifacts["diagnostics_png"].name}" alt="Best case diagnostics">
    <figcaption>Detailed diagnostics for the best case, including search radius, tabu size, voltages, and basin occupancy.</figcaption>
  </figure>

  <p>Raw CSV: <code>{artifacts["summary_csv"].name}</code></p>
</body>
</html>
"""
    html_path = output_path / "report.html"
    html_path.write_text(html_report, encoding="utf-8")
    return {"report_md": markdown_path, "report_html": html_path}


def run_suite_and_save(
    output_dir: str | Path, cases: list[tuple[str, dict[str, object]]] | None = None
):
    """Convenience wrapper used by pytest/script entrypoints."""
    summary_df, histories, recorders = run_suite(cases=cases)
    artifacts = save_visualizations(
        summary_df=summary_df, histories=histories, output_dir=output_dir
    )
    artifacts.update(
        save_report(summary_df=summary_df, output_dir=output_dir, artifacts=artifacts)
    )
    return summary_df, histories, recorders, artifacts
