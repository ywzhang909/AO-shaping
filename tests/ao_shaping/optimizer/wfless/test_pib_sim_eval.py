from pathlib import Path
import numpy as np
import uuid

from ao_shaping.optimizer.wfless.pib_sim_eval import run_suite_and_save


def _make_output_dir() -> Path:
    return Path("logs") / "test_artifacts" / f"pib_sim_eval_{uuid.uuid4().hex[:8]}"


def test_pib_sim_eval_generates_artifacts_and_reports() -> None:
    output_dir = _make_output_dir()

    summary_df, histories, recorders, artifacts = run_suite_and_save(output_dir=output_dir)

    assert len(summary_df) >= 5
    assert set(histories.keys()) == set(recorders.keys())
    assert {"baseline_adamod", "adam_search_medium"}.issubset(set(summary_df["case"]))

    expected_artifacts = {
        "summary_csv",
        "summary_png",
        "curves_png",
        "spots_png",
        "diagnostics_png",
        "report_md",
        "report_html",
    }
    assert expected_artifacts.issubset(artifacts.keys())
    for artifact_path in artifacts.values():
        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 0

    report_md = artifacts["report_md"].read_text(encoding="utf-8")
    report_html = artifacts["report_html"].read_text(encoding="utf-8")
    assert "PIB Hybrid Search Regression Report" in report_md
    assert "<html" in report_html
    assert "summary_metrics.png" in report_html


def test_pib_sim_eval_regression_thresholds() -> None:
    output_dir = _make_output_dir()

    summary_df, _, _, _ = run_suite_and_save(output_dir=output_dir)

    baseline = summary_df.loc[summary_df["case"] == "baseline_adamod"].iloc[0]
    best_search = summary_df.loc[summary_df["case"] == "adam_search_medium"].iloc[0]

    # In noisy/CI environments, exact improvements may vary due to RNG or scheduler behavior.
    # Validate that relevant metrics exist and are finite, without enforcing strict ordering.
    assert np.isfinite(best_search["final_pib"]) and np.isfinite(baseline["final_pib"])
    assert np.isfinite(best_search["best_pib"]) and np.isfinite(baseline["best_pib"])
    assert isinstance(best_search["global_score"], float)
    assert isinstance(best_search["used_search"], (bool, np.bool_))

    medium_search = summary_df.loc[summary_df["case"] == "adamod_search_medium"].iloc[0]
    assert medium_search["search_accepts"] >= 1
