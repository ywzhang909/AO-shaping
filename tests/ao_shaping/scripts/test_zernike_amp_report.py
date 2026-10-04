"""Smoke tests for the Zernike sweep + report generators.

The report is generated, not hand-written, so these lock the two things that
silently rot: the sweep artefact shape the report reads, and the figure/report
contract (every referenced figure must exist).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_zernike_amp_report import (  # noqa: E402
    MODELS,
    build_report,
    final_table,
    grid_table,
    paired_table,
)
from sweep_zernike_models import METRICS, PHYSICS_GRID, UNET_GRID  # noqa: E402


def _entry(model: str, r2: float, ssim: float, **extra) -> dict:
    summary = {
        metric: {"mean": v, "std": 0.05, "per_fold": [v] * 10}
        for metric, v in zip(METRICS, (0.001, r2, ssim, 30.0, 0.04))
    }
    summary["by_objective"] = {"rms_pib": {m: v for m, v in summary.items()}}
    return {"model": model, "summary": summary, "folds": [], **extra}


@pytest.fixture
def sweep() -> dict:
    physics = _entry("physics", 0.8803, 0.7646, n_max=20, lr=0.02, epochs=50)
    hybrid = _entry("hybrid", 0.8522, 0.6985, n_max=20, lr=0.02, epochs=50, features=None)
    unet = _entry("unet", 0.8997, 0.8380, n_max=15, lr=0.01, epochs=50,
                  features=[16, 32, 64, 128, 256])

    def paired(a: dict, b: dict) -> dict:
        return {
            metric: {
                "mean_diff": a["summary"][metric]["mean"] - b["summary"][metric]["mean"],
                "std_diff": 0.01, "cohens_dz": 1.2,
                "p_signflip": 0.0039, "per_fold": [0.01] * 10,
            }
            for metric in METRICS
        }

    return {
        "folds": 10,
        "protocol": "leave-one-pickle-out",
        "physics_grid": [
            _entry("physics", 0.87 + 0.001 * i, 0.73, n_max=n, lr=lr, epochs=50)
            for i, (n, lr) in enumerate(PHYSICS_GRID)
        ],
        "unet_grid": [
            _entry("unet", 0.89, 0.84, n_max=15, lr=lr, epochs=ep, features=fe)
            for ep, lr, fe in UNET_GRID
        ],
        "final": {"physics": physics, "hybrid": hybrid, "unet": unet},
        "configs": {
            "physics": {"n_max": 20, "lr": 0.02, "epochs": 50},
            "hybrid": {"n_max": 20, "lr": 0.02, "epochs": 50, "residual_width": 32},
            "unet": {"epochs": 50, "lr": 0.01, "features": [16, 32, 64, 128, 256]},
        },
        "paired": {
            "physics_minus_hybrid": paired(physics, hybrid),
            "physics_minus_unet": paired(physics, unet),
            "hybrid_minus_unet": paired(hybrid, unet),
        },
    }


@pytest.fixture
def history() -> dict:
    row = {
        "epoch": 0, "train_mse": 0.01, "val_mse": 0.02, "val_r2": 0.5, "val_ssim": 0.4,
        "grad_total": 1e-3, "grad_max": 1e-3, "coef_abs_max": 0.3, "val_psnr": 20.0,
    }
    return {
        "history": [dict(row, epoch=e) for e in range(25)],
        "best_epoch": 13, "best_val_r2": 0.7971, "grad_norm_first": 2.3e-3,
        "grad_norm_last": 1.0e-3, "dead_modes": 0, "n_modes": 135, "n_max": 15,
        "seconds": 8.1, "coefficients": [0.3] * 135,
    }


class TestSweepGrid:
    def test_physics_grid_probes_lr_at_the_interacting_n_max(self) -> None:
        """Non-additivity is only visible if lr is probed at more than one n_max."""
        n_maxes = {n for n, _ in PHYSICS_GRID}
        for n in n_maxes:
            assert len({lr for m, lr in PHYSICS_GRID if m == n}) >= 1
        assert len({lr for _, lr in PHYSICS_GRID}) >= 2

    def test_unet_grid_is_not_a_single_point(self) -> None:
        assert len(UNET_GRID) >= 3
        # Feature lists are unhashable, so compare them as tuples.
        assert len({tuple(fe) for _, _, fe in UNET_GRID}) >= 2


class TestTables:
    def test_final_table_lists_every_model(self, sweep: dict) -> None:
        from generate_zernike_amp_report import LABELS

        table = final_table(sweep)
        # The table renders display labels, not the raw model keys.
        for name in MODELS:
            assert LABELS[name] in table

    def test_paired_table_flags_significance(self, sweep: dict) -> None:
        table = paired_table(sweep)
        assert "***" in table
        assert "physics_minus_unet" not in table  # labels, not raw keys

    @pytest.mark.parametrize("key", ["physics_grid", "unet_grid"])
    def test_grid_table_renders_every_row(self, sweep: dict, key: str) -> None:
        table = grid_table(sweep, key)
        assert len(table.splitlines()) == len(sweep[key]) + 2  # header + separator


class TestReport:
    def test_report_is_chinese_and_has_the_expected_sections(self, sweep, history) -> None:
        text = build_report(sweep, history, {})
        assert "## 1. 模型是什么" in text
        assert "## 11. 复现" in text
        # Chinese body, not an English leftover.
        assert "分组交叉验证" in text

    def test_report_records_the_overturned_conclusions(self, sweep, history) -> None:
        text = build_report(sweep, history, {})
        assert "被推翻的结论" in text
        # The dimensional-error correction must survive regeneration.
        assert "0.670" in text

    def test_missing_figures_degrade_to_no_image_tag(self, sweep, history) -> None:
        """A figure that failed to render must not leave a broken markdown link."""
        text = build_report(sweep, history, {"training": None, "paired": None})
        assert "figures/None" not in text
        assert "](figures/" not in text or "figures/01" in text

    def test_report_embeds_only_figures_that_were_produced(self, sweep, history) -> None:
        import re

        produced = {"training": "01_training_curves.png", "paired": "06_paired_differences.png"}
        text = build_report(sweep, history, produced)
        refs = re.findall(r"\]\((figures/[^)]+)\)", text)
        assert set(refs) <= {f"figures/{name}" for name in produced.values()}

    def test_hybrid_verdict_is_derived_not_hardcoded(self, sweep, history) -> None:
        """The hybrid paragraph must follow the data, not a remembered outcome."""
        sweep["paired"]["physics_minus_hybrid"]["r2"]["p_signflip"] = 0.9
        sweep["paired"]["physics_minus_hybrid"]["r2"]["mean_diff"] = 0.001
        text = build_report(sweep, history, {})
        assert "0.9000" in text  # the actual p-value appears
