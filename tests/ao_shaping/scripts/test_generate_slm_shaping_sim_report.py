"""Regression tests for ``scripts/generate_slm_shaping_sim_report.py``.

Three contracts matter here, and all three were violated at least once during
development:

1. **The DM family must never be presented as optimisation performance.**
   ``SimPibSystem.far_field()`` reads only ``self._phase`` (written solely by
   ``set_phase_rad``), so ``pib`` / ``combined`` drive a DM whose voltage never
   reaches the model. Their loops execute; their objective is uncoupled noise. A
   report that tabulates them next to the SLM runners invites the reader to
   conclude "SPGD fails on DM-PIB", which is false.

2. **Best and last must be reported separately.** The last epoch is routinely
   worse than the best; collapsing them overstates the result. (The sibling
   ``generate_slm_pib_sim_report.py`` had exactly this bug.)

3. **A short run must carry a caveat.** At a smoke budget the "best" value is a
   noise-driven single-point maximum that usually lands in the first few epochs,
   so quoting it bare is an overclaim.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_generator():
    script = ROOT / "scripts" / "generate_slm_shaping_sim_report.py"
    spec = importlib.util.spec_from_file_location("_slm_shaping_sim_report", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slm_pib_run(epochs: int = 13, best_at: int = 1):
    """A slm-pib-style recorder: J / _p% plus image frames."""
    data = {}
    for e in range(epochs):
        pct = 0.63 + (0.005 if e == best_at else -0.04 * e / max(epochs - 1, 1))
        data[e] = {
            "J": 7845.0 / (e + 1),
            "_p%": pct,
            "_max_r": 30.0 + e,
            "max_brt": 99.0,
            "_img": np.full((16, 16), 10.0 + e, dtype=float),
        }
    return list(range(epochs)), data, None


def _gsnet_run(epochs: int = 13):
    data = {}
    for e in range(epochs):
        data[e] = {
            "J": -4.9 + 0.001 * e,
            "quality": 0.588 + 0.002 * (e % 3),
            "cv": 4.8,
            "ee": 0.98,
            "ar": 1.0 + 0.01 * e,
            "_img": np.full((16, 16), 20.0 + e, dtype=float),
        }
    return list(range(epochs)), data, None


def _render(tmp_path, runs, manifest=None):
    gen = _load_generator()
    out = tmp_path / "report.md"
    gen.build_markdown(runs, [], out, manifest)
    return out.read_text(encoding="utf-8")


class TestHonestyAboutDmFamily:
    """The DM runners are smoke tests; the report must say so."""

    def test_report_warns_that_dm_is_not_modelled(self, tmp_path):
        text = _render(tmp_path, {"slm-pib": _slm_pib_run()})
        assert "只作为控制环冒烟测试" in text or "不画收敛曲线" in text, (
            "the report must state that DM runners are smoke tests only"
        )

    def test_smoke_table_carries_the_warning(self, tmp_path):
        manifest = {
            "runs": [
                {
                    "runner": "pib",
                    "family": "dm",
                    "ok": True,
                    "returncode": 0,
                    "wall_s": 33.8,
                    "artefacts": ["x.csv"],
                    "note": "control-loop smoke test only",
                }
            ]
        }
        text = _render(tmp_path, {"slm-pib": _slm_pib_run()}, manifest)
        assert "pib" in text
        assert "冒烟测试" in text
        assert "不能" in text, "the report must warn against reading DM as convergence"

    def test_dm_runner_is_absent_from_the_results_table(self, tmp_path):
        """A DM runner must not appear in the SLM results table at all."""
        runs = {"slm-pib": _slm_pib_run(), "pib": _slm_pib_run()}
        gen = _load_generator()
        out = tmp_path / "r.md"
        gen.build_markdown(runs, [], out, None)
        text = out.read_text(encoding="utf-8")
        table_line = next(
            (ln for ln in text.splitlines() if ln.startswith("| `pib`")), ""
        )
        assert table_line == "", (
            f"pib is a DM runner and must not be tabulated as a result: {table_line!r}"
        )


class TestBestVersusLast:
    def test_best_and_last_are_both_present(self, tmp_path):
        text = _render(tmp_path, {"slm-pib": _slm_pib_run(epochs=13, best_at=1)})
        assert "最佳" in text and "末轮" in text and "初始" in text

    def test_best_uses_the_maximum_not_the_last_row(self, tmp_path):
        """best_at=1 while the last row regresses -> best must be the max."""
        gen = _load_generator()
        epochs, data, _ = _slm_pib_run(epochs=13, best_at=1)
        y = gen._scalar_series(data, epochs, "_p%")
        assert y[1] == y.max(), "fixture must actually peak at epoch 1"
        assert y[-1] < y[1], "fixture must actually regress by the last epoch"

        text = _render(tmp_path, {"slm-pib": (epochs, data, _)})
        best_row = next(
            (ln for ln in text.splitlines() if ln.startswith("| `slm-pib`")), ""
        )
        assert f"{y.max():.4g}" in best_row
        assert f"{y[-1]:.4g}" in best_row


class TestShortBudgetCaveat:
    def test_short_run_gets_a_caveat(self, tmp_path):
        text = _render(tmp_path, {"slm-pib": _slm_pib_run(epochs=13)})
        assert "冒烟预算" in text and "不构成收敛性结论" in text

    def test_long_run_has_no_short_budget_caveat(self, tmp_path):
        text = _render(tmp_path, {"slm-pib": _slm_pib_run(epochs=120)})
        assert "冒烟预算" not in text


class TestImageSeriesRobustness:
    def test_scalar_values_are_rejected(self):
        """A CSV history parses ``_img`` to a scalar NaN; that must not become a frame.

        Regression guard: this produced ``TypeError: Invalid shape () for image
        data`` from ``imshow`` before the ndim guard was added.
        """
        gen = _load_generator()
        frames = gen._image_series({0: {"_img": float("nan")}}, [0])
        assert frames == [None]

    def test_real_frames_are_kept(self):
        gen = _load_generator()
        arr = np.zeros((4, 4))
        frames = gen._image_series({0: {"_img": arr}}, [0])
        assert frames[0] is not None and frames[0].shape == (4, 4)

    def test_missing_key_is_none(self):
        gen = _load_generator()
        assert gen._image_series({0: {}}, [0]) == [None]


class TestNoArtefactsDegradesGracefully:
    def test_empty_runs_still_writes_a_report(self, tmp_path):
        text = _render(tmp_path, {})
        assert "SLM" in text
        assert "未找到" in text or "请先运行" in text

    def test_manifest_absent_is_tolerated(self, tmp_path):
        gen = _load_generator()
        out = tmp_path / "r.md"
        gen.build_markdown({"slm-pib": _slm_pib_run()}, [], out, None)
        assert out.is_file()

    def test_generator_knows_only_the_slm_runners(self):
        """The report's allow-list must be the three SLM-driven runners."""
        gen = _load_generator()
        assert set(gen.HEADLINE) == {"slm-pib", "slm-gsnet", "spgd-square"}
        assert "pib" not in gen.HEADLINE
        assert "combined" not in gen.HEADLINE

    def test_batch_driver_classifies_the_five_runners(self):
        """``FAMILY`` lives in the batch driver and marks the DM pair."""
        import importlib.util as iu

        spec = iu.spec_from_file_location(
            "_run_sim_bench", ROOT / "scripts" / "run_sim_bench.py"
        )
        mod = iu.module_from_spec(spec)
        # A module using @dataclass resolves its own module through
        # sys.modules, so register before executing.
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
            assert set(mod.FAMILY) == {
                "pib",
                "combined",
                "spgd-square",
                "slm-gsnet",
                "slm-pib",
            }
            assert mod.FAMILY["pib"] == "dm"
            assert mod.FAMILY["slm-pib"] == "slm"
        finally:
            sys.modules.pop(spec.name, None)
