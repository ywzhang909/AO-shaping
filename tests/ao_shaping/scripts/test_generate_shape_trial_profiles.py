"""Offline tests for ``scripts/slm/generate_shape_trial_profiles.py``.

The script's job is to re-derive report figures from recorded raw frames, so the
tests guard the parts that would silently produce *wrong* plots rather than just
crash: the provenance resolver, the ``best_frame`` fallback when Step B is
rejected, the provenance subtitle, and the output-name collision that three runs
sharing ``best_far_field.png`` would otherwise cause.

The two-panel rendering and the cut-point convention are **not** owned by this
script any more: they live in ``ao_shaping.display.beam_profile``, the single
implementation shared with a run's ``Recorder`` debug artefacts. The tests that
guard them here call the shared ``cut_point`` rather than a private copy, and
``test_script_delegates_instead_of_reimplementing`` is the anti-drift guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "generate_shape_trial_profiles.py"

sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.display import beam_profile


def _load_module():
    spec = importlib.util.spec_from_file_location("shape_trial_profiles", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()

# 这些用例需要 data/debug/ 下的原始帧 (gitignore `data/*`, 只在采集机上存在)。
# 没有数据的机器 (CI / 干净克隆) 上 resolver 走 unresolved 分支, 断言必挂,
# 故按仓库惯例 (skipif not is_dir) 在缺数据时跳过, 在采集机上照常执行。
_MIL = ROOT / "data" / "debug" / "slm_model_in_loop"
_REPLAY = ROOT / "data" / "debug" / "slm_model_in_loop_replay"
_GS = ROOT / "data" / "debug" / "slm_gs_refine"
_NEEDS_20261019 = not (_MIL / "20261019_103208").is_dir()
_NEEDS_REPLAY_044 = not (_REPLAY / "20261016_1044_uniformity_weights").is_dir()
_NEEDS_REPLAY_115 = not (_REPLAY / "20261014_1115_zero_vs_refit").is_dir()
_NEEDS_GS_2058 = not (_GS / "20261015_205848").is_dir()
_NEEDS_MIL_RUNS = not (
    all(
        (_MIL / r).is_dir()
        for r in ("20261013_114650", "20261013_145819", "20261013_152341")
    )
)


class TestResolution:
    @pytest.mark.skipif(
        _NEEDS_20261019, reason="requires data/debug/slm_model_in_loop/20261019_103208"
    )
    def test_direct_farfield_figure_maps_to_its_run(self):
        spec = mod._resolve_one("figures/20261019_103208_farfield.png")
        assert spec.container.endswith("20261019_103208/preflight/flat.npz")
        assert spec.key == "frame"
        assert spec.verify_mode == "byte"

    @pytest.mark.skipif(
        _NEEDS_REPLAY_044,
        reason="requires data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights",
    )
    def test_doubled_date_prefix_replay_resolves(self):
        spec = mod._resolve_one(
            "figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_01_A_farfield.png"
        )
        assert spec.verify_mode == "structural"
        assert spec.container == (
            "data/debug/slm_model_in_loop_replay/20261016_1044_uniformity_weights/01_A.npy"
        )

    @pytest.mark.skipif(
        _NEEDS_REPLAY_115,
        reason="requires data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit",
    )
    def test_single_date_prefix_replay_resolves(self):
        """The undoubled form must not be broken by stripping its date prefix."""
        spec = mod._resolve_one(
            "figures/model_in_loop_live_20261014_1115_zero_vs_refit_05_A_farfield.png"
        )
        assert spec.container == (
            "data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit/05_A.npy"
        )

    @pytest.mark.skipif(
        _NEEDS_GS_2058, reason="requires data/debug/slm_gs_refine/20261015_205848"
    )
    def test_gs_issue_figure_resolves_to_pkl(self):
        spec = mod._resolve_one("figures/gs_issue_live_farfield_20261015_205848.png")
        assert spec.container.endswith(
            "20261015_205848/slm_gs_refine_20261015_205848.pkl"
        )
        assert spec.verify_mode == "content"

    def test_figures_without_raw_data_are_reported_not_guessed(self):
        for figure in (
            "figures/gs_pad3_best_far_field.png",
            "figures/gs_pad3_debug_summary.png",
        ):
            spec = mod._resolve_one(figure)
            assert spec.verify_mode == "unresolved"
            assert spec.container == ""
            assert spec.note


class TestBestFrameFallback:
    @pytest.mark.skipif(
        _NEEDS_20261019, reason="requires data/debug/slm_model_in_loop/20261019_103208"
    )
    def test_rejected_run_plots_the_flat_baseline(self):
        """A run that never clears the holdout gate must resolve to the baseline.

        ``slm_model_in_loop`` seeds ``best_frame`` with the flat frame, so
        plotting a Step-B frame here would attribute a baseline image to a
        shaping result.
        """
        container, key, why = mod._mil_best_frame("20261019_103208")
        assert container.endswith("preflight/flat.npz")
        assert key == "frame"
        assert "平相位" in why

    def test_missing_run_resolves_to_nothing(self):
        assert mod._mil_best_frame("20991231_235959") is None


class TestOutputNaming:
    @pytest.mark.skipif(
        _NEEDS_MIL_RUNS, reason="requires data/debug/slm_model_in_loop/20261013_* runs"
    )
    def test_generic_stem_is_disambiguated_by_run(self):
        """Three runs all link ``best_far_field.png``; they must not collide."""
        paths = set()
        for run in ("20261013_114650", "20261013_145819", "20261013_152341"):
            spec = mod._resolve_one(
                f"../../../data/debug/slm_model_in_loop/{run}/best_far_field.png"
            )
            paths.add(mod.out_path_for(spec).name)
        assert len(paths) == 3
        assert {p.split("_xy_profile")[0] for p in paths} == {
            "20261013_114650",
            "20261013_145819",
            "20261013_152341",
        }

    def test_gui_panels_get_distinct_names(self):
        names = {
            mod.out_path_for(
                mod.FrameSpec(
                    figure=f"images/gui_shaping_20261014/fig3_far_field.png#{panel}",
                    label=panel,
                )
            ).name
            for panel in ("A_flat", "A_best", "C_flat", "C_best")
        }
        assert len(names) == 4


class TestCutPoint:
    """The cut-point convention now lives in ``ao_shaping.display.beam_profile``.

    This script delegates to the canonical renderer (the same one a run's
    ``Recorder`` uses for its debug artefacts), so the convention is guarded
    there rather than here. The scenario is kept verbatim: it is the regression
    that stops someone "simplifying" the centroid back to ``argmax``.
    """

    def test_cut_is_centroid_not_argmax(self):
        """On a speckle field the brightest grain is not the beam centre.

        The cut must follow the intensity centroid; ``argmax`` is only a
        reported cross-check. One bright grain 12 px off a broad beam must pull
        ``argmax`` all the way over while barely moving the centroid.
        """
        frame = np.zeros((80, 80), dtype=float)
        frame[22:42, 22:42] = 1.0  # the broad beam, 400 counts total
        frame[10, 34] = 5.0  # one brighter grain, inside the beam's span
        cut = beam_profile.cut_point(frame)
        # the beam alone centres on 31.5; the off-centre grain shifts it <0.5 px
        assert (cut.row, cut.col) == pytest.approx((31.5, 31.5), abs=0.5)
        # argmax jumps clean to the grain instead -- 21 px away from the cut
        assert (cut.argmax_row, cut.argmax_col) == (10.0, 34.0)
        assert abs(cut.argmax_row - cut.row) > 15.0

    def test_background_pedestal_is_removed(self):
        frame = np.full((40, 40), 7.0)
        frame[20, 20] = 9.0
        cut = beam_profile.cut_point(frame)
        assert cut.background_median == pytest.approx(7.0)
        assert cut.peak == pytest.approx(2.0)

    def test_script_delegates_instead_of_reimplementing(self):
        """No second copy of the convention may creep back into the script."""
        assert mod.cut_point is beam_profile.cut_point
        assert mod.save_xy_profile is beam_profile.save_xy_profile
        assert not hasattr(mod, "_spot_centre")


class TestProvenanceSubtitle:
    def test_subtitle_names_the_source_container_key_shape_and_tier(self):
        """The footer is the script's own value-add; it must stay on the figure."""
        spec = mod.FrameSpec(
            figure="figures/x_farfield.png",
            label="x",
            container="data/debug/slm_model_in_loop/20261019_103208/preflight/flat.npz",
            key="frame",
            verify_mode="byte",
        )
        sub = mod.provenance_subtitle(spec, np.zeros((300, 300)))
        assert "data/debug/slm_model_in_loop/20261019_103208/preflight/flat.npz" in sub
        assert "· frame" in sub
        assert "(300, 300)" in sub
        assert "PNG 逐字节校验通过" in sub


class TestDiscovery:
    def test_every_referenced_spot_figure_is_accounted_for(self):
        specs = mod.discover()
        assert specs
        for spec in specs:
            assert spec.label, spec.figure
            if spec.verify_mode == "unresolved":
                assert spec.note, spec.figure
            else:
                assert spec.container, spec.figure

    def test_target_patterns_are_excluded(self):
        """``*_farfield_target.png`` are synthesised targets with no recorded array."""
        assert not any("target" in s.figure for s in mod.discover())
