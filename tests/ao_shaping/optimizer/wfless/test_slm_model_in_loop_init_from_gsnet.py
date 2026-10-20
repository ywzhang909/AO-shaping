"""Tests for the ``--init-from-gsnet`` Step-B warm start and its three gates.

Device-free by construction: the loader and the pure decision helper
``_gsnet_init_admissible`` carry the adoption contract, so the gates are
exercised without a bench. The runner CLI is exercised through click's
test runner and the params->config builders.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _gsnet_init_admissible,
    _load_gsnet_phase,
)


def _write_phase(path, side: int, value: float = 0.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((side, side), value, dtype=np.float64))
    return str(path)


class TestLoadGsnetPhase:
    def test_returns_float64_array_for_valid_file(self, tmp_path):
        file = tmp_path / "gsnet_phase.npy"
        path = _write_phase(file, 8, value=0.1)

        phase = _load_gsnet_phase(path, 8)

        assert phase.shape == (8, 8)
        assert phase.dtype == np.float64
        assert np.allclose(phase, 0.1)

    def test_wrong_shape_raises(self, tmp_path):
        path = _write_phase(tmp_path / "bad.npy", 7)

        with pytest.raises(ValueError, match="shape"):
            _load_gsnet_phase(path, 8)

    def test_non_npy_suffix_raises(self, tmp_path):
        file = tmp_path / "not_phase.txt"
        file.write_text("0.1 0.2\n", encoding="utf-8")

        with pytest.raises(ValueError):
            _load_gsnet_phase(str(file), 8)

    def test_unreadable_npy_raises(self, tmp_path):
        file = tmp_path / "corrupt.npy"
        file.write_bytes(b"definitely not npy")

        with pytest.raises(ValueError, match="readable"):
            _load_gsnet_phase(str(file), 8)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _load_gsnet_phase(str(tmp_path / "absent.npy"), 8)

    def test_non_finite_phase_raises(self, tmp_path):
        file = tmp_path / "nan.npy"
        np.save(file, np.full((8, 8), np.nan))

        with pytest.raises(ValueError, match="non-finite"):
            _load_gsnet_phase(str(file), 8)


class TestConfigValidation:
    def test_missing_path_raises_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="init_from_gsnet"):
            SlmModelInLoopConfig(init_from_gsnet=str(tmp_path / "absent.npy"))

    def test_wrong_shape_raises_at_construction(self, tmp_path):
        path = _write_phase(tmp_path / "wrong.npy", 16)

        with pytest.raises(ValueError, match="init_from_gsnet"):
            SlmModelInLoopConfig(init_from_gsnet=path)

    def test_valid_path_is_accepted_and_kept(self, tmp_path):
        path = _write_phase(tmp_path / "ok.npy", 64)

        config = SlmModelInLoopConfig(init_from_gsnet=path)

        assert config.init_from_gsnet == path

    def test_default_none_keeps_historical_path(self):
        assert SlmModelInLoopConfig().init_from_gsnet is None


class TestGsRefineConfigValidation:
    def test_missing_path_raises_at_construction(self, tmp_path):
        from ao_shaping.optimizer.wfless.slm_gs_refine import SlmGsRefineConfig

        with pytest.raises(ValueError, match="init_from_gsnet"):
            SlmGsRefineConfig(init_from_gsnet=str(tmp_path / "absent.npy"))

    def test_default_none(self):
        from ao_shaping.optimizer.wfless.slm_gs_refine import SlmGsRefineConfig

        assert SlmGsRefineConfig().init_from_gsnet is None


class TestGsnetInitAdmissible:
    """The three mandatory gates: trust region, measured bake-off, Pearson."""

    def test_all_gates_pass_is_adopted(self):
        assert _gsnet_init_admissible(
            phase_l2=0.3,
            trust_region_c_l2=0.5,
            score_candidate=0.61,
            score_flat=0.58,
            holdout_correlation=0.72,
            min_holdout_correlation=0.5,
        ) == "adopted"

    def test_trust_region_violation_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.9,
            trust_region_c_l2=0.5,
            score_candidate=0.9,
            score_flat=0.1,
            holdout_correlation=1.0,
            min_holdout_correlation=0.5,
        ) == "trust_region_rejected"

    def test_bake_off_loss_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.5,
            score_flat=0.58,
            holdout_correlation=0.9,
            min_holdout_correlation=0.5,
        ) == "flat_bake_off_rejected"

    def test_bake_off_tie_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.58,
            score_flat=0.58,
            holdout_correlation=0.9,
            min_holdout_correlation=0.5,
        ) == "flat_bake_off_rejected"

    def test_holdout_correlation_below_floor_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.9,
            score_flat=0.58,
            holdout_correlation=0.31,
            min_holdout_correlation=0.5,
        ) == "holdout_correlation_rejected"

    def test_missing_bake_off_evidence_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=None,
            score_flat=0.58,
            holdout_correlation=0.9,
            min_holdout_correlation=0.5,
        ) == "flat_bake_off_rejected"

    def test_missing_flat_reference_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.9,
            score_flat=None,
            holdout_correlation=0.9,
            min_holdout_correlation=0.5,
        ) == "flat_bake_off_rejected"

    def test_missing_holdout_evidence_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.9,
            score_flat=0.58,
            holdout_correlation=None,
            min_holdout_correlation=0.5,
        ) == "holdout_correlation_rejected"

    def test_nan_evidence_rejects(self):
        assert _gsnet_init_admissible(
            phase_l2=0.1,
            trust_region_c_l2=0.5,
            score_candidate=0.9,
            score_flat=0.58,
            holdout_correlation=float("nan"),
            min_holdout_correlation=0.5,
        ) == "holdout_correlation_rejected"

    def test_trust_region_zero_disables_only_that_gate(self):
        assert _gsnet_init_admissible(
            phase_l2=10.0,
            trust_region_c_l2=0.0,
            score_candidate=None,
            score_flat=0.58,
            holdout_correlation=0.9,
            min_holdout_correlation=0.5,
        ) == "flat_bake_off_rejected"

    def test_trust_region_checked_before_the_measurement_gates(self):
        assert _gsnet_init_admissible(
            phase_l2=0.9,
            trust_region_c_l2=0.5,
            score_candidate=None,
            score_flat=None,
            holdout_correlation=None,
            min_holdout_correlation=0.5,
        ) == "trust_region_rejected"


class TestCliWiring:
    def test_builders_map_the_field(self, tmp_path):
        from ao_shaping.runners.runner_common import (
            SlmGsRefineParams,
            SlmModelInLoopParams,
            build_gs_refine_config,
            build_model_in_loop_config,
        )
        from ao_shaping.utils.slm.phase_embed import pupil_side

        mil_path = _write_phase(tmp_path / "mil.npy", 64)
        gs_side = pupil_side(450.0)
        gs_path = _write_phase(tmp_path / "gs.npy", gs_side)

        mil_config = build_model_in_loop_config(
            SlmModelInLoopParams(init_from_gsnet=mil_path), debug=False
        )
        gs_config = build_gs_refine_config(
            SlmGsRefineParams(init_from_gsnet=gs_path), debug=False
        )

        assert mil_config.init_from_gsnet == mil_path
        assert gs_config.init_from_gsnet == gs_path

    @pytest.mark.parametrize(
        "command",
        ["slm-model-in-loop", "slm-gs-refine"],
    )
    def test_registered_commands_expose_the_option(self, command):
        from ao_shaping.main import cli

        result = CliRunner().invoke(cli, [command, "--help"])

        assert result.exit_code == 0, result.output
        assert "--init-from-gsnet" in result.output
