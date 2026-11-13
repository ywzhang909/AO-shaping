"""Shape guard for the sweep-calibration split, issue #205 batch 1.

#205 registered the repo-wide long-function / deep-nesting debt and asked for
per-family batches, each characterization-first and each sealed with a shape
guard so a split cannot grow back. Batch 1 (wfless family) split
``calibrate_bench_geometry_from_sweep`` (662 lines, nesting 4) into nine
module-level helpers by **verbatim** block moves: the bit-for-bit numeric
goldens already in ``test_model_in_loop_sweep_diagnostics.py``
(``REF_ALL_NORMAL_NOTES`` / ``REF_ALL_NORMAL``) pin that the move did not move
a single digit.

The goldens pin equality for today; they do not stop any of the ten functions
from growing back. So the ten are pinned here by AST: span (end_lineno -
lineno + 1) and maximum nesting depth (if / for / while / try / with), the
same convention as ``test_issue167_function_shape.py``. Ceilings are measured
value plus headroom -- except the nesting ceilings at the #167 criterion (<=3)
where the measured value already sits: the two filter/fit blocks moved
verbatim measure nesting 4, so their ceiling is 4 with flattening left to a
later #205 batch, not hidden here by raising the bar repo-wide.

Scope: these functions only. The rest of #205's inventory
(``optimize_slm_square`` at 1287, ``optimize_pib`` at nesting 8, ...) lands in
its own batches with its own ceilings -- pinning it here would turn this guard
into a repo-wide lint that no single batch can pay.

Batch 2 (scripts/slm family) added the ``build_markdown`` split of
``generate_diffraction_efficiency_sim_report.py`` (596 lines, nesting 0): the
section f-strings moved line-exact into three helpers behind a rebind header,
sealed here with the same span/nesting convention. Paths are repo-relative
from batch 2 on, since the guard now covers ``scripts/`` as well as ``src/``.
Batch 3 (same family) added the ``render_retest_section`` split of
``_hw_gs_square_scans_sections.py`` (537 lines, nesting 5) by the same
rebind-header pattern; its nesting-5 zernike block moved verbatim and keeps
ceiling 5 -- flattening is a later batch, not a raised bar here.
Batch 4 (wfless family) lifted the 193-line tqdm epoch block of
``optimize_pib`` (582 lines, nesting 8 -- the repo's deepest) into
``_run_spgd_epochs``: read-only closure names enter as keyword params, the six
outer locals the loop mutates are threaded back through a returned namespace,
and the per-epoch sha256 characterization in
``test_spid_pib_path_characterization.py`` is the behavior net. Nesting 8 fell
to 5 in the orchestrator and 6 in the lifted block -- the block's own deep
nesting moved verbatim and keeps its measured ceiling.
Batch 5 (same family, ``combined_optimizer``) lifted the 225-line
open_pib_bench With block the same way -- 16 read-only params, 5 threaded
mutables; the orchestrator went from nesting 6 to 0.
Batch 6 (tools/slm/transfer) lifted the 325-line open-camera With block of
``validate.run`` (535 lines, nesting 9) into ``_run_validate_session``; the
net is 7 transfer suites (test_transfer_workflow.py drives run() end-to-end),
and the resume-gate source pin in ``test_validate_blocks.py`` moved to two
hops (run -> session -> gate). Nesting 9 in the lifted block keeps its
measured ceiling -- flattening is a later batch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_FILE = "src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py"
_PIB = "src/ao_shaping/optimizer/wfless/pib.py"
_COMB = "src/ao_shaping/optimizer/combined_optimizer.py"
_VAL = "src/ao_shaping/tools/slm/transfer/validate.py"
_SIM2F = "src/ml/sim2f/loop.py"
_DM_RESPONSE_MATRIX = "src/ao_shaping/optimizer/wf/dm_response_matrix.py"
_BENCH = "src/ml/sim2f/bench.py"
_ZERNIKE_MATRIX_RUNNER = "src/ao_shaping/runners/slm/zernike_matrix_runner.py"
_TRANSFER_SHAPES_RUNNER = "src/ao_shaping/runners/slm/transfer_shapes_runner.py"
_PANEL_LOCATE = "src/ao_shaping/tools/slm/panel/panel_locate.py"
_CAPTURE = "src/ao_shaping/tools/slm/transfer/capture.py"
_ZERNIKE_DEBUG_VIEWER = "src/ao_shaping/gui/zernike/zernike_debug_viewer.py"
_THORLAB_WFS = "src/ao_shaping/drivers/wfs/_thorlab_wfs.py"
_TRAIN_AMP = "src/ml/zernike/train_amp.py"
_TRAIN_COEFF = "src/ml/zernike/train_coeff.py"
_WF = "src/ao_shaping/tools/slm/transfer/workflow.py"
_LUT = "src/ao_shaping/tools/slm/calibration/lut_runner.py"
_IO = "src/ao_shaping/utils/io/file.py"
_DM = "src/ao_shaping/tools/kernels/device_mutex.py"
_PAT = "src/ao_shaping/utils/image/target/patterns.py"
_MEAS = "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py"
_SZP = "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py"
_RMS = "src/ao_shaping/optimizer/wf/rms_by_zernike.py"
_MIL = "src/ao_shaping/optimizer/wfless/slm_model_in_loop.py"
_SQ = "src/ao_shaping/optimizer/wfless/slm_square_shaping.py"
_SCRIPT = "scripts/slm/generate_diffraction_efficiency_sim_report.py"
_SECTIONS = "scripts/slm/_hw_gs_square_scans_sections.py"
_WORKFLOW = "src/ao_shaping/tools/slm/transfer/workflow.py"
_DIFFERENTIABLE_BEAM = "src/ao_shaping/optimizer/wfless/differentiable_beam.py"
_ZERNIKE_RESPONSE_MATRIX_UI = "src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py"
_PHASE_CAPTURE = "src/ao_shaping/tools/slm/calibration/phase_capture.py"
_MICRO_DM_UI = "src/ao_shaping/gui/dm/micro_dm_ui.py"
_CALIBRATE = "src/ml/sim2f/calibrate.py"
_TRAIN_DATA_COLLECT = "src/ao_shaping/tools/slm/calibration/train_data_collect.py"
_CCD_ANALYZER = "src/ao_shaping/gui/ccd/ccd_analyzer.py"
_PROBE_ACTIVE = "src/ml/sim2f/probe_active.py"
_SIM_SPGD = "src/ao_shaping/optimizer/wfless/sim_spgd.py"
_CERAMIC_VIEWER = "src/ao_shaping/gui/dm/ceramic_viewer.py"
_BENCH_TWIN_CALIBRATE = "src/ao_shaping/tools/slm/bench_twin_calibrate.py"
_GA_ZERNIKE = "src/ao_shaping/optimizer/wf/ga_zernike.py"
_R50_CERAMIC_VIEWER = "src/ao_shaping/gui/r50/ceramic_viewer.py"
_FOCUS_PROBE = "src/ao_shaping/tools/slm/panel/focus_probe.py"
_TRAIN = "src/ml/sim2f/train.py"
_VOLTAGE_RUNNER = "src/ao_shaping/runners/micro_drive/voltage_runner.py"
_MODEL_IN_LOOP_SHAPING = "src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py"
_ZCO = "src/ao_shaping/algorithm/signal_processing/zernike_coefficient_optimizer.py"
_SHAPING_TRAIN = "src/ml/shaping/train.py"
_ZERNIKE_RESPONSE = "src/ao_shaping/tools/wfs/zernike_response.py"
_ZERNIKE_RESPONSE_MATRIX = "src/ao_shaping/optimizer/wf/zernike_response_matrix.py"
_ML_TRAIN = "src/ml/train.py"
_MICRO_DM_IMAGE_COLLECT = "src/ao_shaping/tools/micro_dm/micro_dm_image_collect.py"
_SLM_CALIBRATION_UI = "src/ao_shaping/gui/slm/slm_calibration_ui.py"
_MODEL_IN_LOOP_RUNNER = "src/ao_shaping/runners/slm/model_in_loop_runner.py"
_EXPOSURE_CONTROL = "src/ao_shaping/utils/image/exposure_control.py"
_SLM_ZERNIKE_PIB = "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py"
_ADAM = "src/ao_shaping/algorithm/gradient/adam.py"
_RECORDING = "src/ao_shaping/tools/slm/transfer/recording.py"
_RUNNER_COMMON = "src/ao_shaping/runners/runner_common.py"
_CALIBRATION = "src/ao_shaping/tools/slm/calibration/calibration.py"
_DATASET = "src/ml/hwdataset/dataset.py"
_AXIS_BEAM_RUNNER = "src/ao_shaping/runners/nlight_dm/axis_beam_runner.py"

#: (relative path under src, function name, span ceiling, nesting ceiling,
#:  the span measured when the guard was written)
TARGETS = [
    (_FILE, "calibrate_bench_geometry_from_sweep", 214, 3, 186),
    (_FILE, "_milsg_validate_sweeps", 30, 3, 26),
    (_FILE, "_milsg_focal_crosschecks", 32, 3, 27),
    (_FILE, "_milsg_fit_waist", 30, 3, 26),
    (_FILE, "_milsg_build_notes", 26, 3, 22),
    (_FILE, "_milsg_log_modes", 19, 3, 16),
    (_FILE, "_tilt_axis_slopes", 70, 3, 44),
    (_FILE, "_ramp_focal_scale", 70, 3, 39),
    (_FILE, "_fov_ratio_cross_check", 80, 3, 46),
    (_FILE, "_tilt_physical_cross_check", 80, 3, 48),
    (_FILE, "_sweep_width_groups", 140, 4, 100),
    (_FILE, "_fit_curvature_offsets", 100, 3, 60),
    (_FILE, "_defocus_asymmetry", 50, 3, 14),
    (_FILE, "_fit_joint_waist", 110, 4, 65),
    (_FILE, "_sweep_calibration_notes", 80, 3, 35),
    (_SCRIPT, "build_markdown", 90, 2, 45),
    (_SCRIPT, "_md_extend", 40, 2, 6),
    (_SCRIPT, "_md_metrics", 90, 2, 42),
    (_SCRIPT, "_md_sections_core", 230, 2, 160),
    (_SCRIPT, "_md_sections_studies", 240, 2, 171),
    (_SCRIPT, "_md_sections_tail", 150, 0, 113),
    (_SCRIPT, "_md_conclusions", 120, 0, 80),
    (_SCRIPT, "_md_falsified", 100, 0, 64),
    (_SECTIONS, "render_retest_section", 60, 1, 14),
    (_SECTIONS, "_rt_extend", 40, 2, 6),
    (_SECTIONS, "_rt_preamble", 40, 1, 12),
    (_SECTIONS, "_rt_intro", 90, 2, 39),
    (_SECTIONS, "_rt_caliber", 160, 3, 95),
    (_SECTIONS, "_rt_atomicity", 120, 3, 58),
    (_SECTIONS, "_rt_blaze", 100, 4, 47),
    (_SECTIONS, "_rt_zernike", 110, 5, 52),
    (_SECTIONS, "_rt_verdict", 230, 1, 139),
    (_SECTIONS, "_rt_jitter", 150, 2, 88),
    (_SECTIONS, "_rt_contract", 120, 1, 69),
    (_PIB, "optimize_pib", 200, 2, 153),
    (_PIB, "_run_pib_bench_session", 192, 6, 175),
    (_PIB, "_pib_bench_epochs", 56, 1, 47),
    (_PIB, "_pib_bench_optimizer", 15, 1, 12),
    (_PIB, "_pib_bench_best_seed", 14, 1, 11),
    (_PIB, "_pib_record_init_row", 29, 2, 25),
    (_PIB, "_pib_make_search_fn", 92, 2, 80),
    (_PIB, "_pib_init_tabu_state", 32, 2, 27),
    (_PIB, "_pib_update_iter_setup", 20, 4, 17),
    (_PIB, "_run_spgd_epochs", 230, 5, 195),
    (_PIB, "_apply_adaptive_search_epoch", 120, 4, 100),
    (_COMB, "optimize_pib", 150, 1, 92),
    (_COMB, "_run_combined_epochs", 80, 1, 54),
    (_COMB, "_run_combined_session", 217, 7, 188),
    (_COMB, "_comb_setup_masks", 50, 3, 43),
    (_COMB, "_comb_record_init_row", 25, 2, 21),
    (_COMB, "_comb_render_factory", 18, 2, 15),
    (_VAL, "run", 250, 2, 212),
    (_VAL, "_validate_prepare_shapes", 110, 5, 69),
    (_VAL, "_run_validate_session", 230, 5, 170),
    (_VAL, "_session_capture", 110, 2, 92),
    (_VAL, "_run_validate_shapes", 130, 3, 105),
    (_VAL, "_run_validate_blocks", 200, 5, 181),
    (_VAL, "_probe_direction_asymmetry", 60, 1, 47),
    (_VAL, "_observe_jacobian_guard", 45, 1, 31),
    (_VAL, "_dump_shape_history_and_timing", 60, 4, 47),
    (_VAL, "_hold_best_phase_or_flatten", 35, 2, 21),
    (_SIM2F, "run_loop", 241, 2, 193),
    (_SIM2F, "_run_probe_stage", 91, 1, 66),
    (_SIM2F, "_run_closed_loop", 108, 2, 83),
    (_SIM2F, "_append_sample_csv_rows", 62, 2, 37),
    (_SIM2F, "_append_sample_record", 85, 0, 60),
    (_SIM2F, "_build_loop_payload", 111, 0, 86),
            (_TRAIN_COEFF, "train", 190, 3, 158),
    (_TRAIN_COEFF, "_train_one_epoch", 80, 3, 63),
    (_TRAIN_COEFF, "_seed_environment", 40, 1, 25),
    (_TRAIN_COEFF, "_build_datasets", 90, 1, 71),
    (_TRAIN_COEFF, "_build_geometric_ops", 50, 4, 37),
    (_TRAIN_COEFF, "_write_summary", 50, 0, 37),
        (_TRAIN_AMP, "train", 240, 4, 205),
    (_TRAIN_AMP, "_prepare_environment", 40, 1, 25),
    (_TRAIN_AMP, "_build_corpus", 60, 1, 39),
    (_TRAIN_AMP, "_build_model_setup", 50, 1, 32),
    (_TRAIN_AMP, "_build_loss_mask", 40, 1, 25),
    (_TRAIN_AMP, "_finalize", 70, 1, 50),
    (_THORLAB_WFS, "load_dll", 391, 0, 351),
    (_ZERNIKE_DEBUG_VIEWER, "_render_main", 389, 7, 349),
    (_CAPTURE, "capture", 281, 7, 244),
    (_CAPTURE, "_cap_live_reader", 37, 2, 32),
    (_CAPTURE, "_cap_load_checkpoint", 27, 4, 23),
    (_CAPTURE, "_cap_spot_xy", 36, 4, 31),
    (_CAPTURE, "_cap_sample_row", 32, 4, 27),
    (_CAPTURE, "_cap_record_dataset", 19, 2, 16),
    (_CAPTURE, "_cap_reuse_guard", 21, 3, 18),
    (_CAPTURE, "_cap_verify_light", 17, 2, 14),
    (_PANEL_LOCATE, "scan_panel", 250, 2, 206),
    (_PANEL_LOCATE, "_scan_panel_session", 190, 4, 150),
    (_TRANSFER_SHAPES_RUNNER, "execute_campaign", 204, 3, 177),
    (_TRANSFER_SHAPES_RUNNER, "_ts_geometry_models", 56, 3, 48),
    (_TRANSFER_SHAPES_RUNNER, "_ts_replay_stage", 25, 4, 21),
    (_TRANSFER_SHAPES_RUNNER, "_ts_make_recorder", 13, 3, 11),
    (_TRANSFER_SHAPES_RUNNER, "_ts_records_stage", 20, 5, 17),
    (_TRANSFER_SHAPES_RUNNER, "_campaign_auto_reuse", 50, 3, 28),
    (_TRANSFER_SHAPES_RUNNER, "_campaign_model_dirs", 80, 2, 58),
    (_TRANSFER_SHAPES_RUNNER, "_campaign_seed_choice", 70, 4, 47),
    (_ZERNIKE_MATRIX_RUNNER, "run", 194, 4, 168),
    (_ZERNIKE_MATRIX_RUNNER, "_zmr_calibration_sweep", 110, 4, 95),
    (_ZERNIKE_MATRIX_RUNNER, "_zmr_finalize_calibration", 36, 4, 31),
    (_ZERNIKE_MATRIX_RUNNER, "_zmr_prepare_dither_reference", 22, 2, 19),
    (_ZERNIKE_MATRIX_RUNNER, "_zmr_echo_device_info", 14, 1, 12),
    (_ZERNIKE_MATRIX_RUNNER, "_zmr_pupil_guidance", 28, 2, 24),
    (_BENCH, "run_session", 165, 2, 143),
    (_BENCH, "_session_metrics_gate", 56, 2, 48),
    (_BENCH, "_session_payload", 70, 2, 60),
    (_BENCH, "_session_write_artifacts", 32, 3, 27),
    (_BENCH, "_run_session_samples", 90, 4, 69),
    (_DM_RESPONSE_MATRIX, "calibrate_dm_response_matrix", 240, 3, 196),
    (_DM_RESPONSE_MATRIX, "_run_calibration_sweep", 170, 4, 130),
    (_WF, "run_session", 220, 3, 178),
    (_WF, "_run_session_rounds", 280, 4, 241),
    (_WF, "_acquire_session_dataset", 95, 4, 78),
    (_WF, "_round_geometry_preflight", 60, 1, 42),
    (_WF, "_offline_synthesize", 130, 3, 108),
    (_WF, "_round_calibration_stages", 70, 1, 52),
    (_LUT, "run", 380, 3, 329),
    (_LUT, "_run_gray_scan", 115, 2, 91),
    (_IO, "render_recorder_gif", 160, 4, 113),
    (_IO, "_gif_frames", 200, 5, 149),
    (_IO, "_render_gif_top_panels", 100, 6, 72),
    (_DM, "wait_until_idle", 150, 1, 89),
    (_DM, "_wait_idle_loop", 150, 4, 116),
    (_DM, "_resolve_wait_state", 70, 3, 51),
    (_DM, "_resolve_probe_state", 150, 4, 136),
    (_PAT, "create_target_shape", 150, 1, 91),
    (_PAT, "_dispatch_shape", 60, 1, 28),
    (_PAT, "_pat_gaussian", 60, 0, 24),
    (_PAT, "_pat_disk", 60, 0, 23),
    (_PAT, "_pat_square", 60, 0, 26),
    (_PAT, "_pat_annular", 60, 0, 27),
    (_ZCO, "__init__", 169, 2, 147),
    (_ZCO, "_zco_init_validate", 80, 4, 69),
    (_ZCO, "_zco_init_coeffs", 40, 4, 34),
    (_ZCO, "_zco_init_affine_travel", 32, 3, 27),
    (_PAT, "_pat_grid", 60, 1, 32),
    (_PAT, "_pat_cross", 60, 0, 24),
    (_PAT, "_pat_rectangle", 60, 0, 28),
    (_PAT, "_pat_pentagon", 60, 1, 32),
    (_PAT, "_pat_poly", 60, 1, 33),
    (_MEAS, "run_measurement", 200, 1, 173),
    (_MEAS, "_run_measurement_session", 286, 4, 248),
    (_MEAS, "_measure_focus_gate", 102, 3, 88),
    (_MEAS, "_measure_scan_bounds", 27, 3, 23),
    (_MEAS, "_measure_saturation_warn", 15, 4, 13),
    (_MEAS, "_measure_drift_warn", 14, 4, 12),
    (_MEAS, "_measure_defocus_phase", 18, 2, 15),
    (_SZP, "optimize_slm_zernike_pib", 280, 2, 250),
    (_SZP, "_run_pib_session", 267, 5, 232),
    (_SZP, "_pib_arm_blaze", 24, 4, 21),
    (_SZP, "_pib_geometry", 44, 2, 38),
    (_SZP, "_pib_build_shaping", 52, 2, 45),
    (_SZP, "_pib_record_row0", 40, 3, 35),
    (_SZP, "_pib_open_session", 38, 2, 33),
    (_SZP, "_run_spgd", 160, 1, 119),
    (_SZP, "_run_spgd_session", 268, 5, 233),
    (_SZP, "_spgd_record_improved", 61, 3, 53),
    (_SZP, "_spgd_apply_update", 42, 1, 36),
    (_SZP, "_spgd_batch_objective", 53, 2, 46),
    (_SZP, "_spgd_capture_abba", 44, 1, 38),
    (_SQ, "optimize_slm_square", 500, 4, 460),
    (_SQ, "_run_square_session", 259, 3, 225),
    (_SQ, "_session_init_iteration", 230, 3, 200),
    (_SQ, "_session_wire_energy_guard", 73, 2, 63),
    (_SQ, "_session_arm_exposure", 52, 3, 45),
    (_SQ, "_run_square_search_stage", 287, 6, 249),
    (_SQ, "_sq_display_epoch", 20, 4, 17),
    (_SQ, "_sq_record_epoch", 29, 2, 25),
    (_SQ, "_sq_adapt_lr", 42, 5, 36),
    (_SQ, "_sq_epoch_metrics", 33, 2, 28),
    (_SQ, "_sq_saturation_resample", 19, 3, 16),
    (_SQ, "_sq_measure_pair", 34, 2, 29),
    (_SQ, "_sq_adaptive_weights", 18, 2, 15),
    (_SQ, "_run_square_heuristic_stage", 210, 3, 168),
    (_SQ, "_square_stage_weights", 95, 3, 52),
    (_SQ, "_square_perturbation_measurement", 95, 0, 52),
    (_SQ, "_select_better_square_perturbation", 135, 1, 92),
    (_RMS, "optimizer_rms_slm", 200, 1, 155),
    (_RMS, "_run_rms_session", 158, 6, 138),
    (_RMS, "_rmsr_schedules", 19, 1, 14),
    (_RMS, "_rmsr_minibatch_gradient", 68, 3, 57),
    (_RMS, "_rmsr_stagnation_boost", 26, 3, 21),
    (_RMS, "_rmsr_epoch_log", 35, 2, 29),
    (_RMS, "_rms_heuristic_stage", 72, 3, 62),
    (_RMS, "_rms_initial_state", 59, 2, 51),
    (_RMS, "_rms_init_coefficients", 61, 4, 53),
    (_MIL, "optimize_slm_model_in_loop", 130, 1, 84),
    (_MIL, "_run_model_in_loop_session", 85, 3, 74),
    (_MIL, "_session_open_bench", 172, 3, 151),
    (_MIL, "_session_measure_flat", 142, 2, 124),
    (_MIL, "_session_round_state_init", 84, 1, 73),
    (_MIL, "_run_model_in_loop_rounds", 208, 4, 181),
    (_MIL, "_milr_round_probe_setup", 28, 2, 23),
    (_MIL, "_milr_round_score_after", 12, 1, 9),
    (_MIL, "_milr_step_b_measure", 111, 2, 96),
    (_MIL, "_milr_step_a_clamp", 73, 1, 63),
    (_MIL, "_milr_accept_verdict", 67, 1, 58),
    (_MIL, "_milr_step_a_setup", 49, 0, 42),
    (_MIL, "_milr_round_display", 30, 1, 26),
    (_MIL, "_model_in_loop_round_row", 70, 0, 39),
    (_MIL, "_model_in_loop_round_holdout", 85, 2, 55),
    (_MIL, "_model_in_loop_round_probes", 110, 2, 81),
    (_WORKFLOW, "main", 191, 5, 165),
    (_WORKFLOW, "_add_session_commands", 157, 4, 136),
    (_DIFFERENTIABLE_BEAM, "optimize_beam_shaping", 200, 1, 153),
    (_DIFFERENTIABLE_BEAM, "_run_simulation_phase", 80, 3, 68),
    (_DIFFERENTIABLE_BEAM, "_run_hardware_phase", 100, 4, 88),
    (_ZERNIKE_RESPONSE_MATRIX_UI, "render_sidebar", 319, 4, 289),
    (_PHASE_CAPTURE, "run", 316, 3, 286),
    (_MICRO_DM_UI, "_tab2_fragment", 309, 7, 279),
    (_CALIBRATE, "calibrate_bench", 164, 5, 142),
    (_CALIBRATE, "_calib_param_vector", 60, 4, 52),
    (_CALIBRATE, "_calib_result_blocks", 53, 3, 46),
    (_CALIBRATE, "_calib_start_scores", 26, 3, 22),
    (_CALIBRATE, "_calib_verdict", 20, 3, 17),
    (_CALIBRATE, "_calib_bounds_stages", 21, 2, 18),
    (_TRAIN_DATA_COLLECT, "main", 307, 4, 277),
    (_CCD_ANALYZER, "main", 295, 4, 265),
    (_PROBE_ACTIVE, "probe_active", 242, 4, 210),
    (_PROBE_ACTIVE, "_probe_validate_inputs", 38, 3, 33),
    (_PROBE_ACTIVE, "_probe_init_state", 30, 2, 26),
    (_SIM_SPGD, "optimize_spgd", 215, 2, 171),
    (_SIM_SPGD, "_spgd_run_loop", 120, 5, 93),
    (_CERAMIC_VIEWER, "render_tab_grid", 287, 5, 257),
    (_BENCH, "gate0", 220, 2, 177),
    (_BENCH, "_gate0_select_exposure", 90, 1, 58),
    (_BENCH, "_gate0_check_geometry", 70, 3, 38),
    (_ZERNIKE_MATRIX_RUNNER, "closed_loop_run", 178, 4, 154),
    (_ZERNIKE_MATRIX_RUNNER, "_clr_build_controller", 30, 1, 26),
    (_ZERNIKE_MATRIX_RUNNER, "_clr_set_initial_reference", 10, 1, 8),
    (_ZERNIKE_MATRIX_RUNNER, "_clr_save_results", 59, 2, 51),
    (_ZERNIKE_MATRIX_RUNNER, "_clr_plot_convergence", 37, 3, 32),
    (_SIM_SPGD, "optimize_spgd_zernike", 215, 1, 171),
    (_SIM_SPGD, "_spgd_zernike_run_loop", 120, 5, 90),
    (_BENCH_TWIN_CALIBRATE, "fit_twin", 126, 3, 109),
    (_BENCH_TWIN_CALIBRATE, "_fit_stage_runs", 170, 4, 147),
    (_GA_ZERNIKE, "optimizer_ga", 206, 6, 188),
    (_GA_ZERNIKE, "_ga_init_population", 65, 5, 54),
    (_GA_ZERNIKE, "_ga_record_initial_state", 22, 1, 18),
    (_R50_CERAMIC_VIEWER, "render_tab_grid", 280, 6, 250),
    (_FOCUS_PROBE, "sweep_focus", 176, 5, 153),
    (_FOCUS_PROBE, "_fp_pick_best", 57, 5, 49),
    (_FOCUS_PROBE, "_fp_exposure_alt", 48, 5, 41),
    (_FOCUS_PROBE, "_fp_flat_gain", 9, 2, 7),
    (_FOCUS_PROBE, "_fp_blocked_verdict", 17, 3, 14),
    (_ZERNIKE_DEBUG_VIEWER, "_compute_response_matrix_from_debug", 276, 8, 246),
    (_TRAIN, "train", 274, 3, 244),
    (_VOLTAGE_RUNNER, "run", 271, 6, 241),
    (_MODEL_IN_LOOP_SHAPING, "calibrate_shared_aberration", 271, 2, 241),
    (_SHAPING_TRAIN, "train", 266, 2, 236),
    (_ZERNIKE_RESPONSE, "main", 264, 5, 234),
    (_ZERNIKE_RESPONSE_MATRIX, "calibrate_zernike_response_matrix", 178, 4, 162),
    (_ZERNIKE_RESPONSE_MATRIX, "_zrm_auto_amplitude", 28, 4, 23),
    (_ZERNIKE_RESPONSE_MATRIX, "_zrm_buffers", 28, 2, 23),
    (_ZERNIKE_RESPONSE_MATRIX, "_zrm_initial_spots", 19, 1, 15),
    (_ZERNIKE_RESPONSE_MATRIX, "_zrm_build_result", 33, 1, 27),
    (_ML_TRAIN, "train_model", 262, 5, 232),
    (_MICRO_DM_IMAGE_COLLECT, "run", 260, 5, 230),
    (_MODEL_IN_LOOP_SHAPING, "calibrate_bench_geometry", 190, 5, 173),
    (_MODEL_IN_LOOP_SHAPING, "_cbg_measure_references", 30, 2, 27),
    (_MODEL_IN_LOOP_SHAPING, "_cbg_scan_seeds", 15, 1, 12),
    (_MODEL_IN_LOOP_SHAPING, "_cbg_finalize", 33, 2, 30),
    (_SLM_CALIBRATION_UI, "render_sidebar", 254, 4, 224),
    (_MODEL_IN_LOOP_RUNNER, "run", 252, 2, 222),
    (_EXPOSURE_CONTROL, "estimate_exposure_ceiling_ms", 197, 5, 176),
    (_EXPOSURE_CONTROL, "_ecc_collect_pairs", 23, 3, 19),
    (_EXPOSURE_CONTROL, "_ecc_init_accumulators", 15, 1, 12),
    (_EXPOSURE_CONTROL, "_ecc_reset_fit_triplet", 14, 2, 11),
    (_EXPOSURE_CONTROL, "_ecc_final_bound", 23, 3, 19),
    (_SLM_ZERNIKE_PIB, "_prepare_geometry", 163, 3, 141),
    (_SLM_ZERNIKE_PIB, "_pibg_window_estimate", 64, 2, 55),
    (_SLM_ZERNIKE_PIB, "_pibg_bucket_init", 37, 3, 32),
    (_MICRO_DM_UI, "_tab1_fragment", 250, 7, 220),
    (_ADAM, "search_optimal_delta", 154, 4, 128),
    (_ADAM, "_sod_init_scan", 19, 2, 15),
    (_ADAM, "_sod_fine_bounds", 26, 2, 21),
    (_ADAM, "_sod_fine_sweep", 68, 4, 56),
    (_ADAM, "_sod_best_fine", 14, 2, 11),
    (_RECORDING, "render_process_gif", 146, 5, 121),
    (_RECORDING, "_rpg_prepare_frames", 43, 1, 36),
    (_RECORDING, "_rpg_prepare_series", 37, 2, 31),
    (_RECORDING, "_rpg_draw_readout", 16, 1, 13),
    (_RECORDING, "_rpg_draw_trajectories", 24, 2, 20),
    (_RECORDING, "_rpg_draw_heading", 14, 2, 12),
    (_RUNNER_COMMON, "save_model_in_loop_results", 152, 3, 136),
    (_RUNNER_COMMON, "_smil_prepare_outputs", 16, 1, 13),
    (_RUNNER_COMMON, "_smil_save_error_terms", 48, 3, 41),
    (_RUNNER_COMMON, "_smil_save_artifacts", 40, 2, 34),
    (_CALIBRATION, "main_shift_calib", 238, 4, 208),
    (_DATASET, "create_hw_dataloaders", 236, 2, 206),
    (_AXIS_BEAM_RUNNER, "run", 233, 2, 203),
    (_MODEL_IN_LOOP_SHAPING, "shape_phase_with_frozen_aberration", 233, 3, 203),
]

NESTING_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Deepest if/for/while/try/with chain inside ``node``."""
    best = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, NESTING_NODES) else depth
        best = max(best, _max_nesting(child, child_depth))
    return best


def _measure(rel_path: str, func_name: str) -> tuple[int, int]:
    module = ast.parse((REPO / rel_path).read_text(encoding="utf-8"))
    hits = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
    ]
    assert len(hits) == 1, (
        f"expected exactly one {func_name} in {rel_path}, found {len(hits)} -- "
        "the guard would silently measure the wrong one"
    )
    node = hits[0]
    return node.end_lineno - node.lineno + 1, _max_nesting(node)


@pytest.mark.parametrize(("rel_path", "func_name", "span_max", "nest_max", "measured"), TARGETS)
def test_the_split_function_stays_within_its_shape_budget(
    rel_path: str, func_name: str, span_max: int, nest_max: int, measured: int
) -> None:
    span, nesting = _measure(rel_path, func_name)

    assert nesting <= nest_max, (
        f"{rel_path}::{func_name} nests {nesting} deep, over the #205 ceiling of "
        f"{nest_max}. Split the new branch out instead of nesting further."
    )
    assert span <= span_max, (
        f"{rel_path}::{func_name} is {span} lines, over its #205 ceiling of "
        f"{span_max} (measured {measured} when the guard was written). It is "
        "growing back -- split it again rather than raising the ceiling."
    )


def test_the_depth_measurement_is_not_silently_zero() -> None:
    """The walker must still see depth where depth exists.

    Pinned against one of the two blocks that genuinely sit at nesting 4
    (verbatim-moved filters, flattening left to a later batch): if the AST walk
    ever degraded to 0, every nesting ceiling above would pass vacuously.
    """
    _span, nesting = _measure(_FILE, "_sweep_width_groups")
    assert nesting == 4, (
        f"expected _sweep_width_groups to measure exactly 4 deep, got {nesting}; "
        "the nesting walker changed and the ceilings above mean nothing"
    )
