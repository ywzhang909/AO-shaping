"""slm-model-in-loop config / result persistence helpers.

Extracted verbatim from :mod:`ao_shaping.runners.runner_common` (#205 slicing).
The public names stay importable from ``runner_common`` via re-export; the single
implementation lives here, so ``runner_common`` no longer owns the on-disk layout
of a model-in-loop run.

``runner_common`` is imported **only** under ``TYPE_CHECKING``: the functions
below duck-type the params/result objects, and a runtime import would create a
cycle with the re-export in ``runner_common``.

Behaviour notes that the tests pin and that must not be "simplified":

* ``best_phase.npy`` is the phase finally ADOPTED (rolls back to flat when the
  comparison rejects the correction); ``shaped_phase.npy`` is the phase really
  uploaded during Step B, and is written only when a shaped probe carries one.
* ``run_config.json`` stores the error terms' ``to_dict()`` form, because the
  ``--error-params`` reader must be able to load the file back.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ao_shaping.optimizer.wfless.slm_model_in_loop import (
        ModelInLoopResult,
        SlmModelInLoopConfig,
    )
    from ao_shaping.runners.runner_common import SlmModelInLoopParams

__all__ = [
    "ModelInLoopSaveResult",
    "save_model_in_loop_config",
    "save_model_in_loop_results",
]


@dataclass
class ModelInLoopSaveResult:
    """``save_model_in_loop_results`` 的落盘产物路径。

    ``phase_file`` 是最终采纳下发相位（``best_phase.npy``）—— 对比择优被拒时它是
    全零的平场相位。``shaped_phase_file`` 是 Step B 实际下发的那条整形相位
    （``split == "shaped"`` 的最后一条探针，``shaped_phase.npy``）；只有当 probe
    records 里真有形状化探针时才写盘，否则为 ``None``。
    """

    save_dir: Path
    csv_file: Path
    phase_file: Path | None
    shaped_phase_file: Path | None
    geometry_file: Path | None
    image_file: Path | None
    recorder_png: Path


def save_model_in_loop_config(config: SlmModelInLoopConfig, save_dir: Path) -> Path:
    """Persist the run configuration before hardware preflight can fail."""
    import json

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "run_config.json"
    run_config = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name != "callback"
    }
    # The error terms are a dataclass; ``default=str`` would file them away as
    # their repr, which cannot be read back. Their dictionary form is what
    # ``--error-params`` accepts, so the run config stays loadable.
    terms = run_config.get("error_terms")
    if terms is not None and hasattr(terms, "to_dict"):
        run_config["error_terms"] = terms.to_dict()
    path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    return path


def _model_in_loop_json_payload(result: ModelInLoopResult) -> dict[str, Any]:
    """Build the JSON sidecar of a model-in-loop run.

    The status and the config are always there. The **self-calibration
    trajectory** is added only when that feature actually ran, so a run with
    the self-calibration off keeps exactly the payload it had.

    Args:
        result: The run's result.

    Returns:
        A JSON-serialisable dictionary.
    """
    payload: dict[str, Any] = {
        "status": result.status,
        "config": str(result.config),
    }
    trajectory = [
        {
            "round": row.get("round"),
            "affine": row.get("selfcal_affine"),
            "travel": row.get("selfcal_travel"),
            "theta_step": row.get("selfcal_theta_step"),
            "gamma_step": row.get("selfcal_gamma_step"),
            "violated": row.get("selfcal_violated"),
        }
        for row in result.recorder.history
        if row.get("selfcal_violated") is not None
    ]
    if trajectory:
        payload["selfcal"] = trajectory
    return payload


def _smil_prepare_outputs(debug_run_dir, gen_date_str, result, root_dir, save_dir):
    # Save dir + csv + config persistence lifted verbatim
    save_dir = (
        save_dir
        if save_dir is not None
        else debug_run_dir(root_dir, "slm_model_in_loop")
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_file = save_dir / f"slm_model_in_loop_{gen_date_str()}.csv"
    result.recorder.save_dataframe(csv_file)

    save_model_in_loop_config(result.config, save_dir)
    return csv_file, save_dir


def _smil_save_error_terms(debug, json, result, save_dir):
    # Error-term and probe-debug persistence lifted verbatim
    error_terms = getattr(result.config, "error_terms", None)
    if error_terms is not None:
        from ao_shaping.utils.wavefront.model_error_terms import save_error_terms

        save_error_terms(error_terms, save_dir / "model_error_terms.json")

    if debug and result.probe_records:
        probes_dir = save_dir / "probes"
        probes_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for record in result.probe_records:
            name = (
                f"round_{record['round']:02d}_{record['split']}_"
                f"{record['index']:02d}.npz"
            )
            arrays = {
                key: np.asarray(record[key], dtype=np.float32)
                for key in ("phase", "raw_frame", "frame", "model_grid", "prediction")
                if record.get(key) is not None
            }
            np.savez_compressed(probes_dir / name, **arrays)
            manifest.append(
                {
                    "file": name,
                    "round": record["round"],
                    "split": record["split"],
                    "family": record.get("family"),
                    "index": record["index"],
                    "exposure_ms": record.get("exposure_ms"),
                    "timestamp_utc": record.get("timestamp_utc"),
                    "correlation": record.get("correlation"),
                    "frame_reference_exposure_ms": 0.02,
                    "raw_frame_note": "separate CCD read after the settled frame",
                }
            )
        (probes_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return


def _smil_save_artifacts(json, params, result, save_best_image, save_dir):
    # Geometry image and recorder debug artifacts lifted verbatim
    geometry_file = None
    if result.geometry is not None:
        geometry_file = save_dir / "bench_geometry.json"
        geometry_file.write_text(
            json.dumps(
                {
                    "panel_disc_radius": result.geometry.panel_disc_radius,
                    "region": result.geometry.region,
                    "beam_waist_panel_px": result.geometry.beam_waist_panel_px,
                    "far_field_size": result.geometry.far_field_size,
                    "camera_px_per_model_px": result.geometry.camera_px_per_model_px,
                    "spot_fwhm_camera_px": result.geometry.spot_fwhm_camera_px,
                    "spot_fwhm_model_px": result.geometry.spot_fwhm_model_px,
                    "correlation": result.geometry.correlation,
                    "method": result.geometry.method,
                    "calibration_notes": result.geometry.calibration_notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    image_file = None
    if params.save_best_image and result.best_frame is not None:
        image_file = save_dir / "best_far_field.png"
        save_best_image(
            result.best_frame,
            image_file,
            title=f"slm-model-in-loop best: score={result.best_score:.4f} (flat {result.flat_score:.4f})",
            hide_axes=True,
        )
    return geometry_file, image_file


def save_model_in_loop_results(
    result: ModelInLoopResult,
    params: SlmModelInLoopParams,
    *,
    root_dir: Path,
    debug: bool,
    save_dir: Path | None = None,
) -> ModelInLoopSaveResult:
    """Persist a model-in-loop run's recorder and debug artefacts.

    Extracted verbatim from ``runners/slm/model_in_loop_runner.py`` so the GUI
    adapter reuses the same disk layout without duplicating the CLI runner.
    """
    import json

    from ao_shaping.display.frames import save_best_image
    from ao_shaping.utils.io.file import (
        debug_run_dir,
        gen_date_str,
        save_recorder_debug_artifacts,
    )

    csv_file, save_dir = _smil_prepare_outputs(debug_run_dir, gen_date_str, result, root_dir, save_dir)

    if result.fitted_coefficients is not None:
        np.save(save_dir / "fitted_coefficients.npy", result.fitted_coefficients)
    if result.source_amplitude is not None:
        np.save(save_dir / "source_amplitude.npy", result.source_amplitude)

    # The error terms the run actually used, so the next run can reuse them
    # (``--error-params <this file>``). Written even when they are the ideal
    # system's zeros, because "the model was ideal" is itself a fact worth
    # having on disk next to the run it belongs to.
    _smil_save_error_terms(debug, json, result, save_dir)

    phase_file = None
    best_phase = result.best_phase
    if best_phase is not None:
        phase_file = save_dir / "best_phase.npy"
        np.save(phase_file, np.asarray(best_phase, dtype=np.float64))

    # The phase actually written to the SLM during Step B (split == "shaped"),
    # the LAST such dropped probe. best_phase.npy is the phase finally ADOPTED
    # (rolls back to a flat zero field when bake-off fails), which is NOT the
    # shaping phase that was really uploaded. The two differ exactly when
    # bake_off_rejected.  Only written when a shaped probe with a phase exists —
    # never fabricate an empty file.
    shaped_phase_file = None
    shaped_phase = None
    for record in result.probe_records:
        if record.get("split") == "shaped" and record.get("phase") is not None:
            shaped_phase = record["phase"]
    if shaped_phase is not None:
        shaped_phase_file = save_dir / "shaped_phase.npy"
        np.save(shaped_phase_file, np.asarray(shaped_phase, dtype=np.float64))
    blaze_params = getattr(result, "blaze_params", None)
    if best_phase is not None and blaze_params is not None:
        from ao_shaping.tools.kernels.bench_kernels import phase_to_panel
        from ao_shaping.utils.wavefront.zero_order_overlay import apply_zero_order_overlay

        panel_phase = phase_to_panel(
            np.asarray(best_phase, dtype=np.float64),
            int(round(result.config.panel_span_px / 2)), result.config.pupil_center,
        )
        np.save(
            save_dir / "best_panel_phase_with_blaze.npy",
            apply_zero_order_overlay(panel_phase, blaze_params),
        )
        from dataclasses import asdict

        choice = getattr(result, "blaze_choice", None)
        (save_dir / "blaze_choice.json").write_text(
            json.dumps(
                asdict(choice) if choice is not None else {
                    "period_px": blaze_params.blaze_period_px,
                    "direction": blaze_params.blaze_direction,
                },
                indent=2,
            ), encoding="utf-8",
        )

    geometry_file, image_file = _smil_save_artifacts(json, params, result, save_best_image, save_dir)

    recorder_png = save_recorder_debug_artifacts(
        result.recorder,
        str(root_dir),
        "slm_model_in_loop",
        scalar_keys=(
            "round",
            "accepted",
            "loss_before",
            "loss_after",
            "step_a_iterations",
            "step_b_iterations",
            "coeff_step_l2",
            "coeff_step_applied",
            "clamp_applied",
            "coeff_norm",
            "score_before",
            "score_after",
            "cv_after",
            "ee_after",
            "uniformity_cv",
            "probe_count",
            "step_a_lr",
            "rejection_streak",
            "geometry_correlation",
            "n_calibration_probes",
            "holdout_correlation",
            "exposure_ms",
            "selfcal_theta_step",
            "selfcal_gamma_step",
            "selfcal_violated",
        ),
        img_keys=("_img",),
        text_keys=("timestamp_utc",),
        object_keys=(
            "step_a_trace",
            "step_b_trace",
            "selfcal_affine",
            "selfcal_travel",
        ),
        output_dir=save_dir,
        json_payload=_model_in_loop_json_payload(result),
        title="slm-model-in-loop",
    )

    return ModelInLoopSaveResult(
        save_dir=save_dir,
        csv_file=csv_file,
        phase_file=phase_file,
        shaped_phase_file=shaped_phase_file,
        geometry_file=geometry_file,
        image_file=image_file,
        recorder_png=recorder_png,
    )
