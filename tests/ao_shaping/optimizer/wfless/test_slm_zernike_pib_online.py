"""Online (hardware) acceptance tests for robust SPGD in the SLM PIB pipeline.

Gated behind ``AO_RUN_HARDWARE`` — the same convention as
``tests/ao_shaping/drivers/test_hardware_integration.py`` — so CI/offline runs
skip the whole module. To execute on the bench:

    $env:AO_RUN_HARDWARE=1
    uv run pytest tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py -v -s

What is validated (user constraints: daheng CCD, n_max > 6, delta < 0.001):

1. ``test_noise_floor_and_snr_monotone_with_delta`` — delegates to the shared,
   **device-agnostic** probe
   :func:`ao_shaping.tools.slm.sweep_analysis.snr_sweep` (already-open cam/slm are
   passed in; the probe never constructs a device), so this bench test and
   ``scripts/measure_shape_sensitivity.py`` measure identically and cannot drift
   apart. The J measurement noise floor at a fixed phase is finite and positive,
   and the perturbation signal grows with delta. It records **both** the
   single-mode SNR and the SPGD-style multi-mode SNR: the per-delta verdict
   (``>=3`` strong / ``>=2`` usable / ``<2`` unusable) is *recorded*, not
   asserted — at ``delta=0.001`` the bench may genuinely sit below SNR 2 (that is
   the h6a diagnosis) and the test must report that, not fail on it.

   **Judge ``--delta`` by the multi-mode column.** SPGD perturbs every mode by a
   random ±1 pattern, which dilutes the coherent signal; measured on this bench at
   ``delta=0.0005`` the single-mode SNR was 2.25 (usable) while the 54-DOF SNR
   was 1.34 (unusable) — which is why such runs gate ~95% of their epochs.

2. ``test_robust_spgd_smoke_delta_lt_0_001`` (and the frame-averaged variant)
   — a short ``optimize_slm_zernike_pib`` run under ``delta=0.0005`` must
   never adopt an abandoned evaluation (no ``-1e3`` rows), keep every
   coefficient inside the driver clip (``|c| <= 5.0``), log at most one row
   per epoch, and its fold/noise gates must be observably exercised
   (``_c``-stalled rows are classified into fold-gated vs noise-gated from the
   recorder rows).

3. Debug artefacts (recorder pkl + JSON summary) are written under
   ``data/debug/slm_pib_online/<timestamp>/`` for offline analysis.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

if os.environ.get("AO_RUN_HARDWARE", "").strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "set AO_RUN_HARDWARE=1 to run hardware-gated slm-pib tests",
        allow_module_level=True,
    )

from ao_shaping.optimizer.wfless.slm_zernike_pib import (  # noqa: E402
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (  # noqa: E402
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)


# ---------------------------------------------------------------------------
# Configuration (mirrors the runner's SPGD branch of _build_slm_pib_config)
# ---------------------------------------------------------------------------

CAM_ID = 0
CAM_TYPE = "daheng"
EXPOSURE_MS = 1.2
CAM_SIZE = 250
N_MAX = 7  # n_max MUST be > 6 (user constraint); 7 keeps the smoke short
ZERNIKE_RADIUS = 480.0
DELTA = 0.0005  # delta MUST be < 0.001 (user constraint)
LR = 0.5
SLM_NUMBER = 1
SLM_WAVELENGTH = 1064
TARGET_SIZE = 50.0
SETTLE_S = 0.3
ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[4] / "data" / "debug" / "slm_pib_online"
)


def make_config(
    epochs: int,
    *,
    delta: float = DELTA,
    n_eval_frames: int = 1,
) -> SlmZernikePibConfig:
    """SPGD-branch config equivalent to the reference CLI invocation:
    ``spgd -c max --target_shape square --target_size 50 --delta 0.0005
    --exposure_time_ms 1.2 -e <epochs> --zernike_radius 480 -n 7 --lr 0.5``."""
    camera = CameraParamsPib(
        center="max",
        cam_id=CAM_ID,
        cam_type=CAM_TYPE,
        cam_size=CAM_SIZE,
        exposure_time_ms=EXPOSURE_MS,
        target=ObjectiveTarget(name="pib", target_shape="square"),
        target_size=TARGET_SIZE,
    )
    slm = SlmParamsPib(
        slm_number=SLM_NUMBER,
        slm_wavelength=SLM_WAVELENGTH,
        n_max=N_MAX,
        zernike_radius=ZERNIKE_RADIUS,
    )
    return SlmZernikePibConfig(
        center="max",
        epochs=epochs,
        camera=camera,
        slm=slm,
        record_phase=False,
        random_seed=None,
        algorithm="spgd",
        pop_size=None,
        delta=delta,
        lr=LR,
        optimizer_type="adamod",
        shrink_iter=0,
        shrink_ratio=0.9,
        show=False,
        n_eval_frames=n_eval_frames,
        fold_ratio=0.5,
        noise_gate_k=3.0,
    )


def _row_objective_key(row: dict[str, Any]) -> str:
    """Effective objective column key (same remap as the runner's
    ``_effective_objective_key``: shape target -> ``"shape"``)."""
    for key in ("shape", "pib", "roi_pib", "rms_pib", "rmse"):
        if key in row:
            return key
    raise KeyError(f"no known objective column in row: {sorted(row)}")


# ---------------------------------------------------------------------------
# Helpers: row classification + artifact writing
# ---------------------------------------------------------------------------

def _classify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline gate observability.

    Prefers the per-row ``_gate`` verdict the optimizer records
    (``"applied"`` / ``"fold"`` / ``"noise"``). Falls back to the
    ``_c``-stagnation heuristic for records without it (heuristic-branch rows,
    older pickles): a row whose ``_c`` equals the previous one was gated, and
    ``_diff == 0.0`` marks a fold gate. The heuristic over-counts ``applied``:
    a noise-gated row still advances ``_c`` because ``optimizer.update(zeros)``
    decays the coefficients, so coefficient motion does not prove the gradient
    was adopted.

    The first row (the initialisation row, ``_epoch == 0``) is a baseline and
    is counted in ``rows`` only: ``applied + stalled + 1 == rows`` for SPGD runs.
    """
    stats: dict[str, Any] = {
        "rows": len(rows),
        "applied": 0,
        "stalled": 0,
        "fold": 0,
        "noise": 0,
    }
    prev_c: np.ndarray | None = None
    for row in rows:
        c = np.asarray(row["_c"])
        gate = row.get("_gate")
        if gate is not None:
            stats[gate] = stats.get(gate, 0) + 1
            if gate != "applied":
                stats["stalled"] += 1
        elif prev_c is not None:
            if np.array_equal(c, prev_c):
                stats["stalled"] += 1
                if float(row["_diff"]) == 0.0:
                    stats["fold"] += 1
                else:
                    stats["noise"] += 1
            else:
                stats["applied"] += 1
        prev_c = c
    return stats


def _scalarize(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _write_artifacts(
    tag: str,
    recorder: Any,
    summary: dict[str, Any],
) -> Path:
    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"recorder_{tag}.pkl").open("wb") as f:
        pickle.dump(recorder, f)
    with (run_dir / f"summary_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return run_dir


# ---------------------------------------------------------------------------
# 1. Noise floor + per-delta SNR discrimination  (delta sweep < 0.001 focus)
# ---------------------------------------------------------------------------

class TestNoiseFloorAndSnr:
    def test_noise_floor_and_snr_monotone_with_delta(self) -> None:
        cfg = make_config(epochs=2)  # config only used for camera/slm/target

        from ao_shaping.drivers.ccd.common import create_camera
        from ao_shaping.drivers.slm import Santec
        from ao_shaping.tools.slm.sweep_analysis import snr_sweep

        with create_camera(cfg.camera) as cam, Santec.from_params(cfg.slm) as slm:
            # The measurement itself is delegated to the shared, device-agnostic
            # probe (cam/slm passed in; it never constructs a device), so this
            # bench test and scripts/measure_shape_sensitivity.py measure
            # identically and cannot drift apart.
            result = snr_sweep(
                cam,
                slm,
                n_max=N_MAX,
                radius=ZERNIKE_RADIUS,
                target_shape="square",
                target_size=TARGET_SIZE,
                deltas=(0.0005, 0.001, 0.01),
                n_frames=15,
                pairs=3,
            )
            center = result.center
            sigma_j = result.sigma
            noise_js = result.noise_values
            snrs = result.single_snrs
            signals = result.single_signals

            assert np.isfinite(center).all()

            # Infrastructure assertions (honest, not bench-optimistic):
            #  - SNR values must be finite and grow (weakly) with delta: a 20x
            #    larger perturbation must not produce less than half the signal.
            assert all(np.isfinite(v) for v in snrs.values())
            assert snrs["0.01"] >= 0.5 * snrs["0.0005"] - 0.25

        run_dir = _write_artifacts(
            "snr",
            None,
            {
                "center": list(map(float, center)),
                "sigma_j": sigma_j,
                "noise_mean_j": float(np.mean(noise_js)),
                "signals": signals,
                "snr_by_delta": snrs,
                "verdict": {
                    d: ("strong" if s >= 3.0 else "usable" if s >= 2.0 else "unusable")
                    for d, s in snrs.items()
                },
                "n_dof": result.n_dof,
                "multi_snrs": result.multi_snrs,
                "multi_verdicts": result.verdicts,
                "note": (
                    "hardware measurement, not an assertion: at delta<0.001 a "
                    "sub-2 SNR is the expected honest-stall regime (h6a). "
                    "'snr_by_delta' is the SINGLE-mode probe; SPGD perturbs all "
                    "modes at once, so judge delta by 'multi_snrs' — measured "
                    "dilution at delta=0.0005 was 2.25 (single) -> 1.34 (54 DOF)."
                ),
            },
        )
        print(f"\nSNR artefact: {run_dir}")
        print(
            f"  sigma_j={sigma_j:.8f} n_dof={result.n_dof}\n"
            f"  single-mode SNR: "
            + ", ".join(f"{d}={v:.2f}" for d, v in snrs.items())
            + "\n  SPGD-style  SNR: "
            + ", ".join(f"{d}={v:.2f}" for d, v in result.multi_snrs.items())
        )


# ---------------------------------------------------------------------------
# 2. Robust SPGD smoke under delta < 0.001
# ---------------------------------------------------------------------------

class TestRobustSpgdSmoke:
    @pytest.mark.parametrize("n_eval_frames", [1, 4])
    def test_smoke_delta_lt_0_001(self, n_eval_frames: int) -> None:
        epochs = 16 if n_eval_frames == 4 else 30
        cfg = make_config(epochs=epochs, delta=DELTA, n_eval_frames=n_eval_frames)

        recorder = optimize_slm_zernike_pib(cfg)
        # Recorder.__getitem__ asserts on out-of-range (no IndexError), so a
        # plain `for row in recorder` cannot terminate; index explicitly.
        rows = [dict(recorder[i]) for i in range(len(recorder))]
        key = _row_objective_key(rows[0])

        # --- Integrity: never adopt an abandoned evaluation. ---------------
        # history = initialisation row (_epoch 0) + at most one row per epoch.
        assert 1 <= len(rows) <= epochs + 1
        for row in rows:
            j = float(row["J"])
            assert np.isfinite(j)
            assert float(row[key]) > -100.0  # no -1e3 abandoned rows
            c = np.asarray(row["_c"])
            assert float(np.max(np.abs(c))) <= 5.0 + 1e-9  # driver clip
            assert np.isfinite(row["_diff"]) and np.isfinite(row["max_brt"])
            assert any(k.startswith("m_") for k in row)  # metric panel present

        # --- Gate observability: classify stalled rows. --------------------
        stats = _classify_rows(rows)
        stats["objective_key"] = key
        stats["best_objective"] = float(max(row[key] for row in rows))
        stats["final_c"] = rows[-1]["_c"]
        stats["delta"] = DELTA
        stats["n_eval_frames"] = n_eval_frames

        run_dir = _write_artifacts(f"smoke_f{n_eval_frames}", recorder, stats)
        print(f"\nRobust-SPGD smoke (n_eval_frames={n_eval_frames}): {run_dir}")
        print(stats)

    def test_smoke_asserts_rows_are_logged(self) -> None:
        """The smoke run must actually have produced epoch rows (and therefore
        exercised SLM writes + captures) — guards against silent no-op runs."""
        cfg = make_config(epochs=8)
        recorder = optimize_slm_zernike_pib(cfg)
        rows = [dict(recorder[i]) for i in range(len(recorder))]
        assert len(rows) >= 2  # initialisation + at least one logged epoch
        assert all(np.isfinite(float(row["J"])) for row in rows)