"""Offline, camera-independent review of one model-in-loop session."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def summarize_session(
    session_dir: Path, *, max_cv: float, min_ee: float, min_iou: float,
) -> dict:
    """Assess recorded measurements against application-specific thresholds.

    The last accepted row need not be the saved best phase: the optimizer retains
    the best measured score. Report both rather than attributing a row to the
    saved phase without a separate replay.
    """
    session_dir = Path(session_dir)
    config = json.loads((session_dir / "run_config.json").read_text(encoding="utf-8"))
    csv_files = sorted(session_dir.glob("slm_model_in_loop_*.csv"))
    if len(csv_files) != 1:
        raise ValueError(f"expected one run CSV in {session_dir}, found {len(csv_files)}")
    with csv_files[0].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rounds = [row for row in rows if row.get("stage") == "round"]
    baseline = next((row for row in rows if row.get("stage") == "calibration"), None)
    accepted = [row for row in rounds if row.get("accepted", "").lower() == "true"]
    eligible = [row for row in accepted if _number(row.get("score_after")) is not None]
    best = max(eligible, key=lambda row: float(row["score_after"])) if eligible else None
    geometry_path = session_dir / "bench_geometry.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.exists() else {}
    metrics = {
        "cv": _number(best.get("cv_after")) if best else None,
        "ee": _number(best.get("ee_after")) if best else None,
        "iou": _number(best.get("shape_iou_after")) if best else None,
        "holdout_correlation": _number(best.get("holdout_correlation")) if best else None,
        "geometry_correlation": _number(str(geometry.get("correlation"))),
    }
    checks = {
        "phase_saved": (session_dir / "best_phase.npy").exists(),
        "accepted_round": bool(accepted),
        "cv": metrics["cv"] is not None and metrics["cv"] <= max_cv,
        "ee": metrics["ee"] is not None and metrics["ee"] >= min_ee,
        "iou": metrics["iou"] is not None and metrics["iou"] >= min_iou,
        "geometry": metrics["geometry_correlation"] is not None
        and metrics["geometry_correlation"] >= float(config.get("min_geometry_correlation", 0.5)),
        "holdout": metrics["holdout_correlation"] is not None
        and metrics["holdout_correlation"] >= float(config.get("min_holdout_correlation", 0.5)),
        "beats_flat": best is not None and baseline is not None
        and _number(best.get("score_after")) is not None
        and _number(baseline.get("score_after")) is not None
        and float(best["score_after"]) > float(baseline["score_after"]),
    }
    return {
        "session_dir": str(session_dir.resolve()),
        "target": {"shape": config["target_shape"], "side_camera_px": config["target_side"]},
        "thresholds": {"max_cv": max_cv, "min_ee": min_ee, "min_iou": min_iou},
        "rounds": len(rounds), "accepted_rounds": len(accepted),
        "best_recorded_round": int(best["round"]) if best else None,
        "metrics_at_best_recorded_round": metrics,
        "checks": checks,
        "verdict": "passes_recorded_gates" if all(checks.values()) else "review_required",
        "qualification": "Recorded round metrics only; replay saved phase at fixed exposure for final optical acceptance.",
    }
