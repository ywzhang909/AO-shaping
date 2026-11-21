# -*- coding: utf-8 -*-
"""Offline strength gate on a frozen transfer checkpoint (no device access).

Implements the a priori criterion frozen in
``docs/dev/transfer_upstream_qualification_plan_20261119.md`` §2:

* held-out rows only (``split in {"test", "validation"}``), each row its own
  independent pass, worst pass decides;
* both sides in counts/ms (``comps.images`` already is; the prediction follows
  ``fit.fit_loss``: ``model.prediction()`` -> optional ``conv2d(kernel,
  padding=12)`` -> ``gain*optical + background + bias_counts/exposure``);
* radiometry fitted on ``split == "train"`` rows only, never on holdout rows;
* whole sampling window minus the zero-order disk (``--zero-radius-px``, in
  window pixels; ``0`` = no exclusion);
* a real dark frame **at the probe exposure** subtracted from the measured
  side (as counts/ms, never scaled by an exposure), and
  ``bias_counts/exposure`` subtracted from the prediction side (§2.3);
  a dark whose recorded exposure differs from the probe exposure is
  ``not_assessable``, and **without** ``--dark`` the gate is
  ``not_assessable`` (prereg §3.4), reporting only the diagnostic median
  fallback (clearly marked ``diagnostic_only``).

Also reported per row: ``rel_l2_fullwindow_unmasked``, the plain
``fit.metrics`` whole-window number with no DC and no mask. It is the
correctness self-check against the archived refit scores (replicate_a test
0.7232375337861096, extend_b test 0.5054810240467212, both at span 160), and
it is *not* the gate.

Exit codes: 0 = pass, 1 = fail, 2 = not assessable (or unusable input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ao_shaping.drivers.ccd import dark_frames as ccd_dark_frames
from ao_shaping.tools.slm.transfer import fit as transfer_fit
from ao_shaping.tools.slm.transfer import protocol as transfer_protocol
from ao_shaping.tools.slm.transfer import radiometry as transfer_radiometry
from ao_shaping.tools.slm.transfer.strength_gate import (
    STRENGTH_GATE_DELTA,
    gate_rel_l2,
    masked_absolute_rel_l2,
    match_dark_exposure,
    window_zero_mask,
)

#: Holdout splits the frozen plan scores. Calibration rows (reference_start and
#: the signed ramps) are never holdout probes.
HOLDOUT_SPLITS = ("test", "validation")
#: The prediction entry point this runner mirrors (``build_fit_model`` +
#: ``fit_loss``'s post-chain); recorded in the evidence for traceability.
GATE_MODEL_SOURCE = "ao_shaping.tools.slm.transfer.fit.build_fit_model"


class DarkUnavailable(Exception):
    """The supplied dark frame cannot be aligned with the probe window."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_dataset(spec: Path) -> Path:
    """The directory that actually holds ``manifest.json``.

    Accepts the capture root itself (``replicate_a_20261118``), a ``capture``
    subdirectory, or a run directory whose single child capture holds it.
    """
    spec = Path(spec)
    candidates = [spec, spec / "capture"]
    if spec.is_dir():
        candidates += sorted(child for child in spec.iterdir() if child.is_dir())
    found: list[Path] = []
    for candidate in candidates:
        if (candidate / "manifest.json").is_file() and candidate.resolve() not in {
            path.resolve() for path in found
        }:
            found.append(candidate)
    if not found:
        raise SystemExit(f"no capture with manifest.json under {spec}")
    if len(found) > 1:
        listed = ", ".join(str(path) for path in found)
        raise SystemExit(f"ambiguous capture roots under {spec}: {listed}")
    return found[0]


def resolve_checkpoint(spec: Path) -> tuple[Path, Path]:
    """Return ``(fit_dir, frozen_parameters.json)`` from a dir or the file."""
    spec = Path(spec)
    if spec.is_dir():
        fit_dir, frozen = spec, spec / "frozen_parameters.json"
    else:
        fit_dir, frozen = spec.parent, spec
    if not frozen.is_file():
        raise SystemExit(f"missing frozen parameters: {frozen}")
    return fit_dir, frozen


def forward_optical(comps) -> np.ndarray:
    """The optical forward in model units, exactly as ``fit.fit_loss`` builds it.

    ``model.prediction()`` then the frozen blaze kernel with ``padding=12``.
    The master ``FitComponents`` carries no static basis, so there is no static
    term (the same limitation ``refit_provenance.forward_api`` records).
    """
    import torch
    from torch.nn.functional import conv2d

    with torch.no_grad():
        prediction = comps.model.prediction()
        kernel = comps.kernel
        if kernel is not None:
            prediction = conv2d(prediction[:, None], kernel, padding=12)[:, 0]
        return prediction.detach().cpu().numpy().astype(np.float64)


def prediction_counts(optical: np.ndarray, radiometry: dict, exposure_ms) -> np.ndarray:
    """``gain*optical + background + bias_counts/exposure`` in counts/ms."""
    exposure = np.asarray(exposure_ms, dtype=float)[:, None, None]
    return (optical * float(radiometry["gain"])
            + float(radiometry["background"])
            + float(radiometry["bias_counts"]) / exposure)


def _frames_of(payload) -> np.ndarray:
    frames = np.asarray(payload, dtype=float)
    if frames.ndim == 3:
        frames = frames.mean(axis=0)
    if frames.ndim != 2:
        raise DarkUnavailable(f"dark frame must be 2D or a stack of 2D frames, got {frames.shape}")
    return frames


def _dark_rate(frames: np.ndarray, exposure_ms: float | None) -> np.ndarray:
    """Dark signal in counts/ms; a raw frame is divided by its own exposure."""
    if exposure_ms is None:
        raise DarkUnavailable(
            "dark frame holds raw counts but carries no exposure_ms; cannot form counts/ms"
        )
    exposure = float(exposure_ms)
    if not np.isfinite(exposure) or exposure <= 0:
        raise DarkUnavailable(f"dark exposure must be positive and finite, got {exposure_ms!r}")
    return frames / exposure


def _dark_window(rate: np.ndarray, window_shape: tuple[int, int], xy: np.ndarray) -> np.ndarray:
    """Project a dark frame (window or full-frame) onto the probe window."""
    rate = np.asarray(rate, dtype=float)
    if rate.shape == tuple(window_shape):
        return rate
    if rate.ndim != 2:
        raise DarkUnavailable(f"dark frame must be 2D, got {rate.shape}")
    rows, cols = xy[..., 1].astype(int), xy[..., 0].astype(int)
    if rows.max() >= rate.shape[0] or cols.max() >= rate.shape[1] or rows.min() < 0 or cols.min() < 0:
        raise DarkUnavailable(
            f"dark frame {rate.shape} does not cover the probe window "
            f"(y {rows.min()}..{rows.max()}, x {cols.min()}..{cols.max()})"
        )
    return rate[rows, cols]


def load_dark(path: Path, *, window_shape, xy, exposure_override_ms=None) -> tuple[np.ndarray, dict]:
    """Load a dark frame and return ``(counts/ms on the window, evidence)``.

    Supported inputs:

    * ``.npz`` with ``exposure_ms`` (scalar) plus one of ``dark_frames`` /
      ``raw_frames`` / ``frames`` / ``dark`` (or exactly one array). The frame
      is divided by ``exposure_ms``; a ``dark_rate`` array is taken as counts/ms
      directly.
    * ``.json`` with ``{"frames"|"dark"|"counts": 2D/3D, "exposure_ms": float}``
      or a bare 2D list. Same division rule.

    ``exposure_override_ms`` supplies the exposure when the file itself does not
    carry one (a CCD dark-library ``mean``-keyed ``.npz`` gets its exposure from
    the manifest, not from the array). When it is None the single-dark behaviour
    is byte-for-byte unchanged.

    No dark file means *not assessable*: this loader never invents one.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"dark frame not found: {path}")
    suffix = path.suffix.lower()
    rate: np.ndarray
    # An explicit override wins; it is required for library ``mean`` npz files.
    exposure_ms: float | None = None if exposure_override_ms is None \
        else float(exposure_override_ms)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            keys = set(saved.files)
            if "exposure_ms" in keys and exposure_override_ms is None:
                exposure_ms = float(np.asarray(saved["exposure_ms"]).reshape(-1)[0])
            if "dark_rate" in keys:
                rate = np.asarray(saved["dark_rate"], dtype=float)
            else:
                frame_keys = [key for key in ("dark_frames", "raw_frames", "frames", "dark", "mean")
                              if key in keys]
                if not frame_keys and len(keys) == 1:
                    frame_keys = [next(iter(keys))]
                if not frame_keys:
                    raise DarkUnavailable(f"{path} has no dark frame array (keys: {sorted(keys)})")
                rate = _dark_rate(_frames_of(saved[frame_keys[0]]), exposure_ms)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if payload.get("exposure_ms") is not None:
                exposure_ms = float(payload["exposure_ms"])
            if payload.get("dark_rate") is not None:
                rate = np.asarray(payload["dark_rate"], dtype=float)
            else:
                frame_keys = [key for key in ("frames", "dark", "counts", "mean") if key in payload]
                if not frame_keys:
                    raise DarkUnavailable(f"{path} has no dark frame array (keys: {sorted(payload)})")
                rate = _dark_rate(_frames_of(payload[frame_keys[0]]), exposure_ms)
        elif isinstance(payload, list):
            rate = _dark_rate(_frames_of(payload), exposure_ms)
        else:
            raise DarkUnavailable(f"unsupported dark JSON payload type: {type(payload).__name__}")
    else:
        raise DarkUnavailable(f"unsupported dark frame format {suffix!r}; use .npz or .json")

    window = _dark_window(rate, window_shape, xy)
    evidence = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "exposure_ms": exposure_ms,
        "shape": list(np.asarray(rate).shape),
        "dark_rate_median_counts_per_ms": float(np.median(window)),
    }
    return window, evidence


def load_optical_path_record(path: Path) -> dict:
    """Read ``camera.bit_depth`` / ``camera.full_scale`` from a frozen record.

    光路记录的位深/满量程位于 ``devices.camera`` (见
    ``ao_shaping.utils.io.optical_path_record.CAMERA_FIELDS``)。这里只读这两个
    字段, 缺失或非法一律返回 ``None``, 绝不臆造值 —— 位深一致性判据宁可
    ``not_assessable``, 也不能靠猜测放行。
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"optical path record not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = (payload.get("devices") or {}).get("camera") or {}
    bit_depth = camera.get("bit_depth")
    full_scale = camera.get("full_scale")
    parsed_depth = None if bit_depth is None else int(bit_depth)
    parsed_scale = None if full_scale is None else float(full_scale)
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "bit_depth": parsed_depth, "full_scale": parsed_scale}


def load_dark_directory(path: Path) -> dict[float, dict]:
    """Build ``{exposure_ms: dark entry}`` from a dark directory.

    Two layouts are supported (no globbing of a library's arrays — the manifest
    is its only index, per ``dark_frames.py``'s contract):

    * a CCD dark library: ``manifest.json`` with the v1 schema plus
      ``dark-exp-*ms.npz`` files holding ``mean`` (no embedded ``exposure_ms``);
    * loose ``*.npz`` dark files carrying their own ``exposure_ms``.

    Each entry is ``{"path": Path, "exposure_ms": float|None, "bit_depth": ...,
    "full_scale": ...}``. A library entry's exposure and bit-depth metadata come
    from the manifest; a loose file's exposure must be embedded in the npz.
    """
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"dark directory not found: {path}")
    manifest_path = path / "manifest.json"
    entries: dict[float, dict] = {}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != ccd_dark_frames.SCHEMA:
            raise SystemExit(
                f"dark manifest schema mismatch: {payload.get('schema')!r} != "
                f"{ccd_dark_frames.SCHEMA!r}"
            )
        for raw in payload.get("entries", []):
            try:
                entry = ccd_dark_frames.DarkFrameEntry.from_dict(raw)
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(f"invalid dark manifest entry in {manifest_path}: {error}")
            array_path = path / entry.file
            if not array_path.is_file():
                raise SystemExit(f"dark manifest points to a missing array: {array_path}")
            entries[float(entry.exposure_ms)] = {
                "path": array_path,
                "exposure_ms": float(entry.exposure_ms),
                "bit_depth": entry.bit_depth,
                "full_scale": entry.full_scale,
            }
        return entries

    # Loose npz files: exposure must live inside the file, never in its name.
    for npz_path in sorted(path.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as saved:
            if "exposure_ms" not in saved.files:
                raise SystemExit(
                    f"loose dark {npz_path} has no exposure_ms; use a dark library manifest"
                )
            exposure = float(np.asarray(saved["exposure_ms"]).reshape(-1)[0])
        entries[exposure] = {
            "path": npz_path, "exposure_ms": exposure,
            "bit_depth": None, "full_scale": None,
        }
    return entries


def _metadata_mismatch(entry: dict, data_bit_depth, data_full_scale) -> str | None:
    """Return a reason when the dark's bit depth/full scale disagrees, else None.

    ``None`` on either side is a mismatch, never an assumption: a dark without
    known metadata cannot prove it matches the frozen dataset's quantisation.
    """
    if data_bit_depth is None or data_full_scale is None:
        return None  # no frozen dataset quantisation to compare against
    dark_depth = entry.get("bit_depth")
    dark_scale = entry.get("full_scale")
    if dark_depth is None or dark_scale is None:
        return (f"dark bit_depth/full_scale unknown (None/None) vs frozen "
                f"bit_depth={data_bit_depth}, full_scale={data_full_scale}")
    if int(dark_depth) != int(data_bit_depth) or float(dark_scale) != float(data_full_scale):
        return (f"dark bit_depth={dark_depth}, full_scale={dark_scale} != frozen "
                f"bit_depth={data_bit_depth}, full_scale={data_full_scale}")
    return None


def _pass_summary(rows: list[dict], split: str) -> dict:
    scored = [row for row in rows if row["split"] == split]
    full_window = [row["rel_l2_fullwindow_unmasked"] for row in scored]
    summary = {
        "split": split,
        "n_rows": len(scored),
        # Diagnostic only: the archived refit scores are whole-window means, so
        # this is the number the correctness self-check compares against.
        "rel_l2_fullwindow_unmasked_mean": float(np.mean(full_window)),
        "rel_l2_fullwindow_unmasked_median": float(np.median(full_window)),
    }
    valid = [row for row in scored
             if row["rel_l2_gate"] is not None and row["n_pixels_gate"] > 0]
    if not valid:
        summary.update({
            "n_assessable_rows": 0,
            "rel_l2_gate": None,
            "rel_l2_gate_median": None,
            "verdict": "not_assessable",
            "reason": "no assessable holdout row (dark frame missing or empty mask)",
        })
        return summary
    worst = max(valid, key=lambda row: row["rel_l2_gate"])
    summary.update({
        "n_assessable_rows": len(valid),
        "rel_l2_gate": float(worst["rel_l2_gate"]),
        "rel_l2_gate_worst_label": worst["label"],
        "rel_l2_gate_median": float(np.median([row["rel_l2_gate"] for row in valid])),
    })
    # 预注册 §3.4 / 计划 §6.5-6.6: 只要有一个 holdout 行不可评估（暗场缺失、
    # 曝光簿记不符、饱和、空 mask）, 该 split 就是 not_assessable —— 不能只把
    # 这几行丢掉再按剩下的行判 pass/fail, 那等于允许"删掉不利的行"通过门。
    # worst/median 仍作为诊断值保留, 但不产生 pass/fail。
    unassessable = [row for row in scored if row not in valid]
    if unassessable:
        summary.update({
            "verdict": "not_assessable",
            "unassessable_labels": [row["label"] for row in unassessable],
            "reason": f"{len(unassessable)}/{len(scored)} holdout rows not assessable: "
                      + "; ".join(f"{row['label']}: {row['not_assessable_reason']}"
                                  for row in unassessable),
        })
    else:
        summary["verdict"] = "pass" if worst["rel_l2_gate"] <= STRENGTH_GATE_DELTA else "fail"
    return summary


def _overall_verdict(passes: list[dict], *, dark_supplied: bool,
                     dark_error: str | None) -> str:
    """Combine per-split verdicts: any unassessable split blocks the whole gate.

    §6.5 要求"缺一档就整份门判 not_assessable"；因此只要有一个 split 不可评估,
    整体就是 not_assessable, 不因为另一个 split 通过而放行。
    """
    if not dark_supplied or dark_error is not None:
        return "not_assessable"
    if not passes or any(item["verdict"] == "not_assessable" for item in passes):
        return "not_assessable"
    if any(item["verdict"] == "fail" for item in passes):
        return "fail"
    return "pass"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path,
                        help="capture directory holding manifest.json (root, capture/ or run dir)")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="frozen_parameters.json (or the fit directory that holds it)")
    parser.add_argument("--span", type=int, default=110,
                        help="sampling window half-width in samples (archived refits: 160)")
    parser.add_argument("--window-step", type=float, default=2.0,
                        help="sensor pixels per window sample (archived campaigns: 2)")
    parser.add_argument("--zero-radius-px", type=float, default=0.0,
                        help="zero-order exclusion radius in window pixels; 0 = keep the whole window")
    dark_source = parser.add_mutually_exclusive_group()
    dark_source.add_argument("--dark", type=Path, default=None,
                             help="single dark frame (.npz or .json) at the probe exposure; "
                                  "mutually exclusive with --dark-dir")
    dark_source.add_argument("--dark-dir", type=Path, default=None,
                             help="directory of darks: a CCD dark library (manifest.json + "
                                  "dark-exp-*ms.npz) or loose *.npz frames; one dark per probe "
                                  "exposure is matched strictly")
    parser.add_argument("--optical-path-record", type=Path, default=None,
                        help="optical_path_record.json; supplies data bit_depth/full_scale "
                             "from devices.camera")
    parser.add_argument("--data-bit-depth", type=int, default=None,
                        help="explicit dataset bit depth (fallback when no optical-path record)")
    parser.add_argument("--data-full-scale", type=float, default=None,
                        help="explicit dataset full scale (fallback when no optical-path record)")
    parser.add_argument("--origin", choices=("measured", "fitted"), default="measured",
                        help="model origin: the dataset's reference_start spot (as "
                             "build_fit_model builds it, and how the archived refits were "
                             "scored) or the checkpoint's frozen fitted origin")
    parser.add_argument("--radiometry", choices=("train", "checkpoint"), default="train",
                        help="gain/background/bias fitted on split=='train' rows of this dataset "
                             "(plan §2.1) or taken verbatim from the checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, type=Path, help="evidence JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = resolve_dataset(args.dataset)
    fit_dir, frozen_path = resolve_checkpoint(args.checkpoint)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for key in ("gain", "background", "bias_counts", "origin"):
        if key not in frozen:
            raise SystemExit(f"{frozen_path} lacks {key!r}")

    comps = transfer_fit.build_fit_model(dataset, fit_dir, span=args.span,
                                         device=args.device, adaptation="full")
    if args.origin == "fitted":
        import torch
        with torch.no_grad():
            comps.model.origin.copy_(torch.as_tensor(np.asarray(frozen["origin"], dtype=float),
                                                    device=comps.model.origin.device))
    optical = forward_optical(comps)

    if args.radiometry == "train":
        _, radiometry = transfer_radiometry.fit_radiometry(
            optical, comps.images, comps.exposure_values, comps.train, mode="exposure")
        radiometry_source = "fit on split=='train' rows of --dataset (plan §2.1)"
    else:
        radiometry = {key: float(frozen[key]) for key in ("gain", "background", "bias_counts")}
        radiometry_source = "frozen_parameters.json (already train-only at fit time)"
    prediction = prediction_counts(optical, radiometry, comps.exposure_values)

    window_shape = tuple(np.asarray(comps.images[0]).shape)
    mask = window_zero_mask(window_shape, window_step=args.window_step,
                            zero_radius_px=args.zero_radius_px)
    zero_radius_full_px = float(args.zero_radius_px) * float(args.window_step)

    # Frozen dataset quantisation: prefer the optical-path record's
    # devices.camera fields, fall back to explicit CLI values. Nothing is
    # guessed — if neither is given the consistency check simply cannot run.
    optical_record = None
    data_bit_depth = args.data_bit_depth
    data_full_scale = args.data_full_scale
    if args.optical_path_record is not None:
        optical_record = load_optical_path_record(args.optical_path_record)
        if data_bit_depth is None:
            data_bit_depth = optical_record["bit_depth"]
        if data_full_scale is None:
            data_full_scale = optical_record["full_scale"]

    dark_rate = None
    dark_evidence = None
    dark_error = None
    # For --dark-dir: exposure -> {"rate": ndarray, "entry": metadata dict}.
    dark_by_exposure: dict[float, dict] = {}
    if args.dark is not None:
        try:
            dark_rate, dark_evidence = load_dark(args.dark, window_shape=window_shape, xy=comps.xy)
        except DarkUnavailable as error:
            dark_error = str(error)
    elif args.dark_dir is not None:
        dark_entries = load_dark_directory(args.dark_dir)
        for exposure, entry in dark_entries.items():
            try:
                rate, evidence = load_dark(
                    entry["path"], window_shape=window_shape, xy=comps.xy,
                    exposure_override_ms=entry["exposure_ms"])
            except DarkUnavailable:
                # One bad dark must not drop the whole verdict; leave this
                # exposure unmatched so its rows become not_assessable.
                continue
            evidence["bit_depth"] = entry["bit_depth"]
            evidence["full_scale"] = entry["full_scale"]
            dark_by_exposure[exposure] = {"rate": rate, "evidence": evidence, "entry": entry}

    holdout = [(index, row) for index, row in enumerate(comps.rows)
               if row.get("split") in HOLDOUT_SPLITS]
    if not holdout:
        raise SystemExit(f"no holdout rows (split in {HOLDOUT_SPLITS}) in {dataset}")

    rows: list[dict] = []
    dark_selection: dict[str, float | None] = {}
    used_dark_entries: dict[float, dict] = {}  # exposure -> evidence (dedup by exposure)
    validity_darks_ok = True         # every matched dark agreed with frozen metadata
    all_rows_have_matched_dark = True  # no holdout row lacked a dark at its exposure
    for index, row in holdout:
        exposure = float(row["exposure_ms"])
        measured = np.asarray(comps.images[index], dtype=float)
        predicted = np.asarray(prediction[index], dtype=float)
        full_window = float(transfer_fit.metrics([predicted], [measured])[0]["relative_l2"])

        median_pred = float(np.median(predicted[mask])) if mask.any() else 0.0
        median_meas = float(np.median(measured[mask])) if mask.any() else 0.0
        fallback = masked_absolute_rel_l2(predicted, measured, mask,
                                          dc_pred=median_pred, dc_meas=median_meas)

        # Prereg §3 items 3: the recorded exposure must match every frame, and
        # a saturated frame makes the pass not assessable (never a pass/fail).
        frame_exposures = {float(frame["exposure_ms"]) for frame in row.get("frames", [])
                           if frame.get("exposure_ms") is not None}
        setting = row.get("exposure_setting") or {}
        exposure_consistent = bool(frame_exposures) and frame_exposures == {exposure} \
            and setting.get("exposure_ms") in (None, exposure)
        saturated = int(row.get("saturation_pixels", 0) or 0)

        gate = None
        reason = None
        row_dark_rate = dark_rate           # --dark single-file path
        dark_exposure = None if dark_evidence is None else dark_evidence.get("exposure_ms")
        if args.dark_dir is not None:
            # Per-row dark: match strictly at this row's probe exposure. One
            # unmatched row makes the row not_assessable AND (per 计划 §6.5) the
            # whole split not_assessable -- a holdout row must never be dropped
            # from the worst-case max just because its dark is missing.
            matched_key = match_dark_exposure(dark_by_exposure, exposure)
            dark_selection[str(row["label"])] = matched_key
            if matched_key is None:
                reason = f"no dark at probe exposure {exposure:g} ms (prereg §3.4)"
                all_rows_have_matched_dark = False
            else:
                selected = dark_by_exposure[matched_key]
                row_dark_rate = selected["rate"]
                dark_exposure = selected["evidence"].get("exposure_ms")
                used_dark_entries[matched_key] = selected["evidence"]
                mismatch = _metadata_mismatch(selected["entry"], data_bit_depth, data_full_scale)
                if mismatch is not None:
                    reason = mismatch
                    validity_darks_ok = False
        if reason is None and row_dark_rate is None:
            reason = (dark_error
                      or ("no dark supplied (prereg §3.4 not assessable)"
                          if args.dark_dir is not None
                          else "no --dark supplied (prereg §3.4 not assessable)"))
        if reason is None and dark_exposure is not None and not np.isclose(
                dark_exposure, exposure, rtol=0.0, atol=1e-9):
            reason = (f"dark exposure {dark_exposure:.6g} ms != probe exposure {exposure:.6g} ms "
                      "(prereg §3.4 requires a dark at the probe exposure)")
        if exposure_consistent is False and reason is None:
            reason = "exposure bookkeeping mismatch: " \
                f"row={exposure} frames={sorted(frame_exposures)}"
        elif saturated > 0 and reason is None:
            reason = f"{saturated} saturated pixels in the probe frame"
        elif reason is None:
            bias_counts = float(radiometry["bias_counts"])
            # dark_rate is already counts/ms at the probe exposure; bias_counts/
            # exposure is counts/ms. gate_rel_l2 subtracts both directly and
            # never multiplies the dark by an exposure.
            gate = gate_rel_l2(predicted, measured, mask,
                               bias_rate=bias_counts / exposure,
                               dark_rate=row_dark_rate)
            if gate["n_pixels"] == 0:
                reason = "empty window mask"
                gate = None
        rows.append({
            "label": row["label"],
            "split": row["split"],
            "exposure_ms": exposure,
            "rel_l2_gate": None if gate is None else gate["rel_l2"],
            "energy_meas_gate": None if gate is None else gate["energy_meas"],
            "n_pixels_gate": 0 if gate is None else gate["n_pixels"],
            "not_assessable_reason": reason,
            "exposure_consistent": exposure_consistent,
            "saturation_pixels": saturated,
            "rel_l2_median_fallback": fallback["rel_l2"],
            "diagnostic_only": True,
            "rel_l2_fullwindow_unmasked": full_window,
        })

    passes = [_pass_summary(rows, split) for split in HOLDOUT_SPLITS
              if any(row["split"] == split for row in rows)]
    verdict = _overall_verdict(
        passes,
        dark_supplied=not (args.dark is None and args.dark_dir is None),
        dark_error=dark_error,
    )

    bit_depths = sorted({row["bit_depth"] for row in comps.rows if row.get("bit_depth") is not None})
    full_scales = sorted({row["full_scale"] for row in comps.rows if row.get("full_scale") is not None})
    exposure_ok = all(row["exposure_consistent"] for row in rows)
    saturation_ok = all(row["saturation_pixels"] == 0 for row in rows)

    # Prereg §3's six measurement-validity prerequisites, item by item. ``ok``
    # is None where this offline runner cannot see the evidence at all; the
    # detail says where it lives. Only the dark frame is a hard blocker for the
    # gate itself (task spec: missing --dark => not_assessable).
    validity = [
        {"item": "1_model_definition_and_split", "ok": True,
         "detail": f"checkpoint sha256 {sha256_file(frozen_path)[:16]}; "
                   f"holdout splits {HOLDOUT_SPLITS}; train rows used for radiometry"},
        {"item": "2_roi_mask_geometry", "ok": int(mask.sum()) > 0,
         "detail": f"window {list(window_shape)} step {args.window_step}; "
                   f"{int(mask.sum())} of {mask.size} samples kept; "
                   f"zero_radius={args.zero_radius_px} window px "
                   f"({zero_radius_full_px} sensor px)"},
        {"item": "3_exposure_accounting", "ok": exposure_ok and saturation_ok,
         "detail": f"per-row exposure == every frame exposure: {exposure_ok}; "
                   f"no saturated pixels: {saturation_ok}"},
        {"item": "4_dark_bitdepth_fullscale",
         "ok": ((dark_rate is not None) if args.dark_dir is None
                else (all_rows_have_matched_dark and validity_darks_ok)),
         "detail": ((f"dark: {dark_evidence}" if dark_evidence is not None
                     else f"dark missing: {dark_error or 'no --dark supplied'}")
                    if args.dark_dir is None
                    else f"darks: {list(used_dark_entries.values())}; "
                         f"all rows matched={all_rows_have_matched_dark}; "
                         f"data bit_depth={data_bit_depth}; data full_scale={data_full_scale}") +
                   f"; bit_depth={bit_depths or None}; full_scale={full_scales or None}"},
        {"item": "5_settle_and_dropped_frames", "ok": None,
         "detail": "not derivable from the capture manifest; read the session log"},
        {"item": "6_script_hashes", "ok": True,
         "detail": "runner + strength_gate + fit + protocol recorded in script_sha256"},
        {"item": "delta_frozen", "ok": True,
         "detail": f"STRENGTH_GATE_DELTA={STRENGTH_GATE_DELTA} (read-only pre-registration)"},
    ]

    evidence = {
        "kind": "transfer_strength_gate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(frozen_path),
        "script_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "strength_gate": sha256_file(Path(__file__).resolve().parents[2]
                                         / "src/ao_shaping/tools/slm/transfer/strength_gate.py"),
            "fit": sha256_file(Path(transfer_fit.__file__).resolve()),
            "protocol": sha256_file(Path(transfer_protocol.__file__).resolve()),
        },
        "model_source": GATE_MODEL_SOURCE,
        "span": args.span,
        "window_step": args.window_step,
        "zero_radius_px": args.zero_radius_px,
        "zero_radius_full_px": zero_radius_full_px,
        "window_shape": list(window_shape),
        "bit_depth": bit_depths or None,
        "full_scale": full_scales or None,
        "dark_source": (None if args.dark is None and args.dark_dir is None
                        else str(Path(args.dark if args.dark is not None
                                      else args.dark_dir).resolve())),
        "dark": (dark_evidence if args.dark_dir is None
                 else [used_dark_entries[key] for key in sorted(used_dark_entries)]),
        "dark_selection": dark_selection if args.dark_dir is not None else None,
        "data_bit_depth": data_bit_depth,
        "data_full_scale": data_full_scale,
        "optical_path_record": optical_record,
        "dark_error": dark_error,
        "origin": args.origin,
        "radiometry": radiometry,
        "radiometry_source": radiometry_source,
        "radiometry_checkpoint": {key: float(frozen[key])
                                  for key in ("gain", "background", "bias_counts")},
        "delta": STRENGTH_GATE_DELTA,
        "metric": "masked absolute relL2 = ||(p-dc_p)-(t-dc_t)||_mask / max(||t-dc_t||_mask, 1e-9)",
        "passes": passes,
        "rows": rows,
        "verdict": verdict,
        "validity": validity,
        "notes": {
            "rel_l2_fullwindow_unmasked": "fit.metrics whole-window number, no DC and no mask; "
                                          "correctness self-check only, NOT the gate",
            "rel_l2_median_fallback": "each side minus its own median inside the mask; "
                                      "diagnostic_only, never part of the verdict",
            "exit_codes": "0 pass, 1 fail, 2 not assessable",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    for item in passes:
        print(f"{item['split']}: {item['verdict']} "
              f"(n={item['n_assessable_rows']}/{item['n_rows']}, "
              f"worst relL2={item['rel_l2_gate']})")
        print(f"  {item['split']} full-window unmasked relL2 "
              f"mean={item['rel_l2_fullwindow_unmasked_mean']:.6f} "
              f"median={item['rel_l2_fullwindow_unmasked_median']:.6f} (diagnostic)")
    print(f"verdict: {verdict}")
    print(f"wrote {args.out}")
    if verdict == "pass":
        return 0
    return 1 if verdict == "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
