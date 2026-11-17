"""Issue #235 §2(1)(a)(b), offline: does the forward model see the phase the panel shows?

Why this exists
---------------
The 14-frame diagnostic of 2026-11-17 fed ``FrozenDefocusForward`` the *requested*
phase grid (radians, 640x640). The panel received ``create_phase_from_array``
output: mod-2*pi, quantized to 1023 levels, nearest-neighbour expanded to
1920x1200 by ``to_panel``. Nothing in the archived session checked that the two
agree, so "the model predicts the wrong direction" and "the model was given the
wrong input" were indistinguishable.

This script closes that gap without a device, from the archived session:

1. **Encode check** -- rebuild the gray the driver should have written from the
   archived requested phase and compare it with the archived ``panel_gray``.
   That validates the reconstruction before anything is concluded from it.
2. **Decode residual** -- invert ``panel_gray`` to radians under two 2*pi scales:

   * ``--encoding-2pi-gray`` (1023, the driver's encoding scale): the residual is
     pure quantization (one LSB = 2*pi/1023 = 6.1 mrad).
   * ``--physical-2pi-gray`` (993, the phase table read back at 1064 nm): the
     residual is quantization **plus** a 3% phase-scale compression, which at a
     0.03 rad probe is no longer small.

3. **Gain comparison** -- predict the N4 / RX / RY odd responses three times
   (requested phase, phase decoded at 1023, phase decoded at 993) and fit each
   against the measured, drift-corrected response. ``odd_affine.gain`` and
   ``cosine`` say how much of the measured-vs-predicted gap the *input* can carry.

No device is opened. Everything is read from ``--session``; the output is a JSON
summary, printed and (with ``--out``) written.

    python scripts/slm/issue235_panel_gray_inversion_20261117.py \
        --session data/debug/square1000_model_diag_20261117_140741

``--session`` must contain ``config.json``, ``manifest.json``, ``capture_*.npz``
(with ``model_phase`` / ``panel_gray`` / ``raw_frames`` / ``camera_xy``),
``probe_phases.npz`` and ``predictions.npz`` -- the layout
``square_1000_model_diagnostic_20261030.py`` writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT, ROOT / "scripts" / "slm"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

DEFAULT_SESSION = ROOT / "data/debug/square1000_model_diag_20261117_140741"
#: Panel geometry the driver expands to (``to_panel``: pitch 3 for a 640 grid).
PANEL_H, PANEL_W = 1200, 1920
#: ``to_panel``/``panel_grid`` pitch: 1920 // grid_size.
GRID_SIZE = 640
#: 2*pi in gray as the driver encodes it (``create_phase_from_array`` default).
ENCODING_2PI_GRAY = 1023
#: 2*pi in gray as the SLM phase table reports it at 1064 nm (diagnostic only).
PHYSICAL_2PI_GRAY = 993
#: Probe amplitude used by the 14-frame session (needed to scale the responses).
PROBE_AMPLITUDE_RAD = 0.03
#: 98% of the 12-bit full scale: the session's saturation rule, quoted for context.
FULL_SCALE = 4095.0
#: Capture indices of the anchor frames used for the drift reference.
ANCHOR_INDICES = (0, 1, 4, 7, 10, 13)


def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap radians into (-pi, pi]."""
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def decode_panel_phase(gray: np.ndarray, two_pi_gray: float) -> np.ndarray:
    """Invert ``panel_gray`` to radians: ``gray * 2*pi / two_pi_gray``."""
    return np.asarray(gray, dtype=np.float64) * (2.0 * np.pi / float(two_pi_gray))


def to_grid(panel_phase: np.ndarray, size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Sample a panel-sized array back onto the model's control grid.

    ``to_panel`` repeats each grid cell over a ``pitch``-wide block, so reading
    one pixel per block inverts it exactly. The model grid is square while the
    panel is 1200 rows tall, so grid rows at or past ``PANEL_H // pitch`` never
    reached the panel; they are returned as an all-False mask rather than zeros,
    because zeros there would read as "phase 0 measured".
    """
    pitch = PANEL_W // size
    panel = np.asarray(panel_phase, dtype=np.float64)
    rows = min(size, PANEL_H // pitch)
    grid = np.zeros((size, size), dtype=np.float64)
    grid[:rows] = panel[::pitch, ::pitch][:rows]
    valid = np.zeros((size, size), dtype=bool)
    valid[:rows] = True
    return grid, valid


def encode_gray(phase_rad: np.ndarray, two_pi_gray: int = ENCODING_2PI_GRAY) -> np.ndarray:
    """The driver's phase -> gray mapping, without the panel expansion."""
    wrapped = np.mod(np.asarray(phase_rad, dtype=np.float64), 2.0 * np.pi)
    return np.mod(np.floor(wrapped / (2.0 * np.pi) * two_pi_gray), two_pi_gray).astype(np.uint16)


def expand_to_panel(gray: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    """``to_panel`` for a uint16 grid, so the encode check runs on panel pixels."""
    pitch = PANEL_W // size
    return np.repeat(np.repeat(gray, pitch, axis=0), pitch, axis=1)[:PANEL_H]


def residual_stats(requested: np.ndarray, decoded: np.ndarray, valid: np.ndarray) -> dict:
    """Wrapped difference statistics between a requested and a decoded phase grid."""
    delta = _wrap(decoded - requested)[valid]
    if delta.size == 0:
        raise RuntimeError("no valid grid pixels to compare")
    return {
        "pixels": int(delta.size),
        "wrapped_rms_rad": float(np.sqrt(np.mean(delta**2))),
        "mean_rad": float(np.mean(delta)),
        "p95_abs_rad": float(np.percentile(np.abs(delta), 95)),
        "max_abs_rad": float(np.max(np.abs(delta))),
        "fraction_abs_gt_0p1_rad": float(np.mean(np.abs(delta) > 0.1)),
        "fraction_abs_gt_half_pi": float(np.mean(np.abs(delta) > 0.5 * np.pi)),
        "lsb_rad": float(2.0 * np.pi / ENCODING_2PI_GRAY),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.vdot(a, b).real / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))


def affine(reference: np.ndarray, measured: np.ndarray) -> dict:
    gain, offset = np.polyfit(reference.ravel(), measured.ravel(), 1)
    residual = measured - (gain * reference + offset)
    return {
        "gain": float(gain),
        "offset": float(offset),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(measured), 1e-20)),
        "correlation": float(np.corrcoef(reference.ravel(), measured.ravel())[0, 1]),
    }


def binned_rate(raw: np.ndarray, xy: np.ndarray, exposure_ms: float, bin_size: int = 1) -> np.ndarray:
    """Counts per millisecond on the camera grid, the analyzer's own sampler."""
    frame = np.asarray(raw, dtype=np.float64).mean(axis=0)
    if bin_size == 1:
        return frame[xy[..., 1].astype(int), xy[..., 0].astype(int)] / exposure_ms
    half = bin_size // 2
    offsets = np.arange(-half, bin_size - half)
    y = xy[..., 1].astype(int)[:, :, None, None] + offsets[None, None, None, :]
    x = xy[..., 0].astype(int)[:, :, None, None] + offsets[None, None, :, None]
    block = np.asarray(raw, dtype=np.float64)[:, y, x].mean(axis=(3, 4))
    return block.mean(axis=0) / exposure_ms


def _build_forward(config: dict, camera_xy: np.ndarray, exposure_ms: float):
    """The frozen forward model the session was analysed with."""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    params = load_frozen_parameters(
        Path(config["checkpoint"]),
        fft_size=int(config.get("fft_size", 2048)),
        pixel_samples=int(config.get("pixel_samples", 3)),
    )
    forward = FrozenDefocusForward(params, camera_xy)
    forward.exposure_ms = float(exposure_ms)
    return forward


def _predict(forward, phase: dict[str, np.ndarray], exposure_ms: float) -> dict[str, np.ndarray]:
    import torch

    out = {}
    forward.exposure_ms = float(exposure_ms)
    with torch.no_grad():
        for name, values in phase.items():
            tensor = torch.as_tensor(values, dtype=torch.float32)
            out[name] = forward.predict(tensor, exposure_ms=float(exposure_ms)).detach().cpu().numpy()
    return out


def _odd(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    return 0.5 * (plus - minus)


def analyse(session: Path, *, physical_2pi: int, encoding_2pi: int) -> dict:
    config = json.loads((session / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    phases = {k: v for k, v in np.load(session / "probe_phases.npz").items()}
    archived_predictions = {k: v for k, v in np.load(session / "predictions.npz").items()}

    rows = {int(row["index"]): row for row in manifest}
    role_of = {int(row["index"]): row["role"] for row in manifest}
    capture: dict[int, dict] = {}
    encode_rows = []
    for index in sorted(rows):
        with np.load(session / rows[index]["file"]) as saved:
            requested = np.asarray(saved["model_phase"], dtype=np.float64)
            gray = np.asarray(saved["panel_gray"], dtype=np.uint16)
            raw = np.asarray(saved["raw_frames"], dtype=np.float64)
            xy = np.asarray(saved["camera_xy"], dtype=np.float64)
        rebuilt = expand_to_panel(encode_gray(requested, encoding_2pi))
        same_shape = rebuilt.shape == gray.shape
        diff = np.abs(rebuilt.astype(np.int32) - gray.astype(np.int32)) if same_shape else None
        encode_rows.append(
            {
                "index": index,
                "role": role_of[index],
                "shape_matches": bool(same_shape),
                "max_abs_gray_error": int(diff.max()) if diff is not None else None,
                "mismatch_fraction": float(np.mean(diff != 0)) if diff is not None else None,
                "gray_max": int(gray.max()),
                "gray_unique": int(np.unique(gray).size),
            }
        )
        capture[index] = {"requested": requested, "gray": gray, "raw": raw, "xy": xy}

    anchor = capture[manifest[0]["index"]]
    requested_grid = anchor["requested"]
    valid = to_grid(np.zeros((PANEL_H, PANEL_W)))[1]
    decoded = {}
    residuals = {}
    for name, two_pi in (("encoding", encoding_2pi), ("physical", physical_2pi)):
        grid, valid = to_grid(decode_panel_phase(anchor["gray"], two_pi))
        decoded[name] = grid
        residuals[name] = residual_stats(requested_grid, grid, valid)
        # The slope of the decoded grid against the requested phase is the phase
        # scale the panel actually applied (1.0 == the model's input is what shone).
        residuals[name]["slope_decoded_over_requested"] = float(
            np.polyfit(_wrap(requested_grid)[valid], grid[valid], 1)[0]
        )

    # Measured, drift-corrected odd responses, exactly as the analyzer builds them.
    rates = {
        int(row["index"]): binned_rate(
            np.asarray(capture[int(row["index"])]["raw"], dtype=np.float64),
            capture[int(row["index"])]["xy"],
            float(row["exposure_ms"]),
        )
        for row in manifest
    }
    drift = {}
    for index in rates:
        left = max(a for a in ANCHOR_INDICES if a <= index)
        right = min(a for a in ANCHOR_INDICES if a >= index)
        weight = 0.0 if left == right else (index - left) / (right - left)
        drift[index] = (1.0 - weight) * rates[left] + weight * rates[right]

    # Rebuild the phase grid behind every captured role from *its own* panel_gray.
    def role_phase(two_pi: int) -> dict[str, np.ndarray]:
        out = {}
        for index, role in role_of.items():
            grid, _ = to_grid(decode_panel_phase(capture[index]["gray"], two_pi))
            out.setdefault(f"{role}@{index}", grid)
        return out

    exposure_ms = float(manifest[0]["exposure_ms"])
    forward = _build_forward(config, capture[manifest[0]["index"]]["xy"], exposure_ms)

    by_role: dict[str, list[int]] = {}
    for index, role in role_of.items():
        by_role.setdefault(role, []).append(index)
    role_index = {role: sorted(indices)[0] for role, indices in by_role.items()}

    actual = {}
    for two_pi in (encoding_2pi, physical_2pi):
        phase = {
            role: to_grid(decode_panel_phase(capture[role_index[role]]["gray"], two_pi))[0]
            for role in ("A", "N4+", "N4-", "RX+", "RX-", "RY+", "RY-")
        }
        actual[str(two_pi)] = _predict(forward, phase, exposure_ms)

    requested_pred = _predict(forward, {role: phases[role] for role in phases}, exposure_ms)
    prediction_drift = {
        role: float(
            np.sqrt(np.mean((requested_pred[role] - archived_predictions[role]) ** 2))
            / max(np.sqrt(np.mean(archived_predictions[role] ** 2)), 1e-20)
        )
        for role in requested_pred
    }

    gains = {}
    for label, (plus, minus) in {"N4": ("N4+", "N4-"), "RX": ("RX+", "RX-"), "RY": ("RY+", "RY-")}.items():
        measured = _odd(rates[role_index[plus]], rates[role_index[minus]])
        drift_corrected = _odd(
            rates[role_index[plus]] - drift[role_index[plus]],
            rates[role_index[minus]] - drift[role_index[minus]],
        )
        entry = {
            "probe_amplitude_rad": PROBE_AMPLITUDE_RAD,
            "measured_rms": float(np.sqrt(np.mean(measured**2))),
            "drift_corrected_rms": float(np.sqrt(np.mean(drift_corrected**2))),
            "predicted": {},
        }
        variants = {"requested": requested_pred}
        for two_pi in (encoding_2pi, physical_2pi):
            variants[f"actual_{two_pi}"] = actual[str(two_pi)]
        for name, pred in variants.items():
            odd = _odd(pred[plus], pred[minus])
            entry["predicted"][name] = {
                "rms": float(np.sqrt(np.mean(odd**2))),
                "cosine_panel": cosine(odd, measured),
                "cosine_drift_corrected": cosine(odd, drift_corrected),
                "odd_affine": affine(odd, drift_corrected),
            }
        gains[label] = entry

    return {
        "session": str(session),
        "device_free": True,
        "scales": {
            "encoding_2pi_gray": encoding_2pi,
            "physical_2pi_gray": physical_2pi,
            "encoding_scale_ratio": physical_2pi / encoding_2pi,
        },
        "config": {
            "seed_run": config.get("seed_run"),
            "seed_key": config.get("seed_key"),
            "checkpoint": config.get("checkpoint"),
            "probe_amplitude_rad": config.get("probe_amplitude_rad"),
            "base_exposure_ms": config.get("base_exposure_ms"),
        },
        "encode_check": encode_rows,
        "decode_residual": residuals,
        "prediction_reproducibility_relative_rms": prediction_drift,
        "gains": gains,
        "notes": [
            "decode_residual['encoding'] is pure quantization at 2*pi/1023 = 6.1 mrad per LSB.",
            "decode_residual['physical'] adds the 993-vs-1023 phase-scale compression "
            "(panel phase = gray * 2*pi / 993).",
            "odd_affine.gain is measured per predicted unit: 1.0 means the model's response "
            "amplitude already matches the measurement.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    parser.add_argument("--physical-2pi-gray", type=int, default=PHYSICAL_2PI_GRAY)
    parser.add_argument("--encoding-2pi-gray", type=int, default=ENCODING_2PI_GRAY)
    args = parser.parse_args(argv)
    if not args.session.is_dir():
        parser.error(f"session directory does not exist: {args.session}")
    summary = analyse(
        args.session,
        physical_2pi=int(args.physical_2pi_gray),
        encoding_2pi=int(args.encoding_2pi_gray),
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
