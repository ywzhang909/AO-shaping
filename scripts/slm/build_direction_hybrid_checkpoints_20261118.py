"""Build hybrid forward-model checkpoints for the offline direction gate (issue #234).

The two epochs disagree about nearly everything in the optical block:

* 2026-11-07 fit (``defocus_forward_20261107_focus_r2_fit``) -- passes the
  direction gate (median 0.618/0.693/0.671 at 0.02/0.05/0.10 rad) but its
  absolute intensity is off by ~7.5x in closed loop (``anchored_m3_20261116``).
* 2026-11-18 fit (``axis_mismatch_validation_20261118``) -- absolute CV is much
  closer (3.29 vs 3.83 measured) but it fails the direction gate
  (0.469/0.368/0.400) when replayed on the 11-07 single-mode screens with
  epoch-correct registration.

This script writes small factorial hybrids so the gate can attribute *which*
block carries the direction information: defocus, envelope width, or envelope
centre.  Gain/background are affine and provably cancel in the Pearson gate, so
they always come from the 11-18 fit and only matter for absolute intensity.

Ramp geometry stays at the 2026-11-07 value: the replay frames are 11-07 frames,
so that ramp is the geometrically correct one for them.

Output: one ``frozen_parameters.json``-shaped file per hybrid under ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD_FIT = Path(
    "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json"
)
DEFAULT_FRESH = Path(
    "data/debug/slm_transfer/axis_mismatch_validation_20261118/fit/frozen_parameters.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ramp-source",
        choices=("old", "fresh"),
        default="old",
        help="old = 2026-11-07 ramp geometry (correct for replaying 11-07 frames); "
        "fresh = 2026-11-18 ramp (only needed when a consumer ignores this field anyway)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=128,
        help="control-grid size; must equal the capture's model_phase shape for fit.py "
        "(320 for the 2026-11-18 capture, 640 for the 2026-11-07 screens)",
    )
    args = parser.parse_args()

    old_fit = json.loads(OLD_FIT.read_text(encoding="utf-8"))
    old = old_fit["models"]["M3_illumination"]
    fresh = json.loads(args.fresh.read_text(encoding="utf-8"))
    ramp = old_fit["ramp_matrix"] if args.ramp_source == "old" else fresh["ramp_matrix"]

    common = {
        # Explicit zero-valued modes keep fit.py's residual-basis construction
        # defined, while preserving the no-residual baseline behavior.
        "residual_noll_radians": {str(j): 0.0 for j in range(5, 12)},
        "sigma_ccd_px": 0.0,
        "bias_counts": 0.0,
        "ramp_matrix": ramp,
        "grid_size": args.grid,
        "fft_size": 1024,
        "pixel_samples": int(fresh.get("pixel_samples") or 3),
        "registration_shift_ccd_px": None,
    }

    def build(
        defocus: float,
        center: list[float],
        width: list[float],
        residual: dict | None = None,
    ) -> dict:
        return {
            "defocus_diopter": defocus,
            "origin": fresh["origin"],
            "amplitude_center": center,
            "amplitude_width": width,
            # absolute-intensity block from the 2026-11-18 fit
            "gain": fresh["gain"],
            "background": fresh["background"],
            **common,
            "residual_noll_radians": residual or {},
        }

    variants = {
        # full fresh optical block (reference: already measured at 0.469/0.368/0.400)
        "hybrid_fresh_all": build(
            fresh["defocus_diopter"],
            fresh["amplitude_center"],
            fresh["amplitude_width"],
        ),
        # old defocus + fresh envelope
        "hybrid_defocus_old": build(
            old["defocus_diopter"], fresh["amplitude_center"], fresh["amplitude_width"]
        ),
        # fresh defocus + old envelope
        "hybrid_envelope_old": build(
            fresh["defocus_diopter"], old["amplitude_center"], old["amplitude_width"]
        ),
        # old everything (baseline; equals M3_old apart from gain/background)
        "hybrid_old_all": build(
            old["defocus_diopter"], old["amplitude_center"], old["amplitude_width"]
        ),
        # old defocus + fresh width, old envelope centre
        "hybrid_width_fresh": build(
            old["defocus_diopter"], old["amplitude_center"], fresh["amplitude_width"]
        ),
        # old defocus + old width, fresh envelope centre
        "hybrid_center_fresh": build(
            old["defocus_diopter"], fresh["amplitude_center"], old["amplitude_width"]
        ),
        # direction-carrying block (old defocus + old envelope) plus the 2026-11-18
        # residual Zernike screen, which is the only block the 11-18 fit adds and the
        # 11-07 M3 checkpoint lacks. This is the candidate double-gate checkpoint.
        "hybrid_oldgeom_resid": build(
            old["defocus_diopter"],
            old["amplitude_center"],
            old["amplitude_width"],
            residual=fresh["residual_noll_radians"],
        ),
        "hybrid_center_fresh_resid": build(
            old["defocus_diopter"],
            fresh["amplitude_center"],
            old["amplitude_width"],
            residual=fresh["residual_noll_radians"],
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in variants.items():
        path = args.out / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{name}: defocus={payload['defocus_diopter']:.5f} "
              f"center={payload['amplitude_center']} width={payload['amplitude_width']}")


if __name__ == "__main__":
    main()
