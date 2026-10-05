"""Which simulator padding makes the trained model and the independent sim comparable?

The inverse panel was comparing a model fitted at ``far_field_padding=10`` against a
simulator at padding 1. Padding sets the angular scale of the centre-cropped output, so
the two cannot be compared pixel-for-pixel unless they agree. Rather than assume, this
scores the agreement at every padding and reports which one the data prefers.

Agreement is measured two ways, because they fail differently:
  * Pearson correlation -- insensitive to a linear scale factor, so it isolates SHAPE.
  * Spearman -- insensitive to any monotone rescale, so it isolates RANKING.
plus the second moment ratio, which is the physical observable the shaping cares about.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import second_moments  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from generate_inverse_design_report import _norm, load_trained  # noqa: E402

CKPT = ROOT / "logs" / "zernike_amp_final_nmax20" / "best_coefficients.pt"


def main() -> None:
    model, blob = load_trained(CKPT)
    print(f"checkpoint: n_max={blob['n_max']} padding={blob['far_field_padding']} "
          f"grid={blob['grid']}")
    designed = inv.gs_phase(inv.SIZE_FRAC, inv.ASPECT)

    # What the model predicts for that phase, at the model's own padding.
    with torch.no_grad():
        phasor = torch.polar(torch.ones(1, 1, inv.GRID, inv.GRID),
                             torch.as_tensor(designed, dtype=torch.float32)[None, None])
        phasor = phasor * inv.inscribed_mask(inv.GRID)
        pred = _norm(model(phasor.real, phasor.imag)[0, 0].numpy())

    var_p = float(second_moments(torch.as_tensor(pred)[None, None])[2])
    print(f"\npred: var_r={var_p:.2f}")
    print(f"{'sim pad':>9}{'pearson':>10}{'spearman':>10}{'var_r sim':>11}"
          f"{'var ratio':>11}{'score(pred)':>13}{'score(sim)':>12}")

    best = None
    for pad in (1, 2, 4, 8, 10, 12, 16):
        true = _norm(inv.sim_far_field(designed, pad))
        var_t = float(second_moments(torch.as_tensor(true)[None, None])[2])
        pear = inv.pearson(pred, true)
        spear = inv.spearman(pred.ravel(), true.ravel())
        # scores need the same padding for the ROI terms to sit on the same pixels
        s_pred = inv.score_phase(designed, inv.SIZE_FRAC, inv.ASPECT, pad)
        s_true = inv.score_phase(designed, inv.SIZE_FRAC, inv.ASPECT, pad)
        print(f"{pad:>9}{pear:>10.4f}{spear:>10.4f}{var_t:>11.2f}"
              f"{var_p / var_t:>11.3f}{s_pred:>13.4f}{s_true:>12.4f}")
        if best is None or pear > best[1]:
            best = (pad, pear, var_t)

    print(f"\nbest padding by shape agreement: {best[0]} (pearson {best[1]:.4f}, "
          f"var_r {best[2]:.2f} vs pred {var_p:.2f})")
    print(
        "The model is trained against real CCD frames whose spots are broad; the sim's "
        "focus is far tighter. If var_r(model) >> var_r(sim) at every padding, the "
        "disagreement is a MODEL limitation, not a scale mismatch."
    )


if __name__ == "__main__":
    main()
