"""Why did the second-moment term fail to move its own metric?

Measured result to explain: with `w_spot_moment` swept 0 -> 2, the validation moment gap
stayed at 0.036-0.041 while R2 fell monotonically. A term that cannot move the quantity it
optimises is either uninformative or mis-measured, and those need different fixes -- so the
metric is tested directly before any new loss term is built on top of it.

The test is a sensitivity check, not a correlation: deliberately perturb a prediction's spot
size and confirm the reported gap responds by the expected amount. A metric that is
insensitive to its own input is broken, and no amount of tuning the loss will help.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ml.zernike.losses import (  # noqa: E402
    second_moments,
    spot_moment_gap_term,
    roi_mask,
)
from ml.zernike.metrics import roi_shape_terms  # noqa: E402

GRID = 64


def gaussian_spot(centre=(32.0, 32.0), sigma: float = 3.0, grid: int = GRID) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32) + 0.5,
        torch.arange(grid, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.exp(-(((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2) / (2 * sigma**2)))[None, None]


def main() -> None:
    print("=" * 96)
    print("SENSITIVITY OF THE SPOT-SIZE METRIC: does it respond to a known change?")
    print("=" * 96)

    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", 0.375 * GRID, 4 / 3)
    target = gaussian_spot(sigma=3.0)

    print(f"\nROI mask: sum={float(mask.sum()):.0f} of {GRID*GRID} px "
          f"({100*float(mask.sum())/(GRID*GRID):.1f}%)")
    print(f"\n{'pred sigma':>11}{'var_r(pred)':>13}{'var_r(ref)':>12}"
          f"{'gap term':>11}{'expect':>10}")
    expected = None
    for sigma in (1.5, 2.0, 3.0, 4.0, 6.0, 9.0):
        pred = gaussian_spot(sigma=sigma)
        v_p = float(second_moments(pred, mask)[2])
        v_t = float(second_moments(target, mask)[2])
        gap = float(spot_moment_gap_term(pred, target, mask).mean())
        if expected is None and abs(sigma - 3.0) < 1e-9:
            expected = gap
        # analytic relative error on the radial variance, which is sigma^2
        exp_rel = abs(sigma**2 - 3.0**2) / 3.0**2
        print(f"{sigma:>11.1f}{v_p:>13.2f}{v_t:>12.2f}{gap:>11.4f}{exp_rel:>10.4f}")

    print("\n1. does the gap peak at zero error? (it must, or the anchor is wrong)")
    gaps = [
        float(spot_moment_gap_term(gaussian_spot(sigma=s), target, mask).mean())
        for s in (1.0, 2.0, 3.0, 5.0, 8.0)
    ]
    print(f"   sigmas 1/2/3/5/8 -> gaps {[round(g, 4) for g in gaps]}")
    print(f"   minimum at sigma={[1.0, 2.0, 3.0, 5.0, 8.0][int(np.argmin(gaps))]} "
          f"(expected 3.0)")

    print("\n2. is the gradient of the gap non-trivial? (a dead term cannot be optimised)")
    pred = gaussian_spot(sigma=5.0).clone().requires_grad_(True)
    spot_moment_gap_term(pred, target, mask).mean().backward()
    g = pred.grad
    print(f"   |grad| max={float(g.abs().max()):.3e} mean={float(g.abs().mean()):.3e} "
          f"nonzero={int((g != 0).sum())}/{g.numel()}")
    rel = float(g.abs().max()) / max(float(pred.abs().max()), 1e-12)
    print(f"   |grad|max / |image|max = {rel:.3e}")

    print("\n3. scale of the term vs the fidelity term it competes with")
    mse = float(((gaussian_spot(sigma=5.0) - target) ** 2).mean())
    gap = float(spot_moment_gap_term(gaussian_spot(sigma=5.0), target, mask).mean())
    print(f"   mse={mse:.6f}   gap={gap:.4f}   ratio gap/mse={gap/max(mse,1e-12):.1f}")
    print("   a ratio >> 1 means an unweighted moment term would dominate the objective,")
    print("   which is consistent with R2 collapsing as w_spot_moment rose")

    print("\n4. does the ROI exclude the spot? (if the mask misses it, the term is noise)")
    inside = float((mask[0, 0] * target).sum() / target.sum())
    print(f"   fraction of the reference spot inside the ROI = {inside:.4f}")
    print("   a small value means the ROI and the spot do not overlap, so the gap is")
    print("   measured on background and carries little information")

    print("\nconclusion is stated by the numbers above; nothing is assumed here")


if __name__ == "__main__":
    main()
