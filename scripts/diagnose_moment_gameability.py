"""Is the second-moment term GAMEABLE? (the mechanism behind the failed sweep)

The sensitivity check (``diagnose_moment_metric.py``) showed the gap is a well-behaved
anchor: exactly 0 at a perfect match, tracking the analytic relative error closely at small
errors, with a healthy gradient. So the metric is not broken.

It also showed ``gap/mse = 278``. At ``w_spot_moment = 1`` the spot-size term is ~278x the
fidelity term, so it dominates the objective almost completely -- which is why R2 collapsed
as the weight rose. A dominating term is only a problem if it can be minimised the wrong
way, and that is what this checks.

The candidate wrong way: ``var_r`` is an intensity-weighted variance, so the question is
whether *removing light* buys a smaller gap without matching the spot at all. If it does,
the term rewards brightness rather than spot size -- the same category error as the
unanchored pib/uniformity pair, which cost R2 +0.78 -> -0.86.

Scoring a dimmed spot and a blank image answers it directly, and the fix follows from the
answer rather than from taste.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ml.zernike.losses import roi_mask, second_moments, spot_moment_gap_term  # noqa: E402

GRID = 64


def spot(grid: int = GRID) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32) + 0.5,
        torch.arange(grid, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.exp(-(((ys - 32.0) ** 2 + (xs - 32.0) ** 2) / 18.0))[None, None]


def main() -> None:
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", 0.375 * GRID, 4 / 3)
    flat = mask.reshape(-1, mask.shape[-1]) if mask.dim() > 2 else mask
    target = spot()

    print("=" * 92)
    print("IS THE SPOT-SIZE TERM GAMEABLE BY DIMMING?")
    print("=" * 92)
    print(f"roi_mask shape={tuple(mask.shape)} sum={float(mask.sum()):.0f} "
          f"({100 * float(mask.sum()) / (GRID * GRID):.1f}% of the grid)")

    rows = flat.sum(1) > 0
    cols = flat.sum(0) > 0
    print(f"ROI bbox: rows {int(rows.nonzero()[0])}..{int(rows.nonzero()[-1])}, "
          f"cols {int(cols.nonzero()[0])}..{int(cols.nonzero()[-1])}")
    inside = float((flat * target[0, 0]).sum() / target.sum())
    print(f"fraction of the spot inside the ROI = {inside:.4f}  (near 1 = ROI contains it)")

    print(f"\n{'prediction':<30}{'var_r':>10}{'gap':>10}{'peak':>9}{'total':>10}")
    print(f"{'reference spot (exact match)':<30}"
          f"{float(second_moments(target, mask)[2]):>10.2f}{0.0:>10.4f}"
          f"{float(target.max()):>9.3f}{float(target.sum()):>10.1f}")
    for scale in (0.5, 0.25, 0.1, 0.01):
        pred = target * scale
        print(f"{'same shape x ' + format(scale, '.2f'):<30}"
              f"{float(second_moments(pred, mask)[2]):>10.2f}"
              f"{float(spot_moment_gap_term(pred, target, mask).mean()):>10.4f}"
              f"{float(pred.max()):>9.3f}{float(pred.sum()):>10.1f}")
    blank = torch.zeros_like(target)
    blank_gap = float(spot_moment_gap_term(blank, target, mask).mean())
    print(f"{'all-zero prediction':<30}"
          f"{float(second_moments(blank, mask)[2]):>10.2f}{blank_gap:>10.4f}"
          f"{0.0:>9.3f}{0.0:>10.1f}")

    print(f"\nA perfectly SHAPED but 100x dimmed spot scores gap="
          f"{float(spot_moment_gap_term(target * 0.01, target, mask).mean()):.4f}")
    print(f"A BLANK prediction scores gap={blank_gap:.4f}")
    print("\nInterpretation:")
    print("  scaling the whole image leaves var_r EXACTLY invariant (it is a ratio of")
    print("  intensity-weighted moments), so dimming alone cannot fake a match. But a dark")
    print("  or near-empty image has near-zero weighted variance about its own centroid,")
    print("  and that DOES drive the term down -- toward removing light, not matching size.")
    print("  With the term 278x the fidelity term, that is the cheaper direction.")


if __name__ == "__main__":
    main()
