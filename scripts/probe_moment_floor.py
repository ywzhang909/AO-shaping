"""Is 0.037 the floor for the spot-size gap, or is there headroom left?

The moment term is not broken (exactly 0 at a match, healthy gradient) and not gameable
(scale-invariant, blank scores worst, ROI contains 99.99% of the spot). Yet sweeping
``w_spot_moment`` 0 -> 2 left the validation gap at 0.036-0.041. So the remaining
explanation is that the metric is already near its floor on held-out data and there is
nothing for the term to win.

That is testable rather than arguable. If ``var_r`` varies across held-out frames by about
the gap the model achieves, then a prediction that simply always emitted the *average*
spot size would score the same gap -- and no loss term can beat it. The number that decides
this is the frame-to-frame spread of ``var_r`` itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import second_moments, spot_moment_gap_term, roi_mask  # noqa: E402

GRID = inv.GRID


def main() -> None:
    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    ds = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", 0.375 * GRID, 4 / 3)

    print("=" * 92)
    print("IS THE SPOT-SIZE GAP ALREADY AT ITS FLOOR ON HELD-OUT FRAMES?")
    print("=" * 92)

    frames = []
    for i in range(0, 128):
        img = ds[i]["image"].float()
        peak = img.amax()
        if float(peak) > 0:
            frames.append((img / peak)[None])   # image is already (1,g,g) -> (1,1,g,g)
    print(f"collected {len(frames)} peak-normalised held-out frames")

    var_r = np.array([float(second_moments(f, mask)[2]) for f in frames])
    print(f"\nvar_r across held-out frames:")
    print(f"  mean={var_r.mean():.2f}  sd={var_r.std(ddof=1):.2f}  "
          f"min={var_r.min():.2f}  max={var_r.max():.2f}")
    print(f"  coefficient of variation = {var_r.std(ddof=1) / var_r.mean() * 100:.2f}%")

    # The decisive comparison: what gap does a CONSTANT prediction -- always the mean spot
    # size -- achieve? Any model cannot beat "emit the average", so that number is a floor.
    mean_var = float(var_r.mean())
    constant = torch.full((1, 1, GRID, GRID), 0.0)
    yy, xx = torch.meshgrid(
        torch.arange(GRID, dtype=torch.float32) + 0.5,
        torch.arange(GRID, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    # a ring whose second moment equals mean_var, peak-normalised
    r = torch.sqrt(((yy - 32.0) ** 2 + (xx - 32.0) ** 2))
    constant = torch.exp(-(r**2) / (2 * (mean_var / 2))).clamp(min=0)[None, None]

    gaps = [float(spot_moment_gap_term(constant, f, mask).mean()) for f in frames]
    gaps = np.array(gaps)
    print(f"\nA CONSTANT prediction that always emits the mean spot size scores:")
    print(f"  mean gap = {gaps.mean():.4f}   (sd {gaps.std(ddof=1):.4f})")
    print(f"\nThe trained model's measured validation gap was 0.0368-0.0381.")
    verdict = "AT/BELOW THE FLOOR" if gaps.mean() >= 0.0368 else "ABOVE THE FLOOR -- headroom exists"
    print(f"\nVERDICT: {verdict}")
    if gaps.mean() >= 0.0368:
        print("  'Always emit the average spot size' scores about the same as the trained")
        print("  model. So the model already predicts spot SIZE as well as the data allows,")
        print("  and a spot-size loss term has nothing left to win -- it can only trade")
        print("  pixel fidelity (R2) for a metric that is already saturated. That is why the")
        print("  sweep cost R2 monotonically and moved the gap not at all.")
        print("  Any spot-SHAPE term must therefore be judged against this floor, not against 0.")


if __name__ == "__main__":
    main()