"""Unit-check the ellipse term against cases the radial second moment cannot distinguish."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))

import math  # noqa: E402

import torch  # noqa: E402

from ml.zernike.losses import ellipse_gap_term, ellipse_parameters, roi_mask, spot_moment_gap_term  # noqa: E402

G = 64


def spot(cx=32.0, cy=32.0, sx=3.0, sy=5.0, theta=0.0):
    ys, xs = torch.meshgrid(
        torch.arange(G, dtype=torch.float32) + 0.5,
        torch.arange(G, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x, y = xs - cx, ys - cy
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = x * c + y * s, -x * s + y * c
    return torch.exp(-(xr**2 / (2 * sx**2) + yr**2 / (2 * sy**2)))[None, None]


mask = roi_mask((G, G), (G / 2, G / 2), "rectangle", 0.375 * G, 4 / 3)
ref = spot()

print("=" * 88)
print("ELLIPSE TERM: does it see what the radial second moment cannot?")
print("=" * 88)

cases = {
    "exact match": spot(),
    "shifted 2 px": spot(cx=34.0),
    "same spread, rotated 45deg": spot(theta=0.7854),
    "var_x <-> var_y swapped": spot(sx=5.0, sy=3.0),
    "10% wider": spot(sx=3.3),
    "blank": torch.zeros_like(ref),
}
print(f"{'case':<30}{'ellipse gap':>14}{'radial moment gap':>20}")
for name, img in cases.items():
    e = float(ellipse_gap_term(img, ref, mask)["_mean_ellipse"])
    m = float(spot_moment_gap_term(img, ref, mask).mean())
    print(f"{name:<30}{e:>14.4f}{m:>20.4f}")

c = ellipse_parameters(ref, mask)
print("\nrecovered parameters of a sigma=(3,5) spot centred at (32,32):")
print(f"  centroid  = ({float(c[0][0]):.3f}, {float(c[1][0]):.3f})   want (32, 32)")
print(f"  var_x     = {float(c[2][0]):.3f}   want ~9  (sigma^2)")
print(f"  var_y     = {float(c[3][0]):.3f}   want ~25 (sigma^2)")
c45 = ellipse_parameters(spot(theta=0.7854), mask)
print(f"  cov at 45deg = {float(c45[4][0]):.3f}   want 0 (a symmetric ellipse has none)")
c30 = ellipse_parameters(spot(theta=0.5236), mask)
print(f"  cov at 30deg = {float(c30[4][0]):.3f}   nonzero -- tilt is visible")

print("\nreading: the rotated case is the point. A 45-degree rotation leaves var_x+var_y")
print("unchanged, so the radial moment term scores it 0 while the ellipse term does not.")
print("That is the extra information this term carries.")