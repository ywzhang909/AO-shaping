import numpy as np
import pickle

# Load the original best_far_field.png
from PIL import Image

# Check the original best_far_field.png
orig = np.array(
    Image.open("data/debug/slm_gs_refine/20261015_211345/best_far_field.png")
)
print("best_far_field.png shape:", orig.shape, "dtype:", orig.dtype)

# Check if it's RGBA
print(
    "Channel 0 (R) unique values (sample):", np.unique(orig[200:300, 300:400, 0])[:20]
)
print("Channel 0 (R) range:", orig[:, :, 0].min(), orig[:, :, 0].max())

# Crop the center region (avoid white border)
# The image is 985x946. Let's find the non-white region.
gray = orig[:, :, :3].mean(axis=2)
# Find rows/cols that are not all white (>250)
non_white_rows = np.where(gray.mean(axis=1) < 240)[0]
non_white_cols = np.where(gray.mean(axis=0) < 240)[0]
if len(non_white_rows) > 0:
    rmin, rmax = non_white_rows.min(), non_white_rows.max()
else:
    rmin, rmax = 0, gray.shape[0]
if len(non_white_cols) > 0:
    cmin, cmax = non_white_cols.min(), non_white_cols.max()
else:
    cmin, cmax = 0, gray.shape[1]
print(f"\nData region: rows {rmin}-{rmax}, cols {cmin}-{cmax}")
print(f"Data region size: {rmax - rmin + 1} x {cmax - cmin + 1}")

# Within the data region, find the beam center
data_region = gray[rmin : rmax + 1, cmin : cmax + 1]
print(f"Data region min/max: {data_region.min():.1f} / {data_region.max():.1f}")

# The inferno colormap: dark (low intensity) = low RGB average, bright (high intensity) = high RGB average
# But white border is also high. Let's find the peak in the data region.
peak_y, peak_x = np.unravel_index(np.argmax(data_region), data_region.shape)
print(
    f"Peak in data region: local ({peak_y}, {peak_x}) -> global ({rmin + peak_y}, {cmin + peak_x}), value: {data_region[peak_y, peak_x]:.1f}"
)

# Check cross-section through the peak in the data region
print(f"\nX cross-section through peak (y={rmin + peak_y}):")
row = gray[rmin + peak_y, cmin : cmax + 1]
for i in range(0, len(row), max(1, len(row) // 20)):
    bar = "#" * int((row[i] - row.min()) / (row.max() - row.min() + 1e-9) * 50)
    print(f"  x={cmin + i:4d}: {row[i]:8.1f} {bar}")

# Check the pkl _img
with open(
    "data/debug/slm_gs_refine/20261015_211345/slm_gs_refine_20261015_211345.pkl", "rb"
) as f:
    pkl = pickle.load(f)
print("\nPkl record 1 (gs stage):")
rec = pkl[1]
print(f"  cv: {rec['cv']}")
print(
    f"  _img shape: {rec['_img'].shape}, dtype: {rec['_img'].dtype}, min: {rec['_img'].min()}, max: {rec['_img'].max()}"
)
print(f"  _img unique: {np.unique(rec['_img'])}")
