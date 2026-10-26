import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, json, os
from pathlib import Path
from PIL import Image
from matplotlib.animation import PillowWriter

# Set Chinese font
import matplotlib.font_manager as fm
font_path = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["font.sans-serif"] = [prop.get_name(), "SimHei", "SimSun"]
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("data/debug/slm_transfer/cold_start_20261026_1004")
FIG_DIR = Path("report/shape_trials/cold_start/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Test forward_intensity with model_phase
from ao_shaping.algorithm.signal_processing.square_shaping import SquareShapingConfig, forward_intensity

z = np.load(BASE / "random_no_abba" / "capture_0000.npz", allow_pickle=True)
model_phase = z["model_phase"]
print("model_phase shape:", model_phase.shape, "min:", model_phase.min(), "max:", model_phase.max())

# Try with n_grid=128
cfg = SquareShapingConfig(
    n_grid=128,
    aperture_size=128 * 8e-6,  # 8 um per pixel
    wavelength=1064e-9,
    focal_length=0.125,
    target_side_px=80,
    far_field_padding=3,
)
print("cfg created")
try:
    intensity = forward_intensity(model_phase, cfg)
    print("intensity shape:", intensity.shape, "min:", intensity.min(), "max:", intensity.max())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
" > /dev/null 2>&1
echo "Script written"