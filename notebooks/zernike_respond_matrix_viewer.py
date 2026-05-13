# %%
from itertools import cycle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analyze_debug_structure import load_debug_data

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

base_path = Path("../data/zernike_response_matrix")
raw_data, info = load_debug_data(base_path)

# %%
assert raw_data

n_mode = len(raw_data.keys())
n_dev = raw_data[0][0]['plus']['dev_x'][0].shape
n_dev = n_dev[0]*n_dev[1]

n_zernike = raw_data[0][0]['plus']['zernike'][0].shape[0]

deviation_matrix = np.zeros((n_mode, 2*n_dev))
zernike_matrix = np.zeros((n_mode, n_zernike))

cycle_ = 0
dev_masks = np.ones_like(raw_data[0][0]['plus']['dev_x'][0], dtype=bool)

def calc_deviations_oneside(dev_samples):
    global dev_masks
    dev_masks = dev_masks & np.all([~np.isnan(x) for x in dev_samples], axis=0)
    _dev_samples = np.array([np.nan_to_num(x, nan=0.0, copy=False) for x in dev_samples])
    return np.mean(_dev_samples, axis=0).reshape(-1)

def calc_deviation(mode:int, xy:str):
    global raw_data
    mode_data = raw_data[mode][cycle_]
    diff_m = calc_deviations_oneside(mode_data['plus'][f'dev_{xy}']) - calc_deviations_oneside(mode_data['minus'][f'dev_{xy}'])
    return diff_m/2
# mode = 0
# mode_data = raw_data[mode]

for mode, mode_data in raw_data.items():
    # divations
    deviation_matrix[mode, :n_dev] = calc_deviation(mode, 'x')
    deviation_matrix[mode, n_dev:] = calc_deviation(mode, 'y')

    # zernike
    zernike_plus_samples = mode_data[cycle_]['plus']['zernike']
    zernike_minus_spamles = mode_data[cycle_]['minus']['zernike']
    zernike_matrix[mode, :] = np.mean(zernike_plus_samples, axis=0) - np.mean(zernike_minus_spamles, axis=0)

_, [ax1, ax2, ax3] =  plt.subplots(1, 3)

ax1.imshow(deviation_matrix, aspect='auto')
ax2.imshow(dev_masks)
ax3.imshow(zernike_matrix, aspect='auto')
# %%
