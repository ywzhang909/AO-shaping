"""独立调试脚本: slm-pib spgd, objective=rmse_out, target_shape=square, target_size=100.
后端: Santec SLM + Daheng CCD (真实硬件).

对应 CLI:
    python src/ao_shaping/main.py slm-pib spgd --show --debug -c max \
        --auto-exposure --objective rmse_out --target_shape square \
        --target_size 100 --delta 0.01 --lr 0.5

本脚本**不解析参数、不做自动探测** (auto-exposure / center probe 全部跳过),
所有数值硬编码。设备后端为真实 Santec SLM + Daheng CCD。
"""

from __future__ import annotations

import numpy as np

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)

# ---------------------------------------------------------------------------
# 硬编码参数 (与上面 CLI 命令一一对应)
# ---------------------------------------------------------------------------
EPOCHS = 50
N_MAX = 4
CENTER = "max"                       # -c max: argmax 定 0-order, 无需 probe
TARGET_SHAPE = "square"              # --target_shape square
TARGET_SIZE = 100.0                  # --target_size 100
DELTA = 0.01                         # --delta 0.01
LR = 0.5                             # --lr 0.5
OBJECTIVE = "rmse_out"               # --objective rmse_out
CAM_TYPE = "daheng"                  # 真实大恒 CCD
CAM_ID = 0                           # Far_Cam_ID
CAM_SIZE = 300
# 真实硬件需要固定曝光; 0 会被 Daheng 驱动钳到设备最小值 (~0.02 ms)。
# 按实际 bench 调节 (例如 0.5~5 ms), 这里给一个保守默认。
EXPOSURE_MS = 1.0
TARGET_MAX_BRIGHTNESS = 40           # 仅 auto-exposure 路径用, 此处不自动调曝光

# ---------------------------------------------------------------------------
# 组装 config (纯 dataclass, 无 Click)
# ---------------------------------------------------------------------------
camera = CameraParamsPib(
    target=ObjectiveTarget(name=OBJECTIVE, target_shape=TARGET_SHAPE),
    cam_type=CAM_TYPE,
    cam_id=CAM_ID,
    cam_size=CAM_SIZE,
    exposure_time_ms=EXPOSURE_MS,
    center=CENTER,
    target_size=TARGET_SIZE,
    target_max_brightness=TARGET_MAX_BRIGHTNESS,
)

slm = SlmParamsPib(
    n_max=N_MAX,
    slm_number=1,
    slm_wavelength=1064,             # nm; 0 = 问设备
    zernike_radius=600.0,
)

config = SlmZernikePibConfig(
    center=CENTER,
    epochs=EPOCHS,
    algorithm="spgd",
    delta=DELTA,
    lr=LR,
    optimizer_type="adamod",
    camera=camera,
    slm=slm,
    show=False,          # notebook 里显示交给自定义渲染, 这里关掉 pygame
    record_phase=True,
)

print("config:", config)
recorder = optimize_slm_zernike_pib(config)

# ---------------------------------------------------------------------------
# 简单查看结果
# ---------------------------------------------------------------------------
import pandas as pd

df = recorder.dataframe
print(df.head())
print(df.tail())

best = df.iloc[df["rmse_out"].idxmin()] if "rmse_out" in df else df.iloc[-1]
print("\nbest row:")
print(best)
print("\nbest coeffs:", best["_c"])