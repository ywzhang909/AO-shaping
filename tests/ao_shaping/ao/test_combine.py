import os
import pytest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from ao_shaping.wfless.DM_cam import optimize_pib
except ImportError:
    optimize_pib = None
from ao_shaping.wf.DM_wfs import optimizer_rms


@pytest.mark.skipif(optimize_pib is None, reason="DM_cam module not available")
def test_optimize():
    cam_id = os.environ["Far_Cam_ID"]
    exposure_time_ms = 500
    epochs = 4_000
    r_bucket = 0
    cam_size = 160
    rms_threshold = 0.17
    pupil_dia = 2.7

    init_V = [0 for _ in range(64)]
    wf_res_list = optimizer_rms(
        init_v=init_V,
        pupil_diameter=pupil_dia,
        wfs_res="768",
        early_stop_threshold=rms_threshold,
        epochs=20_000,
    )

    wf_df = pd.DataFrame(wf_res_list)
    min_id = wf_df["J"].argmin()
    min_iter = wf_df.iloc[min_id]
    init_v = min_iter["_v"]
    init_wf, min_wf = wf_df.iloc[1]["_wavefront"][0], min_iter["_wavefront"][0]

    dm_res_list = optimize_pib(
        cam_id=cam_id,
        center=None,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        epochs=epochs,
        r_bucket=r_bucket,
        lr=0.9,
        delta=0.9,
        shrink_iter=20,
        shrink_ratio=0.8,
        init_v=init_v,
        show=False,
    )
    res_df = pd.DataFrame(dm_res_list)
    max_j_id = res_df.iloc[1:]["pib"].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]["pib"]

    fig, ax = plt.subplots(2, 4, figsize=(12, 8))
    # init image
    ax[0, 0].imshow(res_df.iloc[0]["_img"])
    ax[0, 0].set_title(f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
    ax[0, 0].axis("off")
    # best image
    ax[1, 0].imshow(res_df.iloc[max_j_id]["_img"])
    ax[1, 0].set_title(f"Best Image, pib={max_j:.3f}")
    ax[1, 0].axis("off")
    # pib history
    ax[0, 1].plot(res_df.iloc[1:]["pib"])
    ax[0, 1].set_title("PIB History")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("PIB")
    # rms history
    ax[1, 1].plot(wf_df.iloc[1:]["J"])
    ax[1, 1].set_title("RMS History")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("RMS")

    # init wf
    ax[0, 2].imshow(init_wf)
    ax[0, 2].set_title("Init WF")
    ax[0, 2].set_xlabel("Pixel ID")
    ax[0, 2].set_ylabel("Amplitude")
    # best wf
    ax[1, 2].imshow(min_wf)
    ax[1, 2].set_title("Best WF")
    ax[1, 2].set_xlabel("Pixel ID")
    ax[1, 2].set_ylabel("Amplitude")
    # best voltages plot bar
    ax[0, 3].bar(range(64), init_v, color="r")
    ax[0, 3].bar(range(64), last_V, color="b")
    ax[0, 3].set_title("Best Voltages")
    ax[0, 3].set_xlabel("Unit ID")
    ax[0, 3].set_ylabel("Voltage")
    # voltage history
    ax[1, 3].imshow(np.array(res_df.iloc[1:]["_v"].tolist()).T)
    ax[1, 3].set_title("Voltage History")
    ax[1, 3].set_xlabel("Epoch")
    ax[1, 3].set_ylabel("Voltage")

    plt.tight_layout()
    plt.show()
