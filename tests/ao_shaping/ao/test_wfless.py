import os

import matplotlib.pyplot as plt
import pandas as pd

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.file import get_init_V_by_rms

def test_optimize_pib():

    cam_id = int(os.environ['Far_Cam_ID'])
    center = None
    exposure_time_ms = 400
    epochs = 4_000
    r_bucket = 10
    cam_size = 200
    init_v = []

    res_list = optimize_pib(
        cam_id=cam_id, center=center, exposure_time_ms=exposure_time_ms, cam_size=cam_size,
        epochs=epochs, r_bucket=r_bucket, init_v=init_v,)
    res_df = res_list.dataframe
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # init image
    ax[0, 0].imshow(res_df.iloc[0]["_img"])
    ax[0, 0].set_title(f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
    ax[0, 0].axis("off")
    # best image
    ax[0, 1].imshow(res_df.iloc[max_j_id]["_img"])
    ax[0, 1].set_title(f"Best Image, pib={max_j:.3f}")
    ax[0, 1].axis("off")
    # pib history
    ax[1, 0].plot(res_df["pib"])
    ax[1, 0].set_title("PIB History")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("PIB")
    # best voltages plot bar
    ax[1, 1].bar(range(64), last_V)
    ax[1, 1].set_title("Best Voltages")
    ax[1, 1].set_xlabel("Unit ID")
    ax[1, 1].set_ylabel("Voltage")
        
    plt.tight_layout()
    plt.show()