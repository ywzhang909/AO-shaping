import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.optimizer.wf.rms import optimizer_rms
from ao_shaping.utils.display import plot_funcs

def test_optimize():
    cam_id = os.environ['Far_Cam_ID']
    exposure_time_ms = 70
    epochs = 8_000
    r_bucket = 0
    cam_size = 160
    rms_threshold = 0.12
    pupil_dia = 2.7
    
    init_V = [0 for _ in range(64)]
    wf_records = optimizer_rms(
        init_v=init_V,
        pupil_diameter=pupil_dia,
        wfs_res='768',
        early_stop_threshold=rms_threshold,
        epochs=20_000)

    min_iter, (min_epoch, min_rms) = wf_records.get_best_iter()
    init_v = min_iter["_v"]
    init_wf, min_wf = wf_records.first["_wavefront"][0], min_iter["_wavefront"][0]

    dm_available = np.ones(64, dtype=bool)
    dm_available[0] = False
    dm_available[21:] = False

    ccd_records = optimize_pib(
        cam_id=cam_id, center=None, exposure_time_ms=exposure_time_ms, cam_size=cam_size,
        dm_unit_mask=dm_available,
        epochs=epochs, r_bucket=r_bucket, lr=0.9, delta=0.9, shrink_iter=20, shrink_ratio=0.8,
        init_v=init_v, show=False)
    max_pid_iter, (max_epoch, max_pib) = ccd_records.get_best_iter()
    last_V = max_pid_iter["_v"]

    fig, ax = plt.subplots(2, 4, figsize=(12, 8))
    # init image
    img_plot_func = plot_funcs['img']
    img_plot_func(ccd_records.first["_img"], ax[0, 0], f"Init Image, pib={ccd_records.first['pib']:.3f}")
    # best image
    img_plot_func(max_pid_iter["_img"], ax[1, 0], f"Best Image, pib={max_pib:.3f}")
    
    # rms history
    ax[0, 1].plot(wf_records.get_sublist())
    ax[0, 1].scatter(min_epoch, min_rms, color='r', marker='*', label='Min RMS')
    ax[0, 1].text(min_epoch, min_rms, f"{min_rms:.4f}", color='r')
    ax[0, 1].set_title("RMS History")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("RMS")
    # pib history
    ax[1, 1].plot(ccd_records.get_sublist()[1:])
    ax[1, 1].scatter(max_epoch, max_pib, color='r', marker='*', label='Max PIB')
    ax[1, 1].text(max_epoch, max_pib, f"{max_pib:.4f}", color='r')
    ax[1, 1].set_title("PIB History")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("PIB")
    
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
    ax[0, 3].bar(range(64), init_v, color='r')
    ax[0, 3].bar(range(64), last_V, color='b')
    ax[0, 3].set_title("Best Voltages")
    ax[0, 3].set_xlabel("Unit ID")
    ax[0, 3].set_ylabel("Voltage")
    # voltage history
    voltages = np.array(wf_records.get_sublist("_v")+ccd_records.get_sublist("_v")[1:])
    ax[1, 3].imshow(voltages.T, aspect='auto')
    ax[1, 3].set_title("Voltage History")
    ax[1, 3].set_xlabel("Epoch")
    ax[1, 3].set_ylabel("Voltage")
        
    plt.tight_layout()
    plt.show()