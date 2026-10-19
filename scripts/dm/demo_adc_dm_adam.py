#!/usr/bin/env python3
"""ADC-DM Adam optimization demo (extracted from
``ao_shaping.optimizer.wfless.adc_dm_adam`` — see issue #90).

Runs one Adam optimization pass on the NLight DM with the NI-DAQ feedback
path, plots the history, and persists the last voltages (requires hardware:
NlightDM + nidaqmx).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.optimizer.wfless.adc_dm_adam import (
    Last_voltage_path,
    R,
    ROOT_DIR,
    SAVED,
    gen_file_name,
    optimizer,
)


def main() -> None:
    import time

    for sample_num in range(1):
        dfhistory = optimizer(R)

        disp_grid = (1, 2)
        fig, ax = plt.subplots(*disp_grid)
        ax[0].plot(dfhistory["J"].to_list())
        ax[1].imshow(
            np.stack(dfhistory["_v"].to_list()).transpose(),
            interpolation="nearest",
            aspect="auto",
        )

        if SAVED:
            saved_dir = gen_file_name(ROOT_DIR)
            plt.savefig(gen_file_name(saved_dir, "png"))
            dfhistory.to_pickle(gen_file_name(saved_dir, "pkl"), compression="zip")

        else:
            plt.show()
        plt.close("all")
        last_voltage: np.ndarray = dfhistory.iloc[-1]["_v"]

        last_J = dfhistory.iloc[-1]["J"]
        print(f"{last_J=}")
        np.savetxt(Last_voltage_path, np.round(last_voltage), fmt="%d")
        time.sleep(2)


if __name__ == "__main__":
    main()
