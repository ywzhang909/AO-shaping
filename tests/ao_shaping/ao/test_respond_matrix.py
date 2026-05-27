import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.drivers.wfs import WFSManager

Zern_n = 3
Magnitude = 1.0


def test_respond_matrix():
    with (
        ZernikeSLM(slm_number=1, use_120hz=True, wavelength=1_064) as slm,
        WFSManager("768", high_speed=False) as wfs,
    ):
        assert Zern_n < slm.length
        
        x = np.zeros((slm.length,))
        slm.send_zernike(x)
        resp_array_0 = wfs.get_zernike()

        x[Zern_n] = Magnitude
        slm.send_zernike(x)
        resp_array_plus = wfs.get_zernike()

        x[Zern_n] = -Magnitude
        slm.send_zernike(x)
        resp_array_minus = wfs.get_zernike()

        indices = np.arange(len(resp_array_0))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(indices - width, resp_array_0, width, label="Zero input")
        plt.bar(indices, resp_array_plus, width, label="+Magnitude input")
        plt.bar(indices + width, resp_array_minus, width, label="-Magnitude input")
        plt.xlabel("Zernike Index")
        plt.ylabel("Response")
        plt.title(f"WFS Response for Zernike Mode {Zern_n}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        assert resp_array_plus != resp_array_0 and resp_array_minus != resp_array_0
