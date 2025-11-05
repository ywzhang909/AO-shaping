from ao_shaping.drivers import NlightDM
import numpy as np

def test_dm():
    with NlightDM() as dm:
        voltages = np.zeros((dm.DM_Num))
        for i in range(10):
            v = np.sin(2 * np.pi * i / 1_000_000) * 100
            voltages[1] = v
            dm.send_voltages(voltages, 0)
        
def test_turn_off_dm():
    with NlightDM(keep_when_exit=False) as dm:
        dm.send_voltages(np.zeros((dm.DM_Num)))
        dm.reset_all()