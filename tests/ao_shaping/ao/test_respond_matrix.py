import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.drivers.wfs import ThorlabWFS

magnitude = 1.0

def measure_zern_once(slm, wfs, n):
    x = np.zeros((slm.n_modes,))
    slm.send_zernike(x)
    resp_array_0 = wfs.get_zernike()

    x[n] = magnitude
    slm.send_zernike(x)
    resp_array_plus = wfs.get_zernike()

    x[n] = -magnitude
    slm.send_zernike(x)
    resp_array_minus = wfs.get_zernike()
    
    return resp_array_0, resp_array_plus, resp_array_minus

def measure_dev_once(slm, wfs:ThorlabWFS, n):
    def dev_to_array():
        dx, dy = wfs.get_spot_deviation(cancel_tile=True)
        return np.concat([dx.flatten(), dy.flatten()])
    
    x = np.zeros((slm.n_modes,))
    slm.send_zernike(x)
    resp_array_0 = dev_to_array()

    x[n] = magnitude
    slm.send_zernike(x)
    resp_array_plus = dev_to_array()

    x[n] = -magnitude
    slm.send_zernike(x)
    resp_array_minus = dev_to_array()
    
    return resp_array_0, resp_array_plus, resp_array_minus


def test_respond_dev_array(zern_n = 3):
    with (
        ZernikeSLM(slm_number=1, use_120hz=True, wavelength=532) as slm,
        ThorlabWFS("512", high_speed=False) as wfs,
    ):
        assert zern_n < slm.n_modes
        
        x = np.zeros((slm.n_modes,))
        slm.send_zernike(x)
        wfs.save_user_ref()
        wfs.set_ref_plane(custom = True)
        resp_array_0 ,resp_array_plus, resp_array_minus = measure_dev_once(slm, wfs, zern_n)
        
        wfs.set_ref_plane(custom = False)
        x = np.zeros((slm.n_modes,))
        slm.send_zernike(x)
        resp_array,_,_ = measure_dev_once(slm, wfs, zern_n)

        # Plot the three response arrays
        n_modes = len(resp_array_0)
        ind = np.arange(n_modes)  # the x locations for the groups
        width = 0.25  # the width of the bars

        fig, [ax1, ax2] = plt.subplots(2,1)
        rects1 = ax1.bar(ind - width, resp_array_0, width, label="Zero")
        rects2 = ax1.bar(ind, resp_array_plus, width, label=f"+{magnitude}")
        rects3 = ax1.bar(ind + width, resp_array_minus, width, label=f"-{magnitude}")

        ax1.set_ylabel("Zernike Coefficient")
        ax1.set_title("Zernike Response Matrix Test")
        ax1.set_xticks(ind)
        ax1.set_xticklabels([f"Z{i+3}" for i in range(n_modes)])
        ax1.legend()

        ax2.bar(ind, resp_array, width*3)
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"test_deviation_array_{zern_n}_plot.png")
        plt.close()


def test_respond_zern_array(zern_n = 3):
    with (
        ZernikeSLM(slm_number=1, use_120hz=True, wavelength=532) as slm,
        ThorlabWFS("512", high_speed=False) as wfs,
    ):
        assert zern_n < slm.n_modes
        
        x = np.zeros((slm.n_modes,))
        slm.send_zernike(x)
        wfs.save_user_ref()
        wfs.set_ref_plane(custom = True)
        resp_array_0 ,resp_array_plus, resp_array_minus = measure_zern_once(slm, wfs, zern_n)
        
        wfs.set_ref_plane(custom = False)
        x = np.zeros((slm.n_modes,))
        slm.send_zernike(x)
        resp_array = wfs.get_zernike()[3:]

        # Plot the three response arrays
        n_modes = len(resp_array_0)
        ind = np.arange(n_modes)  # the x locations for the groups
        width = 0.25  # the width of the bars

        fig, [ax1, ax2] = plt.subplots(2,1)
        rects1 = ax1.bar(ind - width, resp_array_0, width, label="Zero")
        rects2 = ax1.bar(ind, resp_array_plus, width, label=f"+{magnitude}")
        rects3 = ax1.bar(ind + width, resp_array_minus, width, label=f"-{magnitude}")

        ax1.set_ylabel("Zernike Coefficient")
        ax1.set_title("Zernike Response Matrix Test")
        ax1.set_xticks(ind)
        ax1.set_xticklabels([f"Z{i+3}" for i in range(n_modes)])
        ax1.legend()

        ax2.bar(ind, resp_array, width*3)
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"test_zernike_array_{zern_n}_plot.png")
        plt.close()

def test_respond_matrix(n_max:int = 4):
    with (
        ZernikeSLM(slm_number=1, use_120hz=True, wavelength=532, n_max=n_max) as slm,
        ThorlabWFS("512", high_speed=False) as wfs,
    ):
        
        # init
        x = np.zeros((slm.n_modes,))
        slm.send_zernike(x)
        
        init_zer,_,_ = measure_dev_once(slm, wfs, 0)
        
        wfs.save_user_ref()
        wfs.set_ref_plane(custom = True)
        
        respond_matrix = np.zeros((slm.n_modes-3, len(init_zer)))
        for n in range(3, slm.n_modes):
            resp_array_0, resp_array_plus, resp_array_minus = measure_dev_once(slm, wfs, n)
            resp_array = resp_array_plus - resp_array_minus
            respond_matrix[n-3, :] = resp_array / (2*magnitude)
            
        plt.imshow(respond_matrix, aspect='equal')
        plt.tight_layout()
        plt.savefig("test_respond_matrix_plot.png")
        plt.close()