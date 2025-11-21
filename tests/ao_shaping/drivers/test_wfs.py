import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.drivers import Thorlab_WFS, MlaRes

def test_wfs():
    with Thorlab_WFS(MlaRes.Res512, exp_time=0.029) as wfs:
        opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
        if 0.001 < opt_exp_time < 87:
            wfs.exposure_time = opt_exp_time
        else:
            print("no usable image. exit now..")
            exit()

        print(f"optimize_pupil: {wfs.optimize_pupil()}")

        for _ in range(1):
            wfs.take_image()
            
            spots_filed = wfs.get_spotfiled_image()
            plt.imshow(spots_filed)
            plt.show()
            
            x, y = wfs.get_spot_deviation()
            intensity, _ = wfs.get_spots_statics()
            wf, statics = wfs.get_wavefront()
            print(f"{statics=}")

            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(x)
            ax[0,0].set_title("spot deviation x")

            ax[0,1].imshow(y)
            ax[0,1].set_title("spot deviation y")
            
            ax[1,0].imshow(intensity)
            ax[1,0].set_title("spot intensity")
            
            ax[1,1].imshow(wf)
            ax[1,1].set_title("wavefront")
            plt.show()
            
        wfs.high_speed = True
        for _ in range(10):
            wfs.take_image()
            x,y = wfs.get_spot_deviation()
            print(y[0,:])
            # print(wfs.get_zernike(3))
            print(wfs.get_wavefront()[0][0,:])
            
def test_rms():
    rms_hist = []
    with Thorlab_WFS(MlaRes.Res768) as wfs:
        wfs.high_speed = True
        for _ in range(100):
            wfs.take_image(n_sample=10)
            
            # dx, dy = wfs.get_spot_deviation()
            # rms = np.sqrt(np.nanmean(dx**2+dy**2))
            zernike_coeff = wfs.get_zernike(10)
            print(zernike_coeff)
            rms_hist.append(np.mean(np.sqrt(np.sum(zernike_coeff**2))))
    return rms_hist

def test_high_speed_d():
    rms_hist = []
    with Thorlab_WFS(MlaRes.Res768) as wfs:
        last_dx, last_dy = None, None
        for _ in range(100):
            wfs.take_image(n_sample=10)
            
            dx, dy = wfs.get_spot_deviation()
            assert last_dx != dx and last_dy != dy

            last_dx, last_dy = dx, dy

            