import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from ao_shaping.drivers import Thorlab_WFS, MlaRes


def test_get_image():
    with Thorlab_WFS(MlaRes.Res1024, exp_time=4) as wfs:
        opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
        if 0.001 < opt_exp_time < 87:
            wfs.exposure_time = opt_exp_time
        else:
            print("no usable image. exit now..")
            sys.exit()

        print(f"optimize_pupil: {wfs.optimize_pupil()}")
        wfs.take_image()
        # Create figure with multiple subplots
        image = wfs.get_spotfiled_image()
        plt.imshow(image, cmap='gray')
        plt.show()


def test_calc_wf():
    with Thorlab_WFS(MlaRes.Res1024, exp_time=4) as wfs:
        opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
        if 0.001 < opt_exp_time < 87:
            wfs.exposure_time = opt_exp_time
        else:
            print("no usable image. exit now..")
            sys.exit()

        print(f"optimize_pupil: {wfs.optimize_pupil()}")
        wfs.take_image()
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3)

        ax_spot = fig.add_subplot(gs[0, 0])
        ax_wf = fig.add_subplot(gs[0, 1])
        ax_dev_x = fig.add_subplot(gs[0, 2])
        ax_dev_y = fig.add_subplot(gs[0, 3])

        # Initialize images
        im_spot = ax_spot.imshow(wfs.get_spots_statics()[0], cmap='viridis')
        wf, statics = wfs.get_wavefront()
        x, y = wfs.get_spot_deviation()
        im_wf = ax_wf.imshow(wf, cmap='RdBu')
        im_dev_x = ax_dev_x.imshow(x, cmap='RdBu')
        im_dev_y = ax_dev_y.imshow(y, cmap='RdBu')

        ax_spot.set_title("Spot Field")
        ax_wf.set_title("Wavefront")
        ax_dev_x.set_title("Deviation X")
        ax_dev_y.set_title("Deviation X")

        # Add colorbars
        fig.colorbar(im_spot, ax=ax_spot, shrink=0.8)
        fig.colorbar(im_wf, ax=ax_wf, shrink=0.8)
        fig.colorbar(im_dev_y, ax=[ax_dev_x, ax_dev_y], shrink=0.8)

        plt.show()

def test_rms():
    rms_hist = []
    with Thorlab_WFS(MlaRes.Res1024, use_custom_ref=False) as wfs:
        for _ in range(10):
            wfs.take_image(n_sample=1)
            dx, dy = wfs.get_spot_deviation()
            rms = np.sqrt(np.nanmean(dx**2+dy**2))
            # x, y = wfs.get_spot_deviation()
            # zernike_coeff = wfs.get_zernike(10)
            # rms_hist.append(np.mean(np.sqrt(np.sum(x**2 + y**2))))
            rms_hist.append(rms)
    plt.plot(rms_hist)
    plt.show()

def test_zernike():
    rms_hist = []
    with Thorlab_WFS(MlaRes.Res1024, use_custom_ref=False) as wfs:
        for _ in range(10):
            zernike_coeff = wfs.get_zernike(10)
            rms_hist.append(np.mean(np.sqrt(np.sum(zernike_coeff**2))))
    fig, [ax1, ax2] = plt.subplots(2,1)
    ax1.plot(rms_hist)
    ax2.bar(np.arange(len(rms_hist[0])), rms_hist[0])
    plt.show()

def test_high_speed_d():
    rms_hist = []
    with Thorlab_WFS(MlaRes.Res768) as wfs:
        last_dx, last_dy = None, None
        for _ in range(100):
            wfs.take_image(n_sample=10)

            dx, dy = wfs.get_spot_deviation()
            assert last_dx != dx and last_dy != dy

            last_dx, last_dy = dx, dy

def test_spot_deviation():
    with Thorlab_WFS(MlaRes.Res512, pupil_diameter=2.5, high_speed=True) as wfs:
        # Create figure for animation
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 10))

        # Initialize with empty images
        im1 = ax1.imshow(np.zeros((100, 100)), cmap='RdBu', vmin=-5, vmax=5)
        im2 = ax2.imshow(np.zeros((100, 100)), cmap='RdBu', vmin=-5, vmax=5)
        ax1.set_title("X Deviation")
        ax2.set_title("Y Deviation")
        fig.tight_layout()

        def update(frame):
            wfs.take_image()
            x, y = wfs.get_spot_deviation()

            im1.set_data(x)
            im2.set_data(y)

            # Update colorbar limits dynamically
            vmax = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)), 1e-6)
            im1.set_clim(-vmax, vmax)
            im2.set_clim(-vmax, vmax)

            ax1.set_title(f"X Deviation (frame={frame})")
            ax2.set_title(f"Y Deviation (frame={frame})")

            return im1, im2

        ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
        plt.show()
