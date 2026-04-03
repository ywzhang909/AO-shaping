import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ao_shaping.drivers import Thorlab_WFS, MlaRes

def test_get_images():
    with Thorlab_WFS(MlaRes.Res512, exp_time=0.029) as wfs:
        opt_exp_time, _ = wfs.optimize_exposure_time_and_gain()
        if 0.001 < opt_exp_time < 87:
            wfs.exposure_time = opt_exp_time
        else:
            print("no usable image. exit now..")
            exit()

        print(f"optimize_pupil: {wfs.optimize_pupil()}")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax_spot = fig.add_subplot(gs[0, 0])
        ax_wf = fig.add_subplot(gs[0, 1])
        ax_dev = fig.add_subplot(gs[0, 2])
        ax_intensity = fig.add_subplot(gs[1, :])
        
        # Initialize images
        im_spot = ax_spot.imshow(np.zeros((100, 100)), cmap='viridis')
        im_wf = ax_wf.imshow(np.zeros((100, 100)), cmap='RdBu')
        im_dev = ax_dev.imshow(np.zeros((100, 100)), cmap='RdBu')
        
        ax_spot.set_title("Spot Field")
        ax_wf.set_title("Wavefront")
        ax_dev.set_title("Deviation (RMS)")
        ax_intensity.set_title("Intensity Profile")
        
        # Add colorbars
        fig.colorbar(im_spot, ax=ax_spot, shrink=0.8)
        fig.colorbar(im_wf, ax=ax_wf, shrink=0.8)
        fig.colorbar(im_dev, ax=ax_dev, shrink=0.8)

        def update(frame):
            wfs.take_image()
            
            spots_field = wfs.get_spotfiled_image()
            x, y = wfs.get_spot_deviation()
            intensity, _ = wfs.get_spots_statics()
            wf, statics = wfs.get_wavefront()
            
            # Update spot field
            im_spot.set_data(spots_field)
            vmax_spot = np.nanmax(spots_field)
            im_spot.set_clim(0, vmax_spot * 0.9 if vmax_spot > 0 else 1)
            
            # Update wavefront
            im_wf.set_data(wf)
            vmax_wf = np.nanmax(np.abs(wf))
            im_wf.set_clim(-vmax_wf, vmax_wf)
            
            # Update deviation (RMS map)
            deviation = np.sqrt(x**2 + y**2)
            im_dev.set_data(deviation)
            vmax_dev = np.nanmax(deviation)
            im_dev.set_clim(0, vmax_dev * 0.9 if vmax_dev > 0 else 1)
            
            # Update intensity profile
            ax_intensity.clear()
            ax_intensity.plot(intensity[0, :], 'b-', linewidth=1)
            ax_intensity.set_ylim(0, np.nanmax(intensity) * 1.1)
            ax_intensity.set_xlabel("Spot Index")
            ax_intensity.set_ylabel("Intensity")
            ax_intensity.set_title(f"Intensity Profile | statics={statics}, std(wf)={np.std(wf):.4f}")
            ax_intensity.grid(True, alpha=0.3)
            
            return im_spot, im_wf, im_dev

        ani = animation.FuncAnimation(fig, update, interval=80, blit=False)
        plt.show()
        
        # Original loop preserved below for reference
        # for _ in range(1):
        #     wfs.take_image()
        #     
        #     spots_filed = wfs.get_spotfiled_image()
        #     plt.imshow(spots_filed)
        #     plt.show()
        #     
        #     x, y = wfs.get_spot_deviation()
        #     intensity, _ = wfs.get_spots_statics()
        #     wf, statics = wfs.get_wavefront()
        #     print(f"{statics=}")
        #     print(f'{np.std(wf)}')
            
        # wfs.high_speed = True
        # for _ in range(10):
        #     wfs.take_image()
        #     x,y = wfs.get_spot_deviation()
        #     print(y[0,:])
        #     # print(wfs.get_zernike(3))
        #     print(wfs.get_wavefront()[0][0,:])
            
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
