"""
基础大气传输模板
"""


import matplotlib.pyplot as plt
import pickle
import time

import numpy as np

import digitaltwin as dt
import spgd


t = 0  # 运行时间
Cn2 = []
J = []
for p in range(20):
    rand = np.random.uniform(1, 10) * 1e-14
    Cn2.append(rand)
    env = dt.base.Environment()
    env.set_value(
        absorb=5e-6,
        scatter=60e-6,
        wind_x=1,
        wind_y=0,
        Cn2=Cn2[p],
        density=1.177,
        Cp=1005,
        Cv=717,
        temperature=300,
        nT=-8.6e-7,
        atm=1,
        Cs2=347.2**2,
        gravity=9.81,
        L0=4,
        l0=8.4e-3
    )

    laser1 = dt.laser.ContinuousLaser(
        npix=512, dpix=2e-3, power=1e3, wavelength=1064e-9, aperture=0.8, env=env,
        source_type='Gauss', radius=None, charges=0, beam_quality=1.0, stokes=(0, 0, 0), wf_customized=None
    )
    focus = dt.optics.Lens(
        focus_length=3200
    )
    atp = dt.atp.ATP(
        prop_dist=3200, layers=10, env_init=env,
        height=0, emission_angle=0, set_env='auto', env_array=None,
        Thermal=False, Turbulent=True, tl_harmonic=0, tb_mode='Green'
    )
    ccd = dt.display.CCD(
        npix=512, dpix=2e-3, figsize=(6.5, 6), length_unit='cm', intensity_unit='KW/cm2',
        bucket_cycle=False, bucket_center=None, bucket_radius=2, cycle_linewidth=2,
        polar_state=False, n_ellipse=10, ellipse_size=1, polar_linewidth=1,
        lim=None, cmap='jet'
    )
    # ----------------------------------------------------------------------------------------------

    r_bucket = 1.6e-2
    delta = 0.001  # 0.01
    DM1 = spgd.voltageAdam(21, 0.8, 0.8, 2, laser1.wave.x, laser1.wave.y, delta, -2, 0.9, 0.999, r_bucket)
    # ------------------------------------------------------------
    it = 500
    max_J, count = 0, 0
    for i in range(it * 2):
        if i < 500:
            DM1.r_bucket = 3.2e-3 * 10
        elif i < 700:
            DM1.r_bucket = 3.2e-3 * 8
        elif i < 800:
            DM1.r_bucket = 3.2e-3 * 6
        elif i < 900:
            DM1.r_bucket = 3.2e-3 * 4
        else:
            DM1.r_bucket = 3.2e-3 * 3

        wave1 = laser1.out()
        DM1.out(wave1)
        focus.out(wave1)
        atp.out(wave1)
        start = time.perf_counter()
        flag1, J1, u1, phase1 = DM1.adamax(wave1)
        end = time.perf_counter()
        t = t + end - start
        J.append(J1)

        # if i % 2 == 0:
        #     if max_J <= J1:
        #         max_J = J1
        #         count = 0
        #     else:
        #         count = count + 1
        #         if count >= 5:
        #             plt.figure(figsize=(8, 6))
        #             plt.pcolor(wave1.x, wave1.y, DM1.phase, cmap='jet')
        #             plt.colorbar()
        #
        #             wave1 = laser1.out()
        #             focus.out(wave1)
        #             atp.out(wave1)
        #             I_raw_out = wave1.intensity
        #             # ccd.out(wave1)
        #
        #             wave1 = laser1.out()
        #             DM1.out(wave1)
        #             focus.out(wave1)
        #             atp.out(wave1)
        #             I_spgd_out = wave1.intensity
        #             DM_phase = DM1.phase
        #             voltage = DM1.voltage
        #             # ccd.out(wave1)
        #             # plt.show()
        #             break

        if i == it * 2 - 1:
            # plt.figure(figsize=(8, 6))
            # plt.pcolor(wave1.x, wave1.y, DM1.phase, cmap='jet')
            # plt.colorbar()

            # plt.figure(figsize=(8, 6))
            # plt.plot(J)
            # plt.show()

            wave1 = laser1.out()
            focus.out(wave1)
            atp.out(wave1)
            I_raw_out = wave1.intensity
            # ccd.out(wave1)

            wave1 = laser1.out()
            DM1.out(wave1)
            focus.out(wave1)
            atp.out(wave1)
            I_spgd_out = wave1.intensity
            DM_phase = DM1.phase
            voltage = DM1.voltage
            # ccd.out(wave1)
            # plt.show()

        if i % 2 == 0:
            print('{4:.1f}... {0:.1f}, J = {1:.2f}, um = {2:.2f}, pm={3:.2f}, time={5:.2f}'.format(i/2, J1, DM1.voltage.max(),
                  DM1.phase.max() - DM1.phase.min(), p, t))

    np.savetxt('data_adamax/Iraw{0}.txt'.format(p), I_raw_out)
    np.savetxt('data_adamax/Ispgd{0}.txt'.format(p), I_spgd_out)
    np.savetxt('data_adamax/DMphase{0}.txt'.format(p), DM_phase)
    np.savetxt('data_adamax/zernike{0}.txt'.format(p), voltage)
    np.savetxt('data_adamax/J{0}.txt'.format(p), J)
np.savetxt('data_adamax/Cn2.txt', Cn2)

plt.show()
