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
    laser2 = dt.laser.ContinuousLaser(
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

    r_bucket = 0.7
    delta = 0.001  # 0.01
    screen = dt.screens.TurbulentScreen(10, env, atp.tl_harmonic)
    opd, _ = screen.get_opd(1064e-9, 512, 2e-3, 1024e-3)
    DM1 = spgd.AdamDM(7, laser1.wave.x, laser1.wave.y, delta, -1, 0.9, 0.999, r_bucket)
    DM2 = spgd.AdamDM(7, laser1.wave.x, laser1.wave.y, delta, -1, 0.9, 0.999, r_bucket)
    # ------------------------------------------------------------
    it = 500
    for i in range(it * 2):
        wave1 = laser1.out()
        DM1.out(wave1)
        focus.out(wave1)
        atp.out(wave1)
        start = time.perf_counter()
        flag1, J1, c1, phase1 = DM1.adamax(wave1)
        end = time.perf_counter()
        t = t + end - start
        J.append(J1)

        wave2 = laser2.out()
        DM2.out(wave2)
        focus.out(wave2)
        atp.out(wave2)
        flag2, J2, c2, phase2 = DM2.adamax(wave2)

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
            ccd.out(wave1)

            wave1 = laser1.out()
            DM1.out(wave1)
            focus.out(wave1)
            atp.out(wave1)
            I_spgd_out = wave1.intensity
            DM_phase = DM1.phase
            zer = DM1.c
            ccd.out(wave1)
            plt.show()

            # wave2 = laser2.out()
            # DM2.out(wave2)
            # focus.out(wave2)
            # atp.out(wave2)
            # I_spgd_out = wave2.intensity
            # DM_phase = DM2.phase
            # zer = DM2.c
            # # ccd.out(wave2)
        if i % 2 == 0:
            print('{4:.1f}... {0:.1f}, J = {1:.2f}, cm = {2:.2f}, pm={3:.2f}, time={5:.2f}'.format(i/2, J1, DM1.c.max(),
                  DM1.phase.max() - DM1.phase.min(), p, t))

    np.savetxt('data_adamax/Iraw{0}.txt'.format(p), I_raw_out)
    np.savetxt('data_adamax/Ispgd{0}.txt'.format(p), I_spgd_out)
    np.savetxt('data_adamax/DMphase{0}.txt'.format(p), DM_phase)
    np.savetxt('data_adamax/zernike{0}.txt'.format(p), zer)
    np.savetxt('data_adamax/J{0}.txt'.format(p), J)
np.savetxt('data_adamax/Cn2.txt', Cn2)

plt.show()
