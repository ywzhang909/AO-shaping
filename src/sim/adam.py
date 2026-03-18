"""
基础大气传输模板
"""

import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import pickle

import numpy as np

import sim.digitaltwin as dt
import sim.spgd as spgd


Cn2 = []
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

    r_bucket = 3.2e-2 / 3
    delta = 0.001  # 0.01
    screen = dt.screens.TurbulentScreen(10, env, atp.tl_harmonic)
    opd, _ = screen.get_opd(1064e-9, 512, 2e-3, 1024e-3)
    DM1 = spgd.AdamDM(15, laser1.wave.x, laser1.wave.y, delta, -0.03, 0.9, 0.99, r_bucket, aber_phi=opd * np.pi * 2 / 1064e-9)
    DM2 = spgd.AdamDM(15, laser1.wave.x, laser1.wave.y, delta, -0.03, 0.9, 0.99, r_bucket, aber_phi=opd * np.pi * 2 / 1064e-9)
    # ------------------------------------------------------------
    it = 2
    for i in range(it * 2):
        if i < 200:
            DM1.r_bucket = 3.2e-3 * 10
            DM2.r_bucket = 3.2e-3 * 10
        elif i < 400:
            DM1.r_bucket = 3.2e-3 * 8
            DM2.r_bucket = 3.2e-3 * 8
        elif i < 600:
            DM1.r_bucket = 3.2e-3 * 6
            DM2.r_bucket = 3.2e-3 * 6
        elif i < 800:
            DM1.r_bucket = 3.2e-3 * 4
            DM2.r_bucket = 3.2e-3 * 4
        else:
            DM1.r_bucket = 3.2e-3 * 3
            DM2.r_bucket = 3.2e-3 * 3

        wave1 = laser1.out()
        DM1.out(wave1)
        focus.out(wave1)
        atp.out(wave1)
        flag1, J1, c1, phase1 = DM1.adam(wave1)
        wave2 = laser2.out()
        DM2.out(wave2)
        focus.out(wave2)
        atp.out(wave2)
        flag2, J2, c2, phase2 = DM2.adam(wave2)

        if i == it * 2 - 1:
            # plt.figure(figsize=(8, 6))
            # plt.pcolor(wave1.x, wave1.y, DM1.phase, cmap='jet')
            # plt.colorbar()

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
            zer = DM1.c
            # ccd.out(wave1)
            # plt.show()

            # wave2 = laser2.out()
            # DM2.out(wave2)
            # focus.out(wave2)
            # atp.out(wave2)
            # I_spgd_out = wave2.intensity
            # DM_phase = DM2.phase
            # zer = DM2.c
            # # ccd.out(wave2)
        if i % 2 == 0:
            print('{4:.1f}... {0:.1f}, J = {1:.2f}, Imax = {2:.2f}, pm={3:.2f}'.format(i/2, J1, wave1.intensity.max() / 1e7,
                  DM1.phase.max() - DM1.phase.min(), p))

    np.savetxt('data/sim/Iraw{0}.txt'.format(p), I_raw_out)
    np.savetxt('data/sim/Ispgd{0}.txt'.format(p), I_spgd_out)
    np.savetxt('data/sim/DMphase{0}.txt'.format(p), DM_phase)
    np.savetxt('data/sim/zernike{0}.txt'.format(p), zer)
np.savetxt('data/sim/Cn2.txt', Cn2)

plt.show()
