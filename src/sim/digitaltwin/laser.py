"""
激光器光源
"""

import numpy as np
from copy import deepcopy

from sim.digitaltwin import base
from sim.digitaltwin import source
from sim.digitaltwin import screens
from sim.digitaltwin import utilities as utils
from sim.digitaltwin import params


class ContinuousLaser:
    def __init__(self, npix, dpix, power, wavelength, aperture, env, beam_quality=1.0, stokes=(0, 0, 0),
                 source_type='Gauss', radius=None, charges=0, law=1.2, phase_var=2000, source_z=1, wf_customized=None
                 ):
        """
        连续光激光器

        :param npix: 波前面像素数
        :param dpix: 波前面像素尺寸
        :param power: 激光功率
        :param wavelength: 激光波长
        :param aperture: 孔径
        :param env: 激光器所处环境
        :param beam_quality: 光束质量，默认1.0
        :param stokes: 斯托克斯参数，默认(0, 0, 0)，平方和小于等于1
        :param source_type: 光源种类，默认'Gauss'，可选'FlatTop', 'Laguerre', 'Pin', 'Customized'
        :param radius: 平顶光半径或高斯光束腰半径，默认None，表示为孔径/2或孔径/2/sqrt(2)
        :param charges: 拉盖尔高斯光的拓扑荷数，默认0，仅拉盖尔高斯光需要设置
        :param law: 锋芒光束的幂律指数，默认1.2，仅锋芒光束需要设置
        :param phase_var: 相位变化速度，默认2000，仅锋芒光束需要设置
        :param source_z: 距离点光源距离，默认1m，仅球面波波前需要设置
        :param wf_customized: 自定义波前，默认None，仅自定义波需要设置
        """
        self.wave = base.Wave()
        self.wave.change_grid(npix, dpix)
        self.wave.wavelength = wavelength
        self.wave.refractive = utils.get_air_refractive(wavelength, env.temperature, env.atm)
        self.wave.wavefront = np.ones((npix, npix), dtype='complex')

        self.power = power
        self.wavelength = wavelength
        self.aperture = aperture
        self.source_type = source_type
        self.radius = radius
        self.charges = charges
        self.law = law
        self.phase_var = phase_var
        self.source_z = source_z
        self.wf_customized = wf_customized
        self.beam_quality = beam_quality
        self.stokes = stokes
        if stokes[0] ** 2 + stokes[1] ** 2 + stokes[2] ** 2 > 1+1e-10:
            raise ValueError('偏振度大于1')

        self.env = env

    def get_source(self):
        """
        根据光源类型设置完全非偏振波前

        :return wavefront: 波前矩阵
        """

        r = self.wave.r
        radius = self.radius
        theta = np.arctan2(self.wave.y, self.wave.x)

        if radius is None and (self.source_type == 'Gauss' or self.source_type == 'Laguerre'
                               or self.source_type == 'Pin'):
            radius = self.aperture / 2 / np.sqrt(2)
        elif radius is None and self.source_type == 'FlatTop':
            radius = self.aperture / 2

        if self.source_type == 'Gauss':
            wavefront = source.gauss(r, radius)
        elif self.source_type == 'FlatTop':
            wavefront = source.flat(r, radius)
        elif self.source_type == 'Laguerre':
            wavefront = source.laguerre_gaussian(r, theta, radius, self.charges)
        elif self.source_type == 'Sphere':
            wavefront = source.sphere(r, self.wave.lamd, self.source_z)
        elif self.source_type == 'Cylinder':
            wavefront = source.cylinder(r, self.wave.lamd, self.source_z)
        elif self.source_type == 'Pin':
            wavefront = source.pin_like(r, self.wave.lamd, radius, self.law, self.phase_var)
        elif self.source_type == 'Customized':
            wavefront = source.customized(self.wf_customized)
        else:
            raise TypeError('source_type error')

        self.wave.wavefront = wavefront * (np.sign(self.aperture / 2 - r) + 1) / 2
        self.wave.scale_power(self.power)

    def quality(self, beam_quality):
        """
        根据光束质量设置波前畸变

        :param beam_quality: 光束质量
        :return aberration: 畸变相位
        """

        r0 = self.aperture / ((beam_quality ** 2 - 1) / 0.62) ** (3/5)
        Cn2 = params.TurbulentIndex.Cn2(r0, self.wave.lamd, 1)  # r0计算Cn2

        env = base.Environment()
        env.L0 = self.env.L0  # 外尺度
        env.l0 = self.env.l0  # 内尺度
        env.Cn2 = Cn2

        wave = deepcopy(self.wave)
        screen = screens.TurbulentScreen(1, env)  # 根据Cn2，L0，l0计算湍流相位屏相位
        screen.out(wave)
        aberration = screen.opd * 2 * np.pi / wave.lamd

        return aberration

    def polarization(self, stokes):
        """
        根据斯托克斯参量设置激光偏振

        :param stokes: 斯托克斯参量s1~s3
        :return wavefront, ex, ey: 偏振光的琼斯矢量和完全非偏振光
        """

        phase = np.angle(self.wave.wavefront)

        DOP = (stokes[0] ** 2 + stokes[1] ** 2 + stokes[2] ** 2) ** 0.5
        s1 = stokes[0] / DOP
        s2 = stokes[1] / DOP
        s3 = stokes[2] / DOP

        if np.abs(s1) == 1.0:
            delta_phase = 0
        elif s3 == 0.0:
            delta_phase = np.pi * (1 - np.sign(s2)) / 2
        else:
            delta_phase = np.arccos(s2 / (1 - s1 ** 2) ** 0.5) * np.sign(s3)

        e_sqr = np.abs(self.wave.wavefront) ** 2
        wavefront = ((1 - DOP) * e_sqr) ** 0.5 * np.exp(1j * phase)
        ex = ((1 + s1) / 2) ** 0.5 * (DOP * e_sqr) ** 0.5 * np.exp(1j * phase)
        ey = ((1 - s1) / 2) ** 0.5 * np.exp(1j * delta_phase) * (DOP * e_sqr) ** 0.5 * np.exp(1j * phase)

        return wavefront, ex, ey

    def out(self):
        self.get_source()
        if self.stokes != (0, 0, 0):
            self.wave.wavefront, self.wave.ex, self.wave.ey = self.polarization(self.stokes)

        if self.beam_quality != 1.0:
            aberration = self.quality(self.beam_quality)
            self.wave.change_wf(phase=aberration)

        return self.wave


class TimeContinuousLaser(ContinuousLaser):
    def __init__(self, npix, dpix, power, wavelength, aperture, env, beam_quality=1.0, stokes=(0, 0, 0),
                 source_type='Gauss', radius=None, charges=0, law=1.2, phase_var=2000, source_z=1, wf_customized=None
                 ):
        """
        时间连续光激光器

        :param npix: 波前面像素数
        :param dpix: 波前面像素尺寸
        :param power: 激光功率
        :param wavelength: 激光波长
        :param aperture: 孔径
        :param env: 激光器所处环境
        :param beam_quality: 光束质量，默认1.0
        :param stokes: 斯托克斯参数，默认(0, 0, 0)，平方和小于等于1
        :param source_type: 光源种类，默认'Gauss'，可选'FlatTop'，'Laguerre', 'Customized'
        :param radius: 平顶光半径，默认None，仅平顶光需要设置，表示为孔径/2
        :param charges: 拉盖尔高斯光的拓扑荷数，默认0，仅拉盖尔高斯光需要设置
        :param law: 锋芒光束的幂律指数，默认1.2，仅锋芒光束需要设置
        :param phase_var: 相位变化速度，默认2000，仅锋芒光束需要设置
        :param source_z: 距离点光源距离，默认1m，仅球面波波前需要设置
        :param wf_customized: 自定义波前，默认None，仅自定义波需要设置
        """

        super(TimeContinuousLaser, self).__init__(
            npix, dpix, power, wavelength, aperture, env, beam_quality, stokes,
            source_type, charges, wf_customized)

        self.wave = base.TimeWave()
        self.wave.change_grid(npix, dpix)
        self.wave.wavelength = wavelength
        self.wave.refractive = utils.get_air_refractive(wavelength, env.temperature, env.atm)
        self.wave.wavefront = np.ones((npix, npix), dtype='complex')

        self.power = power
        self.wavelength = wavelength
        self.aperture = aperture
        self.source_type = source_type
        self.radius = radius
        self.charges = charges
        self.law = law
        self.phase_var = phase_var
        self.source_z = source_z
        self.wf_customized = wf_customized
        self.beam_quality = beam_quality
        self.stokes = stokes
        if stokes[0] ** 2 + stokes[1] ** 2 + stokes[2] ** 2 > 1+1e-10:
            raise ValueError('偏振度大于1')

        self.env = env

    def out(self):
        self.get_source()
        if self.stokes != (0, 0, 0):
            self.wave.wavefront, self.wave.ex, self.wave.ey = self.polarization(self.stokes)

        if self.beam_quality != 1.0:
            aberration = self.quality(self.beam_quality)
            self.wave.change_wf(phase=aberration)

        return self.wave
