"""
模拟各种光学元件，对激光器的光后处理
"""


import numpy as np
from copy import deepcopy

from sim.digitaltwin import utilities as utils


class Lens:
    def __init__(self, focus_length):
        """
        聚焦透镜

        :param focus_length: 焦距，正数为聚焦，负数为发散
        """

        self.focus_length = focus_length

    def set(self, focus_length):
        self.focus_length = focus_length

    def out(self, wave):
        if not isinstance(wave, list):
            focus_phase = -np.pi * wave.r ** 2 / wave.lamd / self.focus_length
            wave.change_wf(phase=focus_phase)
        else:
            for i in range(len(wave)):
                focus_phase = -np.pi * wave[i].r ** 2 / wave[i].lamd / self.focus_length
                wave[i].change_wf(phase=focus_phase)


class SLM:
    def __init__(self, phase, dpix):
        """
        相位片

        :param phase: 施加的相位
        """

        self.phase = phase
        self.npix = len(phase)
        self.dpix = dpix

    def set(self, phase, dpix):
        self.phase = phase
        self.npix = len(phase)
        self.dpix = dpix

    def out(self, wave):
        if self.dpix != wave.dpix or self.npix != wave.npix:
            phase = utils.matrix_size_trans(self.phase, self.dpix, wave.npix, wave.dpix)
        else:
            phase = self.phase

        if not isinstance(wave, list):
            wave.change_wf(phase=phase)
        else:
            for i in range(len(wave)):
                wave[i].change_wf(phase=phase)


class Expander:
    def __init__(self, multiply, npix_mul, dpix_mul):
        """
        扩束器

        :param multiply: 扩束器倍数
        :param npix_mul: 扩束后波前矩阵像素数
        :param dpix_mul: 扩束后波前矩阵像素尺寸
        """

        self.multiply = multiply
        self.npix_mul = npix_mul
        self.dpix_mul = dpix_mul

    def set(self, multiply, npix_mul, dpix_mul):
        self.multiply = multiply
        self.npix_mul = npix_mul
        self.dpix_mul = dpix_mul

    def out(self, wave):
        if not isinstance(wave, list):
            power = wave.power
            wave.wavefront = utils.matrix_size_trans(wave.wavefront, wave.dpix * self.multiply,
                                                        self.npix_mul, self.dpix_mul)
            if wave.ex is not None:
                wave.ex = utils.matrix_size_trans(wave.ex, wave.dpix * self.multiply,
                                                     self.npix_mul, self.dpix_mul)
                wave.ey = utils.matrix_size_trans(wave.ey, wave.dpix * self.multiply,
                                                     self.npix_mul, self.dpix_mul)
            wave.change_grid(self.npix_mul, self.dpix_mul)
            wave.scale_power(power)
        else:
            for i in range(len(wave)):
                power = wave[i].power
                wave[i].wavefront = utils.matrix_size_trans(wave[i].wavefront, wave[i].dpix * self.multiply,
                                                         self.npix_mul, self.dpix_mul)
                if wave[i].ex is not None:
                    wave[i].ex = utils.matrix_size_trans(wave[i].ex, wave[i].dpix * self.multiply,
                                                      self.npix_mul, self.dpix_mul)
                    wave[i].ey = utils.matrix_size_trans(wave[i].ey, wave[i].dpix * self.multiply,
                                                      self.npix_mul, self.dpix_mul)
                wave[i].change_grid(self.npix_mul, self.dpix_mul)
                wave[i].scale_power(power)


class Aperture:
    def __init__(self, radius):
        """

        :param radius: 孔径半径，正数为孔径，负数为遮拦
        """

        self.radius = radius

    def set(self, radius):
        self.radius = radius

    def out(self, wave):
        if not isinstance(wave, list):
            if self.radius > 0:
                mask = (np.sign(self.radius - wave.r) + 1) / 2
            elif self.radius < 0:
                mask = (np.sign(wave.r + self.radius) + 1) / 2
            else:
                mask = 1
            wave.wavefront = np.real(wave.wavefront) * mask + 1j * np.imag(wave.wavefront) * mask
            if wave.ex is not None:
                wave.ex = np.real(wave.ex) * mask + 1j * np.imag(wave.ex) * mask
                wave.ey = np.real(wave.ey) * mask + 1j * np.imag(wave.ey) * mask
        else:
            if self.radius > 0:
                mask = (np.sign(self.radius - wave[0].r) + 1) / 2
            elif self.radius < 0:
                mask = (np.sign(wave[0].r + self.radius) + 1) / 2
            else:
                mask = 1
            for i in range(len(wave)):
                wave[i].wavefront = np.real(wave[i].wavefront) * mask + 1j * np.imag(wave[i].wavefront) * mask
                if wave[i].ex is not None:
                    wave[i].ex = np.real(wave[i].ex) * mask + 1j * np.imag(wave[i].ex) * mask
                    wave[i].ey = np.real(wave[i].ey) * mask + 1j * np.imag(wave[i].ey) * mask


class Polarizer:
    def __init__(self, theta):
        """
        偏振片

        :param theta: 偏振片偏振方向与x轴夹角，单位：°，范围：(-90，90)
        """

        if theta > 90 or theta < -90:
            raise ValueError('Theta is between -90 and 90 degree')

        self.theta = theta / 180 * np.pi

    def set(self, theta):
        if theta > 90 or theta < -90:
            raise ValueError('Theta is between -90 and 90 degree')

        self.theta = theta / 180 * np.pi

    def out(self, wave):
        Jones = [[np.cos(self.theta)**2, np.sin(self.theta*2)/2], [np.sin(self.theta*2)/2, np.sin(self.theta)**2]]

        if not isinstance(wave, list):
            ex = wave.wavefront * 2 ** 0.5 / 2 * np.cos(self.theta)
            ey = wave.wavefront * 2 ** 0.5 / 2 * np.sin(self.theta)
            if wave.ex is not None:
                ex = ex + Jones[0][0] * wave.ex + Jones[0][1] * wave.ey
                ey = ey + Jones[1][0] * wave.ex + Jones[1][1] * wave.ey
            wave.wavefront = wave.wavefront * 0
            wave.ex = ex
            wave.ey = ey
        else:
            for i in range(len(wave)):
                ex = wave[i].wavefront * 2 ** 0.5 / 2 * np.cos(self.theta)
                ey = wave[i].wavefront * 2 ** 0.5 / 2 * np.sin(self.theta)
                if wave[i].ex is not None:
                    ex = ex + Jones[0][0] * wave[i].ex + Jones[0][1] * wave[i].ey
                    ey = ey + Jones[1][0] * wave[i].ex + Jones[1][1] * wave[i].ey
                wave[i].wavefront = wave[i].wavefront * 0
                wave[i].ex = ex
                wave[i].ey = ey


class WavePlate:
    def __init__(self, gamma, theta):
        """

        :param gamma: 快轴与慢轴相位延迟/2pi，半波片为0.5，gamma=1/lamd*|ne-no|d
        :param theta: 波片快轴与x轴的夹角，单位：°，范围：(-90，90)
        """

        if theta > 90 or theta < -90:
            raise ValueError('Theta is between -90 and 90 degree')

        self.gamma = gamma
        self.theta = theta

    def set(self, gamma, theta):
        if theta > 90 or theta < -90:
            raise ValueError('Theta is between -90 and 90 degree')

        self.gamma = gamma
        self.theta = theta

    def out(self, wave):
        gamma = self.gamma / 2
        theta = self.theta * 2
        Jones = [[np.cos(gamma)-1j*np.sin(gamma)*np.cos(theta), -1j*np.sin(gamma)*np.sin(theta)],
                 [-1j*np.sin(gamma)*np.sin(theta), np.cos(gamma)+1j*np.sin(gamma)*np.cos(theta)]]

        if not isinstance(wave, list):
            wf_x = 2 ** 0.5 / 2 * wave.wavefront * (np.cos(theta) - np.sin(theta))
            wf_y = 2 ** 0.5 / 2 * wave.wavefront * (np.cos(theta) + np.sin(theta))
            if wave.ex is not None:
                ex, ey = wave.ex, wave.ey
            else:
                ex, ey = 0, 0
            ex = Jones[0][0] * (ex + wf_x) + Jones[0][1] * (ey + wf_y)
            ey = Jones[1][0] * (ex + wf_x) + Jones[1][1] * (ey + wf_y)
            wave.wavefront *= 0
            wave.ex = ex
            wave.ey = ey
        else:
            for i in range(len(wave)):
                wf_x = 2 ** 0.5 / 2 * wave[i].wavefront * (np.cos(theta) - np.sin(theta))
                wf_y = 2 ** 0.5 / 2 * wave[i].wavefront * (np.cos(theta) + np.sin(theta))
                if wave[i].ex is not None:
                    ex, ey = wave[i].ex, wave[i].ey
                else:
                    ex, ey = 0, 0
                ex = Jones[0][0] * (ex + wf_x) + Jones[0][1] * (ey + wf_y)
                ey = Jones[1][0] * (ex + wf_x) + Jones[1][1] * (ey + wf_y)
                wave[i].wavefront *= 0
                wave[i].ex = ex
                wave[i].ey = ey


class Combiner:
    def __init__(self, percent1=1, percent2=1):
        """
        相干合束，可用于矢量光叠加

        :param percent1: wave1通过合束器后能量占原总能量百分比，与合束后同方向的波，默认1
        :param percent2: wave2通过合束器后能量占原总能量百分比，与合束后不同方向的波，默认1
        """

        self.wave1 = None
        self.wave2 = None
        self.percent1 = percent1
        self.percent2 = percent2

    def set(self, wave1, wave2, percent1, percent2):
        self.wave1 = wave1
        self.wave2 = wave2
        self.percent1 = percent1
        self.percent2 = percent2

    def out(self):
        wave1 = self.wave1
        wave2 = self.wave2
        wave = deepcopy(wave2)
        if not isinstance(wave1, list) and not isinstance(wave2, list):
            if wave1.npix != wave2.npix or wave1.dpix != wave2.dpix:
                wave.wavefront = utils.matrix_size_trans(wave2.wavefront, wave2.dpix, wave1.npix, wave1.dpix)
                if wave2.ex is not None:
                    wave.ex = utils.matrix_size_trans(wave2.ex, wave2.dpix, wave1.npix, wave1.dpix)
                    wave.ey = utils.matrix_size_trans(wave2.ey, wave2.dpix, wave1.npix, wave1.dpix)
                wave.scale_power(wave2.power)

            if wave1.ex is None:
                ex1, ey1 = 0, 0
            else:
                ex1, ey1 = wave1.ex, wave1.ey
            if wave2.ex is None:
                ex2, ey2 = 0, 0
            else:
                ex2, ey2 = wave.ex, wave.ey
            wave.wavefront = wave1.wavefront * self.percent1 ** 0.5 + wave.wavefront * self.percent2 ** 0.5
            if wave1.ex is not None and wave2.ex is not None:
                wave.ex = ex1 * self.percent1 ** 0.5 + ex2 * self.percent2 ** 0.5
                wave.ey = ey1 * self.percent1 ** 0.5 + ey2 * self.percent2 ** 0.5

        else:
            raise TypeError('Combiner can not input wave list')

        return wave


class Splitter:
    def __init__(self, percent1, percent2):
        """
        分束器

        :param percent1: 通道1能量占原总能量百分比，与分束后同方向的波
        :param percent2: 通道2能量占原总能量百分比，与分束后不同方向的波
        """

        self.wave = None
        self.percent1 = percent1
        self.percent2 = percent2

    def set(self, wave, percent1, percent2):
        self.wave = wave
        self.percent1 = percent1
        self.percent2 = percent2

    def out(self):
        wave = deepcopy(self.wave)
        if not isinstance(wave, list):
            wave1 = deepcopy(wave)
            wave2 = deepcopy(wave)
            wave1.change_wf(scale=self.percent1**0.5)
            wave2.change_wf(scale=self.percent1 ** 0.5)

            return wave1, wave2
        else:
            wave1 = []
            wave2 = []
            for i in range(len(wave)):
                wave = deepcopy(wave[i])
                wave.change_wf(scale=self.percent1 ** 0.5)
                wave1.append(wave)
                wave = deepcopy(wave[i])
                wave.change_wf(scale=self.percent2 ** 0.5)
                wave2.append(wave)

            return wave1, wave2
