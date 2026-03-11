"""
相位屏
"""
import numpy as np
import math

from sim.digitaltwin import utilities as utils
from sim.digitaltwin import params


class ThermalScreen:
    def __init__(self, dist, env, solve_mode='FFT_non_Isobaric'):
        """
        热晕相位屏

        :param dist: 相位屏等效的长度
        :param env: 相位屏所处环境
        :param solve_mode: 热晕相位屏的求解器，默认'Green'，可选'FFT_Isobaric'，'FFT_non_Isobaric'
        """
        self.opd = None

        self.dist = dist
        self.absorb = env.absorb
        self.wind_x = env.wind_x
        self.wind_y = env.wind_y
        self.density = env.density
        self.Cp = env.Cp
        self.Cv = env.Cv
        self.temperature = env.temperature
        self.Cs2 = env.Cs2
        self.gravity = env.gravity

        self.solve_mode = solve_mode

    def nat_conv_vel(self, power):
        """
        自然对流风速

        :param power: 功率
        :return v: 自然对流风速
        """
        v = (2 * self.absorb * power * self.gravity / (self.density * self.Cp * self.temperature)) ** (1 / 3)

        return v

    def rho2n(self, refractive, rho1):
        """
        密度变化量转变为折射率变化量

        :param refractive: 环境折射率
        :param rho1: 密度变化量
        :return n1: 折射率变化量
        """
        n1 = (refractive - 1) / self.density * rho1

        return n1

    def n2rho(self, refractive, n1):
        """
        折射率变化量转变为密度变化量

        :param refractive: 环境折射率
        :param n1: 折射率变化量
        :return rho1: 密度变化量
        """
        rho1 = n1 * self.density / (refractive - 1)

        return rho1

    def get_rho(self, qx, qy, dpix, intensity):
        """
        求解密度变化量

        :param qx: 空间频率域x
        :param qy: 空间频率域y
        :param dpix: 像素尺寸
        :param intensity: 光强度
        :return rho1: 密度变化量
        """
        if self.solve_mode == 'Green':
            rho1 = self.green(dpix, intensity)
        elif self.solve_mode == 'FFT_Isobaric':
            rho1 = self.fft_isobaric(dpix, intensity)
        elif self.solve_mode == 'FFT_non_Isobaric':
            rho1 = self.fft_non_isobaric(qx, qy, dpix, intensity)
        else:
            raise ValueError('solve_mode is wrong set')

        return rho1

    def green(self, dpix, intensity):
        """
        格林函数求解密度变化量

        :param dpix: 像素尺寸
        :param intensity: 光强度
        :return rho1: 密度变化量
        """
        alpha = self.absorb
        cp = self.Cp
        t = self.temperature
        vx = self.wind_x
        vy = self.wind_y
        v = np.sqrt(vx ** 2 + vy ** 2)

        mu = alpha / cp / t / v
        rho1 = -mu * utils.line_integral(intensity, vx, vy) * dpix

        return rho1

    def fft_isobaric(self, dpix, intensity):
        """
        FFT求解等压近似密度变化量

        :param dpix: 像素尺寸
        :param intensity: 光强度
        :return rho1: 密度变化量
        """
        gamma = self.Cp / self.Cv
        cs2 = self.Cs2
        alpha = self.absorb
        vx = self.wind_x
        vy = self.wind_y
        v = np.sqrt(vx ** 2 + vy ** 2)

        rho1 = -(gamma - 1.) * alpha / cs2 / np.abs(v) * utils.line_integral(intensity, vx, vy) * dpix

        return rho1

    def fft_non_isobaric(self, qx, qy, dpix, intensity):
        """
        FFT求解非等压近似密度变化量

        :param qx: 空间频率域x
        :param qy: 空间频率域y
        :param dpix: 像素尺寸
        :param intensity: 光强度
        :return rho1: 密度变化量
        """
        qx[0, 0] = 1e-31
        qy[0, 0] = 1e-31
        gamma = self.Cp / self.Cv
        cs2 = self.Cs2
        alpha = self.absorb
        vx = self.wind_x
        vy = self.wind_y
        v = np.sqrt(vx ** 2 + vy ** 2)
        m2 = v ** 2 / cs2

        phi_fft = -(gamma - 1) * alpha / cs2 / np.abs(v) * \
                   (qx ** 2 + qy ** 2) / ((1 - m2) * qx ** 2 + qy ** 2) * np.fft.fft2(intensity) * dpix
        phi_fft[0, 0] = 0
        phi = np.fft.ifft2(phi_fft).real

        rho1 = utils.line_integral(phi, vx, vy)

        return rho1

    def out(self, wave):
        intensity = 0
        power = 0
        if not isinstance(wave, list):
            power = wave.power
            refractive = wave.refractive
            dpix = wave.dpix
            qx = wave.qx
            qy = wave.qy
            intensity = wave.intensity
        else:
            refractive = wave[0].refractive
            dpix = wave[0].dpix
            qx = wave[0].qx
            qy = wave[0].qy
            for i in range(len(wave)):
                intensity = intensity + wave[i].intensity
                power = power + wave[i].power

        if self.wind_x == 0.0 and self.wind_y == 0.0:
            self.wind_y = -self.nat_conv_vel(power)
        elif self.wind_x != 0.0 and self.wind_y != 0.0:
            raise ValueError('wind direction must be x or y')

        rho1 = self.get_rho(qx, qy, dpix, intensity)
        n1 = self.rho2n(refractive, rho1)
        self.opd = n1 * self.dist

        if not isinstance(wave, list):
            phase = self.opd * 2 * np.pi / wave.lamd
            wave.change_wf(1, phase)
        else:
            for i in range(len(wave)):
                phase = self.opd * 2 * np.pi / wave[i].lamd
                wave[i].change_wf(1, phase)


class TimeThermalScreen(ThermalScreen):
    def __init__(self, delta_time, dist, env):
        """
        热晕相位屏

        :param dist: 相位屏等效的长度
        :param env: 相位屏所处环境
        """
        super(TimeThermalScreen, self).__init__(dist, env, 'Green')

        self.time = 0
        self.delta_time = delta_time

        self.intensity_time = 0

    def update_opd_time(self, refractive, intensity, npix, dpix):
        """
        更新相位屏
        :param refractive: 环境折射率
        :param intensity: 光强
        :param npix: 相位屏像素数
        :param dpix: 相位屏尺寸
        :return opd_time: 当前时刻的相位屏
        """
        q = np.fft.fftfreq(npix, dpix) * np.pi * 2
        q[0] += 1e-3
        qx, qy = np.meshgrid(q, q)
        vk = self.wind_x * qx + self.wind_y * qy + 1e-16

        exp = np.exp(-1j * vk * self.delta_time)
        eff = -(self.Cp / self.Cv - 1) / self.Cs2 * self.absorb
        an = 1 / (1j * vk) * ((1 - exp) / (1j * vk * self.delta_time) - exp)
        bn = 1 / (1j * vk) * (1 - (1 - exp) / (1j * vk * self.delta_time))

        if self.opd is None:
            self.opd = np.zeros((npix, npix), dtype='float')
        rho1_time = self.n2rho(refractive, self.opd / self.dist)

        rho1_t_fft = np.fft.fft2(rho1_time)
        f_t_fft = np.fft.fft2(self.intensity_time * eff)
        f_tdt_fft = np.fft.fft2(intensity * eff)

        rho1_tdt_fft = exp * rho1_t_fft + an * f_t_fft + bn * f_tdt_fft
        rho1_time = np.real(np.fft.ifft2(rho1_tdt_fft))
        opd_time = self.rho2n(refractive, rho1_time) * self.dist

        opd_time[:, 0: math.ceil(self.delta_time * (self.wind_x ** 2 + self.wind_y ** 2) ** 0.5 / dpix)] = 0

        return opd_time

    def opd_shift(self, opd, dpix):
        """
        由风速带来的相位屏移动
        :param opd: 原相位屏
        :param dpix: 相位屏尺寸
        :return opd_shift: 移动后相位屏
        """
        npix = len(opd)
        opd_shift = np.zeros((npix, npix), dtype='float')
        x_shift = np.abs(round(self.wind_x * self.delta_time / dpix))
        y_shift = np.abs(round(self.wind_y * self.delta_time / dpix))

        if self.wind_x >= 0 and self.wind_y >= 0:
            opd_shift[y_shift: npix, x_shift: npix] = opd[0: npix - y_shift, 0: npix - x_shift]
        elif self.wind_x >= 0 >= self.wind_y:
            opd_shift[0: npix - y_shift, x_shift: npix] = opd[y_shift: npix, 0: npix - x_shift]
        elif self.wind_x <= 0 <= self.wind_y:
            opd_shift[y_shift: npix, 0: npix - x_shift] = opd[0: npix - y_shift, x_shift: npix]
        else:
            opd_shift[0: npix-y_shift, 0: npix-x_shift] = opd[y_shift: npix, x_shift: npix]

        return opd_shift

    def time_update(self, delta_time):
        """
        时间更新

        :param delta_time: 更新的时间
        """
        self.time += delta_time

    def out(self, wave):
        intensity = 0
        power = 0
        if not isinstance(wave, list):
            power = wave.power
            refractive = wave.refractive
            intensity = wave.intensity
            npix, dpix = wave.npix, wave.dpix
        else:
            refractive = wave[0].refractive
            npix, dpix = wave[0].npix, wave[0].dpix
            for i in range(len(wave)):
                intensity = intensity + wave[i].intensity
                power = power + wave[i].power

        if self.wind_x == 0.0 and self.wind_y == 0.0:
            self.wind_y = -self.nat_conv_vel(power)
        elif self.wind_x != 0.0 and self.wind_y != 0.0:
            raise ValueError('wind direction must be x or y')

        if self.opd is None:
            self.opd = np.zeros((npix, npix), dtype='float')
        else:
            self.opd = self.update_opd_time(refractive, intensity, npix, dpix)

        if not isinstance(wave, list):
            phase = self.opd * 2 * np.pi / wave.lamd
            wave.change_wf(1, phase)
        else:
            for i in range(len(wave)):
                phase = self.opd * 2 * np.pi / wave[i].lamd
                wave[i].change_wf(1, phase)

        self.intensity_time = params.WaveIndex.total_intensity(wave)


class SinglePulseThermalScreen(ThermalScreen):
    def __init__(self, delta_time, dist, env):
        """
        热晕相位屏

        :param dist: 相位屏等效的长度
        :param env: 相位屏所处环境
        """
        super(SinglePulseThermalScreen, self).__init__(dist, env, 'Green')

        self.time = 0
        self.delta_time = delta_time

    def update_opd_time(self, refractive, wave):
        """
        更新相位屏
        :param refractive: 环境折射率
        :param wave: 波
        :return opd_time: 当前时刻的相位屏
        """
        intensity = wave.intensity
        radius = params.WaveIndex.radius(intensity, wave.x, wave.y, energy=0.865)
        th = radius / self.Cs2 ** 0.5

        if self.opd is None:
            self.opd = np.zeros((wave.npix, wave.npix), dtype='float')
        rho1_time = self.n2rho(refractive, self.opd / self.dist)

        if wave.duration_time > th:
            delta_rho1 = -(self.Cp / self.Cv - 1) / self.Cs2 * self.absorb * intensity * self.delta_time
        else:
            lap_x = np.gradient(np.gradient(intensity, wave.dpix, axis=0), wave.dpix, axis=0)
            lap_y = np.gradient(np.gradient(intensity, wave.dpix, axis=1), wave.dpix, axis=1)
            delta_rho1 = (self.Cp / self.Cv - 1) * self.absorb * self.delta_time ** 3 / 6 * (lap_x + lap_y)

        rho1_time = rho1_time + delta_rho1
        opd_time = self.rho2n(refractive, rho1_time) * self.dist

        return opd_time

    def time_update(self, delta_time):
        """
        时间更新

        :param delta_time: 更新的时间
        """
        self.time += delta_time

    def out(self, wave):
        refractive = wave.refractive
        npix, dpix = wave.npix, wave.dpix

        if self.opd is None:
            self.opd = np.zeros((npix, npix), dtype='float')
        else:
            self.opd = self.update_opd_time(refractive, wave)

        phase = self.opd * 2 * np.pi / wave.lamd
        wave.change_wf(1, phase)


class TurbulentScreen:
    def __init__(self, dist, env, harmonic=1):
        """
        湍流相位屏

        :param dist: 相位屏等效的长度
        :param env: 相位屏所处环境
        :param harmonic: 湍流屏的次谐波相位屏阶数，默认1
        """

        self.opd = None
        self.dpix = 0

        self.dist = dist
        self.Cn2 = env.Cn2
        self.L0 = env.L0
        self.l0 = env.l0
        self.harmonic = harmonic

    @staticmethod
    def rand_symmetry(npix, sign):
        """
        高斯分布随机数

        :param npix: 相位屏像素数
        :param sign: 正负号
        :return: a: 随机数
        """
        if np.abs(sign) != 1:
            raise ValueError("sign must be either +1 or -1")

        sign = float(sign)
        seed = np.random.randint(0, 1e8)

        random_numbers = np.random.RandomState()
        random_numbers.seed(seed)
        a = random_numbers.normal(size=(npix, npix))

        a[0, int(npix/2)+1:npix] = sign * a[0, 1:int(npix/2)][::-1]
        a[int(npix/2)+1:npix, 0] = sign * a[1:int(npix/2), 0][::-1]
        a[int(npix/2)+1:npix, int(npix/2)+1:npix] = sign * np.rot90(a[1:int(npix/2), 1:int(npix/2)], 2)
        a[int(npix/2)+1:npix, 1:int(npix/2)] = sign * np.rot90(a[1:int(npix/2), int(npix/2)+1:npix], 2)

        a[0, 0] = 0.0

        return a

    def rand_turbulent(self, npix):
        """
        复数随机数

        :param npix: 相位屏像素数
        :return: c: 随机数矩阵
        """
        a = self.rand_symmetry(npix, 1)
        b = self.rand_symmetry(npix, -1)

        c = (a + 1j * b) / np.sqrt(2.0)

        return c

    def power_spectrum(self, npix, dpix):
        """
        功率谱

        :param npix: 相位屏像素数
        :param dpix: 相位屏尺寸
        :return phi: 功率谱
        """
        k02 = (1 / self.L0) ** 2
        km2 = (5.92 / self.l0) ** 2
        _, _, _, _, _, qr = utils.grid(npix, dpix)

        phi = 0.0330054 * self.Cn2 * (qr ** 2 + k02) ** (-11 / 6) * np.exp(-qr ** 2 / km2)

        return phi

    def sub_harmonic(self, lamd, npix, side_length, order):
        """
        湍流次谐波

        :param lamd: 波长
        :param npix: 相位屏像素数
        :param side_length: 相位屏长度
        :param order: 谐波结束
        :return: opd: 次谐波相位屏
        """
        k = 2 * np.pi / lamd
        dq = 1 / side_length

        r0 = (0.423 * k ** 2 * self.Cn2 * self.dist) ** (-3.0 / 5.0)
        f0 = 1 / self.L0
        power_law_exp = 11 / 3
        na = power_law_exp / 6.0
        Bnum = math.gamma(na / 2)
        Bdenom = (2 ** (2 - na)) * np.pi * na * math.gamma(-na / 2)
        Bfac = (2 * np.pi) ** (2 - na) * Bnum / Bdenom
        cone = (2 * (8 / (na - 2) * math.gamma(2 / (na - 2))) ** ((na - 2) / 2))

        harmonic_phase = np.zeros((npix, npix), dtype='complex')

        x = np.linspace(-0.5, 0.5, npix)
        x, y = np.meshgrid(x, -1 * np.transpose(x))
        xp = np.linspace(-2.5, 2.5, 6)
        xp, yp = np.meshgrid(xp, -1 * np.transpose(xp))

        for n in range(1, order + 1):
            temp_phase = np.zeros((npix, npix), dtype='complex')

            dq_n = dq / (3.0 ** n)
            f = np.sqrt((xp ** 2 + yp ** 2)) * 3 ** (-n) * dq
            psd_phi = cone * Bfac * (f ** 2 + f0 ** 2) ** (-na / 2) * r0 ** (2 - na)

            w = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
            covariances = w * np.sqrt(psd_phi) * dq_n

            temp_shape = np.shape(covariances)
            for i in range(0, temp_shape[0]):
                for j in range(0, temp_shape[1]):
                    index_map = (xp[i, j] * x + yp[i, j] * y)
                    temp_phase += covariances[j, i] * np.exp(1j * 2 * np.pi * 3 ** -n * index_map)
            harmonic_phase += temp_phase

        harmonic_phase = np.real(harmonic_phase) - np.mean(np.real(harmonic_phase))
        opd = harmonic_phase / k

        return opd

    def get_opd(self, lamd, npix, dpix, side_length):
        """
        计算湍流相位屏

        :param lamd: 波长
        :param npix: 相位屏像素数
        :param dpix: 相位屏尺寸
        :param side_length: 相位屏长度
        :return: opd, dpix: 湍流相位屏，像素尺寸
        """
        a = self.rand_turbulent(npix)
        phi = self.power_spectrum(npix, dpix)
        opd_fft = 2 * np.pi / side_length * a * np.sqrt(2 * np.pi * self.dist * phi)
        opd = np.real(np.fft.ifft2(opd_fft) * npix ** 2)
        if self.harmonic != 0:
            opd += self.sub_harmonic(lamd, npix, side_length, self.harmonic)

        return opd, dpix

    def out(self, wave):
        if not isinstance(wave, list):
            lamd = wave.lamd
            side_length = wave.side_length
            npix = wave.npix
            dpix = wave.dpix
        else:
            lamd = wave[0].lamd
            side_length = wave[0].side_length
            npix = wave[0].npix
            dpix = wave[0].dpix

        if self.opd is None:
            self.opd, self.dpix = self.get_opd(lamd, npix, dpix, side_length)
        else:
            if self.dpix != dpix or len(self.opd) != npix:
                self.opd = utils.matrix_size_trans(self.opd, self.dpix, npix, dpix)
                self.dpix = dpix

        if not isinstance(wave, list):
            phase = self.opd * 2 * np.pi / wave.lamd
            wave.change_wf(1, phase)
        else:
            for i in range(len(wave)):
                phase = self.opd * 2 * np.pi / wave[i].lamd
                wave[i].change_wf(1, phase)


class TimeTurbulentScreen(TurbulentScreen):
    def __init__(self, total_time, delta_time, dist, env, harmonic=1):
        """
        时间湍流相位屏

        :param total_time: 湍流计算总时间
        :param delta_time: 湍流时间间隔
        :param dist: 相位屏等效的长度
        :param env: 相位屏所处环境
        :param harmonic: 湍流屏的次谐波相位屏阶数，默认1
        """

        super(TimeTurbulentScreen, self).__init__(dist, env, harmonic)

        self.time = 0
        self.total_time = total_time
        self.delta_time = delta_time
        self.vx = env.wind_x
        self.vy = env.wind_y
        self.opd_large = None

    def cut(self, npix, dpix):
        """
        剪切湍流屏

        :param npix: 剪切后的像素数
        :param dpix: 剪切湍流屏的像素尺寸
        :return opd_cut: 剪切后的光程矩阵
        """
        total_npix = len(self.opd_large)
        shift_x = np.abs(math.floor(self.time * self.vx / dpix))
        shift_y = np.abs(math.floor(self.time * self.vy / dpix))

        if self.vx >= 0 and self.vy >= 0:
            opd_cut = self.opd_large[total_npix - npix - shift_y: total_npix - shift_y,
                      total_npix - npix - shift_x: total_npix - shift_x]
        elif self.vx >= 0 >= self.vy:
            opd_cut = self.opd_large[shift_y: npix + shift_y,
                      total_npix - npix - shift_x: total_npix - shift_x]
        elif self.vx <= 0 <= self.vy:
            opd_cut = self.opd_large[total_npix - npix - shift_y: total_npix - shift_y,
                      shift_x: npix + shift_x]
        else:
            opd_cut = self.opd_large[shift_y: npix + shift_y,
                      shift_x: npix + shift_x]

        return opd_cut

    def time_update(self, delta_time):
        """
        时间更新

        :param delta_time: 更新的时间
        """
        self.time += delta_time

    def out(self, wave):
        if not isinstance(wave, list):
            lamd = wave.lamd
            side_length = wave.side_length
            dpix = wave.dpix
            cut_npix = wave.npix
        else:
            lamd = wave[0].lamd
            side_length = wave[0].side_length
            dpix = wave[0].dpix
            cut_npix = wave[0].npix

        if int((np.max((self.vx, self.vy)) * self.total_time + side_length) / dpix * 6 / 5) > 4096:
            self.opd, self.dpix = self.get_opd(lamd, cut_npix, dpix, side_length)
        else:
            if self.opd_large is None:
                v = np.max((self.vx, self.vy))
                npix = int((v * self.total_time + side_length) / dpix * 6 / 5)
                if npix % 2 != 0:
                    npix += 1
                self.opd_large, self.dpix = self.get_opd(lamd, npix, dpix, npix*dpix)
            else:
                if self.dpix != dpix:
                    npix = math.floor(len(self.opd_large) * self.dpix / dpix)
                    self.opd_large = utils.matrix_size_trans(self.opd_large, self.dpix, npix, dpix)
                    self.dpix = dpix

            self.opd = self.cut(cut_npix, dpix)

        if not isinstance(wave, list):
            phase = self.opd * 2 * np.pi / wave.lamd
            wave.change_wf(1, phase)
        else:
            for i in range(len(wave)):
                phase = self.opd * 2 * np.pi / wave[i].lamd
                wave[i].change_wf(1, phase)
