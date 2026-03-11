"""
各种参数计算
包括：闪烁指数
"""

import numpy as np
from copy import deepcopy


class TurbulentIndex:
    @staticmethod
    def rytov_variance(Cn2, lamd, z):
        """
        Rytov方差

        :param Cn2: 折射率结构常数
        :param lamd: 波长
        :param z: 传输路径总长度
        :return: Rytov方差
        """
        if not isinstance(Cn2, list):
            return 1.23 * Cn2 * (2 * np.pi / lamd) ** (7/6) * z ** (11/6)
        else:
            k = 2 * np.pi / lamd
            dist = z / len(Cn2)
            rytov = 0
            for i in range(len(Cn2)):
                rytov = rytov + 1.23 * 11 / 6 * k ** (7/6) * Cn2[i] * (z - dist * (i + 0.5)) ** (5/6) * dist
            return rytov

    @staticmethod
    def coherent_length(Cn2, lamd, z):
        """
        相干长度

        :param Cn2: 折射率结构常数
        :param lamd: 波长
        :param z: 传输路径总长度
        :return: 相干长度r0
        """
        if not isinstance(Cn2, list):
            return (0.423 * (2 * np.pi / lamd) ** 2 * Cn2 * z) ** (-3/5)
        else:
            k = 2 * np.pi / lamd
            dist = z / len(Cn2)
            r0 = 0
            for i in range(len(Cn2)):
                r0 = r0 + 0.423 * k ** 2 * Cn2[i] * dist
            return r0 ** (-3/5)

    @staticmethod
    def Cn2(r0, lamd, z):
        """
        折射率结构常数，假设Cn2在路径z上均匀

        :param r0: 大气相干常数
        :param lamd: 波长
        :param z: 传输路径长度
        :return: 折射率结构常数Cn2
        """

        return r0 ** (-5/3) / (2 * np.pi / lamd) ** 2 / z / 0.423 * 8 / 3

    @staticmethod
    def Cn2_height(Cn2_ground, wind, height):
        """
        某高度风速下的Cn2

        :param Cn2_ground: 地面的Cn2
        :param wind: 风速
        :param height: 高度
        :return: Cn2
        """
        v = (wind ** 2 + 30.69 * wind + 348.91) ** 2
        Cn2 = (Cn2_ground + 8.148e-56 * v ** 2 * height ** 10) * np.exp(-height / 100) + \
                    2.7e-16 * np.exp(-height / 1500)

        return Cn2

    @staticmethod
    def outer_scale(height):
        """
        外尺度

        :param height: 高度
        :return: 外尺度
        """
        if height <= 1:
            L0 = 0.4
        elif height <= 25:
            L0 = 0.4 * height
        else:
            L0 = 2 * height ** 0.5

        return L0

    @staticmethod
    def inner_scale(thermal_diffusivity, dissipation_rate):
        """
        内尺度

        :param thermal_diffusivity: 分子热扩散率，m2/s
        :param dissipation_rate: 能量耗散率，W/m2
        :return: 内尺度
        """

        return 5.8 * (thermal_diffusivity ** 3 / dissipation_rate) ** 0.25

    @staticmethod
    def scintillation_index(intensity_list):
        """
        闪烁指数

        :param intensity_list: 多次测量的强度列表
        :return SI: 闪烁指数
        """
        num = len(intensity_list)

        intensity, intensity2 = 0, 0
        for i in range(num):
            index = int(len(intensity_list[i]) / 2)
            intensity = intensity + intensity_list[i][index, index] / num
            intensity2 = intensity2 + intensity_list[i][index, index] ** 2 / num

        SI = (intensity2 - intensity ** 2) / intensity ** 2

        return SI


class ThermalIndex:
    @staticmethod
    def Nd(power, lamd, aperture, z, wind=1, absorb=5e-6, nT=-8.6e-7, rho=1.177, cp=1005):
        """
        准直光束热畸变常数，均匀大气，衰减很小

        :param power: 功率
        :param lamd: 波长
        :param aperture: 孔径
        :param z: 传输距离
        :param wind: 风速
        :param absorb: 吸收系数
        :param nT: 折射率温度系数
        :param rho: 密度
        :param cp: 定压比热
        :return nd: 热畸变常数
        """

        k = 2 * np.pi / lamd
        nd = 4 * 2 ** 0.5 * -nT * k * power * absorb * z / rho / cp / wind / aperture

        return nd

    @staticmethod
    def N(source, power, lamd, aperture, z, quality=1, wind=1, absorb=5e-6,
          scatter=5e-5, nT=-8.6e-7, rho=1.177, cp=1005, omega=0):
        """
        聚焦光束热畸变常数，均匀大气，衰减很小

        :param source: 光源类型，可选'Gauss'，'FlatTop'
        :param power: 功率
        :param lamd: 波长
        :param aperture: 孔径
        :param z: 传输距离
        :param quality: 光束质量
        :param wind: 风速
        :param absorb: 吸收系数
        :param scatter: 散射系数
        :param nT: 折射率温度系数
        :param rho: 密度
        :param cp: 定压比热
        :param omega: 光速扫描角速度
        :return: 热畸变常数
        """
        if wind == 0:
            wind = ThermalIndex.nat_conv_vel(power, absorb)

        if source == 'Gauss':
            Nf = (aperture / 2) ** 2 / lamd / z / quality
        elif source == 'FlatTop':
            Nf = (aperture / 2) ** 2 / lamd / z / 2.1 / quality
        else:
            raise ValueError('Distortion number can only be Gauss or FlatTop type')
        Ne = (absorb + scatter) * z
        Nw = omega * z / wind

        Nc = ThermalIndex.Nd(power, lamd, aperture, z, wind, absorb, nT, rho, cp) / (2 * np.pi * Nf)
        fNe = 2 / Ne ** 2 * (Ne - 1 + np.exp(-Ne))
        if omega == 0:
            sNw = 1
        else:
            sNw = 2 / Nw ** 2 * ((Nw + 1) * np.log(Nw + 1) - Nw)
        qNf = 2 * Nf ** 2 / (Nf - 1) * (1 - np.log(Nf) / (Nf - 1))

        return Nc * fNe * sNw * qNf

    @staticmethod
    def nat_conv_vel(power, absorb, gravity=9.81, density=1.177, Cp=1005, temperature=300):
        """
        自然对流风速

        :param power: 功率
        :param absorb: 吸收系数
        :param gravity: 重力
        :param density: 密度
        :param Cp: 定压比热
        :param temperature: 温度
        :return v: 自然对流风速
        """

        v = (2 * absorb * power * gravity / (density * Cp * temperature)) ** (1 / 3)

        return v


class WaveIndex:
    @staticmethod
    def diffraction_limit(lamd, aperture, dist):
        """
        衍射极限光斑直径

        :param lamd: 波长
        :param aperture: 发射光孔径
        :param dist: 传输距离
        :return: 衍射极限光斑直径
        """

        return 1.22 * lamd * dist / aperture

    @staticmethod
    def jitter_diameter(lamd, aperture, dist):
        """
        光束抖动直径，用于自适应光学

        :param lamd: 波长
        :param aperture: 发射光孔径
        :param dist: 传输距离
        :return diameter: 光束抖动直径
        """
        if dist < 1e4:
            diameter = 1e-6 * dist
        else:
            diameter = 3 * lamd * dist / aperture

        return diameter

    @staticmethod
    def centroid(intensity, x, y):
        """
        光强的质心位置

        :param intensity: 强度分布
        :param x: x坐标矩阵
        :param y: y坐标矩阵
        :return center_x, center_y: 光强的质心
        """
        center_x = (x * intensity).sum() / intensity.sum()
        center_y = (y * intensity).sum() / intensity.sum()

        return center_x, center_y

    @staticmethod
    def peak_position(intensity, x, y):
        """
        光强峰值处的坐标

        :param intensity: 强度分布
        :param x: x坐标矩阵
        :param y: y坐标矩阵
        :return xp, yp: 光强峰值处的坐标
        """
        index = np.unravel_index(intensity.argmax(), intensity.shape)
        xp, yp = x[index[0], index[1]], y[index[0], index[1]]

        return xp, yp

    @staticmethod
    def radius(intensity, x, y, center='centroid', energy=0.99):
        """
        以center为圆心，占总能量百分比为energy的圆的半径

        :param intensity: 强度分布
        :param x: x坐标矩阵
        :param y: y坐标矩阵
        :param center: 圆心，默认'centroid'，可选'peak'，'origin'，坐标，如(0, 0)
        :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
        :return radius: 圆的半径
        """
        npix = len(x)
        dpix = x[0, 1] - x[0, 0]

        if type(center) is str:
            if center == 'peak':
                x0, y0 = WaveIndex.peak_position(intensity, x, y)
            elif center == 'centroid':
                x0, y0 = WaveIndex.centroid(intensity, x, y)
            elif center == 'origin':
                x0, y0 = 0, 0
            else:
                raise ValueError('center is wrong set')
        else:
            x0, y0 = center[0], center[1]

        power_in_circle = intensity.sum() * energy
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        radius = npix * dpix / 2
        radius_change = npix * dpix / 4

        for i in range(30):
            mask = (np.sign(radius - r) + 1) / 2
            power = np.sum(intensity * mask)

            if power - power_in_circle < -1e-10 * power_in_circle:
                radius += radius_change
            elif power - power_in_circle > 1e-10 * power_in_circle:
                radius -= radius_change
            else:
                break

            radius_change /= 2
            if radius_change < dpix / 50:
                break
            if i == 29:
                raise StopIteration('Maximal number of iterations reached while calculating beam radius.')

        return radius

    @staticmethod
    def effective_radius(intensity, dpix, clip):
        """
        有效光斑半径

        :param intensity: 强度分布
        :param dpix: 网格尺寸
        :param clip: 阈值，光强大于峰值*clip的像素计入有效光斑面积
        :return radius: 光斑有效半径
        """
        intensity = intensity
        threshold = intensity.max() * clip
        above = (np.sign((intensity - threshold)) + 1) / 2
        d_effective = (above.sum() * dpix ** 2 * 4 / np.pi) ** 0.5

        return d_effective / 2

    @staticmethod
    def power_bucket(intensity, x, y, center, r_bucket):
        """
        以center为圆心，r_bucket为半径的桶中功率

        :param intensity: 强度分布
        :param x: x坐标矩阵
        :param y: y坐标矩阵
        :param center: 桶的圆心，默认'centroid'，可选'peak'，'origin'，坐标，如(0, 0)
        :param r_bucket: 桶半径
        :return: 桶中功率
        """
        dpix = x[0, 1] - x[0, 0]
        radius = r_bucket

        if type(center) is str:
            if center == 'peak':
                x0, y0 = WaveIndex.peak_position(intensity, x, y)
            elif center == 'centroid':
                x0, y0 = WaveIndex.centroid(intensity, x, y)
            elif center == 'origin':
                x0, y0 = 0, 0
            else:
                raise ValueError('center is wrong set')
        else:
            x0, y0 = center[0], center[1]

        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        mask = (np.sign(radius - r) + 1) / 2
        intensity_in_bucket = intensity * mask
        power_in_bucket = intensity_in_bucket.sum() * dpix ** 2

        return power_in_bucket

    @staticmethod
    def total_intensity(wave):
        """
        非相干波叠加计算光强

        :param wave: 波或波列表
        :return intensity: 叠加后强度
        """

        intensity = 0
        if not isinstance(wave, list):
            intensity = wave.intensity
        else:
            for i in range(len(wave)):
                intensity = intensity + wave[i].intensity

        return intensity

    @staticmethod
    def power(intensity, dpix):
        """
        强度计算功率

        :param intensity: 强度矩阵
        :param dpix: 像素大小
        :return power: 功率
        """
        power = intensity.sum() * dpix ** 2

        return power

    @staticmethod
    def RMS(wavefront, side_length, radius):
        """
        规则圆形波面的RMS，波面外相位为0

        :param wavefront: 波面
        :param side_length: 波面像边长
        :param radius: 波面半径
        :return RMS: 单位为lambda
        """

        npix, dpix = len(wavefront), side_length / len(wavefront)
        sl = npix * dpix
        x = np.linspace(-sl / 2, sl / 2 - dpix, npix)
        x, y = np.meshgrid(x, x)
        r = (x ** 2 + y ** 2) ** 0.5

        N = ((np.sign(radius - r) + 1) / 2).sum()
        phase = np.unwrap(np.angle(wavefront), axis=0)
        phase = np.unwrap(phase, axis=1)
        phase_lamd = phase / np.pi / 2
        mean = phase_lamd.sum() / N

        RMS = np.sqrt(((phase_lamd - mean) ** 2).sum() / N)

        return RMS

    @staticmethod
    def MTF(wavefront, qr):
        """
        规则圆形波面的RMS，波面外相位为0

        :param wavefront: 波面
        :param qr: 波前的空间频率
        :returns freq, mtf: 空间频率，MTF
        """
        otf = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(wavefront)) ** 2))
        mtf = np.abs(otf[int(len(otf)/2), int(len(otf)/2):])
        freq = np.fft.fftshift(qr)

        return freq[int(len(otf)/2), int(len(otf)/2):], mtf / mtf.max()

    @staticmethod
    def AOA(Cn2, lamd, dist, wind, freq):
        """
        到达角功率谱

        :param Cn2: 折射率结构常数
        :param lamd: 波长
        :param dist: 传输距离
        :param wind: 风速
        :param freq: 时间频率
        :returns freq, s_alpha: 湍流频率，到达角功率谱
        """
        k = np.pi * 2 / lamd
        s_phi = 0.016 * k ** 2 * Cn2 * dist * wind ** (5/3) * freq ** (-8/3)
        s_alpha = (lamd / wind) ** 2 * freq ** 2 * s_phi

        return s_alpha

    @staticmethod
    def arrival_spectrum(intensity_list, x, y, prop_dist, delta_time):
        arrival_angle = np.zeros(len(intensity_list))
        for i in range(len(intensity_list)):
            centroid = WaveIndex.centroid(intensity_list[i], x, y)
            shift = (centroid[0] ** 2 + centroid[1] ** 2) ** 0.5
            arrival_angle[i] = shift / prop_dist

        freq = np.fft.fftfreq(1024, len(arrival_angle) * delta_time / 1024)
        cor_aoa = np.correlate(arrival_angle, arrival_angle, 'same')
        power_spectrum = np.abs(np.fft.fft(cor_aoa, 1024)) / 1024 / len(arrival_angle)

        return freq[1: len(freq) // 2], power_spectrum[1: len(power_spectrum) // 2]


class EnvParams:
    @staticmethod
    def temperature_height(T0, height):
        if height <= 11e3:
            T = T0 - 0.00649 * height
        else:
            T = T0 - 0.00649 * 11e3

        return T

    @staticmethod
    def absorb_height(absorb0, height):
        if height < 12e3:
            absorb = absorb0 * np.exp(-5e-4 * height)
        elif height < 80e3:
            absorb = absorb0 * np.exp(-5e-4 * 12e3)
        else:
            absorb = 0

        return absorb

    @staticmethod
    def scatter_height(scatter0, height):
        if height < 12e3:
            scatter = scatter0 * np.exp(-8e-4 * height)
        elif height < 80e3:
            scatter = scatter0 * np.exp(-8e-4 * 12e3)
        else:
            scatter = 0

        return scatter

    @staticmethod
    def wind_height(wind0, height):
        wind = wind0 + 30 * (np.exp(-((height - 9400) / 4800) ** 2) - np.exp(-(94 / 48) ** 2))

        return wind

    @staticmethod
    def Cn2_height(Cn20, wind0, height):
        v = (wind0 ** 2 + 30.69 * wind0 + 348.91) ** 2
        Cn2 = (Cn20 + 8.148e-56 * v ** 2 * height ** 10) * np.exp(-height / 100) + \
                    2.7e-16 * np.exp(-height / 1500)

        return Cn2

    @staticmethod
    def cs_height(T0, height):
        T = EnvParams.temperature_height(T0, height)
        Cs2 = 331.3 ** 2 * T / 273.15

        return Cs2 ** 0.5

    @staticmethod
    def density_height(rho0, height):
        rho = rho0 * np.exp(-height / 8430)

        return rho

    @staticmethod
    def atm_height(rho0, T0, height):
        rho = EnvParams.density_height(rho0, height)
        T = EnvParams.temperature_height(T0, height)
        atm = rho * T / 273.15 / 1.293

        return atm

    @staticmethod
    def L0_height(height):
        if height <= 1:
            L0 = 0.4
        elif height <= 25:
            L0 = 0.4 * height
        else:
            L0 = 2 * height ** 0.5

        return L0

    @staticmethod
    def get_env_height(env0, height):
        env = deepcopy(env0)

        env.absorb = EnvParams.absorb_height(env0.absorb, height)
        env.scatter = EnvParams.scatter_height(env0.scatter, height)
        env.density = EnvParams.density_height(env0.density, height)
        env.temperature = EnvParams.temperature_height(env0.temperature, height)
        env.atm = EnvParams.atm_height(env0.density, env0.temperature, height)
        env.Cs2 = EnvParams.cs_height(env.temperature, height) ** 2
        env.L0 = EnvParams.L0_height(height)

        wind = env0.wind_x if env0.wind_x != 0 else env0.wind_y
        if env0.wind_x == 0 and env0.wind_y == 0:
            env.wind_x = EnvParams.wind_height(wind, height)
        elif env.wind_x == 0:
            env.wind_y = EnvParams.wind_height(wind, height)
        else:
            env.wind_x = EnvParams.wind_height(wind, height)
        env.Cn2 = EnvParams.Cn2_height(env0.Cn2, wind, height)

        return env
