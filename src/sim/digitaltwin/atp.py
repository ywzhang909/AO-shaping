"""
绘大气传输模型，设置相位屏
"""
import pickle

import numpy as np
import math
from copy import deepcopy

from sim.digitaltwin import screens
from sim.digitaltwin import utilities as utils


class ATP:
    def __init__(self, env_init, prop_dist, layers, height=0, emission_angle=0, set_env='auto', env_array=None,
                 Turbulent=False, Thermal=False, tl_harmonic=1, tb_mode='Green'):
        """
        ATP传输模拟，由于折射率与密度关系，仅能计算气体环境的传输

        :param env_init: 发射处大气环境参数
        :param prop_dist: 传输距离
        :param layers: 将路径分为layers层
        :param height: 发射处海拔高度，默认0m
        :param emission_angle: 发射方向与水平面夹角，默认0°
        :param set_env: 环境设置方法，默认'auto'，可选'manual'
        :param env_array: 仅当set_env为'manual'，表示手动输入的环境列表，默认None
        :param Turbulent: 是否计算湍流，默认False
        :param Thermal: 是否计算热晕，默认False
        :param tl_harmonic: 湍流屏的次谐波相位屏阶数，默认1
        :param tb_mode: 热晕相位屏的求解器，默认'Green'，可选'FFT_Isobaric'，'FFT_non_Isobaric'
        """

        self.wave = None

        self.prop_dist = prop_dist
        self.layers = layers
        self.layer_length = prop_dist / layers

        self.Turbulent = Turbulent
        self.tl_harmonic = tl_harmonic
        self.Thermal = Thermal
        self.tb_mode = tb_mode

        if set_env == 'auto':
            self.env_array = self.auto_set_env(env_init, height, emission_angle)
        elif set_env == 'manual':
            self.env_array = env_array
        else:
            raise ValueError('The value of set_env is wrong set')

        self.tb_screens = []
        self.tl_screens = []

    def auto_set_env(self, env_init, height, angle):
        """
        根据近地面Cn2计算相位屏法不同高度的每一层的Cn2

        :param env_init: 初始环境
        :param height: 初始海拔
        :param angle: 传输角度
        :return env_array: 环境列表
        """

        height_array = np.linspace(self.layer_length / 2, self.layers * self.layer_length - self.layer_length / 2,
                                   self.layers) * np.sin(angle/180*np.pi) + height
        if height_array.min() < 0:
            raise ValueError('height is less than 0')

        Cn2_array, L0_array, T_array = self.Cn2_L0_T(env_init, height_array)
        Cs2_array = 331.3 ** 2 * T_array / 273.15
        density_array = env_init.density * np.exp(-height_array / 8430)
        atm_array = density_array * T_array / 273.15 / 1.293
        wind = env_init.wind_x if env_init.wind_x != 0 else env_init.wind_y
        wind_array = wind + 30 * (np.exp(-((height_array - 9400) / 4800) ** 2) - np.exp(-(94/48) ** 2))
        absorb_array, scatter_array = self.absorb_scatter(env_init, height_array)

        env_array = []
        for i in range(self.layers):
            env = deepcopy(env_init)
            env.absorb = absorb_array[i]
            env.scatter = scatter_array[i]
            env.density = density_array[i]
            env.temperature = T_array[i]
            env.atm = atm_array[i]
            env.Cs2 = Cs2_array[i]
            env.Cn2 = Cn2_array[i]
            env.L0 = L0_array[i]
            if env.wind_x == 0 and env.wind_y == 0:
                env.wind_x = wind_array[i]
            elif env.wind_x == 0:
                env.wind_y = wind_array[i]
            else:
                env.wind_x = wind_array[i]

            env_array.append(env)

        return env_array

    def Cn2_L0_T(self, env_init, height_array):
        """
        根据近地面Cn2计算相位屏法不同高度的每一层的Cn2

        :param env_init: 初始环境
        :param height_array: 海拔列表
        :return Cn2_array, L0_array, T_array: 不同高度的每一层的Cn2和L0和温度
        """

        wind = env_init.wind
        v = (wind ** 2 + 30.69 * wind + 348.91) ** 2
        Cn2_array = (env_init.Cn2 + 8.148e-56 * v ** 2 * height_array ** 10) * np.exp(-height_array / 100) + \
            2.7e-16 * np.exp(-height_array / 1500)

        L0_array = np.zeros(self.layers)
        for i in range(self.layers):
            if height_array[i] <= 1:
                L0_array[i] = 0.4
            elif height_array[i] <= 25:
                L0_array[i] = 0.4 * height_array[i]
            else:
                L0_array[i] = 2 * height_array[i] ** 0.5

        T_array = np.zeros(self.layers)
        for i in range(self.layers):
            if height_array[i] <= 11e3:
                T_array[i] = env_init.temperature - 0.00649 * height_array[i]
            else:
                T_array[i] = env_init.temperature - 0.00649 * 11e3

        return Cn2_array, L0_array, T_array

    def absorb_scatter(self, env_init, height_array):
        """
        根据近地面环境计算相位屏法不同高度的每一层的吸收散射系数

        :param env_init: 初始环境
        :param height_array: 海拔列表
        :return alpha_array, scatter_array: 不同高度的每一层的吸收和散射系数
        """
        alpha = env_init.absorb
        alpha_array = np.zeros(self.layers)
        for i in range(self.layers):
            if height_array[i] < 12e3:
                alpha_array[i] = alpha * np.exp(-5e-4 * height_array[i])
            elif height_array[i] < 80e3:
                alpha_array[i] = alpha * np.exp(-5e-4 * 12e3)
            else:
                alpha_array[i] = 0

        scatter = env_init.scatter
        scatter_array = np.zeros(self.layers)
        for i in range(self.layers):
            if height_array[i] < 12e3:
                scatter_array[i] = (scatter + alpha) * np.exp(-8e-4 * height_array[i]) - alpha_array[i]
            elif height_array[i] < 80e3:
                scatter_array[i] = (scatter + alpha) * np.exp(-8e-4 * 12e3) - alpha_array[i]
            else:
                scatter_array[i] = 0

        return alpha_array, scatter_array

    def free_propagation(self, dist, extinction):
        """
        自由空间传输，矩阵法限制：lamd*z/d0是两个光斑之间的距离，需要大于光斑直径

        :param dist: 传输距离
        :param extinction: 消光系数
        """
        if not isinstance(self.wave, list):
            utils.wave_angle_spectrum(self.wave, dist, extinction)
            # utils.wave_matrix_prop(self.wave, dist, self.wave.npix, self.wave.dpix)
        else:
            for i in range(len(self.wave)):
                utils.wave_angle_spectrum(self.wave[i], dist, extinction)

    # def free_propagation(self, dist, extinction):
    #     """
    #     自由空间传输，矩阵法限制：lamd*z/d0是两个光斑之间的距离，需要大于光斑直径
    #
    #     :param dist: 传输距离
    #     :param extinction: 消光系数
    #     """
    #
    #     if not isinstance(self.wave, list):
    #         for i in range(10):
    #             wave = deepcopy(self.wave)
    #             utils.wave_angle_spectrum(wave, dist, extinction)
    #             diameter = params.WaveIndex.radius(wave.intensity, wave.x, wave.y) * 2
    #             if diameter > wave.side_length * 0.8:
    #                 self.wave.resize(wave.npix, wave.dpix * 2)
    #             elif diameter < wave.side_length * 0.2:
    #                 self.wave.resize(wave.npix, wave.dpix / 2)
    #             else:
    #                 utils.wave_angle_spectrum(self.wave, dist, extinction)
    #                 break
    #     else:
    #         for i in range(10):
    #             wave = deepcopy(self.wave)
    #             for j in range(len(wave)):
    #                 utils.wave_angle_spectrum(wave[j], dist, extinction)
    #             intensity = params.WaveIndex.total_intensity(wave)
    #             diameter = params.WaveIndex.radius(intensity, wave[0].x, wave[0].y) * 2
    #             if diameter > wave[0].side_length * 0.8:
    #                 for j in range(len(wave)):
    #                     self.wave[j].resize(wave[j].npix, wave[j].dpix * 2)
    #             elif diameter < wave[0].side_length * 0.2:
    #                 for j in range(len(wave)):
    #                     self.wave[j].resize(wave[j].npix, wave[j].dpix / 2)
    #             else:
    #                 for j in range(len(self.wave)):
    #                     utils.wave_angle_spectrum(self.wave[j], dist, extinction)
    #                 break

    # def free_propagation(self, dist, extinction):
    #     """
    #     自由空间传输，矩阵法限制：lamd*z/d0是两个光斑之间的距离，需要大于光斑直径
    #
    #     :param dist: 传输距离
    #     :param extinction: 消光系数
    #     """
    #
    #     if not isinstance(self.wave, list):
    #         dpix, diameter = self.wave.dpix, 0
    #         for i in range(10):
    #             wave = deepcopy(self.wave)
    #             if i != 0:
    #                 wave.resize(wave.npix, dpix)
    #             utils.wave_angle_spectrum(wave, dist, extinction)
    #             diameter = params.WaveIndex.radius(wave.intensity, wave.x, wave.y) * 2
    #             plt.figure()
    #             plt.pcolor(wave.x, wave.y, wave.intensity)
    #             plt.show()
    #             if diameter / wave.side_length > 0.75:
    #                 dpix = dpix * 2
    #             elif diameter / wave.side_length < 0.25:
    #                 dpix = dpix / 2
    #             else:
    #                 break
    #
    #         utils.wave_matrix_prop(self.wave, dist, self.wave.npix, self.wave.dpix, diameter, extinction)
    #         plt.figure()
    #         plt.pcolor(self.wave.x, self.wave.y, self.wave.intensity)
    #         plt.show()
    #     else:
    #         dpix, diameter = self.wave[0].dpix, 0
    #         for i in range(10):
    #             wave = deepcopy(self.wave)
    #             for j in range(len(wave)):
    #                 if i != 0:
    #                     wave[j].resize(wave[j].npix, dpix)
    #                 utils.wave_angle_spectrum(wave[j], dist, extinction)
    #             intensity = params.WaveIndex.total_intensity(wave)
    #             diameter = params.WaveIndex.radius(intensity, wave[0].x, wave[0].y) * 2
    #             if diameter / wave[0].side_length > 0.75:
    #                 dpix = dpix * 2
    #             elif diameter / wave[0].side_length < 0.25:
    #                 dpix = dpix / 2
    #             else:
    #                 break
    #
    #         for i in range(len(self.wave)):
    #             utils.wave_matrix_prop(self.wave[i], dist, self.wave[i].npix, self.wave[i].dpix, diameter, extinction)

    def set_screen(self):
        """
        创建相位屏
        """

        if self.Thermal:
            for i in range(self.layers):
                self.tb_screens.append(screens.ThermalScreen(self.layer_length, self.env_array[i], self.tb_mode))

        if self.Turbulent:
            for i in range(self.layers):
                self.tl_screens.append(screens.TurbulentScreen(self.layer_length, self.env_array[i], self.tl_harmonic))

    def atp_prop(self):
        """
        大气传输
        """
        for i in range(self.layers):
            self.free_propagation(self.layer_length / 2, self.env_array[i].extinction)

            if self.Thermal:
                self.tb_screens[i].out(self.wave)
                with open('temp/tb_opd l{0:d}.pkl'.format(i), 'wb') as file:
                    pickle.dump(self.tb_screens[i].opd, file)
            if self.Turbulent:
                self.tl_screens[i].out(self.wave)
                with open('temp/tl_opd l{0:d}.pkl'.format(i), 'wb') as file:
                    pickle.dump(self.tl_screens[i].opd, file)

            self.free_propagation(self.layer_length / 2, self.env_array[i].extinction)

            with open('temp/wave l{0:d}.pkl'.format(i), 'wb') as file:
                pickle.dump(self.wave, file)

    def restrict(self, Cn2, L0, l0):
        """
        检查限制要求，相位屏间隔限制，采样尺寸小于内尺度

        :param Cn2: 折射率结构常数
        :param L0: 湍流外尺度
        :param l0: 湍流内尺度
        """

        if self.Turbulent:
            if not isinstance(self.wave, list):
                lamd = self.wave.lamd
                dpix = self.wave.dpix
            else:
                lamd, dpix = 0, 0
                for i in range(len(self.wave)):
                    if lamd < self.wave[i].lamd:
                        lamd = self.wave[i].lamd
                        dpix = self.wave[i].dpix

            bottom = L0
            top = (0.1 / 1.23 / Cn2 / (2 * np.pi / lamd) ** (7/6)) ** (6/11)

            try:
                if self.layer_length < bottom or self.layers > top:
                    raise Warning('Distance between screens should between ({:.2f}, {:.2f})'.format(bottom, top))
            except Warning:
                print('Distance between screens should between ({:.2f}, {:.2f})'.format(bottom, top))
            try:
                if l0 < dpix:
                    raise Warning('Pix scale {:.4f} is larger than inner scale {:.4f}'.format(self.wave.dpix, l0))
            except Warning:
                print('Pix scale {:.4f} is larger than inner scale {:.4f}'.format(self.wave.dpix, l0))

    def out(self, wave):
        self.wave = wave

        for i in range(self.layers):
            self.restrict(self.env_array[i].Cn2, self.env_array[i].L0, self.env_array[i].l0)

        if (self.Turbulent and len(self.tl_screens) == 0) or (self.Thermal and len(self.tb_screens) == 0):
            self.set_screen()

        self.atp_prop()

        for i in range(self.layers):
            self.restrict(self.env_array[i].Cn2, self.env_array[i].L0, self.env_array[i].l0)


class TimeATP(ATP):
    def __init__(self, total_time, delta_time, env_init, prop_dist, layers,
                 height=0, emission_angle=0, set_env='auto', env_array=None,
                 Turbulent=False, Thermal=False, tl_harmonic=1, save_mode='brief'):
        """
        时间ATP传输模拟，由于折射率与密度关系，仅能计算气体环境的传输

        :param total_time: 计算总时间
        :param delta_time: 时间间隔，在时间间隔内大气环境及激光状态近似不变
        :param env_init: 发射处大气环境参数
        :param prop_dist: 传输距离
        :param layers: 将路径分为layers层
        :param height: 发射处海拔高度，默认0m
        :param emission_angle: 发射方向与水平面夹角，默认0°
        :param set_env: 环境设置方法，默认'auto'，可选'manual'
        :param env_array: 仅当set_env为'manual'，表示手动输入的环境列表，默认None
        :param Turbulent: 是否计算湍流，默认False
        :param Thermal: 是否计算热晕，默认False
        :param tl_harmonic: 湍流屏的次谐波相位屏阶数，默认1
        :param save_mode: 保存模式，可选'brief'和'all'，表示保存最后时刻的结果/全保存
        """

        super(TimeATP, self).__init__(env_init, prop_dist, layers, height, emission_angle,
                                      set_env, env_array, Turbulent, Thermal, tl_harmonic, 'Green')

        self.time = 0
        self.total_time = total_time
        self.delta_time = delta_time

        self.save_mode = save_mode

    def set_screen(self):
        """
        创建相位屏
        """

        if self.Thermal:
            for i in range(self.layers):
                self.tb_screens.append(screens.TimeThermalScreen(
                    self.delta_time, self.layer_length, self.env_array[i]))

        if self.Turbulent:
            for i in range(self.layers):
                self.tl_screens.append(screens.TimeTurbulentScreen(
                    self.total_time, self.delta_time, self.layer_length, self.env_array[i], self.tl_harmonic))

    def atp_prop(self):
        time_num = math.floor(self.total_time / self.delta_time + 1)
        for t in range(time_num):
            if t != 0:
                self.time += self.delta_time
                if not isinstance(self.wave, list):
                    self.wave.time_update(self.delta_time)
                else:
                    for i in range(len(self.wave)):
                        self.wave[i].time_update(self.delta_time)
                for i in range(self.layers):
                    if self.Thermal:
                        self.tb_screens[i].time_update(self.delta_time)
                    if self.Turbulent:
                        self.tl_screens[i].time_update(self.delta_time)

            for i in range(self.layers):
                self.free_propagation(self.layer_length / 2, self.env_array[i].extinction)

                if self.Thermal:
                    self.tb_screens[i].out(self.wave)
                    if self.save_mode == 'all' or t == time_num:
                        with open('temp/tb_opd l{0:d} t{1:.6f}.pkl'.format(i, self.time), 'wb') as file:
                            pickle.dump(self.tb_screens[i].opd, file)
                if self.Turbulent:
                    self.tl_screens[i].out(self.wave)
                    if self.save_mode == 'all' or t == time_num:
                        with open('temp/tl_opd l{0:d} t{1:.6f}.pkl'.format(i, self.time), 'wb') as file:
                            pickle.dump(self.tl_screens[i].opd, file)
                with open('temp/wave l{0:d} t{1:.6f}.pkl'.format(i, self.time), 'wb') as file:
                    pickle.dump(self.wave, file)

                self.free_propagation(self.layer_length / 2, self.env_array[i].extinction)

            if self.save_mode == 'all' or t == time_num:
                with open('temp/wave l{0:d} t{1:.6f}.pkl'.format(self.layers, self.time), 'wb') as file:
                    pickle.dump(self.wave, file)

            print('time {0:.3f} s'.format(self.time))

    def out(self, wave):
        self.wave = wave
        if not isinstance(wave, list):
            self.wave.wave0 = deepcopy(wave)
        else:
            for i in range(len(wave)):
                self.wave[i].wave0 = deepcopy(wave[i])

        for i in range(self.layers):
            self.restrict(self.env_array[i].Cn2, self.env_array[i].L0, self.env_array[i].l0)

        if (self.Turbulent and len(self.tl_screens) == 0) or (self.Thermal and len(self.tb_screens) == 0):
            self.set_screen()

        self.atp_prop()

        for i in range(self.layers):
            self.restrict(self.env_array[i].Cn2, self.env_array[i].L0, self.env_array[i].l0)
