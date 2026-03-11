from zernike import RZern
import numpy as np
import sim.digitaltwin as dt
import math
##  优化代码
##  桶半径自适应
#  γ自适应  文献表明固定增益快于自适应增益（牺牲校正精度加强校正速度）  《基于Zernike模式的随机并行梯度下降算法的收敛速率》


class DM:
    def __init__(self, Rn, x, y, delta, gamma, r_bucket, phase_init=None, aber_phi=None):
        self.cart = RZern(Rn)
        self.cart.make_cart_grid(x, y)
        if phase_init is None:
            self.phase = np.zeros((len(x), len(y)))
        else:
            self.phase = phase_init
        self.c = self.phase2zernike(self.phase)
        if aber_phi is None:
            aber_phi = np.zeros((len(x), len(y)))
        self.c_scale = np.abs(self.phase2zernike(aber_phi))
        self.c_scale = np.ones(self.cart.nk)
        self.disturb_c = self.c_scale

        self.delta = delta
        self.gamma = gamma
        self.flag = 0
        self.pos = 0
        self.neg = 0
        self.J = 0

        self.r_bucket = r_bucket

    def zernike2phase(self, c):
        phi = np.array(self.cart.eval_grid(c, matrix=True))
        return phi

    def phase2zernike(self, phi):
        c = self.cart.fit_cart_grid(phi)[0]
        return c

    def disturb_initial(self, N):
        disturb_init = np.random.binomial(1, 0.5, N)
        disturb_init[disturb_init == 0] = -1
        disturb_init = self.delta * disturb_init * self.c_scale
        return disturb_init

    def loss(self, wave, r_bucket):
        pb = dt.params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, 'origin', r_bucket)
        return pb

    def spgd(self, wave):
        if self.flag == 0:
            self.disturb_c = self.disturb_initial(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.c = self.c - self.disturb_c
            self.phase = self.zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            self.c = self.c + self.gamma * (self.pos - self.neg) * self.disturb_c + self.disturb_c / 2
            self.phase = self.zernike2phase(self.c)
            self.flag = 0

        return self.flag, self.J, self.c, self.phase

    def out(self, wave):
        if not isinstance(wave, list):
            wave.change_wf(phase=self.phase)
        else:
            for i in range(len(wave)):
                wave[i].change_wf(phase=self.phase)


class AdamDM:
    def __init__(self, Rn, x, y, delta, alpha, beta1, beta2, r_bucket, phase_init=None, aber_phi=None):
        """

        :param Rn:
        :param x:
        :param y:
        :param delta:
        :param alpha: spgd的γ，adam的α
        :param beta1:
        :param beta2:
        :param r_bucket: 固定桶中功率的半径，自适应桶中功率半径/光斑86.5%半径的比
        :param phase_init: DM初始相位
        :param aber_phi: 初始像差相位
        """
        self.cart = RZern(Rn)
        self.cart.make_cart_grid(x, y)
        if phase_init is None:
            self.phase = np.zeros((len(x), len(y)))
        else:
            self.phase = phase_init
        self.c = self.phase2zernike(self.phase)
        
        # Apply initial aberration if provided
        if aber_phi is not None:
            c_aber = self.phase2zernike(aber_phi)
            self.c = self.c + c_aber
            self.phase = self.zernike2phase(self.c)

        self.disturb_c = np.zeros(self.cart.nk)
        self.m = np.zeros(self.cart.nk)
        self.v = np.ones(self.cart.nk)
        self.t = 0
        self.delta = delta
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.flag = 0
        self.pos = 0
        self.neg = 0
        self.J = 0

        self.r_bucket = r_bucket

    def zernike2phase(self, c):
        phi = np.array(self.cart.eval_grid(c, matrix=True))
        return phi

    def phase2zernike(self, phi):
        c = self.cart.fit_cart_grid(phi)[0]
        return c

    def disturb_initial(self, N):
        disturb_init = np.random.binomial(1, 0.5, N)
        disturb_init[disturb_init == 0] = -1
        disturb_init = self.delta * disturb_init
        return disturb_init

    # 桶中功率
    # def loss(self, wave, r_bucket):
    #     pb = dt.params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, 'origin', r_bucket)
    #     return pb

    # 自适应桶中功率
    def loss(self, wave, radius_scale):
        radius = dt.params.WaveIndex.radius(wave.intensity, wave.x, wave.y, 'origin', 0.865) * radius_scale
        pb = dt.params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, 'origin', radius)
        return pb

    def spgd(self, wave):
        if self.flag == 0:
            self.disturb_c = self.disturb_initial(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.c = self.c - self.disturb_c
            self.phase = self.zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            self.c = self.c + self.alpha * (self.pos - self.neg) * self.disturb_c + self.disturb_c / 2
            self.phase = self.zernike2phase(self.c)
            self.flag = 0

    def adam(self, wave):
        if self.flag == 0:
            self.t = self.t + 1
            self.disturb_c = self.disturb_initial(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.c = self.c - self.disturb_c
            self.phase = self.zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            grad = (self.pos - self.neg) / self.disturb_c
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = self.v * self.beta2 + (1 - self.beta2) * grad ** 2
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
            self.c = self.c - self.alpha * m / (v ** 0.5 + 1e-8) + self.disturb_c / 2
            self.phase = self.zernike2phase(self.c)
            self.flag = 0

        return self.flag, self.J, self.c, self.phase

    def adamax(self, wave):
        if self.flag == 0:
            self.t = self.t + 1
            self.disturb_c = self.disturb_initial(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.c = self.c - self.disturb_c
            self.phase = self.zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            grad = (self.pos - self.neg) / self.disturb_c
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = np.max((self.beta2 * self.v, np.abs(grad)), axis=0)
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
            self.c = self.c - self.alpha / (1 - self.beta1 ** self.t) * m / v + self.disturb_c / 2
            self.phase = self.zernike2phase(self.c)
            self.flag = 0

        return self.flag, self.J, self.c, self.phase

    def out(self, wave):
        if not isinstance(wave, list):
            wave.change_wf(phase=self.phase)
        else:
            for i in range(len(wave)):
                wave[i].change_wf(phase=self.phase)


class voltageAdam:
    def __init__(self, layers, diam, omega, law, x, y, delta, alpha, beta1, beta2, r_bucket, phase_init=None):
        """

        :param layers: 驱动器层数，奇数
        :param diam: DM直径
        :param omega: DM参数，0.08
        :param law: DM参数，2
        :param x:
        :param y:
        :param delta:
        :param alpha: spgd的γ，adam的α
        :param beta1:
        :param beta2:
        :param r_bucket: 固定桶中功率的半径，自适应桶中功率半径/光斑86.5%半径的比
        :param phase_init: DM初始相位
        """
        self.x, self.y = x, y
        self.xi, self.yi, self.dimension, self.d = self.position(layers, diam)
        self.omega = omega
        self.law = law

        if phase_init is None:
            self.phase = np.zeros((len(self.x), len(self.x)))
        else:
            self.phase = phase_init

        self.voltage = np.zeros(self.dimension)
        self.disturb_u = np.zeros(self.dimension)
        self.m = np.zeros(self.dimension)
        self.v = np.ones(self.dimension)
        self.t = 0
        self.delta = delta
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.flag = 0
        self.pos = 0
        self.neg = 0
        self.J = 0

        self.r_bucket = r_bucket


    @staticmethod
    def position(layers, D):
        half_layer = int((layers - 1) / 2)
        a = D / half_layer * 2 / 13 ** 0.5
        yd = a / 2 * 3 ** 0.5
        N = 3 * half_layer ** 2 + 3 * half_layer + 1
        xi = np.zeros(N)
        yi = np.zeros(N)
        m = 0
        for i in range(half_layer):
            num = i + half_layer + 1
            for j in range(num):
                xi[m] = -(num - 1) / 2 * a + a * j
                yi[m] = (half_layer - i) * yd
                xi[m + 1] = -(num - 1) / 2 * a + a * j
                yi[m + 1] = -(half_layer - i) * yd
                m = m + 2
        for i in range(2 * half_layer + 1):
            xi[m] = -half_layer * a + i * a
            yi[m] = 0
            m = m + 1
        return xi, yi, N, a

    def voltage2phase(self, voltage):
        opd = 0
        for i in range(self.dimension):
            V = np.exp(np.log(self.omega) * (((self.x - self.xi[i]) ** 2 + (self.y - self.yi[i]) ** 2) ** 0.5 / self.d) ** self.law)
            opd = opd + V * voltage[i]
        return opd

    def disturb_initial(self, N):
        disturb_init = np.random.binomial(1, 0.5, N)
        disturb_init[disturb_init == 0] = -1
        disturb_init = self.delta * disturb_init
        return disturb_init

    # 桶中功率
    def loss(self, wave, r_bucket):
        pb = dt.params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, 'origin', r_bucket)
        return pb

    # 自适应桶中功率
    # def loss(self, wave, radius_scale):
    #     radius = dt.params.WaveIndex.radius(wave.intensity, wave.x, wave.y, 'origin', 0.865) * radius_scale
    #     pb = dt.params.WaveIndex.power_bucket(wave.intensity, wave.x, wave.y, 'origin', radius)
    #     return pb

    def spgd(self, wave):
        if self.flag == 0:
            self.disturb_u = self.disturb_initial(self.dimension)
            self.voltage = self.voltage + self.disturb_u / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.voltage = self.voltage - self.disturb_u
            self.phase = self.voltage2phase(self.voltage)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            self.voltage = self.voltage + self.alpha * (self.pos - self.neg) * self.disturb_u + self.disturb_u / 2
            self.phase = self.voltage2phase(self.voltage)
            self.flag = 0

    def adam(self, wave):
        if self.flag == 0:
            self.t = self.t + 1
            self.disturb_u = self.disturb_initial(self.dimension)
            self.voltage = self.voltage + self.disturb_u / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.voltage = self.voltage - self.disturb_u
            self.phase = self.voltage2phase(self.voltage)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            grad = (self.pos - self.neg) / self.disturb_u
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = self.v * self.beta2 + (1 - self.beta2) * grad ** 2
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
            self.voltage = self.voltage - self.alpha * m / (v ** 0.5 + 1e-8) + self.disturb_u / 2
            self.phase = self.voltage2phase(self.voltage)
            self.flag = 0

        return self.flag, self.J, self.voltage, self.phase

    def adamax(self, wave):
        if self.flag == 0:
            self.t = self.t + 1
            self.disturb_u = self.disturb_initial(self.dimension)
            self.voltage = self.voltage + self.disturb_u / 2
            self.flag = 1

        if self.flag == 1:
            self.pos = self.loss(wave, self.r_bucket)
            self.voltage = self.voltage - self.disturb_u
            self.phase = self.voltage2phase(self.voltage)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(wave, self.r_bucket)
            self.J = (self.pos + self.neg) / 2
            grad = (self.pos - self.neg) / self.disturb_u
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = np.max((self.beta2 * self.v, np.abs(grad)), axis=0)
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
            self.voltage = self.voltage - self.alpha / (1 - self.beta1 ** self.t) * m / v + self.disturb_u / 2
            self.phase = self.voltage2phase(self.voltage)
            self.flag = 0

        return self.flag, self.J, self.voltage, self.phase

    def out(self, wave):
        if not isinstance(wave, list):
            wave.change_wf(phase=self.phase)
        else:
            for i in range(len(wave)):
                wave[i].change_wf(phase=self.phase)
