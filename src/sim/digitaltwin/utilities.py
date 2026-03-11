"""
各种公用函数
"""
import math
import numpy as np
from scipy import constants as const
from scipy import ndimage


def tri(N, M=0, k=0, dtype=int):
    """
    Create a lower triangular matrix.
    This is a replacement for the deprecated scipy.linalg.tri function.
    
    :param N: Number of rows
    :param M: Number of columns (default: N)
    :param k: Diagonal offset (k=0 is main diagonal, k>0 is above, k<0 is below)
    :param dtype: Data type
    :return: Triangular matrix
    """
    if M == 0:
        M = N
    # Create a matrix with ones in the lower triangle
    arr = np.arange(1, N + 1, dtype=dtype)[:, None] > np.arange(M, dtype=dtype)
    # Shift by k positions
    if k > 0:
        arr = np.hstack((np.ones((N, k), dtype=dtype), arr))[:, :M]
    elif k < 0:
        arr = np.vstack((np.ones((-k, M), dtype=dtype), arr))[(-k):, :]
    return arr.astype(dtype)


def grid(npix, dpix):
    """
    计算空间域和频域的网格

    :param npix: 像素数
    :param dpix: 像素尺寸
    :return: x, y, r, qx, qy, qr: 空间域和频域的网格
    """
    sl = npix * dpix
    x = np.linspace(-sl / 2, sl / 2 - dpix, npix)
    x, y = np.meshgrid(x, x)

    qx = np.fft.fftfreq(npix, dpix) * 2 * np.pi
    qx, qy = np.meshgrid(qx, qx)

    r = (x ** 2 + y ** 2) ** 0.5
    qr = (qx ** 2 + qy ** 2) ** 0.5

    return x, y, r, qx, qy, qr


def get_air_refractive(wavelength, temperature=288.15, atm=1, humidity=0, co2=450):
    """
    计算对应波长、温度、压强下的空气折射率
    Original data: Birch and Downs 1994, https://doi.org/10.1088/0026-1394/31/4/006

    :param wavelength: 波长, [0.3, 1.69] e3 nm
    :param temperature: 温度, [-40, 100] - 273.15 K
    :param atm: 压强, [80000, 120000] / 101325 atm
    :param humidity: 湿度, [0, 1]
    :param co2: 二氧化碳含量, [0, 2000] ppm
    """
    pa = atm * 101324.9966
    lamd = wavelength * 1e6
    t = temperature - 273.15
    esT = 6.112e-3 * np.exp(17.62 * t / (243.12 + t))
    pv = esT * humidity

    A, B, C, D, E, F, G = 8342.54, 2406147, 15998, 96095.43, 0.601, 0.00972, 0.003661

    ns = 1 + 1e-8 * (A + B / (130 - lamd ** (-2)) + C / (38.9 - lamd ** (-2)))
    ntp = 1 + pa * (ns - 1) * (1 + 1e-8 * (E - F * t) * pa) / (1 + G * t) / D
    n_air = ntp - pv * (3.73345 - 0.0401 / lamd ** 2) * 1e-10

    return n_air


def tl_dz_limit(Cn2, lamd):
    """

    :param Cn2: 折射率结构常数
    :param lamd: 波长
    :return: dz，湍流相位屏间隔的最大距离
    """

    k = 2 * np.pi / lamd
    dz = (0.1 / 1.23 / Cn2 / k ** (7/6)) ** (6/11)

    return dz


def wf2intensity(wavefront, refractive):
    """
    计算光强矩阵

    :param wavefront: 波前矩阵
    :param refractive: 环境折射率
    :return: intensity: 光强矩阵
    """
    intensity = const.c * const.epsilon_0 * refractive * np.abs(wavefront) ** 2 / 2
    return intensity


def wf2power(wavefront, dpix, refractive):
    """
    计算功率

    :param wavefront: 波前矩阵
    :param dpix: 波前像素尺寸
    :param refractive: 环境折射率
    :return: power: 总功率
    """
    intensity = wf2intensity(wavefront, refractive)
    power = dpix ** 2 * intensity.sum()

    return power



# 请确保vx和vy不同时不为0
def line_integral(func, vx, vy):
    """
    沿x/y风速方向积分

    :param func: 待积分函数
    :param vx: x轴风速
    :param vy: y轴风速
    :return integral: 积分结果
    """
    if np.abs(vx) > 1e-10 and np.abs(vy) > 1e-10:
        raise ValueError('one of vx and vy must be zero')

    npix = len(func)
    integral = np.zeros((npix, npix))

    if vx > 1e-10:
        for i in range(npix):
            for j in range(npix):
                integral[i, j] = np.sum(func[i, 0:j])
    elif vx < -1e-10:
        for i in range(npix):
            for j in range(npix):
                integral[i, j] = np.sum(func[i, j + 1:])

    elif vy > 1e-10:
        for i in range(npix):
            for j in range(npix):
                integral[i, j] = np.sum(func[0:i, j])
    elif vy < -1e-10:
        for i in range(npix):
            for j in range(npix):
                integral[i, j] = np.sum(func[i + 1:, j])

    return integral


def meter2unit(value, unit):
    """
    单位转换，将m转换为对应单位

    :param value: 单位为m的值
    :param unit: 待转换的单位，可选值：'m'，'cm'，'mm'，'km'
    :return transform: 单位转换后的值
    """
    if unit == 'm':
        scale = 1
    elif unit == 'cm':
        scale = 100
    elif unit == 'mm':
        scale = 1e3
    elif unit == 'km':
        scale = 1e-3
    else:
        raise ValueError('unit is cm, mm, or km')

    transform = value * scale

    return transform


def w_sqr_m2unit(value, unit):
    """
    单位转换，将W/m2转换为对应单位

    :param value: 单位为W/m2的值
    :param unit: 待转换的单位，可选值：'KW/cm2'，'W/cm2'
    :return transform: 单位转换后的值
    """
    if unit == 'KW/cm2':
        scale = 1e-7
    elif unit == 'W/cm2':
        scale = 1e-4
    else:
        raise ValueError('unit is KW/cm2 or W/cm2')

    transform = value * scale

    return transform


class _Wave:
    def __init__(self):
        self.npix = None
        self.dpix = None
        self.x = None
        self.y = None
        self.r = None
        self.qx = None
        self.qy = None
        self.qr = None

        self.wavefront = None
        self.ex = None
        self.ey = None

        self.refractive = None
        self.wavelength = None

    @property
    def side_length(self):
        return self.npix * self.dpix

    @property
    def intensity(self):
        intensity = wf2intensity(self.wavefront, self.refractive)
        return intensity

    @property
    def power(self):
        power = self.intensity.sum() * self.dpix ** 2
        return power

    @property
    def lamd(self):
        lamd = self.wavelength / self.refractive
        return lamd

    @property
    def wavenumber(self):
        k = 2 * np.pi / self.lamd
        return k

    @property
    def freq(self):
        freq = const.c / self.wavelength
        return freq

    def change_grid(self, npix, dpix):
        self.x, self.y, self.r, self.qx, self.qy, self.qr = grid(npix, dpix)
        self.npix = npix
        self.dpix = dpix

    def change_wf(self, scale=1, phase=0):
        self.wavefront = scale * self.wavefront * np.exp(1j * phase)


def wf2wave(wavefront, wavelength, npix, dpix, refractive):
    """
    根据波前产生wave，方便传参

    :param wavefront: 波前
    :param wavelength: 波长
    :param npix: 波前像素数
    :param dpix: 波前像素尺寸
    :param refractive: 波折射率
    :return wave: 产生wave
    """
    wave = _Wave()
    wave.change_grid(npix, dpix)
    wave.wavelength = wavelength
    wave.refractive = refractive
    wave.wavefront = wavefront

    return wave


class _TimeWave(_Wave):
    def __init__(self):
        super(_TimeWave, self).__init__()
        self.time = 0
        self.wave0 = None

    @property
    def omega(self):
        omega = 2 * np.pi * self.freq
        return omega

    def time_update(self, delta_time):
        if isinstance(self.wave0, _Wave):
            raise TypeError('wave0 must be Wave type')

        self.time += delta_time
        self.wavefront = self.wave0.wavefront * np.exp(1j * self.omega * self.time)
        self.change_grid(self.wave0.npix, self.wave0.dpix)


def wf2timewave(wavefront, wavelength, npix, dpix, refractive):
    """
    根据波前产生wave，方便传参

    :param wavefront: 波前
    :param wavelength: 波长
    :param npix: 波前像素数
    :param dpix: 波前像素尺寸
    :param refractive: 波折射率
    :return wave: 产生wave
    """

    wave = _TimeWave()
    wave.change_grid(npix, dpix)
    wave.wavelength = wavelength
    wave.refractive = refractive
    wave.wavefront = wavefront

    return wave


def phase2wave(phase, wavelength, npix, dpix, refractive):
    """
    根据相位产生wave，方便传参

    :param phase: 相位，单位rad
    :param wavelength: 波长
    :param npix: 波前像素数
    :param dpix: 波前像素尺寸
    :param refractive: 波折射率
    :return wave: 产生wave
    """

    wave = _Wave()
    wave.change_grid(npix, dpix)
    wave.wavelength = wavelength
    wave.refractive = refractive
    wave.wavefront = np.exp(1j * phase)

    return wave


def intensity2wf(intensity, refractive):
    """
    根据光强计算波前实部

    :param intensity: 光强矩阵
    :param refractive: 环境折射率
    :return: wavefront: 波前矩阵
    """
    wavefront = (intensity / const.c / const.epsilon_0 / refractive * 2) ** 0.5

    return wavefront


def wf_shift(matrix, shift_pix):
    """
    矩阵平移像素

    :param matrix: 待平移矩阵
    :param shift_pix: 需要平移的像素数，(-3, 5)，第一位代表x方向，正为右，第二位代表y方向，正为上，
    :return matrix_trans: 平移后矩阵
    """

    npix = len(matrix)
    x_matrix = tri(npix, npix, -shift_pix[0], dtype=int) - tri(npix, npix, -shift_pix[0]-1, dtype=int)
    y_matrix = tri(npix, npix, -shift_pix[1], dtype=int) - tri(npix, npix, -shift_pix[1]-1, dtype=int)

    matrix_trans = np.dot(matrix, x_matrix)
    matrix_trans = np.dot(y_matrix, matrix_trans)

    return matrix_trans


def wf_size_double(wavefront):
    """
    波前尺寸占比放大2倍（dpix为0.5倍）

    :param wavefront: 波前
    :return wf_trans: 变换后波前
    """
    npix = len(wavefront)
    if npix % 2 != 0:
        raise ValueError('npix of wf must be even')

    npix_half = int(npix / 2)
    temp = ndimage.zoom(wavefront, 2, order=0)
    wf_trans = temp[npix_half: npix + npix_half, npix_half: npix + npix_half]

    return wf_trans


def matrix_size_trans(matrix, dpix_init, npix_trans, dpix_trans):
    """
    矩阵尺寸分辨率变换，不适合波前，适合相位

    :param matrix: 待变换波前或相位矩阵
    :param dpix_init: 初始矩阵像素尺寸
    :param npix_trans: 变换后矩阵像素数
    :param dpix_trans: 变换后矩阵像素尺寸
    :return matrix_trans: 变换后矩阵
    """
    factor = dpix_init / dpix_trans
    temp = ndimage.zoom(matrix, factor, order=3)

    npix_diff = len(temp) - npix_trans
    npix_half = int(np.round(npix_diff / 2))
    if npix_diff < 0:
        matrix_trans = np.pad(temp, (-npix_half, -npix_diff + npix_half),
                              mode='constant', constant_values=(0,))
    elif npix_diff > 0:
        matrix_trans = temp[npix_half: npix_trans + npix_half,
                            npix_half: npix_trans + npix_half]
    else:
        matrix_trans = temp

    return matrix_trans


def wf_add_zeros(wavefront, zeros_layers):
    """
    波前矩阵添0使得npix为4的倍数

    :param wavefront: 波前矩阵
    :param zeros_layers: 左/上和右/下添0的数量，如(2,1)表示左和上2层右和下1层，一个数表示层数相同
    :return wf_trans: 添0后矩阵
    """
    if isinstance(zeros_layers, tuple):
        wf_trans = np.pad(wavefront, (zeros_layers[0], zeros_layers[1]),
                          mode='constant', constant_values=(0,))
    else:
        wf_trans = np.pad(wavefront, (zeros_layers, zeros_layers),
                          mode='constant', constant_values=(0,))

    return wf_trans


def wave_angle_spectrum(wave, dist, extinction=0):
    """
    角谱法自由空间传输

    :param wave: 传输波
    :param dist: 传输距离
    :param extinction: 环境消光系数
    """
    k = 2 * np.pi / wave.lamd

    transfer_mat = np.exp(1j * dist * np.sqrt(k ** 2 - wave.qr ** 2 + 0j))
    if np.abs(wave.wavefront).sum() > 1e-10:
        wf_fft = np.fft.fft2(wave.wavefront)
        wavefront = np.fft.ifft2(wf_fft * transfer_mat)
        wave.wavefront = wavefront * np.exp(-extinction * dist)
        wave.wavefront *= _scatter_boundary(wave.r, wave.npix * wave.dpix)

    if wave.ex is not None:
        wf_fft = np.fft.fft2(wave.ex)
        wavefront = np.fft.ifft2(wf_fft * transfer_mat)
        wave.ex = wavefront * np.exp(-extinction * dist)
        wave.ex *= _scatter_boundary(wave.r, wave.npix * wave.dpix)

        wf_fft = np.fft.fft2(wave.ey)
        wavefront = np.fft.ifft2(wf_fft * transfer_mat)
        wave.ey = wavefront * np.exp(-extinction * dist)
        wave.ey *= _scatter_boundary(wave.r, wave.npix * wave.dpix)


def _scatter_boundary(r, side_length):
    """
    散射边界条件

    :param r: 空间域距离坐标
    :param side_length: 屏边长
    :return scatter_mat: 吸收矩阵
    """
    sigma = 0.451 * side_length
    scatter_mat = np.exp(-(r / sigma) ** 16.0)

    return scatter_mat


def wave_matrix_prop(wave, dist, npix, dpix, extinction=0):
    """
    矩阵法自由空间传输

    :param wave: 传输波
    :param dist: 传输距离
    :param npix: 输出面像素数
    :param dpix: 输出面像素尺寸
    :param extinction: 环境消光系数
    """
    k = 2 * np.pi / wave.lamd

    xi, yi = np.mat(wave.x), np.mat(wave.y)
    xo, yo, ro, _, _, _ = grid(npix, dpix)
    centroid = wave.centroid
    r = np.sqrt((xo - centroid[0]) ** 2 + (yo - centroid[1]) ** 2)
    mask = (np.sign(wave.lamd * dist / wave.dpix / 2 - r) + 1) / 2
    xo, yo = np.mat(xo), np.mat(yo)

    mat_x = np.exp(-1j * k / dist * xo[0].T * xi[0])
    mat_y = np.exp(-1j * k / dist * yi[:, 0] * yo[:, 0].T)

    if np.abs(wave.wavefront).sum() > 1e-10:
        mat = np.mat(wave.wavefront * np.exp(1j * k / dist / 2 * (wave.r ** 2)))
        dot = np.array(mat_x * mat * mat_y) * wave.dpix ** 2
        amplitude = -1j / wave.lamd / dist * np.exp(1j * k * dist) * \
                    np.exp(1j * k / 2 / dist * ro ** 2)
        wave.wavefront = amplitude * dot * np.exp(-extinction * dist / 2) * mask

    if wave.ex is not None:
        mat = np.mat(wave.ex * np.exp(1j * k / dist / 2 * (wave.r ** 2)))
        dot = np.array(mat_x * mat * mat_y * wave.dpix ** 2)
        amplitude = -1j / wave.lamd / dist * np.exp(1j * k * dist) * \
                    np.exp(1j * k / 2 / dist * ro ** 2)
        wave.ex = amplitude * dot * np.exp(-extinction * dist / 2) * mask

        mat = np.mat(wave.ey * np.exp(1j * k / dist / 2 * (wave.r ** 2)))
        dot = np.array(mat_x * mat * mat_y * wave.dpix ** 2)
        amplitude = -1j / wave.lamd / dist * np.exp(1j * k * dist) * \
                    np.exp(1j * k / 2 / dist * ro ** 2)
        wave.ey = amplitude * dot * np.exp(-extinction * dist / 2) * mask

    wave.change_grid(npix, dpix)
