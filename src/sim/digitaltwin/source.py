"""
提供激光器的各种强度分布函数
包括：
高斯光、平顶光、拉盖尔高斯光
"""
import copy

import numpy as np


def gauss(r, twist):
    """
    高斯光源分布

    :param r: 到原点距离网格
    :param twist: 高斯束腰半径
    :return: source: 高斯分布
    """
    source = np.exp(-(r / twist) ** 2)

    return source


def flat(r, twist):
    """
    平顶光源分布

    :param r: 到原点距离网格
    :param twist: 平顶光半径
    :return: source: 平顶分布
    """
    source = (np.sign(twist - r) + 1) / 2

    return source


def laguerre_gaussian(r, theta, twist, charges):
    """
    拉盖尔高斯光源分布

    :param r: 到原点距离网格
    :param theta: 角度网格
    :param twist: 高斯束腰半径
    :param charges: 拓扑荷数
    :return: source: 拉盖尔高斯分布
    """

    source = (r / twist) ** np.abs(charges) * np.exp(-(r / twist) ** 2) * np.exp((1j * charges * theta))

    return source


def sphere(r, lamd, z):
    """
    点光源在距离z处平面的波前

    :param r: 到原点距离网格
    :param lamd: 波长
    :param z: 波前面到点光源距离
    :return: source: 球面波波前
    """
    k = np.pi * 2 / lamd
    dist = (r ** 2 + z ** 2) ** 0.5
    source = 1 / dist * np.exp(1j * k * dist)

    return source


def cylinder(r, lamd, z):
    """
    点光源在距离z处平面的波前

    :param r: 到原点距离网格
    :param lamd: 波长
    :param z: 波前面到点光源距离
    :return: source: 球面波波前
    """
    k = np.pi * 2 / lamd
    dist = (r ** 2 + z ** 2) ** 0.5
    source = 1 / dist ** 0.5 * np.exp(1j * k * dist)

    return source


def pin_like(r, lamd, twist, law, phase_var):
    """
    锋芒光源分布

    :param r: 到原点距离网格
    :param lamd: 波长
    :param twist: 高斯束腰半径
    :param law: 幂律指数
    :param phase_var: 相位变化因子
    :return: source: 锋芒光束分布
    """

    k = 2 * np.pi / lamd

    r = copy.deepcopy(r)
    r[np.where(r < 1e-15)] = r[int(len(r)/2 + 1), int(len(r)/2)] / 2
    amp = r ** (-law / 2)
    phi = - k * phase_var * (r / twist) ** law
    source = amp * np.exp(1j * phi)

    return source


def customized(wf):
    """
    自定义波前

    :param wf: 自定义波前
    :return: source: 自定义波前分布
    """

    source = wf

    return source
