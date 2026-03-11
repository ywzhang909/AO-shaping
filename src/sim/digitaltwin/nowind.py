import numpy as np


def zero_wind_prop(intensity, r, alpha, distance, D, power, lamd, refractive, nT):
    k = 2 * np.pi / lamd
    a = D / 2 / np.sqrt(2)

    Dc = -nT * power * alpha * distance ** 2 / 2 / np.pi / k / refractive / a ** 2
    Iz = intensity * np.exp(-alpha * distance - Dc * np.exp(-r ** 2 / a ** 2))

    return Iz
