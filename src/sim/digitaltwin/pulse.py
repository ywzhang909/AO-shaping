import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as const


def intensity(wf, refractive):
    I = const.c * refractive * const.epsilon_0 * np.abs(wf) ** 2 / 2
    return I


def power(I, d):
    p = d ** 2 * I.sum()
    return p


def get_opd(I):
    rho1 = get_rho(I)
    opd = (n0 - 1.) / rho0 * rho1 * dz
    return opd


def get_Np():
    print(math.floor(total_time * fp), math.floor(D / vx * fp))
    return min(math.floor(total_time * fp), math.floor(D / vx * fp))


def get_rho(I):
    vk = vx * qx
    num = get_Np()
    intensity_fft = np.fft.fft2(I)
    c = -(gamma - 1) / cs2 * absorb

    s = 0
    for j in range(num):
        s = s + np.exp(-1j * (num + 1) / fp * vk)

    rho1_fft = c * intensity_fft * s
    rho1 = np.real(np.fft.ifft2(rho1_fft))
    return rho1


def propagate(wf, z, alpha):
    k = np.pi * 2 / lamd
    H = np.exp(1j * z * (k ** 2 - qr ** 2) ** 0.5)
    wf_fft = np.fft.fft2(wf)
    iwf = np.fft.ifft2(wf_fft * H)
    iwf *= np.sqrt(np.exp(-alpha * z))
    return iwf


total_time = 0.2
D = 0.8
fp = 100
tp = 0.01
w0 = D / 2 / np.sqrt(2)
P = 100e3
L = 3e3
layer = 10
dz = L / layer
num_screen = 10
absorb = 1e-5
n0 = 1.000309
gamma = 1.4
cs2 = 340 ** 2
rho0 = 1.293
lamd = 1.064e-6
vx = 2.


# 渡越时间
tH = w0 / 340
# 稳态时间
flow_time = D / vx
# 稳态脉冲数
Np = fp * flow_time
# 最大功率（方波）
Pm = P / fp / tp



npix, dx = 512, 1e-3
sl = npix * dx
x = np.linspace(-sl / 2, sl / 2 - dx, npix)
x, y = np.meshgrid(x, x)
r = np.sqrt(x**2 + y**2)
q = np.fft.fftfreq(npix, dx) * np.pi * 2
q[0] = q[0] + 1e-7
qx, qy = np.meshgrid(q, q)
qr = np.sqrt(qx**2 + qy**2)


wavefront = np.exp(- (r / w0) ** 2)
wavefront = wavefront * np.exp(-1j * np.pi * r ** 2 / lamd / L)
wavefront *= np.sqrt(Pm / power(intensity(wavefront, n0), dx))


for i in range(layer + 1):
    wavefront = propagate(wavefront, dz, absorb)

    if i != layer:
        wavefront = wavefront * np.exp(1j * np.pi * 2 / lamd * get_opd(intensity(wavefront, n0)))
#
# for m in range(10):

#
#     Np = w0 * 2 / vx / T
#
#     wf = wavefront.Wave(lamd1, n0, dx, num_pix)
#     wf *= optics.Gauss(w0)
#     wf *= optics.Focus(L)
#     wf.scale_power(Pm)
#
#     dz_array = timecontrol.TimeController.dz_array(L, num_screen)
#
#     ps_array = []
#     for i in range(num_screen):
#         ps_array.append(timesObj.PulseThermalScreen(tp, T, dz_array[i], vx, vy, dx, num_pix,
#                                                     num_pulse=num_time, alpha=alpha, rho0=rho0, cs2=cs2, gamma=gamma))
#
#     for i in range(num_screen + 1):
#         wf.propagate(dz_array[i], alpha)
#
#         if i != num_screen:
#             wf *= ps_array[i]
#
#     I[m] = wf.intensity.max()/1e7
#
#     plt.figure()
#     plt.pcolor(wf.x, wf.y, wf.intensity/1e4, cmap='jet')
#     plt.xlim(-0.2, 0.2)
#     plt.ylim(-0.2, 0.2)
#     plt.colorbar()
#     # plt.title("time: {0}s".format(Np * dt))
#     plt.show()
#
# plt.figure()
# plt.plot(N_flow_time, I/1e4)
# plt.xlim(1, 10)
# plt.show()




# tH = w0 / 340
# tp = np.linspace(0, 10*tH, 51)
# Pm = P / (tp + 1e-15)
# I = np.zeros(51)
# for m in range(51):
#
#     dx = 1e-3
#     num_pix = 500
#
#
#     wf = wavefront.Wave(lamd1, n0, dx, num_pix)
#     wf *= optics.Gauss(w0)
#     wf *= optics.Focus(L)
#     wf.scale_power(P)
#
#     dz_array = timecontrol.TimeController.dz_array(L, num_screen)
#
#     ps_array = []
#     for i in range(num_screen):
#         ps_array.append(timesObj.SinglePulseThermalScreen(tp[m], dz_array[i], vx, vy, dx, num_pix,
#                                                      alpha=alpha, rho0=rho0, cs2=cs2, gamma=gamma))
#
#     for i in range(num_screen + 1):
#         wf.propagate(dz_array[i], alpha)
#
#         if i != num_screen:
#             wf *= ps_array[i]
#
#     I[m] = wf.intensity.max()/1e7
#     if m == 3 or m == 50:
#         plt.figure()
#         plt.pcolor(wf.x, wf.y, wf.intensity, cmap='jet')
#         plt.xlim(-0.2, 0.2)
#         plt.ylim(-0.2, 0.2)
#         plt.colorbar()
#         plt.show()
#
# plt.figure()
# plt.plot(tp / tH, I)
# plt.xlim(0, 10)
plt.figure()
plt.pcolor(intensity(wavefront, n0))

plt.show()
