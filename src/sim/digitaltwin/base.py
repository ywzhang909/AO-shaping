"""
基础类
"""

from scipy import constants as const
import numpy as np

from sim.digitaltwin import utilities as utils


class Wave:
    def __init__(self):
        self.npix = None
        self.dpix = None

        self.wavefront = None
        self.ex = None
        self.ey = None

        self.refractive = None
        self.wavelength = None

    @property
    def x(self):
        a, _, _, _, _, _ = utils.grid(self.npix, self.dpix)
        return a

    @property
    def y(self):
        _, a, _, _, _, _ = utils.grid(self.npix, self.dpix)
        return a

    @property
    def r(self):
        _, _, a, _, _, _ = utils.grid(self.npix, self.dpix)
        return a

    @property
    def qx(self):
        _, _, _, a, _, _ = utils.grid(self.npix, self.dpix)
        return a

    @property
    def qy(self):
        _, _, _, _, a, _ = utils.grid(self.npix, self.dpix)
        return a

    @property
    def qr(self):
        _, _, _, _, _, a = utils.grid(self.npix, self.dpix)
        return a

    @property
    def side_length(self):
        return self.npix * self.dpix

    @property
    def intensity(self):
        intensity = utils.wf2intensity(self.wavefront, self.refractive)
        if self.ex is not None:
            intensity = intensity + utils.wf2intensity(self.ex, self.refractive) \
                        + utils.wf2intensity(self.ey, self.refractive)
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

    @property
    def centroid(self):
        center_x = (self.x * self.intensity).sum() / self.intensity.sum()
        center_y = (self.y * self.intensity).sum() / self.intensity.sum()

        return center_x, center_y

    @property
    def peak_position(self):
        intensity = self.intensity
        index = np.unravel_index(intensity.argmax(), intensity.shape)
        xp, yp = self.x[index[0], index[1]], self.y[index[0], index[1]]

        return xp, yp

    def change_grid(self, npix, dpix):
        self.npix = npix
        self.dpix = dpix

    def change_wf(self, scale=1, phase=0):
        self.wavefront = scale * self.wavefront * np.exp(1j * phase)
        if self.ex is not None:
            self.ex = scale * self.ex * np.exp(1j * phase)
            self.ey = scale * self.ey * np.exp(1j * phase)

    def scale_power(self, desire_power):
        power = self.power
        self.change_wf(scale=np.sqrt(desire_power / power))

    # def zoom_double(self):
    #     if self.npix % 2 != 0:
    #         self.npix = self.npix + 1
    #         self.wavefront = np.pad(self.wavefront, (1, 0), mode='constant', constant_values=(0,))
    #         if self.ex is not None:
    #             self.ex = np.pad(self.ex, (1, 0), mode='constant', constant_values=(0,))
    #             self.ey = np.pad(self.ey, (1, 0), mode='constant', constant_values=(0,))
    #
    #     self.wavefront = utils.wf_size_double(self.wavefront)
    #     if self.ex is not None:
    #         self.ex = utils.wf_size_double(self.ex)
    #         self.ey = utils.wf_size_double(self.ey)
    #     self.change_grid(self.npix, self.dpix / 2)

    def resize(self, npix, dpix):
        power = self.power
        self.wavefront = utils.matrix_size_trans(self.wavefront, self.dpix, npix, dpix)
        if self.ex is not None:
            self.ex = utils.matrix_size_trans(self.ex, self.dpix, npix, dpix)
            self.ey = utils.matrix_size_trans(self.ey, self.dpix, npix, dpix)
        self.change_grid(npix, dpix)
        self.scale_power(power)


class TimeWave(Wave):
    def __init__(self):
        super(TimeWave, self).__init__()
        self.time = 0
        self.wave0 = None

        self.modular_fun = None

    @property
    def omega(self):
        omega = 2 * np.pi * self.freq
        return omega

    def time_update(self, delta_time):
        if not isinstance(self.wave0, Wave):
            raise TypeError('wave0 must be Wave type')

        self.time += delta_time
        self.wavefront = self.wave0.wavefront
        self.ex = self.wave0.ex
        self.ey = self.wave0.ey
        self.change_wf(scale=1 if self.modular_fun is None else self.modular_fun(self.time), phase=self.omega*self.time)
        self.change_grid(self.wave0.npix, self.wave0.dpix)


class PulseWave(Wave):
    def __init__(self):
        super(PulseWave, self).__init__()
        self.time = 0
        self.wave0 = None

        self.duration_time = 0
        self.repetition_rate = 0
        self.average_power = 0
        self.shape = None

    @property
    def energy(self):
        energy = self.average_power / self.repetition_rate
        return energy

    @property
    def intensity_max(self):
        if self.shape == 'tri' or self.shape == 'sin2':
            power_max = self.energy * 2 / self.duration_time
        elif self.shape == 'square':
            power_max = self.energy / self.duration_time
        elif self.shape == 'sin':
            power_max = self.energy / 2 * np.pi / self.duration_time
        else:
            raise ValueError('Pulse shape is wrong set')

        return power_max / self.dpix ** 2

    def scale_power(self, desire_power):
        power = self.power
        self.change_wf(scale=np.sqrt(self.average_power / power))

    def time_update(self, delta_time):
        if not isinstance(self.wave0, Wave):
            raise TypeError('wave0 must be Wave type')

        self.time += delta_time
        self.wavefront = self.wave0.wavefront
        self.ex = self.wave0.ex
        self.ey = self.wave0.ey
        self.change_grid(self.wave0.npix, self.wave0.dpix)


class SinglePulseWave(Wave):
    def __init__(self):
        super(SinglePulseWave, self).__init__()
        self.time = 0
        self.wave0 = None

        self.duration_time = 0
        self.shape = None
        self.energy = 0

    @property
    def intensity_max(self):
        if self.shape == 'tri' or self.shape == 'sin2':
            power_max = self.energy * 2 / self.duration_time
        elif self.shape == 'square':
            power_max = self.energy / self.duration_time
        elif self.shape == 'sin':
            power_max = self.energy / 2 * np.pi / self.duration_time
        else:
            raise ValueError('Pulse shape is wrong set')

        return power_max / self.dpix ** 2

    def pulse_func(self, time):
        if time < self.duration_time:
            if self.shape == 'tri':
                scale = np.sqrt((1 - np.abs(2 / self.duration_time * time - 1)))
            elif self.shape == 'square':
                scale = 1
            elif self.shape == 'sin':
                scale = np.sqrt(np.sin((np.pi / self.duration_time) * time))
            elif self.shape == 'sin2':
                scale = np.sin((np.pi / self.duration_time) * time)
            else:
                raise ValueError('Pulse shape is wrong set')
        else:
            scale = 0

        return scale

    def scale_power(self, desire_power):
        power = self.power
        power_max = self.intensity_max * self.dpix ** 2
        self.change_wf(scale=np.sqrt(power_max / power))

    def time_update(self, delta_time):
        if not isinstance(self.wave0, Wave):
            raise TypeError('wave0 must be Wave type')

        self.time += delta_time
        self.wavefront = self.wave0.wavefront
        self.ex = self.wave0.ex
        self.ey = self.wave0.ey
        self.change_wf(scale=self.pulse_func(self.time), phase=0)
        self.change_grid(self.wave0.npix, self.wave0.dpix)


class Environment:
    def __init__(self):

        self.absorb = None
        self.scatter = None
        self.wind_x = None
        self.wind_y = None
        self.density = None
        self.Cp = None
        self.Cv = None
        self.temperature = None
        self.nT = None
        self.atm = None
        self.Cs2 = None
        self.gravity = None
        self.Cn2 = None
        self.L0 = None
        self.l0 = None

    def set_value(self, absorb, scatter, wind_x, wind_y, density, Cp, Cv, temperature, nT, atm, Cs2, gravity,
                  Cn2, L0, l0):
        self.absorb = absorb
        self.scatter = scatter
        self.wind_x = wind_x
        self.wind_y = wind_y
        self.density = density
        self.Cp = Cp
        self.Cv = Cv
        self.temperature = temperature
        self.nT = nT
        self.atm = atm
        self.Cs2 = Cs2
        self.gravity = gravity
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0

    @property
    def wind(self):
        return (self.wind_x ** 2 + self.wind_y ** 2) ** 0.5

    @property
    def extinction(self):
        return self.absorb + self.scatter
