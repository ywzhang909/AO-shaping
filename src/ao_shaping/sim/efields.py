from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import zoom



# 定义配置类
class OptConfig:
    """
    Optical configuration parameters
    """
    pol = 1, 1
    shift = 0, 0


class GlDim:
    """
    Grid dimensions
    """
    x = 1
    y = 1


@dataclass
class FDTDIntervalConfig:
    """
    FDTD output field density interval
    """
    x: int
    y: int
    z: int

    def __post_init__(self):
        self.x = int(max(self.x, 1))
        self.y = int(max(self.y, 1))
        self.z = int(max(self.z, 1))


FDTDInterval = {"x": 50, "y": 50, "z": 50}
N_expansion = 1
FTT_expande_time = 1
PI = np.pi

@dataclass(frozen=True)
class OpticParameters:
    """
    Optical parameters for calculations
    """
    _num_elements_x: int  # Number of elements
    _num_elements_y: int  # Number of elements
    period: float  # Period
    wavelength: float  # Wavelength
    focal_length: float  # Focal length

    @property
    def num_elements_x(self):
        return int(self._num_elements_x)

    @property
    def num_elements_y(self):
        return int(self._num_elements_y)

    @property
    def k0(self):
        return 1 / self.wavelength

    @property
    def Px(self):
        return self.num_elements_x * self.period

    @property
    def Py(self):
        return self.num_elements_y * self.period

    @cached_property
    def str_x(self):
        return np.linspace(-self.Px / 2, self.Px / 2, self.num_elements_x)

    @cached_property
    def str_y(self):
        return np.linspace(-self.Py / 2, self.Py / 2, self.num_elements_y)
    @cached_property
    def meshed_X(self):
        X, _ = np.meshgrid(self.str_x, self.str_y)
        return X

    @cached_property
    def meshed_Y(self):
        _, Y = np.meshgrid(self.str_x, self.str_y)
        return Y

    @property
    def R(self):
        return min(self.Px, self.Py) / 2

    # 出射场密度
    @property
    def x_fdtd(self):
        x_count = np.ceil(self.Px / FDTDInterval.x).astype(int) // 2 * 2 + 1
        return np.linspace(-self.Px / 2, self.Px / 2, x_count)

    @property
    def y_fdtd(self):
        y_count = np.ceil(self.Py / FDTDInterval.y).astype(int) // 2 * 2 + 1
        return np.linspace(-self.Py / 2, self.Py / 2, y_count)

    @property
    def F(self):
        return self.focal_length / (2 * self.R)

    @property
    def NA(self):
        return np.sin(np.arctan(self.R / self.focal_length))


@dataclass
class ElectronicField:
    """
    Electronic field representation
    """
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray

    @property
    def G(self):
        """
        Calculate the intensity of the electronic field
        """
        g = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2
        g[g < 0] = 0
        return g

def phase_to_electronic_field(phase, x, y):
    """
    Convert phase to electronic field

    Args:
        phase: Phase distribution
        x: Size in x direction
        y: Size in y direction

    Returns:
        Electronic field
    """
    g = np.exp(-1j * phase)
    g1 = np.zeros_like(phase)
    datax = np.linspace(-x / 2, x / 2, phase.shape[1])
    datay = np.linspace(-y / 2, y / 2, phase.shape[0])
    X, Y = np.meshgrid(datax, datay)
    r = np.sqrt(X**2 + Y**2)
    R = min(x, y) / 2
    g1[r <= R] = 1
    g = g * g1

    Ex = g * 1
    Ey = g * 1j
    Ez = np.zeros_like(Ex, dtype=np.complex64)
    return ElectronicField(Ex, Ey, Ez)


def expande_electronic_field(init_phase, Px, Py, fdtd_x, fdtd_y):
    """
    Expand electronic field

    Args:
        init_phase: Initial phase
        Px: Size in x direction
        Py: Size in y direction
        fdtd_x: FDTD x coordinates
        fdtd_y: FDTD y coordinates

    Returns:
        Expanded electronic field
    """
    # Initialize optical field
    g = np.exp(-1j * init_phase)
    datax = np.linspace(-Px / 2, Px / 2, init_phase.shape[1])
    datay = np.linspace(-Py / 2, Py / 2, init_phase.shape[0])
    X, Y = np.meshgrid(datax, datay)
    g = np.where(np.sqrt(X**2 + Y**2) <= min(Px, Py) / 2, g, 0)

    Ex = g * 1
    Ey = g * 1j
    Ez = np.zeros_like(Ex, dtype=np.complex64)

    def interpolate_and_expand(E):
        E_amp, E_phase = np.abs(E), np.angle(E)
        E_amp_interpolated = RectBivariateSpline(datax, datay, E_amp, kx=1, ky=1)(
            fdtd_x, fdtd_y
        )
        E_phase_interpolated = RectBivariateSpline(datax, datay, E_phase, kx=1, ky=1)(
            fdtd_x, fdtd_y
        )
        E_interpolated = E_amp_interpolated * np.exp(1j * E_phase_interpolated)
        E_expanded = np.tile(E_interpolated, (N_expansion, 1))
        return E_expanded

    Ex_expanded = interpolate_and_expand(Ex)
    Ey_expanded = interpolate_and_expand(Ey)
    Ez_expanded = interpolate_and_expand(Ez)
    return ElectronicField(Ex_expanded, Ey_expanded, Ez_expanded)


def interp_phase(num_structure_x, num_structure_y, Px, Py, ws, ps, f, lambda_) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate phase values

    Args:
        num_structure_x: Number of structures in x direction
        num_structure_y: Number of structures in y direction
        Px: Size in x direction
        Py: Size in y direction
        ws: Widths
        ps: Phases
        f: Focal length
        lambda_: Wavelength

    Returns:
        Interpolated phase values
    """
    # Generate grid
    x = np.linspace(-Px / 2, Px / 2, num_structure_x)
    y = np.linspace(-Py / 2, Py / 2, num_structure_y)
    X, Y = np.meshgrid(x, y)
    k = 360 / lambda_
    r2 = X**2 + Y**2
    phase = k * (np.sqrt(r2 + f**2) - f)
    phase = np.mod(phase, 360)

    # Interpolation
    P_a = interp1d(ps, ws, kind="nearest", fill_value=np.nan)(phase)
    phase_map = dict(zip(ws, np.deg2rad(ps)))
    dist_phase = np.vectorize(phase_map.get)(P_a)
    return P_a, dist_phase

def trans_beam(
    init_phase,
    params: OpticParameters,
    mon_dist: float,
) -> tuple[ElectronicField, float]:
    """
    Transform beam and calculate efficiency

    Args:
        init_phase: Initial phase
        params: Optical parameters
        mon_dist: Monitor distance
        figure_path: Path to save figures

    Returns:
        tuple: Efficiency and paths to 2D and 1D plots
    """
    expanded_Ef = expande_electronic_field(
        init_phase,
        params.Px,
        params.Py,
        params.x_fdtd,
        params.y_fdtd,
    )

    x_FDTD = params.x_fdtd
    y_num = (len(x_FDTD) - 1) * N_expansion + 1
    y_FDTD = np.linspace(-params.Py / 2, params.Py / 2, y_num)

    Eo, jmX, jmY = perform_fft(
        expanded_Ef, x_FDTD, y_FDTD, params.k0, params.focal_length, mon_dist
    )
    efficiency, (fl, fr) = calculate_efficiency(Eo.G, jmX[0, :], jmY[:, 0], params.R)

    return Eo, efficiency

def calculate_efficiency(G, x, y, R) -> tuple[float, tuple[float, float]]:
    """
    Calculate focusing efficiency

    Args:
        G: Intensity distribution
        x: X coordinates
        y: Y coordinates
        R: Radius

    Returns:
        Efficiency and FWHM values
    """
    A_half = G.shape[0] // 2
    If = G[A_half, :]
    If0 = If / np.max(G[A_half, :])

    pq = np.argmax(If0)
    # pq = A_half
    fl = interp1d(np.flip(If0[0 : pq + 1]), np.flip(x[0 : pq + 1]), kind="linear", fill_value=np.nan)(0.5)
    fr = interp1d(If0[pq:], x[pq:], kind="linear", fill_value=np.nan)(0.5)
    FWHM = abs(fl - fr)

    xc, yc = OptConfig.shift
    rm1 = 3 * FWHM
    rm2 = min(np.max(x), np.max(y))
    rm1 = min(rm1, rm2)

    zmX, zmY = np.meshgrid(x, y)
    cycle = (zmX - xc) ** 2 * GlDim.x + (zmY - yc) ** 2 * GlDim.y
    I1 = np.where(cycle <= rm1**2, G , 0)
    I2 = np.where(cycle <= rm2**2, G , 0)
    efficiency = np.sum(I1) / np.sum(I2)

    return efficiency, (fl, fr)

def perform_fft(e_field: ElectronicField, x_FDTD, y_FDTD, k0, focal_length, mon_dist, recover_scale=False):
    """
    Perform FFT transformation and calculate light intensity

    Args:
        e_field: Electronic field
        x_FDTD: FDTD x coordinates
        y_FDTD: FDTD y coordinates
        k0: Wave number
        focal_length: Focal length
        mon_dist: Monitor distance
        recover_scale: Whether to recover scale

    Returns:
        Electronic field and coordinates
    """
    # Each pixel size
    px = abs(x_FDTD[-1] - x_FDTD[0]) / (len(x_FDTD) - 1)
    py = abs(y_FDTD[-1] - y_FDTD[0]) / (len(y_FDTD) - 1)

    # Expand field of view, a*b -> (jm*a)*3 * (jm*b)*3
    def expand(E, jm):
        E = zoom(E, jm, order=3) if jm != 1 else E
        a, b = E.shape
        E = np.pad(E,
                   ((FTT_expande_time*a, FTT_expande_time*a), (FTT_expande_time*b, FTT_expande_time*b)),
                   mode="constant", constant_values=0)
        return E

    jm = N_expansion
    jExi = expand(e_field.Ex, jm)
    jEyi = expand(e_field.Ey, jm)
    A, B = jExi.shape

    # FFT parameters
    lmaxX = (B / jm - 1) * px
    lmaxY = (A / jm - 1) * py
    jmx = np.linspace(-lmaxX / 2, lmaxX / 2, B)
    jmy = np.linspace(-lmaxY / 2, lmaxY / 2, A)
    jmX, jmY = np.meshgrid(jmx, jmy)
    jfx = PI * B / lmaxX
    jfy = PI * A / lmaxY
    jfmx = np.linspace(-jfx, jfx, B) / (2 * PI)
    jfmy = np.linspace(-jfy, jfy, A) / (2 * PI)
    jkx, jky = np.meshgrid(jfmx, jfmy)
    kr = np.sqrt(jkx**2 + jky**2) / k0

    # FFT transformation
    Ax = np.fft.fftshift(np.fft.fft2(jExi))
    Ay = np.fft.fftshift(np.fft.fft2(jEyi))
    Ax[kr > 1] = 0
    Ay[kr > 1] = 0
    q = np.sqrt(k0**2 - jkx**2 - jky**2, dtype=np.complex64)
    Az = -(jkx * Ax + jky * Ay) / q
    Az[kr > 1] = 0

    # Calculate focal plane light intensity
    zz = focal_length - mon_dist
    qq = np.exp(1j * 2 * PI * q * zz)

    Exo = np.fft.ifft2(np.fft.ifftshift(Ax * qq))
    Eyo = np.fft.ifft2(np.fft.ifftshift(Ay * qq))
    Ezo = np.fft.ifft2(np.fft.ifftshift(Az * qq))

    return (
        ElectronicField(Exo, Eyo, Ezo),
        jmX,
        jmY,
    )

def rescale_EF(E:ElectronicField, jmX, jmY):
    """
    Rescale electronic field

    Args:
        E: Electronic field
        jmX: X coordinates
        jmY: Y coordinates

    Returns:
        Rescaled electronic field and coordinates
    """
    A = E.Ex.shape[0]
    expand_boarder = A //  int(2*FTT_expande_time+1)
    Exo = E.Ex[
        expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
    ]
    Eyo = E.Ey[
        expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
    ]
    Ezo = E.Ez[
        expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
    ]
    return (
        ElectronicField(Exo, Eyo, Ezo),
        jmX[expand_boarder:-expand_boarder, expand_boarder:-expand_boarder],
        jmY[expand_boarder:-expand_boarder, expand_boarder:-expand_boarder],
    )