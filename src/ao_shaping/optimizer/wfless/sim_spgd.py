"""
Simulation-based SPGD optimization for adaptive optics wavefront correction.

Uses the ao_shaping.sim simulation tools, integrating Adam/Momentum optimizers
and heuristic search algorithms (PSO, GA, SA).
"""

import numpy as np
import tqdm

from ao_shaping.drivers.sim.compat import AOConfig, TraditionalAOSystem
from ao_shaping.algorithm.adam import Base, AdaMOD, Adam, AdamW, SGD, Muno, MunoW
from ao_shaping.utils.spots_calc import power_bucket, radius
from ao_shaping.utils import logger, Recorder

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}

def _create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs) -> Base:
    """Create optimizer instance."""
    opt_class = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    filtered_kwargs = {}
    import inspect
    sig = inspect.signature(opt_class.__init__)
    for key, value in kwargs.items():
        if key in sig.parameters:
            filtered_kwargs[key] = value
    return opt_class(dim, lr=lr, **filtered_kwargs)


def optimize_spgd(
    epochs: int,
    r_bucket: float = 0,
    delta: float = 0.1,
    gamma: float = 1e-4,
    n_grid: int = 256,
    aperture: float = 0.1,
    wavelength: float = 1550e-9,
    Cn2: float = 1e-9,
    dm_actuators: int = 8,
    dm_stroke: float = 5e-6,
    propagation_distance: float = 1000.0,
    learning_schedule: bool = False,
    show: bool = False,
    init_v: np.ndarray | None = None,
    aber_strength: float = 1.0,
    target_max_brightness: float = 1.0,
    power_ratio_threshold: float = 0.99,
    seed: int | None = None,
    optimizer_type: str = "spgd",
    beta1: float = 0.9,
    beta2: float = 0.99,
    beta3: float = 0.9999,
    use_momentum: bool = True,
    **kwargs
):
    """SPGD-based optimization using simulated AO system.

    Args:
        epochs: Number of iterations.
        r_bucket: Power-in-bucket radius. If 0, auto-computed from initial image.
        delta: Perturbation amplitude for each actuator.
        gamma: SPGD gain parameter (effective learning rate = gamma * delta).
        n_grid: Simulation grid size.
        aperture: Physical aperture size (m).
        wavelength: Wavelength (m).
        Cn2: Refractive index structure constant.
        dm_actuators: DM actuators per dimension.
        dm_stroke: DM stroke (m).
        propagation_distance: Propagation distance (m).
        learning_schedule: Use adaptive learning rate schedule.
        show: Show images.
        init_v: Initial DM voltages.
        aber_strength: Aberration strength multiplier for turbulence phase.
        target_max_brightness: Target maximum brightness.
        power_ratio_threshold: Power ratio threshold for convergence.
        seed: Random seed.
        optimizer_type: Optimizer type. "spgd" uses fixed gain (reference approach).
        beta1: Momentum coefficient (for use_momentum=True).
        beta2: Second moment decay.
        beta3: AdaMOD long-term buffer coefficient.
        use_momentum: Smooth gradient estimates with exponential moving average.
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization recorder.
    """

    if seed is not None:
        np.random.seed(seed)

    delta = abs(delta)
    epochs = int(epochs)

    recorder = Recorder(mark="sim_spgd", mode="max")

    config = AOConfig(
        N=n_grid,
        L=aperture,
        wavelength=wavelength,
        Cn2=Cn2,
        dm_actuators=dm_actuators,
        dm_stroke=dm_stroke,
        propagation_distance=propagation_distance,
    )

    ao_sys = TraditionalAOSystem(config=config)
    total_actuators = ao_sys.dm.total_actuators

    if init_v is None:
        _init_v = np.zeros(total_actuators, dtype=np.float64)
    else:
        _init_v = np.array(init_v, dtype=np.float64)
        if len(_init_v) != total_actuators:
            logger.warning(
                f"init_v length {len(_init_v)} != total_actuators {total_actuators}, "
                "padding with zeros."
            )
            _new_v = np.zeros(total_actuators, dtype=np.float64)
            _new_v[: len(_init_v)] = _init_v
            _init_v = _new_v

    px = np.arange(n_grid)
    py = np.arange(n_grid)
    PX, PY = np.meshgrid(px, py)

    if Cn2 > 0:
        turb = ao_sys.turbulence.get_phase_screen()
        _aber_phase = turb * aber_strength if aber_strength != 1.0 else turb
        ao_sys._turbulence_phase = _aber_phase
        ao_sys.set_dm_voltages(_init_v)
    else:
        _aber_phase = np.zeros((n_grid, n_grid))
        ao_sys._turbulence_phase = None
        ao_sys.set_dm_voltages(_init_v)

    R0 = n_grid / 2
    _ = radius(ao_sys.get_image(), center=(R0, R0), energy=0.865)

    if r_bucket <= 0:
        _img = ao_sys.get_image()
        r_bucket = radius(_img, center=(R0, R0), energy=0.99)
        _fix_bucket = False
    else:
        _fix_bucket = True

    ideal_r = n_grid / 30
    current_gamma = gamma
    current_delta = delta

    m_momentum = np.zeros(total_actuators) if use_momentum else None

    def calc_pib(img: np.ndarray) -> tuple[float, float]:
        pb = power_bucket(img, PX, PY, (R0, R0), r_bucket, use_dpix_scaling=False)
        total_power = np.sum(img)
        return pb, pb / (total_power + 1e-10)

    J0, pib_ratio0 = calc_pib(ao_sys.get_image())

    disturb_v = np.zeros(total_actuators)
    pos_pib, neg_pib = 0.0, 0.0
    flag = 0
    J = J0
    diff = 0.0
    gradient = np.zeros(total_actuators)
    pos_img = ao_sys.get_image()

    _init_r = r_bucket
    update_iter = max(int(epochs * 0.8 / (np.log(ideal_r / r_bucket) / np.log(0.9))), 1) if not _fix_bucket else epochs + 1

    init_img = ao_sys.get_image()
    init_phase = np.angle(init_img)
    init_rms = np.sqrt(np.mean(init_phase ** 2))
    _strehl_init = np.exp(-init_rms ** 2) if init_rms < 10 else 0.001

    recorder.append(
        {
            "sim_spgd": "init",
            "J": J0,
            "pib": J0,
            "_p%": pib_ratio0,
            "_max_r": _init_r,
            "_v": _init_v.copy(),
            "_img": init_img,
            "_diff": 0.0,
            "gamma": current_gamma,
            "r": r_bucket,
            "delta": current_delta,
            "_epoch": 0,
            "strehl": _strehl_init,
            "_grad": np.zeros_like(_init_v),
        }
    )

    use_fixed_gain = optimizer_type.lower() == "spgd"

    with tqdm.tqdm(total=epochs, desc=f"sim_spgd iter {epochs}", dynamic_ncols=True) as bar:
        for epoch in range(1, epochs + 1):
            if flag == 0:
                disturb_v = np.random.binomial(1, 0.5, (total_actuators,)).astype(float) * 2.0 - 1.0
                disturb_v = disturb_v * current_delta
                _init_v = _init_v + disturb_v / 2
                flag = 1

            if flag == 1:
                ao_sys.set_dm_voltages(_init_v)
                pos_img = ao_sys.get_image()
                pos_pib, pos_ratio = calc_pib(pos_img)
                _init_v = _init_v - disturb_v
                flag = -1
            elif flag == -1:
                ao_sys.set_dm_voltages(_init_v)
                neg_img = ao_sys.get_image()
                neg_pib, neg_ratio = calc_pib(neg_img)
                J = (pos_pib + neg_pib) / 2
                diff = pos_pib - neg_pib
                gradient = diff * disturb_v

                if use_momentum and m_momentum is not None:
                    m_momentum = beta1 * m_momentum + (1 - beta1) * gradient
                    gradient = m_momentum

                if use_fixed_gain:
                    update = current_gamma * gradient
                else:
                    optimizer = _create_optimizer(
                        optimizer_type, total_actuators, lr=current_gamma * 0.01,
                        beta1=beta1, beta2=beta2, beta3=beta3, **kwargs
                    )
                    update = optimizer.update(gradient)

                _init_v = np.clip(_init_v + update + disturb_v / 2, -1.0, 1.0)
                ao_sys.set_dm_voltages(_init_v)
                pos_img = ao_sys.get_image()
                flag = 0
            else:
                pos_img = ao_sys.get_image()

            pib, pib_ratio = calc_pib(pos_img)

            phase = np.angle(pos_img)
            phase_rms = np.sqrt(np.mean(phase ** 2))
            strehl = np.exp(-phase_rms ** 2) if phase_rms < 10 else 0.001

            if epoch % update_iter == update_iter - 1 and not _fix_bucket:
                _init_r = max(_init_r * 0.9, ideal_r)

            if (epoch % update_iter == update_iter - 1 or
                 epoch % max(update_iter // 2, 1) == max(update_iter // 2, 1) - 1 or
                 pib_ratio >= power_ratio_threshold) and not _fix_bucket and pib > 0:
                power_r = radius(pos_img, center=(R0, R0), energy=0.8)
                _pr = power_r * 0.9
                _r = max(r_bucket * 0.9 + 1, ideal_r, r_bucket)
                r_bucket = min(_r, _pr, _init_r)
                if learning_schedule:
                    current_gamma = gamma

            log = {
                "sim_spgd": epoch,
                "J": J,
                "_p%": pib_ratio,
                "_max_r": _init_r,
                "pib": pib,
                "_diff": diff,
                "gamma": current_gamma,
                "r": r_bucket,
                "delta": current_delta,
                "_epoch": epoch,
                "_v": _init_v.copy(),
                "_img": pos_img,
                "strehl": strehl,
                "_grad": gradient,
            }
            recorder.append(log)
            bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
            bar.update(1)

    return recorder



def optimize_spgd_zernike(
    epochs: int,
    n_max: int = 6,
    r_bucket: float = 0,
    delta: float = 0.1,
    gamma: float = 1e-3,
    n_grid: int = 256,
    aperture: float = 0.1,
    wavelength: float = 1550e-9,
    Cn2: float = 1e-9,
    propagation_distance: float = 1000.0,
    learning_schedule: bool = False,
    aber_strength: float = 1.0,
    power_ratio_threshold: float = 0.99,
    seed: int | None = None,
    beta1: float = 0.9,
    beta2: float = 0.99,
    beta3: float = 0.9999,
    use_momentum: bool = True,
    **kwargs
):
    """SPGD-based optimization using Zernike polynomial modes.

    Args:
        epochs: Number of iterations.
        n_max: Maximum Zernike radial order.
        r_bucket: Power-in-bucket radius. If 0, auto-computed.
        delta: Perturbation amplitude.
        gamma: SPGD gain parameter.
        n_grid: Simulation grid size.
        aperture: Physical aperture size (m).
        wavelength: Wavelength (m).
        Cn2: Refractive index structure constant.
        propagation_distance: Propagation distance (m).
        learning_schedule: Use adaptive learning rate schedule.
        aber_strength: Aberration strength multiplier.
        power_ratio_threshold: Power ratio threshold for convergence.
        seed: Random seed.
        beta1: Momentum coefficient.
        beta2: Second moment decay.
        beta3: AdaMOD long-term buffer coefficient.
        use_momentum: Smooth gradient estimates.
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization recorder.
    """

    try:
        from zernike import RZern
    except ImportError:
        logger.error("zernike package not installed. Run: pip install zernike")
        raise

    if seed is not None:
        np.random.seed(seed)

    delta = abs(delta)
    epochs = int(epochs)

    recorder = Recorder(mark="sim_spgd_zernike", mode="max")

    config = AOConfig(
        N=n_grid,
        L=aperture,
        wavelength=wavelength,
        Cn2=Cn2,
        propagation_distance=propagation_distance,
    )
    ao_sys = TraditionalAOSystem(config=config)

    x = np.linspace(-aperture / 2, aperture / 2, n_grid)
    y = np.linspace(-aperture / 2, aperture / 2, n_grid)
    X, Y = np.meshgrid(x, y)

    px = np.arange(n_grid)
    py = np.arange(n_grid)
    PX, PY = np.meshgrid(px, py)

    if Cn2 > 0:
        turb = ao_sys.turbulence.get_phase_screen()
        _aber_phase = turb * aber_strength if aber_strength != 1.0 else turb
    else:
        _aber_phase = np.zeros((n_grid, n_grid))

    if np.any(_aber_phase != 0):
        ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

    cart = RZern(n_max)
    cart.make_cart_grid(X, Y)

    def phase2zernike(phi: np.ndarray) -> np.ndarray:
        return cart.fit_cart_grid(phi)[0]

    def zernike2phase(c: np.ndarray) -> np.ndarray:
        return np.array(cart.eval_grid(c, matrix=True))

    R0 = n_grid / 2
    if r_bucket <= 0:
        _img = ao_sys.get_image()
        r_bucket = radius(_img, center=(R0, R0), energy=0.99)
        _fix_bucket = False
    else:
        _fix_bucket = True

    ideal_r = n_grid / 30
    update_iter = max(int(epochs * 0.8 / (np.log(ideal_r / r_bucket) / np.log(0.9))), 1) if not _fix_bucket else epochs + 1
    nk = cart.nk

    current_gamma = gamma
    current_delta = delta

    m_momentum = np.zeros(nk) if use_momentum else None

    def calc_pib(img: np.ndarray) -> tuple[float, float]:
        pb = power_bucket(img, PX, PY, (R0, R0), r_bucket, use_dpix_scaling=False)
        total_power = np.sum(img)
        return pb, pb / (total_power + 1e-10)

    c = np.zeros(nk)
    disturb_c = np.zeros(nk)
    phase = np.zeros((n_grid, n_grid))
    pos_pib, neg_pib = 0.0, 0.0
    flag = 0
    J = 0.0
    diff = 0.0
    gradient = np.zeros(nk)
    pos_img = ao_sys.get_image()

    _init_r = r_bucket

    _strehl_init = ao_sys.reset()["strehl"]
    if _aber_phase is not None and np.any(_aber_phase != 0):
        ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

    recorder.append(
        {
            "sim_spgd_zernike": "init",
            "J": J,
            "pib": J,
            "_p%": 0.0,
            "_max_r": _init_r,
            "_c": c.copy(),
            "_phase": phase.copy(),
            "_img": ao_sys.get_image(),
            "_diff": diff,
            "gamma": current_gamma,
            "r": r_bucket,
            "delta": current_delta,
            "_epoch": 0,
            "strehl": _strehl_init,
            "_grad": np.zeros(nk),
        }
    )

    with tqdm.tqdm(total=epochs, desc=f"sim_spgd_zernike iter {epochs}", dynamic_ncols=True) as bar:
        for epoch in range(1, epochs + 1):
            if flag == 0:
                disturb_c = np.random.binomial(1, 0.5, nk)
                disturb_c[disturb_c == 0] = -1
                disturb_c = current_delta * disturb_c
                c = c + disturb_c / 2
                phase = zernike2phase(c)
                flag = 1

            if flag == 1:
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * phase * 2 * np.pi / wavelength)
                pos_img = ao_sys.get_image()
                pos_pib, pos_ratio = calc_pib(pos_img)
                c = c - disturb_c
                phase = zernike2phase(c)
                flag = -1
            elif flag == -1:
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * phase * 2 * np.pi / wavelength)
                neg_img = ao_sys.get_image()
                neg_pib, neg_ratio = calc_pib(neg_img)
                J = (pos_pib + neg_pib) / 2
                diff = pos_pib - neg_pib
                gradient = diff * disturb_c

                if use_momentum and m_momentum is not None:
                    m_momentum = beta1 * m_momentum + (1 - beta1) * gradient
                    gradient = m_momentum

                update = current_gamma * gradient
                c = np.clip(c + update + disturb_c / 2, -5.0, 5.0)
                phase = zernike2phase(c)
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * phase * 2 * np.pi / wavelength)
                pos_img = ao_sys.get_image()
                flag = 0
            else:
                pos_img = ao_sys.get_image()

            pib, pib_ratio = calc_pib(pos_img)
            strehl_dict = ao_sys.reset()
            strehl = strehl_dict["strehl"]
            if _aber_phase is not None and np.any(_aber_phase != 0):
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

            if epoch % update_iter == update_iter - 1 and not _fix_bucket:
                _init_r = max(_init_r * 0.9, ideal_r)

            if (epoch % update_iter == update_iter - 1 or
                 epoch % max(update_iter // 2, 1) == max(update_iter // 2, 1) - 1 or
                 pib_ratio >= power_ratio_threshold) and not _fix_bucket and pib > 0:
                power_r = radius(pos_img, center=(R0, R0), energy=0.8)
                _pr = power_r * 0.9
                _r = max(r_bucket * 0.9 + 1, ideal_r, r_bucket)
                r_bucket = min(_r, _pr, _init_r)
                if learning_schedule:
                    current_gamma = gamma

            log = {
                "sim_spgd_zernike": epoch,
                "J": J,
                "_p%": pib_ratio,
                "_max_r": _init_r,
                "pib": pib,
                "_diff": diff,
                "gamma": current_gamma,
                "r": r_bucket,
                "delta": current_delta,
                "_epoch": epoch,
                "_c": c.copy(),
                "_phase": phase.copy(),
                "_img": pos_img,
                "strehl": strehl,
                "_grad": gradient,
            }
            recorder.append(log)
            bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
            bar.update(1)

    return recorder



def optimize_pso(
    epochs: int,
    n_particles: int = 20,
    r_bucket: float = 0,
    n_grid: int = 256,
    aperture: float = 0.1,
    wavelength: float = 1550e-9,
    Cn2: float = 1e-9,
    dm_actuators: int = 8,
    dm_stroke: float = 5e-6,
    propagation_distance: float = 1000.0,
    aber_strength: float = 1.0,
    power_ratio_threshold: float = 0.99,
    seed: int | None = None,
    w_inertia: float = 0.7,
    c1_cognitive: float = 1.4,
    c2_social: float = 1.4,
    show: bool = False,
    **kwargs
):
    """Particle Swarm Optimization for AO wavefront correction."""
    if seed is not None:
        np.random.seed(seed)

    epochs = int(epochs)
    recorder = Recorder(mark="sim_pso", mode="max")

    config = AOConfig(
        N=n_grid, L=aperture, wavelength=wavelength, Cn2=Cn2,
        dm_actuators=dm_actuators, dm_stroke=dm_stroke,
        propagation_distance=propagation_distance,
    )

    ao_sys = TraditionalAOSystem(config=config)
    total_actuators = ao_sys.dm.total_actuators

    px = np.arange(n_grid)
    py = np.arange(n_grid)
    PX, PY = np.meshgrid(px, py)
    R0 = n_grid / 2

    if Cn2 > 0:
        turb = ao_sys.turbulence.get_phase_screen()
        _aber_phase = turb * aber_strength if aber_strength != 1.0 else turb
    else:
        _aber_phase = np.zeros((n_grid, n_grid))

    if np.any(_aber_phase != 0):
        ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

    if r_bucket <= 0:
        _img = ao_sys.get_image()
        r_bucket = radius(_img, center=(R0, R0), energy=0.99)

    def calc_pib(v: np.ndarray) -> float:
        ao_sys.set_dm_voltages(np.clip(v, -1.0, 1.0))
        img = ao_sys.get_image()
        return power_bucket(img, PX, PY, (R0, R0), r_bucket, use_dpix_scaling=False)

    particles = np.random.uniform(-1.0, 1.0, (n_particles, total_actuators))
    velocities = np.random.uniform(-0.1, 0.1, (n_particles, total_actuators))
    personal_best = particles.copy()
    personal_best_pib = np.array([calc_pib(p) for p in particles])

    global_idx = np.argmax(personal_best_pib)
    global_best = particles[global_idx].copy()
    global_best_pib = personal_best_pib[global_idx]

    init_pib = calc_pib(np.zeros(total_actuators))
    _strehl_init = ao_sys.reset()["strehl"]

    recorder.append({
        "sim_pso": "init", "pib": init_pib, "_p%": 0.0,
        "_v": np.zeros(total_actuators), "_epoch": 0, "strehl": _strehl_init,
    })

    with tqdm.tqdm(total=epochs, desc=f"sim_pso iter {epochs}", dynamic_ncols=True) as bar:
        for epoch in range(1, epochs + 1):
            for i in range(n_particles):
                r1, r2 = np.random.rand(total_actuators), np.random.rand(total_actuators)
                velocities[i] = (
                    w_inertia * velocities[i]
                    + c1_cognitive * r1 * (personal_best[i] - particles[i])
                    + c2_social * r2 * (global_best - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], -1.0, 1.0)
                fitness = calc_pib(particles[i])
                if fitness > personal_best_pib[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_pib[i] = fitness

            global_idx = np.argmax(personal_best_pib)
            if personal_best_pib[global_idx] > global_best_pib:
                global_best = personal_best[global_idx].copy()
                global_best_pib = personal_best_pib[global_idx]

            pib = global_best_pib
            strehl_dict = ao_sys.reset()
            strehl = strehl_dict["strehl"]
            if _aber_phase is not None and np.any(_aber_phase != 0):
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

            recorder.append({
                "sim_pso": epoch, "pib": pib, "_p%": 0.0,
                "_v": global_best.copy(), "_epoch": epoch, "strehl": strehl,
            })
            bar.set_postfix(pib=f"{pib:.1f}")
            bar.update(1)

    return recorder


def optimize_ga(
    epochs: int,
    pop_size: int = 20,
    r_bucket: float = 0,
    n_grid: int = 256,
    aperture: float = 0.1,
    wavelength: float = 1550e-9,
    Cn2: float = 1e-9,
    dm_actuators: int = 8,
    dm_stroke: float = 5e-6,
    propagation_distance: float = 1000.0,
    aber_strength: float = 1.0,
    power_ratio_threshold: float = 0.99,
    seed: int | None = None,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    tournament_k: int = 3,
    show: bool = False,
    **kwargs
):
    """Genetic Algorithm for AO wavefront correction."""
    if seed is not None:
        np.random.seed(seed)

    epochs = int(epochs)
    recorder = Recorder(mark="sim_ga", mode="max")

    config = AOConfig(
        N=n_grid, L=aperture, wavelength=wavelength, Cn2=Cn2,
        dm_actuators=dm_actuators, dm_stroke=dm_stroke,
        propagation_distance=propagation_distance,
    )

    ao_sys = TraditionalAOSystem(config=config)
    total_actuators = ao_sys.dm.total_actuators

    px = np.arange(n_grid)
    py = np.arange(n_grid)
    PX, PY = np.meshgrid(px, py)
    R0 = n_grid / 2

    if Cn2 > 0:
        turb = ao_sys.turbulence.get_phase_screen()
        _aber_phase = turb * aber_strength if aber_strength != 1.0 else turb
    else:
        _aber_phase = np.zeros((n_grid, n_grid))

    if np.any(_aber_phase != 0):
        ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

    if r_bucket <= 0:
        _img = ao_sys.get_image()
        r_bucket = radius(_img, center=(R0, R0), energy=0.99)

    def calc_pib(v: np.ndarray) -> float:
        ao_sys.set_dm_voltages(np.clip(v, -1.0, 1.0))
        img = ao_sys.get_image()
        return power_bucket(img, PX, PY, (R0, R0), r_bucket, use_dpix_scaling=False)

    population = np.random.uniform(-1.0, 1.0, (pop_size, total_actuators))
    fitness = np.array([calc_pib(ind) for ind in population])
    best_idx = np.argmax(fitness)
    best_ind = population[best_idx].copy()
    best_pib = fitness[best_idx]

    init_pib = calc_pib(np.zeros(total_actuators))
    _strehl_init = ao_sys.reset()["strehl"]

    recorder.append({
        "sim_ga": "init", "pib": init_pib, "_p%": 0.0,
        "_v": np.zeros(total_actuators), "_epoch": 0, "strehl": _strehl_init,
    })

    def tournament_select(pop, fit, k):
        selected = np.random.choice(len(pop), k, replace=False)
        return pop[selected[np.argmax(fit[selected])]].copy()

    def crossover(p1, p2):
        if np.random.rand() < crossover_prob:
            mask = np.random.rand(len(p1)) > 0.5
            return np.where(mask, p1, p2), np.where(mask, p2, p1)
        return p1.copy(), p2.copy()

    def mutate(ind):
        mask = np.random.rand(len(ind)) < mutation_prob
        ind = ind.copy()
        ind[mask] = np.random.uniform(-1.0, 1.0, np.sum(mask))
        return np.clip(ind, -1.0, 1.0)

    with tqdm.tqdm(total=epochs, desc=f"sim_ga iter {epochs}", dynamic_ncols=True) as bar:
        for epoch in range(1, epochs + 1):
            new_pop = []
            for _ in range(pop_size // 2):
                p1 = tournament_select(population, fitness, tournament_k)
                p2 = tournament_select(population, fitness, tournament_k)
                c1, c2 = crossover(p1, p2)
                new_pop.append(mutate(c1))
                new_pop.append(mutate(c2))

            while len(new_pop) < pop_size:
                new_pop.append(mutate(population[np.random.randint(pop_size)]))

            population = np.array(new_pop[:pop_size])
            fitness = np.array([calc_pib(ind) for ind in population])

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_pib:
                best_ind = population[best_idx].copy()
                best_pib = fitness[best_idx]

            pib = best_pib
            strehl_dict = ao_sys.reset()
            strehl = strehl_dict["strehl"]
            if _aber_phase is not None and np.any(_aber_phase != 0):
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

            recorder.append({
                "sim_ga": epoch, "pib": pib, "_p%": 0.0,
                "_v": best_ind.copy(), "_epoch": epoch, "strehl": strehl,
            })
            bar.set_postfix(pib=f"{pib:.1f}")
            bar.update(1)

    return recorder


def optimize_sa(
    epochs: int,
    r_bucket: float = 0,
    n_grid: int = 256,
    aperture: float = 0.1,
    wavelength: float = 1550e-9,
    Cn2: float = 1e-9,
    dm_actuators: int = 8,
    dm_stroke: float = 5e-6,
    propagation_distance: float = 1000.0,
    aber_strength: float = 1.0,
    power_ratio_threshold: float = 0.99,
    seed: int | None = None,
    T_init: float = 1.0,
    T_min: float = 1e-6,
    cooling_rate: float = 0.995,
    step_size: float = 0.05,
    show: bool = False,
    **kwargs
):
    """Simulated Annealing for AO wavefront correction."""
    if seed is not None:
        np.random.seed(seed)

    epochs = int(epochs)
    recorder = Recorder(mark="sim_sa", mode="max")

    config = AOConfig(
        N=n_grid, L=aperture, wavelength=wavelength, Cn2=Cn2,
        dm_actuators=dm_actuators, dm_stroke=dm_stroke,
        propagation_distance=propagation_distance,
    )

    ao_sys = TraditionalAOSystem(config=config)
    total_actuators = ao_sys.dm.total_actuators

    px = np.arange(n_grid)
    py = np.arange(n_grid)
    PX, PY = np.meshgrid(px, py)
    R0 = n_grid / 2

    if Cn2 > 0:
        turb = ao_sys.turbulence.get_phase_screen()
        _aber_phase = turb * aber_strength if aber_strength != 1.0 else turb
    else:
        _aber_phase = np.zeros((n_grid, n_grid))

    if np.any(_aber_phase != 0):
        ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

    if r_bucket <= 0:
        _img = ao_sys.get_image()
        r_bucket = radius(_img, center=(R0, R0), energy=0.99)

    def calc_pib(v: np.ndarray) -> float:
        ao_sys.set_dm_voltages(np.clip(v, -1.0, 1.0))
        img = ao_sys.get_image()
        return power_bucket(img, PX, PY, (R0, R0), r_bucket, use_dpix_scaling=False)

    current = np.zeros(total_actuators)
    current_pib = calc_pib(current)
    best = current.copy()
    best_pib = current_pib
    T = T_init

    init_pib = current_pib
    _strehl_init = ao_sys.reset()["strehl"]

    recorder.append({
        "sim_sa": "init", "pib": init_pib, "_p%": 0.0,
        "_v": current.copy(), "_epoch": 0, "strehl": _strehl_init, "_T": T,
    })

    with tqdm.tqdm(total=epochs, desc=f"sim_sa iter {epochs}", dynamic_ncols=True) as bar:
        for epoch in range(1, epochs + 1):
            neighbor = current + np.random.uniform(-step_size, step_size, total_actuators)
            neighbor = np.clip(neighbor, -1.0, 1.0)
            neighbor_pib = calc_pib(neighbor)
            delta = neighbor_pib - current_pib

            if delta > 0 or np.random.rand() < np.exp(delta / (T + 1e-10)):
                current = neighbor
                current_pib = neighbor_pib
                if current_pib > best_pib:
                    best = current.copy()
                    best_pib = current_pib

            T = max(T * cooling_rate, T_min)

            pib = best_pib
            strehl_dict = ao_sys.reset()
            strehl = strehl_dict["strehl"]
            if _aber_phase is not None and np.any(_aber_phase != 0):
                ao_sys.E_corrected = ao_sys.E_corrected * np.exp(1j * _aber_phase)

            recorder.append({
                "sim_sa": epoch, "pib": pib, "_p%": 0.0,
                "_v": best.copy(), "_epoch": epoch, "strehl": strehl, "_T": T,
            })
            bar.set_postfix(pib=f"{pib:.1f}", T=f"{T:.2e}")
            bar.update(1)

    return recorder
