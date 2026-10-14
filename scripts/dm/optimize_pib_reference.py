"""
PIB Optimization using drivers/sim device layer.

Goal: With turbulence applied, optimize with SPGD/PSO/GA/SA to make PIB
reach 80% of the ideal (no-turbulence) value.

Physics:
- Plane wave → circular aperture → thin lens → turbulence phase screen → focal plane
- Angular spectrum propagation via ao_shaping.drivers.sim
- PIB metric from ao_shaping.drivers.sim (wrapping sim.digitaltwin)
- Zernike-based DM control (from reference src/sim/spgd.py)
"""
from __future__ import annotations

import numpy as np
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from zernike import RZern
from ao_shaping.drivers.sim import (
    SimulatedTurbulentScreen,
    create_wave,
    apply_aperture,
    apply_focus,
    propagate,
    power_bucket,
    radius_metric,
    Environment,
)
from sim.digitaltwin import params as dt_params


@dataclass
class SimConfig:
    npix: int = 256
    aperture: float = 0.005
    wavelength: float = 1550e-9
    focal_length: float = 0.5
    propagation_distance: float = 1000.0
    Cn2: float = 1e-9
    L0: float = 10.0
    l0: float = 0.01
    dm_actuators: int = 8
    dm_stroke: float = 1e-6
    r_frac: float = 0.5
    subharmonics: int = 0


@dataclass
class SimState:
    wave: Any = None
    ideal_wave: Any = None
    turb: SimulatedTurbulentScreen | None = None
    env: Environment | None = None
    mask: np.ndarray | None = None
    focus_phase: np.ndarray | None = None
    r_bucket: float = 0.0
    ideal_pib: float = 0.0
    ideal_total: float = 0.0


class AOSim:
    state: SimState

    def __init__(self, config: SimConfig, seed: int | None = None):
        self.cfg = config
        if seed is not None:
            np.random.seed(seed)
        self._init_state()
    
    def _init_state(self) -> None:
        cfg = self.cfg
        dpix = cfg.aperture / cfg.npix
        ar = cfg.aperture / 2
        
        wave = create_wave(cfg.npix, dpix, cfg.wavelength)
        mask = (np.sign(ar - wave.r) + 1) / 2
        wave.wavefront = wave.wavefront * mask
        focus_phase = -np.pi * wave.r ** 2 / wave.lamd / cfg.focal_length
        wave.change_wf(phase=focus_phase)
        propagate(wave, cfg.focal_length)
        
        ideal_wave = create_wave(cfg.npix, dpix, cfg.wavelength)
        ideal_wave.wavefront = ideal_wave.wavefront * mask.copy()
        ideal_wave.change_wf(phase=focus_phase.copy())
        propagate(ideal_wave, cfg.focal_length)
        
        r_bucket = radius_metric(
            ideal_wave.intensity, ideal_wave.x, ideal_wave.y, 'origin', cfg.r_frac
        )
        ideal_pib = power_bucket(
            ideal_wave.intensity, ideal_wave.x, ideal_wave.y, 'origin', r_bucket
        )
        ideal_total = ideal_wave.intensity.sum() * dpix ** 2
        
        env = Environment()
        env.Cn2 = cfg.Cn2
        env.L0 = cfg.L0
        env.l0 = cfg.l0
        
        turb = SimulatedTurbulentScreen(
            dist=cfg.propagation_distance, Cn2=cfg.Cn2,
            L0=cfg.L0, l0=cfg.l0, harmonic=cfg.subharmonics,
        )
        
        self.state = SimState(
            wave=wave,
            ideal_wave=ideal_wave,
            turb=turb,
            env=env,
            mask=mask,
            focus_phase=focus_phase,
            r_bucket=r_bucket,
            ideal_pib=ideal_pib,
            ideal_total=ideal_total,
        )
    
    def get_turb_image(self, dm_phase: np.ndarray | None = None) -> np.ndarray:
        wave = create_wave(self.cfg.npix, self.cfg.aperture / self.cfg.npix, self.cfg.wavelength)
        wave.wavefront = wave.wavefront * self.state.mask.copy()
        wave.change_wf(phase=self.state.focus_phase.copy())
        self.state.turb.process(wave)
        if dm_phase is not None:
            wave.change_wf(phase=dm_phase)
        propagate(wave, self.cfg.focal_length)
        return wave.intensity.copy()
    
    def eval_pib(self, dm_phase: np.ndarray) -> float:
        img = self.get_turb_image(dm_phase)
        return self.pib(img)
    
    def get_image(self, voltages: np.ndarray | None = None) -> np.ndarray:
        phase = self._voltages_to_phase(voltages) if voltages is not None else None
        return self.get_turb_image(phase)
    
    def _voltages_to_phase(self, voltages: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n = cfg.npix
        n_act_sq = cfg.dm_actuators ** 2
        if len(voltages) != n_act_sq:
            voltages = np.zeros(n_act_sq)
        
        ar = cfg.aperture / 2
        x_acts = np.linspace(-ar, ar, cfg.dm_actuators)
        y_acts = np.linspace(-ar, ar, cfg.dm_actuators)
        dx = cfg.aperture / cfg.dm_actuators
        sigma = dx * 1.2
        
        x = self.state.wave.x
        y = self.state.wave.y
        
        phase = np.zeros((n, n), dtype=float)
        for i, v in enumerate(voltages):
            ix = i % cfg.dm_actuators
            iy = i // cfg.dm_actuators
            cx, cy = x_acts[ix], y_acts[iy]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
            phase += v * cfg.dm_stroke * gaussian
        
        phase *= (2 * np.pi / cfg.wavelength)
        return phase
    
    def pib(self, img: np.ndarray) -> float:
        dpix = self.cfg.aperture / self.cfg.npix
        power = img.sum() * dpix ** 2
        r_bucket = self.state.r_bucket
        wave_x = self.state.ideal_wave.x
        wave_y = self.state.ideal_wave.y
        return power_bucket(img, wave_x, wave_y, 'origin', r_bucket)
    
    def strehl(self, img: np.ndarray) -> float:
        return np.max(img) / np.max(self.state.ideal_wave.intensity)
    
    def __repr__(self) -> str:
        cfg = self.cfg
        s = self.state
        return (f"AOSim(npix={cfg.npix}, D={cfg.aperture*1e3:.1f}mm, "
                f"lambda={cfg.wavelength*1e9:.0f}nm, f={cfg.focal_length}m, "
                f"Cn2={cfg.Cn2:.0e}, r_bucket={s.r_bucket/self.cfg.aperture*self.cfg.npix:.0f}px, "
                f"ideal_pib={s.ideal_pib/self.state.ideal_total:.4f})")


class ZernikeDM:
    def __init__(self, n_modes: int, x: np.ndarray, y: np.ndarray, delta: float, alpha: float,
                 beta1: float = 0.9, beta2: float = 0.99, r_bucket: float = 0.0,
                 phase_init: np.ndarray | None = None, aber_phase: np.ndarray | None = None):
        self.cart = RZern(n_modes)
        self.cart.make_cart_grid(x, y)
        
        if phase_init is None:
            self.phase = np.zeros((len(x), len(y)))
        else:
            self.phase = phase_init
        
        self.c = self._phase2zernike(self.phase)
        
        if aber_phase is not None:
            c_aber = self._phase2zernike(aber_phase)
            self.c = self.c + c_aber
            self.phase = self._zernike2phase(self.c)
        
        self.disturb_c = np.zeros(self.cart.nk)
        self.m = np.zeros(self.cart.nk)
        self.v = np.ones(self.cart.nk)
        self.t = 0
        self.delta = delta
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.flag = 0
        self.pos = 0.0
        self.neg = 0.0
        self.J = 0.0
        self.r_bucket = r_bucket
    
    def _zernike2phase(self, c: np.ndarray) -> np.ndarray:
        return np.array(self.cart.eval_grid(c, matrix=True))
    
    def _phase2zernike(self, phi: np.ndarray) -> np.ndarray:
        if phi.ndim == 2:
            return self.cart.fit_cart_grid(phi)[0]
        return self.cart.fit_cart_grid(phi.ravel())[0]
    
    def disturb_init(self, n: int) -> np.ndarray:
        d = np.random.binomial(1, 0.5, n).astype(float) * 2.0 - 1.0
        return self.delta * d
    
    def loss(self, intensity: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        r = radius_metric(intensity, x, y, 'origin', 0.865) * self.r_bucket
        pb = power_bucket(intensity, x, y, 'origin', r)
        return pb
    
    def spgd_step(self, intensity: np.ndarray, x: np.ndarray, y: np.ndarray):
        if self.flag == 0:
            self.disturb_c = self.disturb_init(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1
        
        if self.flag == 1:
            self.pos = self.loss(intensity, x, y)
            self.c = self.c - self.disturb_c
            self.phase = self._zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(intensity, x, y)
            self.J = (self.pos + self.neg) / 2
            self.c = self.c + self.alpha * (self.pos - self.neg) * self.disturb_c + self.disturb_c / 2
            self.phase = self._zernike2phase(self.c)
            self.flag = 0
    
    def adam_step(self, intensity: np.ndarray, x: np.ndarray, y: np.ndarray):
        if self.flag == 0:
            self.t += 1
            self.disturb_c = self.disturb_init(self.cart.nk)
            self.c = self.c + self.disturb_c / 2
            self.flag = 1
        
        if self.flag == 1:
            self.pos = self.loss(intensity, x, y)
            self.c = self.c - self.disturb_c
            self.phase = self._zernike2phase(self.c)
            self.flag = -1
        elif self.flag == -1:
            self.neg = self.loss(intensity, x, y)
            self.J = (self.pos + self.neg) / 2
            grad = (self.pos - self.neg) / (self.disturb_c + 1e-10)
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = self.v * self.beta2 + (1 - self.beta2) * grad ** 2
            m_h = self.m / (1 - self.beta1 ** self.t + 1e-10)
            v_h = self.v / (1 - self.beta2 ** self.t + 1e-10)
            self.c = self.c - self.alpha * m_h / (np.sqrt(v_h) + 1e-8) + self.disturb_c / 2
            self.phase = self._zernike2phase(self.c)
            self.flag = 0
    
    def out(self, wave: Any) -> None:
        wave.change_wf(phase=self.phase)


def run_spgd_zernike(ao: AOSim, n_modes: int = 11, delta: float = 0.1,
                     alpha: float = 1e-3, beta1: float = 0.9, beta2: float = 0.99,
                     epochs: int = 500, target_ratio: float = 0.80,
                     verbose: bool = True) -> dict:
    x = ao.state.ideal_wave.x
    y = ao.state.ideal_wave.y
    ideal_pib = ao.state.ideal_pib
    target_pib = ideal_pib * target_ratio
    
    cart = RZern(n_modes)
    cart.make_cart_grid(x, y)
    
    def eval_grid(c_vals):
        return np.array(cart.eval_grid(c_vals, matrix=True))
    
    def fit_grid(phi):
        return cart.fit_cart_grid(phi)[0]
    
    c = np.zeros(cart.nk)
    phase = eval_grid(c)
    
    m = np.zeros(cart.nk)
    v = np.ones(cart.nk)
    t = 0
    
    pib_vals, ratio_vals, epochs_list = [], [], []
    best_pib, best_c = 0.0, c.copy()
    stagnant, flag = 0, 0
    
    for epoch in range(1, epochs + 1):
        if flag == 0:
            disturb = delta * (np.random.binomial(1, 0.5, cart.nk).astype(float) * 2.0 - 1.0)
            phase_pos = eval_grid(c + disturb / 2)
            flag = 1
        
        if flag == 1:
            pos_pib = ao.eval_pib(phase_pos)
            phase_neg = eval_grid(c - disturb / 2)
            flag = -1
        elif flag == -1:
            neg_pib = ao.eval_pib(phase_neg)
            J = (pos_pib + neg_pib) / 2
            
            t += 1
            grad = (pos_pib - neg_pib) / (disturb + 1e-10)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_h = m / (1 - beta1 ** t + 1e-10)
            v_h = v / (1 - beta2 ** t + 1e-10)
            c = c - alpha * m_h / (np.sqrt(v_h) + 1e-8) + disturb / 2
            phase = eval_grid(c)
            
            ratio = J / ideal_pib if ideal_pib > 0 else 0
            pib_vals.append(J)
            ratio_vals.append(ratio)
            epochs_list.append(epoch)
            
            if J > best_pib:
                best_pib = J
                best_c = c.copy()
                stagnant = 0
            else:
                stagnant += 1
            
            if ratio >= target_ratio:
                if verbose:
                    print(f"  [SPGD] Converged at epoch {epoch}: PIB={J:.4e}, ratio={ratio:.4f}")
                break
            if stagnant > 100:
                if verbose:
                    print(f"  [SPGD] Stagnant at epoch {epoch}: best={best_pib:.4e}")
                break
            
            flag = 0
    
    final_phase = eval_grid(best_c)
    final_img = ao.get_turb_image(final_phase)
    final_pib = ao.pib(final_img)
    final_ratio = final_pib / ideal_pib
    
    if verbose:
        print(f"  [SPGD] Final: PIB={final_pib:.4e}, ratio={final_ratio:.4f}, "
              f"ideal={ideal_pib:.4e}, target={target_pib:.4e}")
    
    return {
        'pib_curve': np.array(pib_vals),
        'ratio_curve': np.array(ratio_vals),
        'epochs': np.array(epochs_list),
        'best_pib': final_pib,
        'final_ratio': final_ratio,
        'converged': final_ratio >= target_ratio,
        'final_img': final_img,
        'ideal_pib': ao.state.ideal_pib,
        'ideal_total': ao.state.ideal_total,
        'r_bucket': ao.state.r_bucket,
    }


def run_spgd_voltage(ao: AOSim, delta: float = 0.5, gamma: float = 1e-3,
                     epochs: int = 500, target_ratio: float = 0.80,
                     seed: int | None = None, verbose: bool = True) -> dict:
    if seed is not None:
        np.random.seed(seed)
    
    total_act = ao.cfg.dm_actuators ** 2
    voltages = np.zeros(total_act)
    m_mom = np.zeros(total_act)
    
    pib_vals, ratio_vals, epochs_list = [], [], []
    best_pib, best_v = 0.0, voltages.copy()
    stagnant, flag = 0, 0
    pos_pib, neg_pib = 0.0, 0.0
    perturb_v = np.zeros(total_act)
    
    target_pib = ao.state.ideal_pib * target_ratio
    
    for epoch in range(1, epochs + 1):
        if flag == 0:
            perturb_v = (np.random.binomial(1, 0.5, total_act).astype(float) * 2.0 - 1.0) * delta
            voltages = voltages + perturb_v / 2
            flag = 1
        
        if flag == 1:
            img = ao.get_image(voltages)
            pos_pib = ao.pib(img)
            voltages = voltages - perturb_v
            flag = -1
        elif flag == -1:
            img = ao.get_image(voltages)
            neg_pib = ao.pib(img)
            
            diff = pos_pib - neg_pib
            grad = diff * perturb_v
            m_mom = 0.9 * m_mom + 0.1 * grad
            update = gamma * m_mom
            voltages = np.clip(voltages + update + perturb_v / 2, -1.0, 1.0)
            
            J = (pos_pib + neg_pib) / 2
            ratio = J / ao.state.ideal_pib if ao.state.ideal_pib > 0 else 0
            pib_vals.append(J)
            ratio_vals.append(ratio)
            epochs_list.append(epoch)
            
            if J > best_pib:
                best_pib = J
                best_v = voltages.copy()
                stagnant = 0
            else:
                stagnant += 1
            
            if ratio >= target_ratio:
                if verbose:
                    print(f"  [SPGD-V] Converged at epoch {epoch}: PIB={J:.4e}, ratio={ratio:.4f}")
                break
            if stagnant > 100:
                if verbose:
                    print(f"  [SPGD-V] Stagnant at epoch {epoch}")
                break
            
            flag = 0
    
    final_img = ao.get_image(best_v)
    final_pib = ao.pib(final_img)
    final_ratio = final_pib / ao.state.ideal_pib
    
    if verbose:
        print(f"  [SPGD-V] Final: PIB={final_pib:.4e}, ratio={final_ratio:.4f}")
    
    return {
        'pib_curve': np.array(pib_vals),
        'ratio_curve': np.array(ratio_vals),
        'epochs': np.array(epochs_list),
        'best_pib': final_pib,
        'final_ratio': final_ratio,
        'converged': final_ratio >= target_ratio,
        'final_img': final_img,
        'ideal_pib': ao.state.ideal_pib,
        'ideal_total': ao.state.ideal_total,
        'r_bucket': ao.state.r_bucket,
    }


def run_pso(ao: AOSim, n_particles: int = 20, epochs: int = 300,
            w: float = 0.7, c1: float = 1.4, c2: float = 1.4,
            target_ratio: float = 0.80, seed: int | None = None,
            verbose: bool = True) -> dict:
    if seed is not None:
        np.random.seed(seed)
    
    total_act = ao.cfg.dm_actuators ** 2
    
    def fitness(v):
        return ao.pib(ao.get_image(v))
    
    particles = np.random.uniform(-1, 1, (n_particles, total_act))
    velocities = np.random.uniform(-0.2, 0.2, (n_particles, total_act))
    fitnesses = np.array([fitness(p) for p in particles])
    
    best_idx = np.argmax(fitnesses)
    g_best = particles[best_idx].copy()
    p_best = particles.copy()
    p_best_f = fitnesses.copy()
    
    pib_vals = [float(np.max(fitnesses))]
    ratio_vals = [float(np.max(fitnesses)) / ao.state.ideal_pib]
    epochs_list = [0]
    best_ever = float(np.max(fitnesses))
    stagnant = 0
    
    for epoch in range(1, epochs + 1):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (p_best[i] - particles[i])
                             + c2 * r2 * (g_best - particles[i]))
            velocities[i] = np.clip(velocities[i], -0.5, 0.5)
            particles[i] = np.clip(particles[i] + velocities[i], -1, 1)
            f = fitness(particles[i])
            if f > p_best_f[i]:
                p_best[i] = particles[i].copy()
                p_best_f[i] = f
                if f > fitness(g_best):
                    g_best = particles[i].copy()
        
        best_f = float(np.max(p_best_f))
        pib_vals.append(best_f)
        ratio_vals.append(best_f / ao.state.ideal_pib)
        epochs_list.append(epoch)
        
        if best_f > best_ever:
            best_ever = best_f
            stagnant = 0
        else:
            stagnant += 1
        
        if best_f / ao.state.ideal_pib >= target_ratio:
            if verbose:
                print(f"  [PSO] Converged at epoch {epoch}: PIB={best_f:.4e}, ratio={best_f/ao.state.ideal_pib:.4f}")
            break
        if stagnant > 80:
            if verbose:
                print(f"  [PSO] Stagnant at epoch {epoch}")
            break
    
    final_img = ao.get_image(g_best)
    final_ratio = best_ever / ao.state.ideal_pib
    
    if verbose:
        print(f"  [PSO] Best: PIB={best_ever:.4e}, ratio={final_ratio:.4f}")
    
    return {
        'pib_curve': np.array(pib_vals),
        'ratio_curve': np.array(ratio_vals),
        'epochs': np.array(epochs_list),
        'best_pib': best_ever,
        'final_ratio': final_ratio,
        'converged': final_ratio >= target_ratio,
        'final_img': final_img,
        'ideal_pib': ao.state.ideal_pib,
        'ideal_total': ao.state.ideal_total,
        'r_bucket': ao.state.r_bucket,
    }


def run_ga(ao: AOSim, pop_size: int = 30, epochs: int = 300,
           crossover_prob: float = 0.8, mutation_prob: float = 0.15,
           target_ratio: float = 0.80, seed: int | None = None,
           verbose: bool = True) -> dict:
    if seed is not None:
        np.random.seed(seed)
    
    total_act = ao.cfg.dm_actuators ** 2
    
    def fitness(v):
        return ao.pib(ao.get_image(v))
    
    population = np.random.uniform(-0.5, 0.5, (pop_size, total_act))
    fitnesses = np.array([fitness(p) for p in population])
    
    best_idx = np.argmax(fitnesses)
    best_chrom = population[best_idx].copy()
    best_f = float(fitnesses[best_idx])
    
    pib_vals = [best_f]
    ratio_vals = [best_f / ao.state.ideal_pib]
    epochs_list = [0]
    stagnant = 0
    
    for epoch in range(1, epochs + 1):
        new_pop = [best_chrom.copy()]
        for _ in range(pop_size - 1):
            if np.random.rand() < crossover_prob:
                p1 = population[np.random.randint(pop_size)]
                p2 = population[np.random.randint(pop_size)]
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = population[np.random.randint(pop_size)].copy()
            if np.random.rand() < mutation_prob:
                child = np.clip(child + np.random.uniform(-0.15, 0.15, total_act), -1, 1)
            new_pop.append(child)
        population = np.array(new_pop)
        fitnesses = np.array([fitness(p) for p in population])
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_f:
            best_f = float(fitnesses[best_idx])
            best_chrom = population[best_idx].copy()
            stagnant = 0
        else:
            stagnant += 1
        pib_vals.append(best_f)
        ratio_vals.append(best_f / ao.state.ideal_pib)
        epochs_list.append(epoch)
        if best_f / ao.state.ideal_pib >= target_ratio:
            if verbose:
                print(f"  [GA] Converged at epoch {epoch}: PIB={best_f:.4e}, ratio={best_f/ao.state.ideal_pib:.4f}")
            break
        if stagnant > 80:
            if verbose:
                print(f"  [GA] Stagnant at epoch {epoch}")
            break
    
    final_img = ao.get_image(best_chrom)
    final_ratio = best_f / ao.state.ideal_pib
    
    if verbose:
        print(f"  [GA] Best: PIB={best_f:.4e}, ratio={final_ratio:.4f}")
    
    return {
        'pib_curve': np.array(pib_vals),
        'ratio_curve': np.array(ratio_vals),
        'epochs': np.array(epochs_list),
        'best_pib': best_f,
        'final_ratio': final_ratio,
        'converged': final_ratio >= target_ratio,
        'final_img': final_img,
        'ideal_pib': ao.state.ideal_pib,
        'ideal_total': ao.state.ideal_total,
        'r_bucket': ao.state.r_bucket,
    }


def run_sa(ao: AOSim, T_init: float = 1.0, T_min: float = 1e-6,
           cooling: float = 0.99, step_size: float = 0.2, epochs: int = 300,
           target_ratio: float = 0.80, seed: int | None = None,
           verbose: bool = True) -> dict:
    if seed is not None:
        np.random.seed(seed)
    
    total_act = ao.cfg.dm_actuators ** 2
    
    def fitness(v):
        return ao.pib(ao.get_image(v))
    
    current_v = np.zeros(total_act)
    current_E = fitness(current_v)
    best_v = current_v.copy()
    best_E = current_E
    T = T_init
    
    pib_vals = [current_E]
    ratio_vals = [current_E / ao.state.ideal_pib]
    epochs_list = [0]
    
    for epoch in range(1, epochs + 1):
        candidate = np.clip(current_v + np.random.uniform(-step_size, step_size, total_act), -1, 1)
        E_new = fitness(candidate)
        dE = E_new - current_E
        if dE > 0 or np.random.rand() < np.exp(dE / (T + 1e-10)):
            current_v = candidate
            current_E = E_new
            if current_E > best_E:
                best_E = current_E
                best_v = current_v.copy()
        T *= cooling
        if T < T_min:
            T = T_min
        pib_vals.append(best_E)
        ratio_vals.append(best_E / ao.state.ideal_pib)
        epochs_list.append(epoch)
        if best_E / ao.state.ideal_pib >= target_ratio:
            if verbose:
                print(f"  [SA] Converged at epoch {epoch}: PIB={best_E:.4e}, ratio={best_E/ao.state.ideal_pib:.4f}")
            break
    
    final_img = ao.get_image(best_v)
    final_ratio = best_E / ao.state.ideal_pib
    
    if verbose:
        print(f"  [SA] Best: PIB={best_E:.4e}, ratio={final_ratio:.4f}")
    
    return {
        'pib_curve': np.array(pib_vals),
        'ratio_curve': np.array(ratio_vals),
        'epochs': np.array(epochs_list),
        'best_pib': best_E,
        'final_ratio': final_ratio,
        'converged': final_ratio >= target_ratio,
        'final_img': final_img,
        'ideal_pib': ao.state.ideal_pib,
        'ideal_total': ao.state.ideal_total,
        'r_bucket': ao.state.r_bucket,
    }


def plot_convergence(results: dict, save_path: str = "pib_convergence.png"):
    ideal_pib = results['spgd']['ideal_pib']
    ideal_total = results['spgd']['ideal_total']
    target = 0.80 * ideal_pib
    target_ratio = 0.80
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.axhline(ideal_pib / ideal_total, color='green', linestyle='--', linewidth=2,
               label=f'Ideal PIB={ideal_pib/ideal_total:.4f}')
    ax.axhline(target / ideal_total, color='orange', linestyle='--', linewidth=2,
               label=f'Target (80%)={target/ideal_total:.4f}')
    
    for name, res in results.items():
        if name in ('ideal_pib', 'ideal_total', 'r_bucket'):
            continue
        if len(res['epochs']) == 0:
            continue
        ax.plot(res['epochs'], res['pib_curve'] / ideal_total,
                label=f'{name.upper()} (final={res["pib_curve"][-1]/ideal_total:.4f}, '
                      f'ratio={res["final_ratio"]:.4f})',
                linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PIB / Total Power')
    ax.set_title('PIB Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal (100%)')
    ax.axhline(target_ratio, color='orange', linestyle='--', linewidth=2, label='Target (80%)')
    
    for name, res in results.items():
        if name in ('ideal_pib', 'ideal_total', 'r_bucket'):
            continue
        if len(res['epochs']) == 0:
            continue
        ax.plot(res['epochs'], res['ratio_curve'],
                label=f'{name.upper()}', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PIB / Ideal PIB')
    ax.set_title('PIB Ratio vs Ideal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved to {save_path}")
    plt.close()


def plot_spots(results: dict, ao: AOSim, save_path: str = "pib_spots.png"):
    s = ao.state
    cfg = ao.cfg
    
    ideal_img = s.ideal_wave.intensity
    turb_img = ao.get_turb_image(None)
    
    vmax = np.max(ideal_img)
    cx = cfg.npix // 2
    r_px = s.r_bucket / (cfg.aperture / cfg.npix)
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    axes[0].imshow(ideal_img, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title(f'Ideal\nPIB={ao.pib(ideal_img)/s.ideal_total:.4f}')
    axes[0].axis('off')
    axes[0].add_patch(Circle((cx, cx), r_px, fill=False, color='cyan', linewidth=1.5))
    
    axes[1].imshow(turb_img, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(f'+Turb (Cn2={cfg.Cn2:.0e})\nPIB={ao.pib(turb_img)/s.ideal_total:.4f}')
    axes[1].axis('off')
    axes[1].add_patch(Circle((cx, cx), r_px, fill=False, color='cyan', linewidth=1.5))
    
    names = [n for n in results.keys() if n not in ('ideal_pib', 'ideal_total', 'r_bucket')]
    for i, name in enumerate(names[:3]):
        res = results[name]
        axes[2 + i].imshow(res['final_img'], cmap='hot', vmin=0, vmax=vmax)
        axes[2 + i].set_title(f'{name.upper()}\nPIB={res["best_pib"]/s.ideal_total:.4f}, '
                               f'ratio={res["final_ratio"]:.4f}')
        axes[2 + i].axis('off')
        axes[2 + i].add_patch(Circle((cx, cx), r_px, fill=False, color='cyan', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Spots saved to {save_path}")
    plt.close()


def main():
    np.random.seed(42)
    
    NPIX = 128
    APERTURE = 0.005
    WAVELENGTH = 1550e-9
    FOCAL_LENGTH = 0.5
    DIST = 1000.0
    CN2 = 1e-9
    DM_ACTUATORS = 8
    TARGET_RATIO = 0.80
    EPOCHS_SPGD = 200
    EPOCHS_HEURISTIC = 100
    
    print("=" * 70)
    print("AO PIB Optimization with Validated digitaltwin Physics")
    print("=" * 70)
    print(f"Grid: {NPIX}x{NPIX}, Aperture: {APERTURE*1e3:.1f}mm, "
          f"Lambda: {WAVELENGTH*1e9:.0f}nm, f: {FOCAL_LENGTH}m")
    print(f"Turbulence: Cn2={CN2:.0e}, Dist={DIST}m, Subharmonics: OFF")
    print(f"Target: PIB reaches {TARGET_RATIO*100:.0f}% of ideal")
    print()
    
    cfg = SimConfig(
        npix=NPIX,
        aperture=APERTURE,
        wavelength=WAVELENGTH,
        focal_length=FOCAL_LENGTH,
        propagation_distance=DIST,
        Cn2=CN2,
        dm_actuators=DM_ACTUATORS,
        subharmonics=0,
        r_frac=0.5,
    )
    
    ao = AOSim(cfg, seed=42)
    s = ao.state
    
    print(f"System: {ao}")
    print(f"r_bucket: {s.r_bucket/(APERTURE/NPIX):.0f}px")
    print(f"ideal_pib: {s.ideal_pib/s.ideal_total:.4f} (absolute: {s.ideal_pib:.4e})")
    print()
    
    turb_img = ao.get_turb_image()
    turb_pib = ao.pib(turb_img)
    turb_strehl = ao.strehl(turb_img)
    print(f"Turbulent: PIB={turb_pib/s.ideal_total:.4f} ({turb_pib/s.ideal_pib:.4f}x ideal), "
          f"Strehl={turb_strehl:.4f}")
    print(f"Target PIB: {TARGET_RATIO * s.ideal_pib/s.ideal_total:.4f} ({TARGET_RATIO*100:.0f}% of ideal)")
    print()
    
    results = {}
    
    print("=" * 70)
    print("STEP 1: SPGD with Zernike modes (11 modes, Adam)")
    print("=" * 70)
    np.random.seed(42)
    best_spgd = run_spgd_zernike(
        ao, n_modes=11, delta=0.1, alpha=5e-3,
        epochs=EPOCHS_SPGD, target_ratio=TARGET_RATIO, verbose=True
    )
    results['spgd'] = best_spgd
    
    print()
    print("=" * 70)
    print("STEP 2: SPGD with voltage-based DM")
    print("=" * 70)
    np.random.seed(42)
    spgd_v = run_spgd_voltage(ao, delta=0.5, gamma=1e-3, epochs=EPOCHS_SPGD,
                              target_ratio=TARGET_RATIO, seed=42, verbose=True)
    results['spgd_v'] = spgd_v
    
    print()
    print("=" * 70)
    print("STEP 3: Heuristic algorithms (PSO, GA, SA)")
    print("=" * 70)
    
    np.random.seed(42)
    res_pso = run_pso(ao, n_particles=15, epochs=EPOCHS_HEURISTIC,
                      target_ratio=TARGET_RATIO, seed=42, verbose=True)
    results['pso'] = res_pso
    
    np.random.seed(42)
    res_ga = run_ga(ao, pop_size=20, epochs=EPOCHS_HEURISTIC,
                    target_ratio=TARGET_RATIO, seed=42, verbose=True)
    results['ga'] = res_ga
    
    np.random.seed(42)
    res_sa = run_sa(ao, T_init=1.0, cooling=0.99, step_size=0.3, epochs=EPOCHS_HEURISTIC,
                    target_ratio=TARGET_RATIO, seed=42, verbose=True)
    results['sa'] = res_sa
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ideal PIB:       {s.ideal_pib/s.ideal_total:.4f} (absolute: {s.ideal_pib:.4e})")
    print(f"Turbulent PIB:   {turb_pib/s.ideal_total:.4f} (ratio: {turb_pib/s.ideal_pib:.4f})")
    print(f"Target (80%):    {TARGET_RATIO * s.ideal_pib/s.ideal_total:.4f}")
    print()
    print(f"{'Algorithm':<10} {'Final PIB':>12} {'Ratio':>8} {'Converged':>10}")
    print("-" * 45)
    for name in ('spgd', 'spgd_v', 'pso', 'ga', 'sa'):
        r = results[name]
        status = "YES" if r['converged'] else "NO"
        print(f"{name.upper():<10} {r['best_pib']/s.ideal_total:>12.4f} {r['final_ratio']:>8.4f} {status:>10}")
    
    plot_convergence(results)
    plot_spots(results, ao)
    
    return results


if __name__ == '__main__':
    main()
