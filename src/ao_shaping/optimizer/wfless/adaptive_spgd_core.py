from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple

import math
import numpy as np


@dataclass(frozen=True)
class AdaptiveSpgdConfig:
    stage_weights: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.6, 0.2, 0.2),
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
    )
    stagnate_win: int = 10
    stagnate_boost: float = 1.5
    enable_stagnate_boost: bool = True
    boost_w_u_only: bool = True
    min_improve_frac: float = 0.01


def get_stage_weights(cfg: AdaptiveSpgdConfig, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
    if total_epochs <= 0:
        return cfg.stage_weights[0]
    t = epoch / total_epochs if total_epochs > 0 else 0.0
    if t < 1.0 / 3.0:
        return cfg.stage_weights[0]
    if t < 2.0 / 3.0:
        return cfg.stage_weights[1]
    return cfg.stage_weights[2]


def should_boost_stagnation(cv_history, win: int = 10, min_improve_frac: float = 0.01) -> bool:
    if win <= 0 or len(cv_history) == 0:
        return False
    hist = list(cv_history)
    if len(hist) < win + 1:
        return False
    best_cur_win = min(hist[-win:])
    best_prev_win = min(hist[-win - 1:-1])
    if best_prev_win <= 0:
        return False
    improve = (best_prev_win - best_cur_win) / best_prev_win if best_prev_win > 0 else 0.0
    if improve < min_improve_frac:
        return True
    return False


class GradientBalancer:
    def __init__(self, n_terms: int = 3, ema_decay: float = 0.9, eps: float = 1e-9):
        self.n_terms = int(n_terms)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.gbar = np.ones(self.n_terms, dtype=float)

    def reset(self) -> None:
        self.gbar = np.ones(self.n_terms, dtype=float)

    def compute(self, dJ_terms: Sequence[float] | np.ndarray, da: Sequence[float] | np.ndarray | None = None,
                weights: Sequence[float] | np.ndarray | None = None) -> float:
        dJ = np.asarray(dJ_terms, dtype=float)
        if dJ.size != self.n_terms:
            # pad or truncate if needed
            if dJ.size < self.n_terms:
                dJ = np.pad(dJ, (0, self.n_terms - dJ.size))
            else:
                dJ = dJ[:self.n_terms]
        g = np.abs(dJ)
        self.gbar = self.ema_decay * self.gbar + (1.0 - self.ema_decay) * g
        w = np.ones(self.n_terms, dtype=float)
        if weights is not None:
            w_in = np.asarray(weights, dtype=float)
            if w_in.size == self.n_terms:
                w = w_in
            elif w_in.size > 0:
                if w_in.size < self.n_terms:
                    w[:w_in.size] = w_in
                else:
                    w = w_in[:self.n_terms]
        denom = np.sum(self.gbar * w)
        if math.isfinite(denom) and denom > self.eps:
            return float(np.sum(dJ * w / denom))
        return float(np.sum(dJ * w))


def adapt_step_size(delta: float, accept_history: Sequence[bool] | list[bool] | tuple[bool, ...],
                    window: int = 10, lo: float = 0.02, hi: float = 0.5) -> float:
    if not accept_history:
        return float(delta)
    hist = list(accept_history)
    if window > 0 and len(hist) > window:
        hist = hist[-window:]
    succ = sum(1 for x in hist if x)
    total = len(hist)
    if total <= 0:
        rate = 0.0
    else:
        rate = succ / total
    nd = float(delta)
    if rate > 0.6:
        nd = nd * 1.4
    elif rate < 0.3:
        nd = nd * 0.6
    if math.isfinite(nd):
        nd = float(np.clip(nd, lo, hi))
    else:
        nd = float(delta)
    return nd


def snr_signal(dj_pos_neg_mag: float, sigma_j: float, sqrt2: float = 1.4142135623730951) -> float:
    sj = float(sigma_j)
    if sj <= 0.0:
        return float("inf")
    return float(dj_pos_neg_mag) / (sqrt2 * sj + 1e-18)


def sigma_scale_from_snr(snr: float, sigma_min: float = 0.01, sigma_max: float = 0.3,
                         enlarge_if_snr_lt: float = 2.0, shrink_if_snr_gt: float = 8.0,
                         enlarge_factor: float = 1.3, shrink_factor: float = 0.85) -> float:
    s = float(snr)
    if math.isfinite(s) and s < enlarge_if_snr_lt:
        return float(enlarge_factor)
    if math.isfinite(s) and s > shrink_if_snr_gt:
        return float(shrink_factor)
    return 1.0
