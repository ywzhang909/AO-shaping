from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math

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
    if len(hist) < win + 1 and len(hist) >= win:
        # need prev window comparison - if we just started, check vs best so far? but test says need 10 epoch window comparison
        pass  # handle
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
