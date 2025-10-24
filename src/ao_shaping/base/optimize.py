from typing import Callable

import numpy as np

from ao_shaping.algorithm.adam import Base

class DM_Optimizer:
    def __init__(self, delta, opt_algri:Base, target_func:Callable[[np.ndarray], np.ndarray]):
        self.delta = delta
        self.opt_algri = opt_algri
        self.target_func = target_func
        
    def optimize(self, init_v:np.ndarray, epochs:int):
        v = init_v.copy()
        for _ in range(epochs):
            grad = self.target_func(v)
            v -= self.opt_algri.update(grad)
        return v
        
