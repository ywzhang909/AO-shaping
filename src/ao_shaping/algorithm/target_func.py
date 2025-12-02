import numpy as np

class ImageTargetFunc:
    def __init__(self, x, y):
        assert x.type == y.type, "x and y must have the same type"

        if isinstance(x, int):
            self.xv, self.yv = np.ogrid[:x, :y]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                self.xv, self.yv = x, y
            elif x.ndim == 1:
                self.xv, self.yv = np.meshgrid(x, y)

        self.dist_mat = np.sqrt(self.xv**2 + self.yv**2)

    def __get_pib_mask(self, pib_radius):
        pib_mask = self.dist_mat <= pib_radius
        return pib_mask

    def pib(self, img, pib_radius):
        pib_mask = self.__get_pib_mask(pib_radius)
        pib_ratio = calc_pib_ratio(img, pib_mask)
        return pib_ratio