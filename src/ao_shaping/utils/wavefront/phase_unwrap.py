from __future__ import annotations

from enum import Enum, auto
from typing import Literal

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import convolve2d


class UnwrapStrategy(Enum):
    GOLDSTEIN = auto()
    QUALITY_GUIDED = auto()
    ITERATIVE = auto()
    LEAST_SQUARES = auto()
    REGION_GROWING = auto()
    SIMPLE = auto()


def wrap(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase, 2 * np.pi)


def unwrap_1d(phase: np.ndarray) -> np.ndarray:
    diff = np.diff(phase)
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    cumsum = np.cumsum(diff)
    result = np.concatenate([[phase[0]], phase[0] + cumsum])
    return result


class PhaseUnwrapper:
    def __init__(
        self,
        resolution: tuple[int, int] = (512, 512),
        strategy: UnwrapStrategy = UnwrapStrategy.ITERATIVE,
    ):
        self.resolution = resolution
        self.strategy = strategy
        self._cached_vars: dict = {}
        self._init_cached_variables()

    def _init_cached_variables(self):
        h, w = self.resolution
        self._cached_vars = {
            'shape': (h, w),
            'y_grid': np.arange(h, dtype=np.float64),
            'x_grid': np.arange(w, dtype=np.float64),
            'yy': np.empty((h, w), dtype=np.float64),
            'xx': np.empty((h, w), dtype=np.float64),
            'yy_grad': np.empty((h, w), dtype=np.float64),
            'xx_grad': np.empty((h, w), dtype=np.float64),
            'mask': np.ones((h, w), dtype=np.bool_),
            'row_idx': np.arange(h, dtype=np.int32),
            'col_idx': np.arange(w, dtype=np.int32),
            'wrap_table': np.arange(2 * np.pi, dtype=np.float64),
        }
        yy, xx = np.meshgrid(self._cached_vars['y_grid'], self._cached_vars['x_grid'], indexing='ij')
        self._cached_vars['yy'] = yy
        self._cached_vars['xx'] = xx
        self._cached_vars['yy_grad'] = yy.astype(np.float32)
        self._cached_vars['xx_grad'] = xx.astype(np.float32)

    def compute_gradient(self, wrapped: np.ndarray):
        h, w = wrapped.shape
        gy = np.zeros((h, w), dtype=np.float64)
        gx = np.zeros((h, w), dtype=np.float64)
        gy[:-1, :] = wrapped[1:, :] - wrapped[:-1, :]
        gx[:, :-1] = wrapped[:, 1:] - wrapped[:, :-1]
        gy = np.mod(gy + np.pi, 2 * np.pi) - np.pi
        gx = np.mod(gx + np.pi, 2 * np.pi) - np.pi
        return gx, gy

    def find_residues(self, wrapped: np.ndarray):
        h, w = wrapped.shape
        gy, gx = self.compute_gradient(wrapped)
        gy_pad = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
        gx_pad = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
        cx = np.zeros((h, w), dtype=np.float64)
        cy = np.zeros((h, w), dtype=np.float64)
        cx[:, 1:] = gx_pad[:, 1:] - gx_pad[:, :-1]
        cy[1:, :] = gy_pad[1:, :] - gy_pad[:-1, :]
        residues = np.mod(np.round((cx + cy) / (2 * np.pi)), 2 * np.pi)
        residues = np.where(np.abs(residues) > 0.5, np.sign(residues), 0)
        return residues

    def goldstein_unwrap(self, wrapped: np.ndarray, branch_cut_threshold: float = 5.0):
        h, w = wrapped.shape
        residues = self.find_residues(wrapped)
        mask = np.ones((h, w), dtype=np.bool_)
        residue_map = np.abs(residues) > 0.5
        branch_cuts = self._place_branch_cuts(residue_map, threshold=branch_cut_threshold)
        mask[branch_cuts] = False
        unwrapped = self._path_follow_unwrap(wrapped, mask)
        return unwrapped

    def _place_branch_cuts(self, residue_map: np.ndarray, threshold: float = 5.0):
        from scipy.ndimage import binary_dilation, generate_binary_structure
        h, w = residue_map.shape
        cuts = residue_map.copy()
        struct = generate_binary_structure(2, 2)
        for _ in range(int(threshold)):
            cuts = binary_dilation(cuts, structure=struct)
        return cuts

    def _path_follow_unwrap(self, wrapped: np.ndarray, mask: np.ndarray):
        h, w = wrapped.shape
        unwrapped = np.zeros((h, w), dtype=np.float64)
        visited = np.zeros((h, w), dtype=np.bool_)
        queue = [(h // 2, w // 2)]
        visited[h // 2, w // 2] = True
        unwrapped[h // 2, w // 2] = wrapped[h // 2, w // 2]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                    visited[ny, nx] = True
                    if abs(dy) + abs(dx) == 1:
                        diff = wrapped[ny, nx] - wrapped[cy, cx]
                        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                        unwrapped[ny, nx] = unwrapped[cy, cx] + diff
                    else:
                        diff1 = wrapped[ny, cx] - wrapped[cy, cx]
                        diff2 = wrapped[ny, nx] - wrapped[ny, cx]
                        diff1 = np.mod(diff1 + np.pi, 2 * np.pi) - np.pi
                        diff2 = np.mod(diff2 + np.pi, 2 * np.pi) - np.pi
                        unwrapped[ny, nx] = unwrapped[cy, cx] + diff1 + diff2
                    queue.append((ny, nx))
        return unwrapped

    def compute_quality_map(self, wrapped: np.ndarray):
        gx, gy = self.compute_gradient(wrapped)
        gmag = np.sqrt(gx**2 + gy**2)
        quality = gaussian_filter(gmag, sigma=2)
        return quality

    def quality_guided_unwrap(self, wrapped: np.ndarray):
        quality = self.compute_quality_map(wrapped)
        h, w = wrapped.shape
        unwrapped = np.zeros((h, w), dtype=np.float64)
        visited = np.zeros((h, w), dtype=np.bool_)
        flat_quality = quality.flatten()
        sorted_indices = np.argsort(flat_quality)[::-1]
        h_idx, w_idx = np.unravel_index(sorted_indices, (h, w))
        start_y, start_x = h_idx[0], w_idx[0]
        unwrapped[start_y, start_x] = wrapped[start_y, start_x]
        visited[start_y, start_x] = True
        queue = [(start_y, start_x)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    diff = wrapped[ny, nx] - wrapped[cy, cx]
                    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                    unwrapped[ny, nx] = unwrapped[cy, cx] + diff
                    queue.append((ny, nx))
        return unwrapped

    def iterative_unwrap(self, wrapped: np.ndarray, num_iterations: int = 10):
        h, w = wrapped.shape
        gx, gy = self.compute_gradient(wrapped)
        gx_cum = np.zeros((h, w), dtype=np.float64)
        gy_cum = np.zeros((h, w), dtype=np.float64)
        for i in range(num_iterations):
            gx_cum = gx_cum + gx
            gy_cum = gy_cum + gy
            gx = np.roll(gx, 1, axis=1)
            gy = np.roll(gy, 1, axis=0)
        cumsum_y = np.cumsum(gy_cum, axis=0)
        cumsum_x = np.cumsum(gx_cum, axis=1)
        phase_est = cumsum_y + cumsum_x - gx_cum[0, 0]
        offset = wrapped[0, 0] - phase_est[0, 0]
        phase_est = phase_est + offset
        residual = self._compute_residual(wrapped, phase_est)
        phase_est = phase_est + self._propagate_corrections(residual)
        return phase_est

    def _compute_residual(self, wrapped: np.ndarray, unwrapped: np.ndarray):
        diff = unwrapped - wrapped
        residual = np.mod(diff + np.pi, 2 * np.pi) - np.pi
        return residual

    def _propagate_corrections(self, residual: np.ndarray):
        from scipy.ndimage import gaussian_filter
        corrections = gaussian_filter(residual, sigma=5)
        return corrections

    def least_squares_unwrap(self, wrapped: np.ndarray):
        h, w = wrapped.shape
        gx, gy = self.compute_gradient(wrapped)
        from scipy.fftpack import fft2, ifft2, fftfreq
        kx = fftfreq(w, d=1.0).reshape(1, -1)
        ky = fftfreq(h, d=1.0).reshape(-1, 1)
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1e-10
        gx_fft = fft2(gx)
        gy_fft = fft2(gy)
        denom = -(2 * np.pi) * 1j * kx * gx_fft - (2 * np.pi) * 1j * ky * gy_fft
        phase_fft = denom / k_sq
        unwrapped = np.real(ifft2(phase_fft))
        offset = wrapped[0, 0] - unwrapped[0, 0]
        unwrapped = unwrapped + offset
        return unwrapped

    def region_growing_unwrap(self, wrapped: np.ndarray, quality_threshold: float = 0.1):
        quality = self.compute_quality_map(wrapped)
        h, w = wrapped.shape
        unwrapped = np.zeros((h, w), dtype=np.float64)
        visited = np.zeros((h, w), dtype=np.bool_)
        flat_quality = quality.flatten()
        sorted_indices = np.argsort(flat_quality)[::-1]
        h_idx, w_idx = np.unravel_index(sorted_indices, (h, w))
        for i in range(len(h_idx)):
            if flat_quality[sorted_indices[i]] < quality_threshold:
                break
            cy, cx = h_idx[i], w_idx[i]
            if not visited[cy, cx]:
                if i == 0:
                    unwrapped[cy, cx] = wrapped[cy, cx]
                else:
                    seeds = self._find_nearby_seeds(cy, cx, visited, unwrapped)
                    if seeds:
                        nearest = seeds[0]
                        diff = wrapped[cy, cx] - wrapped[nearest[0], nearest[1]]
                        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                        unwrapped[cy, cx] = unwrapped[nearest[0], nearest[1]] + diff
                visited[cy, cx] = True
        return unwrapped

    def _find_nearby_seeds(self, cy: int, cx: int, visited: np.ndarray, unwrapped: np.ndarray):
        h, w = visited.shape
        seeds = []
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and visited[ny, nx]:
                    seeds.append((ny, nx))
        return seeds[:1]

    def simple_unwrap(self, wrapped: np.ndarray):
        h, w = wrapped.shape
        gy, gx = self.compute_gradient(wrapped)
        cumsum_y = np.cumsum(gy, axis=0)
        cumsum_x = np.cumsum(gx, axis=1)
        unwrapped = cumsum_y + cumsum_x - gy[0, 0] - gx[0, 0] + wrapped[0, 0]
        return unwrapped

    def unwrap(self, wrapped: np.ndarray, strategy: UnwrapStrategy | None = None) -> np.ndarray:
        strategy = strategy or self.strategy
        if strategy == UnwrapStrategy.GOLDSTEIN:
            return self.goldstein_unwrap(wrapped)
        elif strategy == UnwrapStrategy.QUALITY_GUIDED:
            return self.quality_guided_unwrap(wrapped)
        elif strategy == UnwrapStrategy.ITERATIVE:
            return self.iterative_unwrap(wrapped)
        elif strategy == UnwrapStrategy.LEAST_SQUARES:
            return self.least_squares_unwrap(wrapped)
        elif strategy == UnwrapStrategy.REGION_GROWING:
            return self.region_growing_unwrap(wrapped)
        elif strategy == UnwrapStrategy.SIMPLE:
            return self.simple_unwrap(wrapped)
        else:
            return self.iterative_unwrap(wrapped)

    def unwrap_fast(self, wrapped: np.ndarray) -> np.ndarray:
        h, w = wrapped.shape
        if h != self._cached_vars['shape'][0] or w != self._cached_vars['shape'][1]:
            self.resolution = (h, w)
            self._init_cached_variables()
        return self._fast_iterative_unwrap(wrapped)

    def _fast_iterative_unwrap(self, wrapped: np.ndarray):
        h, w = wrapped.shape
        y_grid = self._cached_vars['y_grid']
        x_grid = self._cached_vars['x_grid']
        yy = self._cached_vars['yy']
        xx = self._cached_vars['xx']
        gy = np.zeros((h, w), dtype=np.float64)
        gx = np.zeros((h, w), dtype=np.float64)
        gy[:-1, :] = wrapped[1:, :] - wrapped[:-1, :]
        gx[:, :-1] = wrapped[:, 1:] - wrapped[:, :-1]
        gy = np.mod(gy + np.pi, 2 * np.pi) - np.pi
        gx = np.mod(gx + np.pi, 2 * np.pi) - np.pi
        gx_cum = gx.copy()
        gy_cum = gy.copy()
        for _ in range(3):
            gx_cum = np.cumsum(gx_cum, axis=1)
            gy_cum = np.cumsum(gy_cum, axis=0)
        phase_est = gy_cum + np.roll(gx_cum, 1, axis=1) - gx[0, 0]
        offset = wrapped[0, 0] - phase_est[0, 0]
        phase_est = phase_est + offset
        return phase_est

    def get_cached_gradient_ops(self):
        return {
            'compute_diff_y': lambda arr: np.mod(np.diff(arr, axis=0, prepend=arr[-1:, :]) + np.pi, 2 * np.pi) - np.pi,
            'compute_diff_x': lambda arr: np.mod(np.diff(arr, axis=1, prepend=arr[:, -1:]) + np.pi, 2 * np.pi) - np.pi,
            'cumsum_y': lambda arr: np.cumsum(arr, axis=0),
            'cumsum_x': lambda arr: np.cumsum(arr, axis=1),
        }


def unwrap_phase(
    wrapped: np.ndarray,
    strategy: Literal["goldstein", "quality_guided", "iterative", "least_squares", "region_growing", "simple", "fast"] = "iterative",
    resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    if resolution is None:
        resolution = tuple(wrapped.shape)
    strategy_map = {
        "goldstein": UnwrapStrategy.GOLDSTEIN,
        "quality_guided": UnwrapStrategy.QUALITY_GUIDED,
        "iterative": UnwrapStrategy.ITERATIVE,
        "least_squares": UnwrapStrategy.LEAST_SQUARES,
        "region_growing": UnwrapStrategy.REGION_GROWING,
        "simple": UnwrapStrategy.SIMPLE,
        "fast": None,
    }
    if strategy == "fast":
        unwrapper = PhaseUnwrapper(resolution=resolution)
        return unwrapper.unwrap_fast(wrapped)
    unwrapper = PhaseUnwrapper(resolution=resolution, strategy=strategy_map[strategy])
    return unwrapper.unwrap(wrapped)