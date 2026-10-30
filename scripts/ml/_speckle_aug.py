"""Shared image-augmentation kernels for the two ``scripts/ml`` loss benchmarks.

``closed_loop_loss_taxonomy.py`` and ``speckle_loss_bench.py`` carried
byte-identical copies of ``_gauss_kernel`` and ``_blur``; the corruption
variants they score losses against (``over_smooth`` / ``missing_speckle``)
call ``_blur`` with the same sigmas, so a drifted copy would make the two
benchmark tables measure different images. These are augmentation kernels,
not evaluation statistics -- they deliberately do **not** live in
``ml/zernike/eval_stats.py``.
"""

from __future__ import annotations

import torch


def _gauss_kernel(sigma: float) -> torch.Tensor:
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss_kernel(sigma)
    r = (k.numel() - 1) // 2
    n, c, h, w = img.shape
    f = img.reshape(n * c, 1, h, w)
    f = torch.nn.functional.pad(f, (r, r, 0, 0), mode="reflect")
    f = torch.nn.functional.conv2d(f, k.view(1, 1, 1, -1))
    f = torch.nn.functional.pad(f, (0, 0, r, r), mode="reflect")
    f = torch.nn.functional.conv2d(f, k.view(1, 1, -1, 1))
    return f.reshape(n, c, h, w)
