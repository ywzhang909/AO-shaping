#!/usr/bin/env python3
"""Wavefront reconstruction demo (extracted from
``ao_shaping.algorithm.signal_processing.wavefront`` — see issue #90).

Builds a synthetic tilt+sphere+noise wavefront, recovers it from the measured
slopes via ``reconstruct_wavefront``, and plots the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.algorithm.signal_processing.wavefront import reconstruct_wavefront


def main() -> None:
    N = 32
    # -------------- 伪造：X tilt + 球面 + 噪声 --------------
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # 1. X 倾斜（只随 x 线性变化）
    tx = 0.5 * X

    # 2. 球面项（r²）
    sphere = 0.4 * (X**2 + Y**2)

    # 3. 高频随机噪声
    noise = 0.05 * np.random.randn(N, N)

    # 合成真实波前
    true_phi = tx + sphere + noise
    # 数值差分得到斜率
    sx = 0.5*(np.roll(true_phi, -1, axis=1) - np.roll(true_phi, 1, axis=1))[:, 1:-1]
    sy = 0.5*(np.roll(true_phi, -1, axis=0) - np.roll(true_phi, 1, axis=0))[1:-1, :]
    sx = sx[1:-1, :]              # 内部 N-2×N-2，补零到 N×N
    sy = sy[:, 1:-1]
    sx = np.pad(sx, 1, mode='constant')
    sy = np.pad(sy, 1, mode='constant')

    phi, rms = reconstruct_wavefront(sx, sy, True)
    phi_center = 0.25*(phi[:-1,:-1] + phi[1:,:-1] + phi[:-1,1:] + phi[1:,1:])

    plt.figure(figsize=(9,3))
    plt.subplot(231)
    plt.imshow(true_phi)
    plt.title('true (tilt+sphere+noise)')
    plt.subplot(232)
    plt.imshow(phi)
    plt.title('reconstructed')
    plt.subplot(233)
    plt.imshow(true_phi-phi_center, vmin=-0.05, vmax=0.05)
    plt.title("Recovered RMS = %.3f" % rms)
    plt.subplot(234)
    plt.imshow(sx)
    plt.title("sx")
    plt.subplot(235)
    plt.imshow(sy)
    plt.title("sy")
    plt.show()


if __name__ == "__main__":
    main()
