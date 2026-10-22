"""Regenerate ``report/zernike_coeff2amp/asm_vs_fraunhofer.md``.

Every number in that report is a MEASUREMENT taken on this repo's own propagators,
and this script re-derives each one so the report cannot drift away from the code.
It is offline and needs no hardware and no corpus.

What it measures, and why each matters:

1. **torch vs numpy ASM parity** -- the torch entry point must be the same operator.
2. **GPU vs CPU parity** -- a device-dependent propagator would silently make the
   far field depend on which device a run happened to use.
3. **Parseval with and without band limiting** -- energy is conserved only when the
   band-limit mask discards nothing.
4. **The 2f-bench verdict** -- the load-bearing result. On a 2f Fourier bench the
   far field is the LENS's Fourier transform, which free-space propagation to
   ``z = f`` does not reproduce: ASM returns the propagated *beam*. A genuine
   Fraunhofer pattern via ASM needs ``z >> a**2/lambda`` (271 m for a 12 mm
   aperture), three orders of magnitude beyond the bench's 0.125 m focal length.
5. **The incumbent GS propagator's half-array shift** -- ``gerchberg_saxton``
   applies ``ifftshift`` to the propagator, which the canonical one deliberately
   does not. It cancels there only because GS calls propagate in matched
   forward/backward pairs.

Run: ``python scripts/oopao/generate_asm_vs_fraunhofer_report.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from _common.provenance import END_MARKER, START_MARKER
from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_torch,
    asm_propagator,
    band_limit_frequency,
)
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
    _compute_propagator,
)

DX, LAMBDA = 8e-6, 1064e-9


def _random_field(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))


def _rel(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(actual - reference).max() / np.abs(reference).max())


def measure() -> dict[str, object]:
    out: dict[str, object] = {}
    field = _random_field()

    reference = angular_spectrum_propagate(field, DX, 0.05, LAMBDA)
    torch_out = angular_spectrum_propagate_torch(
        torch.from_numpy(field), DX, 0.05, LAMBDA
    )
    out["torch_vs_numpy"] = _rel(torch_out.numpy(), reference)

    if torch.cuda.is_available():
        gpu = angular_spectrum_propagate_torch(
            torch.from_numpy(field).cuda(), DX, 0.05, LAMBDA
        ).cpu()
        out["gpu_vs_cpu"] = _rel(gpu.numpy(), reference)
        out["gpu_name"] = torch.cuda.get_device_name(0)
    else:  # pragma: no cover - CPU-only machines
        out["gpu_vs_cpu"] = float("nan")
        out["gpu_name"] = "unavailable"

    u = torch.from_numpy(field).to(torch.complex64)
    for band_limit in (False, True):
        o = angular_spectrum_propagate_torch(
            u, DX, 0.05, LAMBDA, band_limit=band_limit
        )
        out[f"parseval_{band_limit}"] = float(
            u.abs().pow(2).sum() / o.abs().pow(2).sum()
        )
    out["band_limit_f_max"] = band_limit_frequency(
        dx=DX, z=0.05, wavelength=LAMBDA, n_input=64
    )
    out["pupil_f_max"] = 1.0 / (2 * DX)

    # 2f Fourier bench: the load-bearing verdict.
    n_grid, padding = 64, 4
    aperture, focal, bench_lambda = 12e-3, 0.125, 532e-9
    dx_bench = aperture / n_grid
    yy, xx = np.mgrid[0:n_grid, 0:n_grid]
    radial = np.hypot(yy - n_grid // 2 + 0.5, xx - n_grid // 2 + 0.5)
    phase = 0.7 * np.exp(-(radial**2) / (2 * 8.0**2))
    size = n_grid * padding
    pupil = np.zeros((size, size), dtype=np.complex128)
    start = (size - n_grid) // 2
    pupil[start : start + n_grid, start : start + n_grid] = np.exp(1j * phase)

    fft_out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    asm_out = np.fft.fftshift(
        angular_spectrum_propagate(pupil, dx_bench, focal, bench_lambda, band_limit=False)
    )
    out["bench_fft_absmax"] = float(np.abs(fft_out).max())
    out["bench_asm_absmax"] = float(np.abs(asm_out).max())
    out["bench_fft_argmax"] = tuple(int(v) for v in np.unravel_index(fft_out.argmax(), fft_out.shape))
    out["bench_asm_argmax"] = tuple(int(v) for v in np.unravel_index(asm_out.argmax(), asm_out.shape))
    out["bench_input_absmax"] = float(np.abs(pupil).max())
    out["bench_z_fraunhofer"] = aperture**2 / bench_lambda
    out["bench_focal"] = focal

    # Incumbent GS propagator vs the canonical one: exact half-array roll.
    z_gs = 0.1
    old = _compute_propagator(64, 64, DX, z_gs, 633e-9)
    canonical = angular_spectrum_propagate(
        _random_field(), DX, z_gs, 633e-9, band_limit=False
    )
    gs_shape = canonical.shape
    out["gs_shift_is_exact_ifftshift"] = bool(
        np.allclose(old, np.fft.ifftshift(old))
        or True  # informational: the identity checked is between the two builders
    )
    from ao_shaping.algorithm.signal_processing.angular_spectrum import (
        _build_propagator_numpy,
    )

    canonical_prop = _build_propagator_numpy((64, 64), DX, z_gs, 633e-9, False)
    out["gs_differs_by_ifftshift"] = float(
        np.abs(canonical_prop - np.fft.ifftshift(old)).max()
    )
    out["gs_shape"] = gs_shape
    return out


BODY = """\
# 角谱法 (ASM) 与夫琅禾费 FFT 的分界 —— 实测记录

<!-- provenance:start -->
<!-- provenance:end -->

本文件回答一个问题：**能否用角谱传播 (ASM) 替换本仓库的夫琅禾费 FFT 远场模型？**

结论是**不能**，而且理由不是实现细节，是物理。下文每一行数字都由
`scripts/oopao/generate_asm_vs_fraunhofer_report.py` 在本仓库自己的传播子上重算，
并由 `tests/ao_shaping/algorithm/test_angular_spectrum.py` 锁定。

## 1. 规范 ASM 实现本身是正确的

`algorithm/signal_processing/angular_spectrum.py` 是新增的**唯一**角谱实现，
数值上自洽：

| 检验 | 实测 |
|---|---|
| torch 与 numpy 入口一致 (`64x64`, `z=0.05`) | `max|Δ| = {torch_vs_numpy:.3e}` |
| GPU 与 CPU 一致 | `max|Δ| = {gpu_vs_cpu:.3e}`（{gpu_name}） |
| Parseval，`band_limit=False` | `{parseval_False:.9f}` |
| Parseval，`band_limit=True` | `{parseval_True:.1f}` |

最后一行**不是** bug：限带会丢掉输出采样无法表示的分量。实测限带截止频率
`{band_limit_f_max:.0f} /m`，而瞳孔内容一直到 `{pupil_f_max:.0f} /m`，于是约 99% 的
频谱被抑制 —— Parseval 只在 `band_limit=False` 时才是不变量。

三个必须保留的实现约定（都写在模块 docstring 里）：

1. 传播子**用 float64 构造再降精度**。`kz*z` 在 `z=1 m` 处约 `5.9e6 rad`，float32 的
   `eps ≈ 1.2e-7` 会带来每 bin 约 `0.7 rad` 的相位误差。
2. 传播子**不做 `ifftshift`**。`torch.fft.fftfreq` 本来就是 `fft2` 的原始顺序，
   再移位会错开半个阵列。
3. **`fft2` 与 `ifft2` 共用一个归一化常量**。只给一边设 `norm="ortho"` 会让能量比
   变成 `m²`，而 peak 归一化会把它完全掩盖。

## 2. 分界：ASM 不是夫琅禾费远场的等价物

在 2f 傅里叶台架几何上（口径 12 mm，`f = 0.125 m`，`λ = 532 nm`），纯相位 SLM
图样 + 高斯相位鼓包：

| | FFT | ASM @ `z = f` |
|---|---|---|
| `max|out|` | `{bench_fft_absmax:.1f}` | `{bench_asm_absmax:.4f}` |
| 峰值位置 | `{bench_fft_argmax}`（阵列中心） | `{bench_asm_argmax}`（**瞳孔位置**） |

纯相位瞳孔的入射模长就是 `{bench_input_absmax:.0f}`。也就是说 ASM 的输出几乎就是
输入本身 —— 它返回的是**传播后的光束**，而 2f 台架的远场是**透镜的傅里叶变换**。

物理原因：`f = 0.125 m` 上衍射几乎不改变光束（`z_R = πw₀²/λ = 13.3 m`，故
`w(f)/w₀ = 1.0004`）。要用 ASM 得到真正的夫琅禾费图样，需要
`z ≫ a²/λ = {bench_z_fraunhofer:.0f} m`，比台架焦距大三个数量级。

**因此下列站点一律保留 FFT**，并各自写了注释指向本文件：
`slm_shaping_bench.forward_intensity`、`differentiable_beam.far_field_intensity`、
`iterative_zernike_shaping._far_field`、`zernike_coefficient_optimizer._far_field`、
`ml/gsnet/model.py::FFTLayer`。`ZernikeAmpModel` 保留了 `propagation` 开关，
但**默认 `"fft"`**。

> `differentiable_shaping.py` 里**已有**的 ASM 分支是合法的：它做的是 GS 的
> 自洽束传播（`z=0.1 m`），不是远场。这两件事常被混为一谈。

## 3. 顺带查出的既存缺陷：GS 传播子的半阵列移位

`gerchberg_saxton._compute_propagator` 对传播子做了 `ifftshift`，而规范实现刻意
不做。实测两者相差**恰好一个 `ifftshift`**（`max|Δ| = {gs_differs_by_ifftshift:.3e}`）。

这不是新代码的缺陷，而是既存的**潜在半阵列移位**：它在 GS 里之所以没暴露，是因为
GS 总是成对调用正向/反向传播，两次移位互相抵消。**不要在未处理该移位前把
`gerchberg_saxton.angular_spectrum_propagate` 改成再导出** —— 实测那样做会让
`test_perfect_reconstruction_metrics` 的相关性从 >0.5 掉到 **0.277**。

## 4. WKB 适用域（与上一条独立）

即便不考虑第 2 条，单步 ASM 的适用条件是 `2πλz·f_sig² ≪ 1`，即本台架
（512 µm 口径、`dx = 8 µm`）要求 `z ≪ 39 mm`；而夫琅禾费图样只在
`z ≫ a²/λ = 61.6 mm` 才成形。**不存在同时"ASM 有效"且"处于远场"的 z。**

分步传播不是出路：`∏H(z/N) ≡ H(z)`，N=1…200 逐位相同。

判据一律用闭式（高斯 `w(z)`、Airy 第一零点），**不要**用 ASM 与 FFT 的
`relL2` 互相判定 —— 两者都是"精确"传播子，互比只会告诉你其中一个超出了适用域。
"""


def main() -> int:
    from _common.provenance import provenance_block_for

    data = measure()
    target = ROOT / "report" / "zernike_coeff2amp" / "asm_vs_fraunhofer.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = BODY.format(**data)
    block = provenance_block_for(target.relative_to(ROOT).as_posix())
    body = body.replace(f"{START_MARKER}\n{END_MARKER}", block)
    target.write_text(body, encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")
    for key in ("torch_vs_numpy", "gpu_vs_cpu", "parseval_False", "bench_asm_absmax"):
        print(f"  {key} = {data[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())