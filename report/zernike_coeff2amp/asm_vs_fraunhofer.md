# 角谱法 (ASM) 与夫琅禾费 FFT 的分界 —— 实测记录

<!-- provenance:start -->
> **生成脚本**: [`scripts/generate_asm_vs_fraunhofer_report.py`](../../scripts/generate_asm_vs_fraunhofer_report.py)
> **复现命令**: `python scripts/generate_asm_vs_fraunhofer_report.py`
> **数据/关联脚本**: [`tests/ao_shaping/algorithm/test_angular_spectrum.py`](../../tests/ao_shaping/algorithm/test_angular_spectrum.py)
> **运行环境**: 离线
> **说明**: 角谱法与夫琅禾费 FFT 的分界: 2f 台架实测 ASM 返回传播光束而非远场; 含 Parseval/限带/平移约定与 GS 半阵列移位
<!-- provenance:end -->

**本轮结论**：在本仓库 2f 傅里叶台架上，**ASM 不能替换夫琅禾费 FFT 远场模型，且理由是物理而非实现**：(1) 规范 ASM 实现数值自洽（torch/numpy 入口 `max|Δ|=4.226e-16`，GPU/CPU `4.215e-16`；`band_limit=False` 时 Parseval `1.000000119`），实现不是问题；(2) 2f 台架几何（口径 12 mm、`f=0.125 m`、`λ=532 nm`）下 FFT 与 ASM@`z=f` 输出峰值位置分别为 `(128,128)`（阵列中心）与 `(30,33)`（瞳孔位置），`max|out|` 为 `4057.0` vs `1.1160` —— ASM 返回**传播后的光束**（`z_R=13.3 m`，`w(f)/w₀=1.0004`），远场图样需 `z ≫ a²/λ = 271 m`，比焦距大三个数量级；(3) WKB 适用域（`2πλz·f_sig² ≪ 1` ⇒ `z ≪ 39 mm`）与夫琅禾费域（`z ≫ 61.6 mm`）在 512 µm 口径下**无交集**，分步传播 `∏H(z/N) ≡ H(z)`（N=1…200 逐位相同）亦无出路；(4) 顺带查出 `gerchberg_saxton._compute_propagator` 的既存潜在半阵列移位（与规范实现恰差一个 `ifftshift`，GS 正反向成对调用时互相抵消），未处理前不得改成再导出（实测会让完美重构相关性从 >0.5 掉到 0.277）。判定：五个远场站点（`slm_shaping_bench.forward_intensity`、`differentiable_beam.far_field_intensity`、`iterative_zernike_shaping._far_field`、`zernike_coefficient_optimizer._far_field`、`ml/gsnet/model.py::FFTLayer`）一律保留 FFT；判据用闭式（高斯 `w(z)`、Airy 第一零点），不用 ASM 与 FFT 的 `relL2` 互比。

本文件回答一个问题：**能否用角谱传播 (ASM) 替换本仓库的夫琅禾费 FFT 远场模型？**

结论是**不能**，而且理由不是实现细节，是物理。下文每一行数字都由
`scripts/generate_asm_vs_fraunhofer_report.py` 在本仓库自己的传播子上重算，
并由 `tests/ao_shaping/algorithm/test_angular_spectrum.py` 锁定。

## 1. 规范 ASM 实现本身是正确的

`algorithm/signal_processing/angular_spectrum.py` 是新增的**唯一**角谱实现，
数值上自洽：

| 检验 | 实测 |
|---|---|
| torch 与 numpy 入口一致 (`64x64`, `z=0.05`) | `max|Δ| = 4.226e-16` |
| GPU 与 CPU 一致 | `max|Δ| = 4.215e-16`（NVIDIA GeForce GTX 1070 with Max-Q Design） |
| Parseval，`band_limit=False` | `1.000000119` |
| Parseval，`band_limit=True` | `170.2` |

最后一行**不是** bug：限带会丢掉输出采样无法表示的分量。实测限带截止频率
`4812 /m`，而瞳孔内容一直到 `62500 /m`，于是约 99% 的
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
| `max|out|` | `4057.0` | `1.1160` |
| 峰值位置 | `(128, 128)`（阵列中心） | `(30, 33)`（**瞳孔位置**） |

纯相位瞳孔的入射模长就是 `1`。也就是说 ASM 的输出几乎就是
输入本身 —— 它返回的是**传播后的光束**，而 2f 台架的远场是**透镜的傅里叶变换**。

物理原因：`f = 0.125 m` 上衍射几乎不改变光束（`z_R = πw₀²/λ = 13.3 m`，故
`w(f)/w₀ = 1.0004`）。要用 ASM 得到真正的夫琅禾费图样，需要
`z ≫ a²/λ = 271 m`，比台架焦距大三个数量级。

**因此下列站点一律保留 FFT**，并各自写了注释指向本文件：
`slm_shaping_bench.forward_intensity`、`differentiable_beam.far_field_intensity`、
`iterative_zernike_shaping._far_field`、`zernike_coefficient_optimizer._far_field`、
`ml/gsnet/model.py::FFTLayer`。`ZernikeAmpModel` 保留了 `propagation` 开关，
但**默认 `"fft"`**。

> `differentiable_shaping.py` 里**已有**的 ASM 分支是合法的：它做的是 GS 的
> 自洽束传播（`z=0.1 m`），不是远场。这两件事常被混为一谈。

## 3. 顺带查出的既存缺陷：GS 传播子的半阵列移位

`gerchberg_saxton._compute_propagator` 对传播子做了 `ifftshift`，而规范实现刻意
不做。实测两者相差**恰好一个 `ifftshift`**（`max|Δ| = 0.000e+00`）。

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
