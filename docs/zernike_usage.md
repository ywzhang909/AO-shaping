# Zernike 使用指南

> 本文件从 README.md 拆出（2026-10-13）。

> **Canonical 入口 (单一事实源)**: 全项目所有 Zernike 纯数学 (模式枚举 / 索引换算 / 相位生成 / 系数解析 / 单位换算) 统一走 `utils/wavefront/` 两个模块。**任何脚本、runner、工具、GUI 不得自行实现** Noll↔(n,m) 查表、Zernike 多项式求值或相位生成 —— 2026-09 已完成去重重构, 此前散落在 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 的重复实现已全部收敛到这两层。

## 两层入口

| 层 | 模块 | 公开 API | 何时用 |
|---|---|---|---|
| API 层 (首选) | `ao_shaping.utils.wavefront.zernike_utils` | `parse_zernike_coefficients` (Noll dict / (n,m) dict / Noll 序数组 → {(n,m):amp}), `generate_zernike_phase` (系数 → **raw 弧度**相位图), `list_zernike_modes` (→ [(noll,n,m,name)]), `coefficients_to_array` ((n,m) dict → Noll 序数组), `um_to_waves` (WFS µm→λ) | 绝大多数场景: 解析 / 生成 / 枚举 / 单位换算 |
| 引擎层 (复用/底层) | `ao_shaping.utils.wavefront.zernike_calc` | `ZernikeGenerator` (带网格缓存: `generate_noll` / `generate_polynomial` / `generate` / `fit`), `noll_to_nm` / `ZernikeGenerator.nm_to_noll` (支持任意 Noll), `zernike_modes` / `noll_indices` (模式枚举), `calc_n_zernike_terms`, `get_zernike_name`, `fit_zernike` | 同一分辨率反复生成相位 (复用实例, 省 RZern 建网格开销), 或需要底层索引换算 / 拟合 |

单向依赖: `zernike_utils` → `zernike_calc` (API 层内部创建/复用引擎)。上层 (optimizer / runner / scripts / GUI) 只 import 这两层, 不得反向。

## 最小示例

```python
from ao_shaping.utils.wavefront.zernike_utils import (
    parse_zernike_coefficients,
    generate_zernike_phase,
    list_zernike_modes,
    um_to_waves,
)

# 1) 解析系数 — 3 种输入格式, 输出统一为 {(n, m): amplitude}
coeffs = parse_zernike_coefficients({"5": 1.0, "13": 0.5})   # Noll 索引 dict
coeffs = parse_zernike_coefficients({(2, 0): 1.0, (4, 0): 0.5})  # (n, m) dict
coeffs = parse_zernike_coefficients([0.0, 0.0, 0.0, 1.0])    # Noll 序扁平数组

# 2) 生成相位 — 输出 **raw 未包裹弧度** float64 (孔径外 NaN), 非 uint16
phase = generate_zernike_phase(coeffs, resolution=(1920, 1200), n_max=4)

# 3) 枚举模式 → [(noll, n, m, name), ...] (Noll 序)
for noll, n, m, name in list_zernike_modes(4):
    print(noll, (n, m), name)
```

同一分辨率反复生成时, 直接持有 `ZernikeGenerator` 实例复用 (内部缓存 RZern 坐标网格):

```python
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

gen = ZernikeGenerator((1920, 1200), n_orders=4)   # 一次建网格, 多次复用
gen.set_bits(10)
img = gen.generate_polynomial({(2, 0): 1.0, (4, 0): 0.5})
```

## 红线 (违反 = 重新引入重复 / 约定漂移 / 单位 bug)

1. **禁止自写 Noll↔(n,m) 查表或模式枚举**。历史教训: `noll_to_nm_legacy` (那里 Noll 5=(2,0), 与 canonical 相反) 曾与 canonical 并存导致两套索引混用, 已删除; 新代码一律 `zernike_calc.noll_to_nm()` / `zernike_utils.list_zernike_modes()`。
2. **禁止生成器自行 `mod 2π`**。`generate_zernike_phase()` / `ZernikeGenerator.*` 返回 **raw 未包裹弧度**; 唯一 wrap 点在 SLM 驱动 `Santec.create_phase_from_array()` (弧度→灰度)。弧度→灰度统一走 `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)`。
3. **禁止 min-max 归一化相位**。`(p−min)/(max−min)` 归一化使图案**尺度无关** (系数 ×1 与 ×4 输出字节完全相同, 幅度不可控), 是已知反模式, 不得用于新代码。
   历史例子: `ZernikeDM.generate_phase` (于 2026-09 修复) 与 `PatternHelper._zernike_to_uint16` (**于 2026-09 删除**)。
4. **WFS 系数单位必须统一**。WFS `get_zernike()` 返回 **µm**; 参与响应矩阵 / 矫正运算前必须 `um_to_waves()` (µm→λ, ÷0.532); 反解出的 λ 系数在喂给 `generate_zernike_phase` / `make_phase` 前必须 **×2π** (λ→rad)。两个真实 bug (2026-09-16, 均因单位混用: 系数放大 1.88× / 相位缩小 6.28×) 修复后闭环 RMS 改善 13.8% → 42.1%。
5. **Noll 约定 = Noll 1976 (aotools)**: Noll 4=(2,0) defocus, Noll 5=(2,-2) astig, Noll 11=(4,0) spherical, Noll 13=(4,-2) ⚠ (不是 (2,0))。`zernike_calc.noll_indices` (Noll 序) 与 `zernike_modes` ((n,m) 字典序)**顺序不同, 不可互换**; 完整前 15 阶映射表见 `zernike_utils` 模块 docstring。
