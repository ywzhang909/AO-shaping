# Zernike 使用规范 (canonical 入口)

> **单一事实源**: 全项目所有 Zernike 纯数学 (模式枚举 / 索引换算 / 相位生成 / 系数解析 / 单位换算) 统一走 `utils/wavefront/` 两层。**任何脚本、runner、tools、GUI、optimizer 不得自行实现** Noll↔(n,m) 查表、Zernike 多项式求值或相位生成 (2026-09 去重重构, 此前散落在 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 的重复实现已全部收敛)。详见 README `## Zernike 使用指南`。

| 层 | 模块 | 公开 API | 何时用 |
|---|---|---|---|
| API 层 (首选) | `utils/wavefront/zernike_utils.py` | `parse_zernike_coefficients`, `generate_zernike_phase`, `list_zernike_modes` (4元组 noll,n,m,name), `coefficients_to_array`, `um_to_waves`, `LAMBDA_UM` | 绝大多数场景 |
| 引擎层 (复用/底层) | `utils/wavefront/zernike_calc.py` | `ZernikeGenerator` (网格缓存: `generate_noll`/`generate_polynomial`/`generate`/`fit`), `noll_to_nm`/`nm_to_noll`, `zernike_modes`/`noll_indices`, `calc_n_zernike_terms`, `get_zernike_name`, `fit_zernike` | 同分辨率反复生成 (复用实例), 或底层索引/拟合 |

单向依赖: `zernike_utils` → `zernike_calc`; 上层只 import 这两层, 不得反向。

**红线** (对应 ANTI-PATTERNS):
1. 自写 Noll↔(n,m) 查表 / 模式枚举 — `noll_to_nm_legacy` (Noll 5=(2,0), 与 canonical 相反) 已删除, 勿再引入第二套索引
2. 生成器自行 `mod 2π` — 一律返回 **raw 未包裹弧度**; 唯一 wrap 点 `Santec.create_phase_from_array()`
3. min-max 归一化相位 — 已知反模式 (尺度无关, 幅度不可控): `ZernikeDM.generate_phase`（于 2026-09 修复）。同樾的 `PatternHelper._zernike_to_uint16` 于 2026-09 **已删除**（仅作历史参考，不再是现存代码）
4. WFS µm 系数未经 `um_to_waves()` 直接参与运算; λ 系数未 ×2π 直接喂 `make_phase`/`generate_zernike_phase`
5. `zernike_calc.noll_indices` (Noll 序) 与 `zernike_modes` ((n,m) 字典序) 顺序不同, 不可互换

