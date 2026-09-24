# slm_zernike_pib.py 待办问题清单（合并后对账版）

生成时间：2026-09-25（对账 2026-09-24 原版；曾于 2026-09-25 被 remote `a3130a7` 重写）
文件：`src/ao_shaping/optimizer/wfless/slm_zernike_pib.py`（**2179 行**）

> **合并影响（2026-09-25, remote a3130a7 `SlmZernikePibConfig` 收敛）**：签名收敛为
> dataclass（`optimize_slm_zernike_pib` 首参 `config=SlmZernikePibConfig`），内部重命名
> `test_pib → ideal_pib_ratio`、`intellij_center → _smart_center`，`to_min` 标量删除 →
> `objective_mode`/`_spgd_sign` 派生，7 个 objective 闭包提取为 `_calc_objective_*` 前置步骤。
> **除行号外，下述全部 TODO 项均未被合并解决** —— 全部仍存在于当前文件。

---

## 一、P0 — 先修的疑似 bug（改架构前先止血）

### P0-1：`ALGORITHM_CHOICES` 默认值加固

- **位置**：L146
- **状态**：**仍未加固**（`heuristic_algorithm_choices()` 默认调用），此前核实**不成立**——
  默认含 `"spgd"`，守卫 L754 对默认路径是 no-op。
- **待办**：写死 `include_spgd=True` + 补 `test_default_algorithm_runs`（sim 后端）锁定（低优先级）。

### P0-2：能量守卫对 `objective="pib"` 失效 ⚠️ 真 bug（仍存在）

- **位置（当前）**：L1532 `best_objective = float(ideal_pib_ratio(init_img)) if objective == "pib" else float(j)`；
  L1801（heuristic 分支 `obj_val = float(ideal_pib_ratio(img)) if ...`）；L1956（SPGD 分支同模式）；
  惩罚定义 L1505 `bad = j - 1e3 if objective_mode == "max" else j + 1e3`。
- **问题**：`calc_objective`（L1483-1507）返回值含 `-1e3` 守卫惩罚，但三处 `best_objective`/
  `obj_val` 从 `ideal_pib_ratio` 取（**不含惩罚**）。守卫触发后 `best_c` 仍可能采纳"被放弃"的评估，
  硬件上 SLM 退出时会留在守卫不允许的状态。L1528-1531 注释补充了"track the objective's OWN value"
  的理由，但未解决惩罚不一致。
- **修法**：`calc_objective` 改返回三元组 `(J_effective, ratio, headline)`，守卫触发时一并惩罚
  headline；`best_objective` 统一从 headline 取，消灭三处 `objective == "pib"` 特判。

### P0-3：`_log_row` 漏记 `w_ee` / `ee_term`（仍存在）

- **位置（当前）**：`_row0` L1611-1639（写 `w_pib/w_rms/w_ee/pib_term/rms_term/ee_term` +
  `_metric_panel` 全量）vs `_log_row` L1641-1669（仅 `J/_p%/_max_r/objective/_diff/lr/r/delta/_epoch/_c/_img/exp_t...`，**无 `w_*`/`*_term`**）。
- **问题**：DataFrame 第一行有 `ee_term` 等，其余全 NaN。
- **修法**：收敛进 `_update_dynamic_weights` 状态对象统一输出（见 P2-2）。

### P0-4：`optimizer.lr, delta = learning_schedule(radius(init_img, center=center, energy=0.8), ...)`

- **位置（当前）**：L1522（同族：L1990 SPGD 分支 `radius(pos_img, center=center, ...)`）。
- **问题**：`center` 是 `reset_window` 返回的**全帧坐标**，而 `init_img`/`pos_img` 已是窗口局部图。
  与已修的 `r_bucket`（`reference_center`）同类 bug。
- **修法**：改用 `reference_center`（窗口局部坐标）。

### P0-5：饱和判定硬编码 8 bit 且两分支不一致（仍存在）

- **位置（当前）**：L1779（heuristic 分支 `float(np.max(img)) >= 255`）vs L1933（SPGD 分支
  `max_brightness == 255 and exposure_time_ms == 0`）。
- **问题**：都假设 uint8；某些像素格式位深不同会静默漏判。
- **修法**：统一 `np.iinfo(img.dtype).max`，抽 `is_saturated(img) -> bool`。

---

## 二、P1 — 架构重构（解决根本问题）

### P1-1：目标函数 if/elif 链 → Objective 类族（未做）

- **位置（当前）**：L1300-1368（7 种 objective 闭包 `_calc_objective_pib/roi_pib/rms_pib/shape/rmse/radiu/avg`
  + `_raw_calc_objective` 分发） + `calc_objective` 守卫包装 L1483。合并已把闭包提取成独立函数
  名（P1-1 的准备工作），但仍是 if 分发 + `objective_mode` 符号 + `last_terms` nonlocal。
- **修法**：Objective 类族 + GuardedObjective（见下）。

### P1-2：两个搜索分支共享 `BenchSession`（未做）

- **位置（当前）**：SPGD 分支（~L1930-1990）与 heuristic 分支（L1770-1800）重复
  "clip → 相位 → `_display` → sleep → 采图 → 饱和处理 → 算目标"序列；饱和策略不一致（P0-5）。
- **修法**：抽 `BenchSession` context manager（`measure(coeffs)/settle()/apply_best()`）。

### P1-3：巨型函数 → 编排器 + 纯函数段（未做）

- **位置（当前）**：`optimize_slm_zernike_pib`（~L760-1700+，仍 ~900 行）；setup 段反复覆写
  `center` 变量（mass 定位 → clamp → reset_window 全帧坐标 → 当窗口坐标用）仍是坐标系 bug 策源地。
- **修法**：`prepare_geometry` 返回不可变 dataclass（字段区分 `window_center_full_frame` /
  `reference_center_window_local`），编排器 + `_run_spgd`/`_run_heuristic`。

### P1-4：硬件安全 try/finally（未做）

- **位置（当前）**：L1060 `with (create_camera(...) as cam, Santec(...) as slm, display_ctx as live_display):`
- **问题**：任何异常（相机掉线、越界、KeyboardInterrupt）都会让 SLM 停在随机相位。
- **修法**：编排器外层 try/except，异常路径也执行"恢复 flat 或 best"。

---

## 三、P2 — 一致性与清理

| # | 项 | 当前位置 | 状态 |
|---|---|---|---|
| P2-1 | 常量替换字面量（`-5.0/5.0` clip ×6, `1e3` L1505, `1e-4`）| L1505 等 | 未做 → `GUARD_PENALTY=1e3`/`IMPROVE_EPS=1e-4`/复用 `ZERNIKE_CLIP` |
| P2-2 | `_update_dynamic_weights` → `AdaptiveWeights` dataclass | L251-375 | 未做（裸 dict setdefault + 2/3-tuple 联合返回）→ 顺带修 P0-3 |
| P2-3 | `_create_optimizer` inspect.signature 创可贴 | L422-430 | 未做 → 显式 `OptimizerConfig` |
| P2-4a | 死代码 `gauss_center` | L153-203 | **确认零生产调用**（src 内仅定义处；可删） |
| P2-4b | 死代码 `best_j/best_objective_ratio/best_img`、`shape_schedule` | SPGD 分支 | 待核（合并未动） |
| P2-5 | 300px 光阑踩坑文档裸字符串 | L447-460 | 仍错位（挂在 `TARGET_BOX_WAIST_FACTOR` 后）→ 移到 `ZERNIKE_APERTURE_RADIUS`(L433) 上方作注释 |
| P2-6 | 文件尾 argparse `__main__` 块 | L2139-2180 | **仍存在**（已改用 `config=SlmZernikePibConfig` 但双入口继续分叉）→ 删除，统一 main.py click |
| P2-7 | 日志 f-string/占位符混用 | L2173/2179 等 | 仍混用 → 统一花括号占位 |
| P2-8 | `_metric_panel` 每 epoch 全量六套指标 | L1544-… | 未做 → `panel_every_n: int = 1` |
| P2-9 | `_apply_best_on_exit` 往 Recorder 挂属性 | ~L1584 域 | 未做 → 显式 `RawReport` 字段 |
| P2-10 | `SLM_WIDTH/HEIGHT` 与驱动 `Panel_Res` 重复 | L129-131 | 未做 → 读驱动常量 |

---

## 四、P3 — 可测试性与 GS 热启动

### P3-1：sim 台架（未做）
已有 `sim` 相机后端（2f-Fourier），无对称 SimBench。给定相位 → FFT → 平方得 CCD 图，
接口与 BenchSession 一致后，P0 全 bug/自适应权重/learning_schedule/GS 热启动可无硬件回归。

### P3-2：离线 GS 作闭环初值（未做）
- **位置（当前）**：L1058 `# TODO： 先跑一次离线 gs() 把结果作为闭环初值…`
- **修法**：落地为 `--init-gs`：`gs_warm_start(geo, cfg)` = 实测平场振幅 → 目标形状模板振幅 →
  GS 20~50 内循环 → `zernike_lstsq_fit` 投影回 Zernike 基 → 作为 `init_c` 入口。

---

## 五、与入口文件（main.py / slm_pib_runner）的联动

| 入口动作 | 依赖本文件改动 |
|---|---|
| `_optimizer_kwargs` 40 行映射删除 | 已完成（合并 a3130a7 收敛 `SlmZernikePibConfig`；runner 侧 `_optimizer_kwargs` 为兼容保留，可再精简） |
| 共享参数上移 group / ctx.obj | 本文件已接收单个 config |
| 新增搜索算法只需薄命令 | `_run_heuristic` 已由 `run_heuristic_search` 统一驱动 |
| `--config YAML` + default_map | config dataclass 字段与 YAML 一一对应（已具备） |
| `--resume-config` | 编排器返回 recorder/config 可反序列化重建（已具备） |

---

## 六、建议实施顺序（更新）

1. **第 1 步（半天）**：修 P0-2/P0-4/P0-5 三个真 bug + `gauss_center` 删除 + P2-5 注释归位
   （纯行为修正，先让行为正确）。
2. **第 2 步（1-2 天）**：Objective 类族 + GuardedObjective + P0-3（`_row0`/`_log_row` 收敛），
   消灭 if/elif 与符号特判 —— 回归风险最高，靠 sim 台架兜底。
3. **第 3 步（1 天）**：BenchSession + prepare_geometry + 编排器拆分 + P1-4 try/finally。
4. **第 4 步（半天）**：P2-1/2/3/6/7/8/9/10 清理 + argparse 删除。
5. **第 5 步**：P3-1 sim 台架 + P3-2 `--init-gs` 热启动。
6. **独立项**：`docs/TODO.md` 本身是审计产物（untracked），随第 1 步落地后可由
   `scripts/generate_*_report.py` 惯例迁入正式报告（如需）。