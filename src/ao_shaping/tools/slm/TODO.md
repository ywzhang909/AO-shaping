# `tools/slm/` 重构 TODO

> 状态：**方案已定，尚未动手**（本文件不改动任何代码）
> 调研日期：2026-10-01
> 范围：`src/ao_shaping/tools/slm/`（22 个模块 + `cartographer/` 子包，~530 KB）

---

## 一、三个核心结论

- [x] **功能确实重复，但重复集中在「CLI 管道」与「设备会话」两层；测量内核与分析早已收敛。**
      15 个模块手写 6–27 个 `@click.option`（约 200 个 flag 声明），17 个文件各自构造 `Santec`，
      13 种相机打开写法。而 `slm_bench_probe` / `slm_scan_analysis` / `SlotRotator`（已下沉进驱动）/
      `open_camera` 说明前几轮重构已把这三层收完。

- [x] **可以合并，但只合并 CLI 层，不合并命令体。**
      6 个探针共享的只是"打开 SLM + 打开相机 + 读一帧"，物理实验各不相同。
      合并命令体会挪动 settle/slot/dark-frame 这些**位置敏感**步骤 → 产生看似可信的错数据。

- [x] **可以复用 `runners/runner_common.py`，但只能复用「机制」，不能直接 import 那个模块。**
      机制段 `runner_common.py:118–323` 只依赖 `functools/dataclasses/pathlib/types/typing/click`，
      **零 `ao_shaping` 导入**，是干净叶子。但 `88–109` 污染（`list_dm_types()` + `PANEL_RES` +
      `asyn_micro` 注册副作用），且 `runners/slm/zernike_matrix_runner.py:95,111` **已反向依赖
      `tools.slm`** —— 直接 import 即成环。

---

## 二、已被前几轮重构解决（复用，勿重做）

- [x] `slm_bench_probe.py` — 纯测量内核，**设备由参数注入** → 可脱机单测
- [x] `slm_scan_analysis.py` — 纯 numpy 分析助手
- [x] `SlotRotator` / `SLOT_MIN` / `SLOT_MAX` — 已下沉进 Santec 驱动（原 `utils/slm_slot.py` 已删）
- [x] `open_camera` — `utils/image/hardware_utils.py:644`（`utils/hardware_utils.py` 只是 512 B别名 shim）

---

## 三、仍然重复（本次重构对象）

- [ ] **CLI 管道** — 15 个模块手写 click；1 个用 argparse（`slm_zernike_sweep_probe.py`）；**0个**用 `with_params`
- [ ] **设备会话** — `Santec(...)` 散落 17 个文件；相机 13 种打开写法
- [ ] **曝光默认值** — 0.02 / 0.03 / 1.1 / 1.2 / 2.0 / 3.0 / 4.0 ms 共 7 种并存
- [ ] **内存槽轮换** — 3 种写法：驱动自动轮换 / 自建 `SlotRotator` / 手工 `current_slot`
- [ ] **抓帧助手** — 6 个各自实现（见下）
- [ ] **光斑定位** — canonical `zero_order_center` **仅 1 个文件在用**（`slm_snr_probe.py:55,364`），其余裸 `np.argmax`
- [ ] **相位→灰度** — canonical `phase_to_slm_grayscale` 仅 `slm_snr_probe.py:246`；5 处直调 `create_phase_from_array`

---

## 四、顺带查出的 4 个现存缺陷（与重构独立，建议单独 PR）

- [ ] **D1 · 内存槽固件 no-op 违规**
      `cartographer/dynamic_compensation.py:266,272,276,383`
      `memory_slot` 默认 **2**、注释写 "1-128"、反复写同一槽
      → 违反 README「铁律三：固定 `memory_number` 是固件 no-op」

- [ ] **D2 · min-max 归一化反模式**
      `calibration.py:2182` — `(P-P.min())/(np.ptp(P)+1e-12)`
      → 尺度无关（系数 ×1 与 ×4 输出字节相同），幅度不可控

- [ ] **D3 · README 文档自相矛盾**
      `slm-diagnose` 选项表写 `--cam-type`，同页示例写 `--camera-type`，`slm-lut` 写 `--camera-type`
      → 实测代码是 `--camera-type`，选项表是错的

- [ ] **D4 · `tools/slm/__init__.py` docstring 漏 6 个模块**
      未描述：`slm_zernike_response` / `slm_zernike_correction` / `slm_zernike_common` /
      `slm_wfs_probe` / `slm_wfs_reference` / `delta_explorer`

- [ ] **D5 · 清理陈旧 `.pyc`**（可选）
      `__pycache__` 存有 5 个已删模块的 `.pyc`：`_slm_fix_wavelength` / `_slm_health_check` /
      `_slm_reboot_wavelength` / `slm_shift_calib` / `slm_zernike_report`

---

## 五、分阶段步骤

> 每步独立可交付、不断破坏。硬件测试受 `AO_RUN_HARDWARE` 门控（默认 skip）。

### Step 0 · 特征测试（先行，不可跳过）

- [ ] 对每个命令对象冻结 ① `--help` 全文
- [ ] 冻结 ② 固定**非默认** argv 下组装出的 dataclass repr
- [ ] 冻结 ③ 8 个已注册命令的 `--dm_type` 选项列表
- [ ] 断言"固定 argv 下**没有任何字段等于其默认值**"（见风险 R3）
- [ ] 钉住 `pupil_center` 的「二元联合 + callback」耦合（见风险 R4）

**验证**：`pytest` 全绿。**这是后面所有步骤的安全网。** — 0.5–1 d

### Step 1 · 抽机制到新叶子模块

- [ ] 新建 `src/ao_shaping/utils/cli/params.py` + `utils/cli/__init__.py`
- [ ] 搬 `runner_common.py:118–323`：`option` 再导出、`_DelayedCall` / `_DelayedFunction`、
      `ClickGroup`、`_collect_click_annotations`、`_patch_click_types` / `_patch_defaults` /
      `_patch_names` / `_strip_optional` / `_copy_delayed_call`、`with_params`
- [ ] `runner_common.py` 改为从叶子导入（保留 runner 专属 param dataclasses）
- [ ] **禁止**从 `utils/__init__.py` 再导出（它是 eager 的，见 §七）
- [ ] 写死不变式进 docstring：**本模块禁止任何 `ao_shaping.*` 导入**

**验证**：Step 0 快照**逐字节 diff 为空** — 0.5 d

### Step 2 · `tools/slm/params.py` + 一个试点

- [ ] 新建 `tools/slm/params.py`：`SlmBenchParams`
      （`cam_type` / `cam_id` / `exposure_ms` / `slm_number` / `slm_wavelength` / `settle` / `frames` / `seed`）
      + 各探针专属 dataclass
- [ ] 试点迁移 `slm_zernike_sweep_probe.py`（唯一有 `--no-hw`、能脱机跑完整路径的探针）
- [ ] 原模块路径变 3 行 shim，保持 `python -m` 可用

**验证**：`--help` 与 argv→dataclass 快照一致 — 0.5–1 d

### Step 3 · 逐族迁移剩余探针

- [ ] `diagnose` 族：`slm_diagnose` / `slm_snr_probe` / `slm_exposure_check`
- [ ] `probe` 族：`slm_tilt_probe` / `slm_panel_locate` / `slm_beam_extent` /
      `slm_phase_resolution` / `slm_zernike_sweep_probe`
- [ ] `cal` 族：`slm_lut_runner` / `calibration` / `cartographer`
- [ ] 每族一个 commit，每个 commit 结束时旧模块都是 shim

**验证**：每族快照一致 + 现有 pytest 仍绿 — 1–2 d

### Step 4 · 修现存缺陷（独立 PR，排在 Step 0–3 之后）

- [ ] D1 / D2 / D3 / D4（+ 可选 D5）
- [ ] 审计 `slm_zernike_common.py` 是否仍自实现 Zernike 数学 →
      若仍自实现则**删除内部实现改用 canonical**，而非搬文件

**验证**：硬件验证 — 0.5–1 d

### Step 5 · flag 拼写收敛（可选，后置）

- [ ] `--cam-type` / `--camera-type` 二选一
- [ ] 全仓 grep + README 同步

**验证**：`pytest` 全绿 — 0.5 d

---

## 六、明确**不做**的事

- [ ] ❌ **不**抽 `with_slm_bench()` 上下文管理器统一设备会话 → 直接踩台架铁律一/二/三
- [ ] ❌ **不**把 22 个脚本合成一个 `slm` 大组 22 个子命令 → `--help` 不可用，
      且 `main.py` 会从 2 个变成 22 个工具导入
- [ ] ❌ **不**搬 `calibration.py`（109 KB 硬件编排）
- [ ] ❌ **不**在 `tools/slm/` 里复制第二套 click 机制
- [ ] ❌ **不**让 `tools/slm` 直接 `import runners.runner_common`

**保留**：`main.py` 现有 `slm-diagnose` / `slm-lut` 作为**别名**（click 允许同一 Command 对象挂两个名字）；
所有 `python -m ao_shaping.tools.slm.<name>` 入口保持可用。

---

## 七、风险登记册

| ID | 风险 | 后果 | 缓解 |
|---|---|---|---|
| **R1** | settle / slot / dark-frame 步骤**位置**被挪动 | 数据"看起来对"但斜率读错 **3.3×**（同斜坡首读 FWHM 43.2px、3 s 后 12.8px、质心移 62px） | 本次**只动 CLI 层**，命令体逐字不动|
| **R2** | `runners/__init__.py` 改 lazy 会改变 `--help` | `runner_common.py:95-109` 记录 `DM_TYPES` 依赖导入顺序；改 lazy 后 `dm_matrix_runner` 可能在 `asyn_micro` 注册之后才导入 → `--dm_type` **5 项变 6 项** | **本次不碰 lazy**；若要修须单独立项 + 加 `DM_TYPES` 快照测试 |
| **R3** | 跨 `with_params` 的字段名**静默**冲突 | `_collect_click_annotations` 只在单个类树内去重；两个 `@with_params` 各自收集，内层 wrapper 先 pop 同名字段，**外层类静默退回自己默认值**，而 `--help` 完全不变 | Step 0 断言"固定 argv 下无字段等于默认值" |
| **R4** | `pupil_center` 的「二元联合 + callback」耦合被"清理" | `runner_common.py:394-402` 靠 `_patch_click_types:173-180` 在 `callback` 存在时提前 return，绕过会对二元联合 `raise` 的 `_strip_optional:156-164`。注解改成裸 `str` 且删掉 callback → **导入即炸** | Step 1 钉住该耦合的测试 |
| **R5** | argparse → click 改退出码 / stdout | `scripts/generate_*` 若解析探针输出或依赖产物路径会断 | 先 grep `scripts/` 是否消费其 stdout / 产物路径 |
| **R6** | 新叶子模块反向 `import ao_shaping` | 结构性成环 | docstring 写死不变式 + AST/import 扫描测试 |
| **R7** | `utils/cli/params.py` 无法成为"廉价叶子" | `utils/__init__.py:17–27` 是 eager 的（已拉入 display/pygame、slm_lut、zernike_calc），任何 `utils.*` 导入都会付这个代价 | **接受**。抽取的真正理由是**解耦 + 副作用隔离**，不是省导入时间 —— PR 描述里不要写错理由 |

---

## 八、工作量汇总

| 范围 | 工作量 |
|---|---|
| Step 0–3（CLI 层完整重构 + 特征测试） | **3–5 天** |
| Step 4（4 个现存缺陷，独立 PR，需硬件） | 0.5–1 天 |
| Step 5（flag 拼写收敛 + README 修文档 bug） | 0.5 天 |

---

## 九、待决策

- [ ] **Q1** — §四 的 4 个现存缺陷是否纳入本次？
      建议：**独立 PR，排在 Step 0–3 之后**，让 CLI 重构可证明"行为中性"。
- [ ] **Q2** — 是否接受"不合并命令体"？
      若有具体想合并的一对（`slm_panel_locate` + `slm_beam_extent` 最接近），
      可在 Step 3 后加一个受特征测试门控的 Step。
- [ ] **Q3** — flag 拼写是否在本次收敛？
      建议**保留现有拼写**不破坏习惯用法，仅修 README 文档 bug（D3）；拼写收敛另立 Step 5。

---

## 十、参考资料

- `README.md` §「SLM 台架探针」— 四条台架铁律 + 实测台架几何
- `docs/slm/model_in_loop_bench_calibration.md` — 台架常数来源
- `src/ao_shaping/runners/runner_common.py:118–323` — 待抽取的机制本体
- `src/ao_shaping/drivers/slm/AGENTS.md:116` — `calibration.py` 的 shift CLI 用法