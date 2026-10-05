# `tools/slm/` 重构 TODO

> 状态：**已执行完毕（2026-10-02）**。下方结论均已复核并订正，勾选框反映实际结果。
> 执行记录见本文末「执行结果」。调研日期：2026-10-01
> 范围：`src/ao_shaping/tools/slm/`（22 个模块 + `cartographer/` 子包，~530 KB）

---

## 一、三个核心结论

- [x] **功能确实重复，但重复集中在「CLI 管道」与「设备会话」两层；测量内核与分析早已收敛。**
      15 个模块手写 5–25 个 `@click.option`（实测合计 **202** 个 flag 声明；原文「6–27」两端都不准），17 个文件各自构造 `Santec`，
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

- [x] **CLI 管道** — **已收敛**：15 个模块全部走 `with_params`（`cc2efa0` 试点 → `79c57bd` 13 个 → `d45f1e0` 补上 `phase_capture`），`slm_zernike_sweep_probe.py` 已由 argparse 迁为 click；`--help` 逐字节不变（黄金 0 键变化）。`calibration.py` 按 §六 保留手写 click。
- [ ] **设备会话** — `Santec(...)` 散落 17 个文件（**18 处调用点**）；相机打开实为
      **3 个助手族的 15 处调用点**（`open_camera` / `create_camera` / 裸 `DahengCamera(`·`MIICamera(`），
      原文「13 种写法」是把调用点数与写法数混淆
- [ ] **曝光默认值** — 实为 **9 种**并存：0.02 / 0.03 / 0.8 / 1.0 / 1.2 / 2.0 / 3.0 / 4.0 / 20
      （另有 `slm_snr_probe.py` 硬编码 1.2），原文「7 种」漏了 0.8 / 1.0 / 20
- [ ] **内存槽轮换** — 3 种写法：驱动自动轮换 / 自建 `SlotRotator` / 手工 `current_slot`
- [ ] **抓帧助手** — 6 个各自实现（见下）
- [ ] **光斑定位** — canonical `zero_order_center` **仅 1 个文件在用**（`slm_snr_probe.py:55,364`），其余裸 `np.argmax`
- [ ] **相位→灰度** — canonical `phase_to_slm_grayscale` 仅 `slm_snr_probe.py:246`；5 处直调 `create_phase_from_array`

---

## 四、顺带查出的 4 个现存缺陷（与重构独立，建议单独 PR）

- [x] **D1 · 内存槽固件 no-op 违规** — **已修**（`eaba803`）
      `cartographer/dynamic_compensation.py:266,272,276,383`：`memory_slot` 默认 **2**、注释写
      "1-128"、循环内反复写同一槽 → 违反 README「铁律三：固定 `memory_number` 是固件 no-op」。
      根因比原文更重：`driver.py:1224-1226` 的 `if memory_number is not None:` 分支完全绕过
      L1228-1246 的轮换与"跳过已显示槽"逻辑，故第 2 轮起**面板永不刷新**，循环仍记录
      `improvement_rms`（伪造数据）。真实槽数是 127（`constants.py:20 MAX_MEM_SLOTS`），
      非注释所写 128。

- [x] **D2 · min-max 归一化反模式** — **已推翻，不修**
      `calibration.py:2182` 的 `P` 是刀口扫描中**一维标量 CCD 光强数组**（`power()` 返回
      `float(p.sum())`），`[0,1]` 归一化是 `np.interp(0.50, Pn, ss)` 求 CDF 分位点的**必要前提**，
      与相位/尺度无关性无关。另外原文点名的两个反模式**早已修复**：`PatternHelper._zernike_to_uint16`
      已删除，`ZernikeDM.generate_phase` 已于 2026-09-16 改为 raw 直通 —— 是 `AGENTS.md`
      的 ANTI-PATTERNS 表把它们记成"仍存在"而过期。

- [x] **D3 · README 文档自相矛盾** — **已修**（`307d90d`）：仅 `README.md:439,449` 两行把
      `--cam-type` 写成 `--camera-type`。原文暗示 `slm-lut` 也有问题，**该处本就是对的**，勿动。

- [x] **D4 · `tools/slm/__init__.py` docstring 漏 6 个模块** — **已修**（`307d90d`），
      并加 `test_package_docstring.py` 防再漂移。（合并上游后又补 `slm_abba_probe` /
      `slm_drift_probe` / `slm_floor_probe` / `slm_bench_metrics` 四个新模块。）

- [x] **D5 · 清理陈旧 `.pyc`** — 5 个里只有 `slm_shift_calib.cpython-313.pyc` 真实存在；
      `_slm_fix_wavelength` / `_slm_health_check` / `_slm_reboot_wavelength` 在本目录
      **从未存在**（`git log --diff-filter=D` 为空），`slm_zernike_report` 的 `.pyc` 已被清理。
      `__pycache__` 是 git-ignored 构建产物，随手删除即可，无需提交。

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
| **R7** | ~~`utils/cli/params.py` 无法成为"廉价叶子"~~ **诊断错误，已订正** | 原文归因于 `utils/__init__.py:17–27` 的 eager（几百 ms 量级）。**真实主因是 import 期网络 I/O**：`drivers/sim/slm_pib_sim.py` 模块级 `from ao_shaping.config import DM_N_ACTUATORS` → `config.__getattr__` → `MicroDM.is_reachable()` 对 192.168.0.101–126 做 **26 次顺序 1s TCP 探测**，且 `_resolve_dm_n_actuators` **无缓存** → `import ao_shaping` 实测 **27–29s**。这违反 `drivers/AGENTS.md`「import 不得加载 SDK / 做 I/O」 | **已修**（`67599bd`）：触发点下沉为函数内 import + 两个解析器加 `lru_cache` + 探测改并发（**`settimeout(1.0)` 保留**，慢但存活的真机仍可探测）。`import ao_shaping` 29.41s → **1.615s**，连接数 26 → **0**。抽取的真正理由仍是**解耦 + 副作用隔离**，不要写成"省导入时间" |

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
- `report/slm/model_in_loop_bench_calibration.md` — 台架常数来源
- `src/ao_shaping/runners/runner_common.py:118–323` — 待抽取的机制本体
- `src/ao_shaping/drivers/slm/AGENTS.md:116` — `calibration.py` 的 shift CLI 用法
---

## 执行结果（2026-10-02）

| 步骤 | 状态 | 证据 |
|---|---|---|
| **Wave 0** import 期网络 I/O | ✅ `67599bd` | `import ao_shaping` 29.41s → **1.615s**；`connect_ex` 26 → **0**；`DM_N_ACTUATORS` 仍 64 |
| **Step 0** 特征测试（安全网） | ✅ `2d5d0a1` | 109 tests；35 命令 `--help` 黄金快照 + 参数契约 + argv 绑定 + 文档串守卫 |
| **Step 1** 抽机制到 `utils/cli/params.py` | ✅ `e5e5b79` | `runner_common.py` 2078 → 1870 行；**15** 个符号逐字节搬迁（原文只列 11，漏 `_TYPE_INFERENCE` / `_is_click_group` / `_ClickOption` / `_ClickGroupSpec`，按名单挑会 `NameError`） |
| **Step 2** 试点 `slm_zernike_sweep_probe` | ✅ `cc2efa0` | argparse → click；`--no-hw` 仍 exit 0；黄金 **+2 键 / 0 键变化** |
| **Step 3** 剩余 13 个 CLI 迁移 | ✅ `79c57bd` | **黄金 0 键变化** —— 逐字保持 `--help`；`slm-diagnose` / `slm-lut` 注册与 `python -m` 入口均不变 |
| **Step 3b** 补上 `phase_capture`（复审 F19） | ✅ `d45f1e0` | `79c57bd` 漏迁最后 1 个（24 个手写 option + 4 个死 import）。补迁后**黄金 0 键变化**；连带修 R3 守卫自身的缺陷：`_gates_device_open` / `_diverges_before_device_open` 只收 `ast.Name`，迁移后 body 读 `params.<field>`（`ast.Attribute`）→ 结构性豁免集体失效，改为 `_tested_names()` 同收两种 |
| **Step 4** D1 / D3 / D4 | ✅ `eaba803` `307d90d` | D1 RED `[2,2,2] != [None,None,None]` → GREEN；D2 推翻 |
| **合并上游** | ✅ `6f544a2` | 上游 15 提交（ABBA / PBR / 新探针）合入；仅 2 个黄金键因上游新增 `--w-pbr` 等而重基线 |

**经验（留给下一次重构）**

1. **`with_params` 保持 `--help` 顺序**：它按**声明的逆序**施加 option，使 click 累积的
   `__click_params__` 输出为**声明顺序**；因此按原装饰器栈的上下顺序声明字段，`--help`
   逐字节不变。这是"行为中性"能被证明的关键，也是 §五 Step 0 快照存在的意义。
2. **先建安全网再动手**。本仓原先 0 个测试断言任何 CLI 选项面，`with_params` 静默丢字段
   不会被发现。`_collect_click_annotations._add` 只在**单个类树内**去重并 `raise`，所以
   两个 `@with_params` 叠加时**外层**字段仍会静默退回默认值 —— 原文 R3 结论正确，
   但"静默"仅限跨装饰器场景。
3. **R4 成立且更脆**：4 个 union 字段各靠**不同**逃生口（2×`type=`、2×`callback=`），
   全部在**导入期**炸。搬迁时整块 118–323 逐字搬，不可按名单挑。
4. **`DM_TYPES` 依赖导入顺序**：6 → 7（`runner_common` 导入 `asyn_micro_dm` 的注册副作用）。
   断言**子集**而非个数，否则测试本身变成顺序相关。
5. **并发 agent 不要共写同一文件**：一次"黄金测试 70 失败"并非 flaky，而是另一个 agent
   正在对 `runner_common.py` 做证伪实验。证伪必须串行，或在隔离副本里做。
