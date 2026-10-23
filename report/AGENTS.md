# AGENTS.md — `report/` 目录结构与路径规范

> 本文件约束 `report/` 下的目录布局、文件命名、资产归属与溯源登记。
> 与仓库根 `AGENTS.md` 冲突时，**以本文件为准**（`report/` 专用规则）。

---

## 1. 两类文档的分工（红线）

| 位置 | 内容 | 规则 |
|------|------|------|
| `docs/` | **设备说明文档**：设备规格、SDK/驱动用法、装配/操作手册、图集 | 不放实验结论 |
| `report/` | **实验报告**：一次实验/基准/仿真的**结论**（测量值、参数、口径、被推翻的假设） | 结论的唯一事实源 |

每份实验报告必须能追到生成它的脚本，否则读者无法判断数字从哪来。

---

## 2. 目录布局：一份报告 = 一个子目录

**规范**：报告**不**平铺在 `report/<topic>/` 根下。按**生成脚本 / 实验系列**分子目录，报告与其独占资产同住：

```
report/<topic>/
├── <系列名>/                    # 一个生成脚本 / 一条实验系列 = 一个子目录
│   ├── <报告名>.md              # 报告正文
│   ├── figures/                 # 该报告的图 (PNG/GIF)
│   ├── images/                  # 该报告的插图/动画 (与 figures/ 的区别见 §4)
│   ├── artifacts/               # 上游实验脚本的原始产物 (CSV/JSON) —— 报告的输入契约
│   └── assets/                  # 现场抓取的原始资产 (截图等)
├── README.md                    # 该 topic 的索引 (手工维护)
└── <跨系列汇总>.md              # 跨系列的总结/计划 (手工撰写)
```

### 2.1 已确立的子目录（`report/shape_trials/`）

| 子目录 | 生成脚本 | 报告 |
|--------|----------|------|
| `gs_defocus_sim/` | `scripts/slm/generate_gs_defocus_sim_report.py` | `gs_defocus_sim_20261019.md` |
| `hw_defocus_zernike/` | `scripts/slm/generate_hw_defocus_zernike_report.py` | `hw_defocus_zernike_20261020.md`、`hw_defocus_zernike_20261021.md`、`hw_defocus_zernike_20261022.md` |
| `sim_vs_hw/` | `scripts/slm/compare_sim_vs_hw.py` | `sim_vs_hw_20261022.md` |
| `gs_golden_reference/` | `scripts/slm/generate_gs_golden_reference.py` | `gs_golden_reference.md` |
| `gs_refine_fwdmodel/` | `scripts/slm/generate_gs_refine_fwdmodel_report.py` | `gs_refine_fwdmodel_calibration.md` |
| `gs_refine_zernike/` | `scripts/slm/generate_gs_refine_zernike_report.py` | `gs_refine_zernike_uniformity.md` |
| `model_in_loop_cv/` | `scripts/slm/generate_model_in_loop_cv_report.py` | `model_in_loop_cv.md` |
| `online_recal_efficiency/` | `scripts/slm/generate_online_recal_efficiency_report.py` | `online_recal_efficiency.md` |
| `gui_closed_loop_shaping/` | `scripts/generate_gui_shaping_hw_report.py` | `gui_closed_loop_shaping_20261014.md` |
| `fast_recal_adapter/` | `scripts/ml/exp_fast_recal_adapter_report.py` | `fast_recal_adapter.md` |
| `gs_refine/` | 手工撰写（GS 系列） | `gs_issue_live_20261015.md` 等 9 份 |
| `model_in_loop/` | 手工撰写（MIL 系列） | `model_in_loop_live_20261013.md` 等 9 份 |

同一系列在**同一天**追加轮次时，文件名保留日期并加 `_rN` 轮次后缀（如 `hw_defocus_zernike_20261022_r2.md` 表示 10-22 当天的第二轮）；跨天追加则直接换日期（`hw_defocus_zernike_20261020.md` → `..._20261021.md` → `..._20261022.md`）。生成脚本从系列前缀之后的**整段后缀**推导图名戳（`_20261022`、`_20261022_r2`），所以同系列的多份报告不会互相覆盖图。

**每一张图都要带这个戳** —— 逐组图族（远场拼版 / 光斑截面 / 泽尼克逐阶 / 离焦响应）与聚合图一视同仁，否则同系列的两份报告仍会互相覆盖对方的逐组图。同一系列里每份报告**独占**自己的图；正文已定稿、只想把图换回自己那一轮时，用生成器的 `--figures-only` 重画，不要重新生成正文（重新生成会让正文出现与那一轮事实不符的叙述）。

### 2.2 新增报告时

1. **先定归属**：找到生成它的脚本（`scripts/` 下）。同脚本 → 已有子目录；新脚本 → **新建子目录**，目录名 = 脚本语义名（去掉 `generate_` / `experiment_` 前缀）。
2. 报告放子目录根，资产放子目录的 `figures/` / `images/`。
3. 若是**手工撰写**的报告，归入对应的**实验系列**子目录（按报告名的语义前缀），不要另起新目录。
4. **跨系列的总结/计划**留在 `report/<topic>/` 根（如 `shape_trials/README.md`、`experiment_summary_*.md`）。

---

## 3. 溯源登记（硬性要求）

`scripts/_common/provenance.py::REPORTS` 是**唯一映射源**，键 = **仓库根相对路径**（精确匹配，含子目录）。

新增或移动报告后**必须**同步三处：

| 位置 | 改什么 |
|------|--------|
| `scripts/_common/provenance.py::REPORTS` | 键改成新路径；`script` 填生成脚本，`extra_scripts` 填数据生产脚本 |
| `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN` | 键跟着改（基线是路径精确匹配的） |
| 报告正文 | 跑 `python scripts/sync_report_provenance.py` 重生成溯源块与 `report/README.md` 索引 |

**生成脚本路径常量必须写死新目录**。不要用「写到根再手工搬」的方式——那样脚本重跑一次就会把文件写回旧位置：

```python
REPORT_KEY = "report/shape_trials/<子目录>/<报告名>.md"
OUT_DIR   = ROOT / "report" / "shape_trials" / "<子目录>"
FIG_DIR   = OUT_DIR / "figures"
```

---

## 4. 资产归属

| 规则 | 说明 |
|------|------|
| **独占** | 一张图只属于一份报告。跨报告复用同一张图 → 复制到各自目录，不要建 `shared/` |
| **跟随源图** | 由源图派生的产物（如 X/Y 截面图）必须写回**源图所在目录**，不能落到全局 `figures/` |
| **`artifacts/`** | 上游实验脚本的原始 CSV/JSON —— 是报告的**输入契约**，必须与报告同目录 |
| **`figures/` vs `images/`** | `figures/` = 报告正文引用的图；`images/` = 插图/动画（如 GS 循环 GIF） |

**孤儿资产**（无报告引用）不允许留在 `report/` 根：要么归入对应子目录，要么删除。

---

## 5. 链接规则

- 报告内引用**同目录**资产：`figures/xxx.png`（**不带** `../`）
- 引用**同族**（同一子目录内）另一份报告：`xxx.md`（**不带** `../`）
- 引用**其他子目录**的报告：`../<子目录>/xxx.md`
- `report/<topic>/` 根的 `README.md` 引用子目录报告：`<子目录>/xxx.md`（**不带** `../`）
- 引用 `data/` / `docs/`：按实际深度写 `../../`（子目录内是三层）
- **仓库根相对路径**只出现在散文里（`` `report/shape_trials/...` ``），**不要**放进 Markdown 链接——那样换目录就断

---

## 6. 校验

改完路径必须跑：

```bash
python scripts/sync_report_provenance.py --check     # 溯源块与索引无漂移
python -m pytest tests/ao_shaping/scripts/test_report_provenance.py -q
ruff check .
```

`test_report_provenance.py` 会校验：每份报告有且仅有一个溯源块、注册表脚本路径存在、**报告内相对链接全部可解析**、索引链接可解析、基线断链仍被合理豁免。**链接断了它会指名道姓地报出来。**

---

## 7. 反模式

| 反模式 | 为什么禁止 |
|--------|-----------|
| 把报告平铺在 `report/<topic>/` 根 | 资产无处安放；几十份报告的图混在一个 `figures/` 里，无法判断谁属于谁 |
| 生成脚本写死 `report/<topic>/` 根，重跑就覆盖别人的文件 | 子目录是唯一的隔离边界；写到根等于没有隔离 |
| 报告在子目录，图留在 `figures/` 根 | 链接要么断，要么得靠 `../` 绕回去；派生图（如截面图）会写错目录 |
| 手工编辑溯源块 | 溯源块是**生成**的（`insert_header`），手改会被 sync 覆盖，且下次 `--check` 立刻报漂移 |
| 报告搬目录后不更新 `REPORTS` 键 | 键是**精确路径**匹配 → 报告从索引里消失，且拿不到溯源头 |
| 在 `report/` 根留孤儿 CSV/JSON/PNG | 无人认领的产物 = 没人知道它是什么；要么归位要么删 |
| 为省事用 `shared/` 目录放公共图 | 「公共」意味着两个报告都依赖它，改一处断两处；复制成本远低于耦合成本 |
| 把 `docs/` 的设备说明搬进 `report/` | 两类文档职责不同：设备怎么用 vs 这次测出什么 |
