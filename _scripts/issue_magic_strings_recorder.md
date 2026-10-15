# 魔法字符串审计：`Recorder` 记录字典键 + objective 词表 — 重构方案（跨文件，未改代码）

> 2026-10-15 全仓静态扫描（AST + grep，`src/` + `scripts/`），聚焦 `utils/io/file.py` 的 `Recorder` 类及其**字典**魔法字符串。
> 按约定：**args 解析保留字符串**；能用 enum/dataclass 的提出方案；**跨文件 → 本 issue**；**仅影响单文件的已直接修掉**（见文末）。
> 本 issue 只列**方案 + 影响范围**，不改代码。

## 0. 已直接修复（单文件、行为保持、已验证）— 无需在本 issue 处理

**`Recorder.mode` 魔法字符串 `"max"`/`"min"` → `RecordMode(str, Enum)`**
- 已落地：`file.py` 新增 `RecordMode(str, Enum)`（`MAX`/`MIN`）+ `RecordMode.coerce()`；`__init__` 用 `coerce` 归一；`get_best_iter` 改为 `self.mode == RecordMode.MAX`。
- 关键设计点（已验证行为 100% 等价）：
  - `str`-mixin → 所有 ~25 个外部调用点 `Recorder(..., mode="max")` **一字不改**仍工作。
  - 原逻辑是 `self.mode == "max"`（**只有字面量 `"max"`** 走 `argmax`，`"min"`/拼写错/`None` 全走 `argmin`）→ `coerce` 把 `None`/未知值映射到 **`MIN`（不是 MAX）** 以精确复刻。
  - 用 `==` 而非 `is`：`test_recorder_postprocess.py` 存在**构造后** `rec.mode = "max"`（裸字符串）的用法，`is` 会失败、`==` 因 str-mixin 双向相等而保持原语义。
- 验证：`test_files`/`test_recorder_postprocess`/`test_io_subpackage`/`test_import_purity`/`test_ga_zernike`/`test_rms_by_zernike`/`test_slm_train_data_collect` 全过（100 例）。
- `RecordMode` 已按 PEP 562 惰性加载加入 `ao_shaping.utils` 的 `_LAZY` 映射 + `__all__`（与 `Recorder` 一致），可直接 `from ao_shaping.utils import RecordMode`。

> 下面第 1 项是 `mode` 的同源问题（objective 选择方向），但**分散在多个优化器文件**，故入 issue。

## 1. 【跨文件】objective 选择方向 `"max"`/`"min"` 字符串散落在 5+ 优化器

`mode` 本身已枚举化，但决定 `objective_mode` 的**上游词表**仍是裸字符串 + 内联 `==`，且每个优化器各自判断：

| 文件 | 行 | 现状 |
|---|---|---|
| `optimizer/wfless/pib.py` | 385 | `objective_mode = "max" if objective in ("pib","avg_radiu") else "min"`；`Recorder(mark=objective, mode=objective_mode)` |
| `optimizer/wfless/slm_zernike_shaping.py` | 470,868,1024,1077,1156,1159,1227 | 同义 `== "max"` 共 6 处 |
| `optimizer/wfless/slm_zernike_pib.py` | 502,934 | 同义 `== "max"` 共 2 处 |
| `optimizer/wfless/slm_square_shaping.py` | 1194 | `Recorder(mark=_score_column, mode="max")` 硬编码 |
| `optimizer/wf/rms.py`/`rms_by_zernike.py`/`ga_zernike.py`/`greedy_zernike.py` | 各 | `mode="min"` 硬编码 |
| `scripts/slm/generate_slm_pib_sim_report.py` | 448 | `return "max" if objective in MAXIMIZE_OBJECTIVES else "min"`（**重复维护**同一词表） |

**方案**：把"objective → 方向"做成单一真源，供所有优化器与报告复用：

```python
# 建议放 utils/image/target/objective.py（已存在，见下方 3）或新 objective_mode 模块
@functools.lru_cache(maxsize=None)
def objective_direction(objective: str) -> RecordMode:
    return RecordMode.MAX if objective in MAXIMIZE_OBJECTIVES else RecordMode.MIN
```

- 各优化器把 `"max" if ... else "min"` 与 `if objective_mode == "max"` 统一替换为 `objective_direction(objective)` + `RecordMode` 成员判断。
- `MAXIMIZE_OBJECTIVES` 元组提升为模块级常量（目前 `pib.py` 内联 `("pib","avg_radiu")`，报告里又写一份 → 单一来源）。

**影响范围**：`src/ao_shaping/optimizer/wfless/{pib,slm_zernike_shaping,slm_zernike_pib,slm_square_shaping}.py` + `src/ao_shaping/optimizer/wf/{rms,rms_by_zernike,ga_zernike,greedy_zernike}.py` + `scripts/slm/generate_slm_pib_sim_report.py`（约 6 文件）。
**风险**：低-中（纯字符串→枚举替换，无签名变化）；**但必须**跑全部 wfless + wf 优化器回归。
**为何不入"直接修"**：跨 6 文件 + 触及核心优化循环的 `== "max"` 分支，非单文件。

## 2. 【跨文件】`Recorder` 记录字典键 `_id/_img/_c/_v/_epoch` — 真正的"字典魔法字符串"

`Recorder` 的 `history` 是 `list[dict]`，各写入方用**下划线前缀字符串**作键，是本次"字典魔法字符串"的核心。全仓 **275 处**写入/读取，散布 **50+ 文件**：

| 键 | 语义 | file.py 内读取点 |
|---|---|---|
| `_id` | 记录序号 | L417 `int(row.get('_id', idx))`、L438-439 自动补 |
| `_img` | 2D 图像帧 | L705 `data[e]["_img"]`、L856 默认 `img_keys=("_img",)` |
| `_c` | 1D 系数向量（Zernike） | L695-696、L702 |
| `_v` | 1D 系数向量（freeform） | 文档 L773 |
| `_epoch` | epoch 序号 | L927、L940 `data[int(rec["_epoch"])]` |

**现状问题**：
- 键名靠**约定**（下划线前缀 = 元数据），无类型约束，拼错（`_eoch`）静默丢失。
- `save_recorder_debug_artifacts` 用 `img_keys=("_img",)`、`d1_keys`/`d2_keys` 等元组参数**重新用字符串**枚举一遍，与写入方再次靠约定对齐。
- `_DATA_MODE_OBJECTIVE_KEYS`（file.py L610-620）已是**字符串元组常量**，是正确方向，但 objective 词表本身仍散落（见 1）。

**方案（dataclass / 常量，不改落盘格式）**：

```python
# 建议 utils/io/record_keys.py（或并入 file.py）
class RecorderKeys:
    """Recorder 记录字典的元数据键（下划线前缀约定，单一真源）。"""
    ID = "_id"; EPOCH = "_epoch"; IMG = "_img"; COEF = "_c"; VEC = "_v"
```

- 写入方 `rec.append({...})` 改 `rec.append({RecorderKeys.ID: ..., RecorderKeys.EPOCH: ...})`；`file.py` 内 `"_img"`/`"_c"`/`"_epoch"`/`"_id"` 同步替换。
- `save_recorder_debug_artifacts` 默认 `img_keys=(RecorderKeys.IMG,)`。
- **关键约束**：只替换**引用**，**落盘 CSV/JSON/pkl 里的字符串保持 `_id`/`_img` 不变**（str-mixin 或常量值不变），历史产物可继续读取 → 纯机械替换、零格式破坏。

**影响范围**：50+ 文件（`tools/ccd/*`、`tools/slm/*`、`optimizer/wf/*`、`optimizer/wfless/*`、`scripts/*` 全部 `Recorder.append({...})` 调用点）。
**风险**：低（机械替换 + 字符串值不变），但**体量大** → 必须分批 + 全量回归。
**为何入 issue**：50+ 文件，远超"单文件"边界；且需保证落盘格式稳定（有 `test_data_mode_keys_are_subset` 守护，可复用）。
**建议节奏**：先用 `RecorderKeys` 收口 `file.py` 内部 5 处 + `save_recorder_debug_artifacts`（单文件），再批量替换调用方（每 5-10 文件一批，每批跑回归）。

## 3. 【已存在，作为 1/2 的落点】objective 词表常量

`utils/image/target/objective.py` 已有 `SHAPING_OBJECTIVE_CHOICES`；`file.py` 已刻意**独立**声明 `_DATA_MODE_OBJECTIVE_KEYS` 并注释"落盘契约 vs CLI 词表分离"（L600-620）+ `test_data_mode_keys_are_subset` 守护子集关系。**保留该设计**（CLI 词表 vs 落盘契约分离是对的）。
本 issue 的 1（方向词表）与 2（字典键）落地时**复用**该文件作为 objective 相关常量的宿主，避免再造第三份词表。

## 4. 明确"保留字符串、不动"的（符合 args 解析豁免）

- **argparse CLI 选择**：`--objective`/`--mode` 的 `choices=[...]`、`scripts/slm/*.py` 里 `parser.add_argument("--objective", choices=...)` → **args 解析保留字符串**（用户明确豁免）。
- **`Recorder.mark`**：自由标签（`"J"`/`"rms"`/`"ga_zernike"`/`"peak"`…），**非固定词表**，保留 `str`（不是该 enum 化的魔法字符串）。
- **设备类型字符串** `device_type="slm"/"dm"/"ccd"`（`DeviceConfigManager`）：是**目录名/文件名**，直接进路径，保留字符串（改 enum 会动落盘路径，风险>收益）。
- **时间戳格式串** `TimestampParser.PATTERNS` 的 `fmt`（`%Y%m%d_%H%M%S` 等）：是 **strptime 格式**，保留字符串。

## 5. 建议执行顺序（如需推进）

1. ✅ `Recorder.mode` → `RecordMode`（已完成，本 issue 第 0 节）。
2. `objective_direction()` 单一真源 + 替换 6 文件的内联 `== "max"`（第 1 节，中优先级，需 wfless 全量回归）。
3. `RecorderKeys` 收口 `file.py` 内部 + `save_recorder_debug_artifacts`（第 2 节单文件部分，低优先级）。
4. `RecorderKeys` 批量替换 50+ 调用方（第 2 节，按 5-10 文件/批，每批回归）。

> 2、3、4 均为**行为保持**（字符串值不变 / 枚举 str-mixin），无落盘格式破坏；主要成本是回归广度。
