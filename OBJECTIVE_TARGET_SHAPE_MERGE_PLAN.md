# `objective` / `target_shape` 合并方案

> **状态**: 方案设计稿，**尚未实施**。本文档只描述「要改成什么样、怎么改、怎么验证」，不含任何已落地的代码改动。
>
> **目标**: 把 `slm-pib` 体系里并列的 `objective` 与 `target_shape` 两个配置字段合并为**一个嵌套的 objective 结构**，使 `target_shape` 成为 `objective` 的下级字段，并让其他整形算法（方形整形、benchmark 等）能够复用同一套 schema。
>
> **分支**: `banckend`　**基线 commit**: `ecae143`
>
> **重要前置**: 工作区当前存在**另一条无关的未提交改动流**（移除 `SlmSquareParams` / `optimize_slm_square` / `slm_gsnet_runner` / `slm_square_runner` 的 `cam_type` 选项 + import 排序 + `objective.py` 纯格式化）。本方案**不得触碰或回滚**这些改动。

---

## 1. Problem Statement

### 1.1 症状

`objective` 与 `target_shape` 在 `ObjectiveParamsPib`（`src/ao_shaping/runners/runner_common.py:454-537`）中是两个**平级**字段，但它们在语义上不是平级的：

- `objective` 决定**优化什么**（桶内功率 / 半径 / RMSE / 形状 / ROI-PIB / …）
- `target_shape` 只在**部分** objective 下有意义，且角色还会随 objective 变化：

| objective | `target_shape` 的语义 | 未提供时 |
|---|---|---|
| `pib` | 提供后**自动提升**为 `shape` | 非法组合被拒绝 |
| `radiu` / `avg_radiu` | 完全不适用，传入即报错 | — |
| `shape` | 定义目标形状 | 默认 `"rectangle"` |
| `roi_pib` / `rms_pib` / `rmse` | **仅用于选择 ROI**，不提升 objective | 默认 `"rectangle"` |

即：`target_shape` 是 `objective` 的**从属参数**，却和 `objective` 并列声明、并被独立校验。这导致「哪些 objective 接受哪种 shape」这条规则无法表达，只能散落在调用点。

### 1.2 具体证据

**证据 1 — 提升规则被复制了 3 份**

`pib` + `target_shape` → `shape` 的重映射规则同时存在于：

1. `src/ao_shaping/optimizer/wfless/slm_zernike_pib.py:558-595`（38 行内联 if-else 链）
2. `src/ao_shaping/runners/slm_pib_runner.py:100-109` `_effective_objective_key()`（为了挑对 Recorder 列名而**再次实现一遍**）
3. `src/ao_shaping/optimizer/wfless/slm_zernike_shaping.py` 新增的 `ObjectiveSpec.resolve()`（未提交，抽取自第 1 份）

**证据 2 — objective 取值列表被复制了 5 份**

| # | 位置 | 形式 | 内容 |
|---|---|---|---|
| 1 | `runners/runner_common.py:462` | 手写 `click.Choice([...])` | 7 值 |
| 2 | `optimizer/wfless/slm_zernike_shaping.py` `OBJECTIVE_CHOICES` | tuple | 7 值 |
| 3 | `utils/image/target/objective.py:268` `SHAPING_OBJECTIVE_CHOICES` | tuple | 7 值 |
| 4 | `utils/io/file.py:537` `_DATA_MODE_OBJECTIVE_KEYS` | tuple | **保持独立**（当前 8 值：与 data-mode 落盘契约一致，**不与用户可选集合并**），仅加子集断言防漂移 |
| 5 | `scripts/generate_slm_pib_heuristic_hw_report.py:566` | argparse `choices` | **5 值**（缺 `rmse`、`rms_pib`） |

现有测试（`test_slm_zernike_rmse.py:171-178`、`test_slm_zernike_rms_pib.py:374-378`）只对 `--help` 文本做**正则匹配检查「某个值在不在」**，从不校验**集合相等**，因此 #1/#2/#3 之间已经可以互相漂移而不被发现。

**证据 3 — 公共配置层仍是平级的**

共享运行时层 `ShapingObjectiveParams`（`utils/image/target/objective.py:206-246`）**早已是嵌套布局**（`objective: str` + `shape: TargetShape`，L230/L232），两个优化器也都已经以 `ShapingObjectiveParams(objective=..., shape=shape_for_metric, ...)` 喂给它。也就是说：**运行时层是嵌套的，配置层是平级的**，中间这段落差就是本次要消除的东西。

**证据 4 — 校验顺序被测试锁定，不能随意调整**

`tests/ao_shaping/optimizer/wfless/test_slm_zernike_shaping_objective.py:79-83` 明确断言「兼容性检查先于 objective 成员检查」。`radiu + circle` 必须报 `target_shape can only be used with...`，而不是 `objective must be one of...`。

---

## 2. Current Architecture

### 2.1 数据流

```
CLI  --objective  /  --target_shape
  │
  ▼
ObjectiveParamsPib                        runner_common.py:454-537
  ├─ name: str                (L458-465)  ← 平级字段 ①
  ├─ target_shape: str|None   (L483-490)  ← 平级字段 ②
  └─ +17 个几何/权重兄弟字段
  │
  ▼  CameraParamsPib(CameraParams, ObjectiveParamsPib)   runner_common.py:971-1005
     （继承两者，因此 camera.name / camera.target_shape 是唯二访问路径）
     │
     ├──────────────────────────────┬─────────────────────────────────┐
     ▼                              ▼                                 ▼
optimize_slm_zernike_pib()     optimize_slm_zernike_shaping()    slm_pib_runner._execute()
 slm_zernike_pib.py:526-527     shaping.py:590-591 (未提交)       slm_pib_runner.py:346
 L558-595 38 行内联 if-else     ObjectiveSpec.resolve()            _effective_objective_key(
   提升 / 默认 / 校验              （已抽取，未提交）                  camera.name, camera.target_shape)
     │                              │                                 │  ← 第 3 份规则副本
     │  Recorder(mark=objective)    │  Recorder                       │  选 best 行
     │  L643                        │                                 │
     └──────────────┬───────────────┴─────────────────────────────────┘
                    ▼
        ShapingObjectiveParams(objective=..., shape=shape_for_metric, ...)
                     utils/image/target/objective.py:206-246
                     ▲ 运行时层已经是嵌套的
```

### 2.2 关键文件与行号

| 文件 | 行号 | 内容 |
|---|---|---|
| `runners/runner_common.py` | 199-211 | `_collect_click_annotations()` — **只遍历顶层 `fields(cls)`** |
| `runners/runner_common.py` | 214-246 | `with_params()` — L237-241 也只按 `fields(arg_class)` 回填 |
| `runners/runner_common.py` | 454-537 | `ObjectiveParamsPib`，两个平级字段所在 |
| `runners/runner_common.py` | 540-568 | `ObjectiveParamsSquare`（`slm-gsnet` 用，`target_side: int`） |
| `runners/runner_common.py` | 971-1005 | `CameraParamsPib` 融合体 |
| `runners/runner_common.py` | 1028-1043 | `config_payload()` 序列化 JSON 侧车（保留 `name` 作为身份字段） |
| `runners/runner_common.py` | 1190-1199 | `PibRunnerParams.objective` — DM 版 `pib` 命令的**另一套** 3 值 `--objective` |
| `runners/slm_pib_runner.py` | 100-109 | `_effective_objective_key()` — 第 3 份提升规则 |
| `runners/slm_pib_runner.py` | 346 | 调用点 |
| `optimizer/wfless/slm_zernike_pib.py` | 558-595 | 38 行内联解析链（生产路径，未重构） |
| `optimizer/wfless/slm_zernike_shaping.py` | 502-548 | `ObjectiveSpec`（未提交，仅存在于变体文件） |
| `utils/image/target/objective.py` | 206-246 | `ShapingObjectiveParams`（已嵌套） |
| `utils/image/target/objective.py` | 268-276 | `SHAPING_OBJECTIVE_CHOICES` |
| `utils/image/target/metrics.py` | 16-25 | `TARGET_SHAPE_CHOICES`（8 值，**权威源**） |
| `utils/image/target/metrics.py` | 28-37 | `TargetShape` Literal（同样 8 值，**顺序不同**） |

### 2.3 兄弟优化器的 schema 并不一致

「方便其他整形算法使用」是本次的明确目标，因此必须先认清现状**不统一**：

| 优化器 | objective 字段 | 形状字段 | 形状词表 |
|---|---|---|---|
| `slm_zernike_pib` / `_shaping` | `camera.name`（7 值） | `camera.target_shape` | 完整 8 值 `TARGET_SHAPE_CHOICES` |
| `slm_gsnet`（`ObjectiveParamsSquare`） | `name = "square"`（**无 CLI**） | `target_side: int`（像素） | 仅方形 |
| `slm_square_shaping`（`spgd-square`） | 无（硬编码 `square_quality_score` L426） | `target_side: int` + `target_mean_brightness` | 仅方形 |
| `algorithm/signal_processing/beam_shaping_benchmark.py` | 无 | `shape: str` 位置参数 | **3 值子集** `{square, circle, gaussian}`（L61） |
| `algorithm/signal_processing/differentiable_shaping.py` | 无 | 无（直接传入目标数组） | — |
| `optimizer/wfless/pib.py`（DM 版 `pib`） | `objective: str` 3 值，无 shape | 无 | — |
| `gui/ccd/target_shape_helper.py` | 无 | `shape_type` + `shape_params` dict（中文标签） | 第三种表示法 |
| `utils/image/resample.py:57-85` | — | `target_shape: tuple[int,int]` | **假阳性**（重采样输出尺寸） |

**结论**：新的嵌套 schema 必须是现有 schema 的**超集 + 可裁剪**，不能假设所有优化器都用同一套形状词表。

### 2.4 `target_shape` 的 4 种同名异义（防误伤）

| 含义 | 位置 | 是否在本次范围 |
|---|---|---|
| objective 的形状字段 | `runners/runner_common.py:483` | ✅ 是 |
| 显示叠加层 dict key | `display/windows.py:254-341`、`display/frames.py:109-152` | ❌ 否 |
| 局部变量，装 `(h, w)` 网格元组 | `algorithm/signal_processing/differentiable_shaping.py:479-487` | ❌ 否（假阳性） |
| 重采样输出尺寸参数 | `utils/image/resample.py:57-85` | ❌ 否（假阳性） |

全局改名前必须确认只命中第 1 种。

---

## 3. Target Schema

### 3.1 采用的设计（方案推荐，待确认）

**双类设计**，刻意区分「用户输入」与「解析结果」：

```python
# src/ao_shaping/utils/image/target/objective.py  （新增，权威解析层）

class ObjectiveTarget:
    """用户输入的 objective 配置：可编辑、携带 Click 元数据。"""

    name: Annotated[
        str,
        option(
            "--objective",
            type=click.Choice(list(SHAPING_OBJECTIVE_CHOICES)),
            help="目标函数族。",
        ),
    ] = "pib"

    target_shape: Annotated[
        str | None,
        option(
            "--target_shape",
            type=click.Choice(list(TARGET_SHAPE_CHOICES)),
            help="目标形状（objective 的下级字段；仅 shape/roi_pib/rms_pib/rmse 适用）。",
        ),
    ] = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """解析后的 objective：`(name, shape)` 已归一化、已校验、已提升。"""

    name: str
    shape: TargetShape | None

    @classmethod
    def resolve(cls, objective: str, target_shape: str | None) -> "ObjectiveSpec":
        ...  # 逻辑从 slm_zernike_pib.py:558-595 原样搬入，顺序不变
```

**命名理由**：不叫 `ObjectiveParams`，因为 `slm_pib_runner.py:38` 已有 `ObjectiveParams = ObjectiveParamsPib` 的模块级别名，同名会造成混淆。

### 3.2 字段归置表

| 字段（`ObjectiveParamsPib`） | 现行行号 | 目标位置 |
|---|---|---|
| `name` | 458-465 | **移入** `ObjectiveTarget.name` |
| `target_shape` | 483-490 | **移入** `ObjectiveTarget.target_shape` |
| `target_max_brightness` | 466 | 保持平级 |
| `r_bucket` | 469 | 保持平级 |
| `target_size` | 472 | 保持平级 |
| `target_aspect_ratio` | 475 | 保持平级 |
| `target_center_smooth` | 479 | 保持平级 |
| `shape_schedule` | 491 | 保持平级 |
| `max_roi_energy_loss` | 495 | 保持平级 |
| `w_uniformity` | 499 | 保持平级 |
| `w_peak` | 502 | 保持平级 |
| `w_displacement` | 505 | 保持平级 |
| `log_uniformity` | 508 | 保持平级 |
| `w_ema_decay` | 512 | 保持平级 |
| `w_floor` | 516 | 保持平级 |
| `w_temperature` | 520 | 保持平级 |
| `w_pib_init` | 526 | 保持平级 |
| `w_rms_init` | 530 | 保持平级 |
| `w_ee_init` | 534 | 保持平级 |

**只嵌套 2 个字段，17 个兄弟字段保持平级。** 理由：几何/权重字段在 `target_shape` 之外被独立使用（例如 `r_bucket` 也被 DM 版 `pib` 用），把它们一并塞进去会让 `SlmZernikePibConfig(camera=...)` 的构造可读性急剧下降，且迁移面扩大约 6 倍而收益为零。

### 3.3 `ObjectiveParamsPib` 改造后

```python
@dataclass
class ObjectiveParamsPib:
    target: Annotated[
        ObjectiveTarget,
        option_metadata(click_group=True),   # 新增：显式声明"这是一个 Click 组"
    ] = field(default_factory=ObjectiveTarget)

    # ↓ 以下为 17 个原样保留的平级兄弟字段（target_max_brightness ... w_ee_init）
    ...

    # --- 兼容层：只读转发，保留一个发布周期 ---
    @property
    def name(self) -> str:
        return self.target.name

    @property
    def target_shape(self) -> str | None:
        return self.target.target_shape
```

`CameraParamsPib`（L971）无需改动即可继续通过继承暴露 `.name` / `.target_shape`。

### 3.4 复用性：让其他整形算法能用同一套 schema

```python
# 权威解析层（8 个 objective，全部从 SHAPING_OBJECTIVE_CHOICES 派生）
SHAPING_OBJECTIVE_CHOICES: tuple[str, ...] = (
    "pib", "radiu", "avg_radiu", "rmse", "rmse_out", "shape", "roi_pib", "rms_pib",
)

#: 每个 objective 允许的形状子集；None 表示"该 objective 不接受 target_shape"
#:
#: 从 SHAPING_OBJECTIVE_CHOICES x SHAPE_AWARE_OBJECTIVES 派生，而非手写字典
#: （手写版曾漏掉 rmse_out）。不变量：
#:     OBJECTIVE_ALLOWED_SHAPES[name] is None  <=>  resolve(name, <任意 shape>) 抛 ValueError
OBJECTIVE_ALLOWED_SHAPES: dict[str, tuple[str, ...] | None] = {
    name: (tuple(TARGET_SHAPE_CHOICES) if name in SHAPE_AWARE_OBJECTIVES else None)
    for name in SHAPING_OBJECTIVE_CHOICES
}
```

> **⚠️ 修正：`pib` 的取值是"全部形状"，不是 `None`。**
> 本方案早期版本写过 `"pib": None`，理由是"`pib` + shape 会被提升为 `shape`"。
> 但这与 `ObjectiveSpec.resolve` 的实际校验**矛盾**：`resolve` 只在
> `name not in SHAPE_AWARE_OBJECTIVES` 时抛 `ValueError`，而 `pib` **在**
> `SHAPE_AWARE_OBJECTIVES` 内 —— 即 `resolve("pib", "circle")` 是**合法**的
> （返回 `name="shape", shape="circle"`）。因此 `pib -> None` 会让
> `OBJECTIVE_ALLOWED_SHAPES` 与 resolver 行为**自相矛盾**。
>
> 正确区分是**"接受" vs "保留"**：
> - **接受（acceptance）** = `resolve()` 不抛错 = `pib` 接受全部 8 个形状 → 值为全部形状。
> - **保留（survival）** = 解析后该形状是否仍是 objective 自身的形状 → `pib` **不保留**，
>   由 `ROI_ONLY_SHAPE_OBJECTIVES`（`rmse`/`rmse_out`/`roi_pib`/`rms_pib`）决定谁能保留。
>
> 决定"保留"的常量是 `ROI_ONLY_SHAPE_OBJECTIVES`，不是 `OBJECTIVE_ALLOWED_SHAPES`。
> 对应测试：`test_objective_allowed_shapes_matrix` 断言上述双向不变量。


**关于 `target_side`（方形边长，像素）是否入组**：建议**暂不入组**。`target_side` 是「像素尺寸」，`target_shape` 是「形状枚举」，两者语义层级不同；强行塞进同一个 dataclass 会让 `shape: str | None` 与 `side: int | None` 并列，读者无法判断该填哪个。方形系优化器（`ObjectiveParamsSquare` / `spgd-square`）应复用 `ObjectiveTarget` 的**结构**，但保留自己的字段：

```python
# 复用同一 dataclass 骨架，但用方形专属的形状词表
@dataclass
class ObjectiveTargetSquare:
    name: str = "square"                    # 固定，不暴露 --objective
    target_side: int = 0                    # 像素，0=自动
    target_mean_brightness: int = 0
    ...
```

这样「schema 可复用」体现在**同一个 `ObjectiveSpec` 解析入口 + 同一套 Click 组机制**，而不是强行统一字段语义。

**关于 `beam_shaping_benchmark` 的 3 值子集**：`OBJECTIVE_ALLOWED_SHAPES` 的值本身就是元组，因此窄词表天然被支持（传 `("square", "circle", "gaussian")` 即可），无需为它单独造机制。

**关于 `square` 是否进入 `slm-pib` 的 `--objective` 取值**：**否**。`square` 不是合法的 `ShapingObjective`（`objective.py:300-334` 会拒绝），把它放进 `slm-pib` 的选项列表会诱导用户写出必然崩溃的组合。

---

## 4. Canonical Source Consolidation

### 4.1 解析器归属（受叶子层规则约束）

`AGENTS.md` 规定 `utils/` 是叶子层，**不得** import `optimizer/` / `algorithm/`。而当前未提交的 `ObjectiveSpec` 位于 **optimizer 模块**内（`slm_zernike_shaping.py:502`），因此 `utils/` 无法消费它。

**决策**：把 `OBJECTIVE_CHOICES` + `ObjectiveSpec` 迁入 `src/ao_shaping/utils/image/target/objective.py`，紧邻已有的 `ShapingObjectiveParams`（L206），同属一个模块，零新增依赖。

**循环 import 核查**：
- `objective.py:16` 已经从 `metrics.py` import `TargetShape` → 新增 `TargetShape` 相关代码零新增边
- `runner_common.py:81` 已经从 `utils.image.targets` import `TARGET_SHAPE_CHOICES` → 零新增边
- `metrics.py` 不 import `objective.py` → 无环
- 结论：**不产生循环 import**。

**验收动作**：删除 `slm_zernike_shaping.py:484-548` 的本地副本，改 `from ao_shaping.utils.image.target.objective import ObjectiveSpec`；`slm_zernike_pib.py:558-595` 的 38 行内联链整体替换为一次 `ObjectiveSpec.resolve()` 调用。

### 4.2 5 份 objective 列表的归一

| # | 位置 | 改造方式 |
|---|---|---|
| 1 | `runners/runner_common.py:462` | 删掉手写字面量，改为 `click.Choice(list(SHAPING_OBJECTIVE_CHOICES))` |
| 2 | `slm_zernike_shaping.py` `OBJECTIVE_CHOICES` | **删除**（迁入 `objective.py` 后共用 `SHAPING_OBJECTIVE_CHOICES`） |
| 3 | `utils/image/target/objective.py:268` | **保留为唯一权威源** |
| 4 | `utils/io/file.py:537` `_DATA_MODE_OBJECTIVE_KEYS` | **保持独立**。它是 data-mode 录制格式的**落盘契约**，不是用户可选集。改造为断言 `set(...) <= set(SHAPING_OBJECTIVE_CHOICES)`，让漂移可检测，但**不合并** |
| 5 | `scripts/generate_slm_pib_heuristic_hw_report.py:566` | 删掉 5 值 argparse `choices`，改为从权威元组派生（顺带修好缺失的 `rmse`/`rms_pib`） |

### 4.3 顺带修掉的既有缺陷

| 缺陷 | 位置 | 修法 |
|---|---|---|
| `pib` + 形状时读错列 → `KeyError` | `scripts/generate_slm_pib_heuristic_hw_report.py:232` `df[args.objective]` | 改用 `ObjectiveSpec.resolve(...).name` 得到的归一化名 |
| `--target-shape` 无 `choices=` 校验，非法值一路穿透到优化器才炸 | 同文件 L569-571 | 补 `choices=list(TARGET_SHAPE_CHOICES)`，或直接复用 Click 解析器 |
| 持久化 `config.json` 兼容 | `scripts/diff_beam_frame_analysis.py:92` `cfg.get("target_shape")` | 先读新键 `objective.target_shape`，回退旧键 `target_shape`；旧侧车继续可读 |

---

## 5. Migration Steps

> 每个步骤结束时测试套件必须保持绿。提交按依赖顺序切分。

### Step 1 — 权威解析层（C1）

**新增** `utils/image/target/objective.py`：
- `SHAPING_OBJECTIVE_CHOICES` 作为唯一权威源（已有，微调）
- `OBJECTIVE_ALLOWED_SHAPES` 映射表
- `ObjectiveSpec` 冻结 dataclass + `resolve()`（从 `slm_zernike_pib.py:558-595` **原样搬入**，校验顺序不变）
- `ObjectiveTarget` Click 承载 dataclass

**新增测试** `tests/ao_shaping/utils/test_objective_spec.py`：把 `test_slm_zernike_shaping_objective.py:28-42` 的 `RESOLVE_TABLE` 13 行 + `TestValidation` 的错误消息与**顺序断言**整体搬过来。

`refactor(utils): add canonical ObjectiveSpec resolver + ObjectiveTarget click group`

### Step 2 — Click 组递归（⚠️ 最高风险步）

`with_params` / `_collect_click_annotations` 目前**只遍历顶层 `fields(cls)`**（`runner_common.py:199-211`），且 L237-241 也只按顶层字段名回填。

**后果**：直接塞一个嵌套 dataclass 进去而不改这里，`--objective` 和 `--target_shape` 会**静默消失，无任何报错**——所有选项直接丢失。`with_params` 被 17 个 runner 模块使用，这是全案最容易造成灾难性回归的一步。

**方案**：递归必须**显式 opt-in**，用 `metadata={"click_group": True}` 标记，保证所有现存的平铺 dataclass **逐字节不变**：

```python
# 概念示意，非最终实现
def _is_click_group(hint: Any) -> bool:
    return (
        get_origin(hint) is Annotated
        and any(
            isinstance(m, dict) and m.get("click_group")
            for m in get_args(hint)[1:]
        )
    )

# _collect_click_annotations 遇到 click_group 字段时：
#   1. 先收集该嵌套 dataclass 自身的子字段注解
#   2. 再收集本字段（供 wrapper 回填成实例）
```

`wrapper`（L236-242）需同步支持：把子字段的 kwargs 组装成嵌套实例后注入。

**强制验收（门槛条件）**：本步**必须**对 `main.py` 所有已注册命令做 `--help` 输出 diff，确认只有 `slm-pib` 的选项顺序按预期变化、其余命令**逐字符一致**：

```powershell
python src/ao_shaping/main.py slm-pib spgd --help
python src/ao_shaping/main.py slm-pib heuristic --help
# 以及 main.py:100-117 注册的全部命令
```

`feat(runners): support opt-in nested click groups in with_params`

### Step 3 — 生产优化器接入（C3）

`slm_zernike_pib.py`：
- L558-595 的 38 行内联链 → `spec = ObjectiveSpec.resolve(objective, target_shape)`
- L526-527 改为从 `camera_config.target` 读取
- 删除 L103 的本地 `TargetShape = Literal[...]` 重复定义

`slm_zernike_shaping.py`：
- 删除 L484-548 本地副本，改从 `utils` import
- 保持 L622-629 现有调用形态不变

`refactor(wfless): route both slm-zernike optimizers through canonical ObjectiveSpec`

### Step 4 — Runner 接线（C4）

`slm_pib_runner.py`：
- `_effective_objective_key()`（L100-109）**删除**，改调 `ObjectiveSpec.resolve()`
- L346 调用点随之简化
- L138/145 的 `objective.name` 保持（转发属性仍在）

`refactor(runners): drop duplicate remap in slm_pib_runner`

### Step 5 — 废弃转发属性 + 全部调用点（C5）

`ObjectiveParamsPib` 加只读 `.name` / `.target_shape` 转发属性（§3.3）。

> **注意语义**：转发属性让**读取**继续工作，但**构造**会显式失败——`CameraParamsPib(name="pib")` 抛 `TypeError: unexpected keyword argument 'name'`。这是**刻意设计**：静默忽略未迁移的构造参数会掩盖漏改的调用点。

同步修 `scripts/generate_slm_pib_heuristic_hw_report.py`（L232 KeyError + L569-571 缺 `choices`）与 `scripts/diff_beam_frame_analysis.py`（L92 双键读取）。

`refactor(runners): nest target_shape under objective with deprecated forwarders`

### Step 6 — 方形系接入（PR2，单独提交）

让 `ObjectiveParamsSquare` / `slm_square_shaping` 复用同一 Click 组机制与解析入口。

> ⚠️ **强烈建议与无关的 `cam_type` 改动流分离**：`slm_gsnet_runner.py` / `slm_square_runner.py` / `runner_common.py` 三个文件**同时**被本任务和 `cam_type` 流修改，混在一个 PR 里会让 diff 无法审阅。

`refactor(runners): adopt ObjectiveTarget click group for square shaping`

### Step 7 — 文档同步（C7）

| 文件 | 行号 |
|---|---|
| `AGENTS.md` | 109（`CameraParamsPib` 嵌套容器契约） |
| `README.md` | 171 / 322 / 340 / 410 / 448 / 537 / 585 / 627 / 631-632 / 661 / 670 / 673 / 676 / 831 |
| `main.py` | 24（陈旧 docstring，提及已下线的 `gs`） |
| `docs/slm_pib_heuristic_hw/report.md` | 40, 121, 142, 166, 194, 201 |
| `docs/slm_pib_sim/report.md` | 13 |
| `docs/slm_pib_heuristic_hw/sensitivity.md` | — |
| `docs/TODO.md` | 62（引用的 7 分支内联闭包已被 `ShapingObjective` 取代） |
| `scripts/README.md` | 940, 1128 |
| `docs/slm/slm_square_spgd/README.md` | 22-23, 34, 166-191 |
| `docs/slm/slm_shaping_diff/readme.md` | 73, 98, 102 |
| `docs/diff_beam/README.md` | 9, 26, 30 |

`docs(README): document nested objective CLI schema`

---

## 6. Backward Compatibility Strategy

### 6.1 保留一个发布周期的只读转发

`ObjectiveParamsPib.name` / `.target_shape` 以 `@property` 形式保留，节省约 10 处读取点（含 `slm_pib_runner.py:138,145` 与 4 处测试断言）。

| 场景 | 行为 |
|---|---|
| `camera.target_shape` 读取 | ✅ 正常（转发到 `self.target.target_shape`） |
| `CameraParamsPib(name="pib")` 构造 | ❌ `TypeError`（**刻意**，暴露漏改点） |
| `--objective` / `--target_shape` CLI | ✅ 名称、取值、帮助文本全部不变 |
| `config.json` 旧侧车 | ✅ `diff_beam_frame_analysis.py:92` 双键回退 |
| `utils.image.targets` 兼容 shim | ✅ 不动（identity-preserving，`test_targets.py` 依赖） |

### 6.2 不在兼容层内的事

- **不**保留平铺构造的 `**kwargs` 兜底。静默吞掉错误参数会让漏改的调用点在硬件上才暴露。
- **不**改 `--target_shape` 的拼写下划线形式（README 记录了 `snake_case` 与 `kebab-case` 混用的历史，为避免破坏既有脚本调用，保持原样）。

### 6.3 移除时机

下一个发布周期删除转发属性，届时约 6 处读取点需显式改为 `camera.target.name` / `camera.target.target_shape`。

---

## 7. Call-Site Inventory

### 7.1 必须修改（否则 `TypeError`）

| 位置 | 现状 | 改法 |
|---|---|---|
| `scripts/generate_slm_pib_heuristic_hw_report.py:212-222` | `CameraParamsPib(name=..., target_shape=...)` | `target=ObjectiveTarget(name=..., target_shape=...)` |
| `scripts/compare_shape_objectives.py:216-222` | 同上 | 同上 |
| `optimizer/wfless/slm_zernike_shaping.py:1460` | legacy `__main__` argparse | 同上 |
| `optimizer/wfless/slm_zernike_pib.py:1426` | legacy `__main__` argparse | 同上 |
| `tests/.../test_slm_zernike_shaping_objective.py:137` | `CameraParamsPib(name="pib", ...)` | 同上 |
| `tests/.../test_slm_zernike_rmse.py:152` | `CameraParamsPib(name="rmse", target_shape="circle")` | 同上 |
| `tests/.../test_slm_zernike_pib_shape.py:136` | `CameraParamsPib(name="radiu", target_shape="circle")` | 同上 |
| `tests/.../test_slm_zernike_heuristic.py:70-94` | 设置全部 20 个 objective 字段 | 同上（含 20 字段 round-trip 断言 L120-148） |
| `tests/.../test_slm_zernike_heuristic.py:157` | `CameraParamsPib(name="pib", r_bucket=7, cam_type="sim")` | 同上 |
| `tests/.../runners/test_slm_pib_runner_debug.py:138` | `ObjectiveParams(name="rms_pib")` | 同上 |

### 7.2 CLI 字符串构造

| 位置 | 内容 | 改法 |
|---|---|---|
| `scripts/slm_pib_sim_run.py:68-73` | 构造 Click argv：`"-c","shape","--objective","shape","--target_shape","square"` | **无需改动**（CLI 表面不变）——这正是兼容层的价值 |
| `optimizer/wfless/slm_zernike_pib.py:1497` | legacy argparse `"--objective"` | 可保留 |
| `optimizer/wfless/slm_zernike_shaping.py:1531` | legacy argparse `"--objective"` | 可保留 |

### 7.3 序列化产物

| 位置 | 读法 | 改法 |
|---|---|---|
| `runners/runner_common.py:1028-1043` | `config_payload()` 写 `name` | 写嵌套结构 `objective: {name, target_shape}` |
| `scripts/diff_beam_frame_analysis.py:92` | `cfg.get("target_shape")` | 先读新键，回退旧键 |

### 7.4 明确不动

| 位置 | 原因 |
|---|---|
| `runners/runner_common.py:1190-1199` `PibRunnerParams.objective` | DM 版 `pib` 命令的独立 3 值 objective，与 target_shape 无关 |
| `optimizer/wfless/pib.py:228-243` `_objective_to_min()` | 同上 |
| `utils/io/file.py:537` `_DATA_MODE_OBJECTIVE_KEYS` | 落盘格式契约，只加断言 |
| `display/windows.py`、`display/frames.py` | `target_shape` 是叠加层 dict key |
| `algorithm/signal_processing/differentiable_shaping.py:479-487` | 同名局部变量（`(h,w)` 网格） |
| `utils/image/resample.py:57-85` | 同名参数（输出尺寸） |
| `tests/.../runners/test_diff_shaping_runner.py`、`test_diff_beam_runner.py` | 模块已在 commit `fbef192` 删除，测试自 skip |
| `optimizer/wfless/slm_square_shaping.py` 等 4 个文件的 `cam_type` diff | **无关工作流** |

---

## 8. Test Plan

### 8.1 现有回归网（改动中必须全程保持绿）

| 测试 | 锁定内容 | 最先受影响 |
|---|---|---|
| `tests/.../utils/test_target_package.py:33-70` | `EXPECTED_EXPORTS` 逐模块导出面 + `TARGET_SHAPE_CHOICES` identity + 精确 8 值集合 | **最先炸**（任何重命名/新增导出） |
| `tests/.../wfless/test_slm_zernike_shaping_objective.py` | `RESOLVE_TABLE` 13 行、错误消息、**错误顺序**、`FrozenInstanceError`、7 值集合相等 | Step 1 |
| `tests/.../wfless/test_slm_zernike_pib_shape.py:84-127` | `--target_shape` / `--target_size` / `--target_aspect_ratio` / `--target_center_smooth` 出现在 `run spgd --help`；正则解析 choices 并断言与 `TARGET_SHAPE_CHOICES` **相等** | Step 2 |
| `tests/.../wfless/test_slm_zernike_rmse.py:141-178` | `rmse` + circle 合法；`_effective_objective_key` 重映射；`--objective` 含 `rmse` | Step 4（`_effective_objective_key` 删除） |
| `tests/.../wfless/test_slm_zernike_heuristic.py:47-48, 120-148` | 默认 `camera.name == "pib"` 且 `camera.target_shape is None`；20 字段逐一 round-trip | Step 5 |
| `tests/.../wfless/test_slm_zernike_rms_pib.py:294-378` | `ObjectiveParamsPib` 默认值；`--objective` 含 `rms_pib` | Step 5 |
| `tests/.../utils/test_shaping_objective.py` | `ShapingObjective` 全部行为锚点（`pib`/`radiu`/`avg_radiu` 需 `target_func`；能量护栏；权重自适应） | Step 3 |
| `tests/.../utils/test_targets.py` | 8 种形状 + bogus 拒绝 + `target_shape=` kwargs | Step 1 |
| `tests/.../runners/test_slm_pib_runner_debug.py:59, 159` | `("objective","mode")` 录制列；断言 `"_target_shape"` **不在**数据行内 | Step 4 |

### 8.2 新增测试

| 新测试 | 目的 |
|---|---|
| `tests/.../utils/test_objective_spec.py` | `ObjectiveSpec.resolve` 的权威契约测试（从 shaping 变体文件搬入 13 行 `RESOLVE_TABLE` + 顺序断言） |
| **`test_cli_objective_choices_match_canonical`** | **关键缺口**。断言 `slm-pib spgd --help` 里 `--objective [...]` 的选项集合 **等于** `SHAPING_OBJECTIVE_CHOICES`。现有测试只做「某值在不在」的正则匹配，无法捕获 5 份列表之间的漂移 |
| **`test_nested_group_help_is_stable`** | 对所有已注册命令做 `--help` 快照，防 Step 2 的 Click 组递归误伤其他 runner |
| `test_objective_allowed_shapes_matrix` | 断言 `OBJECTIVE_ALLOWED_SHAPES` 覆盖 `SHAPING_OBJECTIVE_CHOICES` 全部键，且 `radiu`/`avg_radiu` 为 `None` |
| `test_forwarders_are_read_only` | 断言 `.name` 赋值抛 `AttributeError`；`CameraParamsPib(name=...)` 抛 `TypeError`（防静默吞参） |
| `test_data_mode_keys_are_subset` | `set(_DATA_MODE_OBJECTIVE_KEYS) <= set(SHAPING_OBJECTIVE_CHOICES)` |
| `test_config_payload_roundtrip_old_and_new` | 新侧车写 `objective.target_shape`；旧侧车（扁平 `target_shape`）仍可被 `diff_beam_frame_analysis` 读取 |

### 8.3 执行命令

```powershell
# 快速回归（每次提交后）
pytest tests/ao_shaping/utils/test_target_package.py `
       tests/ao_shaping/utils/test_targets.py `
       tests/ao_shaping/utils/test_shaping_objective.py `
       tests/ao_shaping/utils/test_objective_spec.py -q

# 目标函数与 CLI 契约
pytest tests/ao_shaping/optimizer/wfless/ -q

# runner 接线
pytest tests/ao_shaping/runners/ -q

# 全量
pytest -q
```

### 8.4 Success Criteria

- 全量 `pytest -q` 通过；**不新增** skip，不删除任何测试
- 全部已注册命令 `--help` 输出与基线一致（`slm-pib` 选项顺序按设计变化）
- `slm_zernike_pib.py:558-595` 内联链**归零**，`slm_zernike_shaping.py:484-548` 本地副本**归零**
- 5 份 objective 列表 → 1 份权威 + 1 份带断言的落盘契约
- 硬件回归：`slm-pib spgd --objective shape --target_shape square --cam_type sim` 数值与改动前一致（同一 seed）

### 8.5 Manual QA（必须人工执行）

1. `python src/ao_shaping/main.py slm-pib spgd --help` — 确认 `--objective` / `--target_shape` **都在**，且 choices 完整
2. `python src/ao_shaping/main.py slm-pib spgd --objective radiu --target_shape circle --debug` — 确认复现原错误消息 `target_shape can only be used with...`
3. `python src/ao_shaping/main.py slm-pib spgd --objective pib --target_shape square --cam_type sim --epochs 5` — 确认 Recorder 列名为 `shape`（不是 `pib`）
4. `python src/ao_shaping/main.py slm-pib spgd --objective bogus` — 确认 CLI 层直接拒绝
5. `python src/ao_shaping/main.py pib --help` — 确认 DM 版 `pib` 的 3 值 `--objective` **未受影响**
6. 抽查一份新生成的 `config.json`，确认 `objective.target_shape` 嵌套结构；再用 `diff_beam_frame_analysis.py` 读旧侧车

---

## 9. Risks and Mitigations

| # | 风险 | 严重度 | 检测方式 | 缓解 |
|---|---|---|---|---|
| R1 | **`_collect_click_annotations` 静默跳过嵌套字段** → `--objective` / `--target_shape` 从 CLI 消失且无报错 | **致命** | Step 2 后对全部已注册命令做 `--help` diff | 递归必须 `metadata={"click_group": True}` 显式 opt-in，保证现存平铺 dataclass 逐字节不变 |
| R2 | `test_target_package.py:33-70` 的 `EXPECTED_EXPORTS` 因新增导出而失败 | 高 | 该测试最先运行 | Step 1 同步更新 `__all__` 与 `EXPECTED_EXPORTS` |
| R3 | 校验顺序被改，错误消息错位，破坏 `test_..._objective.py:79-83` | 高 | 该测试的顺序断言 | `resolve()` 逻辑**逐行搬移**，不改顺序；Step 1 先把契约测试建在新位置 |
| R4 | 5 份 objective 列表继续漂移 | 中 | **新增**集合相等测试（§8.2） | 全部从 `SHAPING_OBJECTIVE_CHOICES` 派生；`_DATA_MODE_OBJECTIVE_KEYS` 加子集断言 |
| R5 | 旧 `config.json` 侧车无法被 `diff_beam_frame_analysis.py` 读取 | 中 | §8.2 round-trip 测试 | 双键读取（新键优先，回退旧键） |
| R6 | 与无关的 `cam_type` 改动流冲突，diff 无法审阅 | 中 | 人工审阅 | 方形系接入（Step 6）单独成 PR；实施前先确认 `cam_type` 流已合入 |
| R7 | `_effective_objective_key` 删除后 `slm_pib_runner.py:346` 行为漂移 | 中 | `test_slm_zernike_rmse.py:161-168` | 直接复用同一 `ObjectiveSpec.resolve()`，不重新实现 |
| R8 | 转发属性让漏改的构造点静默通过 | 低 | `TypeError` 即为告警 | 刻意设计：读取兼容，构造失败 |
| R9 | 全局重命名 `target_shape` 误伤 3 处同名异义 | 中 | 人工核对 §2.4 清单 | 只改 `runners/runner_common.py:483` 一处；`display/`、`differentiable_shaping.py`、`resample.py` 不动 |
| R10 | `scripts/generate_slm_pib_heuristic_hw_report.py:232` 的 `KeyError` 被误认为是新引入的回归 | 低 | §8.4 硬件回归对比 | 该缺陷**先于**本次改动存在，在 Step 5 顺手修并在 commit message 中注明 |

---

## 10. Out of Scope

以下内容**明确不在**本方案范围内：

1. **无关的 `cam_type` 改动流** — `runner_common.py` / `slm_square_shaping.py` / `slm_square_runner.py` / `slm_gsnet_runner.py` 四个文件的未提交改动，本方案不得触碰或回滚
2. **已删除的 runner** — `diff_shaping_runner.py` / `diff_beam_runner.py` / `gs_hologram_runner.py` / `gs_square_runner.py`（commit `fbef192`），其测试自 skip；仅更新文档残留
3. **DM 版 `pib` 的 3 值 objective** — `runners/runner_common.py:1190-1199` + `optimizer/wfless/pib.py:228-243`，与 `target_shape` 无关
4. **`target_shape` 的 3 处同名异义** — `display/windows.py`、`display/frames.py`、`differentiable_shaping.py:479-487`、`resample.py:57-85`
5. **`target_side` 语义统一** — 方形系保留自己的字段，只复用 Click 组机制与解析入口
6. **`_SHAPE_ALIASES`（`target/ccd.py:227`）** — alias 机制保持不变
7. **`ShapingObjective` 内部行为** — 权重自适应、能量护栏等一律不动
8. **CI / lint 配置** — 仓库当前无 CI、无 linter，本方案不引入

---

## 11. Open Decisions

> 以下 4 项会实质改变工作量与 PR 切分。方案正文已按**推荐项**编写，实施前请确认。

### D1 — 嵌套范围：**仅 2 个字段**（推荐，已采用）

`name` + `target_shape` 进入 `ObjectiveTarget`，17 个几何/权重兄弟字段保持平级。

**备选**：把 `target_size` / `target_aspect_ratio` / `target_center_smooth` 一并嵌套。
**不推荐理由**：迁移面扩大约 6 倍，`SlmZernikePibConfig(camera=...)` 可读性显著下降，而 `r_bucket` 等字段本就被 DM 版 `pib` 独立复用，嵌套后反而需要跨组访问。

### D2 — PR 切分：**两个 PR**（推荐，已采用）

- **PR1**：嵌套 `ObjectiveParamsPib` + Click 组递归 + 解析器归位 + 脚本 bug 修复
- **PR2**：方形系（`ObjectiveParamsSquare` / `spgd-square`）接入同一 Click 组机制

**备选**：合成一个 PR。
**不推荐理由**：`slm_gsnet_runner.py` / `slm_square_runner.py` / `runner_common.py` 三个文件同时被本任务和 `cam_type` 流修改，混合后 diff 无法审阅，且冲突解决会引入隐性风险。

### D3 — 转发属性保留 **1 个发布周期**（推荐，已采用）

`.name` / `.target_shape` 以只读 property 保留，构造则显式失败。

**备选**：立即删除（多约 6 处读取点改动），或永久保留（会长期掩盖漏改点）。

### D4 — 脚本 bug **在本次一并修复**（推荐，已采用）

`generate_slm_pib_heuristic_hw_report.py:232` 的 `KeyError` 与 L569-571 缺失的 `choices=` 在 Step 5 修复——该 commit 本就要打开此文件做 choices 归一。

**备选**：拆成独立的 `fix(scripts):` PR 先行。
**不推荐理由**：拆分会让 choices 归一与 bug 修复连续两次触碰同一文件，且 `KeyError` 缺陷在 PR1 期间仍然存在。

### 附：两个已确认的低风险约定

- `square` **不进入** `slm-pib` 的 `--objective` 取值（它不是合法的 `ShapingObjective`）
- 类名采用 `ObjectiveTarget`（可变、承载 Click 元数据）+ `ObjectiveSpec`（冻结、解析结果），以避开 `slm_pib_runner.py:38` 已有的 `ObjectiveParams` 别名

---

## 12. Estimated Change Surface

| 类别 | 文件数 | 说明 |
|---|---|---|
| 权威层 | 2 | `utils/image/target/objective.py`（改）、`__init__.py`（导出） |
| Click 机制 | 1 | `runners/runner_common.py` |
| 优化器 | 2 | `slm_zernike_pib.py`、`slm_zernike_shaping.py` |
| Runner | 1 | `slm_pib_runner.py` |
| 脚本 | 3 | `generate_slm_pib_heuristic_hw_report.py`、`compare_shape_objectives.py`、`diff_beam_frame_analysis.py` |
| 测试 | 8 | 新增 1、改 7 |
| 文档 | 12 | 见 §5 Step 7 |
| **合计（PR1）** | **约 29** | 不含 PR2 的方形系接入 |
