# Agent 注意事项（2026-11-05 当日修改）

> 外迁自根 `AGENTS.md`；按日追加，过期条目迁入 `docs/dev/` 对应专题。
> 覆盖当日两个提交（`c925360f` basetemp 移出 tmpfs、`840f1c8e` 收口三处继承门禁红）与
> 工作区未提交改动（transfer 采集/恢复加固、beam-quality 导航重设计 + 后台预热、SLM 静态整形桌面页）。

## transfer / 恢复语义（`tools/slm/transfer/session_plan.py`）
- 判定键 = **冻结采集组数（`_frozen_groups`）**：`remaining_blocks` 在 `resume=True` 时以
  `_frozen_groups(directory)` 为已完成块数（`range(min(recorded, blocks), blocks)`）；
  `stage_complete` 要求 `manifest.json` 存在且 `_frozen_groups >= blocks`。
- ⚠️ **勿改用 manifest 行数判完成**：`capture()` 每 block 写约 11 行 manifest（A1/B1/B2/A2、
  方向、探索端点、finals），用 `len(manifest) >= blocks + 1` 判完成会在 4 块 recipe 的第 1 块
  后就误判 True、resume 静默跳过其余 3 块。2026-11-05 曾因此改走 manifest 行数（`1b1a7d92`），
  已于 `704fe179` 回退为 `_frozen_groups` 计数——**以 `_frozen_groups` 为准，勿把 manifest
  行数重新引回**。
- `capture.py` 大幅加固采集与恢复（+147 行）：hardening capture and resume.

## transfer-desktop / 冷启动文案（`gui/slm/transfer_desktop/live_view.py`）
- 占位文案「物理预测模型未就绪」→「物理预测模型初始化中」（几何设计初始化）：是**短暂**占位，
  不是错误态；画面本身够用时不需要状态行。

## gui/slm `pattern_specs`（单一真源）
- 新**纯数据**模块 `gui/slm/pattern_specs.py`：`PATTERN_TYPES` / `PARAM_SPECS` /
  `generate_phase_radians` / `specs_for`，Streamlit `helper_tab` 与 PySide6 `static_shaping_page`
  共用。**不得 import streamlit / Qt**；两前端只差「控件怎么渲染」，图案集合与参数语义完全一致，
  不会各写一套而漂移。

## gui/slm `static_shaping_page`（新桌面页）
- 把 Streamlit `helper_tab` 的 `PatternHelper` 交互搬进桌面控制台（`create_phase_from_array` +
  `display_data` 下发 SLM），与光束质量/迁移学习页同住 `DesktopShellApp` 页 2。
- **线程边界**：坐标网格 / 湍流屏 / SLM 打开 + 写入都是秒级阻塞 → 跑在页面**单线程 worker**，
  不得上 Qt 主线程（否则界面冻结）。

## beam-quality shell（导航 + 后台预热）
- 导航栏改 **120 px 单行短标签 + 状态点**；去掉副标题与底部提示（说明交给页面内部空状态文案）。
  长名（如「迁移学习闭环整形」8 汉字 ≈ 96 px）缩成短标签「闭环整形」，全名由页面标题承担，
  悬停提示全名。
- 页 1（迁移学习）**后台预热**：`_start_transfer_warmup` 经 `_WarmTask`（`QThreadPool`）后台
  import + 构造；`_on_transfer_warm_ready` 在**主线程**换占位，`_closing` 守卫；失败 → 占位显示
  「请重新点击导航重试」并重置 `_warm_started`。
- 页 1 / 页 2 惰性创建；看光束质量页的人不为其他页的初始化代价买单。

## 合并损坏教训（`840f1c8e`）
- 坏合并曾**静默丢掉** `_run_validate_blocks` 的 def 签名与 docstring 首行 → 整文件解析损坏
  （ruff `invalid-syntax`），而两个父提交各自都能解析。合并后必须核对 **def 签名 + docstring 首行**
  是否存活；怀疑合并时按父提交逐字补回并证明 AST 与父全等，不要当风格问题「修」。
- `helper_tab.py`：`PatternHelper` 用于返回注解与构造却从未导入 → 构造点运行期 `NameError`
  （注解因 `from __future__ import annotations` 惰性化才没先炸）；按 docstring 规范路径补导入，
  并删未用的 `Literal`（`__all__` 不含它）。

## 术语门禁
- `ccd_live.py` 6 行「实验系统相机→实验系统相机」漂移，`term_unify_20261016.py --apply` 统一，
  dry-run 回 0 文件；新词 §3 登记。全仓仅两处测试含「实验系统」：禁止词清单与门禁自植入的违例样本，
  均不钉本页措辞。

## pytest / CI
- `scripts/build/run_pytest_ci.sh`：basetemp 移出 tmpfs；随之照出并修掉 `test_model_in_loop_hw_ab.py`
  的**目录顺序依赖**。

## 新增测试（工作区未提交）
- `tests/ao_shaping/gui/slm/test_static_shaping_page.py`、
  `tests/ao_shaping/tools/slm/transfer/test_capture_live_prediction.py`、
  `tests/ao_shaping/tools/slm/transfer/test_hardware_retry.py`、
  `tests/ao_shaping/tools/slm/transfer/test_session_plan.py`。

## SLM transfer 复现：capture 饱和 255 的根因（当日 16:05-16:15 三连诊断）
- 现象：`transfer run --execute` 两次在 `reference_start` 失败——最低曝光 0.011 ms 下平场 peak=255、仅 2 饱和像素，
  `light_check` 计数率 5100/ms（全饱和）；而 15:49 的 `slm300_shaping_probe` 平场=51 @ 0.011 ms。
- 根因：**不是 SLM 相位残留，是驱动板（Drive board）间歇性不应答**。日志证据：SLM #1 驱动板 ID 读不到、
  只拿到选板（optionboard 23040015）→ 面板不调制 → 未调制的整束光直接打到相机，故任何图案（含平场）都饱和。
  15:49 那次 probe 驱动板应答（22030090）才正常。运行库警告原话即指此事，且 probe 脚本用 `require_drive_board`
  硬门禁中止（`SLMDriveBoardError` exit 1），transfer capture 却**没有**该门禁、径直采集无效数据。
- 处置：需**物理断电 SLM-300 控制器重置驱动板**。预期稳定应答后 transfer 平场回到 ~51 形态、capture 可过。
- 复现清单：transfer `reference_start` 饱和 → 先查 run log 是否 `驱动板 (Drive board) ID 读不到`；勿误判为光路/曝光问题。

## SLM transfer 复现于 miicam+SLM#1 成功（下午 17:19 run，EXIT=0）
- 完整跑通 capture→fit→response→blur→CCD blocks→audit→summary，数据全在 `data/debug/slm_transfer/20261105_171947/`。
- 前后堵点与解除：驱动板间歇不应答（需物理断电重置，见上"16:05-16:15"节）→ 平场/平方在 0.011ms 钳到单档使
  `exposure_response_analysis` 失败（≥3 档门禁，`utils/image/exposure_response.py:18`，有测试守卫勿删）→ 装 ND 后
  平场/平方各得 4 个不同曝光档，响应拟合通过。
- **科学口径守住**：12/12 块均 `acceptance=model_only`（`session_plan.model_only_decision`）——只表示模型损失下降，
  **不代表 CCCT 收到的确有改善**。audit `status=passed` 同样只是流水线通过。三者不能互替。
- **关键未解**：三个形状 seed→final 的 CCD measured CV/EE/Pearson 几乎不动，而模型 loss 下降、predicted pearson 约
  0.95→0.84；predicted↔measured 相关从 ~0.5-0.6 跌到 ~0.34-0.41。即模型自洽优化，但**预测与实测在优化中背离**。
  方向：量纲（预测 CV 0.1-0.47 vs 实测更均匀 0.5-0.75）、或当前 bench ramp 几何/ND 后光场与模型几何不匹配。
- 若下一步要真实 CCD 改善：喂当前 round ramp 几何/效标后重拟合、给 measured 指标加权、或先定位预测-实测分歧在
  peak/能量/CV 哪个成分最大。

## 已知未解（提交时）
- `ao_shaping.gui.slm.pattern_specs` 提交时不存在于任何 ref/磁盘 → `helper_tab` 及其下游 9 个
  测试模块收集期 `ImportError`。⚠️ 工作区已新增 `pattern_specs.py` / `static_shaping_page.py`
  （未跟踪）——**必须与 helper_tab 一同提交**，否则收集仍断。
