# SLM 整形 GUI 主题配置（`.streamlit/config.toml`）

`.streamlit/` 整目录被 `.gitignore` 忽略（machine-local，同 `.vscode/`），但 SLM 整形界面的
主题（Issue #215 规则 1）只通过这个文件生效——换一台实验机启动 Streamlit 时若该文件缺席，
界面会回落到 Streamlit 默认主题（亮橙强调色、深色模式跟随系统），所有灰阶与语义色约定失效。

因此正本在此存档。新机器上一行命令复原（在仓库根目录执行）：

```powershell
New-Item -ItemType Directory -Force .streamlit | Out-Null; Copy-Item docs/GUI/slm/streamlit_theme_config.toml .streamlit/config.toml -Force
```

配套正本文件：[`streamlit_theme_config.toml`](streamlit_theme_config.toml)（与本目录文档同步维护；
若与机器上的文件冲突，以本正本为准并回改机器副本）。

## 设计要点（为什么是这些值）

- **白底冷蓝操作台**：`base="light"` 显式锁定（否则系统深色模式下浅底深字全部反转），
  主色 `#2E86AB` 是这套控制台的既有身份，本次只补齐统一，不改版。
- **层次靠灰阶而非描边**：8 档冷灰（N0–N7，色相与主色同族），其中 7 档落到 `[theme]`
  真实槽位；N5 `#AAB3BF` 无对应槽位，属登记未用（不做 CSS 注入伪造）。
- **语义色只用于状态，不用于装饰**：红/绿/黄各三件套（色相本体 + 浅底 + 浅底上的深字），
  纯黄在白底不可读故取赭黄，全部满足 AA 对比度。
- **链接用加深档 `#1F6D8C`**：主色 `#2E86AB` 在白底仅 ~3.6:1，正文级链接不够；加深档约
  5.4:1，同色相不破坏统一。
- **图表色不做灰阶**：50 路单元电压柱图与多序列轨迹在灰阶下无法区分，图表序列保留
  主色 + 语义色 + 两个高区分度冷色。
- **圆角统一 0.5rem、输入控件带描边**：形状与可编辑性一一对应。

## `[theme]` 表达不了的部分（有意不实现）

1. 8 档灰阶不是运行期可寻址的 token 体系；页面内第三级底色差需自定义 CSS，本次不做。
2. 字体仅支持三个关键字 / 本地字体文件 / 远程 CSS；仓库无字体资产且实验机不保证外网，
   保持 `sans-serif`。
3. 按钮分层（每屏 ≤1 primary）、空状态引导、分隔线纪律是控件级规则，落在
   `slm_calibration_ui.py` / `multi_slm_controller.py` 代码里，并有 AST 守卫测试钉住，
   与本文件无关。

## 验收口径

代码与配置完成 ≠ 交付：最终以**实验系统上的截图**验收（浅底冷蓝、单主色、灰阶层次），
截图归档到本目录后再勾掉 Issue #215 规则 1。
