# AGENTS.md — `report/` 目录结构与路径规范

> 本文件约束 `report/` 下的目录布局、文件命名、资产归属与溯源登记；
> 与仓库根 `AGENTS.md` 冲突时，**以本文件为准**（`report/` 专用规则）。

## 1. 两类文档分工（红线）

| 位置 | 内容 | 规则 |
|------|------|------|
| `docs/` | 设备说明文档（规格/SDK/手册/图集） | 不放实验结论 |
| `report/` | 实验报告（测量值/参数/口径/被推翻的假设） | 结论的唯一事实源 |

每份实验报告必须能追到生成它的脚本。

## 2. 目录布局：一份报告 = 一个子目录

报告不平铺在 `report/<topic>/` 根下；按生成脚本/实验系列分子目录，报告与其独占资产同住。
布局模板与子目录对照见 [docs/dev/report_subdir_index.md](../docs/dev/report_subdir_index.md)（§2.0 / §2.1）。

## 3. 规则与校验索引

- 溯源登记（硬性）：[docs/dev/report_provenance.md](../docs/dev/report_provenance.md)（`scripts/_common/provenance.py::REPORTS` 唯一映射源；新增/移动后同步三处）。
- 资产归属与链接：[docs/dev/report_asset_rules.md](../docs/dev/report_asset_rules.md)。
- 校验：[docs/dev/report_provenance.md](../docs/dev/report_provenance.md)（`sync_report_provenance.py --check` + `test_report_provenance.py` + `ruff check .`）。
- 反模式：[docs/dev/report_antipatterns.md](../docs/dev/report_antipatterns.md)。
