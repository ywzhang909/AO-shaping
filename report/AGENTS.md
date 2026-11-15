# AGENTS.md — `report/` 目录结构与路径规范

> 本文件约束 `report/` 下的目录布局、文件命名、资产归属与溯源登记；
> 与根 [AGENTS.md](../AGENTS.md) 冲突时，**以本文件为准**（`report/` 专用规则）。

- **目录布局与分工**：[docs/dev/report_subdir_index.md](../docs/dev/report_subdir_index.md)（§1.0 两类文档分工红线 / §2.0 布局模板 / §2.1 已确立子目录 / §2.2 新增报告）。
- **溯源登记（硬性）与校验**：[docs/dev/report_provenance.md](../docs/dev/report_provenance.md)（`scripts/_common/provenance.py::REPORTS` 唯一映射源，新增/移动后同步三处；校验 `sync_report_provenance.py --check` + `test_report_provenance.py` + `ruff check .`）。
- **资产归属与链接**：[docs/dev/report_asset_rules.md](../docs/dev/report_asset_rules.md)。
- **反模式**：[docs/dev/report_antipatterns.md](../docs/dev/report_antipatterns.md)。
