# Report 资产归属与链接规则

> 外迁自 `report/AGENTS.md` §4-§5；完整报告规范见 [report/AGENTS.md](../../report/AGENTS.md)。

## 资产归属
- 独占：一张图只属于一份报告；跨报告复用 → 复制，不建 `shared/`。
- 跟随源图：派生产物写回源图所在目录。
- `artifacts/`：上游 CSV/JSON，与报告同目录。
- `figures/` = 正文图；`images/` = 插图/动画。
- 孤儿资产（无报告引用）不允许留在 `report/` 根。

## 链接规则
- 同目录资产：`figures/xxx.png`（不带 `../`）；同族另一份报告：`xxx.md`。
- 其他子目录：`../<子目录>/xxx.md`；`report/<topic>/` 根的 `README.md`：`<子目录>/xxx.md`。
- 引用 `data/`/`docs/`：按实际深度写 `../../`（子目录内三层）。
- 仓库根相对路径只出现在散文里（`` `report/shape_trials/...` ``），**不要**放进 Markdown 链接。