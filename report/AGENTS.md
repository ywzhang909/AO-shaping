# AGENTS.md — `report/` 目录结构与路径规范

> 本文件约束 `report/` 下的目录布局、文件命名、资产归属与溯源登记。
> 与仓库根 `AGENTS.md` 冲突时，**以本文件为准**（`report/` 专用规则）。
> 详表：子目录对照 [docs/dev/report_subdir_index.md](../docs/dev/report_subdir_index.md)；反模式 [docs/dev/report_antipatterns.md](../docs/dev/report_antipatterns.md)；资产/链接 [docs/dev/report_asset_rules.md](../docs/dev/report_asset_rules.md)。

---

## 1. 两类文档分工（红线）

| 位置 | 内容 | 规则 |
|------|------|------|
| `docs/` | 设备说明文档（规格/SDK/手册/图集） | 不放实验结论 |
| `report/` | 实验报告（测量值/参数/口径/被推翻的假设） | 结论的唯一事实源 |

每份实验报告必须能追到生成它的脚本。

## 2. 目录布局：一份报告 = 一个子目录

报告不平铺在 `report/<topic>/` 根下；按生成脚本/实验系列分子目录，报告与其独占资产同住。结构与对照表见 [docs/dev/report_subdir_index.md](../docs/dev/report_subdir_index.md)。

```
report/<topic>/<系列名>/
├── <报告名>.md
├── figures/       # 正文引用的图
├── images/        # 插图/动画
├── artifacts/     # 上游脚本原始产物 (CSV/JSON)，输入契约
└── assets/        # 现场抓取的原始资产
```

## 3. 溯源登记（硬性）

`scripts/_common/provenance.py::REPORTS` 是唯一映射源，键 = 仓库根相对路径（精确匹配）。

新增/移动报告后同步三处：

| 位置 | 改什么 |
|------|--------|
| `scripts/_common/provenance.py::REPORTS` | 键改新路径；`script`/`extra_scripts` 填脚本 |
| `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN` | 键跟着改 |
| 报告正文 | 跑 `python scripts/sync_report_provenance.py` 重生成溯源块与 `report/README.md` 索引 |

生成脚本路径常量必须写死新目录（`REPORT_KEY`/`OUT_DIR`/`FIG_DIR`）。

## 4. 资产归属与链接

详见 [docs/dev/report_asset_rules.md](../docs/dev/report_asset_rules.md)。

## 5. 校验

```bash
python scripts/sync_report_provenance.py --check
python -m pytest tests/ao_shaping/scripts/test_report_provenance.py -q
ruff check .
```

`test_report_provenance.py` 校验：每份报告有且仅有一个溯源块、注册表脚本路径存在、报告内相对链接可解析、索引链接可解析、基线断链合理豁免。

## 6. 反模式

> 详见 [docs/dev/report_antipatterns.md](../docs/dev/report_antipatterns.md)。