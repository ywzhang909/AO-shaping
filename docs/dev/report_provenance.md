# Report 溯源登记与校验（外迁自 `report/AGENTS.md` §3/§5）

> 完整报告规范见 [report/AGENTS.md](../../report/AGENTS.md)。

## 溯源登记（硬性）

`scripts/_common/provenance.py::REPORTS` 是唯一映射源，键 = 仓库根相对路径（精确匹配）。

新增/移动报告后同步三处：

| 位置 | 改什么 |
|------|--------|
| `scripts/_common/provenance.py::REPORTS` | 键改新路径；`script`/`extra_scripts` 填脚本 |
| `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN` | 键跟着改 |
| 报告正文 | 跑 `python scripts/sync_report_provenance.py` 重生成溯源块与 `report/README.md` 索引 |

生成脚本路径常量必须写死新目录（`REPORT_KEY`/`OUT_DIR`/`FIG_DIR`）。

## 校验

```bash
python scripts/sync_report_provenance.py --check
python -m pytest tests/ao_shaping/scripts/test_report_provenance.py -q
ruff check .
```

`test_report_provenance.py` 校验：每份报告有且仅有一个溯源块、注册表脚本路径存在、报告内相对链接可解析、索引链接可解析、基线断链合理豁免。