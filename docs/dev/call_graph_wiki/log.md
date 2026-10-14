# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`

## [2026-10-13] create | Wiki initialized
- Domain: AO-shaping 仓库开发使用手册（main.py 调用图 + 逐函数文档）
- 结构：SCHEMA.md / index.md / log.md / entry/ / commands/ / functions/ / raw/ / assets/

## [2026-10-13] ingest | call_graph_final.json + docstrings
- AST 静态分析：497 模块，入口 = main.py → cli() → 20 命令
- 调用图：193 已解析节点 / 290 边（`raw/assets/call_graph_final.json`）
- 96/103 未解析名经全库同名唯一匹配恢复；其余为模块内私有 `_*` 辅助或第三方
- 生成 2 entry + 20 command + 25 function 页 + 1 diagram 总览
- 分层调用图 mermaid → kroki SVG + headless Chrome PNG，经 vision 验证渲染正确

## [2026-10-14] create | 全局调用图 + 全局类图（mermaid）
- `diagram/global-call-graph.mmd`：193 函数节点聚合为 31 个包级节点（8 个一级包 subgraph），
  边权=调用次数，脚本 `scripts/call_graph/build_module_graph.py`（幂等，独立重跑）
- `diagram/class-diagram.mmd`：7 域核心类骨架（设备/注册表/模型/优化器/目标/标定/控制），
  全部继承边经 AST 逐一核对（全库 587 个顶层类中保留核心）
- 两图 mmdc 渲染 SVG+PNG 至 `assets/`，vision 验证渲染正确
- `index.md`/`SCHEMA.md` 登记 2 个新 diagram 页
