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
