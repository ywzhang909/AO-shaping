<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

<!-- AO-SHAPING-TERMS_START -->
## 术语规范（必须遵守）

本仓库所有中文文档写作 / 改写 / 摘要，必须遵守
[术语规范表 `docs/terminology_glossary.md`](../docs/terminology_glossary.md)（单一事实源）：

- 口语词禁用、规范词必用：`台架→实验系统`、`实机/真机→实验`、`上机→实验`、
  `平场泄漏→加性背景（杂散光）项`、`瞳孔→瞳面`、`跑通→在线实现`、`打散→弥散`、
  `金标准→参考（真值）`、`围栅→环围`、`板卡→控制卡`、`粉饰→拔高` 等（全表见术语表 §1）。
- 只改中文叙述；**绝不改** 代码标识符 / 围栏代码块 / mermaid / 行内 `code` /
  纯英文 / 厂商型号 / 数值参数 / `docs/dev/call_graph_wiki/**`。
- `pupil（瞳面）` 在中文叙述中**首现夹注一次**，不得双重夹注。`bake-off`
  自 2026-10-17 起中文叙述一律用**对比择优**，不再夹带英文。
- 数据守恒：替换不得改动任何数值 / 单位 / 公式 / 参数 / 路径 / DOI / 引用编号。
- 提交前校验（全绿）：
  - `python3 scripts/_common/term_unify_20261016.py` → 输出 `0 个文件`
  - `git grep -E 'pupil（瞳面）（|bake-off（对比择优）|bake-off（强制对比择优）'` → 空
  - 残留 `平场泄漏|台架|实机|真机|上机|瞳孔|跑通|打散|灌进|钻空子|金标准|围栅`
    的命中须全部落在术语表 §2 DO-NOT-TOUCH 内
- 新口语词先在术语表 §3 登记（T-60 起）再替换。
<!-- AO-SHAPING-TERMS_END -->
