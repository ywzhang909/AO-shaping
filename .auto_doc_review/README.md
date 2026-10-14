# .auto_doc_review — 仓库文档自动审核钩子（polling hook）

本目录是 AO-shaping 仓库的"push/issue 钩子"实现：定时轮询 GitHub，
发现新 commit 或新的 documentation issue 时，触发 Hermes cron 代理
自动 review + 修文档 + push + close。

## 为什么用轮询而不是 GitHub webhook

本容器没有公网入站地址（Docker 容器，无公网 IP，无 ngrok/cloudflared
隧道），GitHub 无法把 webhook POST 进来。但出站访问 GitHub 正常，
所以用**出站轮询**实现同样的"有更新就自动处理"语义，且更稳健
（不依赖任何外部可达性）。

## 组成

| 文件 | 说明 |
|------|------|
| `poll_repo_changes.py` | 轮询器。cron 任务的 `script`；stdout 注入代理 prompt |
| `state.json` | 去重状态（单一事实源）。记录 `last_commit`、`seen_commits[]`、`own_pushes[]`、`seen_issues[]` |

轮询器副本同时放在 `/opt/data/scripts/` 和 `~/.hermes/scripts/`
（cron 脚本从那里解析）；三处副本 `STATE_PATH` 都指向本目录的
`state.json`，保证共用一份去重状态。

## 触发源（只读 GitHub API，脚本本身绝不 push）

1. `banckend` 分支上的**新 commit**（他人推的：实验室 `zhanghao`、
   `Sisyphus` 等）。
2. 新出现的 **`documentation` 标签 open issue**。

## 防循环（关键设计）

git push 在这里的 author 归因不可靠（本自动化 push 出的 commit 会被
GitHub 归到 `ywzhang909`，实验室代理 push 的 commit 的 `author.login`
是 `null`），所以**不用 author 判自己，用 SHA 判自己**：

- 代理每次成功 push 后必须执行
  `python3 poll_repo_changes.py --mark-pushed <sha>...`，
  把刚推的 commit 记进 `own_pushes[]`；
- 轮询时 `own_pushes` 和 `seen_commits` 里的 SHA 一律不当新输入。

因此：代理推一个修复 commit → 标记 → 下次轮询 QUIET，不会无限循环。

## rebase 稳健性

本仓库经常 rebase（实验室机器先推）。存下的 `last_commit` 若不再是
祖先，`compare` 会 404；轮询器改用"最近 N 个 commit 减去已见 SHA"
的方式，对 rebase 不敏感。

## 轮询节奏

cron 任务 `25a5a1f4f4a3`，`*/30 * * * *`（每 30 分钟）。
无变化时脚本 stdout 为空 → 代理回 `[SILENT]` → 不通知、零 token。

## 手动操作

```bash
cd /opt/data/workspace/projects/AO-shaping/repo/.auto_doc_review
python3 poll_repo_changes.py --selftest    # 离线测去重逻辑
python3 poll_repo_changes.py --seed        # 重置状态（重新从当前 HEAD 起算）
python3 poll_repo_changes.py               # 立即轮询一次（看有无新事件）
python3 poll_repo_changes.py --mark-pushed <sha>...   # push 后调用
```
