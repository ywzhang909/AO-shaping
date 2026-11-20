# 贡献指南 (Contributing)

感谢你关注 AO-Shaping。本仓库采用约定式提交（Conventional Commits）与自动化质量门禁，请在提交前阅读以下约定。

## 分支与远程命名

- 分支名请使用**有意义的英文短名**，避免拼写错误（历史上有过 `banckend` 应为 `backend` 的教训）。
- 远程名应为 `github`（历史上有过 `gitub` 拼写错误）。
- 推荐前缀：`feat/`、`fix/`、`refactor/`、`docs/`、`test/`、`exp/`。

## 提交信息规范

遵循 Conventional Commits：`type(scope): 简要中文描述`，并尽量关联 issue 编号，例如：

```
fix(slm): 修正 Santec 连接重试逻辑 (#b48b08f3)
refactor(optimizer): 收敛共享常量至 constants.py (#91)
```

常用 `type`：`feat` / `fix` / `docs` / `refactor` / `test` / `chore`。

## 本地质量门禁（提交前必须全绿）

```bash
ruff check .                 # 强制 E+F 规则，全仓须通过
pytest                       # 默认子集；硬件 / ml / 部分 GUI 测试已被忽略（需真机或重型依赖）
```

- 术语统一：中文叙述必须遵守 `docs/terminology_glossary.md`，提交前运行
  `python3 scripts/_common/term_unify_20261016.py`（dry-run，输出 `0 个文件`）。
- **push 前自检（#251 P2-A）**：`python scripts/push_preflight.py` —— ruff +
  术语 + 模块体积棘轮 + 四个静态守卫（repo-root bootstrap / R-25 helper 复发 /
  dry-run 自包含 / golden 容差），约 60 s。ci169 教训：这四类通道守卫都在，
  但只在 push 后的完整分片才跑到，每个错误付一整段 CI 红窗口。CI 环境对
  `data/`、`libs/gxipy`、绝对路径的保证集见
  [`docs/ci-environment-contract.md`](docs/ci-environment-contract.md)。
- **census 重钉与特性同 commit（#251 P0-A）**：新增 `Santec(`/`create_camera(`
  构造点或新探针模块的 commit，必须同 commit 内含对应守卫重钉
  （`test_cli_argv_binding` 的 census 字典、`test_probe_flags` 的
  `PROBE_MODULES`/flag 面数字、`probe_help_golden.json`）。合并把两者拆进
  不同窗口时，`tests/ao_shaping/census_repin.py` 只把「已登记 TODO 台账」的
  漂移降为黄灯 xfail；未登记的漂移文件仍是红灯。黄灯是留痕不是豁免，
  下一次触碰该守卫的 commit 必须把它转绿。
- 新增依赖请同步更新 `pyproject.toml`，不要手动 `pip install`。

## 测试

- 测试与源码同构镜像在 `tests/ao_shaping/`。
- 需要真实设备的测试请加 `@pytest.mark.hardware`；长耗时测试加 `@pytest.mark.slow`。
- 默认 `pytest` 会跳过 `tests/ao_shaping/drivers`、`tests/ao_shaping/ml` 以及个别会挂起的 GUI 测试；这些区域需要硬件或重型依赖，**不属于普通回归的一部分**。

## Issue / PR

- 提交功能或修复前，请先在 issue 中登记（参考既有 `[R-xx]` / `[H-xx]` / `[code-scan]` 编号体系）。
- PR 描述请说明动机、改动范围与验收方式；CI 的 ruff 门禁必须通过。

## 代码风格

- 统一使用 `ruff` 格式化（双引号、空格缩进、自动换行）。
- 日志统一使用 `loguru` 的 `logger`，不要使用 `print()`。
- 并行迭代请优先使用 `zip(..., strict=True)` 以在长度不匹配时显式失败。
- 公开 API 建议添加类型注解与 docstring（含 `Args` / `Returns`）。
