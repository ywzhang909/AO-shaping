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
