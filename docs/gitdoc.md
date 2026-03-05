# Git 文档规范

**项目名称:** AO-shaping  
**文档版本:** 1.0.0  
**创建日期:** 2026-03-05  
**最后更新:** 2026-03-05

---

## 目录

1. [概述](#1-概述)
2. [仓库设置](#2-仓库设置)
3. [分支策略](#3-分支策略)
4. [提交规范](#4-提交规范)
5. [工作流程](#5-工作流程)
6. [Pull Request 流程](#6-pull-request-流程)
7. [代码审查指南](#7-代码审查指南)
8. [发布流程](#8-发布流程)
9. [常见问题](#9-常见问题)

---

## 1. 概述

本文档定义了 AO-shaping 项目的 Git 工作流标准。通过遵循这些规范，团队成员可以保持代码库的一致性，提高协作效率，并确保代码质量管理。

### 1.1 目标

- 统一团队成员的 Git 操作习惯
- 规范化提交信息，便于追踪历史
- 明确分支管理策略
- 简化代码审查流程

### 1.2 适用对象

- 项目所有贡献者
- 维护者
- 使用 Git 进行版本控制的开发人员

---

## 2. 仓库设置

### 2.1 克隆仓库

```bash
# 克隆主仓库
git clone https://github.com/[organization]/AO-shaping.git

# 进入项目目录
cd AO-shaping

# 查看远程仓库
git remote -v
```

### 2.2 配置 Git

在进行任何操作之前，请先配置 Git 用户信息：

```bash
# 设置用户名
git config user.name "Your Name"

# 设置邮箱
git config user.email "your.email@example.com"

# 设置默认分支名
git config init.defaultBranch master

# 启用拉取时自动变基（推荐）
git config pull.rebase true

# 设置拉取策略
git config rebase.autoStash true
```

### 2.3 开发环境配置

本项目使用 `uv` 进行依赖管理：

```bash
# 安装依赖
uv sync

# 运行测试
pytest

# 代码覆盖率
pytest --cov=ao_shaping --cov-report=html
```

---

## 3. 分支策略

### 3.1 分支类型

| 分支类型 | 命名规则 | 用途 | 生命周期 |
|---------|---------|------|---------|
| master/main | `master` | 生产分支 | 永久 |
| develop | `develop` | 开发主分支 | 永久 |
| feature/* | `feature/功能名称` | 新功能开发 | 临时 |
| bugfix/* | `bugfix/问题描述` | Bug修复 | 临时 |
| hotfix/* | `hotfix/问题描述` | 紧急修复 | 临时 |
| release/* | `release/版本号` | 发布准备 | 临时 |
| refactor/* | `refactor/重构范围` | 代码重构 | 临时 |

### 3.2 分支命名规范

- 使用小写字母
- 使用连字符 `-` 分隔单词
- 保持简洁但具有描述性
- 避免特殊字符和空格

**正确示例：**
```
feature/user-authentication
bugfix/fix-slm-connection-timeout
hotfix/security-patch
refactor/driver-cleanup
```

**错误示例：**
```
Feature/UserAuth  # 大写
bug fix/login    # 空格
fix_bug_123      # 下划线
```

### 3.3 保护分支

以下分支受到保护，禁止直接推送：

- `master` - 生产分支
- `develop` - 开发分支

保护规则：
- 必须通过 Pull Request 合并
- 需要至少 1 人审查通过
- 必须通过所有 CI/CD 检查

---

## 4. 提交规范

### 4.1 提交信息格式

本项目采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### 4.2 类型 (Type)

| 类型 | 描述 | 示例 |
|-----|------|------|
| `feat` | 新功能 | `feat(drivers): add Santec SLM driver` |
| `fix` | Bug 修复 | `fix(algorithm): correct wavefront calculation` |
| `docs` | 文档更新 | `docs: update API documentation` |
| `style` | 代码格式 | `style: format import order` |
| `refactor` | 重构 | `refactor(wf): simplify Zernike polynomials` |
| `perf` | 性能优化 | `perf: optimize convolution kernel` |
| `test` | 测试相关 | `test: add unit tests for DM driver` |
| `chore` | 构建/工具 | `chore: update pytest configuration` |
| `ci` | CI/CD | `ci: add GitHub Actions workflow` |

### 4.3 作用域 (Scope)

可选，用于说明提交影响的范围：

| 作用域 | 描述 |
|-------|------|
| `drivers` | 硬件驱动 |
| `slm` | SLM 相关 |
| `dm` | DM 相关 |
| `wfs` | 波前传感器相关 |
| `wf` | 波前控制算法 |
| `wfless` | 无波前传感算法 |
| `utils` | 工具模块 |
| `display` | 显示/可视化 |
| `algorithm` | 算法模块 |
| `ci` | CI/CD 配置 |
| `docs` | 文档 |

### 4.4 提交信息规则

**Subject（标题）：**
- 使用祈使句，现在时态
- 首字母小写
- 结尾不加句号
- 不超过 50 字符
- 描述做了什么，而不是如何做

**Body（正文）：**
- 隔一行后书写
- 每行不超过 72 字符
- 说明 what 和 why，而不是 how

**Footer（脚注）：**
- 用于关联 Issue：`Closes #123`
- 用于破坏性变更：`BREAKING CHANGE: description`

### 4.5 提交示例

**功能提交：**
```
feat(wfs): add Shack-Hartmann WFS support

Implement Shack-Hartmann wavefront sensor with:
- Centroid calculation
- Zernike mode reconstruction
- Real-time acquisition mode

Closes #42
```

**修复提交：**
```
fix(slm): resolve connection timeout issue

The Santec SLM driver was not properly handling connection
timeouts. Added retry logic with exponential backoff.

Fixes #78
```

**破坏性变更：**
```
feat(algorithm): change optimization API

BREAKING CHANGE: The Optimizer class now requires a
configuration dictionary instead of individual parameters.

Migration:
- Before: Optimizer(learning_rate=0.01, iterations=100)
- After:  Optimizer(config={"lr": 0.01, "iter": 100})
```

---

## 5. 工作流程

### 5.1 开发流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Git 工作流程图                                │
└─────────────────────────────────────────────────────────────────────┘

    master ────────────────────────────────────────────────────────
      │                                                            │
      │  (tag v1.0.0)                                              │
      ▼                                                            │
    release ────────────────────────────────────────────────────────
      │                                                            │
      │  (merge)                                                   │
      ▼                                                            │
    develop ────────────────────────────────────────────────────────
      │                                                            │
      │    │──────────┐                                            │
      │    ▼          ▼                                            │
      │ feature/A  feature/B                                        │
      │    │          │                                            │
      │    └────┬─────┘                                            │
      │         ▼                                                  │
      └─────────┘ (rebase/merge to develop)
```

### 5.2 功能开发流程

```bash
# 1. 确保 develop 分支最新
git checkout develop
git pull origin develop

# 2. 创建功能分支
git checkout -b feature/new-feature-name

# 3. 开发并提交
git add .
git commit -m "feat(scope): description"

# 4. 定期同步上游变更
git fetch origin
git rebase origin/develop

# 5. 完成开发后，推送分支
git push origin feature/new-feature-name
```

### 5.3 Bug 修复流程

```bash
# 1. 从 develop 创建修复分支
git checkout develop
git pull origin develop
git checkout -b bugfix/issue-description

# 2. 修复问题并测试
# ... (修复代码)
pytest  # 运行测试确认修复

# 3. 提交修复
git add .
git commit -m "fix(scope): resolve issue"

# 4. 推送
git push origin bugfix/issue-description
```

### 5.4 紧急修复流程

```bash
# 1. 从 master 创建热修复分支
git checkout master
git checkout -b hotfix/urgent-fix

# 2. 修复并测试
# ... (紧急修复)

# 3. 提交
git commit -m "hotfix: critical security patch"

# 4. 同时推送到 master 和 develop
git push origin hotfix/urgent-fix
git checkout develop
git merge hotfix/urgent-fix
git push origin develop
git checkout master
git merge hotfix/urgent-fix
git push origin master
```

---

## 6. Pull Request 流程

### 6.1 创建 Pull Request

1. 推送分支到远程
2. 在 GitHub/GitLab 打开 PR
3. 填写 PR 模板

### 6.2 PR 模板

```markdown
## 描述
[简要描述这个 PR 做了什么]

## 变更类型
- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 破坏性变更 (BREAKING CHANGE)
- [ ] 文档更新 (docs)

## 测试
- [ ] 单元测试通过
- [ ] 手动测试通过
- [ ] 无测试（仅文档更新）

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 已添加必要的文档
- [ ] 已更新 CHANGELOG（如果需要）
- [ ] 所有测试通过

## 相关 Issue
Closes #xxx
```

### 6.3 PR 标题规范

与提交信息格式相同：

```
feat(drivers): add new SLM driver support
fix(wf): correct Zernike coefficient calculation
docs: update installation guide
```

### 6.4 PR 合并策略

- **Squash Merge**（推荐）：将所有提交合并为一个，保持历史线性
- **Merge Commit**：保留所有提交历史
- **Rebase**：将分支变基到目标分支

```bash
# Squash Merge（通过 GitHub UI 操作）
# 或者命令行
git checkout develop
git merge --squash feature/your-feature
git commit -m "feat(scope): complete feature description"
git push origin develop
```

---

## 7. 代码审查指南

### 7.1 审查者职责

- 检查代码逻辑正确性
- 验证代码风格一致性
- 确保测试覆盖充分
- 识别潜在问题

### 7.2 审查清单

**代码质量：**
- [ ] 代码逻辑正确
- [ ] 没有明显的 Bug
- [ ] 错误处理适当
- [ ] 性能合理

**代码风格：**
- [ ] 遵循 PEP 8 规范
- [ ] 导入顺序正确（stdlib → third-party → local）
- [ ] 命名规范一致
- [ ] 注释适当

**测试：**
- [ ] 有必要的单元测试
- [ ] 测试覆盖新增功能
- [ ] 测试通过

**文档：**
- [ ] 公共 API 有文档字符串
- [ ] 复杂逻辑有注释说明
- [ ] README 已更新（如果需要）

### 7.3 审查意见格式

**推荐格式：**

```markdown
# 必须修改（阻塞合并）
- [ ] 这里有逻辑错误，需要修复
- [ ] 缺少必要的错误处理

# 建议修改（可选）
- [NIT] 可以使用更简洁的写法
- [SUGGESTION] 建议添加类型注解

# 提问
- [QUESTION] 这里为什么要这样做？
```

### 7.4 审查通过标准

- 至少 1 位审查者 approve
- 无阻塞性问题
- 所有 CI 检查通过

---

## 8. 发布流程

### 8.1 版本号规范

采用语义化版本号 (Semantic Versioning)：

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: 不兼容的 API 变更
- **MINOR**: 向后兼容的新功能
- **PATCH**: 向后兼容的 Bug 修复

### 8.2 发布步骤

```bash
# 1. 更新版本号
# 修改 pyproject.toml 中的版本

# 2. 更新 CHANGELOG
# 添加新版本变更说明

# 3. 创建 tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# 4. 推送到远程
git push origin master --tags

# 5. 创建 GitHub Release
# 在 GitHub 页面创建新的 Release
```

### 8.3 CHANGELOG 格式

```markdown
# Changelog

## [1.0.0] - 2026-03-05

### Added
- New SLM driver support (#42)
- Wavefront reconstruction algorithm (#45)

### Changed
- Improved DM response time (#50)

### Fixed
- Connection timeout issue (#48)

### Removed
- Deprecated `old_api` function (#55)

### Breaking Changes
- Migration guide for new optimizer API (see #60)
```

---

## 9. 常见问题

### 9.1 如何撤销提交？

```bash
# 撤销未推送的提交
git reset --soft HEAD~1

# 撤销已推送的提交（谨慎使用）
git revert HEAD~1
git push origin branch-name
```

### 9.2 如何解决合并冲突？

```bash
# 1. 获取最新代码
git fetch origin
git merge origin/develop

# 2. 解决冲突后
git add .
git commit -m "Resolve merge conflicts"

# 或者使用变基
git rebase origin/develop
```

### 9.3 如何清理本地分支？

```bash
# 删除已合并的本地分支
git branch -d feature/old-feature

# 强制删除未合并的分支
git branch -D feature/abandoned

# 删除远程已删除的本地追踪分支
git fetch --prune
```

### 9.4 如何更新 fork？

```bash
# 添加上游仓库
git remote add upstream https://github.com/original/AO-shaping.git

# 获取上游更新
git fetch upstream

# 合并到本地分支
git checkout develop
git merge upstream/develop
```

---

## 附录

### A. Git 常用命令速查

```bash
# 分支操作
git branch                    # 列出分支
git checkout -b branch-name   # 创建并切换
git checkout branch-name      # 切换分支
git branch -d branch-name     # 删除分支

# 提交操作
git status                    # 查看状态
git add .                     # 添加所有修改
git commit -m "message"       # 提交
git log --oneline -10         # 查看提交历史

# 远程操作
git remote -v                 # 查看远程
git fetch                    # 获取
git pull                     # 拉取并合并
git push                     # 推送

# 变基与合并
git rebase develop           # 变基到 develop
git merge branch-name        # 合并分支
git stash                   # 暂存修改
git stash pop               # 恢复暂存
```

### B. 关联工具

- **GitHub**: 代码托管与协作
- **pytest**: 测试框架
- **uv**: 依赖管理
- **GitHub Actions**: CI/CD

### C. 参考资料

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)

---

*本文档由 AI 自动生成，最后更新于 2026-03-05*
