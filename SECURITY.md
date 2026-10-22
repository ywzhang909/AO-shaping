# 安全策略 (Security Policy)

## 支持版本

仅 `master` / 当前活跃开发分支受安全更新支持。

## 报告漏洞

**请勿通过公开 issue 报告安全漏洞。**

请通过以下方式私下联系维护者：

- 邮件：[请在此填入维护者安全联系邮箱]
- 或在 GitHub 上发起 **private vulnerability report**（Security → Report a vulnerability）。

请在报告中尽量包含：

1. 漏洞类型与受影响模块（如 `src/ao_shaping/drivers/...`）；
2. 复现步骤与最小示例；
3. 潜在影响评估；
4. 如可能，给出初步修复建议。

维护者会在 **5 个工作日内**确认收到，并协商修复与披露时间线。

## 范围说明

AO-Shaping 包含**直接控制光学硬件（变形镜、空间光调制器、相机）**的代码。任何可导致设备失控、错误曝光或人身/设备风险的缺陷均视为高优先级。涉及实验路径的代码（`drivers/`、`runners/` 中的硬件分支）请在漏洞报告中明确标注，以便优先复核。

## 依赖安全

依赖声明集中在 `pyproject.toml`。建议定期运行 `uv lock` 与依赖审计（`pip-audit` / `uv export` + 审计工具）以发现已知漏洞。
