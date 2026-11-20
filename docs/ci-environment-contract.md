# CI 环境契约（#251 P1-A）

> 本文件声明 **CI 容器对测试/脚本保证什么、不保证什么**。任何测试或脚本的
> 外部文件依赖必须属于「保证存在」集；否则必须自带合成回退（dry-run 路径由
> `tests/ao_shaping/scripts/test_dry_run_self_contained.py` 静态强制）。
> 立项动因：ci169 的 #2/#5/#6/#8 四红全部是「开发机环境假设未声明」——
> 本机恒绿、CI 全新 checkout 恒红。

## 保证存在

| 项 | 说明 |
|---|---|
| `src/`、`scripts/`、`tests/`、`docs/`、`report/` | 全部 git 跟踪内容；测试可自由读取 |
| `libs/` 目录本身 | git 跟踪（驱动 SDK 说明），但**内容见下** |
| `data/`（仅分片脚本 staging 注入的） | `scripts/build/run_pytest_ci.sh` 的 staging 机制按需复制白名单产物；测试只应依赖 `AO_PYTEST_ROOT` 下已注入的样本，不得假设整棵 `data/` 在场 |
| 依赖组 | CI 镜像按 `pyproject.toml`  extras 安装，含 PySide6 / torch-cpu（本地轻量容器**没有** PySide6，收集错误以 CI 为准） |

## 保证不存在

| 项 | 后果与正确写法 |
|---|---|
| `data/` 全量语料（hwdataset 65 GB、调试 pkl 等） | dry-run/单测读真实语料文件必须带 `is_file()`/`exists()` 判断 + `synthetic_*` 回退（先例：`analyze_fwdmodel_fit.py`、`hw_forward_model_fidelity_ladder.py`，第 170 批） |
| `libs/gxipy`（被 `.gitignore gxipy*` 排除的厂商 SDK 绑定） | 涉及它的测试必须条件 skip（先例：`drivers/ccd/daheng/test_discovery.py::test_missing_env_falls_back_to_bundled_libs`） |
| 任何 `C:\` / `D:\` 机器绝对路径 | 一律 `Path(__file__).resolve().parents[N]` 派生；静态守卫 `test_script_repo_root_bootstrap.py::test_no_script_hardcodes_a_machine_absolute_path` |
| 另一测试先运行的进程纯净性 | 「clean process」类断言必须真起子进程（先例：`test_analyze_model_hw_fidelity.py`，第 170 批） |
| 开发机的 GPU / 真实设备 / 网络 | 设备路径 `@pytest.mark.hardware` 跳过；数值 golden 用「漂移 ×10」标定容差而非逐位相等（`tests/ao_shaping/test_golden_tolerance_policy.py`，第 172 批） |

## 变更本契约

改「保证存在/不存在」两个集合等于改全体测试的地基：必须同 commit 更新对应
守卫并跑红/绿负样本各一次，在 TODO.md 台账登记。
