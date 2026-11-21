# 冻结目标与相位矩阵离线生成报告

## 摘要

已执行 `square, circle, triangle, star, rectangle, annular` × `200, 500, 1000, 1500` CCD px × `gs, model` 的 48 条计划。最终 manifest 含 30 条有效相位文件记录（24 GS + 6 model）；18 条 model 因网格不可表示而明确不生成相位。没有失败条目。本批未 import/打开硬件驱动，未占用 SLM、CCD、激光，未执行任何 `--execute`，未 git add/commit。

产物目录：`data/debug/slm_transfer/dyn_phase_matrix_20261120/`，清单：该目录的 `manifest.json`。

## 任务完成情况

- [x] 新建离线生成脚本及 7 项测试。
- [x] 复用 `create_target_shape`、`gerchberg_saxton`、裸文件 `load_frozen_parameters` 与 `FrozenDefocusForward.predict`。
- [x] 目标、掩膜、相位逐文件 SHA256 核验全部匹配。
- [x] dry-run 不创建输出目录；测试验证重复运行跳过及不可表示项不伪造相位。
- [x] 指定生成命令、pytest、Ruff 完成。

## 命令与退出码

工作目录：`D:\Projects\TIFO\AO-shaping`；生成命令使用环境 `PYTHONPATH=src;libs;libs/OOPAO;.`。

| 命令 | 最终 exit code | 结果 |
|---|---:|---|
| `python scripts/slm/target_phase_matrix_20261120.py --out data/debug/slm_transfer/dyn_phase_matrix_20261120` | 0 | 48 条计划全部完成 |
| `pytest tests/ao_shaping/scripts/test_target_phase_matrix_20261120.py -q` | 0 | 7 passed in 31.02s |
| `python -m ruff check scripts/slm/target_phase_matrix_20261120.py tests/ao_shaping/scripts/test_target_phase_matrix_20261120.py` | 0 | All checks passed |
| `graphify update .` | 0 | 图已更新；pyproject.toml zero-node warning 不影响测试 |

前两次后台 shell 分别报告 `-1` 和 `4294967295`，未作为成功验证。第二次 shell 退出后其 Python 子进程仍在运行，后显式终止本任务子进程树，再以前台同一生成命令按已有 manifest 续跑成功。最终运行日志：`C:\Users\zhangh\AppData\Local\Temp\opencode\matrix_final_20261120.log`。

## Manifest 汇总

| 状态 | 数量 |
|---|---:|
| written | 15 |
| skipped（已有且哈希完整） | 15 |
| overwritten | 0 |
| failed | 0 |
| skipped_unrepresentable | 18 |
| total | 48 |

`representable=false` 均为 **model** 方法：六种形状各自的 `500/1000/1500`，共 18 条。其目标和掩膜保留，`phase_path=null`、`sha256.phase=null`，不伪造有效相位。所有 GS 条目和 model 的六个 200 px 条目可在各自数值网格内表示；这不是硬件资格认证。

## 网格与代表性指标

| 参数 | GS | 冻结 model |
|---|---|---|
| 求解/控制网格 | 600×960 | 128×128 |
| FFT 网格 | 600×960 | 512×512 |
| 控制格间距 | 2 panel px | 15 panel px |
| 相位保存网格 | 1200×1920，复场双线性升采样 ×2 后取 angle | 128×128 |
| FFT bin | x=3.958622、y=6.333795 CCD px | 0.989655 CCD px |
| 可达半跨 | 1900.138450 CCD px | 253.351793 CCD px |
| 8 px 留白后半跨 | 1892.138450 CCD px | 245.351793 CCD px |
| 独立分辨元（较粗轴） | 6.333795 CCD px | 3.958622 CCD px |
| 居中上限跨度 | 3784.276900 CCD px | 490.703587 CCD px |

目标绘制于 CCD 原生 `1944×2592`，中心 `(1235.077880859375, 1104.6370849609375)`。CCD 像元通过 checkpoint 的 `ramp_matrix`、FFT 频率网格及 `origin_xy` 映射；不是将 CCD 图直接缩成控制网格。GS 采用整面板每两像元一点以降低 200 次迭代的 CPU 成本；1500 px 目标半跨 750 px 仍在其可达窗内。GS 200 px 矩形的短边 100 px 对应约 15.79 个较粗轴分辨元；采用最短跨度至少 8 个分辨元的显式启发式阈值，不等同于光学保真保证。

model 200 px 半跨 100 px，余量 145.351793 px；500 px 半跨 250 px 已超过可用半径 245.351793 px，1000/1500 px 更严重。每条记录含窗口外目标面积比例及原因。model FFT 4 倍零填充只增加采样密度，不增加覆盖范围或独立自由度；早期 `14.8448 px/bin` 探针未计入 15 panel px 控制间距，不采用该口径。

## 零级与使用边界

本批 `carrier="none"`、`tilt_applied=false`，目标就在零级。数值估计的 10 CCD px 半径零级核心能量占比范围：GS `0.00000329997–0.02216337410`，model `0.00023565535–0.02516634650`。它们只相对各自计算视场，不含未建模硬件零级泄漏，不能作 CCD 实测能量结论。

`slm-transfer-shapes` 是 CCD 锚定的硬件闭环，不是无需 CCD 的离线生成真值；无 CCD 的模型结果只能 `acceptance="model_only"`。本脚本 GS 标为 `simulation_only`，model 标为 `model_only` 且 `model_phase_diagnostic_only=true`。P3 checkpoint 双资格门 fail，所有 model 相位仅作诊断，任何指标均不暗示通过资格门。`initial-phase` 的 auto/transport/backprojection/gs/plane/random 是初相位种子算法，不是本批方法/优化器；本批只有 GS 与 transfer model 两条方法。`--model-dir` 预期会话布局，本脚本明确使用 `load_frozen_parameters(<bare frozen_parameters.json>)`。

## 溯源

- git HEAD：`ffff098c603bf86a1f787759dccdb8d596e53419`
- 脚本 SHA256：`1dc776b46575d288459c8402ec944e84c6551536b2ac8fc9823c9b1b4a09fdf5`
- checkpoint：`data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json`
- checkpoint SHA256：`380b9b87885fe70a4279dbe00fc500dcc3893b9422ddc84ccd0115e491c6619a`
- GS iterations：200；model iterations：150；seed：0。

工作区原有大量其他修改未处理、未回滚；图索引按项目规则已更新。
