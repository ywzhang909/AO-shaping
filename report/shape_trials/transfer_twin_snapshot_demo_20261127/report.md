# 传输模型三层防泄漏快照与四分支预训练 —— 数字孪生首验 (#254 T3/T4)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/run_transfer_twin_snapshot_demo.py`](../../../scripts/slm/run_transfer_twin_snapshot_demo.py)
> **复现命令**: `python scripts/slm/run_transfer_twin_snapshot_demo.py`
> **运行环境**: 离线（数字孪生语料 generate_dataset, 无需 SLM / CCD / 激光）
> **说明**: #254 T3/T4 数字孪生首验: v1-twin 快照 (calibration 30 / train 18 / validation 11 / final_test 1, root_hash 60dac86018f1d2fa…) → 四分支预注册 spec (基线/+PSF/+几何/+两者, 孪生预演非正式预注册) 统一预算训练 → selection 打分 winner B_psf (score 0.0772, C_geometry 与基线并列 0.0866 无增益, 与 11-24 几何证伪同向) → final_test 一次性打分 (corr 0.9997); 机制与产物链验证, 非硬件结论; 真实 v1 在采集侧同一命令执行
<!-- provenance:end -->

- 生成时间 (UTC): 2026-11-27T14:19:37+00:00
- 快照: `v1-twin`, root_hash `60dac86018f1d2fa…`, 层计数 `{'calibration': 30, 'train': 18, 'validation': 11, 'final_test': 1}`, 封存层 `['final_test']`
- 预算 (全分支共享): iterations=150, span=120, device=cpu; 分支只可覆盖模型旋钮白名单 (spec `budget`/`branches` 分离)
- 候选冻结: winner **B_psf**, spec_sha256 `91efb42297d227dd…`; 冻结时刻 `final_test=None` (契约)
- 封存层一次性打分: correlation=0.9997, relative_l2=0.1214 (审计行已入 final_test_audit.jsonl, 二次打分会被拒)

## 分支 selection 层打分 (唯一晋级依据)

| 分支 | 覆盖 | score | correlation | relative_l2 |
|---|---|---|---|---|
| B_psf | `{'adaptation': 'alignment', 'learn_psf': True}` | 0.0772 | 0.9996 | 0.0767 |
| D_psf_geometry | `{'adaptation': 'alignment', 'learn_psf': True, 'learn_geometry': True}` | 0.0777 | 0.9995 | 0.0772 |
| A_baseline | `{'adaptation': 'alignment'}` | 0.0866 | 0.9989 | 0.0855 |
| C_geometry | `{'adaptation': 'alignment', 'learn_geometry': True}` | 0.0866 | 0.9989 | 0.0855 |

## 口径与边界

- 语料 = `transfer.simulation.generate_dataset` 已知真值孪生 (cap_a/b/c, cap_c 末行改标 final_test), **不是硬件数据**; 本实验验证的是 T3/T4 工具链端到端 (冻结可复核、预算结构性唯一、封存一次性打分、并行同结论), 不构成任何模型质量或效率结论;
- 四分支是 2026-11-24 两项适配 (PSF 通过 opt-in / 几何证伪) 的自然重构, 仅用于孪生预演; 正式四分支集合与打分口径以采集侧 plan.md 为准;
- 真实 v1 冻结与正式训练在持有 `data/debug/slm_transfer` 语料的采集侧执行: `python -m ao_shaping.tools.slm.transfer.snapshot freeze --root data/debug/slm_transfer --id v1 --out data/snapshots [--plan-only]` → `python -m ao_shaping.tools.slm.transfer.branch_fit freeze --spec ... --manifest ...`;
- 训练工作目录 (inputs.npz / iterations.pkl / 各分支 fit 产物) 留在 `data/twin_snapshot_demo/run/`, 不入库; 入库 JSON 仅 spec / 快照清单 / candidates_frozen / final_test_result / audit。
