"""Step B 迭代预算扫描（纯离线仿真，不碰硬件）。

目的
----
GUI「物理预测模型闭环」(MIL) 的 Step B 默认跑 600 步 Adam（可微整形迭代）。
本脚本固定其余一切（轮数 / Step A 步数 / 像差 / 种子 / 目标），只扫
``step_b.iterations``，回答两个问题：

1. **收敛拐点**：Step B 的加权损失在第几步基本平台化？
2. **省多少**：把预算砍到拐点前，省下多少秒，质量损失多少？

物理指标用仿真的远场方斑指标（CV / efficiency），不是模型 loss ——
loss 降了不等于光斑变好。

用法::

    .venv/Scripts/python.exe data/debug/stepb_budget_sweep_20261116/sweep_stepb_budget.py
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    ModelInLoopConfig,
    SimulationEnvParams,
    StepAConfig,
    StepBConfig,
    simulate_iterative_shaping,
)

OUT = Path(__file__).resolve().parent

#: 扫描的 Step B 预算。600 = GUI「标准 6/80/600」预设的现行值。
BUDGETS = (100, 200, 300, 400, 600)

#: 与 GUI 标准预设一致的外围条件。
#:
#: Step A 用**类默认**（500 步 / 8 探针），不是 GUI 预设的 80 步：80 步时像差
#: 拟合发散（实测 coeff error 2–3.5 rad、Step A loss 不降），整条对比会被
#: 发散的基线污染，那样的数字不能用来定 Step B 的预算。
N_ROUNDS = 6
STEP_A_ITERS = 500

#: 固定像差（Zernike Noll 号 → 弧度）与种子：跨预算必须完全一致，
#: 否则比较的不是「同样的问题解得更浅」而是「换了一道题」。
ABERRATIONS = {4: 0.35, 5: 0.30, 8: 0.20}
SEED = 0


def main() -> int:
    base_env = SimulationEnvParams(region=64, noise_enabled=False, n_frames=1, seed=SEED)
    rows: list[dict[str, object]] = []

    for budget in BUDGETS:
        config = ModelInLoopConfig(
            env=base_env,
            step_a=StepAConfig(iterations=STEP_A_ITERS),
            step_b=StepBConfig(iterations=budget),
            aberrations=dict(ABERRATIONS),
            n_rounds=N_ROUNDS,
            seed=SEED,
        )
        started = time.perf_counter()
        result = simulate_iterative_shaping(config)
        elapsed = time.perf_counter() - started

        last = result.rounds[-1]
        rows.append(
            {
                "step_b_budget": budget,
                "wall_s": round(elapsed, 3),
                "step_b_loss_initial": last.step_b_loss_initial,
                "step_b_loss_final": last.step_b_loss_final,
                "loss_ratio": round(last.step_b_loss_final / last.step_b_loss_initial, 6),
                "metrics_after": last.metrics_after,
                "metrics_before": last.metrics_before,
            }
        )
        print(
            f"B={budget:4d} wall={elapsed:7.2f}s "
            f"loss {last.step_b_loss_initial:.4e} -> {last.step_b_loss_final:.4e} "
            f"({last.step_b_loss_final / last.step_b_loss_initial:.4f}x) "
            f"after={last.metrics_after}",
            flush=True,
        )

    payload = {
        "n_rounds": N_ROUNDS,
        "step_a_iterations": STEP_A_ITERS,
        "aberrations": ABERRATIONS,
        "seed": SEED,
        "budgets": list(BUDGETS),
        "rows": rows,
    }
    (OUT / "stepb_budget_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUT / 'stepb_budget_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
