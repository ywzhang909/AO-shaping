"""四阶段 CCD 锚定整形配方表（单一事实源，本模块不碰设备）。

为什么单独成模块
----------------
``STAGE_RECIPES`` 原本长在 ``runners/slm/transfer_shapes_runner.py`` 里，
于是任何"只想读阶段表"的消费者（GUI 的 campaign 计划页、进度读取器、
``scripts/slm/plan_multishape_campaign.py`` 一类免设备脚本）都得 import
整个 runner —— 而 runner 在 import 期就绑定了 ``click`` 与
``tools.slm.transfer.{capture,fit,validate,recording,replay_phases}``。
把表搬到 ``tools/slm/transfer`` 后，消费者只依赖一个纯 ``dataclass`` 模块。
runner 侧改成从这里 import 并保留同名模块级符号，所以
``from ao_shaping.runners.slm.transfer_shapes_runner import STAGE_RECIPES``
（``scripts/slm/generate_multishape1200_report.py:30`` 就是这么用的）仍然成立。

生成规则（逐字段照搬，未做任何改写）
-----------------------------------
四个阶段的块数 4 / 8 / 12 / 4 = **每形状 28 块**、四形状 **112 块**，与
``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md:64``
（§3.2）逐位一致。改这里的任何字段都会改变写出的 manifest/CSV 内容，
必须同时改那份表的核对。

只有第 4 段（``mono12_dense``）用宽载波模型（``wide_model=True``）与更细的
相机采样（``window_step=2.0``）；这是前三段的分辨率上限暴露出来的地方。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["STAGE_RECIPES", "StageRecipe"]


@dataclass(frozen=True)
class StageRecipe:
    """One CCD-anchored optimization stage of the shaping chain."""

    name: str
    blocks: int
    updates: int
    phase_lr: float
    phase_rms_cap: float
    bit_depth: int
    ccd_check: bool
    accept_metric: str
    window_step: float
    core_margin: float | None = None
    wide_model: bool = False
    seed_policy: str = "strict"


#: The four stages of the square1200 chain, in order. Stage 1 starts from the
#: non-GS geometric seed; stages 2-4 continue from the previous stage's
#: accepted phase. Only stage 4 uses the wide-ramp model and the finer camera
#: sampling, which is where the resolution limit of the first three shows up.
STAGE_RECIPES: tuple[StageRecipe, ...] = (
    StageRecipe(name="transport_cv", blocks=4, updates=15, phase_lr=0.03,
                phase_rms_cap=0.25, bit_depth=8, ccd_check=False, accept_metric="loss",
                window_step=4.0),
    StageRecipe(name="mono12_cv", blocks=8, updates=20, phase_lr=0.025,
                phase_rms_cap=0.20, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=4.0),
    StageRecipe(name="mono12_continue", blocks=12, updates=15, phase_lr=0.02,
                phase_rms_cap=0.15, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=4.0),
    StageRecipe(name="mono12_dense", blocks=4, updates=20, phase_lr=0.015,
                phase_rms_cap=0.12, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=2.0, core_margin=8.0, wide_model=True, seed_policy="transfer"),
)
