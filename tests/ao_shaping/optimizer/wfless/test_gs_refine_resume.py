"""Stage A 分块续跑: 块边界必须续上同一条轨迹。

「一次扫描不超过 10 分钟」把 Stage A 切成若干块后, 如果块边界重置了学习率调度、
重测了基线、或者把最佳值清零, 那 250 轮就变成 6 段互不相干的优化 —— 分块的
目的 (「每组 ≤10 分钟但总强度不变」) 就落空了。

这些测试盯住三条不变量:
1. **全局 epoch 编号连续** (学习率按全局 epoch 走 ⇒ 调度不重置);
2. **基线不重测** (续跑块里没有 flat 行 —— 平场是上一块测的, 重测会引入漂移);
3. **最佳值跨块取全局最优** (第二块的 best 不低于第一块)。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_gs_refine import (
    SlmGsRefineConfig,
    optimize_slm_gs_refine,
    resume_state_from_recorder,
)

#: 数字孪生缩到的面板尺寸 (h, w)。真实面板 1200x1920 的一次远场 FFT 要 ~0.6 GB,
#: 而续跑测试只需要「能跑起来」的画面。
SMALL_PANEL = (120, 192)


@pytest.fixture
def small_sim_bench(monkeypatch):
    """把数字孪生缩到 120x192, 测试后把进程级 system 原样还原。

    进程级 system 是**全局**的: 不还原的话, 后面的测试会拿到缩小面板的孪生,
    而它们读的仍是 1200x1920 的面板形状, 于是相位形状对不上直接报错。
    """
    import ao_shaping.drivers.sim.slm_pib_sim as sim
    from ao_shaping.optimizer.wfless import slm_gs_refine as gsr

    class _SmallSystem(sim.SimPibSystem):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs.setdefault("slm_shape", SMALL_PANEL)
            kwargs.setdefault("beam_w0", 40.0)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sim, "SimPibSystem", _SmallSystem)
    monkeypatch.setattr(gsr, "panel_shape_from_slm", lambda slm: SMALL_PANEL)
    saved = sim._SYSTEM  # 进程级单例, 只能这样还原
    sim.reset_system(seed=0)
    try:
        yield
    finally:
        sim._SYSTEM = saved


def _fast_config(**overrides: Any) -> SlmGsRefineConfig:
    """A sim config small enough to run in a few seconds."""
    base = dict(
        cam_type="sim",
        epochs=4,
        phase_grid=4,
        delta=0.4,
        lr=0.05,
        seed=7,
        beam_radius_px=24.0,
        cam_size=100,
        gs_iters=8,
        n_eval_frames=1,
        settle_wait_s=0.001,
        settle_max_wait_s=0.001,
        settle_max_discard=1,
        progress_every=1,
        # 续跑状态要带平场帧, 而帧只在 record_debug 时才进历史 (扫描脚本也是这么配的)
        record_debug=True,
    )
    base.update(overrides)
    return SlmGsRefineConfig(**base)


def _resume_from(recorder: Any) -> dict[str, Any]:
    """用记录器 + 平场帧打包出下一块的起点。

    框中心用记录器冻结的**窗口内**零级位置 —— 全幅传感器坐标不能当框中心用
    (框会整个落在 ROI 窗口之外)。
    """
    flat_row = next(r for r in recorder.history if r["stage"] == "flat")
    flat_frame = np.asarray(flat_row["_img"], dtype=np.float64)
    return resume_state_from_recorder(
        recorder,
        flat_frame=flat_frame,
        center_y=int(recorder.center_y),
        center_x=int(recorder.center_x),
    )


class TestStageAResume:
    def test_resume_continues_the_global_epoch_numbering(self, small_sim_bench):
        """块边界的 epoch 编号必须连续 —— 学习率调度才不会被重置。"""
        first = optimize_slm_gs_refine(_fast_config(epochs=2))
        state = _resume_from(first)

        second = optimize_slm_gs_refine(
            _fast_config(epochs=2, chunk_epochs=2, resume_state=state)
        )

        epochs = [r["epoch"] for r in second.history if r["stage"] == "refine"]
        assert epochs == [2, 3], f"全局 epoch 编号断了: {epochs}"
        assert second.epoch_offset == 2
        assert second.epochs_total == 4

    def test_resume_does_not_re_measure_the_flat_baseline(self, small_sim_bench):
        """续跑块不该重测平场: 平场是上一块测的, 重测会引入漂移。"""
        first = optimize_slm_gs_refine(_fast_config(epochs=2))
        state = _resume_from(first)

        second = optimize_slm_gs_refine(
            _fast_config(epochs=2, chunk_epochs=2, resume_state=state)
        )

        stages = [r["stage"] for r in second.history]
        assert "flat" not in stages, f"续跑块重测了平场: {stages}"
        assert "gs" not in stages, f"续跑块重跑了 GS 热启动: {stages}"

    def test_resume_keeps_the_global_best(self, small_sim_bench):
        """第二块的最佳值不能低于第一块 —— 最佳是跨块的全局最优。"""
        first = optimize_slm_gs_refine(_fast_config(epochs=2))
        state = _resume_from(first)

        second = optimize_slm_gs_refine(
            _fast_config(epochs=2, chunk_epochs=2, resume_state=state)
        )

        assert second.best_score >= first.best_score - 1e-12

    def test_resume_state_carries_the_dof_not_a_fresh_one(self, small_sim_bench):
        """续跑的状态必须带上上一块的 DOF 与基座相位 (否则等于从头再来)。"""
        first = optimize_slm_gs_refine(_fast_config(epochs=2))
        state = _resume_from(first)

        assert state["dof"].shape == first.dof.shape
        assert state["base_panel"].shape == SMALL_PANEL
        assert state["epoch_offset"] == 2
        # 最优相位必须是**未叠加**的原始相位 (零级叠加在显示时才加)。
        assert state["best_phase"].shape == SMALL_PANEL

    def test_chunk_epochs_zero_means_use_epochs(self, small_sim_bench):
        """``chunk_epochs=0`` ⇒ 仍然跑满 ``epochs`` (默认行为不变)。"""
        rec = optimize_slm_gs_refine(_fast_config(epochs=2, chunk_epochs=0))
        assert rec.epochs_executed == 2
        assert rec.epochs_total == 2

    def test_resume_state_records_the_full_frame_centre(self, small_sim_bench):
        """续跑状态要带上**全幅**零级位置 —— 下一块据此摆同一个 ROI 窗口。

        仿真路径没有全幅发现 (没有裁剪窗口那一步), 所以它是 ``None`` ⇒ 下一块
        照旧重新定位。
        """
        first = optimize_slm_gs_refine(_fast_config(epochs=1))
        state = _resume_from(first)

        assert first.full_frame_center is None
        assert state["full_frame_center"] is None

    def test_out_of_frame_mask_centre_names_both_coordinate_frames(
        self, small_sim_bench
    ):
        """框中心落在窗口外 ⇒ 报错必须点名两个坐标系。

        2026-10-21 实测: 分块续跑两秒就挂在 "target square (side=12px at
        (973,1434)) is empty" —— 报错只说「框是空的」, 完全看不出是**全幅**
        传感器坐标被当成了**窗口内**坐标。所以这里锁定: 报错里必须同时出现两个
        坐标系的说法与画幅尺寸。
        """
        first = optimize_slm_gs_refine(_fast_config(epochs=1))
        state = _resume_from(first)
        # 换成**全幅**传感器坐标: 硬件上就是这样混用的 (973, 1434) 落在
        # 256x256 的 ROI 窗口之外。
        state["center_y"] = 973
        state["center_x"] = 1434

        with pytest.raises(ValueError) as excinfo:
            optimize_slm_gs_refine(
                _fast_config(epochs=1, chunk_epochs=1, resume_state=state)
            )

        message = str(excinfo.value)
        assert "WINDOW-LOCAL" in message
        assert "FULL-FRAME" in message
        height, width = state["flat_frame"].shape
        assert f"{height}x{width}" in message
