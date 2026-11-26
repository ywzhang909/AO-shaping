"""``--phase-source transfer``: 被测相位接入最新 transfer 整形终稿。

`run_gs_refine_zernike_sweep` 的 Stage A 在 transfer 来源下**整段跳过**: 被测
相位就是 ``scripts/slm/_transfer_phase.py`` 读到的终稿面板相位 (1200x1920
弧度)。这里覆盖三件事:

* CLI: ``--phase-source`` / ``--transfer-phase`` 的解析与默认值;
* 纯函数: :func:`stage_a_from_transfer` 的交接内容 (相位逐位一致、基线帧为
  ``None``、来源标记与几何说明齐全);
* sim 小规模端到端 (无硬件): ``--stage-a-only`` 落盘的 stage_a.npz 键与元数据、
  ``--stage-a-in`` 复用交接文件跑 Stage B/C 时基线记 ``null``、以及
  「transfer 交接文件不能分块续跑 GS」的守卫。

不打开任何真实设备; 全部走 ``--cam-type sim`` 或纯函数。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.slm._transfer_phase import load_transfer_phase
from scripts.slm.run_gs_refine_zernike_sweep import (
    SweepParams,
    _build_parser,
    describe_transfer_geometry,
    load_stage_a,
    main as sweep_main,
    stage_a_from_transfer,
)

#: 数字孪生在测试里缩到的面板尺寸 (h, w) —— 与 sweep 测试同款缩法。
SIM_PANEL = (120, 192)

#: 测试里的瞳面半径 (面板像素)。transfer 相位本身仍是 640x640 模型网格。
SIM_BEAM_RADIUS = 16.0

SIM_ARGS = [
    "--cam-type",
    "sim",
    "--cam-size",
    "256",
    "--beam-radius-px",
    str(SIM_BEAM_RADIUS),
    "--sim-far-field-padding",
    "4",
]

#: 仿制的 transfer 模型网格 (与真实终稿同为 640x640, 内容只要有限即可)。
MODEL_GRID = 640


@pytest.fixture
def sim_bench(monkeypatch):
    """把数字孪生缩到测试能跑的规模 (无硬件), 与 sweep 测试同一套改绑。"""
    import ao_shaping.drivers.sim.sim_bench_patch as bench_patch
    import ao_shaping.drivers.sim.slm_pib_sim as sim
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt
    from ao_shaping.optimizer.wfless import slm_gs_refine as gsr

    class _SmallSystem(sim.SimPibSystem):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("slm_shape", SIM_PANEL)
            kwargs.setdefault("beam_w0", 40.0)
            super().__init__(*args, **kwargs)

    class _SmallSLM(sim.SimSLMPib):
        Panel_Res = (SIM_PANEL[1], SIM_PANEL[0])

    monkeypatch.setattr(sim, "SimPibSystem", _SmallSystem)
    monkeypatch.setattr(sim, "SimSLMPib", _SmallSLM)
    monkeypatch.setattr(bench_patch, "SimSLMPib", _SmallSLM)
    monkeypatch.setattr(pib_opt, "SLM_RESOLUTION", (SIM_PANEL[1], SIM_PANEL[0]))
    monkeypatch.setattr(gsr, "panel_shape_from_slm", lambda slm: SIM_PANEL)
    monkeypatch.setattr(pib_opt, "Santec", pib_opt.Santec)
    monkeypatch.setattr(pib_opt, "create_camera", pib_opt.create_camera)

    original = gsr.display_and_average

    def _fast_measure(cam, slm, phase_rad, *, n_frames, **kwargs):
        kwargs["settle_wait_s"] = 0.0
        kwargs["settle_max_wait_s"] = 0.0
        return original(cam, slm, phase_rad, n_frames=1, **kwargs)

    monkeypatch.setattr(gsr, "display_and_average", _fast_measure)

    saved_system = sim._SYSTEM
    try:
        yield
    finally:
        sim._SYSTEM = saved_system


@pytest.fixture
def transfer_npy(tmp_path) -> str:
    """一份临时的 640x640 transfer 模型相位 (.npy), 内容确定性。"""
    rng = np.random.default_rng(7)
    phase = rng.uniform(-np.pi, np.pi, size=(MODEL_GRID, MODEL_GRID))
    path = tmp_path / "transfer_best.npy"
    np.save(path, phase.astype(np.float64))
    return str(path)


class TestPhaseSourceCLI:
    """``--phase-source`` / ``--transfer-phase`` 的解析。"""

    def test_default_is_transfer(self):
        args = _build_parser().parse_args([])
        assert args.phase_source == "transfer"
        assert args.transfer_phase is None

    def test_gs_choice_is_kept(self):
        args = _build_parser().parse_args(["--phase-source", "gs"])
        assert args.phase_source == "gs"

    def test_unknown_source_is_rejected(self):
        with pytest.raises(SystemExit) as excinfo:
            _build_parser().parse_args(["--phase-source", "hologram"])
        assert excinfo.value.code == 2

    def test_transfer_phase_is_parsed(self):
        args = _build_parser().parse_args(
            ["--transfer-phase", "data/debug/whatever/star_best.npy"]
        )
        assert args.transfer_phase == "data/debug/whatever/star_best.npy"

    def test_sweep_params_defaults_match_the_cli(self):
        params = SweepParams()
        assert params.phase_source == "transfer"
        assert params.transfer_phase == ""


class TestTransferGeometry:
    """几何说明: 模型网格 → 面板平铺 → 截断窗。"""

    def test_geometry_mentions_grid_pitch_and_window(self):
        text = describe_transfer_geometry((640, 640), SIM_BEAM_RADIUS)
        assert "640x640" in text
        assert "pitch=3px" in text
        # 1200 行 / pitch 3 ⇒ 模型只用前 400 行; 截断窗按 pupil_side 给出。
        assert "前 400 行" in text
        assert "pupil_side(16)=32px" in text


class TestStageAFromTransfer:
    """纯函数交接: 相位逐位一致、没有伪造的基线。"""

    @staticmethod
    def _result(transfer_npy):
        return stage_a_from_transfer(
            transfer_phase=transfer_npy,
            center_xy=(1434, 973),
            w0=4.5,
            side=12.0,
            probe_shape=(256, 256),
            beam_radius_px=SIM_BEAM_RADIUS,
            config={"phase_source": "transfer", "transfer_phase": transfer_npy},
        )

    def test_best_phase_is_the_transfer_panel_bit_for_bit(self, transfer_npy):
        result = self._result(transfer_npy)
        _, _, panel = load_transfer_phase(transfer_npy)
        assert result.best_phase.shape == (1200, 1920)
        np.testing.assert_array_equal(result.best_phase, panel)

    def test_baselines_are_none_and_scores_are_not_fabricated(self, transfer_npy):
        result = self._result(transfer_npy)
        assert result.flat_frame is None
        assert result.best_frame is None
        assert result.best_score is None
        assert result.epochs_executed == 0
        assert result.epochs_total == 0
        # 续跑三件也不存在: transfer 没有 DOF / 基座相位。
        assert result.dof is None
        assert result.base_panel is None
        assert result.best_phase_plain is None

    def test_provenance_fields_carry_the_real_path(self, transfer_npy):
        result = self._result(transfer_npy)
        assert result.phase_source == "transfer"
        assert result.source_path == transfer_npy
        assert result.best_stage == "transfer"
        assert "640x640" in result.geometry_source
        # 几何仍来自平场探测 (Stage C 摆窗口要用), 原样保留。
        assert result.center_xy == (1434, 973)
        assert result.w0 == pytest.approx(4.5)
        assert result.side == pytest.approx(12.0)
        assert result.probe_shape == (256, 256)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            stage_a_from_transfer(
                transfer_phase=str(tmp_path / "nope.npy"),
                center_xy=(0, 0),
                w0=1.0,
                side=8.0,
                probe_shape=(256, 256),
                beam_radius_px=SIM_BEAM_RADIUS,
            )

    def test_none_path_uses_the_default_phase(self):
        # None → DEFAULT_TRANSFER_PHASE (仓库里在盘的最新终稿)。
        from pathlib import Path as _P

        from scripts.slm._transfer_phase import DEFAULT_TRANSFER_PHASE as _DEF

        _root = _P(__file__).resolve().parents[4]
        if not (_root / _DEF).is_file():
            pytest.skip(
                f"冻结 transfer 产物不在本机 (data/debug 硬件数据未入库): {_DEF}"
            )
        result = stage_a_from_transfer(
            transfer_phase=None,
            center_xy=(0, 0),
            w0=1.0,
            side=8.0,
            probe_shape=(256, 256),
            beam_radius_px=SIM_BEAM_RADIUS,
        )
        assert result.phase_source == "transfer"
        assert result.source_path  # 真实路径非空
        assert result.best_phase.shape == (1200, 1920)


class TestTransferStageAOnly:
    """``--stage-a-only --phase-source transfer`` 落盘可复用的交接文件。"""

    @staticmethod
    def _run(tmp_path, transfer_npy, *extra) -> str:
        code = sweep_main(
            [
                *SIM_ARGS,
                "--phase-source",
                "transfer",
                "--transfer-phase",
                str(transfer_npy),
                "--target-side",
                "24",
                "--stage-a-only",
                "--out",
                str(tmp_path),
                "--group",
                "transfer_stage_a",
                *extra,
            ]
        )
        assert code == 0
        return tmp_path / "transfer_stage_a"

    def test_npz_carries_phase_and_provenance_only(
        self, tmp_path, transfer_npy, sim_bench
    ):
        group_dir = self._run(tmp_path, transfer_npy)

        with np.load(group_dir / "stage_a.npz") as data:
            files = set(data.files)
            # 键名沿用 best_phase; 来源元数据进 npz。
            assert "best_phase" in files
            assert "phase_source" in files
            assert "source_path" in files
            # 没有采集基线 ⇒ 干脆不写这两个键 (缺键 = None)。
            assert "flat_frame" not in files
            assert "best_frame" not in files
            # best_phase 是 640x640 模型展开后的面板相位。
            assert data["best_phase"].shape == (1200, 1920)
            _, _, panel = load_transfer_phase(transfer_npy)
            np.testing.assert_array_equal(data["best_phase"], panel)
            assert str(data["phase_source"]) == "transfer"
            assert str(data["source_path"]) == str(transfer_npy)

    def test_json_and_summary_record_the_source_honestly(
        self, tmp_path, transfer_npy, sim_bench
    ):
        group_dir = self._run(tmp_path, transfer_npy)

        payload = json.loads(
            (group_dir / "stage_a.json").read_text(encoding="utf-8")
        )
        assert payload["phase_source"] == "transfer"
        assert payload["source_path"] == str(transfer_npy)
        # 没跑 GS ⇒ 没有评分, 如实记 null, 不写 0 之类的伪值。
        assert payload["best_score"] is None
        assert payload["best_stage"] == "transfer"
        assert "geometry_source" in payload

        summary = json.loads(
            (group_dir / "summary.json").read_text(encoding="utf-8")
        )
        config = summary["config"]
        assert config["phase_source"] == "transfer"
        assert config["transfer_phase"] == str(transfer_npy)
        assert "640x640" in config["geometry_source"]
        assert summary["stage_a"]["best_score"] is None
        assert summary["orders"] == []
        assert "transfer" in summary["baselines_note"]
        # 没跑 GS 这件事也要写进分组清单。
        manifest = json.loads(
            (tmp_path / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest[0]["meta"]["phase_source"] == "transfer"

    def test_missing_transfer_phase_is_an_argument_error(
        self, tmp_path, sim_bench
    ):
        code = sweep_main(
            [
                *SIM_ARGS,
                "--phase-source",
                "transfer",
                "--transfer-phase",
                str(tmp_path / "nope.npy"),
                "--target-side",
                "24",
                "--stage-a-only",
                "--out",
                str(tmp_path),
                "--group",
                "bad",
            ]
        )
        assert code == 2


class TestTransferStageAIn:
    """``--stage-a-in`` 复用 transfer 交接文件跑 Stage B/C。"""

    @staticmethod
    def _handover(tmp_path, transfer_npy, sim_bench) -> str:
        code = sweep_main(
            [
                *SIM_ARGS,
                "--phase-source",
                "transfer",
                "--transfer-phase",
                str(transfer_npy),
                "--target-side",
                "24",
                "--stage-a-only",
                "--out",
                str(tmp_path),
                "--group",
                "transfer_stage_a",
            ]
        )
        assert code == 0
        return str(tmp_path / "transfer_stage_a" / "stage_a.npz")

    def test_loaded_handover_keeps_the_transfer_phase(self, tmp_path, transfer_npy, sim_bench):
        npz_path = self._handover(tmp_path, transfer_npy, sim_bench)
        result = load_stage_a(npz_path)
        assert result.phase_source == "transfer"
        assert result.source_path == str(transfer_npy)
        assert result.flat_frame is None
        assert result.best_frame is None
        assert result.best_score is None
        _, _, panel = load_transfer_phase(transfer_npy)
        np.testing.assert_array_equal(result.best_phase, panel)

    def test_stage_b_c_runs_with_null_baselines(self, tmp_path, transfer_npy, sim_bench):
        npz_path = self._handover(tmp_path, transfer_npy, sim_bench)
        order_dir = tmp_path / "transfer_order_nmax3"
        code = sweep_main(
            [
                *SIM_ARGS,
                "--stage-a-in",
                npz_path,
                "--spgd-epochs",
                "1",
                "--orders",
                "3",
                "--out",
                str(tmp_path),
                "--group",
                "transfer_order_nmax3",
            ]
        )
        assert code == 0

        summary = json.loads(
            (order_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert [entry["n_max"] for entry in summary["orders"]] == [3]
        # 基线帧没有采集 ⇒ null + 明确标注, 不是伪帧也不是缺键。
        assert summary["baselines"]["flat"] is None
        assert summary["baselines"]["gs_freeform"] is None
        assert "transfer" in summary["baselines_note"]
        assert summary["phase_source"] == "transfer"
        assert summary["transfer_phase"] == str(transfer_npy)
        assert summary["stage_a"]["best_score"] is None

    def test_transfer_handover_cannot_resume_a_gs_chunk(
        self, tmp_path, transfer_npy, sim_bench
    ):
        npz_path = self._handover(tmp_path, transfer_npy, sim_bench)
        with pytest.raises(SystemExit) as excinfo:
            sweep_main(
                [
                    *SIM_ARGS,
                    "--epochs",
                    "2",
                    "--chunk-epochs",
                    "1",
                    "--stage-a-only",
                    "--stage-a-in",
                    npz_path,
                    "--out",
                    str(tmp_path),
                    "--group",
                    "wont_resume",
                ]
            )
        assert "参数错误" in str(excinfo.value)
