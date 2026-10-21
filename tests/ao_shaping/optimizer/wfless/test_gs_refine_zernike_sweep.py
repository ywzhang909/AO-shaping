"""Tests for the GS-refine Zernike-order sweep driver.

The driver runs ONE GS-refine optimisation, fits its best phase into Zernike
coefficients at several truncation orders, then re-optimises each coefficient
vector with SPGD and re-scores every run on a single frozen square mask.

These tests exercise only the pure helpers (target-box resolution, sweep plan,
stage-C config assembly, clip warning, offline scoring) - never the CLI or any
hardware. The orchestration entry point is a plain callable so it can be driven
with tiny injected parameters; ``main`` is a thin argparse shell around it.

The segmented-sweep tests (``--stage-a-only`` / ``--stage-a-in`` / ``--orders``)
do drive ``main`` end to end, but only against the **sim** backend at a
deliberately tiny scale.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from scripts._common.group_manifest import (
    group_record,
    load_manifest,
    parse_group_meta,
    update_manifest,
)
from scripts.slm.run_gs_refine_zernike_sweep import (
    CAM_WINDOW_TARGET_MARGIN,
    DEFAULT_SIM_FAR_FIELD_PADDING,
    GROUP_VARIABLE,
    MIN_TARGET_SIDE,
    ZERNIKE_CLIP,
    _build_parser,
    _metrics_identical,
    build_stage_c_config,
    check_zernike_clip,
    group_fields,
    group_output_dir,
    load_stage_a,
    main as sweep_main,
    parse_orders,
    probe_flat_field,
    resolve_orders,
    resolve_target_side,
    save_stage_a,
    score_frame,
    sweep_plan,
)


class TestTargetSide:
    """The shared target box must fit the camera window for every order."""

    def test_shared_target_side_fits_camera_window(self):
        side = resolve_target_side(w0=3.0, cam_size=256)
        assert 4.0 <= side <= 256 / CAM_WINDOW_TARGET_MARGIN

        # Tiny window: the clamp holds and the box still fits inside the frame.
        side_tiny = resolve_target_side(w0=3.0, cam_size=6)
        assert side_tiny < 6
        assert side_tiny <= 6 / CAM_WINDOW_TARGET_MARGIN + 1e-9

    def test_rejects_degenerate_target_box(self):
        with pytest.raises(ValueError):
            resolve_target_side(w0=3.0, cam_size=256, override=300.0)
        with pytest.raises(ValueError):
            resolve_target_side(w0=3.0, cam_size=256, override=256.0)


class TestSweepPlan:
    def test_sweep_plan_orders(self):
        plan = sweep_plan([4, 6, 8])
        assert [p["n_max"] for p in plan] == [4, 6, 8]
        assert [p["n_modes"] for p in plan] == [
            calc_n_zernike_terms(n) for n in (4, 6, 8)
        ]


class TestStageCConfig:
    def test_seed_coefficients_match_zernike_radius(self):
        coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cfg = build_stage_c_config(
            center=(128, 128),
            coefficients=coeffs,
            n_max=4,
            side=40.0,
            beam_radius_px=450.0,
            cam_type="sim",
            cam_id=0,
            exposure_time_ms=1.0,
            cam_size=256,
            slm_number=1,
            epochs=4,
            delta=0.2,
            lr=0.0,
            seed=0,
        )
        assert cfg.slm.zernike_radius == 450.0
        assert np.array_equal(cfg.slm.init_c, coeffs)
        assert cfg.camera.target_size == 40.0
        assert cfg.camera.name == "rms_pib"
        assert cfg.camera.target_shape == "square"
        assert cfg.center == (128, 128)
        assert cfg.epochs == 4

    def test_energy_guard_passthrough(self):
        cfg = build_stage_c_config(
            center=(128, 128),
            coefficients=np.zeros(5),
            n_max=4,
            side=40.0,
            beam_radius_px=450.0,
            cam_type="sim",
            cam_id=0,
            exposure_time_ms=1.0,
            cam_size=256,
            slm_number=1,
            epochs=4,
            delta=0.02,
            lr=0.01,
            seed=0,
            max_energy_loss=0.0,
        )
        assert cfg.camera.max_roi_energy_loss == 0.0


class TestClipWarning:
    def test_zernike_clip_warning(self):
        assert check_zernike_clip(1.0) is None
        warn = check_zernike_clip(ZERNIKE_CLIP + 1.0)
        assert warn is not None
        assert "clip" in warn.lower() or str(ZERNIKE_CLIP) in warn


class TestScoreFrame:
    def test_score_frame_reports_all_metrics(self):
        frame = np.zeros((64, 64), dtype=np.float64)
        frame[28:36, 28:36] = 100.0
        mask = np.zeros((64, 64), dtype=bool)
        mask[28:36, 28:36] = True

        scores = score_frame(frame, mask, center_xy=(32, 32), side=8)

        assert set(scores) >= {"pib", "cv", "ee", "ar", "quality"}
        assert all(np.isfinite(v) for v in scores.values())
        assert scores["pib"] > 0.9
        assert scores["ee"] > 0.9


class TestSimBenchGeometry:
    """The default sim bench must yield a resolvable target box, not the floor.

    The far-field spot size in pixels is ``N * padding / (pi * beam_w0)``, so it
    is invariant under a proportional scaling of the panel and the beam. These
    tests shrink both by 10x to keep the FFT cheap while exercising the exact
    default padding and the real probe path.
    """

    @staticmethod
    def _shrink_sim_bench(monkeypatch):
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        class _SmallSystem(sim.SimPibSystem):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("slm_shape", (120, 192))
                kwargs.setdefault("beam_w0", 40.0)
                super().__init__(*args, **kwargs)

        class _SmallSLM(sim.SimSLMPib):
            # 孪生的面板尺寸必须**两处**一致: system 收相位, SLM 报面板形状。
            # 只缩 system 的话, 平场相位会按 1200x1920 写进 120x192 的孪生里,
            # 形状对不上直接报错。
            Panel_Res = (192, 120)

        monkeypatch.setattr(sim, "SimPibSystem", _SmallSystem)
        monkeypatch.setattr(sim, "SimSLMPib", _SmallSLM)

    def test_default_sim_config_target_side_above_floor(self, monkeypatch):
        self._shrink_sim_bench(monkeypatch)
        _center, w0, shape = probe_flat_field(
            "sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256
        )
        side = resolve_target_side(w0, 256)
        assert shape == (256, 256)
        # The box is ``2 * w0``; it must clear the floor rather than clamp to it.
        assert w0 > MIN_TARGET_SIDE / 2
        assert side > MIN_TARGET_SIDE

    def test_probe_configures_sim_far_field_padding(self, monkeypatch):
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        self._shrink_sim_bench(monkeypatch)
        probe_flat_field("sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256)
        assert sim.get_system().far_field_padding == DEFAULT_SIM_FAR_FIELD_PADDING

    def test_probe_flat_field_passes_cam_size(self, monkeypatch):
        import ao_shaping.drivers.ccd.common as ccd_common

        captured: dict = {}

        class _FakeCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_numpy_image(self, *args, **kwargs):
                yy, xx = np.mgrid[0:256, 0:256]
                return np.exp(-((xx - 128.0) ** 2 + (yy - 128.0) ** 2) / (2 * 8.0**2))

        def fake_create_camera(camera_type, cam_id=0, exposure_time_ms=20.0, **kwargs):
            captured["camera_type"] = camera_type
            captured["kwargs"] = kwargs
            return _FakeCam()

        monkeypatch.setattr(ccd_common, "create_camera", fake_create_camera)
        probe_flat_field("sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256)
        assert captured["kwargs"].get("cam_size") == 256


class TestStageBMonotonicity:
    """The Stage-B fit residual must be non-increasing in ``n_max``.

    This is the invariant the order scan (E3) rests on: the Zernike truncation
    is a *nested* least-squares basis, so widening it can never increase the
    residual. A sweep whose ``fit_rms_rad`` ever rises with order is reporting
    a bug (wrong radius, wrong panel crop, non-nested basis), not a finding.

    The scan is what ranks orders on hardware, so the invariant has to hold
    before any ranking is believed.
    """

    @staticmethod
    def _non_flat_panel(radius_px: float = 450.0) -> np.ndarray:
        """A smooth, non-flat pupil phase on the driver's real panel size."""
        from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

        return generate_zernike_phase(
            {1: 0.4, 4: 0.25, 6: -0.15},
            resolution=(960, 1200),
            n_max=8,
            radius=radius_px,
        )

    def test_fit_rms_non_increasing_in_n_max(self):
        from ao_shaping.optimizer.wfless.gs_refine_zernike_fit import fit_best_phase

        phase = self._non_flat_panel()
        residuals = [
            fit_best_phase(phase, n_max, 450.0).rms_residual_rad
            for n_max in (2, 3, 4, 6, 8)
        ]
        # Nested basis => monotone non-increasing (tolerate float noise only).
        for lo, hi in zip(residuals, residuals[1:]):
            assert hi <= lo + 1e-12, f"residual rose with order: {residuals}"

    def test_sweep_records_one_residual_per_order(self):
        """The driver writes exactly one ``fit_rms_rad`` per swept order."""
        from ao_shaping.optimizer.wfless.gs_refine_zernike_fit import fit_best_phase

        orders = sweep_plan([4, 6, 8])
        phase = self._non_flat_panel()
        residuals = [
            (p["n_max"], fit_best_phase(phase, p["n_max"], 450.0).rms_residual_rad)
            for p in orders
        ]
        assert [n for n, _ in residuals] == [4, 6, 8]
        assert all(isinstance(r, float) and r >= 0.0 for _, r in residuals)


class TestGroupSurface:
    """一次调用只探一组变量 (这一组的自变量 = ``n_max``)。

    The sweep used to write its artefacts straight into ``--out``, so two groups
    of the same experiment could not live side by side. It now writes into
    ``<out>/<group>/`` and keeps a ``manifest.json`` at the root, and the group's
    identity travels into ``summary.json`` too.
    """

    def test_group_and_meta_are_parsed(self):
        args = _build_parser().parse_args(
            [
                "--group",
                "nmax_scan",
                "--group-meta",
                "exposure_ms=0.2",
                "--group-meta",
                "phase=hardware_gs",
            ]
        )
        assert args.group == "nmax_scan"
        assert parse_group_meta(args.group_meta_pairs) == {
            "exposure_ms": "0.2",
            "phase": "hardware_gs",
        }

    def test_group_defaults_to_main(self):
        assert _build_parser().parse_args([]).group == "main"

    def test_group_fields_land_in_the_summary(self):
        fields = group_fields("nmax_scan", {"exposure_ms": "0.2"})
        assert fields["group"] == "nmax_scan"
        assert fields["variable"] == GROUP_VARIABLE == "n_max"
        assert fields["group_meta"] == {"exposure_ms": "0.2"}

    def test_group_output_path_is_out_plus_group(self, tmp_path):
        assert group_output_dir(tmp_path, "nmax_scan") == tmp_path / "nmax_scan"

    def test_manifest_indexes_the_group(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()
        update_manifest(
            root,
            group_record(
                group="nmax_scan",
                variable=GROUP_VARIABLE,
                meta={"exposure_ms": "0.2"},
                n_rows=5,
            ),
        )
        entries = load_manifest(root)
        assert len(entries) == 1
        assert entries[0]["group"] == "nmax_scan"
        assert entries[0]["variable"] == "n_max"
        assert entries[0]["dir"] == "nmax_scan"
        assert entries[0]["n_rows"] == 5


class TestSpgdImprovedIsAlwaysRecorded:
    """``spgd_improved`` must be a real decision, never a missing key.

    ``orders.append`` always writes ``bool(spgd_improved)``, so a Stage-C run
    that never moved records ``false`` rather than omitting the field. A reader
    (the report generator included) can therefore treat ``false`` as "measured,
    did not improve" and never as "not measured".
    """

    def test_identical_metrics_mean_no_improvement(self):
        base = {"cv": 0.5, "pib": 0.4, "ee": 0.9}
        assert _metrics_identical(base, dict(base)) is True

    def test_any_changed_metric_breaks_equality(self):
        before = {"cv": 0.5, "pib": 0.4, "ee": 0.9}
        after = {"cv": 0.5, "pib": 0.4 + 1e-6, "ee": 0.9}
        assert _metrics_identical(before, after) is False

    def test_sub_tolerance_drift_still_counts_as_identical(self):
        before = {"cv": 0.5}
        after = {"cv": 0.5 + 1e-12}
        assert _metrics_identical(before, after) is True


# ---------------------------------------------------------------------------
# 分段扫描 (--stage-a-only / --stage-a-in / --orders)
# ---------------------------------------------------------------------------

#: 数字孪生在测试里缩到的面板尺寸 (h, w)。真实面板 1200x1920 配 padding 16 的
#: 远场 FFT 一次要 9.4 GB, 所以测试必须缩小。
SIM_PANEL = (120, 192)

#: 一组缩到测试规模的命令行参数 (公共部分)。
#:
#: ``--sim-far-field-padding 4`` 是**故意**取小的: 补零倍数直接决定远场 FFT 的
#: 规模 (padding 16 ⇒ 1920x3072 的复数缓冲 ≈ 47 MB/次), 而这些测试要的只是
#: 「能跑通一遍流程」, 不是分辨得开的远场光斑。
SIM_BEAM_RADIUS = 16.0
"""测试里用的瞳面半径 (面板像素); 落盘的拟合相位就是按这个半径还原的。"""

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


@pytest.fixture
def sim_bench(monkeypatch):
    """把数字孪生缩到测试能跑的规模 (无硬件)。

    缩面板、去掉稳定等待 (仿真面板没有弛豫时间可言, 每次测量白等 0.5 s),
    并把 ``run_sweep`` 里的 sim 改绑还原掉 —— 否则后面的测试会拿到缩小面板
    的仿真 SLM。

    🔴 进程级 system 也必须还原: ``probe_flat_field`` 会 ``reset_system`` 建出
    一个缩小面板的孪生, 它是**全局**的, 不还原的话后面按 1200x1920 读面板形状的
    测试会直接撞上相位形状不符。
    """
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
    # ``run_sweep`` 会把这两个名字永久改绑到仿真替身上; 登记原值以便还原。
    monkeypatch.setattr(pib_opt, "Santec", pib_opt.Santec)
    monkeypatch.setattr(pib_opt, "create_camera", pib_opt.create_camera)

    original = gsr.display_and_average

    def _fast_measure(cam, slm, phase_rad, *, n_frames, **kwargs):
        kwargs["settle_wait_s"] = 0.0
        kwargs["settle_max_wait_s"] = 0.0
        return original(cam, slm, phase_rad, n_frames=1, **kwargs)

    monkeypatch.setattr(gsr, "display_and_average", _fast_measure)

    saved_system = sim._SYSTEM  # noqa: SLF001 - 进程级单例, 只能这样还原
    try:
        yield
    finally:
        sim._SYSTEM = saved_system  # noqa: SLF001


class TestOrdersResolution:
    """``--orders`` 优先于 ``--n-max``。"""

    def test_parse_orders_splits_on_commas(self):
        assert parse_orders("3,4,6") == (3, 4, 6)
        assert parse_orders(" 3 , 4 ") == (3, 4)
        assert parse_orders("25") == (25,)

    def test_parse_orders_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_orders("3,x")
        with pytest.raises(ValueError):
            parse_orders("")

    def test_orders_win_over_n_max(self):
        from scripts.slm.run_gs_refine_zernike_sweep import SweepParams

        params = SweepParams(n_max_list=(4, 6, 8), orders=(3,))
        assert resolve_orders(params) == (3,)
        assert resolve_orders(SweepParams(n_max_list=(4, 6, 8))) == (4, 6, 8)


class TestStageAHandover:
    """Stage A 的落盘 / 读回: 分段扫描的接缝。"""

    def test_round_trip_preserves_the_arrays(self, tmp_path):
        from scripts.slm.run_gs_refine_zernike_sweep import StageAResult

        result = StageAResult(
            best_phase=np.arange(12, dtype=np.float64).reshape(3, 4),
            flat_frame=np.ones((5, 6)),
            best_frame=np.full((5, 6), 2.0),
            center_xy=(11, 22),
            w0=3.5,
            side=9.0,
            probe_shape=(5, 6),
            best_stage="refine",
            best_score=0.5,
            epochs_executed=7,
            config={"defocus_focal_mm": 1626.2},
        )
        npz_path, json_path = save_stage_a(result, tmp_path)
        assert npz_path.is_file() and json_path.is_file()

        loaded = load_stage_a(npz_path)
        np.testing.assert_array_equal(loaded.best_phase, result.best_phase)
        np.testing.assert_array_equal(loaded.flat_frame, result.flat_frame)
        np.testing.assert_array_equal(loaded.best_frame, result.best_frame)
        assert loaded.center_xy == (11, 22)
        assert loaded.w0 == 3.5
        assert loaded.side == 9.0
        assert loaded.probe_shape == (5, 6)
        assert loaded.epochs_executed == 7
        assert loaded.config["defocus_focal_mm"] == 1626.2

    def test_missing_arrays_are_omitted_not_written_as_empty(self, tmp_path):
        """值为 None ⇒ **不写这个键** (读回来才是 None)。

        写成空数组的话, 读回来是「长度 0 的数组」而不是 None ——
        「不可续跑」的守卫 (``dof is None``) 就形同虚设, 带着空 DOF 往下跑会在
        更难懂的地方炸掉。缺键 = None, 写读语义必须一致。
        """
        from scripts.slm.run_gs_refine_zernike_sweep import StageAResult

        result = StageAResult(
            best_phase=np.zeros((3, 4)),
            flat_frame=np.ones((5, 6)),
            best_frame=np.full((5, 6), 2.0),
            center_xy=(11, 22),
            w0=3.5,
            side=9.0,
            probe_shape=(5, 6),
            best_stage="refine",
            best_score=0.5,
            epochs_executed=7,
        )
        npz_path, _ = save_stage_a(result, tmp_path)

        with np.load(npz_path) as data:
            assert "dof" not in data.files
            assert "base_panel" not in data.files

        loaded = load_stage_a(npz_path)
        assert loaded.dof is None
        assert loaded.base_panel is None

    def test_missing_files_are_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stage_a(tmp_path / "stage_a.npz")


class TestSegmentedSweep:
    """``--stage-a-only`` 跑一次 Stage A, ``--stage-a-in`` 再按阶跑 Stage B/C。"""

    @staticmethod
    def _stage_a(tmp_path, *extra):
        code = sweep_main(
            [
                *SIM_ARGS,
                "--epochs",
                "2",
                "--gs-iters",
                "4",
                "--spgd-epochs",
                "2",
                "--stage-a-only",
                "--out",
                str(tmp_path),
                "--group",
                "zernike_stage_a",
                *extra,
            ]
        )
        assert code == 0
        return tmp_path / "zernike_stage_a"

    def test_stage_a_only_writes_the_handover_files(self, tmp_path, sim_bench):
        group_dir = self._stage_a(tmp_path)

        npz_path = group_dir / "stage_a.npz"
        json_path = group_dir / "stage_a.json"
        assert npz_path.is_file()
        assert json_path.is_file()

        with np.load(npz_path) as data:
            assert {"best_phase", "flat_frame", "best_frame"} <= set(data.files)
            assert data["best_phase"].ndim == 2
            assert data["flat_frame"].shape == data["best_frame"].shape
            assert np.isfinite(data["best_phase"]).all()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        # 仿真孪生的零级落在画面正中 (噪声会让 argmax 抖一两像素)。
        assert abs(payload["center_xy"][0] - 128) <= 2
        assert abs(payload["center_xy"][1] - 128) <= 2
        assert payload["probe_shape"] == [256, 256]
        assert payload["side"] > 0
        assert payload["w0"] > 0
        assert payload["epochs_executed"] >= 0
        assert "best_score" in payload
        assert "defocus_focal_mm" in payload["config"]

        # Stage A 这一段不跑任何阶。
        summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["orders"] == []

    def test_stage_a_in_runs_only_the_named_order(self, tmp_path, sim_bench):
        group_dir = self._stage_a(tmp_path)
        npz_path = group_dir / "stage_a.npz"

        order_dir = tmp_path / "zernike_order_nmax3"
        code = sweep_main(
            [
                *SIM_ARGS,
                "--spgd-epochs",
                "2",
                "--stage-a-in",
                str(npz_path),
                "--orders",
                "3",
                "--out",
                str(tmp_path),
                "--group",
                "zernike_order_nmax3",
            ]
        )
        assert code == 0

        summary = json.loads(
            (order_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert [entry["n_max"] for entry in summary["orders"]] == [3]
        # 基线是从读回来的帧算出来的 (帧存在 stage_a.npz 里就够)。
        assert set(summary["baselines"]) == {"flat", "gs_freeform"}
        assert summary["stage_a"]["source"] == str(npz_path)
        assert summary["stage_a"]["epochs_executed"] >= 0

        # 逐阶产物保持原样。
        assert (order_dir / "spgd_nmax3.csv").is_file()
        assert (order_dir / "best_phase_nmax3.npy").is_file()
        assert (order_dir / "fit_coefficients_nmax3.npy").is_file()
        # 没被点名的阶不许留下产物。
        assert not (order_dir / "spgd_nmax4.csv").exists()
        assert not (order_dir / "best_phase_nmax4.npy").exists()

    def test_stage_a_only_with_stage_a_in_resumes_a_chunk(
        self, tmp_path, sim_bench
    ):
        """``--stage-a-only`` + ``--stage-a-in`` = **续跑这一块** Stage A。

        「一次扫描不超过 10 分钟」把 Stage A 切成若干块后, 块边界必须能续上:
        累计轮数接着涨 (不是从零重来), 否则 250 轮就变成 N 段互不相干的优化。
        """
        chunk1_dir = tmp_path / "chunk1"
        code = sweep_main(
            [
                *SIM_ARGS,
                "--epochs", "4",
                "--chunk-epochs", "2",
                "--gs-iters", "4",
                "--spgd-epochs", "2",
                "--stage-a-only",
                "--out", str(tmp_path),
                "--group", "chunk1",
            ]
        )
        assert code == 0
        first = json.loads(
            (chunk1_dir / "stage_a.json").read_text(encoding="utf-8")
        )

        code = sweep_main(
            [
                *SIM_ARGS,
                "--epochs", "4",
                "--chunk-epochs", "2",
                "--gs-iters", "4",
                "--spgd-epochs", "2",
                "--stage-a-only",
                "--stage-a-in",
                str(chunk1_dir / "stage_a.npz"),
                "--out", str(tmp_path),
                "--group", "chunk2",
            ]
        )
        assert code == 0
        second = json.loads(
            (tmp_path / "chunk2" / "stage_a.json").read_text(encoding="utf-8")
        )

        # 累计轮数接着涨: 第二块把第一块的轮数接着往下数。
        assert second["epochs_total"] > first["epochs_total"]
        assert second["epochs_total"] == first["epochs_total"] + second["epochs_executed"]

    def test_unreadable_handover_exits_non_zero(self, tmp_path, sim_bench):
        """读不回交接文件 ⇒ 非零退出, 不许静默跑一遍没意义的扫描。

        ``--stage-a-only`` + ``--stage-a-in`` 现在是「续跑这一块」的**合法**组合,
        所以「参数错误」落在**读不回来**的地方: 路径不存在, 或者交接文件是旧版
        格式 (没有续跑需要的 DOF / 基座相位 —— 那种文件只能从头跑)。
        """
        # (1) 路径不存在
        code = sweep_main(
            [
                *SIM_ARGS,
                "--stage-a-in",
                str(tmp_path / "nope" / "stage_a.npz"),
                "--out",
                str(tmp_path),
                "--group",
                "missing",
            ]
        )
        assert code != 0, "读不回来的交接文件必须报错退出, 而不是静默跑一遍"

        # (2) 旧版交接文件: 只有三个数组, 没有续跑需要的 DOF / 基座相位
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        np.savez(
            legacy_dir / "stage_a.npz",
            best_phase=np.zeros((120, 192)),
            flat_frame=np.zeros((256, 256)),
            best_frame=np.zeros((256, 256)),
        )
        (legacy_dir / "stage_a.json").write_text(
            json.dumps(
                {
                    "center_xy": [128, 128],
                    "w0": 2.0,
                    "side": 4.0,
                    "probe_shape": [256, 256],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            sweep_main(
                [
                    *SIM_ARGS,
                    "--stage-a-only",
                    "--stage-a-in",
                    str(legacy_dir / "stage_a.npz"),
                    "--out",
                    str(tmp_path),
                    "--group",
                    "legacy",
                ]
            )
        assert "参数错误" in str(excinfo.value)

    def test_time_budget_truncates_stage_a(self, tmp_path, sim_bench):
        """预算极小 ⇒ Stage A 提前收工, 但仍留下一份可用的最佳相位。"""
        group_dir = self._stage_a(tmp_path, "--time-budget-s", "0.001")

        payload = json.loads(
            (group_dir / "stage_a.json").read_text(encoding="utf-8")
        )
        assert payload["epochs_executed"] == 0
        with np.load(group_dir / "stage_a.npz") as data:
            assert np.isfinite(data["best_phase"]).all()
            assert data["best_phase"].size > 0

    def test_per_order_phase_is_the_stage_b_fit_rebuilt_on_the_panel(
        self, tmp_path, sim_bench
    ):
        """逐阶组必须落盘「这一阶的泽尼克匀化相位」—— 远场补拍组要拍的就是它。

        远场组必须拍**本轮**的相位: 上一轮的相位对应上一轮的焦面矫正, 沿用就等于
        拿别的光路状态去拍这一轮。所以这里锁两件事:

        * 落盘的形状是**真面板** (1200x1920)。仿真里缩小的面板只是测试提速, 写进
          给硬件用的相位文件必须是面板分辨率, 否则离焦探针读不进去。
        * 它与 ``fit_coefficients_nmax{n}.npy`` 用 ``panel_phase_from_coefficients``
          重算出来的**逐位一致**。这条恒等式同时排除了「存成 Stage C 之后的
          freeform 相位」—— 那种相位不可能由拟合系数精确重建。
        """
        from ao_shaping.optimizer.wfless.gs_refine_zernike_fit import (
            fit_best_phase,
            panel_phase_from_coefficients,
        )
        from ao_shaping.utils.slm.bench_constants import SLM_PANEL_H, SLM_PANEL_W

        group_dir = self._stage_a(tmp_path)
        npz_path = group_dir / "stage_a.npz"

        order_dir = tmp_path / "zernike_order_nmax3"
        code = sweep_main(
            [
                *SIM_ARGS,
                "--spgd-epochs",
                "2",
                "--stage-a-in",
                str(npz_path),
                "--orders",
                "3",
                "--out",
                str(tmp_path),
                "--group",
                "zernike_order_nmax3",
            ]
        )
        assert code == 0

        phase_path = order_dir / "phase_nmax3.npy"
        coeff_path = order_dir / "fit_coefficients_nmax3.npy"
        assert phase_path.is_file()
        assert coeff_path.is_file()

        phase = np.load(phase_path)
        assert phase.shape == (SLM_PANEL_H, SLM_PANEL_W)
        assert phase.dtype == np.float64

        coefficients = np.load(coeff_path)
        rebuilt = panel_phase_from_coefficients(
            coefficients, SIM_BEAM_RADIUS, (SLM_PANEL_H, SLM_PANEL_W)
        )
        np.testing.assert_array_equal(phase, rebuilt)

        # 存的是**拟合**而不是 freeform 相位: freeform 相位在 n_max 阶上留有
        # 拟合残差, 两者必定不同 (残差为 0 的退化情形除外 —— 那时两者相同是对的)。
        best_phase = np.load(order_dir / "best_phase_nmax3.npy")
        fit = fit_best_phase(best_phase, 3, SIM_BEAM_RADIUS)
        if fit.rms_residual_rad > 0:
            assert not np.array_equal(phase, best_phase)


class TestChunkHandoverFrames:
    """分块续跑的接缝: 目标框中心必须是**窗口内**坐标。

    2026-10-21 实测: 交接文件里的 ``center_xy`` 是**全幅**传感器坐标
    (1434, 973), 而平场帧是相机裁出来的 256x256 ROI 窗口 —— 拿全幅坐标当框
    中心, 框整个落在窗口之外, 续跑块两秒就挂在 "target square (side=12px at
    (973,1434)) is empty"。

    这些测试用一份「硬件味」的交接文件复现那个现场: ``center_xy`` 给全幅坐标,
    ``flat_frame`` 给窗口帧, 然后要求续跑块照样跑完, 且框中心落在窗口内。
    """

    #: 全幅探测器的画面尺寸 (h, w) —— 硬件上就是这么大。
    FULL_FRAME_SHAPE = (1944, 2592)

    #: 全幅坐标 (x, y): 摆 ROI 窗口用, **不是**框中心。
    FULL_FRAME_CENTRE = (1434, 973)

    #: 窗口内坐标 (x, y): 框中心。
    WINDOW_MASK_CENTRE = (127, 128)

    @pytest.fixture(autouse=True)
    def _small_twin(self, sim_bench):
        """把进程级孪生重建成**测试规模**。

        这些用例走 ``--stage-a-in`` (不做平场探测), 所以孪生系统是上一个用例留下
        的那一个 —— 可能是 1200x1920 的真尺寸, 于是相位形状与面板对不上。
        """
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        sim.reset_system(seed=0)

    @staticmethod
    def _window_frame(size: int = 256, peak_xy: tuple[int, int] = (127, 128)):
        """一张**窗口坐标系**下的平场帧: 零级在 ``peak_xy`` (x, y)。"""
        yy, xx = np.mgrid[0:size, 0:size]
        frame = 200.0 * np.exp(
            -((xx - peak_xy[0]) ** 2 + (yy - peak_xy[1]) ** 2) / (2 * 2.0**2)
        )
        return frame.astype(np.float64)

    def _write_handover(self, tmp_path, *, mask_center_xy=None, dof=None):
        """写一份「硬件味」的 Stage A 交接文件。

        Args:
            tmp_path: 临时目录。
            mask_center_xy: 交接文件里的窗口内框中心; ``None`` = 不写这个字段
                (老版交接文件的样子, 2026-10-21 那批就是)。
            dof: 自由参数量; ``None`` = 不写 ``dof`` 键 (老版交接文件),
                ``"empty"`` = 写一个长度为 0 的数组 (写坏的文件)。

        Returns:
            ``stage_a.npz`` 的路径。
        """
        group_dir = tmp_path / "handed_over"
        group_dir.mkdir(parents=True, exist_ok=True)
        npz_path = group_dir / "stage_a.npz"
        frame = self._window_frame()
        arrays = {
            "best_phase": np.zeros(SIM_PANEL),
            "flat_frame": frame,
            "best_frame": frame,
            "base_panel": np.zeros(SIM_PANEL),
            "best_phase_plain": np.zeros(SIM_PANEL),
        }
        if isinstance(dof, str):
            # 写坏的文件: 键在, 但长度 0。
            arrays["dof"] = np.zeros(0)
        elif dof is not None:
            arrays["dof"] = np.asarray(dof, dtype=np.float64)
        np.savez(npz_path, **arrays)
        payload = {
            # 全幅坐标 —— 摆窗口用, 不能当框中心。
            "center_xy": [
                int(self.FULL_FRAME_CENTRE[0]),
                int(self.FULL_FRAME_CENTRE[1]),
            ],
            "w0": 6.0,
            "side": 12.0,
            "probe_shape": [
                int(self.FULL_FRAME_SHAPE[0]),
                int(self.FULL_FRAME_SHAPE[1]),
            ],
            "best_stage": "refine",
            "best_score": 0.5,
            "epochs_executed": 2,
            "epochs_total": 2,
            "elapsed_s": 1.0,
            "config": {},
        }
        if mask_center_xy is not None:
            payload["mask_center_xy"] = [
                int(mask_center_xy[0]),
                int(mask_center_xy[1]),
            ]
        (group_dir / "stage_a.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return npz_path

    @staticmethod
    def _chunk(tmp_path, npz_path, group: str = "chunk_handover") -> int:
        """用交接文件续跑一块 Stage A。"""
        return sweep_main(
            [
                *SIM_ARGS,
                "--epochs",
                "4",
                "--chunk-epochs",
                "1",
                "--gs-iters",
                "4",
                "--stage-a-only",
                "--stage-a-in",
                str(npz_path),
                "--out",
                str(tmp_path),
                "--group",
                group,
            ]
        )

    def test_chunk_uses_the_window_local_mask_centre(self, tmp_path, sim_bench):
        """交接文件里有窗口内中心 ⇒ 续跑块用它, 块跑得完。"""
        npz_path = self._write_handover(
            tmp_path, mask_center_xy=self.WINDOW_MASK_CENTRE, dof=np.zeros(24 * 24)
        )

        assert self._chunk(tmp_path, npz_path) == 0

        payload = json.loads(
            (tmp_path / "chunk_handover" / "stage_a.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["mask_center_xy"] == [127, 128]
        # 全幅中心仍是全幅的 —— 两个坐标系不可互换。
        assert payload["center_xy"] == [1434, 973]

    def test_legacy_handover_without_a_mask_centre_is_relocated(
        self, tmp_path, sim_bench
    ):
        """老版交接文件 (没有 ``mask_center_xy``) ⇒ 从平场帧重新定位窗口内中心。

        2026-10-21 那批已落盘的 chunk 产物就是这个形状: 只有全幅 ``center_xy``。
        """
        npz_path = self._write_handover(tmp_path, mask_center_xy=None, dof=np.zeros(24 * 24))

        assert self._chunk(tmp_path, npz_path) == 0

        payload = json.loads(
            (tmp_path / "chunk_handover" / "stage_a.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["mask_center_xy"] == [127, 128]

    def test_handover_with_a_full_frame_mask_centre_is_relocated(
        self, tmp_path, sim_bench
    ):
        """``mask_center_xy`` 里写的是全幅坐标 ⇒ 落在窗口外, 必须重新定位。"""
        npz_path = self._write_handover(
            tmp_path, mask_center_xy=self.FULL_FRAME_CENTRE, dof=np.zeros(24 * 24)
        )

        assert self._chunk(tmp_path, npz_path) == 0

        payload = json.loads(
            (tmp_path / "chunk_handover" / "stage_a.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["mask_center_xy"] == [127, 128]

    def test_handover_without_a_dof_cannot_be_resumed(self, tmp_path, sim_bench):
        """交接文件里没有 DOF ⇒ 明确报错退出, 不静默跑一遍。"""
        npz_path = self._write_handover(
            tmp_path, mask_center_xy=self.WINDOW_MASK_CENTRE, dof=None
        )

        with pytest.raises(SystemExit):
            self._chunk(tmp_path, npz_path)

    def test_handover_with_an_empty_dof_cannot_be_resumed(self, tmp_path, sim_bench):
        """DOF 是长度 0 的数组 ⇒ 同样不可续跑 (空 DOF 会让后面炸得更难懂)。"""
        npz_path = self._write_handover(
            tmp_path, mask_center_xy=self.WINDOW_MASK_CENTRE, dof="empty"
        )

        with pytest.raises(SystemExit):
            self._chunk(tmp_path, npz_path)

    def test_the_real_20261021_chunk_artefact_is_consumable(self, tmp_path, sim_bench):
        """今天 15:56 已落盘的那份 chunk 产物必须能被回退路径吃下。

        那份文件 (``data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/``)
        是**旧格式**: 只有全幅 ``center_xy`` (1434, 973), 没有 ``mask_center_xy``;
        而 ``stage_a.npz`` 的 ``flat_frame`` 是 256x256 的**窗口**帧。回退路径必须
        从平场帧重新定位窗口内中心 —— 这正是 15:44-16:01 那轮失败的现场。

        ``data/`` 不在版本控制里, 所以文件不在时跳过 (合成版见上面两条)。
        """
        from pathlib import Path

        from ao_shaping.utils.image.targets import square_mask
        from scripts.slm.run_gs_refine_zernike_sweep import (
            _window_mask_centre,
            load_stage_a,
        )

        root = Path(__file__).resolve().parents[4]
        artefact = (
            root
            / "data/debug/hw_zernike_sweep_20261021/zernike_stage_a_1/stage_a.npz"
        )
        if not artefact.is_file():
            pytest.skip(f"现场产物不在: {artefact}")

        stage_a = load_stage_a(artefact)
        # 老产物: 没有窗口内中心, 也没有全幅中心。
        assert stage_a.mask_center_xy is None
        assert stage_a.full_frame_center is None
        # 全幅坐标是 1000 量级, 平场帧是 256x256 的窗口帧 —— 两个坐标系并存。
        assert stage_a.center_xy[0] > stage_a.flat_frame.shape[1]
        assert stage_a.flat_frame.shape == (256, 256)

        centre, how = _window_mask_centre(
            stage_a.flat_frame, stage_a.mask_center_xy, stage_a.side
        )
        assert how == "relocated"
        mask = square_mask(
            stage_a.flat_frame.shape,
            centre[1],
            centre[0],
            int(round(stage_a.side)),
        )
        assert mask.any(), "重新定位出来的框仍然是空的"

    def test_chunk_summary_records_the_window_local_mask_centre(
        self, tmp_path, sim_bench
    ):
        """``summary.json`` 的 ``mask_center_xy`` 不能照抄全幅 ``center_xy``。"""
        npz_path = self._write_handover(
            tmp_path, mask_center_xy=self.WINDOW_MASK_CENTRE, dof=np.zeros(24 * 24)
        )

        assert self._chunk(tmp_path, npz_path) == 0

        summary = json.loads(
            (tmp_path / "chunk_handover" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["mask_center_xy"] != summary["center_xy"]
        assert summary["mask_center_xy"] == [127, 128]
        assert summary["center_xy"] == [1434, 973]


class TestNoBenchLeak:
    """缩小的孪生不许泄漏给后面的测试。

    数字孪生的 system 是**进程级单例**: 分段扫描的测试把它缩到 120x192 以求快,
    但泄漏一次, 后面按 1200x1920 读面板形状的测试就会撞上「相位形状不符」。
    这个类定义在 :class:`TestSegmentedSweep` **之后**, 所以它看到的是夹具
    还原之后的进程状态。
    """

    def test_the_process_level_system_is_back_to_normal(self):
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        system = sim.get_system()
        assert (system.slm_h, system.slm_w) != SIM_PANEL
