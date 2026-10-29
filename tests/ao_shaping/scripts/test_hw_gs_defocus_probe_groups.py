"""Tests for the defocus probe's one-variable-group-per-invocation mode.

The probe (``scripts/slm/hw_gs_defocus_probe.py``) sweeps one variable group:
the CCD defocus distance ``dz_mm``. Its grouping surface is:

* ``--group NAME`` -- the group's identity (default ``main``); the artefacts land
  in ``<root>/<group>/`` and the root keeps a ``manifest.json`` indexing them.
* ``--group-meta KEY=VALUE`` -- the group's **fixed** parameters, recorded
  verbatim.
* ``--replot`` -- accepts the root **or** a group directory.

The dry-run guarantee ("no device is imported, let alone opened") must hold
**with and without** ``--group``: the grouping surface is on the same code path
as the plan printout, so a stray import there would break the guarantee.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.square_shaping import (
    composite_from_pib_cv,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from scripts._common.group_manifest import (
    group_record,
    load_manifest,
    parse_group_meta,
    update_manifest,
)
from scripts.slm.hw_gs_defocus_probe import (
    GROUP_VARIABLE,
    ChirpTiltCalibrationError,
    OverlayState,
    build_parser,
    calibrate_chirp_tilt,
    compensation_for_dz,
    find_measurement_dirs,
    group_fields,
    group_output_dir,
    learn_ramp_axis_map,
    predict_shift_px,
    ramp_camera_shift_px,
    resolve_probe_target_side,
    run_chirp_tilt_calibration,
    shift_per_mm,
    spot_geometry,
)

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts" / "slm" / "hw_gs_defocus_probe.py"

#: 2026-10-20 那轮实验系统实测留下的原始帧 (根 ``.gitignore`` 忽略 ``data/*``,
#: 所以这批帧**不入库**)。判据依赖这批**具体**帧 —— 缺了就跳过, 不凭记忆重建数字。
HW_GROUPS_ROOT = ROOT / "data" / "debug" / "hw_groups_20261020"

#: 两组期望的自动边长: 硬件 GS 相位打出的光斑 12.9 px 宽 -> 13 px; 模型 GS 相位
#: 22.6 px -> 23 px。
EXPECTED_AUTO_SIDE = {"dz_scan_hw_gs": 13, "dz_scan_model_gs": 23}


def _run_probe(*args: str) -> subprocess.CompletedProcess[str]:
    """跑一次探针 (子进程, 拿真实退出码)。

    像元不再有默认值: 探针要么从 ``--camera-pixel-um`` 取, 要么 (``--execute`` 时)
    从相机型号配置读。子进程里没有真相机, 所以这里注入一个显式值
    (MER2-507-23GM 的 2.2 µm); 调用方自己给了就尊重它。
    """
    extra: tuple[str, ...] = (
        () if "--camera-pixel-um" in args else ("--camera-pixel-um", "2.2")
    )
    return subprocess.run(
        [sys.executable, str(PROBE), *args, *extra],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


class TestDryRunGuarantee:
    """dry-run 的保证, 带不带 ``--group`` 都得成立。"""

    def test_dry_run_without_group(self):
        proc = _run_probe("--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert "dry-run 保证通过" in proc.stdout

    def test_dry_run_with_group(self):
        proc = _run_probe("--dry-run", "--group", "dz_scan")
        assert proc.returncode == 0, proc.stderr
        assert "dry-run 保证通过" in proc.stdout
        assert "dz_scan" in proc.stdout

    def test_dry_run_prints_the_group_and_its_variable(self):
        proc = _run_probe(
            "--dry-run",
            "--group",
            "dz_scan",
            "--group-meta",
            "exposure=0.2",
        )
        assert proc.returncode == 0, proc.stderr
        assert "dz_scan" in proc.stdout
        assert GROUP_VARIABLE in proc.stdout
        assert "exposure=0.2" in proc.stdout

    def test_malformed_group_meta_is_rejected(self):
        proc = _run_probe("--dry-run", "--group-meta", "no_equals_sign")
        assert proc.returncode != 0
        assert "KEY=VALUE" in (proc.stderr + proc.stdout)


class TestGroupParsing:
    def test_group_defaults_to_main(self):
        args = build_parser().parse_args([])
        assert args.group == "main"

    def test_group_meta_is_parsed_into_pairs(self):
        args = build_parser().parse_args(
            ["--group", "dz_scan", "--group-meta", "exposure=0.2",
             "--group-meta", "phase=hardware_gs"]
        )
        assert args.group == "dz_scan"
        assert parse_group_meta(args.group_meta_pairs) == {
            "exposure": "0.2",
            "phase": "hardware_gs",
        }

    def test_group_fields_carry_the_variable(self):
        fields = group_fields("dz_scan", {"exposure": "0.2"})
        assert fields["group"] == "dz_scan"
        assert fields["variable"] == GROUP_VARIABLE == "dz_mm"
        assert fields["group_meta"] == {"exposure": "0.2"}


class TestGroupLayout:
    def test_group_output_dir_is_root_plus_group(self, tmp_path):
        assert group_output_dir(tmp_path, "dz_scan") == tmp_path / "dz_scan"

    def test_manifest_indexes_the_group(self, tmp_path):
        root = tmp_path / "root"
        (root / "dz_scan").mkdir(parents=True)
        update_manifest(
            root,
            group_record(
                group="dz_scan",
                variable=GROUP_VARIABLE,
                meta={"exposure": "0.2"},
                n_rows=11,
            ),
        )
        entries = load_manifest(root)
        assert len(entries) == 1
        assert entries[0]["group"] == "dz_scan"
        assert entries[0]["variable"] == "dz_mm"
        assert entries[0]["meta"] == {"exposure": "0.2"}
        assert entries[0]["dir"] == "dz_scan"
        assert entries[0]["n_rows"] == 11
        assert entries[0]["created_utc"]

    def test_rerunning_a_group_replaces_its_entry(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        for n_rows in (3, 7):
            update_manifest(
                root,
                group_record(
                    group="dz_scan", variable=GROUP_VARIABLE, n_rows=n_rows
                ),
            )
        entries = load_manifest(root)
        assert len(entries) == 1
        assert entries[0]["n_rows"] == 7


class TestReplotResolution:
    """``--replot`` 接受根目录或组目录。"""

    @staticmethod
    def _make_group(root: Path, name: str) -> Path:
        group_dir = root / name
        group_dir.mkdir(parents=True)
        (group_dir / "measurements.json").write_text("[]", encoding="utf-8")
        return group_dir

    def test_group_directory_is_accepted_directly(self, tmp_path):
        group_dir = self._make_group(tmp_path, "dz_scan")
        assert find_measurement_dirs(group_dir) == [group_dir]

    def test_root_directory_is_resolved_through_the_manifest(self, tmp_path):
        root = tmp_path / "root"
        group_dir = self._make_group(root, "dz_scan")
        update_manifest(
            root,
            group_record(group="dz_scan", variable=GROUP_VARIABLE, n_rows=1),
        )
        assert find_measurement_dirs(root) == [group_dir]

    def test_root_directory_falls_back_to_scanning_subdirs(self, tmp_path):
        """清单还没写 (跑到一半崩了) 时, 扫描子目录也要找得到。"""
        root = tmp_path / "root"
        group_dir = self._make_group(root, "dz_scan")
        assert find_measurement_dirs(root) == [group_dir]

    def test_empty_root_yields_nothing(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert find_measurement_dirs(root) == []


@pytest.mark.parametrize("with_group", [False, True])
def test_dry_run_never_imports_the_driver_package(with_group: bool):
    """探针自己会断言这件事; 这里再从外面看一眼, 免得断言被改松。"""
    args = ["--dry-run"]
    if with_group:
        args += ["--group", "dz_scan"]
    proc = _run_probe(*args)
    assert proc.returncode == 0, proc.stderr
    assert "ao_shaping.drivers" not in proc.stderr


def test_manifest_json_is_a_list_of_group_entries(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    update_manifest(
        root,
        group_record(group="a", variable=GROUP_VARIABLE, n_rows=1),
    )
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert set(raw[0]) >= {"group", "variable", "meta", "dir", "n_rows", "created_utc"}


# ---------------------------------------------------------------------------
# 目标框自动定尺
# ---------------------------------------------------------------------------
def _load_group(name: str) -> dict:
    """读一组的 run_config + 逐行原始帧 (指标必须用原始帧重算)。"""
    directory = HW_GROUPS_ROOT / name
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
    frames: dict[int, np.ndarray] = {}
    for row in rows:
        path = directory / f"frame_{int(row['index']):03d}.npz"
        if path.is_file():
            frames[int(row["index"])] = np.load(path)["frame"]
    return {"dir": directory, "config": config, "rows": rows, "frames": frames}


def _rescore(group: dict, side: int) -> list[dict]:
    """按给定边长重算每行的 (Δz, CV, PIB, score), 中心用**冻结的**零级。

    口径与离线报告生成器
    (``scripts/slm/generate_hw_defocus_zernike_report.py``) 一致: 存盘帧先
    ``prepare_frame`` 再算指标, 框心取 ``run_config.json`` 里冻结的零级。
    """
    cx, cy = group["config"]["zero_order_roi_xy_frozen"]
    out: list[dict] = []
    for row in group["rows"]:
        frame = group["frames"].get(int(row["index"]))
        if frame is None:
            continue
        prepared = prepare_frame(frame)
        mask = square_mask(prepared.shape, int(cy), int(cx), int(side))
        cv = float(uniformity_cv(prepared, mask))
        pib = float(power_in_bucket(prepared, mask))
        out.append(
            {
                "index": int(row["index"]),
                "dz_mm": float(row["dz_mm"]),
                "cv": cv,
                "pib": pib,
                "score": float(composite_from_pib_cv(pib, cv)),
            }
        )
    return out


def _positive_branch(scored: list[dict]) -> list[dict]:
    """扫描行里 0 ≤ Δz ≤ 3.75 mm 的那一段 (按 Δz 升序)。

    只取**正**支: 负支 (Δz < 0) 与正支不是严格对称的 (两侧的零级晕斑落点
    不同), 把两支按 |Δz| 混在一起排序, 连噪声都会被算成「不单调」。
    """
    picked = [
        s for s in scored
        if s["index"] >= 1 and -0.01 <= s["dz_mm"] <= 3.76
    ]
    return sorted(picked, key=lambda s: s["dz_mm"])


class TestResolveTargetSide:
    """自动定尺的规则本身 (纯函数, 不需要任何原始帧)。"""

    @staticmethod
    def _blob(fwhm_px: float, shape: tuple[int, int] = (120, 120)) -> np.ndarray:
        """一个高斯光斑 (FWHM 约等于 ``fwhm_px``)。"""
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        cy, cx = shape[0] // 2, shape[1] // 2
        sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))

    def test_explicit_side_is_kept_verbatim(self):
        side, source, _ = resolve_probe_target_side(self._blob(12.0), 250, 20)
        assert (side, source) == (20, "explicit")

    def test_auto_side_is_the_rounded_fwhm(self):
        side, source, fwhm = resolve_probe_target_side(self._blob(12.9), 250, 0)
        assert source == "auto_fwhm"
        assert side == 13
        assert fwhm == pytest.approx(12.9, abs=1.5)

    def test_auto_side_is_clamped_to_the_roi(self):
        # 光斑比 ROI 的 0.6 倍还宽 -> 上界生效。
        side, source, _ = resolve_probe_target_side(self._blob(400.0, (600, 600)), 250, 0)
        assert source == "auto_fwhm"
        assert side == int(np.floor(0.6 * 250))

    def test_auto_side_has_a_lower_bound(self):
        # 下界是 4 px: 框内像素太少时 CV 完全被噪声支配。
        # (实测上 ``measure_spot`` 定位前会做 5x5 盒式模糊, 可测到的最小 FWHM
        # 本身就在 5 px 上下, 所以这条下界是兜底而不是日常路径。)
        side, source, _ = resolve_probe_target_side(self._blob(0.5), 250, 0)
        assert source == "auto_fwhm"
        assert side >= 4

    def test_degenerate_baseline_falls_back_instead_of_crashing(self):
        side, source, _ = resolve_probe_target_side(np.zeros((32, 32)), 250, 0)
        assert (side, source) == (44, "fallback_44")

    def test_spot_geometry_reports_the_spot(self):
        geom = spot_geometry(self._blob(10.0))
        assert set(geom) == {
            "spot_fwhm_px", "spot_centroid_x", "spot_centroid_y", "spot_hollowness",
        }
        assert geom["spot_fwhm_px"] == pytest.approx(10.0, abs=2.0)
        assert geom["spot_centroid_x"] == pytest.approx(60.0, abs=1.5)
        assert geom["spot_centroid_y"] == pytest.approx(60.0, abs=1.5)


@pytest.mark.skipif(
    not HW_GROUPS_ROOT.is_dir(),
    reason="2026-10-20 的实测原始帧不在 (data/* 未入库)",
)
class TestAutoSideOnMeasuredFrames:
    """离线判据: 用**已采集**的原始帧证明自动定尺能恢复离焦响应。"""

    @pytest.mark.parametrize("group", sorted(EXPECTED_AUTO_SIDE))
    def test_auto_side_matches_the_phase_s_own_spot(self, group: str):
        loaded = _load_group(group)
        baseline = loaded["frames"][0]
        roi_px = int(loaded["config"]["roi_px"])
        side, source, fwhm = resolve_probe_target_side(baseline, roi_px, 0)
        assert source == "auto_fwhm"
        assert side == EXPECTED_AUTO_SIDE[group]
        assert fwhm > 0.0

    @pytest.mark.parametrize("group", sorted(EXPECTED_AUTO_SIDE))
    def test_auto_box_response_is_monotone_decreasing(self, group: str):
        """自动框下, 合成分随 Δz 单调下降 (|Δz| ≤ 3.75 mm)。"""
        loaded = _load_group(group)
        side = EXPECTED_AUTO_SIDE[group]
        series = _positive_branch(_rescore(loaded, side))
        assert len(series) >= 7
        scores = [s["score"] for s in series]
        assert all(
            later < earlier for earlier, later in zip(scores, scores[1:])
        ), scores

    def test_fixed_44px_box_cancels_the_hardware_response(self):
        """硬件相位: 44 px 的框让两个合成分**互相抵消**, 响应看起来是平的。

        框比光斑大 3.4 倍时, 离焦弥散把整个框填满 —— CV 下降 (均匀度项
        ``1/(1+CV)`` **上升**) 恰好抵消 PIB 的下降, 合成分于是不降反升。
        """
        loaded = _load_group("dz_scan_hw_gs")
        series = _positive_branch(_rescore(loaded, 44))
        scores = [s["score"] for s in series]
        assert not all(
            later < earlier for earlier, later in zip(scores, scores[1:])
        ), scores
        # 机制: 弥散把框填满 -> 均匀度项上升, 而功率项下降。
        first, later = series[0], next(s for s in series if s["dz_mm"] > 1.0)
        assert later["cv"] < first["cv"]
        assert 1.0 / (1.0 + later["cv"]) > 1.0 / (1.0 + first["cv"])
        assert later["pib"] < first["pib"]

    @pytest.mark.parametrize("group", sorted(EXPECTED_AUTO_SIDE))
    def test_auto_box_is_at_least_as_sensitive_as_the_fixed_box(self, group: str):
        """自动框在最小非零 Δz 上的相对下降 ≥ 44 px 框的相对下降。

        模型相位那一组 44 px 框本来就不抵消 (光斑 22.6 px, 弥散填不满 44 px
        的框), 但它的响应更**平** —— 自动框把这条响应拉陡。
        """
        loaded = _load_group(group)
        auto = _positive_branch(_rescore(loaded, EXPECTED_AUTO_SIDE[group]))
        fixed = _positive_branch(_rescore(loaded, 44))
        assert len(auto) == len(fixed) >= 2

        def _relative_drop(series: list[dict]) -> float:
            first, second = series[0], series[1]
            return (first["score"] - second["score"]) / first["score"]

        assert _relative_drop(auto) >= _relative_drop(fixed)


# ---------------------------------------------------------------------------
# 瞳面斜坡去倾斜 (自标定)
# ---------------------------------------------------------------------------
#: 假实验系统: 面板斜坡 → 相机位移的映射与**符号**都是已知的合成真值。
#:
#: 真值取本实验系统的实际情形 —— 面板 y 斜坡挪的是**相机 x**, 面板 x 斜坡挪的是
#: **相机 y** (面板/相机轴交换 90°), 两个方向的符号都是**负的**。这个符号在仓库
#: 任何地方都没记着, 所以这里把它显式写成真值: 被测代码必须**测**出它, 不许假设。
FAKE_TRUE_MAP: dict[int, tuple[int, float]] = {0: (0, -1.0), 1: (1, -1.0)}

#: 假实验系统的面板/画面尺寸 (小到跑得快, 但光斑与画面边界离得够远)。
FAKE_PANEL_SHAPE = (120, 192)
FAKE_FRAME_SHAPE = (120, 120)
FAKE_CENTER = (60.0, 60.0)

#: 假实验系统里二次相位的转向量 (camera px/mm)。真实验系统实测是
#: 相机 x 方向 -5.2 px/mm。
FAKE_CHIRP_SHIFT_PX_PER_MM = (0.0, -5.2)


class _FakeBench:
    """一台**假**实验系统 —— 不碰任何设备, 帧是现算的高斯光斑。

    假 SLM / 假 CCD 合二为一: :meth:`capture` 收到一个叠加层状态, 按**已知的**
    真值把光斑挪到该挪的位置, 再画一个高斯光斑。

    ``dead_labels`` 让指定的那些状态返回**全黑帧** —— 用来测「量不到光斑就
    放弃去倾斜」那条失败路径。
    """

    def __init__(
        self,
        *,
        chirp_shift_px_per_mm: tuple[float, float] = FAKE_CHIRP_SHIFT_PX_PER_MM,
        scale: float = TILT_SHIFT_SCALE,
        dead_labels: tuple[str, ...] = (),
    ) -> None:
        self.chirp_shift_px_per_mm = chirp_shift_px_per_mm
        self.scale = float(scale)
        self.dead_labels = tuple(dead_labels)
        self.states: list[OverlayState] = []

    def _shift_for(self, state: OverlayState) -> tuple[float, float]:
        """这个叠加层状态让光斑挪了多少 (相机 x/y)。"""
        dx = dy = 0.0
        if state.chirp_dz_mm is not None:
            dx += self.chirp_shift_px_per_mm[0] * state.chirp_dz_mm
            dy += self.chirp_shift_px_per_mm[1] * state.chirp_dz_mm
        for spec in state.ramps:
            camera_axis, sign = FAKE_TRUE_MAP[spec.panel_axis]
            value = spec.sign * sign * self.scale / float(spec.period_px)
            if camera_axis == 0:
                dx += value
            else:
                dy += value
        return dx, dy

    @staticmethod
    def _blob(cx: float, cy: float) -> np.ndarray:
        yy, xx = np.mgrid[: FAKE_FRAME_SHAPE[0], : FAKE_FRAME_SHAPE[1]]
        sigma = 3.0
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))

    def capture(self, state: OverlayState) -> np.ndarray:
        self.states.append(state)
        if state.label in self.dead_labels:
            return np.zeros(FAKE_FRAME_SHAPE, dtype=np.float64)
        dx, dy = self._shift_for(state)
        return self._blob(FAKE_CENTER[0] + dx, FAKE_CENTER[1] + dy)


def _synthetic_axis_map(*, swapped: bool = True):
    """一份**合成**的映射 (直接学出来的, 不是手填的响应表)。

    ``swapped=True`` 复现本实验系统 (面板 y → 相机 x); ``swapped=False`` 故意
    反过来 —— 用来证明被测代码用的是**学到的**映射, 而不是写死的 90° 交换。
    """
    if swapped:
        return learn_ramp_axis_map({0: (-7.4, 0.05), 1: (0.03, -7.4)}, 1000)
    return learn_ramp_axis_map({0: (0.05, -7.4), 1: (-7.4, 0.03)}, 1000)


def _realised_shift(axis_map, specs) -> tuple[float, float]:
    """一组斜坡规格在**相机**上实际造成的位移。"""
    dx = dy = 0.0
    for spec in specs:
        sx, sy = ramp_camera_shift_px(axis_map, spec)
        dx += sx
        dy += sy
    return dx, dy


class TestRampCameraMappingInversion:
    """斜坡 → 相机映射的**反解**: 给定映射, 算出的补偿斜坡必须产生相反的位移。"""

    @pytest.mark.parametrize("swapped", [True, False])
    def test_compensating_ramp_produces_the_opposite_shift(self, swapped: bool):
        """补偿斜坡必须把转向**抵消**掉 (而不是加倍)。"""
        axis_map = _synthetic_axis_map(swapped=swapped)
        chirp_shift = (11.0, -6.5)  # 二次相位带来的相机位移
        calibration = _fake_calibration(axis_map, per_mm=(0.0, 0.0))
        calibration = _with_shift(calibration, chirp_shift)
        _, specs, realised = compensation_for_dz(
            1.0, calibration, FAKE_PANEL_SHAPE
        )
        assert realised[0] == pytest.approx(-chirp_shift[0], rel=0.02)
        assert realised[1] == pytest.approx(-chirp_shift[1], rel=0.02)

    def test_the_period_follows_the_measured_slope(self):
        """周期 = |实测斜率| / |期望位移| —— 不是照抄 bench 常量。"""
        axis_map = _synthetic_axis_map(swapped=True)
        calibration = _with_shift(_fake_calibration(axis_map), (0.0, -5.2))
        _, specs, realised = compensation_for_dz(2.0, calibration, FAKE_PANEL_SHAPE)
        # 期望位移 10.4 px, 实测斜率 7400 ⇒ 周期 ≈ 712 panel px。
        assert len(specs) == 1
        assert specs[0].period_px == pytest.approx(7400.0 / 10.4, abs=1.0)
        assert realised[1] == pytest.approx(10.4, rel=0.02)

    def test_a_synthetic_map_with_a_different_scale_is_respected(self):
        """映射是**学**出来的: 标度差一半, 周期就得差一半。"""
        axis_map = learn_ramp_axis_map({0: (-3.7, 0.0), 1: (0.0, -3.7)}, 1000)
        calibration = _with_shift(_fake_calibration(axis_map), (0.0, -5.2))
        _, specs, _ = compensation_for_dz(2.0, calibration, FAKE_PANEL_SHAPE)
        assert specs[0].period_px == pytest.approx(3700.0 / 10.4, abs=1.0)

    def test_the_ramp_phase_is_finite_and_bounded(self):
        axis_map = _synthetic_axis_map(swapped=True)
        calibration = _with_shift(_fake_calibration(axis_map), (0.0, -5.2))
        phase, _, _ = compensation_for_dz(2.0, calibration, FAKE_PANEL_SHAPE)
        assert phase.shape == FAKE_PANEL_SHAPE
        assert np.all(np.isfinite(phase))
        assert float(np.abs(phase).max()) <= 2.0 * np.pi + 1e-9


def _fake_calibration(axis_map, per_mm=(0.0, 0.0)):
    """造一个 :class:`ChirpTiltCalibration` (只填被测代码真正用到的字段)。"""
    from scripts.slm.hw_gs_defocus_probe import ChirpTiltCalibration

    return ChirpTiltCalibration(
        axis_map=axis_map,
        raw_shift_px_per_mm=per_mm,
        cal_dz_mm=2.0,
        test_period_px=1000,
        cal_centroid_xy=FAKE_CENTER,
    )


def _with_shift(calibration, shift_px):
    """把标定结果里的转向量换成给定的**每 mm** 位移。"""
    from dataclasses import replace

    return replace(calibration, raw_shift_px_per_mm=shift_px)


class TestPerMmShiftExtrapolation:
    """每 mm 位移的折算与外推。"""

    def test_shift_per_mm(self):
        per_mm = shift_per_mm((100.0, 50.0), (90.0, 40.0), 2.0)
        assert per_mm == (-5.0, -5.0)

    def test_predict_scales_linearly_with_dz(self):
        per_mm = (-5.0, 2.0)
        assert predict_shift_px(per_mm, 0.0) == (0.0, 0.0)
        assert predict_shift_px(per_mm, 4.0) == (-20.0, 8.0)
        assert predict_shift_px(per_mm, -1.0) == (5.0, -2.0)

    def test_zero_dz_cannot_be_converted_to_per_mm(self):
        with pytest.raises(ValueError):
            shift_per_mm((0.0, 0.0), (1.0, 1.0), 0.0)

    def test_the_extrapolation_reproduces_the_measured_point(self):
        """在标定点上, 外推必须回到实测值 (否则线性模型就是坏的)。"""
        baseline = (100.0, 50.0)
        measured = (90.0, 40.0)
        per_mm = shift_per_mm(baseline, measured, 2.0)
        back = predict_shift_px(per_mm, 2.0)
        assert (baseline[0] + back[0], baseline[1] + back[1]) == pytest.approx(measured)


class TestChirpTiltCalibration:
    """自标定本身: 学出来的映射、转向量与残差。"""

    def _run(self, bench: _FakeBench, cal_dz_mm: float = 2.0):
        return run_chirp_tilt_calibration(
            bench.capture,
            baseline_xy=FAKE_CENTER,
            focus_panel=None,
            chirp_panel=np.zeros(FAKE_PANEL_SHAPE, dtype=np.float64),
            cal_dz_mm=cal_dz_mm,
            test_period_px=1000,
            panel_shape=FAKE_PANEL_SHAPE,
        )

    def test_the_axis_mapping_is_learned_not_assumed(self):
        bench = _FakeBench()
        calibration, _, _ = self._run(bench)
        assert calibration is not None
        # 假实验系统的真值: 面板 y → 相机 x, 面板 x → 相机 y, 符号都是负的。
        assert calibration.axis_map.for_panel_axis(0).camera_axis == 0
        assert calibration.axis_map.for_panel_axis(1).camera_axis == 1
        assert calibration.axis_map.for_panel_axis(0).shift_px < 0.0
        assert calibration.axis_map.for_panel_axis(1).shift_px < 0.0
        # 实测标度应当对上 bench 常量 (周期 1000 ⇒ 位移 7.4 camera px)。
        assert calibration.axis_map.for_panel_axis(0).expected_ratio == pytest.approx(
            1.0, rel=0.01
        )

    def test_the_steering_is_measured_per_mm(self):
        bench = _FakeBench()
        calibration, _, _ = self._run(bench)
        assert calibration is not None
        assert calibration.raw_shift_px_per_mm[0] == pytest.approx(0.0, abs=0.05)
        assert calibration.raw_shift_px_per_mm[1] == pytest.approx(-5.2, abs=0.05)

    def test_the_residual_is_zero_after_the_compensation(self):
        """补偿之后远场应当**不动** —— 残差 ~ 0。"""
        bench = _FakeBench()
        calibration, residual, note = self._run(bench)
        assert calibration is not None, note
        assert residual is not None, note
        assert abs(residual[0]) < 0.02
        assert abs(residual[1]) < 0.02

    def test_without_the_compensation_the_steering_is_there(self):
        """对照: 不加补偿时, 同样的 Δz 上光斑确实挪了 ~5.2 px/mm。"""
        bench = _FakeBench()
        bench.capture(
            OverlayState("chirp", None, (), 2.0)
        )
        shifted = bench.capture(OverlayState("chirp", None, (), 2.0))
        from scripts.slm.hw_gs_defocus_probe import spot_centroid

        xy = spot_centroid(shifted)
        assert xy is not None
        assert xy[1] - FAKE_CENTER[1] == pytest.approx(-10.4, abs=0.05)


class TestChirpTiltFailSafe:
    """失败保护: 标定不成**不许**让整轮扫描作废。"""

    def _run(self, bench: _FakeBench):
        return run_chirp_tilt_calibration(
            bench.capture,
            baseline_xy=FAKE_CENTER,
            focus_panel=None,
            chirp_panel=np.zeros(FAKE_PANEL_SHAPE, dtype=np.float64),
            cal_dz_mm=2.0,
            test_period_px=1000,
            panel_shape=FAKE_PANEL_SHAPE,
        )

    def test_a_dead_frame_degrades_instead_of_raising(self):
        bench = _FakeBench(dead_labels=("chirp",))
        calibration, residual, note = self._run(bench)
        assert calibration is None
        assert residual is None
        assert "标定失败" in note

    def test_a_dead_ramp_frame_degrades_instead_of_raising(self):
        """二次相位那一帧还好, 但加测试斜坡后光斑没了 —— 照样放弃。"""
        bench = _FakeBench(dead_labels=("chirp+ramp0",))
        calibration, residual, note = self._run(bench)
        assert calibration is None
        assert residual is None
        assert "标定失败" in note

    def test_calibrate_raises_a_dedicated_error(self):
        """底层函数是**抛**的 —— 抛的是专用异常, 不是随便一个 RuntimeError。"""
        bench = _FakeBench(dead_labels=("chirp",))
        with pytest.raises(ChirpTiltCalibrationError):
            calibrate_chirp_tilt(
                bench.capture,
                baseline_xy=FAKE_CENTER,
                focus_panel=None,
                chirp_panel=np.zeros(FAKE_PANEL_SHAPE, dtype=np.float64),
                cal_dz_mm=2.0,
                test_period_px=1000,
                panel_shape=FAKE_PANEL_SHAPE,
            )

    def test_an_implausible_scale_degrades(self):
        """实测标度与 bench 常量差 10 倍 ⇒ 这次标定不可信 ⇒ 放弃。"""
        bench = _FakeBench(scale=TILT_SHIFT_SCALE * 10.0)
        calibration, residual, note = self._run(bench)
        assert calibration is None
        assert residual is None
        assert "标定失败" in note

    def test_a_zero_dz_calibration_is_rejected(self):
        bench = _FakeBench()
        with pytest.raises(ChirpTiltCalibrationError):
            calibrate_chirp_tilt(
                bench.capture,
                baseline_xy=FAKE_CENTER,
                focus_panel=None,
                chirp_panel=np.zeros(FAKE_PANEL_SHAPE, dtype=np.float64),
                cal_dz_mm=0.0,
                test_period_px=1000,
                panel_shape=FAKE_PANEL_SHAPE,
            )


class TestChirpTiltCli:
    """CLI 表面: 默认**关闭** (opt-in)。"""

    def test_the_de_tilt_is_off_by_default(self):
        args = build_parser().parse_args([])
        assert args.chirp_tilt_cancel is False
        assert args.chirp_tilt_cal_dz_mm == pytest.approx(2.0)
        assert args.chirp_tilt_test_period_px == 1000

    def test_the_flags_are_accepted(self):
        args = build_parser().parse_args(
            ["--chirp-tilt-cancel", "--chirp-tilt-cal-dz-mm", "1.5",
             "--chirp-tilt-test-period-px", "800"]
        )
        assert args.chirp_tilt_cancel is True
        assert args.chirp_tilt_cal_dz_mm == pytest.approx(1.5)
        assert args.chirp_tilt_test_period_px == 800

    def test_dry_run_mentions_the_de_tilt_and_its_fail_safe(self):
        proc = _run_probe("--dry-run", "--chirp-tilt-cancel")
        assert proc.returncode == 0, proc.stderr
        assert "去倾斜" in proc.stdout
        assert "标定步骤" in proc.stdout
        assert "失败保护" in proc.stdout

    def test_dry_run_without_the_flag_says_it_is_off(self):
        proc = _run_probe("--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert "去倾斜" in proc.stdout
        assert "关闭 (默认)" in proc.stdout

    def test_every_row_field_is_present_in_the_plan_steps(self):
        """逐行字段必须**每一行**都有 —— 少一个键, 落盘那一步就会 KeyError。

        去倾斜的三列是运行期补的 (标定之后才知道), 所以这里只查非去倾斜的字段。
        """
        from scripts.slm.hw_gs_defocus_probe import (
            STEP_ROW_FIELDS,
            build_optics,
            build_plan,
        )

        args = build_parser().parse_args(["--dry-run", "--camera-pixel-um", "2.2"])
        plan = build_plan(args, build_optics(args))
        for step in plan["steps"]:
            for key in STEP_ROW_FIELDS:
                if key.startswith("chirp_tilt_"):
                    continue
                assert key in step, key


class TestTargetSideRename:
    """探针的定尺函数必须带 ``probe_`` 前缀。

    仓库里另有一个 ``resolve_target_side``
    (``scripts/slm/run_gs_refine_zernike_sweep.py``), 规则是 ``2·w0_flat`` ——
    与探针的「按光斑 FWHM 定尺」是**两套不同的策略**。同名会让人以为它们等价,
    所以只改名, **不**统一规则。
    """

    def test_the_probe_no_longer_exports_the_bare_name(self):
        import scripts.slm.hw_gs_defocus_probe as probe

        assert not hasattr(probe, "resolve_target_side")
        assert hasattr(probe, "resolve_probe_target_side")

    def test_the_sweep_keeps_its_own_name(self):
        from scripts.slm.run_gs_refine_zernike_sweep import resolve_target_side

        assert callable(resolve_target_side)

