"""物理预测模型缺陷拟合分析 (脚本) 的契约测试。

全部离线: 不碰设备、不读写仓库数据目录, 用的都是脚本自己的纯函数与合成数据。
这些测试钉住的是报告**赖以成立**的东西:

* 采样/掩膜的口径 (覆盖掩膜把窗外的零填充排除掉, 目标核罩住实测零级);
* 几何参数的**符号约定** (中心平移的方向、尺度的放大方向) —— 符号写反了, 报告里
  「模型零级错位多少像素」就会变成反的;
* 互相关求平移的约定 (实测是模型右移 s 个采样点 ⇒ 估计出 s);
* 形状残差与仿射对齐 (曝光口径不同不该被算成形状差);
* 退化保护 (模型在核里没信号时, 仿射系数不许给出 1e6 量级的荒唐值);
* 离线契约 (不许 import 设备驱动);
* 合成自检 (脚本 --dry-run 必须把合成真值捞回来)。
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.slm.analyze_fwdmodel_fit import (
    BIG_COST,
    DEFAULT_FIT_PATTERNS,
    DEFAULT_HOLDOUT_PATTERNS,
    FROZEN_GLOBALS,
    GLOBAL_BOUNDS,
    GLOBAL_NAMES,
    affine_fit,
    camera_grid,
    check_synthetic_recovery,
    content_peak,
    coverage_mask,
    disk_mask,
    estimate_align_shift,
    fwhm_profile,
    grid_centre_xy,
    index_to_camera,
    nrmse,
    pearson,
    radial_profile,
    shape_residual,
    spearman,
)

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 统计原语
# ---------------------------------------------------------------------------
def test_pearson_and_spearman_are_self_consistent_and_degenerate_safe() -> None:
    x = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    y = 3.0 * x + 2.0
    assert pearson(x, y) == pytest.approx(1.0, abs=1e-12)
    assert spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert math.isnan(pearson(np.ones((4, 4)), np.ones((4, 4))))
    assert math.isnan(spearman(np.ones((4, 4)), np.ones((4, 4))))
    assert math.isnan(pearson(np.ones((4, 4)), np.ones((3, 3))))


def test_nrmse_is_invariant_to_the_unit_conversion() -> None:
    model = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    assert nrmse(model, model * 1.3e5) == pytest.approx(0.0, abs=1e-12)
    value = nrmse(model, model * 0.5 + 3.0)
    assert math.isfinite(value) and value > 0.0
    assert math.isnan(nrmse(model, np.zeros((8, 8))))


def test_fwhm_profile_measures_half_maximum_width() -> None:
    step = 2.0
    # 三角形 [0, 0.5, 1, 0.5, 0]: 半高在两个 0.5 上 ⇒ FWHM = 2 个采样点
    profile = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    assert fwhm_profile(profile, step) == pytest.approx(2.0 * step, abs=1e-9)
    assert math.isnan(fwhm_profile(np.zeros(5), step))


def test_radial_profile_is_centred_on_the_given_index() -> None:
    image = np.zeros((41, 41))
    image[20, 20] = 10.0
    radii, profile = radial_profile(image, (20.0, 20.0), step=2.0, bin_ccd=4.0)
    assert radii[0] == pytest.approx(2.0)
    assert profile.argmax() == 0  # 峰落在第一个 bin
    assert profile[0] > 0.0
    # 偏心的圆心会把峰推到别的 bin
    _, off = radial_profile(image, (30.0, 30.0), step=2.0, bin_ccd=4.0)
    assert off.argmax() != 0


# ---------------------------------------------------------------------------
# 几何与掩膜
# ---------------------------------------------------------------------------
def _toy_dataset(spot=(100.0, 200.0), half_span=40.0, step=2.0):
    """造一个最小的 Dataset 替身 (只需要几何那几个字段)。"""

    class _Dataset:
        pass

    dataset = _Dataset()
    dataset.spot_xy = spot
    dataset.half_span_px = half_span
    dataset.step_px = step
    return dataset


def test_camera_grid_is_identity_at_the_frozen_parameters() -> None:
    dataset = _toy_dataset()
    grid = camera_grid(dataset)
    centre = grid_centre_xy(grid)
    assert centre == pytest.approx(dataset.spot_xy)
    assert grid.shape == (41, 41, 2)
    # 采样步长 = step_px
    assert grid[0, 1, 0] - grid[0, 0, 0] == pytest.approx(dataset.step_px)


def test_camera_grid_shift_moves_the_sampling_grid_not_the_content() -> None:
    """dx > 0 ⇒ 采样网格往右挪 ⇒ 相机里的固定内容相对窗口**向左**跑。"""
    dataset = _toy_dataset()
    grid = camera_grid(dataset, dx_px=10.0)
    assert grid_centre_xy(grid)[0] == pytest.approx(dataset.spot_xy[0] - 10.0)
    grid = camera_grid(dataset, dy_px=-6.0)
    assert grid_centre_xy(grid)[1] == pytest.approx(dataset.spot_xy[1] + 6.0)


def test_camera_grid_scale_stretches_the_sampling_pitch() -> None:
    dataset = _toy_dataset()
    grid = camera_grid(dataset, scale_x=1.5)
    assert grid[0, 1, 0] - grid[0, 0, 0] == pytest.approx(1.5 * dataset.step_px)


def test_coverage_mask_excludes_samples_outside_the_frame() -> None:
    grid = camera_grid(_toy_dataset(spot=(5.0, 5.0), half_span=40.0))
    covered = coverage_mask(grid, (30, 30))
    assert covered[20, 20]  # 窗心 (5, 5) 在 30×30 帧内
    assert not covered[0, 0]  # 角上 (−35, −35) 在帧外
    assert not covered[-1, -1]


def test_disk_mask_uses_window_index_not_camera_coordinates() -> None:
    """回归测试: 圆心给的是相机坐标, 半径按 CCD px 算 (曾经把下标当坐标用)。

    窗心下标是 (row=20, col=20); 半径 10 CCD px = 5 个采样点 (step=2 px)。
    """
    grid = camera_grid(_toy_dataset(spot=(100.0, 200.0)))
    mask = disk_mask(grid, (100.0, 200.0), radius_ccd_px=10.0, step_px=2.0)
    assert mask[20, 20]
    assert mask[20, 24]  # 4 个采样点 = 8 CCD px ≤ 10
    assert not mask[20, 26]  # 6 个采样点 = 12 CCD px > 10
    assert mask[25, 20]  # 5 个采样点 = 10 CCD px = 半径 (闭区间)
    assert not mask[26, 20]


def test_index_to_camera_round_trips_through_the_grid() -> None:
    grid = camera_grid(_toy_dataset())
    for col, row in ((0, 0), (20, 20), (40, 40)):
        x, y = index_to_camera(grid, float(col), float(row))
        assert x == pytest.approx(grid[row, col, 0])
        assert y == pytest.approx(grid[row, col, 1])


# ---------------------------------------------------------------------------
# 互相关 (中心初值)
# ---------------------------------------------------------------------------
def test_estimate_align_shift_recovers_a_known_translation() -> None:
    """约定: 若 measured[i] == model[i - s] (实测是模型右移 s 个采样点), 返回 (s, 0)。"""
    rng = np.random.default_rng(7)
    model = rng.normal(size=(81, 81))
    model = gaussian_blur(model)
    for shift_x, shift_y in ((0, 0), (3, 0), (0, -4), (5, 6), (-7, 2)):
        measured = np.roll(np.roll(model, shift_x, axis=1), shift_y, axis=0)
        dx, dy = estimate_align_shift(model, measured)
        assert dx == pytest.approx(shift_x, abs=0.35)
        assert dy == pytest.approx(shift_y, abs=0.35)


def gaussian_blur(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """测试用的小高斯 (避免额外依赖)。"""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(image, dtype=np.float64), sigma=sigma)


# ---------------------------------------------------------------------------
# 仿射对齐与形状残差
# ---------------------------------------------------------------------------
def test_shape_residual_absorbs_gain_and_offset() -> None:
    rng = np.random.default_rng(11)
    model = rng.normal(size=(32, 32))
    measured = 17.0 * model + 3.5
    mask = np.ones((32, 32), dtype=bool)
    residual, a, b = shape_residual(model, measured, mask)
    assert residual == pytest.approx(0.0, abs=1e-9)
    assert a == pytest.approx(17.0, rel=1e-9)
    assert b == pytest.approx(3.5, abs=1e-9)


def test_shape_residual_is_scale_invariant() -> None:
    rng = np.random.default_rng(13)
    model = rng.normal(size=(32, 32))
    measured = 4.0 * model
    mask = np.ones((32, 32), dtype=bool)
    assert shape_residual(model, measured, mask)[0] == pytest.approx(0.0, abs=1e-9)
    assert shape_residual(model * 1000.0, measured, mask)[0] == pytest.approx(0.0, abs=1e-9)


def test_affine_fit_refuses_a_degenerate_model() -> None:
    """模型在核里没有信号时, 不许给出 1e6 量级的荒唐系数。"""
    rng = np.random.default_rng(17)
    measured = rng.normal(size=(32, 32))
    mask = np.ones((32, 32), dtype=bool)
    a, b = affine_fit(np.full((32, 32), 2.5), measured, mask)
    assert math.isnan(a) and math.isnan(b)
    residual, _, _ = shape_residual(np.full((32, 32), 2.5), measured, mask)
    assert math.isnan(residual)


def test_content_peak_finds_the_brightest_blob_not_a_speckle() -> None:
    image = np.zeros((61, 61))
    image[30, 30] = 100.0
    image[10, 50] = 3.0  # 散斑
    col, row = content_peak(image, smooth_sigma=2.0)
    assert (col, row) == (30, 30)


# ---------------------------------------------------------------------------
# 参数表与合成自检
# ---------------------------------------------------------------------------
def test_frozen_globals_are_the_identity_of_the_defect_model() -> None:
    assert set(FROZEN_GLOBALS) == set(GLOBAL_NAMES)
    assert FROZEN_GLOBALS["lateral_scale_x"] == 1.0
    assert FROZEN_GLOBALS["lateral_scale_y"] == 1.0
    assert FROZEN_GLOBALS["centre_dx_px"] == 0.0
    assert FROZEN_GLOBALS["centre_dy_px"] == 0.0
    assert FROZEN_GLOBALS["phase_scale"] == 1.0
    assert FROZEN_GLOBALS["blur_sigma_samples"] == 0.0
    assert FROZEN_GLOBALS["defocus_delta_diopter"] == 0.0


def test_every_global_parameter_has_bounds_containing_the_frozen_value() -> None:
    for name, (lo, hi) in GLOBAL_BOUNDS.items():
        assert lo < hi
        assert lo <= FROZEN_GLOBALS[name] <= hi


def test_fit_and_holdout_pattern_sets_are_disjoint() -> None:
    assert set(DEFAULT_FIT_PATTERNS) & set(DEFAULT_HOLDOUT_PATTERNS) == set()


def test_synthetic_recovery_check_only_requires_the_sharp_parameters() -> None:
    truth = dict(FROZEN_GLOBALS)
    truth.update({"centre_dx_px": 8.0, "centre_dy_px": -6.0})
    good = dict(truth)
    check = check_synthetic_recovery(truth, good, baseline_cost=1.0, fitted_cost=0.2)
    assert check["passed"] is True

    bad = dict(truth)
    bad["centre_dx_px"] = 100.0
    check = check_synthetic_recovery(truth, bad, baseline_cost=1.0, fitted_cost=0.2)
    assert check["passed"] is False  # 中心平移是必须项

    # 残差没降也算不过
    check = check_synthetic_recovery(truth, good, baseline_cost=1.0, fitted_cost=0.9)
    assert check["passed"] is False
    assert check["residual_reduction_ok"] is False


def test_big_cost_is_finite_so_powell_never_sees_inf() -> None:
    assert math.isfinite(BIG_COST)
    assert BIG_COST > 1e3


# ---------------------------------------------------------------------------
# 离线契约 + 端到端自检
# ---------------------------------------------------------------------------
def test_the_script_does_not_import_any_device_driver() -> None:
    """静态检查 (AST): 脚本的 import 里不许出现任何设备驱动包。"""
    import ast

    source = (REPO / "scripts" / "slm" / "analyze_fwdmodel_fit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # 相对 import
                continue
            imported.append(module)
    offenders = [name for name in imported if name.split(".")[0] == "ao_shaping"
                 and "drivers" in name.split(".")]
    assert not offenders, f"脚本 import 了设备驱动: {offenders}"
    assert not any(name.startswith("santec") or name.startswith("miicam") for name in imported)


@pytest.mark.slow
def test_dry_run_recovers_the_synthetic_truth(tmp_path: Path) -> None:
    """端到端: ``--dry-run`` 必须把合成真值捞回来 (中心平移是必须项)。"""
    out = tmp_path / "dryrun"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "slm" / "analyze_fwdmodel_fit.py"),
            "--dry-run",
            "--out",
            str(out),
            "--no-figures",
            "--no-joint",
            "--no-sweeps",
            "--max-fit-evals",
            "120",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    import json

    payload = json.loads((out / "fwdmodel_fit.json").read_text(encoding="utf-8"))
    assert payload["synthetic_recovery_passed"] is True
    assert payload["offline"]["driver_modules_imported"] == []
    assert (out / "fwdmodel_fit.md").is_file()
    assert (out / "run.log").is_file()
