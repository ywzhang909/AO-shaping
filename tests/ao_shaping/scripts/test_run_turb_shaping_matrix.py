"""``run_turb_shaping_matrix.py`` (#253) 的 CI-safe 契约: dry-run 零副作用。

矩阵执行本体是实验动作 (分钟级/单元), 不进 CI; 进 CI 的是它的计划面 —— 9 个
单元、参数可构造 ``TurbShapingParams``、目录互不冲突、不写任何文件。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ao_shaping.runners.slm.turb_shaping_runner import TurbShapingParams

_NAME = "turb_matrix_under_test"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location(
        _NAME, ROOT / "scripts/slm/run_turb_shaping_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_axes_are_the_frozen_grid(matrix) -> None:
    assert matrix.VARIANTS == ("spgd", "square", "heuristic", "gs_refine",
                               "model_in_loop")
    assert matrix.CN2_GRID == (1e-14, 2e-13, 2e-12)


def test_slot_bases_reproduce_historical_pairing(matrix) -> None:
    # 基址 {0,3,6} 必须逐位复现已提交 m3/m4b/m5c 的 enumerate 布局; gs_refine 新基址
    assert matrix.slot_of("spgd", 2e-12) == 6
    assert matrix.slot_of("square", 2e-13) == 4
    assert matrix.slot_of("heuristic", 2e-12) == 8
    assert matrix.slot_of("gs_refine", 1e-14) == 12
    # MIL (M5b) 独立基址段 24+: 与 gs_refine 的 12..14 不撞, 老单元槽位不动
    assert matrix.slot_of("model_in_loop", 1e-14) == 24
    assert matrix.slot_of("model_in_loop", 2e-12) == 26
    assert len({matrix.slot_of(v, c)
                for v in matrix.VARIANTS for c in matrix.CN2_GRID}) == 15


def test_cell_plans_construct_params_and_are_unique(matrix, tmp_path: Path) -> None:
    args = matrix.parse_args(["--dry-run", "--out-root", str(tmp_path)])
    plans = [
        matrix.cell_plan(args, variant, cn2, matrix.slot_of(variant, cn2))
        for cn2 in matrix.CN2_GRID for variant in matrix.VARIANTS
    ]
    assert len(plans) == 15
    dirs = {Path(p["out_dir"]).name for p in plans}
    assert len(dirs) == 15, "单元目录互撞会让 artifact 发现挑错目录"
    for plan in plans:
        params = TurbShapingParams(**plan)
        expected_mode = "static" if plan["algorithm"] == "model_in_loop" else "dynamic"
        assert params.turb_mode == expected_mode and params.report is False


def test_slot_of_pairs_sub_sweeps_with_frozen_baseline(matrix) -> None:
    # 冻结网格序号 = cn2 档位 * 3 + 变体位; 子扫掠 (--variants 缩窄) 不改种子
    assert matrix.slot_of("spgd", 1e-14) == 0
    assert matrix.slot_of("heuristic", 2e-12) == 8
    args = matrix.parse_args(["--dry-run", "--variants", "heuristic",
                              "--cn2-grid", "2e-12"])
    sub = matrix.axes(args)
    plan = matrix.cell_plan(args, "heuristic", 2e-12, matrix.slot_of("heuristic", 2e-12))
    full = matrix.cell_plan(matrix.parse_args([]), "heuristic", 2e-12,
                            matrix.slot_of("heuristic", 2e-12))
    assert plan["dist_seed"] == full["dist_seed"] == matrix.BASE_DIST_SEED + 8


def test_dry_run_prints_plan_and_writes_nothing(matrix, tmp_path: Path, capsys, monkeypatch) -> None:
    argv = ["prog", "--dry-run", "--out-root", str(tmp_path / "nope")]
    monkeypatch.setattr(sys, "argv", argv)
    assert matrix.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["cells"]) == 15
    assert not (tmp_path / "nope").exists()
