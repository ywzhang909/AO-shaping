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
    assert matrix.VARIANTS == ("spgd", "square", "heuristic")
    assert matrix.CN2_GRID == (1e-14, 2e-13, 2e-12)


def test_cell_plans_construct_params_and_are_unique(matrix, tmp_path: Path) -> None:
    args = matrix.parse_args(["--dry-run", "--out-root", str(tmp_path)])
    plans = [
        matrix.cell_plan(args, variant, cn2, slot)
        for slot, (cn2, variant) in enumerate(
            (c, v) for c in matrix.CN2_GRID for v in matrix.VARIANTS
        )
    ]
    assert len(plans) == 9
    dirs = {Path(p["out_dir"]).name for p in plans}
    assert len(dirs) == 9, "单元目录互撞会让 artifact 发现挑错目录"
    for plan in plans:
        params = TurbShapingParams(**plan)
        assert params.turb_mode == "dynamic" and params.report is False


def test_dry_run_prints_plan_and_writes_nothing(matrix, tmp_path: Path, capsys, monkeypatch) -> None:
    argv = ["prog", "--dry-run", "--out-root", str(tmp_path / "nope")]
    monkeypatch.setattr(sys, "argv", argv)
    assert matrix.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["cells"]) == 9
    assert not (tmp_path / "nope").exists()
