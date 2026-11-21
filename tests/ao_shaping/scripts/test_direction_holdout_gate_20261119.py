"""``scripts/slm/direction_holdout_gate_20261119.py`` 的离线测试。

只测纯逻辑，不打开硬件、不跑 torch 前向：
- ``pair_rows`` 从合成清单里配出 7 对，某模式缺一个符号时返回 None；
- ``decide`` 的阈值语义：14 对里通过 9 ⇒ fail、10 ⇒ pass；有不可读行 ⇒ not_assessable；
- ``load_checkpoint`` 同时接受裸 ``frozen_parameters.json`` 与 ``{"model", "ramp_matrix"}`` 包。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/slm/direction_holdout_gate_20261119.py"
)


def _load():
    """按路径加载脚本模块。"""
    spec = importlib.util.spec_from_file_location(
        "direction_holdout_gate_20261119_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load()


def _manifest(modes: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)) -> list[dict]:
    """合成清单：模拟 holdout 采集的标签布局。"""
    rows: list[dict] = [
        {"file": "sample_0000.npz", "label": "reference_start", "split": "calibration"}
    ]
    index = 0
    for mode in modes:
        for sign, amp in (("+", "0.05"), ("-", "0.05")):
            rows.append(
                {
                    "file": f"sample_{index + 1:04d}.npz",
                    "label": f"holdout_{index:03d}_Z{mode}_{sign}{amp}",
                    "split": "test",
                    "exposure_ms": 0.043,
                }
            )
            index += 1
    return rows


def test_pair_rows_builds_seven_pairs_from_synthetic_manifest() -> None:
    """7 个模式各配成一对，幅度取正则捕获。"""
    manifest = _manifest()
    pairs = {mode: gate.pair_rows(manifest, mode) for mode in gate.MODES}
    assert all(value is not None for value in pairs.values())
    for mode, value in pairs.items():
        assert value is not None
        minus, plus, amp = value
        assert f"_Z{mode}_-" in minus["label"]
        assert f"_Z{mode}_+" in plus["label"]
        assert amp == "0.05"


def test_pair_rows_returns_none_when_a_sign_is_missing() -> None:
    """缺负向行时必须返回 None，而不是拿正向行凑一对。"""
    manifest = [row for row in _manifest() if "_Z7_-" not in str(row["label"])]
    assert gate.pair_rows(manifest, 7) is None
    assert gate.pair_rows(manifest, 6) is not None


def _pairs(passed: int, total: int = 14) -> list[dict]:
    return [
        {
            "mode": gate.MODES[i % len(gate.MODES)],
            "amp": 0.05,
            "pass": f"pass{i // len(gate.MODES) + 1}",
            "pearson": 0.8 if i < passed else 0.1,
            "passed": i < passed,
            "reason": "synthetic",
        }
        for i in range(total)
    ]


def test_decide_nine_of_fourteen_fails() -> None:
    """9/14 低于 10 的阈值 ⇒ fail。"""
    assert gate.decide(_pairs(9), n_expected=14, min_pairs=10, unreadable=[]) == "fail"


def test_decide_ten_of_fourteen_passes() -> None:
    """10/14 达到阈值 ⇒ pass。"""
    assert gate.decide(_pairs(10), n_expected=14, min_pairs=10, unreadable=[]) == "pass"


def test_decide_missing_mode_is_not_assessable() -> None:
    """某 run 缺一个模式的可读行 ⇒ not_assessable（即便通过数够）。"""
    assert (
        gate.decide(
            _pairs(14), n_expected=14, min_pairs=10, unreadable=["Z7 缺 ± 对"]
        )
        == "not_assessable"
    )
    # 行数不等于期望（例如某个 pass 整段缺失）也视为不可评估。
    assert gate.decide(_pairs(7, total=7), n_expected=14, min_pairs=10, unreadable=[]) == (
        "not_assessable"
    )


_MODEL = {
    "defocus_diopter": 0.01,
    "amplitude_center": [882.0, 572.0],
    "amplitude_width": [368.0, 306.0],
    "gain": 1290.0,
    "background": 20.0,
    "residual_noll_radians": {"5": 0.1},
}
_RAMP = [[155.0, 7570.0], [-7489.0, 196.0]]


def test_load_checkpoint_accepts_bare_frozen_parameters(tmp_path: Path) -> None:
    """裸 frozen_parameters.json：顶层字段 + ramp_matrix。"""
    path = tmp_path / "frozen_parameters.json"
    path.write_text(
        json.dumps({**_MODEL, "ramp_matrix": _RAMP}), encoding="utf-8"
    )
    model, ramp, variant = gate.load_checkpoint(path)
    assert variant == "frozen_parameters"
    assert ramp == _RAMP
    assert model["gain"] == 1290.0
    # 归档脚本的 build_forward 会 int(mode) 转换，故这里保持 JSON 的字符串键。
    assert model["residual_noll_radians"] == {"5": 0.1}


def test_load_checkpoint_accepts_hybrid_bundle(tmp_path: Path) -> None:
    """混合包：{"model": {...}, "ramp_matrix": ...}。"""
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps({"model": _MODEL, "ramp_matrix": _RAMP}), encoding="utf-8"
    )
    model, ramp, variant = gate.load_checkpoint(path)
    assert variant == "hybrid_bundle"
    assert ramp == _RAMP
    assert model["defocus_diopter"] == 0.01
