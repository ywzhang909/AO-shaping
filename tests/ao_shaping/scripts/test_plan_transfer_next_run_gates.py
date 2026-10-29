"""Lane 生成器的探针门禁: 光路不 ok 就退出 3, 不是 1。

为什么这条单独锁: 2026-10-29 实测一条被挡掉一部分的光路 (光通量降到参考的 22%)
照样被判 ``ok``, 探针因此长出了第四态 ``weak_path``; 而一条**完全**没有光的 lane
应当以「光路」的身份失败 (码 3, 与 runner 的 ``OpticalPathInvalid`` 同码), 不能被
脚本里的 ``exit 1`` 混成「脚本自己坏了」。生成器是唯一事实源, 已落盘的 ``.ps1``
属于数据, 不随之改写。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "plan_transfer_next_run.py"


def _load():
    spec = importlib.util.spec_from_file_location("plan_transfer_next_run_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = _load()


def _lane(tmp_path: Path, **overrides) -> str:
    fields = dict(shape="square", out=tmp_path / "out", model_dir=tmp_path / "model",
                  side=1200.0, initial_phase="gs", seed_bakeoff=False,
                  gate=tmp_path / "lane.lock", replay=[20.0, 50.0, 90.0])
    fields.update(overrides)
    return planner.lane_script(**fields)


def test_every_shaping_lane_gates_the_probe_and_exits_three(tmp_path):
    lane = _lane(tmp_path)
    assert "probe_optical_path_validity.py" in lane
    after_gate = lane[lane.index("probe_optical_path_validity.py"):]
    assert "if ($LASTEXITCODE -ne 0)" in after_gate
    assert "exit 3" in after_gate
    assert "exit 1" not in after_gate.split("shaping run")[0]
    # 注释必须点名 weak_path: 那条 lane 之前踩的正是「22% 光通量仍判 ok」。
    assert "weak_path" in lane


def test_the_replay_lane_carries_the_same_gate(tmp_path):
    stage = tmp_path / "s600_fixed" / "shape" / "square" / "transport_cv"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps([{"label": "square_seed"}]),
                                        encoding="utf-8")
    (tmp_path / "lanes").mkdir()
    path = planner.replay_only(tmp_path / "s600_fixed", tmp_path / "replay",
                               lane_dir=tmp_path / "lanes", gate=tmp_path / "lane.lock",
                               exposures=[20.0])
    text = path.read_text(encoding="utf-8")
    assert "probe_optical_path_validity.py" in text
    after_gate = text[text.index("probe_optical_path_validity.py"):]
    assert "exit 3" in after_gate
    assert "weak_path" in text
