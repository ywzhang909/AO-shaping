"""``slm-turb-shaping`` runner (#253) —— 湍流仿真整形闭环的 CI-safe 回归。

覆盖: dry-run 组装 (两个变体, 零副作用); 挂具抵抗 reset 丢失后的**实测**干扰记录;
dynamic 湍流下质量改善 (近似断言, 非逐位); 同种子可复现 (近似); 每次实验都产出
报告所需的全套产物 (run.json / disturbance 伴随 / pkl) 且报告生成器可用 ——
「每次测试都要在报告中展示动态演化」由最后一条把守。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from click.testing import CliRunner  # noqa: E402

from ao_shaping.main import cli  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sim_patch_state():
    """runner 走的是**进程级** sim 补丁 (install_sim_slm 重绑 optimizer 模块的
    ``Santec``, 包装 ``reset_system``); 不还原会污染同分片后续文件
    (test_slm_shaping_runner_sim_backend 的存在性断言实测被打红)。"""
    import ao_shaping.drivers.sim.slm_pib_sim as sim_module
    import ao_shaping.drivers.sim.turbulence_harness as harness
    import ao_shaping.optimizer.wfless.slm_model_in_loop as mil
    import ao_shaping.optimizer.wfless.slm_square_shaping as square_opt
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt

    saved = (square_opt.Santec, pib_opt.Santec, sim_module.reset_system,
             harness._injected, mil._open_bench)
    yield
    (square_opt.Santec, pib_opt.Santec, sim_module.reset_system,
     harness._injected, mil._open_bench) = saved


def _run_arm(tmp_path: Path, algorithm: str, *, epochs: int = 6, seed: int = 7, extra: tuple[str, ...] = ()):
    args = [
        "slm-turb-shaping", "--algorithm", algorithm,
        "--epochs", str(epochs), "--cam-size", "256",
        "--turb-mode", "dynamic", "--dist-seed", str(seed),
        "-d", str(tmp_path), "--no-report", *extra,
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output[-800:]
    return result


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted(p.parent for p in (tmp_path / "debug").rglob("*.pkl"))


def test_dry_run_assembles_both_arms_without_side_effects(tmp_path: Path) -> None:
    for arm in ("square", "spgd"):
        result = CliRunner().invoke(
            cli, ["slm-turb-shaping", "--algorithm", arm, "--dry-run", "-d", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output.strip().splitlines()[-1])
        assert plan["arm"] == arm
        argv = plan["argv"]
        assert "--cam-type" in argv and "sim" in argv
        if arm == "square":
            assert "--slm-type" in argv
            assert argv.index("--debug") < argv.index("square")
    assert not (tmp_path / "debug").exists()


def test_square_arm_records_disturbance_and_report_inputs(tmp_path: Path) -> None:
    _run_arm(tmp_path, "square")
    (run_dir,) = _run_dirs(tmp_path)
    for name in ("run.json", "disturbance.json", "disturbance.npz"):
        assert (run_dir / name).is_file(), name
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    measured = meta["measured"]
    assert meta["turb_mode"] == "dynamic"
    # SPGD 双探针 + 初始帧: 光学求值必须真的被干扰模型看到, 幅度按实测记录。
    assert measured["calls"] >= 13
    assert measured["sigma_total_rad"] > 0
    assert json.loads((run_dir / "disturbance.json").read_text())["measured"]["streaks_used"] > 1


def test_square_arm_improves_quality_under_dynamic_turbulence(tmp_path: Path) -> None:
    _run_arm(tmp_path, "square", epochs=10)
    (run_dir,) = _run_dirs(tmp_path)
    import pickle

    obj = pickle.loads(next(run_dir.glob("*.pkl")).read_bytes())
    rows = [obj[k] for k in sorted(obj, key=lambda x: float(x))]
    q0 = float(rows[0]["quality"])
    best = max(float(r["quality"]) for r in rows)
    # 断言"近似改善": 同种子两次运行也会因设备噪声流不同而差 ~1e-3, 逐位断言禁止。
    assert best > q0 + 1e-4, f"quality did not improve: first={q0} best={best}"


def test_same_dist_seed_is_reproducible(tmp_path: Path) -> None:
    import pickle

    finals = []
    for i in range(2):
        out = tmp_path / f"r{i}"
        _run_arm(out, "square", epochs=6, seed=11)
        (run_dir,) = _run_dirs(out)
        obj = pickle.loads(next(run_dir.glob("*.pkl")).read_bytes())
        rows = [obj[k] for k in sorted(obj, key=lambda x: float(x))]
        finals.append(float(rows[-1]["quality"]))
    assert abs(finals[0] - finals[1]) < 1e-9, finals


def test_spgd_arm_smoke(tmp_path: Path) -> None:
    _run_arm(tmp_path, "spgd", epochs=4)
    (run_dir,) = _run_dirs(tmp_path)
    assert (run_dir / "disturbance.json").is_file()
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["arm"] == "spgd" and meta["exit_code"] == 0
    # 挂具静默说谎回归: 扰动必须真的在光路上被求值 (惰性首建不带扰动的缺陷)。
    assert meta["measured"]["calls"] >= 9, meta["measured"]


def test_heuristic_arm_smoke(tmp_path: Path) -> None:
    _run_arm(
        tmp_path, "heuristic", epochs=2,
        extra=("--search-algorithm", "pso", "--pop-size", "6"),
    )
    (run_dir,) = _run_dirs(tmp_path)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["arm"] == "heuristic" and meta["exit_code"] == 0
    assert meta["measured"]["calls"] >= 6, meta["measured"]


def test_report_rendered_after_experiment(tmp_path: Path) -> None:
    """用户硬约束: 每次测试实验都要在报告中看到相位与光斑的动态演化。"""
    result = CliRunner().invoke(cli, [
        "slm-turb-shaping", "--algorithm", "square",
        "--epochs", "6", "--cam-size", "256", "--turb-mode", "dynamic",
        "--dist-seed", "7", "-d", str(tmp_path),
        "--report", "--report-out", str(tmp_path / "report"),
    ])
    assert result.exit_code == 0, result.output[-800:]
    report_dir = tmp_path / "report"
    for name in ("report.md", "spot_evolution.gif", "phase_evolution.gif", "metrics.png"):
        path = report_dir / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
    body = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "spot_evolution.gif" in body and "phase_evolution.gif" in body
    assert "实测 σ_total" in body


def _model_in_loop_run(tmp_path: Path, *, mode: str, epochs: int) -> Path:
    result = CliRunner().invoke(cli, [
        "slm-turb-shaping", "--algorithm", "model_in_loop",
        "--turb-mode", mode, "--cn2", "2e-13", "--dist-seed", "45",
        "--epochs", str(epochs), "--cam-size", "256", "-n", "8",
        "--target-side", "120", "-d", str(tmp_path), "--no-report",
    ])
    assert result.exit_code == 0, result.output[-800:]
    (run_dir,) = _run_dirs(tmp_path)
    return run_dir


def test_model_in_loop_arm_static_smoke_and_row_contract(tmp_path: Path) -> None:
    """M5b 三合一: 变体在线实现 + 扰动真上光路 + 孪生行补齐 recorder 契约 (_c/_img)。"""
    import pickle

    run_dir = _model_in_loop_run(tmp_path, mode="static", epochs=3)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["arm"] == "model_in_loop" and meta["exit_code"] == 0
    assert meta["n_max"] == 8
    assert meta["measured"]["calls"] >= 4, meta["measured"]
    pkl = next(run_dir.glob("*.pkl"))
    data = pickle.loads(pkl.read_bytes())
    epochs = sorted(data)
    assert epochs[0] == 0, "基线行必须落 pkl (报告首帧)"
    for epoch in epochs:
        row = data[epoch]
        assert row["_img"].shape == (256, 256)
        assert len(row["_c"]) == 45, "n_orders=8 → Noll 1..45 (n≤8 含 piston, 与生成器重建同基)"


def test_model_in_loop_arm_dynamic_completes_under_relaxed_guard(tmp_path: Path) -> None:
    """dynamic 屏会翻盘逐轮验收 —— 放宽的 streak 保险必须让它跑完而非中止。"""
    run_dir = _model_in_loop_run(tmp_path, mode="dynamic", epochs=2)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == 0


def test_model_in_loop_arm_renders_report(tmp_path: Path) -> None:
    """MIL 报告的列名归一 (score_after→quality) 与 n_orders sidecar 兜底。"""
    result = CliRunner().invoke(cli, [
        "slm-turb-shaping", "--algorithm", "model_in_loop",
        "--turb-mode", "static", "--cn2", "2e-13", "--dist-seed", "45",
        "--epochs", "3", "--cam-size", "256", "-n", "8",
        "--target-side", "120", "-d", str(tmp_path),
        "--report", "--report-out", str(tmp_path / "report"),
    ])
    assert result.exit_code == 0, result.output[-800:]
    body = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "quality 首/末/最优" in body and "nan / nan" not in body
    for name in ("spot_evolution.gif", "phase_evolution.gif"):
        assert (tmp_path / "report" / name).stat().st_size > 0, name
