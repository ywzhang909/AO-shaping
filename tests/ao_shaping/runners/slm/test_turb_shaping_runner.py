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
    import ao_shaping.optimizer.wfless.slm_square_shaping as square_opt
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt

    saved = (square_opt.Santec, pib_opt.Santec, sim_module.reset_system, harness._injected)
    yield
    square_opt.Santec, pib_opt.Santec, sim_module.reset_system, harness._injected = saved


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
