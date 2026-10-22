"""``scripts/slm/gsnet_sim_dryrun.py`` 的离线测试。

**为什么这些测试存在.** 预演器是「实验前把整条命令链验证一遍」的工具，它自己
最大的风险是**假装跑过**：某个阶段其实没跑、产物其实没落、判据其实没触发，而
报告照样写着 OK。所以这里钉的是**可观测的事实**：

1. 阶段注册表完整且顺序固定（8 个阶段，少一个就说明漏了一条链）。
2. ``--only probe_fixture --only coupling`` 真跑一遍：fixture 的四个必备产物在，
   耦合拟合的周期 ≈ 993（合成真值）。
3. fixture 能被 ``scripts/slm/fit_bench_twin.py --from-dir`` **接受**（几何契约
   成立、标度来自 fixture 的 ``preflight/geometry.json`` 而不是标称焦距回退）。
4. ``--only dual_aperture`` 能验证通过并产出两个口径各一行。
5. 没有 checkpoint 时 ``export`` / ``transfer`` 判 **BLOCKED** 而不是 FAILED，
   整体退出码仍是 0（缺前置条件 ≠ 链断了）。
6. ``--help`` 正常（文件式启动）。
7. 脚本**不 import** 任何硬件驱动（预演器绝不打开设备）。

**完全离线**: 只跑仿真与纯计算，临时目录，不碰任何设备。
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "gsnet_sim_dryrun.py"
FIT_TWIN = ROOT / "scripts" / "slm" / "fit_bench_twin.py"

EXPECTED_STAGES: tuple[str, ...] = (
    "sim_mil",
    "probe_fixture",
    "twin_fit",
    "coupling",
    "export",
    "transfer",
    "ab_plan",
    "dual_aperture",
)


def _load_script() -> Any:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_gsnet_sim_dryrun", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_gsnet_sim_dryrun"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load_script()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in payload["stages"]:
        if entry["stage"] == name:
            return entry
    raise AssertionError(f"报告里没有阶段 {name}: {payload['stages']}")


@pytest.fixture(scope="module")
def prepared(script: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """跑一次 ``--only probe_fixture``，把 fixture 目录留在磁盘上供后续测试复用。"""
    out = tmp_path_factory.mktemp("gsnet_dryrun_fixture")
    code = script.main(["--out", str(out), "--only", "probe_fixture"])
    assert code == 0, "probe_fixture 阶段没有验证通过"
    assert (out / "fixture" / "probes" / "manifest.json").is_file()
    return out


# ---------------------------------------------------------------------------
# 1. 阶段注册表
# ---------------------------------------------------------------------------


def test_stage_registry_is_complete_and_ordered(script: Any) -> None:
    """8 个阶段一个不少，顺序固定。"""
    assert tuple(script.STAGES) == EXPECTED_STAGES
    assert set(script.STAGE_FUNCS) == set(EXPECTED_STAGES)
    assert set(script.STAGE_TITLES) == set(EXPECTED_STAGES)


# ---------------------------------------------------------------------------
# 2. fixture + 耦合自洽
# ---------------------------------------------------------------------------


def test_probe_fixture_and_coupling_run(script: Any, tmp_path: Path) -> None:
    """``--only probe_fixture --only coupling`` 真跑一遍。"""
    out = tmp_path / "run"
    code = script.main(
        ["--out", str(out), "--only", "probe_fixture", "--only", "coupling"]
    )
    assert code == 0

    fixture = out / "fixture"
    assert (fixture / "probes" / "manifest.json").is_file()
    assert (fixture / "preflight" / "geometry.json").is_file()
    assert (fixture / "source_amplitude.npy").is_file()
    train = sorted((fixture / "probes").glob("round_01_train_*.npz"))
    assert len(train) >= 8, train

    manifest = _read_json(fixture / "probes" / "manifest.json")
    assert isinstance(manifest, list) and manifest
    for entry in manifest:
        for key in (
            "file",
            "round",
            "split",
            "family",
            "index",
            "exposure_ms",
            "timestamp_utc",
            "correlation",
        ):
            assert key in entry, key
        assert (fixture / "probes" / entry["file"]).is_file()

    geometry = _read_json(fixture / "preflight" / "geometry.json")
    model_side = int(np.load(train[0])["model_grid"].shape[0])
    assert int(geometry["far_field_size"]) == model_side, (
        "far_field_size 必须等于 model_side, 否则 fit_bench_twin 会走标度换算分支"
    )
    assert int(geometry["region"]) == int(np.load(train[0])["phase"].shape[0])

    # 耦合：合成真值 c=0.12 / ψ=0.7 / 周期 993，拟合必须自洽。
    coupling = _read_json(out / "coupling.json")
    fit = coupling["amplitude_coupling_fit"]
    assert abs(float(fit["period"]) - 993.0) / 993.0 <= 0.02, fit
    assert abs(float(fit["c"]) - script.SYNTH_C) <= script.COUPLING_C_TOL, fit
    assert abs(float(fit["psi_rad"]) - script.SYNTH_PSI) <= script.COUPLING_PSI_TOL, fit

    payload = _read_json(out / "dryrun.json")
    assert _stage(payload, "probe_fixture")["status"] == "OK"
    assert _stage(payload, "coupling")["status"] == "OK"


# ---------------------------------------------------------------------------
# 3. fixture 能被 fit_bench_twin 接受
# ---------------------------------------------------------------------------


def test_fixture_is_accepted_by_fit_bench_twin(prepared: Path) -> None:
    """``fit_bench_twin.py --from-dir <fixture>`` 必须验证通过，且标度来自 fixture。"""
    out_json = prepared / "bench_twin_probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(FIT_TWIN),
            "--from-dir",
            str(prepared / "fixture"),
            "--out",
            str(out_json),
            "--fit-modes",
            "5,6,7",
            "--fit-lens-modes",
            "5,6",
            "--limit",
            "2",
            "--maxiter",
            "1",
            "--popsize",
            "1",
            "--maxiter-fine",
            "1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert out_json.is_file()
    combined = completed.stdout + completed.stderr
    assert "实测标度" in combined, combined[-2000:]
    assert "标称焦距口径" not in combined, combined[-2000:]
    payload = _read_json(out_json)
    config = payload.get("config", payload)
    assert int(config["region"]) == 64
    assert int(config["far_field_padding"]) == 4


# ---------------------------------------------------------------------------
# 4. 双口径 CV
# ---------------------------------------------------------------------------


def test_dual_aperture_stage_runs_and_writes_two_apertures(
    script: Any, prepared: Path
) -> None:
    """``--only dual_aperture`` 能验证通过并产出两个口径各一行。"""
    code = script.main(["--out", str(prepared), "--only", "dual_aperture"])
    assert code == 0

    csv_path = prepared / "dual_aperture.csv"
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "CSV 是空的"
    assert {row["aperture"] for row in rows} == {"target_full", "plateau_inner"}
    assert len(rows) % 2 == 0

    payload = _read_json(prepared / "dryrun.json")
    stage = _stage(payload, "dual_aperture")
    assert stage["status"] == "OK", stage
    check = _read_json(prepared / "dual_aperture.json")["self_check"]
    assert check["performed"] is True, check
    assert check["reason"] == "ok", check


# ---------------------------------------------------------------------------
# 5. 没有 checkpoint ⇒ export / transfer 是 BLOCKED 而不是 FAILED
# ---------------------------------------------------------------------------


def test_export_and_transfer_are_blocked_without_checkpoint(
    script: Any, tmp_path: Path
) -> None:
    """缺前置条件判 BLOCKED，整体退出码仍是 0。"""
    out = tmp_path / "no_ckpt"
    code = script.main(["--out", str(out), "--only", "export", "--only", "transfer"])
    assert code == 0, "缺 checkpoint 不该让整条链判失败"

    payload = _read_json(out / "dryrun.json")
    for name in ("export", "transfer"):
        stage = _stage(payload, name)
        assert stage["status"] == "BLOCKED", stage
        assert stage["status"] != "FAILED"
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["blocked"] >= 2


# ---------------------------------------------------------------------------
# 6. --help 与「绝不打开设备」
# ---------------------------------------------------------------------------


def test_help_runs_as_a_file() -> None:
    """文件式启动（``python scripts/...``）必须能跑起来。"""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "gsnet_sim_dryrun" in completed.stdout
    for stage in EXPECTED_STAGES:
        assert stage in completed.stdout


def test_script_imports_no_hardware_driver() -> None:
    """预演器绝不 import 硬件驱动。"""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "ao_shaping.drivers"
        ):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names if a.name.startswith("ao_shaping.drivers"))
        for module in modules:
            if not module.startswith("ao_shaping.drivers.sim"):
                offenders.append(module)
    assert not offenders, f"预演器 import 了硬件驱动: {offenders}"


def test_gray_csv_columns_match_the_writer(script: Any, tmp_path: Path) -> None:
    """合成 CSV 的列名必须与 ``validate_flat_phase_gray.py`` 的完全一致。"""
    writer = ROOT / "scripts" / "slm" / "validate_flat_phase_gray.py"
    spec = importlib.util.spec_from_file_location("_validate_flat_phase_gray", writer)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(script.gray_csv_fieldnames()) == tuple(module.CSV_FIELDNAMES)

    path = tmp_path / "gray.csv"
    script._write_synthetic_gray_csv(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or []) == list(module.CSV_FIELDNAMES)
        rows = list(reader)
    assert len(rows) >= 8
    assert "bucket_sum" in rows[0]


def test_dryrun_report_contains_the_must_go_hardware_checklist(
    script: Any, tmp_path: Path
) -> None:
    """报告必须带上「仍然必须在设备上执行」的清单与「仿真 ≠ 实验系统」的注意事项。"""
    out = tmp_path / "report"
    code = script.main(["--out", str(out), "--only", "probe_fixture"])
    assert code == 0
    text = (out / "dryrun.md").read_text(encoding="utf-8")
    assert "仍然必须在设备上执行" in text
    assert "仿真 ≠ 实验系统" in text
    for keyword in ("S1-1 光功率门", "S1-2 灰度扫描", "S1-3 一次 MIL 采集", "S2-1 正式跑"):
        assert keyword in text, keyword
    assert "probes/*.npz" in text
