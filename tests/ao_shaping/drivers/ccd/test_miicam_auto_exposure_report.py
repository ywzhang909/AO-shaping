"""MiiCam ``auto_exposure`` 算法硬件验证报告（设备测试项）。

本测试项**替代**原先的仿真报告 ``test_miicam_simulation_report.py``：后者用
合成高斯光斑生成 ``report/miicam_simulation/``，不接触硬件；本测试项在相机在
线时运行探针 ``scripts/ccd/miicam_auto_exposure_probe.py``，把实测报告写入
``report/miicam_auto_exposure/miicam_auto_exposure_probe/``（``report.md`` +
``figures/`` + ``artifacts/``，布局见 ``report/AGENTS.md``）。

探针覆盖（详见其模块 docstring）：前提校验、传感器响应图与台阶扫描、脏数据
核查（帧序号/时间戳/重复帧/交叉切换）、六类目标场景下的收敛/边界退出/有界
不收敛行为、运行后 SDK 状态。设备安全红线：仅经 miicam SDK 枚举/打开/取帧，
全程不导入 gxipy、不触碰 Daheng GigE 相机。

运行方式（宿主机需显式放行，见根 ``conftest.py``）::

    AO_ALLOW_HOST_PYTEST=1 .venv/Scripts/python.exe -m pytest \\
        tests/ao_shaping/drivers/ccd/test_miicam_auto_exposure_report.py \\
        -m hardware -s

或经报告运行器（``--pytest`` 走 ``test_miicam_report.py``；直接运行模式走本文件）::

    python tests/ao_shaping/utils/run_all_reports.py --device miicam
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE = REPO_ROOT / "scripts" / "ccd" / "miicam_auto_exposure_probe.py"
REPORT_DIR = (
    REPO_ROOT / "report" / "miicam_auto_exposure" / "miicam_auto_exposure_probe"
)
REPORT_MD = REPORT_DIR / "report.md"
FIGURE = REPORT_DIR / "figures" / "auto_exposure_probe.png"
RESULTS_JSON = REPORT_DIR / "artifacts" / "results.json"

#: The probe's whole run must finish well inside this; the six AE scenarios
#: include two 20-iteration dead-zone runs (~17 s each) plus the scans.
PROBE_TIMEOUT_S = 900

#: Contract checks that must exist and pass regardless of the target list.
_DIRTY_DATA_CHECKS = (
    "device_expo_matches_request",
    "frame_seq_advances",
    "frame_timestamps_advance",
    "frames_not_duplicated",
    "switch_step_deterministic",
    "switch_no_lag",
)
_OTHER_CHECKS = (
    "premise_brightness",
    "premise_lit_fraction",
    "sdk_auto_exposure_disabled",
    "exposure_readback_consistent",
)
#: One check of each AE scenario class must have run (the step-shaped response
#: makes all three classes meaningful: a covered band converges, a band between
#: rungs oscillates boundedly, a target below the lowest rung exits at the
#: min-exposure boundary).
_AE_SCENARIO_PREFIXES = ("converged[", "boundary_exit[", "deadzone_bounded[")


def test_miicam_auto_exposure_report() -> None:
    """Run the auto-exposure probe on hardware and regenerate the report."""
    assert PROBE.is_file(), f"probe script missing: {PROBE}"

    proc = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--output-dir",
            str(REPORT_DIR),
            "--layout",
            "report",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode} -- not every check passed.\n"
        f"{tail}\n{proc.stderr}"
    )

    assert RESULTS_JSON.is_file(), f"results.json missing under {REPORT_DIR}"
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    # A vacuous pass is worse than a failure: the camera must really have been
    # opened (sn read back from the device) and the scans must have run.
    cam = results.get("camera") or {}
    assert cam.get("sn"), f"no camera sn recorded: {cam}"
    summary = results["summary"]
    assert summary["all_passed"], (
        f"{summary['n_passed']}/{summary['n_checks']} checks passed; failed: "
        f"{[c['check'] for c in results['checks'] if c['passed'] is False]}"
    )
    assert summary["n_checks"] >= 15, (
        f"only {summary['n_checks']} checks recorded -- the scan sections did "
        "not all run"
    )

    failed = [c["check"] for c in results["checks"] if c["passed"] is False]
    assert not failed, f"failed checks: {failed}"

    names = {c["check"] for c in results["checks"]}
    missing = [n for n in (*_DIRTY_DATA_CHECKS, *_OTHER_CHECKS) if n not in names]
    assert not missing, f"contract checks absent from the run: {missing}"
    for prefix in _AE_SCENARIO_PREFIXES:
        assert any(n.startswith(prefix) for n in names), (
            f"no {prefix}... check ran -- the AE scenario classes were not all "
            "exercised"
        )

    # The report tree itself must be complete and linked to its generator.
    assert REPORT_MD.is_file(), f"report missing: {REPORT_MD}"
    assert FIGURE.is_file(), f"figure missing: {FIGURE}"
    for artefact in ("config.json", "trajectory.csv"):
        assert (REPORT_DIR / "artifacts" / artefact).is_file(), (
            f"artefact missing: {artefact}"
        )
    text = REPORT_MD.read_text(encoding="utf-8")
    assert "<!-- provenance:start -->" in text, (
        "provenance header missing -- the report is not registered in "
        "scripts/_common/provenance.py or the stamp failed"
    )
    assert "figures/auto_exposure_probe.png" in text, (
        "report does not embed the figure"
    )

    print(f"\n[miicam-auto-exposure] sn={cam['sn']} "
          f"{summary['n_passed']}/{summary['n_checks']} checks passed, "
          f"{summary['n_info']} diagnostics, report -> {REPORT_MD}")
    for run in results["runs"]:
        print(f"  {run['label']}: exp {run['start_exposure_ms']} -> "
              f"{run['final_exposure_ms']:.4f} ms, peak "
              f"{run['final_peak_returned']:.0f}, resets={run['n_resets']}, "
              f"converged={run['converged']}")


if __name__ == "__main__":
    test_miicam_auto_exposure_report()
