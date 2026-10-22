"""Tests for the one-variable-group-per-invocation sweep driver.

The driver (``scripts/slm/run_hw_sweep_groups.py``) reads a JSON plan and runs
each variable group as an **independent subprocess**. The contract these tests
pin is the one that makes the split worth anything:

* ``--dry-run`` resolves the plan and executes nothing;
* one group that fails (or exits non-zero) does **not** take the others down;
* an already-finished measurement directory can be *adopted* instead of re-run;
* the index (``groups_manifest.json``) is written and re-readable, and it is
  written **after every group**, so a hard kill still leaves the completed
  groups indexed.

Plus the two "somebody else is holding the device" contracts (2026-10-21/22):

* **device unavailable** (SLM not plugged in / already opened by another
  process / camera taken) **aborts the round** instead of burning every group;
* ``--wait-until-idle`` waits for the device to free up, ``--resume`` picks up
  where the previous (aborted) round stopped.

No hardware is touched: the plans here run ``sys.executable -c "..."``, and the
device probe is a **fake probe** (a script printing a canned verdict).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.slm.run_hw_sweep_groups import (
    GROUPS_MANIFEST_NAME,
    KIND_CAMERA_BUSY,
    KIND_SLM_ABSENT,
    KIND_SLM_BUSY,
    PROBE_IDLE,
    ProcessInfo,
    build_argv,
    check_adopt_dir,
    classify_device_unavailable,
    classify_probe_setup_error,
    main,
    parse_probe_output,
    resolve_specs,
    select_device_using_processes,
    wait_until_idle,
)


def _write_plan(
    path: Path,
    groups: list[dict],
    root: Path,
) -> Path:
    """写一份计划文件 (root + groups)。"""
    path.write_text(
        json.dumps({"root": str(root), "groups": groups}, indent=2),
        encoding="utf-8",
    )
    return path


def _read_manifest(root: Path) -> dict:
    """读回驱动的索引。"""
    return json.loads((root / GROUPS_MANIFEST_NAME).read_text(encoding="utf-8"))


def _harmless(group: str, code: str = "print('x')") -> dict:
    """一组「无害命令」: 当前解释器跑一行 python。"""
    return {
        "group": group,
        "variable": "none",
        "script": sys.executable,
        "args": ["-c", code],
    }


class TestDryRun:
    """``--dry-run`` 只解析计划, 什么都不执行。"""

    def test_lists_groups_and_touches_nothing(self, tmp_path, capsys):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "dz_noise",
                    "variable": "dz_mm",
                    "meta": {"exposure_ms": "0.2"},
                    "script": "scripts/slm/hw_gs_defocus_probe.py",
                    "args": ["--execute", "--dz-mm", "0"],
                },
                {
                    "group": "nmax_scan",
                    "variable": "n_max",
                    "script": "scripts/slm/run_gs_refine_zernike_sweep.py",
                    "args": ["--cam-type", "sim"],
                },
            ],
            root,
        )
        drivers_before = {
            m for m in sys.modules if m.startswith("ao_shaping.drivers")
        }

        rc = main(["--plan", str(plan), "--dry-run"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "dz_noise" in out
        assert "nmax_scan" in out
        assert "dz_mm" in out
        assert "n_max" in out
        assert not root.exists(), "dry-run 不该创建根目录"
        assert not (tmp_path / "groups_manifest.json").exists()
        # dry-run 不许**新**牵进驱动包 (同进程里别的模块引进来的不算)
        drivers_after = {
            m for m in sys.modules if m.startswith("ao_shaping.drivers")
        }
        assert drivers_after == drivers_before

    def test_print_plan_dumps_json(self, tmp_path, capsys):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "dz_noise",
                    "variable": "dz_mm",
                    "script": "scripts/slm/hw_gs_defocus_probe.py",
                    "args": ["--dz-mm", "0"],
                }
            ],
            root,
        )

        rc = main(["--plan", str(plan), "--print-plan"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert [g["group"] for g in payload["groups"]] == ["dz_noise"]
        assert payload["groups"][0]["variable"] == "dz_mm"
        # .py 脚本前面要加当前解释器
        assert payload["groups"][0]["argv"][0] == sys.executable
        assert payload["groups"][0]["argv"][1].endswith("hw_gs_defocus_probe.py")
        assert not root.exists()


class TestRunGroups:
    """一组坏掉不许把后面几组一起带走。"""

    def test_ok_group_is_recorded_with_nonzero_duration(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [_harmless("hello")], root)

        rc = main(["--plan", str(plan)])

        assert rc == 0
        record = _read_manifest(root)["groups"][0]
        assert record["group"] == "hello"
        assert record["status"] == "ok"
        assert record["exit_code"] == 0
        assert record["duration_s"] > 0
        assert record["start_utc"] and record["end_utc"]

    def test_failed_group_does_not_stop_the_next_one(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                _harmless("boom", "import sys; sys.exit(3)"),
                _harmless("after"),
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc == 1, "有组失败时整轮要报失败"
        groups = _read_manifest(root)["groups"]
        assert [g["group"] for g in groups] == ["boom", "after"]
        assert groups[0]["status"] == "failed"
        assert groups[0]["exit_code"] == 3
        # 🔴 关键契约: 坏掉的那组不许带走后面那组
        assert groups[1]["status"] == "ok"
        assert groups[1]["exit_code"] == 0

    def test_only_runs_the_named_group(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [_harmless("first"), _harmless("second")],
            root,
        )

        rc = main(["--plan", str(plan), "--only", "second"])

        assert rc == 0
        groups = _read_manifest(root)["groups"]
        assert [g["group"] for g in groups] == ["second"]

    def test_only_with_unknown_group_is_an_error(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [_harmless("first")], root)

        rc = main(["--plan", str(plan), "--only", "nope"])

        assert rc == 2
        assert not root.exists()

    def test_plan_without_root_and_without_override_is_an_error(self, tmp_path):
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps({"groups": [_harmless("a")]}), encoding="utf-8")

        rc = main(["--plan", str(plan)])

        assert rc == 2


class TestAdoptExisting:
    """已经跑完的目录可以登记进来, 不重跑。"""

    def test_adopt_records_an_existing_directory(self, tmp_path):
        done = tmp_path / "done"
        done.mkdir()
        (done / "measurements.json").write_text("[]", encoding="utf-8")
        (done / "run_config.json").write_text(
            json.dumps({"variable": "dz_mm", "group_meta": {"exposure_ms": "0.2"}}),
            encoding="utf-8",
        )
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [], root)

        rc = main(["--plan", str(plan), "--adopt-existing", "dz_scan", str(done)])

        assert rc == 0
        record = _read_manifest(root)["groups"][0]
        assert record["group"] == "dz_scan"
        assert record["status"] == "adopted"
        assert record["out_dir"] == str(done)
        # 组元数据从目录里读回来
        assert record["variable"] == "dz_mm"
        assert record["meta"] == {"exposure_ms": "0.2"}

    def test_adopt_rejects_a_missing_directory(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [], root)

        rc = main(
            ["--plan", str(plan), "--adopt-existing", "dz_scan", str(tmp_path / "nope")]
        )

        assert rc == 2
        assert not root.exists()

    def test_adopt_rejects_a_directory_without_artefacts(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [], root)

        rc = main(["--plan", str(plan), "--adopt-existing", "dz_scan", str(empty)])

        assert rc == 2
        assert not root.exists()

    def test_adopted_group_in_the_plan_is_not_re_run(self, tmp_path):
        done = tmp_path / "done"
        done.mkdir()
        (done / "summary.json").write_text("{}", encoding="utf-8")
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "dz_scan",
                    "variable": "dz_mm",
                    "script": sys.executable,
                    "args": ["-c", "import sys; sys.exit(9)"],
                }
            ],
            root,
        )

        rc = main(["--plan", str(plan), "--adopt-existing", "dz_scan", str(done)])

        assert rc == 0
        record = _read_manifest(root)["groups"][0]
        assert record["status"] == "adopted"
        assert record["argv"] == []


class TestManifest:
    """索引文件本身。"""

    def test_manifest_is_written_and_re_readable(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [_harmless("a"), _harmless("b", "import sys; sys.exit(1)")],
            root,
        )

        main(["--plan", str(plan)])

        payload = _read_manifest(root)
        assert payload["root"] == str(root)
        assert len(payload["groups"]) == 2
        # 每个组一行摘要
        assert len(payload["summary"]) == 2
        assert payload["summary"][0].startswith("a: ")
        assert payload["summary"][1].startswith("b: ")
        assert payload["created_utc"] and payload["updated_utc"]

    def test_manifest_survives_a_group_that_could_not_start(self, tmp_path):
        """命令根本起不来时, 索引里也必须有这一组。"""
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "missing_binary",
                    "variable": "none",
                    "script": str(tmp_path / "no_such_executable"),
                    "args": [],
                }
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc == 1
        record = _read_manifest(root)["groups"][0]
        assert record["group"] == "missing_binary"
        assert record["status"] == "failed"


class TestGroupBudget:
    """单组的时长预算: 跑完后判定, 超了就点名 (但不杀进程)。"""

    def test_group_over_the_budget_is_named_and_marked(self, tmp_path, capsys):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [_harmless("slow", "import time; time.sleep(1.2)")],
            root,
        )

        rc = main(
            ["--plan", str(plan), "--max-group-seconds", "0.5", "--timeout-s", "60"]
        )

        # 超预算不是失败: 这一组自己跑完了, 退出码仍是 0。
        assert rc == 0
        record = _read_manifest(root)["groups"][0]
        assert record["status"] == "over_budget"
        assert record["exit_code"] == 0
        assert float(record["duration_s"]) > 0.5
        captured = capsys.readouterr()
        assert "slow" in captured.out
        assert "超预算" in captured.err
        # 汇总里必须有一行「最长的一组用了多少秒 / 预算多少秒」
        assert "最长的一组" in captured.out
        assert "预算 0.5s" in captured.out

    def test_group_within_the_budget_stays_ok(self, tmp_path):
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [_harmless("quick")], root)

        rc = main(["--plan", str(plan), "--max-group-seconds", "600"])

        assert rc == 0
        assert _read_manifest(root)["groups"][0]["status"] == "ok"

    def test_default_budgets_are_ten_minutes(self):
        from scripts.slm.run_hw_sweep_groups import _build_parser

        args = _build_parser().parse_args(["--plan", "x.json"])
        assert float(args.timeout_s) == 600.0
        assert float(args.max_group_seconds) == 600.0


class TestHelpers:
    """纯函数。"""

    def test_build_argv_prefixes_the_interpreter_for_python_scripts(self):
        argv = build_argv("scripts/slm/hw_gs_defocus_probe.py", ["--dz-mm", "0"])
        assert argv[0] == sys.executable
        assert argv[1] == "scripts/slm/hw_gs_defocus_probe.py"
        assert argv[2:] == ["--dz-mm", "0"]

    def test_build_argv_uses_a_non_python_command_as_is(self):
        argv = build_argv(sys.executable, ["-c", "print('x')"])
        assert argv == [sys.executable, "-c", "print('x')"]

    def test_check_adopt_dir_reports_the_reason(self, tmp_path):
        assert check_adopt_dir(tmp_path / "nope") is not None
        empty = tmp_path / "empty"
        empty.mkdir()
        assert check_adopt_dir(empty) is not None
        good = tmp_path / "good"
        good.mkdir()
        (good / "summary.json").write_text("{}", encoding="utf-8")
        assert check_adopt_dir(good) is None

    def test_resolve_specs_rejects_a_group_without_a_name(self):
        with pytest.raises(ValueError):
            resolve_specs({"groups": [{"script": "x"}]}, {}, [])

    def test_resolve_specs_filters_by_only(self):
        specs = resolve_specs(
            {
                "groups": [
                    {"group": "a", "script": "x"},
                    {"group": "b", "script": "y"},
                ]
            },
            {},
            ["b"],
        )
        assert [s.group for s in specs] == ["b"]


# ---------------------------------------------------------------------------
# 设备不可用的判别 (2026-10-21 18:23 实测: 27 组在 100 s 内全灭)
# ---------------------------------------------------------------------------
#: 实测到的 SLM 异常文本 (USB 断开) —— 逐字取自失败轮的索引。
SLM_ABSENT_STDERR = (
    "Traceback (most recent call last):\n"
    "  File \"scripts/slm/hw_gs_defocus_probe.py\", line 1617, in run_execute\n"
    "    with guarded(\"SLM 打开\", lambda: open_slm(params), timeout_s=OPEN_TIMEOUT_S) as slm:\n"
    "ao_shaping.drivers.slm.santec.driver.SantecError: 无法打开SLM #1 "
    "(错误码: -10002, 未找到USB设备，请检查设备电源和连接)\n"
)

#: 实测到的 SLM 异常文本 (被别的进程占着) —— 2026-10-22 09:03。
SLM_BUSY_STDERR = (
    "ao_shaping.drivers.slm.santec.driver.SantecError: 无法打开SLM #1 "
    "(错误码: -10003, USB设备已打开)\n"
)


class TestClassifyDeviceUnavailable:
    """「设备不可用」必须分四类 —— 处置不同 (要人插拔 vs 等一会儿)。"""

    def test_slm_absent_is_not_waitable(self):
        verdict = classify_device_unavailable(SLM_ABSENT_STDERR)
        assert verdict is not None
        assert verdict.kind == KIND_SLM_ABSENT
        assert verdict.code == -10002
        # 设备没插上 ⇒ 等是等不来的
        assert not verdict.waitable
        assert "重新插拔" in verdict.advice()

    def test_slm_busy_is_waitable(self):
        verdict = classify_device_unavailable(SLM_BUSY_STDERR)
        assert verdict is not None
        assert verdict.kind == KIND_SLM_BUSY
        assert verdict.code == -10003
        # 别人在用 ⇒ 等一会儿就好
        assert verdict.waitable

    def test_camera_busy_is_waitable(self):
        verdict = classify_device_unavailable(
            "AssertionError: camera not initialized"
        )
        assert verdict is not None
        assert verdict.kind == KIND_CAMERA_BUSY
        assert verdict.waitable

    def test_ordinary_failure_is_not_device_unavailable(self):
        assert classify_device_unavailable("Traceback: ValueError: bad config") is None
        assert classify_device_unavailable("") is None

    def test_the_last_error_wins(self):
        text = SLM_ABSENT_STDERR + SLM_BUSY_STDERR
        verdict = classify_device_unavailable(text)
        assert verdict is not None
        assert verdict.kind == KIND_SLM_BUSY

    def test_the_error_code_alone_is_enough(self):
        verdict = classify_device_unavailable("SantecError: 错误码 -10003")
        assert verdict is not None
        assert verdict.kind == KIND_SLM_BUSY


class TestClassifyProbeSetupError:
    """「探针自己没装好」必须与「设备被占」分开 (2026-10-21 空转 55 分钟)。"""

    def test_missing_gxipy_is_a_setup_error(self):
        reason = classify_probe_setup_error(
            "NameError: name 'gx' is not defined"
        )
        assert reason is not None
        assert "gxipy" in reason
        assert "PYTHONPATH" in reason

    def test_a_busy_device_is_not_a_setup_error(self):
        assert classify_probe_setup_error(SLM_BUSY_STDERR) is None


class TestParseProbeOutput:
    """探针的输出协议。"""

    def test_idle_line(self):
        result = parse_probe_output("DEVICE_PROBE: idle\n")
        assert result.status == PROBE_IDLE

    def test_unavailable_line_with_kind_and_code(self):
        result = parse_probe_output(
            "DEVICE_PROBE_NOTE: SLM 的 USB 句柄已被别的进程打开\n"
            "DEVICE_PROBE: unavailable kind=slm_busy code=-10003\n"
        )
        assert result.status == "unavailable"
        assert result.kind == KIND_SLM_BUSY
        assert result.code == -10003
        assert "USB 句柄" in result.message

    def test_no_verdict_at_all_is_broken(self):
        result = parse_probe_output("", "boom")
        assert result.status == "broken"


class TestProbeMode:
    """``--probe-devices``: 只探一次就退出 (探针本身不碰硬件, 这里也替换掉)。"""

    def test_probe_mode_prints_the_protocol_line(self, monkeypatch, capsys):
        from scripts.slm import run_hw_sweep_groups as driver

        monkeypatch.setattr(
            driver,
            "probe_devices",
            lambda **kwargs: driver.ProbeResult(status=driver.PROBE_IDLE),
        )

        rc = driver.main(["--probe-devices"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "DEVICE_PROBE: idle" in out

    def test_probe_mode_reports_a_busy_device(self, monkeypatch, capsys):
        from scripts.slm import run_hw_sweep_groups as driver

        monkeypatch.setattr(
            driver,
            "probe_devices",
            lambda **kwargs: driver.ProbeResult(
                status=driver.PROBE_UNAVAILABLE,
                kind=KIND_SLM_BUSY,
                code=-10003,
                message="SLM 的 USB 句柄已被别的进程打开",
            ),
        )

        rc = driver.main(["--probe-devices"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "DEVICE_PROBE: unavailable kind=slm_busy code=-10003" in out

    def test_a_missing_plan_is_a_usage_error(self, capsys):
        assert main([]) == 2
        assert "--plan" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 假探针 (不碰硬件)
# ---------------------------------------------------------------------------
def _write_fake_probe(path: Path, body: str) -> Path:
    """写一个假探针脚本 (打印既定的 DEVICE_PROBE 结论)。"""
    path.write_text(body, encoding="utf-8")
    return path


def _busy_then_idle_probe(tmp_path: Path, busy_times: int = 1) -> Path:
    """假探针: 前 ``busy_times`` 次报「被占」, 之后报「可用」。"""
    counter = tmp_path / "probe_counter.txt"
    body = (
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        "n = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(n + 1))\n"
        "if n < "
        f"{busy_times}:\n"
        "    print('DEVICE_PROBE: unavailable kind=slm_busy code=-10003')\n"
        "else:\n"
        "    print('DEVICE_PROBE: idle')\n"
    )
    return _write_fake_probe(tmp_path / "fake_probe_busy.py", body)


def _always_busy_probe(tmp_path: Path) -> Path:
    """假探针: 一直报「被占」。"""
    return _write_fake_probe(
        tmp_path / "fake_probe_always_busy.py",
        "print('DEVICE_PROBE: unavailable kind=slm_busy code=-10003')\n",
    )


def _always_absent_probe(tmp_path: Path) -> Path:
    """假探针: 一直报「设备没插上」。"""
    return _write_fake_probe(
        tmp_path / "fake_probe_absent.py",
        "print('DEVICE_PROBE: unavailable kind=slm_absent code=-10002')\n",
    )


def _broken_probe(tmp_path: Path) -> Path:
    """假探针: 自己坏了 (gxipy 不在 PYTHONPATH 里)。"""
    return _write_fake_probe(
        tmp_path / "fake_probe_broken.py",
        "import sys\n"
        "print(\"NameError: name 'gx' is not defined\", file=sys.stderr)\n"
        "sys.exit(1)\n",
    )


def _no_processes() -> list[ProcessInfo]:
    """进程列表的注入源: 一台「没有别的会话在跑」的机器。"""
    return []


def _counting_probe(tmp_path: Path, verdict: str) -> tuple[Path, Path]:
    """假探针: 每次跑都记一次数 (用来断言「探针到底跑没跑」)。

    Args:
        tmp_path: 临时目录。
        verdict: ``"idle"`` 或 ``"busy"``。

    Returns:
        ``(探针脚本路径, 计数文件路径)``。
    """
    calls = tmp_path / "probe_calls.txt"
    body = (
        "from pathlib import Path\n"
        f"calls = Path({str(calls)!r})\n"
        "n = int(calls.read_text()) if calls.exists() else 0\n"
        "calls.write_text(str(n + 1))\n"
        + (
            "print('DEVICE_PROBE: idle')\n"
            if verdict == "idle"
            else "print('DEVICE_PROBE: unavailable kind=slm_busy code=-10003')\n"
        )
    )
    return _write_fake_probe(tmp_path / "counting_probe.py", body), calls


class TestWaitUntilIdle:
    """``--wait-until-idle``: 被占就等, 空出来再开跑。"""

    def test_waits_until_the_device_is_free(self, tmp_path, capsys):
        probe = _busy_then_idle_probe(tmp_path, busy_times=1)

        code, _reason = wait_until_idle(
            30.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "设备被占" in out
        assert "设备可用" in out

    def test_gives_up_after_the_timeout(self, tmp_path, capsys):
        probe = _always_busy_probe(tmp_path)

        code, reason = wait_until_idle(
            0.2,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 1
        assert reason
        err = capsys.readouterr().err
        assert "仍拿不到设备" in err

    def test_a_missing_device_does_not_wait(self, tmp_path, capsys):
        """-10002 (没插上) 等不来 ⇒ 立刻报错, 不空等。"""
        probe = _always_absent_probe(tmp_path)

        code, _reason = wait_until_idle(
            600.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 1
        err = capsys.readouterr().err
        assert "等不来" in err

    def test_a_broken_probe_stops_immediately(self, tmp_path, capsys):
        """探针自己坏了 ⇒ 立刻报错 (不能当成「相机被占」去空等)。"""
        probe = _broken_probe(tmp_path)

        code, reason = wait_until_idle(
            600.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 2
        assert "gxipy" in reason
        err = capsys.readouterr().err
        assert "探针自己坏了" in err

    def test_the_driver_waits_before_it_starts(self, tmp_path, capsys, monkeypatch):
        """端到端: ``--wait-until-idle`` 等到设备空出来, 然后才开跑。"""
        from scripts.slm import run_hw_sweep_groups as driver

        # 这台机器上别的会话可能有 python 进程在跑 ⇒ 注入一个空的进程列表,
        # 让测试只测「等设备」这条逻辑。
        monkeypatch.setattr(driver, "query_windows_python_processes", _no_processes)
        probe = _busy_then_idle_probe(tmp_path, busy_times=1)
        root = tmp_path / "groups"
        plan = _write_plan(tmp_path / "plan.json", [_harmless("hello")], root)

        rc = main(
            [
                "--plan",
                str(plan),
                "--wait-until-idle",
                "30",
                "--wait-poll-s",
                "0.05",
                "--device-probe-cmd",
                json.dumps([sys.executable, str(probe)]),
            ]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "设备被占" in out
        assert "设备可用" in out
        assert _read_manifest(root)["groups"][0]["status"] == "ok"

    def test_a_probe_that_cannot_start_is_broken(self, tmp_path, capsys):
        """探针命令根本起不来 ⇒ 立刻报错, 不空等。"""
        code, reason = wait_until_idle(
            600.0,
            probe_cmd=[str(tmp_path / "no_such_probe")],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 2
        assert "探针" in reason


# ---------------------------------------------------------------------------
# 公平性: 先看进程, 再探针 (2026-10-22 09:03-09:48 实测)
# ---------------------------------------------------------------------------
def _other_session_processes() -> list[ProcessInfo]:
    """假想「另一个会话」的硬件进程 (4 个进程, 逐字取自那次实测)。"""
    return [
        ProcessInfo(
            pid=100001,
            ppid=100000,
            cmdline="python scripts/slm/model_in_loop_hw_ab.py --stage a",
        ),
        ProcessInfo(
            pid=100002,
            ppid=100000,
            cmdline=(
                "python src/ao_shaping/main.py wf --debug --cam-type daheng "
                "--slm-number 1"
            ),
        ),
    ]


class TestFairness:
    """「探针能打开」不等于「没人用设备」 ⇒ **先看进程, 再探针**。"""

    def test_other_sessions_mean_the_probe_is_never_run(self, tmp_path, capsys):
        """🔴 关键契约: 有别的会话在用 ⇒ **一次都不探针** (探针本身是干扰源)。"""
        probe, calls = _counting_probe(tmp_path, "idle")

        code, reason = wait_until_idle(
            0.3,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_other_session_processes,
        )

        assert code == 1, "等不到就该超时退出"
        assert "别的会话在用设备" in reason
        assert not calls.exists(), "有别的会话在用时, 探针一次都不该跑"
        out = capsys.readouterr().out
        assert "别的会话在用设备" in out
        # 等待过程要说人话: 打印对方是谁 + 已经等了多久
        assert "model_in_loop_hw_ab.py" in out
        assert "已等" in out

    def test_no_other_process_and_the_probe_succeeds(self, tmp_path, capsys):
        """没人用 + 探针成功 ⇒ 开跑, 并说清「凭什么认为设备归我们」。"""
        probe, calls = _counting_probe(tmp_path, "idle")

        code, _reason = wait_until_idle(
            30.0,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 0
        assert calls.exists(), "没人用时探针应该跑过"
        out = capsys.readouterr().out
        assert "设备可用, 且无其它硬件进程 ⇒ 开始采集" in out

    def test_the_probe_says_busy_but_nobody_is_recognised(self, tmp_path, capsys):
        """探针报被占却认不出占用者 ⇒ 继续等, 并说明「没认出占用者」。"""
        probe, _calls = _counting_probe(tmp_path, "busy")

        code, reason = wait_until_idle(
            0.3,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=_no_processes,
        )

        assert code == 1
        assert "没认出占用者" in reason
        out = capsys.readouterr().out
        assert "设备被占但没认出占用者" in out

    def test_our_own_processes_are_not_occupiers(self, tmp_path):
        """自己这条线上的进程 (自身/后代/自己的脚本) 不算占用者。"""
        import os

        self_pid = os.getpid()
        processes = [
            ProcessInfo(pid=self_pid, ppid=1, cmdline="python scripts/slm/x.py"),
            # 我们启动的探针子进程 (它是我们的后代)
            ProcessInfo(pid=self_pid + 1, ppid=self_pid, cmdline="python probe.py"),
            # 同一条线上的另一个实例 (命令行里有我们自己的脚本)
            ProcessInfo(
                pid=self_pid + 2,
                ppid=1,
                cmdline="python scripts/slm/run_hw_sweep_groups.py --plan x.json",
            ),
            # 别的会话的进程 —— 这一个才算
            ProcessInfo(
                pid=self_pid + 3,
                ppid=1,
                cmdline="python scripts/slm/other_lane_probe.py",
            ),
        ]

        hits = select_device_using_processes(
            processes,
            own_paths=[Path(__file__).resolve().parent.parent.parent
                       / "scripts" / "slm" / "run_hw_sweep_groups.py"],
        )

        assert [proc.pid for proc in hits] == [self_pid + 3]

    def test_an_explicitly_owned_pid_is_not_an_occupier(self):
        """``--own-pid`` 声明的进程 (同一条线上的另一个脚本) 不算占用者。"""
        processes = [
            ProcessInfo(pid=500001, ppid=1, cmdline="python scripts/slm/mine.py"),
        ]

        assert select_device_using_processes(processes, exclude_pids=[500001]) == []
        assert len(select_device_using_processes(processes)) == 1

    def test_offline_report_generators_are_not_occupiers(self):
        """纯离线报告生成器住在 scripts/slm/ 下, 但**不碰设备** ⇒ 不算占用者。

        不排除它们的话, 一台正在跑报告的机器会让硬件采集永远等下去 (实测:
        三个 generate_bench_twin_report.py)。
        """
        processes = [
            ProcessInfo(
                pid=600001,
                ppid=1,
                cmdline="python scripts/slm/generate_bench_twin_report.py --run-dir x",
            ),
            ProcessInfo(
                pid=600002,
                ppid=1,
                cmdline="python scripts/slm/hw_gs_defocus_probe.py --execute",
            ),
        ]

        hits = select_device_using_processes(processes)
        assert [proc.pid for proc in hits] == [600002]

    def test_pytest_that_mentions_slm_paths_is_not_an_occupier(self):
        """🔴 回归 (2026-10-22 实测): 别的会话在跑 pytest ⇒ 采集永远等下去。

        pytest 的命令行里**提到** slm / drivers 的测试路径 (还带 ``--ignore=...``),
        裸的 ``slm`` 关键字会把它当成「占用设备的进程」—— 于是一台**没人用设备**的
        机器上, 采集会一直等到超时 (实测: 另一个会话的 pytest 跑着, 采集等了 74 s 还
        在等, 而设备其实是空的)。

        判据必须是「这个进程在**执行**设备代码」, 而不是「命令行里提到设备相关字样」。
        下面两条命令行**逐字取自**那次实测。
        """
        processes = [
            ProcessInfo(
                pid=700001,
                ppid=1,
                cmdline=(
                    "python.exe -m pytest -p no:cacheprovider --rootdir "
                    "C:/Users/me/Worktrees/AO-shaping/x -q --tb=short -ra "
                    '-m "not slow" C:/Users/me/Worktrees/AO-shaping/x/tests '
                    "--ignore=C:/Users/me/Worktrees/AO-shaping/x/tests/ao_shaping/drivers "
                    "--ignore=C:/Users/me/Worktrees/AO-shaping/x/tests/ao_shaping/ml "
                    "--ignore=C:/Users/me/Worktrees/AO-shaping/x/tests/ao_shaping/gui/slm"
                    "/test_zernike_abort_repro.py"
                ),
            ),
            ProcessInfo(
                pid=700002,
                ppid=1,
                cmdline=(
                    "python.exe -m pytest "
                    "D:\\Projects\\TIFO\\AO-shaping\\tests\\ao_shaping\\optimizer\\wfless"
                    "\\test_slm_gs_refine.py -q --no-header"
                ),
            ),
        ]

        assert select_device_using_processes(processes) == []

    def test_a_session_that_starts_during_the_probe_means_we_were_too_early(
        self, tmp_path, capsys
    ):
        """探针开锁的那一瞬间别人开始了 ⇒ 抢早了, 不能开跑。"""
        probe, _calls = _counting_probe(tmp_path, "idle")
        calls = {"n": 0}

        def appearing_processes() -> list[ProcessInfo]:
            # 第一次扫描 (探针之前) 没人; 第二次扫描 (探针之后) 别人开始了。
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            return _other_session_processes()

        code, reason = wait_until_idle(
            0.3,
            probe_cmd=[sys.executable, str(probe)],
            poll_s=0.05,
            process_source=appearing_processes,
        )

        assert code == 1, "抢早了就不该开跑"
        assert "别的会话" in reason
        out = capsys.readouterr().out
        assert "设备可用, 但还有别的硬件进程" in out


class TestAbortOnDeviceUnavailable:
    """设备不可用 ⇒ 中止整轮, 而不是把剩下的组也烧一遍。"""

    def test_round_is_aborted_and_the_rest_is_skipped(self, tmp_path, capsys):
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                _harmless("boom", "import sys; sys.stderr.write(%r); sys.exit(1)"
                          % SLM_ABSENT_STDERR),
                _harmless("after"),
                _harmless("after2"),
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc != 0, "设备不可用时整轮要报失败"
        groups = _read_manifest(root)["groups"]
        assert [g["group"] for g in groups] == ["boom", "after", "after2"]
        # 🔴 关键契约: 致命的那一组不是「failed」, 剩下的组不是「failed」而是 skipped
        assert groups[0]["status"] == "device_unavailable"
        assert groups[0]["device_unavailable"]["kind"] == KIND_SLM_ABSENT
        assert groups[1]["status"] == "skipped"
        assert groups[1]["skip_reason"] == "round_aborted"
        assert groups[2]["status"] == "skipped"
        err = capsys.readouterr().err
        assert "设备不可用" in err
        assert "--resume" in err

    def test_an_ordinary_failure_still_lets_the_round_continue(self, tmp_path):
        """「这一组坏了」≠「设备不可用」: 前者照常跑下一组 (原行为不变)。"""
        root = tmp_path / "groups"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                _harmless("boom", "import sys; sys.stderr.write('ValueError: x'); sys.exit(3)"),
                _harmless("after"),
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc == 1
        groups = _read_manifest(root)["groups"]
        assert groups[0]["status"] == "failed"
        assert groups[1]["status"] == "ok"


class TestResume:
    """``--resume``: 已经有产物的组不重跑, 依赖缺失的组一并跳过。"""

    def test_groups_with_artefacts_are_adopted(self, tmp_path, capsys):
        root = tmp_path / "groups"
        done = root / "already"
        done.mkdir(parents=True)
        (done / "summary.json").write_text("{}", encoding="utf-8")
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "already",
                    "variable": "none",
                    "script": sys.executable,
                    "args": ["-c", "import sys; sys.exit(9)"],
                },
                _harmless("fresh"),
            ],
            root,
        )

        rc = main(["--plan", str(plan), "--resume"])

        assert rc == 0
        groups = _read_manifest(root)["groups"]
        assert groups[0]["status"] == "adopted"
        assert groups[0]["argv"] == []
        assert groups[1]["status"] == "ok"
        assert "已有产物" in capsys.readouterr().out

    def test_a_group_whose_upstream_is_missing_is_skipped(self, tmp_path, capsys):
        """Stage A 的块之间有链: 上游缺产物 ⇒ 这一组不能跑。"""
        root = tmp_path / "groups"
        upstream = root / "block_1" / "stage_a.npz"
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "block_2",
                    "variable": "chunk",
                    "script": sys.executable,
                    "args": [
                        "-c",
                        "print('should not run')",
                        "--stage-a-in",
                        str(upstream),
                    ],
                },
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc == 0, "跳过的组不算失败"
        record = _read_manifest(root)["groups"][0]
        assert record["status"] == "skipped"
        assert record["skip_reason"] == "missing_dependency"
        out = capsys.readouterr().out
        assert "因缺上游产物而跳过" in out
        assert "block_1" in out

    def test_a_group_whose_upstream_exists_runs(self, tmp_path):
        root = tmp_path / "groups"
        upstream = root / "block_1" / "stage_a.npz"
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes(b"npz")
        plan = _write_plan(
            tmp_path / "plan.json",
            [
                {
                    "group": "block_2",
                    "variable": "chunk",
                    "script": sys.executable,
                    "args": ["-c", "print('ok')", "--stage-a-in", str(upstream)],
                },
            ],
            root,
        )

        rc = main(["--plan", str(plan)])

        assert rc == 0
        assert _read_manifest(root)["groups"][0]["status"] == "ok"
