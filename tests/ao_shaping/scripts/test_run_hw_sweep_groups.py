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

No hardware is touched: the plans here run ``sys.executable -c "..."``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.slm.run_hw_sweep_groups import (
    GROUPS_MANIFEST_NAME,
    build_argv,
    check_adopt_dir,
    main,
    resolve_specs,
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
