"""Tests for the third-round sweep-plan generator.

The generator (``scripts/slm/build_hw_sweep_plan.py``) replaces the
hand-written plan JSON of the previous round. The hand-written version failed
in two ways that this suite pins down:

* **A change that has to reach many places reached only some of them.** When the
  focal length moved from 1626.2 mm to 1523.8 mm, twenty ``--defocus-focal-mm``
  arguments had to be edited by hand, one at a time. A generator makes that a
  single parameter, so the test asserts the value appears in *every* group that
  takes it.
* **A wrong path did not fail loudly.** The previous round's plan pointed the
  defocus and far-field groups at the *old* sweep root for their phase files.
  When the per-order groups failed, the defocus groups silently reused the
  phase left over from the previous round and nothing complained. The
  generator's self-check (c) refuses to emit a plan whose phase files live
  outside this round's sweep root.

No hardware is touched: the generator imports nothing from ``ao_shaping``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.slm.build_hw_sweep_plan import (
    DEFAULT_DZ_SCAN_MM,
    DEFAULT_DZ_SCAN_MODEL_MM,
    DEFAULT_ORDERS,
    PlanConfig,
    PlanError,
    arg_value,
    build_plan,
    check_chunk_chain,
    check_epoch_coverage,
    check_paths_under_roots,
    check_unique_names,
    main,
    validate_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_ORDERS = (3, 4, 6, 8, 11, 15, 20, 25)


def _cfg(**overrides) -> PlanConfig:
    """默认配置 (可覆盖个别项)。"""
    return PlanConfig(**overrides)


def _names(plan: dict) -> list[str]:
    return [str(g["group"]) for g in plan["groups"]]


def _group(plan: dict, name: str) -> dict:
    for group in plan["groups"]:
        if group["group"] == name:
            return group
    raise AssertionError(f"计划里没有组 {name}")


def _args(group: dict) -> list[str]:
    return [str(a) for a in group["args"]]


def _dz_points(group: dict) -> list[str]:
    """``--dz-mm`` 后面的全部点 (它是该命令的最后一个参数)。"""
    args = _args(group)
    return args[args.index("--dz-mm") + 1 :]


class TestNoHardware:
    """导入生成器不许牵进驱动包。"""

    def test_import_does_not_pull_in_drivers(self):
        # 在**干净子进程**里 import 生成器再查 sys.modules。主测试进程里
        # 同会话别的测试早已牵动驱动包 (惰性注册副作用), 直接查 sys.modules
        # 会把别人的加载记到生成器头上, 误报泄漏。
        probe = (
            "import sys\n"
            "from scripts.slm import build_hw_sweep_plan\n"
            "leaked = sorted(m for m in sys.modules if m.startswith('ao_shaping.drivers'))\n"
            "print(len(leaked))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"clean import failed:\n{proc.stderr}"
        leaked = int(proc.stdout.strip().splitlines()[-1])
        assert leaked == 0, f"生成器牵进了驱动包 ({leaked} 个模块)"


class TestLayout:
    """组数、组序、块数。"""

    def test_group_count_and_order(self):
        plan = build_plan(_cfg())

        expected = (
            ["focus_calibration"]
            + [f"zernike_stage_a_{i}" for i in range(1, 8)]
            + [f"zernike_order_nmax{n}" for n in EXPECTED_ORDERS]
            + ["dz_noise_hw_gs", "dz_scan_hw_gs", "dz_scan_model_gs"]
            + [f"zernike_farfield_nmax{n}" for n in EXPECTED_ORDERS]
        )
        assert _names(plan) == expected
        assert len(plan["groups"]) == 27

    def test_farfield_groups_are_ascending(self):
        """上一轮留下的 11,15,20,25,3,4,6,8 是手写乱序, 这一轮按升序。"""
        plan = build_plan(_cfg())
        farfield = [n for n in _names(plan) if n.startswith("zernike_farfield_nmax")]
        orders = [int(n.rsplit("nmax", 1)[1]) for n in farfield]
        assert orders == sorted(orders)
        assert orders == list(EXPECTED_ORDERS)

    def test_chunk_count_is_ceil_of_total_over_actual(self):
        """250 / 39 ⇒ 7 块 (6 块 × 39 ≈ 234 轮够不到 250 轮)。"""
        cfg = _cfg()
        assert cfg.total_epochs == 250
        assert cfg.epochs_per_chunk == 39
        assert cfg.chunks == 7

    def test_chunk_count_follows_the_actual_epochs_per_chunk(self):
        """每块实际产出调成 45 ⇒ 6 块。"""
        assert _cfg(epochs_per_chunk=45).chunks == 6
        assert _cfg(epochs_per_chunk=50).chunks == 5
        assert _cfg(total_epochs=100, epochs_per_chunk=39).chunks == 3

    def test_requested_chunk_epochs_is_not_the_actual_one(self):
        """39 是墙钟截断的**结果**, 请求值仍是 45。"""
        cfg = _cfg()
        assert cfg.chunk_epochs == 45
        assert cfg.epochs_per_chunk == 39
        assert cfg.effective_epochs_per_chunk == 39
        # 每块**请求**的轮数写进命令里, 实际产出的 39 只用来算块数。
        group = _group(build_plan(cfg), "zernike_stage_a_1")
        assert arg_value(_args(group), "--chunk-epochs") == "45"
        assert arg_value(_args(group), "--time-budget-s") == "540"
        assert arg_value(_args(group), "--epochs") == "250"

    def test_stage_a_chunk_meta_counts_the_total(self):
        plan = build_plan(_cfg())
        for index in range(1, 8):
            group = _group(plan, f"zernike_stage_a_{index}")
            assert group["meta"]["chunk"] == f"{index}/7"


class TestChunkChain:
    """块链: 第 1 块无 ``--stage-a-in``, 第 N 块读第 N-1 块。"""

    def test_first_chunk_has_no_stage_a_in(self):
        plan = build_plan(_cfg())
        assert (
            arg_value(_args(_group(plan, "zernike_stage_a_1")), "--stage-a-in") is None
        )

    def test_chunk_n_reads_chunk_n_minus_one(self):
        plan = build_plan(_cfg())
        for index in range(2, 8):
            group = _group(plan, f"zernike_stage_a_{index}")
            assert arg_value(_args(group), "--stage-a-in") == (
                f"data/debug/hw_zernike_sweep_20261021_r3/"
                f"zernike_stage_a_{index - 1}/stage_a.npz"
            )

    def test_order_groups_read_the_last_chunk(self):
        plan = build_plan(_cfg())
        for n_max in EXPECTED_ORDERS:
            group = _group(plan, f"zernike_order_nmax{n_max}")
            assert arg_value(_args(group), "--stage-a-in") == (
                "data/debug/hw_zernike_sweep_20261021_r3/zernike_stage_a_7/stage_a.npz"
            )

    def test_a_broken_chain_is_reported(self):
        """块链断了必须报错, 而不是照写一份计划。"""
        plan = build_plan(_cfg())
        group = _group(plan, "zernike_stage_a_3")
        args = _args(group)
        index = args.index("--stage-a-in")
        del args[index : index + 2]
        group["args"] = args

        problems = check_chunk_chain(plan, _cfg())
        assert problems, "断掉的块链必须被报出来"
        assert any("zernike_stage_a_3" in p for p in problems)

    def test_a_stage_a_in_pointing_outside_the_plan_is_reported(self):
        """指向计划外的组 = 读上一轮的遗留文件, 而那不会报错。"""
        plan = build_plan(_cfg())
        group = _group(plan, "zernike_stage_a_2")
        args = _args(group)
        index = args.index("--stage-a-in")
        args[index + 1] = (
            "data/debug/hw_zernike_sweep_20261021_r3/zernike_stage_a_99/stage_a.npz"
        )
        group["args"] = args

        problems = check_chunk_chain(plan, _cfg())
        assert any("计划外" in p for p in problems)

    def test_a_missing_chunk_is_reported(self):
        """块编号断了必须报出来, 而不是崩掉。"""
        plan = build_plan(_cfg())
        plan["groups"] = [
            g for g in plan["groups"] if g["group"] != "zernike_stage_a_3"
        ]

        problems = check_chunk_chain(plan, _cfg())
        assert any("块编号不连续" in p for p in problems)


class TestPhasePaths:
    """离焦/远场组引用的相位文件必须落在本轮 sweep 根下。"""

    def test_defocus_and_farfield_phase_files_are_under_this_round_root(self):
        cfg = _cfg()
        plan = build_plan(cfg)
        root = "data/debug/hw_zernike_sweep_20261021_r3"

        checked = 0
        for group in plan["groups"]:
            args = _args(group)
            phase = arg_value(args, "--gs-phase")
            if phase is None:
                continue
            checked += 1
            assert phase.startswith(root + "/"), (
                f"{group['group']} 的相位路径不在本轮根下"
            )
        # 离焦 2 组 (模型对照组不带相位文件) + 远场 8 组
        assert checked == 10

    def test_defocus_groups_use_the_nmax4_best_phase(self):
        plan = build_plan(_cfg())
        for name in ("dz_noise_hw_gs", "dz_scan_hw_gs"):
            group = _group(plan, name)
            expected = (
                "data/debug/hw_zernike_sweep_20261021_r3/"
                "zernike_order_nmax4/best_phase_nmax4.npy"
            )
            assert arg_value(_args(group), "--gs-phase") == expected
            assert group["meta"]["phase_file"] == expected

    def test_farfield_groups_use_their_own_fitted_phase(self):
        plan = build_plan(_cfg())
        for n_max in EXPECTED_ORDERS:
            group = _group(plan, f"zernike_farfield_nmax{n_max}")
            assert arg_value(_args(group), "--gs-phase") == (
                f"data/debug/hw_zernike_sweep_20261021_r3/"
                f"zernike_order_nmax{n_max}/phase_nmax{n_max}.npy"
            )

    def test_model_group_carries_no_phase_file(self):
        """模型相位对照组不带任何相位文件 (它现场生成)。"""
        plan = build_plan(_cfg())
        group = _group(plan, "dz_scan_model_gs")
        assert arg_value(_args(group), "--gs-phase") is None
        assert "phase_file" not in group["meta"]

    def test_a_phase_outside_the_round_root_is_reported(self):
        """上一轮那个静默坑: 相位文件写在旧根下 ⇒ 必须报错。"""
        plan = build_plan(_cfg())
        group = _group(plan, "dz_scan_hw_gs")
        args = _args(group)
        index = args.index("--gs-phase")
        args[index + 1] = (
            "data/debug/hw_zernike_sweep_20261021/zernike_order_nmax4/best_phase_nmax4.npy"
        )
        group["args"] = args
        group["meta"]["phase_file"] = args[index + 1]

        problems = check_paths_under_roots(plan, _cfg())
        assert any("dz_scan_hw_gs" in p for p in problems)

    def test_output_roots_are_under_this_round_root(self):
        cfg = _cfg()
        plan = build_plan(cfg)
        for group in plan["groups"]:
            args = _args(group)
            out_root = arg_value(args, "--out-root")
            if out_root is not None:
                assert out_root == "data/debug/hw_groups_20261021_r3"
            out = arg_value(args, "--out")
            if out is not None:
                assert out == "data/debug/hw_zernike_sweep_20261021_r3"


class TestEpochCoverage:
    """块链能覆盖总轮数。"""

    def test_default_plan_covers_the_total(self):
        cfg = _cfg()
        assert check_epoch_coverage(build_plan(cfg), cfg) == []

    def test_too_few_chunks_is_reported(self):
        """每块请求值太小 ⇒ 块链够不到总轮数, 必须报出来。"""
        cfg = _cfg(chunk_epochs=10)
        with pytest.raises(PlanError) as excinfo:
            build_plan(cfg)
        assert "覆盖不住总轮数" in str(excinfo.value)

    def test_coverage_check_names_the_numbers(self):
        cfg = _cfg(chunk_epochs=10)
        plan = build_plan(_cfg())
        problems = check_epoch_coverage(plan, cfg)
        assert problems, "覆盖不住总轮数必须被报出来"
        assert any("250" in p for p in problems)


class TestFocalLengthPropagation:
    """焦距改动必须同时出现在所有相关组。"""

    def test_focal_length_reaches_every_group_that_takes_it(self):
        plan = build_plan(_cfg(defocus_focal_mm=1600.0))

        touched = 0
        for group in plan["groups"]:
            args = _args(group)
            value = arg_value(args, "--defocus-focal-mm")
            if value is None:
                # 焦面标定组是**测**焦距的, 不接受焦距。
                assert group["group"] == "focus_calibration"
                continue
            touched += 1
            assert value == "1600", f"{group['group']} 的焦距没跟着改"
            assert group["meta"]["defocus_focal_mm"] == "1600"
        # 7 块 + 8 逐阶 + 3 离焦 + 8 远场
        assert touched == 26

    def test_the_old_focal_length_disappears_entirely(self):
        plan = build_plan(_cfg(defocus_focal_mm=1600.0))
        blob = json.dumps(plan, ensure_ascii=False)
        assert "1523.8" not in blob

    def test_a_single_parameter_moves_everywhere(self):
        """这就是手写计划的失败模式: 一处漏改, 数据就自相矛盾。"""
        plan = build_plan(_cfg(defocus_focal_mm=1234.5))
        for group in plan["groups"]:
            value = arg_value(_args(group), "--defocus-focal-mm")
            if value is not None:
                assert value == "1234.5"


class TestIdentityFields:
    """每组都带自变量与角色。"""

    def test_every_group_has_variable_and_role(self):
        plan = build_plan(_cfg())
        for group in plan["groups"]:
            assert str(group.get("variable", "")).strip(), (
                f"{group['group']} 没有自变量"
            )
            assert str((group.get("meta") or {}).get("role", "")).strip(), (
                f"{group['group']} 没有 meta.role"
            )

    def test_duplicate_group_names_are_reported(self):
        plan = build_plan(_cfg())
        plan["groups"].append(dict(plan["groups"][0]))
        problems = check_unique_names(plan)
        assert problems
        assert any("重复" in p for p in problems)


class TestSelfCheck:
    """自检不过 ⇒ 报错退出, 不写文件。"""

    def test_build_plan_raises_on_a_broken_plan(self):
        cfg = _cfg()
        plan = build_plan(cfg)
        # 造一个「相位文件写在旧根下」的计划, 校验必须把它挑出来。
        group = _group(plan, "dz_noise_hw_gs")
        args = _args(group)
        index = args.index("--gs-phase")
        args[index + 1] = "data/debug/hw_zernike_sweep_20261021/x.npy"
        group["args"] = args
        assert validate_plan(plan, cfg)

    def test_validate_plan_is_clean_for_the_default(self):
        cfg = _cfg()
        assert validate_plan(build_plan(cfg), cfg) == []


class TestCli:
    """命令行: 落盘 / 只打印 / 退出码。"""

    def test_print_plan_writes_nothing(self, tmp_path, capsys):
        out = tmp_path / "plan.json"
        rc = main(["--print-plan", "--out", str(out)])

        assert rc == 0
        assert not out.exists(), "--print-plan 不该落盘"
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["groups"]) == 27

    def test_out_writes_the_plan(self, tmp_path):
        out = tmp_path / "plan.json"
        rc = main(["--out", str(out)])

        assert rc == 0
        assert out.is_file()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["root"] == "data/debug/hw_groups_20261021_r3"
        assert len(payload["groups"]) == 27

    def test_default_out_is_under_the_groups_root(self):
        from scripts.slm.build_hw_sweep_plan import _build_parser

        args = _build_parser().parse_args([])
        assert args.out is None, "默认落盘路径由 groups_root 推导, 不是写死的"
        assert args.groups_root == "data/debug/hw_groups_20261021_r3"
        assert args.sweep_root == "data/debug/hw_zernike_sweep_20261021_r3"

    def test_cli_can_move_the_roots(self, tmp_path):
        out = tmp_path / "plan.json"
        rc = main(
            [
                "--out",
                str(out),
                "--groups-root",
                "data/debug/hw_groups_20261022",
                "--sweep-root",
                "data/debug/hw_zernike_sweep_20261022",
            ]
        )

        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["root"] == "data/debug/hw_groups_20261022"
        for group in payload["groups"]:
            for value in (
                arg_value(_args(group), "--out"),
                arg_value(_args(group), "--stage-a-in"),
                arg_value(_args(group), "--gs-phase"),
            ):
                if value is None:
                    continue
                assert "20261021" not in value

    def test_a_bad_argument_exits_with_code_2(self, tmp_path):
        rc = main(["--out", str(tmp_path / "plan.json"), "--orders", "a,b"])
        assert rc == 2
        assert not (tmp_path / "plan.json").exists()


class TestPointTables:
    """Δz 点表与孔径。"""

    def test_dz_tables_match_the_previous_round(self):
        cfg = _cfg()
        plan = build_plan(cfg)

        assert _dz_points(_group(plan, "dz_noise_hw_gs")) == ["0"] * 9

        # 点数照抄上一轮: 硬件组 23 点, 模型组在那 23 点再加 6 个大 Δz 点 = 29 点。
        scan = _dz_points(_group(plan, "dz_scan_hw_gs"))
        assert scan == [
            "-20",
            "-15",
            "-10",
            "-7.5",
            "-5",
            "-3.75",
            "-2.5",
            "-1.5007",
            "-0.7504",
            "-0.3752",
            "-0.1876",
            "0",
            "0.1876",
            "0.3752",
            "0.7504",
            "1.5007",
            "2.5",
            "3.75",
            "5",
            "7.5",
            "10",
            "15",
            "20",
        ]

        model = _dz_points(_group(plan, "dz_scan_model_gs"))
        assert model[: len(scan)] == scan
        assert model[len(scan) :] == ["25", "30", "35", "40", "45", "50"]

    def test_default_tables_are_the_documented_ones(self):
        assert len(DEFAULT_DZ_SCAN_MM) == 23
        assert len(DEFAULT_DZ_SCAN_MODEL_MM) == 29
        assert DEFAULT_DZ_SCAN_MODEL_MM[:23] == DEFAULT_DZ_SCAN_MM
        assert DEFAULT_ORDERS == EXPECTED_ORDERS

    def test_apertures_are_preserved(self):
        """孔径是**预先登记**的: 硬件 10 px / 模型 44 px, 不可互换。"""
        plan = build_plan(_cfg())
        assert (
            arg_value(_args(_group(plan, "dz_scan_hw_gs")), "--target-side-px") == "10"
        )
        assert (
            arg_value(_args(_group(plan, "dz_scan_model_gs")), "--target-side-px")
            == "44"
        )
        assert arg_value(_args(_group(plan, "dz_scan_hw_gs")), "--exposure-ms") == "0.2"
        assert (
            arg_value(_args(_group(plan, "dz_scan_model_gs")), "--exposure-ms") == "0.4"
        )


class TestDriverDryRun:
    """生成的计划必须能被驱动直接读 (退出码 0, 27 组命令全部打印)。"""

    def test_driver_dry_run_accepts_the_generated_plan(self, tmp_path):
        plan_path = tmp_path / "plan.json"
        assert main(["--out", str(plan_path)]) == 0

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/slm/run_hw_sweep_groups.py",
                "--plan",
                str(plan_path),
                "--dry-run",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
        assert "dry-run" in out
        for name in ("focus_calibration", "zernike_stage_a_7", "dz_scan_model_gs"):
            assert name in out
        assert out.count("命令: ") == 27


def test_plan_error_is_a_runtime_error():
    """自检失败要能被调用方当异常接住。"""
    assert issubclass(PlanError, RuntimeError)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
