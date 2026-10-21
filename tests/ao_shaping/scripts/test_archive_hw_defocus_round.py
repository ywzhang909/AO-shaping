"""Tests for the per-round hardware-archive script.

``scripts/slm/archive_hw_defocus_round.py`` turns one round of hardware
collection into a **snapshot** under
``report/shape_trials/hw_defocus_zernike/artifacts/<round>/``.

Why it exists
-------------
The first two rounds were archived **by hand**: one-off, irreproducible, and
the second round happened to be archived *after* the third round's first
attempt had already overwritten eleven of its group directories — so what
landed in the archive was "whatever is on disk now", not the second round's
original data. From this round on the archive is produced by a script.

What this suite pins down
-------------------------
* the archive **layout** matches the existing convention (group directories
  keep their names, so the archive maps one-to-one onto the data root);
* the **big** files (per-frame ``.npz``, phase ``.npy``, ``stage_a.npz``) are
  *not* archived — they are 17–56 MB each and stay in ``data/debug/``;
* **missing groups are reported**, never silently skipped: a missing group
  means that group did not run in that round, and hiding that fact is the
  exact failure the hand-made archive produced;
* an existing archive is **not overwritten** without ``--force`` — a snapshot
  that can be silently rewritten is not a snapshot.

No hardware is touched and no real report directory is written: every test
builds its own tree under ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.slm.archive_hw_defocus_round import (  # noqa: E402
    ArchivePlan,
    build_plan,
    main,
    repo_rel,
)

ROUND = "round3_test"


# ---------------------------------------------------------------------------
# Fake data root
# ---------------------------------------------------------------------------
def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_bytes(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


def _probe_group(root: Path, name: str) -> Path:
    """A defocus/noise/far-field group: the three-piece record plus a frame."""
    group = root / name
    _write(group / "measurements.json", '[{"index": 0, "dz_mm": 0.0}]')
    _write(group / "measurements.csv", "index,dz_mm\n0,0.0\n")
    _write(group / "run_config.json", '{"target_side_px": 10}')
    _write_bytes(group / "frame_000.npz", 4096)
    _write_bytes(group / "phase_nmax3.npy", 2048)
    return group


def _zernike_group(root: Path, name: str, n_max: int) -> Path:
    """A per-order Zernike group: summary + SPGD csv + before/after png."""
    group = root / name
    _write(group / "summary.json", '{"orders": [{"n_max": %d}]}' % n_max)
    _write(group / f"spgd_nmax{n_max}.csv", "epoch,cv\n0,1.0\n")
    _write_bytes(group / f"before_nmax{n_max}.png", 128)
    _write_bytes(group / f"after_nmax{n_max}.png", 128)
    _write_bytes(group / f"phase_nmax{n_max}.npy", 2048)
    return group


def _make_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a fake round: groups root + sweep root + focus evidence + plan."""
    groups_root = tmp_path / "data" / "debug" / "hw_groups_20261021_r3"
    sweep_root = tmp_path / "data" / "debug" / "hw_zernike_sweep_20261021_r3"

    _probe_group(groups_root, "dz_scan_hw_gs")
    _probe_group(groups_root, "dz_noise_hw_gs")
    _probe_group(groups_root, "zernike_farfield_nmax3")
    _write(groups_root / "manifest.json", '[{"group": "dz_scan_hw_gs"}]')
    _write(groups_root / "groups_manifest.json", '{"groups": []}')

    _zernike_group(sweep_root, "zernike_order_nmax3", 3)
    _zernike_group(sweep_root, "zernike_order_nmax4", 4)

    focus = groups_root / "focus_calibration"
    _write(focus / "slm1_20261021_120000.json", '{"best_diopter": 0.58}')

    plan = {
        "root": "data/debug/hw_groups_20261021_r3",
        "groups": [
            {
                "group": "dz_scan_hw_gs",
                "args": ["--execute", "--out-root", str(groups_root)],
            },
            {
                "group": "dz_noise_hw_gs",
                "args": ["--execute", "--out-root", str(groups_root)],
            },
            {
                "group": "zernike_farfield_nmax3",
                "args": ["--execute", "--out-root", str(groups_root)],
            },
            {
                "group": "zernike_order_nmax3",
                "args": ["--out", str(sweep_root)],
            },
            {
                "group": "zernike_order_nmax4",
                "args": ["--out", str(sweep_root)],
            },
            {
                # Planned but never produced -> must be reported as missing.
                "group": "dz_scan_model_gs",
                "args": ["--execute", "--out-root", str(groups_root)],
            },
        ],
    }
    _write(groups_root / "plan.json", json.dumps(plan, ensure_ascii=False))

    return {
        "groups_root": groups_root,
        "sweep_root": sweep_root,
        "focus": focus,
        "plan": groups_root / "plan.json",
        "out": tmp_path / "report" / "artifacts",
    }


def _build(tmp_path: Path) -> ArchivePlan:
    tree = _make_tree(tmp_path)
    return build_plan(
        round_name=ROUND,
        groups_root=tree["groups_root"],
        sweep_root=tree["sweep_root"],
        focus_dir=tree["focus"],
        out_root=tree["out"],
        plan_path=tree["plan"],
        archive_plan=True,
    )


def _run(tree: dict[str, Path], *extra: str) -> int:
    """Invoke the CLI against the fake tree."""
    argv = [
        "--round",
        ROUND,
        "--groups-root",
        str(tree["groups_root"]),
        "--sweep-root",
        str(tree["sweep_root"]),
        "--focus-dir",
        str(tree["focus"]),
        "--plan",
        str(tree["plan"]),
        "--out",
        str(tree["out"]),
        *extra,
    ]
    return main(argv)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
class TestLayout:
    """归档目录结构与既有惯例一致。"""

    def test_group_directory_names_are_preserved(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        dests = {repo_rel(entry.dest) for entry in plan.entries}
        assert any("dz_scan_hw_gs" in d for d in dests)
        assert any("zernike_order_nmax3" in d for d in dests)

    def test_measurements_triplet_is_archived(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        names = {entry.dest.name for entry in plan.entries}
        assert {"measurements.json", "measurements.csv", "run_config.json"} <= names

    def test_zernike_records_are_archived(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        names = {entry.dest.name for entry in plan.entries}
        assert {"summary.json", "spgd_nmax3.csv", "before_nmax3.png", "after_nmax3.png"} <= names

    def test_focus_evidence_is_archived(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        assert any(entry.dest.name == "slm1_20261021_120000.json" for entry in plan.entries)

    def test_plan_is_archived_by_default(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        assert plan.plan_archived
        assert any(entry.dest.name == "plan.json" for entry in plan.entries)

    def test_plan_can_be_left_out(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        plan = build_plan(
            round_name=ROUND,
            groups_root=tree["groups_root"],
            sweep_root=tree["sweep_root"],
            focus_dir=tree["focus"],
            out_root=tree["out"],
            plan_path=tree["plan"],
            archive_plan=False,
        )
        assert not plan.plan_archived
        assert not any(entry.dest.name == "plan.json" for entry in plan.entries)

    def test_archive_lands_under_the_requested_round_name(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        assert (tree["out"] / ROUND).is_dir()


# ---------------------------------------------------------------------------
# Big files
# ---------------------------------------------------------------------------
class TestBigFilesStayOut:
    """体积大的逐帧/相位文件不入库。"""

    def test_npz_and_npy_are_not_archived(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        archived = {entry.src.name for entry in plan.entries}
        assert "frame_000.npz" not in archived
        assert "phase_nmax3.npy" not in archived

    def test_big_files_are_counted_as_skipped(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        skipped = {path.name for path, _ in plan.skipped}
        assert "frame_000.npz" in skipped
        assert "phase_nmax3.npy" in skipped

    def test_nothing_large_reaches_the_disk(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        for path in (tree["out"] / ROUND).rglob("*"):
            if path.is_file():
                assert path.suffix not in (".npz", ".npy"), f"大文件被归档了: {path}"


# ---------------------------------------------------------------------------
# Missing groups
# ---------------------------------------------------------------------------
class TestMissingGroups:
    """缺组必须被点名, 不能静默跳过。"""

    def test_missing_group_is_reported(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        assert "dz_scan_model_gs" in plan.missing_groups

    def test_missing_group_is_printed(self, tmp_path: Path, capsys) -> None:
        tree = _make_tree(tmp_path)
        _run(tree)
        out = capsys.readouterr().out
        assert "dz_scan_model_gs" in out
        assert "缺组" in out

    def test_present_groups_are_not_reported_missing(self, tmp_path: Path) -> None:
        plan = _build(tmp_path)
        assert "dz_scan_hw_gs" not in plan.missing_groups
        assert "zernike_order_nmax4" not in plan.missing_groups

    def test_missing_group_is_recorded_in_the_manifest(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        manifest = json.loads(
            (tree["out"] / ROUND / "archive_manifest.json").read_text(encoding="utf-8")
        )
        assert "dz_scan_model_gs" in manifest["missing_groups"]


# ---------------------------------------------------------------------------
# Overwrite guard
# ---------------------------------------------------------------------------
class TestOverwriteGuard:
    """归档是快照, 不能被悄悄改写。"""

    def test_second_archive_refuses(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        assert _run(tree, "--write") == 2

    def test_refusal_leaves_the_first_archive_intact(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        marker = tree["out"] / ROUND / "marker.txt"
        marker.write_text("keep me", encoding="utf-8")
        assert _run(tree, "--write") == 2
        assert marker.read_text(encoding="utf-8") == "keep me"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree, "--write") == 0
        marker = tree["out"] / ROUND / "marker.txt"
        marker.write_text("stale", encoding="utf-8")
        assert _run(tree, "--write", "--force") == 0
        # 覆盖 = 删掉旧快照再重建: 上一版多出来的文件不该留在归档里。
        assert not marker.exists()
        assert (tree["out"] / ROUND / "archive_manifest.json").is_file()

    def test_round_name_may_not_escape_the_output_root(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        code = main(
            [
                "--round",
                "../escaped",
                "--groups-root",
                str(tree["groups_root"]),
                "--sweep-root",
                str(tree["sweep_root"]),
                "--out",
                str(tree["out"]),
                "--write",
            ]
        )
        assert code == 2
        assert not (tree["out"].parent / "escaped").exists()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
class TestDryRun:
    """默认只演练, 不碰文件系统。"""

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        tree = _make_tree(tmp_path)
        assert _run(tree) == 0
        assert not (tree["out"] / ROUND).exists()

    def test_dry_run_reports_the_plan(self, tmp_path: Path, capsys) -> None:
        tree = _make_tree(tmp_path)
        _run(tree)
        out = capsys.readouterr().out
        assert "演练" in out
        assert "dz_scan_hw_gs" in out

    def test_an_empty_round_is_refused(self, tmp_path: Path, capsys) -> None:
        """空轮不许归档: 空壳快照读起来像「跑过了但没留下」。

        真相是「这轮根本没跑」—— 两者的区别对读报告的人很重要。
        """
        groups_root = tmp_path / "data" / "debug" / "hw_groups_empty"
        sweep_root = tmp_path / "data" / "debug" / "hw_sweep_empty"
        groups_root.mkdir(parents=True)
        sweep_root.mkdir(parents=True)
        out = tmp_path / "out"
        code = main(
            [
                "--round",
                ROUND,
                "--groups-root",
                str(groups_root),
                "--sweep-root",
                str(sweep_root),
                "--out",
                str(out),
                "--write",
            ]
        )
        assert code == 2
        assert not (out / ROUND).exists()
        assert "空" in capsys.readouterr().err

    def test_a_round_with_only_the_plan_is_still_refused(self, tmp_path: Path) -> None:
        """计划 JSON 与索引文件**不算**组记录。

        拿它们凑数会让「一个组都没跑」的这一轮看起来像归档成功了 ——
        那正是空壳快照的由来。
        """
        groups_root = tmp_path / "data" / "debug" / "hw_groups_only_plan"
        sweep_root = tmp_path / "data" / "debug" / "hw_sweep_only_plan"
        groups_root.mkdir(parents=True)
        sweep_root.mkdir(parents=True)
        _write(groups_root / "plan.json", json.dumps({"groups": []}))
        _write(groups_root / "manifest.json", "[]")
        out = tmp_path / "out"
        code = main(
            [
                "--round",
                ROUND,
                "--groups-root",
                str(groups_root),
                "--sweep-root",
                str(sweep_root),
                "--out",
                str(out),
                "--write",
            ]
        )
        assert code == 2
        assert not (out / ROUND).exists()

    def test_a_focus_only_round_is_archived(self, tmp_path: Path) -> None:
        """只跑了焦面标定的那一轮**有内容**, 不该被空轮守卫拦下。"""
        groups_root = tmp_path / "data" / "debug" / "hw_groups_focus"
        sweep_root = tmp_path / "data" / "debug" / "hw_sweep_focus"
        (groups_root / "focus_calibration").mkdir(parents=True)
        sweep_root.mkdir(parents=True)
        _write(groups_root / "focus_calibration" / "slm1_20261021_120000.json", "{}")
        out = tmp_path / "out"
        code = main(
            [
                "--round",
                ROUND,
                "--groups-root",
                str(groups_root),
                "--sweep-root",
                str(sweep_root),
                "--out",
                str(out),
                "--write",
            ]
        )
        assert code == 0
        assert (out / ROUND / "focus_calibration" / "slm1_20261021_120000.json").is_file()


# ---------------------------------------------------------------------------
# No hardware
# ---------------------------------------------------------------------------
def test_the_script_does_not_import_any_driver() -> None:
    """归档是纯文件搬运: 不许牵进 ``ao_shaping`` (会拖进驱动包)。"""
    import ast
    import inspect

    import scripts.slm.archive_hw_defocus_round as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("ao_shaping") for name in imported), imported


def test_repo_relative_paths_use_forward_slashes() -> None:
    """输出里的路径必须是仓库相对 + 正斜杠。"""
    text = repo_rel(REPO_ROOT / "data" / "debug" / "hw_groups_20261021_r3")
    assert text == "data/debug/hw_groups_20261021_r3"
    assert "\\" not in text


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
