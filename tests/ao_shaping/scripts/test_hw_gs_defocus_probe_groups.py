"""Tests for the defocus probe's one-variable-group-per-invocation mode.

The probe (``scripts/slm/hw_gs_defocus_probe.py``) sweeps one variable group:
the CCD defocus distance ``dz_mm``. Its grouping surface is:

* ``--group NAME`` -- the group's identity (default ``main``); the artefacts land
  in ``<root>/<group>/`` and the root keeps a ``manifest.json`` indexing them.
* ``--group-meta KEY=VALUE`` -- the group's **fixed** parameters, recorded
  verbatim.
* ``--replot`` -- accepts the root **or** a group directory.

The dry-run guarantee ("no device is imported, let alone opened") must hold
**with and without** ``--group``: the grouping surface is on the same code path
as the plan printout, so a stray import there would break the guarantee.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts._common.group_manifest import (
    group_record,
    load_manifest,
    parse_group_meta,
    update_manifest,
)
from scripts.slm.hw_gs_defocus_probe import (
    GROUP_VARIABLE,
    build_parser,
    find_measurement_dirs,
    group_fields,
    group_output_dir,
)

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts" / "slm" / "hw_gs_defocus_probe.py"


def _run_probe(*args: str) -> subprocess.CompletedProcess[str]:
    """跑一次探针 (子进程, 拿真实退出码)。"""
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


class TestDryRunGuarantee:
    """dry-run 的保证, 带不带 ``--group`` 都得成立。"""

    def test_dry_run_without_group(self):
        proc = _run_probe("--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert "dry-run 保证通过" in proc.stdout

    def test_dry_run_with_group(self):
        proc = _run_probe("--dry-run", "--group", "dz_scan")
        assert proc.returncode == 0, proc.stderr
        assert "dry-run 保证通过" in proc.stdout
        assert "dz_scan" in proc.stdout

    def test_dry_run_prints_the_group_and_its_variable(self):
        proc = _run_probe(
            "--dry-run",
            "--group",
            "dz_scan",
            "--group-meta",
            "exposure=0.2",
        )
        assert proc.returncode == 0, proc.stderr
        assert "dz_scan" in proc.stdout
        assert GROUP_VARIABLE in proc.stdout
        assert "exposure=0.2" in proc.stdout

    def test_malformed_group_meta_is_rejected(self):
        proc = _run_probe("--dry-run", "--group-meta", "no_equals_sign")
        assert proc.returncode != 0
        assert "KEY=VALUE" in (proc.stderr + proc.stdout)


class TestGroupParsing:
    def test_group_defaults_to_main(self):
        args = build_parser().parse_args([])
        assert args.group == "main"

    def test_group_meta_is_parsed_into_pairs(self):
        args = build_parser().parse_args(
            ["--group", "dz_scan", "--group-meta", "exposure=0.2",
             "--group-meta", "phase=hardware_gs"]
        )
        assert args.group == "dz_scan"
        assert parse_group_meta(args.group_meta_pairs) == {
            "exposure": "0.2",
            "phase": "hardware_gs",
        }

    def test_group_fields_carry_the_variable(self):
        fields = group_fields("dz_scan", {"exposure": "0.2"})
        assert fields["group"] == "dz_scan"
        assert fields["variable"] == GROUP_VARIABLE == "dz_mm"
        assert fields["group_meta"] == {"exposure": "0.2"}


class TestGroupLayout:
    def test_group_output_dir_is_root_plus_group(self, tmp_path):
        assert group_output_dir(tmp_path, "dz_scan") == tmp_path / "dz_scan"

    def test_manifest_indexes_the_group(self, tmp_path):
        root = tmp_path / "root"
        (root / "dz_scan").mkdir(parents=True)
        update_manifest(
            root,
            group_record(
                group="dz_scan",
                variable=GROUP_VARIABLE,
                meta={"exposure": "0.2"},
                n_rows=11,
            ),
        )
        entries = load_manifest(root)
        assert len(entries) == 1
        assert entries[0]["group"] == "dz_scan"
        assert entries[0]["variable"] == "dz_mm"
        assert entries[0]["meta"] == {"exposure": "0.2"}
        assert entries[0]["dir"] == "dz_scan"
        assert entries[0]["n_rows"] == 11
        assert entries[0]["created_utc"]

    def test_rerunning_a_group_replaces_its_entry(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        for n_rows in (3, 7):
            update_manifest(
                root,
                group_record(
                    group="dz_scan", variable=GROUP_VARIABLE, n_rows=n_rows
                ),
            )
        entries = load_manifest(root)
        assert len(entries) == 1
        assert entries[0]["n_rows"] == 7


class TestReplotResolution:
    """``--replot`` 接受根目录或组目录。"""

    @staticmethod
    def _make_group(root: Path, name: str) -> Path:
        group_dir = root / name
        group_dir.mkdir(parents=True)
        (group_dir / "measurements.json").write_text("[]", encoding="utf-8")
        return group_dir

    def test_group_directory_is_accepted_directly(self, tmp_path):
        group_dir = self._make_group(tmp_path, "dz_scan")
        assert find_measurement_dirs(group_dir) == [group_dir]

    def test_root_directory_is_resolved_through_the_manifest(self, tmp_path):
        root = tmp_path / "root"
        group_dir = self._make_group(root, "dz_scan")
        update_manifest(
            root,
            group_record(group="dz_scan", variable=GROUP_VARIABLE, n_rows=1),
        )
        assert find_measurement_dirs(root) == [group_dir]

    def test_root_directory_falls_back_to_scanning_subdirs(self, tmp_path):
        """清单还没写 (跑到一半崩了) 时, 扫描子目录也要找得到。"""
        root = tmp_path / "root"
        group_dir = self._make_group(root, "dz_scan")
        assert find_measurement_dirs(root) == [group_dir]

    def test_empty_root_yields_nothing(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert find_measurement_dirs(root) == []


@pytest.mark.parametrize("with_group", [False, True])
def test_dry_run_never_imports_the_driver_package(with_group: bool):
    """探针自己会断言这件事; 这里再从外面看一眼, 免得断言被改松。"""
    args = ["--dry-run"]
    if with_group:
        args += ["--group", "dz_scan"]
    proc = _run_probe(*args)
    assert proc.returncode == 0, proc.stderr
    assert "ao_shaping.drivers" not in proc.stderr


def test_manifest_json_is_a_list_of_group_entries(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    update_manifest(
        root,
        group_record(group="a", variable=GROUP_VARIABLE, n_rows=1),
    )
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert set(raw[0]) >= {"group", "variable", "meta", "dir", "n_rows", "created_utc"}
