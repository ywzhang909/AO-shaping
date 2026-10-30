"""检查点索引：登记、幂等、扫描回填与哈希校验 (免设备)。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer.checkpoint_registry import (
    CHECKPOINT_NAME,
    INDEX_NAME,
    register_session,
    scan_checkpoint_index,
    verify_checkpoint,
)
from ao_shaping.tools.slm.transfer.optical_session import save_fingerprint


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _session(root: Path) -> Path:
    """一个带两份检查点的会话目录 (单形状, 有拟合质量与指纹)。"""
    _write(root / "journal.json", [{"stage": "plan", "status": "complete",
                                    "plan": {"shapes": ["square"]}}])
    _write(root / "shape" / "square" / "capture" / "manifest.json", [{"peak": 100.0}])
    body = {"ramp_matrix": [[1, 2], [3, 4]], "grid_size": [320, 320]}
    _write(root / "nominal" / CHECKPOINT_NAME, body)
    _write(root / "model640" / CHECKPOINT_NAME, body)
    _write(root / "fit" / "result.json",
           {"metrics": [{"cv_measured": 0.2, "correlation": 0.9},
                        {"cv_measured": 0.4, "correlation": 0.7}]})
    save_fingerprint({"window_span": 820.0}, root)
    return root


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_kind(entries: list[dict]) -> dict[str, dict]:
    return {str(entry["kind"]): entry for entry in entries}


def test_register_session_records_every_checkpoint(tmp_path):
    root = _session(tmp_path / "session")
    entries = register_session(root)

    assert len(entries) == 2
    record = _by_kind(entries)["model640"]
    file = root / "model640" / CHECKPOINT_NAME
    assert record["file"] == str(file)
    assert record["path"] == str(file.parent)
    assert record["session"] == str(root.resolve())
    assert record["shape"] == "square"
    assert record["stale"] is False
    assert record["sha256"] == _digest(file)
    assert record["cv"] == pytest.approx(0.3)
    assert record["fit_quality"] == pytest.approx(0.8)
    assert record["fingerprint"]["window_span"] == 820.0
    assert record["created_at"]


def test_the_index_is_written_next_to_the_session(tmp_path):
    root = _session(tmp_path / "session")
    entries = register_session(root)

    index = json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))
    assert [entry["file"] for entry in index] == [entry["file"] for entry in entries]


def test_registering_twice_is_idempotent(tmp_path):
    root = _session(tmp_path / "session")
    first = register_session(root)
    second = register_session(root)

    keys = ("file", "sha256", "created_at", "kind")
    assert [[entry[key] for key in keys] for entry in second] == \
           [[entry[key] for key in keys] for entry in first]


def test_a_stale_mark_survives_a_rescan(tmp_path):
    root = _session(tmp_path / "session")
    entries = register_session(root)
    marked = dict(entries[0], stale=True)
    (root / INDEX_NAME).write_text(json.dumps([marked, entries[1]]), encoding="utf-8")

    again = _by_kind(register_session(root))

    assert again[str(marked["kind"])]["stale"] is True
    other = "nominal" if marked["kind"] == "model640" else "model640"
    assert again[other]["stale"] is False


def test_a_session_without_checkpoints_writes_no_index(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    assert register_session(root) == []
    assert not (root / INDEX_NAME).exists()


def test_a_changed_checkpoint_is_marked_stale_by_the_scan(tmp_path):
    root = _session(tmp_path / "session")
    register_session(root)
    touched = root / "model640" / CHECKPOINT_NAME
    touched.write_text(json.dumps({"ramp_matrix": [[9, 9]]}), encoding="utf-8")

    rows = _by_kind(scan_checkpoint_index(root))

    assert rows["model640"]["stale"] is True
    assert rows["model640"]["sha256_ok"] is False
    assert rows["nominal"]["stale"] is False
    assert rows["nominal"]["sha256_ok"] is True


def test_a_deleted_checkpoint_is_marked_stale(tmp_path):
    root = _session(tmp_path / "session")
    register_session(root)
    (root / "nominal" / CHECKPOINT_NAME).unlink()

    rows = _by_kind(scan_checkpoint_index(root))

    assert rows["nominal"]["stale"] is True
    assert rows["nominal"]["sha256_ok"] is False


def test_verify_checkpoint_reports_and_can_persist_the_verdict(tmp_path):
    root = _session(tmp_path / "session")
    entries = _by_kind(register_session(root))
    good, bad = root / "nominal" / CHECKPOINT_NAME, root / "model640" / CHECKPOINT_NAME
    assert verify_checkpoint(entries["nominal"]) is True
    assert verify_checkpoint(good) is True

    bad.write_text(json.dumps({"ramp_matrix": [[7, 7]]}), encoding="utf-8")
    assert verify_checkpoint(entries["model640"]) is False
    assert verify_checkpoint(bad) is False

    # 指定索引时把结论回写: 过期是结论, 不靠下一次扫描重新发现.
    assert verify_checkpoint(entries["model640"], index=root / INDEX_NAME) is False
    index = {entry["kind"]: entry for entry in
             json.loads((root / INDEX_NAME).read_text(encoding="utf-8"))}
    assert index["model640"]["stale"] is True
    assert index["model640"]["sha256_ok"] is False
    assert index["nominal"]["stale"] is False


def test_a_session_that_was_never_registered_is_synthesised_on_scan(tmp_path):
    root = tmp_path / "legacy"
    checkpoint = root / "model640" / CHECKPOINT_NAME
    _write(checkpoint, {"ramp_matrix": [[1, 2]]})

    rows = scan_checkpoint_index(root)

    assert len(rows) == 1
    assert rows[0]["file"] == str(checkpoint)
    assert rows[0]["sha256"] == _digest(checkpoint)
    assert rows[0]["stale"] is False
    assert rows[0]["sha256_ok"] is True
    # 扫描是只读动作: 没登记过的目录不会凭空多出一份索引.
    assert not (root / INDEX_NAME).exists()


def test_the_scan_can_be_limited_and_verified_off(tmp_path):
    root = tmp_path / "campaign"
    _session(root / "one")
    _write(root / "two" / "model640" / CHECKPOINT_NAME, "tampered")

    all_rows = scan_checkpoint_index(root, verify=False)

    assert len(all_rows) == 3
    assert all("sha256_ok" not in row for row in all_rows)
    assert len(scan_checkpoint_index(root, limit=2)) == 2
