"""#254 T3 快照冻结器 (免设备)。

覆盖清单承重的四件事: 哈希可复核 (篡改必须被 verify 抓到)、三层口径校验
(rounds 整轮同层的违反必须拒冻)、不可覆写 (同 id 不同内容报错, 字节相同幂等)、
``--plan-only`` 零副作用。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import snapshot as snap


def _capture(root: Path, name: str, *, scheme="legacy", rows: list[tuple[str, str, int | None]] | None = None) -> Path:
    """一个最小采集目录: 每行一个 npz + manifest.json。"""
    capture = root / name
    capture.mkdir(parents=True)
    manifest = []
    for i, (label, split, rnd) in enumerate(rows or [("probe_a", "train", None),
                                                     ("probe_b", "validation", None)]):
        file = f"sample_{i:04d}.npz"
        np.savez_compressed(capture / file, frame=np.zeros((4, 4), np.uint8))
        row = dict(file=file, label=label, split=split, exposure_ms=1.0)
        if scheme != "legacy":
            row["split_scheme"] = scheme
            row["probe_round"] = rnd
        manifest.append(row)
    (capture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return capture


def test_freeze_writes_manifest_with_hashes_and_layers(tmp_path):
    root = tmp_path / "corpus"
    _capture(root, "cap_a")
    _capture(root, "cap_b", rows=[("ramp_x_64_1", "calibration", None),
                                  ("probe_c", "train", None),
                                  ("probe_d", "validation", None),
                                  ("probe_e", "final_test", None)])
    manifest = snap.freeze(root, tmp_path / "snap", "v1")
    assert manifest["snapshot_id"] == "v1"
    assert [c["name"] for c in manifest["captures"]] == ["cap_a", "cap_b"]
    assert manifest["layer_counts"]["final_test"] == 1
    assert manifest["sealed_layers"] == ["final_test"]
    assert len(manifest["root_hash"]) == 64
    entry = manifest["captures"][0]
    assert entry["files"][0]["sha256"]
    assert snap.verify(tmp_path / "snap" / "v1" / "snapshot_manifest.json")["ok"]


def test_plan_only_writes_nothing(tmp_path):
    root = tmp_path / "corpus"
    _capture(root, "cap_a")
    out = tmp_path / "snap"
    manifest = snap.freeze(root, out, "v1", plan_only=True)
    assert manifest["captures"]
    assert not out.exists()


def test_round_consistency_violation_refuses_freeze(tmp_path):
    root = tmp_path / "corpus"
    _capture(root, "cap_r", scheme="rounds", rows=[
        ("probe_000_a", "train", 0), ("probe_006_a", "validation", 0),  # 同轮跨层
        ("probe_001_a", "train", 0), ("probe_002_a", "validation", 0)])
    with pytest.raises(ValueError, match="整轮必须同层"):
        snap.freeze(root, tmp_path / "snap", "v1")


def test_missing_layer_and_unregistered_file_refuse_freeze(tmp_path):
    root = tmp_path / "corpus"
    cap = _capture(root, "cap_no_val", rows=[("probe_a", "train", None)])
    with pytest.raises(ValueError, match="缺必需层"):
        snap.freeze(root, tmp_path / "snap", "v1")
    # 补上 validation 行 → 再塞一个清单外的 npz → 仍拒
    rows = json.loads((cap / "manifest.json").read_text())
    np.savez_compressed(cap / "sample_0001.npz", frame=np.zeros((4, 4), np.uint8))
    rows.append(dict(file="sample_0001.npz", label="probe_b", split="validation",
                     exposure_ms=1.0))
    (cap / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    np.savez_compressed(cap / "sample_9999.npz", frame=np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="清单未登记"):
        snap.freeze(root, tmp_path / "snap", "v1")


def test_refreeze_same_id_is_idempotent_then_immutable(tmp_path):
    root = tmp_path / "corpus"
    _capture(root, "cap_a")
    snap.freeze(root, tmp_path / "snap", "v1")
    again = snap.freeze(root, tmp_path / "snap", "v1")  # 字节相同 → 幂等
    path = tmp_path / "snap" / "v1" / "snapshot_manifest.json"
    first_text = path.read_text(encoding="utf-8")
    assert again["root_hash"] == json.loads(first_text)["root_hash"]
    # 语料变了再冻同名 → 拒绝覆写
    _capture(root, "cap_b")
    with pytest.raises(FileExistsError):
        snap.freeze(root, tmp_path / "snap", "v1")


def test_verify_catches_content_tampering_and_moves(tmp_path):
    root = tmp_path / "corpus"
    cap = _capture(root, "cap_a")
    snap.freeze(root, tmp_path / "snap", "v1")
    manifest_path = tmp_path / "snap" / "v1" / "snapshot_manifest.json"
    sample = sorted(cap.glob("sample_*.npz"))[0]
    sample.write_bytes(sample.read_bytes() + b"x")
    result = snap.verify(manifest_path)
    assert not result["ok"] and any("内容哈希漂移" in p for p in result["problems"])
    # 搬移语料 (内容不动): 按旧绝对路径复核失败, 按 --root 重定位成功
    (tmp_path / "moved").mkdir()
    moved = tmp_path / "moved" / "cap_a"
    sample.write_bytes(sample.read_bytes()[:-1])  # 还原
    cap.rename(moved)
    assert not snap.verify(manifest_path)["ok"]
    assert snap.verify(manifest_path, root=tmp_path / "moved")["ok"]
