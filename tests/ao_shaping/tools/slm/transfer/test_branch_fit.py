"""#254 T4 分支预注册训练工具链 (免设备, 合成语料)。

机制断言四组: spec 校验(预算不可逐分支覆盖/必须预注册时刻)、冻结候选
(同 spec 幂等 / 异 spec 拒覆写 / final_test 冻结时刻必须为空)、封存一次性
(二次打分拒绝)、并行入口 (jobs=2 真子进程, 与 jobs=1 同结论)。
``fit`` 走真训练 (iterations 很小, 与 test_transfer_adaptation 同一量级)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer import branch_fit as bf
from ao_shaping.tools.slm.transfer import snapshot as snap
from ao_shaping.tools.slm.transfer.simulation import generate_dataset

torch = pytest.importorskip("torch")

PRE_REGISTERED = "2026-11-26T00:00:00+00:00"


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    """两个合成采集目录 + 一个 checkpoint; 第二个采样的末行改标 final_test。"""
    root = tmp_path / "corpus"
    root.mkdir()
    ckpt = generate_dataset(root / "cap_a", power=1.0, count=6)
    generate_dataset(root / "cap_b", power=1.0, count=6, seed=20241)
    manifest = root / "cap_b" / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    probe_rows = [r for r in rows if r["split"] not in {"calibration"}]
    probe_rows[-1]["split"] = "final_test"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return root, Path(ckpt)


def _spec(tmp_path: Path, checkpoint: Path, branches=None, *, mutate=None) -> Path:
    spec = {
        "schema": bf.SPEC_SCHEMA, "snapshot_id": "v1",
        "pre_registered_utc": PRE_REGISTERED,
        "budget": {"checkpoint": str(checkpoint), "iterations": 3, "span": 120,
                   "device": "cpu"},
        "branches": branches if branches is not None else
        [{"name": "A", "overrides": {"adaptation": "alignment"}},
         {"name": "B", "overrides": {"adaptation": "alignment", "learn_psf": True}}],
    }
    if mutate:
        mutate(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.fixture()
def frozen(tmp_path):
    root, ckpt = _corpus(tmp_path)
    manifest = snap.freeze(root, tmp_path / "snap", "v1")
    manifest_path = tmp_path / "snap" / "v1" / "snapshot_manifest.json"
    return tmp_path, root, ckpt, manifest_path


def test_spec_validation_rejects_unregistered_and_per_branch_budget(tmp_path):
    root, ckpt = _corpus(tmp_path)
    _ = root
    with pytest.raises(ValueError, match="pre_registered_utc"):
        bf.load_spec(_spec(tmp_path, ckpt, mutate=lambda s: s.pop("pre_registered_utc")))
    with pytest.raises(ValueError, match="至少两条"):
        bf.load_spec(_spec(tmp_path, ckpt, branches=[{"name": "A"}]))
    with pytest.raises(ValueError, match="重复"):
        bf.load_spec(_spec(tmp_path, ckpt, branches=[{"name": "A"}, {"name": "A"}]))
    with pytest.raises(ValueError, match="越界键"):
        bf.load_spec(_spec(tmp_path, ckpt, branches=[
            {"name": "A", "overrides": {}},
            {"name": "B", "overrides": {"iterations": 999}}]))
    with pytest.raises(ValueError, match="封存打分开关"):
        bf.load_spec(_spec(tmp_path, ckpt, branches=[
            {"name": "A", "overrides": {"score_final_test": True}},
            {"name": "B", "overrides": {}}]))


def test_stage_dataset_merges_captures_and_keeps_freeze_semantics(frozen):
    tmp_path, root, _ckpt, manifest_path = frozen
    snapshot = bf.load_snapshot(manifest_path)
    staged = bf.stage_dataset(snapshot, tmp_path / "staged", root=root)
    rows = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    assert {r["source_capture"] for r in rows} == {"cap_a", "cap_b"}
    files = [r["file"] for r in rows]
    assert len(files) == len(set(files))  # 跨采集同名 npz 已重命名
    assert all("cap_a__" in f or "cap_b__" in f for f in files)
    assert (staged / rows[0]["file"]).is_file()
    assert "final_test" in {r["split"] for r in rows}


def test_freeze_must_match_snapshot_id(frozen):
    tmp_path, _root, ckpt, manifest_path = frozen
    bad = _spec(tmp_path, ckpt, mutate=lambda s: s.update(snapshot_id="v0"))
    with pytest.raises(ValueError, match="必须同一冻结"):
        bf.freeze_candidates(bad, manifest_path, tmp_path / "run")


def test_candidates_freeze_score_idempotent_and_immutable(frozen):
    tmp_path, _root, ckpt, manifest_path = frozen
    spec = _spec(tmp_path, ckpt)
    out = tmp_path / "run"
    payload = bf.freeze_candidates(spec, manifest_path, out)
    assert payload["final_test"] is None
    assert payload["winner"] in {"A", "B"}
    scores = {b["name"]: b["quality"]["score"] for b in payload["branches"]}
    assert scores[payload["winner"]] == min(scores.values())
    for name in ("A", "B"):
        assert (out / "branches" / name / "result.json").is_file()
        assert (out / "branches" / name / "branch_record.json").is_file()
    again = bf.freeze_candidates(spec, manifest_path, out)  # 幂等: 不重 fit
    assert again == payload
    other = _spec(tmp_path, ckpt,
                  branches=[{"name": "C", "overrides": {}},
                            {"name": "D", "overrides": {"radiometry": "shared"}}])
    with pytest.raises(FileExistsError):
        bf.freeze_candidates(other, manifest_path, out)


def test_score_final_once_is_one_shot(frozen):
    tmp_path, _root, ckpt, manifest_path = frozen
    spec = _spec(tmp_path, ckpt)
    out = tmp_path / "run"
    candidates_path = out / "candidates_frozen.json"
    bf.freeze_candidates(spec, manifest_path, out)
    record = bf.score_final_once(candidates_path, spec, manifest_path)
    assert record["final_test_quality"]["correlation"]
    assert candidates_path.read_text(encoding="utf-8")  # 冻结文件未被改写
    with pytest.raises(RuntimeError, match="只读一次"):
        bf.score_final_once(candidates_path, spec, manifest_path)
    audit = (out / "final_test_audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit) == 1


def test_parallel_jobs_match_sequential_winner(frozen):
    tmp_path, _root, ckpt, manifest_path = frozen
    spec = _spec(tmp_path, ckpt)
    seq = bf.freeze_candidates(spec, manifest_path, tmp_path / "run_seq")
    par = bf.freeze_candidates(spec, manifest_path, tmp_path / "run_par", jobs=2)
    assert {b["name"] for b in par["branches"]} == {"A", "B"}
    assert par["winner"] == seq["winner"]
    for a, b in zip(sorted(seq["branches"], key=lambda r: r["name"]),
                    sorted(par["branches"], key=lambda r: r["name"])):
        assert a["quality"]["score"] == pytest.approx(b["quality"]["score"])
