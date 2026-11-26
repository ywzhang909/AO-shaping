"""#254 T4: spec 预注册的 N 分支并行训练 + 统一打分 + 候选冻结。

分支语义与 #254 的防泄漏纪律同源: 预注册计划文档 (plan.md, 仅存采集侧机器)
定义分支集合与统一预算, 本模块**不硬编码任何分支名单** —— 分支集合是输入
(JSON spec), 工具链只保证机制:

* **同预算强制**: ``iterations`` / ``span`` / 数据集只存在于 spec 的 ``budget``
  段, 分支只能覆盖 :data:`BRANCH_OVERRIDE_WHITELIST` 里的模型旋钮
  (learn_psf / learn_geometry / adaptation / radiometry / ...) —— 结构性地
  杜绝「某分支多给训练步数」这类不可比比较 (11-24 PSF 报告的局限 2 的教训)。
* **晋级只读 selection 层**: 每分支 fit 后按 :func:`.adaptation.validation_quality`
  (仅 ``split == validation`` 行)打分, 封存行在 :func:`.fit.load_dataset` 层面
  就不进训练也不进打分。
* **候选冻结不可覆写 + final_test 一次性**: ``candidates_frozen.json`` 写入后
  字节级幂等; 封存打分只在冻结**之后**运行一次, 审计追加进
  ``final_test_audit.jsonl`` —— 同哈希二次打分会被拒, 这是「final_test 只读
  一次」的机制实现, 不是口头约定。
* **并行**: ``--jobs>1`` 对每分支 spawn 独立子进程 (torch 进程态不共享),
  聚合仍走同一冻结路径; ``--jobs=1`` 顺序跑, 结果逐字节相同 (打分与
  执行顺序无关是聚合的前提, 有测试钉)。

数字孪生先行: 采集语料与 plan.md 都在持有数据的采集侧机器上; 本仓库用合成
语料做全流程回归 (``tests/ao_shaping/tools/slm/transfer/test_branch_fit.py``,
含一次真 ``fit.fit`` 端到端)。在采集侧执行时: 先 ``snapshot freeze --plan-only``
核对划分, 再 ``freeze``, 最后 ``branch_fit freeze --spec ... --manifest ...``。

spec 最小形状::

    {"schema": "transfer-branch-spec-v1",
     "snapshot_id": "v1", "pre_registered_utc": "...",
     "budget": {"checkpoint": "ckpt.json", "iterations": 100, "span": 160,
                 "device": "auto"},
     "branches": [{"name": "A", "overrides": {}},
                  {"name": "B", "overrides": {"learn_psf": true}}, ...]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from ao_shaping.tools.slm.transfer import fit as fit_module
from ao_shaping.tools.slm.transfer.adaptation import final_test_quality, validation_quality
from ao_shaping.tools.slm.transfer.capture import clock
from ao_shaping.tools.slm.transfer.snapshot import SCHEMA as SNAPSHOT_SCHEMA

SPEC_SCHEMA = "transfer-branch-spec-v1"
CANDIDATES_SCHEMA = "transfer-candidates-v1"

#: 分支允许覆盖的模型旋钮。预算 (iterations/span/checkpoint/dataset) 不在列 ——
#: 分支间可比性靠「预算只有 budget 一份」保证, 不靠自觉。
BRANCH_OVERRIDE_WHITELIST = frozenset({
    "adaptation", "amplitude_center_lr", "learn_geometry", "learn_psf",
    "psf_sigma_init", "psf_sigma_max", "radiometry",
})


@dataclass
class BranchSpec:
    path: Path
    sha256: str
    snapshot_id: str
    pre_registered_utc: str
    budget: dict
    branches: list = field(default_factory=list)


def load_spec(path: Path) -> BranchSpec:
    """读取并校验分支预注册 spec; 校验失败抛 ValueError (全量问题列表)。"""
    path = Path(path)
    raw_bytes = path.read_bytes()
    spec = json.loads(raw_bytes.decode("utf-8"))
    problems: list[str] = []
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema 应为 {SPEC_SCHEMA!r}, 实为 {spec.get('schema')!r}")
    if not spec.get("snapshot_id"):
        problems.append("缺 snapshot_id (必须钉住 T3 冻结的快照)")
    if not spec.get("pre_registered_utc"):
        problems.append("缺 pre_registered_utc —— 没有预注册时刻的分支集合不算预注册")
    budget = spec.get("budget") or {}
    for key in ("checkpoint", "iterations", "span"):
        if key not in budget:
            problems.append(f"budget 缺 {key!r}")
    if not isinstance(budget.get("iterations"), int) or budget.get("iterations", 0) <= 0:
        problems.append("budget.iterations 必须是正整数")
    names = [b.get("name") for b in spec.get("branches") or []]
    if len(names) < 2:
        problems.append("branches 至少两条 (单分支不需要本工具链)")
    if len(set(names)) != len(names):
        problems.append(f"分支名重复: {names}")
    for branch in spec.get("branches") or []:
        extra = set(branch.get("overrides") or {}) - BRANCH_OVERRIDE_WHITELIST
        if extra:
            problems.append(f"分支 {branch.get('name')!r} 覆盖越界键 {sorted(extra)} "
                            f"(预算类键禁止逐分支覆盖)")
        if "score_final_test" in (branch.get("overrides") or {}):
            problems.append("分支不得自带封存打分开关 (一次性打分走 score-final)")
    if problems:
        raise ValueError("分支 spec 校验未通过:\n  " + "\n  ".join(problems))
    return BranchSpec(path=path, sha256=hashlib.sha256(raw_bytes).hexdigest(),
                      snapshot_id=spec["snapshot_id"],
                      pre_registered_utc=spec["pre_registered_utc"],
                      budget=dict(budget), branches=[dict(b) for b in spec["branches"]])


def load_snapshot(manifest_path: Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"快照清单 schema 不符: {manifest.get('schema')!r}")
    return manifest


def stage_dataset(snapshot: dict, out_dir: Path, root: Path | None = None) -> Path:
    """把快照内多个采集目录合并成一个 fit 可用的数据集目录。

    采集目录之间 ``sample_XXXX.npz`` 文件名可重复, 合并时统一重命名为
    ``<capture>__<file>`` 并改写行内 ``file`` 字段; ``label`` 加同名前缀保持
    可追溯。行字段逐字保留 (split / probe_round / split_scheme 是冻结语义,
    不允许在暂存阶段被稀释)。
    """
    out_dir = Path(out_dir)
    if any(out_dir.glob("sample_*.npz")):
        raise FileExistsError(f"{out_dir} 已有 sample 文件, 拒绝二次暂存")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict] = []
    for entry in snapshot["captures"]:
        capture = Path(root or snapshot["root"]) / entry["name"] if root \
            else Path(entry["dir"])
        rows = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
        frozen_files = {f["name"] for f in entry["files"]}
        for row in rows:
            if row["file"] not in frozen_files:
                raise ValueError(
                    f"{capture.name}: 行 {row['file']!r} 不在冻结清单里 —— "
                    f"语料在冻结后被改动, 拒绝暂存 (先 verify 快照)")
            src = capture / row["file"]
            staged_name = f"{entry['name']}__{row['file']}"
            dest = out_dir / staged_name
            if not dest.is_file():
                try:
                    dest.symlink_to(src.resolve())
                except OSError:  # Windows 无符号链接权限时退化为拷贝 (语义相同)
                    import shutil
                    shutil.copy2(src, dest)
            merged_row = dict(row)
            merged_row["file"] = staged_name
            merged_row["source_capture"] = entry["name"]
            merged.append(merged_row)
    (out_dir / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return out_dir


def _run_fit_for_branch(spec: BranchSpec, overrides: dict, dataset: Path,
                        out: Path) -> dict:
    """跑一个分支的 fit 并返回打分; 单独成函数以便子进程入口复用。"""
    args = SimpleNamespace(
        dataset=Path(dataset),
        checkpoint=Path(spec.budget["checkpoint"]),
        out=Path(out),
        iterations=int(spec.budget["iterations"]),
        span=int(spec.budget["span"]),
        device=spec.budget.get("device", "auto"),
        samples=None,
        train_all=False,
        score_final_test=False,
        **{k: v for k, v in overrides.items()})
    started = time.monotonic()
    fit_module.fit(args)
    result = json.loads((args.out / "result.json").read_text(encoding="utf-8"))
    rows = json.loads((args.out / "provenance.json").read_text(encoding="utf-8"))["rows"]
    return dict(quality=validation_quality(result, rows),
                fit_seconds=round(time.monotonic() - started, 3),
                manifest_sha256=json.loads(
                    (args.out / "provenance.json").read_text(encoding="utf-8")
                )["manifest_sha256"])


def run_single_branch(spec: BranchSpec, branch: dict, dataset: Path, out_root: Path) -> dict:
    """名义入口: 训练单分支并落 ``<out_root>/branches/<name>/``。"""
    out = Path(out_root) / "branches" / branch["name"]
    run = _run_fit_for_branch(spec, dict(branch.get("overrides") or {}), dataset, out)
    record = dict(name=branch["name"],
                  overrides=dict(branch.get("overrides") or {}),
                  out=str(out.resolve()), **run)
    (out / "branch_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _candidates_path(out_root: Path) -> Path:
    return Path(out_root) / "candidates_frozen.json"


def freeze_candidates(spec_path: Path, manifest_path: Path, out_root: Path,
                      jobs: int = 1, root: Path | None = None) -> dict:
    """暂存数据集 → 训练全部分支 → 写不可覆写的 ``candidates_frozen.json``。"""
    spec = load_spec(Path(spec_path))
    snapshot = load_snapshot(Path(manifest_path))
    if spec.snapshot_id != snapshot["snapshot_id"]:
        raise ValueError(f"spec 钉的是 {spec.snapshot_id!r}, 快照是 "
                         f"{snapshot['snapshot_id']!r} —— 必须同一冻结")
    out_root = Path(out_root)
    final_path = _candidates_path(out_root)
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("spec_sha256") == spec.sha256 \
                and existing.get("snapshot_root_hash") == snapshot["root_hash"]:
            return existing  # 幂等: 同 spec + 同快照哈希 = 同一冻结
        raise FileExistsError(f"{final_path} 已存在且属于另一次冻结, 拒绝覆写")
    dataset = stage_dataset(snapshot, out_root / "dataset", root=root)
    if jobs <= 1:
        records = []
        for branch in spec.branches:
            done = out_root / "branches" / branch["name"] / "branch_record.json"
            if done.is_file():  # 崩溃重跑: 已完成分支直接复用, 不重 fit
                records.append(json.loads(done.read_text(encoding="utf-8")))
            else:
                records.append(run_single_branch(spec, branch, dataset, out_root))
    else:
        records = _run_parallel(spec, dataset, out_root, jobs)
    ranked = sorted(records, key=lambda r: (r["quality"]["score"], r["name"]))
    payload = dict(
        schema=CANDIDATES_SCHEMA, spec_sha256=spec.sha256, spec_path=str(Path(spec_path)),
        snapshot_id=spec.snapshot_id, snapshot_root_hash=snapshot["root_hash"],
        dataset_manifest_sha256=hashlib.sha256(
            (dataset / "manifest.json").read_bytes()).hexdigest(),
        budget=dict(spec.budget), pre_registered_utc=spec.pre_registered_utc,
        branches=ranked, winner=ranked[0]["name"],
        final_test=None,  # 冻结时刻封存在打分必须为空; score-final 后另文件记录
        jobs=int(jobs), **clock())
    text = _canonical(payload)
    if final_path.is_file():
        if final_path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{final_path} 内容不一致, 拒绝覆写")
        return payload
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(text, encoding="utf-8")
    return payload


def _run_parallel(spec: BranchSpec, dataset: Path, out_root: Path, jobs: int) -> list[dict]:
    """每分支一个子进程 (独立 torch 进程态), 聚合读回 branch_record.json。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(fit_module.__file__).resolve().parents[4]), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    pending = list(spec.branches)
    running: dict[subprocess.Popen, str] = {}
    records: dict[str, dict] = {}
    while pending or running:
        while pending and len(running) < jobs:
            branch = pending.pop(0)
            out = out_root / "branches" / branch["name"]
            if (out / "branch_record.json").is_file():
                records[branch["name"]] = json.loads(
                    (out / "branch_record.json").read_text(encoding="utf-8"))
                continue
            proc = subprocess.Popen(
                [sys.executable, "-m", "ao_shaping.tools.slm.transfer.branch_fit",
                 "run-branch", "--spec", str(spec.path), "--branch", branch["name"],
                 "--dataset", str(dataset), "--out-root", str(out_root)], env=env)
            running[proc] = branch["name"]
        if not running:
            continue
        done, _ = _wait_any(running)
        for proc in done:
            name = running.pop(proc)
            if proc.returncode != 0:
                raise RuntimeError(f"分支 {name!r} 子进程退出码 {proc.returncode}")
            records[name] = json.loads(
                (out_root / "branches" / name / "branch_record.json").read_text(encoding="utf-8"))
    return [records[b["name"]] for b in spec.branches]


def _wait_any(running: dict) -> tuple[list, list]:
    import time as _time
    while True:
        done = [p for p in running if p.poll() is not None]
        if done:
            return done, []
        _time.sleep(0.5)


def score_final_once(candidates_path: Path, spec_path: Path, manifest_path: Path,
                     root: Path | None = None) -> dict:
    """候选冻结后的一次性封存打分; 同一 candidates 哈希只允许一次。

    训练折叠到 train+validation (``train_all=True`` —— T2 修复保证封存行永不
    进训练 mask), 打分只读 ``final_test`` 行。结果写 ``final_test_result.json``
    并向 ``final_test_audit.jsonl`` 追加审计行; 不改写 ``candidates_frozen.json``
    (不可覆写纪律)。
    """
    candidates_path = Path(candidates_path)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    spec = load_spec(Path(spec_path))
    snapshot = load_snapshot(Path(manifest_path))
    out_root = candidates_path.parent
    audit = out_root / "final_test_audit.jsonl"
    candidates_sha = hashlib.sha256(candidates_path.read_bytes()).hexdigest()
    if audit.is_file():
        for line in audit.read_text(encoding="utf-8").splitlines():
            if line and json.loads(line).get("candidates_sha256") == candidates_sha:
                raise RuntimeError(
                    f"该候选冻结已在 {json.loads(line).get('scored_utc')} 打过分 —— "
                    f"final_test 每候选只读一次 (#254 验收边界)")
    if candidates.get("final_test") is not None:
        raise RuntimeError("candidates_frozen.json 已带 final_test 字段, 口径异常")
    dataset = out_root / "dataset"
    if not (dataset / "manifest.json").is_file():
        dataset = stage_dataset(snapshot, out_root / "dataset_final", root=root)
    winner = next(b for b in candidates["branches"] if b["name"] == candidates["winner"])
    args = SimpleNamespace(
        dataset=dataset, checkpoint=Path(spec.budget["checkpoint"]),
        out=out_root / "final_test_fit",
        iterations=int(spec.budget["iterations"]), span=int(spec.budget["span"]),
        device=spec.budget.get("device", "auto"), samples=None,
        train_all=True, score_final_test=True, **dict(winner.get("overrides") or {}))
    fit_module.fit(args)
    result = json.loads((args.out / "result.json").read_text(encoding="utf-8"))
    rows = json.loads((args.out / "provenance.json").read_text(encoding="utf-8"))["rows"]
    quality = final_test_quality(result, rows)
    record = dict(candidates_sha256=candidates_sha, spec_sha256=spec.sha256,
                  winner=candidates["winner"], final_test_quality=quality, **clock())
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "final_test_result.json").write_text(_canonical(record), encoding="utf-8")
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_freeze = sub.add_parser("freeze", help="训练全部预注册分支并冻结候选")
    p_freeze.add_argument("--spec", type=Path, required=True)
    p_freeze.add_argument("--manifest", type=Path, required=True)
    p_freeze.add_argument("--out", type=Path, required=True)
    p_freeze.add_argument("--root", type=Path, default=None,
                          help="覆盖快照记录的采集根目录 (语料搬移后使用)")
    p_freeze.add_argument("--jobs", type=int, default=1)
    p_branch = sub.add_parser("run-branch", help=argparse.SUPPRESS)  # 子进程内部入口
    p_branch.add_argument("--spec", type=Path, required=True)
    p_branch.add_argument("--branch", required=True)
    p_branch.add_argument("--dataset", type=Path, required=True)
    p_branch.add_argument("--out-root", type=Path, required=True)
    p_final = sub.add_parser("score-final", help="候选冻结后的一次性 final_test 打分")
    p_final.add_argument("--candidates", type=Path, required=True)
    p_final.add_argument("--spec", type=Path, required=True)
    p_final.add_argument("--manifest", type=Path, required=True)
    p_final.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        payload = freeze_candidates(args.spec, args.manifest, args.out,
                                    jobs=args.jobs, root=args.root)
        print(json.dumps({k: payload[k] for k in
                          ("snapshot_id", "winner", "final_test")}, indent=2), flush=True)
        for branch in payload["branches"]:
            print(f"  {branch['name']}: score={branch['quality']['score']:.6f} "
                  f"corr={branch['quality']['correlation']:.4f}", flush=True)
        return 0
    if args.cmd == "run-branch":
        spec = load_spec(args.spec)
        branch = next(b for b in spec.branches if b["name"] == args.branch)
        record = run_single_branch(spec, branch, args.dataset, args.out_root)
        print(json.dumps(record["quality"], indent=2), flush=True)
        return 0
    record = score_final_once(args.candidates, args.spec, args.manifest, root=args.root)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
