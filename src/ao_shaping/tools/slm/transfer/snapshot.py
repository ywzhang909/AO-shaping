"""#254 T3: 把闭环采集语料冻结成三层快照 v1(哈希清单, 封存口径)。

数字孪生先行的边界: 本模块**不采集、不训练、不碰设备** —— 它只读取
``capture`` 已落盘的采集目录(``manifest.json`` + ``sample_*.npz``), 逐文件
sha256, 校验三层分组口径(#254 T2 的 ``train / validation / final_test``,
见 :mod:`.splits`), 产出一份不可覆写的冻结清单。真正的 v1 冻结在持有语料的
采集侧机器上跑同一条命令; 本仓库内用合成语料做全流程回归(
``tests/ao_shaping/tools/slm/transfer/test_snapshot.py``)。

清单承重三件事:

1. **可审计**: 每个 ``sample_*.npz`` 与每份 ``manifest.json`` 都有内容哈希;
   ``root_hash`` 是所有采集(名字 + 文件哈希)的规范化 JSON 哈希, 复核时
   :func:`verify` 逐文件重算即可判定语料是否被移动后改动。
2. **防泄漏口径不漂移**: 冻结前逐采集校验 ``split`` 取值合法、``rounds`` 方案
   下同一 ``probe_round`` 必须同层(T2 的整轮预注册不变量)。校验不过就拒绝
   冻结, 而不是把脏划分冻进快照。
3. **不可覆写**: 同一清单路径上, 内容不一致的二次冻结直接报错 ——
   「v1 冻结后再来一版 v1」是三层防泄漏里最容易自欺的口子; 字节级相同则
   幂等跳过(与 provenance 同步脚本同一纪律)。

用法::

    python -m ao_shaping.tools.slm.transfer.snapshot freeze \\
        --root data/debug/slm_transfer --id v1 --out data/snapshots [--plan-only]
    python -m ao_shaping.tools.slm.transfer.snapshot verify \\
        --manifest data/snapshots/v1/snapshot_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ao_shaping.tools.slm.transfer.capture import clock

#: 清单 schema 版本; 字段级不兼容改动必须换号, 读取方按号分支。
SCHEMA = "transfer-snapshot-v1"

#: 合法的逐行 split 取值(与 fit.load_dataset / splits.py 同一词表)。
KNOWN_SPLITS = frozenset({"train", "validation", "test", "final_test", "calibration"})

#: 必须至少各有一行才能进快照的层; ``final_test`` 允许缺失(历史语料在 T2
#: 之前采集, 缺封存层要如实记录而非阻塞冻结), ``test`` 同理可选。
REQUIRED_SPLITS = frozenset({"train", "validation"})

CHUNK = 1 << 20


def discover_captures(root: Path) -> list[Path]:
    """``root`` 下含 ``manifest.json`` 且至少一个 ``sample_*.npz`` 的目录, 按名排序。"""
    root = Path(root)
    found = []
    for path in sorted(root.rglob("manifest.json")):
        capture = path.parent
        if any(capture.glob("sample_*.npz")):
            found.append(capture)
    return found


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _round_consistency_problems(rows: list[dict]) -> list[str]:
    """rounds 方案: 同一 ``probe_round`` 的所有行必须同层(T2 整轮预注册不变量)。"""
    problems: list[str] = []
    by_round: dict[int, str] = {}
    for row in rows:
        if row.get("split_scheme") != "rounds":
            continue
        rnd = row.get("probe_round")
        if rnd is None:
            problems.append(f"row {row.get('label')!r}: rounds 行缺 probe_round")
            continue
        split = row.get("split")
        if rnd in by_round and by_round[rnd] != split:
            problems.append(
                f"round {rnd}: split 漂移 {by_round[rnd]!r} -> {split!r} "
                f"(整轮必须同层, 见 splits.assign_round_splits)")
        by_round.setdefault(rnd, split)
    return problems


def capture_problems(capture: Path, rows: list[dict]) -> list[str]:
    """单个采集目录的冻结前校验; 返回问题列表, 空列表 = 可冻结。"""
    problems: list[str] = []
    labels: set[str] = set()
    files: set[str] = set()
    for row in rows:
        name = row.get("file")
        if not name or name in files:
            problems.append(f"row {row.get('label')!r}: file 字段缺失或重复 ({name!r})")
        files.add(str(name))
        if row.get("label") in labels:
            problems.append(f"duplicate label {row.get('label')!r}")
        labels.add(str(row.get("label")))
        if row.get("split") not in KNOWN_SPLITS:
            problems.append(f"row {row.get('label')!r}: 未知 split {row.get('split')!r}")
        elif not (capture / str(name)).is_file():
            problems.append(f"row {row.get('label')!r}: 清单引用的 {name} 不存在")
    registered = {str(row.get("file")) for row in rows}
    for extra in sorted(p.name for p in capture.glob("sample_*.npz")):
        if extra not in registered:
            problems.append(f"{extra}: 磁盘存在但清单未登记 (冻结后不可寻址)")
    problems.extend(_round_consistency_problems(rows))
    layers = {row.get("split") for row in rows}
    missing = REQUIRED_SPLITS - layers
    if missing:
        problems.append(f"缺必需层 {sorted(missing)} —— 该采集无法参与训练/选择")
    return problems


def _load_rows(capture: Path) -> list[dict]:
    rows = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{capture}: manifest.json 不是行列表")
    return rows


def build_manifest(root: Path, snapshot_id: str) -> dict:
    """扫描并校验语料, 产出清单 dict(不落盘)。任何采集有问题 → ValueError(全量报告)。"""
    root = Path(root)
    captures = discover_captures(root)
    if not captures:
        raise ValueError(f"{root} 下没有可冻结的采集目录 (manifest.json + sample_*.npz)")
    problems: list[str] = []
    entries: list[dict] = []
    totals: dict[str, int] = {}
    for capture in captures:
        rows = _load_rows(capture)
        problems.extend(f"{capture.name}: {p}" for p in capture_problems(capture, rows))
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row.get("split"))] = counts.get(str(row.get("split")), 0) + 1
            totals[str(row.get("split"))] = totals.get(str(row.get("split")), 0) + 1
        files = sorted(p.name for p in capture.glob("sample_*.npz"))
        entries.append(dict(
            name=capture.name,
            dir=str(capture.resolve()),
            manifest_sha256=_sha256_file(capture / "manifest.json"),
            files=[dict(name=n, sha256=_sha256_file(capture / n),
                        bytes=(capture / n).stat().st_size) for n in files],
            rows=len(rows),
            split_counts=counts,
        ))
    if problems:
        raise ValueError("快照校验未通过:\n  " + "\n  ".join(problems))
    sealed = [s for s in sorted(totals) if s not in {"train", "validation", "calibration"}]
    root_hash = hashlib.sha256(json.dumps(
        [{k: e[k] for k in ("name", "manifest_sha256", "files")} for e in entries],
        sort_keys=True).encode("utf-8")).hexdigest()
    return dict(schema=SCHEMA, snapshot_id=snapshot_id, root=str(root),
                root_hash=root_hash, layer_counts=totals,
                sealed_layers=sealed, captures=entries, **clock())


def render_plan(manifest: dict) -> str:
    """人读的预注册计划表(``--plan-only`` 的输出; 采集/哈希先行, 落盘在后)。"""
    lines = [f"snapshot {manifest['snapshot_id']!r}  root_hash={manifest['root_hash'][:16]}…",
             f"层计数: {manifest['layer_counts']}  封存层: {manifest['sealed_layers'] or '无'}"]
    for entry in manifest["captures"]:
        lines.append(f"  {entry['name']}: {entry['rows']} 行, "
                     f"{len(entry['files'])} 文件, {entry['split_counts']}")
    return "\n".join(lines)


def _canonical(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True)


#: 冻结时刻随每次构建变化, 不参与「同一冻结」判定 —— 否则相同语料重复冻结
#: 永远字节不同, 幂等分支变成死代码 (2026-11-26 首测即踩中)。
_CLOCK_KEYS = frozenset({"timestamp_utc", "monotonic_ns"})


def _strip_clock(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k not in _CLOCK_KEYS}


def freeze(root: Path, out_dir: Path, snapshot_id: str, *,
           plan_only: bool = False) -> dict:
    """冻结快照; 返回清单 dict。``plan_only`` 只构建并返回, 不写任何文件。

    不可覆写: 目标清单已存在且字节不同 → FileExistsError; 字节相同 → 幂等返回
    既有清单(不重写)。
    """
    manifest = build_manifest(Path(root), snapshot_id)
    if plan_only:
        return manifest
    path = Path(out_dir) / snapshot_id / "snapshot_manifest.json"
    text = _canonical(manifest)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _strip_clock(existing) == _strip_clock(manifest):
            return existing  # 同一冻结: 保留首次冻结时刻, 不重写
        raise FileExistsError(
            f"{path} 已存在且内容不同 —— 快照一经冻结不可改写; "
            f"语料更新请换新的 --id, 复核用 verify 子命令")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(manifest), encoding="utf-8")
    return manifest


def verify(manifest_path: Path, root: Path | None = None) -> dict:
    """逐文件重算哈希复核语料; 返回 ``{ok, problems}``, 不抛。

    ``root`` 给出后按 ``<root>/<capture name>`` 重定位 —— 语料搬移后仍能复核
    内容哈希 (搬移不改字节, 只改路径; 路径本身不在 ``root_hash`` 里)。
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema 不符: {manifest.get('schema')!r}")
    for entry in manifest.get("captures", []):
        capture = Path(root) / entry["name"] if root else Path(entry["dir"])
        if not (capture / "manifest.json").is_file():
            problems.append(f"{entry['name']}: 目录不存在 {capture}")
            continue
        if _sha256_file(capture / "manifest.json") != entry["manifest_sha256"]:
            problems.append(f"{entry['name']}: manifest.json 哈希漂移")
        recorded = {f["name"]: f["sha256"] for f in entry["files"]}
        present = {p.name for p in capture.glob("sample_*.npz")}
        for missing in sorted(set(recorded) - present):
            problems.append(f"{entry['name']}: 缺失 {missing}")
        for extra in sorted(present - set(recorded)):
            problems.append(f"{entry['name']}: 冻结后新增 {extra}")
        for name, sha in sorted(recorded.items()):
            file_path = capture / name
            if file_path.is_file() and _sha256_file(file_path) != sha:
                problems.append(f"{entry['name']}: {name} 内容哈希漂移")
    return dict(ok=not problems, problems=problems, snapshot_id=manifest.get("snapshot_id"),
                root_hash=manifest.get("root_hash"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_freeze = sub.add_parser("freeze", help="冻结语料为三层快照清单")
    p_freeze.add_argument("--root", type=Path, required=True)
    p_freeze.add_argument("--out", type=Path, required=True)
    p_freeze.add_argument("--id", dest="snapshot_id", required=True)
    p_freeze.add_argument("--plan-only", action="store_true",
                          help="只打印冻结计划(采集、层计数、哈希), 不落盘")
    p_verify = sub.add_parser("verify", help="按清单逐文件重算哈希复核")
    p_verify.add_argument("--manifest", type=Path, required=True)
    p_verify.add_argument("--root", type=Path, default=None,
                          help="按 <root>/<采集名> 重定位后复核 (语料搬移场景)")
    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        manifest = freeze(args.root, args.out, args.snapshot_id,
                          plan_only=args.plan_only)
        print(render_plan(manifest), flush=True)
        if args.plan_only:
            print("(plan-only: 未写任何文件)", flush=True)
        else:
            path = args.out / args.snapshot_id / "snapshot_manifest.json"
            print(f"清单: {path}", flush=True)
        return 0
    result = verify(args.manifest, root=args.root)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
