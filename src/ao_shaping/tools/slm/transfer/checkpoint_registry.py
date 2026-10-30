"""检查点索引：把会话目录里的模型文件登记成一张可扫描、可校验的表。

为什么需要索引
--------------
会话目录里 ``frozen_parameters.json`` 会出现好几份（``nominal`` / ``model640`` /
``model_wide640``），而"这一份还能不能用"取决于三件索引里才有的事：它是哪条光路的
产物（:mod:`ao_shaping.tools.slm.transfer.optical_session` 的指纹）、它当初落盘时的
哈希、以及它是不是已经被判定过期。历史浏览器要在一屏里比较多个会话，所以扫描不能
靠"把每个文件都当模型加载一遍"，只能读一份小的索引。

三件事分开做
------------
* :func:`register_session` —— 把一个会话目录里的全部检查点登记进它自己的
  ``checkpoint_index.json``（原子替换、按解析后的路径幂等）；
* :func:`scan_checkpoint_index` —— 从一个根目录扫回所有索引（没有索引的旧目录
  现场合成条目），按创建时间倒序，并在返回前对每一条做哈希校验；
* :func:`verify_checkpoint` —— 单条校验；失配时把这条在索引里标 ``stale``，
  而不是删掉它 —— 「过期」本身是有用的信息。

索引只写 ``session/checkpoint_index.json``，绝不改会话里的模型文件。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ao_shaping.tools.slm.transfer.optical_session import load_fingerprint

__all__ = [
    "CHECKPOINT_NAME",
    "INDEX_NAME",
    "register_session",
    "scan_checkpoint_index",
    "verify_checkpoint",
]

#: 会话目录里索引文件的固定名字。
INDEX_NAME = "checkpoint_index.json"
#: 检查点文件的固定名字（``geometric_seed.export_frozen_parameters`` 的产物）。
CHECKPOINT_NAME = "frozen_parameters.json"
#: 会话根到检查点的最大层数（``nominal`` / ``model640`` / ``model_wide640`` 走一层）。
_MAX_DEPTH = 2


# --------------------------------------------------------------------------- #
# 读取与小工具（全部容忍缺失）
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str | None:
    """文件的 sha256；读不到返回 ``None``（不存在与读失败在调用侧等价）。"""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    """文件的修改时间（UTC ISO）；取不到时用当前时间，保证排序键总是有值。"""
    try:
        moment = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        moment = datetime.now(timezone.utc)
    return moment.isoformat(timespec="microseconds")


def _write_index(index_path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    """原子替换索引文件（先写临时文件再 ``replace``，扫描侧永远读到完整内容）。"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp")
    temporary.write_text(
        json.dumps([dict(entry) for entry in entries], indent=2, ensure_ascii=False,
                   sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(index_path)


# --------------------------------------------------------------------------- #
# 会话侧：找检查点、拼元数据
# --------------------------------------------------------------------------- #
def _checkpoint_files(session: Path) -> list[Path]:
    """会话目录里的全部检查点文件（深度不超过 :data:`_MAX_DEPTH`）。"""
    found: list[Path] = []
    for path in session.rglob(CHECKPOINT_NAME):
        if not path.is_file():
            continue
        depth = len(path.relative_to(session).parts) - 1
        if depth <= _MAX_DEPTH:
            found.append(path)
    return sorted(found)


def _session_of(checkpoint: Path) -> Path:
    """一个检查点文件所属的会话目录（往上找 ``journal.json`` / ``shape/`` 那一层）。"""
    for parent in [checkpoint.parent, *checkpoint.parents]:
        if (parent / "journal.json").is_file() or (parent / "shape").is_dir():
            return parent
    return checkpoint.parent


def _shape_of(session: Path) -> str | None:
    """会话的形状键：单形状会话给出名字，多形状 campaign 给 ``None``（不硬选一个）。"""
    journal = _read_json(session / "journal.json")
    if isinstance(journal, Sequence) and not isinstance(journal, (str, bytes)):
        for record in journal:
            if not isinstance(record, Mapping) or record.get("stage") != "plan":
                continue
            plan = record.get("plan")
            shapes = plan.get("shapes") if isinstance(plan, Mapping) else None
            if isinstance(shapes, Sequence) and not isinstance(shapes, (str, bytes)):
                names = [str(name) for name in shapes if str(name).strip()]
                return names[0] if len(names) == 1 else None
    shape_root = session / "shape"
    if shape_root.is_dir():
        names = sorted(path.name for path in shape_root.iterdir() if path.is_dir())
        return names[0] if len(names) == 1 else None
    return None


def _fit_quality(session: Path) -> tuple[float | None, float | None]:
    """会话拟合质量 → ``(cv, quality)``；取不到时两项都是 ``None``。

    ``fit/result.json`` 的 ``metrics`` 是逐样本的字典序列，这里取
    ``cv_measured`` 的均值作 CV、``correlation`` 的均值作质量：
    「预测得像不像」和「光斑本身长什么样」是两个数，不混成一个。
    """
    payload = _read_json(session / "fit" / "result.json")
    if not isinstance(payload, Mapping):
        return None, None
    metrics = payload.get("metrics")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        return _number(payload.get("cv")), _number(payload.get("quality"))
    cvs = [value for value in (_number(row.get("cv_measured")) for row in metrics
                               if isinstance(row, Mapping)) if value is not None]
    correlations = [value for value in (_number(row.get("correlation")) for row in metrics
                                        if isinstance(row, Mapping)) if value is not None]
    return (sum(cvs) / len(cvs) if cvs else None,
            sum(correlations) / len(correlations) if correlations else None)


def _entry(session: Path, checkpoint: Path, *, stale: bool = False) -> dict[str, Any]:
    """一个检查点 → 一条索引记录（字段与历史浏览器的列一一对应）。"""
    cv, quality = _fit_quality(session)
    digest = _sha256(checkpoint)
    return {
        "path": str(checkpoint.parent),
        "file": str(checkpoint),
        "session": str(session),
        "kind": checkpoint.parent.name,
        "shape": _shape_of(session),
        "created_at": _iso_mtime(checkpoint),
        "cv": cv,
        "fit_quality": quality,
        "sha256": digest,
        "fingerprint": load_fingerprint(session),
        "stale": bool(stale or digest is None),
    }


# --------------------------------------------------------------------------- #
# 注册 / 扫描 / 校验
# --------------------------------------------------------------------------- #
def register_session(session_dir: str | Path) -> list[dict[str, Any]]:
    """登记一个会话目录里的全部检查点，返回这些记录（同时也是索引的内容）。

    幂等：同一解析路径重复登记得到同形的记录（``created_at`` 取文件 mtime、
    ``sha256`` 取当下内容），已有的 ``stale`` 标记会被保留——过期是结论，不是
    可以靠重扫抹掉的状态。没有检查点的目录不写索引，返回空列表。
    """
    session = Path(session_dir).resolve()
    checkpoints = _checkpoint_files(session)
    if not checkpoints:
        return []
    index_path = session / INDEX_NAME
    previous = _load_index(index_path)
    stale = {str(Path(entry.get("file", ""))): bool(entry.get("stale"))
             for entry in previous if entry.get("file")}
    entries = [_entry(session, checkpoint, stale=stale.get(str(checkpoint), False))
               for checkpoint in checkpoints]
    _write_index(index_path, entries)
    return entries


def _load_index(index_path: Path) -> list[dict[str, Any]]:
    """读一份索引；不是列表就返回空（坏索引等同于没有索引）。"""
    payload = _read_json(index_path)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [dict(entry) for entry in payload if isinstance(entry, Mapping)]


def _recorded_digest(file: Path, index: Path | None) -> str | None:
    """从索引里找回某检查点登记时的 sha256；记录里没有就返回 ``None``。"""
    if index is None:
        index = _session_of(file) / INDEX_NAME
    if not index.is_file():
        return None
    return next(
        (str(row.get("sha256")) for row in _load_index(index)
         if str(row.get("file") or row.get("path") or "") == str(file)
         and row.get("sha256")),
        None,
    )


def _verified(entry: dict[str, Any]) -> dict[str, Any]:
    """按当下的文件内容校验一条记录；失配（或文件没了）标 ``stale``。"""
    file = Path(str(entry.get("file") or entry.get("path") or ""))
    actual = _sha256(file)
    ok = actual is not None and (entry.get("sha256") in (None, actual))
    entry["sha256_ok"] = ok
    if not ok:
        entry["stale"] = True
    return entry


def scan_checkpoint_index(root: str | Path, *, verify: bool = True,
                          limit: int | None = None) -> list[dict[str, Any]]:
    """从 ``root`` 扫回全部检查点记录，按创建时间倒序返回。

    优先读各会话自己的索引；对没有索引的旧目录（或索引读不出来）现场合成记录，
    所以"从没登记过"的历史不会从列表里消失。``verify=True`` 时每条都做哈希校验，
    失配的在返回值里标 ``stale``（**不**回写索引——扫描是只读动作）。
    """
    base = Path(root)
    entries: dict[str, dict[str, Any]] = {}
    indexes = [base] if base.is_file() else sorted(base.rglob(INDEX_NAME))
    for index_path in indexes:
        if not index_path.is_file():
            continue
        for entry in _load_index(index_path):
            file = str(entry.get("file") or entry.get("path") or "")
            if file:
                entries[file] = entry
    if not base.is_file():
        for checkpoint in sorted(base.rglob(CHECKPOINT_NAME)):
            if not checkpoint.is_file():
                continue
            key = str(checkpoint)
            if key in entries:
                continue
            session = _session_of(checkpoint)
            entries[key] = _entry(session, checkpoint)
    rows = [_verified(entry) if verify else dict(entry) for entry in entries.values()]
    rows.sort(key=lambda entry: str(entry.get("created_at") or ""), reverse=True)
    return rows[:limit] if limit is not None else rows


def verify_checkpoint(target: str | Path | Mapping[str, Any], *,
                      index: str | Path | None = None) -> bool:
    """校验一个检查点（条目或路径）当前内容是否与登记时一致。

    Args:
        target: 一条索引记录，或检查点文件/目录的路径。
        index: 给定索引文件时，校验失败会把对应条目标 ``stale`` 并回写；不给就
            只返回结论。

    Returns:
        文件存在且哈希一致（或该条从未记录哈希）时为 ``True``。
    """
    if isinstance(target, Mapping):
        entry = dict(target)
    else:
        path = Path(target)
        file = path / CHECKPOINT_NAME if path.is_dir() else path
        entry = {"file": str(file), "sha256": None}
        recorded = _recorded_digest(file, Path(index) if index is not None else None)
        if recorded is not None:
            entry["sha256"] = recorded
    verified = _verified(entry)
    index_path: Path | None = None
    if index is not None:
        index_path = Path(index)
    elif entry.get("session"):
        candidate = Path(str(entry["session"])) / INDEX_NAME
        index_path = candidate if candidate.is_file() else None
    elif recorded is not None:
        index_path = _session_of(Path(str(entry["file"]))) / INDEX_NAME
    if not verified.get("sha256_ok") and index_path is not None:
        rows = _load_index(index_path)
        for row in rows:
            if str(row.get("file")) == str(verified.get("file")):
                row["stale"] = True
                row["sha256_ok"] = False
        _write_index(index_path, rows)
    return bool(verified.get("sha256_ok"))
