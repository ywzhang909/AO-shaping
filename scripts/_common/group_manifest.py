"""一次只跑一组变量的探测: 分组元数据与分组清单 (group manifest)。

为什么
----

两个硬件探针脚本 (``scripts/slm/hw_gs_defocus_probe.py`` 扫 Δz,
``scripts/slm/run_gs_refine_zernike_sweep.py`` 扫 Zernike 阶数) 以前都是
「一次把整个实验做完」, 于是:

* 两组变量混在同一次运行里, 谁贡献了哪一点说不清;
* 一组失败就把整轮作废;
* 产物没有「哪一组」的身份, 离线聚合只能靠目录名猜。

现在每次调用只探**一组**变量: 这一组的身份由 ``--group`` 给出, 这一组的固定参数由
``--group-meta KEY=VALUE`` 给出, 落盘目录是 ``<root>/<group>/``, 而
``<root>/manifest.json`` 按组列出本根目录下记录过的每一组。

清单条目 (两个探针共用同一形状)::

    {"group": "dz_scan", "variable": "dz_mm", "meta": {...},
     "dir": "dz_scan", "n_rows": 11, "created_utc": "2026-10-20T02:11:00+00:00"}

``variable`` 是这一组扫的变量名 (离焦探针 = ``dz_mm``, 泽尼克扫描 = ``n_max``),
``meta`` 是这一组的**固定**参数, 逐字记录。

🔴 **清单必须在每组写完后立刻更新**: 后一组崩掉 (或者整轮被强杀) 时, 前一组必须
已经被索引 —— 否则「跑了一半」这件事在磁盘上不留痕迹, 只留在操作者的记忆里。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_NAME",
    "group_output_dir",
    "group_record",
    "load_manifest",
    "manifest_path",
    "parse_group_meta",
    "update_manifest",
    "utc_now_iso",
]

#: 分组清单的文件名 (写在**根目录**, 与各组子目录平级)。
MANIFEST_NAME = "manifest.json"

#: 组子目录名的默认身份 (不带 ``--group`` 时的行为)。
DEFAULT_GROUP = "main"


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO-8601 字符串 (秒精度)。

    清单条目里的 ``created_utc`` 用它 —— 探针一次运行动辄几十分钟, 秒精度足够,
    而且不依赖机器的本地时区。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_group_meta(pairs: list[str] | None) -> dict[str, str]:
    """把 ``--group-meta KEY=VALUE`` 列表解析成 ``{KEY: VALUE}``。

    值**逐字保留** (只去掉首尾空白): ``exposure=0.2`` 记成 ``"0.2"``, 不做类型
    推断 —— 清单是这一组的**事实记录**, 推断出来的类型会让读的人以为参数是被
    归一化过的。

    Args:
        pairs: 形如 ``["exposure=0.2", "phase=hardware_gs"]`` 的原始字符串列表。

    Returns:
        ``{KEY: VALUE}`` 字典; 同一个 KEY 后写的覆盖先写的。

    Raises:
        ValueError: 某个元素不含 ``=``, 或者 KEY 为空。
    """
    meta: dict[str, str] = {}
    for raw in pairs or []:
        key, sep, value = str(raw).partition("=")
        if not sep:
            raise ValueError(f"--group-meta 需要 KEY=VALUE 形式, 收到 {raw!r}")
        key = key.strip()
        if not key:
            raise ValueError(f"--group-meta 的 KEY 不能为空: {raw!r}")
        meta[key] = value.strip()
    return meta


def group_output_dir(root: str | Path, group: str) -> Path:
    """组的输出目录 = ``<root>/<group>/``。

    两个探针脚本都用它, 所以「一组 = 一个子目录」这条约定只有一处定义。

    Args:
        root: 根目录 (离焦探针是 ``report/shape_trials/hw_gs_defocus_probe_<UTC>``,
            泽尼克扫描是 ``--out``)。
        group: 组名 (``--group``)。

    Returns:
        组目录路径 (不保证已存在)。
    """
    return Path(root) / str(group)


def manifest_path(root: str | Path) -> Path:
    """根目录下的分组清单路径 (``<root>/manifest.json``)。"""
    return Path(root) / MANIFEST_NAME


def load_manifest(root: str | Path) -> list[dict[str, Any]]:
    """读回根目录的分组清单。

    清单文件缺失 / 损坏时返回空列表而不是抛异常: 清单是**索引**, 索引读不出来
    不该让探针跑不下去 —— 真正的测量数据在组目录里, 那才是要紧的。
    """
    path = manifest_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def group_record(
    *,
    group: str,
    variable: str,
    meta: dict[str, str] | None = None,
    dir_name: str | None = None,
    n_rows: int = 0,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """组装一条分组清单条目 (两个探针共用同一形状)。

    Args:
        group: 组名 (``--group``)。
        variable: 这一组扫的变量名 (如 ``dz_mm`` / ``n_max``)。
        meta: 这一组的固定参数, 逐字记录。
        dir_name: 组目录名 (相对根目录); 缺省等于 ``group``。
        n_rows: 这一组记录了多少行测量 (离焦探针 = Δz 点数, 泽尼克扫描 = 阶数)。
        created_utc: 写入时刻; 缺省取当前 UTC 时间。

    Returns:
        可 JSON 序列化的条目。
    """
    return {
        "group": str(group),
        "variable": str(variable),
        "meta": dict(meta or {}),
        "dir": str(dir_name if dir_name is not None else group),
        "n_rows": int(n_rows),
        "created_utc": created_utc or utc_now_iso(),
    }


def update_manifest(root: str | Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """把一条分组条目并入根目录的 ``manifest.json`` 并立刻落盘。

    同一个组名**就地替换** (重跑同一组不会留下两条), 新组**追加**在末尾 ——
    清单的顺序就是实验的顺序。

    Args:
        root: 根目录。
        entry: :func:`group_record` 生成的条目。

    Returns:
        落盘后的完整清单 (列表)。
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(root_path)
    group = str(entry.get("group", ""))
    merged: list[dict[str, Any]] = []
    replaced = False
    for existing in entries:
        if str(existing.get("group", "")) == group:
            merged.append(dict(entry))
            replaced = True
        else:
            merged.append(existing)
    if not replaced:
        merged.append(dict(entry))
    manifest_path(root_path).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged
