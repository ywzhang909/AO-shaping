"""#251 P0-A: census / probe-list 黄灯态元守卫。

「加新构造点/新探针」与「守卫重钉」分属两个 commit 时, 中间窗口 CI 必红
(ci169 缺陷 1: `open_loop`/`capture_darks` 各制造一段假红)。约定是二者同
commit (CONTRIBUTING.md「本地质量门禁」); 当合并把两者拆进不同窗口时, 本模块
把「已在 TODO 台账登记的漂移」降级为 xfail(黄, 不阻塞 blocking 分片但留痕),
把「未登记的未知漂移」保持为 fail(红)。

判据 = 漂移文件的名字 stem 出现在 ``TODO.md`` 台账正文里。登记是历史事实,
不是可以随口写的免责声明 —— 没进台账的新模块就是未经审查的变更, 必须红。
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

_TODO = Path(__file__).resolve().parents[2] / "TODO.md"


def registered_in_ledger(path: str) -> bool:
    """漂移文件是否已在 TODO.md 台账留痕。

    匹配单位是「路径尾段」(``transfer/open_loop.py`` 级别) 而非裸文件名 ——
    ``santec``/``measure`` 这类泛名在台账里作为别的模块路径的子串大量出现,
    按 stem 匹配会把陌生新文件误判成已登记。
    """
    text = _TODO.read_text(encoding="utf-8", errors="replace")
    parts = [p for p in path.split("/") if p]
    for k in range(len(parts), 1, -1):
        if "/".join(parts[-k:]) in text:
            return True
    # probe-list 守卫只能拿到模块裸名 (磁盘扫描跨多个子包): 按 ``<name>.py`` 匹配。
    if len(parts) == 1:
        return f"{parts[0]}.py" in text
    return False


def yellow_or_red(drift_paths: set[str], detail: str) -> NoReturn:
    """census/probe 断言失败的统一出口: 登记过 → 黄; 否则 → 红。"""
    unregistered = sorted(p for p in drift_paths if not registered_in_ledger(p))
    if unregistered:
        pytest.fail(
            f"{detail}\n未登记的漂移文件 (真未知变更, 重钉必须与特性同 commit): "
            f"{unregistered}"
        )
    pytest.xfail(
        "re-baseline pending, drift registered in TODO ledger: "
        + ", ".join(sorted(drift_paths))
    )
