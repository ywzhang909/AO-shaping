"""DM 致动器邻接矩阵的单一加载器。

该矩阵过去会被读取两次, 且行为已经分化:

* ``drivers/dm/nlight/driver.py`` 在其 **类体** 里读 ``"data/dm_adj.txt"``,
  于是读取发生在 *import* 期, 工作目录不对时会让
  ``import ao_shaping`` 直接以 ``FileNotFoundError`` 失败。
* ``drivers/sim/dm/simulated_dm.py`` 在 ``__init__`` 里读同一路径, 但缺失时
  会回退到合成网格。

路径相对已安装的包解析, 而非进程 CWD。
``PATHS.root_dir`` 刻意是 CWD 相对的, 因为它存放运行 *输出*;
本文件是仓库 *输入*, 因此不能随调用方变动。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

#: ``src/ao_shaping/drivers/dm/_adjacency.py`` -> 仓库根目录。
_REPO_ROOT = Path(__file__).resolve().parents[4]
ADJACENCY_PATH: Path = _REPO_ROOT / "data" / "dm_adj.txt"

_ACTUATOR_COUNT = 64

#: ``np.loadtxt`` 产出 float64; 合成回退产出 int。两者都只经由
#: ``== 1`` / 布尔索引消费, 因此缓存按两者的公共数值父类型标注,
#: 而不是做强制转换 (那会改变取值)。
_cache: npt.NDArray[np.number] | None = None


def _synthetic_grid(size: int = _ACTUATOR_COUNT) -> npt.NDArray[np.intp]:
    """4-邻居网格, 仅在该资产确实不可用时使用。"""
    side = int(np.sqrt(size))
    if side * side != size:
        raise ValueError(f"{size} is not a square grid")
    adj = np.zeros((size, size), dtype=int)
    for i in range(size):
        row, col = divmod(i, side)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < side and 0 <= c < side:
                adj[i, r * side + c] = 1
    return adj


def _cache_clear() -> None:
    """丢弃已记忆化的矩阵 (供替换路径的测试使用)。"""
    global _cache
    _cache = None


def load_adjacency() -> npt.NDArray[np.number]:
    """返回 ``(64, 64)`` 的致动器邻接矩阵, 只加载一次。

    解析顺序: 包内资产, 然后是 CWD 相对的 ``data/`` 副本 (供把该文件
    放在别处的检出), 最后是合成的 4-邻居网格。最后一步意味着资产缺失时
    邻居安全检查会降级, 而不会让驱动无法构造。
    """
    global _cache
    if _cache is not None:
        return _cache

    candidates = [ADJACENCY_PATH, Path("data") / "dm_adj.txt"]
    for candidate in candidates:
        try:
            matrix = np.loadtxt(candidate)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        _cache = matrix
        return _cache

    _cache = _synthetic_grid()
    return _cache


class lazy_adjacency:
    """把矩阵读取推迟到首次属性访问的描述符。

    在类体里赋值 :func:`load_adjacency` 的结果会在 import 期间读文件。
    描述符既让 ``NLight.Units_Adj_Mat`` 与 ``self.Units_Adj_Mat`` 继续可用,
    又把 I/O 移到首次使用时。
    """

    def __get__(
        self, obj: object, objtype: type | None = None
    ) -> npt.NDArray[np.number]:
        return load_adjacency()


__all__ = ["ADJACENCY_PATH", "lazy_adjacency", "load_adjacency"]
