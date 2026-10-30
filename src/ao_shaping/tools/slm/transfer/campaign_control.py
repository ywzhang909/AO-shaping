"""段间暂停闸：把「本段结束后停下」这个请求写进目录，而不是写进内存。

为什么不用 ``threading.Event``
------------------------------
runner 的停止粒度本来就是**段**（``execute_campaign`` 里单块是原子事务，段末的
``_final`` 重测与验收都还在这段之内），所以「停止」和「段间暂停」共用同一个停止点，
差别的只有**谁读得到这个请求**：内存里的 flag 换个页面、重开浏览器、换一台机器就没了；
落进目录的请求还在，下次「开始」或「从断点续跑」照样能看见。

文件约定与 :mod:`ao_shaping.tools.slm.transfer.parallel_fit` 的
``stop_acquisition.request`` 完全一致：一个 JSON 小文件，存在即代表请求已发出。读侧
只看文件在不在，不看内容——内容坏了也不该让暂停请求凭空失效。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ao_shaping.tools.slm.transfer.capture import clock

__all__ = [
    "STOP_AFTER_STAGE_FILE",
    "clear_stage_stop",
    "request_stage_stop",
    "stage_stop_requested",
]

#: 段间暂停请求的文件名（落在 campaign 输出目录根部）。
STOP_AFTER_STAGE_FILE = "stop_after_stage.request"


def _marker(out_dir: str | Path) -> Path:
    """请求文件的完整路径（不建目录、不碰盘）。"""
    return Path(out_dir) / STOP_AFTER_STAGE_FILE


def request_stage_stop(out_dir: str | Path, *, reason: str = "", **extra: Any) -> Path:
    """在 ``out_dir`` 里写一个段间暂停请求，返回请求文件路径。

    Args:
        out_dir: campaign 输出目录（``--out`` 那一层）。
        reason: 发起方自己的理由，原样落盘，方便事后看是谁按的。
        **extra: 其它要一起落盘的上下文（例如 ``requested_by``）。

    请求文件**只增不改**：重复请求只是覆盖同一条内容，仍然只是"一个请求"。
    """
    path = _marker(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(reason=str(reason), **extra, **clock())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True,
                               default=str), encoding="utf-8")
    return path


def clear_stage_stop(out_dir: str | Path) -> bool:
    """清掉暂停请求；返回是否真的删掉了文件（幂等，无请求时返回 ``False``）。"""
    path = _marker(out_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


def stage_stop_requested(out_dir: str | Path) -> bool:
    """目录里是否有段间暂停请求（只看存在性，不解析内容）。"""
    return _marker(out_dir).is_file()
