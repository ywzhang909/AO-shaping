"""设备互斥: 「谁占着设备」从**猜**变成**可判定的锁**。

为什么有它 (2026-10-22 21:18-21:27 实测)
----------------------------------------

那天用后台方式起 ``scripts/slm/hw_differentiable_phase_opt.py --execute
--wait-until-idle 1800``, **同一个命令出现了两个 python 进程** (命令行完全相同),
于是排队循环在 22 轮 / 455 s 里**一直**报「别的会话在用设备」; 而**直连验证**
SLM 与大恒相机都能正常打开 (Santec 打开成功、driveboardID 22030108、正常关闭;
``create_camera("daheng", 0)`` 打开成功并出帧)。同一时段并发会话的
``hw_gs_defocus_probe.py --execute`` 也是**两个同名进程** ⇒ 「一对同名进程」是
这类脚本的常态 (Windows spawn / 子进程), **不是**「别的会话」。

根因有两条, 都在这里被拆掉:

1. **占用者是猜的。** 旧判据是「命令行里含 ``scripts/slm/`` 之类的关键字 ⇒
   有人在用设备」, 而**自己这一棵进程树**启动的子进程恰好也满足这条关键字 ⇒
   永远抢不到设备。现在占用者由**跨进程锁**判定: 锁的持有者 pid 写进锁文件,
   比对**进程祖先链**后, 自己/父/子进程持有 ⇒ **不是冲突**。
2. **「SDK 报忙」有两种, 处置相反。** 一种是**别人真的持有** (等一会儿就空出来);
   另一种是 **SDK 陈旧状态** —— SDK 说句柄已被打开, 但**没有任何活着的持有者**
   (上一个人的进程已经退出, 句柄却还留在 SDK 侧)。后者**排队等不来**,
   仓库 ``AGENTS.md`` 与 ``src/ao_shaping/drivers/slm/AGENTS.md`` 记录的处置是
   **给 SLM 控制器物理断电重置**。旧判据把两种都归成「别的会话占用」⇒
   排 1800 s 排不到。

归类 (处置不同, 所以必须分开)
------------------------------

**能等**的两类 (别人在用, 等一会儿就空出来):

=========================  ==========================================  ==============
归类                       判据                                        能不能等
=========================  ==========================================  ==============
``slm_busy``               ``-10003`` / 「USB设备已打开」                能 (别人在用)
``camera_busy``            ``camera not initialized``                   能 (别人在用)
=========================  ==========================================  ==============

**等不来**的五类:

=========================  ==========================================  ==============
``sdk_stale``              **锁空** 但 SDK 报忙 (陈旧句柄)              **不能** (**断电**)
``slm_absent``             ``-10002`` 未找到USB设备                     不能 (要人插拔)
``camera_absent``          ``Camera ID not found``                      不能 (要人检查)
``light_path_blocked``     设备都在, 光束没到 CCD                       不能 (要人查光路)
``probe_setup``            探针自己没装好 (gxipy 不在 PYTHONPATH)      不能 (**立刻报错**)
=========================  ==========================================  ==============

模块级**不 import 任何驱动/SDK**: 这里只做字符串分类与文件锁, 脚本 import 它
不会碰到设备 (:func:`probe_devices` 才在**函数体内** import 驱动)。

公开 API
--------

* 分类: :func:`classify_device_unavailable` / :func:`classify_probe_setup_error` /
  :func:`escalate_stale_state`, 以及 ``KIND_*`` / ``WAITABLE_KINDS`` /
  ``DEVICE_UNAVAILABLE_INFO``。
* 探针: :func:`probe_devices` / :func:`run_device_probe` / :func:`parse_probe_output`。
* 锁: :class:`DeviceLock` (:meth:`DeviceLock.acquire` / ``release`` / ``holder`` /
  ``is_held_by_me``), :func:`default_lock_path`, :func:`process_tree_pids`。
* 等待: :func:`wait_until_idle` → :class:`WaitVerdict`。
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 仓库根 (探针子进程要 PYTHONPATH)
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    """仓库根目录。

    从本文件往上找「含 ``src/ao_shaping`` 的那一层」; 找不到 (非 editable
    安装) 就退到当前工作目录 —— 探针只是要 PYTHONPATH, 找不到时也不会崩。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "ao_shaping").is_dir():
            return parent
    return Path.cwd()


ROOT = _repo_root()


# ---------------------------------------------------------------------------
# 设备不可用的判别 (模块级纯函数) —— **唯一事实源**
# ---------------------------------------------------------------------------
#: 「设备不可用」的归类。**处置不同, 所以必须分开**:
#: 断电类要人重新插拔 (等是等不来的); 被占类别人在用, 等一会儿就空出来。
KIND_SLM_ABSENT = "slm_absent"
KIND_SLM_BUSY = "slm_busy"
KIND_CAMERA_BUSY = "camera_busy"
KIND_CAMERA_ABSENT = "camera_absent"
#: 光路不通: 设备都在 (SLM 能开、相机能开), 但**相机上一个光斑都没有** ⇒
#: 光束没到相机 (光路被挡/被挪/光源关了)。
KIND_LIGHT_PATH_BLOCKED = "light_path_blocked"
#: 探针自己没装好 (大恒 SDK 不在 PYTHONPATH 里) —— 这不是「设备被占」,
#: **必须立刻报错**, 绝不能当成「设备被占」去空等 (2026-10-21 空转 55 分钟)。
KIND_PROBE_SETUP = "probe_setup"
#: **SDK 陈旧状态** (2026-10-22 新增): SDK 报句柄已被打开, 但**设备锁是空的**
#: —— 没有任何活着的持有者 ⇒ 上一轮留下的陈旧句柄。**排队等不来**, 处置是
#: **给 SLM 控制器物理断电重置**。
KIND_SDK_STALE = "sdk_stale"

#: 可以自动等的那几类 (别人在用 ⇒ 等一会儿就好)。
WAITABLE_KINDS = frozenset({KIND_SLM_BUSY, KIND_CAMERA_BUSY})

#: 「陈旧句柄」的处置**按设备**给: SLM 的句柄只能靠控制器断电清掉; 相机的句柄
#: 挂在 gxipy 侧, 重开 SDK / 重启相机即可 (必要时也给 SLM 控制器断电)。
SDK_STALE_REMEDY: dict[str, str] = {
    KIND_SLM_BUSY: (
        "给 SLM 控制器物理断电重置 (断电几秒再上电) —— 排队等不来, "
        "陈旧句柄只能靠断电清掉"
    ),
    KIND_CAMERA_BUSY: (
        "相机句柄停在 gxipy 侧: 重开相机 SDK 或重启相机 (必要时也给 SLM "
        "控制器物理断电重置) —— 排队等不来"
    ),
}

#: 每种归类的中文说明与处置办法 (操作者看到的就是这两句)。
DEVICE_UNAVAILABLE_INFO: dict[str, tuple[str, str]] = {
    KIND_SLM_ABSENT: (
        "SLM 未找到USB设备: 设备没插上或没通电",
        "需要人重新插拔 SLM 控制器的电源/USB —— 等是等不来的",
    ),
    KIND_SLM_BUSY: (
        "SLM 的 USB 句柄已被别的进程打开",
        "别的进程正在用这台 SLM, 等它跑完就空出来 (可加 --wait-until-idle 自动等)",
    ),
    KIND_CAMERA_BUSY: (
        "相机没初始化成功 (camera not initialized): 相机被别的进程占着",
        "别的进程正在用这台相机, 等它跑完就空出来 (可加 --wait-until-idle 自动等)",
    ),
    KIND_CAMERA_ABSENT: (
        "相机不在 (Camera ID not found)",
        "需要人检查相机的供电/网线 —— 等是等不来的",
    ),
    KIND_LIGHT_PATH_BLOCKED: (
        "相机上没有任何光斑 (SLM 与相机都打开着, 但探测器上没读到光斑)",
        "先确认**激光器是否开着** (2026-10-22 12:15-12:59 那 45 分钟就是这样: "
        "光源被关掉了, 属正常操作); 再查实验系统的三段光路: 光束 → SLM 面板 → 2f 透镜 → CCD "
        "(面板被挪/被挡) —— 等是等不来的",
    ),
    KIND_PROBE_SETUP: (
        "探针自己没装好 (大恒 SDK gxipy 不在 PYTHONPATH 里)",
        "这不是「设备被占」, 等不来: 请把 libs/ 加进 PYTHONPATH 后重试",
    ),
    KIND_SDK_STALE: (
        "SDK 报设备被占, 但设备锁是空的 (没有任何活着的持有者) —— 陈旧句柄",
        SDK_STALE_REMEDY[KIND_SLM_BUSY],
    ),
}

#: 光路受阻时给操作者的**三段光路**检查清单。
LIGHT_PATH_CHECKLIST = "光束 → SLM 面板 → 2f 透镜 → CCD"

#: 设备不可用的特征串。**数字错误码与中文短语都认**: 驱动异常文本带码
#: (「错误码: -10002」), 而 GUI 的中文提示只带短语。
_SLM_ABSENT_RE = re.compile(r"未找到USB设备|错误码[:：]?\s*-10002(?!\d)")
_SLM_BUSY_RE = re.compile(r"USB设备已打开|错误码[:：]?\s*-10003(?!\d)")
_CAMERA_BUSY_RE = re.compile(r"camera not initialized|REPEAT_OPENED|device has been open")
_CAMERA_ABSENT_RE = re.compile(r"camera not found|Camera ID \d+ not found")
#: 光路不通的特征串 (探针的中文提示 + 曝光阶梯估不出上限时的英文报错)。
_LIGHT_PATH_BLOCKED_RE = re.compile(r"光路受阻|光束没有到达相机|no ladder frame carries signal")

#: 异常文本里的 Santec 错误码 (「错误码: -10003」/「错误码 -10003」)。
_SANTEC_CODE_RE = re.compile(r"错误码[:：]?\s*(-?\d+)")

#: 「**探针自己**没装好」的特征 (不是设备被占, 是环境不对) —— 这类必须
#: **立刻**报错退出, 绝不能当成「设备被占」去空等 (2026-10-21 空转 55 分钟)。
PROBE_SETUP_ERROR_RES = (
    re.compile(r"name 'gx' is not defined"),
    re.compile(r"No module named ['\"]gxipy['\"]"),
    re.compile(r"missing SDK"),
    re.compile(r"ModuleNotFoundError"),
    re.compile(r"ImportError"),
)

#: 「判定理由原文」最多留多少字符 (够看清一句报错, 又不至于把整页日志塞进索引)。
REASON_TEXT_CHARS = 400


@dataclass(frozen=True)
class DeviceUnavailable:
    """「设备不可用」的判定结果: 哪一类设备 + 为什么 + 该怎么办。"""

    kind: str
    """归类 (:data:`KIND_SLM_ABSENT` 等)。"""

    code: int | None = None
    """Santec 错误码 (相机侧没有码, 是 ``None``)。"""

    detail: str = ""
    """一句话说清根因 (给操作者看)。"""

    remedy: str = ""
    """该怎么处置。"""

    reason_text: str = ""
    """**判定理由原文** —— 命中判定的那一行 (从子进程输出里摘出来的)。

    为什么留着原文: 「光路不通」这句话本身不说明**是哪一段**断了, 而操作者要
    看的正是探针给出的那一句 (它带着三段光路的检查清单与实测数字)。"""

    @property
    def waitable(self) -> bool:
        """这一类能不能靠「等」解决 (别人在用 vs 设备没插上 vs 陈旧句柄)。"""
        return self.kind in WAITABLE_KINDS

    def headline(self) -> str:
        """根因的一句话 (带错误码)。"""
        code = f" (错误码 {self.code})" if self.code is not None else ""
        return f"{self.detail}{code}"

    def advice(self) -> str:
        """处置建议。"""
        return self.remedy


def _reason_text(text: str, pattern: re.Pattern[str]) -> str:
    """摘出命中判定的那一行原文。

    Args:
        text: 子进程输出 (stderr + stdout 拼起来)。
        pattern: 判定用的正则。

    Returns:
        命中处所在的**整行** (去掉首尾空白, 最多 :data:`REASON_TEXT_CHARS` 字符);
        没命中则空字符串。
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return ""
    start = matches[-1].start()
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end].strip()[:REASON_TEXT_CHARS]


def _extract_santec_error_code(text: str) -> int | None:
    """取出文本里最后一个 Santec 错误码 (没有则 ``None``)。

    Args:
        text: 待解析文本。

    Returns:
        错误码整数, 或 ``None``。
    """
    codes = _SANTEC_CODE_RE.findall(text)
    if not codes:
        return None
    try:
        return int(codes[-1])
    except ValueError:  # pragma: no cover - 正则保证是整数
        return None


def classify_device_unavailable(text: str) -> DeviceUnavailable | None:
    """判定一段子进程输出是不是「设备不可用」, 并归类。

    为什么要有它: 2026-10-21 18:23 实测, SLM 的 USB 当时是断开的, 驱动把
    **27 组**全当普通失败记进索引, 100 s 内全灭 —— 根因 (设备根本不可用) 在
    索引里完全看不出来。「设备不可用」与「这一组坏了」是**两件不同的事**:
    前者要中止整轮, 后者跑下一组。

    五类归类 (处置不同, 所以必须分开): 断电类 (``slm_absent`` /
    ``camera_absent``) 要人处理; 被占类 (``slm_busy`` / ``camera_busy``)
    等一会儿就好; **光路不通** (``light_path_blocked``) 是实验系统光路的问题 ——
    设备都在, 但光束没到 CCD, 等是等不来的, 而且**必须立刻中止整轮**
    (实测 2026-10-22 12:5x: 驱动把剩下 8 组逐个跑了一遍, 每组都报
    「光束没有到达相机」, 根因在索引里被埋掉了)。

    ⚠️ **本函数只看文本**, 不看锁 —— 「SDK 陈旧状态」(:data:`KIND_SDK_STALE`)
    需要锁的状态才能判定, 由 :func:`escalate_stale_state` 负责。

    Args:
        text: 子进程的输出 (stderr + stdout 拼起来)。

    Returns:
        判定结果; 不是设备不可用时返回 ``None``。
    """
    if not text:
        return None
    code = _extract_santec_error_code(text)
    best_pos = -1
    best: DeviceUnavailable | None = None
    for kind, pattern in (
        (KIND_SLM_ABSENT, _SLM_ABSENT_RE),
        (KIND_SLM_BUSY, _SLM_BUSY_RE),
        (KIND_CAMERA_BUSY, _CAMERA_BUSY_RE),
        (KIND_CAMERA_ABSENT, _CAMERA_ABSENT_RE),
        # 🔴 光路不通: 设备都在, 光束没到 CCD ⇒ 立刻中止整轮, 不等、不续跑。
        (KIND_LIGHT_PATH_BLOCKED, _LIGHT_PATH_BLOCKED_RE),
    ):
        positions = [m.start() for m in pattern.finditer(text)]
        if not positions:
            continue
        detail, remedy = DEVICE_UNAVAILABLE_INFO[kind]
        # 光路不通**不是**设备错误码能解释的: 输出里若混着别的 Santec 码
        # (例如上一次失败留下的), 挂上去只会把根因指歪 ⇒ 这一类不带码。
        candidate = DeviceUnavailable(
            kind=kind,
            code=None if kind == KIND_LIGHT_PATH_BLOCKED else code,
            detail=detail,
            remedy=remedy,
            reason_text=_reason_text(text, pattern),
        )
        # 取**最后**出现的那一处: 一组失败时最后一条错误才是致命原因。
        if positions[-1] > best_pos:
            best_pos = positions[-1]
            best = candidate
    return best


def classify_probe_setup_error(text: str) -> str | None:
    """判定一段输出是不是「探针自己没装好」(而不是设备被占)。

    Args:
        text: 探针的 stderr (或异常文本)。

    Returns:
        给操作者看的中文原因; 不是环境问题则返回 ``None``。
    """
    if not text:
        return None
    for pattern in PROBE_SETUP_ERROR_RES:
        if pattern.search(text):
            return (
                "探针自己没装好 (大恒 SDK gxipy 不在 PYTHONPATH 里 —— 它只随仓库"
                "分发在 libs/ 下)。这不是「设备被占」, 不能靠等解决: 请把 "
                "libs/ 加进 PYTHONPATH 后重试。"
            )
    return None


def escalate_stale_state(verdict: DeviceUnavailable) -> DeviceUnavailable:
    """把「被占」升级成「SDK 陈旧状态」。

    前提由调用方保证: **设备锁是空的** (没有任何活着的持有者) 且**没有别的
    硬件进程**。此时 SDK 仍然报句柄已被打开 ⇒ 只剩一种解释: 上一轮留下的
    **陈旧句柄** ⇒ **排队等不来**, 处置是**给 SLM 控制器物理断电重置**。

    Args:
        verdict: 文本分类得到的判定 (只对「被占」两类有意义)。

    Returns:
        归类为 :data:`KIND_SDK_STALE` 的判定; 输入不是「被占」类时原样返回。
    """
    if verdict is None or verdict.kind not in WAITABLE_KINDS:
        return verdict
    detail, remedy = DEVICE_UNAVAILABLE_INFO[KIND_SDK_STALE]
    return DeviceUnavailable(
        kind=KIND_SDK_STALE,
        code=verdict.code,
        detail=detail,
        remedy=SDK_STALE_REMEDY.get(verdict.kind, remedy),
        reason_text=verdict.reason_text,
    )


# ---------------------------------------------------------------------------
# 进程表 (祖先链 / 后代)
# ---------------------------------------------------------------------------
#: 进程表缓存的有效期 (s) —— 等待循环每轮都要问「谁持有」, 不必每次重扫。
_PROCESS_TABLE_TTL_S = 0.5

_PROCESS_TABLE_CACHE: dict[str, Any] = {"at": 0.0, "table": {}}


def _windows_process_table() -> dict[int, tuple[int, str]]:
    """Windows: 用 ``CreateToolhelp32Snapshot`` 列出 ``pid -> (ppid, 可执行文件名)``。

    为什么不用 powershell/CIM: 那条路一次要几百毫秒到几秒, 而等待循环每轮
    都要问「自己这条进程树上还有谁」⇒ 太慢。快照是**微秒级**的, 且能看见
    **非 python** 的中间层 (cmd.exe) —— 而那正是假阳性的根源 (自己启动的
    子进程经由 cmd.exe 起了 python, 旧判据认不出来 ⇒ 报「别的会话占用」)。
    """
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot is None or snapshot == INVALID_HANDLE_VALUE:
        return {}
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                table[int(entry.th32ProcessID)] = (
                    int(entry.th32ParentProcessID),
                    str(entry.szExeFile),
                )
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def _posix_process_table() -> dict[int, tuple[int, str]]:
    """POSIX: 从 ``/proc`` 读 ``pid -> (ppid, comm)`` (读不到就退化到 ``ps``)。"""
    table: dict[int, tuple[int, str]] = {}
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # comm 字段可能带空格与括号 ⇒ 从最后一个 ')' 之后开始切。
            cut = stat.rfind(")")
            if cut < 0:
                continue
            comm = stat[stat.find("(") + 1: cut]
            fields = stat[cut + 2:].split()
            if len(fields) < 2:
                continue
            try:
                table[pid] = (int(fields[1]), comm)
            except ValueError:
                continue
        return table
    try:
        completed = subprocess.run(  # 固定命令
            ["ps", "-eo", "pid=,ppid=,comm="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in completed.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            table[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
        except ValueError:
            continue
    return table


def process_table_detail(*, fresh: bool = False) -> dict[int, tuple[int, str]]:
    """当前系统的 ``pid -> (ppid, 可执行文件名)`` 表 (带 0.5 s 缓存)。

    Args:
        fresh: 忽略缓存, 强制重扫。

    Returns:
        ``{pid: (ppid, exe)}``; 查不到时是空字典。
    """
    now = time.monotonic()
    if not fresh:
        cached = _PROCESS_TABLE_CACHE
        if now - float(cached["at"]) < _PROCESS_TABLE_TTL_S:
            return dict(cached["table"])
    table = _windows_process_table() if os.name == "nt" else _posix_process_table()
    _PROCESS_TABLE_CACHE["at"] = now
    _PROCESS_TABLE_CACHE["table"] = table
    return dict(table)


def process_table(*, fresh: bool = False) -> dict[int, int]:
    """当前系统的 ``pid -> ppid`` 表 (带缓存)。

    Args:
        fresh: 忽略缓存, 强制重扫。

    Returns:
        ``{pid: ppid}``; 查不到时是空字典。
    """
    return {
        pid: ppid
        for pid, (ppid, _exe) in process_table_detail(fresh=fresh).items()
    }


def _is_python_process(exe: str) -> bool:
    """这个可执行文件名像不像 python 解释器。"""
    name = Path(str(exe)).name.lower()
    return name.startswith("python") or name.startswith("py.exe")


def process_tree_pids(self_pid: int | None = None, *, fresh: bool = False) -> set[int]:
    """「自己这一棵进程树」的 pid 集合 = 自身 + 祖先 + 后代。

    两条规则是**故意不一样**的:

    * **祖先**只沿 **python** 进程往上走。祖先若是 ``cmd.exe`` / ``powershell.exe``
      之类, 它启动的其它进程未必属于我们这条线 (同一个终端里可以开好几个会话)
      ⇒ 不能把它们一并当成自己的。
    * **后代**沿完整进程表往下走, **穿过** ``cmd.exe`` 这样的中间层。这正是
      2026-10-22 那个假阳性的形态: 自己启动的子进程经由 cmd.exe 起了 python,
      旧判据只认「列表里的 pid」⇒ 认不出来 ⇒ 报「别的会话占用」。

    Args:
        self_pid: 自己的 pid; ``None`` = 当前进程。
        fresh: 忽略进程表缓存, 强制重扫 (刚起的子进程可能不在缓存里)。

    Returns:
        pid 集合 (至少含自己)。
    """
    pid = os.getpid() if self_pid is None else int(self_pid)
    table = process_table_detail(fresh=fresh)
    own = {pid}
    # 祖先 (只穿 python)
    current = pid
    guard = 0
    while guard < 32:
        guard += 1
        parent, _exe = table.get(current, (0, ""))
        if not parent or parent in own:
            break
        if not _is_python_process(table.get(parent, (0, ""))[1]):
            break
        own.add(parent)
        current = parent
    # 后代 (穿任何中间层)
    children: dict[int, list[int]] = {}
    for child, (parent, _exe) in table.items():
        children.setdefault(parent, []).append(child)
    stack = list(own)
    while stack:
        current = stack.pop()
        for child in children.get(current, ()):
            if child not in own:
                own.add(child)
                stack.append(child)
    return own


def pid_alive(pid: int) -> bool:
    """某个 pid 还活着吗。

    Args:
        pid: 进程号。

    Returns:
        活着 ⇒ ``True``; 查不到进程表时**保守地**当作活着 (宁可多等, 也不
        误判「没人持有」)。
    """
    if pid <= 0:
        return False
    table = process_table()
    if not table:
        return True
    return pid in table


def current_cmdline() -> str:
    """当前进程的命令行 (自己记, 不用问系统 ⇒ 精确且便宜)。"""
    argv = [sys.executable, *sys.argv]
    if os.name == "nt":
        return subprocess.list2cmdline([str(a) for a in argv])
    return " ".join(shlex.quote(str(a)) for a in argv)


def _host_name() -> str:
    """主机名 (锁元数据里用来分辨是不是同一台机器)。"""
    if os.name == "nt":
        return os.environ.get("COMPUTERNAME", "")
    uname = getattr(os, "uname", None)
    return uname().nodename if callable(uname) else ""


# ---------------------------------------------------------------------------
# 占用者进程 (旧判据: 命令行关键字) —— 锁没覆盖到的老脚本靠它兜底
# ---------------------------------------------------------------------------
#: 占用者进程的命令行里出现这些关键字就算「可能在用设备」。
#:
#: 🔴 **刻意不含裸的 ``ao_shaping``**: 本仓库里**每一个** python 进程 (pytest、
#: 编辑器、报告生成器) 的命令行里都有它 ⇒ 假阳性满天飞, 等待永远等不到。
#:
#: 🔴 **也刻意不含裸的 ``slm``**: 它同样会命中**只是在测试路径里提到 slm** 的进程 ——
#: 实测另一个会话的 ``pytest .../tests/ao_shaping/gui/slm/test_zernike_abort_repro.py``
#: 与 ``pytest .../tests/ao_shaping/optimizer/wfless/test_slm_gs_refine.py`` 都被当成了
#: 「占用设备的进程」, 于是一台**没人用设备**的机器上, 采集永远等下去 (2026-10-22 实测)。
OCCUPIER_KEYWORDS: tuple[str, ...] = (
    "scripts/slm/",
    "src/ao_shaping/gui/",
    "ao_shaping.drivers.santec",
    "ao_shaping.drivers.ccd",
    "ao_shaping.tools.slm",
    "daheng",
    "santec",
    "gxipy",
    "ao_shaping/main.py",
    "-m ao_shaping.main",
)

#: **纯离线报告生成器不算占用者** (仓库约定: 报告生成脚本一律叫
#: ``scripts/generate_*_report.py``)。
NON_DEVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"generate_[a-z0-9_]*report", re.IGNORECASE),
    re.compile(r"build_[a-z0-9_]*report", re.IGNORECASE),
)

#: 显式声明「这些进程是我们自己的」的环境变量 (逗号分隔的 pid)。
OWN_PIDS_ENV = "AO_HW_OWN_PIDS"


@dataclass(frozen=True)
class ProcessInfo:
    """一个 python 进程 (pid / 父 pid / 命令行)。"""

    pid: int
    ppid: int = 0
    """父进程 pid (用来算进程树 —— 我们自己启动的进程不算占用者)。"""

    cmdline: str = ""
    """命令行 (可能为空: 有些进程的 CommandLine 拿不到)。"""


def query_windows_python_processes() -> list[ProcessInfo]:
    """用 CIM 列出 python 进程 (pid, 父 pid, 命令行)。

    非 Windows 或查询失败时返回空列表 —— 查不到占用者不该让整轮失败, 退化成
    「只靠探针」即可。

    Returns:
        ``[ProcessInfo, ...]``。
    """
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' } | "
        "ForEach-Object { "
        "\"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        completed = subprocess.run(  # 固定命令, 不带用户输入
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    processes: list[ProcessInfo] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, rest = line.partition("\t")
        ppid_text, _, cmdline = rest.partition("\t")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        try:
            ppid = int(ppid_text)
        except ValueError:
            ppid = 0
        processes.append(ProcessInfo(pid=pid, ppid=ppid, cmdline=cmdline))
    return processes


def own_process_ids(
    processes: Sequence[ProcessInfo],
    *,
    self_pid: int | None = None,
    extra_pids: Sequence[int] = (),
    tree_table: dict[int, int] | None = None,
) -> set[int]:
    """算「我们自己」这条线上的进程 (自身 + 祖先 + 后代 + 显式声明的)。

    为什么: 「别的会话的硬件进程」与「我们自己启动的」必须分开 —— 我们自己
    启动的探针子进程 (它**会**开一次 SLM) 不能算成「别人在用设备」, 否则等待
    永远等不到。

    Args:
        processes: 进程列表 (通常是 python 进程)。
        self_pid: 自己的 pid; ``None`` = 当前进程。
        extra_pids: 额外算作「我们自己的」pid (例如同一条线上的另一个脚本)。
        tree_table: 完整的 ``pid -> ppid`` 表 (含 cmd.exe 之类的中间层);
            给了它, 后代就能穿过非 python 的中间层被认出来。

    Returns:
        pid 集合。
    """
    pid = os.getpid() if self_pid is None else int(self_pid)
    by_pid = {proc.pid: proc for proc in processes}
    own: set[int] = {pid}
    own.update(int(extra) for extra in extra_pids)
    # 祖先: 启动我们的那个 **python** 进程 (同一条线上的脚本) 也不算占用者。
    # 只沿**列表里有的**进程往上走: 祖先若是 cmd.exe 之类, 它的其它子进程未必是
    # 我们这条线的 (别的会话可能共用同一个终端) ⇒ 不能把它们一并当成自己的。
    current = by_pid.get(pid)
    while current is not None and current.ppid:
        if current.ppid in own or current.ppid not in by_pid:
            break
        own.add(current.ppid)
        current = by_pid.get(current.ppid)
    if tree_table:
        # 后代: 走**完整进程表** ⇒ 经由 cmd.exe 起的孙进程也算我们自己的。
        children: dict[int, list[int]] = {}
        for child, parent in tree_table.items():
            children.setdefault(parent, []).append(child)
        stack = list(own)
        while stack:
            current_pid = stack.pop()
            for child in children.get(current_pid, ()):
                if child not in own:
                    own.add(child)
                    stack.append(child)
        return own
    # 后代: 我们启动的探针子进程 (以及我们那条线上的脚本启动的子进程)。
    changed = True
    while changed:
        changed = False
        for proc in processes:
            if proc.ppid and proc.ppid in own and proc.pid not in own:
                own.add(proc.pid)
                changed = True
    return own


def select_device_using_processes(
    processes: Sequence[ProcessInfo],
    *,
    keywords: Sequence[str] = OCCUPIER_KEYWORDS,
    exclude_pids: Sequence[int] = (),
    own_paths: Sequence[str | Path] = (),
    self_pid: int | None = None,
    tree_table: dict[int, int] | None = None,
) -> list[ProcessInfo]:
    """从进程列表里挑出「**别的会话的**硬件进程」。

    自己这条线上的进程一律排除: 自身、祖先、后代 (探针子进程), 以及命令行里
    出现**我们自己的脚本**的那些 (同一条线上的另一个实例)。

    Args:
        processes: 进程列表 (可注入 ⇒ 测试不用真查 CIM)。
        keywords: 命令行里要匹配的关键字。
        exclude_pids: 额外要排除的 pid。
        own_paths: 我们自己的脚本路径 (命令行里出现它们的不算占用者)。
        self_pid: 自己的 pid; ``None`` = 当前进程。
        tree_table: 完整进程表 (让后代能穿过 cmd.exe 这样的中间层)。

    Returns:
        ``[ProcessInfo, ...]``。
    """
    own = own_process_ids(
        processes,
        self_pid=self_pid,
        extra_pids=exclude_pids,
        tree_table=tree_table,
    )
    needles: list[str] = []
    for path in own_paths:
        text = str(path)
        needles.append(text.lower())
        needles.append(Path(text).name.lower())
    hits: list[ProcessInfo] = []
    for proc in processes:
        if proc.pid in own:
            continue
        lowered = proc.cmdline.lower()
        if any(needle in lowered for needle in needles):
            continue
        if any(pattern.search(proc.cmdline) for pattern in NON_DEVICE_PATTERNS):
            # 纯离线报告生成器 —— 不住设备, 不算占用者。
            continue
        if any(str(keyword).lower() in lowered for keyword in keywords):
            hits.append(proc)
    return hits


def find_device_using_processes(
    keywords: Sequence[str] = OCCUPIER_KEYWORDS,
    *,
    exclude_pids: Sequence[int] = (),
    own_paths: Sequence[str | Path] = (),
) -> list[ProcessInfo]:
    """挑出命令行里含设备关键字、且**不是我们自己**的进程。

    Args:
        keywords: 命令行里要匹配的关键字。
        exclude_pids: 额外要排除的 pid。
        own_paths: 我们自己的脚本路径。

    Returns:
        ``[ProcessInfo, ...]``。
    """
    return select_device_using_processes(
        query_windows_python_processes(),
        keywords=keywords,
        exclude_pids=exclude_pids,
        own_paths=own_paths,
        tree_table=process_table(),
    )


def format_occupier_lines(hits: Sequence[ProcessInfo]) -> list[str]:
    """把占用者列表排成给人看的行。"""
    return [f"  pid {proc.pid}: {proc.cmdline}" for proc in hits]


# ---------------------------------------------------------------------------
# 设备可用性探针
# ---------------------------------------------------------------------------
#: 探针子进程输出的协议行前缀 (后面跟 status 与 key=value 字段)。
PROBE_LINE_PREFIX = "DEVICE_PROBE:"
#: 探针子进程输出的说明行前缀 (人看的, 可以含空格)。
PROBE_NOTE_PREFIX = "DEVICE_PROBE_NOTE:"

#: 探针的三种结论。
PROBE_IDLE = "idle"                 # 设备可用
PROBE_UNAVAILABLE = "unavailable"   # 设备不可用 (kind 说的是哪一类)
PROBE_BROKEN = "broken"             # 探针自己没装好 —— **不是**设备的问题

#: 探针结论的**来源**。
#:
#: 🔴 只有**真探针** (直接问 SDK) 的「被占」才是「SDK 陈旧状态」的证据;
#: 外部注入的探针 (测试替身 / 别的探针脚本) 的结论**不**作陈旧判定依据。
PROBE_SOURCE_SDK = "sdk"
PROBE_SOURCE_INJECTED = "injected"

#: 探针每隔多久重试一次 (s)。
PROBE_POLL_S = 15.0
#: 探针子进程的超时 (s) —— 探针不该把驱动挂住。
PROBE_TIMEOUT_S = 120.0
#: 探针命令的环境变量名 (测试与非常规环境用它换成别的探针)。
PROBE_CMD_ENV = "AO_HW_DEVICE_PROBE_CMD"

#: 等待时每隔多久汇报一次 (s) —— 别刷屏。
WAIT_REPORT_INTERVAL_S = 60.0


@dataclass(frozen=True)
class ProbeResult:
    """探针的结论。"""

    status: str
    """``idle`` / ``unavailable`` / ``broken``。"""

    kind: str = ""
    """设备不可用时的归类 (见 :data:`KIND_SLM_BUSY` 等)。"""

    code: int | None = None
    """Santec 错误码 (若有)。"""

    message: str = ""
    """人看的说明 (可含空格, 走 NOTE 行)。"""

    source: str = PROBE_SOURCE_INJECTED
    """结论来源 (:data:`PROBE_SOURCE_SDK` = 真问过 SDK)。"""


def format_probe_line(result: ProbeResult) -> str:
    """把探针结论格式化成协议行 (字段值里不许有空格)。"""
    parts = [PROBE_LINE_PREFIX, result.status]
    if result.kind:
        parts.append(f"kind={result.kind}")
    if result.code is not None:
        parts.append(f"code={int(result.code)}")
    return " ".join(parts)


def parse_probe_output(stdout: str, stderr: str = "") -> ProbeResult:
    """解析探针子进程的输出。

    Args:
        stdout: 子进程的标准输出。
        stderr: 子进程的标准错误 (探针自己崩了时结论在这里)。

    Returns:
        探针结论; 没给出结论时是 ``broken`` (附上原因)。
    """
    line: str | None = None
    note = ""
    for raw in stdout.splitlines():
        text = raw.strip()
        if text.startswith(PROBE_LINE_PREFIX):
            line = text[len(PROBE_LINE_PREFIX):].strip()
        elif text.startswith(PROBE_NOTE_PREFIX):
            note = text[len(PROBE_NOTE_PREFIX):].strip()
    if line is None:
        reason = classify_probe_setup_error(stderr)
        if reason is None:
            tail = stderr.strip()
            reason = (
                f"探针没有给出任何结论 (stderr 末尾: {tail[-300:]})"
                if tail
                else "探针没有给出任何结论"
            )
        return ProbeResult(status=PROBE_BROKEN, message=reason)
    tokens = line.split()
    status = tokens[0] if tokens else ""
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    code: int | None = None
    if "code" in fields:
        try:
            code = int(fields["code"])
        except ValueError:
            code = None
    return ProbeResult(
        status=status,
        kind=fields.get("kind", ""),
        code=code,
        message=note or fields.get("message", ""),
        source=fields.get("source", PROBE_SOURCE_INJECTED),
    )


def _ensure_repo_paths() -> None:
    """把仓库的四个搜索路径塞进 ``sys.path`` (探针子进程用)。

    为什么不能只靠环境变量: ``gxipy`` 只随仓库分发在 ``libs/`` 下, 不在
    site-packages 里。漏了它, 相机会在半途抛 ``NameError: name 'gx' is not
    defined`` —— 于是「相机被占」与「相机没装好」分不开, 而等设备的逻辑会永远
    等下去 (2026-10-21 实测空转 55 分钟)。所以这里**显式**补一遍。
    """
    for path in (ROOT, ROOT / "src", ROOT / "libs", ROOT / "libs" / "OOPAO"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_probe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """探针子进程的环境变量 = 当前环境 + 仓库的四个搜索路径。

    ``gxipy`` 只随仓库分发在 ``libs/`` 下, 不在 site-packages 里 ⇒ 漏了它,
    相机会在半途抛 ``NameError: name 'gx' is not defined``。
    """
    env = os.environ.copy()
    wanted = [
        str(ROOT),
        str(ROOT / "src"),
        str(ROOT / "libs"),
        str(ROOT / "libs" / "OOPAO"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in wanted if p not in existing.split(os.pathsep)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    for key, value in (extra or {}).items():
        env[str(key)] = str(value)
    return env


def probe_devices(
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
) -> ProbeResult:
    """真开一次 SLM + 相机再关 —— 只探可用性。

    三条硬约束:

    1. **恒用 memory 模式**开 SLM: DVI 模式的 ``open()`` 可能挂死, 挂死后
       必须物理断电才恢复 (``drivers/slm/AGENTS.md``)。
    2. **自己带 PYTHONPATH**: 见 :func:`_ensure_repo_paths`。
    3. 🔴 **返回前必须释放句柄**: 探针自己握着的句柄会让**下一次**开锁拿到
       -10003 ⇒ 于是「设备被占」的结论有一部分是探针自己造的。SLM 与相机
       都在 ``finally`` 里关掉, 关不掉也要吞掉异常 (不让关闭失败盖掉结论)。

    Args:
        slm_number: SLM 编号。
        cam_id: 相机编号。
        cam_type: 相机类型 (默认大恒)。
        exposure_ms: 开相机时的曝光 (ms)。

    Returns:
        探针结论。
    """
    _ensure_repo_paths()
    # -- SLM: 恒用 memory 模式 --------------------------------------------
    try:
        from ao_shaping.drivers.slm.santec import Santec

        slm = Santec(slm_number=int(slm_number), video_mode=0)
        try:
            slm.open()
        finally:
            with contextlib.suppress(Exception):
                slm.close()
    except Exception as exc:  # 探针要把任何异常都翻译成结论
        return _probe_failure(f"{type(exc).__name__}: {exc}")
    # -- 相机 --------------------------------------------------------------
    try:
        from ao_shaping.drivers.ccd.common import create_camera

        cam = create_camera(
            cam_type, cam_id=int(cam_id), exposure_time_ms=float(exposure_ms)
        )
        try:
            cam.open()
        finally:
            with contextlib.suppress(Exception):
                cam.close()
    except Exception as exc:  # 探针契约: 相机侧 open() 失败同样翻译成结论, 不向巡检抛异常
        return _probe_failure(f"{type(exc).__name__}: {exc}")
    return ProbeResult(status=PROBE_IDLE, source=PROBE_SOURCE_SDK)


def _probe_failure(text: str) -> ProbeResult:
    """把探针里的异常翻译成结论。"""
    setup = classify_probe_setup_error(text)
    if setup is not None:
        return ProbeResult(
            status=PROBE_BROKEN,
            message=setup,
            source=PROBE_SOURCE_SDK,
        )
    verdict = classify_device_unavailable(text)
    if verdict is not None:
        return ProbeResult(
            status=PROBE_UNAVAILABLE,
            kind=verdict.kind,
            code=verdict.code,
            message=verdict.detail,
            source=PROBE_SOURCE_SDK,
        )
    return ProbeResult(
        status=PROBE_BROKEN,
        message=f"探针异常: {text}",
        source=PROBE_SOURCE_SDK,
    )


def default_probe_cmd(
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
) -> list[str]:
    """内置探针的命令 (本模块自己, 走 ``--probe-devices`` 模式)。"""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-devices",
        "--probe-slm-number",
        str(int(slm_number)),
        "--probe-cam-id",
        str(int(cam_id)),
        "--probe-cam-type",
        str(cam_type),
        "--probe-exposure-ms",
        str(float(exposure_ms)),
    ]


def parse_probe_cmd(text: str) -> list[str]:
    """把探针命令的文本解析成 argv。

    JSON 数组优先 (路径里有空格时只有 JSON 靠得住), 否则按空白切。

    Args:
        text: 命令文本。

    Returns:
        argv 列表 (空文本 → 空列表)。
    """
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if isinstance(data, list):
            return [str(item) for item in data]
    return stripped.split()


def resolve_probe_cmd(explicit: str | None = None) -> list[str] | None:
    """探针命令: 命令行参数优先, 其次环境变量, 都没有则用内置探针。"""
    if explicit:
        parsed = parse_probe_cmd(explicit)
        if parsed:
            return parsed
    env_text = os.environ.get(PROBE_CMD_ENV, "")
    if env_text.strip():
        parsed = parse_probe_cmd(env_text)
        if parsed:
            return parsed
    return None


def run_device_probe(
    probe_cmd: Sequence[str] | None = None,
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
) -> ProbeResult:
    """跑一次探针子进程, 解析它的结论。

    Args:
        probe_cmd: 探针命令; ``None`` = 用内置探针 (**真问过 SDK** ⇒
            :data:`PROBE_SOURCE_SDK`)。
        timeout_s: 探针子进程的超时 (s)。
        slm_number: SLM 编号 (内置探针用)。
        cam_id: 相机编号 (内置探针用)。
        cam_type: 相机类型 (内置探针用)。
        exposure_ms: 曝光 (ms, 内置探针用)。

    Returns:
        探针结论。
    """
    builtin = probe_cmd is None
    cmd = (
        list(probe_cmd)
        if probe_cmd
        else default_probe_cmd(slm_number, cam_id, cam_type, exposure_ms)
    )
    try:
        completed = subprocess.run(  # 命令来自调用者/环境, 故意如此
            list(cmd),
            cwd=str(ROOT),
            env=build_probe_env(),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            status=PROBE_BROKEN,
            message=f"探针超时 (>{float(timeout_s):.0f}s) —— 设备状态不明, 不等",
        )
    except OSError as exc:
        return ProbeResult(status=PROBE_BROKEN, message=f"探针起不来: {exc}")
    result = parse_probe_output(completed.stdout, completed.stderr)
    if builtin:
        # 内置探针 = 本模块自己的 ``--probe-devices`` 模式 ⇒ 真问过 SDK。
        result = replace(result, source=PROBE_SOURCE_SDK)
    return result


# ---------------------------------------------------------------------------
# 跨进程设备锁
# ---------------------------------------------------------------------------
#: 锁文件路径的环境变量名 (换机器/换用户时用它指到共享位置)。
LOCK_PATH_ENV = "AO_DEVICE_MUTEX_LOCK"

#: 锁文件里元数据区的起始偏移 (第 0 个字节是**互斥字节**, 永远不写内容)。
_METADATA_OFFSET = 1

#: 本进程持有的锁 (按解析后的路径) —— 同进程内第二个句柄去抢同一把锁时,
#: 靠操作系统语义的差异来判会不稳定, 自己记一份最省事。
_LOCAL_HOLDERS: dict[str, int] = {}


def default_lock_path(label: str = "bench") -> Path:
    """默认锁文件路径。

    默认放在**用户临时目录**下 (不是仓库里): 仓库的 ``data/`` 会被容器挂载,
    于是「宿主机上跑的硬件会话」与「容器里跑的单元测试」会抢同一把锁。

    Args:
        label: 锁的名字 (一台实验系统可以有多把锁, 例如按设备分)。

    Returns:
        锁文件路径 (文件**不一定存在** —— 只有真的加过锁才创建)。
    """
    override = os.environ.get(LOCK_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "ao_shaping" / "device_mutex" / f"{label}.lock"


@dataclass(frozen=True)
class LockHolder:
    """锁的持有者。"""

    pid: int
    """持有者的 pid。"""

    cmdline: str = ""
    """持有者的命令行 (持有者**自己**记的 ⇒ 精确)。"""

    acquired_at: str = ""
    """拿到锁的时刻 (ISO 8601)。"""

    acquired_at_epoch: float = 0.0
    """拿到锁的时刻 (Unix 时间戳)。"""

    ppid: int = 0
    """持有者的父进程 pid。"""

    unknown: bool = False
    """持有者**认不出来** (锁被占, 但元数据没写全)。"""

    def describe(self) -> str:
        """一行人话 (pid + 命令行)。"""
        if self.unknown or self.pid <= 0:
            return "持有者未知 (锁被占, 但元数据不完整)"
        text = self.cmdline or "(没有记下命令行)"
        return f"pid {self.pid}: {text}"


def _lock_handle(handle: Any) -> None:
    """给句柄加**排他**锁 (第 0 个字节)。

    Windows: ``msvcrt.locking`` (按**字节**锁, 非阻塞)。POSIX: ``fcntl.flock``。

    Raises:
        OSError: 锁已被别人持有。
    """
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle: Any) -> None:
    """解除 :func:`_lock_handle` 加的锁。"""
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DeviceLock:
    """实验系统的**跨进程**互斥锁。

    锁文件里有两个区:

    * 第 0 个字节 —— **互斥区**。用操作系统的排他锁占住 (Windows ``msvcrt``,
      POSIX ``fcntl``); 谁占住谁就是持有者。进程退出时操作系统自动释放 ⇒
      崩了不会留下死锁。
    * 第 1 个字节往后 —— **元数据区** (JSON): 持有者的 pid / 命令行 / 拿到锁
      的时刻。持有者**自己**写 (命令行自己最清楚), 读的人只读这一段, 不碰
      互斥区。

    典型用法::

        lock = DeviceLock.default()
        if not lock.acquire(timeout=1800):
            raise RuntimeError(f"设备一直被占: {lock.holder()}")
        try:
            ...用设备...
        finally:
            lock.release()
    """

    def __init__(self, path: str | Path | None = None, *, label: str = "bench") -> None:
        """建一把锁。

        Args:
            path: 锁文件路径; ``None`` = :func:`default_lock_path`。
            label: 锁的名字 (只用于默认路径)。
        """
        self._path = Path(path) if path is not None else default_lock_path(label)
        self._label = label
        self._handle: Any = None

    # -- 基本属性 ---------------------------------------------------------
    @property
    def path(self) -> Path:
        """锁文件路径。"""
        return self._path

    @property
    def label(self) -> str:
        """锁的名字。"""
        return self._label

    @classmethod
    def default(cls, label: str = "bench") -> "DeviceLock":
        """默认路径上的锁。"""
        return cls(default_lock_path(label), label=label)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"DeviceLock(path={str(self._path)!r}, held={self._held})"

    # -- 加锁 / 解锁 ------------------------------------------------------
    @property
    def _held(self) -> bool:
        """本对象当前是否握着锁。"""
        return self._handle is not None

    def _open(self) -> Any:
        """打开锁文件 (不存在就创建), 返回句柄。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with contextlib.suppress(OSError):
                self._path.touch()
        return open(self._path, "r+b")

    def _try_lock(self) -> bool:
        """试一次加锁 (非阻塞)。成功返回 ``True``。"""
        if self._held:
            return True
        if str(self._path.resolve()) in _LOCAL_HOLDERS:
            # 同进程里另一个对象已经持有 ⇒ 当作被占 (不依赖操作系统的句柄语义)。
            return False
        try:
            handle = self._open()
        except OSError:
            return False
        try:
            _lock_handle(handle)
        except OSError:
            with contextlib.suppress(OSError):
                handle.close()
            return False
        self._handle = handle
        _LOCAL_HOLDERS[str(self._path.resolve())] = os.getpid()
        return True

    def acquire(self, timeout: float = 0.0, *, poll_s: float = 0.5) -> bool:
        """加锁 (最多等 ``timeout`` 秒)。

        Args:
            timeout: 最长等多久 (s); ``0`` = 不等, 抢不到就返回 ``False``。
            poll_s: 重试间隔 (s)。

        Returns:
            拿到锁 ⇒ ``True``; 超时 ⇒ ``False`` (调用方可以问 :meth:`holder`
            到底是谁占着)。
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            if self._try_lock():
                self._write_metadata()
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(float(poll_s), remaining))

    def release(self) -> None:
        """放锁 (幂等)。**释放一定要在 ``finally`` 里。**"""
        handle, self._handle = self._handle, None
        _LOCAL_HOLDERS.pop(str(self._path.resolve()), None)
        if handle is None:
            return
        try:
            self._clear_metadata(handle)
        finally:
            try:
                _unlock_handle(handle)
            finally:
                with contextlib.suppress(OSError):
                    handle.close()

    def __enter__(self) -> "DeviceLock":
        """加锁 (拿不到就抛)。"""
        if not self.acquire(timeout=0.0):
            raise TimeoutError(f"设备锁抢不到: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """放锁。"""
        self.release()

    # -- 元数据 -----------------------------------------------------------
    def _write_metadata(self) -> None:
        """把持有者信息写进元数据区。"""
        handle = self._handle
        if handle is None:
            return
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "cmdline": current_cmdline(),
                "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "acquired_at_epoch": time.time(),
                "label": self._label,
                "host": _host_name(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            handle.seek(_METADATA_OFFSET)
            handle.write(payload)
            handle.truncate(_METADATA_OFFSET + len(payload))
            handle.flush()
        except OSError:
            # 元数据写不进去 (例如只读目录) ⇒ 锁仍然有效, 只是认不出持有者。
            pass

    @staticmethod
    def _clear_metadata(handle: Any) -> None:
        """清掉元数据 (放锁时)。"""
        if handle is None:
            return
        try:
            handle.seek(_METADATA_OFFSET)
            handle.truncate(_METADATA_OFFSET)
            handle.flush()
        except OSError:
            pass

    def _read_metadata(self) -> dict[str, Any] | None:
        """读元数据区 (不碰互斥区)。"""
        try:
            with open(self._path, "rb") as handle:
                handle.seek(_METADATA_OFFSET)
                blob = handle.read()
        except OSError:
            return None
        if not blob:
            return None
        try:
            data = json.loads(blob.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def write_holder_metadata(
        self,
        pid: int,
        *,
        cmdline: str = "",
        acquired_at: str = "",
    ) -> None:
        """**故障注入用**: 直接写持有者元数据 (不真的加锁)。

        用途: 单测里构造「锁的持有者是自己的父进程」这种情形 —— 真正的持有者
        是别的进程, 我们不能替它加锁, 但可以替它「说」自己是谁。

        Args:
            pid: 持有者的 pid。
            cmdline: 持有者的命令行。
            acquired_at: 拿到锁的时刻 (ISO 8601)。
        """
        payload = json.dumps(
            {
                "pid": int(pid),
                "ppid": 0,
                "cmdline": cmdline,
                "acquired_at": acquired_at or datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "acquired_at_epoch": time.time(),
                "label": self._label,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with contextlib.suppress(OSError):
                self._path.touch()
        with open(self._path, "r+b") as handle:
            handle.seek(_METADATA_OFFSET)
            handle.write(payload)
            handle.truncate(_METADATA_OFFSET + len(payload))
            handle.flush()

    # -- 查询 -------------------------------------------------------------
    def is_locked(self) -> bool:
        """锁现在**被占着**吗 (不关心是谁)。"""
        if self._held:
            return True
        if str(self._path.resolve()) in _LOCAL_HOLDERS:
            return True
        if not self._path.exists():
            return False
        try:
            handle = self._open()
        except OSError:
            return False
        try:
            _lock_handle(handle)
        except OSError:
            return True
        try:
            _unlock_handle(handle)
        except OSError:
            return True
        finally:
            with contextlib.suppress(OSError):
                handle.close()
        return False

    def holder(self) -> LockHolder | None:
        """当前持有者。

        Returns:
            持有者 (:class:`LockHolder`); **锁空**时返回 ``None``。锁被占但
            元数据不完整时返回一个 ``unknown`` 的持有者 (pid = 0)。
        """
        if not self._path.exists():
            return None
        data = self._read_metadata()
        locked = self.is_locked()
        if data is None:
            if locked:
                return LockHolder(pid=0, unknown=True)
            return None
        try:
            pid = int(data.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        if pid <= 0:
            return LockHolder(pid=0, unknown=True) if locked else None
        if not locked and not pid_alive(pid):
            # 陈旧元数据: 持有者已经没了 (崩了) ⇒ 锁是空的。
            return None
        return LockHolder(
            pid=pid,
            cmdline=str(data.get("cmdline", "")),
            acquired_at=str(data.get("acquired_at", "")),
            acquired_at_epoch=float(data.get("acquired_at_epoch", 0.0) or 0.0),
            ppid=int(data.get("ppid", 0) or 0),
        )

    def is_held_by_me(self, holder: LockHolder | None = None) -> bool:
        """锁是不是**自己这条进程树**持有 (自己 / 父 / 子)?

        这是假阳性的关键: 自己启动的探针子进程 (它**会**开一次 SLM) 持有锁时,
        **不是冲突** ⇒ 等待必须立刻放行。

        Args:
            holder: 已知的持有者; ``None`` = 自己查。

        Returns:
            是 ⇒ ``True``。
        """
        info = holder if holder is not None else self.holder()
        if info is None or info.pid <= 0:
            return False
        if info.pid in process_tree_pids():
            return True
        # 进程表的缓存可能是在子进程出生**之前**拍的一张 ⇒ 再拍一张 (快照是
        # 微秒级的), 否则刚起的探针子进程会被误判成「别人」。
        return info.pid in process_tree_pids(fresh=True)


# ---------------------------------------------------------------------------
# 规范等待
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaitVerdict:
    """等设备空闲的**结构化**结论。"""

    ok: bool
    """设备可用 (可以开跑)。"""

    status: str
    """``idle`` / ``timeout`` / ``unavailable`` / ``broken`` / ``stale``。"""

    exit_code: int
    """0 = 可用; 1 = 等不来/超时; 2 = 探针自己坏了。"""

    waited_s: float = 0.0
    """等了多久 (s)。"""

    reason: str = ""
    """一句话结论 (兼容旧的 ``(退出码, 说明)`` 形态)。"""

    detail: str = ""
    """根因描述。"""

    remedy: str = ""
    """处置建议。"""

    kind: str = ""
    """设备不可用的归类 (若有)。"""

    code: int | None = None
    """Santec 错误码 (若有)。"""

    holder: LockHolder | None = None
    """锁的持有者 (若有)。"""

    holder_is_me: bool = False
    """持有者是不是我们自己这条进程树。"""

    attempts: int = 0
    """等了多少轮。"""

    lock: DeviceLock | None = None
    """成功时**已经拿在手里**的锁 (调用方负责 ``release``)。"""

    def headline(self) -> str:
        """根因的一句话。"""
        if self.detail:
            code = f" (错误码 {self.code})" if self.code is not None else ""
            return f"{self.detail}{code}"
        return self.reason


def _other_hardware_processes(
    process_source: Callable[[], Sequence[ProcessInfo]] | None,
    *,
    keywords: Sequence[str],
    own_pids: Sequence[int],
    own_paths: Sequence[str | Path],
) -> list[ProcessInfo]:
    """当前**别的会话**在用设备的进程 (自己这条线上的已排除)。

    非 Windows (没有 CIM) ⇒ 空列表 ⇒ 退化成「只靠探针」, 不崩。

    ``tree_table`` **恒**传完整进程表: 进程列表可能来自注入 (测试 / 别的来源),
    但「自己这一棵进程树」是**本机**的事实, 与列表来源无关 —— 少传它就会退回
    「只认列表里的 pid」, 于是经由 cmd.exe 起的子进程又会被当成「别的会话」
    (2026-10-22 那个假阳性)。
    """
    table = process_table()
    if process_source is None:
        return find_device_using_processes(
            keywords, exclude_pids=own_pids, own_paths=own_paths
        )
    return select_device_using_processes(
        list(process_source()),
        keywords=keywords,
        exclude_pids=own_pids,
        own_paths=own_paths,
        tree_table=table,
    )


def wait_until_idle(
    seconds: float,
    *,
    lock: DeviceLock | None = None,
    probe_cmd: Sequence[str] | None = None,
    probe: Callable[[], ProbeResult] | None = None,
    poll_s: float = PROBE_POLL_S,
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
    keywords: Sequence[str] = OCCUPIER_KEYWORDS,
    own_pids: Sequence[int] = (),
    own_paths: Sequence[str | Path] = (),
    process_source: Callable[[], Sequence[ProcessInfo]] | None = None,
    report_interval_s: float = WAIT_REPORT_INTERVAL_S,
) -> WaitVerdict:
    """开跑前等设备空出来 —— **先看锁, 再看进程, 最后才探针**。

    🔴 **顺序很重要** (2026-10-22 09:03-09:48 与 21:18-21:27 实测)

    1. **锁**是权威: 锁被**别人**持有 ⇒ 等 (并报出 pid + 命令行); 锁被**自己
       这棵进程树**持有 ⇒ **不是冲突**, 立刻放行 (旧判据在这里把「自己启动的
       子进程」当成「别的会话」⇒ 永远抢不到设备)。
    2. **没人在用互斥锁**的老脚本靠**进程扫描**兜底: 有别的会话的硬件进程 ⇒
       **一次都不探针** (探针本身要开一次 SLM, 是干扰源), 直接等下一轮。
    3. 才**探针**: 探针成功 ⇒ **再扫一次**进程 (探针开锁的那一瞬间可能刚好有
       别的会话开始) —— 仍没人 ⇒ 放行。
    4. 探针报**被占**: 若这是**真探针** (直接问过 SDK) 且**锁空**、也**没有别的
       硬件进程** ⇒ 判为 **SDK 陈旧状态** ⇒ **立刻停**并给出「给 SLM 控制器
       物理断电重置」的提示 (排队等不来); 否则继续等。

    Args:
        seconds: 最长等多久 (s)。
        lock: 设备锁; ``None`` = 用默认路径上那把。
        probe_cmd: 探针命令; ``None`` = 用内置探针 (真问过 SDK)。
        probe: 直接给一个探针可调用对象 (测试用; 给了就**不**跑子进程)。
        poll_s: 重试间隔 (s)。
        slm_number: SLM 编号。
        cam_id: 相机编号。
        cam_type: 相机类型。
        exposure_ms: 曝光 (ms)。
        keywords: 占用者进程的命令行关键字。
        own_pids: 额外算作「我们自己的」的 pid (同一条线上的另一个脚本)。
        own_paths: 我们自己的脚本路径 (命令行里出现它们的不算占用者)。
        process_source: 进程列表的来源 (可注入 ⇒ 测试不用真查 CIM);
            ``None`` = 用 :func:`query_windows_python_processes`。
        report_interval_s: 汇报间隔 (s) —— 状态变了就立刻报, 否则最多每分钟一行。

    Returns:
        :class:`WaitVerdict`; ``verdict.ok`` 为真时设备可用。
    """
    deadline = time.monotonic() + max(0.0, float(seconds))
    started = time.monotonic()
    paths = list(own_paths) or [Path(__file__).resolve()]
    the_lock = lock if lock is not None else DeviceLock.default()
    attempt = 0
    last_report_at = float("-inf")
    last_state = ""
    last_reason = ""
    detail_lines: list[str] = []

    def _run_probe() -> ProbeResult:
        if probe is not None:
            return probe()
        return run_device_probe(
            probe_cmd,
            slm_number=slm_number,
            cam_id=cam_id,
            cam_type=cam_type,
            exposure_ms=exposure_ms,
        )

    while True:
        attempt += 1
        waited = time.monotonic() - started

        # -- 1) 锁 (权威) --------------------------------------------------
        holder = the_lock.holder()
        if holder is not None and the_lock.is_held_by_me(holder):
            # 🔴 自己这条进程树持有 ⇒ **不是冲突** (假阳性的正身)。
            print(
                f"[等待] 设备锁由我方进程树持有 ({holder.describe()}) ⇒ 不是冲突, 放行",
                flush=True,
            )
            if not the_lock.is_locked():
                the_lock.acquire(timeout=0.0)
            return WaitVerdict(
                ok=True,
                status="idle",
                exit_code=0,
                waited_s=waited,
                reason="",
                detail="设备锁由我方进程树持有",
                holder=holder,
                holder_is_me=True,
                attempts=attempt,
                lock=the_lock,
            )
        if holder is not None:
            state = "locked_by_other"
            last_reason = f"设备锁被别的进程持有 ({holder.describe()})"
            detail_lines = [f"  {holder.describe()}"]
        else:
            # -- 2) 进程扫描 (没参与互斥的老脚本) --------------------------
            others = _other_hardware_processes(
                process_source,
                keywords=keywords,
                own_pids=own_pids,
                own_paths=paths,
            )
            if others:
                # 🔴 有别的会话在用 ⇒ **不探针** (探针要开一次 SLM, 会打断对方)。
                state = "others_using"
                last_reason = "别的会话在用设备"
                detail_lines = format_occupier_lines(others)
            else:
                # -- 3) 探针 -------------------------------------------------
                result = _run_probe()
                if result.status == PROBE_IDLE:
                    # 探针开锁的那一瞬间可能刚好有别的会话开始 ⇒ 再扫一次。
                    others_after = _other_hardware_processes(
                        process_source,
                        keywords=keywords,
                        own_pids=own_pids,
                        own_paths=paths,
                    )
                    holder_after = the_lock.holder()
                    if others_after:
                        state = "idle_but_others"
                        last_reason = "设备可用, 但还有别的硬件进程 (抢早了)"
                        detail_lines = format_occupier_lines(others_after)
                    elif holder_after is not None and not the_lock.is_held_by_me(
                        holder_after
                    ):
                        state = "locked_by_other"
                        last_reason = (
                            f"设备锁被别的进程持有 ({holder_after.describe()})"
                        )
                        detail_lines = [f"  {holder_after.describe()}"]
                    else:
                        # 放行之前把锁**拿在手里** (调用方负责 release) ——
                        # 否则两个会话可能同时通过检查。
                        if not the_lock.acquire(timeout=0.0):
                            state = "locked_by_other"
                            last_reason = "设备锁刚被别人抢走"
                            detail_lines = []
                        else:
                            print(
                                "[等待] 设备可用, 且无其它硬件进程 ⇒ 开始采集",
                                flush=True,
                            )
                            return WaitVerdict(
                                ok=True,
                                status="idle",
                                exit_code=0,
                                waited_s=time.monotonic() - started,
                                reason="",
                                detail="设备可用",
                                attempts=attempt,
                                lock=the_lock,
                            )
                elif result.status == PROBE_BROKEN:
                    print(
                        f"[等待] 探针自己坏了, 无法判断设备是否被占: {result.message}",
                        file=sys.stderr,
                    )
                    return WaitVerdict(
                        ok=False,
                        status="broken",
                        exit_code=2,
                        waited_s=waited,
                        reason=result.message,
                        detail=result.message,
                        attempts=attempt,
                    )
                else:
                    # 设备不可用: 分成「能等」与「等不来」两类。
                    kind = result.kind or KIND_SLM_BUSY
                    info = DEVICE_UNAVAILABLE_INFO.get(kind)
                    verdict = DeviceUnavailable(
                        kind=kind,
                        code=result.code,
                        detail=result.message or (info[0] if info else "设备不可用"),
                        remedy=info[1] if info else "",
                    )
                    if not verdict.waitable:
                        print(
                            f"[等待] 设备不可用且等不来: {verdict.headline()} —— "
                            f"{verdict.advice()}",
                            file=sys.stderr,
                        )
                        return WaitVerdict(
                            ok=False,
                            status="unavailable",
                            exit_code=1,
                            waited_s=waited,
                            reason=verdict.headline(),
                            detail=verdict.detail,
                            remedy=verdict.remedy,
                            kind=verdict.kind,
                            code=verdict.code,
                            attempts=attempt,
                        )
                    # 可等的两类 (别人在用)。若这是**真探针**说的, 且**锁空**、
                    # 且**没有别的硬件进程** ⇒ SDK 陈旧状态 (排队等不来)。
                    if (
                        result.source == PROBE_SOURCE_SDK
                        and the_lock.holder() is None
                        and not others
                    ):
                        stale = escalate_stale_state(verdict)
                        print(
                            f"[等待] {stale.headline()} —— {stale.advice()}",
                            file=sys.stderr,
                        )
                        return WaitVerdict(
                            ok=False,
                            status="stale",
                            exit_code=1,
                            waited_s=waited,
                            reason=stale.headline(),
                            detail=stale.detail,
                            remedy=stale.remedy,
                            kind=stale.kind,
                            code=stale.code,
                            attempts=attempt,
                        )
                    state = "busy_unidentified"
                    last_reason = f"设备被占但没认出占用者 ({verdict.headline()})"
                    detail_lines = []

        remaining = deadline - time.monotonic()
        waited = time.monotonic() - started
        if remaining <= 0:
            hits = _other_hardware_processes(
                process_source,
                keywords=keywords,
                own_pids=own_pids,
                own_paths=paths,
            )
            print(
                f"[等待] 等了 {float(seconds):.0f}s 仍拿不到设备: {last_reason}",
                file=sys.stderr,
            )
            holder_now = the_lock.holder()
            if holder_now is not None:
                print("设备锁的持有者:", file=sys.stderr)
                print(f"  {holder_now.describe()}", file=sys.stderr)
            if hits:
                print("占用它的进程 (命令行):", file=sys.stderr)
                for line in format_occupier_lines(hits):
                    print(line, file=sys.stderr)
            elif holder_now is None:
                print(
                    "  (没找到可疑的占用进程 —— 用任务管理器看看是不是有残留的 "
                    "python 进程还握着设备)",
                    file=sys.stderr,
                )
            return WaitVerdict(
                ok=False,
                status="timeout",
                exit_code=1,
                waited_s=waited,
                reason=last_reason,
                detail=last_reason,
                kind=KIND_SLM_BUSY if state == "busy_unidentified" else "",
                holder=holder_now,
                attempts=attempt,
            )
        sleep_s = min(float(poll_s), remaining)
        now = time.monotonic()
        if state != last_state or (now - last_report_at) >= float(report_interval_s):
            print(
                f"[等待] {last_reason}; 已等 {waited:.0f}s, {sleep_s:.0f}s 后再看"
                f" (第 {attempt} 轮)",
                flush=True,
            )
            for line in detail_lines:
                print(line, flush=True)
            last_report_at = now
            last_state = state
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# 探针的 CLI 入口 (内置探针子进程跑的就是这里)
# ---------------------------------------------------------------------------
def _probe_main(argv: list[str]) -> int:
    """``--probe-devices`` 模式: 探一次设备, 打一行协议行。

    这一模式**只**探可用性然后退出: 真正的等待逻辑在父进程里, 子进程不持有
    任何句柄 (见 :func:`probe_devices` 的第三条硬约束)。

    Returns:
        0 = 设备可用; 1 = 不可用 / 探针自己坏了。
    """
    import argparse

    parser = argparse.ArgumentParser(description="设备可用性探针")
    parser.add_argument("--probe-devices", action="store_true")
    parser.add_argument("--probe-slm-number", type=int, default=1)
    parser.add_argument("--probe-cam-id", type=int, default=0)
    parser.add_argument("--probe-cam-type", default="daheng")
    parser.add_argument("--probe-exposure-ms", type=float, default=0.2)
    args = parser.parse_args(argv)
    if not args.probe_devices:
        parser.error("本模块的 CLI 只有 --probe-devices 一个模式")
    result = probe_devices(
        slm_number=args.probe_slm_number,
        cam_id=args.probe_cam_id,
        cam_type=args.probe_cam_type,
        exposure_ms=args.probe_exposure_ms,
    )
    if result.message:
        print(f"{PROBE_NOTE_PREFIX} {result.message}", flush=True)
    print(format_probe_line(result), flush=True)
    return 0 if result.status == PROBE_IDLE else 1


if __name__ == "__main__":
    raise SystemExit(_probe_main(sys.argv[1:]))
