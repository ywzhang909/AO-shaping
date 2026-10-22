#!/usr/bin/env python
"""一次只跑一组变量的硬件探测: 按计划 (plan) 分组驱动。

为什么
----

两个硬件探针脚本各扫一组变量:

* ``scripts/slm/hw_gs_defocus_probe.py`` —— 自变量 **Δz** (离焦间距, mm);
* ``scripts/slm/run_gs_refine_zernike_sweep.py`` —— 自变量 **n_max** (泽尼克阶数)。

把它们放进**同一次运行**有两个坏处: 一组失败整轮作废; 产物说不清是哪一组贡献的。
所以每次调用只探一组 (``--group`` / ``--group-meta``), 而**编排**交给本脚本:
它读一份 JSON 计划, 把每一组当作**独立子进程**按顺序跑完。

🔴 **为什么是独立子进程**: 一组卡死 (Santec SDK 的原生等待不受 Python 侧超时保护)
或崩溃时, 不能把后面几组一起带走; 而且每一组都要有自己的设备生命周期 (open →
测量 → close), 同一进程里连着跑两组会共用残留的驱动状态。

计划文件 (JSON)::

    {
      "root": "data/debug/hw_groups_20261020",
      "groups": [
        {"group": "dz_noise", "variable": "dz_mm", "meta": {"exposure_ms": "0.2"},
         "script": "scripts/slm/hw_gs_defocus_probe.py",
         "args": ["--execute", "--defocus-focal-mm", "1597.4",
                  "--exposure-ms", "0.2", "--dz-mm", "0"]},
        {"group": "dz_scan", "variable": "dz_mm", "script": "scripts/slm/hw_gs_defocus_probe.py",
         "args": ["--execute", "--dz-mm", "0", "5", "10"]}
      ]
    }

* ``root`` —— 这一轮的根目录; 每一组的产物目录是 ``<root>/<组名>/``, 驱动自己的
  索引写在 ``<root>/groups_manifest.json``。
* ``script`` —— 要跑的命令。**以 ``.py`` 结尾时用当前解释器 (``sys.executable``)
  启动**, 计划里写的是仓库相对路径, 用哪个解释器该由跑计划的环境决定; 其余情况
  (``script`` 本身就是可执行文件) 原样使用。
* ``args`` —— 追加给脚本的参数。

用法::

    # 只看计划, 什么都不执行
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json --dry-run

    # 真跑
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json --timeout-s 3600

    # 把已经跑完的目录登记进来 (不重跑)
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json \\
        --adopt-existing dz_scan report/shape_trials/hw_gs_defocus_probe_20261020_015426/dz_scan

单组的时长预算
--------------

``--timeout-s`` 是**硬超时** (超时 = 杀掉子进程, ``status="timeout"``), 而
``--max-group-seconds`` 是**软预算**: 一组跑完后再比它的耗时, 超过预算就记
``status="over_budget"`` 并在汇总里点名。

🔴 **为什么不直接按软预算杀进程**: 杀掉一个正在测量的组会把那一组的测量写坏
(半张图、半个 epoch), 拿到的数据既不能算也不能复查。所以软预算只**判定**不**中断**
—— 硬约束靠「每一组都必须在 10 分钟内跑完」的编排纪律 + 跑完后的点名来保证。

默认两个预算都是 600 s: 「一次扫描不超过 10 分钟」这条约束由驱动器自己盯着,
而不是靠人掐表。

设备不可用 ⇒ **中止整轮** (2026-10-21 18:23 实测)
-------------------------------------------

SLM 的 USB 当时是**断开**的, 于是每一组都在 2-5 s 内死于

    SantecError: 无法打开SLM #1 (错误码: -10002, 未找到USB设备，请检查设备电源和连接)

而驱动把每一组都当成**普通失败**记进索引, 继续跑下一组 ⇒ **27 组在 100 s 内全灭**,
一轮机时全丢, 而且索引里全是「failed」, 看不出根因是**设备根本不可用**。

所以现在有两条独立的判别:

* 「**这一组坏了**」⇒ 记 ``failed``, 继续跑下一组 (原行为, 不变);
* 「**设备不可用**」⇒ 记 ``device_unavailable``, **中止整轮**, 剩下的组记
  ``skipped``, 退出码非 0, 并打印一句说清根因的中文。

设备不可用又分四类, 因为**处置不同**:

======================  ==========================================  ==============
归类                   特征                                        能不能等
======================  ==========================================  ==============
``slm_absent``         ``-10002`` 未找到USB设备 (没插上/没通电)     不能 (要人插拔)
``slm_busy``           ``-10003`` USB设备已打开 (被别的进程占着)   能 (等一会儿)
``camera_busy``        ``camera not initialized`` (相机被占)        能 (等一会儿)
``camera_absent``      相机不在 (``Camera ID not found``)          不能 (要人处理)
======================  ==========================================  ==============

判别逻辑在**模块级纯函数** :func:`classify_device_unavailable` 里 (不散在主流程中),
单元测试直接测它。

``--wait-until-idle SECONDS``: 开跑前等设备空出来
--------------------------------------------------

被别的进程占着 (-10003 / 相机) 是**可以自动等**的那一类。所以加了这个开关:
开跑前**先探一次设备** (真开一次 SLM + 相机再关), 被占着就**每 15 s 重探**,
直到设备空出来再开跑; 超过 SECONDS 仍拿不到 ⇒ 报错退出 (退出码非 0), 并打印
**占用它的进程命令行** (用 CIM 查 python 进程, 挑出命令行里含设备关键字的那些),
让人知道**是谁**在占。

探针有两条硬约束:

* **恒用 memory 模式**开 SLM —— DVI 模式的 ``open()`` 可能挂死, 挂死后必须物理
  断电才恢复;
* **自己带 PYTHONPATH** (仓库根 + ``src`` + ``libs`` + ``libs/OOPAO``) ——
  ``gxipy`` 只随仓库分发在 ``libs/`` 下, 不在 site-packages 里。漏了它, 相机会在
  半途抛 ``NameError: name 'gx' is not defined``, 于是「相机被占」与「相机没装好」
  分不开, 而等设备的逻辑会永远等下去 (2026-10-21 实测空转 55 分钟)。所以探针把
  「自己没装好」单列成一类, **立刻**报错退出, 不等。

``--resume``: 设备恢复后续跑
----------------------------

上一轮已经产出的组**不该重跑**。``--resume`` 跳过**已经有产物**的组 (判据与仓库
既有的 ``check_adopt_dir`` 一致: 目录里有 ``measurements.json`` / ``summary.json``),
在索引里登记为 **``adopted``** 而不是「failed」—— 复用 ``--adopt-existing`` 的登记
能力, 不另造一套。

⚠ Stage A 的块之间有**链** (第 N 块读第 N-1 块的 ``stage_a.npz``)。续跑时若中间
某块缺产物, 后面依赖它的块**不能**直接跑 (会读到不存在的 npz) ⇒ 依赖不满足的组
一并记 ``skipped``, 并在输出里说清「因缺第 N 块而跳过」。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common.group_manifest import (  # noqa: E402
    group_output_dir,
    load_manifest,
    utc_now_iso,
)

#: 驱动自己的索引文件名 (与两个探针脚本的 ``manifest.json`` 分开, 免得互相覆盖)。
GROUPS_MANIFEST_NAME = "groups_manifest.json"

#: 一组「已经跑完」的判据: 目录里至少有下面之一 (两个探针各自的落盘标志)。
ARTEFACT_MARKERS = ("measurements.json", "summary.json")

#: 组命令的输出留多少尾巴进索引 (前部被截掉 —— 崩在最后一行才是有用的信息)。
OUTPUT_TAIL_CHARS = 4000


# ---------------------------------------------------------------------------
# 设备不可用的判别 (模块级纯函数)
# ---------------------------------------------------------------------------
#: 「设备不可用」的四种归类。**处置不同, 所以必须分开**:
#: 断电类要人重新插拔 (等是等不来的); 被占类别人在用, 等一会儿就空出来。
KIND_SLM_ABSENT = "slm_absent"
KIND_SLM_BUSY = "slm_busy"
KIND_CAMERA_BUSY = "camera_busy"
KIND_CAMERA_ABSENT = "camera_absent"

#: 可以自动等的那几类 (别人在用 ⇒ 等一会儿就好)。
WAITABLE_KINDS = frozenset({KIND_SLM_BUSY, KIND_CAMERA_BUSY})

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
}

#: 设备不可用的特征串。**数字错误码与中文短语都认**: 驱动异常文本带码
#: (「错误码: -10002」), 而 GUI 的中文提示只带短语。
_SLM_ABSENT_RE = re.compile(r"未找到USB设备|错误码[:：]?\s*-10002(?!\d)")
_SLM_BUSY_RE = re.compile(r"USB设备已打开|错误码[:：]?\s*-10003(?!\d)")
_CAMERA_BUSY_RE = re.compile(r"camera not initialized|REPEAT_OPENED|device has been open")
_CAMERA_ABSENT_RE = re.compile(r"camera not found|Camera ID \d+ not found")

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

    @property
    def waitable(self) -> bool:
        """这一类能不能靠「等」解决 (别人在用 vs 设备没插上)。"""
        return self.kind in WAITABLE_KINDS

    def headline(self) -> str:
        """根因的一句话 (带错误码)。"""
        code = f" (错误码 {self.code})" if self.code is not None else ""
        return f"{self.detail}{code}"

    def advice(self) -> str:
        """处置建议。"""
        return self.remedy


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
    ):
        positions = [m.start() for m in pattern.finditer(text)]
        if not positions:
            continue
        detail, remedy = DEVICE_UNAVAILABLE_INFO[kind]
        candidate = DeviceUnavailable(
            kind=kind, code=code, detail=detail, remedy=remedy
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


# ---------------------------------------------------------------------------
# 设备可用性探针 (独立子进程)
# ---------------------------------------------------------------------------
#: 探针子进程输出的协议行前缀 (后面跟 status 与 key=value 字段)。
PROBE_LINE_PREFIX = "DEVICE_PROBE:"
#: 探针子进程输出的说明行前缀 (人看的, 可以含空格)。
PROBE_NOTE_PREFIX = "DEVICE_PROBE_NOTE:"

#: 探针的三种结论。
PROBE_IDLE = "idle"                 # 设备可用
PROBE_UNAVAILABLE = "unavailable"   # 设备不可用 (kind 说的是哪一类)
PROBE_BROKEN = "broken"             # 探针自己没装好 —— **不是**设备的问题

#: 探针每隔多久重试一次 (s)。
PROBE_POLL_S = 15.0
#: 探针子进程的超时 (s) —— 探针不该把驱动挂住。
PROBE_TIMEOUT_S = 120.0
#: 探针命令的环境变量名 (测试与非常规环境用它换成别的探针)。
PROBE_CMD_ENV = "AO_HW_DEVICE_PROBE_CMD"

#: 占用者进程的命令行里出现这些关键字就算「可能在用设备」。
#: 收得比较紧 (``camera`` / ``ccd`` 太宽, 满屏都是), 但要能盖住两种真实占用者:
#: ``scripts/slm/*.py`` 那一类探针, 以及 ``python src/ao_shaping/main.py ... --debug``。
OCCUPIER_KEYWORDS: tuple[str, ...] = (
    "slm",
    "daheng",
    "santec",
    "gxipy",
    "ao_shaping",
)


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


def probe_devices(
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
) -> ProbeResult:
    """真开一次 SLM + 相机再关 —— 只探可用性。

    两条硬约束:

    1. **恒用 memory 模式**开 SLM: DVI 模式的 ``open()`` 可能挂死, 挂死后
       必须物理断电才恢复 (``drivers/slm/AGENTS.md``)。
    2. **自己带 PYTHONPATH**: 见 :func:`_ensure_repo_paths`。

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
    except Exception as exc:  # noqa: BLE001 - 探针要把任何异常都翻译成结论
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
    except Exception as exc:  # noqa: BLE001 - 同上
        return _probe_failure(f"{type(exc).__name__}: {exc}")
    return ProbeResult(status=PROBE_IDLE)


def _probe_failure(text: str) -> ProbeResult:
    """把探针里的异常翻译成结论。"""
    setup = classify_probe_setup_error(text)
    if setup is not None:
        return ProbeResult(status=PROBE_BROKEN, message=setup)
    verdict = classify_device_unavailable(text)
    if verdict is not None:
        return ProbeResult(
            status=PROBE_UNAVAILABLE,
            kind=verdict.kind,
            code=verdict.code,
            message=verdict.detail,
        )
    return ProbeResult(status=PROBE_BROKEN, message=f"探针异常: {text}")


def default_probe_cmd(
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
) -> list[str]:
    """内置探针的命令 (本脚本自己, 走 ``--probe-devices`` 模式)。"""
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
        probe_cmd: 探针命令; ``None`` = 用内置探针。
        timeout_s: 探针子进程的超时 (s)。
        slm_number: SLM 编号 (内置探针用)。
        cam_id: 相机编号 (内置探针用)。
        cam_type: 相机类型 (内置探针用)。
        exposure_ms: 曝光 (ms, 内置探针用)。

    Returns:
        探针结论。
    """
    cmd = (
        list(probe_cmd)
        if probe_cmd
        else default_probe_cmd(slm_number, cam_id, cam_type, exposure_ms)
    )
    try:
        completed = subprocess.run(  # noqa: S603 - 命令来自调用者/环境, 故意如此
            list(cmd),
            cwd=str(ROOT),
            env=build_env(),
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
    return parse_probe_output(completed.stdout, completed.stderr)


def query_windows_python_processes() -> list[tuple[int, str]]:
    """用 CIM 列出 python 进程 (pid, 命令行)。

    非 Windows 或查询失败时返回空列表 —— 查不到占用者不该让整轮失败。

    Returns:
        ``[(pid, 命令行), ...]``。
    """
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' } | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    try:
        completed = subprocess.run(  # noqa: S603 - 固定命令, 不带用户输入
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
    processes: list[tuple[int, str]] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, cmdline = line.partition("\t")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        processes.append((pid, cmdline))
    return processes


def find_device_using_processes(
    keywords: Sequence[str] = OCCUPIER_KEYWORDS,
    *,
    exclude_pids: Sequence[int] = (),
) -> list[tuple[int, str]]:
    """挑出命令行里含设备关键字的进程 (排除自己)。

    Args:
        keywords: 命令行里要匹配的关键字。
        exclude_pids: 额外要排除的 pid。

    Returns:
        ``[(pid, 命令行), ...]``。
    """
    skip = {int(pid) for pid in exclude_pids}
    skip.update({os.getpid(), os.getppid()})
    hits: list[tuple[int, str]] = []
    for pid, cmdline in query_windows_python_processes():
        if pid in skip:
            continue
        lowered = cmdline.lower()
        if any(str(keyword).lower() in lowered for keyword in keywords):
            hits.append((pid, cmdline))
    return hits


def format_occupier_lines(hits: Sequence[tuple[int, str]]) -> list[str]:
    """把占用者列表排成给人看的行。"""
    return [f"  pid {pid}: {cmdline}" for pid, cmdline in hits]


def wait_until_idle(
    seconds: float,
    *,
    probe_cmd: Sequence[str] | None = None,
    poll_s: float = PROBE_POLL_S,
    slm_number: int = 1,
    cam_id: int = 0,
    cam_type: str = "daheng",
    exposure_ms: float = 0.2,
    keywords: Sequence[str] = OCCUPIER_KEYWORDS,
) -> tuple[int, str]:
    """开跑前等设备空出来。

    只有**可以等**的那两类 (被别的进程占着) 才会等; 设备没插上 / 探针自己坏了
    立刻报错退出 —— 等是等不来的。

    Args:
        seconds: 最长等多久 (s)。
        probe_cmd: 探针命令; ``None`` = 用内置探针。
        poll_s: 重试间隔 (s)。
        slm_number: SLM 编号。
        cam_id: 相机编号。
        cam_type: 相机类型。
        exposure_ms: 曝光 (ms)。
        keywords: 占用者进程的关键字。

    Returns:
        ``(退出码, 中文说明)``; 退出码 0 = 设备可用, 可以开跑。
    """
    deadline = time.monotonic() + max(0.0, float(seconds))
    attempt = 0
    while True:
        attempt += 1
        result = run_device_probe(
            probe_cmd,
            slm_number=slm_number,
            cam_id=cam_id,
            cam_type=cam_type,
            exposure_ms=exposure_ms,
        )
        if result.status == PROBE_IDLE:
            print(f"[等待] 设备可用 (第 {attempt} 次探测)", flush=True)
            return 0, ""
        if result.status == PROBE_BROKEN:
            print(
                f"[等待] 探针自己坏了, 无法判断设备是否被占: {result.message}",
                file=sys.stderr,
            )
            return 2, result.message
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
            return 1, verdict.headline()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            hits = find_device_using_processes(keywords)
            print(
                f"[等待] 等了 {float(seconds):.0f}s 仍拿不到设备: {verdict.headline()}",
                file=sys.stderr,
            )
            if hits:
                print("占用它的进程 (命令行):", file=sys.stderr)
                for line in format_occupier_lines(hits):
                    print(line, file=sys.stderr)
            else:
                print(
                    "  (没找到可疑的占用进程 —— 用任务管理器看看是不是有残留的 "
                    "python 进程还握着设备)",
                    file=sys.stderr,
                )
            return 1, verdict.headline()
        sleep_s = min(float(poll_s), remaining)
        print(
            f"[等待] 设备被占 ({verdict.kind}): {verdict.headline()}; "
            f"{sleep_s:.0f}s 后重试 (还剩 {remaining:.0f}s)",
            flush=True,
        )
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------
@dataclass
class GroupSpec:
    """计划里的一组 (要跑什么, 以及它是哪一组)。"""

    group: str
    variable: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    script: str = ""
    args: list[str] = field(default_factory=list)
    adopted_dir: str | None = None
    """非空 = 这一组已经跑完, 直接登记这个目录, 不重跑。"""
    env: dict[str, str] = field(default_factory=dict)
    """这一组子进程的环境变量覆盖 (例如**显式的曝光上限**)。

    为什么要有它: 曝光上限 ``AO_DAHENG_MAX_EXPOSURE_MS`` 在两类组里必须不同 ——
    焦面标定组要能一路加到 20 ms (它自己在扫曝光阶梯), 而整形测量组必须压在
    0.4 ms 以下, 否则画面会饱和, 半个 epoch 的数据就废了。内置的加倍阶梯估计在
    **暗**实验系统上会给出假上限 (实测假 2.07 ms), 所以这个值必须由计划**显式**
    写死, 而不是靠现场猜。"""


def load_plan(path: Path) -> dict[str, Any]:
    """读计划文件。

    Args:
        path: 计划 JSON 的路径。

    Returns:
        计划字典 (含 ``root`` 与 ``groups``)。

    Raises:
        ValueError: 文件不是合法 JSON, 或缺 ``groups`` 列表。
        OSError: 文件读不出来。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象")
    if not isinstance(data.get("groups"), list):
        raise ValueError(f"{path} 缺 'groups' 列表")
    return data


def build_argv(script: str, args: list[str]) -> list[str]:
    """组命令 = ``[脚本] + 参数``。

    ``script`` 以 ``.py`` 结尾时前面加当前解释器 (``sys.executable``): 计划里写的
    是仓库相对路径, 用哪个解释器必须由**跑计划的那个环境**决定, 而不是由写计划
    的人决定 (换机器 / 换 venv 时计划不用改)。

    Args:
        script: 命令本体。
        args: 追加的参数。

    Returns:
        可直接交给 :func:`subprocess.run` 的参数列表。
    """
    tail = [str(a) for a in args]
    if str(script) == "-m":
        # 模块调用 (``python -m ao_shaping.tools.slm.panel.focus_probe``):
        # 焦面标定是**模块**不是脚本, 计划里就得能这样写。
        return [sys.executable, "-m", *tail]
    if str(script).lower().endswith(".py"):
        return [sys.executable, str(script), *tail]
    return [str(script), *tail]


def build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """子进程的环境变量 = 当前环境 + 仓库的四个搜索路径 + 计划给的覆盖。

    探针脚本要 import ``ao_shaping`` / ``scripts.*``, 而计划里写的是仓库相对
    路径, 所以 PYTHONPATH 必须带上仓库根与 ``src``。**只在缺失时追加**, 不覆盖
    调用者已有的设置。
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
    # 计划给的覆盖放在最后: 显式写死的值必须赢过调用者的环境。
    for key, value in (extra or {}).items():
        env[str(key)] = str(value)
    return env


def resolve_base_dir(spec: GroupSpec, root: Path) -> Path:
    """这一组的**根目录** (组目录建在它下面)。

    计划里显式给了 ``--out-root`` (离焦探针) 或 ``--out`` (泽尼克扫描) 就用它,
    否则用驱动自己的根目录 —— 那时组目录就是 ``<root>/<组名>/``。
    """
    for flag in ("--out-root", "--out"):
        if flag in spec.args:
            idx = spec.args.index(flag)
            if idx + 1 < len(spec.args):
                return Path(str(spec.args[idx + 1]))
    return Path(root)


def resolve_out_dir(spec: GroupSpec, root: Path) -> Path:
    """这一组的产物目录 = ``<根目录>/<组名>/``。"""
    return group_output_dir(resolve_base_dir(spec, root), spec.group)


# ---------------------------------------------------------------------------
# 登记 (adopt)
# ---------------------------------------------------------------------------
def read_group_metadata(directory: Path) -> tuple[str, dict[str, str]]:
    """从已完成的组目录里读回 ``(variable, meta)``。

    两个探针都把自己的组元数据写进组目录 (``run_config.json`` / ``summary.json``),
    所以登记一个计划外的组时, 它的自变量与固定参数不必在计划里重抄一遍。

    Args:
        directory: 组目录。

    Returns:
        ``(variable, meta)``; 读不到时是 ``("", {})``。
    """
    for name in ("run_config.json", "summary.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        raw_meta = data.get("group_meta")
        meta = (
            {str(k): str(v) for k, v in raw_meta.items()}
            if isinstance(raw_meta, dict)
            else {}
        )
        return str(data.get("variable", "") or ""), meta
    return "", {}


def check_adopt_dir(directory: Path) -> str | None:
    """检查一个待登记的目录: 存在, 且有落盘产物。

    Args:
        directory: 待登记目录。

    Returns:
        ``None`` 表示可以登记; 否则是给操作者看的中文原因。
    """
    if not directory.exists():
        return f"目录不存在: {directory}"
    if not directory.is_dir():
        return f"不是目录: {directory}"
    if not any((directory / marker).is_file() for marker in ARTEFACT_MARKERS):
        return (
            f"目录里没有落盘产物 (需要 {' 或 '.join(ARTEFACT_MARKERS)} 之一): {directory}"
        )
    return None


# ---------------------------------------------------------------------------
# 续跑 (--resume) 与组间的依赖
# ---------------------------------------------------------------------------
#: 组之间「上一块的产物」这类硬输入的开关 (值 = 上游产物文件路径)。
DEPENDENCY_FLAGS: tuple[str, ...] = ("--stage-a-in",)


def group_artefact_dirs(spec: GroupSpec, root: Path) -> list[Path]:
    """这一组**可能的**产物目录 (按仓库既有的口径)。

    - 探针组 (``--out-root`` / ``--out``): 产物在 ``<base>/<组名>/``;
    - 焦面标定组 (``--debug-dir``): 证据**平铺**在那个目录里 (不加组名一层),
      但两种布局都认 (与 ``scripts/slm/archive_hw_defocus_round.py`` 一致);
    - 都没有: 落在驱动的根目录下的 ``<组名>/``。

    Args:
        spec: 这一组的计划。
        root: 驱动自己的根目录。

    Returns:
        候选目录列表 (按可能性排序, 不重复)。
    """
    dirs: list[Path] = []
    for flag in ("--out-root", "--out"):
        if flag in spec.args:
            idx = spec.args.index(flag)
            if idx + 1 < len(spec.args):
                dirs.append(
                    group_output_dir(Path(str(spec.args[idx + 1])), spec.group)
                )
            break
    else:
        if "--debug-dir" in spec.args:
            idx = spec.args.index("--debug-dir")
            if idx + 1 < len(spec.args):
                base = Path(str(spec.args[idx + 1]))
                dirs.append(base)
                dirs.append(group_output_dir(base, spec.group))
        if not dirs:
            dirs.append(group_output_dir(root, spec.group))
    unique: list[Path] = []
    for directory in dirs:
        if directory not in unique:
            unique.append(directory)
    return unique


def find_resume_artefact_dir(spec: GroupSpec, root: Path) -> Path | None:
    """续跑: 这一组**已经有的**产物目录 (没有则 ``None``)。

    判据与仓库既有的 :func:`check_adopt_dir` 一致 (目录里有
    ``measurements.json`` / ``summary.json`` 就算跑过了)。

    Args:
        spec: 这一组的计划。
        root: 驱动自己的根目录。

    Returns:
        可登记的产物目录; 没有则 ``None``。
    """
    for directory in group_artefact_dirs(spec, root):
        if check_adopt_dir(directory) is None:
            return directory
    return None


def apply_resume(specs: list[GroupSpec], root: Path) -> list[GroupSpec]:
    """续跑: 已经有产物的组登记为 adopted (不重跑), 其余照常。

    复用 ``--adopt-existing`` 的登记能力 (``adopted_dir``), 不另造一套。

    Args:
        specs: 这一轮要处理的组。
        root: 驱动自己的根目录。

    Returns:
        同一份列表 (``adopted_dir`` 被就地填上)。
    """
    for spec in specs:
        if spec.adopted_dir:
            continue
        directory = find_resume_artefact_dir(spec, root)
        if directory is None:
            print(f"[续跑] {spec.group}: 没有产物 → 这一组要跑", flush=True)
            continue
        spec.adopted_dir = str(directory)
        print(
            f"[续跑] {spec.group}: 已有产物 ({directory}) → 登记为 adopted, 不重跑",
            flush=True,
        )
    return specs


def group_dependency_paths(spec: GroupSpec) -> list[Path]:
    """这一组读的上游产物文件 (空 = 没有依赖)。

    Args:
        spec: 这一组的计划。

    Returns:
        上游产物路径列表 (相对仓库根, 与计划里写的一致)。
    """
    paths: list[Path] = []
    for flag in DEPENDENCY_FLAGS:
        if flag in spec.args:
            idx = spec.args.index(flag)
            if idx + 1 < len(spec.args):
                paths.append(Path(str(spec.args[idx + 1])))
    return paths


def missing_dependencies(spec: GroupSpec) -> list[Path]:
    """还没产出的上游产物。

    相对仓库根解析 —— 探针子进程就是在仓库根里跑的 (``cwd=ROOT``)。

    Args:
        spec: 这一组的计划。

    Returns:
        缺失的上游产物路径 (空 = 依赖都满足)。
    """
    missing: list[Path] = []
    for path in group_dependency_paths(spec):
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            missing.append(path)
    return missing


def describe_missing_dependency(path: Path) -> str:
    """说清缺的是哪一块的产物 (「因缺第 N 块而跳过」)。"""
    producer = path.parent.name or path.name
    return f"因缺上游产物而跳过: {path} (应由组 {producer} 产出)"


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _tail(value: Any, limit: int = OUTPUT_TAIL_CHARS) -> str:
    """把子进程输出截成尾巴 (bytes / str / None 都能吃)。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[-limit:]


def _manifest_n_rows(base: Path, group: str) -> int | None:
    """从组自己的清单里读这一组的行数 (读不到就是 ``None``)。"""
    for entry in load_manifest(base):
        if str(entry.get("group", "")) == group:
            try:
                return int(entry.get("n_rows", 0))
            except (TypeError, ValueError):
                return None
    return None


def _driver_modules() -> frozenset[str]:
    """当前已加载的 ``ao_shaping.drivers.*`` 模块名。"""
    return frozenset(
        m for m in sys.modules if m.startswith("ao_shaping.drivers")
    )


def assert_no_devices(before: frozenset[str]) -> None:
    """dry-run 的硬保证: 计划解析过程中**没有新**导入 ``ao_shaping.drivers``。

    比的是「开始之前」与「现在」的差集, 而不是「整个进程里一个驱动模块都没有」:
    后者会把**同进程里别的模块**引进来的驱动包算到 dry-run 头上 —— 例如测试
    进程里先 import 了 ``scripts.slm.run_gs_refine_zernike_sweep`` (它要
    ``ao_shaping.drivers.slm.factory``), 那是误判。判据是 dry-run 自己**没有
    牵进**驱动包。
    """
    leaked = sorted(m for m in _driver_modules() - before)
    if leaked:
        raise SystemExit(
            "dry-run 违约: 计划解析过程中导入了驱动包:\n  " + "\n  ".join(leaked)
        )


def run_group(
    spec: GroupSpec,
    root: Path,
    timeout_s: float | None,
    max_group_seconds: float | None = None,
) -> dict[str, Any]:
    """把一组当作独立子进程跑完, 返回它的索引条目。

    **失败不中断整轮**: 退出码非零记 ``failed``, 超时记 ``timeout``, 两种情况
    都继续跑下一组 —— 一组坏掉不该让后面几组陪葬。

    **软预算不杀进程**: 跑完之后才比耗时, 超过 ``max_group_seconds`` 就记
    ``over_budget``。杀掉一个正在测量的组会把那一组的测量写坏。

    Args:
        spec: 这一组的计划。
        root: 驱动自己的根目录 (组目录的缺省位置)。
        timeout_s: 单组超时 (s); ``None`` = 不限。
        max_group_seconds: 单组的时长预算 (s); ``None`` = 不判。超了就记
            ``over_budget``。

    Returns:
        索引条目 (字段见模块文档)。
    """
    argv = build_argv(spec.script, spec.args)
    base = resolve_base_dir(spec, root)
    out_dir = group_output_dir(base, spec.group)
    started = time.time()
    record: dict[str, Any] = {
        "group": spec.group,
        "variable": spec.variable,
        "meta": dict(spec.meta),
        "script": spec.script,
        "argv": list(argv),
        "start_utc": utc_now_iso(),
        "end_utc": None,
        "duration_s": 0.0,
        "exit_code": None,
        "status": "failed",
        "out_dir": str(out_dir),
        "out_dir_exists": out_dir.is_dir(),
        "n_rows": None,
        # 这一组的环境变量覆盖 (显式的曝光上限就在里面): 测量参数必须跟着
        # 数据一起被记下来, 否则复现时不知道当时拍的是多少曝光上限。
        "env": dict(spec.env),
    }
    print(f"[组] {spec.group} (自变量 {spec.variable or '?'}) -> {' '.join(argv)}",
          flush=True)
    try:
        completed = subprocess.run(  # noqa: S603 - 命令来自计划文件, 本来就要执行
            argv,
            cwd=str(ROOT),
            env=build_env(spec.env),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        record["exit_code"] = int(completed.returncode)
        record["status"] = "ok" if completed.returncode == 0 else "failed"
        record["stdout_tail"] = _tail(completed.stdout)
        record["stderr_tail"] = _tail(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        record["status"] = "timeout"
        record["exit_code"] = None
        record["stdout_tail"] = _tail(exc.stdout)
        record["stderr_tail"] = _tail(exc.stderr)
    except OSError as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["end_utc"] = utc_now_iso()
    record["duration_s"] = round(max(time.time() - started, 0.0), 6)
    record["out_dir_exists"] = out_dir.is_dir()
    rows = _manifest_n_rows(base, spec.group)
    if rows is not None:
        record["n_rows"] = rows
    # 软预算: 跑完之后才判 (不杀进程 —— 杀掉正在测量的组会把测量写坏)。
    if (
        max_group_seconds is not None
        and record["status"] == "ok"
        and float(record["duration_s"]) > float(max_group_seconds)
    ):
        record["status"] = "over_budget"
        record["budget_s"] = float(max_group_seconds)
    print(f"[组] {spec.group}: {record['status']} "
          f"({record['duration_s']:.1f}s, exit={record['exit_code']})", flush=True)
    return record


def adopt_group(spec: GroupSpec, directory: Path) -> dict[str, Any]:
    """把一个已完成的测量目录登记为某一组 (不重跑)。

    Args:
        spec: 这一组的计划 (组名 / 自变量 / 固定参数; 计划里没写的从目录里读)。
        directory: 已完成的组目录。

    Returns:
        索引条目, ``status`` 为 ``"adopted"``。
    """
    variable, meta = read_group_metadata(directory)
    started = time.time()
    record = {
        "group": spec.group,
        "variable": spec.variable or variable,
        "meta": dict(spec.meta) if spec.meta else meta,
        "script": spec.script,
        "argv": [],
        "start_utc": utc_now_iso(),
        "end_utc": utc_now_iso(),
        "duration_s": round(max(time.time() - started, 0.0), 6),
        "exit_code": None,
        "status": "adopted",
        "out_dir": str(directory),
        "out_dir_exists": True,
        "n_rows": _manifest_n_rows(directory.parent, spec.group),
    }
    print(f"[组] {spec.group}: adopted (已有目录 {directory})", flush=True)
    return record


def skipped_group(
    spec: GroupSpec,
    root: Path,
    reason: str,
    *,
    detail: str = "",
) -> dict[str, Any]:
    """一个「**没跑**」的组: 状态 ``skipped``, 带跳过原因。

    两种跳过:

    - ``missing_dependency``: 上游那一块还没产出 (续跑时中间缺了一块);
    - ``round_aborted``: 整轮因为设备不可用被中止。

    Args:
        spec: 这一组的计划。
        root: 驱动自己的根目录。
        reason: 跳过原因 (机器可读)。
        detail: 人看的说明。

    Returns:
        索引条目, ``status`` 为 ``"skipped"``。
    """
    record = {
        "group": spec.group,
        "variable": spec.variable,
        "meta": dict(spec.meta),
        "script": spec.script,
        "argv": [],
        "start_utc": utc_now_iso(),
        "end_utc": utc_now_iso(),
        "duration_s": 0.0,
        "exit_code": None,
        "status": "skipped",
        "skip_reason": reason,
        "out_dir": str(resolve_out_dir(spec, root)),
        "out_dir_exists": resolve_out_dir(spec, root).is_dir(),
        "n_rows": None,
    }
    if detail:
        record["skip_detail"] = detail
    print(f"[组] {spec.group}: skipped ({detail or reason})", flush=True)
    return record


def write_groups_manifest(
    root: Path,
    records: list[dict[str, Any]],
    *,
    plan_path: str,
    created_utc: str,
) -> Path:
    """写 (或覆盖) ``<root>/groups_manifest.json``。

    **每组跑完就写一次**: 整轮被强杀时, 已经跑完的那几组仍然在索引里。

    Args:
        root: 根目录。
        records: 已完成的组条目。
        plan_path: 计划文件路径 (记进索引, 便于复现)。
        created_utc: 本轮索引的创建时刻 (第一次写时定下, 之后不变)。

    Returns:
        索引文件路径。
    """
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "plan": str(plan_path),
        "created_utc": created_utc,
        "updated_utc": utc_now_iso(),
        "groups": records,
        "summary": [
            f"{r['group']}: {r['status']} ({float(r['duration_s']):.1f}s)"
            for r in records
        ],
    }
    path = root / GROUPS_MANIFEST_NAME
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", default=None,
                        help="计划 JSON 的路径 (探针模式不需要)")
    parser.add_argument("--root", default=None,
                        help="覆盖计划里的 root (这一轮的根目录)")
    parser.add_argument("--only", action="append", default=[], metavar="GROUP",
                        help="只跑这一组 (可重复); 计划里没有这一组就报错")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划 (组 / 自变量 / 命令), 什么都不执行")
    parser.add_argument("--print-plan", action="store_true",
                        help="把解析后的计划按 JSON 打到 stdout 然后退出")
    parser.add_argument("--adopt-existing", action="append", nargs=2,
                        default=[], metavar=("GROUP", "DIR"),
                        help="把一个已完成的测量目录登记为这一组 (不重跑); 可重复")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help=(
            "单组硬超时 (s); 超时杀掉子进程并记 status=timeout。"
            "默认 600 (10 分钟)。"
        ),
    )
    parser.add_argument(
        "--max-group-seconds",
        type=float,
        default=600.0,
        help=(
            "单组时长预算 (s); 跑完后超过预算就记 status=over_budget 并在汇总里"
            "点名。**不杀进程** —— 杀掉正在测量的组会把那一组的测量写坏。"
            "默认 600 (10 分钟)。"
        ),
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "开跑前先探一次设备 (真开一次 SLM+相机再关); 被别的进程占着"
            " (-10003 / 相机) 就每 15 s 重探, 直到空出来再开跑。超过 SECONDS "
            "仍拿不到就报错退出, 并打印占用它的进程命令行。0 = 不探 (默认)。"
        ),
    )
    parser.add_argument(
        "--wait-poll-s",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="等设备时的重探间隔 (s); 默认 15。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "续跑: 跳过已经有产物的组 (在索引里登记为 adopted, 不重跑), 只跑缺的。"
            "依赖上一块产物的组若上游缺失, 一并跳过。"
        ),
    )
    parser.add_argument(
        "--device-probe-cmd",
        default=None,
        help=(
            "设备探针的命令 (覆盖内置探针; 路径里有空格时用 JSON 数组)。"
            f"也可用环境变量 {PROBE_CMD_ENV} 指定。"
        ),
    )
    parser.add_argument(
        "--probe-devices",
        action="store_true",
        help="只探一次设备可用性然后退出 (内部用; --wait-until-idle 调它)",
    )
    parser.add_argument("--probe-slm-number", type=int, default=1,
                        help="探针用的 SLM 编号")
    parser.add_argument("--probe-cam-id", type=int, default=0,
                        help="探针用的相机编号")
    parser.add_argument("--probe-cam-type", default="daheng",
                        help="探针用的相机类型")
    parser.add_argument("--probe-exposure-ms", type=float, default=0.2,
                        help="探针开相机时的曝光 (ms)")
    return parser


def plan_device_params(specs: Sequence[GroupSpec]) -> dict[str, Any]:
    """从计划的参数里读出设备参数 (探针要用同一套设备)。

    Args:
        specs: 这一轮要处理的组。

    Returns:
        ``{"slm_number", "cam_id", "cam_type", "exposure_ms"}``。
    """
    params: dict[str, Any] = {
        "slm_number": 1,
        "cam_id": 0,
        "cam_type": "daheng",
        "exposure_ms": 0.2,
    }
    for spec in specs:
        for key, flag in (
            ("slm_number", "--slm-number"),
            ("cam_id", "--cam-id"),
            ("cam_type", "--cam-type"),
            ("exposure_ms", "--exposure-time-ms"),
        ):
            if flag in spec.args:
                idx = spec.args.index(flag)
                if idx + 1 < len(spec.args):
                    raw = str(spec.args[idx + 1])
                    try:
                        if key in ("slm_number", "cam_id"):
                            params[key] = int(raw)
                        elif key == "exposure_ms":
                            params[key] = float(raw)
                        else:
                            params[key] = raw
                    except ValueError:
                        continue
    return params


def resolve_specs(
    plan: dict[str, Any],
    adopt_map: dict[str, Path],
    only: list[str],
) -> list[GroupSpec]:
    """把计划 + 登记请求合成一张「要处理的组」的表。

    Args:
        plan: 计划字典。
        adopt_map: ``{组名: 已完成目录}``。
        only: 只处理这些组 (空 = 全部)。

    Returns:
        组列表 (计划顺序; 计划外的登记组追加在末尾)。

    Raises:
        ValueError: 某一组没有组名, 或 ``--only`` 指名的组不在计划里。
    """
    plan_env = plan.get("env") or {}
    if not isinstance(plan_env, dict):
        raise ValueError("计划的 'env' 必须是对象")
    specs: list[GroupSpec] = []
    for raw in plan.get("groups", []):
        if not isinstance(raw, dict):
            raise ValueError("计划里的每一组都必须是 JSON 对象")
        group = str(raw.get("group", "")).strip()
        if not group:
            raise ValueError("计划里的每一组都必须有非空的 'group'")
        raw_meta = raw.get("meta") or {}
        if not isinstance(raw_meta, dict):
            raise ValueError(f"组 {group} 的 'meta' 必须是对象")
        specs.append(
            GroupSpec(
                group=group,
                variable=str(raw.get("variable", "") or ""),
                meta={str(k): str(v) for k, v in raw_meta.items()},
                script=str(raw.get("script", "") or ""),
                args=[str(a) for a in (raw.get("args") or [])],
                # 计划级的 env 是默认, 组级的 env 覆盖它。
                env={
                    **{str(k): str(v) for k, v in (plan_env or {}).items()},
                    **{str(k): str(v) for k, v in (raw.get("env") or {}).items()},
                },
            )
        )
    known = {s.group for s in specs}
    for group, directory in adopt_map.items():
        if group not in known:
            specs.append(GroupSpec(group=group, adopted_dir=str(directory)))
    for spec in specs:
        if spec.group in adopt_map:
            spec.adopted_dir = str(adopt_map[spec.group])
    if only:
        wanted = set(only)
        missing = sorted(wanted - {s.group for s in specs})
        if missing:
            raise ValueError(f"计划里没有这些组: {missing}")
        specs = [s for s in specs if s.group in wanted]
    return specs


def main(argv: list[str] | None = None) -> int:
    """按计划逐组跑 (或只打印计划)。

    Returns:
        进程退出码: 0 = 全部组成功 (或只是打印计划); 1 = 有组失败/超时, 或
        **设备不可用** (整轮已中止), 或等设备等到超时; 2 = 用法或计划有问题
        (含「探针自己没装好」)。
    """
    args = _build_parser().parse_args(argv)

    # -- 探针模式: 只探一次设备可用性然后退出 (--wait-until-idle 调它) -------
    if args.probe_devices:
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

    if not args.plan:
        print("缺少 --plan (计划 JSON 的路径)", file=sys.stderr)
        return 2

    # dry-run 的「没有牵进驱动包」检查: 比的是**开始之前**与**现在**的差集。
    drivers_before = _driver_modules()
    try:
        plan = load_plan(Path(args.plan))
    except (OSError, ValueError) as exc:
        print(f"计划读取失败: {exc}", file=sys.stderr)
        return 2

    root_text = str(args.root) if args.root else str(plan.get("root", "") or "")
    if not root_text:
        print("计划里没有 root, 请用 --root 指定根目录", file=sys.stderr)
        return 2
    root = Path(root_text)

    adopt_map: dict[str, Path] = {}
    for group, directory in args.adopt_existing:
        path = Path(str(directory))
        problem = check_adopt_dir(path)
        if problem is not None:
            print(f"登记失败 [{group}]: {problem}", file=sys.stderr)
            return 2
        adopt_map[str(group)] = path

    try:
        specs = resolve_specs(plan, adopt_map, list(args.only))
    except ValueError as exc:
        print(f"计划有问题: {exc}", file=sys.stderr)
        return 2

    if args.print_plan:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "plan": str(args.plan),
                    "timeout_s": args.timeout_s,
                    "max_group_seconds": args.max_group_seconds,
                    "groups": [
                        {
                            "group": s.group,
                            "variable": s.variable,
                            "meta": s.meta,
                            "script": s.script,
                            "argv": build_argv(s.script, s.args),
                            "adopted_dir": s.adopted_dir,
                        }
                        for s in specs
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        assert_no_devices(drivers_before)
        return 0

    if args.dry_run:
        print(f"根目录: {root}")
        for spec in specs:
            argv = build_argv(spec.script, spec.args)
            print(f"  组 {spec.group} (自变量 {spec.variable or '?'})")
            print(f"    命令: {' '.join(argv)}")
            if spec.adopted_dir:
                print(f"    登记已有目录: {spec.adopted_dir}")
        print("dry-run: 没有执行任何命令, 也没有打开任何设备")
        assert_no_devices(drivers_before)
        return 0

    # -- 续跑: 已经有产物的组登记为 adopted (不重跑) ------------------------
    if args.resume:
        specs = apply_resume(specs, root)

    # -- 开跑前等设备空出来 -------------------------------------------------
    if float(args.wait_until_idle) > 0:
        device_params = plan_device_params(specs)
        code, reason = wait_until_idle(
            float(args.wait_until_idle),
            probe_cmd=resolve_probe_cmd(args.device_probe_cmd),
            poll_s=float(args.wait_poll_s),
            slm_number=int(device_params["slm_number"]),
            cam_id=int(device_params["cam_id"]),
            cam_type=str(device_params["cam_type"]),
            exposure_ms=float(device_params["exposure_ms"]),
        )
        if code != 0:
            print(
                f"设备没拿到, 本轮不开跑 ({reason or '设备不可用'}); "
                "设备恢复后可用 --resume 续跑",
                file=sys.stderr,
            )
            return code

    created_utc = utc_now_iso()
    records: list[dict[str, Any]] = []
    aborted: DeviceUnavailable | None = None
    for index, spec in enumerate(specs):
        if spec.adopted_dir:
            record = adopt_group(spec, Path(spec.adopted_dir))
        else:
            missing = missing_dependencies(spec)
            if missing:
                # 上游那一块还没产出 ⇒ 这一组不能跑 (会读到不存在的 npz)。
                record = skipped_group(
                    spec,
                    root,
                    "missing_dependency",
                    detail="; ".join(
                        describe_missing_dependency(p) for p in missing
                    ),
                )
                record["missing_dependencies"] = [str(p) for p in missing]
            else:
                record = run_group(
                    spec, root, args.timeout_s, args.max_group_seconds
                )
                verdict = None
                if record["status"] in ("failed", "timeout"):
                    verdict = classify_device_unavailable(
                        f"{record.get('stderr_tail', '')}\n{record.get('stdout_tail', '')}"
                    )
                if verdict is not None:
                    # 🔴 设备不可用 ≠ 这一组坏了: 中止整轮, 别把剩下的组也烧一遍
                    # (2026-10-21 18:23 实测 27 组在 100 s 内全灭)。
                    record["status"] = "device_unavailable"
                    record["device_unavailable"] = {
                        "kind": verdict.kind,
                        "code": verdict.code,
                        "detail": verdict.detail,
                        "remedy": verdict.remedy,
                    }
                    aborted = verdict
                    print(
                        f"[组] {spec.group}: device_unavailable "
                        f"({verdict.kind}) —— 中止整轮",
                        flush=True,
                    )
        records.append(record)
        # 每组跑完立刻写 —— 整轮被强杀时, 已完成的组仍然在索引里。
        write_groups_manifest(
            root, records, plan_path=str(args.plan), created_utc=created_utc
        )
        if aborted is not None:
            # 剩下的组一律记 skipped (不是 failed —— 它们根本没跑)。
            for rest in specs[index + 1:]:
                records.append(
                    skipped_group(
                        rest,
                        root,
                        "round_aborted",
                        detail="因整轮中止 (设备不可用) 而跳过",
                    )
                )
            break

    manifest_path = write_groups_manifest(
        root, records, plan_path=str(args.plan), created_utc=created_utc
    )
    failed = [
        r
        for r in records
        if r["status"] in ("failed", "timeout", "device_unavailable")
    ]
    over_budget = [r for r in records if r["status"] == "over_budget"]
    skipped = [r for r in records if r["status"] == "skipped"]
    longest = max(records, key=lambda r: float(r["duration_s"]), default=None)
    print(f"\n索引: {manifest_path}")
    for line in (
        f"{r['group']}: {r['status']} ({float(r['duration_s']):.1f}s)"
        for r in records
    ):
        print(f"  {line}")
    if longest is not None:
        print(
            f"最长的一组: {longest['group']} 用了 {float(longest['duration_s']):.1f}s / "
            f"预算 {float(args.max_group_seconds):.1f}s"
        )
    if over_budget:
        # 点名: 超预算的组必须出现在汇总里, 否则「每组不超过 10 分钟」这条约束
        # 就没有人盯着。
        print(
            f"超预算的组 ({len(over_budget)}): "
            + ", ".join(
                f"{r['group']} ({float(r['duration_s']):.1f}s)" for r in over_budget
            ),
            file=sys.stderr,
        )
    if skipped:
        print(f"有 {len(skipped)} 组被跳过 (见索引里的 skip_reason)", file=sys.stderr)
    if aborted is not None:
        print(
            f"[中止] 设备不可用: {aborted.headline()} —— {aborted.advice()}。"
            "本轮已中止 (剩余组标记为 skipped); 设备恢复后可用 --resume 续跑",
            file=sys.stderr,
        )
        return 1
    if failed:
        print(f"有 {len(failed)} 组失败/超时 (其余组已照常完成并记录)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
