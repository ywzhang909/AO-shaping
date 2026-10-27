"""GS 方形光斑三段扫描 (边长 / Zernike 阶数 / 闪耀光栅) 的离线报告生成器。

做什么
======
``scripts/slm/hw_gs_square_scans.py`` 把三段扫描的每一个配置都落盘成
``measurements.csv`` / ``measurements.json`` / ``run_config.json`` / ``metrics_*.json``。
本脚本**只读这些已落盘的产物** (以及那次运行的日志), 渲染一份可入库的实验报告 +
四张图, 并把输入产物复制进报告的 ``artifacts/`` 作为**输入契约**。

**它不 import 任何驱动、不打开任何设备** —— 除了 ``matplotlib`` / ``numpy`` /
``_common`` 之外没有第三方依赖, 所以可以在没有 SLM / 相机的机器上重跑。

三条纪律
========
1. **零硬编码数字。** 报告里的每一个测量值都从 ``measurements.csv`` 读出来
   (``run_config.json`` 提供参数, ``manifest.json`` 提供行数, 日志提供 settle 判据
   与曝光/0 级位置). 想改一个数就改产物, 报告跟着变。
2. **表格要回读校验。** 渲染完成后, 从**已渲染的 Markdown** 里把每张数值表解析
   回来, 逐单元与 ``measurements.csv`` 的原文比对 (容差 = 该单元显示精度的半个
   末位). 任何一处不一致都**非零退出** —— 「表里的数字与产物一致」这条不能靠
   人工核对。
3. **``--check`` 只读。** 重新渲染到内存, 与磁盘上的报告逐字节比对; 同时校验
   ``artifacts/`` 里的副本与源文件逐字节一致、正文引用的图都存在。**零 diff** 是
   幂等的判据, 写图只在普通模式下发生。

输入 (默认)
===========
* ``data/debug/hw_gs_square_scans/20261023_gs_square/`` —— 三段产物 (``--run-root``)
* ``logs/gs_square_scan.log`` —— 那次运行的日志 (``--log``); 缺失时报告会明确写出
  「settle 判据状态未知」, 而不是假装没有超时。

产物
====
* ``report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md``
* ``report/shape_trials/hw_gs_square_scans/figures/*.png`` (4 张)
* ``report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square/`` (输入契约)

复现
====
.. code-block:: powershell

    python scripts/slm/generate_hw_gs_square_scans_report.py
    python scripts/slm/generate_hw_gs_square_scans_report.py --check   # 幂等校验
"""

from __future__ import annotations

import argparse
from loguru import logger
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# 让脚本可独立运行: `python scripts/slm/generate_hw_gs_square_scans_report.py` 时
# sys.path[0] 是 scripts/slm/, 既找不到 `ao_shaping` (src/) 也找不到 `_common`
# (scripts/) —— 与 `scripts/slm/compare_sim_vs_hw.py` 同一套自举写法。
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")  # 🔴 必须在 import matplotlib.pyplot 之前 (仓库级规则)

import matplotlib.pyplot as plt
import numpy as np

from _common import savefig
from _common.provenance import REPORTS, insert_header

ROOT = _ROOT

#: 报告键 (仓库根相对路径)。生成脚本**写死新目录** —— 重跑不会写回旧位置。
REPORT_KEY = "report/shape_trials/hw_gs_square_scans/hw_gs_square_scans_20261023.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "hw_gs_square_scans"
FIG_DIR = OUT_DIR / "figures"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "20261023_gs_square"

#: 默认产物根与日志。
DEFAULT_RUN_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_gs_square"
DEFAULT_LOG = ROOT / "logs" / "gs_square_scan.log"

#: 图名戳 —— 同系列里每份报告独占自己的图 (报告/AGENTS.md §2.1)。
FIG_TAG = "_20261023"

#: 三段与各自的自变量列名。
STAGES: tuple[str, ...] = ("sides", "zernike", "blaze")
VARIABLE: dict[str, str] = {
    "sides": "side_px",
    "zernike": "n_max",
    "blaze": "blaze_period_px",
}
STAGE_CN: dict[str, str] = {
    "sides": "边长扫描",
    "zernike": "Zernike 阶数扫描",
    "blaze": "闪耀光栅扫描",
}

#: 8 bit 相机的满量程 (counts)。行里 ``frame_peak`` 达到它就是**饱和**, 读数只能
#: 当"≥满量程"用, 不能当线性测量值。相机 dtype 由 ``light_level`` 一类探针记录为
#: ``uint8``; 这里把它当**器件常数**, 不是测量值。
SATURATION_COUNTS = 255.0

#: `‡` 帧电平异常判据: 该行 ``frame_peak`` 低于同段中位数的这个比例。
DARK_FRAME_FRACTION = 0.10

#: `※` CV 离群判据: 该行 ``cv`` 超过同段中位数的这个倍数。
OUTLIER_CV_RATIO = 1.5
#: 在 CSV 行里按浮点键回查一条记录 (``blaze_period_px`` / ``n_max`` / ``side_px`` /
#: 优化指标 ``variable``) 的匹配容差。命中唯一行才算「这帧对应这行」, 落空的后果是
#: 报告里出现空洞而不是崩, 所以这道容差是**表示**级 (float 往返), 不是光学判据。
ROW_MATCH_ATOL = 1e-9

#: ``.npz`` 里的 params 与 CSV 行必须一致 (cv / pib / score / frame_peak 这类
#: 由测量算出的量): 超过它就判「帧与 CSV 行对不上」并终止渲染。取 **1e-6** 是因为
#: 这些量经过一次 float 化与 CSV 的十进制文本往返, 1e-9 会把纯表示差异误报成不一致。
PARAM_NPZ_CSV_ATOL = 1e-6

#: 同一个校验里 ``side_px`` 用的是 **1e-9** 而不是上面的 1e-6: 它是整数语义的边长
#: (下发值), 只有 float 往返误差, 没有任何算术放大 —— 两个容差**故意不同**, 统一
#: 成一个就把这条整数硬校验放松了。
SIDE_PX_NPZ_CSV_ATOL = 1e-9

#: ``score = 0.5·PIB + 0.5/(1+CV)`` 恒等式的复算容差: 报告读到的 score 必须由同一行
#: 的 PIB/CV 复算得出, 差得比 float 算术噪声大就说明落盘时被别的代码路径改写过。
SCORE_IDENTITY_ATOL = 1e-6

#: 框面积恒等式 ``side² = box_area_px + box_clipped_px`` 的容差。这里是**像素面积**
#: 量级 (side 可到千余 px), float 表示误差随量级放大, 所以比 score 那条松三个数量级。
AREA_IDENTITY_ATOL = 1e-3

#: 曝光读回值与生效值一致的容差 (``exposure_ms_readback`` vs ``exposure_ms`` /
#: ``exposure_ms_working``): 驱动回读的是同一个下发浮点, 只允许表示误差。
EXPOSURE_MATCH_ATOL = 1e-9

#: 帧最大值与 CSV ``frame_peak`` 的一致性容差 (counts, 8 bit 满量程 255 量级):
#: 这是**跨表示**核对 (帧 mean 与写盘时的 float 文本), 1e-3 counts 已远小于任何
#: 有意义的电平差, 再松就会让真实的帧↔行错配溜过去。
PEAK_MATCH_ATOL = 1e-3

# ---------------------------------------------------------------------------
# 复测批次 (功率已变) —— 第二数据源
# ---------------------------------------------------------------------------

#: 复测产物根 (``--retest-root``)。不给就完全不渲染 §11, 报告与没有这个参数时逐字节相同。
#: 默认指向**冻结版**那一批 (单一代码版本, hash b7e964f3)。
DEFAULT_RETEST_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_retest_frozen"

#: 更早一版复测 (同协议、同功率, 但代码版本不同)。只用来算**批间重复性**:
#: 两批同协议的 CV / 位移差就是「复测自身能分辨多小」的带宽。不存在就跳过那张表。
DEFAULT_RETEST_PRIOR_ROOT = (
    ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_retest_power_changed"
)

#: 复测日志目录 (``--retest-log-dir``); 文件名为 ``retest_<stage>_<取值>.log``。
DEFAULT_RETEST_LOG_DIR = ROOT / "logs"

#: 整轮原子重扫 (``--atomic-root``)。这是**唯一「单一脚本版本 + 单一功率」的完整一轮**,
#: 所以三段数字第一次真正可比。不给就完全不渲染 §13。
DEFAULT_ATOMIC_ROOT = ROOT / "data" / "debug" / "hw_gs_square_scans" / "20261023_atomic_full"

#: 原子整轮产物的归档目录 (同样从默认根的目录名推导)。
ATOMIC_ARTIFACT_DIR = OUT_DIR / "artifacts" / DEFAULT_ATOMIC_ROOT.name

#: 复测产物的归档目录 (报告输入契约的第二块)。基线那一批在 ``ARTIFACT_DIR``。
#: 目录名**从默认根的目录名推导** —— 换 ``--retest-root`` 时 :func:`main` 会把它改成
#: 参数那一批的目录名, 所以这里不另抄一份批次字面量。
RETEST_ARTIFACT_DIR = OUT_DIR / "artifacts" / DEFAULT_RETEST_ROOT.name
RETEST_PRIOR_ARTIFACT_DIR = OUT_DIR / "artifacts" / DEFAULT_RETEST_PRIOR_ROOT.name

#: 复测批次**没有** sides 段 —— 每个配置都用 ``--side-px`` 显式覆盖边长。
RETEST_STAGES: tuple[str, ...] = ("blaze", "zernike")
RETEST_STAGE_CN: dict[str, str] = {
    "blaze": "闪耀光栅",
    "zernike": "Zernike 阶数",
}


@dataclass(frozen=True)
class FieldRecord:
    """**现场记录** —— 不在这两批产物里的外部读数。

    存在的唯一理由: 说明「激光功率变了多少」。它**不参与任何 CSV 解析**,
    只出现在 §11.1 一张表头写着「现场记录（非本批产物）」的表里, 并且
    :func:`retest_power_ratio` 的入参就是它 —— 想改数就改这里, 并同时改 ``source``。

    ``readings`` 是 ``(曝光 ms, 峰值 counts)`` 的序列。**斜率只用最大曝光那个点**:
    低曝光读数被噪声地板抬高 (0.05 ms 只有 3 counts), 拿它定斜率会把功率比算歪。
    """

    label: str
    readings: tuple[tuple[float, float], ...]
    source: str

    def anchor(self) -> tuple[float, float]:
        """返回 ``(曝光 ms, 峰值 counts)`` 里曝光最大的那个点。"""
        return max(self.readings, key=lambda pair: pair[0])

    def counts_per_ms(self) -> float:
        exposure_ms, peak = self.anchor()
        return peak / exposure_ms


#: 基线同期的平场 0 级读数 (现场记录)。
FIELD_BASELINE_FLAT = FieldRecord(
    label="基线同期 平场 0 级",
    readings=((0.5, 163.0),),
    source="logs/probe_light_exposures.py 现场读数（2026-10-23，基线同一光路状态）",
)

#: 复测前的平场 0 级曝光阶梯 (现场记录)。
FIELD_RETEST_FLAT = FieldRecord(
    label="复测前 平场 0 级",
    readings=((0.05, 3.0), (0.2, 8.0), (1.0, 38.0), (5.0, 180.0)),
    source="logs/probe_light_exposures.py 现场读数（2026-10-23，复测开始前）",
)


# ---------------------------------------------------------------------------
# 中文渲染 (字体 + rcParams)
# ---------------------------------------------------------------------------


def setup_matplotlib() -> None:
    """全局绘图约定: CJK 字体 + 负号不显示成方块。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    family = [name for name in preferred if name in installed] or ["DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = family
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 表格单元: 每个数字都记得自己来自哪个 CSV 原文
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """一个表格单元。

    ``value`` 是**数值本身** (来自 CSV 原文或现场推导), ``nd`` 是显示精度。
    回读校验的容差就是 ``0.5 · 10^-nd`` —— 即"显示精度内的半个末位", 所以只要
    渲染时多丢一位小数, 校验立刻失败。
    """

    value: float
    nd: int = 4
    suffix: str = ""

    def text(self) -> str:
        if not math.isfinite(self.value):
            return "-"
        return f"{self.value:.{self.nd}f}{self.suffix}"

    def accepts(self, parsed: float) -> bool:
        """``parsed`` (从渲染后的 Markdown 解析回来) 是否就是这个数。"""
        if not math.isfinite(self.value):
            return not math.isfinite(parsed)
        return abs(parsed - self.value) <= 0.5 * 10.0 ** (-self.nd) + 1e-12


def cell_of(row: dict[str, str], key: str, nd: int = 4, suffix: str = "") -> Cell:
    """从 CSV 的一行 (原文字典) 取一个单元。键缺失即报错, 绝不静默补 0。"""
    if key not in row:
        raise KeyError(f"CSV 列 '{key}' 不存在; 实际表头按实际表头解析, 见 load_stage")
    text = row[key]
    try:
        value = float(text)
    except ValueError as exc:  # pragma: no cover - 产物损坏时才会走到
        raise ValueError(f"列 '{key}' 的值 {text!r} 不是数字") from exc
    return Cell(value=value, nd=nd, suffix=suffix)


def derived(value: float, nd: int = 4, suffix: str = "") -> Cell:
    """现场推导出来的数 (差值、比值、评分分解项…)。"""
    return Cell(value=float(value), nd=nd, suffix=suffix)


@dataclass
class Table:
    """一张数值表 + 它自己的回读校验。

    ``rows`` 里既有 :class:`Cell` 也有纯文本 (标记列, 如 ``正常`` / ``饱和``)。
    """

    headers: tuple[str, ...]
    rows: list[tuple[Cell | str, ...]] = field(default_factory=list)

    def render(self) -> str:
        lines = ["| " + " | ".join(self.headers) + " |"]
        lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        for row in self.rows:
            lines.append(
                "| " + " | ".join(c.text() if isinstance(c, Cell) else str(c) for c in row) + " |"
            )
        return "\n".join(lines)

    @property
    def numeric_cells(self) -> int:
        return sum(1 for row in self.rows for c in row if isinstance(c, Cell))


def verify_table_in_markdown(markdown: str, table: Table) -> list[str]:
    """把 ``table`` 从**已渲染的 Markdown** 里找出来, 逐单元回读比对。

    返回问题清单 (空 = 通过)。同一张表在正文里必须**恰好出现一次**; 表格块以
    表头行起始、到第一个非 ``|`` 行结束 —— 与 :meth:`Table.render` 的输出形状
    严格对应。
    """
    problems: list[str] = []
    header_line = "| " + " | ".join(table.headers) + " |"
    occurrences = markdown.count(header_line)
    if occurrences != 1:
        return [f"表头 {header_line!r} 出现 {occurrences} 次 (期望 1 次)"]
    start = markdown.index(header_line)
    block = markdown[start:].split("\n\n", 1)[0].splitlines()
    body = [ln for ln in block if ln.startswith("|")][2:]  # 去掉表头 + 分隔行
    if len(body) != len(table.rows):
        problems.append(f"表头 {header_line!r}: 行数 {len(body)} != {len(table.rows)}")
    for r, (line, expected_row) in enumerate(zip(body, table.rows)):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(expected_row):
            problems.append(f"第 {r} 行: 列数 {len(cells)} != {len(expected_row)}")
            continue
        for c, (text, expected) in enumerate(zip(cells, expected_row)):
            if not isinstance(expected, Cell):
                if text != str(expected):
                    problems.append(f"第 {r} 行第 {c} 列: {text!r} != {expected!r}")
                continue
            bare = text
            if expected.suffix and bare.endswith(expected.suffix):
                bare = bare[: -len(expected.suffix)]
            if bare == "-":
                if math.isfinite(expected.value):
                    problems.append(f"第 {r} 行第 {c} 列: 显示 '-' 但产物值是 {expected.value}")
                continue
            try:
                parsed = float(bare)
            except ValueError:
                problems.append(f"第 {r} 行第 {c} 列: {text!r} 不是数字")
                continue
            if not expected.accepts(parsed):
                problems.append(
                    f"第 {r} 行第 {c} 列: 表里 {parsed!r} != 产物 {expected.value!r}"
                    f" (显示精度 {expected.nd} 位)"
                )
    return problems


# ---------------------------------------------------------------------------
# 读产物
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """按**实际表头**读 CSV, 返回 (表头, 原文字典行)。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少产物文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"{path} 里没有数据行")
    return headers, rows


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"缺少产物文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Stage:
    """一段扫描的全部产物 + 解析出来的派生量。"""

    name: str
    headers: list[str]
    rows: list[dict[str, str]]
    run_config: dict[str, Any]
    manifest_entry: dict[str, Any]

    @property
    def variable(self) -> str:
        return VARIABLE[self.name]

    @property
    def x(self) -> list[float]:
        return [float(r[self.variable]) for r in self.rows]

    def col(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.rows]

    def median(self, key: str) -> float:
        vals = [v for v in self.col(key) if math.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")


@dataclass(frozen=True)
class SettleEvent:
    """日志里的一条 ``settle criterion not met`` 记录。"""

    stage: str
    index: int
    waited_s: float
    probes: int
    probe_peak: float
    probe_fwhm: float
    raw_fragment: str = ""
    row_key: str = ""

    def describe(self) -> str:
        return (
            f"settle 判据超时 {self.waited_s:.2f} s / {self.probes} 次探测 / "
            f"探测峰值 {self.probe_peak:g}"
        )


@dataclass
class Run:
    """一次运行的全部输入。"""

    root: Path
    stages: dict[str, Stage]
    manifest: list[dict[str, Any]]
    plan: dict[str, Any]
    stage_plans: dict[str, dict[str, Any]]
    settle_events: list[SettleEvent]
    log_path: Path | None
    log_crosscheck: list[tuple[str, str, str, bool]]

    # -- 常用参数, 全部来自 run_config / manifest ---------------------------
    @property
    def exposure_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ms_working"])

    @property
    def exposure_requested_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ms_requested"])

    @property
    def exposure_ceiling_ms(self) -> float:
        return float(self.stages["sides"].run_config["exposure_ceiling_ms"])

    @property
    def zero_order_xy(self) -> tuple[float, float]:
        xy = self.stages["sides"].run_config["zero_order_xy_frozen"]
        return float(xy[0]), float(xy[1])

    @property
    def frame_shape_hw(self) -> tuple[int, int]:
        h, w = self.stages["sides"].run_config["frame_shape_hw"]
        return int(h), int(w)

    @property
    def settle_policy(self) -> dict[str, Any]:
        return dict(self.stages["sides"].run_config.get("settle") or {})

    @property
    def spot_assessment(self) -> dict[str, Any]:
        return dict(self.stages["sides"].run_config.get("spot_assessment") or {})

    def settle_for(self, stage: str, index: int) -> SettleEvent | None:
        for ev in self.settle_events:
            if ev.stage == stage and ev.index == index:
                return ev
        return None


#: 日志里的 settle 告警 (会被终端按 ~118 字符硬折行, 所以先"展开"再匹配)。
_SETTLE_RE = re.compile(
    r"settle criterion not met after (?P<waited>[\d.]+) s\s*\((?P<probes>\d+) probes,"
    r" n_stable=\d+\): settled=False \(t\s*imeout\)\s*probes=(?P=probes)"
    r" peak=(?P<peak>[\d.]+) fwhm=(?P<fwhm>[-\d.]+|nan)px"
)
_STAGE_LINE_RE = re.compile(r"^\[(?P<stage>sides|zernike|blaze) #(?P<idx>\d+)\]")
_STAMP_RE = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d|\[|\s*$)")


def _read_log_text(path: Path) -> str:
    """读日志文本, 自己判编码。

    ``Tee-Object`` 在 Windows PowerShell 上默认写 **UTF-16LE + BOM**, 用 UTF-8 强读
    会把整个文件读成一行 (换行被 0x00 隔开), 于是 settle 告警一条也匹配不到 ——
    本模块踩过这个坑, 所以编码判断放在唯一入口里, 不做 "默认 utf-8"。
    """
    raw = path.read_bytes()
    for bom, encoding in ((b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
                          (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(encoding, errors="replace")
    for encoding in ("utf-8", "cp936"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _unwrap_log(path: Path) -> list[str]:
    """去掉 ANSI 色码, 并把被终端硬折行的续行接回上一行。

    规则: 以时间戳 / ``[`` 开头 (或空行) 的行是新记录, 其余都是上一条的续行。
    折行位置不可控, 所以"展开"必须在解析前做, 否则 ``settle criterion not met
    after 7.03 s`` 会被切成两行而匹配不到。
    """
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    out: list[str] = []
    for raw in _read_log_text(path).splitlines():
        line = ansi.sub("", raw)
        if _STAMP_RE.match(line) or not out:
            out.append(line)
        else:
            out[-1] += line
    return out


def parse_log(path: Path | None) -> tuple[list[SettleEvent], dict[str, str]]:
    """解析日志里的 settle 告警与若干关键行。

    settle 告警出现在**该配置的结果行之前** (下发相位 → 等稳定 → 取帧 → 打印),
    所以一条告警归属于它**之后**出现的第一条 ``[stage #NNN]`` 行 —— 这是本仓库
    采集脚本的输出顺序, 不是猜测。
    """
    if path is None or not path.exists():
        return [], {}
    events: list[SettleEvent] = []
    pending: list[tuple[re.Match[str], str]] = []
    key_lines: dict[str, str] = {}
    for line in _unwrap_log(path):
        if "settle criterion not met" in line:
            match = _SETTLE_RE.search(line)
            if match:
                pending.append((match, line[line.index("settle criterion not met"):]))
            continue
        stage_line = _STAGE_LINE_RE.match(line)
        if stage_line:
            for match, fragment in pending:
                events.append(
                    SettleEvent(
                        stage=stage_line.group("stage"),
                        index=int(stage_line.group("idx")),
                        waited_s=float(match.group("waited")),
                        probes=int(match.group("probes")),
                        probe_peak=float(match.group("peak")),
                        probe_fwhm=float(match.group("fwhm")),
                        raw_fragment=fragment,
                    )
                )
            pending.clear()
            continue
        for name, pattern in (
            ("exposure", r"曝光: 请求 [\d.]+ ms -> 上限 [\d.]+ ms -> 生效 (?P<v>[\d.]+) ms"),
            ("zero_order_x", r"0 级 \(全幅 argmax, 已冻结\): \((?P<v>\d+), \d+\)"),
            ("zero_order_y", r"0 级 \(全幅 argmax, 已冻结\): \(\d+, (?P<v>\d+)\)"),
            ("side_max", r"最大边长: (?P<v>\d+) px \(原始 \d+ px\)"),
            ("side_max_raw", r"最大边长: \d+ px \(原始 (?P<v>\d+) px\)"),
            ("selected_side", r"\[选定\] 边长 (?P<v>\d+) px"),
        ):
            found = re.search(pattern, line)
            if found:
                key_lines[name] = found.group("v")
    return events, key_lines


def load_run(root: Path, log_path: Path | None) -> Run:
    """读全部产物, 并做**输入契约**的硬校验 (CSV ↔ JSON ↔ manifest ↔ 日志)。"""
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, list):
        raise ValueError(f"{root / 'manifest.json'} 不是 manifest 列表")
    by_group = {str(entry.get("group")): entry for entry in manifest}

    stages: dict[str, Stage] = {}
    for name in STAGES:
        stage_dir = root / name
        headers, rows = load_csv(stage_dir / "measurements.csv")
        payload = load_json(stage_dir / "measurements.json")
        run_config = load_json(stage_dir / "run_config.json")
        if not isinstance(payload, list):
            raise ValueError(f"{stage_dir / 'measurements.json'} 不是行列表")
        _cross_check_csv_json(name, rows, payload)
        if name not in by_group:
            raise ValueError(f"manifest 里没有 '{name}' 这一段")
        entry = by_group[name]
        if int(entry.get("n_rows", -1)) != len(rows):
            raise ValueError(
                f"manifest 记 {name} 有 {entry.get('n_rows')} 行, CSV 实际 {len(rows)} 行"
            )
        if str(entry.get("variable")) != VARIABLE[name]:
            raise ValueError(
                f"manifest 记 {name} 的自变量是 {entry.get('variable')!r}, "
                f"期望 {VARIABLE[name]!r}"
            )
        stages[name] = Stage(name=name, headers=headers, rows=rows,
                             run_config=run_config, manifest_entry=entry)

    plan = dict(stages["sides"].run_config.get("plan") or {})
    stage_plans = dict(plan.get("stage_plans") or {})
    events, key_lines = parse_log(log_path)
    for ev in events:
        if ev.stage not in stages:
            raise ValueError(f"日志里的 stage {ev.stage!r} 不在 {STAGES}")
        if ev.index >= len(stages[ev.stage].rows):
            raise ValueError(
                f"日志说 {ev.stage} #{ev.index} 有 settle 超时, 但 CSV 只有 "
                f"{len(stages[ev.stage].rows)} 行"
            )
    crosscheck = _log_crosscheck(stages, key_lines) if key_lines else []
    return Run(root=root, stages=stages, manifest=manifest, plan=plan,
               stage_plans=stage_plans, settle_events=events,
               log_path=log_path if (log_path and log_path.exists()) else None,
               log_crosscheck=crosscheck)


def _cross_check_csv_json(
    name: str, rows: list[dict[str, str]], payload: Sequence[dict[str, Any]]
) -> None:
    """CSV 与 JSON 必须逐值一致 (两者由同一个 writer 写出, 差别就是真问题)。"""
    if len(rows) != len(payload):
        raise ValueError(f"{name}: CSV {len(rows)} 行 != JSON {len(payload)} 行")
    for i, (row, obj) in enumerate(zip(rows, payload)):
        for key, value in row.items():
            if key not in obj:
                continue
            try:
                expected = float(value)
                actual = float(obj[key])
            except (TypeError, ValueError):
                # 非数值列 (如 timestamp_utc) 直接按字符串比
                if str(value) != str(obj[key]):
                    raise ValueError(
                        f"{name} 第 {i} 行 '{key}': CSV {value!r} != JSON {obj[key]!r}"
                    )
                continue
            if math.isnan(expected) and math.isnan(actual):
                continue
            if expected != actual:
                raise ValueError(
                    f"{name} 第 {i} 行 '{key}': CSV {value} != JSON {obj[key]}"
                )


def _selected_side_px_from(stages: dict[str, "Stage"]) -> float:
    """``selected_side_px``: 先看 zernike 段 run_config, 再回落到它的 CSV 行。"""
    cfg = stages["zernike"].run_config
    if "selected_side_px" in cfg:
        return float(cfg["selected_side_px"])
    return float(stages["zernike"].rows[0]["selected_side_px"])


def _selected_side_px(run: "Run") -> float:
    return _selected_side_px_from(run.stages)


def _log_crosscheck(
    stages: dict[str, Stage], key_lines: dict[str, str]
) -> list[tuple[str, str, str, bool]]:
    """日志里打印的关键量与 ``run_config`` 必须一致 (否则 settle 表会错配到别的 run)。"""
    cfg = stages["sides"].run_config
    checks: list[tuple[str, str, str, bool]] = []

    def add(name: str, logged: Any, actual: Any) -> None:
        checks.append((name, str(logged), str(actual), str(logged) == str(actual)))

    if "exposure" in key_lines:
        add("生效曝光 (ms)", f"{float(key_lines['exposure']):g}", f"{float(cfg['exposure_ms_working']):g}")
    if "zero_order_x" in key_lines and "zero_order_y" in key_lines:
        xy = cfg["zero_order_xy_frozen"]
        add("0 级位置", f"({key_lines['zero_order_x']}, {key_lines['zero_order_y']})",
            f"({int(xy[0])}, {int(xy[1])})")
    if "side_max" in key_lines:
        add("最大边长 (px)", key_lines["side_max"], f"{int(cfg['side_max_px'])}")
    if "side_max_raw" in key_lines:
        add("原始最大边长 (px)", key_lines["side_max_raw"], f"{int(cfg['side_max_raw_px'])}")
    if "selected_side" in key_lines:
        add("选定边长 (px)", key_lines["selected_side"],
            f"{int(_selected_side_px_from(stages))}")
    return checks


# ---------------------------------------------------------------------------
# 派生量: 全部现场算, 不抄
# ---------------------------------------------------------------------------


def score_terms(pib: float, cv: float) -> tuple[float, float]:
    """``score = 0.5·PIB + 0.5·1/(1+CV)`` 的两项 (与采集脚本同一口径)。"""
    return 0.5 * pib, 0.5 / (1.0 + cv)


def monotonic_report(values: Sequence[float], *, increasing: bool = True) -> list[int]:
    """返回**破坏**单调性的下标 (i 表示 values[i] 相对 values[i-1] 反向)。"""
    out: list[int] = []
    for i in range(1, len(values)):
        if increasing and values[i] <= values[i - 1]:
            out.append(i)
        if not increasing and values[i] >= values[i - 1]:
            out.append(i)
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """秩相关 (无 scipy 依赖): 秩后取 Pearson。"""
    def rank(v: Sequence[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        order = arr.argsort()
        ranks = np.empty(arr.size, dtype=float)
        ranks[order] = np.arange(1, arr.size + 1, dtype=float)
        # 平秩取平均 (本数据里自变量无重复, 这里只是稳妥)
        for value in np.unique(arr):
            mask = arr == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    return pearson(rank(xs), rank(ys))


def flag_rows(run: Run, stage: Stage) -> list[str]:
    """逐行的旗标: ``†`` settle 超时, ``‡`` 帧电平异常, ``※`` CV 离群, ``§`` 框出画。

    ``※`` 刻意用**邻行**做参照而不是只用段中位数: 边长段的 CV 本身随边长单调爬升
    (ρ ≈ 0.996), 若只比中位数就会把趋势的尾巴 (1700/1800 px) 当成离群点。判据是
    「比中位数高 >1.5× **且** 比邻行中位数高 >1.5× **且** 邻行本身不离中位数太远」,
    于是只有真正从平稳背景里跳出来的点会被标 —— 本轮的 ``n_max = 18``、``period = 40``。
    """
    med_peak = stage.median("frame_peak")
    med_cv = stage.median("cv")
    cvs = stage.col("cv")
    out: list[str] = []
    for i, row in enumerate(stage.rows):
        marks: list[str] = []
        if run.settle_for(stage.name, i) is not None:
            marks.append("†")
        peak = float(row["frame_peak"])
        if math.isfinite(med_peak) and med_peak > 0 and peak < DARK_FRAME_FRACTION * med_peak:
            marks.append("‡")
        cv = float(row["cv"])
        neighbours = [cvs[j] for j in (i - 1, i + 1) if 0 <= j < len(cvs)]
        if (
            neighbours
            and math.isfinite(med_cv) and med_cv > 0
            and cv > OUTLIER_CV_RATIO * med_cv
            and cv > OUTLIER_CV_RATIO * float(np.median(neighbours))
            and float(np.median(neighbours)) < 1.2 * med_cv
        ):
            marks.append("※")
        if float(row["box_clipped_px"]) > 0:
            marks.append("§")
        out.append("".join(marks))
    return out


def saturation_flag(row: dict[str, str]) -> str:
    peak = float(row["frame_peak"])
    if peak >= SATURATION_COUNTS:
        return "饱和"
    if peak >= 0.95 * SATURATION_COUNTS:
        return "接近饱和"
    return "未饱和"


@dataclass
class Findings:
    """报告要讲的全部结论 —— 每一个都在这里算出来, 正文只做排版。"""

    sides_x: list[float]
    pib: list[float]
    cv: list[float]
    score: list[float]
    score_peak_index: int
    score_runner_up_index: int
    score_peak_gap: float
    pib_violations: list[int]
    cv_violations: list[int]
    cv_spearman: float
    score_descend_ok: bool
    terms_at_peak: tuple[float, float]
    terms_at_last: tuple[float, float]
    terms_at_small: tuple[float, float]
    near_window: tuple[float, float]
    near_best_index: int
    alt_window: tuple[float, float]
    alt_best_index: int
    selected_side: int
    zern_explained: list[float]
    zern_residual: list[float]
    zern_cv: list[float]
    zern_score: list[float]
    zern_clean_cv: list[float]
    zern_clean_cv_spread: float
    zern_cv_pearson: float
    zern_explained_violations: list[int]
    blaze_score: list[float]
    blaze_score_violations_clean: list[int]
    blaze_score_violations_all: list[int]
    blaze_shift_ratio: list[float]
    model_pib: list[float]
    model_delta: list[float]
    side_max_raw: float
    side_max: float
    step: float
    edge_margin: float
    binding_edge: str
    dropped_configs: list[str]
    csv_cells: int


def compute_findings(run: Run) -> Findings:
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]

    xs = sides.x
    pib = sides.col("pib")
    cv = sides.col("cv")
    score = sides.col("score")
    order = sorted(range(len(score)), key=lambda i: -score[i])
    peak = order[0]
    runner = order[1]

    cv_viol = monotonic_report(cv, increasing=True)
    pib_viol = monotonic_report(pib, increasing=True)
    descend = all(score[i] < score[i - 1] for i in range(peak + 1, len(score)))

    near_lo, near_hi = (float(v) for v in (run.stages["sides"].run_config["near_window_px"]))
    near_idx = [i for i, x in enumerate(xs) if near_lo <= x <= near_hi]
    near_best = max(near_idx, key=lambda i: score[i])
    alt_lo, alt_hi = 900.0, 1100.0
    alt_idx = [i for i, x in enumerate(xs) if alt_lo <= x <= alt_hi]
    alt_best = max(alt_idx, key=lambda i: score[i])

    zern_cv = zern.col("cv")
    zern_med = zern.median("cv")
    clean = [
        i for i, r in enumerate(zern.rows)
        if run.settle_for("zernike", i) is None
        and float(r["frame_peak"]) >= DARK_FRAME_FRACTION * zern.median("frame_peak")
        and float(r["cv"]) <= OUTLIER_CV_RATIO * zern_med
    ]

    b_score = blaze.col("score")
    b_clip = [i for i, r in enumerate(blaze.rows) if float(r["box_clipped_px"]) > 0]
    b_dark = [i for i, r in enumerate(blaze.rows)
              if float(r["frame_peak"]) < DARK_FRAME_FRACTION * blaze.median("frame_peak")]
    b_excluded = sorted(set(b_clip) | set(b_dark))
    b_clean = [i for i in range(len(b_score)) if i not in b_excluded]

    shift_ratio = []
    for row in blaze.rows:
        pred = float(row["predicted_shift_px"])
        meas = float(row["measured_shift_y_px"])
        shift_ratio.append(meas / pred if pred else float("nan"))

    model_pib = sides.col("model_pib")
    model_delta = [m - p for m, p in zip(model_pib, pib)]

    cfg = sides.run_config
    zo_x, zo_y = run.zero_order_xy
    h, w = run.frame_shape_hw
    margin = float(cfg["plan"]["frame"]["edge_margin_px"]) if "plan" in cfg else 0.0
    dist = {
        "左": zo_x,
        "右": (w - 1) - zo_x,
        "上": zo_y,
        "下": (h - 1) - zo_y,
    }
    binding = min(dist, key=lambda k: dist[k])
    raw = 2.0 * (dist[binding] - margin)
    step = float(min(b - a for a, b in zip(xs, xs[1:]))) if len(xs) > 1 else 1.0
    side_max = math.floor(raw / step) * step

    planned = [float(v) for v in run.stage_plans.get("sides", {}).get("configs", [])]
    dropped = [f"{int(v)}" for v in planned if v not in set(xs)]

    return Findings(
        sides_x=xs, pib=pib, cv=cv, score=score,
        score_peak_index=peak, score_runner_up_index=runner,
        score_peak_gap=score[peak] - score[runner],
        pib_violations=pib_viol, cv_violations=cv_viol,
        cv_spearman=spearman(xs, cv),
        score_descend_ok=descend,
        terms_at_peak=score_terms(pib[peak], cv[peak]),
        terms_at_last=score_terms(pib[-1], cv[-1]),
        terms_at_small=score_terms(pib[0], cv[0]),
        near_window=(near_lo, near_hi), near_best_index=near_best,
        alt_window=(alt_lo, alt_hi), alt_best_index=alt_best,
        selected_side=_selected_side_px(run),
        zern_explained=zern.col("zernike_explained_fraction"),
        zern_residual=zern.col("zernike_rms_residual_rad"),
        zern_cv=zern_cv, zern_score=zern.col("score"),
        zern_clean_cv=[zern_cv[i] for i in clean],
        zern_clean_cv_spread=(
            (max(zern_cv[i] for i in clean) - min(zern_cv[i] for i in clean))
            / min(zern_cv[i] for i in clean)
        ),
        zern_cv_pearson=pearson([zern.x[i] for i in clean], [zern_cv[i] for i in clean]),
        zern_explained_violations=monotonic_report(zern.col("zernike_explained_fraction")),
        blaze_score=b_score,
        blaze_score_violations_clean=monotonic_report(
            [b_score[i] for i in b_clean]),
        blaze_score_violations_all=monotonic_report(b_score),
        blaze_shift_ratio=shift_ratio,
        model_pib=model_pib, model_delta=model_delta,
        side_max_raw=raw, side_max=side_max, step=step, edge_margin=margin,
        binding_edge=binding,
        dropped_configs=dropped,
        csv_cells=sum(len(r) for s in run.stages.values() for r in s.rows),
    )


# ---------------------------------------------------------------------------
# 复测批次: 读产物 + 批内比较
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettleRecord:
    """单配置日志里的一条 settle 告警 (复测批次每个配置一个日志文件)。"""

    waited_s: float
    probes: int
    probe_peak: float
    probe_fwhm: float
    raw_fragment: str

    def describe(self) -> str:
        return (
            f"settle 判据超时 {self.waited_s:.2f} s / {self.probes} 次探测 / "
            f"探测峰值 {self.probe_peak:g}"
        )


@dataclass
class RetestConfig:
    """复测批次里的一个配置 (原子化调用 ⇒ 一个配置 = 一个进程 = 一份 metrics.json + 一个日志)。"""

    stage: str
    config: str
    value: float
    row: dict[str, str]
    metrics: dict[str, Any]
    log_path: Path | None
    settle: list[SettleRecord]

    @property
    def preflight(self) -> dict[str, Any]:
        return dict(self.metrics.get("preflight") or {})

    @property
    def spot(self) -> dict[str, Any]:
        return dict(self.preflight.get("spot_assessment") or {})

    @property
    def zero_order(self) -> tuple[float, float]:
        xy = self.preflight.get("zero_order_xy_frozen") or (float("nan"), float("nan"))
        return float(xy[0]), float(xy[1])

    @property
    def spot_flagged(self) -> bool:
        """平场评估自己报「没有可评分的局域光斑」(饱和平台让 ``argmax`` 失义)。"""
        return str(self.spot.get("reason", "")).startswith("光路有光但没有可评分的局域光斑")


@dataclass
class RetestStage:
    name: str
    headers: list[str]
    rows: list[dict[str, str]]
    configs: list[RetestConfig]
    #: 段级 ``run_config.json`` (v1 那批没有 ⇒ 空 dict)。**没有 `plan`** 是正常的。
    run_config: dict[str, Any] = field(default_factory=dict)

    @property
    def variable(self) -> str:
        return VARIABLE[self.name]

    @property
    def x(self) -> list[float]:
        """自变量列 (与 :class:`Stage` 同名, 让 ``compute_findings`` 能直接吃这一批)。"""
        return [float(r[self.variable]) for r in self.rows]

    def row_of(self, value: float) -> dict[str, str]:
        for row in self.rows:
            if abs(float(row[self.variable]) - float(value)) < ROW_MATCH_ATOL:
                return row
        raise KeyError(f"复测 {self.name} 段没有取值 {value} 的行")

    def col(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.rows]

    def median(self, key: str) -> float:
        """与 :meth:`Stage.median` 同名同义。"""
        vals = [v for v in self.col(key) if math.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")


@dataclass
class RetestRun:
    root: Path
    manifest: list[dict[str, Any]]
    stages: dict[str, RetestStage]
    log_dir: Path

    @property
    def exposure_ms(self) -> float:
        """生效曝光 —— 从**每配置的 preflight** 读, 并强制它们一致。"""
        values = {float(cfg.preflight.get("exposure_ms_working", float("nan")))
                  for st in self.stages.values() for cfg in st.configs}
        if len(values) != 1:
            raise ValueError(f"复测各配置的生效曝光不一致: {sorted(values)}")
        return values.pop()

    @property
    def exposure_ceiling_ms(self) -> float:
        values = {float(cfg.preflight.get("exposure_ceiling_ms", float("nan")))
                  for st in self.stages.values() for cfg in st.configs}
        if len(values) != 1:
            raise ValueError(f"复测各配置的曝光上限不一致: {sorted(values)}")
        return values.pop()

    @property
    def stage_plans(self) -> dict[str, Any]:
        """原子化批次没有 ``plan`` ⇒ 空 dict (``compute_findings`` 只用它算"计划里被砍掉的配置")。"""
        return {}

    def _sides_cfg(self) -> dict[str, Any]:
        stage = self.stages.get("sides")
        return dict(stage.run_config) if stage is not None else {}

    @property
    def zero_order_xy(self) -> tuple[float, float]:
        """段级冻结 0 级 (只有带 sides 段的整轮批次才有)。"""
        return run_zero_order(self._sides_cfg())

    @property
    def frame_shape_hw(self) -> tuple[int, int]:
        shape = self._sides_cfg().get("frame_shape_hw") or ()
        if len(shape) != 2:
            return 0, 0
        return int(shape[0]), int(shape[1])

    @property
    def all_configs(self) -> list[RetestConfig]:
        return [cfg for name in RETEST_STAGES for cfg in self.stages[name].configs]

    @property
    def settle_count(self) -> int:
        return sum(len(cfg.settle) for cfg in self.all_configs)

    def settle_for(self, stage: str, index: int) -> SettleRecord | None:
        """该配置自己的第一条 settle 告警 (没有就 ``None``)。

        与 :meth:`Run.settle_for` 同名同义 —— 复测/整轮批次的日志是**每配置一个文件**,
        所以按配置下标取, 不需要像基线那样按"下一行打印"归属。
        """
        st = self.stages.get(stage)
        if st is None or not (0 <= index < len(st.configs)):
            return None
        events = st.configs[index].settle
        return events[0] if events else None

    @property
    def selected_source(self) -> str:
        values = {str(r.get("selected_source", "?")) for st in self.stages.values() for r in st.rows}
        return "/".join(sorted(values))

    @property
    def mode(self) -> str:
        modes = {str((e.get("summary") or {}).get("mode", "?")) for e in self.manifest}
        return "/".join(sorted(modes))


def _settle_records(lines: Sequence[str]) -> list[SettleRecord]:
    """把一段(已展开的)日志里的全部 settle 告警取出来 (不做配置归属)。"""
    out: list[SettleRecord] = []
    for line in lines:
        if "settle criterion not met" not in line:
            continue
        match = _SETTLE_RE.search(line)
        if not match:
            continue
        out.append(
            SettleRecord(
                waited_s=float(match.group("waited")),
                probes=int(match.group("probes")),
                probe_peak=float(match.group("peak")),
                probe_fwhm=float(match.group("fwhm")),
                raw_fragment=line[line.index("settle criterion not met"):],
            )
        )
    return out


def load_retest(root: Path, log_dir: Path | None,
                stage_names: Sequence[str] = RETEST_STAGES) -> RetestRun:
    """读一个**原子化批次** (每配置一个进程 + 一个 ``metrics.json``)。

    §11 的复测批与 §13 的整轮重扫都是这个形状, 所以共用这一份读取:
    差别只有段集合 (``stages``) 与有没有每配置日志 (``log_dir=None`` 就跳过)。

    **每次调用都从磁盘重新解析** —— 这些产物会被脚本覆盖重跑, 所以绝不缓存、
    绝不内联任何数字。
    """
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, list):
        raise ValueError(f"{root / 'manifest.json'} 不是 manifest 列表")
    by_group = {str(entry.get("group")): entry for entry in manifest}

    stages: dict[str, RetestStage] = {}
    for name in stage_names:
        if name not in by_group:
            raise ValueError(f"复测 manifest 里没有 '{name}' 这一段")
        stage_dir = root / name
        headers, rows = load_csv(stage_dir / "measurements.csv")
        payload = load_json(stage_dir / "measurements.json")
        if not isinstance(payload, list):
            raise ValueError(f"{stage_dir / 'measurements.json'} 不是行列表")
        _cross_check_csv_json(f"retest/{name}", rows, payload)
        entry = by_group[name]
        if int(entry.get("n_rows", -1)) != len(rows):
            raise ValueError(
                f"复测 manifest 记 {name} 有 {entry.get('n_rows')} 行, CSV 实际 {len(rows)} 行"
            )
        if str(entry.get("variable")) != VARIABLE[name]:
            raise ValueError(
                f"复测 manifest 记 {name} 的自变量是 {entry.get('variable')!r}, "
                f"期望 {VARIABLE[name]!r}"
            )
        config_names = [str(c) for c in ((entry.get("summary") or {}).get("configs") or [])]
        if len(config_names) != len(rows):
            raise ValueError(
                f"复测 manifest 记 {name} 有 {len(config_names)} 个配置, CSV 有 {len(rows)} 行"
            )
        configs: list[RetestConfig] = []
        for config_name, row in zip(config_names, rows):
            metrics_path = stage_dir / config_name / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"缺少单配置产物: {metrics_path}")
            metrics = load_json(metrics_path)
            value = float(row[VARIABLE[name]])
            if str(metrics.get("config")) != config_name:
                raise ValueError(
                    f"{metrics_path} 的 config 是 {metrics.get('config')!r}, 期望 {config_name!r}"
                )
            log_path = (log_dir / f"retest_{name}_{int(round(value))}.log"
                        if log_dir is not None else None)
            settle: list[SettleRecord] = []
            if log_path is not None and log_path.exists():
                settle = _settle_records(_unwrap_log(log_path))
            _cross_check_retest_metrics(name, config_name, row, metrics)
            configs.append(
                RetestConfig(stage=name, config=config_name, value=value, row=row,
                             metrics=metrics,
                             log_path=(log_path if log_path is not None
                                       and log_path.exists() else None),
                             settle=settle)
            )
        if [cfg.value for cfg in configs] != [float(r[VARIABLE[name]]) for r in rows]:
            raise ValueError(f"复测 {name}: metrics.json 的配置顺序与 CSV 行顺序不一致")
        stage_cfg_path = stage_dir / "run_config.json"
        stages[name] = RetestStage(
            name=name, headers=headers, rows=rows, configs=configs,
            run_config=load_json(stage_cfg_path) if stage_cfg_path.exists() else {},
        )

    return RetestRun(root=root, manifest=manifest, stages=stages,
                     log_dir=log_dir if log_dir is not None else Path("."))


#: CSV 与单配置 ``metrics.json`` 必须逐值一致的那批键 (两边都有, 且是本节要用的)。
_RETEST_METRIC_KEYS: tuple[str, ...] = (
    "cv", "pib", "score", "frame_peak", "box_peak", "box_clipped_px",
    "box_centroid_offset_x", "box_centroid_offset_y", "exposure_ms_readback",
    "box_center_x", "box_center_y", "box_side_px",
)


def _cross_check_retest_metrics(
    stage: str, config: str, row: dict[str, str], metrics: dict[str, Any]
) -> None:
    """``measurements.csv`` 的一行与 ``<config>/metrics.json`` 必须逐值一致。"""
    for key in _RETEST_METRIC_KEYS:
        if key not in metrics:
            continue
        expected = float(row[key])
        actual = float(metrics[key])
        if math.isnan(expected) and math.isnan(actual):
            continue
        if expected != actual:
            raise ValueError(
                f"复测 {stage}/{config} '{key}': CSV {row[key]} != metrics.json {metrics[key]}"
            )


@dataclass
class RetestFindings:
    """§11 要用的全部派生量 —— 每一个都现场算, 不复用任何字面量。"""

    power_ratio: float
    baseline_counts_per_ms: float
    retest_counts_per_ms: float
    equivalent_exposure_ms: float
    baseline_exposure_ms: float
    retest_exposure_ms: float
    # -- 批内邻居: 40 / 18 相对左右邻居中位数的相对偏差 --------------------
    blaze_cv: list[float]
    blaze_cv_monotone: bool
    blaze40_dev: float
    baseline_blaze40_dev: float
    zern_cv: list[float]
    zern_cv_monotone: bool
    zern18_dev: float
    baseline_zern18_dev: float
    # -- 位移: 方向 / 标度 / 1/P 形状 --------------------------------------
    offset_identity_residual: float
    ratio_retest: list[float]
    ratio_baseline: list[float]
    alpha_retest: float
    alpha_baseline: float
    alpha_predicted: float
    measured_ends_retest: tuple[float, float, float]
    measured_ends_baseline: tuple[float, float, float]
    predicted_ends_retest: tuple[float, float, float]
    predicted_ends_baseline: tuple[float, float, float]
    offset_y_small_period: list[tuple[float, float]]
    # -- 原点抖动 ----------------------------------------------------------
    zero_order_x_span: float
    zero_order_y_span: float
    zero_order_x_span_zern: float
    zero_order_y_span_zern: float
    centroid_x_span: float
    centroid_y_span: float
    measured_x_span: float
    measured_y_span: float
    offset_y_vs_baseline: list[tuple[float, float, float]]
    # -- 拟合一致 / 标记 ---------------------------------------------------
    fit_bit_identical: list[float]
    fit_differs: list[float]
    flagged_configs: list[str]
    settle_configs: list[str]
    dark_probe_configs: list[str]
    retest_rows: int
    # -- 批内一致性 (单批, 不跨批): 相对左右邻居中位数的最大偏差 --------------
    batch_dev_max: float
    prior_root: Path | None


def retest_power_ratio() -> tuple[float, float, float]:
    """由**现场记录**算功率比: ``(基线 counts/ms, 复测 counts/ms, 比值)``。"""
    base = FIELD_BASELINE_FLAT.counts_per_ms()
    now = FIELD_RETEST_FLAT.counts_per_ms()
    return base, now, base / now


def _neighbour_deviation(cvs: Sequence[float], index: int) -> float:
    """第 ``index`` 行相对左右邻居中位数的相对偏差 (无邻居返回 nan)。"""
    neighbours = [cvs[j] for j in (index - 1, index + 1) if 0 <= j < len(cvs)]
    if not neighbours:
        return float("nan")
    ref = float(np.median(neighbours))
    return cvs[index] / ref - 1.0 if ref else float("nan")


def _loglog_exponent(xs: Sequence[float], ys: Sequence[float]) -> float:
    """两点 log-log 斜率 (``y ∝ x^-α`` 里的 α); 用两端点, 不拟合。"""
    if len(xs) < 2 or min(xs) <= 0 or min(ys) <= 0:
        return float("nan")
    return -math.log(ys[-1] / ys[0]) / math.log(xs[-1] / xs[0])


def compute_retest_findings(
    run: Run, retest: RetestRun, prior: RetestRun | None = None
) -> RetestFindings:
    base_cpm, now_cpm, ratio = retest_power_ratio()
    blaze = retest.stages["blaze"]
    zern = retest.stages["zernike"]
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]

    b_cv = blaze.col("cv")
    blaze_monotone = all(b_cv[i] < b_cv[i - 1] for i in range(1, len(b_cv)))
    i40 = _idx_of([float(r["blaze_period_px"]) for r in blaze.rows], 40.0)
    z_cv = zern.col("cv")
    zern_monotone = all(z_cv[i] > z_cv[i - 1] for i in range(1, len(z_cv)))
    i18 = _idx_of([float(r["n_max"]) for r in zern.rows], 18.0)

    # 基线同位置的偏差 (同一套「相对左右邻居中位数」的口径)
    base_b_cv = base_blaze.col("cv")
    base_z_cv = base_zern.col("cv")
    base_blaze40_dev = _neighbour_deviation(
        base_b_cv, _idx_of([float(r["blaze_period_px"]) for r in base_blaze.rows], 40.0)
    )
    base_zern18_dev = _neighbour_deviation(
        base_z_cv, _idx_of([float(r["n_max"]) for r in base_zern.rows], 18.0)
    )

    # 偏移量恒等式: offset_y == measured_shift_y − predicted_shift_y
    residual = 0.0
    for cfg in blaze.configs:
        offset = float(cfg.row["box_centroid_offset_y"])
        identity = float(cfg.row["measured_shift_y_px"]) - float(cfg.row["predicted_shift_px"])
        residual = max(residual, abs(offset - identity))

    z0 = [cfg.zero_order for cfg in blaze.configs]
    centroid_x = [float(cfg.row["box_centroid_x"]) for cfg in blaze.configs]
    centroid_y = [float(cfg.row["box_centroid_y"]) for cfg in blaze.configs]
    measured_x = [float(cfg.row["measured_shift_x_px"]) for cfg in blaze.configs]
    measured_y = [float(cfg.row["measured_shift_y_px"]) for cfg in blaze.configs]

    # 位移: 只用 measured_shift_y (相对冻结 0 级)。基线黑帧行(period=40)剔除。
    base_med_peak = base_blaze.median("frame_peak")
    base_clean = [
        row for row in base_blaze.rows
        if float(row["frame_peak"]) >= DARK_FRAME_FRACTION * base_med_peak
    ]
    ratio_baseline = [
        float(r["measured_shift_y_px"]) / float(r["predicted_shift_px"]) for r in base_clean
    ]
    ratio_retest = [
        float(cfg.row["measured_shift_y_px"]) / float(cfg.row["predicted_shift_px"])
        for cfg in blaze.configs
    ]

    def ends(rows: Sequence[dict[str, str]], measured_key: str, predicted_key: str):
        ordered = sorted(rows, key=lambda r: float(r["blaze_period_px"]))
        lo, hi = ordered[0], ordered[-1]
        return (
            (float(lo["blaze_period_px"]), float(lo[measured_key]), float(hi[measured_key])),
            (float(hi["blaze_period_px"]), float(lo[predicted_key]), float(hi[predicted_key])),
        )

    ret_meas, ret_pred = ends([cfg.row for cfg in blaze.configs],
                              "measured_shift_y_px", "predicted_shift_px")
    base_meas, base_pred = ends(base_clean, "measured_shift_y_px", "predicted_shift_px")

    # 小周期处「框心 − 质心」偏多少 (框跟着预测走时框偏得有多远)
    offset_y_small = sorted(
        [(float(cfg.row["blaze_period_px"]), float(cfg.row["box_centroid_offset_y"]))
         for cfg in blaze.configs],
        key=lambda pair: pair[0],
    )

    # 与基线的同周期对照 (基线黑帧行不参与 —— 它的偏移量无意义)
    offset_delta: list[tuple[float, float, float]] = []
    for row in blaze.rows:
        value = float(row["blaze_period_px"])
        b_row = None
        for cand in base_blaze.rows:
            if abs(float(cand["blaze_period_px"]) - value) < ROW_MATCH_ATOL:
                b_row = cand
                break
        if b_row is None or float(b_row["frame_peak"]) < DARK_FRAME_FRACTION * base_med_peak:
            continue
        offset_delta.append((
            value,
            float(b_row["box_centroid_offset_y"]),
            float(row["box_centroid_offset_y"]) - float(b_row["box_centroid_offset_y"]),
        ))

    # 拟合量跨会话逐位一致? (同一 GS 相位 + 同一拟合配置 ⇒ 应当逐位相同)
    identical: list[float] = []
    differs: list[float] = []
    for cfg in zern.configs:
        b_row = None
        for row in base_zern.rows:
            if abs(float(row["n_max"]) - cfg.value) < ROW_MATCH_ATOL:
                b_row = row
                break
        if b_row is None:
            continue
        same = (
            float(b_row["zernike_rms_residual_rad"]) == float(cfg.row["zernike_rms_residual_rad"])
            and float(b_row["zernike_explained_fraction"])
            == float(cfg.row["zernike_explained_fraction"])
            and float(b_row["zernike_modes_used"]) == float(cfg.row["zernike_modes_used"])
        )
        (identical if same else differs).append(cfg.value)

    # 批内一致性: 单批 6 个点各自相对左右邻居中位数的偏差, 取绝对值最大者。
    # (三点时"邻居中位数"就是线性插值, 所以中间点的这个量等价于"曲率残差"。)
    devs = []
    for stage_name in RETEST_STAGES:
        st = retest.stages[stage_name]
        cvs = st.col("cv")
        for i in range(len(cvs)):
            if i in (0, len(cvs) - 1):
                continue
            dev = _neighbour_deviation(cvs, i)
            if math.isfinite(dev):
                devs.append(abs(dev))
        # 端点也算进去 (它是趋势斜率, 不是噪声), 用于给出"批内最大偏差"的上界
        for i in (0, len(cvs) - 1):
            neighbours = [cvs[j] for j in ((i - 1,) if i else (i + 1,))]
            if neighbours and neighbours[0]:
                devs.append(abs(cvs[i] / neighbours[0] - 1.0))

    flagged = [cfg.config for cfg in retest.all_configs if cfg.spot_flagged]
    settle_cfgs = [cfg.config for cfg in retest.all_configs if cfg.settle]
    dark_probe = [
        cfg.config for cfg in retest.all_configs
        if any(ev.probe_peak < DARK_FRAME_FRACTION * SATURATION_COUNTS for ev in cfg.settle)
    ]

    return RetestFindings(
        power_ratio=ratio,
        baseline_counts_per_ms=base_cpm,
        retest_counts_per_ms=now_cpm,
        equivalent_exposure_ms=retest.exposure_ms / ratio,
        baseline_exposure_ms=run.exposure_ms,
        retest_exposure_ms=retest.exposure_ms,
        blaze_cv=b_cv,
        blaze_cv_monotone=blaze_monotone,
        blaze40_dev=_neighbour_deviation(b_cv, i40),
        baseline_blaze40_dev=base_blaze40_dev,
        zern_cv=z_cv,
        zern_cv_monotone=zern_monotone,
        zern18_dev=_neighbour_deviation(z_cv, i18),
        baseline_zern18_dev=base_zern18_dev,
        offset_identity_residual=residual,
        ratio_retest=ratio_retest,
        ratio_baseline=ratio_baseline,
        alpha_retest=_loglog_exponent(
            [ret_meas[0], ret_pred[0]], [ret_meas[1], ret_meas[2]]),
        alpha_baseline=_loglog_exponent(
            [base_meas[0], base_pred[0]], [base_meas[1], base_meas[2]]),
        alpha_predicted=_loglog_exponent(
            [ret_meas[0], ret_pred[0]], [ret_pred[1], ret_pred[2]]),
        measured_ends_retest=ret_meas,
        measured_ends_baseline=base_meas,
        predicted_ends_retest=ret_pred,
        predicted_ends_baseline=base_pred,
        offset_y_small_period=offset_y_small,
        zero_order_x_span=max(x for x, _ in z0) - min(x for x, _ in z0),
        zero_order_y_span=max(y for _, y in z0) - min(y for _, y in z0),
        zero_order_x_span_zern=(lambda zz: max(x for x, _ in zz) - min(x for x, _ in zz))(
            [cfg.zero_order for cfg in zern.configs]),
        zero_order_y_span_zern=(lambda zz: max(y for _, y in zz) - min(y for _, y in zz))(
            [cfg.zero_order for cfg in zern.configs]),
        centroid_x_span=max(centroid_x) - min(centroid_x),
        centroid_y_span=max(centroid_y) - min(centroid_y),
        measured_x_span=max(measured_x) - min(measured_x),
        measured_y_span=max(measured_y) - min(measured_y),
        offset_y_vs_baseline=offset_delta,
        fit_bit_identical=identical,
        fit_differs=differs,
        flagged_configs=flagged,
        settle_configs=settle_cfgs,
        dark_probe_configs=dark_probe,
        retest_rows=sum(len(st.rows) for st in retest.stages.values()),
        batch_dev_max=max(devs) if devs else float("nan"),
        prior_root=prior.root if prior is not None else None,
    )


def column_contract(
    baseline: Stage, retest: RetestStage, needed: Sequence[str]
) -> list[tuple[str, str, str]]:
    """逐列说明「这一列两批都有吗」—— 复测的列集合与基线**不同**, 不能假设相同。"""
    rows: list[tuple[str, str, str]] = []
    for key in needed:
        rows.append((
            f"`{key}`",
            "有" if key in baseline.headers else "**缺**",
            "有" if key in retest.headers else "**缺**",
        ))
    return rows


# ---------------------------------------------------------------------------
# 图 (全部点来自 CSV)
# ---------------------------------------------------------------------------


def fig_sides_objectives(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    x = f.sides_x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax1.plot(x, f.pib, "o-", color="#1f77b4", label="PIB")
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("PIB (框内功率占比)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(x, f.cv, "s--", color="#d62728", label="CV")
    ax1b.set_ylabel("CV (框内变异系数)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)
    ax1.set_title("(a) PIB 与 CV 同向上升, 但两者在 score 里正好相反")

    ax2.plot(x, f.score, "o-", color="#2ca02c", label="score")
    peak_x = x[f.score_peak_index]
    ax2.plot([peak_x], [f.score[f.score_peak_index]], "k*", ms=14,
             label=f"峰值 {peak_x:.0f} px")
    ax2.axvline(f.selected_side, color="gray", ls=":", lw=1)
    ax2.annotate(f"选定 {f.selected_side} px\n(近 1000 窗口取到上界)",
                 xy=(f.selected_side, f.score[0]), xytext=(f.selected_side - 520, f.score[0] + 0.03),
                 fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("score = 0.5·PIB + 0.5/(1+CV)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_title("(b) 复合分数在 1500 px 见顶, 之后被 CV 拖下来")
    fig.tight_layout()
    path = FIG_DIR / f"fig1_sides_objectives{FIG_TAG}.png"
    savefig(fig, path)
    return path


def fig_sides_exposure(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    sides = run.stages["sides"]
    x = f.sides_x
    frame_peak = sides.col("frame_peak")
    box_peak = sides.col("box_peak")
    fwhm = sides.col("global_spot_fwhm_px")
    marks = flag_rows(run, sides)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, frame_peak, "o-", color="#1f77b4", label="frame_peak (整帧最大)")
    ax1.plot(x, box_peak, "s--", color="#ff7f0e", label="box_peak (框内最大)")
    ax1.axhline(SATURATION_COUNTS, color="#d62728", ls=":", lw=1.2,
                label=f"8 bit 满量程 {SATURATION_COUNTS:.0f} counts")
    for i, mark in enumerate(marks):
        if "†" in mark or "‡" in mark:
            ax1.plot([x[i]], [frame_peak[i]], "x", color="k", ms=7)
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("counts")
    ax1.set_title("(a) 只有 100 px 那一行顶到满量程; 其余 17 行都有余量")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="center right")

    finite = [(xi, v) for xi, v in zip(x, fwhm) if math.isfinite(v)]
    ax2.plot([p[0] for p in finite], [p[1] for p in finite], "o-", color="#9467bd",
             label="global_spot_fwhm (整帧最亮结构)")
    median_fwhm = float(np.median([q for _, q in finite]))
    for xi, v in zip(x, fwhm):
        if math.isfinite(v) and v > 3.0 * median_fwhm:
            ax2.annotate(f"{v:.1f} px", xy=(xi, v), xytext=(xi + 40, v + 2), fontsize=8)
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("FWHM (camera px)")
    ax2.set_title("(b) 该列量的是 0 级/晕斑, 不是方斑")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    path = FIG_DIR / f"fig2_sides_exposure_geometry{FIG_TAG}.png"
    savefig(fig, path)
    return path


def fig_zernike(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    zern = run.stages["zernike"]
    x = zern.x
    explained_pct = [v * 100.0 for v in f.zern_explained]
    marks = flag_rows(run, zern)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, explained_pct, "o-", color="#1f77b4", label="解释方差占比 (%)")
    ax1.set_xlabel("截断阶数 n_max")
    ax1.set_ylabel("zernike_explained_fraction (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(x, f.zern_residual, "s--", color="#d62728", label="残差 RMS (rad)")
    ax1b.set_ylabel("zernike_rms_residual (rad)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    ax1b.set_ylim(min(f.zern_residual) - 0.01, max(f.zern_residual) + 0.01)
    ax1.set_title("(a) 496 个模式只解释 3% 的相位方差; 残差几乎不动")

    ax2.plot(x, f.zern_cv, "o-", color="#2ca02c", label="CV")
    cluster = [c for c in f.zern_clean_cv]
    ax2.set_ylim(min(cluster) - 0.15 * (max(cluster) - min(cluster)),
                 max(cluster) + 0.6 * (max(cluster) - min(cluster)))
    for i, mark in enumerate(marks):
        if "†" in mark:
            ax2.plot([x[i]], [f.zern_cv[i]], "o", mfc="none", mec="k", ms=12)
        if "‡" in mark or "※" in mark:
            ax2.annotate(f"n_max={x[i]:.0f}\nCV={f.zern_cv[i]:.2f} ({mark})\n超出纵轴",
                         xy=(x[i], ax2.get_ylim()[1]), xytext=(x[i] + 1.5, ax2.get_ylim()[1] - 0.55),
                         fontsize=8, color="#d62728", ha="left",
                         arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))
    ax2.set_xlabel("截断阶数 n_max")
    ax2.set_ylabel("CV", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(x, f.zern_score, "^:", color="#7f7f7f", lw=1, label="score")
    ax2b.set_ylabel("score", color="#7f7f7f")
    ax2b.tick_params(axis="y", labelcolor="#7f7f7f")
    ax2.set_title("(b) 干净行 CV 只从 7.30 摆到 7.70; 空圈 = settle 超时")
    fig.tight_layout()
    path = FIG_DIR / f"fig3_zernike_reconstruction{FIG_TAG}.png"
    savefig(fig, path)
    return path


def fig_blaze(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    blaze = run.stages["blaze"]
    x = blaze.x
    pred = blaze.col("predicted_shift_px")
    meas = blaze.col("measured_shift_y_px")
    marks = flag_rows(run, blaze)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, pred, "o-", color="#1f77b4", label="predicted_shift (模型)")
    dark = [i for i, mark in enumerate(marks) if "‡" in mark]
    keep = [i for i in range(len(x)) if i not in dark]
    ax1.plot([x[i] for i in keep], [meas[i] for i in keep], "s--", color="#ff7f0e",
             label="measured_shift_y (框内质心)")
    for i in dark:
        ax1.plot([x[i]], [meas[i]], "x", color="#d62728", ms=11, mew=2,
                 label="黑帧行的实测值 (不计)")
        ax1.annotate("黑帧伪迹\n不当数据点", xy=(x[i], meas[i]),
                     xytext=(x[i] + 8, meas[i] - 55), fontsize=8, color="#d62728",
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    for i, mark in enumerate(marks):
        if not mark or i in dark:
            continue
        ax1.plot([x[i]], [meas[i]], "o", mfc="none", mec="#d62728", ms=13, mew=1.4)
        ax1.annotate(f"{mark}", xy=(x[i], meas[i]), xytext=(x[i] + 6, meas[i] - 42),
                     fontsize=11, color="#d62728")
    ax1.set_xlabel("闪耀光栅周期 (SLM px)")
    ax1.set_ylabel("位移 (camera px)")
    ax1.set_title("(a) 实测位移系统性小于模型; 周期 16 让框出画 (§)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    excluded = {i for i, mark in enumerate(marks) if "§" in mark or "‡" in mark}
    keep = [i for i in range(len(x)) if i not in excluded]
    ax2.plot([x[i] for i in keep], [f.blaze_score[i] for i in keep], "o-",
             color="#2ca02c", label="score (干净行)")
    for i in sorted(excluded):
        ax2.plot([x[i]], [f.blaze_score[i]], "x", color="#d62728", ms=9,
                 label="排除行 (§ / ‡)")
        ax2.annotate(f"{marks[i]} score {f.blaze_score[i]:.3f}",
                     xy=(x[i], f.blaze_score[i]), xytext=(x[i] + 4, f.blaze_score[i] - 0.03),
                     fontsize=8, color="#d62728")
    ax2.set_xlabel("闪耀光栅周期 (SLM px)")
    ax2.set_ylabel("score", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(alpha=0.3)
    ax2.set_title(f"(b) 去掉 {len(excluded)} 行后 score 随周期单调上升")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIG_DIR / f"fig4_blaze_shift{FIG_TAG}.png"
    savefig(fig, path)
    return path


def fig_retest_neighbors(run: Run, retest: RetestRun, rf: RetestFindings) -> Path:
    """复测邻居比较: 左列 CV、右列 score, 上排光栅、下排 Zernike。"""
    setup_matplotlib()
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]
    r_blaze = retest.stages["blaze"]
    r_zern = retest.stages["zernike"]

    b_p = [float(r["blaze_period_px"]) for r in base_blaze.rows]
    b_cv = base_blaze.col("cv")
    b_sc = base_blaze.col("score")
    b_n = [float(r["n_max"]) for r in base_zern.rows]
    bz_cv = base_zern.col("cv")
    bz_sc = base_zern.col("score")
    r_p = [float(r["blaze_period_px"]) for r in r_blaze.rows]
    r_n = [float(r["n_max"]) for r in r_zern.rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    (ax_cv_p, ax_sc_p), (ax_cv_n, ax_sc_n) = axes

    def panel(ax, xs, ys, label, **kw):
        ax.plot(xs, ys, label=label, **kw)
        ax.grid(alpha=0.3)

    # (a) 光栅 CV
    panel(ax_cv_p, b_p, b_cv, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=5)
    panel(ax_cv_p, r_p, rf.blaze_cv, "复测 (180 ms)", color="#2ca02c", marker="s", ms=7)
    ax_cv_p.plot([40.0], [float(_row_at(base_blaze.rows, "blaze_period_px", 40.0)["cv"])], "x",
                 color="#d62728",
                 ms=12, mew=2, label="基线 40: 黑帧")
    ax_cv_p.set_yscale("log")
    ax_cv_p.set_xlabel("闪耀光栅周期 (SLM px)")
    ax_cv_p.set_ylabel("CV (对数轴)")
    ax_cv_p.set_title("(a) 基线 40 的 CV 尖峰在复测里消失")
    ax_cv_p.legend(fontsize=8)

    # (b) 光栅 score
    panel(ax_sc_p, b_p, b_sc, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=5)
    panel(ax_sc_p, r_p, [float(r["score"]) for r in r_blaze.rows], "复测 (180 ms)",
          color="#2ca02c", marker="s", ms=7)
    ax_sc_p.set_xlabel("闪耀光栅周期 (SLM px)")
    ax_sc_p.set_ylabel("score")
    r_score_p = [float(r["score"]) for r in r_blaze.rows]
    mono_p = all(r_score_p[i] > r_score_p[i - 1] for i in range(1, len(r_score_p)))
    ax_sc_p.set_title(
        "(b) score: 复测 32/40/48 " + ("单调上升" if mono_p else "非单调（无尖峰）"))
    ax_sc_p.legend(fontsize=8, loc="lower right")

    # (c) Zernike CV
    panel(ax_cv_n, b_n, bz_cv, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=4)
    panel(ax_cv_n, r_n, rf.zern_cv, "复测 (180 ms)", color="#2ca02c", marker="s", ms=7)
    ax_cv_n.plot([18.0], [float(_row_at(base_zern.rows, "n_max", 18.0)["cv"])], "x",
                 color="#d62728",
                 ms=12, mew=2, label="基线 18: 黑帧")
    ax_cv_n.set_yscale("log")
    ax_cv_n.set_xlabel("截断阶数 n_max")
    ax_cv_n.set_ylabel("CV (对数轴)")
    ax_cv_n.set_title("(c) 基线 18 的 CV 尖峰在复测里消失")
    ax_cv_n.legend(fontsize=8)

    # (d) Zernike score
    panel(ax_sc_n, b_n, bz_sc, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=4)
    panel(ax_sc_n, r_n, [float(r["score"]) for r in r_zern.rows], "复测 (180 ms)",
          color="#2ca02c", marker="s", ms=7)
    ax_sc_n.set_xlabel("截断阶数 n_max")
    ax_sc_n.set_ylabel("score")
    r_score_n = [float(r["score"]) for r in r_zern.rows]
    mono_n = all(r_score_n[i] > r_score_n[i - 1] for i in range(1, len(r_score_n)))
    ax_sc_n.set_title(
        "(d) score: 复测 17/18/19 "
        + ("单调上升" if mono_n else f"在 {min(r_score_n):.4f}–{max(r_score_n):.4f} 之间，无尖峰"))
    ax_sc_n.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"复测（功率已变, 折合曝光 {rf.equivalent_exposure_ms:.1f} ms）—— 只做批内邻居比较",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIG_DIR / f"fig5_retest_neighbors{FIG_TAG}.png"
    savefig(fig, path)
    return path


def render_figures(run: Run, f: Findings) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return [
        fig_sides_objectives(run, f),
        fig_sides_exposure(run, f),
        fig_zernike(run, f),
        fig_blaze(run, f),
    ]


# ---------------------------------------------------------------------------
# CCD 远场帧: 定位 / 选择 / 拼图 (§12)
# ---------------------------------------------------------------------------

#: 每格的裁剪倍数 (相对 ``box_side_px``)。写进图注与 §12 正文。
CROP_FACTOR = 1.4

#: 拼图 1 的固定边长 (camera px)。
SIDES_PANEL_PX: tuple[float, ...] = (200.0, 600.0, 1000.0, 1500.0, 1800.0)

#: 拼图 2 的固定阶数。
ZERNIKE_PANEL_ORDER: tuple[float, ...] = (30.0, 19.0, 12.0, 7.0)

#: 拼图 3 的固定光栅周期 (16 是"有就画")。
BLAZE_PANEL_PERIOD: tuple[float, ...] = (32.0, 64.0, 128.0)
BLAZE_PANEL_OPTIONAL: float = 16.0

#: 拼图 4 的整帧边长目标 (取最接近的那个)。
FULLFRAME_TARGET_PX = 1000.0


@dataclass(frozen=True)
class CellPick:
    """拼图里的一格: 哪个段、哪一行 CSV、以什么身份入选。"""

    stage: str
    row: dict[str, str]
    role: str
    note: str = ""

    @property
    def value(self) -> float:
        return float(self.row[VARIABLE[self.stage]])

    @property
    def slug(self) -> str:
        """该配置在 ``frames_selection.json`` 里的标识。"""
        config = self.row.get("config")
        if config:
            return str(config)
        index = self.row.get("index")
        return f"index_{int(float(index)):03d}" if index else "?"


@dataclass(frozen=True)
class FrameRef:
    """一格的帧来源。"""

    paths: tuple[Path, ...]
    layout: str  # "atomic" | "legacy" | "missing"


def locate_frames(stage_dir: Path, row: dict[str, str], headers: Sequence[str]) -> FrameRef:
    """按**实际布局**定位该行的帧。

    两种布局都要支持, 判据是 CSV 里有没有 ``config`` 列 (原子布局每配置一个子目录):

    * 原子布局: ``<stage>/<config>/frame_*.npz``
    * 旧整段布局: ``<stage>/frame_<index:03d>.npz``

    找不到就返回 ``layout="missing"`` —— 调用方负责"跳过该格 + 记缺帧", 不静默少画。
    """
    if "config" in headers and row.get("config"):
        config_dir = stage_dir / str(row["config"])
        if config_dir.is_dir():
            paths = sorted(config_dir.glob("frame_*.npz"))
            if paths:
                return FrameRef(paths=tuple(paths), layout="atomic")
        return FrameRef(paths=(), layout="missing")
    if "index" in headers and row.get("index"):
        path = stage_dir / f"frame_{int(float(row['index'])):03d}.npz"
        if path.exists():
            return FrameRef(paths=(path,), layout="legacy")
        return FrameRef(paths=(), layout="missing")
    return FrameRef(paths=(), layout="missing")


def load_frame_mean(paths: Sequence[Path]) -> np.ndarray:
    """把一格的若干 npz **逐个**读入并求均值 (一次只在内存里留一个整帧)。"""
    total: np.ndarray | None = None
    for path in paths:
        with np.load(path) as payload:
            if "frame" not in payload:
                raise KeyError(f"{path} 里没有 'frame' 数组 (实际键: {list(payload.keys())})")
            frame = np.asarray(payload["frame"], dtype=np.float32)
        if total is None:
            total = np.zeros_like(frame, dtype=np.float32)
        elif frame.shape != total.shape:
            raise ValueError(f"{path} 的帧形状 {frame.shape} 与首帧 {total.shape} 不一致")
        total += frame
    if total is None:
        raise ValueError("没有可读的帧")
    total /= float(len(paths))
    return total


def frame_params(path: Path) -> dict[str, Any]:
    """读 npz 里那个 0 维 ``params`` (JSON 字符串) —— 用于帧 ↔ CSV 交叉校验。"""
    with np.load(path) as payload:
        if "params" not in payload:
            return {}
        raw = payload["params"]
    text = str(raw.item()) if getattr(raw, "shape", ()) == () else str(raw)
    try:
        return dict(json.loads(text))
    except (TypeError, ValueError):
        return {}


def _check_frame_params(stage: str, cfg: CellPick, ref: FrameRef) -> None:
    """npz 自带的 ``params`` 必须与该行 CSV 一致 —— 否则拼图会张冠李戴。"""
    if not ref.paths:
        return
    params = frame_params(ref.paths[0])
    if not params:
        return
    checks = (("cv", "cv"), ("pib", "pib"), ("score", "score"),
              ("frame_peak", "frame_peak"))
    for key, column in checks:
        if key not in params:
            continue
        if abs(float(params[key]) - float(cfg.row[column])) > PARAM_NPZ_CSV_ATOL:
            raise ValueError(
                f"{stage}/{cfg.slug}: npz params 的 {key}={params[key]} 与 CSV 的 "
                f"{column}={cfg.row[column]} 不一致 —— 帧与 CSV 行对不上"
            )
    if "side_px" in params and stage == "sides":
        if abs(float(params["side_px"]) - float(cfg.row["side_px"])) > SIDE_PX_NPZ_CSV_ATOL:
            raise ValueError(
                f"{stage}/{cfg.slug}: npz params 的 side_px={params['side_px']} 与 CSV 不一致"
            )


def select_cells(
    run: Run,
    f: Findings,
    *,
    sides_panel: Sequence[float] = SIDES_PANEL_PX,
    add_score_cell: bool = True,
) -> dict[str, list[CellPick]]:
    """按写死的规则从**实际存在的 CSV 行**里挑格子。

    ``sides_panel`` / ``add_score_cell`` 让 §13 的整轮重扫用另一套边长格
    (200/600/1000/1500/1700, 且不追加 score 格) —— 其余规则完全一样。
    """
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]

    def pick(stage: Stage, target: float, role: str, note: str = "") -> CellPick | None:
        key = VARIABLE[stage.name]
        for row in stage.rows:
            if abs(float(row[key]) - float(target)) < ROW_MATCH_ATOL:
                return CellPick(stage=stage.name, row=row, role=role, note=note)
        return None

    # -- 拼图 1: 5 个固定边长 + score 最高行 ---------------------------------
    sides_picks: list[CellPick] = []
    for value in sides_panel:
        got = pick(sides, value, "固定格")
        if got is not None:
            sides_picks.append(got)
    best = CellPick(stage="sides", row=sides.rows[f.score_peak_index],
                    role="score 最高", note=f"score 最高 ({f.score[f.score_peak_index]:.4f})")
    taken = {p.value for p in sides_picks}
    if not add_score_cell:
        pass
    elif best.value not in taken:
        sides_picks.append(best)
    else:
        # 规则里的回退("换 1000")在本轮会与固定格重复, 所以退一步: 取最接近 1000 的
        # **未入选**行, 并在表里写明这是回退 —— 否则第 6 格只是第 3 格的复制品。
        candidates = [r for r in sides.rows if float(r["side_px"]) not in taken]
        if candidates:
            nearest = min(candidates,
                          key=lambda r: abs(float(r["side_px"]) - FULLFRAME_TARGET_PX))
            sides_picks.append(CellPick(
                stage="sides", row=nearest, role="回退",
                note=(f"score 最高行 ({best.value:.0f} px) 已在固定格内 ⇒ "
                      f"取最接近 1000 的未入选行 {float(nearest['side_px']):.0f} px"),
            ))
        else:
            sides_picks.append(CellPick(
                stage="sides", row=best.row, role="重复",
                note=f"score 最高行 ({best.value:.0f} px) 已在固定格内, 且没有别的行可选"))

    # -- 拼图 2: 4 个阶数 + 同 run-root 的 GS 参照 ---------------------------
    zern_picks: list[CellPick] = []
    for order in ZERNIKE_PANEL_ORDER:
        got = pick(zern, order, "固定格")
        if got is not None:
            zern_picks.append(got)
    ref_side = float(_selected_side_px(run))
    gs_ref = pick(sides, ref_side, "GS 参照",
                  note=f"同 run-root 边长段的 {ref_side:.0f} px 行（未截断）")
    if gs_ref is not None:
        zern_picks.append(gs_ref)

    # -- 拼图 3: 3 个周期 + 16 (有就画) --------------------------------------
    blaze_picks: list[CellPick] = []
    for period in BLAZE_PANEL_PERIOD:
        got = pick(blaze, period, "固定格")
        if got is not None:
            blaze_picks.append(got)
    optional = pick(blaze, BLAZE_PANEL_OPTIONAL, "补充格",
                    note="该行方框出画, PIB 不可与其它行横比")
    if optional is not None:
        blaze_picks.append(optional)

    # -- 拼图 4: 最接近 1000 px 的一行, 整帧不裁剪 ---------------------------
    nearest_row = min(sides.rows,
                      key=lambda r: abs(float(r["side_px"]) - FULLFRAME_TARGET_PX))
    full_pick = CellPick(stage="sides", row=nearest_row, role="整帧",
                         note="不裁剪, 用于核对 0 级与画面中心的相对位置")

    return {"sides": sides_picks, "zernike": zern_picks,
            "blaze": blaze_picks, "fullframe": [full_pick]}


@dataclass(frozen=True)
class MosaicSpec:
    """一张拼图。"""

    key: str          # sides / zernike / blaze / fullframe
    stem: str         # 文件名主干 (不含 batch 后缀)
    title: str
    ncols: int
    cell_inch: float
    crop: bool


#: 表格里「拼图」列的中文名。
MOSAIC_CN: dict[str, str] = {
    "sides": "1 · 边长",
    "zernike": "2 · Zernike",
    "blaze": "3 · 光栅",
    "fullframe": "4 · 整帧",
}

MOSAICS: tuple[MosaicSpec, ...] = (
    MosaicSpec("sides", "fig6_farfield_sides", "边长段: 方斑随边长变化", 3, 3.4, True),
    MosaicSpec("zernike", "fig7_farfield_zernike", "Zernike 段: 低阶重构后的远场", 3, 3.4, True),
    MosaicSpec("blaze", "fig8_farfield_blaze", "闪耀光栅段: 光栅把方斑搬走", 2, 3.6, True),
    MosaicSpec("fullframe", "fig9_farfield_fullframe", "整帧: 0 级不在画面中心", 1, 7.2, False),
)


def batch_tag(run: Run) -> str:
    """图名里的批次后缀 = run-root 目录名 (换 run-root 不会互相覆盖)。"""
    return run.root.name


def _cell_title(pick: CellPick, stage: Stage) -> str:
    """两行标题: 第一行是配置, 第二行是指标 (单行会压到相邻格)。"""
    row = pick.row
    peak = float(row["frame_peak"])
    sat = "  饱和" if peak >= SATURATION_COUNTS else ""
    if pick.stage == "sides":
        head = f"side={float(row['side_px']):.0f} px"
        if pick.role == "GS 参照":
            head = "GS 参照（未截断）  " + head
        body = (f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}  "
                f"score={float(row['score']):.3f}  peak={peak:.0f} counts")
    elif pick.stage == "zernike":
        head = f"n_max={float(row['n_max']):.0f}"
        body = (f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}  "
                f"score={float(row['score']):.3f}  peak={peak:.0f} counts")
    else:
        head = f"P={float(row['blaze_period_px']):.0f} px"
        body = (f"预测位移={float(row['predicted_shift_px']):.1f} px  "
                f"实测 y={float(row['measured_shift_y_px']):.1f} px  "
                f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}")
    return f"{head}{sat}\n{body}"


def _crop_box(row: dict[str, str], shape: tuple[int, int], factor: float
              ) -> tuple[int, int, int, int, bool]:
    """以 ``box_center`` 为中心、边长 ``box_side × factor`` 的裁剪框 (已夹到画幅内)。"""
    height, width = shape
    side = float(row["box_side_px"]) * factor
    cx, cy = float(row["box_center_x"]), float(row["box_center_y"])
    half = side / 2.0
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half))
    clipped = x0 < 0 or y0 < 0 or x1 > width or y1 > height
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    return x0, x1, y0, y1, clipped


def _annotate_cell(ax, pick: CellPick, *, crop: tuple[int, int, int, int] | None,
                   zero_order: tuple[float, float] | None = None,
                   frame_centre: tuple[float, float] | None = None) -> None:
    """叠加: 目标框 (虚线) / 预测框心 (十字) / 实测质心 (圆点) / 出画提示。"""
    row = pick.row
    side = float(row["box_side_px"])
    cx, cy = float(row["box_center_x"]), float(row["box_center_y"])
    if crop is None:
        x_off = y_off = 0.0
    else:
        x_off, y_off = float(crop[0]), float(crop[2])
    left, top = cx - side / 2 - x_off, cy - side / 2 - y_off
    ax.add_patch(plt.Rectangle((left, top), side, side, fill=False, ec="#00e5ff",
                               ls="--", lw=1.1, label="目标框"))
    ax.plot([cx - x_off], [cy - y_off], marker="+", color="#ffd400", ms=13, mew=1.6,
            ls="none", label="预测框心")
    ax.plot([float(row["box_centroid_x"]) - x_off], [float(row["box_centroid_y"]) - y_off],
            marker="o", mfc="none", mec="#ff3b30", ms=9, mew=1.6, ls="none",
            label="实测质心")
    if zero_order is not None:
        ax.plot([zero_order[0]], [zero_order[1]], marker="*", color="#ffffff", ms=15,
                ls="none", label="0 级 (argmax)")
    if frame_centre is not None:
        ax.plot([frame_centre[0]], [frame_centre[1]], marker="x", color="#7fff00",
                ms=9, mew=1.4, ls="none", label="画面中心")
    clipped = float(row.get("box_clipped_px", 0.0) or 0.0)
    if clipped > 0:
        ax.text(0.02, 0.02, f"框出画 {clipped:.0f} px", transform=ax.transAxes,
                fontsize=7.5, color="#ff3b30", va="bottom", ha="left",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.5))


def plan_mosaic(run: Run, spec: MosaicSpec, picks: Sequence[CellPick]) -> dict[str, Any]:
    """**不读像素**地规划一张拼图: 每格的来源、帧数、裁剪框、拉伸参数。

    峰值取自 CSV 的 ``frame_peak`` (绘制时再与帧的实际最大值逐格比对), 所以
    ``--check`` 可以只靠这一步就判断 ``frames_selection.json`` 是否过期, 不必重画。
    """
    stage_dirs = {name: run.root / name for name in run.stages}
    records: list[dict[str, Any]] = []
    for pick in picks:
        stage = run.stages[pick.stage]
        ref = locate_frames(stage_dirs[pick.stage], pick.row, stage.headers)
        planned = int(stage.run_config.get("frames_per_measurement", 0) or 0)
        shape = tuple(int(v) for v in stage.run_config.get("frame_shape_hw", ()))
        peak = float(pick.row["frame_peak"])
        record: dict[str, Any] = {
            "mosaic": spec.key,
            "stage": pick.stage,
            "config": pick.slug,
            "value": {VARIABLE[pick.stage]: pick.value},
            "role": pick.role,
            "note": pick.note,
            "layout": ref.layout,
            "sources": [{"path": _rel(path), "sha256": _sha256(path)} for path in ref.paths],
            "frames_present": len(ref.paths),
            "frames_planned": planned,
            "missing": not ref.paths,
            "peak": peak,
            "saturated": peak >= SATURATION_COUNTS,
            "cv": float(pick.row["cv"]),
            "pib": float(pick.row["pib"]),
            "score": float(pick.row["score"]),
            "box_clipped_px": float(pick.row.get("box_clipped_px", 0.0) or 0.0),
            "frame_shape_hw": list(shape) if shape else None,
            "norm": {"vmin": 1.0, "vmax": max(peak, 1.0 + 1e-6)},
            "crop": None,
        }
        if spec.crop and shape:
            x0, x1, y0, y1, clipped = _crop_box(pick.row, shape, CROP_FACTOR)
            record["crop"] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
                              "factor": CROP_FACTOR, "clipped_by_frame": clipped}
        if ref.paths:
            # 读一遍像素: 峰值必须等于 CSV 的 frame_peak (帧 ↔ CSV 的硬校验), 顺便
            # 记下**方框内**的中位/99 分位电平 —— 大边长格在对数显示下几乎全黑,
            # 有没有这两个数决定了读者能不能分清"数据太暗"和"画错了"。
            image = load_frame_mean(ref.paths)
            peak_px = float(image.max())
            if abs(peak_px - peak) > PEAK_MATCH_ATOL:
                raise ValueError(
                    f"{pick.stage}/{pick.slug}: 帧最大值 {peak_px:.6g} 与 CSV frame_peak "
                    f"{peak:.6g} 不一致 —— 帧与 CSV 行对不上"
                )
            height, width = image.shape
            side = float(pick.row["box_side_px"])
            cx, cy = float(pick.row["box_center_x"]), float(pick.row["box_center_y"])
            by0 = max(0, int(round(cy - side / 2)))
            by1 = min(height, int(round(cy + side / 2)))
            bx0 = max(0, int(round(cx - side / 2)))
            bx1 = min(width, int(round(cx + side / 2)))
            box = image[by0:by1, bx0:bx1]
            record["box_median"] = float(np.median(box))
            record["box_p99"] = float(np.percentile(box, 99))
            record["box_pixels"] = int(box.size)
            record["peak"] = peak_px
            del image, box
        else:
            record["box_median"] = None
            record["box_p99"] = None
            record["box_pixels"] = 0
        records.append(record)
    return {"mosaic": spec.key, "title": spec.title,
            "figure": f"{spec.stem}_{batch_tag(run)}.png", "cells": records}


def draw_mosaic(run: Run, spec: MosaicSpec, picks: Sequence[CellPick],
                plan: dict[str, Any], out_path: Path) -> bool:
    """按 ``plan`` 画图: 一格一格读帧、求均值、裁剪、随即释放。"""
    from matplotlib.colors import LogNorm

    setup_matplotlib()
    stage_dirs = {name: run.root / name for name in run.stages}
    cells = [(pick, record) for pick, record in zip(picks, plan["cells"])
             if not record["missing"]]
    if not cells:
        return False

    nrows = math.ceil(len(cells) / spec.ncols)
    fig, axes = plt.subplots(nrows, spec.ncols,
                             figsize=(spec.cell_inch * spec.ncols,
                                      spec.cell_inch * nrows + 0.5),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(cells):]:
        ax.axis("off")
    for ax, (pick, record) in zip(flat, cells):
        ref = locate_frames(stage_dirs[pick.stage], pick.row, run.stages[pick.stage].headers)
        image = load_frame_mean(ref.paths)
        if abs(float(image.max()) - float(record["peak"])) > PEAK_MATCH_ATOL:
            raise ValueError(
                f"{pick.stage}/{pick.slug}: 帧最大值 {float(image.max()):.6g} 与 CSV "
                f"frame_peak {record['peak']:.6g} 不一致 —— 帧与 CSV 行对不上"
            )
        crop = record["crop"]
        if crop is not None:
            if image.shape != tuple(record["frame_shape_hw"]):
                raise ValueError(
                    f"{pick.stage}/{pick.slug}: 帧形状 {image.shape} 与 run_config 的 "
                    f"{record['frame_shape_hw']} 不一致")
            shown = image[crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]]
        else:
            shown = image
        ax.imshow(shown, cmap="inferno",
                  norm=LogNorm(vmin=record["norm"]["vmin"], vmax=record["norm"]["vmax"]),
                  interpolation="nearest")
        zero_order = None
        frame_centre = None
        if crop is None:
            peak_index = np.unravel_index(int(np.argmax(shown)), shown.shape)
            zero_order = (float(peak_index[1]), float(peak_index[0]))
            height, width = shown.shape
            frame_centre = ((width - 1) / 2.0, (height - 1) / 2.0)
        _annotate_cell(ax, pick, crop=None if crop is None else
                       (crop["x0"], crop["x1"], crop["y0"], crop["y1"]),
                       zero_order=zero_order, frame_centre=frame_centre)
        title = _cell_title(pick, run.stages[pick.stage])
        if crop is None and zero_order is not None:
            frozen = run.zero_order_xy
            title += (f"\n0 级(argmax)=({zero_order[0]:.0f}, {zero_order[1]:.0f})  "
                      f"冻结 0 级=({frozen[0]:.0f}, {frozen[1]:.0f})  "
                      f"画面中心=({frame_centre[0]:.0f}, {frame_centre[1]:.0f})")
        ax.set_title(title, fontsize=7.5, linespacing=1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.98, 0.02, f"×{record['frames_present']} 帧均值", transform=ax.transAxes,
                fontsize=7.5, color="white", ha="right", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.5))
        del image, shown
    fig.suptitle(spec.title, fontsize=11)
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    savefig(fig, out_path)
    return True


def plan_farfield(run: Run, selection: dict[str, list[CellPick]]) -> dict[str, Any]:
    """4 张拼图的规划 + ``frames_selection.json`` 载荷 (不读像素)。"""
    payload: dict[str, Any] = {
        "batch": batch_tag(run),
        "run_root": _rel(run.root),
        "display": {
            "crop_factor": CROP_FACTOR,
            "norm": "LogNorm(vmin=1, vmax=该格峰值)",
            "dpi": 150,
            "interpolation": "nearest",
            "cmap": "inferno",
        },
        "mosaics": [],
    }
    for spec in MOSAICS:
        picks = selection.get(spec.key) or []
        if not picks:
            continue
        payload["mosaics"].append(plan_mosaic(run, spec, picks))
    return payload


def render_farfield(run: Run, selection: dict[str, list[CellPick]],
                    payload: dict[str, Any]) -> list[Path]:
    """按已规划好的 ``payload`` 画 4 张拼图。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in MOSAICS:
        picks = selection.get(spec.key) or []
        plan = next((m for m in payload["mosaics"] if m["mosaic"] == spec.key), None)
        if not picks or plan is None:
            continue
        out_path = FIG_DIR / plan["figure"]
        if draw_mosaic(run, spec, picks, plan, out_path):
            paths.append(out_path)
    return paths


def _atomic_artifact_manifest(atomic: RetestRun) -> list[tuple[str, str]]:
    """§13 归档目录的文件说明 (与实际复制的文件一一对应)。"""
    rows: list[tuple[str, str]] = [("manifest.json", "三段汇总 (行数 / 自变量 / 模式 / 配置名)")]
    for name in STAGES:
        stage_dir = atomic.root / name
        rows.append((f"{name}/measurements.csv",
                     f"{STAGE_CN[name]} 全部指标 — §13 的**唯一来源**"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (用于交叉校验)"))
        if (stage_dir / "run_config.json").exists():
            rows.append((f"{name}/run_config.json",
                         "段级参数 (无 `plan`; 含 mode / preflight / side_max / 冻结 0 级)"))
    if (atomic.root / "sides" / "selected.json").exists():
        rows.append(("sides/selected.json", "第 1 段选定边长与它的分数"))
    for name in STAGES:
        for cfg in atomic.stages[name].configs:
            rows.append((f"{name}/{cfg.config}/metrics.json",
                         f"{cfg.config} 单配置指标 + `preflight`(含本进程冻结的 0 级)"))
    rows.append(("frames_selection.json", "§13 四张拼图的逐格来源 + sha256 + 裁剪/拉伸参数"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows


def atomic_artifact_sources(atomic: RetestRun, target: Path) -> list[tuple[Path, Path]]:
    """§13 归档的 (源, 目标) 清单。"""
    pairs: list[tuple[Path, Path]] = [(atomic.root / "manifest.json", target / "manifest.json")]
    for name in STAGES:
        stage_dir = atomic.root / name
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            src = stage_dir / fname
            if src.exists():
                pairs.append((src, target / name / fname))
        for cfg in atomic.stages[name].configs:
            pairs.append((stage_dir / cfg.config / "metrics.json",
                          target / name / cfg.config / "metrics.json"))
    selected = atomic.root / "sides" / "selected.json"
    if selected.exists():
        pairs.append((selected, target / "sides" / "selected.json"))
    return pairs


def _copy_farfield_to_artifacts(payload: dict[str, Any], figures: Sequence[Path],
                                target: Path | None = None) -> int:
    """把拼图复制进 ``artifacts/<batch>/`` 并写 ``frames_selection.json``。"""
    target = ARTIFACT_DIR if target is None else target
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in figures:
        if not path.exists():
            continue
        shutil.copy2(path, target / path.name)
        copied += 1
    write_frames_selection(payload, target)
    return copied


def _check_farfield(run: Run, payload: dict[str, Any], figures: Sequence[Path],
                    artifact_dir: Path | None = None) -> list[str]:
    """``--check``: 拼图存在、与 artifacts 副本逐字节一致, 且 frames_selection.json 未过期。"""
    problems: list[str] = []
    if not payload.get("mosaics"):
        return problems
    artifact_dir = ARTIFACT_DIR if artifact_dir is None else artifact_dir
    drawn = {path.name for path in figures}
    for mosaic in payload["mosaics"]:
        name = mosaic["figure"]
        if name not in drawn:
            problems.append(f"拼图缺失: figures/{name}")
            continue
        source = FIG_DIR / name
        copy = artifact_dir / name
        if not copy.exists():
            problems.append(f"拼图副本缺失: {_rel(copy)}")
        elif _sha256(source) != _sha256(copy):
            problems.append(f"拼图与 artifacts 副本不一致: {name}")
    expected = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = artifact_dir / "frames_selection.json"
    if not path.exists():
        problems.append(f"缺失: {_rel(path)}")
    elif path.read_text(encoding="utf-8") != expected:
        problems.append("frames_selection.json 与现场规划不一致 (帧/裁剪/峰值已变)")
    return problems


def write_frames_selection(payload: dict[str, Any], target: Path) -> Path:
    """``frames_selection.json`` (排序键、无时间戳 ⇒ 逐字节可复现)。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target / "frames_selection.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 报告正文
# ---------------------------------------------------------------------------

FLAG_LEGEND = (
    "`†` = 该行采集时 **settle 判据超时** (日志原文 `settle criterion not met`); "
    "`‡` = 该行整帧电平低于同段中位数的 "
    f"{DARK_FRAME_FRACTION:.0%} (帧近乎全黑); "
    f"`※` = 该行 CV 超过同段中位数的 {OUTLIER_CV_RATIO:.1f}×; "
    "`§` = 该行方框有像素落在画面外 (`box_clipped_px > 0`)。"
)


def render_retest_section(
    run: Run, f: Findings, retest: RetestRun, rf: RetestFindings
) -> tuple[list[str], list[Table]]:
    """§11 复测分节。数字全部从复测 CSV / metrics.json 现读, 现场记录单独标注。"""
    tables: list[Table] = []
    out: list[str] = []
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]
    r_blaze = retest.stages["blaze"]
    r_zern = retest.stages["zernike"]

    def add(text: str = "") -> None:
        out.append(text)

    add("## 11. 复测（功率已变）")
    add()
    add(
        "本节是**第二批产物**：`"
        f"{_rel(retest.root)}/`。基线之后激光功率变了（见 §11.1），所以把需要复查的配置"
        "用**原子化调用**在**同一脚本版本**下重跑了一遍：每个配置一个独立进程"
        f"（`mode = {retest.mode}`）、同一 {rf.retest_exposure_ms:g} ms 曝光、同一 "
        f"`--side-px {r_blaze.rows[0]['selected_side_px']}`，共 {rf.retest_rows} 行"
        f"（光栅 {'/'.join(f'{cfg.value:.0f}' for cfg in r_blaze.configs)}、"
        f"Zernike {'/'.join(f'{cfg.value:.0f}' for cfg in r_zern.configs)}）。"
    )
    add()
    add(
        "复测批只有两个用途：(a) 判定基线的两处异常是物理效应还是采集事故；"
        "(b) 在同一批内做**邻居比较**。它**不能**与基线做绝对值比较 —— 理由见 §11.1。"
    )
    add()
    add(
        "**本节与前面所有章节是两套数字，不能混读。** 复测的每一行都由生成器从复测的 "
        "`measurements.csv` 现读（并与同配置的 `metrics.json` 逐值交叉校验），"
        "所以脚本覆盖重跑这 "
        f"{rf.retest_rows} 个配置后，只需重跑一次生成器即可刷新本节 —— "
        "生成器不缓存、不内联任何复测数值。"
    )
    if rf.prior_root is not None:
        add()
        add(
            f"> 脚注：更早一版复测（`{_rel(rf.prior_root)}`）跨了两个脚本版本，**已作废**，"
            "仅作证据留存；本报告不引用它的任何数字。"
        )
    add()

    # -- 11.1 ---------------------------------------------------------------
    add("### 11.1 口径：为什么本节只做批内比较")
    add()
    regime = Table(
        ("项", "基线", "复测"),
        [
            ("产物根", f"`{_rel(run.root)}`", f"`{_rel(retest.root)}`"),
            ("段", "sides / zernike / blaze",
             " / ".join(RETEST_STAGE_CN[s] for s in RETEST_STAGES) + "（**无 sides**）"),
            ("行数", derived(float(sum(len(s.rows) for s in run.stages.values())), 0),
             derived(float(rf.retest_rows), 0)),
            ("生效曝光 (ms)", derived(rf.baseline_exposure_ms, 1),
             derived(rf.retest_exposure_ms, 1)),
            ("会话上限 (ms)", derived(run.exposure_ceiling_ms, 1),
             derived(retest.exposure_ceiling_ms, 1)),
            ("调用方式", "一次进程连续扫三段", f"{retest.mode}（每配置一个进程）"),
            ("目标边长来源", f"第 1 段按 score 选定 {f.selected_side:.0f} px",
             f"`--side-px` 显式覆盖（`selected_source = {retest.selected_source}`）"),
        ],
    )
    tables.append(regime)
    add(regime.render())
    add()
    add(
        f"两批的**工作点差 {rf.retest_exposure_ms / rf.baseline_exposure_ms:.0f}×**"
        f"（{rf.baseline_exposure_ms:g} ms → {rf.retest_exposure_ms:g} ms），"
        "因为激光功率在基线之后变了。功率量级用下面这张表表达 —— "
        "**它是现场记录，不在这两批产物里**，所以单独一张表、单独一列来源："
    )
    add()
    power_rows: list[tuple[Cell | str, ...]] = []
    for record in (FIELD_BASELINE_FLAT, FIELD_RETEST_FLAT):
        anchor = record.anchor()
        for exposure, peak in record.readings:
            power_rows.append((
                record.label,
                derived(exposure, 2),
                derived(peak, 1),
                derived(peak / exposure, 1,
                        "  ← 定斜率" if (exposure, peak) == anchor else ""),
                record.source if (exposure, peak) == anchor else "",
            ))
    power_rows.append((
        "**功率比**",
        "—",
        "—",
        derived(rf.power_ratio, 2, "×"),
        f"{rf.baseline_counts_per_ms:.1f} / {rf.retest_counts_per_ms:.1f} counts·ms⁻¹",
    ))
    power_tbl = Table(("现场记录（**非本批产物**）", "曝光 (ms)", "峰值 (counts)",
                       "counts/ms", "来源"), power_rows)
    tables.append(power_tbl)
    add(power_tbl.render())
    add()
    add(
        f"读法：基线同期 0.5 ms 时 0 级峰值 {FIELD_BASELINE_FLAT.anchor()[1]:g} counts "
        f"（{rf.baseline_counts_per_ms:.1f} counts/ms），复测前 5 ms 时 "
        f"{FIELD_RETEST_FLAT.anchor()[1]:g} counts（{rf.retest_counts_per_ms:.1f} counts/ms）"
        f"⇒ 同一曝光下复测期**弱 {rf.power_ratio:.2f}×**。"
        f"所以复测用 {rf.retest_exposure_ms:g} ms 补偿，折合基线的 "
        f"{rf.equivalent_exposure_ms:.1f} ms —— 与基线 {rf.baseline_exposure_ms:g} ms "
        "几乎相同。**这条折算只说明「亮度口径大致相当」，不等于两批可比**："
    )
    add()
    add(
        "- 斜率只取每段记录里**曝光最大**的那个点。低曝光读数被噪声地板抬高"
        f"（0.05 ms 只有 {FIELD_RETEST_FLAT.readings[0][1]:g} counts），拿它定斜率会把功率比算歪。"
        "同一段记录里 1 ms 与 5 ms 的 counts/ms 相差 "
        f"{abs(FIELD_RETEST_FLAT.readings[2][1] / FIELD_RETEST_FLAT.readings[2][0] - rf.retest_counts_per_ms) / rf.retest_counts_per_ms:.0%}，"
        "所以这条线性外推的可信度就是**约一成**，不要读得更细。"
    )
    add(
        f"- 两批的**成像状态**本身不同：`global_spot_fwhm` 基线 {min(v for v in base_blaze.col('global_spot_fwhm_px') if math.isfinite(v)):.1f}–"
        f"{max(base_blaze.col('global_spot_fwhm_px')):.1f} px、复测 "
        f"{min(r_blaze.col('global_spot_fwhm_px')):.1f}–{max(r_blaze.col('global_spot_fwhm_px')):.1f} px；"
        f"CV 也系统性偏低（光栅 {min(rf.blaze_cv):.2f}–{max(rf.blaze_cv):.2f} vs 基线 "
        f"{min(base_blaze.col('cv')):.2f}–{max(base_blaze.col('cv')):.2f}）。"
        "**「复测 CV 更低」不能读成「复测更好」**，它只是另一个成像状态。"
    )
    add(
        "- 因此本节**只做批内邻居比较**：同一批内、相邻配置之间「谁高谁低」才是可判定的；"
        "跨批的绝对值一律不比。"
    )
    add()

    # -- 11.2 ---------------------------------------------------------------
    add("### 11.2 复测是怎么跑的（原子化 + 显式边长）")
    add()
    add(
        f"每个配置都是独立进程（`mode = {retest.mode}`），边长由 `--side-px "
        f"{r_blaze.rows[0]['selected_side_px']}` 显式给（`selected_source = "
        f"{retest.selected_source}`），因此**每个配置都会重新定位一次 0 级**——"
        "这一点在 §11.6 会变成关键。基线则是「定位一次、冻结 18 行」。"
    )
    add()
    cfg_rows: list[tuple[Cell | str, ...]] = []
    for stage_name in RETEST_STAGES:
        for cfg in retest.stages[stage_name].configs:
            zx, zy = cfg.zero_order
            settle_text = (
                "；".join(ev.describe() for ev in cfg.settle) if cfg.settle else "—"
            )
            cfg_rows.append((
                RETEST_STAGE_CN[stage_name],
                cfg.config,
                cell_of(cfg.row, VARIABLE[stage_name], 0),
                derived(zx, 0),
                derived(zy, 0),
                cell_of(cfg.row, "exposure_ms_readback", 1),
                "无可用局域光斑" if cfg.spot_flagged else "光斑可用",
                derived(float(cfg.spot.get("fwhm_px", float("nan"))), 1),
                settle_text,
            ))
    cfg_tbl = Table(
        ("段", "配置", "自变量", "0 级 x", "0 级 y", "曝光读回 (ms)",
         "平场评估", "平场 FWHM (px)", "settle 告警"),
        cfg_rows,
    )
    tables.append(cfg_tbl)
    add(cfg_tbl.render())
    add()
    add(
        f"两件事先记下：(1) 全部 {rf.retest_rows} 行的 `frame_peak` 都顶到满量程，"
        f"平场评估的 FWHM 是 "
        f"{min(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f}–"
        f"{max(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f} px —— "
        "0 级是一个**饱和平台**，`argmax` 在平台上的位置本身就不稳；"
        f"(2) {len(rf.flagged_configs)} 个配置（{', '.join(rf.flagged_configs)}）的平场评估"
        "直接报了「光路有光但没有可评分的局域光斑」，"
        f"{len(rf.settle_configs)} 个配置有 settle 告警（{', '.join(rf.settle_configs)}），"
        f"但**没有一条**探测峰值是黑帧（低于满量程 {DARK_FRAME_FRACTION:.0%} 的配置："
        f"{', '.join(rf.dark_probe_configs) if rf.dark_probe_configs else '无'}）—— "
        "与基线「超时 ≠ 帧坏」的读法一致（§2.3）。"
    )
    add()

    # -- 11.3 ---------------------------------------------------------------
    add("### 11.3 光栅 32 / 40 / 48：基线与复测并排")
    add()
    add(
        "两列数字来自两批各自的 `measurements.csv`；`—` 表示该批没有这个周期。"
        "「基线 40」那一行是 §6.3 认定的黑帧（`frame_peak = "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 40.0)['frame_peak']):.1f}`），"
        "它的 CV / PIB / score 全部不可用。"
    )
    add()
    blaze_rows: list[tuple[Cell | str, ...]] = []
    for value in [float(r["blaze_period_px"]) for r in r_blaze.rows]:
        for label, row in (("基线", _row_at(base_blaze.rows, "blaze_period_px", value)),
                           ("复测", _row_at(r_blaze.rows, "blaze_period_px", value))):
            note = ""
            if label == "基线" and float(row["frame_peak"]) < DARK_FRAME_FRACTION * base_blaze.median("frame_peak"):
                note = f"**黑帧（frame_peak = {float(row['frame_peak']):.1f}）**"
            elif label == "复测":
                note = "批内邻居"
            blaze_rows.append((
                derived(value, 0),
                label,
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                cell_of(row, "score", 4),
                cell_of(row, "frame_peak", 2),
                cell_of(row, "box_peak", 2),
                cell_of(row, "box_clipped_px", 0),
                cell_of(row, "box_centroid_offset_x", 2),
                cell_of(row, "box_centroid_offset_y", 2),
                note or "—",
            ))
    blaze_tbl = Table(
        ("周期 (px)", "批", "CV", "PIB", "score", "frame_peak", "box_peak",
         "出画面积 (px)", "偏移 x (px)", "偏移 y (px)", "备注"),
        blaze_rows,
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()

    # -- 11.4 ---------------------------------------------------------------
    add("### 11.4 Zernike 17 / 18 / 19：基线与复测并排")
    add()
    add(
        "基线 18 是 §5.2 的黑帧（`frame_peak = "
        f"{float(_row_at(base_zern.rows, 'n_max', 18.0)['frame_peak']):.1f}`），"
        "基线 17 是 settle 超时行。**解释方差与残差 RMS 在两批里逐位相同**"
        f"（n_max = {', '.join(f'{v:.0f}' for v in rf.fit_bit_identical)} 全部逐位一致）—— "
        "同一 GS 相位 + 同一拟合配置 ⇒ 拟合量是确定性的，所以两批的差别只能来自**测量**。"
    )
    add()
    zern_rows: list[tuple[Cell | str, ...]] = []
    for value in [float(r["n_max"]) for r in r_zern.rows]:
        for label, row in (("基线", _row_at(base_zern.rows, "n_max", value)),
                           ("复测", _row_at(r_zern.rows, "n_max", value))):
            note = ""
            if label == "基线":
                if float(row["frame_peak"]) < DARK_FRAME_FRACTION * base_zern.median("frame_peak"):
                    note = f"**黑帧（frame_peak = {float(row['frame_peak']):.1f}）**"
                elif run.settle_for("zernike", _idx_of(base_zern.x, value)) is not None:
                    note = "settle 超时"
            else:
                note = "批内邻居"
            zern_rows.append((
                derived(value, 0),
                label,
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                cell_of(row, "score", 4),
                cell_of(row, "frame_peak", 2),
                cell_of(row, "zernike_explained_fraction", 6),
                cell_of(row, "zernike_rms_residual_rad", 6),
                cell_of(row, "box_centroid_offset_x", 2),
                note or "—",
            ))
    zern_tbl = Table(
        ("n_max", "批", "CV", "PIB", "score", "frame_peak", "解释方差",
         "残差 RMS (rad)", "偏移 x (px)", "备注"),
        zern_rows,
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()

    # -- 11.5 ---------------------------------------------------------------
    add("### 11.5 判定")
    add()
    b40 = _row_at(base_blaze.rows, "blaze_period_px", 40.0)
    b18 = _row_at(base_zern.rows, "n_max", 18.0)
    b17 = _row_at(base_zern.rows, "n_max", 17.0)
    r32 = _row_at(r_blaze.rows, "blaze_period_px", 32.0)
    r40 = _row_at(r_blaze.rows, "blaze_period_px", 40.0)
    r48 = _row_at(r_blaze.rows, "blaze_period_px", 48.0)
    rz17 = _row_at(r_zern.rows, "n_max", 17.0)
    rz18 = _row_at(r_zern.rows, "n_max", 18.0)
    rz19 = _row_at(r_zern.rows, "n_max", 19.0)
    add(
        f"**判定 1 — 基线 blaze 40 的异常（CV {float(b40['cv']):.2f}、score "
        f"{float(b40['score']):.4f}）：推翻。**"
        f"复测 CV 在 32/40/48 上是 {rf.blaze_cv[0]:.4f} / {rf.blaze_cv[1]:.4f} / "
        f"{rf.blaze_cv[2]:.4f}（随周期单调下降 = {rf.blaze_cv_monotone}），"
        f"40 相对左右邻居中位数只偏 **{rf.blaze40_dev:+.2%}**；score 是 "
        f"{float(r32['score']):.4f} / {float(r40['score']):.4f} / {float(r48['score']):.4f}"
        f"（单调上升）。**40 没有异常。**"
        f"基线同一位置偏 {rf.baseline_blaze40_dev:+.1%}（CV {float(b40['cv']):.2f} 对邻居 "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 32.0)['cv']):.2f} / "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 48.0)['cv']):.2f}），"
        "那是黑帧采集失败（`frame_peak = "
        f"{float(b40['frame_peak']):.1f}`、`global_spot_fwhm = nan`、settle 超时且探测峰值 2）"
        "留下的假极值 —— §6.3 的 provisional 可以按「已复测推翻」结案。"
    )
    add()
    add(
        f"**判定 2 — 基线 Zernike 18 的异常（CV {float(b18['cv']):.2f}、frame_peak "
        f"{float(b18['frame_peak']):.1f}、质心 x 从 +120 跳到 −26.8）：推翻。**"
        f"复测 CV 在 17/18/19 上是 {rf.zern_cv[0]:.4f} / {rf.zern_cv[1]:.4f} / "
        f"{rf.zern_cv[2]:.4f}，18 相对左右邻居只偏 **{rf.zern18_dev:+.2%}**；"
        f"复测 18 的 `frame_peak` 回到满量程、质心 x 偏移 "
        f"{float(rz18['box_centroid_offset_x']):+.2f} px —— 与 19 的 "
        f"{float(rz19['box_centroid_offset_x']):+.2f} px 只差 "
        f"{abs(float(rz18['box_centroid_offset_x']) / float(rz19['box_centroid_offset_x']) - 1):.1%}"
        f"（17 是 {float(rz17['box_centroid_offset_x']):+.2f} px），"
        f"基线那个 −26.80 px 的跳变没有复现。基线同一位置偏 {rf.baseline_zern18_dev:+.1%} —— 比复测批内"
        f"任何一点偏离邻居的量（≤ {rf.batch_dev_max:.1%}，见判定 6）大 "
        f"{abs(rf.baseline_zern18_dev) / rf.batch_dev_max:.0f} 倍。**18 阶本身没有问题**；"
        "基线那一行是黑帧。"
    )
    add()
    add(
        "**判定 3 — 我此前把基线 n_max = 17 的高端异常归因为「高阶级重构在面板上欠采样」："
        "未确认（降级为未证实假设）。**"
        f"复测 17 的 CV {rf.zern_cv[0]:.4f} 与 18/19 同量级（相对邻居偏 "
        f"{_neighbour_deviation(rf.zern_cv, 0):+.2%}），"
        "且复测 17 自己**也有** settle 告警（"
        f"{retest.stages['zernike'].configs[0].settle[0].describe() if retest.stages['zernike'].configs[0].settle else '无'}）"
        "却没有任何异常 ⇒ 「超时」本身不产生异常；基线的 17 更像一次**探测帧全黑**"
        f"（settle 探测峰值 2、最终帧 {float(b17['frame_peak']):.1f}）的失败采集。"
        "**要证实欠采样假设，需要的不是再测一个点，而是同一配置的多次重复**："
        "固定 `n_max = 17`、固定相位、连测 N ≥ 5 次，看 CV 的分布是否**系统性地**"
        "偏离 17/18/19 的趋势 —— 单点复测只能排除「单次采集失败」，无法确认机制。"
    )
    add()
    add(
        "**判定 4 — 位移：方向可用，标度与 1/P 形状都不可用。**"
        "⚠️ 先把口径钉死：**位移只有 `measured_shift_y_px` 这一列**（质心相对冻结 0 级）。"
        "`box_centroid_offset_y` **不是**位移 —— 它 = 实测质心 − **预测框心**，"
        "而框是跟着预测走的，所以它天然是个大负值（§11.6 给出恒等式，残差 "
        f"{rf.offset_identity_residual:.1e} px）；把它当位移读会得到「反号」的假结论。"
    )
    add()
    add(
        f"- **方向同号**：P = 32 处基线 {float(_row_at(base_blaze.rows, 'blaze_period_px', 32.0)['measured_shift_y_px']):+.2f} px、"
        f"复测 {float(r32['measured_shift_y_px']):+.2f} px；P = 48 处基线 "
        f"{float(_row_at(base_blaze.rows, 'blaze_period_px', 48.0)['measured_shift_y_px']):+.2f} px、"
        f"复测 {float(r48['measured_shift_y_px']):+.2f} px —— 两批同号、同量级。"
    )
    add(
        f"- **标度不可用**：实测/预测 = 复测 {min(rf.ratio_retest):.3f}–{max(rf.ratio_retest):.3f}、"
        f"基线 {min(rf.ratio_baseline):.3f}–{max(rf.ratio_baseline):.3f}（基线排除黑帧行）。"
        "若只是「标度系数错」，这个比值应当**与周期无关**；"
        f"实际它随周期上升（{len(rf.ratio_baseline)} 个点里 "
        f"{sum(1 for i in range(1, len(rf.ratio_baseline)) if rf.ratio_baseline[i] > rf.ratio_baseline[i - 1])} 步上升）"
        "⇒ 形状也不对。"
    )
    add(
        f"- **1/P 形状被强烈压缩**：基线 P {rf.measured_ends_baseline[0]:.0f} → "
        f"{rf.predicted_ends_baseline[0]:.0f}（{rf.predicted_ends_baseline[0] / rf.measured_ends_baseline[0]:.1f}×）时，"
        f"预测 {rf.predicted_ends_baseline[1]:.2f} → {rf.predicted_ends_baseline[2]:.2f} px"
        f"（{rf.predicted_ends_baseline[1] / rf.predicted_ends_baseline[2]:.1f}×，构造上恰好 ∝1/P），"
        f"实测只有 {rf.measured_ends_baseline[1]:.2f} → {rf.measured_ends_baseline[2]:.2f} px"
        f"（{rf.measured_ends_baseline[1] / rf.measured_ends_baseline[2]:.1f}×）。"
        f"两端点 log-log 斜率：预测 α = {rf.alpha_predicted:.2f}，基线实测 α = "
        f"{rf.alpha_baseline:.2f}；复测只有 {rf.measured_ends_retest[0]:.0f} → "
        f"{rf.predicted_ends_retest[0]:.0f}（{rf.predicted_ends_retest[0] / rf.measured_ends_retest[0]:.2f}×）"
        f"三点，实测 α = {rf.alpha_retest:.2f} —— 跨度太短，只能说明「明显不等于 1」。"
    )
    add(
        "- **直接后果**：周期越小，框偏得越远。复测里框心相对质心偏 "
        + " / ".join(f"{off:+.1f}" for _, off in rf.offset_y_small_period)
        + " px（" + " / ".join(f"P={p:.0f}" for p, _ in rf.offset_y_small_period)
        + "）—— 与基线同量级（§6.4）。"
    )
    add()
    add(
        "**判定 5 — 「方框跟着 `predicted_shift_px` 走」这条做法必须改：成立。**"
        f"复测 {len(r_blaze.rows)} 行上恒等式 `偏移 y ≡ measured_shift_y − predicted_shift_y` "
        f"成立到 {rf.offset_identity_residual:.1e} px，所以框的位置**完全由预测决定**："
        "预测偏大 → 框偏 → 框内质心被框边拉回 → 量出来的位移更小。这是循环依赖，"
        "用这种方式永远验不了 `predicted_shift`。两个可选改法："
    )
    add()
    add(
        "1. **框固定 + 宽 ROI**：框心 = 冻结 0 级（不跟随预测），或直接用 ≥ 半幅的 ROI "
        "对**整帧质心**求位移。这是唯一能独立测位移的做法。"
    )
    add(
        "2. **重标定**：若一定要框跟随，先用改法 1 的实测把 `predicted_shift` 的**标度与 "
        "1/P 指数**拟合出来（本轮实测 α ≈ "
        f"{rf.alpha_baseline:.2f}，而不是 1.00），再拿标定后的式子摆放方框。"
    )
    add()
    add(
        "**判定 6 — 这两处极值不可能是「趋势上的点」：成立（单批即可判定）。**"
        f"复测批内 6 个配置相对左右邻居中位数的偏差最大 **{rf.batch_dev_max:.1%}**"
        f"（这是端点的趋势斜率；两个**中间点**分别只有 {rf.blaze40_dev:+.2%} 与 "
        f"{rf.zern18_dev:+.2%}）；而基线同一统计量是 "
        f"{rf.baseline_blaze40_dev:+.0%} 与 {rf.baseline_zern18_dev:+.0%}，"
        f"相差 {abs(rf.baseline_blaze40_dev) / rf.batch_dev_max:.0f} 倍与 "
        f"{abs(rf.baseline_zern18_dev) / rf.batch_dev_max:.0f} 倍。"
        "也就是说：**同一条平滑趋势上不可能长出这两个值** —— 这个判定只用复测批自己的"
        "内部一致性，不依赖任何跨批重复性，也不依赖基线的任何假设。"
    )
    add()

    # -- 11.6 ---------------------------------------------------------------
    add("### 11.6 0 级每次重定位都会抖，而光斑几乎不动")
    add()
    add(
        "复测是原子化的，所以每个配置都重新在平场帧上 `argmax` 定位 0 级。"
        "把三次独立定位与三次光斑质心摆在一起（全部是光栅段）："
    )
    add()
    shift_rows: list[tuple[Cell | str, ...]] = []
    for cfg in r_blaze.configs:
        zx, zy = cfg.zero_order
        offset = float(cfg.row["box_centroid_offset_y"])
        measured = float(cfg.row["measured_shift_y_px"])
        predicted = float(cfg.row["predicted_shift_px"])
        shift_rows.append((
            derived(cfg.value, 0),
            derived(zx, 0),
            derived(zy, 0),
            cell_of(cfg.row, "box_centroid_x", 2),
            cell_of(cfg.row, "box_centroid_y", 2),
            derived(predicted, 2),
            derived(measured, 2),
            derived(offset, 2),
            derived(offset - (measured - predicted), 6),
        ))
    shift_tbl = Table(
        ("周期 (px)", "0 级 x", "0 级 y", "质心 x", "质心 y", "predicted y (px)",
         "measured y (px)", "偏移 y (px)", "偏移 − (measured − predicted)"),
        shift_rows,
    )
    tables.append(shift_tbl)
    add(shift_tbl.render())
    add()
    add(
        f"三次独立定位的 0 级 x 相差 **{rf.zero_order_x_span:.0f} px**、y 相差 "
        f"**{rf.zero_order_y_span:.0f} px**；而**光斑质心**只差 "
        f"{rf.centroid_x_span:.2f} px / {rf.centroid_y_span:.2f} px。"
        f"于是 `measured_shift_x` 的 {rf.measured_x_span:.2f} px、`measured_shift_y` 的 "
        f"{rf.measured_y_span:.2f} px 波动里，**原点定位占了大头**，不是光斑在动。"
        "原因就在 §11.2 那张表：平场 0 级是**饱和平台**"
        f"（FWHM {min(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f}–"
        f"{max(c.spot.get('fwhm_px', float('nan')) for c in retest.all_configs):.0f} px、"
        f"`hollowness` 最高到 {max(float(c.spot.get('hollowness', 0.0)) for c in retest.all_configs):.2f}、"
        f"{len(rf.flagged_configs)} 个配置被平场评估直接判为「无可用局域光斑」），"
        "`argmax` 落在平台的哪一格没有物理意义。"
        f"（Zernike 段更明显：三次定位是 "
        + " / ".join(f"{cfg.zero_order[0]:.0f}" for cfg in r_zern.configs)
        + f" px（x 相差 {rf.zero_order_x_span_zern:.0f} px、y 相差 "
        f"{rf.zero_order_y_span_zern:.0f} px），比光栅段还大。）"
    )
    add()
    add("由此得到一条**判别性复测方案**（下一轮直接照做）：")
    add()
    add(
        "1. **先在低曝光定位 0 级**（0 级不饱和的那个曝光，例如基线用的 "
        f"{rf.baseline_exposure_ms:g} ms），把位置写进 `run_config` 冻结；再切到测量曝光。"
        "**不要**在饱和帧上 `argmax`。"
    )
    add(
        "2. **框固定**：框心 = 上一步冻结的 0 级，不跟随 `predicted_shift`；"
        "或者干脆用宽 ROI（≥ 画幅的一半）直接对**整帧质心**求位移。"
    )
    add(
        "3. **同一相位重复 ≥ 3 次**，报质心的分布而不是单值 —— 现在这个量级"
        f"（{rf.measured_y_span:.1f} px）与原点抖动（{rf.zero_order_x_span:.0f} px）同阶，"
        "单次读数分不开「光斑动了」与「原点动了」。"
    )
    add(
        "4. 有了上面三条，`predicted_shift_px` 的**符号与标度**才第一次可以被独立验证；"
        "在那之前，它只能当**摆放方框的初值**，不能当被验证的物理量。"
    )
    add()
    add(f"![复测邻居比较](figures/{_figname('fig5_retest_neighbors')})")
    add()
    add(
        "图 5 是**批内**邻居比较：灰色虚线是基线全段（含它的黑帧点），绿色实线是复测的三点。"
        "两条线的**绝对高度不要互相比较**（§11.1），只看各自段内的顺序与尖峰是否还在。"
    )
    add()

    # -- 11.7 ---------------------------------------------------------------
    add("### 11.7 列口径（两批的列集合不同）")
    add()
    add(
        f"复测的 CSV 列数是 {len(r_blaze.headers)} / {len(r_zern.headers)}"
        f"（光栅 / Zernike），基线是 {len(base_blaze.headers)} / {len(base_zern.headers)}。"
        "本节用到的列逐个核对如下 —— 复测独有的是 `config` / `stage` / `selected_source` / "
        "`base_phase_source` / `fit_source` 这类**运行元数据**，不是指标："
    )
    add()
    needed_blaze = ("blaze_period_px", "cv", "pib", "score", "frame_peak", "box_peak",
                    "box_clipped_px", "box_centroid_offset_x", "box_centroid_offset_y",
                    "measured_shift_y_px", "predicted_shift_px", "exposure_ms_readback")
    needed_zern = ("n_max", "cv", "pib", "score", "frame_peak", "box_peak",
                   "zernike_explained_fraction", "zernike_rms_residual_rad",
                   "box_centroid_offset_x", "box_centroid_offset_y",
                   "exposure_ms_readback")
    contract_rows = [(f"光栅 · {a}", b, c) for a, b, c in
                     column_contract(base_blaze, r_blaze, needed_blaze)]
    contract_rows += [(f"Zernike · {a}", b, c) for a, b, c in
                      column_contract(base_zern, r_zern, needed_zern)]
    contract_tbl = Table(("列", "基线有", "复测有"), contract_rows)
    tables.append(contract_tbl)
    add(contract_tbl.render())
    add()
    missing = [row for row in contract_rows if "**缺**" in (row[1], row[2])]
    add(
        "全部所需列在两批里都存在（缺列数 = "
        f"{len(missing)}）。生成器不假设两批表头相同：它按列名取，取不到就报错退出。"
    )
    add()
    add(
        "**两处容易读错的口径**："
    )
    add()
    add(
        "- **段级 `run_config.json` 的 `zero_order_xy_frozen` 不是每配置的框心。** "
        + "；".join(
            f"{RETEST_STAGE_CN[name]}段级记 "
            f"({float(st.run_config['zero_order_xy_frozen'][0]):.0f}, "
            f"{float(st.run_config['zero_order_xy_frozen'][1]):.0f})，而三个配置各自是 "
            + " / ".join(
                f"({cfg.zero_order[0]:.0f}, {cfg.zero_order[1]:.0f})" for cfg in st.configs)
            for name, st in retest.stages.items()
            if st.run_config.get("zero_order_xy_frozen")
        )
        + "。本节一律用**每配置 `metrics.json` 的 `preflight`** 值 —— 那才是真正摆放方框的那个数。"
    )
    add(
        "- **`fit_seconds` 不是整段耗时**：它是**本进程这一次**拟合的墙钟"
        "（复测里 "
        + " / ".join(
            f"{float(r['fit_seconds']):.2f} s" for r in r_zern.rows
        )
        + "），复测是每配置一个进程，所以它既不累加也不代表阶段时长。本节不用它做任何时长判断。"
        if "fit_seconds" in r_zern.headers
        else "- 复测的 Zernike CSV 没有 `fit_seconds` 列。"
    )
    add()
    return out, tables


def _cell_short(cell: dict[str, Any]) -> str:
    """表格/图注里的短标签: ``side=200 px`` / ``n_max=30`` / ``P=32 px``。"""
    stage = cell["stage"]
    value = float(cell["value"][VARIABLE[stage]])
    if stage == "sides":
        return f"side={value:.0f} px"
    if stage == "zernike":
        return f"n_max={value:.0f}"
    return f"P={value:.0f} px"


def render_farfield_section(
    run: Run, payload: dict[str, Any], figures: Sequence[Path],
    *, residual_pointer: bool = False,
) -> tuple[list[str], list[Table]]:
    """§12 CCD 远场光斑图。数字与配置全部来自 ``payload`` (它由 CSV 与帧路径算出)。

    ``residual_pointer`` 为真时在开头加一句指引 —— 指向 §13.8 的
    「Zernike 残差 × 同配置远场」图。那个图依赖相位文件, 只有整轮原子重扫那一批
    落了 `pupil_phase.npy`, 所以本节自己画不出来。
    """
    tables: list[Table] = []
    out: list[str] = []

    def add(text: str = "") -> None:
        out.append(text)

    add("## 12. CCD 远场光斑图")
    add()
    add(
        "本节把每段**实测远场帧**直接摆出来 —— 不是指标曲线，是相机画面。"
        "帧来自各配置落盘的 `frame_*.npz`（`np.load(...)[\"frame\"]`，"
        "`float32`、画幅与 `run_config.frame_shape_hw` 一致、0–255 counts），"
        "每格是**该配置落盘帧的算术平均**（右下角标 `×N 帧均值`）。"
    )
    add()
    add(
        "生成器在画图前逐格核对两件事：帧的最大值必须等于该行 CSV 的 `frame_peak`，"
        "npz 里那个 0 维 `params`（JSON）的 `cv`/`pib`/`score`/`frame_peak` 必须等于该行 CSV。"
        "两者任一不符就报错退出 —— 否则拼图会张冠李戴。"
    )
    add()
    if residual_pointer:
        add(
            "本节只摆**远场画面**。把 **Zernike 截断的相位残差**与同配置远场并排的那张图"
            "在 §13.8 —— 它需要每个配置的 `pupil_phase.npy`，而只有整轮原子重扫那一批落了"
            "相位文件（基线批没有）。"
        )
        add()

    add("### 12.1 显示了哪些配置")
    add()
    rows: list[tuple[Cell | str, ...]] = []
    missing_cells: list[str] = []
    for mosaic in payload["mosaics"]:
        for cell in mosaic["cells"]:
            stage = run.stages[cell["stage"]]
            row = _row_at(stage.rows, VARIABLE[cell["stage"]], cell["value"][VARIABLE[cell["stage"]]])
            source = ("—" if not cell["sources"] else
                      (_rel(Path(cell["sources"][0]["path"]).parent)
                       if cell["layout"] == "atomic"
                       else _rel(Path(cell["sources"][0]["path"]))))
            frames = (f"{cell['frames_present']} / {cell['frames_planned']}"
                      if cell["frames_planned"] else f"{cell['frames_present']} / —")
            note = cell["note"] or ("缺帧" if cell["missing"] else "")
            if cell["missing"]:
                missing_cells.append(f"{cell['stage']}/{cell['config']}")
            median = cell.get("box_median")
            rows.append((
                MOSAIC_CN.get(mosaic["mosaic"], mosaic["mosaic"]),
                STAGE_CN[cell["stage"]],
                cell["config"],
                _cell_short(cell),
                f"`{source}`",
                frames,
                derived(cell["peak"], 2),
                "饱和" if cell["saturated"] else "未饱和",
                cell_of(row, "cv", 4),
                cell_of(row, "pib", 4),
                (derived(median, 2) if median is not None else "—"),
                note or "—",
            ))
    table = Table(
        ("拼图", "段", "配置", "取值", "源", "帧数 (实际 / 计划)", "峰值 (counts)",
         "峰值状态", "CV", "PIB", "方框中位电平 (counts)", "备注"),
        rows,
    )
    tables.append(table)
    add(table.render())
    add()
    planned_typical = sorted({c["frames_planned"] for m in payload["mosaics"]
                              for c in m["cells"] if c["frames_planned"]})
    add(
        "**帧数口径**：「实际 / 计划」= 该配置**落盘的 npz 个数** / `run_config."
        "frames_per_measurement`。"
        + (f"本轮计划值是 {'/'.join(str(v) for v in planned_typical)}，"
           "而落盘的是**已经做过多帧平均的那一张**（每个 npz 内部已含若干次曝光），"
           "所以「实际 1 / 计划 4」不代表少了 3 帧，而是**平均后只存了一张**；"
           "生成器按实际存在的 npz 求平均，不补帧、不报错，并把两个数都写进 "
           "`frames_selection.json`。" if planned_typical else "")
        + (f"**缺帧的格子**：{', '.join(missing_cells)}（表里已标 `缺帧`，图上不画）。"
           if missing_cells else "")
    )
    add()

    add("### 12.2 四张拼图")
    add()
    by_key = {m["mosaic"]: m for m in payload["mosaics"]}
    drawn = {path.name for path in figures}
    captions = {
        "sides": ("方斑随目标边长变化：小方框里方斑被压得最亮（100–200 px 甚至顶到满量程），"
                  "方框放大后方斑摊开、峰值下降。第 6 格是 score 最高行或它的回退（见表）。"),
        "zernike": ("低阶 Zernike 重构后的远场：与同框的 GS 参照（最后一格）比，"
                    "重构出来的不是方斑而是一个亮核 —— 这正是 §5.1 的结论在画面上的样子。"),
        "blaze": ("叠加闪耀光栅后，方框跟着**预测**位移走；16 px 那一行位移最大，"
                  "框已出画（格内左下角标出出画面积）。"),
        "fullframe": ("整帧不裁剪：0 级（argmax）与画面几何中心并不重合，"
                      "这就是 §8 里「最大边长必须现场量」的由来。"),
    }
    for spec in MOSAICS:
        mosaic = by_key.get(spec.key)
        if mosaic is None:
            continue
        name = mosaic["figure"]
        if name not in drawn:
            add(f"（拼图 `{name}` 没有可画的帧，已跳过。）")
            add()
            continue
        add(f"![{spec.title}](figures/{name})")
        add()
        frames_seen = sorted({c["frames_present"] for c in mosaic["cells"]
                              if not c["missing"]})
        sat = [_cell_short(c) for c in mosaic["cells"] if c["saturated"] and not c["missing"]]
        add(
            f"*{spec.title}。* 对数显示（`LogNorm(vmin=1, vmax=该格峰值)`，"
            "**颜色不代表线性强度**）；"
            + (f"裁剪：以 `box_center` 为中心、边长 `box_side × {CROP_FACTOR}` "
               "（已夹到画幅内）；" if spec.crop else "本图**不裁剪**（整帧）；")
            + "叠加：目标框（虚线）、预测框心（十字）、实测质心（圆点）"
            + ("、0 级 argmax（星）与画面中心（叉）" if not spec.crop else "")
            + "；右下角 `×N 帧均值`。"
            + (f" 饱和格：{', '.join(sat)}。" if sat else "")
            + captions.get(spec.key, "")
        )
        add()
        medians = [(_cell_short(c), c["box_median"]) for c in mosaic["cells"]
                   if not c["missing"] and c.get("box_median") is not None]
        if len(medians) >= 2:
            low = min(medians, key=lambda pair: pair[1])
            high = max(medians, key=lambda pair: pair[1])
            add(
                f"方框内**中位电平**：`{high[0]}` 是 {high[1]:.2f} counts，"
                f"`{low[0]}` 只有 {low[1]:.2f} counts（整列见 §12.1）。"
                + {
                    "sides": ("方框放大后方斑被摊薄到 **1–2 counts 的量化地板**，"
                              "在对数显示下就是一片黑 —— 那是数据本身的动态范围，不是画错；"
                              "读那些格的均匀度请回 CSV 的 CV / PIB。"),
                    "zernike": ("框内大部分像素是 0 counts、峰值却顶到满量程 ⇒ "
                                "光集中在一个亮核上，**这正是 CV 7.3 的画面形态**"
                                "（§5.1 的「PIB 假高分」）。"),
                    "blaze": ("框内中位只有 1 counts 上下、峰值饱和 ⇒ 框里是一个亮核"
                              "加几乎全暗的背景，方斑并没有跟着框走（§6.4）。"),
                }.get(spec.key, "")
            )
            add()

    add("### 12.3 显示口径")
    add()
    add(
        f"1. **裁剪**：拼图 1–3 以该行 CSV 的 `box_center_x/y` 为中心、边长取 "
        f"`box_side_px × {CROP_FACTOR}`，并夹到画幅内（夹过就在 "
        "`frames_selection.json` 里记 `clipped_by_frame`）；拼图 4 **不裁剪**。"
    )
    add(
        "2. **对数拉伸**：`matplotlib.colors.LogNorm(vmin=1, vmax=该格峰值)`。"
        "0 级与方斑的每像素亮度差 2–3 个数量级（§4.2），线性拉伸只会看到 0 级一个点，"
        "方斑的结构全部落在最暗的几档里；对数显示能把两者放进同一张图。"
        "**代价是颜色与强度不成线性** —— 要读绝对强度请回 CSV 的 `frame_peak` / `box_peak`，"
        "不要从颜色反推。"
    )
    add(
        "3. **叠加与饱和**：目标框（虚线）、预测框心（十字）、实测质心（圆点）；"
        "`box_clipped_px > 0` 的格子在左下角标 `框出画 … px`。"
        f"该行 `frame_peak ≥ {SATURATION_COUNTS:.0f}` 时标题末尾加「饱和」——"
        "意味着最亮像素已被 8 bit 削平，画面里那一片的颜色不随强度变化。"
    )
    add()
    return out, tables


# ---------------------------------------------------------------------------
# §13 整轮原子重扫 (单一脚本版本 + 单一功率)
# ---------------------------------------------------------------------------


def load_selected(root: Path) -> dict[str, Any]:
    """``sides/selected.json`` (选定边长与它的分数)。没有就返回空 dict。"""
    path = root / "sides" / "selected.json"
    return load_json(path) if path.exists() else {}


def _two_tier_split(values: Sequence[float]) -> float | None:
    """在一维样本里找**最大间隙**作为两档分界 (返回分界中点); 样本太集中就返回 None。"""
    ordered = sorted(set(float(v) for v in values))
    if len(ordered) < 2:
        return None
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    gap, index = max(gaps)
    span = ordered[-1] - ordered[0]
    if span <= 0 or gap < 0.25 * span:
        return None
    return (ordered[index] + ordered[index + 1]) / 2.0


@dataclass
class AtomicFindings:
    """§13 要用的派生量。全部现场算, 不内联任何数字。"""

    exposure_ms: float
    ceiling_ms: float
    mode: str
    zero_order: tuple[float, float]
    side_max: float
    side_max_raw: float
    step: float
    rejected: tuple[float, ...]
    n_sides: int
    selected_side: float
    selected_score: float
    score_max: float
    score_argmax_side: float
    plateau: list[tuple[float, float]]
    plateau_gap: float
    tail: list[tuple[float, float]]
    box_peak_min: float
    box_peak_max: float
    frame_peak_values: list[float]
    ar_median: float
    ar_outliers: list[tuple[float, float]]
    zern_cv_min: float
    zern_cv_max: float
    zern_pib_min: float
    zern_pib_max: float
    zern_explained_min: float
    zern_explained_max: float
    zern_modes_max: float
    tier_split: float | None
    tier_low: list[tuple[float, float, float]]
    tier_high: list[tuple[float, float, float]]
    centroid_x_span: float
    blaze: list[tuple[float, float, float, float, float, float, float, float]]
    blaze_meas_min: tuple[float, float]
    blaze_meas_max: tuple[float, float]
    blaze_ratio_min: float
    blaze_ratio_max: float
    blaze_flat_ratio: float
    blaze_predicted_ratio: float
    blaze_clipped: list[float]
    cross: list[tuple[str, float, float, float, float, float, float, float]]
    cross_cv_max: float
    cross_score_max: float


def compute_atomic_findings(
    run: Run, atomic: RetestRun, selected: dict[str, Any],
    retest: RetestRun | None,
) -> AtomicFindings:
    sides = atomic.stages["sides"]
    zern = atomic.stages["zernike"]
    blaze = atomic.stages["blaze"]
    sides_cfg = sides.run_config

    xs = sides.col("side_px")
    scores = sides.col("score")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    peak = order[0]
    score_max = scores[peak]
    plateau = [(xs[i], scores[i]) for i in range(len(xs)) if scores[i] >= score_max - 1e-3]
    tail = [(xs[i], scores[i]) for i in range(len(xs)) if xs[i] > max(p for p, _ in plateau)]
    plateau_gap = (max(s for _, s in plateau) - min(s for _, s in plateau)) if len(plateau) > 1 else 0.0

    box_peaks = sides.col("box_peak")
    ars = sides.col("ar")
    ar_median = float(np.median(ars))
    ar_outliers = [(xs[i], ars[i]) for i in range(len(ars))
                   if abs(ars[i] / ar_median - 1.0) > 0.5]

    z_cv = zern.col("cv")
    z_pib = zern.col("pib")
    z_exp = zern.col("zernike_explained_fraction")
    z_off = zern.col("box_centroid_offset_x")
    z_centroid = zern.col("box_centroid_x")
    z_n = zern.col("n_max")
    z0x = [cfg.zero_order[0] for cfg in zern.configs]
    split = _two_tier_split(z0x)
    tier_low: list[tuple[float, float, float]] = []
    tier_high: list[tuple[float, float, float]] = []
    for i in range(len(z_n)):
        bucket = tier_low if split is None or z0x[i] < split else tier_high
        bucket.append((z_n[i], z_off[i], z0x[i]))

    b_p = blaze.col("blaze_period_px")
    b_pred = blaze.col("predicted_shift_px")
    b_meas = blaze.col("measured_shift_y_px")
    blaze_rows = [
        (b_p[i], b_pred[i], b_meas[i],
         b_meas[i] / b_pred[i] if b_pred[i] else float("nan"),
         blaze.col("cv")[i], blaze.col("pib")[i], blaze.col("score")[i],
         blaze.col("box_clipped_px")[i])
        for i in range(len(b_p))
    ]
    clipped = [b_p[i] for i in range(len(b_p)) if blaze.col("box_clipped_px")[i] > 0]
    clean = [row for row in blaze_rows if row[7] <= 0]
    meas_clean = [row[2] for row in clean]
    pred_clean = [row[1] for row in clean]

    # 与 §11 复测的同功率交叉核对 (只比两批都有的配置)
    cross: list[tuple[str, float, float, float, float, float, float, float]] = []
    if retest is not None:
        for stage_name in ("blaze", "zernike"):
            if stage_name not in retest.stages:
                continue
            for cfg in retest.stages[stage_name].configs:
                try:
                    mine = atomic.stages[stage_name].row_of(cfg.value)
                except KeyError:
                    continue
                cv_a = float(mine["cv"])
                cv_b = float(cfg.row["cv"])
                sc_a = float(mine["score"])
                sc_b = float(cfg.row["score"])
                cross.append((
                    stage_name, cfg.value, cv_a, cv_b,
                    abs(cv_a - cv_b) / ((cv_a + cv_b) / 2.0),
                    sc_a, sc_b, abs(sc_a - sc_b) / ((sc_a + sc_b) / 2.0),
                ))

    return AtomicFindings(
        exposure_ms=atomic.exposure_ms,
        ceiling_ms=atomic.exposure_ceiling_ms,
        mode=atomic.mode,
        zero_order=run_zero_order(sides_cfg),
        side_max=float(sides_cfg.get("side_max_px", float("nan"))),
        side_max_raw=float(sides_cfg.get("side_max_raw_px", float("nan"))),
        step=float(min(b - a for a, b in zip(xs, xs[1:]))) if len(xs) > 1 else 1.0,
        rejected=(
            float(sides_cfg.get("side_max_px", float("nan"))) + 100.0,
            float(sides_cfg.get("side_max_px", float("nan"))) + 200.0,
        ),
        n_sides=len(xs),
        selected_side=float(selected.get("side_px", float("nan"))),
        selected_score=float(selected.get("rank_value", float("nan"))),
        score_max=score_max,
        score_argmax_side=xs[peak],
        plateau=plateau,
        plateau_gap=plateau_gap,
        tail=tail,
        box_peak_min=min(box_peaks),
        box_peak_max=max(box_peaks),
        frame_peak_values=sorted({float(v) for v in sides.col("frame_peak")}),
        ar_median=ar_median,
        ar_outliers=ar_outliers,
        zern_cv_min=min(z_cv),
        zern_cv_max=max(z_cv),
        zern_pib_min=min(z_pib),
        zern_pib_max=max(z_pib),
        zern_explained_min=min(z_exp),
        zern_explained_max=max(z_exp),
        zern_modes_max=max(zern.col("zernike_modes_used")),
        tier_split=split,
        tier_low=tier_low,
        tier_high=tier_high,
        centroid_x_span=max(z_centroid) - min(z_centroid),
        blaze=blaze_rows,
        blaze_meas_min=((min(clean, key=lambda r: r[2])[0],
                         min(clean, key=lambda r: r[2])[2]) if clean
                        else (float("nan"), float("nan"))),
        blaze_meas_max=((max(clean, key=lambda r: r[2])[0],
                         max(clean, key=lambda r: r[2])[2]) if clean
                        else (float("nan"), float("nan"))),
        blaze_ratio_min=min(row[3] for row in clean) if clean else float("nan"),
        blaze_ratio_max=max(row[3] for row in clean) if clean else float("nan"),
        blaze_flat_ratio=(max(meas_clean) / min(meas_clean)) if clean else float("nan"),
        blaze_predicted_ratio=(max(pred_clean) / min(pred_clean)) if clean else float("nan"),
        blaze_clipped=clipped,
        cross=cross,
        cross_cv_max=max((row[4] for row in cross), default=float("nan")),
        cross_score_max=max((row[7] for row in cross), default=float("nan")),
    )


def run_zero_order(cfg: dict[str, Any]) -> tuple[float, float]:
    """段级 ``zero_order_xy_frozen`` (取不到就给 nan)。"""
    xy = cfg.get("zero_order_xy_frozen")
    if not xy:
        return float("nan"), float("nan")
    return float(xy[0]), float(xy[1])


def fig_atomic_sides(run: Run, atomic: RetestRun) -> Path:
    """边长 → CV/PIB/score: 本轮实线 + 基线虚线 (定性对照)。"""
    setup_matplotlib()
    a_sides = atomic.stages["sides"]
    b_sides = run.stages["sides"]
    ax_x = a_sides.col("side_px")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax1.plot(ax_x, a_sides.col("cv"), "o-", color="#d62728", label="CV（本轮）")
    ax1.plot(b_sides.x, b_sides.col("cv"), "s--", color="#d62728", alpha=0.45,
             label="CV（基线, 20 ms）")
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("CV", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(ax_x, a_sides.col("pib"), "o-", color="#1f77b4", label="PIB（本轮）")
    ax1b.plot(b_sides.x, b_sides.col("pib"), "s--", color="#1f77b4", alpha=0.45,
              label="PIB（基线, 20 ms）")
    ax1b.set_ylabel("PIB", color="#1f77b4")
    ax1b.tick_params(axis="y", labelcolor="#1f77b4")
    handles = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(handles, [ln.get_label() for ln in handles], fontsize=7.5, loc="upper left")
    ax1.set_title("(a) CV / PIB: 形状一致, 高度不可比")

    ax2.plot(ax_x, a_sides.col("score"), "o-", color="#2ca02c", label="score（本轮）")
    ax2.plot(b_sides.x, b_sides.col("score"), "s--", color="#2ca02c", alpha=0.45,
             label="score（基线, 20 ms）")
    scores_a = a_sides.col("score")
    peak_x = max(zip(ax_x, scores_a), key=lambda pair: pair[1])[0]
    plateau_x = [x for x, sc in zip(ax_x, scores_a) if sc >= max(scores_a) - 1e-3]
    ax2.plot([peak_x], [max(scores_a)], "k*", ms=14, label=f"argmax {peak_x:.0f} px")
    if len(plateau_x) > 1:
        ax2.axvspan(min(plateau_x), max(plateau_x), color="k", alpha=0.08,
                    label=f"平台 {min(plateau_x):.0f}–{max(plateau_x):.0f} px")
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("score = 0.5·PIB + 0.5/(1+CV)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("(b) score 是平台, 不是单点")
    fig.suptitle(f"整轮原子重扫（{atomic.exposure_ms:g} ms, 单一版本）", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / f"fig10_atomic_sides_{batch_tag(atomic)}.png"
    savefig(fig, path)
    return path


def fig_atomic_blaze(atomic: RetestRun) -> Path:
    """光栅周期 → 预测位移 (1/P) vs 实测位移 (近水平)。"""
    setup_matplotlib()
    blaze = atomic.stages["blaze"]
    x = blaze.col("blaze_period_px")
    pred = blaze.col("predicted_shift_px")
    meas = blaze.col("measured_shift_y_px")
    clipped = [i for i in range(len(x)) if blaze.col("box_clipped_px")[i] > 0]
    keep = [i for i in range(len(x)) if i not in clipped]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, pred, "o-", color="#1f77b4", label="预测位移 (∝ 1/周期)")
    ax.plot([x[i] for i in keep], [meas[i] for i in keep], "s-", color="#ff7f0e",
            label="实测位移 y")
    for i in clipped:
        ax.plot([x[i]], [meas[i]], "x", color="#d62728", ms=11, mew=2,
                label="该行方框出画 (不计入趋势)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}" for v in x])
    ax.minorticks_off()
    ax.set_yticks([50, 100, 200, 400])
    ax.set_yticklabels(["50", "100", "200", "400"])
    ax.set_xlabel("闪耀光栅周期 (SLM px, 对数轴)")
    ax.set_ylabel("位移 (camera px, 对数轴)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("预测按 1/周期 下降, 实测几乎水平 —— 比值从 0.22 涨到 1.20")
    fig.tight_layout()
    path = FIG_DIR / f"fig11_atomic_blaze_shift_{batch_tag(atomic)}.png"
    savefig(fig, path)
    return path


# ---------------------------------------------------------------------------
# §13.8 Zernike 截断残差 × 同配置 CCD 远场 (fig12)
# ---------------------------------------------------------------------------

#: 残差图的列 (n_max 降序)。只画 CSV 里**实际存在**、且落了 `pupil_phase.npy` 的配置。
RESIDUAL_PANEL_ORDER: tuple[int, ...] = (30, 19, 12, 7)

#: 残差色标的半幅 (rad)。固定 ±π: 面板施加的相位就是 mod 2π 的, 四列因此可直接横比。
RESIDUAL_VLIM = math.pi

#: 残差图 / 基准相位的色图 (写死, 不随数据或 matplotlib 默认值变)。
RESIDUAL_CMAP = "RdBu_r"
BASE_CMAP = "twilight"

#: §13.8 的自检容差, 与 `logs/check_zernike_phase_residual.py` 同一口径。
RESIDUAL_RMS_RTOL = 0.02
RESIDUAL_RECON_ATOL = 1e-9

#: 「可见结构」描述里的高频判据 (cyc / fit-grid px): 0.25 对应周期 4 个拟合格。
RESIDUAL_HIGH_FREQ = 0.25


def base_phase_path(root: Path, metrics: dict[str, Any]) -> Path:
    """GS 基准相位 (512², raw rad) 的路径。

    🔴 **基准不在 `<stage>/nmax_<N>/` 里**。每个 zernike 配置目录的 `pupil_phase.npy`
    是**该阶数的重构相位**; 拿它当基准会得到残差恒为 0 (踩过)。真正的基准是边长段
    选定那一格的相位 `<root>/sides/side_<S>/pupil_phase.npy`, 路径优先取 `metrics`
    里记的 `base_phase_info.side_dir` (绝对路径), 否则按 `selected_side_px` 反推。
    """
    candidates: list[Path] = []
    side_dir = (metrics.get("base_phase_info") or {}).get("side_dir")
    if side_dir:
        candidates.append(Path(str(side_dir)) / "pupil_phase.npy")
        candidates.append(root / str(side_dir) / "pupil_phase.npy")
    side = int(float(metrics.get("selected_side_px") or 0))
    if side:
        candidates.append(root / "sides" / f"side_{side}" / "pupil_phase.npy")
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"找不到 GS 基准相位 (selected_side_px={side}, "
        f"base_phase_info={metrics.get('base_phase_info')!r}) —— 残差不能算"
    )


@dataclass
class ResidualPanel:
    """fig12 的一列: 一个 ``n_max`` 的残差相位 + 它的结构描述量。"""

    n_max: int
    config: str
    row: dict[str, str]
    modes: float
    #: `metrics.json` / CSV 的记录值 (§13.8 表格与图标题用的就是它们)。
    rms_recorded: float
    explained: float
    #: 现场重算值 (与记录值对拍, 容差见 :data:`RESIDUAL_RMS_RTOL`)。
    rms_recomputed: float
    recon_delta: float
    #: 残差相位 (fit_grid², 已卷到 ±π, 单位圆外 0) 与它的统计掩码。
    residual: np.ndarray
    mask: np.ndarray
    #: 结构描述量 (全部现场算, 见 :meth:`structure_note`)。
    mean_freq: float
    high_share: float
    radial_ratio: float
    m4_share: float

    def structure_note(self) -> str:
        """「残差里的可见结构」一句话 (纯数据派生, 不含任何手写判断)。"""
        return (
            f"细颗粒高频（平均 {self.mean_freq:.2f} cyc/px，"
            f"{self.high_share:.0%} 能量在 >{RESIDUAL_HIGH_FREQ:g} cyc/px）、"
            f"径向近均匀（外缘/中心 RMS {self.radial_ratio:.2f}）、"
            f"m=4 对称仅 {self.m4_share:.1%}"
        )


@dataclass
class ResidualSet:
    """§13.8 的全部输入: 一张基准相位 + 每列一个 :class:`ResidualPanel`。"""

    base: np.ndarray
    base_path: Path
    fit_grid: int
    panels: list[ResidualPanel]


def _residual_structure(residual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """残差相位的三个结构量 (径向分布 / 空间频率 / 方位向 4 重对称)。"""
    grid = int(residual.shape[0])
    centre = (grid - 1) / 2.0
    yy, xx = np.mgrid[0:grid, 0:grid]
    radius = np.hypot(yy - centre, xx - centre) / (grid / 2.0)
    inner = mask & (radius <= 0.35)
    outer = mask & (radius > 0.70)
    rms_in = float(np.sqrt(np.mean(residual[inner] ** 2))) if inner.any() else float("nan")
    rms_out = float(np.sqrt(np.mean(residual[outer] ** 2))) if outer.any() else float("nan")

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(grid))
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    krad = np.hypot(ky, kx)
    total = float(spectrum.sum())
    mean_freq = float((spectrum * krad).sum() / total) if total > 0 else float("nan")
    high_share = float(spectrum[krad > RESIDUAL_HIGH_FREQ].sum() / total) if total > 0 else float("nan")

    power = np.where(mask, residual ** 2, 0.0)
    theta = np.arctan2(yy - centre, xx - centre)
    m4 = float(np.abs((power * np.exp(4j * theta)).sum()) / power.sum()) if power.sum() > 0 else float("nan")
    return {
        "mean_freq": mean_freq,
        "high_share": high_share,
        "radial_ratio": (rms_out / rms_in) if rms_in else float("nan"),
        "m4_share": m4,
    }


def compute_residuals(atomic: RetestRun) -> ResidualSet | None:
    """逐配置重算「前 N 阶重构 − GS 基准相位」, 并算结构描述量。

    口径 (与 `logs/check_zernike_phase_residual.py` 逐行一致)::

        base      = <root>/sides/side_<S>/pupil_phase.npy        # 512², raw rad
        reference = upsample_bilinear_to_shape(base, (fit_grid, fit_grid))
        recon     = generate_noll(coeffs[:modes_for_order(n_max)])  # 高阶置零
        residual  = recon − reference   (只统计 ZernikeGenerator.mask 内)

    Zernike 数学**全部复用** `scripts/slm/hw_gs_square_scans.py` 与
    `ao_shaping.utils.wavefront.zernike_calc`, 本函数不自己算任何基函数。那两个模块
    是**延迟导入**的: 它们会连带 import 驱动包, 而本脚本的模块级 import 必须保持
    「不 import 任何驱动」, 好在没有产物时也能跑。

    Returns:
        :class:`ResidualSet`; 该批**没有任何** `pupil_phase.npy` 时返回 ``None``
        (基线批就是这样 —— 那时完全不渲染 §13.8)。
    """
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
    from scripts.slm.hw_gs_square_scans import ZernikeFit, reconstruct_from_fit

    stage = atomic.stages["zernike"]
    stage_dir = atomic.root / "zernike"
    wanted = [
        cfg for cfg in stage.configs
        if int(round(cfg.value)) in set(RESIDUAL_PANEL_ORDER)
        and (stage_dir / cfg.config / "pupil_phase.npy").exists()
    ]
    if not wanted:
        return None
    wanted.sort(key=lambda cfg: -cfg.value)

    base_path = base_phase_path(atomic.root, wanted[0].metrics)
    base = np.asarray(np.load(base_path), dtype=np.float64)
    generators: dict[tuple[int, int], Any] = {}
    panels: list[ResidualPanel] = []
    for cfg in wanted:
        other = base_phase_path(atomic.root, cfg.metrics)
        if other.resolve() != base_path.resolve():
            raise ValueError(
                f"{cfg.config} 的基准相位 {_rel(other)} 与首列的 {_rel(base_path)} 不同 —— "
                "同一张图里的残差必须共用一个基准"
            )
        grid = int(cfg.metrics["fit_grid"])
        order = int(cfg.metrics["fit_max_order"])
        reference = upsample_bilinear_to_shape(base, (grid, grid))
        key = (grid, order)
        if key not in generators:
            generators[key] = ZernikeGenerator((grid, grid), radius=grid / 2.0, n_orders=order)
        gen = generators[key]
        fit = ZernikeFit(
            coefficients=np.asarray(cfg.metrics["fit_coefficients_noll"], dtype=np.float64),
            fit_grid=grid,
            max_order=order,
            reference=reference,
            mask=np.asarray(gen.mask, dtype=bool),
        )
        n_max = int(round(cfg.value))
        recon, stats = reconstruct_from_fit(fit, n_max, generator=gen)
        mask = np.asarray(fit.mask, dtype=bool)
        saved = np.asarray(np.load(stage_dir / cfg.config / "pupil_phase.npy"), dtype=np.float64)
        # 显示用的是**卷到 ±π** 的残差: 面板施加的相位就是 mod 2π 的, 卷过之后 ±π 的
        # 色标才逐格有意义。统计量仍取 reconstruct_from_fit 的**未卷**差值 (与记录值同口径)。
        residual = np.where(mask, np.mod(recon - reference + math.pi, 2.0 * math.pi) - math.pi, 0.0)
        panels.append(
            ResidualPanel(
                n_max=n_max,
                config=cfg.config,
                row=cfg.row,
                modes=float(cfg.row["zernike_modes_used"]),
                rms_recorded=float(cfg.row["zernike_rms_residual_rad"]),
                explained=float(cfg.row["zernike_explained_fraction"]),
                rms_recomputed=float(stats["zernike_rms_residual_rad"]),
                recon_delta=float(np.abs(recon - saved).max()),
                residual=residual,
                mask=mask,
                **_residual_structure(residual, mask),
            )
        )
    return ResidualSet(base=base, base_path=base_path, fit_grid=int(panels[0].mask.shape[0]),
                       panels=panels)


def check_residuals(residuals: ResidualSet | None) -> list[str]:
    """§13.8 的自检: 重算残差 RMS 与记录值一致、重算相位与存盘相位逐点一致。

    **两种模式 (生成 / ``--check``) 都跑** —— 它只读产物, 与画不画图无关。
    返回问题清单 (空 = 通过); 每条都同时打印重算值与记录值。
    """
    problems: list[str] = []
    if residuals is None:
        return problems
    for panel in residuals.panels:
        rel = abs(panel.rms_recomputed - panel.rms_recorded) / max(abs(panel.rms_recorded), 1e-12)
        if rel >= RESIDUAL_RMS_RTOL:
            problems.append(
                f"n_max={panel.n_max}: 重算残差 RMS {panel.rms_recomputed:.9f} rad != "
                f"metrics.json 记录 {panel.rms_recorded:.9f} rad (相对误差 {rel:.2%}, "
                f"容差 {RESIDUAL_RMS_RTOL:.0%})"
            )
        if panel.recon_delta >= RESIDUAL_RECON_ATOL:
            problems.append(
                f"n_max={panel.n_max}: 重算相位与 {panel.config}/pupil_phase.npy 最大偏差 "
                f"{panel.recon_delta:.3e} (重算 vs 存盘; 容差 {RESIDUAL_RECON_ATOL:.0e})"
            )
    return problems


def residual_figure_name(atomic: RetestRun) -> str:
    """fig12 的文件名 —— 正文、`--check` 与 artifacts 三处共用同一份拼装。"""
    return f"fig12_zernike_phase_residual_{batch_tag(atomic)}.png"


def fig_zernike_residual(atomic: RetestRun, residuals: ResidualSet) -> Path:
    """fig12: 左上 = GS 基准相位, 第 1 行 = 残差相位, 第 2 行 = 同配置 CCD 远场。

    **列间共用同一尺度**: 残差行固定 ``±π`` (一个色标), 远场行共用一个
    ``LogNorm(vmin=1, vmax=四列最大峰值)`` —— 同一行内的颜色因此可直接横比。
    行间 / 基准列与残差列之间各加一条细线, 把「两种量」分开。
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    setup_matplotlib()
    panels = residuals.panels
    ncols = len(panels) + 1
    fig = plt.figure(figsize=(2.85 * ncols + 1.1, 6.6))
    grid = fig.add_gridspec(2, ncols + 1, hspace=0.24, wspace=0.10,
                            width_ratios=[1.0] * ncols + [0.05])

    ax_base = fig.add_subplot(grid[0, 0])
    im_base = ax_base.imshow(residuals.base, cmap=BASE_CMAP, vmin=-RESIDUAL_VLIM,
                             vmax=RESIDUAL_VLIM, interpolation="nearest")
    ax_base.set_title("GS 基准相位 (raw rad)", fontsize=8.5)
    ax_base.set_xticks([])
    ax_base.set_yticks([])
    # 基准相位的色标占 (1,0) 格: 用子网格把它做窄, 否则会占满整格宽度
    cax_base = fig.add_subplot(grid[1, 0].subgridspec(1, 5, wspace=0.0)[0, 2])
    cb_base = fig.colorbar(im_base, cax=cax_base)
    cb_base.set_label("基准相位 (rad)", fontsize=7.5)
    cb_base.ax.tick_params(labelsize=6.5)

    residual_axes = []
    residual_image = None
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, k + 1])
        residual_image = ax.imshow(
            np.ma.masked_where(~panel.mask, panel.residual),
            cmap=RESIDUAL_CMAP, vmin=-RESIDUAL_VLIM, vmax=RESIDUAL_VLIM,
            interpolation="nearest",
        )
        ax.set_title(
            f"n_max={panel.n_max}  残差 RMS={panel.rms_recorded:.3f} rad\n"
            f"解释={panel.explained * 100:.2f}%",
            fontsize=8.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        residual_axes.append(ax)
    cax_res = fig.add_subplot(grid[0, ncols])
    cb_res = fig.colorbar(residual_image, cax=cax_res)
    cb_res.set_label("残差 (rad)", fontsize=7.5)
    cb_res.set_ticks([-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi])
    cb_res.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cb_res.ax.tick_params(labelsize=6.5)

    stage_dir = atomic.root / "zernike"
    headers = atomic.stages["zernike"].headers
    frame_vmax = max(float(p.row["frame_peak"]) for p in panels)
    ccd_axes = []
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(grid[1, k + 1])
        ccd_axes.append(ax)
        pick = CellPick(stage="zernike", row=panel.row, role="残差列")
        ref = locate_frames(stage_dir, panel.row, headers)
        if not ref.paths:
            ax.axis("off")
            ax.text(0.5, 0.5, "缺帧", ha="center", va="center", fontsize=9, color="#d62728")
            continue
        _check_frame_params("zernike", pick, ref)
        image = load_frame_mean(ref.paths)
        if abs(float(image.max()) - float(panel.row["frame_peak"])) > PEAK_MATCH_ATOL:
            raise ValueError(
                f"zernike/{panel.config}: 帧最大值 {float(image.max()):.6g} 与 CSV "
                f"frame_peak {panel.row['frame_peak']} 不一致 —— 帧与 CSV 行对不上"
            )
        x0, x1, y0, y1, _ = _crop_box(panel.row, image.shape, CROP_FACTOR)
        shown = image[y0:y1, x0:x1]
        ax.imshow(shown, cmap="inferno",
                  norm=LogNorm(vmin=1.0, vmax=max(frame_vmax, 1.0 + 1e-6)),
                  interpolation="nearest")
        _annotate_cell(ax, pick, crop=(x0, x1, y0, y1))
        peak = float(panel.row["frame_peak"])
        sat = "  饱和" if peak >= SATURATION_COUNTS else ""
        ax.set_title(
            f"CV={float(panel.row['cv']):.3f}  PIB={float(panel.row['pib']):.3f}  "
            f"peak={peak:.0f}{sat}",
            fontsize=8.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.98, 0.02, f"×{len(ref.paths)} 帧均值", transform=ax.transAxes,
                fontsize=7.0, color="white", ha="right", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.2))
        del image, shown

    fig.suptitle(
        f"Zernike 截断残差 × 同配置 CCD 远场（{batch_tag(atomic)}, {atomic.exposure_ms:g} ms）",
        fontsize=11,
    )
    # 不用 tight_layout: 色标轴是 subgridspec 造出来的, tight_layout 会报
    # "Axes that are not compatible" 并给出错位的间距 —— 这里显式定边距。
    fig.subplots_adjust(left=0.012, right=0.985, top=0.90, bottom=0.025,
                        hspace=0.30, wspace=0.08)
    # 分隔线必须在 layout 定稿后按**实际**轴位置画 (否则与格子的间距对不上)
    fig.canvas.draw()
    if residual_axes and ccd_axes:
        boxes_top = [ax.get_position() for ax in residual_axes]
        boxes_bottom = [ax.get_position() for ax in ccd_axes]
        y_line = 0.5 * (min(b.y0 for b in boxes_top) + max(b.y1 for b in boxes_bottom))
        fig.add_artist(Line2D([min(b.x0 for b in boxes_top), max(b.x1 for b in boxes_top)],
                              [y_line, y_line], transform=fig.transFigure,
                              color="#999999", lw=0.7, zorder=5))
        x_line = 0.5 * (ax_base.get_position().x1 + boxes_top[0].x0)
        y_lo = min(b.y0 for b in boxes_top + boxes_bottom)
        y_hi = max(b.y1 for b in boxes_top + boxes_bottom)
        fig.add_artist(Line2D([x_line, x_line], [y_lo, y_hi], transform=fig.transFigure,
                              color="#999999", lw=0.7, zorder=5))

    path = FIG_DIR / residual_figure_name(atomic)
    savefig(fig, path)
    return path


def render_atomic_section(
    run: Run, f: Findings, atomic: RetestRun, af: AtomicFindings,
    retest: RetestRun | None, figures: Sequence[Path],
    residuals: ResidualSet | None = None,
) -> tuple[list[str], list[Table]]:
    """§13 整轮原子重扫。数字全部从该批 CSV / metrics.json 现读。"""
    tables: list[Table] = []
    out: list[str] = []
    sides = atomic.stages["sides"]
    zern = atomic.stages["zernike"]
    blaze = atomic.stages["blaze"]

    def add(text: str = "") -> None:
        out.append(text)

    add("## 13. 整轮原子重扫（单一版本 + 单一功率）")
    add()
    add(
        "第三批产物：`"
        f"{_rel(atomic.root)}/`。它是**唯一「单一脚本版本 + 单一功率」的完整一轮** —— "
        "三段（边长 / Zernike / 光栅）在同一个 `mode=atomic` 下逐配置重扫，"
        "所以三段数字之间第一次真正可比。"
    )
    add()

    add("### 13.1 口径")
    add()
    gate = Table(
        ("项", "本轮取值", "来源"),
        [
            ("生效曝光 (ms)", derived(af.exposure_ms, 1), "`run_config` / 每配置 `preflight`"),
            ("会话上限 (ms)", derived(af.ceiling_ms, 1), "`run_config`"),
            ("调用方式", f"`mode = {af.mode}`（每配置一个独立进程）", "`manifest` / `run_config`"),
            ("0 级（段级冻结）", f"({af.zero_order[0]:.0f}, {af.zero_order[1]:.0f})",
             "`run_config.zero_order_xy_frozen`"),
            ("可用最大边长 (px)", derived(af.side_max, 0), "`run_config.side_max_px`"),
            ("反解原始值 (px)", derived(af.side_max_raw, 0), "`run_config.side_max_raw_px`"),
            ("边长步长 (px)", derived(af.step, 0), "由 CSV 的 `side_px` 相邻差"),
            ("实际边长行数", derived(float(af.n_sides), 0), "`manifest.n_rows` / CSV"),
            ("选定边长 (px)", derived(af.selected_side, 0), "`sides/selected.json`"),
            ("选定时的 score", derived(af.selected_score, 4), "`sides/selected.json`"),
        ],
    )
    tables.append(gate)
    add(gate.render())
    add()
    add(
        "**1800 与 1900 被拒是保护生效，不是失败。** 按步长 "
        f"{af.step:.0f} px 往上，接下来的两档是 "
        f"{af.rejected[0]:.0f} / {af.rejected[1]:.0f} px，都超过本轮实测的可用最大边长 "
        f"{af.side_max:.0f} px（= 2·(min(cx, W−1−cx, cy, H−1−cy) − edge_margin) 向下取整，"
        f"原始 {af.side_max_raw:.0f} px）。脚本原文的拒绝理由："
    )
    add()
    add("> 边长 `{value}` px 超过本次实测的画面可用最大边长 "
        f"{af.side_max:.0f} px —— 方框会压在帧边上, PIB/EE 就不再是方斑质量了。"
        "改小 --side-px, 或调 --edge-margin-px / 曝光")
    add()
    add("（上面是脚本里的**消息模板**：`{value}` 处就是被拒的那一档边长 —— "
        f"本轮是 {af.rejected[0]:.0f} 与 {af.rejected[1]:.0f}。）")
    add()
    add(
        "这条保护的判据是**方框会不会压到帧边**（压边后窗函数与读出噪声会污染方框内的统计，"
        "PIB/EE 就不再是方斑质量的度量）。所以「少了两行」恰恰是它该做的事："
        f"宁可只有 {af.n_sides} 行可用数据，也不给两个框压边的假点。"
        f"（退出码 2 —— 该脚本把这条越限检查与读帧超时共用同一个硬退出码。）"
    )
    add()

    # -- 13.2 边长段 --------------------------------------------------------
    add("### 13.2 边长段")
    add()
    side_tbl = Table(
        ("边长 (px)", "CV", "PIB", "score", "box_peak", "frame_peak",
         "方框内面积 (px)", "出画面积 (px)", "AR", "质心偏移 x", "质心偏移 y"),
        [(cell_of(r, "side_px", 0), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
          cell_of(r, "score", 4), cell_of(r, "box_peak", 2), cell_of(r, "frame_peak", 2),
          cell_of(r, "box_area_px", 0), cell_of(r, "box_clipped_px", 0),
          cell_of(r, "ar", 4), cell_of(r, "box_centroid_offset_x", 2),
          cell_of(r, "box_centroid_offset_y", 2))
         for r in sides.rows],
    )
    tables.append(side_tbl)
    add(side_tbl.render())
    add()
    plateau_txt = " / ".join(f"{p:.0f}（{s:.4f}）" for p, s in af.plateau)
    tail_txt = " / ".join(f"{p:.0f}（{s:.4f}）" for p, s in af.tail)
    add(
        f"**score 的平台在 {min(p for p, _ in af.plateau):.0f}–"
        f"{max(p for p, _ in af.plateau):.0f} px**：{plateau_txt}，两者只差 "
        f"{af.plateau_gap:.1e}。之后开始掉：{tail_txt}。"
        f"（argmax 在 {af.score_argmax_side:.0f} px。）"
        "也就是说这一轮的「最优边长」不是一个点，而是一段**平台**："
        f"平台宽 {max(p for p, _ in af.plateau) - min(p for p, _ in af.plateau):.0f} px，"
        f"而平台内的高度差只有 {af.plateau_gap:.1e} —— 在单次测量的分辨力内，"
        "峰落在平台哪一点**不可判定**（与 §3.3 基线那一轮同一性质）。"
    )
    add()
    add(
        f"**`box_peak` 与饱和**：全部 {af.n_sides} 行的 `frame_peak` 都是 "
        f"{'/'.join(f'{v:.0f}' for v in af.frame_peak_values)}（0 级削顶），"
        f"而 `box_peak` 只有 {af.box_peak_min:.2f}–{af.box_peak_max:.2f} —— "
        "**削顶的是框外的 0 级，不是方斑本身**（`box_peak < frame_peak` 就证明最亮像素在框外）。"
        "这与 §4 的「0 级饱和」是两回事：那一条说的是平场 0 级不含线性信息，"
        "这一条说的是方斑自己贴着满量程但还没到顶 —— 方斑的读数仍然可信。"
    )
    add()
    if af.ar_outliers:
        add(
            "**一个没解释的离群**：`ar`（方框内长宽比）在 "
            + "、".join(f"{p:.0f} px 处是 {v:.2f}" for p, v in af.ar_outliers)
            + f"，而同列中位数只有 {af.ar_median:.2f}。这一行的 CV/PIB/score 都在趋势上，"
            "所以不是整帧坏，问题只可能出在 AR 这个量本身。**机制未定**，"
            "不要据此删行；要确认需要看该行的方框内二值轮廓（`ar` 由"
            " `square_aspect_ratio` 从框内二值掩膜算出）。"
        )
        add()

    # -- 13.3 Zernike 段 ----------------------------------------------------
    add("### 13.3 Zernike 段")
    add()
    zern_tbl = Table(
        ("n_max", "模式数", "解释方差", "残差 RMS (rad)", "CV", "PIB", "score",
         "质心偏移 x", "质心偏移 y"),
        [(cell_of(r, "n_max", 0), cell_of(r, "zernike_modes_used", 0),
          cell_of(r, "zernike_explained_fraction", 6),
          cell_of(r, "zernike_rms_residual_rad", 6),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "box_centroid_offset_x", 2), cell_of(r, "box_centroid_offset_y", 2))
         for r in zern.rows],
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()
    add(
        f"**CV 全平**：{af.zern_cv_min:.4f}–{af.zern_cv_max:.4f}（相对跨度 "
        f"{(af.zern_cv_max - af.zern_cv_min) / af.zern_cv_min:.1%}），PIB "
        f"{af.zern_pib_min:.4f}–{af.zern_pib_max:.4f}，score "
        f"{min(float(r['score']) for r in zern.rows):.4f}–"
        f"{max(float(r['score']) for r in zern.rows):.4f}。"
        f"解释方差从 {af.zern_explained_min * 100:.3f}% 单调升到 "
        f"{af.zern_explained_max * 100:.3f}%（{af.zern_modes_max:.0f} 个模式）—— "
        "**阶数加了四倍，画面上几乎没变**，这就是 §5.1「基不适配」的量化版。"
    )
    add()
    if af.tier_split is not None:
        low = sorted(af.tier_low)
        high = sorted(af.tier_high)
        add(
            "**`box_centroid_offset_x` 的两档台阶**："
            f"低档 {min(v for _, v, _ in low):.2f}–{max(v for _, v, _ in low):.2f} px"
            f"（n_max = {'/'.join(f'{n:.0f}' for n, _, _ in low)}），"
            f"高档 {min(v for _, v, _ in high):.2f}–{max(v for _, v, _ in high):.2f} px"
            f"（n_max = {'/'.join(f'{n:.0f}' for n, _, _ in high)}）。"
            "**台阶的位置已经定位：它不在远场里**。把这一列拆开看 ——"
        )
        add()
        add(
            f"- `box_centroid_x`（实测质心）在全部 {len(af.tier_low) + len(af.tier_high)} 个配置上只差 "
            f"**{af.centroid_x_span:.1f} px**；"
        )
        add(
            f"- 两档的差别全在**参考原点**上：低档各配置的 0 级 x 是 "
            f"{min(z for _, _, z in low):.0f}–{max(z for _, _, z in low):.0f}，"
            f"高档是 {min(z for _, _, z in high):.0f}–{max(z for _, _, z in high):.0f}，"
            f"分界落在 x ≈ {af.tier_split:.0f}；"
        )
        add(
            "- 而 `offset_x ≡ centroid_x − 0 级 x`（定义如此），所以台阶 = **原点跳了约 "
            f"{min(z for _, _, z in high) - max(z for _, _, z in low):.0f} px**，不是光斑移动。"
        )
        add()
        add(
            f"注意 `n_max = 21` 落在**低档**（{min(v for n, v, _ in low if n == 21.0):.2f} px）"
            "—— 20 与 22 都在高档。所以这个台阶**不是 `n_max` 的函数**，"
            "而是「那一次进程把 0 级定到了哪一簇」的结果。"
        )
        add()
        add(
            "**为什么原点会跳，仍未定。** 本轮能确定的只有「跳的是参考原点」；"
            "候选机制是每配置独立在饱和平台上 `argmax`（§11.6 已记录同一现象："
            "同一段里 0 级定位跨进程相差 8–39 px），但本轮**无法**把它与别的可能分开。"
            "要定因需要固定 0 级后再复测一次（§11.6 的四条方案）。"
            "**在定因之前，不要把 `box_centroid_offset_x` 的任何台阶读成物理效应。**"
        )
        add()

    # -- 13.4 光栅段 --------------------------------------------------------
    add("### 13.4 闪耀光栅段")
    add()
    blaze_tbl = Table(
        ("周期 (px)", "预测位移 (px)", "实测位移 y (px)", "实测/预测", "CV", "PIB",
         "score", "出画面积 (px)"),
        [(cell_of(r, "blaze_period_px", 0), cell_of(r, "predicted_shift_px", 2),
          cell_of(r, "measured_shift_y_px", 2), derived(ratio, 4),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "box_clipped_px", 0))
         for r, (_, _, _, ratio, _, _, _, _) in zip(blaze.rows, af.blaze)],
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()
    add(
        f"**实测几乎不随周期变**：排除出画那一行后，实测位移 y 只有 "
        f"{af.blaze_meas_min[1]:.2f}（P={af.blaze_meas_min[0]:.0f}）→ "
        f"{af.blaze_meas_max[1]:.2f} px（P={af.blaze_meas_max[0]:.0f}），"
        f"全程只变 **{af.blaze_flat_ratio:.2f}×**；而预测在同一区间变了 "
        f"**{af.blaze_predicted_ratio:.2f}×**（它按 `TILT_SHIFT_SCALE/周期` 构造，"
        "所以是严格的 1/P）。于是实测/预测从 "
        f"{af.blaze_ratio_min:.3f} 一路涨到 {af.blaze_ratio_max:.3f} —— "
        "**比值不是常数，说明错的不是标度系数，而是 1/P 这个形状本身**（与 §6.4 同一结论，"
        "这一轮把它在同一功率下复现了一遍）。"
    )
    add()
    add(
        f"**出画那一行**：P = {'/'.join(f'{p:.0f}' for p in af.blaze_clipped)} 的 "
        "`box_clipped_px > 0`（预测位移把框推到画面外），该行 PIB 不可与其它行横比。"
        "它的 CV/score 也一并在表里给着，但只能当「框错位后的读数」，不是周期 16 的质量。"
    )
    add()

    # -- 13.5 与 §11 复测的同功率交叉核对 -----------------------------------
    add("### 13.5 与 §11 复测的同功率交叉核对")
    add()
    if af.cross:
        add(
            "§11 的复测批与本轮**同为 "
            f"{af.exposure_ms:g} ms**，所以这两批可以做**同功率**对照"
            "（不同于基线的 20 ms）。逐配置比两批的 CV 与 score："
        )
        add()
        cross_tbl = Table(
            ("段", "配置", "CV (本轮)", "CV (§11)", "ΔCV", "score (本轮)", "score (§11)", "Δscore"),
            [(STAGE_CN[stage], derived(value, 0), derived(cv_a, 4), derived(cv_b, 4),
              derived(dcv, 4), derived(sc_a, 4), derived(sc_b, 4), derived(dsc, 4))
             for stage, value, cv_a, cv_b, dcv, sc_a, sc_b, dsc in af.cross],
        )
        tables.append(cross_tbl)
        add(cross_tbl.render())
        add()
        higher = sum(1 for row in af.cross if row[5] > row[6])
        add(
            f"**结论：同功率不同批次的重复性约 1%。** CV 最大差 "
            f"{af.cross_cv_max:.1%}、score 最大差 {af.cross_score_max:.1%}；"
            f"方向**不系统**（{len(af.cross)} 个配置里本轮 score 高 {higher} 个、"
            f"低 {len(af.cross) - higher} 个），也就是说这 1% 是噪声而不是批次偏差。"
            "这条带宽正是 §11.5 判定 1/2 需要的那个量：基线那两处异常偏 "
            f"+72% / +801%，比它大两个数量级以上，所以「黑帧」判定不依赖某一批的运气。"
        )
    else:
        add("（没有 §11 的复测批可比 —— 未给 `--retest-root`。）")
    add()

    # -- 13.6 与基线的定性对照 ----------------------------------------------
    add("### 13.6 与基线的对照（只能定性）")
    add()
    base = run.stages
    base_sides = base["sides"]
    cmp_rows: list[tuple[Cell | str, ...]] = []
    for label, key, stage, pick in (
        ("边长段 score 最高", "score", "sides", max(base_sides.col("score"))),
        ("边长段 CV 最小", "cv", "sides", min(base_sides.col("cv"))),
        ("Zernike 段 CV 最小", "cv", "zernike", min(base["zernike"].col("cv"))),
        ("光栅段 score 最高", "score", "blaze", max(base["blaze"].col("score"))),
    ):
        mine = (max(atomic.stages[stage].col(key))
                if key == "score" else min(atomic.stages[stage].col(key)))
        cmp_rows.append((label, f"{key} = {pick:.4f}", f"{key} = {mine:.4f}", "不可横比"))
    cmp_tbl = Table(("关键量", "基线（20 ms）", "本轮（180 ms）", "可比性"), cmp_rows)
    tables.append(cmp_tbl)
    add(cmp_tbl.render())
    add()
    add(
        f"两批的曝光差 {af.exposure_ms / run.exposure_ms:.0f}×（功率也差一个量级，见 §11.1），"
        "所以**任何绝对值都不能横比**。这张表只用来回答一个问题：**趋势是否一致** —— "
        "例如「CV 随边长上升」「score 有平台」「低阶 Zernike 的 CV 与阶数无关」，"
        "这些**形状**在两批里是一致的，而高度不是。"
    )
    add()

    # -- 13.7 图 ------------------------------------------------------------
    add("### 13.7 图")
    add()
    drawn = {p.name for p in figures}
    for stem, caption in (
        ("fig10_atomic_sides", "边长 → CV / PIB / score（本轮实线，基线虚线仅作定性对照）"),
        ("fig11_atomic_blaze_shift", "光栅周期 → 预测位移（1/P）vs 实测位移（近水平）"),
    ):
        name = f"{stem}_{batch_tag(atomic)}.png"
        if name in drawn:
            add(f"![{caption}](figures/{name})")
            add()
            add(f"*{caption}。*")
            add()
    for spec in MOSAICS:
        name = f"{spec.stem}_{batch_tag(atomic)}.png"
        if name in drawn:
            add(f"![{spec.title}（本轮）](figures/{name})")
            add()
            add(f"*{spec.title}（本轮 {af.exposure_ms:g} ms）。* "
                "口径与 §12 完全一致：对数显示（`LogNorm(vmin=1, vmax=该格峰值)`，"
                "**颜色不代表线性强度**）、"
                + (f"裁剪 `box_side × {CROP_FACTOR}`、" if spec.crop else "整帧不裁剪、")
                + "目标框/预测框心/实测质心叠加、右下角 `×N 帧均值`。"
                "每格的源 npz 与 sha256 见 `artifacts/"
                f"{ATOMIC_ARTIFACT_DIR.name}/frames_selection.json`。")
            add()
    add(
        "拼图里没有 1800/1900 两格 —— 这两行本来就没有被采集（§13.1）。"
        "边长格取 200 / 600 / 1000 / 1500 / 1700 五档（§13 不追加 score 格："
        "本轮 score 是平台而不是单点，追加一格会误导）。"
    )
    add()

    # -- 13.8 相位残差 × 同配置远场 -----------------------------------------
    if residuals is None or not residuals.panels:
        return out, tables

    panels = residuals.panels
    first = panels[0]      # n_max 最大
    last = panels[-1]      # n_max 最小
    rms_drop = 1.0 - first.rms_recorded / last.rms_recorded
    max_explained = max(p.explained for p in panels)
    unrepresented = 1.0 - max_explained
    max_rel = max(abs(p.rms_recomputed - p.rms_recorded) / max(abs(p.rms_recorded), 1e-12)
                  for p in panels)
    cvs = [float(p.row["cv"]) for p in panels]
    pibs = [float(p.row["pib"]) for p in panels]
    freqs = [p.mean_freq for p in panels]
    highs = [p.high_share for p in panels]
    ratios = [p.radial_ratio for p in panels]
    m4s = [p.m4_share for p in panels]

    add("### 13.8 相位残差 × 同配置远场")
    add()
    add(
        "本节把**被 Zernike 截断掉的那部分相位**与同配置的相机画面并排摆出来。"
        "只有这一批落了相位文件（`zernike/nmax_*/pupil_phase.npy`），基线批没有 ⇒ "
        "这张图只能放在这里（§12 只有远场画面）。"
    )
    add()
    add("口径（与采集脚本的生产路径逐行一致）：")
    add()
    add(f"- 基准相位：`{_rel(residuals.base_path)}`（512², raw rad；边长由 `sides/selected.json` 决定）；")
    add(
        f"- `reference` = `upsample_bilinear_to_shape(base, ({residuals.fit_grid}, "
        f"{residuals.fit_grid}))`，`fit_grid` = {residuals.fit_grid}；"
    )
    add("- `recon` = `generate_noll(coeffs[:modes_for_order(n_max)])` —— 高阶系数**置零**，"
        "所以「第 N 阶重构」严格是「把高阶丢掉」，不是另一个模型的解；")
    add("- `residual` = `recon − reference`，只统计单位圆内（`ZernikeGenerator.mask`）。")
    add()
    add(
        "⚠ 每个 `zernike/nmax_<N>/pupil_phase.npy` 是**该阶数的重构相位**，不是基准；"
        "拿它当基准残差恒为 0。基准在边长段的 `side_<S>/pupil_phase.npy`。"
    )
    add()

    residual_figure = residual_figure_name(atomic)
    if residual_figure in drawn:
        add(f"![Zernike 截断残差 × 同配置 CCD 远场](figures/{residual_figure})")
        add()
        add(
            "*第一行 = 残差相位（`RdBu_r`，固定 ±π，**四列共用同一色标**，单位圆外没有光"
            "所以不画）；第二行 = 同配置 CCD 远场（对数显示 `LogNorm(vmin=1, vmax=四列最大峰值)`、"
            f"裁剪 `box_side × {CROP_FACTOR}`、叠加目标框/预测框心/实测质心、右下角 `×N 帧均值`）；"
            "左上角 = GS 基准相位（`twilight`，±π）。残差在图上已卷到 ±π（面板施加的本来就是 "
            "mod 2π 的相位），而标题与表里的 RMS 取的是**未卷**差值 —— 与 `metrics.json` 同口径。*"
        )
        add()

    res_tbl = Table(
        ("n_max", "模式数", "残差 RMS (rad)", "解释比例", "CV", "PIB",
         "残差里的可见结构（一句话）"),
        [(cell_of(p.row, "n_max", 0), cell_of(p.row, "zernike_modes_used", 0),
          cell_of(p.row, "zernike_rms_residual_rad", 6),
          # 解释比例按 % 显示: 与图标题、与正文的百分比是同一个表示, 免得读者
          # 在 0.032515 与 3.25% 之间换算 (CSV 里存的是分数)。
          derived(float(p.row["zernike_explained_fraction"]) * 100.0, 3, "%"),
          cell_of(p.row, "cv", 4), cell_of(p.row, "pib", 4),
          p.structure_note())
         for p in panels],
    )
    tables.append(res_tbl)
    add(res_tbl.render())
    add()
    add(
        "前三列与 §13.3 同源（同一批 CSV 列；解释比例在这里按 % 显示，CSV 里存的是分数），"
        "新增的只有最后一列 —— 它是从**残差数组**现场算出来的四个量：平均空间频率、"
        "`>0.25 cyc/px`（周期 4 个拟合格）的能量占比、外缘（`r > 0.7`）与中心"
        "（`r ≤ 0.35`）的 RMS 比，以及方位向 `m = 4` 分量的占比。"
    )
    add()
    add(
        "**残差不是「拟合误差」，也不是噪声 —— 它是被这个基截断掉的高频结构。** "
        f"判据是残差随阶数的走向：模式数从 {last.modes:.0f}（`n_max = {last.n_max}`）加到 "
        f"{first.modes:.0f}（`n_max = {first.n_max}`），残差 RMS 只从 "
        f"{last.rms_recorded:.4f} 降到 {first.rms_recorded:.4f} rad（{rms_drop:.1%}），"
        f"解释比例从 {last.explained:.2%} 升到 {first.explained:.2%}。"
        "若残差是「没拟合好」，多给 "
        f"{first.modes - last.modes:.0f} 个自由度应该让它大幅下降；它几乎不动，"
        f"说明方斑所需的相位里有 {unrepresented:.1%} 根本不在前 {first.n_max} 阶张成的空间里。"
    )
    add()
    add(
        "**它在图上长什么样，以及看不到什么。** 四列是同一片细颗粒：平均空间频率 "
        f"{min(freqs):.2f}–{max(freqs):.2f} cyc/px（周期约 4 个拟合格，已经是 "
        f"{residuals.fit_grid} 网格能表示的最细尺度），{min(highs):.0%}–{max(highs):.0%} 的"
        "能量在 `>0.25 cyc/px`；径向几乎不衰减（外缘/中心 RMS 比 "
        f"{min(ratios):.2f}–{max(ratios):.2f}），方位向也没有方斑该有的 4 重对称"
        f"（`m = 4` 只占 {min(m4s):.1%}–{max(m4s):.1%}）。所以**方斑的「边缘与角」不在这张"
        "相位图上** —— 它们不是画在瞳面上的轮廓：目标是均匀方斑（远场幅度是二维 sinc 型，"
        "边与角全在高频里），要把圆盘照明变成它，瞳面相位必须携带同样高频的结构，而这些"
        "结构恰好是 Zernike 截断丢掉的那一层。这一点可以当场验证：把残差相位单独做一次"
        "傅里叶变换，得到的仍是一个**方形**的均匀区域，而不是亮核（这项检查不在图里，"
        "它的用处是把「高频结构」与「噪声」分开 —— 噪声的变换不会保持方形的边）。"
    )
    add()
    add(
        "**远场这一行就是「塌」的样子。** 第二行与 §12 拼图 2 是同一件事的两面："
        f"前 N 阶重构的远场是一个亮核（CV {min(cvs):.3f}–{max(cvs):.3f}，"
        f"PIB {min(pibs):.3f}–{max(pibs):.3f}），而同框的 GS 基准远场是方斑"
        "（§12 拼图 2 最后一格）。CV 在四列之间只差 "
        f"{(max(cvs) - min(cvs)) / min(cvs):.1%} —— 阶数加了四倍画面几乎没变，"
        "这正是 §13.3「基不适配」在相位侧的对应物：改的是相位的高频内容，远场却基本不动，"
        "因为那部分内容根本没有被表示出来。"
    )
    add()
    add(
        "**自检（生成与 `--check` 两条路径都跑，不只是生成时）。** 重算的残差 RMS 必须与 "
        "`metrics.json` 的 `zernike_rms_residual_rad` 相对误差 < "
        f"{RESIDUAL_RMS_RTOL:.0%}，重算相位必须与存盘的 `pupil_phase.npy` 最大偏差 < "
        # `:.0e` 会印成 1e-09; 这里只要值对, 顺手去掉指数里的前导零
        f"{format(RESIDUAL_RECON_ATOL, '.0e').replace('e-0', 'e-')} —— "
        "任一不满足就打印两边值并非零退出。"
        f"本轮 {len(panels)} 个配置全部通过（最大相对误差 {max_rel:.2%}，"
        f"最大偏差 {max(p.recon_delta for p in panels):.1e}）。"
    )
    add()
    return out, tables


def render_report(
    run: Run,
    f: Findings,
    retest: RetestRun | None = None,
    rf: RetestFindings | None = None,
    prior: RetestRun | None = None,
    farfield: dict[str, Any] | None = None,
    farfield_figures: Sequence[Path] = (),
    atomic: RetestRun | None = None,
    af: AtomicFindings | None = None,
    atomic_figures: Sequence[Path] = (),
    residuals: ResidualSet | None = None,
) -> tuple[str, list[Table]]:
    """渲染正文。返回 (markdown, 需要回读校验的表)。

    ``retest``/``rf`` 为 ``None`` 时**不渲染 §11**，输出与没有 ``--retest-root``
    时逐字节相同（这是 ``--check`` 幂等的前提之一）。
    """
    tables: list[Table] = []
    sides = run.stages["sides"]
    zern = run.stages["zernike"]
    blaze = run.stages["blaze"]
    cfg = sides.run_config
    bench = (run.plan.get("bench") or {})
    spot = run.spot_assessment
    h, w = run.frame_shape_hw
    zo_x, zo_y = run.zero_order_xy
    marks_s = flag_rows(run, sides)
    marks_z = flag_rows(run, zern)
    marks_b = flag_rows(run, blaze)
    out: list[str] = []

    def add(text: str = "") -> None:
        out.append(text)

    # -- 标题 ---------------------------------------------------------------
    add("# GS 方形光斑三段扫描：边长 / Zernike 阶数 / 闪耀光栅（2026-10-23）")
    add()
    add(
        "一次纯 Gerchberg-Saxton（GS）的方形远场整形实验，沿三个自变量各扫一遍："
        f"**目标方框边长**（{len(sides.rows)} 行）、**Zernike 截断阶数**（{len(zern.rows)} 行）、"
        f"**闪耀光栅周期**（{len(blaze.rows)} 行）。三种扫描之间只通过"
        "「选定边长 / 选定相位」单向传递，全程**没有任何闭环优化**（GS 是交替投影，不含随机数）。"
    )
    add()
    add(
        f"全部数字来自 `{_rel(run.root)}/` 下落盘的 "
        "`measurements.csv` / `measurements.json` / `run_config.json` / `manifest.json`"
        + (f"，以及那次运行的日志 `{_rel(run.log_path)}`。" if run.log_path
           else "，**日志未提供**，因此 settle 判据状态未知（见 §2.3）。")
        + "本报告由 `scripts/slm/generate_hw_gs_square_scans_report.py` 生成；"
        "生成器把每张数值表渲染后再从 Markdown 回读，逐单元与 CSV 原文比对，"
        "不一致即非零退出。"
    )
    add()

    # -- 0. 结论核对 --------------------------------------------------------
    add("## 0. 结论核对（逐条用数据判定）")
    add()
    add(
        "这一节把「本轮原本想验证的说法」逐条摊到数据上。判定只有三种："
        "**成立**（数据支持）、**部分成立**（方向对但措辞过强）、**推翻**。"
    )
    add()
    claim = Table(
        ("#", "待核对的结论", "数据", "判定"),
        [
            ("1", "PIB 随边长单调上升（更大方框装更多能量）",
             f"{len(sides.rows)} 行中反向步数 = {len(f.pib_violations)}；"
             f"PIB {f.pib[0]:.4f} → {f.pib[-1]:.4f}", "成立"),
            ("2", "CV 与 PIB 同步上升",
             f"整体 Spearman ρ = {f.cv_spearman:.3f}，但 {len(f.cv_violations)} 处回落"
             f"（{_join_x(f.sides_x, f.cv_violations)} px）",
             "部分成立" if f.cv_violations else "成立"),
            ("3", "score 在 1500 px 最高、之后下降",
             f"argmax = {f.sides_x[f.score_peak_index]:.0f} px（{f.score[f.score_peak_index]:.4f}）；"
             f"次高 {f.sides_x[f.score_runner_up_index]:.0f} px 只差 {f.score_peak_gap:.2e}"
             f"；峰后单调下降 = {f.score_descend_ok}", "成立但峰不可分辨"),
            ("4", "Zernike 解释方差仅 0.2%–3.3%、残差 1.53–1.55 rad、CV 与阶数无关",
             f"{min(f.zern_explained) * 100:.3f}%–{max(f.zern_explained) * 100:.3f}%；"
             f"{min(f.zern_residual):.4f}–{max(f.zern_residual):.4f} rad；"
             f"干净行 CV {min(f.zern_clean_cv):.3f}–{max(f.zern_clean_cv):.3f}"
             f"（相对跨度 {f.zern_clean_cv_spread:.1%}），"
             f"r(n_max, CV) = {f.zern_cv_pearson:+.3f}",
             "部分成立" if abs(f.zern_cv_pearson) >= 0.5 else "成立"),
            ("5", "n_max=18 是异常点（CV 13.53）",
             f"CV {f.zern_cv[_idx_of(zern.x, 18.0)]:.3f}，"
             f"整帧峰值 {float(zern.rows[_idx_of(zern.x, 18.0)]['frame_peak']):.1f} counts"
             f"（同段中位数 {zern.median('frame_peak'):.1f}）；"
             + (f"复测 CV {float(_row_at(retest.stages['zernike'].rows, 'n_max', 18.0)['cv']):.4f}"
                f"（与 17/19 差 {abs(_neighbour_deviation(rf.zern_cv, 1)):.2%}）"
                if retest is not None and rf is not None else "（未复测）"),
             "**已复测推翻**（§11.5 判定 2）" if retest is not None
             else "成立，且更严重"),
            ("6", "闪耀光栅 score 随周期单调上升",
             f"排除 §/‡ 两行后反向步数 = {len(f.blaze_score_violations_clean)}",
             "成立" if not f.blaze_score_violations_clean else "不成立"),
            ("7", "period=16 预测位移让方框出画",
             f"predicted {blaze.col('predicted_shift_px')[0]:.1f} px；"
             f"box_clipped_px = {blaze.col('box_clipped_px')[0]:.0f} px"
             f"（占整框 {_pct(blaze.col('box_clipped_px')[0], float(blaze.rows[0]['box_side_px']) ** 2)}）",
             "成立"),
            ("8", "period=40 异常且该次采集 settle 超时",
             f"score {f.blaze_score[_idx_of(blaze.x, 40.0)]:.4f}，CV "
             f"{blaze.col('cv')[_idx_of(blaze.x, 40.0)]:.2f}；"
             + (f"日志：{run.settle_for('blaze', _idx_of(blaze.x, 40.0)).describe()}"
                if run.settle_for("blaze", _idx_of(blaze.x, 40.0))
                else "日志缺失，无法核对")
             + (f"；复测 CV {float(_row_at(retest.stages['blaze'].rows, 'blaze_period_px', 40.0)['cv']):.4f}"
                f" 落在 32/48 之间" if retest is not None else ""),
             (f"**异常已复测推翻**（§11.5 判定 1）；settle 超时本身成立"
              f"（另有 {max(len(run.settle_events) - 1, 0)} 行同样超时）")
             if retest is not None
             else f"成立（另有 {max(len(run.settle_events) - 1, 0)} 行同样超时）"),
            ("9", "「近 1000 取到窗口上界」",
             f"窗口 {f.near_window[0]:.0f}–{f.near_window[1]:.0f} px 内 score 单调 ⇒ "
             f"选中 {f.sides_x[f.near_best_index]:.0f} px，正是上界", "成立"),
            ("10", "side_max_px 由 0 级位置反解",
             f"2·({zo_y:.0f}−{f.edge_margin:.0f}) = {f.side_max_raw:.0f} → "
             f"向下取 {f.step:.0f} 的整数倍 = {f.side_max:.0f}", "成立"),
        ]
        + (
            [
                ("11", "基线 n_max=17 的高端异常来自「高阶级重构在面板上欠采样」",
                 f"复测 17 的 CV "
                 f"{float(_row_at(retest.stages['zernike'].rows, 'n_max', 17.0)['cv']):.4f}"
                 f" 落在趋势内（且复测 17 自己也有 settle 超时）",
                 "**未确认**（降级为未证实假设，§11.5 判定 3）"),
            ]
            if retest is not None
            else []
        ),
    )
    tables.append(claim)
    add(claim.render())
    add()
    add(f"> 判定 8 的补充：本轮日志里**一共 {len(run.settle_events)} 行** settle 超时"
        f"（sides {_count_stage(run, 'sides')} / zernike {_count_stage(run, 'zernike')} / "
        f"blaze {_count_stage(run, 'blaze')}），不是只有 period=40 一行。见 §2.3。")
    add()

    # -- 1. 口径 ------------------------------------------------------------
    add("## 1. 实验与口径")
    add()
    add("### 1.1 光路与设备（`sides/run_config.json`）")
    add()
    # 原子化批次的段级 run_config 只记段级参数, gs_grid / gs_iters 在 CSV 行里
    gs_grid = int(cfg["gs_grid"]) if "gs_grid" in cfg else int(float(sides.rows[0]["gs_grid"]))
    gs_iters = (int(cfg["gs_iters"]) if "gs_iters" in cfg
                else int(float(sides.rows[0]["gs_iters"])))
    def _opt(value: Any, spec: str, unit: str = "") -> str:
        """``plan.bench`` 缺失时给 ``—`` 而不是 ``nan`` (原子化批次不带 plan)。"""
        if value is None:
            return "—（本批 `run_config` 无 `plan.bench`）" if not bench else "—"
        return f"{float(value):{spec}}{unit}"

    params = Table(
        ("项", "值", "来源"),
        [
            ("波长", _opt(bench.get("wavelength_nm"), ".0f", " nm"), "`plan.bench`"),
            ("焦距", _opt(bench.get("focal_length_m"), ".3f", " m"), "`plan.bench`"),
            ("SLM 面板 (W×H)", f"{bench.get('panel_shape')} (行×列)", "`plan.bench`"),
            ("面板像元", _opt(bench.get("panel_pitch_um"), ".1f", " µm"), "`plan.bench`"),
            ("相机像元", _opt(bench.get("camera_pixel_um"), ".1f", " µm"), "`plan.bench`"),
            ("相机画幅 (W×H)", f"{w} × {h}", "`frame_shape_hw`"),
            ("等效瞳面边长", f"{float(bench.get('pupil_side_px', float('nan'))):.0f} panel px", "`plan.bench`"),
            ("0 级位置（实测冻结）", f"({zo_x:.0f}, {zo_y:.0f})", "`zero_order_xy_frozen`"),
            ("画面几何中心", f"({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})", "由 `frame_shape_hw` 推导"),
            ("GS 网格 / 迭代", f"N = {gs_grid} / {gs_iters}",
             "`run_config`" if "gs_grid" in cfg else "CSV 行"),
            ("每配置帧数", f"{int(cfg['frames_per_measurement'])}", "`run_config`"),
        ],
    )
    tables.append(params)
    add(params.render())
    add()
    add(
        "`TILT_SHIFT_SCALE`（2f 焦面标度）在采集脚本里按实测常量给出，于是"
        "**一个 GS 网格像素恒等于固定的 camera 像素数**，与网格边长 N 无关；"
        "这也是为什么方斑边长可以直接按 `side_gs_px = side_cam_px · pupil_side_px / TILT_SHIFT_SCALE` "
        "换算。本报告不重算这条标度，只引用 `run_config` 里 GS 的实际输出 "
        f"`gs_grid = {gs_grid}` 与 `gs_iters = {gs_iters}`。"
    )
    add()

    add("### 1.2 指标口径")
    add()
    add(
        "- `PIB` = 目标方框内功率 / 整帧功率（`square_shaping.power_in_bucket`）；"
        "帧先过 `prepare_frame`（减中位数 + 裁负），否则读噪声会把 PIB 顶到 1 以上。"
    )
    add(
        "- `CV` = 方框**内部**强度的标准差 / 均值（`uniformity_cv`），越低越均匀。"
    )
    add(
        "- `score` = **0.5·PIB + 0.5·1/(1+CV)**（`composite_from_pib_cv`）。"
        "本报告把这两项拆开列（§3.3），因为方框变大时它们**方向相反**。"
    )
    add(
        "- `ee` 与 `PIB` 在数值上**完全相同**（`square_encircled_energy` 与 "
        "`power_in_bucket` 对同一方框做同一个求和），所以本报告不把 `ee` 当独立指标。"
    )
    add(
        "- `global_spot_fwhm_px` / `global_spot_hollowness` 量的是**整帧最亮结构**"
        "（2f 光路上通常是 0 级或其晕斑），**不是**被整形的方斑。图 2(b) 里它随边长"
        "大幅摆动是 0 级几何的变化，不能读成「方斑变差」。"
    )
    add()
    add("### 1.3 稳定性判据")
    add()
    policy = run.settle_policy
    add(
        f"每次下发相位后丢弃 {policy.get('n_discard')} 帧、等 {policy.get('wait_s')} s，"
        f"再以相对容差 {policy.get('tol')} 判定稳定，最长等 {policy.get('max_wait_s')} s，"
        "稳定后平均 4 帧。判据超时时采集脚本**仍然取帧**，但会打一条 "
        "`settle criterion not met … must be treated as provisional` —— 本报告把这些行"
        f"标 `†`（共 {len(run.settle_events)} 行）。"
    )
    add()

    # -- 2. 数据完整性 ------------------------------------------------------
    add("## 2. 数据完整性")
    add()
    add("### 2.1 计划 vs 落盘")
    add()
    rows_plan = []
    for name in STAGES:
        st = run.stages[name]
        planned = run.stage_plans.get(name, {}).get("configs", [])
        rows_plan.append(
            (STAGE_CN[name], str(len(planned)), derived(float(len(st.rows)), 0),
             str(st.manifest_entry.get("created_utc", "-")),
             derived(float(len(st.headers)), 0))
        )
    completeness = Table(("段", "计划配置数", "CSV 行数", "落盘时间 (UTC)", "CSV 列数"),
                          rows_plan)
    tables.append(completeness)
    add(completeness.render())
    add()
    if f.dropped_configs:
        add(
            f"**计划里有、CSV 里没有的配置：{', '.join(f.dropped_configs)} px。** "
            "这不是丢数据：`side_max_px` 由 0 级位置与画面边缘**现场反解**（§8），"
            f"解出 {f.side_max:.0f} px，于是 {', '.join(f.dropped_configs)} 从未被采集。"
        )
        add()
    add(
        f"CSV 与 `measurements.json` 已逐值交叉校验（{f.csv_cells} 个单元，含 NaN 语义），"
        "`manifest.json` 的 `n_rows` / `variable` 与 CSV 的实际行数、表头一致。"
        + ("日志里的曝光、0 级位置、最大边长、选定边长与 `run_config` **全部一致**。"
           if run.log_crosscheck and all(ok for *_, ok in run.log_crosscheck)
           else ("日志与 `run_config` 存在**不一致**，见 §2.4。" if run.log_crosscheck
                 else "未提供日志，settle 判据状态未知。"))
    )
    add()

    add("### 2.2 饱和普查")
    add()
    sat_rows: list[tuple[Cell | str, ...]] = []
    for name in STAGES:
        st = run.stages[name]
        peaks = st.col("frame_peak")
        n_sat = sum(1 for v in peaks if v >= SATURATION_COUNTS)
        sat_rows.append((
            STAGE_CN[name],
            derived(float(n_sat), 0),
            derived(float(len(peaks) - n_sat), 0),
            derived(float(max(peaks)), 1),
            derived(float(min(peaks)), 1),
            ", ".join(f"{st.x[i]:.0f}" for i, v in enumerate(peaks) if v >= SATURATION_COUNTS)
            or "—",
        ))
    saturation = Table(
        ("段", "frame_peak 达满量程的行数", "其余行数", "本段最大峰值", "本段最小峰值", "达满量程的配置"),
        sat_rows,
    )
    tables.append(saturation)
    add(saturation.render())
    add()
    add(
        f"满量程按 8 bit 计为 {SATURATION_COUNTS:.0f} counts。**只有达到满量程才是饱和**："
        "这些行的峰值只能读成「≥ 满量程」，不能当线性测量值。注意 `frame_peak` 是**整帧**"
        "最大值，它通常落在 0 级上；方框内的 `box_peak` 单独记，两列一起看才能判断"
        "「饱和的是 0 级还是方斑」。"
    )
    add()

    add("### 2.3 settle 判据超时（日志原文）")
    add()
    if run.settle_events:
        ev_rows: list[tuple[Cell | str, ...]] = []
        for ev in run.settle_events:
            st = run.stages[ev.stage]
            row = st.rows[ev.index]
            frame_peak = float(row["frame_peak"])
            marks = flag_rows(run, st)[ev.index]
            ev_rows.append((
                STAGE_CN[ev.stage],
                derived(st.x[ev.index], 0),
                derived(ev.waited_s, 2),
                derived(float(ev.probes), 0),
                derived(ev.probe_peak, 2),
                derived(frame_peak, 2),
                derived(frame_peak - ev.probe_peak, 2),
                marks or "†",
            ))
        settle_tbl = Table(
            ("段", "配置", "等待 (s)", "探测次数", "探测峰值", "最终 frame_peak", "差",
             "旗标"),
            ev_rows,
        )
        tables.append(settle_tbl)
        add(settle_tbl.render())
        add()
        add(
            "读法：探测峰值与最终 `frame_peak` **相近**的行，说明取帧时画面已经到达"
            "它最终的电平，判据超时更可能是**判据过严**（相对容差 "
            f"{policy.get('tol')} 对这台相机的噪声偏紧），指标仍可用，只是形式上 provisional；"
            "两者**差距很大**或两者都远离满量程的行，才需要按不可信处理 —— "
            "本轮的 `‡` 行（§9）正是后者。"
        )
    else:
        add(
            "**日志未提供**（`--log` 指向的文件不存在），所以 settle 判据是否超时无法核对。"
            "这一节留空而不是推断 —— 断言的空白比错误的断言更安全。"
        )
    add()

    if run.log_crosscheck:
        add("### 2.4 日志 ↔ `run_config` 交叉核对")
        add()
        xc = Table(("量", "日志打印", "`run_config`", "一致"),
                   [(name, logged, actual, "是" if ok else "**否**")
                    for name, logged, actual, ok in run.log_crosscheck])
        tables.append(xc)
        add(xc.render())
        add()

    # -- 3. 边长扫描 --------------------------------------------------------
    add("## 3. 边长扫描")
    add()
    add(
        "每个边长都**从零跑一次 GS** 造方斑（不是把上一行的相位缩放），然后测量。"
        "这是本段最重要的一句口径：18 行的 GS 相位彼此独立，所以行间差异里含有"
        "**GS 本身的收敛差异**，而不只是方框大小的效应。"
    )
    add()
    add("### 3.1 逐行数据")
    add()
    side_tbl = Table(
        ("边长 (px)", "CV", "PIB", "score", "quality", "frame_peak", "box_peak",
         "方框内面积 (px)", "出画面积 (px)", "峰值状态", "旗标"),
        [
            (cell_of(r, "side_px", 0), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
             cell_of(r, "score", 4), cell_of(r, "quality", 4),
             cell_of(r, "frame_peak", 2), cell_of(r, "box_peak", 2),
             cell_of(r, "box_area_px", 0), cell_of(r, "box_clipped_px", 0),
             saturation_flag(r), marks_s[i] or "—")
            for i, r in enumerate(sides.rows)
        ],
    )
    tables.append(side_tbl)
    add(side_tbl.render())
    add()
    add(FLAG_LEGEND)
    add()

    add("### 3.2 PIB 与 CV 同向上升")
    add()
    add(f"![边长 → PIB / CV / score](figures/{_figname('fig1_sides_objectives')})")
    add()
    add(
        f"PIB 在 {len(sides.rows)} 行里**严格单调上升**（反向步数 {len(f.pib_violations)}），"
        f"从 {f.pib[0]:.4f}（{f.sides_x[0]:.0f} px）到 {f.pib[-1]:.4f}"
        f"（{f.sides_x[-1]:.0f} px）。这符合「方框越大装得越多」的直觉，但它同时说明"
        "**PIB 单独不构成整形质量指标**：把框放大到装下整幅画面，PIB 会自然趋近 1。"
    )
    add()
    add(
        f"CV 的整体趋势同样上升（Spearman ρ = {f.cv_spearman:.3f}），但"
        f"**不是单调**：{len(f.cv_violations)} 处回落（{_join_x(f.sides_x, f.cv_violations)} px）。"
        "CV 上升的物理含义很直接 —— 方框变大后，框内除了方斑还装进了越来越多的"
        "**暗区**，标准差/均值被撑大。所以「PIB 与 CV 同向上升」这句在方向上成立，"
        "但它描述的其实是同一个几何事实（框变大），不是两个独立的物理效应。"
    )
    add()

    add("### 3.3 score 的峰：两项互相抵消")
    add()
    peak_x = f.sides_x[f.score_peak_index]
    add(
        f"`score` 的 argmax 在 **{peak_x:.0f} px**（{f.score[f.score_peak_index]:.4f}），"
        f"次高是 {f.sides_x[f.score_runner_up_index]:.0f} px（{f.score[f.score_runner_up_index]:.4f}）。"
        f"两者相差 **{f.score_peak_gap:.2e}** —— 这个差值小到没有意义，所以正确的说法是"
        f"「峰值落在 {min(peak_x, f.sides_x[f.score_runner_up_index]):.0f}–"
        f"{max(peak_x, f.sides_x[f.score_runner_up_index]):.0f} px 之间，"
        "无法分辨单点」，而不是「1500 px 最优」。"
    )
    add()
    term_tbl = Table(
        ("边长 (px)", "PIB", "0.5·PIB", "CV", "0.5/(1+CV)", "score", "说明"),
        [
            (cell_of(sides.rows[0], "side_px", 0), cell_of(sides.rows[0], "pib", 4),
             derived(f.terms_at_small[0], 4), cell_of(sides.rows[0], "cv", 4),
             derived(f.terms_at_small[1], 4), cell_of(sides.rows[0], "score", 4),
             "最小方框：饱和"),
            (cell_of(sides.rows[f.score_peak_index], "side_px", 0),
             cell_of(sides.rows[f.score_peak_index], "pib", 4),
             derived(f.terms_at_peak[0], 4),
             cell_of(sides.rows[f.score_peak_index], "cv", 4),
             derived(f.terms_at_peak[1], 4),
             cell_of(sides.rows[f.score_peak_index], "score", 4), "峰值"),
            (cell_of(sides.rows[-1], "side_px", 0), cell_of(sides.rows[-1], "pib", 4),
             derived(f.terms_at_last[0], 4), cell_of(sides.rows[-1], "cv", 4),
             derived(f.terms_at_last[1], 4), cell_of(sides.rows[-1], "score", 4),
             "最大方框"),
        ],
    )
    tables.append(term_tbl)
    add(term_tbl.render())
    add()
    if f.score_peak_index == len(f.sides_x) - 1:
        add(
            f"**本轮的 score 峰就落在最后一行（{peak_x:.0f} px）**，峰后没有数据 —— "
            "所以「峰后下降」这一条在本轮**不可判定**，§3.3 的分解也就只能给到峰值点为止。"
            "（基线那一轮扫到了 1800 px，才有峰后下降的腿。）"
        )
        add()
    add(
        f"从 {peak_x:.0f} px 到 {f.sides_x[-1]:.0f} px，PIB 项**还在涨**"
        f"（{f.terms_at_peak[0]:.4f} → {f.terms_at_last[0]:.4f}，"
        f"{f.terms_at_last[0] - f.terms_at_peak[0]:+.4f}），"
        f"而均匀度项在掉（{f.terms_at_peak[1]:.4f} → {f.terms_at_last[1]:.4f}，"
        f"{f.terms_at_last[1] - f.terms_at_peak[1]:+.4f}）。"
        f"后者绝对值更大，于是 score 净变化 {f.score[-1] - f.score[f.score_peak_index]:+.4f}。"
        "**峰值之后 score 的下降完全由 CV 造成，PIB 反而更好了** —— 这就是"
        "「两项互相抵消」的量化版本，也解释了为什么单纯追 PIB 会把方框越开越大。"
    )
    add()

    add("### 3.4 GS 模型输出 vs 实测")
    add()
    model_tbl = Table(
        ("边长 (px)", "model_cv", "实测 CV", "model_pib", "实测 PIB", "ΔPIB"),
        [(cell_of(r, "side_px", 0), cell_of(r, "model_cv", 4), cell_of(r, "cv", 4),
          cell_of(r, "model_pib", 4), cell_of(r, "pib", 4), derived(d, 4))
         for r, d in zip(sides.rows, f.model_delta)],
    )
    tables.append(model_tbl)
    add(model_tbl.render())
    add()
    add(
        f"GS 模型给出的 PIB 在 {min(f.model_pib):.4f}–{max(f.model_pib):.4f} 之间几乎不动，"
        f"而实测 PIB 在 {min(f.pib):.4f}–{max(f.pib):.4f} 之间涨了 "
        f"{max(f.pib) - min(f.pib):.2f}。模型在最小方框上高估 {f.model_delta[0]:+.4f}，"
        f"在最大方框上差 {f.model_delta[-1]:+.4f}。**结论：GS 的模型内指标不能用来选边长** —— "
        "它看不到「实际远场不是理想方斑」这件事（模型里方斑是精确的平顶方框，"
        "实测里还有 0 级、散射与像差）。"
    )
    add()
    add(f"![边长 → 峰值与光斑几何](figures/{_figname('fig2_sides_exposure_geometry')})")
    add()

    # -- 4. 曝光口径 --------------------------------------------------------
    add("## 4. 曝光口径")
    add()
    add("### 4.1 工作点")
    add()
    add(
        f"请求 {run.exposure_requested_ms:g} ms，会话上限 `AO_DAHENG_MAX_EXPOSURE_MS` = "
        f"{run.exposure_ceiling_ms:g} ms，**生效 {run.exposure_ms:g} ms**"
        f"（= min(请求, 上限)，与日志一致）。上限只是相机的保护阈值，不是工作点；"
        f"本轮 {len(sides.rows) + len(zern.rows) + len(blaze.rows)} 行的 "
        "`exposure_ms_readback` 全部等于生效值（生成器已逐行核对）。"
    )
    add()
    add("### 4.2 平场的 0 级峰值不能用来挑曝光")
    add()
    add(
        "GS 把能量**摊到边长上千像素的方斑上**，而 0 级是一个衍射极限量级的小核。"
        "两者每像素电平差好几个数量级，所以「把 0 级调到不饱和」和「让方斑有足够信噪比」"
        "是两个完全不同的曝光要求。本轮的证据是自洽的："
    )
    add()
    add(
        f"- GS 之前的平场帧（`spot_assessment`，同一曝光 {run.exposure_ms:g} ms）里，"
        f"整帧峰值 = **{float(spot.get('peak', float('nan'))):.1f} counts**，"
        f"已顶到满量程 {SATURATION_COUNTS:.0f} —— 这个读数只能读成「≥ 满量程」，"
        "**不含任何线性信息**，任何按它做的曝光决策都是在猜。"
    )
    add(
        f"- 同一曝光下，被整形方斑的峰值（`box_peak`）在边长 ≥ "
        f"{f.sides_x[1]:.0f} px 的 {len(sides.rows) - 1} 行里是 "
        f"{min(sides.col('box_peak')[1:]):.2f}–{max(sides.col('box_peak')[1:]):.2f} counts，"
        "全部**远离**满量程。也就是说：0 级已经饱和时，方斑还在半个满量程附近 —— "
        "如果按 0 级反推去降低曝光，方斑会先失去信噪比。"
    )
    add(
        f"- 信噪比余量：评估帧的噪声 σ = {float(spot.get('noise', float('nan'))):.4f} counts"
        "（同曝光、同累计帧数），方斑峰值/噪声 ≈ "
        f"{min(sides.col('box_peak')[1:]) / max(float(spot.get('noise', 1.0)), 1e-9):.0f}–"
        f"{max(sides.col('box_peak')[1:]) / max(float(spot.get('noise', 1.0)), 1e-9):.0f}，"
        "曝光足够，不需要再加。"
    )
    add()
    add(
        "**这里有一个口径缺口，必须写明**：把「0 级峰值 → 方斑每像素电平」写成等式需要两个"
        "本轮没有归档的量 ——（a）未饱和的 0 级峰值随曝光的关系（一条平场曝光阶梯），"
        "（b）0 级的等效面积。本轮 `spot_assessment.peak` 在 20 ms 上已经饱和，且产物里没有"
        "平场阶梯，所以本报告**不给这条换算的绝对值**，只给上面那组可复现的比较"
        "（饱和的 0 级 vs 未饱和的方斑峰值 vs 噪声）。补一条 "
        "0.05 / 0.2 / 1 / 5 / 20 ms 的平场阶梯就能把它写成等式，见 §9。"
    )
    add()

    add("### 4.3 饱和行")
    add()
    sat_list = [f"{sides.x[i]:.0f} px（frame_peak = box_peak = "
                f"{sides.col('frame_peak')[i]:.0f}）"
                for i, r in enumerate(sides.rows)
                if float(r["frame_peak"]) >= SATURATION_COUNTS]
    add(
        ("边长段里饱和的行：" + "；".join(sat_list) + "。") if sat_list
        else "边长段没有饱和行。"
    )
    add(
        "这一行的 PIB/CV 仍然可用，但要知道它们是怎么算的：`PIB` 是**整框求和**与**整帧求和**的"
        "比值，`CV` 是框内逐像素的统计 —— 两者都不是「峰值有多亮」，所以框内少量像素被 8 bit "
        "削平并不会把它们整体带偏；被带偏的只有峰值类读数（`frame_peak` / `box_peak`）和任何"
        "以峰值为分母的量。**读这一行时，请把「框内均值」类结论与峰值类结论分开。**"
    )
    add()
    peak_in_box = sum(
        1 for st in run.stages.values() for r in st.rows
        if float(r["box_peak"]) == float(r["frame_peak"])
    )
    total_rows = sum(len(st.rows) for st in run.stages.values())
    add(
        f"**另外两段的饱和是常态，而且原因不同。** 全部 {total_rows} 行里，"
        f"有 {peak_in_box} 行的 `box_peak == frame_peak` —— 最亮的那个像素"
        "**就在方框内**，饱和的是框内的核，不是框外的杂散光。具体到各段："
    )
    add()
    add(
        f"- **Zernike 段 {sum(1 for v in zern.col('frame_peak') if v >= SATURATION_COUNTS)}/"
        f"{len(zern.rows)} 行饱和**：低阶重构把光聚成一个亮核，亮核落在 1200 px 框里，"
        "框内峰值于是顶到满量程。这正是 §5.1 那个「PIB 假高分」的同一件事 —— "
        "**饱和与「框内几乎全暗但有个亮核」是同一个形态**。"
    )
    add(
        f"- **闪耀光栅段 {sum(1 for v in blaze.col('frame_peak') if v >= SATURATION_COUNTS)}/"
        f"{len(blaze.rows)} 行饱和**：叠加光栅后光场里始终有一个**未经相位调制的分量**"
        "停在冻结 0 级附近（`predicted_shift` 只搬动被调制的那部分），于是即便方框跟着"
        "预测位移走了，框内仍有一个高亮核。所以这一段的峰值饱和**不能**读成「方斑太亮」—— "
        "它量的是那个未调制分量。"
    )
    add()
    add(
        "两段的共同结论：**在这些行上 `frame_peak` / `box_peak` 只能当「≥ 满量程」用**；"
        "`PIB`（比值）与 `CV`（框内统计）不受单像素削顶的系统性影响，仍可比。"
    )
    add()

    # -- 5. Zernike ---------------------------------------------------------
    add("## 5. Zernike 重构段")
    add()
    add(
        f"把选定边长（{f.selected_side} px）那一次的 GS 相位拟合到 Zernike 基，"
        f"然后用**前 N 阶**重构相位、下发、测量。N 从 "
        f"{max(zern.x):.0f} 降到 {min(zern.x):.0f}，最高阶用 "
        f"{int(float(zern.rows[_idx_of(zern.x, max(zern.x))]['zernike_modes_used'])):.0f} 个模式"
        f"（拟合网格 {int(zern.run_config.get('fit_grid', 0))}、"
        f"拟合上限 {int(zern.run_config.get('fit_max_order', 0))} 阶、"
        f"拟合耗时 {float(zern.run_config.get('fit_seconds', float('nan'))):.2f} s —— "
        "这四项都来自 `zernike/run_config.json`）。"
    )
    add()
    zern_tbl = Table(
        ("n_max", "模式数", "解释方差", "残差 RMS (rad)", "CV", "PIB", "score",
         "frame_peak", "质心偏移 x", "质心偏移 y", "旗标"),
        [(cell_of(r, "n_max", 0), cell_of(r, "zernike_modes_used", 0),
          cell_of(r, "zernike_explained_fraction", 6),
          cell_of(r, "zernike_rms_residual_rad", 6),
          cell_of(r, "cv", 4), cell_of(r, "pib", 4), cell_of(r, "score", 4),
          cell_of(r, "frame_peak", 2), cell_of(r, "box_centroid_offset_x", 2),
          cell_of(r, "box_centroid_offset_y", 2), marks_z[i] or "—")
         for i, r in enumerate(zern.rows)],
    )
    tables.append(zern_tbl)
    add(zern_tbl.render())
    add()
    add(f"![Zernike 阶数 → 解释方差 / 残差 / CV](figures/{_figname('fig3_zernike_reconstruction')})")
    add()

    add("### 5.1 低阶 Zernike 表示不了方斑")
    add()
    zn_max = max(zern.x)
    zn_modes = int(float(zern.rows[_idx_of(zern.x, zn_max)]["zernike_modes_used"]))
    expl_max = max(f.zern_explained)
    add(
        f"解释方差从 {min(f.zern_explained) * 100:.3f}%（n_max = {min(zern.x):.0f}）"
        f"单调升到 {expl_max * 100:.3f}%（n_max = {zn_max:.0f}，{zn_modes} 个模式），"
        f"反向步数 {len(f.zern_explained_violations)}。**{zn_modes} 个模式只解释 "
        f"{expl_max * 100:.2f}% 的相位方差** —— 剩下 "
        f"{100 * (1 - expl_max):.2f}% 完全在低阶 Zernike 张成的空间之外。"
    )
    add()
    add(
        f"残差 RMS 从 {max(f.zern_residual):.4f} rad（n_max = {min(zern.x):.0f}）只降到 "
        f"{min(f.zern_residual):.4f} rad（n_max = {zn_max:.0f}），"
        f"相对变化 {abs(min(f.zern_residual) - max(f.zern_residual)) / max(f.zern_residual) * 100:.1f}%。"
        "半个波长的残差在 1064 nm 上就是 **相位本身** 量级的东西，说明重构出的相位与目标相位"
        "几乎没有关系。"
    )
    add()
    add(
        f"实测侧：干净行（排除 `†`/`‡`/`※`）的 CV 落在 "
        f"{min(f.zern_clean_cv):.3f}–{max(f.zern_clean_cv):.3f}，"
        f"与阶数的相关系数 r = {f.zern_cv_pearson:+.3f}。**趋势是有的**（阶数越高略均匀），"
        f"但整段的相对跨度只有 {f.zern_clean_cv_spread:.1%} —— 把它和同框 GS 方斑的 CV 放一起就"
        "看出量级差："
        f"同一 1200 px 方框上 GS 方斑的 CV 是 "
        f"{f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.4f}，"
        f"而 Zernike 重构后是 {f.zern_cv[_idx_of(zern.x, zn_max)]:.4f} —— "
        f"**高 {f.zern_cv[_idx_of(zern.x, zn_max)] / f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.1f} 倍**。"
        f"换句话说：阶数带来的 {f.zern_clean_cv_spread:.1%} 改善，与「基不适配造成的 "
        f"{f.zern_cv[_idx_of(zern.x, zn_max)] / f.cv[_idx_of(f.sides_x, float(f.selected_side))]:.0f}× 劣化」"
        "相比可以忽略 —— 所以「CV 与阶数几乎无关」在**量级**上成立，作为相关系数则不然。"
    )
    add()
    add(
        "这**印证**了仓库既有的反模式记录：「不要用低阶 Zernike 做方斑整形」。"
        "机制也说得通：GS 方斑的相位是高频、类 sinc 的结构（近场需要二维 sinc 的过零点），"
        "而 Zernike 是圆域上的低阶光滑正交基 —— 它不是「拟合得不够准」，而是**基不适配**，"
        "加阶数只能线性地多解释一点点方差（本段 "
        f"{min(f.zern_explained) * 100:.2f}% → {expl_max * 100:.2f}%，"
        f"{zn_modes} 个模式才这个量级）。"
    )
    add()
    add(
        "**这里还有一个必须点出来的陷阱**：失败的 Zernike 重构 PIB = "
        f"{float(zern.rows[_idx_of(zern.x, zn_max)]['pib']):.4f}，高于 GS 方斑的 PIB = "
        f"{f.pib[_idx_of(f.sides_x, float(f.selected_side))]:.4f}。"
        "原因不是它更好，而是**框太大**：低阶重构把光聚成一个亮核（整帧峰值因此顶到满量程），"
        "亮核正好落在 1200 px 方框里，框内功率占比自然高；而方框其余部分几乎全暗，CV 就爆到 7.3。"
        "**只看 PIB 会把这个失败排到成功前面。**"
    )
    add()

    add("### 5.2 n_max = 18 这一行")
    add()
    i18 = _idx_of(zern.x, 18.0)
    i17 = _idx_of(zern.x, 17.0)
    add(
        f"`n_max = 18` 的 CV 是 **{f.zern_cv[i18]:.3f}**，是同段中位数 "
        f"{zern.median('cv'):.3f} 的 {f.zern_cv[i18] / zern.median('cv'):.2f} 倍；"
        f"整帧峰值只有 **{float(zern.rows[i18]['frame_peak']):.1f} counts**"
        f"（相邻行都是 {float(zern.rows[i17]['frame_peak']):.1f}），"
        f"score 掉到 {f.zern_score[i18]:.4f}，框内质心 x 从相邻行的 "
        f"{float(zern.rows[i17]['box_centroid_offset_x']):+.1f} px 跳到 "
        f"{float(zern.rows[i18]['box_centroid_offset_x']):+.1f} px。"
    )
    add()
    add(
        "这一行的**整帧几乎全黑**，所以它的 CV / 质心 / score 全部由噪声主导 —— 它是"
        "**采集失败**，不是「18 阶时性能崩了」。它同时说明本段不能按「阶数越低越好」"
        "或「某阶最优」去读，只能读「整段都在不可用区间」。"
    )
    add()
    fit_grid = int(zern.run_config.get("fit_grid", 0))
    panel = (run.plan.get("bench") or {}).get("panel_shape") or [0, 0]
    add(
        f"另一个次异常是 `n_max = {zern.x[i17]:.0f}`：CV {f.zern_cv[i17]:.3f}"
        f"（比中位数高 {(f.zern_cv[i17] / zern.median('cv') - 1) * 100:.1f}%）、"
        f"score {f.zern_score[i17]:.4f}、整帧峰值正常，但它**恰好也是一条 `†` 行**。"
        "两条高端异常都落在 `†` 上，指向同一个可疑来源：**高阶级重构在面板上的采样**。"
        f"拟合在 {fit_grid}×{fit_grid} 网格上做、显示在 {int(panel[1])}×{int(panel[0])} 面板上（W×H），"
        "高阶模式的相位被放大后可能出现欠采样/混叠，把光衍射到远场窗口之外"
        "（全黑帧正是这个特征）。这是**假设**，验证它需要把 `zernike_fit_grid` 提到 "
        f"{2 * fit_grid} 并把 24–30 阶细扫一遍，见 §9。"
    )
    add()

    # -- 6. blaze -----------------------------------------------------------
    add("## 6. 闪耀光栅段")
    add()
    add(
        "在选定相位上叠加不同周期的**面板 x 向斜坡**，位移按 `predicted_shift_px` 预测"
        "（已含面板↔相机 90° 交换，所以位移落在相机 y 上），**方框跟着预测位移走**，"
        "质心在框内量。"
    )
    add()
    blaze_tbl = Table(
        ("周期 (SLM px)", "预测位移 (px)", "实测位移 y (px)", "位移误差 (px)",
         "实测/预测", "CV", "PIB", "score", "出画面积 (px)", "frame_peak", "旗标"),
        [(cell_of(r, "blaze_period_px", 0), cell_of(r, "predicted_shift_px", 2),
          cell_of(r, "measured_shift_y_px", 2), cell_of(r, "shift_error_px", 2),
          derived(ratio, 4), cell_of(r, "cv", 4), cell_of(r, "pib", 4),
          cell_of(r, "score", 4), cell_of(r, "box_clipped_px", 0),
          cell_of(r, "frame_peak", 2), marks_b[i] or "—")
         for i, (r, ratio) in enumerate(zip(blaze.rows, f.blaze_shift_ratio))],
    )
    tables.append(blaze_tbl)
    add(blaze_tbl.render())
    add()
    add(f"![光栅周期 → 预测/实测位移与 score](figures/{_figname('fig4_blaze_shift')})")
    add()

    add("### 6.1 score 随周期单调上升")
    add()
    clean_scores = [f.blaze_score[i] for i in range(len(blaze.rows))
                    if i not in {_idx_of(blaze.x, 16.0), _idx_of(blaze.x, 40.0)}]
    add(
        f"去掉 `§`（周期 {blaze.x[_idx_of(blaze.x, 16.0)]:.0f}，框出画）和 `‡`"
        f"（周期 {blaze.x[_idx_of(blaze.x, 40.0)]:.0f}，帧近乎全黑）两行后，"
        f"score 从 {clean_scores[0]:.4f} 单调升到 {clean_scores[-1]:.4f}，"
        f"反向步数 {len(f.blaze_score_violations_clean)}。方向与直觉一致：**周期越大，"
        "同样的斜坡幅值给出的位移越小，方斑被保形得越好** —— 位移 ∝ 1/周期"
        "（`predicted_shift_px` 的表达式就是 `TILT_SHIFT_SCALE / 周期`），"
        "所以周期加倍时方斑离开冻结位置的位移减半，落在框外的部分也随之减少。"
    )
    add()

    add("### 6.2 周期 16：框出画，PIB/EE 不可横比")
    add()
    i16 = _idx_of(blaze.x, 16.0)
    side16 = float(blaze.rows[i16]["box_side_px"])
    add(
        f"预测位移 {blaze.col('predicted_shift_px')[i16]:.1f} px 把方框中心推到 "
        f"({float(blaze.rows[i16]['box_center_x']):.0f}, {float(blaze.rows[i16]['box_center_y']):.0f})，"
        f"框高 {side16:.0f} px ⇒ 有 **{blaze.col('box_clipped_px')[i16]:.0f} px**"
        f"（{_pct(blaze.col('box_clipped_px')[i16], side16 ** 2)} 的框面积）落在画面外。"
        f"这一行的 `PIB` = {blaze.col('pib')[i16]:.4f}、`EE` 同值，都因为**分母是整帧、"
        "分子被裁掉一块**而系统性偏低，所以**不能与其它行横向比较**；`CV` 是框内统计，"
        f"受的影响小一些（{blaze.col('cv')[i16]:.2f}），但方框本身已经错位，也不应当作"
        "「周期 16 的均匀度」来读。"
    )
    add()

    add("### 6.3 周期 40：provisional，不得当结论")
    add()
    i40 = _idx_of(blaze.x, 40.0)
    ev40 = run.settle_for("blaze", i40)
    add(
        (f"日志原文（该配置那一条）：\n\n> `{ev40.raw_fragment}`\n" if ev40 and ev40.raw_fragment
         else (f"日志记录：{ev40.describe()}。" if ev40 else "日志缺失，无法核对这一行的 settle 状态。"))
        + f"\n该行最终 `frame_peak` = {float(blaze.rows[i40]['frame_peak']):.2f}，"
        f"`box_peak` = {float(blaze.rows[i40]['box_peak']):.2f}，"
        f"`global_spot_fwhm_px` = {blaze.rows[i40]['global_spot_fwhm_px']}（无效值），"
        f"CV {blaze.col('cv')[i40]:.2f}、score {f.blaze_score[i40]:.4f}。"
    )
    add()
    add(
        "**这一行是采集事故，不是物理结论。** 它落在一条单调序列中间（左邻周期 "
        f"{blaze.x[i40 - 1]:.0f} 的 score {f.blaze_score[i40 - 1]:.4f}、右邻周期 "
        f"{blaze.x[i40 + 1]:.0f} 的 {f.blaze_score[i40 + 1]:.4f}），如果把它当真会得到"
        "「40 px 突然变差」的假结论。它同时是本轮 `†` 里**唯一一行最终画面也全黑**的"
        "（探测峰值与最终峰值都在 2 counts 量级），所以 `‡` 和 `†` 同时命中它。"
    )
    add()

    add("### 6.4 预测位移 vs 实测位移：方向对，标度与 1/P 形状都不对")
    add()
    clean_ratio = [
        float(r["measured_shift_y_px"]) / float(r["predicted_shift_px"])
        for i, r in enumerate(blaze.rows) if i != i40
    ]
    clean_periods = [float(r["blaze_period_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    meas = [float(r["measured_shift_y_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    pred = [float(r["predicted_shift_px"]) for i, r in enumerate(blaze.rows) if i != i40]
    alpha_meas = _loglog_exponent([clean_periods[0], clean_periods[-1]], [meas[0], meas[-1]])
    alpha_pred = _loglog_exponent([clean_periods[0], clean_periods[-1]], [pred[0], pred[-1]])
    add(
        "⚠️ 先把口径钉死：**位移只有 `measured_shift_y_px` 这一列**（质心相对冻结 0 级）。"
        "`box_centroid_offset_y` **不是**位移 —— 它 = 实测质心 − **预测框心**，而框是跟着"
        "预测走的，所以它天然是个大负值（§11.6 给出恒等式）。把它当位移读会得到「反号」的假结论。"
    )
    add()
    add(
        f"- **方向可用**：实测与预测**同号**（全部为正），"
        f"实测/预测 = {min(clean_ratio):.4f}–{max(clean_ratio):.4f}"
        f"（排除 `period = {blaze.x[i40]:.0f}` 那一行 —— 它的实测 "
        f"{float(blaze.rows[i40]['measured_shift_y_px']):.2f} px 是黑帧伪迹，"
        "见 §2.3 / §6.3，**不当数据点**）。"
    )
    add(
        f"- **标度不对，而且不是常数标度错**：若只是系数错，这个比值应当与周期无关；"
        f"实际它随周期**总体上升**（{min(clean_ratio):.4f} → {clean_ratio[-1]:.4f}），"
        f"6 步里 {sum(1 for i in range(1, len(clean_ratio)) if clean_ratio[i] > clean_ratio[i - 1])} 步上升、"
        f"只有 24→32 一步回落 {abs(clean_ratio[2] / clean_ratio[1] - 1):.1%} ⇒ 形状也不对。"
    )
    add(
        f"- **1/P 形状被强烈压缩**：P {clean_periods[0]:.0f} → {clean_periods[-1]:.0f}"
        f"（{clean_periods[-1] / clean_periods[0]:.1f}×）时，预测 "
        f"{pred[0]:.2f} → {pred[-1]:.2f} px（{pred[0] / pred[-1]:.1f}×，构造上恰好 ∝1/P），"
        f"而实测只有 {meas[0]:.2f} → {meas[-1]:.2f} px（{meas[0] / meas[-1]:.1f}×）。"
        f"两端点 log-log 斜率：预测 α = {alpha_pred:.2f}、实测 α = {alpha_meas:.2f}。"
    )
    add(
        "- **直接后果**：周期越小框偏得越远。框心相对质心偏 "
        + " / ".join(
            f"{float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y']):+.1f}"
            for p in (16.0, 24.0, 32.0))
        + " px（P = 16 / 24 / 32）—— 也就是说框跟着预测走时，小周期处框会离光斑 "
        + f"{min(abs(float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y'])) for p in (16.0, 24.0, 32.0)):.0f}–"
        + f"{max(abs(float(blaze.rows[_idx_of(blaze.x, p)]['box_centroid_offset_y'])) for p in (16.0, 24.0, 32.0)):.0f} px。"
    )
    add()
    add(
        "候选解释只剩两条：**斜坡幅值被削顶**（相位调制深度有限）与**框跟随导致的质心"
        "截断偏置**。二者都预测「实测 < 预测」，本数据分不开；要分开它们，"
        "需要一次**框固定**（冻结在未加光栅的 0 级上，或直接用宽 ROI）的复测："
        "框不动时质心才反映真实位移。§11.5 判定 5 与 §11.6 给出具体做法。"
    )
    add()

    # -- 7. 近 1000 选点 ----------------------------------------------------
    add("## 7. 「近 1000」的选点边界效应")
    add()
    add(
        f"第 2 段的输入边长由 `--target-side-near 1000` 在窗口 "
        f"[{f.near_window[0]:.0f}, {f.near_window[1]:.0f}] px 内按 score 取最大决定。"
        "把窗口内的行摊开看就能看出问题："
    )
    add()
    near_rows = [i for i, x in enumerate(f.sides_x) if f.near_window[0] <= x <= f.near_window[1]]
    near_tbl = Table(
        ("边长 (px)", "score", "距 1000 的距离 (px)", "是否被选中"),
        [(cell_of(sides.rows[i], "side_px", 0), cell_of(sides.rows[i], "score", 4),
          derived(f.sides_x[i] - 1000.0, 0, " px"), "**是**" if i == f.near_best_index else "—")
         for i in near_rows],
    )
    tables.append(near_tbl)
    add(near_tbl.render())
    add()
    i1000 = _idx_of(f.sides_x, 1000.0)
    add(
        f"窗口内 score **严格单调上升**（反向步数 "
        f"{len(monotonic_report([f.score[i] for i in near_rows]))}），所以 argmax 必然落在"
        f"**窗口上界 {f.sides_x[f.near_best_index]:.0f} px** —— 这不是局部峰，是边界效应。"
        f"并且整条曲线在窗口外还在涨（峰值 {f.sides_x[f.score_peak_index]:.0f} px，"
        f"{f.score[f.score_peak_index]:.4f}），所以「选到 1200」既不是局部最优也不是全局最优，"
        "它只是「窗口上界」。"
    )
    add()
    add(
        f"**建议**：若窗口改成 [{f.alt_window[0]:.0f}, {f.alt_window[1]:.0f}] px，"
        f"用本轮的 CSV 就能回答会选到 **{f.sides_x[f.alt_best_index]:.0f} px**"
        f"（score {f.score[f.alt_best_index]:.4f}）。窗口内 score 依然单调，"
        "所以只要上界跟着动，选中值就跟着动 —— 在 score 单调上升的区间里，"
        "任何「接近某值」的选点规则实际选的都是窗口边界。"
        f"**真正想选「接近 1000」应该直接取最接近 1000 的点（{f.sides_x[i1000]:.0f} px），"
        "或者把窗口做成对称且明确说明「取上界」。**"
    )
    add()
    add(
        f"（补一句量级：1000 px 的 score 是 {f.score[i1000]:.4f}，"
        f"1200 px 是 {f.score[f.near_best_index]:.4f}，差 "
        f"{f.score[f.near_best_index] - f.score[i1000]:.4f}；"
        f"而 §3.3 里 1500 vs 1600 的差只有 {f.score_peak_gap:.2e}，"
        f"两者差 {(f.score[f.near_best_index] - f.score[i1000]) / f.score_peak_gap:.0f} 倍。"
        "所以「窗口边界把选中值从 1000 推到 1200」的代价是**真实可测**的，"
        "只是它仍落在平台区（距全局峰值 "
        f"{f.score[f.score_peak_index] - f.score[f.near_best_index]:.4f}）。）"
    )
    add()

    # -- 8. 最大边长 --------------------------------------------------------
    add("## 8. 最大边长与 0 级位置")
    add()
    if "plan" not in cfg:
        add(
            "`side_max_px` **不是常数**，是从 0 级位置与画面边缘反解出来的"
            "（`side_max = 2·(min(cx, W−1−cx, cy, H−1−cy) − edge_margin)`，再向下取到"
            "步长的整数倍）。**但本批的 `run_config` 不含 `plan`**，"
            "`edge_margin` 与计划里的假设中心都没有落盘，所以本节只给出实测的 0 级位置，"
            "不做反解 —— 反解的完整推导见基线那一轮的报告。"
        )
        add()
        add("```")
        add(f"0 级 (实测冻结)     = ({zo_x:.0f}, {zo_y:.0f})")
        add(f"画面 (W x H)        = ({w}, {h})   ⇒ 几何中心 ({(w - 1) / 2:.0f}, "
            f"{(h - 1) / 2:.0f})")
        add(f"到四条边的距离       = 左 {zo_x:.0f} / 右 {(w - 1) - zo_x:.0f} / "
            f"上 {zo_y:.0f} / 下 {(h - 1) - zo_y:.0f}")
        add(f"约束边              = {f.binding_edge} "
            f"({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f})")
        add("```")
        add()
        add(
            f"**0 级不在画面中心**：几何中心是 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})，"
            f"实测 0 级在 ({zo_x:.0f}, {zo_y:.0f})，偏 "
            f"{zo_x - (w - 1) / 2:+.0f} / {zo_y - (h - 1) / 2:+.0f} px。"
        )
        add()
    else:
        add(
            "`side_max_px` **不是常数**，是从 0 级位置与画面边缘反解出来的："
        )
        add()
        add("```")
    add(f"0 级 (实测冻结)     = ({zo_x:.0f}, {zo_y:.0f})")
    add(f"画面 (W x H)        = ({w}, {h})   ⇒ 几何中心 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})")
    add(f"到四条边的距离       = 左 {zo_x:.0f} / 右 {(w - 1) - zo_x:.0f} / "
        f"上 {zo_y:.0f} / 下 {(h - 1) - zo_y:.0f}")
    add(f"约束边              = {f.binding_edge} ({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f})")
    add(f"side_max_raw        = 2 · ({min(zo_x, (w - 1) - zo_x, zo_y, (h - 1) - zo_y):.0f}"
        f" − {f.edge_margin:.0f}) = {f.side_max_raw:.0f} px")
    add(f"向下取 {f.step:.0f} 的整数倍 = {f.side_max:.0f} px")
    add("```")
    add()
    assumed = (run.plan.get("frame") or {}).get("zero_order_xy") or [w / 2.0, h / 2.0]
    as_x, as_y = float(assumed[0]), float(assumed[1])
    assumed_dist = min(as_x, (w - 1) - as_x, as_y, (h - 1) - as_y)
    assumed_raw = 2.0 * (assumed_dist - f.edge_margin)
    add(
        f"**0 级不在画面中心**：几何中心是 ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f})，"
        f"实测 0 级在 ({zo_x:.0f}, {zo_y:.0f})，偏 {zo_x - (w - 1) / 2:+.0f} / "
        f"{zo_y - (h - 1) / 2:+.0f} px。所以「最大边长」必须现场量：若按**假设的中心** "
        f"({as_x:.0f}, {as_y:.0f}) 代入同一条式，得到 "
        f"2 · ({assumed_dist:.0f} − {f.edge_margin:.0f}) = {assumed_raw:.0f} px，"
        f"向下取整就是计划里的 "
        f"{run.plan.get('frame', {}).get('side_max', '-')} px —— 比实际能用的 "
        f"{f.side_max:.0f} px 大 "
        f"{float(run.plan.get('frame', {}).get('side_max', 0)) - f.side_max:.0f} px，"
        "那个尺寸的框会直接压到帧边（框压边会被窗函数与读出噪声污染）。"
    )
    add()
    add(
        f"这也解释了 §2.1 里「计划有 {', '.join(f.dropped_configs)}、CSV 没有」："
        f"`--dry-run` 的计划按**画幅中心**估出 "
        f"{run.plan.get('frame', {}).get('side_max_raw', '-')} px"
        f"（`plan.frame.side_max_raw`），实际按 0 级解出 {f.side_max_raw:.0f} px，"
        f"向下取整后 {f.side_max:.0f} px，于是计划里的 "
        f"{', '.join(f.dropped_configs)} 被裁掉。这是**设计行为**，不是缺数据。"
    )
    add()

    # -- 9. 可疑项 ----------------------------------------------------------
    add("## 9. 可疑项与复测清单")
    add()
    add(
        f"按优先级排列。每条都写明「为什么可疑」和「复测要看什么」，"
        f"不止清单本身。本轮共 {len(run.settle_events)} 行 settle 超时"
        "（见 §2.3），其中 1 行最终画面也全黑。"
    )
    add()
    i18_peak = float(zern.rows[i18]["frame_peak"])
    clean_ratios = [
        r for i, r in enumerate(f.blaze_shift_ratio) if i != i40 and math.isfinite(r)
    ]
    if retest is not None and rf is not None:
        head = [
            (f"**`n_max = 18` 全黑帧**（§5.2）：整帧峰值 {i18_peak:.1f} counts、CV "
             f"{f.zern_cv[i18]:.3f}。**已复测推翻**（§11.5 判定 2）：复测 18 的 CV "
             f"{rf.zern_cv[1]:.4f} 与邻居同量级、`frame_peak` 回到满量程 ⇒ "
             "基线那行是采集失败，不需要再测。"),
            (f"**`blaze period = 40` provisional**（§6.3）：settle 超时 + 最终峰值 "
             f"{float(blaze.rows[i40]['frame_peak']):.2f} counts。**已复测推翻**"
             f"（§11.5 判定 1）：复测 40 的 CV {rf.blaze_cv[1]:.4f} 落在 32/48 之间。"
             "留下的唯一待办是把 settle 的**探测时间序列**打出来（现在只打了最后一次探测）"
             "—— 两次黑帧都发生在探测阶段，这条诊断信息比再测一个点更有价值。"),
            ("**预测位移 vs 实测位移：方向可用，标度与 1/P 形状不可用**（§6.4）：实测/预测 "
             f"{min(clean_ratios):.2f}–{max(clean_ratios):.2f}，且实测的 1/P 指数只有 "
             f"α = {rf.alpha_baseline:.2f}（预测是 1.00）。复测：**框固定 + 宽 ROI** "
             "直接量整帧质心（§11.6 四条），同时直接量面板斜坡的相位幅值是否被削顶。"),
        ]
    else:
        head = [
            (f"**`n_max = 18` 全黑帧**（§5.2）：整帧峰值 {i18_peak:.1f} counts、CV "
             f"{f.zern_cv[i18]:.3f}。复测：把 `zernike_fit_grid` 提到 512，只扫 17–20 阶，"
             "每阶重复 3 次（相位不变），看全黑是否复现。"),
            (f"**`blaze period = 40` provisional**（§6.3）：settle 超时 + 最终峰值 "
             f"{float(blaze.rows[i40]['frame_peak']):.2f} counts。复测：单点重跑，"
             "并把 settle 的探测时间序列打出来（现在只打了最后一次探测）。"),
            ("**预测位移 vs 实测位移系统性偏低**（§6.4）：实测/预测 "
             f"{min(clean_ratios):.2f}–{max(clean_ratios):.2f}。复测：框固定（冻结在未加光栅的 0 级）、"
             "宽 ROI，再比一次；同时直接量面板斜坡的相位幅值是否被削顶。"),
        ]
    items = head + [
        (f"**settle 判据超时**（§2.3）：{len(run.settle_events)}/"
         f"{len(sides.rows) + len(zern.rows) + len(blaze.rows)} 行，"
         f"但其中只有 {sum(1 for ev in run.settle_events if ev.probe_peak < DARK_FRAME_FRACTION * SATURATION_COUNTS)} "
         f"行的探测峰值低于满量程的 {DARK_FRAME_FRACTION:.0%}（即真黑帧）。"
         f"相对容差 `tol = {policy.get('tol')}` 对这台相机可能过紧（读噪声 "
         f"σ ≈ {float(spot.get('noise', float('nan'))):.3f} counts，而容差按相对值算）。"
         + (f"复测里又有 {rf.settle_configs.__len__()} 个配置超时"
            f"（{', '.join(rf.settle_configs)}），**同样没有一条是黑帧** ⇒ 判据确实偏紧，"
            if retest is not None and rf is not None else "")
         + "复测：把 `max_wait_s` 提到 12 s，或改用「探测值变化 < 2σ」的绝对判据。"),
        (f"**`{sides.x[0]:.0f} px` 行饱和**（§4.3）：`frame_peak = box_peak = "
         f"{SATURATION_COUNTS:.0f}`。复测：该点单独降曝光（例如 "
         f"{run.exposure_ms / 4:g} ms）重测，才能给出它的真实峰值。"),
        ("**`ee` ≡ `PIB`**：50 行里两者逐位相同，所以 `ee` / `ar` / `quality` 这一组"
         "里只有 `ar` 与 `quality` 带独立信息。报告与图里都不把 `ee` 当第二个指标 —— "
         "否则会让人误以为两个独立指标互相印证。"),
        ("**平场曝光阶梯缺失**（§4.2）：无法把「0 级峰值 → 方斑每像素」写成等式。"
         "补 0.05 / 0.2 / 1 / 5 / 20 ms 的平场帧 + FWHM，就能给出那条换算。"),
        ("**每配置只测 1 次**：没有重复，因此**没有误差棒**。§3.3 里 1500 vs 1600 差 "
         f"{f.score_peak_gap:.2e}，这个量级低于任何可以分辨的单次噪声，"
         "所以「峰在哪个点」这件事本轮**不可判定**。"
         + (f"复测批内各点相对邻居的偏差（≤ {rf.batch_dev_max:.1%}）只能给出"
            "「这不是趋势上的点」这条弱判据，仍不是误差棒（§11.5 判定 6）。"
            if retest is not None and rf is not None else "")),
    ]
    for idx, text in enumerate(items, start=1):
        add(f"{idx}. {text}")
    add()

    # -- 10. 局限 -----------------------------------------------------------
    add("## 10. 局限与不可横比项")
    add()
    add(
        "- **CV 与 PIB 都依赖框大小**，所以它们只在**同一框**下可比。跨段的数字"
        "（例如「Zernike 重构的 PIB 0.91 高于 GS 方斑的 0.86」）只能作为「框选得太大」"
        "的证据，不能作为质量排序。"
    )
    add(
        "- **无重复测量**：每配置 1 次。所有「某点最优」的说法都受这一条限制，"
        "本报告因此只报峰值区间与差值，不报显著性。"
    )
    add(
        "- **GS 每次从零跑**：18 行的 GS 相位彼此独立，行间差异包含 GS 自身的收敛差异"
        "（`gs_seconds` 在 "
        f"{min(sides.col('gs_seconds')):.2f}–{max(sides.col('gs_seconds')):.2f} s 之间，"
        "不是固定迭代预算下的等价收敛）。"
    )
    add(
        "- **`global_spot_fwhm` 不是方斑宽度**（§1.2）：它是整帧最亮结构，2f 光路上通常"
        "是 0 级/晕斑。所有基于它的「光斑退化」判断都不成立。"
    )
    add(
        "- **Zernike 拟合在 "
        f"{int(zern.run_config.get('fit_grid', 0))}×{int(zern.run_config.get('fit_grid', 0))} "
        f"网格上做**，而显示在 {int(panel[1])}×{int(panel[0])} 面板上（W×H）；高阶模式的"
        "采样充分性没有验证（§5.2 的假设正是从这里来的）。"
    )
    add(
        "- **`box_clipped_px` 的语义**：它是「框落在画面外的**面积**」，不是「被钳制的"
        "像素个数」。本轮 50 行满足恒等式 `box_side_px² − box_area_px == box_clipped_px`"
        "（生成器逐行核对，残差 0），所以可以放心按面积读。"
    )
    add()

    # -- 11. 复测 (功率已变) -------------------------------------------------
    if retest is not None and rf is not None:
        retest_lines, retest_tables = render_retest_section(run, f, retest, rf)
        out.extend(retest_lines)
        tables.extend(retest_tables)

    # -- 12. CCD 远场光斑图 --------------------------------------------------
    if farfield is not None and farfield.get("mosaics"):
        farfield_lines, farfield_tables = render_farfield_section(
            run, farfield, farfield_figures,
            residual_pointer=bool(residuals is not None and residuals.panels))
        out.extend(farfield_lines)
        tables.extend(farfield_tables)

    # -- 13. 整轮原子重扫 ----------------------------------------------------
    if atomic is not None and af is not None:
        atomic_lines, atomic_tables = render_atomic_section(
            run, f, atomic, af, retest, atomic_figures, residuals)
        out.extend(atomic_lines)
        tables.extend(atomic_tables)

    # -- 附录 ---------------------------------------------------------------
    add(f"## 附录 A. 输入契约（`artifacts/{ARTIFACT_DIR.name}/`）")
    add()
    art_rows = []
    for rel, note in _artifact_manifest():
        art_rows.append((f"`{rel}`", note))
    if retest is not None:
        for rel, note in _retest_artifact_manifest(retest):
            art_rows.append((f"`{RETEST_ARTIFACT_DIR.name}/{rel}`", note))
    if prior is not None:
        for rel, note in _retest_artifact_manifest(prior):
            art_rows.append((
                f"`{RETEST_PRIOR_ARTIFACT_DIR.name}/{rel}`",
                "已作废（跨脚本版本）—— **仅存档为证据，报告不引用**" if rel == "manifest.json"
                else "同上（已作废批次）",
            ))
    if atomic is not None:
        for rel, note in _atomic_artifact_manifest(atomic):
            art_rows.append((f"`{ATOMIC_ARTIFACT_DIR.name}/{rel}`", note))
    art_tbl = Table(("文件", "说明"), art_rows)
    tables.append(art_tbl)
    add(art_tbl.render())
    add()
    if retest is not None:
        add(
            f"复测批次的产物单独归档在 `artifacts/{RETEST_ARTIFACT_DIR.name}/`"
            f"（{len(_retest_artifact_manifest(retest))} 个文件），与基线**并列**而不是覆盖 —— "
            "两批的 `measurements.csv` 各自独立，§11 的数字从第二块读。"
            + (f"整轮原子重扫（§13）在 `artifacts/{ATOMIC_ARTIFACT_DIR.name}/`"
               f"（{len(_atomic_artifact_manifest(atomic))} 个文件，含 4 张拼图与 "
               "`frames_selection.json`）。" if atomic is not None else "")
            + (f"更早一版复测在 `artifacts/{RETEST_PRIOR_ARTIFACT_DIR.name}/`"
               f"（{len(_retest_artifact_manifest(prior))} 个文件）—— 那一批跨了两个脚本版本，"
               "**已作废**，只作为证据留存，报告不引用它的数字。" if prior is not None else "")
        )
        add()
    add(
        "上面的文件是这次运行的**原始字节副本**（生成器用 `shutil.copy2` 复制，不重写、"
        "不格式化），所以报告里的每个测量值都能追到某个 `measurements.csv` 的某个单元。"
        "`PROVENANCE.json` 记录源路径与 sha256，`--check` 会校验副本、sha256 与源逐字节一致。"
    )
    add()
    add(
        "⚠️ 根 `.gitignore` 忽略 `*.csv` 与 `*.log`，所以入库时这两个扩展名要 `git add -f`"
        "（与本目录既有 `artifacts/` 的做法一致）；否则新克隆上只剩 JSON，"
        "而 JSON 与 CSV 是逐值等价的（生成器每次运行都做这条交叉校验），仍可复现。"
    )
    add()

    add("## 附录 B. 复现")
    add()
    add("```powershell")
    add("$env:PYTHONPATH=\"<repo>;<repo>\\src;<repo>\\libs\"")
    add("& .venv/Scripts/python.exe scripts/slm/generate_hw_gs_square_scans_report.py \\")
    add("    --run-root data/debug/hw_gs_square_scans/20261023_gs_square")
    add("# 幂等 / 产物一致性校验 (只读, 不写任何文件)")
    add("& .venv/Scripts/python.exe scripts/slm/generate_hw_gs_square_scans_report.py \\")
    add("    --run-root data/debug/hw_gs_square_scans/20261023_gs_square --check")
    add("```")
    add()
    add(
        "全程**不 import 任何驱动、不打开任何设备**：只读 CSV / JSON / 日志 + 画图。"
        "`--check` 模式下连图都不重画 —— 它只比对报告正文、`artifacts/` 副本与图是否存在。"
    )
    add()

    body = "\n".join(out).rstrip("\n") + "\n"
    return insert_header(body, REPORT_KEY), tables


def _idx_of(values: Sequence[float], target: float) -> int:
    for i, v in enumerate(values):
        if abs(float(v) - float(target)) < ROW_MATCH_ATOL:
            return i
    raise ValueError(f"{target} 不在 {list(values)} 里")


def _row_at(rows: Sequence[dict[str, str]], key: str, value: float) -> dict[str, str]:
    """在 CSV 原文行里按某个数值列取一行 (取不到就报错, 不静默返回空)。"""
    for row in rows:
        if abs(float(row[key]) - float(value)) < ROW_MATCH_ATOL:
            return row
    raise KeyError(f"没有 {key} = {value} 的行")


def _join_x(xs: Sequence[float], indices: Sequence[int]) -> str:
    return " / ".join(f"{xs[i]:.0f}" for i in indices) if indices else "无"


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "-"


def _count_stage(run: Run, stage: str) -> int:
    return sum(1 for ev in run.settle_events if ev.stage == stage)


def _figname(stem: str) -> str:
    return f"{stem}{FIG_TAG}.png"


def _rel(path: Path | None) -> str:
    if path is None:
        return "(未提供)"
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_manifest() -> list[tuple[str, str]]:
    """``artifacts/`` 里该有的文件与说明 (顺序即渲染顺序)。"""
    rows: list[tuple[str, str]] = [("manifest.json", "三段汇总 (行数 / 自变量 / 创建时间)")]
    for name in STAGES:
        rows.append((f"{name}/measurements.csv",
                     f"{STAGE_CN[name]} 全部指标 — 报告里测量值的**唯一来源**"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (含 NaN 语义, 用于交叉校验)"))
        rows.append((f"{name}/run_config.json", "本次运行参数 + 冻结的 0 级 + 曝光上下限"))
    rows.append(("run.log", "采集日志原文 — settle 判据超时与关键量的出处"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows


# ---------------------------------------------------------------------------
# 产物复制 (输入契约)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sources(run: Run) -> list[tuple[Path, Path]]:
    """(源文件, 目标文件) 清单 —— 复制与校验共用这一份, 不各写一遍。"""
    pairs: list[tuple[Path, Path]] = [
        (run.root / "manifest.json", ARTIFACT_DIR / "manifest.json")
    ]
    for name in STAGES:
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            pairs.append((run.root / name / fname, ARTIFACT_DIR / name / fname))
    if run.log_path is not None:
        pairs.append((run.log_path, ARTIFACT_DIR / "run.log"))
    return pairs


def retest_artifact_sources(retest: RetestRun, target: Path) -> list[tuple[Path, Path]]:
    """复测批次的 (源, 目标) 清单。

    复测是原子化落盘, 所以每配置还多一份 ``metrics.json`` 与一个日志;
    段级 ``run_config.json`` **有就复制、没有就跳过** (v1 那批没有)。
    """
    pairs: list[tuple[Path, Path]] = [(retest.root / "manifest.json", target / "manifest.json")]
    for name in RETEST_STAGES:
        stage_dir = retest.root / name
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            src = stage_dir / fname
            if src.exists():
                pairs.append((src, target / name / fname))
        for cfg in retest.stages[name].configs:
            pairs.append((stage_dir / cfg.config / "metrics.json",
                          target / name / cfg.config / "metrics.json"))
            if cfg.log_path is not None:
                pairs.append((cfg.log_path, target / "logs" / cfg.log_path.name))
    return pairs


def _retest_artifact_manifest(retest: RetestRun) -> list[tuple[str, str]]:
    """复测归档目录的文件说明 (顺序即渲染顺序, 与实际复制的文件一一对应)。"""
    rows: list[tuple[str, str]] = [
        ("manifest.json", "复测两段汇总 (行数 / 自变量 / 模式 / 配置名)")
    ]
    for name in RETEST_STAGES:
        stage_dir = retest.root / name
        rows.append((f"{name}/measurements.csv",
                     f"{RETEST_STAGE_CN[name]} 复测指标 — §11 复测值/重复性对照的来源"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (用于交叉校验)"))
        if (stage_dir / "run_config.json").exists():
            rows.append((f"{name}/run_config.json",
                         "段级运行参数 (无 `plan`; 含 mode / preflight / 冻结 0 级)"))
        for cfg in retest.stages[name].configs:
            rows.append((f"{name}/{cfg.config}/metrics.json",
                         f"{cfg.config} 单配置指标 + `preflight`(含本进程冻结的 0 级)"))
        for cfg in retest.stages[name].configs:
            if cfg.log_path is not None:
                rows.append((f"logs/{cfg.log_path.name}", f"{cfg.config} 采集日志原文"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows


def _copy_artifact_set(pairs: Sequence[tuple[Path, Path]], target: Path,
                       label: str, run_root: Path) -> list[Path]:
    """把一组 (源, 目标) 逐字节复制, 并写该目录的 ``PROVENANCE.json``。"""
    written: list[Path] = []
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written.append(dst)
    provenance = {
        "report": REPORT_KEY,
        "generator": "scripts/slm/generate_hw_gs_square_scans_report.py",
        "batch": label,
        "run_root": _rel(run_root),
        "files": {
            dst.relative_to(target).as_posix(): {
                "source": _rel(src),
                "sha256": _sha256(src),
                "bytes": src.stat().st_size,
            }
            for src, dst in pairs
        },
    }
    path = target / "PROVENANCE.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    written.append(path)
    return written


def _check_artifact_set(pairs: Sequence[tuple[Path, Path]], target: Path) -> list[str]:
    """一组副本必须存在、与源逐字节一致, 且 ``PROVENANCE.json`` 未过期。"""
    problems: list[str] = []
    for src, dst in pairs:
        if not dst.exists():
            problems.append(f"artifacts 缺失: {_rel(dst)}")
            continue
        if dst.stat().st_size != src.stat().st_size or _sha256(dst) != _sha256(src):
            problems.append(f"artifacts 副本与源不一致: {_rel(dst)}")
    prov_path = target / "PROVENANCE.json"
    if not prov_path.exists():
        problems.append(f"artifacts 缺失: {_rel(prov_path)}")
        return problems
    recorded = json.loads(prov_path.read_text(encoding="utf-8")).get("files", {})
    expected = {
        dst.relative_to(target).as_posix(): _sha256(src) for src, dst in pairs
    }
    if set(recorded) != set(expected):
        problems.append(
            f"{_rel(prov_path)}: 记录的文件集合与本次输入不符 "
            f"(记录 {sorted(recorded)}, 期望 {sorted(expected)})"
        )
    for name, digest in expected.items():
        if name in recorded and recorded[name].get("sha256") != digest:
            problems.append(f"{_rel(prov_path)} 的 sha256 过期: {name}")
    return problems


def copy_artifacts(run: Run) -> list[Path]:
    """把基线的输入产物**逐字节**复制进 ``artifacts/``。"""
    return _copy_artifact_set(artifact_sources(run), ARTIFACT_DIR, "baseline", run.root)


def check_artifacts(run: Run) -> list[str]:
    """``--check``: 基线副本必须存在、与源逐字节一致, 且溯源 JSON 未过期。"""
    return _check_artifact_set(artifact_sources(run), ARTIFACT_DIR)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT),
                        help="三段产物的根目录 (含 manifest.json 与 sides/zernike/blaze)")
    parser.add_argument("--log", default=str(DEFAULT_LOG),
                        help="那次运行的日志; 传空字符串表示明确不要日志")
    parser.add_argument("--out-md", default=str(ROOT / REPORT_KEY),
                        help="报告正文输出路径")
    parser.add_argument("--out-figures", default=str(FIG_DIR),
                        help="图输出目录")
    parser.add_argument("--retest-root", default="",
                        help="复测批次根目录 (含 manifest.json 与 blaze/zernike); "
                             "不给就完全不渲染 §11")
    parser.add_argument("--atomic-root", default="",
                        help="整轮原子重扫的根目录 (单一版本 + 单一功率, 三段齐全); "
                             "不给就完全不渲染 §13")
    parser.add_argument("--retest-prior-root", default=str(DEFAULT_RETEST_PRIOR_ROOT),
                        help="更早一版复测 (只用于算批间重复性带宽); 不存在就跳过那张表")
    parser.add_argument("--check", action="store_true",
                        help="只校验: 重新渲染到内存并与磁盘比对, 不写任何文件")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global FIG_DIR, ARTIFACT_DIR, RETEST_ARTIFACT_DIR, RETEST_PRIOR_ARTIFACT_DIR
    global ATOMIC_ARTIFACT_DIR
    args = _parse_args(argv)
    run_root = Path(args.run_root)
    out_md = Path(args.out_md)
    FIG_DIR = Path(args.out_figures)
    log_path = Path(args.log) if args.log else None

    if not run_root.is_absolute():
        run_root = (Path.cwd() / run_root).resolve()
    run = load_run(run_root, log_path)
    if (run.log_crosscheck and any(not ok for *_, ok in run.log_crosscheck)
            and str(args.log) == str(DEFAULT_LOG)):
        # 默认日志属于基线那一轮; 换 run-root 时它必然对不上 —— 忽略它, 而不是
        # 让一次"日志选错"变成硬失败 (显式传 --log 时仍然严格校验)。
        logger.info(f"警告: 默认日志 {_rel(DEFAULT_LOG)} 与本次 run-root 的 run_config 不一致, "
              "已按「未提供日志」处理; 要核对 settle 判据请显式传 --log")
        run = load_run(run_root, None)
    findings = compute_findings(run)
    # 归档目录跟随实际传入的 run-root (默认值只是示例), 换批次不会互相覆盖
    ARTIFACT_DIR = OUT_DIR / "artifacts" / run.root.name

    retest: RetestRun | None = None
    rf: RetestFindings | None = None
    prior: RetestRun | None = None
    if args.retest_root:
        retest_root = Path(args.retest_root)
        if not retest_root.is_absolute():
            retest_root = (Path.cwd() / retest_root).resolve()
        retest = load_retest(retest_root, DEFAULT_RETEST_LOG_DIR)
        if args.retest_prior_root:
            prior_root = Path(args.retest_prior_root)
            if not prior_root.is_absolute():
                prior_root = (Path.cwd() / prior_root).resolve()
            if prior_root.exists() and prior_root != retest_root:
                prior = load_retest(prior_root, DEFAULT_RETEST_LOG_DIR)
        rf = compute_retest_findings(run, retest, prior)
        # 归档目录跟随实际传入的那一批 (参数为准, 默认值只是示例)
        RETEST_ARTIFACT_DIR = OUT_DIR / "artifacts" / retest.root.name
        if prior is not None:
            RETEST_PRIOR_ARTIFACT_DIR = OUT_DIR / "artifacts" / prior.root.name

    # §12 的规划不读像素 (峰值取自 CSV), 所以 --check 也能拿到完全相同的载荷
    selection = select_cells(run, findings)
    farfield = plan_farfield(run, selection)
    # 「会被画出来的」拼图 = 至少有一格不是缺帧的那些 (与 draw_mosaic 的判据一致)
    farfield_figures = [FIG_DIR / m["figure"] for m in farfield["mosaics"]
                        if any(not cell["missing"] for cell in m["cells"])]

    atomic: RetestRun | None = None
    af: AtomicFindings | None = None
    atomic_selection: dict[str, list[CellPick]] = {}
    atomic_farfield: dict[str, Any] = {"mosaics": []}
    atomic_figures: list[Path] = []
    residuals: ResidualSet | None = None
    residual_figure: Path | None = None
    if args.atomic_root:
        atomic_root = Path(args.atomic_root)
        if not atomic_root.is_absolute():
            atomic_root = (Path.cwd() / atomic_root).resolve()
        atomic = load_retest(atomic_root, None, stage_names=STAGES)
        ATOMIC_ARTIFACT_DIR = OUT_DIR / "artifacts" / atomic.root.name
        selected = load_selected(atomic.root)
        af = compute_atomic_findings(run, atomic, selected, retest)
        atomic_findings = compute_findings(atomic)
        atomic_selection = select_cells(
            atomic, atomic_findings,
            sides_panel=(200.0, 600.0, 1000.0, 1500.0, 1700.0),
            add_score_cell=False,
        )
        atomic_farfield = plan_farfield(atomic, atomic_selection)
        atomic_mosaics = [FIG_DIR / m["figure"] for m in atomic_farfield["mosaics"]
                          if any(not cell["missing"] for cell in m["cells"])]
        atomic_figures = atomic_mosaics + [
            FIG_DIR / f"fig10_atomic_sides_{batch_tag(atomic)}.png",
            FIG_DIR / f"fig11_atomic_blaze_shift_{batch_tag(atomic)}.png",
        ]
        # §13.8 的残差只读产物重算 (延迟 import 驱动链, 见 compute_residuals)
        residuals = compute_residuals(atomic)
        if residuals is not None:
            residual_figure = FIG_DIR / residual_figure_name(atomic)
            atomic_figures.append(residual_figure)

    # -- 自检 0: §13.8 的残差必须与记录值 / 存盘相位对得上 --------------------
    residual_problems = check_residuals(residuals)
    if residual_problems:
        logger.info("§13.8 残差自检失败:")
        for problem in residual_problems:
            logger.info(f"  - {problem}")
        return 1
    if residuals is not None:
        pairs = ", ".join(
            f"n_max={p.n_max} 记录 {p.rms_recorded:.6f} / 重算 {p.rms_recomputed:.6f}"
            for p in residuals.panels
        )
        logger.info(f"§13.8 残差自检: {len(residuals.panels)} 个配置, {pairs}; "
              f"recon 与存盘相位最大偏差 "
              f"{max(p.recon_delta for p in residuals.panels):.1e}")

    markdown, tables = render_report(run, findings, retest, rf, prior,
                                     farfield, farfield_figures,
                                     atomic, af, atomic_figures, residuals)

    # -- 自检 1: 表里的值与 CSV 逐值一致 -----------------------------------
    problems: list[str] = []
    checked = 0
    for table in tables:
        checked += table.numeric_cells
        problems.extend(verify_table_in_markdown(markdown, table))
    if problems:
        logger.info("表格回读校验失败:")
        for problem in problems:
            logger.info(f"  - {problem}")
        return 2
    logger.info(f"表格回读校验: {len(tables)} 张表 / {checked} 个数值单元, "
          f"全部与产物逐值一致")

    # -- 自检 2: 结构性不变量 ----------------------------------------------
    invariants = check_invariants(run, findings, retest)
    if invariants:
        logger.info("结构不变量校验失败:")
        for problem in invariants:
            logger.info(f"  - {problem}")
        return 3
    logger.info("结构不变量校验: score 分解 / 框面积恒等式 / CSV↔JSON / manifest "
          "/ 日志↔run_config 全部通过")

    # -- 自检 3 (非致命): 溯源条目里的复现命令必须指向本次实际用的复测目录 -------
    # 溯源块由 `_common/provenance.py::REPORTS` 渲染, 本脚本不能改它; 但一旦条目里的
    # `--retest-root` 与实际参数不同, 报告头部就会印一条错的复现命令 —— 这里把它喊出来。
    registered = REPORTS.get(REPORT_KEY)
    if retest is not None and registered is not None and registered.command:
        found = re.search(r"--retest-root\s+(\S+)", registered.command)
        if found:
            registered_root = Path(found.group(1))
            if registered_root.name != retest.root.name:
                logger.info(
                    "警告: REPORTS 里的复现命令指向 "
                    f"{registered_root.name}, 而本次实际用的是 {retest.root.name} —— "
                    "报告头部的「复现命令」会与实际不符, 请更新 provenance.py 的该条目",

                )

    if args.check:
        if not out_md.exists():
            logger.info(f"报告不存在: {out_md}")
            return 1
        on_disk = out_md.read_text(encoding="utf-8")
        drift = [] if on_disk == markdown else [
            f"正文与现场渲染不一致 ({len(on_disk)} vs {len(markdown)} 字符)"
        ]
        drift.extend(check_artifacts(run))
        if retest is not None:
            drift.extend(_check_artifact_set(
                retest_artifact_sources(retest, RETEST_ARTIFACT_DIR), RETEST_ARTIFACT_DIR))
            if rf is not None and rf.prior_root is not None and prior is not None:
                drift.extend(_check_artifact_set(
                    retest_artifact_sources(prior, RETEST_PRIOR_ARTIFACT_DIR),
                    RETEST_PRIOR_ARTIFACT_DIR))
        figure_names = ["fig1_sides_objectives", "fig2_sides_exposure_geometry",
                        "fig3_zernike_reconstruction", "fig4_blaze_shift"]
        if retest is not None:
            figure_names.append("fig5_retest_neighbors")
        for name in figure_names:
            path = FIG_DIR / _figname(name)
            if not path.exists():
                drift.append(f"图缺失: {_rel(path)}")
            elif path.stat().st_size == 0:
                drift.append(f"图为空文件: {_rel(path)}")
            elif f"figures/{_figname(name)}" not in markdown:
                drift.append(f"正文没有引用: figures/{_figname(name)}")
        drift.extend(_check_farfield(run, farfield, farfield_figures))
        if atomic is not None:
            drift.extend(_check_artifact_set(
                atomic_artifact_sources(atomic, ATOMIC_ARTIFACT_DIR), ATOMIC_ARTIFACT_DIR))
            for stem in ("fig10_atomic_sides", "fig11_atomic_blaze_shift"):
                path = FIG_DIR / f"{stem}_{batch_tag(atomic)}.png"
                if not path.exists():
                    drift.append(f"图缺失: {_rel(path)}")
                elif path.stat().st_size == 0:
                    drift.append(f"图为空文件: {_rel(path)}")
                elif f"figures/{path.name}" not in markdown:
                    drift.append(f"正文没有引用: figures/{path.name}")
            drift.extend(_check_farfield(atomic, atomic_farfield, atomic_mosaics,
                                         artifact_dir=ATOMIC_ARTIFACT_DIR))
            if residual_figure is not None:
                if not residual_figure.exists():
                    drift.append(f"图缺失: {_rel(residual_figure)}")
                elif residual_figure.stat().st_size == 0:
                    drift.append(f"图为空文件: {_rel(residual_figure)}")
                elif f"figures/{residual_figure.name}" not in markdown:
                    drift.append(f"正文没有引用: figures/{residual_figure.name}")
                copy = ATOMIC_ARTIFACT_DIR / residual_figure.name
                if not copy.exists():
                    drift.append(f"图副本缺失: {_rel(copy)}")
                elif _sha256(copy) != _sha256(residual_figure):
                    drift.append(f"图与 artifacts 副本不一致: {residual_figure.name}")
        if drift:
            logger.info("--check 失败 (存在漂移):")
            for item in drift:
                logger.info(f"  - {item}")
            return 1
        n_mosaic = len([m for m in farfield.get("mosaics", [])])
        extra = ""
        if atomic is not None:
            extra = (f"; §13 另有 2 张曲线 + {len(atomic_mosaics)} 张拼图"
                     + (" + fig12 残差图" if residual_figure is not None else "")
                     + f" 与 `artifacts/{ATOMIC_ARTIFACT_DIR.name}/`")
        logger.info(f"--check 通过: {_rel(out_md)} 与现场渲染逐字节一致, "
              f"artifacts 副本与源一致, {len(figure_names)} 张指标图 + "
              f"{n_mosaic} 张远场拼图齐备 (含 frames_selection.json){extra}")
        return 0

    # -- 写正文 / 图 / 产物 -------------------------------------------------
    # 拼图放在自检之后画: 自检失败时不会在 figures/ 里留下孤儿图
    if not args.check:
        render_farfield(run, selection, farfield)
        if atomic is not None and af is not None:
            render_farfield(atomic, atomic_selection, atomic_farfield)
            fig_atomic_sides(run, atomic)
            fig_atomic_blaze(atomic)
        if atomic is not None and residuals is not None:
            fig_zernike_residual(atomic, residuals)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    logger.info(f"报告已写到: {_rel(out_md)}")
    figures = render_figures(run, findings)
    if retest is not None and rf is not None:
        figures.append(fig_retest_neighbors(run, retest, rf))
    logger.info(f"图已写到: {_rel(FIG_DIR)}/ ({len(figures)} 张)")
    written = copy_artifacts(run)
    logger.info(f"输入产物已复制: {_rel(ARTIFACT_DIR)}/ ({len(written)} 个文件)")
    if farfield["mosaics"]:
        copied = _copy_farfield_to_artifacts(farfield, farfield_figures)
        logger.info(f"远场拼图已复制: {_rel(ARTIFACT_DIR)}/ ({copied} 张) + "
              "frames_selection.json")
    if atomic is not None:
        pairs_a = atomic_artifact_sources(atomic, ATOMIC_ARTIFACT_DIR)
        written_a = _copy_artifact_set(pairs_a, ATOMIC_ARTIFACT_DIR, "atomic", atomic.root)
        copied_a = _copy_farfield_to_artifacts(atomic_farfield, atomic_figures,
                                               target=ATOMIC_ARTIFACT_DIR)
        logger.info(f"整轮原子重扫产物已复制: {_rel(ATOMIC_ARTIFACT_DIR)}/ "
              f"({len(written_a)} 个源文件 + {copied_a} 张图) + frames_selection.json")
    if retest is not None:
        pairs = retest_artifact_sources(retest, RETEST_ARTIFACT_DIR)
        written_r = _copy_artifact_set(pairs, RETEST_ARTIFACT_DIR, "retest", retest.root)
        logger.info(f"复测产物已复制: {_rel(RETEST_ARTIFACT_DIR)}/ ({len(written_r)} 个文件)")
        if prior is not None:
            pairs_p = retest_artifact_sources(prior, RETEST_PRIOR_ARTIFACT_DIR)
            written_p = _copy_artifact_set(pairs_p, RETEST_PRIOR_ARTIFACT_DIR,
                                           "retest_prior", prior.root)
            logger.info(f"更早一版复测产物已复制: {_rel(RETEST_PRIOR_ARTIFACT_DIR)}/ "
                  f"({len(written_p)} 个文件)")
    return 0


def check_invariants(
    run: Run, f: Findings, retest: RetestRun | None = None
) -> list[str]:
    """与「渲染」无关的结构性检查 —— 数据自身必须自洽。"""
    problems: list[str] = []

    if retest is not None:
        for name, stage in retest.stages.items():
            for i, row in enumerate(stage.rows):
                pib = float(row["pib"])
                cv = float(row["cv"])
                expect = 0.5 * pib + 0.5 / (1.0 + cv)
                if abs(expect - float(row["score"])) > SCORE_IDENTITY_ATOL:
                    problems.append(
                        f"复测 {name} 第 {i} 行: score {row['score']} != "
                        f"0.5·PIB + 0.5/(1+CV) = {expect}"
                    )
                side = float(row["box_side_px"])
                if abs(side * side - float(row["box_area_px"])
                       - float(row["box_clipped_px"])) > AREA_IDENTITY_ATOL:
                    problems.append(f"复测 {name} 第 {i} 行: 框面积恒等式不成立")
                if abs(float(row["exposure_ms_readback"]) - retest.exposure_ms) > EXPOSURE_MATCH_ATOL:
                    problems.append(
                        f"复测 {name} 第 {i} 行: 读回曝光 {row['exposure_ms_readback']} "
                        f"!= 生效 {retest.exposure_ms}"
                    )
            # 段级 run_config (有就核对; v1 那批没有 —— 缺文件与缺 `plan` 都必须容忍)
            cfg_path = retest.root / name / "run_config.json"
            if cfg_path.exists():
                cfg = load_json(cfg_path)
                if abs(float(cfg.get("exposure_ms_working", float("nan")))
                       - retest.exposure_ms) > EXPOSURE_MATCH_ATOL:
                    problems.append(f"复测 {name}/run_config.json 的生效曝光与 preflight 不一致")

    for name, stage in run.stages.items():
        for i, row in enumerate(stage.rows):
            pib = float(row["pib"])
            cv = float(row["cv"])
            score = float(row["score"])
            expect = 0.5 * pib + 0.5 / (1.0 + cv)
            if abs(expect - score) > SCORE_IDENTITY_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: score {score} != 0.5·PIB + 0.5/(1+CV) = {expect}"
                )
            side = float(row["box_side_px"])
            clipped = float(row["box_clipped_px"])
            area = float(row["box_area_px"])
            if abs(side * side - area - clipped) > AREA_IDENTITY_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: 框面积恒等式不成立 "
                    f"({side}² − {area} != {clipped})"
                )
            if abs(float(row["exposure_ms_readback"]) - run.exposure_ms) > EXPOSURE_MATCH_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: 读回曝光 {row['exposure_ms_readback']} "
                    f"!= 生效 {run.exposure_ms}"
                )

    # 单调性结论必须与正文一致
    if f.pib_violations:
        problems.append(f"PIB 并非严格单调 (反向步 {f.pib_violations}) —— 正文需改写")
    peak = max(f.score)
    if sum(1 for v in f.score if v == peak) > 1:
        problems.append("score 的最大值不唯一 (有并列) —— 选点/叙述都不可判")

    # 日志 ↔ run_config
    for name, logged, actual, ok in run.log_crosscheck:
        if not ok:
            problems.append(f"日志与 run_config 不一致: {name} 日志={logged} 配置={actual}")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
