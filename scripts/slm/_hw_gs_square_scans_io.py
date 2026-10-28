"""GS 方扫报告的**落盘产物读取层**。

只做一件事：把 ``measurements.csv`` / ``measurements.json`` / ``run_config.json``
/ ``manifest.json`` 与那次运行的日志读成 :mod:`_hw_gs_square_scans_model` 里的
数据结构，并就地做 CSV↔JSON 交叉校验。

**它不渲染、不画图、不算指标** —— 报告里"数字从哪来"全部收敛在这里，
下游拿到的都是已经交叉核对过的对象。
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence


# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.slm._hw_gs_square_scans_model import (
    Cell,
    RETEST_STAGES,
    RetestConfig,
    RetestRun,
    RetestStage,
    Run,
    STAGES,
    SettleEvent,
    SettleRecord,
    Stage,
    Table,
    VARIABLE,
)

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
