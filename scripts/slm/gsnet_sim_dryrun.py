"""issue #165 硬件链路的**全链离线预演器**（实验前把命令链验证一遍）。

**它是什么.** issue #165 的实测部分要在实验系统上跑，方案见
``docs/slm/gsnet_hw_plan_min_time.md``（两段独占设备的会话 S1/S2 + 一大段离线块）。
设备稀缺 ⇒ 实验前必须先把**整条命令链**在离线环境里验证通过：命令能跑、产物路径对、
上游依赖的脚本参数名不写错、门控判据能触发。本脚本就是那个「预演器」。

**它不是什么.**

* **不打开任何设备。** 相机 / SLM 一律不碰；本文件不 import 任何硬件驱动
  （``ao_shaping.drivers.daheng`` / ``santec`` …），只 import 仓库的**仿真**孪生
  （``ao_shaping.drivers.sim.bench_twin``）与纯数学工具。
* **不是实验系统上的预演。** 探针数据是**仿真**的，耦合曲线是**合成**的；
  报告里逐条写明「仿真 ≠ 实验系统」。

**为什么要有阶段 2（``probe_fixture``）.** 仿真路径（``--cam-type sim``）**不写**
``probes/*.npz``、也**不写** ``preflight/geometry.json``（探针记录只在硬件路径上
落盘，见 ``src/ao_shaping/optimizer/wfless/slm_model_in_loop.py`` 的探针落盘段与
几何预检段）⇒ 阶段 3（拟合孪生）**不能**直接吃仿真运行目录。阶段 2 造一份
「形状与硬件完全一致」的探针数据，好让 ``fit_bench_twin.py --from-dir`` 与
``gsnet_transfer.py --from-dir`` 这两个离线脚本能被真实地跑一遍。

**阶段清单（固定顺序，每阶段一条 verdict）**

=========================  ====================================================
阶段                       做什么
=========================  ====================================================
``sim_mil``               跑一次仿真的 model-in-loop；校验 4 个产物；**并记录
                          仿真路径不落 probes/ 与 preflight/geometry.json**
``probe_fixture``         造一份形状与硬件一致的**仿真**探针数据（probes/ +
                          manifest.json + preflight/geometry.json +
                          source_amplitude.npy）
``twin_fit``              ``fit_bench_twin.py --from-dir``；校验几何来自
                          fixture 的 geometry.json（**不是**标称焦距回退）
``coupling``              合成灰度响应 CSV → ``fit_slm_amplitude_coupling.py``；
                          校验拟合周期 ≈ 993 且 c / ψ 与合成值相符
``export``                ``export_gsnet_phase.py --forward calibrated``；
                          无 checkpoint ⇒ BLOCKED
``transfer``              ``gsnet_transfer.py --from-dir --gate``；
                          无 checkpoint ⇒ BLOCKED
``ab_plan``               ``gsnet_init_hw_ab.py --plan``（**绝不打开设备**）；
                          校验臂顺序交错且第 i 对按奇偶翻转
``dual_aperture``         ``dual_aperture_cv.py --from-dir``；校验两个口径各一行
=========================  ====================================================

**绝不静默跳过.** 任何没跑的阶段都记成 ``BLOCKED`` 并写明原因（``--only`` /
``--skip`` 的选择、以及上游阶段失败都会写进 ``notes``）。

用法::

    python scripts/slm/gsnet_sim_dryrun.py
    python scripts/slm/gsnet_sim_dryrun.py --only probe_fixture --only coupling
    python scripts/slm/gsnet_sim_dryrun.py --skip sim_mil --timeout-s 600
    python scripts/slm/gsnet_sim_dryrun.py --strict

退出码::

    0  没有阶段 FAILED（BLOCKED 不算失败，除非 --strict）
    1  有阶段 FAILED（或 --strict 下有 BLOCKED）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# ``src`` 必须排在 ``ROOT`` 前面：``ml`` 只在 ``src/`` 下有（``scripts/ml`` 会遮蔽它）。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts._common import markdown_table  # noqa: E402

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
MAIN = ROOT / "src" / "ao_shaping" / "main.py"
FIT_TWIN = ROOT / "scripts" / "slm" / "fit_bench_twin.py"
FIT_COUPLING = ROOT / "scripts" / "slm" / "fit_slm_amplitude_coupling.py"
EXPORT_PHASE = ROOT / "scripts" / "slm" / "export_gsnet_phase.py"
GSNET_TRANSFER = ROOT / "scripts" / "slm" / "gsnet_transfer.py"
HW_AB = ROOT / "scripts" / "slm" / "gsnet_init_hw_ab.py"
DUAL_APERTURE = ROOT / "scripts" / "slm" / "dual_aperture_cv.py"

# ---------------------------------------------------------------------------
# 阶段注册表（顺序固定 —— 测试按这个顺序断言）
# ---------------------------------------------------------------------------
STAGES: tuple[str, ...] = (
    "sim_mil",
    "probe_fixture",
    "twin_fit",
    "coupling",
    "export",
    "transfer",
    "ab_plan",
    "dual_aperture",
)

STAGE_TITLES: dict[str, str] = {
    "sim_mil": "仿真 model-in-loop（不打开设备）",
    "probe_fixture": "造形状一致的仿真探针数据",
    "twin_fit": "拟合孪生（fit_bench_twin --from-dir）",
    "coupling": "灰度→振幅耦合的离线自洽检查",
    "export": "用拟合前向导出相位（export_gsnet_phase）",
    "transfer": "迁移学习（gsnet_transfer --from-dir --gate）",
    "ab_plan": "闭环 A/B 计划（--plan，绝不打开设备）",
    "dual_aperture": "双口径 CV 离线后处理（dual_aperture_cv）",
}

#: 阶段之间的依赖：依赖阶段没有 OK ⇒ 本阶段 BLOCKED。
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "sim_mil": (),
    "probe_fixture": (),
    "twin_fit": ("probe_fixture",),
    "coupling": (),
    "export": ("probe_fixture", "twin_fit"),
    "transfer": ("probe_fixture", "twin_fit"),
    "ab_plan": (),
    "dual_aperture": ("probe_fixture",),
}

#: 依赖阶段「已经跑成功」之外，还可以用磁盘上**已经存在**的产物顶上
#: （例如上一次预演留下的 fixture）。只有依赖阶段**失败**时一律不许复用。
DEPENDENCY_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "probe_fixture": ("fixture",),
    "twin_fit": ("twin_json",),
}

#: 这两个阶段没有 checkpoint 就没法跑 —— 判决里要写明原因。
CHECKPOINT_STAGES: tuple[str, ...] = ("export", "transfer")

STATUS_OK = "OK"
STATUS_BLOCKED = "BLOCKED"
STATUS_FAILED = "FAILED"

# ---------------------------------------------------------------------------
# 几何常量（fixture 与阶段 1 的仿真 MIL 共用同一套 —— 一致才谈得上「预演」）
# ---------------------------------------------------------------------------
REGION = 64
"""瞳面网格边长（模型采样点数）；与阶段 1 的仿真 MIL 同网格。"""

MIL_FAR_FIELD_PADDING = 8
"""阶段 1（仿真 MIL）的远场补零因子 —— 任务书给定的命令。"""

FIXTURE_PADDING = 4
"""fixture 的远场补零因子。

取 **4** 而不是 8 是**算力**原因：反演的目标函数每评估一次要跑一遍
``region × padding`` 的复数 FFT 加一次 numpy SSIM，网格边长翻倍 ⇒ 单次评估
慢 4 倍，而 L-BFGS-B 的有限差分梯度每次迭代要 2×参数个数次评估。离线预演
必须能在几分钟内跑完，所以 fixture 用 256×256 的模型网格（实验系统是
``region 128 × padding 8``）。口径差别已写进报告。
"""

MODEL_SIDE = REGION * FIXTURE_PADDING
"""fixture 里 ``model_grid`` 的边长（模型采样点）= 256。"""

CAMERA_PX_PER_MODEL_PX = float(FIXTURE_PADDING)
"""相机像元 / 模型采样点。

仿真实验系统的 ``camera_px_per_model_px`` 恰等于 ``far_field_padding``
（``slm_model_in_loop.py`` 的 ``_SimBench.true_geometry``：「整幅裁剪下
``far_field_padding`` 个相机像素对应 1 个模型像素」）。fixture 沿用同一口径。
"""

CAM_SIZE = 400
"""相机窗口边长（相机像元）。"""

TARGET_SIDE = 288
"""目标方框边长（相机像元）；与阶段 1 的 ``--target-side`` 同值。"""

PROBE_SPREAD = 4.0
"""探针相位 std（rad）；运行时 ``probe_spread < 3.0`` 会直接报错。"""

N_TRAIN_PROBES = 8
N_HOLDOUT_PROBES = 2

GRAY_LEVELS = 993
"""器件在 1064 nm 的 2π 灰度数（**不是** 10-bit 满量程 1023）。"""

#: 面板 / 透镜要反演的 Zernike 模式（Noll 序）。刻意避开 Noll 4（离焦）：
#: 面板离焦与相机离焦在远场上近似简并，混在一起会让两者的份额不可辨识。
FIT_PANEL_MODES = "5,6,7"
FIT_LENS_MODES = "5,6"

#: 拟合孪生的两阶段预算（离线预演要快；实验那一步用脚本默认的 12/8/60）。
TWIN_LIMIT = 6
TWIN_MAXITER = 1
TWIN_POPSIZE = 2
TWIN_MAXITER_FINE = 2

#: 合成耦合曲线的真值（c / ψ）与拟合容差。
SYNTH_C = 0.12
SYNTH_PSI = 0.70
COUPLING_PERIOD_TOL = 0.02
"""拟合周期与 2π 灰度数的最大相对偏差（与 ``fit_slm_amplitude_coupling.py``
的 ``PERIOD_LEVELS_TOL`` 同口径）。"""

COUPLING_C_TOL = 0.01
COUPLING_PSI_TOL = 0.05
"""c / ψ 的拟合容差（**本预演自定**，报告里写明）。"""

#: 灰度→振幅耦合 CSV 的列名 —— **从 ``validate_flat_phase_gray.py`` 直接读**
#: （它才是写这份 CSV 的那个脚本），不复制一份 —— 复制就会漂移。
GRAY_CSV_SCRIPT = ROOT / "scripts" / "slm" / "validate_flat_phase_gray.py"

#: ``validate_flat_phase_gray.py`` 不可用时的兜底列名（内容必须与上游一致）。
GRAY_CSV_FIELDNAMES_FALLBACK: tuple[str, ...] = (
    "gray_value",
    "max_brightness",
    "min_brightness",
    "sum_brightness",
    "bucket_peak",
    "bucket_sum",
    "zero_order_x",
    "zero_order_y",
    "frame_dtype",
    "frame_shape",
    "slm_max_gray",
    "wavelength_nm",
    "exposure_ms",
    "cam_type",
    "cam_id",
    "slm_serial",
    "timestamp",
)

#: ``gsnet_init_hw_ab.py`` 的两臂与交错协议（第 i 对按 i 的奇偶翻转）。
AB_ARMS = ("gsnet", "physical")


def ab_expected_order(pairs: int) -> list[tuple[int, str]]:
    """交错重放的期望执行顺序（与 ``gsnet_init_hw_ab.stage_order`` 同规则）。"""
    order: list[tuple[int, str]] = []
    for index in range(pairs):
        arms = list(AB_ARMS) if index % 2 == 0 else list(reversed(AB_ARMS))
        for arm in arms:
            order.append((index, arm))
    return order


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """一次子进程调用的结果。"""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    @property
    def command_line(self) -> str:
        """报告里显示的命令：解释器写成 ``python``，仓库内的路径写成相对路径。

        实际执行用的是 ``sys.executable`` 与绝对路径；显示版只为了让人能直接
        复制到命令行里跑（报告不该带某台机器的绝对路径）。
        """
        parts: list[str] = ["python"]
        for token in self.command[1:]:
            parts.append(_display_path(token))
        return subprocess.list2cmdline(parts)


def _display_path(token: str) -> str:
    """把一个绝对路径 token 换成相对仓库根的形式（换不了就原样返回）。"""
    if not token or not Path(token).is_absolute():
        return token
    try:
        return str(Path(token).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return token


@dataclass
class StageResult:
    """一个阶段的判决。"""

    stage: str
    status: str
    seconds: float
    command: str
    artifacts: list[str] = field(default_factory=list)
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "seconds": round(float(self.seconds), 3),
            "command": self.command,
            "artifacts": list(self.artifacts),
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


@dataclass
class Context:
    """阶段之间共享的状态。"""

    out_dir: Path
    checkpoint: Path | None
    checkpoint_note: str
    timeout_s: float
    results: dict[str, StageResult] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        return self.out_dir.joinpath(*parts)

    @property
    def fixture_dir(self) -> Path:
        return self.out_dir / "fixture"


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def run_command(
    command: list[str], timeout_s: float, cwd: Path = ROOT
) -> CommandResult:
    """跑一条命令并抓下 stdout / stderr（**永不**打开设备：调用方自己保证）。"""
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):  # pragma: no cover - 平台差异兜底
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):  # pragma: no cover - 平台差异兜底
            stderr = stderr.decode("utf-8", "replace")
        return CommandResult(
            command, -1, stdout, stderr, time.time() - started, timed_out=True
        )
    return CommandResult(
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.time() - started,
    )


def _session_dir_from_output(stdout: str) -> Path | None:
    """从 MIL 的 stdout 里解析运行目录（照 ``gsnet_init_hw_ab.py`` 的做法）。"""
    matches = re.findall(r"(?:Recorder pkl/json\s*:|Preflight evidence\s*:)\s*(.+)", stdout)
    if not matches:
        return None
    candidate = Path(matches[-1].strip().strip('"'))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _newest_dir(root: Path) -> Path | None:
    """``root`` 下最新的子目录（按 mtime）。"""
    if not root.is_dir():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _rel(path: Path) -> str:
    """产物路径写成相对仓库根的形式（报告里要能直接点开）。"""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _grep(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def find_checkpoint(explicit: Path | None) -> tuple[Path | None, str]:
    """按「``--checkpoint`` → ``data/gsnet_pretrain`` → ``data/gsnet_train``」找预训练权重。

    Returns:
        ``(checkpoint, 说明)``；找不到时 ``checkpoint is None`` 且说明写明原因。
    """
    if explicit is not None:
        if explicit.is_file():
            return explicit, "--checkpoint 显式给定"
        return None, f"--checkpoint 给的路径不存在: {explicit}"
    roots = [ROOT / "data" / "gsnet_pretrain", ROOT / "data" / "gsnet_train"]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(p for p in root.rglob("*.pt") if p.is_file()))
    if not found:
        return None, (
            "没有预训练 checkpoint（需要先离线跑一次预训练："
            "没有它就没有 S2 的闭环 A/B）"
        )

    def _rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "checkpoint" in name:
            return (0, str(path))
        if "best" in name:
            return (1, str(path))
        return (2, str(path))

    found.sort(key=_rank)
    return found[0], (
        f"在 {' / '.join(_rel(r) for r in roots)} 下找到 {len(found)} 个 .pt，"
        f"取 {found[0].name}"
    )


# ---------------------------------------------------------------------------
# 阶段 1：仿真的 model-in-loop
# ---------------------------------------------------------------------------


def stage_sim_mil(ctx: Context) -> StageResult:
    """跑一次 ``--cam-type sim`` 的 MIL，并校验 4 个产物。

    **同时记录一条关键事实**：仿真路径**不写** ``probes/*.npz``、也**不写**
    ``preflight/geometry.json`` —— 这就是阶段 2 存在的理由。
    """
    root = ctx.out_dir / "sim_mil"
    command = [
        sys.executable,
        str(MAIN),
        "--debug",
        "--dir",
        str(root),
        "slm-model-in-loop",
        "--cam-type",
        "sim",
        "--n-rounds",
        "2",
        "--probe-count",
        "4",
        "--holdout-probe-count",
        "2",
        "--step-a-iterations",
        "40",
        "--step-b-iterations",
        "50",
        "--region",
        str(REGION),
        "--far-field-padding",
        str(MIL_FAR_FIELD_PADDING),
        "--cam-size",
        str(CAM_SIZE),
        "--target-side",
        str(TARGET_SIDE),
        "--seed",
        "19",
    ]
    result = run_command(command, ctx.timeout_s)
    artifacts: list[str] = []
    notes: list[str] = []

    if result.timed_out:
        return StageResult(
            "sim_mil",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"仿真 MIL 超时（>{ctx.timeout_s:g} s）",
            notes + ["超时 ⇒ 该阶段判 FAILED，依赖它的阶段 BLOCKED"],
        )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-800:]
        return StageResult(
            "sim_mil",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"仿真 MIL 退出码 {result.returncode}",
            notes + [f"stderr 尾部: {tail}"],
        )

    run_dir = _session_dir_from_output(result.stdout)
    if run_dir is None or not run_dir.exists():
        fallback = _newest_dir(root / "debug" / "slm_model_in_loop")
        if fallback is None:
            fallback = _newest_dir(ROOT / "data" / "debug" / "slm_model_in_loop")
        run_dir = fallback
    if run_dir is None or not run_dir.exists():
        return StageResult(
            "sim_mil",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "仿真 MIL 跑完了，但找不到运行目录",
            notes,
        )

    artifacts.append(_rel(run_dir))
    missing: list[str] = []
    for pattern in ("run_config.json", "slm_model_in_loop_*.csv"):
        if not list(run_dir.glob(pattern)):
            missing.append(pattern)
    for name in ("best_phase.npy", "source_amplitude.npy"):
        if not (run_dir / name).is_file():
            missing.append(name)
    if missing:
        return StageResult(
            "sim_mil",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"运行目录缺产物: {missing}",
            notes,
        )

    for name in ("best_phase.npy", "source_amplitude.npy", "run_config.json"):
        artifacts.append(_rel(run_dir / name))
    csvs = sorted(run_dir.glob("slm_model_in_loop_*.csv"))
    artifacts.append(_rel(csvs[-1]))

    probes_dir = run_dir / "probes"
    geometry = run_dir / "preflight" / "geometry.json"
    sim_writes_probes = probes_dir.is_dir()
    sim_writes_geometry = geometry.is_file()
    notes.append(
        "仿真路径**不写** `probes/*.npz`、也**不写** `preflight/geometry.json` "
        "（探针记录只在硬件路径上落盘）"
        f" —— 本次实测: probes/ 存在={sim_writes_probes}, "
        f"preflight/geometry.json 存在={sim_writes_geometry}"
    )
    notes.append(
        "⇒ 阶段 3（拟合孪生）**不能**直接吃仿真运行目录；"
        "阶段 2 的 fixture 就是为此存在"
    )
    ctx.scratch["sim_run_dir"] = run_dir
    ctx.scratch["best_phase"] = run_dir / "best_phase.npy"

    verdict = (
        "MIL 仿真链路验证通过，4 个产物齐全；"
        "但仿真不落探针/几何 ⇒ 孪生拟合必须吃阶段 2 的 fixture"
    )
    return StageResult(
        "sim_mil", STATUS_OK, result.seconds, result.command_line, artifacts, verdict, notes
    )


# ---------------------------------------------------------------------------
# 阶段 2：造一份形状与硬件一致的仿真探针数据
# ---------------------------------------------------------------------------


def _structured_probe_phase(
    rng: np.random.Generator, region: int, spread: float
) -> np.ndarray:
    """低阶 Zernike 组合出的**结构化随机相位**，std 归一到 ``spread``（rad）。

    真实探针由 ``UncertaintyProbeSelector`` 在 Zernike 基上生成；这里用仓库的
    canonical 生成器（``utils/wavefront/zernike_utils``）复现同一族形状。
    """
    from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

    coefficients = {noll: float(rng.normal(0.0, 1.0)) for noll in range(2, 8)}
    raw = generate_zernike_phase(
        coefficients, resolution=(region, region), n_max=4, radius=region / 2.0
    )
    phase = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    std = float(np.std(phase))
    if std <= 0.0:  # pragma: no cover - 概率为 0
        return phase
    return phase * (spread / std)


def _camera_frame_from_model(
    model_grid: np.ndarray, scale: int, cam_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """把远场模型网格变成「相机形状的裁剪帧」。

    相机比模型**过采样** ``scale`` 倍（相机像元 / 模型采样点），所以一格模型
    对应 ``scale × scale`` 个相机像元。这里按块复制上采样，再以 0 阶为中心裁出
    ``cam_size`` 见方的窗口 —— 于是「块平均降采样」恰好能还原模型窗口
    （与 ``slm_model_in_loop._to_model_grid`` 的整数标度分支互逆）。

    Returns:
        ``(frame, (模型网格里的 0 阶行, 列))``。
    """
    win_cells = max(1, cam_size // int(scale))
    grid = np.asarray(model_grid, dtype=np.float64)
    rows, cols = grid.shape
    peak_y, peak_x = np.unravel_index(int(np.argmax(grid)), grid.shape)
    y0 = int(min(max(peak_y - win_cells // 2, 0), max(rows - win_cells, 0)))
    x0 = int(min(max(peak_x - win_cells // 2, 0), max(cols - win_cells, 0)))
    window = grid[y0 : y0 + win_cells, x0 : x0 + win_cells]
    frame = np.repeat(np.repeat(window, int(scale), axis=0), int(scale), axis=1)
    return frame, (peak_y, peak_x)


def _build_fixture(ctx: Context) -> dict[str, Any]:
    """写出 fixture 目录，返回一份自述（供报告引用）。"""
    from ao_shaping.drivers.sim.bench_twin import (
        BenchTwin,
        BenchTwinConfig,
        TwinErrorModel,
    )
    from ao_shaping.utils.image.beam_metrics import (
        compute_square_metrics,
        prepare_frame,
        zero_order_center,
    )

    fixture = ctx.fixture_dir
    if fixture.exists():
        shutil.rmtree(fixture)
    probes_dir = fixture / "probes"
    preflight_dir = fixture / "preflight"
    probes_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)

    # --- 真值孪生：面板静态相差 + 装调误差都随机化，让拟合有东西可拟合 -----
    rng = np.random.default_rng(20261022)
    truth_errors = TwinErrorModel(
        lut_gamma=1.0,
        lut_phase_max_rad=2.0 * math.pi,
        gray_levels=GRAY_LEVELS,
        slm_zernike={5: float(rng.uniform(-0.12, 0.12)),
                     6: float(rng.uniform(-0.12, 0.12)),
                     7: float(rng.uniform(-0.12, 0.12))},
        slm_zernike_n_max=3,
        slm_roughness_sigma_rad=0.01,
        slm_roughness_seed=7,
        defocus_m=float(rng.uniform(-1.5e-4, 1.5e-4)),
        tilt_x_rad=float(rng.uniform(-4.0e-5, 4.0e-5)),
        tilt_y_rad=float(rng.uniform(-4.0e-5, 4.0e-5)),
        lens_zernike={5: float(rng.uniform(-0.08, 0.08)),
                      6: float(rng.uniform(-0.08, 0.08))},
        lens_zernike_n_max=2,
    )
    config = BenchTwinConfig.from_measured_scale(
        CAMERA_PX_PER_MODEL_PX,
        region=REGION,
        region_span_px=900.0,
        far_field_padding=FIXTURE_PADDING,
        beam_w0_px=250.0,
        crop_px=MODEL_SIDE,
        peak_adu=100.0,
        noise_adu=0.0,
        seed=20261022,
        errors=truth_errors,
    )
    twin = BenchTwin(config)

    manifest: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    def _record(
        filename: str,
        *,
        round_index: int,
        split: str,
        index: int,
        phase: np.ndarray,
        family: str | None,
    ) -> dict[str, Any]:
        model_grid = twin.far_field(phase)
        frame_raw, _ = _camera_frame_from_model(
            model_grid, int(CAMERA_PX_PER_MODEL_PX), CAM_SIZE
        )
        frame = prepare_frame(frame_raw)
        np.savez_compressed(
            probes_dir / filename,
            phase=np.asarray(phase, dtype=np.float32),
            model_grid=np.asarray(model_grid, dtype=np.float32),
            frame=np.asarray(frame, dtype=np.float32),
        )
        entry = {
            "file": filename,
            "round": round_index,
            "split": split,
            "family": family,
            "index": index,
            "exposure_ms": 0.02,
            "timestamp_utc": timestamp,
            "correlation": _frame_model_correlation(frame, model_grid),
            "frame_reference_exposure_ms": 0.02,
            "raw_frame_note": "simulated fixture (no device) — not a measurement",
        }
        manifest.append(entry)
        return {"frame": frame, "model_grid": model_grid, "phase": phase}

    # 平场（round 0, split=flat）：中心定位的依据。
    flat = _record(
        "round_00_flat_00.npz",
        round_index=0,
        split="flat",
        index=0,
        phase=np.zeros((REGION, REGION), dtype=np.float64),
        family=None,
    )
    # 训练探针（round 1, split=train）。
    for index in range(N_TRAIN_PROBES):
        _record(
            f"round_01_train_{index:02d}.npz",
            round_index=1,
            split="train",
            index=index,
            phase=_structured_probe_phase(rng, REGION, PROBE_SPREAD),
            family="probe",
        )
    # 验证集探针（round 1, split=holdout）。
    for index in range(N_HOLDOUT_PROBES):
        _record(
            f"round_01_holdout_{index:02d}.npz",
            round_index=1,
            split="holdout",
            index=index,
            phase=_structured_probe_phase(rng, REGION, PROBE_SPREAD),
            family="probe",
        )
    # 整形帧（round 1, split=shaped）：双口径 CV 的统计对象。
    shaped_phase = _structured_probe_phase(rng, REGION, 1.2)
    shaped = _record(
        "round_01_shaped_00.npz",
        round_index=1,
        split="shaped",
        index=0,
        phase=shaped_phase,
        family="shaped",
    )

    (probes_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 几何预检文件：far_field_size 必须等于 model_side -------------------
    geometry = {
        "camera_px_per_model_px": CAMERA_PX_PER_MODEL_PX,
        "region": REGION,
        "far_field_size": MODEL_SIDE,
        "panel_disc_radius": 450,
        "beam_waist_panel_px": 450.0,
        "spot_fwhm_camera_px": None,
        "spot_fwhm_model_px": None,
        "correlation": 1.0,
        "method": "gsnet-sim-dryrun-fixture",
        "calibration_notes": (
            "仿真 fixture 的几何是**构造出来**的，不是实测；"
            "camera_px_per_model_px = far_field_padding（仿真实验系统口径）。"
        ),
    }
    (preflight_dir / "geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 瞳面照明振幅（网络 source 的输入） ---------------------------------
    amplitude = np.asarray(twin.pupil_amplitude(), dtype=np.float32)
    np.save(fixture / "source_amplitude.npy", amplitude)

    # --- run_config.json：dual_aperture_cv 从这里读 target_side -------------
    run_config = {
        "cam_type": "sim",
        "region": REGION,
        "far_field_padding": FIXTURE_PADDING,
        "cam_size": CAM_SIZE,
        "target_side": TARGET_SIDE,
        "panel_span_px": 900.0,
        "n_rounds": 1,
        "probe_count": N_TRAIN_PROBES,
        "holdout_probe_count": N_HOLDOUT_PROBES,
        "seed": 20261022,
        "source": "gsnet_sim_dryrun fixture — SIMULATED, not measured on the bench",
    }
    (fixture / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- 历史 CSV：双口径 CV 的自检要拿它比对口径 B -------------------------
    center = zero_order_center(flat["frame"], refine=False)
    metrics = compute_square_metrics(shaped["frame"], TARGET_SIDE, center)
    history_name = f"slm_model_in_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with (fixture / history_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "round",
                "accepted",
                "reason",
                "uniformity_cv",
                "ee_after",
                "cv_after",
            ]
        )
        writer.writerow(
            [
                "calibration",
                0,
                "True",
                "baseline",
                f"{metrics['uniformity_cv']:.10g}",
                f"{metrics['encircled_energy']:.10g}",
                f"{metrics['uniformity_cv']:.10g}",
            ]
        )
        writer.writerow(
            [
                "round",
                1,
                "True",
                "accepted",
                f"{metrics['uniformity_cv']:.10g}",
                f"{metrics['encircled_energy']:.10g}",
                f"{metrics['uniformity_cv']:.10g}",
            ]
        )

    return {
        "manifest_entries": len(manifest),
        "center": center,
        "shaped_cv": float(metrics["uniformity_cv"]),
        "history": history_name,
        "truth_errors": truth_errors,
    }


def _frame_model_correlation(frame: np.ndarray, model_grid: np.ndarray) -> float:
    """帧与模型网格的 Pearson 相关（块平均降采样后比对）—— 仿真里恒为 1。"""
    scale = int(CAMERA_PX_PER_MODEL_PX)
    rows, cols = model_grid.shape
    win_rows = min(rows, frame.shape[0] // scale)
    win_cols = min(cols, frame.shape[1] // scale)
    if win_rows < 2 or win_cols < 2:
        return 1.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(model_grid)), model_grid.shape)
    y0 = int(min(max(peak_y - win_rows // 2, 0), max(rows - win_rows, 0)))
    x0 = int(min(max(peak_x - win_cols // 2, 0), max(cols - win_cols, 0)))
    window = np.asarray(model_grid, dtype=np.float64)[
        y0 : y0 + win_rows, x0 : x0 + win_cols
    ].ravel()
    block = frame[: win_rows * scale, : win_cols * scale]
    block = block.reshape(win_rows, scale, win_cols, scale).mean(axis=(1, 3)).ravel()
    if window.std() <= 0 or block.std() <= 0:
        return 1.0
    return float(np.corrcoef(window, block)[0, 1])


def stage_probe_fixture(ctx: Context) -> StageResult:
    """造一份「形状与硬件完全一致」的**仿真**探针数据。"""
    started = time.time()
    try:
        info = _build_fixture(ctx)
    except Exception as exc:  # noqa: BLE001 - 失败是判决，不是崩溃
        return StageResult(
            "probe_fixture",
            STATUS_FAILED,
            time.time() - started,
            "(in-process)",
            [],
            f"造 fixture 失败: {type(exc).__name__}: {exc}",
            [],
        )

    fixture = ctx.fixture_dir
    required = [
        fixture / "probes" / "manifest.json",
        fixture / "preflight" / "geometry.json",
        fixture / "source_amplitude.npy",
        fixture / "run_config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    npz_files = sorted((fixture / "probes").glob("*.npz"))
    if missing:
        return StageResult(
            "probe_fixture",
            STATUS_FAILED,
            time.time() - started,
            "(in-process)",
            [_rel(p) for p in npz_files],
            f"fixture 缺文件: {missing}",
            [],
        )
    train_files = sorted((fixture / "probes").glob("round_01_train_*.npz"))
    if len(train_files) < 8:
        return StageResult(
            "probe_fixture",
            STATUS_FAILED,
            time.time() - started,
            "(in-process)",
            [_rel(p) for p in npz_files],
            f"训练探针只有 {len(train_files)} 个（要求 ≥ 8）",
            [],
        )

    artifacts = [_rel(fixture)]
    artifacts += [_rel(p) for p in train_files]
    artifacts.append(_rel(fixture / "probes" / "manifest.json"))
    artifacts.append(_rel(fixture / "preflight" / "geometry.json"))
    artifacts.append(_rel(fixture / "source_amplitude.npy"))

    notes = [
        "这些是**仿真**探针（用仓库规范的 `drivers/sim/bench_twin.py` 前向生成），"
        "**不是**实验系统实测数据",
        f"几何：camera_px_per_model_px={CAMERA_PX_PER_MODEL_PX}, "
        f"region={REGION}, far_field_size={MODEL_SIDE}（== model_side ⇒ "
        "拟合孪生不会走「标度换算」分支）",
        f"探针：{N_TRAIN_PROBES} 个 train + {N_HOLDOUT_PROBES} 个 holdout + "
        "1 flat + 1 shaped；相位 std≈{:.1f} rad（运行时 probe_spread<3.0 会报错）".format(
            PROBE_SPREAD
        ),
        "面板静态相差 / 装调误差用 `TwinErrorModel` 随机化 ⇒ 拟合孪生有东西可拟合",
        "帧 = 远场模型网格按 camera_px_per_model_px 块上采样后以 0 阶为中心裁出"
        f" {CAM_SIZE}×{CAM_SIZE} 的相机窗口（已 `prepare_frame`）",
    ]
    verdict = (
        f"fixture 就绪：{info['manifest_entries']} 条探针记录，"
        f"整形帧口径 B 的 CV={info['shaped_cv']:.4f}"
    )
    ctx.scratch["fixture_info"] = info
    return StageResult(
        "probe_fixture",
        STATUS_OK,
        time.time() - started,
        "(in-process)",
        artifacts,
        verdict,
        notes,
    )


# ---------------------------------------------------------------------------
# 阶段 3：拟合孪生
# ---------------------------------------------------------------------------


def stage_twin_fit(ctx: Context) -> StageResult:
    """``fit_bench_twin.py --from-dir <fixture>``，并校验标度来源。"""
    command = [
        sys.executable,
        str(FIT_TWIN),
        "--from-dir",
        str(ctx.fixture_dir),
        "--out",
        str(ctx.out_dir / "bench_twin.json"),
        "--fit-modes",
        FIT_PANEL_MODES,
        "--fit-lens-modes",
        FIT_LENS_MODES,
        "--limit",
        str(TWIN_LIMIT),
        "--maxiter",
        str(TWIN_MAXITER),
        "--popsize",
        str(TWIN_POPSIZE),
        "--maxiter-fine",
        str(TWIN_MAXITER_FINE),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts: list[str] = []
    notes: list[str] = []

    if result.timed_out:
        return StageResult(
            "twin_fit",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"拟合孪生超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "twin_fit",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"拟合孪生退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    out_json = ctx.out_dir / "bench_twin.json"
    if not out_json.is_file():
        return StageResult(
            "twin_fit",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "拟合孪生没有写出 bench_twin.json",
            notes + [(combined[-800:])],
        )
    artifacts.append(_rel(out_json))

    scale_line = _grep(combined, r"焦面标度口径:\s*(.+)")
    used_nominal = "标称焦距口径" in combined
    if used_nominal:
        notes.append(
            "⚠️ 脚本走了**标称焦距口径**回退 —— 说明几何没被读到，"
            "拟合出的系数全部存疑"
        )
    notes.append(
        "标度来源（脚本 stdout 原文）: " + (scale_line or "（未解析到该行）")
    )
    if used_nominal:
        return StageResult(
            "twin_fit",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "几何来自**标称焦距回退**而不是 fixture 的 preflight/geometry.json",
            notes,
        )

    initial_loss = _grep(combined, r"初始损失\s*=\s*([0-9.eE+-]+)")
    final_loss = _grep(combined, r"最终联合损失\s*=\s*([0-9.eE+-]+)")
    residual = _grep(combined, r"残差 RMS 均值\s*=\s*([0-9.eE+-]+)")
    corr = _grep(combined, r"Pearson 相关均值\s*=\s*([0-9.eE+-]+)")
    ssim = _grep(combined, r"SSIM 均值\s*=\s*([0-9.eE+-]+)")
    improved = "拟合把联合损失从" in combined
    notes.append(
        "拟合质量: 初始损失={} → 最终联合损失={}, 残差 RMS={}, "
        "SSIM={}, Pearson={}".format(
            initial_loss or "?", final_loss or "?", residual or "?", ssim or "?", corr or "?"
        )
    )
    if not improved:
        notes.append(
            "⚠️ 脚本自己报告「拟合没有比初始配置更好」—— 这**不是**一次成功的反演"
        )

    # 报告拟合出的关键参数（从 JSON 读，不靠正则刨 stdout）。
    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        config = payload.get("config", payload)
        errors = config.get("errors", {})
        notes.append(
            "拟合出的参数: LUT Γ={:.4f}, 满灰度相位={:.4f} rad, "
            "相机离焦={:.3e} m, tilt=({:.3e}, {:.3e}) rad, 面板 Zernike={}, "
            "透镜 Zernike={}".format(
                float(errors.get("lut_gamma", float("nan"))),
                float(errors.get("lut_phase_max_rad", float("nan"))),
                float(errors.get("defocus_m", float("nan"))),
                float(errors.get("tilt_x_rad", float("nan"))),
                float(errors.get("tilt_y_rad", float("nan"))),
                errors.get("slm_zernike", {}),
                errors.get("lens_zernike", {}),
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        notes.append(f"（读回 JSON 失败: {exc}）")

    ctx.scratch["twin_json"] = out_json
    verdict = "孪生拟合验证通过，几何来自 fixture 的 preflight/geometry.json（实测标度口径）"
    return StageResult(
        "twin_fit", STATUS_OK, result.seconds, result.command_line, artifacts, verdict, notes
    )


# ---------------------------------------------------------------------------
# 阶段 4：灰度→振幅耦合的离线自洽检查
# ---------------------------------------------------------------------------


def gray_csv_fieldnames() -> tuple[str, ...]:
    """灰度响应 CSV 的列名 —— 从 ``validate_flat_phase_gray.py`` 现读。

    那份 CSV 的**写方**是 ``scripts/slm/validate_flat_phase_gray.py``，读方是
    ``scripts/slm/fit_slm_amplitude_coupling.py``；复制一份列名迟早会漂移
    （漂移的后果是耦合拟合悄悄退回全帧能量列）。所以这里按文件路径加载写方的
    模块，直接取它的 ``CSV_FIELDNAMES``。
    """
    if not GRAY_CSV_SCRIPT.is_file():  # pragma: no cover - 上游脚本被删
        return GRAY_CSV_FIELDNAMES_FALLBACK
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_validate_flat_phase_gray", GRAY_CSV_SCRIPT
    )
    if spec is None or spec.loader is None:  # pragma: no cover - 兜底
        return GRAY_CSV_FIELDNAMES_FALLBACK
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - 上游脚本 import 失败就用兜底
        return GRAY_CSV_FIELDNAMES_FALLBACK
    names = getattr(module, "CSV_FIELDNAMES", None)
    if not names:
        return GRAY_CSV_FIELDNAMES_FALLBACK
    return tuple(str(name) for name in names)


def _write_synthetic_gray_csv(path: Path) -> dict[str, float]:
    """写一份**合成**的灰度响应 CSV。

    ``bucket_sum`` 按 ``I(g) = 1 + c·sin(2π·g/993 + ψ)`` 生成（993 = 2π 灰度数）。
    """
    offset = 1000.0
    stamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for gray in range(0, GRAY_LEVELS, 25):
        intensity = 1.0 + SYNTH_C * math.sin(
            2.0 * math.pi * gray / GRAY_LEVELS + SYNTH_PSI
        )
        rows.append(
            {
                "gray_value": gray,
                "max_brightness": f"{0.92 * offset * intensity:.6f}",
                "min_brightness": f"{0.08 * offset * intensity:.6f}",
                "sum_brightness": f"{4.0 * offset * intensity:.6f}",
                "bucket_peak": f"{1.20 * offset * intensity:.6f}",
                "bucket_sum": f"{offset * intensity:.6f}",
                "zero_order_x": 200,
                "zero_order_y": 200,
                "frame_dtype": "float32",
                "frame_shape": f"{CAM_SIZE}x{CAM_SIZE}",
                "slm_max_gray": 1023,
                "wavelength_nm": 1064,
                "exposure_ms": 0.4,
                "cam_type": "sim",
                "cam_id": 0,
                "slm_serial": "SIMULATED",
                "timestamp": stamp,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gray_csv_fieldnames()))
        writer.writeheader()
        writer.writerows(rows)
    return {"c": SYNTH_C, "psi_rad": SYNTH_PSI, "levels": float(GRAY_LEVELS)}


def stage_coupling(ctx: Context) -> StageResult:
    """合成灰度 CSV → ``fit_slm_amplitude_coupling.py`` → 校验周期 / c / ψ。"""
    csv_path = ctx.out_dir / "gray_synthetic.csv"
    truth = _write_synthetic_gray_csv(csv_path)
    command = [
        sys.executable,
        str(FIT_COUPLING),
        "--csv",
        str(csv_path),
        "--out",
        str(ctx.out_dir / "coupling.json"),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts = [_rel(csv_path)]
    notes = [
        "这是**离线自洽性**检查：CSV 是合成的，拟合结果只证明「脚本 ↔ 脚本」这条"
        "链自洽，**不是**实测耦合；实测耦合仍需 S1 的灰度扫描",
        f"合成真值: c={SYNTH_C}, ψ={SYNTH_PSI} rad, 周期={GRAY_LEVELS} 灰度",
    ]

    if result.timed_out:
        return StageResult(
            "coupling",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"耦合拟合超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "coupling",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"耦合拟合退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    out_json = ctx.out_dir / "coupling.json"
    if not out_json.is_file():
        return StageResult(
            "coupling",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "耦合拟合没有写出 coupling.json",
            notes + [(combined[-800:])],
        )
    artifacts.append(_rel(out_json))

    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        fit = payload["amplitude_coupling_fit"]
        period = float(fit["period"])
        c_fit = float(fit["c"])
        psi_fit = float(fit["psi_rad"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return StageResult(
            "coupling",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"coupling.json 读不出拟合参数: {exc}",
            notes,
        )

    period_dev = abs(period - GRAY_LEVELS) / GRAY_LEVELS
    c_dev = abs(c_fit - truth["c"])
    psi_dev = abs((psi_fit - truth["psi_rad"] + math.pi) % (2.0 * math.pi) - math.pi)
    notes.append(
        "拟合: 周期={:.3f} (偏差 {:.2%}, 容差 {:.0%}), c={:.5f} (Δ={:.5f}, 容差 "
        "{:.3f}), ψ={:.5f} rad (Δ={:.5f}, 容差 {:.3f})".format(
            period,
            period_dev,
            COUPLING_PERIOD_TOL,
            c_fit,
            c_dev,
            COUPLING_C_TOL,
            psi_fit,
            psi_dev,
            COUPLING_PSI_TOL,
        )
    )
    problems: list[str] = []
    if period_dev > COUPLING_PERIOD_TOL:
        problems.append(
            f"拟合周期 {period:.1f} 偏离 2π 灰度数 {GRAY_LEVELS} 超过 "
            f"{COUPLING_PERIOD_TOL:.0%}"
        )
    if c_dev > COUPLING_C_TOL:
        problems.append(f"c 偏差 {c_dev:.5f} > 容差 {COUPLING_C_TOL}")
    if psi_dev > COUPLING_PSI_TOL:
        problems.append(f"ψ 偏差 {psi_dev:.5f} rad > 容差 {COUPLING_PSI_TOL}")
    if problems:
        return StageResult(
            "coupling",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "；".join(problems),
            notes,
        )

    verdict = (
        f"耦合拟合自洽：周期 {period:.1f}≈{GRAY_LEVELS}, "
        f"c={c_fit:.4f}（真值 {SYNTH_C}）, ψ={psi_fit:.3f} rad（真值 {SYNTH_PSI}）"
    )
    return StageResult(
        "coupling", STATUS_OK, result.seconds, result.command_line, artifacts, verdict, notes
    )


# ---------------------------------------------------------------------------
# 阶段 5：导出相位
# ---------------------------------------------------------------------------


def stage_export(ctx: Context) -> StageResult:
    """``export_gsnet_phase.py --forward calibrated``（需要预训练 checkpoint）。"""
    if ctx.checkpoint is None:
        return StageResult(
            "export",
            STATUS_BLOCKED,
            0.0,
            "(not run)",
            [],
            f"没有预训练 checkpoint：{ctx.checkpoint_note}",
            [
                "阶段 5/6 的 BLOCKED **不是**失败：只是缺一件必须先离线做完的事",
                "补救：先在离线块跑一次预训练（--samples 2000 --epochs 24 量级）",
            ],
        )

    twin_json = ctx.scratch.get("twin_json") or (ctx.out_dir / "bench_twin.json")
    if not Path(twin_json).is_file():
        return StageResult(
            "export",
            STATUS_BLOCKED,
            0.0,
            "(not run)",
            [],
            "依赖阶段 twin_fit 没有产出 bench_twin.json",
            [],
        )

    target = ctx.out_dir / "target.npy"
    if not target.is_file():
        source_npz = sorted((ctx.fixture_dir / "probes").glob("round_01_train_*.npz"))
        if not source_npz:
            return StageResult(
                "export",
                STATUS_BLOCKED,
                0.0,
                "(not run)",
                [],
                "没有可用的占位目标远场（fixture 里找不到探针）",
                [],
            )
        with np.load(source_npz[0], allow_pickle=False) as data:
            placeholder = np.asarray(data["model_grid"], dtype=np.float32)
        np.save(target, placeholder)
        ctx.scratch["target_is_placeholder"] = True

    command = [
        sys.executable,
        str(EXPORT_PHASE),
        "--checkpoint",
        str(ctx.checkpoint),
        "--source",
        str(ctx.fixture_dir / "source_amplitude.npy"),
        "--target",
        str(target),
        "--twin-config",
        str(twin_json),
        "--forward",
        "calibrated",
        "--out",
        str(ctx.out_dir / "phase.npy"),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts = [_rel(target)]
    notes = [
        "``--target`` 用的是 fixture 里某张 ``model_grid`` 存成的**占位目标**"
        "（不是真实目标远场）",
    ]

    if result.timed_out:
        return StageResult(
            "export",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"导出相位超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "export",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"导出相位退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    phase = ctx.out_dir / "phase.npy"
    if not phase.is_file():
        return StageResult(
            "export",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "导出相位没有写出 phase.npy",
            notes + [(combined[-800:])],
        )
    artifacts.append(_rel(phase))
    ctx.scratch["phase_npy"] = phase
    return StageResult(
        "export",
        STATUS_OK,
        result.seconds,
        result.command_line,
        artifacts,
        "用拟合前向（--forward calibrated）导出相位成功",
        notes,
    )


# ---------------------------------------------------------------------------
# 阶段 6：迁移学习
# ---------------------------------------------------------------------------


def stage_transfer(ctx: Context) -> StageResult:
    """``gsnet_transfer.py --from-dir <fixture> --gate``（需要预训练 checkpoint）。"""
    if ctx.checkpoint is None:
        return StageResult(
            "transfer",
            STATUS_BLOCKED,
            0.0,
            "(not run)",
            [],
            f"没有预训练 checkpoint：{ctx.checkpoint_note}",
            [
                "阶段 5/6 的 BLOCKED **不是**失败：只是缺一件必须先离线做完的事",
            ],
        )

    twin_json = ctx.scratch.get("twin_json") or (ctx.out_dir / "bench_twin.json")
    if not Path(twin_json).is_file():
        return StageResult(
            "transfer",
            STATUS_BLOCKED,
            0.0,
            "(not run)",
            [],
            "依赖阶段 twin_fit 没有产出 bench_twin.json",
            [],
        )

    out_dir = ctx.out_dir / "transfer"
    command = [
        sys.executable,
        str(GSNET_TRANSFER),
        "--checkpoint",
        str(ctx.checkpoint),
        "--from-dir",
        str(ctx.fixture_dir),
        "--twin-config",
        str(twin_json),
        "--freeze",
        "freeze-decoder",
        "--gate",
        "--epochs",
        "3",
        "--out",
        str(out_dir),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts: list[str] = []
    notes: list[str] = []

    if result.timed_out:
        return StageResult(
            "transfer",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"迁移学习超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "transfer",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"迁移学习退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        return StageResult(
            "transfer",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "迁移学习没有写出 summary.json",
            notes + [(combined[-800:])],
        )
    artifacts.append(_rel(summary_path))

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return StageResult(
            "transfer",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"summary.json 读不出来: {exc}",
            notes,
        )

    history = summary.get("history") or []
    chance = float(summary.get("phase_loss_chance_level", math.pi**2 / 3.0))
    if history:
        phase_first = float(history[0]["phase_loss"])
        phase_last = float(history[-1]["phase_loss"])
        shaping_first = float(history[0].get("shaping_loss", float("nan")))
        shaping_last = float(history[-1].get("shaping_loss", float("nan")))
    else:  # pragma: no cover - 3 个 epoch 一定有 history
        phase_first = phase_last = shaping_first = shaping_last = float("nan")
    notes.append(
        "phase_loss 首={:.4f} 末={:.4f}（随机猜测水平 π²/3={:.4f}，"
        "末值{}该水平）".format(
            phase_first,
            phase_last,
            chance,
            "低于" if phase_last < chance else "**尚未低于**",
        )
    )
    notes.append(
        "shaping_loss 首={:.4f} 末={:.4f}".format(shaping_first, shaping_last)
    )
    notes.append(
        "--gate 判决={}（部署权重={}）".format(
            summary.get("gate"), summary.get("deployed_weights")
        )
    )
    notes.append(
        "可训练参数={} / 总参数={}".format(
            summary.get("trainable_params"), summary.get("total_params")
        )
    )
    return StageResult(
        "transfer",
        STATUS_OK,
        result.seconds,
        result.command_line,
        artifacts,
        "迁移学习验证通过（--gate 判决见 notes）",
        notes,
    )


# ---------------------------------------------------------------------------
# 阶段 7：闭环 A/B 的计划
# ---------------------------------------------------------------------------


def stage_ab_plan(ctx: Context) -> StageResult:
    """``gsnet_init_hw_ab.py --plan``（**绝不打开设备**）。"""
    phase = ctx.scratch.get("phase_npy")
    source_note = "阶段 5 导出的 phase.npy"
    if phase is None or not Path(phase).is_file():
        best = ctx.scratch.get("best_phase")
        if best is not None and Path(best).is_file():
            phase = Path(best)
            source_note = "**用仿真 best_phase 占位**（阶段 5 没有产出 phase.npy）"
        else:
            return StageResult(
                "ab_plan",
                STATUS_BLOCKED,
                0.0,
                "(not run)",
                [],
                "没有可用的初值相位文件（阶段 5 的 phase.npy 与阶段 1 的 "
                "best_phase.npy 都没有）",
                [],
            )

    out_root = ctx.out_dir / "ab_plan"
    command = [
        sys.executable,
        str(HW_AB),
        "--plan",
        "--gsnet-phase",
        str(phase),
        "--pairs",
        "3",
        "--out-root",
        str(out_root),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts = [_rel(phase)]
    notes = [f"初值相位来源: {source_note}"]

    if result.timed_out:
        return StageResult(
            "ab_plan",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"A/B 计划超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "ab_plan",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"A/B 计划退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    plan_json = out_root / "plan.json"
    if plan_json.is_file():
        artifacts.append(_rel(plan_json))

    if "no device is opened" not in combined:
        notes.append("⚠️ 计划文本里没有「no device is opened」的声明")

    parsed = [
        (int(pair), arm)
        for pair, arm in re.findall(r"\[pair(\d+)_(gsnet|physical)\]", result.stdout)
    ]
    expected = ab_expected_order(3)
    if not parsed:
        return StageResult(
            "ab_plan",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "计划渲染成功但解析不出臂顺序",
            notes,
        )
    if parsed != expected:
        return StageResult(
            "ab_plan",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"臂顺序不是交错且按奇偶翻转: 实际 {parsed}，期望 {expected}",
            notes,
        )
    notes.append(
        "臂顺序校验通过: " + " → ".join(f"p{p}/{a}" for p, a in parsed)
        + "（第 i 对按 i 的奇偶翻转）"
    )
    return StageResult(
        "ab_plan",
        STATUS_OK,
        result.seconds,
        result.command_line,
        artifacts,
        "A/B 计划渲染成功，交错顺序与奇偶翻转校验通过（--plan 不打开设备）",
        notes,
    )


# ---------------------------------------------------------------------------
# 阶段 8：双口径 CV
# ---------------------------------------------------------------------------


def stage_dual_aperture(ctx: Context) -> StageResult:
    """``dual_aperture_cv.py --from-dir <fixture>``。"""
    script = DUAL_APERTURE
    if not script.is_file():
        return StageResult(
            "dual_aperture",
            STATUS_BLOCKED,
            0.0,
            "(not run)",
            [],
            "dual_aperture_cv.py 不存在",
            [f"期望路径: {_rel(script)}"],
        )

    csv_path = ctx.out_dir / "dual_aperture.csv"
    json_path = ctx.out_dir / "dual_aperture.json"
    command = [
        sys.executable,
        str(script),
        "--from-dir",
        str(ctx.fixture_dir),
        "--csv",
        str(csv_path),
        "--json",
        str(json_path),
    ]
    result = run_command(command, ctx.timeout_s)
    combined = result.stdout + "\n" + result.stderr
    artifacts: list[str] = []
    notes: list[str] = []

    if result.timed_out:
        return StageResult(
            "dual_aperture",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"双口径统计超时（>{ctx.timeout_s:g} s）",
            notes,
        )
    if result.returncode != 0:
        return StageResult(
            "dual_aperture",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"双口径统计退出码 {result.returncode}",
            notes + [(combined[-800:])],
        )

    if not csv_path.is_file():
        return StageResult(
            "dual_aperture",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            "双口径统计没有写出 CSV",
            notes + [(combined[-800:])],
        )
    artifacts.append(_rel(csv_path))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    apertures = sorted({row.get("aperture", "") for row in rows})
    if apertures != ["plateau_inner", "target_full"]:
        return StageResult(
            "dual_aperture",
            STATUS_FAILED,
            result.seconds,
            result.command_line,
            artifacts,
            f"CSV 里的口径不是两个（实际 {apertures}）",
            notes,
        )
    notes.append(f"CSV 口径: {apertures}（每个统计对象各一行）")

    if json_path.is_file():
        artifacts.append(_rel(json_path))
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            check = payload.get("self_check", {})
            notes.append(
                "自检: performed={} reason={} 最大相对偏差={}".format(
                    check.get("performed"),
                    check.get("reason"),
                    check.get("max_rel_delta"),
                )
            )
            if not check.get("performed"):
                return StageResult(
                    "dual_aperture",
                    STATUS_FAILED,
                    result.seconds,
                    result.command_line,
                    artifacts,
                    f"口径 B 的自检**未做**（reason={check.get('reason')}）",
                    notes,
                )
        except json.JSONDecodeError as exc:
            return StageResult(
                "dual_aperture",
                STATUS_FAILED,
                result.seconds,
                result.command_line,
                artifacts,
                f"dual_aperture.json 读不出来: {exc}",
                notes,
            )

    return StageResult(
        "dual_aperture",
        STATUS_OK,
        result.seconds,
        result.command_line,
        artifacts,
        "双口径 CV 验证通过：两个口径各一行，自检已做",
        notes,
    )


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------

STAGE_FUNCS = {
    "sim_mil": stage_sim_mil,
    "probe_fixture": stage_probe_fixture,
    "twin_fit": stage_twin_fit,
    "coupling": stage_coupling,
    "export": stage_export,
    "transfer": stage_transfer,
    "ab_plan": stage_ab_plan,
    "dual_aperture": stage_dual_aperture,
}


def _selection_reason(
    stage: str, only: set[str], skip: set[str]
) -> str | None:
    """该阶段是否应当执行；不执行时给出原因。"""
    if only and stage not in only:
        return "未被 --only 选中"
    if stage in skip:
        return "被 --skip 排除"
    return None


def _artifact_path(ctx: Context, key: str) -> Path | None:
    """依赖产物在磁盘上的位置（``None`` = 不存在 / 不适用）。"""
    if key == "fixture":
        return ctx.fixture_dir
    if key == "twin_json":
        return ctx.out_dir / "bench_twin.json"
    return None


def _dependency_block(
    ctx: Context, stage: str, only: set[str], skip: set[str]
) -> tuple[str, list[str]] | None:
    """检查阶段的依赖；不满足时返回 ``(原因, 备注)``，满足时返回 ``None``。

    规则（绝不静默跳过）：

    * 依赖阶段**失败** ⇒ 一律不许复用磁盘上的旧产物；
    * 依赖阶段**成功** ⇒ 满足；
    * 依赖阶段**没执行**（被 ``--only`` / ``--skip`` 排除）⇒ 只有磁盘上已有
      对应产物时才满足，否则 BLOCKED。
    """
    reasons: list[str] = []
    notes: list[str] = []
    for dep in DEPENDENCIES.get(stage, ()):
        dep_result = ctx.results.get(dep)
        if dep_result is not None and dep_result.status == STATUS_OK:
            continue
        if dep_result is not None and dep_result.status == STATUS_FAILED:
            reasons.append(f"{dep}=FAILED")
            notes.append(f"依赖阶段 {dep} 失败 ⇒ 不复用磁盘上的旧产物")
            continue
        # 依赖阶段没跑：看磁盘上有没有现成的产物。
        reusable = []
        for key in DEPENDENCY_ARTIFACTS.get(dep, ()):
            path = _artifact_path(ctx, key)
            if path is not None and path.exists():
                reusable.append(key)
        if reusable:
            notes.append(
                f"依赖阶段 {dep} 本次未执行（磁盘上已有产物 {reusable}，直接复用）"
            )
            continue
        reason = _selection_reason(dep, only, skip) or "未执行"
        reasons.append(f"{dep}（{reason}，且磁盘上没有它的产物）")
    if not reasons:
        return None
    if stage in CHECKPOINT_STAGES and ctx.checkpoint is None:
        notes.append(
            f"另外：{stage} 还需要预训练 checkpoint —— {ctx.checkpoint_note}"
        )
    return "依赖阶段未成功：" + ", ".join(reasons), notes


def run_stages(
    ctx: Context, only: set[str], skip: set[str]
) -> list[StageResult]:
    """按固定顺序跑全部阶段，每个阶段记一条判决。"""
    results: list[StageResult] = []
    for stage in STAGES:
        reason = _selection_reason(stage, only, skip)
        if reason is not None:
            notes = ["未跑的阶段一律记 BLOCKED（绝不静默跳过）"]
            if stage in CHECKPOINT_STAGES and ctx.checkpoint is None:
                notes.append(
                    f"另外：{stage} 还需要预训练 checkpoint —— {ctx.checkpoint_note}"
                )
            result = StageResult(
                stage,
                STATUS_BLOCKED,
                0.0,
                "(not run)",
                [],
                f"未执行：{reason}",
                notes,
            )
            results.append(result)
            ctx.results[stage] = result
            continue

        blocked = _dependency_block(ctx, stage, only, skip)
        if blocked is not None:
            reason, notes = blocked
            result = StageResult(
                stage,
                STATUS_BLOCKED,
                0.0,
                "(not run)",
                [],
                reason,
                notes + ["上游没就绪 ⇒ 本阶段不执行（绝不静默跳过）"],
            )
            results.append(result)
            ctx.results[stage] = result
            continue

        result = STAGE_FUNCS[stage](ctx)
        results.append(result)
        ctx.results[stage] = result
    return results


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

MUST_GO_HARDWARE: tuple[str, ...] = (
    "**S1-0 环境自检**（0 帧）：确认没有残留 python 进程占着 SLM 的 USB；"
    "本会话内设 `AO_DAHENG_MAX_EXPOSURE_MS=0.4`（不要留空）。",
    "**S1-1 光功率门**（≈ 5–9 帧，< 30 s）：`diagnose_light_level.py --cam-type daheng "
    "--cam-id 0 --slm-number 1 --slm-wavelength 1064 "
    "--exposures 0.02,0.05,0.1,0.2,0.4`。不过就必须停 —— 没有光就没有任何可整形的东西。",
    "**S1-2 灰度扫描**（40 点 × 4 帧 ≈ 160 帧，≈ 40 s）：`validate_flat_phase_gray.py "
    "--scan --gray-step 25 --exposure-ms <S1-1 定下的档位> --cam-type daheng --metric "
    "bucket --csv data/slm_gray_response_daheng.csv`。40 点 = range(0, 993, 25)；"
    "`--gray-step 50` 会漏掉周期峰谷，不要图省事放大。**实测的 c / ψ 只能从这里来。**",
    "**S1-3 一次 MIL 采集**（42 次采集 ≈ 290 帧，≈ 2–3 min）：`main.py --debug "
    "slm-model-in-loop --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 "
    "--exposure-time-ms <现定> --region 128 --far-field-padding 8 --cam-size 300 "
    "--target-side 100 --n-rounds 3 --probe-count 8 --holdout-probe-count 2 "
    "--step-a-iterations 40 --step-b-iterations 200 --dir data`。一次采集同时产出"
    "「几何预检 / 探针对 / 源面强度 / 平场与整形帧」四件下游要的东西。",
    "**离线块（0 帧 / 0 设备时间）**：`fit_slm_amplitude_coupling.py --csv <S1-2 的 CSV>` → "
    "`fit_bench_twin.py --from-dir <S1-3 的运行目录> --coupling-config <耦合 JSON>` → "
    "`export_gsnet_phase.py` → `gsnet_transfer.py --from-dir`（**不要**用 `--hardware`，"
    "它会再采一遍数据）。预演器覆盖的是这条链的**命令与产物路径**，数据仍来自 S1。",
    "**预训练 checkpoint**：本工作树 `data/` 下没有（已核实无 `data/gsnet_pretrain`）。"
    "会话前必须先在离线块跑一次预训练（`--samples 2000 --epochs 24` 量级）。"
    "**没有 checkpoint 就没有 S2** —— 这是前置阻塞项，本次预演的 export / transfer "
    "两个阶段因此判 BLOCKED。",
    "**S2-0 先出计划**（0 帧，不打开设备）：`gsnet_init_hw_ab.py --plan --gsnet-phase "
    "<phase.npy> --pairs 3`，确认交错顺序与臂参数（实验系统参数：target-side 100 / "
    "cam-size 300 / region 128，与预演用的 288 / 400 / 64 不同）。",
    "**S2-1 正式跑**：同一命令去掉 `--plan`，**`--timeout-s` 必须从 5400 压到 600**"
    "（一次挂起 90 min × 6 个 stage = 最坏 9 小时）。",
    "**门 1 阈值核对**：确认 `--init-from-gsnet-trust-region-c-l2` 默认 2.5 rad 生效。"
    "闭环自身最优相位 rms 1.678 rad > 0.5 rad；阈值不对 ⇒ 6 次 A/B 全被判 "
    "`trust_region_rejected`，8 分钟零信息。",
    "**止损表**（出现即停，不要在设备上试错）：光功率门不过 ⇒ 停 S1 查光路；"
    "几何预检 `correlation < 0.5` ⇒ 停 S1（提 `--probe-spread` 或加 "
    "`--n-calibration-probes` 后**重开一次**）；帧里找不到目标方框 ⇒ 用 `argmax` 锚定，"
    "别用画幅中心；单次采集 > 4 s ⇒ 面板黏滞，物理断电复位后再来。",
)

SIM_NOT_BENCH_NOTES: tuple[str, ...] = (
    "**仿真路径不落探针、不落几何**：仿真运行目录里没有 `probes/*.npz`、也没有 "
    "`preflight/geometry.json`（探针记录只在硬件路径上落盘）。所以拟合孪生**不能**"
    "直接吃仿真运行目录 —— 阶段 2 的 fixture 就是为此存在的替代物。",
    "**阶段 2 的探针是仿真的**：它用仓库规范的 `drivers/sim/bench_twin.py` 前向生成，"
    "形状与硬件一致，但**不是**实验系统实测数据。它验证的是「命令链 + 产物格式 + "
    "门控判据」，不验证任何物理结论。",
    "**阶段 4 的耦合是自洽不是实测**：CSV 是合成的（`I(g)=1+c·sin(2πg/993+ψ)`），"
    "拟合结果只证明「`fit_slm_amplitude_coupling.py` ↔ 数据」这条链自洽。"
    "实测的 c / ψ 必须由 S1-2 的灰度扫描给出。",
    "**焦面标度口径不同**：fixture 的 `camera_px_per_model_px` 是**构造出来**的"
    "（= `far_field_padding`）；实验系统 2026-10-19 实测是 **2.614022**，"
    "比标称焦距口径（1.027778）差 **2.54 倍**。fixture 里的孪生系数**不能**与硬件的孪生"
    "系数互相比较。",
    "**fixture 的 `model_grid` 是补零后的远场网格**（`model_side = region × padding`），"
    "与硬件运行目录里 `region × region` 的 `model_grid` 不是同一个口径 —— 这正是"
    "fixture 刻意让 `far_field_size == model_side` 的原因（否则 `fit_bench_twin.py` "
    "会走「标度换算」分支）。",
    "**fixture 的补零因子是 4，不是 8**（阶段 1 的仿真 MIL 用 8，实验系统用 "
    "`region 128 × padding 8`）：反演的目标函数每次评估要跑一遍 `region × padding` "
    "的复数 FFT 加一次 numpy SSIM，而 L-BFGS-B 的有限差分梯度每次迭代要 "
    "2×参数个数次评估 —— 网格边长翻倍 ⇒ 单次评估慢 4 倍。离线预演必须在几分钟内"
    "跑完，所以 fixture 取 256×256 的模型网格。**这只是算力取舍，不影响任何命令"
    "与产物的形状契约。**",
    "**跨会话数字不可比**：8 分钟内功率可漂 21 %。S1 与 S2 的任何数值都不能互相比较；"
    "单轮 best 分数也不得作为可复现增益声明（本实验系统分数呈双峰）。",
    "**仿真天然验证不了的**：探针落盘格式与 manifest 字段、几何相关性求解与 "
    "`correlation < 0.5` 的止损门、帧率/曝光/读噪声/16-bit 量化本底、面板黏滞"
    "（单次采集 > 4 s）、0 阶是否真的落在画幅中心、目标方框是否锚在光斑上、"
    "光功率门的真实阈值 —— 这些**只有上实验系统才知道**。",
)


def render_markdown(ctx: Context, results: list[StageResult]) -> str:
    """渲染 ``dryrun.md``。"""
    lines: list[str] = []
    lines.append("# issue #165 全链离线预演报告")
    lines.append("")
    lines.append(f"- 输出目录：`{_rel(ctx.out_dir)}`")
    lines.append(
        f"- 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append(
        f"- checkpoint：`{ctx.checkpoint}`（{ctx.checkpoint_note}）"
        if ctx.checkpoint
        else f"- checkpoint：**无** —— {ctx.checkpoint_note}"
    )
    lines.append(f"- 每阶段超时上限：{ctx.timeout_s:.0f} s")
    lines.append("")
    lines.append(
        "> 本报告由 `scripts/slm/gsnet_sim_dryrun.py` 生成；**全程不打开任何设备**。"
    )
    lines.append("")

    lines.append("## 1. 阶段表")
    lines.append("")
    lines.append(
        markdown_table(
            ["阶段", "说明", "状态", "秒数", "判据"],
            [
                [
                    f"`{result.stage}`",
                    STAGE_TITLES.get(result.stage, ""),
                    result.status,
                    f"{result.seconds:.1f}",
                    result.verdict,
                ]
                for result in results
            ],
        )
    )
    lines.append("")

    lines.append("## 2. 逐阶段细节")
    lines.append("")
    for result in results:
        lines.append(f"### `{result.stage}` — {result.status}")
        lines.append("")
        lines.append(f"- 判据：{result.verdict}")
        lines.append(f"- 秒数：{result.seconds:.1f}")
        if result.command and result.command != "(not run)":
            lines.append(f"- 命令：`{result.command}`")
        if result.artifacts:
            lines.append("- 产物：")
            for artifact in result.artifacts:
                lines.append(f"  - `{artifact}`")
        if result.notes:
            lines.append("- 备注：")
            for note in result.notes:
                lines.append(f"  - {note}")
        lines.append("")

    lines.append("## 3. 仍然必须在设备上执行的清单")
    lines.append("")
    lines.append(
        "下面每一条**都不是**本次预演能覆盖的（照 "
        "`docs/slm/gsnet_hw_plan_min_time.md` 的 S1 / S2 阶段写）："
    )
    lines.append("")
    for item in MUST_GO_HARDWARE:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 4. 仿真 ≠ 实验系统")
    lines.append("")
    for note in SIM_NOT_BENCH_NOTES:
        lines.append(f"- {note}")
    lines.append("")

    counts = {
        STATUS_OK: sum(1 for r in results if r.status == STATUS_OK),
        STATUS_BLOCKED: sum(1 for r in results if r.status == STATUS_BLOCKED),
        STATUS_FAILED: sum(1 for r in results if r.status == STATUS_FAILED),
    }
    lines.append("## 5. 汇总")
    lines.append("")
    lines.append(
        markdown_table(
            ["OK", "BLOCKED", "FAILED", "合计"],
            [[counts[STATUS_OK], counts[STATUS_BLOCKED], counts[STATUS_FAILED], len(results)]],
        )
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "issue #165 硬件链路的全链离线预演器: 在不打开任何设备的前提下, "
            "把 MIL 仿真 / 探针 fixture / 孪生拟合 / 耦合自检 / 导出相位 / "
            "迁移学习 / A/B 计划 / 双口径 CV 这 8 个阶段逐个走一遍, "
            "给出 OK / BLOCKED / FAILED 判决与「仍然必须在设备上执行」的清单."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录 (默认 data/debug/gsnet_sim_dryrun/<时间戳>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="预训练 checkpoint; 不给则在 data/gsnet_pretrain 与 data/gsnet_train 下找",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="STAGE",
        help=f"只跑这些阶段 (可重复); 合法值: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="STAGE",
        help=f"跳过这些阶段 (可重复); 合法值: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="每个阶段的超时上限 (秒, 默认 1800)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="有 BLOCKED 也算失败 (退出码 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    unknown_only = [s for s in args.only if s not in STAGES]
    unknown_skip = [s for s in args.skip if s not in STAGES]
    if unknown_only or unknown_skip:
        parser.error(
            f"未知的阶段名: only={unknown_only} skip={unknown_skip}; "
            f"合法值: {', '.join(STAGES)}"
        )
    if args.timeout_s <= 0:
        parser.error("--timeout-s 必须为正")

    out_dir = args.out or (
        ROOT / "data" / "debug" / "gsnet_sim_dryrun" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, checkpoint_note = find_checkpoint(args.checkpoint)
    ctx = Context(
        out_dir=out_dir,
        checkpoint=checkpoint,
        checkpoint_note=checkpoint_note,
        timeout_s=float(args.timeout_s),
    )

    results = run_stages(ctx, set(args.only), set(args.skip))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_note": checkpoint_note,
        "timeout_s": float(args.timeout_s),
        "stages": [result.to_dict() for result in results],
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.status == STATUS_OK),
            "blocked": sum(1 for r in results if r.status == STATUS_BLOCKED),
            "failed": sum(1 for r in results if r.status == STATUS_FAILED),
        },
    }
    (out_dir / "dryrun.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "dryrun.md").write_text(
        render_markdown(ctx, results), encoding="utf-8"
    )

    print("")
    for result in results:
        print(
            f"[{result.status:<7}] {result.stage:<14} {result.seconds:7.1f}s  "
            f"{result.verdict}"
        )
    print("")
    print(f"产物: {_rel(out_dir / 'dryrun.json')}")
    print(f"产物: {_rel(out_dir / 'dryrun.md')}")

    if payload["summary"]["failed"]:
        return 1
    if args.strict and payload["summary"]["blocked"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
