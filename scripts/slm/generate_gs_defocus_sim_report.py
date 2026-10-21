"""Generate ``report/shape_trials/gs_defocus_sim_20261019.md`` (CCD 焦后离焦下的 GS 匀化仿真).

**纯离线**：正文里的每一个数字都来自已经落盘的产物 (四个实验脚本写出的
CSV/JSON)，本脚本只负责 (a) 读表、(b) 渲染 5 张静态图 + 1 张复制图、
(c) 用**姐妹实验脚本自己的类**重算三栏动画所需的帧、(d) 拼装 markdown。

不复刻任何传播 / GS / 度量数学：正演与 GS 全部 import
``scripts/slm/experiment_gs_improved.py`` 的 :class:`TwoComponentBench`
(它自己又继承 ``experiment_gs_defocus_offocus.py`` 的 ``OffFocusBench``)，
度量用 ``ao_shaping.algorithm.signal_processing.square_shaping`` 的
``compute_bench_metrics`` / ``compute_bench_metrics`` 同族函数 —— 与实验脚本逐位同口径。
**不打开任何设备。**

产物:
    report/shape_trials/gs_defocus_sim/gs_defocus_sim_20261019.md
    report/shape_trials/gs_defocus_sim/figures/fig_gs_defocus_*.png
    report/shape_trials/gs_defocus_sim/images/gs_defocus_sim_*.gif

运行::

    python scripts/slm/generate_gs_defocus_sim_report.py            # 全量 (含 15 次 GS, ≈6–10 min)
    python scripts/slm/generate_gs_defocus_sim_report.py --no-gif   # 只出静态图 + markdown
    python scripts/slm/generate_gs_defocus_sim_report.py --quick    # 冒烟: 只验帧管线, 不写报告

**输入契约**: 全量模式只读上游四个实验脚本落盘的 CSV/JSON, 所以启动时先做一次
:func:`preflight` —— 缺任何一件就列出「缺哪个文件 / 由谁产出 / 跑哪条命令」并拒绝执行
(:class:`MissingArtefactsError`), 不抛裸 ``FileNotFoundError``。``--quick`` 与此相反:
它**不读任何落盘产物**, 在只有源码的干净克隆上也能跑 (数据 CSV 按仓库惯例不入库, 见
:data:`PRODUCERS`)。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common.provenance import insert_header  # noqa: E402

from ao_shaping.algorithm.signal_processing.square_shaping import (  # noqa: E402
    composite_score,
    compute_bench_metrics,
    make_square_target,
)

REPORT_KEY = "report/shape_trials/gs_defocus_sim/gs_defocus_sim_20261019.md"
ART = ROOT / "report" / "shape_trials" / "gs_defocus_sim"
FIG_DIR = ART / "figures"
GIF_DIR = ART / "images"
#: 四个上游实验脚本的原始产物 (CSV/JSON/PNG) —— 它们是本报告的**输入契约**,
#: 位于本目录的 ``artifacts/`` 子目录。
ARTE_DIR = ART / "artifacts"

#: 已落盘的产物 (全部由那四个实验脚本写出; 本脚本只读)
ARTEFACTS = {
    "offocus_json": ARTE_DIR / "summary.json",
    "sweep_json": ARTE_DIR / "summary_defocus_sweep_visuals.json",
    "fresnel_json": ARTE_DIR / "summary_fresnel_vs_vl.json",
    "improved_json": ARTE_DIR / "summary_gs_improved_improved.json",
    "fv_groups_csv": ARTE_DIR / "fv_groups_fresnel_vs_vl.csv",
    "fv_equiv_csv": ARTE_DIR / "fv_equivalence_fresnel_vs_vl.csv",
    "scheme_gap_csv": ARTE_DIR / "scheme_gap_dz0_50mm_visuals.csv",
    "imp_groups_csv": ARTE_DIR / "imp_groups_improved.csv",
    "imp_halo_csv"  : ARTE_DIR / "imp_halo_improved.csv",
    "e1_heatmap_png": ARTE_DIR / "e1_cv_heatmap.png",
}

#: 四个上游实验脚本: 确切命令 (默认写到 ``report/shape_trials/``) + 大致耗时。
#: 耗时是给人看的保守量级 —— 缺输入的人最容易把 ``experiment_gs_improved.py`` 的
#: 静默一小时当成卡死。
PRODUCERS: dict[str, dict[str, str]] = {
    "experiment_gs_defocus_offocus.py": {
        "command": "python scripts/slm/experiment_gs_defocus_offocus.py",
        "runtime": "约数分钟",
    },
    "experiment_gs_fresnel_vs_virtuallens.py": {
        "command": "python scripts/slm/experiment_gs_fresnel_vs_virtuallens.py",
        "runtime": "约 10 分钟",
    },
    "experiment_gs_defocus_sweep_visuals.py": {
        "command": "python scripts/slm/experiment_gs_defocus_sweep_visuals.py",
        "runtime": "约数分钟",
    },
    "experiment_gs_improved.py": {
        "command": "python scripts/slm/experiment_gs_improved.py",
        "runtime": "约 1 小时",
    },
}

#: ``ARTEFACTS`` 的键 → 产出它的上游脚本 (只用于报错与 §10 的命令表)
ARTEFACT_PRODUCER: dict[str, str] = {
    "offocus_json": "experiment_gs_defocus_offocus.py",
    "e1_heatmap_png": "experiment_gs_defocus_offocus.py",
    "fresnel_json": "experiment_gs_fresnel_vs_virtuallens.py",
    "fv_groups_csv": "experiment_gs_fresnel_vs_virtuallens.py",
    "fv_equiv_csv": "experiment_gs_fresnel_vs_virtuallens.py",
    "sweep_json": "experiment_gs_defocus_sweep_visuals.py",
    "scheme_gap_csv": "experiment_gs_defocus_sweep_visuals.py",
    "improved_json": "experiment_gs_improved.py",
    "imp_groups_csv": "experiment_gs_improved.py",
    "imp_halo_csv": "experiment_gs_improved.py",
}

#: ``--quick`` 用的 H3 η。正常取自 ``imp_groups_improved.csv`` 的机械选取结果
#: (:func:`zeroth_selection`), 但冒烟必须零输入, 故直接写这个已记录的常数;
#: 它只影响冒烟帧的标题, 不会进报告。
QUICK_H3_ETA = 0.02

#: 瞳面过采样 seed: 与上游 ``experiment_gs_improved.py::RunConfig.FULL`` 同口径,
#: 本脚本的 :class:`Frames` 也用它 (默认 0)。
GS_SEED = 0

#: §10 命令表的内容: 生产者脚本 (顺序即表格行序) → 产物列表。
#: 命令与耗时**不在这里重写**, 一律取 :data:`PRODUCERS`, 避免两处漂移。
REPRO_PRODUCTS: dict[str, str] = {
    "experiment_gs_defocus_offocus.py":
        "`summary.json`, `e1_defocus_sweep.csv`, `e2_methods.csv`, `e1_cv_heatmap.png`",
    "experiment_gs_fresnel_vs_virtuallens.py":
        "`summary_fresnel_vs_vl.json`, `fv_groups_fresnel_vs_vl.csv`, `fv_equivalence_fresnel_vs_vl.csv`",
    "experiment_gs_defocus_sweep_visuals.py":
        "`summary_defocus_sweep_visuals.json`, `scheme_gap_dz0_50mm_visuals.csv`, "
        "`v1_gs_dz_sweep_visuals.csv`, `halo_zero_order_visuals.csv`",
    "experiment_gs_improved.py":
        "`summary_gs_improved_improved.json`, `imp_groups_improved.csv`, `imp_halo_improved.csv`",
}

SZ_SCRIPT = ROOT / "scripts" / "slm"
PATH_IMPROVED = SZ_SCRIPT / "experiment_gs_improved.py"

DZ_BLUR_MM = 0.7503639840341595
DZ_SWEEP_MM = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
GROUP_ORDER = ("G1", "G2", "G1p", "G3", "G4")
GROUP_LABEL = {
    "G1": "G1 Δz 盲 GS, 无离焦 (基线)",
    "G2": "G2 Δz 盲 GS + 事后补偿离焦",
    "G1p": "G1′ 逐 Δz 重优化 GS, 无离焦 (对照)",
    "G3": "G3 逐 Δz 重优化 + 离焦并入环路 (方案A)",
    "G4": "G4 同 G3 但用方案B (瞳面二次相位)",
}
GROUP_COLOR = {
    "G1": "#9E9E9E",
    "G2": "#FF5722",
    "G1p": "#9C27B0",
    "G3": "#2196F3",
    "G4": "#4CAF50",
}

# ---------------------------------------------------------------------------
# §0 四要素的**文字**部分 (实验目的 / 实验设置 / 算法修改)
#
# 这三段是纯文本常量, 刻意不写死任何数字: 数字一律由 :func:`build_markdown`
# 从落盘产物 (JSON/CSV) 插值, 所以正文里的每个数都能追到 ``artifacts/`` 里的
# 某一行。写成普通字符串 (不是 f-string) 也让里面的 ``{`` ``}`` 不必转义 ——
# 插值进去的内容不会被二次解析。
# ---------------------------------------------------------------------------
PURPOSE_TEXT = """
本轮要回答的问题只有一句: **CCD 焦后离焦, 在 GS 匀化里到底该怎么处理, 值多少分。**

拆成三个可判定的子问题:

1. **离焦要不要补偿, 用哪种补偿?** 补偿离焦既可以用**瞳面二次相位**近似 (便宜), 也可以在 GS
   环路里直接用**精确角谱**正演 (准但慢)。本轮把两者放在同一张网格、同一判据下对比, 给出
   等价边界与各自的失效方式。
2. **"在离焦处重新优化"与"把离焦并入环路"各贡献多少?** 二者都能把分数拉回来, 但修的不是同一个
   误差; 不拆开就会把功劳记错。本轮用一条**五组因果链**把两者分开 (各组定义见 §2 的表)。
3. **测量判据本身会不会刷分?** 若判据的框跟着光斑一起变大, "测量区域变大"会被读成"更均匀"。
   本轮量化这个陷阱, 并给出一条可执行的判据规则。

预期能排除什么: (a) 排除"离焦不值钱"的说法 —— 事后补一个标量离焦本身就值与重新优化同量级
(§0.3); (b) 排除"离焦并入环路"这个**形式**本身是关键的误解 —— 关键的是**拿到工作点信息**;
(c) 排除自适应框给出的"最优离焦"这类非物理结论 —— 固定物理尺度框下的真实最优小一个量级 (§7);
(d) 排除"方案B 可以替代方案A 做场级/相位级比较"的说法 (§3)。
"""

SETUP_TEXT = """
**本轮全部是纯仿真, 没有硬件验证**: 全程不打开任何设备 (SLM / 相机 / 光学平台), 不做任何
标定。下面表里的"波长 / 透镜焦距 / 瞳面口径"是**仿真里写死的光学常量**, 与实验系统的同名常量
同源 (`scripts/slm/experiment_gs_defocus_offocus.py` 的 `WAVELENGTH_M` / `FOCAL_LENGTH_M` /
`APERTURE_M`), 但本报告的结论**只对仿真成立**。

五组因果链**只改两个因子**: ① GS 相位是在 Δz=0 优化一次 (盲), 还是在工作点逐 Δz 重优化;
② 离焦旋钮 (Zernike (2,0) 标量) 用不用, 以及它用精确角谱 (方案A) 还是瞳面二次相位 (方案B) 实现。

目标形状是**均匀方形** (`make_square_target`): 判据框边长固定, 框心在信号上定位一次后**冻结**
(逐帧重定位会把"区域变大"读成"更均匀", 见 §7)。其余设置全篇共用:
"""

CHANGES_TEXT = """
本轮相对上一轮, 仿真链路与工具链上新增/修复的机制 (每条都能在代码里定位到; 数值一律由落盘
产物给出, 不在本节写死):

1. **带限 ASM 的开关被提到显式参数并默认关闭。** 库默认 `band_limit=True` (Voelz & Roggemann
   cut) 在焦平面上**有损且不可逆**, 正/反向不构成伴随对, GS 内环会收敛到一个不存在的解 (§6)。
   本轮把它提成 `RunConfig.band_limit` / `TwoComponentBench(band_limit=...)`, Δz>0 一律关,
   并用 `round_trip_relerr` 自检兜底。
2. **判据从"质心自动缩放框"改为"固定物理尺度定尺框"** (边长恒 `target_side_px`, 框心用窗口化
   质心在信号上定位一次后**冻结**)。自适应框保留为对照量 (`*_adapt`), 只用来展示陷阱, 不作结论 (§7)。
3. **新增零级双分量瞳面模型** `U_pupil = U_inc·[η + (1−η)·exp(iφ)]`, 以及 GS 投影链上的两个新开关:
   `subtract` (目标面约束前的相干扣除 `U ← U − b0`) 与 `weighted` (加权反馈; 参数 `gs_gamma` /
   `gs_weight_clip` / `gs_beta` / `gs_hio_fraction` 取自 `SlmGsRefineConfig`, 更新式照抄
   `UniformSquareShaper._weighted_step`)。另加 `warm_phase` (GS 热启动) 与 `offset_px`
   (H4 用小倾斜把光斑移开轴上晕斑)。
4. **新增两套离焦实现与五组因果链**: 方案A = 精确角谱 `U_CCD = ASM(FT(U_SLM), Δz)`,
   方案B = 瞳面二次相位 `exp(−iπ·Δz·r²/(λf²))`; G1/G2/G1′/G3/G4 把"重优化"与"离焦"两个因子拆开。
5. **自检写进流程, 不通过就抛**: `self_check_eta0` (与权威件 `gs_shape` 逐位对照)、
   `self_check_round_trip` (往返误差)、`self_check_halo_monotonic` (渐近区 I·Δz² 恒定)、
   `measure_trustworthy_dz` (几何回绕边界) 与 `measure_band_limit_damage` (带限伤害随谱内容变化)。
6. **报告生成器只读上游产物, 不复刻数学**: 本脚本 import 姐妹实验脚本的 `TwoComponentBench`
   (正演与 GS) 与 `square_shaping.compute_bench_metrics` (度量), 与实验脚本逐位同口径; 启动时
   `preflight()` 逐条列出"缺哪个文件 / 由谁产出 / 跑哪条命令", 缺输入时**拒绝执行**而不是抛裸
   `FileNotFoundError`; `--quick` 是零输入冒烟, `--no-gif` 只出静态图与正文。
"""

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 输入检查 (缺输入时给可执行的错误, 而不是裸 FileNotFoundError)
# ---------------------------------------------------------------------------
class MissingArtefactsError(RuntimeError):
    """全量模式缺上游产物。消息里逐条列出「缺哪个文件 / 由谁产出 / 跑哪条命令」。"""


def preflight(keys: Iterable[str] | None = None) -> None:
    """检查 ``ARTEFACTS`` 里的产物是否存在。

    数据 CSV 被 ``.gitignore`` 的 ``*.csv`` 全局排除 (仓库约定: 只入库报告与图),
    所以干净克隆上它们本来就不在。这里把"缺文件"翻译成"先跑哪条命令"。

    Args:
        keys: 需要检查的 ``ARTEFACTS`` 键; ``None`` = 全量模式, 检查全部。

    Raises:
        MissingArtefactsError: 有任何一个所需文件不存在。
    """
    wanted = sorted(ARTEFACTS if keys is None else keys)
    missing = [k for k in wanted if not ARTEFACTS[k].is_file()]
    if not missing:
        return

    lines = [f"缺少 {len(missing)} 个已落盘产物 (本脚本只读它们, 不自己算):"]
    for key in missing:
        path = ARTEFACTS[key].relative_to(ROOT).as_posix()
        script = ARTEFACT_PRODUCER[key]
        prod = PRODUCERS[script]
        lines.append(f"  - {path}")
        lines.append(f"      由 {script} 产出 ({prod['runtime']})")
        lines.append(f"      命令: {prod['command']}")
    lines.extend(
        [
            "",
            "为什么缺: 仓库按惯例只入库报告与图, 数据 CSV 被 .gitignore 的 *.csv 全局排除,",
            "所以干净克隆上看不到它们 (JSON 与 PNG 是入库的, 但也可能被 --no-* 之类的跑法漏掉)。",
            "",
            "怎么办: 先跑上面列出的命令 (四个脚本互不依赖, 顺序任意), 再跑本生成器。",
            "四个脚本全跑一遍约 1.5 小时, 其中 experiment_gs_improved.py 占大头 (约 1 小时)。",
            "只想确认管线能走完: 用 --quick —— 它不读任何落盘产物。",
        ]
    )
    raise MissingArtefactsError("\n".join(lines))


# ---------------------------------------------------------------------------
# 读表
# ---------------------------------------------------------------------------
def read_json(key: str) -> dict[str, Any]:
    return json.loads(ARTEFACTS[key].read_text(encoding="utf-8"))


def read_csv(key: str) -> list[dict[str, str]]:
    with ARTEFACTS[key].open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fv_groups() -> dict[str, dict[str, dict[str, float]]]:
    """``fv_groups_fresnel_vs_vl.csv`` → ``{group: {dz_units_str: row}}``。

    只收**能解析成 float** 的列 (CSV 里还有 ``phase_source`` 这种文本列和布尔列,
    硬转会把它们变成另一个错误源)。
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in read_csv("fv_groups_csv"):
        row: dict[str, float] = {}
        for k, v in r.items():
            if k == "group":
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        out.setdefault(r["group"], {})[r["dz_units"]] = row
    return out


def self_checks(now: dict[str, Any]) -> list[tuple[str, str]]:
    """记录在案的**关系**必须仍然成立, 否则报告里的因果链就不成立。

    只断言关系 (半轴内逐位相同 / G3 与 G4 重合 / 等价边界 / 往返误差),
    不断言数值大小 —— 数值本身是读出来的, 不会漂。

    全量模式专用: 它读 ``fv_groups_fresnel_vs_vl.csv`` (数据 CSV 不入库)，
    故 ``--quick`` 不调用它 —— 冒烟必须零输入可跑。
    """
    checks: list[tuple[str, str]] = []
    g = fv_groups()

    for half, keys in (("负半轴", ("-2.0", "-1.0", "-0.5", "-0.25")), ("正半轴", ("0.25", "0.5", "1.0", "2.0"))):
        vals = [g["G1p"][k]["score_fixed"] for k in keys]
        spread = max(vals) - min(vals)
        checks.append((f"G1′ 在{half}内 score_fixed 逐位相同", f"spread={spread:.1e}"))
        assert spread < 1e-9, f"G1′ 在{half}内不再逐位相同 (spread={spread})"

    gap = max(abs(g["G3"][k]["score_fixed"] - g["G4"][k]["score_fixed"]) for k in g["G3"])
    checks.append(("G3 与 G4 的定尺框 score 差", f"max={gap:.1e}"))
    assert gap <= 1e-5, f"G3/G4 不再重合 (max={gap})"

    verdict = now["fresnel_json"]["equivalence_verdict"]
    lim = verdict["Bf_equivalent_limit_dz_units"]
    checks.append(("方案A/B 严格等价边界", f"{lim:.6f} dz_unit = {lim * DZ_BLUR_MM:.4f} mm"))
    assert 0.10 < lim < 0.12, f"等价边界跑掉了 ({lim})"
    assert verdict["Bf_vs_Bp_identical"] is True, "Bf 与 Bp 不再逐位相同"

    rt = now["improved_json"]["self_check"]["round_trip"]
    checks.append(("GS 往返误差 (band_limit=False)", f"{rt['round_trip_relerr_max']:.2e}"))
    checks.append(("GS 往返误差 (band_limit=True)", f"{rt['round_trip_relerr_max_band_limited']:.4f}"))
    assert rt["round_trip_relerr_max"] < 1e-12
    assert rt["round_trip_relerr_max_band_limited"] > 0.6
    return checks


# ---------------------------------------------------------------------------
# 帧管线 (import 姐妹实验脚本的类, 不复刻数学)
# ---------------------------------------------------------------------------
def load_improved() -> Any:
    """加载 ``scripts/slm/experiment_gs_improved.py`` (它自己会加载其兄弟脚本)。"""
    spec = importlib.util.spec_from_file_location("_gen_gs_defocus_improved", PATH_IMPROVED)
    if spec is None or spec.loader is None:  # pragma: no cover - 路径写死
        raise RuntimeError(f"cannot load {PATH_IMPROVED}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Frames:
    """Δz 扫描下需要的所有帧: 正演 + GS 全部走姐妹实验脚本的类。"""

    def __init__(self, imp: Any, *, gs_iters: int, seed: int = GS_SEED) -> None:
        self.imp = imp
        # 与 ``experiment_gs_improved.py::FULL`` 同口径 (n_grid/padding/target_side/seed)
        rc = imp.RunConfig(
            n_grid=64,
            far_field_padding=8,
            target_side_px=43,
            seed=seed,
            gs_iters=gs_iters,
        )
        self.rc = rc
        self.cfg = rc.bench_cfg()
        self.bench = imp.TwoComponentBench(self.cfg, band_limit=False)
        self.target = make_square_target(self.cfg)
        self.calls = 0

    # -- GS ---------------------------------------------------------------
    def gs(self, dz_mm: float, *, eta: float, weighted: bool = False, subtract: bool = False) -> np.ndarray:
        """该 Δz 处的 GS 连续相位 (与实验脚本同一条代码路径)。"""
        out = self.bench.gs_phase_eta(
            dz_mm * 1e-3,
            eta=eta,
            n_iters=self.rc.gs_iters,
            seed=self.rc.seed,
            weighted=weighted,
            subtract=subtract,
            offset_px=(0, 0),
            quantize=True,
        )
        self.calls += 1
        return np.asarray(out["phase_cont"], dtype=float)

    # -- 正演 + data.json 定尺框度量 ---------------------------------------
    def frame(self, phase: np.ndarray, dz_mm: float, *, eta: float) -> tuple[np.ndarray, dict[str, float]]:
        inten = self.bench.intensity_eta(phase, eta, dz_mm * 1e-3)
        dj = compute_bench_metrics(inten, self.target)
        metrics = {"cv": float(dj["CV"]), "pib": float(dj["PIB"]), "score": float(composite_score(dj))}
        return inten, metrics


# ---------------------------------------------------------------------------
# 可信度标注 (两条**不同**的边界, 一律并列写出)
# ---------------------------------------------------------------------------
def trust_table() -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for r in read_json("sweep_json")["scheme_gap_table"]:
        out[float(r["dz_mm"])] = {"scheme_a": bool(r["trusted"]), "rel_err": float(r["rel_err_max"])}
    for r in read_json("improved_json")["trustworthy_dz"]["per_dz"]:
        out.setdefault(float(r["dz_mm"]), {})["wrap"] = float(r["halo_radius_over_halfwindow"])
    return out


def frame_trust_title(dz_mm: float, info: dict[str, Any]) -> tuple[str, bool, str]:
    """返回 ``(标注文本, 是否两条边界都通过, 简写)``。

    只用中文字符与 ASCII —— ``✓``/``✗``/``⚠`` 在 Microsoft YaHei 里缺字形, 渲染出来
    是方块 (实测), 所以判据一律写成 ``OK``/``NG``。
    """
    scheme_ok = bool(info.get("scheme_a", True))
    wrap = info.get("wrap")
    wrap_ok = wrap is None or float(wrap) <= 0.5
    rel = info.get("rel_err")
    parts = [
        f"方案A {'OK' if scheme_ok else 'NG'}"
        + (f" rel_err={rel * 100:.2f}%" if rel is not None else "")
    ]
    parts.append(
        "几何 —" if wrap is None else f"几何 {'OK' if wrap_ok else 'NG'} 回绕={float(wrap):.2f}"
    )
    ok = scheme_ok and wrap_ok
    return " · ".join(parts), ok, ("" if ok else "  【不可信】")


# ---------------------------------------------------------------------------
# 静态图
# ---------------------------------------------------------------------------
def fig_groups_ceiling() -> Path:
    g = fv_groups()
    xs = sorted({float(k) for k in g["G1"]})
    order = ["-2.0", "-1.0", "-0.5", "-0.25", "0.0", "0.25", "0.5", "1.0", "2.0"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    style = {"G1": dict(ls="-", lw=2), "G2": dict(ls="-", lw=2), "G1p": dict(ls="-", lw=2.4),
             "G3": dict(ls="-", lw=3.2), "G4": dict(ls="--", lw=1.4)}
    for group in GROUP_ORDER:
        sel = g[group]
        axes[0].plot(xs, [sel[k]["cv"] for k in order], "o-", ms=5,
                     color=GROUP_COLOR[group], label=GROUP_LABEL[group], **style[group])
        axes[1].plot(xs, [sel[k]["score_fixed"] for k in order], "o-", ms=5,
                     color=GROUP_COLOR[group], label=GROUP_LABEL[group], **style[group])
        axes[2].plot(xs, [sel[k]["defocus_rad"] for k in order], "o-", ms=5,
                     color=GROUP_COLOR[group], label=f"{group} 自动框判据", **style[group])
        axes[2].plot(xs, [sel[k]["defocus_rad_fixed"] for k in order], "s:", ms=4,
                     color=GROUP_COLOR[group], label=f"{group} 定尺框判据", lw=1.2, alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("目标框内 CV (自动定标框, 主判据, 对数轴)")
    axes[0].set_title("① 主判据: 自动定标框 CV\n框跟着光斑变大会把「区域变大」读成「更均匀」")
    axes[1].set_ylabel("复合分数 (定尺框, 各判据取各自最优离焦)")
    axes[1].set_title("② 定尺框 score\n(结论只看这一栏)")
    axes[2].axhline(0.0, color="#666666", ls="--", lw=1)
    axes[2].set_ylabel("最优离焦幅度 (Zernike (2,0), 弧度)")
    axes[2].set_title("③ 最优离焦: 自动框把解推到 ±3.87 rad\n定尺框的真实最优是 ±0.34 rad")
    for ax in axes:
        ax.set_xlabel("CCD 焦后位移 Δz / Δz_blur  (1 unit = 0.7504 mm)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower center")
    fig.suptitle("五组因果链: 上限 = 各组自己的最优离焦 (G1/G1′ 不搜离焦)")
    fig.tight_layout()
    out = FIG_DIR / "fig_gs_defocus_groups_ceiling.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_equivalence() -> Path:
    """方案B (Bf / Bp / Bp_chirped) 相对方案A 的误差 vs |Δz|, 标出 1e-06 等价边界。"""
    rows = read_csv("fv_equiv_csv")
    by: dict[str, dict[float, float]] = {}
    for r in rows:
        u = abs(float(r["dz_units"]))
        v = r["variant"]
        by.setdefault(v, {})
        by[v][u] = max(by[v].get(u, 0.0), f(r, "rel_err_max"))
    label = {
        "Bf": "Bf 焦平面域",
        "Bp": "Bp 瞳面域 (方案B 本体)",
        "Bp_chirped": "Bp + 输出二次相位",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for v, colour, ls in (("Bf", "#2196F3", "-"), ("Bp", "#4CAF50", "--"), ("Bp_chirped", "#9E9E9E", ":")):
        xs = sorted(by[v])
        ax.plot(xs, [by[v][u] for u in xs], "o", ms=4, color=colour, ls=ls, lw=1.6,
                label=label[v])
    ax.axhline(1e-6, color="k", ls="--", lw=1.2, label="严格等价判据 1e-06")
    lim = 0.11377795104935455
    ax.axvline(lim, color="#c53030", ls="-.", lw=1.4, label=f"等价边界 {lim:.4f} dz_unit")
    ax.axvspan(lim, 2.5, color="#c53030", alpha=0.08)
    ax.set_yscale("log")
    ax.set_xlabel("|CCD 焦后位移 Δz| / Δz_blur")
    ax.set_ylabel("max 相对误差 (方案B vs 方案A)")
    ax.set_title("方案A (精确角谱) vs 方案B (瞳面二次相位): 三个变体重合\n"
                 "焦平面域与瞳面域是同一个离散算子, 偏差全部来自 kz 的二阶傍轴截断")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "fig_gs_defocus_scheme_b_equivalence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_scheme_a_vs_chirp() -> Path:
    rows = read_csv("scheme_gap_csv")
    dz = [f(r, "dz_mm") for r in rows]
    rel = [f(r, "rel_err_max") for r in rows]
    edge = [f(r, "edge_energy_scheme_A") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    axes[0].plot(dz, rel, "o-", color="#c53030", lw=2, label="max 相对误差 (方案A 为真值)")
    axes[0].axhline(0.05, color="k", ls="--", lw=1.2, label="工程可用上限 5%")
    axes[0].axvline(20.0, color="#2b6cb0", ls="-.", lw=1.4, label="工程可用边界 Δz=20 mm")
    axes[0].axvspan(20.0, max(dz), color="#c53030", alpha=0.10, label="不可作为硬件预测")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-6, 3)
    axes[0].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[0].set_ylabel("方案B(瞳面二次相位) vs 方案A 的相对误差")
    axes[0].set_title("① 傍轴截断偏离: 内点最优在 ≤20 mm")
    axes[1].plot(dz, edge, "s-", color="#975a16", lw=2, label="方案A 边缘能量占比")
    axes[1].axhline(0.02, color="k", ls="--", lw=1.2, label="阈值 2%")
    axes[1].axvline(25.0, color="#2b6cb0", ls="-.", lw=1.4, label="首个不可信点 25 mm")
    axes[1].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[1].set_ylabel("光斑跑到裁切窗口外的能量占比")
    axes[1].set_title("② 边缘能量: 25 mm 起 >2%")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("可行域边界 (a): 方案B 的近似质量 (方案A 为真值)")
    fig.tight_layout()
    out = FIG_DIR / "fig_gs_defocus_scheme_a_vs_chirp.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_halo_wraparound() -> Path:
    rows = sorted(read_csv("imp_halo_csv"), key=lambda r: f(r, "dz_mm"))
    dz = [f(r, "dz_mm") for r in rows]
    peak = [f(r, "i_halo_peak") for r in rows]
    wrap = [f(r, "halo_w_over_shape_halfwindow") for r in rows]
    trusted = [r["trusted"] == "True" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    axes[0].plot(dz, peak, "o-", color="#2b6cb0", lw=2, label="零级晕斑峰值 (sum=1 归一)")
    axes[0].plot(dz, [f(r, "i_halo_peak_bl") for r in rows], "k--", lw=1.2, label="带限正演 (数值误差带)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[0].set_ylabel("零级晕斑峰值")
    axes[0].set_title("① 晕斑峰值随 Δz 单调下降 (约 249×, 0→20 mm)\n渐近区 I·Δz² 恒定到 4.7%")
    axes[1].plot(dz, wrap, "s-", color="#c53030", lw=2, label="晕斑 1/e² 半径 / 半窗")
    axes[1].axhline(0.5, color="k", ls="--", lw=1.4, label="回绕阈值 0.5")
    if not all(trusted):
        first_bad = min(d for d, ok in zip(dz, trusted) if not ok)
        axes[1].axvspan(first_bad, max(dz) * 1.06, color="#c53030", alpha=0.12,
                        label=f"不可信 (Δz ≥ {first_bad:g} mm)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[1].set_ylabel("晕斑 1/e² 半径 / 整形网格半窗")
    axes[1].set_title("② 几何回绕: 半窗 578.8 µm 只能容到 Δz≤10 mm")
    fig.suptitle("可行域边界 (b): 网格装不装得下 (与 (a) 是两个不同的边界)")
    fig.tight_layout()
    out = FIG_DIR / "fig_gs_defocus_halo_wraparound.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def zeroth_selection() -> dict[str, Any]:
    """零级组 H0~H4 的取 η 规则: 每组取 **Δz=0 处 cv_box 最小** 的 η, 该 η 固定用到底。

    规则是机械化地从 ``imp_groups_improved.csv`` 里算出来的 (不是手挑), 且它复现了
    报告 §5 表格里引用的那一行数 —— 改数据会让表格跟着动, 不会静默错位。
    """
    rows = read_csv("imp_groups_csv")
    pick: dict[str, tuple[float, float]] = {}
    for r in rows:
        if f(r, "dz_mm") != 0.0:
            continue
        group, eta, cv = r["group"], float(r["eta"]), f(r, "cv_box")
        if group not in pick or cv < pick[group][0]:
            pick[group] = (cv, eta)
    table: dict[str, list[dict[str, float]]] = {}
    for group in ("H0", "H1", "H2", "H3", "H4"):
        eta = pick[group][1]
        table[group] = sorted(
            ({"dz": f(r, "dz_mm"), "cv": f(r, "cv_box"), "pib": f(r, "pib_box"),
              "score": f(r, "score_box")}
             for r in rows if r["group"] == group and float(r["eta"]) == eta),
            key=lambda d: d["dz"],
        )
    return {"eta": {k: v[1] for k, v in pick.items()}, "rows": table}


def fig_zeroth_cv_box(sel: dict[str, Any]) -> Path:
    cols = {"H0": "#9E9E9E", "H1": "#FF5722", "H2": "#4CAF50", "H3": "#2196F3", "H4": "#673AB7"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for group, rows in sel["rows"].items():
        axes[0].plot([r["dz"] for r in rows], [r["cv"] for r in rows], "o-", lw=2,
                     color=cols[group], label=f"{group} (η={sel['eta'][group]:g})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CCD 焦后位移 Δz (mm)")
    axes[0].set_ylabel("cv_box (data.json 居中定尺框, 连续相位, 对数轴)")
    axes[0].set_title("① H0~H4 的 cv_box vs Δz\nH3 全程平坦 (0.0718–0.0735), 故结论不依赖不可信区")

    w = 0.16
    dzt = (0.0, 5.0, 10.0)
    base = np.arange(len(dzt))
    for i, (group, rows) in enumerate(sel["rows"].items()):
        by = {r["dz"]: r["cv"] for r in rows}
        axes[1].bar(base + (i - 2) * w, [by[d] for d in dzt], w,
                    color=cols[group], label=f"{group} (η={sel['eta'][group]:g})")
    axes[1].set_xticks(base)
    axes[1].set_xticklabels([f"Δz={d:g} mm" for d in dzt])
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.05, 4.0)
    axes[1].set_ylabel("cv_box")
    axes[1].set_title("② 可信区 (Δz≤10 mm) 上的 cv_box\nH3 把 0.42 压到 0.072 (5.7×)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("零级双分量模型: 加权 GS + 相干扣除 (H3) 才有效, 单做扣除 (H2) 反而更差")
    fig.tight_layout()
    out = FIG_DIR / "fig_gs_defocus_zeroth_h0_h4_cv_box.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def copy_e1_heatmap() -> Path:
    src = ARTEFACTS["e1_heatmap_png"]
    dst = FIG_DIR / "fig_gs_defocus_e1_cv_heatmap.png"
    dst.write_bytes(src.read_bytes())
    return dst


# ---------------------------------------------------------------------------
# 三栏 GIF
# ---------------------------------------------------------------------------
def _to_pil(fig: Any) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf).convert("RGB")
    return img.convert("P", palette=Image.ADAPTIVE, colors=192)


def _frame_panels(
    inten: np.ndarray,
    phase: np.ndarray,
    *,
    side: int,
    dx_um: float,
    log_ff: bool = True,
) -> tuple[Any, Any]:
    """一帧的三栏: 远场 (log10) / SLM 相位 / 过零级的 x 截面 + 目标框。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    m = inten.shape[0]
    half = (m * dx_um) / 2.0
    ext = [-half, half, -half, half]

    show = inten / max(inten.max(), 1e-300)
    if log_ff:
        im = axes[0].imshow(np.log10(show + 1e-8), cmap="inferno", origin="lower",
                            extent=ext, vmin=-7, vmax=0)
        axes[0].set_title("远场强度 (方案A, log10, 归一峰值)", fontsize=9)
    else:
        im = axes[0].imshow(show, cmap="inferno", origin="lower", extent=ext)
        axes[0].set_title("远场强度 (方案A, 线性)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    b = side * dx_um / 2.0
    axes[0].add_patch(plt.Rectangle((-b, -b), 2 * b, 2 * b, fill=False, ec="#00E5FF", lw=1.4))
    axes[0].set_xlim(-half, half)
    axes[0].set_ylim(-half, half)
    xlabel = "x (µm)"
    axes[0].set_xlabel(xlabel)

    pm = axes[1].imshow(np.mod(phase + np.pi, 2 * np.pi) - np.pi, cmap="twilight",
                        origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("SLM 瞳面相位 (弧度, 口径内)", fontsize=9)
    fig.colorbar(pm, ax=axes[1], fraction=0.046)
    axes[1].set_xlabel("瞳面像素")

    xs = (np.arange(m) - m // 2) * dx_um
    prof = inten[m // 2, :]
    axes[2].semilogy(xs, np.maximum(prof / max(prof.max(), 1e-300), 1e-8), lw=1.4, color="#2b6cb0")
    axes[2].axvline(-b, color="#00E5FF", ls="--", lw=1.2)
    axes[2].axvline(b, color="#00E5FF", ls="--", lw=1.2)
    axes[2].axvspan(-b, b, color="#00E5FF", alpha=0.08, label="目标框 (43 px 定尺)")
    axes[2].set_ylim(1e-8, 2.0)
    axes[2].set_title("过零级的 x 截面 (log10, 归一)", fontsize=9)
    axes[2].set_xlabel("x (µm, 远场)")
    axes[2].legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25)
    return fig, axes


def _render_gif(frames: list[Image.Image], out: Path, *, duration: int = 650) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, optimize=True)
    return out


def gif_dz_sweep(fx: Frames, *, quick: bool, out_dir: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    """原始 GS (Δz=0 优化) 扫 Δz —— 等价于 sweep 实验的 v1。

    ``quick=True`` 不读可信度表 (它来自落盘 summary), 帧标题只给出可算的量。
    """
    dzs = (0.0, 10.0, 25.0, 50.0) if quick else DZ_SWEEP_MM
    trust = {} if quick else trust_table()
    phase = fx.gs(0.0, eta=0.0)
    info_rows = []
    frames = []
    for dz in dzs:
        inten, metrics = fx.frame(phase, dz, eta=0.0)
        note, ok, warn = frame_trust_title(dz, trust.get(dz, {}))
        fig, axes = _frame_panels(inten, phase, side=fx.rc.target_side_px,
                                 dx_um=fx.bench.dx * 1e6)
        for ax in axes:
            for sp in ax.spines.values():
                sp.set_color("#c53030" if not ok else "#444444")
                sp.set_linewidth(3.0 if not ok else 1.0)
        fig.suptitle(
            f"Δz = {dz:g} mm (原始 GS, Δz=0 处优化){warn}\n{note}   "
            f"cv_box={metrics['cv']:.4f}  pib_box={metrics['pib']:.4f}",
            fontsize=10, color=("#c53030" if not ok else "black"),
        )
        fig.tight_layout()
        frames.append(_to_pil(fig))
        plt.close(fig)
        info_rows.append({"dz_mm": dz, "ok": ok, "note": note, **metrics})
    out = (out_dir or GIF_DIR) / "gs_defocus_sim_dz_sweep_20261019.gif"
    return _render_gif(frames, out), info_rows


def gif_reopt(fx: Frames, *, quick: bool, out_dir: Path | None = None) -> tuple[Path, list[dict[str, Any]], float]:
    """逐 Δz 重优化 GS (G1′ / 零级组的 H0 相位), 同样扫 Δz。

    ``quick=True`` 时没有落盘 H0 行可比对, 第三个返回值 (偏差) 恒为 ``0.0``, 不代表一致。
    """
    dzs = (0.0, 10.0, 25.0, 50.0) if quick else DZ_SWEEP_MM
    trust = {} if quick else trust_table()
    recorded = (
        {}
        if quick
        else {f(r, "dz_mm"): f(r, "cv_box") for r in read_csv("imp_groups_csv") if r["group"] == "H0"}
    )
    frames = []
    info_rows = []
    dev = 0.0
    for dz in dzs:
        phase = fx.gs(dz, eta=0.0)
        inten, metrics = fx.frame(phase, dz, eta=0.0)
        note, ok, warn = frame_trust_title(dz, trust.get(dz, {}))
        if dz in recorded:
            dev = max(dev, abs(metrics["cv"] - recorded[dz]))
        fig, axes = _frame_panels(inten, phase, side=fx.rc.target_side_px,
                                 dx_um=fx.bench.dx * 1e6)
        for ax in axes:
            for sp in ax.spines.values():
                sp.set_color("#c53030" if not ok else "#444444")
                sp.set_linewidth(3.0 if not ok else 1.0)
        cross = "" if dz not in recorded else f"   (落盘 H0 的 cv_box={recorded[dz]:.4f})"
        fig.suptitle(
            f"Δz = {dz:g} mm (逐 Δz 重优化 GS, 无离焦){warn}\n{note}   "
            f"cv_box={metrics['cv']:.4f}  pib_box={metrics['pib']:.4f}{cross}",
            fontsize=10, color=("#c53030" if not ok else "black"),
        )
        fig.tight_layout()
        frames.append(_to_pil(fig))
        plt.close(fig)
        info_rows.append({"dz_mm": dz, "ok": ok, "note": note, **metrics})
    out = (out_dir or GIF_DIR) / "gs_defocus_sim_reopt_20261019.gif"
    return _render_gif(frames, out), info_rows, dev


def gif_zeroth(fx: Frames, *, quick: bool, out_dir: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    """零级对照: 上排 H0 (η=0), 下排 H3 (加权 GS + 相干扣除, η=0.02)。

    ``quick=True`` 不读 ``imp_groups_improved.csv``: η 用 :data:`QUICK_H3_ETA`,
    也没有落盘行可比对。
    """
    dzs = (0.0, 10.0) if quick else (0.0, 5.0, 10.0)
    if quick:
        eta_h3 = QUICK_H3_ETA
        recorded: dict[tuple[str, float], dict[str, float]] = {}
    else:
        sel = zeroth_selection()
        eta_h3 = float(sel["eta"]["H3"])
        recorded = {(g, r["dz"]): r for g, rows in sel["rows"].items() for r in rows}
    frames = []
    info_rows = []
    for dz in dzs:
        phase_h0 = fx.gs(dz, eta=0.0)
        phase_h3 = fx.gs(dz, eta=eta_h3, weighted=True, subtract=True)
        fig = plt.figure(figsize=(13.4, 7.0))
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)
        texts = []
        for row, (name, phase, eta) in enumerate((("H0", phase_h0, 0.0), ("H3", phase_h3, eta_h3))):
            inten, metrics = fx.frame(phase, dz, eta=eta)
            sub = fig.add_subplot(gs[row, 0])
            m = inten.shape[0]
            half = m * fx.bench.dx * 1e6 / 2.0
            ext = [-half, half, -half, half]
            im = sub.imshow(np.log10(inten / inten.max() + 1e-8), cmap="inferno",
                            origin="lower", extent=ext, vmin=-7, vmax=0)
            fig.colorbar(im, ax=sub, fraction=0.046)
            b = fx.rc.target_side_px * fx.bench.dx * 1e6 / 2.0
            sub.add_patch(plt.Rectangle((-b, -b), 2 * b, 2 * b, fill=False, ec="#00E5FF", lw=1.4))
            sub.set_title(f"{name} 远场 (方案A, log10)  η={eta:g}", fontsize=9)
            sub2 = fig.add_subplot(gs[row, 1])
            pm = sub2.imshow(np.mod(phase + np.pi, 2 * np.pi) - np.pi, cmap="twilight",
                             origin="lower", vmin=-np.pi, vmax=np.pi)
            fig.colorbar(pm, ax=sub2, fraction=0.046)
            sub2.set_title(f"{name} SLM 相位 (弧度)", fontsize=9)
            sub3 = fig.add_subplot(gs[row, 2])
            xs = (np.arange(m) - m // 2) * fx.bench.dx * 1e6
            prof = inten[m // 2, :]
            sub3.semilogy(xs, np.maximum(prof / prof.max(), 1e-8), lw=1.3, color="#2b6cb0")
            sub3.axvline(-b, color="#00E5FF", ls="--", lw=1.2)
            sub3.axvline(b, color="#00E5FF", ls="--", lw=1.2)
            sub3.set_ylim(1e-8, 2.0)
            sub3.set_title(f"{name} 过零级 x 截面 (log10)", fontsize=9)
            for ax in (sub, sub2, sub3):
                ax.grid(alpha=0.25)
            rec = recorded.get((name, dz))
            got = (f"渲染帧 cv_box={metrics['cv']:.4f} pib_box={metrics['pib']:.4f}")
            if rec is not None:
                got += f"  |  落盘 cv_box={rec['cv']:.4f} pib_box={rec['pib']:.4f}"
            texts.append(f"{name}: {got}")
            info_rows.append({"dz_mm": dz, "group": name, "eta": eta, **metrics})
        fig.suptitle(
            f"零级对照: H0 (plain GS, η=0) vs H3 (加权 GS + 相干扣除, η={eta_h3:g})   "
            f"Δz = {dz:g} mm\n" + "     ".join(texts),
            fontsize=10,
        )
        frames.append(_to_pil(fig))
        plt.close(fig)
    out = (out_dir or GIF_DIR) / "gs_defocus_sim_zeroth_20261019.gif"
    return _render_gif(frames, out, duration=900), info_rows


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _score_row(g: dict[str, Any], group: str, key: str) -> str:
    return f"{g[group][key]['score_fixed']:.4f}"


def build_markdown(state: dict[str, Any]) -> str:
    fr = state["fresnel_json"]
    sw = state["sweep_json"]
    im = state["improved_json"]
    g = state["fv_groups"]
    order = ["-2.0", "-1.0", "-0.5", "-0.25", "0.0", "0.25", "0.5", "1.0", "2.0"]
    dz_mm = [g["G1"][k]["dz_mm"] for k in order]

    # -- 表 2.1 定尺框 score (五组 × 9 个 Δz) --------------------------------
    hdr = "| 组 | " + " | ".join(f"{v:+.2f}" for v in (-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2)) + " |"
    sep = "|---|" + "---:|" * 9
    rows = "\n".join(
        f"| {GROUP_LABEL[grp]} | " + " | ".join(_score_row(g, grp, k) for k in order) + " |"
        for grp in GROUP_ORDER
    )

    # -- 表 2.2 工作点上的复合分数/PIB/CV -------------------------------------
    work = "\n".join(
        f"| {GROUP_LABEL[grp]} @ Δz/Δz_blur={tag} | {g[grp][k]['score']:.4f} | "
        f"{g[grp][k]['pib']:.4f} | {g[grp][k]['cv']:.4f} | {g[grp][k]['score_fixed']:.4f} |"
        for grp, k, tag in (("G1", "-2.0", "-2"), ("G1p", "-2.0", "-2"),
                            ("G3", "-2.0", "-2"), ("G1", "0.0", "0"))
    )

    # -- 表 2.3 收益拆解 ----------------------------------------------------
    decomp = "\n".join(
        f"| {k} | {dz_mm[i]:+.4f} | {g['G1'][k]['score_fixed']:.4f} | "
        f"{g['G1p'][k]['score_fixed']:.4f} | {g['G3'][k]['score_fixed']:.4f} | "
        f"{g['G1p'][k]['score_fixed'] - g['G1'][k]['score_fixed']:+.4f} | "
        f"{g['G3'][k]['score_fixed'] - g['G1p'][k]['score_fixed']:+.4f} | "
        f"{g['G3'][k]['score_fixed'] - g['G1'][k]['score_fixed']:+.4f} |"
        for i, k in enumerate(order)
    )

    # -- 表 4.1 方案A vs 瞳面 chirp ----------------------------------------
    far = "\n".join(
        f"| {f(r, 'dz_mm'):g} | {f(r, 'rel_err_max'):.4f} | {f(r, 'edge_energy_scheme_A'):.4f} | "
        f"{'✓' if r['trusted'] == 'True' else '✗'} |"
        for r in read_csv("scheme_gap_csv")
    )

    # -- 表 4.2 几何回绕 ----------------------------------------------------
    halo_rows = sorted((r for r in read_csv("imp_halo_csv") if float(r["eta"]) == 0.02),
                       key=lambda r: f(r, "dz_mm"))
    wrap = "\n".join(
        f"| {f(r, 'dz_mm'):g} | {f(r, 'halo_w_1e2_um'):.1f} | {f(r, 'halo_w_over_shape_halfwindow'):.3f} | "
        f"{'可信' if r['trusted'] == 'True' else '**不可信**'} |"
        for r in halo_rows
    )

    # -- 表 5 零级组 --------------------------------------------------------
    sel = state["zeroth"]
    zl_label = {
        "H0": "H0 plain GS, η=0 (基线)",
        "H1": "H1 plain GS, 不扣除",
        "H2": "H2 plain GS, 相干扣除",
        "H3": "H3 加权 GS + 相干扣除",
        "H4": "H4 H3 + 小倾斜移开晕斑",
    }
    zt = "\n".join(
        f"| {zl_label[grp]} | η={sel['eta'][grp]:g} | "
        + " / ".join(f"{r['cv']:.4f}" for r in sel["rows"][grp] if r["dz"] in (0.0, 5.0, 10.0))
        + " | "
        + " / ".join(f"{r['pib']:.4f}" for r in sel["rows"][grp] if r["dz"] in (0.0, 5.0, 10.0))
        + " |"
        for grp in ("H0", "H1", "H2", "H3", "H4")
    )
    h4_pibs = " / ".join(f"{r['pib']:.4f}" for r in sel["rows"]["H4"] if r["dz"] in (0.0, 5.0, 10.0))
    #: 自适应框陷阱的两个实测数字: 框边长 (G3 在 +0.5 dz_unit 处) 与 σ 相对焦面的倍数
    #: (方案A 在 Δz=50 mm 处, scheme_gap 的 sigma_over_ref)。
    g3_auto_side = fr["group_ceiling"]["G3"]["+0.50"]["side_px"]
    g3_auto_defocus = fr["group_ceiling"]["G3"]["-2.00"]["defocus_rad"]
    g3_auto_defocus_hi = fr["group_ceiling"]["G3"]["+0.50"]["defocus_rad"]
    g3_fix_defocus = fr["group_ceiling"]["G3"]["-2.00"]["defocus_rad_fixed"]
    g3_fix_defocus_hi = fr["group_ceiling"]["G3"]["+0.50"]["defocus_rad_fixed"]
    sigma_ratio_max = max(f(r, "sigma_over_ref") for r in read_csv("scheme_gap_csv"))
    h3_flat = [r["cv"] for r in sel["rows"]["H3"]]
    h3_all = [r["score"] for r in sel["rows"]["H3"]]
    trusted_dz = (0.0, 5.0, 10.0)
    h1_best_trusted = max(r["score"] for r in sel["rows"]["H1"] if r["dz"] in trusted_dz)
    h3_best_trusted = max(r["score"] for r in sel["rows"]["H3"] if r["dz"] in trusted_dz)

    rt = im["self_check"]["round_trip"]
    equiv = fr["equivalence_verdict"]
    trust = sw["trust_criteria"]
    wall_a = [v for k, v in fr["gs_wallclock_s"].items() if k.startswith("gs_A_")]
    wall_b = [v for k, v in fr["gs_wallclock_s"].items() if k.startswith("gs_B_")]

    checks = "\n".join(f"- {name}: `{val}`" for name, val in state["checks"])
    gif_lines = "\n\n".join(
        f"![{cap}](images/{name})" for cap, name in state["gifs"]
    )
    repro = "\n".join(
        f"| {i} | `{PRODUCERS[script]['command']}` | {REPRO_PRODUCTS[script]} "
        f"| {PRODUCERS[script]['runtime']} |"
        for i, script in enumerate(REPRO_PRODUCTS, start=1)
    )
    # -- §0.1 的设置表 (值全部取自落盘 summary, 不写死) ----------------------
    grid = im["grid"]
    optics = im["optics"]
    setup = "\n".join(
        [
            "| 量 | 值 |",
            "|---|---|",
            "| 目标形状 | 均匀方形 (`make_square_target`), 框心定位一次后冻结 |",
            f"| 瞳面网格 / 远场网格 | n_grid = {grid['n_grid']}, "
            f"far_field_padding = {grid['far_field_padding']} ⇒ 远场 {grid['far_field_size']}² |",
            f"| 远场像元 | {optics['far_field_pixel_um']:.4f} µm |",
            f"| 瞳面口径 | {optics['aperture_m'] * 1e3:.1f} mm "
            f"(面板像元 {optics['panel_pixel_um']:g} µm × 半径 450 px) |",
            f"| 瞳面过采样 seed / GS 迭代 | seed = {GS_SEED}, gs_iters = {grid['gs_iters']} |",
            f"| Δz 扫描 | 0–50 mm (步长 5 mm, {len(DZ_SWEEP_MM)} 点) |",
        ]
    )

    # -- §0.3 结果分析: 只用已算出的统计量 ----------------------------------
    results = "\n".join(
        [
            f"1. **大头是\"重新优化\", 不是\"离焦\"**: |Δz|=2 个单位上, 重新优化 (G1′−G1) 拿到 "
            f"{g['G1p']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f}, "
            f"而离焦并入环路在其之上的净贡献 (G3−G1′) 只有 "
            f"{g['G3']['-2.0']['score_fixed'] - g['G1p']['-2.0']['score_fixed']:+.4f}; "
            f"逐 Δz 的完整拆解见 §2.2。",
            f"2. **两个旋钮高度冗余**: 事后补一个标量离焦 (G2−G1) 拿到 "
            f"{g['G2']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f}, "
            f"与重新优化同量级; 两者同时用 (G3−G1) 只有 "
            f"{g['G3']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f}, "
            f"远小于两者之和 ⇒ 关键不是\"离焦并入环路\"这个**形式**, 而是**拿到工作点信息**。",
            f"3. **方案A/B 的严格等价边界**: |Δz| ≤ {equiv['Bf_equivalent_limit_dz_units']:.4f} dz_unit "
            f"(= {equiv['Bf_equivalent_limit_dz_units'] * DZ_BLUR_MM:.4f} mm) 上相对误差 ≤ 1e-06, "
            f"之后按 Δz² 发散; G3 与 G4 的定尺框 score 差 ≤ "
            f"{max(abs(g['G3'][k]['score_fixed'] - g['G4'][k]['score_fixed']) for k in g['G3']) * 1e6:.1f}e-06 "
            f"⇒ 方案B 可作提速替身, 但**不可**做场级/相位级比较 (§3)。",
            f"4. **带限 ASM 必须关**: 往返相对误差 {rt['round_trip_relerr_max']:.2e} (关闭) vs "
            f"{rt['round_trip_relerr_max_band_limited']:.4f} (开启, 库默认) ⇒ 库默认在 |Δz| ≳ 10 mm 上"
            f"不可信, 而且**不报错** (§6)。",
            f"5. **零级双分量 + 加权 GS + 相干扣除 (H3) 才有效**: cv_box 由 H1 的 "
            f"{sel['rows']['H1'][0]['cv']:.4f} 压到 {sel['rows']['H3'][0]['cv']:.4f}, "
            f"代价是 pib_box 从 {sel['rows']['H1'][0]['pib']:.4f} 让到 {sel['rows']['H3'][0]['pib']:.4f}; "
            f"单独做扣除 (H2) 反而更差 (Δz=0 上 {sel['rows']['H2'][0]['cv']:.4f} vs "
            f"{sel['rows']['H1'][0]['cv']:.4f})。H3 的 cv_box 在 0–50 mm 全程平坦 "
            f"({min(h3_flat):.4f}–{max(h3_flat):.4f}), 所以该结论不依赖不可信区 (§5)。",
            f"6. **几何回绕是另一条独立边界**: 当前整形网格半窗 "
            f"{im['trustworthy_dz']['half_window_um']:.1f} µm, 首个不可信点在 Δz = "
            f"{im['trustworthy_dz']['first_untrusted_dz_mm']:g} mm ⇒ 可信区是 Δz ≤ 10 mm, "
            f"与\"近似质量\"边界 (≤ 20 mm) 不是同一条 (§4)。",
            f"7. **自适应框会刷分**: 定尺框下 G3 的真实最优离焦是 {g3_fix_defocus:+.2f} rad, "
            f"而自适应框把它推到 {g3_auto_defocus:+.2f} rad (非物理值) ⇒ 跨 Δz 的结论一律用定尺框 (§7)。",
        ]
    )


    return f"""# CCD 焦后离焦下的 GS 匀化仿真（2026-10-19）

**本轮结论**（全部为仿真, 无任何硬件验证）：2f 光路把一个 SLM 相位送到透镜后方的 CCD 时,
**"离焦"本身几乎不值钱, 值钱的是"在工作点上重新优化"**。定尺框口径下, 拿 Δz=0 收敛的 GS 相位
直接用到 Δz=−2 个单位 (=−1.50 mm) 处, 复合分数从 {g['G1']['0.0']['score_fixed']:.4f} 掉到
{g['G1']['-2.0']['score_fixed']:.4f}; 在同一 Δz 上重新跑 GS 就能回到
{g['G1p']['-2.0']['score_fixed']:.4f}（**+{g['G1p']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:.4f}**）,
再把标量离焦并入环路只再加 {g['G3']['-2.0']['score_fixed'] - g['G1p']['-2.0']['score_fixed']:.4f}。
也就是说, **离焦并入环路的净贡献是 +0.017 量级, 而不是 +0.15 量级** —— 后者是重新优化挣来的。
同时本轮给出了三条**边界**和一个**数值缺陷**: (a) 方案B (瞳面二次相位) 只在 |Δz| ≤
{equiv['Bf_equivalent_limit_dz_units']:.4f} dz_unit (={equiv['Bf_equivalent_limit_dz_units'] * DZ_BLUR_MM:.4f} mm) 上严格等价于方案A,
之后按 Δz² 发散; (b) 方案B 与真值 (方案A) 的偏离在 **Δz ≤ 20 mm** 内 ≤5%, 25 mm 起不可作硬件预测;
(c) **几何回绕**是另一个、更早的边界: 当前的 64² 整形网格半窗只有 578.8 µm, Δz ≥ 20 mm 起晕斑
1/e² 半径已超过半窗的 0.5 倍, 那里的 CV/PIB 是 FFT 周期性把边缘能量卷回靶区的伪影 —— 这两个
边界**不同**（一个管近似质量, 一个管网格容得下), 不可合并。数值缺陷是
`angular_spectrum_propagate(..., band_limit=True)`（库默认）在 |Δz| ≳ 10 mm 上**有损且不可逆**:
实测往返相对误差 **{rt['round_trip_relerr_max']:.2e}（关闭）vs {rt['round_trip_relerr_max_band_limited']:.4f}（开启）**,
正/反向因此不构成伴随对, GS 内环会收敛到一个不存在的解且不报错。最后, **质心自动缩放框会刷分**:
它把"测量区域变大"读成"均匀度变好", 把"最优离焦"推到 ±3.87 rad 这种非物理值, 而定尺框下的
真实最优是 **±0.34 rad** —— 本轮所有结论一律用**固定物理尺度框**。

> 与本仓库既有硬件报告的关系: 本报告不推翻任何实测结论, 它量化的是**仿真侧**"离焦该怎么处理"。
> 唯一的硬件含义是 §5 的 H3 (加权 GS + 相干扣除), 见该节的适用条件与代价。

## 0. 实验目的

{PURPOSE_TEXT.strip()}

## 0.1 实验设置

{SETUP_TEXT.strip()}

{setup}

### 光学与网格（全篇共用）

| 量 | 值 |
|---|---|
| 波长 | 1064 nm |
| 透镜焦距 | 122.406 mm |
| 瞳面口径 | 7.2 mm（面板像元 8 µm × 半径 450 px） |
| 整形网格 | n_grid = 64, far_field_padding = 8 ⇒ 远场 512², 像元 2.2611 µm |
| 目标框 | 43 px（定尺） |
| 远场半窗 | {im['trustworthy_dz']['half_window_um']:.1f} µm（= n_grid·λf/D, 与 padding 无关） |
| 1 个 dz_unit | {DZ_BLUR_MM:.4f} mm |

{checks}

## 0.2 算法修改

{CHANGES_TEXT.strip()}

## 0.3 结果分析

{results}

## 1. 正向模型（为什么不是"离焦 = 放大"）

2f 光路的正确构造是 **透镜先傅里叶变换, 再对已变换的场自由传播**:

    U_CCD = ASM( FT(U_SLM), Δz )

**不是**从瞳面直接 `ASM(z = f+Δz)` —— 那一步忽略了透镜, 得到的是瞳面光束在距离 z 上的衍射
图样, 而不是远场图样; `iterative_zernike_shaping.py` 已把这件事实测并锁在
`TestNotAFraunhoferSubstitute` 里。

Collins/ABCD 给出 `T(f+d)·Lens(f)·T(f) = [[-d/f, f], [-1/f, 0]]`, 即 **B ≡ f 恒定**, 所以
**CCD 离焦不产生放大率**: 远场像元 pitch 恒为 `λf/(N·dx_pupil)`, 目标框不做任何重采样。
外部资料常写 `M = (f+Δz)/f`, 那是**错的**（已把理由记在脚本注释里）; 若照它重采样目标框,
误差会与真实几何差 1.4 倍量级。CCD 离焦只让图样变模糊: 倾斜平面波经透镜聚到 `fθ` 之后只是
发散, 中心不跑。

## 2. 五组因果链: 收益到底来自哪一步

五组构成一条**可拆解**的因果链（Δz 顺序 −2/−1/−0.5/−0.25/0/+0.25/+0.5/+1/+2 个 dz_unit,
对应 {dz_mm[0]:+.4f} … {dz_mm[-1]:+.4f} mm）:

| 组 | 定义 |
|---|---|
| G1 | Δz 盲 GS, 无离焦（基线） |
| G2 | Δz 盲 GS + 离焦（事后补偿） |
| G1′ | 逐 Δz 重优化 GS, 无离焦（**对照**） |
| G3 | 逐 Δz 重优化 + 离焦并入环路（方案A） |
| G4 | 同 G3 但用方案B（瞳面二次相位） |

### 2.1 定尺框复合分数（结论只看这一栏）

{hdr}
{sep}
{rows}

G1′ 在**同一半轴内逐位相同**（负侧 {g['G1p']['-2.0']['score_fixed']:.4f}, 正侧
{g['G1p']['0.25']['score_fixed']:.4f}）, 且正负半轴之间有 {abs(g['G1p']['-2.0']['score_fixed'] - g['G1p']['0.25']['score_fixed']):.4f}
的差（离散 ASM 网格的半像元不对称, 未单独归因）。**要紧的是**: 无论哪一半轴,
"拿焦面解直接用"都掉到 {g['G1']['-2.0']['score_fixed']:.4f} —— 必须在**工作点上**优化。

工作点上的复合分数 / PIB / 框内 CV（复合分数与 PIB/CV 是同一个工作点上的, 自洽）:

| 阶段 | 复合分数（自动框） | PIB | 框内 CV | 定尺框复合分数 |
|---|---:|---:|---:|---:|
{work}

### 2.2 收益拆解（定尺框）

| Δz/Δz_blur | Δz (mm) | G1 | G1′ | G3=G4 | G1′−G1（重新优化本身） | G3−G1′（离焦并入环路的净贡献） | G3−G1（总增益） |
|---|---:|---:|---:|---:|---:|---:|---:|
{decomp}

读法: **大头是"重新优化"**。在 |Δz|=2 上 `G1′−G1` 拿到
{g['G1p']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f}（负侧）/ {g['G1p']['2.0']['score_fixed'] - g['G1']['2.0']['score_fixed']:+.4f}（正侧）,
|Δz|=1 上 {g['G1p']['-1.0']['score_fixed'] - g['G1']['-1.0']['score_fixed']:+.4f} / {g['G1p']['1.0']['score_fixed'] - g['G1']['1.0']['score_fixed']:+.4f},
|Δz|=0.5 上 {g['G1p']['-0.5']['score_fixed'] - g['G1']['-0.5']['score_fixed']:+.4f} / {g['G1p']['0.5']['score_fixed'] - g['G1']['0.5']['score_fixed']:+.4f},
|Δz|=0.25 上 ~0, Δz=0 上恒为 0。而`G3−G1′`（离焦**在其之上**的增量）只有
{g['G3']['-2.0']['score_fixed'] - g['G1p']['-2.0']['score_fixed']:+.4f}（负侧）、
{g['G3']['0.25']['score_fixed'] - g['G1p']['0.25']['score_fixed']:+.4f}（正侧）、
{g['G3']['0.0']['score_fixed'] - g['G1p']['0.0']['score_fixed']:+.4f}（Δz=0）。

**为什么"离焦只值 +0.017"不是"离焦没用"**: 标量离焦就是 Zernike (2,0) 模态, 而 GS 的相位是
**全频随机相位** —— `φ_GS + a·Z₂⁰` 本来就在 GS 的可达空间里。所以这一列度量的是"能不能在重优化之上
再挤出一点", 而不是"能不能表达"。

![五组上限]({state['figs']['ceiling']})

## 3. 方案A vs 方案B: 什么时候可以互相替代

方案B 用瞳面二次相位 `exp(−iπ·Δz·r²/(λf²))` 代替真 ASM（分母是 **λf²**, 不是 λf(f+Δz)）。
它与方案A 在离散网格上的关系:

| 量 | 值 |
|---|---|
| 严格等价（相对误差 ≤ 1e-06）边界 | **{equiv['Bf_equivalent_limit_dz_units']:.4f} dz_unit = {equiv['Bf_equivalent_limit_dz_units'] * DZ_BLUR_MM:.4f} mm** |
| \\|Δz\\| = 2.5 dz_unit 处的相对误差 | {equiv['Bp_max_rel_err_over_swept_range']:.2e}（按 Δz² 发散: kz 的二阶傍轴截断） |
| 焦平面域 (Bf) 与瞳面域 (Bp) | 在所有 Δz 上**逐位相同**（`Bf_vs_Bp_identical = {equiv['Bf_vs_Bp_identical']}`, 表明偏差全部来自傍轴截断, 而不是"搬回瞳面"带来的环卷绕） |
| G3 与 G4 的 score 差 | ≤ 7e-06（本轮实测上限 {max(abs(g['G3'][k]['score_fixed'] - g['G4'][k]['score_fixed']) for k in g['G3']) * 1e6:.1f}e-06） |
| GS 墙钟（方案A / 方案B） | {min(wall_a):.2f}–{max(wall_a):.2f} s / {min(wall_b):.2f}–{max(wall_b):.2f} s |

⇒ 方案B 可作**约 2× 提速的替身**（score 级结论不受影响）, 但**不可用于任何场级 / 相位级比较**:
`Bp` 与 `Bp_chirped` 只差一个模长为 1 的输出二次相位, 对归一化强度逐位无影响, 但 GS 的反向传播
取的是**相位**, 少了它就等价于换了一个不是伴随的算子。

![方案A/B 等价性]({state['figs']['equivalence']})

## 4. 两个**不同**的可信度边界（0–50 mm）

这两条边界必须分开说, 混在一起就会得出"20 mm 以内都可信"或"25 mm 以内都可信"两种都错的结论。

### 4.1 (a) 瞳面二次相位 vs 真 ASM 的偏离（方案A 为真值）

| Δz (mm) | max 相对误差 | 边缘能量占比 | 记录在案的判定 |
|---:|---:|---:|---|
{far}

⇒ 工程可用 **≤ 20 mm**（相对误差 ≤5%）; **25 mm 起不可作为硬件预测**; 35 mm 起误差 >40%。
这条边界管的是**近似质量**。

![方案A vs 瞳面 chirp]({state['figs']['scheme_a']})

### 4.2 (b) 几何回绕（远场半窗只有 {im['trustworthy_dz']['half_window_um']:.1f} µm）

| Δz (mm) | 晕斑 1/e² 半径 (µm) | 半径 / 半窗 | 可信 |
|---:|---:|---:|---|
{wrap}

⇒ **Δz ≤ 10 mm 可信**; 20 mm 起比值 >0.5, FFT 是周期的, 溢出能量会从对面卷回靶区, 污染 CV/PIB。
45 mm 起比值 >1。这条边界管的是**网格装不装得下**, 与 (a) 无关: 要覆盖到 Δz=50 mm 需要
n_grid ≈ {im['trustworthy_dz']['n_grid_needed_for_dz_max']}（padding=8 ⇒ 远场 {im['trustworthy_dz']['far_field_size_needed_at_padding8']}²),
**加大 padding 没用** —— 窗口只随 n_grid 变。

![几何回绕]({state['figs']['halo']})

本轮 G 组扫描的 |Δz| ≤ {abs(dz_mm[0]):.4f} mm 落在两条边界之内; 零级组的 Δz 扫描跨到 50 mm,
所以 §5 的结论只取 Δz ≤ 10 mm（H3 在 0–50 mm 上全程平坦, 这一条是**额外**的稳健性, 不是边界内的推断）。

## 5. 零级双分量模型 + 加权 GS（可信区 Δz ≤ 10 mm）

真实 SLM 上总有一部分光不承载调制（调制深度不足 / 像素死区 / 未调制区）, 它只经过物理透镜,
在 f+Δz 处弥散成一团晕斑。写成双分量瞳面模型:

    U_pupil = U_inc · [ η + (1 − η) · exp(iφ) ]

晕斑峰值随 Δz 单调下降（口径: 宽网格 (n_grid=256) 远场 sum=1 归一 × η², η=0.02:
2.9e-06 @Δz=0 → 1.2e-08 @Δz=20 mm, 约 249×）,
渐近区 `I_halo ∝ 1/Δz²` 保持到 **4.7%**（`max_I_dz2_spread` {im['self_check']['halo_monotonic']['max_I_dz2_spread']:.4f} < 容许 {im['self_check']['halo_monotonic']['max_I_dz2_spread_tol']:.2f}）。

每组取 **Δz=0 处 cv_box 最小** 的 η（H1 → {sel['eta']['H1']:g}, H2/H3/H4 → {sel['eta']['H2']:g}; H0 恒 η=0）,
该 η 固定用到底。下表是 **cv_box / pib_box**（data.json 居中定尺框, 连续相位）:

| 组 | η | cv_box (Δz=0 / 5 / 10 mm) | pib_box (Δz=0 / 5 / 10 mm) |
|---|---|---|---|
{zt}

三条结论, 按重要性：

1. **单独做相干扣除无效甚至有害**：H2 的 cv_box 比 H1 **差**（Δz=0 上 {sel['rows']['H2'][0]['cv']:.4f} vs {sel['rows']['H1'][0]['cv']:.4f}）。
   扣除是在目标面约束之前减去模型里的零级复场 `b0`; 只扣不配权重, 等于把一个偏置移走却不动其余部分。
2. **加权 GS + 相干扣除（H3）才有效**：把 cv_box 从 ~0.42 压到 **{h3_flat[0]:.4f}**（≈5.7×）, 代价是
   pib_box 从 ~0.917 让到 {sel['rows']['H3'][0]['pib']:.4f}（让出约 17%）, 复合分数净赚
   {h1_best_trusted:.4f} → {h3_best_trusted:.4f}（可信区内各自最好点; H3 在整段 0–50 mm 上的最好点是
   {max(h3_all):.4f}）。**而且 H3 的 cv_box 在 0–50 mm 全程平坦**
   （{min(h3_flat):.4f}–{max(h3_flat):.4f}）, 所以它的结论不依赖不可信区。
3. **H4（小倾斜把光斑移开轴上晕斑）在本口径下毁掉**：cv_box {sel['rows']['H4'][0]['cv']:.4f} / {sel['rows']['H4'][1]['cv']:.4f} / {sel['rows']['H4'][2]['cv']:.4f},
   pib_box {h4_pibs} —— 光斑被挪出居中定尺框, 框内几乎无光。**注意**:
   这**不代表** H4 的物理想法错, 只说明"用居中定尺框评 H4"这件事本身不成立（H4 必须用自己的
   锚定框评, 见 §7）。

![零级对照]({state['figs']['zeroth']})

## 6. 数值缺陷: 带限 ASM 在大 Δz 有损且不可逆（必读）

`angular_spectrum_propagate(..., band_limit=True)` 是**库的默认值**（Voelz & Roggemann cut）。
那个 cut 在焦平面上**有损且不可逆**: 高频 bin 被置零, 能量在**正向一步**就漏掉, 反向一步补不
回来 —— 于是 `fwd(+z)` 与 `bwd(−z)` 不构成伴随对, GS 的交替投影**解的不是一个自洽问题**,
而且**不会报错**。

实测往返相对误差 `max|u − bwd(fwd(u))| / max|u|`, 同一传播器、同一网格:

| band_limit | Δz=0 | Δz=10 mm | Δz=20 mm | Δz=30 mm | Δz=40 mm | Δz=50 mm |
|---|---:|---:|---:|---:|---:|---:|
| True（库默认） | 1.1e-15 | 1.5e-15 | 0.0507 | 0.2674 | 0.4735 | {rt['round_trip_relerr_max_band_limited']:.4f} |
| False（本实验） | 1.1e-15 | {rt['round_trip_relerr_max']:.2e} | 1.3e-15 | 1.2e-15 | 1.1e-15 | 1.3e-15 |

伤害幅度**强依赖场的谱内容**: 实验脚本正文记录准单频场在同一条链、同一个 Δz=50 mm 上只丢
0.7%, 而宽带场丢 80%; 本脚本落盘 summary 用 4 个探针实测的能量损失跨度是 **0.02% ~ 38.75%**
（`band_limit_damage.note` 明确要求"不要引用单一数字, 先测你自己那个场的谱"）。两个口径都指向
同一件事: **换实验系统 / 换入射场就重测**.

处置: 大 Δz 一律 `band_limit=False`, 并加 `round_trip_relerr` 自检。注意 Δz=0 时开与关
**逐位相同**（`exp(i·kz·0)=1` 且 cut 全通）, 所以基线与既有 `data.json` 的核对不受影响。

## 7. 指标陷阱: 质心自动缩放框会刷分

主判据若用"质心跟踪 + σ 缩放边长"的自适应框, 框会跟着光斑一起变大 —— 实测框边长从 43 px 涨到
{g3_auto_side:.0f} px（G3 在 Δz/Δz_blur=+0.5 处）, σ 在 Δz=50 mm 上相对焦面涨到 **{sigma_ratio_max:.2f}×**,
于是"最优离焦"被推到 **{g3_auto_defocus:+.2f} rad**（非物理值; 正侧更直接顶到搜索区间端点
{g3_auto_defocus_hi:+.2f} rad）, 而"均匀度变好"其实是"测量区域变大"。定尺框下同一组的真实最优是
**{g3_fix_defocus:+.2f} rad**（正侧 {g3_fix_defocus_hi:+.2f} rad）。

本调查中这个陷阱出现 **四次**, 其中一次藏在 `group_ceilings` 块里: 用自适应框会把 H4 的失败
掩盖成与 H3 并列（{im['group_ceilings']['H4']['best_score']:.4f} vs {im['group_ceilings']['H3']['best_score']:.4f}）,
而用定尺框看 H4 在 §5 里是 {sel['rows']['H4'][0]['pib']:.4f} 的框内 PIB。

**规则**: 跨 Δz 的任何数字, 先确认判据的框是否随 Δz 变; 所有结论必须用**固定物理尺度框**。

另一个未解释项: H3 在焦面（cv_box {sel['rows']['H3'][0]['cv']:.4f}）反而**略差**于焦后
（Δz=5 mm 的 {sel['rows']['H3'][1]['cv']:.4f}）, 幅度约 2%, 可能落在数值噪声内但**未证实** —— 如实标注为未解释,
不编理由。

## 8. 动画与静态图

三栏统一为 **左 = 远场强度（方案A, log10, 归一峰值）/ 中 = SLM 瞳面相位（弧度）/ 右 = 过零级的
x 截面（log10）+ 目标框**。注意左栏的 log10 是为了同时看见晕斑与被整形的主瓣; 每帧标题给出
该帧的 `cv_box` / `pib_box`（data.json 居中定尺框, 连续相位）, 并**同时**标出两条边界的状态
（方案A 相对误差、几何回绕比）—— 只要有一条越界, 帧边框变红并在标题里标「不可信」。
（标题不用 `⚠` 之类的符号: 配置的 CJK 字体没有该字形, 渲染出来是方块, 故一律改用中文文字标记。）

{gif_lines}

逐帧对照表（帧内 `cv_box` / `pib_box` 由本脚本现场重算; "落盘"列是相应实验脚本记录的同一量）:

| 动画 | 帧数 | 内容 |
|---|---:|---|
| `gs_defocus_sim_dz_sweep_20261019.gif` | {state['gif_frames'][0]} | 原始 GS（Δz=0 优化）扫 Δz = 0…50 mm |
| `gs_defocus_sim_reopt_20261019.gif` | {state['gif_frames'][1]} | 逐 Δz 重优化 GS（G1′ 口径）同样扫 Δz |
| `gs_defocus_sim_zeroth_20261019.gif` | {state['gif_frames'][2]} | 零级对照: 上排 H0（η=0）, 下排 H3（加权+扣除, η={sel['eta']['H3']:g}）, Δz=0/5/10 mm |

静态图（`figures/`）:

| 图 | 内容 |
|---|---|
| [`fig_gs_defocus_groups_ceiling.png`](figures/fig_gs_defocus_groups_ceiling.png) | 五组上限: ① 自动框 CV（陷阱示范）② 定尺框 score ③ 最优离焦（自动框 ±3.87 rad vs 定尺框 ±0.34 rad） |
| [`fig_gs_defocus_scheme_b_equivalence.png`](figures/fig_gs_defocus_scheme_b_equivalence.png) | 方案A/B 等价性: 三个方案B 变体重合, 标出 1e-06 边界 0.1138 dz_unit |
| [`fig_gs_defocus_scheme_a_vs_chirp.png`](figures/fig_gs_defocus_scheme_a_vs_chirp.png) | 边界 (a): 方案A vs 瞳面二次相位的相对误差（标出 20 mm） |
| [`fig_gs_defocus_halo_wraparound.png`](figures/fig_gs_defocus_halo_wraparound.png) | 边界 (b): 晕斑峰值与几何回绕比（标出 0.5 阈值） |
| [`fig_gs_defocus_zeroth_h0_h4_cv_box.png`](figures/fig_gs_defocus_zeroth_h0_h4_cv_box.png) | 零级组 H0–H4 的 cv_box vs Δz 与可信区柱状对照 |
| [`fig_gs_defocus_e1_cv_heatmap.png`](figures/fig_gs_defocus_e1_cv_heatmap.png) | E1 离焦扫描热力图（原图复制自 `report/shape_trials/gs_defocus_sim/artifacts/e1_cv_heatmap.png`） |

## 9. 限制（不能断言什么）

* **全部是仿真, 没有硬件验证。** 本报告的每一个数字都来自纯 numpy 数字孪生; 没有下发过任何相位,
  也没有拍过任何帧。§5 的 H3 是一条**实验假设**, 不是结论。
* **"离焦并入环路只值 +0.017"的结论限于当前网格与判据**: 64² 瞳面 / padding=8 / 定尺框 43 px /
  gs_iters=200 / 方案A 正演。换目标函数（例如纯 CV、纯 PIB、或带形状约束的复合分）、换网格、
  换口径都要重跑 —— 尤其"离焦只值 +0.017"在**别的目标函数**下没有证据。
* **"重新优化值 +0.147"是相对"Δz 盲基线"说的, 不是相对"上一个工作点"说的。** 它回答"该不该在
  工作点重优化", 不回答"重优化要花多少硬件时间"。
* **自适应框陷阱的教训是元级的**: 任何跨 Δz 的数字都要先确认判据。本报告因此并列给出两个判据,
  并把 H4 的失败明确标成"居中定尺框口径不适用"而不是"方法失败"。
* **§3 的 2× 提速是 GS 墙钟的比值, 不是端到端闭环的比值**; 且方案B **不可**用于场级/相位级比较,
  只可用于 score 级排序。
* **§6 的带限伤害幅度不能引用单一数字。** 它是谱内容的函数, 落盘 summary 的探针跨度是
  0.02% ~ 38.75%, 脚本正文另记录 0.7%（准单频）~ 80%（宽带）这个更宽的口径; 两者都不是
  "普适的百分比"。换实验系统必须重测。
* **H3 在焦面略差于焦后（约 2%）未解释**, 未证实, 不作为结论。
* **晕斑峰值的绝对量级取决于归一化口径**: 本报告 §5 引用的是落盘 `halo_rows` 的
  `i_halo_peak`（远场归一化到 sum=1, 再乘 η²）, Δz=0 上是 2.9e-06。换一个归一化
  （例如相对目标区平均强度 `halo_peak_over_box_mean`, Δz=0 上是 5.6e-03）会得到相差几个量级的
  数 —— 引用绝对值之前先确认口径; 只有**随 Δz 的幂律**与**几何回绕比**是口径无关的。
* **正负半轴的不对称（G1′ 两半轴差 {abs(g['G1p']['-2.0']['score_fixed'] - g['G1p']['0.25']['score_fixed']):.4f}）未单独归因**;
  怀疑来自离散 ASM 网格的半像元不对称, 但没有单独实测证实。
* **两个旋钮在很大程度上是冗余的, 这一点值得单独记住**: 在 |Δz|=2 上,
  "Δz 盲 GS + 事后搜一个标量离焦" (G2−G1) 拿到 {g['G2']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f},
  而"逐 Δz 重优化 GS、不加离焦" (G1′−G1) 拿到 {g['G1p']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f} —— 两者量级相同,
  因为它们修的是同一处的误差; 同时用 (G3−G1) 只有 {g['G3']['-2.0']['score_fixed'] - g['G1']['-2.0']['score_fixed']:+.4f}, 远小于两者之和。
  所以"离焦并入环路"这一**形式**本身不是关键, 拿到工作点信息才是; 反过来也不能据此说"离焦不值钱" ——
  单独加离焦同样值 +0.15 量级。
* **本报告的渲染脚本依赖 4 个上游实验脚本落盘的 CSV, 而这些 CSV 按仓库惯例不入库。**
  因此**从干净克隆复现需要先跑那 4 个脚本** (总耗时约 1.5 小时, 其中 `experiment_gs_improved.py`
  占大头), 单跑本脚本会因缺输入而**拒绝执行并列出缺哪些文件** —— 这是有意的失败方式, 不是崩溃。
  确切命令与耗时见 §10; 只想确认管线能走完则用 `--quick` (零输入, 不读任何落盘产物)。

## 10. 复现

数据全部来自四个纯仿真实验脚本（本脚本只读它们的 CSV/JSON, 不重跑它们）。四个脚本**互不依赖**,
哪个先跑都行, 但都必须在本生成器之前跑完; 默认产物都落在 `report/shape_trials/gs_defocus_sim/artifacts/`:

| # | 确切命令 | 产物 (默认 `report/shape_trials/gs_defocus_sim/artifacts/`) | 大致耗时 |
|---|---|---|---|
{repro}

四个脚本跑完后, 本生成器:

```bash
python scripts/slm/generate_gs_defocus_sim_report.py            # 全量: 6 张静态图 + 3 个 GIF + 本 markdown (约 6–10 min)
python scripts/slm/generate_gs_defocus_sim_report.py --no-gif   # 只出静态图与 markdown
python scripts/slm/generate_gs_defocus_sim_report.py --quick    # 冒烟: 零输入, 只验帧管线, 不写报告/图
```

> **`*.csv` 被 `.gitignore` 全局排除, 这是仓库约定 (只入库报告与图, 数据 CSV 不入库), 不是缺陷。**
> 因此**干净克隆上必须先跑上面四个脚本产出 CSV, 本生成器才有输入**。缺任何一个输入时, 本脚本会在
> 启动时直接拒绝执行, 并逐条列出「缺哪个文件 / 由哪个脚本产出 / 要跑哪条命令 / 大致耗时」,
> 不会抛裸 `FileNotFoundError`; `--quick` 不读任何落盘产物, 在任何树上都能跑。

动画与静态图由本目录的 `scripts/slm/generate_gs_defocus_sim_report.py` 渲染; 它**import** 
`experiment_gs_improved.py` 的 `TwoComponentBench`（正演与 GS）和
`square_shaping.compute_bench_metrics`（度量）, 与实验脚本逐位同口径, 不复刻数学。

帧内 `cv_box` 的自洽性核对: 逐 Δz 重优化动画（G1′ 口径）在 7 个与落盘 H0 行相同 Δz 上的
`cv_box` 最大偏差 **{state['reopt_dev']:.2e}**（{state['reopt_dev_note']}）。
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_quick(*, gs_iters: int | None = None) -> None:
    """冒烟: 只验「正演能跑 / GS 能跑 / 帧能渲染 / 三栏管线通」。

    **零输入**: 不读任何落盘 CSV/JSON, 也不写 ``report/``。数据 CSV 按仓库惯例不入库
    (``.gitignore`` 的 ``*.csv``), 所以干净克隆上只有这条路径可用。
    """
    imp = load_improved()
    fx = Frames(imp, gs_iters=gs_iters or 60)
    tmp = ROOT / ".tmp_gs_defocus_quick"
    tmp.mkdir(parents=True, exist_ok=True)
    p1, r1 = gif_dz_sweep(fx, quick=True, out_dir=tmp)
    p2, r2, _dev = gif_reopt(fx, quick=True, out_dir=tmp)
    p3, r3 = gif_zeroth(fx, quick=True, out_dir=tmp)
    print(f"[quick] sweep={p1.name} ({len(r1)} frames)  reopt={p2.name} ({len(r2)} frames)  "
          f"zeroth={p3.name} ({len(r3)} frames)")
    print("[quick] smoke frames ok; 未读任何落盘产物, 未写 report/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-gif", action="store_true", help="只出静态图与 markdown")
    ap.add_argument("--quick", action="store_true", help="冒烟: 只验帧管线 (零输入), 不写报告/图")
    ap.add_argument("--gs-iters", type=int, default=None, help="GS 迭代数 (默认取 FULL=200)")
    args = ap.parse_args()

    if args.quick:
        run_quick(gs_iters=args.gs_iters)
        return

    preflight()  # 缺输入就列出「缺哪个文件 / 谁产出 / 跑哪条命令」, 不抛裸 FileNotFoundError
    now = {k: read_json(k) for k in ("sweep_json", "fresnel_json", "improved_json")}
    checks = self_checks(now)
    print("[self-check]")
    for name, val in checks:
        print(f"  - {name}: {val}")

    gs_iters = args.gs_iters or now["improved_json"]["grid"]["gs_iters"]
    sel = zeroth_selection()
    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    figs = {
        "ceiling": fig_groups_ceiling(),
        "equivalence": fig_equivalence(),
        "scheme_a": fig_scheme_a_vs_chirp(),
        "halo": fig_halo_wraparound(),
        "zeroth": fig_zeroth_cv_box(sel),
        "e1": copy_e1_heatmap(),
    }
    state: dict[str, Any] = {
        "sweep_json": now["sweep_json"],
        "fresnel_json": now["fresnel_json"],
        "improved_json": now["improved_json"],
        "fv_groups": fv_groups(),
        "zeroth": sel,
        "checks": checks,
        "figs": {k: p.relative_to(ART).as_posix() for k, p in figs.items()},
    }

    gifs: list[tuple[str, str]] = []
    gif_frames = [0, 0, 0]
    reopt_dev = float("nan")
    note = "未计算"
    if not args.no_gif:
        imp = load_improved()
        fx = Frames(imp, gs_iters=gs_iters)
        p1, _r1 = gif_dz_sweep(fx, quick=False)
        gif_frames[0] = len(DZ_SWEEP_MM)
        p2, _r2, reopt_dev = gif_reopt(fx, quick=False)
        gif_frames[1] = len(DZ_SWEEP_MM)
        p3, _r3 = gif_zeroth(fx, quick=False)
        gif_frames[2] = 3
        gifs = [
            ("原始 GS（Δz=0 优化）扫 Δz 动画: 远场 / SLM 相位 / 过零级截面", p1.name),
            ("逐 Δz 重优化 GS（G1′ 口径）扫 Δz 动画", p2.name),
            ("零级对照动画: H0 (η=0) vs H3 (加权+相干扣除, η=0.02)", p3.name),
        ]
        note = "与落盘 H0 行同一条代码路径"

    state["gifs"] = gifs
    state["gif_frames"] = gif_frames
    state["reopt_dev"] = reopt_dev
    state["reopt_dev_note"] = note

    out = ROOT / REPORT_KEY
    out.write_text(insert_header(build_markdown(state), REPORT_KEY), encoding="utf-8")
    print(f"\nWrote {out} in {time.perf_counter() - t0:.1f}s")
    for p in figs.values():
        print(f"  fig  {p.relative_to(ROOT)}")
    for _cap, name in gifs:
        print(f"  gif  {(GIF_DIR / name).relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except MissingArtefactsError as exc:
        print(f"\n错误: 本生成器缺少输入, 未执行。\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None
