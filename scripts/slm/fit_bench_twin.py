"""从 slm-model-in-loop ``--debug`` 落盘的 (相位 → 帧) 探针对**反演 2f 实验系统数字孪生**。

补齐 ``docs/slm/gsnet_transfer_hw_runbook.md`` §7 缺的那一步：runbook 让操作员
「先反演孪生、再让网络用它」，``gsnet_transfer.py`` / ``export_gsnet_phase.py``
的 ``--twin-config`` 能**消费** ``bench_twin_calibrate.save_fit`` 的 JSON，但仓库
里此前没有任何脚本**产出**它。本脚本就是那个产出者：读探针目录 → 构造与数据
几何一致、误差模式按调用方声明的 ``BenchTwinConfig`` → 调
``ao_shaping.tools.slm.bench_twin_calibrate.fit_twin`` → ``save_fit`` 写 JSON →
打印中文摘要。

**本脚本完全离线**：只读文件，不打开任何硬件。

数据从哪来
----------
``slm-model-in-loop --debug`` 的运行目录（默认 ``data/debug/slm_model_in_loop/<ts>``）::

    probes/<i>.npz        每条探针: phase (瞳面网格命令相位, rad)
                                   + model_grid (相机帧映射到远场模型网格) ← 拟合的“实测画面”
    source_amplitude.npy  标定出来的瞳面照明振幅 (缺失 = 该运行不是 --debug 跑的)

要反演哪些模式（``fit_twin`` 的约定）
-------------------------------------
``fit_twin`` 反演的 Zernike 模式 = 配置里 ``slm_zernike`` / ``lens_zernike``
**已经存在键**的那些（哪怕值为 0.0）。所以本脚本按 ``--fit-modes`` /
``--fit-lens-modes``（Noll 序号）把这些键以 0.0 建好，让拟合去确定它们的值；
不给这两个开关 = 不拟合对应族的像差。粗参数（LUT Γ、满灰度相位、相机离焦）
与光路倾斜是 ``fit_twin`` 两阶段里**始终**反演的量，无需开关。
⚠️ 灰度→振幅耦合**不从探针拟合**：model-in-loop 的探针命令的相位几乎相同，
耦合在该数据上不可辨识。耦合是用 ``scripts/slm/fit_slm_amplitude_coupling.py``
在灰度响应 CSV 上**单独测出来**的；``--coupling-config`` 指向它写出的
``save_fit`` 形状 JSON，把其中的耦合参数**合并**进本拟合（摘要里会注明耦合
来自测量而非本次拟合，并在合并后**重评一次损失**）。

几何契约（下游 ``gsnet_transfer._twin_from_fit`` 会核验，不符即报错）
--------------------------------------------------------------------
* 孪生的 ``region`` ＝ 探针相位的边长（瞳面网格）；
* 孪生 ``far_field`` 的输出边长（中心裁剪 ``crop_side``）＝ 目标帧
  （``model_grid``）的边长 —— 本脚本据此设
  ``crop_px = model_grid 边长``、``far_field_padding = model_grid 边长 // region``。
  模型网格边长必须 ≥ region 且是 region 的**整数倍**，否则孪生远场与实测帧
  逐像素不可比，拟合出的参数没有物理意义 —— 本脚本会**明确报错**而不是
  静默继续。

用法
----
::

    python scripts/slm/fit_bench_twin.py \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --fit-modes 4,5,6 --fit-lens-modes 4,5

    # 合并单独测出来的灰度→振幅耦合:
    python scripts/slm/fit_bench_twin.py \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --coupling-config data/debug/slm_amplitude_coupling/<ts>/twin_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIBS = ROOT / "libs"
for _p in (SRC, LIBS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ao_shaping.drivers.sim.bench_twin import (  # noqa: E402
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ao_shaping.tools.slm.bench_twin_calibrate import (  # noqa: E402
    TwinFitResult,
    TwinSample,
    fit_twin,
    save_fit,
    twin_loss,
)
from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: E402
    get_zernike_name,
    noll_to_nm,
)

# 器件在 1064 nm 的 2π 灰度数由驱动从 EEPROM 查询得到 (993); 1023 只是 10-bit
# 满量程, **不是**相位周期数 —— 孪生的灰度域周期必须用前者。
DEFAULT_LEVELS = 993


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description=(
            "从 slm-model-in-loop --debug 的 (相位→帧) 探针对反演 2f 实验系统"
            "数字孪生, 写出 save_fit 形状的 JSON (供 --twin-config 消费)。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from-dir", type=Path, required=True,
                   help="slm-model-in-loop --debug 产出的运行目录 "
                        "(含 probes/*.npz 与 source_amplitude.npy)")
    p.add_argument("--limit", type=int, default=None,
                   help="最多用多少对探针 (默认: 全部)")
    p.add_argument("--fit-modes", default=None,
                   help="要反演的**面板** Zernike 模式 (Noll 序号, 逗号分隔, "
                        "如 '4,5,6')。不写 = 不拟合面板像差")
    p.add_argument("--fit-lens-modes", default=None,
                   help="要反演的**透镜** Zernike 模式 (Noll 序号, 逗号分隔)。"
                        "不写 = 不拟合透镜像差")
    p.add_argument("--region", type=int, default=None,
                   help="瞳面网格边长 (默认: 从探针相位边长推断)")
    p.add_argument("--far-field-padding", type=int, default=None,
                   help="远场补零因子 (默认: 远场帧边长 // region; 必须使 "
                        "region × padding 恰等于远场帧边长, 否则网格标度不可比)")
    p.add_argument("--region-span-px", type=float, default=900.0,
                   help="瞳面网格覆盖的 SLM 面板边长 (px, 默认 900)")
    p.add_argument("--beam-w0-px", type=float, default=250.0,
                   help="瞳面入射高斯光束的 1/e² 强度半径 (面板 px, 默认 250)")
    p.add_argument("--levels", type=int, default=DEFAULT_LEVELS,
                   help=f"器件的 2π 灰度数 (1064 nm 下 {DEFAULT_LEVELS}; "
                        f"不是 10-bit 满量程 1023), 默认 {DEFAULT_LEVELS}")
    p.add_argument("--coupling-config", type=Path, default=None,
                   help="可选: 另一份 save_fit 形状 JSON "
                        "(scripts/slm/fit_slm_amplitude_coupling.py 的产物), "
                        "把其中**测出来的**灰度→振幅耦合合并进拟合结果")
    p.add_argument("--maxiter", type=int, default=12,
                   help="Stage-1 differential_evolution 代数上限 (默认 12)")
    p.add_argument("--popsize", type=int, default=8,
                   help="Stage-1 种群规模乘数 (默认 8)")
    p.add_argument("--maxiter-fine", type=int, default=60,
                   help="Stage-2 L-BFGS-B 迭代上限 (默认 60)")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子 (DE 初始化; 同 seed 逐位可复现)")
    p.add_argument("--out", type=Path, default=None,
                   help="输出 JSON (默认 data/debug/gsnet_bench_twin_fit/"
                        "<时间戳>/twin_fit.json)")
    p.add_argument("--verbose", action="store_true",
                   help="打开 fit_twin 的逐阶段日志")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        p.error("--limit 必须 >= 1")
    if args.levels < 1:
        p.error("--levels 必须 >= 1")
    return args


def parse_noll_modes(spec: str | None, flag: str) -> list[int]:
    """解析 ``--fit-modes`` / ``--fit-lens-modes`` 的 Noll 序号列表。

    Raises:
        SystemExit: 格式非法或序号 < 1。
    """
    if spec is None:
        return []
    try:
        modes = [int(tok) for tok in spec.replace("，", ",").split(",") if tok.strip()]
    except ValueError as exc:
        sys.exit(f"错误: {flag} 不是逗号分隔的 Noll 序号 (如 '4,5,6'): {exc}")
    if not modes:
        sys.exit(f"错误: {flag} 给了空列表; 不想要该族像差就不要写这个开关")
    if any(m < 1 for m in modes):
        sys.exit(f"错误: {flag} 的 Noll 序号必须 >= 1, 收到 {modes}")
    return sorted(set(modes))


def mode_n_max(modes: list[int]) -> int:
    """Noll 序号列表所需的最大径向阶数 n (canonical ``noll_to_nm``, 禁止自写查表)。"""
    return max(noll_to_nm(j)[0] for j in modes) if modes else 0


def describe_mode(j: int) -> str:
    """Noll 序号 → 'Noll j (n, m, 名称)' 便于操作员核对。"""
    n, m = noll_to_nm(j)
    return f"Noll {j} (n={n}, m={m}, {get_zernike_name(n, m)})"


def load_probe_pairs(
    run_dir: Path, limit: int | None
) -> tuple[list[TwinSample], int, int, np.ndarray | None]:
    """读运行目录里的 (命令相位, model_grid 帧) 探针对。

    Returns:
        ``(samples, region, model_side, source_amplitude)``。

    Raises:
        SystemExit: 目录/文件缺失、没有可用记录、或网格几何自相矛盾。
    """
    if not run_dir.is_dir():
        sys.exit(f"错误: 运行目录不存在: {run_dir}")
    probes_dir = run_dir / "probes"
    if not probes_dir.is_dir():
        sys.exit(
            f"错误: {run_dir} 下没有 probes/ 目录 —— --from-dir 要指向 "
            f"slm-model-in-loop --debug 的运行目录 (data/debug/slm_model_in_loop/<ts>)"
        )
    amplitude_path = run_dir / "source_amplitude.npy"
    if not amplitude_path.exists():
        sys.exit(
            f"错误: 缺少 {amplitude_path} —— 该运行不是带 --debug 跑的 "
            "(或中途崩溃): 没有标定的瞳面照明振幅, 无法确认探针几何"
        )
    amplitude = np.load(amplitude_path, allow_pickle=False)

    samples: list[TwinSample] = []
    region: int | None = None
    model_side: int | None = None
    for path in sorted(probes_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "phase" not in data or "model_grid" not in data:
                continue
            phase = np.asarray(data["phase"], dtype=np.float64)
            frame = np.asarray(data["model_grid"], dtype=np.float64)
        if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
            sys.exit(f"错误: {path} 的 phase 不是方形 2-D 瞳面网格: {phase.shape}")
        if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
            sys.exit(f"错误: {path} 的 model_grid 不是方形 2-D 远场网格: {frame.shape}")
        if region is None:
            region = int(phase.shape[0])
            model_side = int(frame.shape[0])
        elif phase.shape != (region, region):
            sys.exit(
                f"错误: {path} 的 phase 边长 {phase.shape[0]} != 首条探针的 "
                f"{region}: 一次运行里瞳面网格必须一致"
            )
        elif frame.shape != (model_side, model_side):
            sys.exit(
                f"错误: {path} 的 model_grid 边长 {frame.shape[0]} != 首条探针的 "
                f"{model_side}: 远场模型网格必须一致"
            )
        samples.append(TwinSample(phase_rad=phase, image=frame, label=path.stem))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        sys.exit(
            f"错误: {probes_dir} 下没有任何同时带 phase 与 model_grid 的 .npz 记录"
        )
    assert region is not None and model_side is not None
    return samples, region, model_side, amplitude


def check_geometry(
    region: int, model_side: int, region_override: int | None,
    padding_override: int | None,
) -> None:
    """核验并声明几何契约; 不符时明确报错 (而不是让拟合静默产出无意义的参数)。

    Raises:
        SystemExit: 模型网格边长不是 region 的整数倍 (且 ≥ region), 或显式
            ``--region`` / ``--far-field-padding`` 与数据矛盾。
    """
    if region_override is not None and region_override != region:
        sys.exit(
            f"错误: --region {region_override} != 探针相位的边长 {region}: "
            f"孪生的 region 必须等于数据实际的瞳面网格 (下游 --twin-config 也按此核验)"
        )
    if model_side < region:
        sys.exit(
            f"错误: 远场模型网格边长 {model_side} < 瞳面 region {region}: "
            f"model_grid 与相位网格的对应关系不对, 请确认 --from-dir 是 "
            f"model-in-loop --debug 的产物"
        )
    if model_side % region != 0:
        sys.exit(
            f"错误: 远场模型网格边长 {model_side} 不是瞳面 region {region} 的整数倍: "
            f"孪生的 far_field 与实测帧逐像素不可比, 拟合会静默产出无意义的参数。"
            f" 请用与这批数据同一 region / far_field_padding 的运行, 或先把帧重采样到"
            f" region × 整数补零因子 的网格。"
        )
    computed = model_side // region
    if padding_override is not None and padding_override != computed:
        sys.exit(
            f"错误: --far-field-padding {padding_override} 与数据不符: 这批 model_grid "
            f"的网格边长 = region × {computed} = {model_side}; 补零因子改变远场的 "
            f"像素标度 (p_fft = λf / (N_pad·pitch)), 换 padding 后仿真与实测不再是 "
            f"同一张网格。默认值就是 {computed}。"
        )


def build_initial_config(
    args: argparse.Namespace, region: int, model_side: int
) -> BenchTwinConfig:
    """按数据几何 + 声明的待反演模式构造初始配置。

    ``TwinErrorModel`` 的 Zernike 字典里**写入哪些 Noll 键**就是「要反演哪些
    模式」(``fit_twin`` 的约定, 值为 0.0 只是初值)。
    """
    panel_modes = parse_noll_modes(args.fit_modes, "--fit-modes")
    lens_modes = parse_noll_modes(args.fit_lens_modes, "--fit-lens-modes")
    errors = TwinErrorModel(
        lut_gamma=1.0,
        lut_phase_max_rad=2.0 * np.pi,
        gray_levels=int(args.levels),
        slm_zernike={j: 0.0 for j in panel_modes},
        slm_zernike_n_max=mode_n_max(panel_modes),
        lens_zernike={j: 0.0 for j in lens_modes},
        lens_zernike_n_max=mode_n_max(lens_modes),
    )
    return BenchTwinConfig(
        region=region,
        region_span_px=float(args.region_span_px),
        far_field_padding=model_side // max(1, region),
        crop_px=model_side,
        beam_w0_px=float(args.beam_w0_px),
        peak_adu=100.0,
        noise_adu=0.0,
        seed=int(args.seed),
        errors=errors,
    )


def merge_coupling(
    result: TwinFitResult, coupling_path: Path, levels: int
) -> tuple[float, float, int]:
    """把**单独测出来的**灰度→振幅耦合合并进拟合结果 (原样改 config.errors)。

    Returns:
        ``(c, psi_rad, coupling_gray_levels)``。

    Raises:
        SystemExit: 该 JSON 不是合法的孪生配置。
    """
    if not coupling_path.is_file():
        sys.exit(f"错误: --coupling-config 不存在: {coupling_path}")
    try:
        payload = json.loads(coupling_path.read_text(encoding="utf-8"))
        source_twin = BenchTwin.from_dict(payload.get("config", payload))
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        sys.exit(f"错误: --coupling-config 不是合法的 save_fit 形状孪生配置: {exc}")
    err = source_twin.errors
    result.config.errors.slm_amplitude_coupling = float(err.slm_amplitude_coupling)
    result.config.errors.slm_amplitude_coupling_phase_rad = float(
        err.slm_amplitude_coupling_phase_rad
    )
    if int(err.gray_levels) != int(levels):
        print(
            f"⚠️  警告: 耦合配置的 gray_levels = {err.gray_levels} 与本次拟合的 "
            f"--levels {levels} 不同 —— 振幅耦合的灰度域周期钉在 gray_levels 上, "
            f"两者不一致时 ψ 的灰度域位置会错位; 请核对器件的 2π 灰度数 "
            f"(1064 nm 下应为 993)。"
        )
    return (
        float(err.slm_amplitude_coupling),
        float(err.slm_amplitude_coupling_phase_rad),
        int(err.gray_levels),
    )


def format_summary(
    result: TwinFitResult,
    samples: list[TwinSample],
    config: BenchTwinConfig,
    coupling: tuple[float, float, int] | None,
) -> None:
    """打印中文摘要: 用了多少对、每个拟合参数 (带单位)、各阶段损失、残差。"""
    errors = config.errors
    print("\n=== 反演出的孪生参数 (物理单位) ===")
    print(f"  LUT 灰度→相位 Gamma        = {errors.lut_gamma:.6f}   [无量纲]")
    print(f"  满灰度最大相位             = {errors.lut_phase_max_rad:.6f} rad "
          f"(= {errors.lut_phase_max_rad / np.pi:.4f}·π)")
    print(f"  2π 灰度数 gray_levels      = {errors.gray_levels} [级]")
    print(f"  相机离焦                   = {errors.defocus_m:.6e} m")
    print(f"  光路倾斜 tilt_x / tilt_y   = {errors.tilt_x_rad:.6e} / "
          f"{errors.tilt_y_rad:.6e} rad")
    if errors.slm_zernike:
        print("  面板 (SLM) 静态相差 Zernike:")
        for j in sorted(errors.slm_zernike):
            print(f"    {describe_mode(j):<34s} = {errors.slm_zernike[j]:+.6f} rad")
    else:
        print("  面板 (SLM) 静态相差 Zernike: 未参与反演 (--fit-modes 未给)")
    if errors.lens_zernike:
        print("  透镜像差 Zernike:")
        for j in sorted(errors.lens_zernike):
            print(f"    {describe_mode(j):<34s} = {errors.lens_zernike[j]:+.6f} rad")
    else:
        print("  透镜像差 Zernike: 未参与反演 (--fit-lens-modes 未给)")
    if coupling is not None:
        c, psi, c_levels = coupling
        print(f"  灰度→振幅耦合 c          = {c:.6f} [无量纲],  "
              f"ψ = {psi:.6f} rad  ← **来自单独测量** ({c_levels} 灰度周期), "
              f"不是从探针拟合的")
    else:
        print("  灰度→振幅耦合: 未合并 (耦合应从 fit_slm_amplitude_coupling.py 测量, "
              "探针数据上不可辨识)")

    print("\n=== 拟合质量 ===")
    stages = result.stages
    print(f"  使用探针配对             = {len(samples)} 对")
    print(f"  初始损失                 = {stages.get('initial_loss', float('nan')):.6g}")
    if "stage1_loss" in stages:
        print(f"  Stage-1 (DE 粗调) 损失  = {stages['stage1_loss']:.6g} "
              f"({stages.get('n_evaluations_stage1', 0):.0f} 次评估)")
    if "stage2_loss" in stages:
        print(f"  Stage-2 (L-BFGS-B) 损失 = {stages['stage2_loss']:.6g} "
              f"({stages.get('n_evaluations_stage2', 0):.0f} 次评估, "
              f"{stages.get('n_restarts_stage2', 1):.0f} 个起点)")
    print(f"  最终联合损失             = {result.loss:.6g}")
    print(f"  目标函数总评估次数       = {result.n_evaluations}")
    print(f"  拟合时孪生 peak_adu      = {stages.get('peak_adu', float('nan')):.6g} ADU")

    if result.per_sample:
        rms = float(np.mean([e["residual_rms"] for e in result.per_sample]))
        ssim = float(np.mean([e["ssim"] for e in result.per_sample]))
        corr = float(np.mean([e["corr"] for e in result.per_sample]))
        energy = float(np.mean([abs(e["energy_rel"]) for e in result.per_sample]))
        print("  实测 vs 仿真 (peak 归一):")
        print(f"    残差 RMS 均值          = {rms:.6g}")
        print(f"    SSIM 均值              = {ssim:.6f}")
        print(f"    Pearson 相关均值       = {corr:.6f}")
        print(f"    相对能量偏差均值       = {energy:.6f}")

    initial = float(stages.get("initial_loss", result.loss))
    if result.loss >= initial:
        print("\n⚠️  **拟合没有比初始配置更好** —— 保存的是初始 (裁剪到边界内的) "
              "参数; 这**不是**一次成功的反演, 请先检查探针数据质量 "
              "(信噪比 / 探针相位是否有差异) 再决定是否使用这个 JSON。")
    else:
        print(f"\n拟合把联合损失从 {initial:.6g} 降到 {result.loss:.6g} "
              f"(下降 {100.0 * (initial - result.loss) / max(initial, 1e-12):.1f}%)。")


def main(argv: list[str] | None = None) -> int:
    """入口: 读探针 → 建孪生 → fit_twin → save_fit (+ 可选耦合合并) → 摘要。"""
    args = parse_args(argv)
    samples, region, model_side, amplitude = load_probe_pairs(
        args.from_dir, args.limit
    )
    check_geometry(region, model_side, args.region, args.far_field_padding)
    if amplitude is not None and amplitude.ndim == 2:
        if amplitude.shape == (model_side, model_side):
            amp_note = f"{model_side}×{model_side} (远场模型网格)"
        elif amplitude.shape == (region, region):
            amp_note = f"{region}×{region} (瞳面网格, 该运行是 sim 后端)"
        else:
            amp_note = (
                f"{amplitude.shape} —— 既不是远场网格 ({model_side}×{model_side}) "
                f"也不是瞳面网格 ({region}×{region}), 请核对数据来源"
            )
        print(f"提示: source_amplitude.npy 形状 = {amp_note} (本脚本只核验几何, "
              f"孪生的入射幅度由 --beam-w0-px 的高斯参数化)")

    config = build_initial_config(args, region, model_side)
    n_panel = len(config.errors.slm_zernike)
    n_lens = len(config.errors.lens_zernike)
    print(f"数据: {args.from_dir}")
    print(f"  探针配对: {len(samples)} 对; 瞳面 region = {region}; "
          f"远场模型网格 = {model_side}×{model_side} "
          f"(far_field_padding = {config.far_field_padding}, crop_px = {model_side})")
    print(f"  待反演: LUT Γ / 满灰度相位 / 相机离焦 (全局) + 面板 Zernike "
          f"{n_panel} 项 + 倾斜 2 项 + 透镜 Zernike {n_lens} 项 (空间)")

    result = fit_twin(
        samples,
        config,
        maxiter=int(args.maxiter),
        popsize=int(args.popsize),
        maxiter_fine=int(args.maxiter_fine),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )

    coupling: tuple[float, float, int] | None = None
    if args.coupling_config is not None:
        coupling = merge_coupling(result, args.coupling_config, config.errors.gray_levels)
        # 合并后的孪生与拟合时的不是同一台: 重评一次损失, 别让摘要里的损失
        # 冒充「含耦合的孪生」的损失。
        merged_loss, _ = twin_loss(BenchTwin(result.config), samples)
        print(f"提示: 合并测量耦合后, 同一批探针上的联合损失 = {merged_loss:.6g} "
              f"(拟合期的 {result.loss:.6g} 是不含耦合的孪生)")

    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data") / "debug" / "gsnet_bench_twin_fit" / ts / "twin_fit.json"
    written = save_fit(result, out)
    if coupling is not None:
        # 在 save_fit 形状上追加溯源说明 (load_fit 忽略多余顶层键)。
        payload = json.loads(written.read_text(encoding="utf-8"))
        c, psi, c_levels = coupling
        payload["coupling_source"] = {
            "note": "灰度→振幅耦合来自单独测量, 不是从探针反演的",
            "coupling_config": str(args.coupling_config),
            "slm_amplitude_coupling": c,
            "slm_amplitude_coupling_phase_rad": psi,
            "coupling_gray_levels": c_levels,
        }
        written.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    format_summary(result, samples, result.config, coupling)

    twin = BenchTwin(result.config)
    print("\n几何契约 (gsnet_transfer / export_gsnet_phase 的 --twin-config 核验):")
    print(f"  孪生 region = {twin.region} = 探针相位边长 {region} ✓")
    print(f"  孪生 crop_side = {twin.crop_side} = 目标帧边长 {model_side} ✓")
    print(f"\n拟合结果已写出 (save_fit 形状, load_fit 可读回): {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
