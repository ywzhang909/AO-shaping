# -*- coding: utf-8 -*-
"""fouriergsnet_sim_train.py — FourierGSNet 纯仿真场景矩阵训练 (离线, 无硬件).

对 TARGET SHAPE x ABERRATION SET x TURBULENCE LEVEL 的每个组合 (cell) 运行
真实的 FourierGSNet 闭环整形管线 (``fouriergsnet_optimize.py``), 但用
``SimFourierGSNetEnv`` 数字孪生替代 SLM+CCD 硬件:

* 每个 cell 独立种子 (确定性可复现), 独立 env (含 CCD 泊松/读出噪声 +
  标定噪声注入), 独立静态像差预设与可选的 OU 湍流漂移;
* 管线流程与硬件 ``run`` 命令一致: ``adaptive_gs_init`` (设备自适应 GS 初值)
  -> ``FourierGSNetLite`` -> ``closed_loop`` (默认关闭在线 replay, 避免
  随机权重网络触发慢速 collect+finetune);
* 逐闭环步保存动态帧 ``far_%04d.npy`` (K×K float32 干净远场) 与
  ``phase_%04d.npy`` (N×N float32 整形相位), 供报告生成器渲染
  "整形相位 + 目标光斑演化" 动画 (frame 0 = 初值, 1..steps = 每闭环步之后);
* 每 cell 输出 config.json / metrics.csv / final.json, 顶层输出
  config.json / summary.json / README.md。

用法示例::

    python scripts/fouriergsnet_sim_train.py
    python scripts/fouriergsnet_sim_train.py --shapes square --aberrations none,defocus \\
        --turbulence off --steps 30 --k-px 512 --no-replay
    python scripts/fouriergsnet_sim_train.py --shapes square --aberrations none \\
        --turbulence off --steps 3 --k-px 128 --no-replay --out /tmp/fgn_sim_smoke
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.drivers.sim.fouriergsnet_env import (
    BeamParams,
    CalibNoise,
    SimFourierGSNetEnv,
)
from ao_shaping.utils.image.beam_metrics import compute_metrics

# ---------------------------------------------------------------------------
# 场景预设
# ---------------------------------------------------------------------------
# 目标形状 -> 管线 target_fn (half 参数即归一化半宽 / 高斯 sigma)。
TARGET_FNS: dict[str, object] = {
    "square": None,  # 延迟到管线模块加载后绑定 (make_square_target)
    "circle": None,
    "gaussian": None,
}

# 静态像差预设 (Noll 索引 -> 系数, 单位 rad)。
ABERRATION_PRESETS: dict[str, dict[int, float]] = {
    "none": {},
    "defocus": {4: 0.6},
    "astig": {5: -0.4, 6: 0.3},
    "mixed": {4: 0.5, 5: -0.4, 6: 0.3, 11: 0.2, 13: -0.1},
    "severe": {4: 0.8, 5: -0.6, 6: 0.5, 7: 0.3, 8: -0.3, 11: 0.3, 12: 0.2, 14: -0.2},
}

# 湍流预设 (OU 过程参数; None = 关闭)。
TURBULENCE_PRESETS: dict[str, dict | None] = {
    "off": None,
    "slow": {"sigma": 0.15, "tau": 50.0, "nolls": (4, 5, 6)},
    "fast": {"sigma": 0.3, "tau": 12.0, "nolls": (4, 5, 6, 11, 13)},
}

# 仿真环境固定参数 (与 gate 测试同量级, 模拟真实 CCD 噪声 + 标定失配)。
ENV_PEAK_PHOTONS = 50000.0
ENV_READ_NOISE_E = 2.0
ENV_CALIB_NOISE = dict(deltaK_fraction=0.02, center_offset_px=2.0, rotation_deg=0.3)

# 闭环前设备自适应 GS 迭代数 (硬件默认 15; 仿真中 5 轮已足够收敛)。
INIT_GS_ITERS = 5


def _load_pipeline():
    """从仓库根加载独立脚本 fouriergsnet_optimize.py (不加入 sys.modules 缓存)."""
    spec = importlib.util.spec_from_file_location(
        "fouriergsnet_optimize", ROOT / "fouriergsnet_optimize.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten_multi(values: tuple[str, ...]) -> list[str]:
    """展开 click multiple 选项, 支持逗号分隔 (--shapes square,circle)."""
    out: list[str] = []
    for v in values:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


def _cell_seed(base_seed: int, scenario: str) -> int:
    """按场景名派生确定性种子 (同场景跨运行可复现)."""
    digest = hashlib.sha256(scenario.encode()).hexdigest()[:8]
    return base_seed + int(digest, 16) % 2**31


def _scalar_record(record: dict) -> dict:
    """剔除 ndarray 字段, 保留可写 CSV/JSON 的标量."""
    return {k: v for k, v in record.items() if not isinstance(v, np.ndarray)}


def _json_safe(value):
    """递归把 numpy 标量/数组转成 JSON 可序列化类型."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_cell(
    fg,
    shape: str,
    aberration: str,
    turbulence: str,
    k_px: int,
    steps: int,
    replay: bool,
    ft_samples: int,
    ft_epochs: int,
    cell_seed: int,
    device: str,
    save_frames: bool,
    cell_dir: Path,
    frames_dir: Path,
    native: bool,
) -> dict:
    """运行单个场景 cell, 返回 summary 条目 (含最终指标与耗时)."""
    t0 = time.time()
    scenario = f"{shape}__{aberration}__{turbulence}"
    logger.info("=== cell: {} (seed={}) ===", scenario, cell_seed)

    np.random.seed(cell_seed)
    torch.manual_seed(cell_seed)

    # 管线模块全局补丁: DEV 在 ShapingSystem 构造/调用时读取, SETTLE_S 消除
    # 每次 display 的 0.2s 睡眠 (仿真无需 settle)。
    fg.DEV = device
    fg.SETTLE_S = 0.0

    # ---- 数字孪生环境 ----
    # native 模式: 64x64 方板, 模型全息图以原生像素节距放置 (无带限上采样),
    # env FFT 与 fouriergsnet_optimize.prop() 一致 -> 方形目标可被真实整形。
    # 非 native: 1920x1200 各向异性板, region=k_px 窄束 (旧行为)。
    if native:
        beam = BeamParams(region=64, w0=28.0, native=True)
    else:
        beam = BeamParams(region=k_px, w0=1.0)
    env = SimFourierGSNetEnv(
        K_px=k_px,
        beam=beam,
        noise_enabled=True,
        peak_photons=ENV_PEAK_PHOTONS,
        read_noise_e=ENV_READ_NOISE_E,
        calib_noise=CalibNoise(**ENV_CALIB_NOISE),
        seed=cell_seed,
    )
    env.aberrations = dict(ABERRATION_PRESETS[aberration])
    turb = TURBULENCE_PRESETS[turbulence]
    if turb is not None:
        env.configure_turbulence(
            seed=cell_seed + 1,
            sigma=turb["sigma"],
            tau=turb["tau"],
            dt=1.0,
            nolls=turb["nolls"],
        )

    # ---- 理想标定 (env 内部已注入标定噪声到渲染远场 = 真实失配) ----
    if native:
        beam_center = np.array([64.0, 64.0])
    else:
        beam_center = np.array([960.0, 600.0])
    calib = {
        "Kx": float(k_px),
        "Ky": float(k_px),
        "center": np.array([k_px / 2.0, k_px / 2.0]),
        "crop_side": int(k_px),
        "rotation_deg": 0.0,
        "beam_center": beam_center,
    }
    lut_result = {"phase_range": 2.0 * np.pi, "gray_of_phase": np.linspace(0.0, 255.0, 256)}

    def acquire(ccd, n_sample: int = 1):
        """采图回调: 先推进湍流一步 (未配置时 no-op), 再取 CCD 帧."""
        if env.turbulence_active:
            env.advance_time()
        return ccd.get_numpy_image(n_sample=n_sample)

    target_fn = getattr(fg, f"make_{shape}_target")
    sys_ = fg.ShapingSystem(env.slm, env.ccd, acquire, calib, lut_result, target_fn=target_fn)
    net = fg.FourierGSNetLite(fg.K_UNROLL, fg.N_ZERN, fg.CH, sys_.src_mask).to(fg.DEV)

    # ---- 初值: 设备自适应 GS (此时不装帧捕获包装, 不保存逐 GS 迭代帧) ----
    phi = sys_.adaptive_gs_init(iters=INIT_GS_ITERS)

    # frame 0 = 初值状态 (phi0, 其远场)。
    frames_far = [env.render_intensity().astype(np.float32)]
    frames_phase = [phi.detach().cpu().numpy().astype(np.float32)]

    # ---- 微调 (升级2域随机化, 对齐硬件 run 的 collect+finetune 流程) ----
    # 注意: finetune 内部损失硬编码 make_square_target (fouriergsnet_optimize.py
    # L886), 因此该阶段仅对 square 目标语义正确; FT_BATCH=16 以下的样本数会
    # 让批次循环为空 (range(0, n-FT_BATCH+1, FT_BATCH) = 空) -> 零训练步。
    ft_info: dict | None = None
    if ft_samples > 0:
        if shape != "square":
            logger.warning("finetune 损失硬编码 square 目标, cell {} 跳过微调", scenario)
        elif ft_samples < fg.FT_BATCH:
            logger.warning("ft_samples={} < FT_BATCH={} -> 微调零训练步, cell {} 跳过微调",
                           ft_samples, fg.FT_BATCH, scenario)
        else:
            t_ft = time.time()
            ft_data = sys_.collect(phi, ft_samples)
            net = sys_.finetune(net, ft_data, phi, epochs=ft_epochs, tag="sim-ft")
            ft_info = {
                "samples": ft_samples,
                "epochs": ft_epochs,
                "batch": fg.FT_BATCH,
                "lr": fg.FT_LR,
                "wall_time_s": round(time.time() - t_ft, 2),
            }
            logger.info("微调完成: {} 样本 x {} epochs ({:.1f}s)",
                        ft_samples, ft_epochs, ft_info["wall_time_s"])

    recorder = fg.Recorder("uniformity", "max")

    # ---- 闭环: 仅此阶段捕获逐步动态帧 (display 后 = 该步相位及其远场) ----
    if save_frames:
        orig_display = sys_.display

        def wrapped_display(phi_nx):
            orig_display(phi_nx)
            frames_far.append(env.render_intensity().astype(np.float32))
            frames_phase.append(phi_nx.detach().cpu().numpy().astype(np.float32))

        sys_.display = wrapped_display

    if replay:
        net, phi = sys_.closed_loop(net, phi, steps, recorder, replay=True)
    else:
        # 默认 --no-replay: 推理模式闭环。closed_loop 内部对 phi 做
        # phi.cpu().numpy() (fouriergsnet_optimize.py:938), 而 net 前向输出
        # phi 带 requires_grad (c_hat 来自可训练卷积层), 训练模式下直接
        # numpy() 会抛 RuntimeError。用 no_grad 包裹 (replay 微调路径需要
        # autograd, 故仅对非 replay 生效)。
        with torch.no_grad():
            net, phi = sys_.closed_loop(net, phi, steps, recorder, replay=False)

    # ---- 收尾整帧指标 (复刻 _append_final_record 内联; 不调用原函数,
    #      它会写模块级 OUT_DIR = data/shaping_test/history.csv) ----
    final = sys_.geo.workzone(np.asarray(acquire(env.ccd), np.float64), fg.N)
    metrics = compute_metrics(final, sys_.I_tgt.cpu().numpy())
    I_t = torch.from_numpy(final.astype(np.float32)).to(fg.DEV)
    uni, ee = fg.ShapingSystem._metrics(I_t, sys_.roi)
    recorder.append({"step": steps, "ccd": final, "uniformity": uni, "encircled": ee, **metrics})
    logger.info(
        "整帧: mse={:.5f} corr={:.4f} eff={:.4f} uniformity={:.3f} encircled={:.3f}",
        metrics["mse"], metrics["correlation"], metrics["efficiency"], uni, ee,
    )

    # ---- 保存 cell 产物 ----
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_config = {
        "scenario": scenario,
        "shape": shape,
        "aberration": aberration,
        "turbulence": turbulence,
        "k_px": k_px,
        "steps": steps,
        "replay": replay,
        "ft_samples": ft_samples,
        "ft_epochs": ft_epochs,
        "finetune": ft_info,
        "seed": cell_seed,
        "device": device,
        "init_gs_iters": INIT_GS_ITERS,
        "aberrations_rad": ABERRATION_PRESETS[aberration],
        "turbulence_params": turb,
        "env": {
            "peak_photons": ENV_PEAK_PHOTONS,
            "read_noise_e": ENV_READ_NOISE_E,
            "calib_noise": ENV_CALIB_NOISE,
        },
        "calib": _json_safe(calib),
    }
    with open(cell_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cell_config, f, indent=2, ensure_ascii=False)

    scalar_rows = [_scalar_record(rec) for rec in recorder.history]
    pd.DataFrame(scalar_rows).to_csv(cell_dir / "metrics.csv", index=False)

    best_uni = float(max(r["uniformity"] for r in recorder.history))
    best_ee = float(max(r["encircled"] for r in recorder.history))
    final_json = {
        "scenario": scenario,
        "final": _json_safe(_scalar_record(recorder.history[-1])),
        "best_uniformity": best_uni,
        "best_encircled": best_ee,
        "wall_time_s": round(time.time() - t0, 2),
    }
    with open(cell_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, (far, ph) in enumerate(zip(frames_far, frames_phase)):
            np.save(frames_dir / f"far_{i:04d}.npy", far)
            np.save(frames_dir / f"phase_{i:04d}.npy", ph)
        logger.info("已保存 {} 帧 -> {}", len(frames_far), frames_dir)

    logger.info("cell {} 完成 ({:.1f}s)", scenario, time.time() - t0)
    return {
        "scenario": scenario,
        "shape": shape,
        "aberration": aberration,
        "turbulence": turbulence,
        "seed": cell_seed,
        "ok": True,
        "ft_samples": ft_samples,
        "ft_epochs": ft_epochs,
        "finetune": ft_info,
        "final_uniformity": float(uni),
        "final_encircled": float(ee),
        "best_uniformity": best_uni,
        "best_encircled": best_ee,
        "wall_time_s": round(time.time() - t0, 2),
    }


def _write_readme(out_dir: Path, summary: dict) -> None:
    """顶层 README.md: 目录结构 / 指标列含义 / 帧语义 / 复现命令."""
    cells = summary["cells"]
    ok = [c for c in cells if c.get("ok")]
    ft_line = (f", 微调 {summary['ft_samples']} 样本×{summary['ft_epochs']} 轮"
               if summary["ft_samples"] > 0 else "")
    lines = [
        "# FourierGSNet 仿真场景矩阵训练结果",
        "",
        f"- 生成时间: {summary['generated_at']}",
        f"- 命令: `{summary['command']}`",
        f"- 网格 K={summary['k_px']}, 闭环步数 {summary['steps']}, "
        f"在线 replay={summary['replay']}, 基础种子 {summary['base_seed']}, "
        f"设备 {summary['device']}{ft_line}",
        f"- 场景矩阵: {len(cells)} 个 cell ({len(ok)} 成功)",
        "",
        "## 目录结构",
        "",
        "```",
        "<out>/",
        "├── config.json          # 顶层运行配置",
        "├── summary.json         # 每 cell 最终指标汇总",
        "├── README.md            # 本文件",
        "├── frames/<scenario>/   # 动态帧 (报告动画素材)",
        "│   ├── far_%04d.npy     #   K×K float32 干净远场强度 (render_intensity)",
        "│   └── phase_%04d.npy   #   N×N float32 整形相位 (rad)",
        "└── <scenario>/          # 每 cell 产物",
        "    ├── config.json      #   cell 配置 (种子/像差/湍流/标定/env)",
        "    ├── metrics.csv      #   逐步标量指标",
        "    └── final.json       #   最终整帧指标 + 最优值",
        "```",
        "",
        "## metrics.csv 列含义",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        "| step | 闭环步号 (0..steps-1; 收尾整帧记录为 steps) |",
        "| uniformity | 目标 ROI 内均匀度 (min/mean, 1=完美平顶) |",
        "| encircled | 目标 ROI 内封闭能量 (0~1) |",
        "| inference_ms | 网络单步推理耗时 (ms) |",
        "| mse / correlation / efficiency | 整帧 vs 目标强度图的 MSE / Pearson 相关 / 重叠能量比 |",
        "",
        "## 帧语义 (frames/<scenario>/)",
        "",
        "- `far_0000.npy` / `phase_0000.npy`: 初值状态 (adaptive_gs_init 之后, "
        "闭环第 0 步之前)。",
        "- `far_%04d.npy` / `phase_%04d.npy` (i=1..steps): 第 i 个闭环步 "
        "`display(phi)` 之后的整形相位及其远场 —— 逐帧播放即「整形相位 + "
        "目标光斑演化」动画。",
        "- 帧为干净远场 (`env.render_intensity()`), 非 CCD 含噪 uint16 帧, "
        "便于报告直接渲染。",
        "",
        "## 复现",
        "",
        f"```bash",
        f"python scripts/fouriergsnet_sim_train.py {summary['command'].split('python scripts/fouriergsnet_sim_train.py', 1)[-1].strip()}",
        f"```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--out",
    default=None,
    help="输出根目录 (默认 data/fouriergsnet_sim/<时间戳>)",
)
@click.option("--k-px", "k_px", default=512, show_default=True, type=int,
              help="仿真 CCD 帧边长 K (FFT 网格取 max(K, 光束区 512))")
@click.option("--shapes", multiple=True, default=("square", "circle", "gaussian"),
              help="目标形状 (可多次 / 逗号分隔)")
@click.option("--aberrations", multiple=True, default=("none", "defocus", "mixed"),
              help="静态像差预设 (可多次 / 逗号分隔)")
@click.option("--turbulence", multiple=True, default=("off", "slow", "fast"),
              help="湍流预设 (可多次 / 逗号分隔)")
@click.option("--steps", default=30, show_default=True, type=int, help="闭环步数")
@click.option("--replay/--no-replay", default=False, show_default=True,
              help="启用在线 replay (默认关闭: 随机权重网络可能触发慢速 collect+finetune)")
@click.option("--ft-samples", "ft_samples", default=0, show_default=True, type=int,
              help="闭环前微调样本数 (0=不微调; 仅 square 目标语义正确, 需 >=16 才有训练步)")
@click.option("--ft-epochs", "ft_epochs", default=12, show_default=True, type=int,
              help="微调轮数 (FT_EPOCHS)")
@click.option("--seed", default=42, show_default=True, type=int, help="基础随机种子")
@click.option("--device", default="cpu", show_default=True,
              type=click.Choice(["cpu", "cuda"]), help="计算设备")
@click.option("--save-frames/--no-save-frames", default=True, show_default=True,
              help="保存逐步动态帧 (far/phase .npy)")
@click.option("--native/--no-native", default=False, show_default=True,
              help="原生像素节距方形光阑 (64x64, 无带限上采样) — 让方形目标可被真实整形")
def cli(out, k_px, shapes, aberrations, turbulence, steps, replay, ft_samples,
        ft_epochs, seed, device, save_frames, native):
    """FourierGSNet 纯仿真场景矩阵训练 (无硬件, 数字孪生)."""
    fg = _load_pipeline()
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda 不可用, 回退到 cpu")
        device = "cpu"

    shapes = _flatten_multi(shapes)
    aberrations = _flatten_multi(aberrations)
    turbulences = _flatten_multi(turbulence)

    unknown = [s for s in shapes if s not in ("square", "circle", "gaussian")]
    if unknown:
        raise click.BadParameter(f"未知目标形状: {unknown} (可选 square/circle/gaussian)")
    unknown = [a for a in aberrations if a not in ABERRATION_PRESETS]
    if unknown:
        raise click.BadParameter(f"未知像差预设: {unknown} (可选 {list(ABERRATION_PRESETS)})")
    unknown = [t for t in turbulences if t not in TURBULENCE_PRESETS]
    if unknown:
        raise click.BadParameter(f"未知湍流预设: {unknown} (可选 {list(TURBULENCE_PRESETS)})")

    out_dir = (
        Path(out)
        if out
        else Path("data/fouriergsnet_sim") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = out_dir / "frames"

    cells = [(s, a, t) for s in shapes for a in aberrations for t in turbulences]
    logger.info("场景矩阵: {} 形状 x {} 像差 x {} 湍流 = {} 个 cell -> {}",
                len(shapes), len(aberrations), len(turbulences), len(cells), out_dir)

    summary: dict = {
        "command": " ".join(sys.argv),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "k_px": k_px,
        "steps": steps,
        "replay": replay,
        "ft_samples": ft_samples,
        "ft_epochs": ft_epochs,
        "base_seed": seed,
        "device": device,
        "native": native,
        "init_gs_iters": INIT_GS_ITERS,
        "cells": [],
    }

    for idx, (shape, aberration, turb) in enumerate(cells):
        scenario = f"{shape}__{aberration}__{turb}"
        cell_seed = _cell_seed(seed, scenario)
        cell_dir = out_dir / scenario
        frames_dir = frames_root / scenario
        try:
            entry = run_cell(
                fg, shape, aberration, turb, k_px, steps, replay, ft_samples,
                ft_epochs, cell_seed, device, save_frames, cell_dir, frames_dir,
                native,
            )
        except Exception as exc:  # noqa: BLE001 — 单 cell 失败不中断矩阵
            logger.exception("cell {} 失败: {}", scenario, exc)
            entry = {
                "scenario": scenario,
                "shape": shape,
                "aberration": aberration,
                "turbulence": turb,
                "seed": cell_seed,
                "ok": False,
                "error": str(exc),
            }
        summary["cells"].append(entry)
        logger.info("进度 {}/{}", idx + 1, len(cells))

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe({k: v for k, v in summary.items() if k != "cells"}),
                  f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    _write_readme(out_dir, summary)

    ok = [c for c in summary["cells"] if c.get("ok")]
    logger.info("完成: {}/{} cell 成功 -> {}", len(ok), len(cells), out_dir)


if __name__ == "__main__":
    cli()