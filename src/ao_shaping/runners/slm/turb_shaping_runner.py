"""湍流仿真光束整形实验 runner —— ``slm-turb-shaping`` (#253)。

在 2f 傅里叶**数字孪生**上给整形闭环挂时变波前扰动 (``SimDisturbance``),
产出报告需要的全套伴随产物 (recorder pkl + ``disturbance.json``/``npz`` +
``run.json``), 再离线渲染相位/光斑演化动画。**不触碰任何硬件**: 两个变体都强制
``--cam-type sim`` (square 变体另带 ``--slm-type sim``), 设备补丁与
``slm_pib_sim_run.py`` 同源。

变体 (M1 两个, 共用同一干扰挂具):

* ``spgd``  —— ``slm-pib spgd`` (Zernike 系数 SPGD, ``--objective shape``
  目标=方形远场);
* ``square`` —— ``slm-pib square`` (方形远场 CV+EE+AR 综合质量分)。

时变机制的口径 (为什么 "dynamic = 每次求值全新独立屏" 是本环路的正确渐近)
见 :mod:`ao_shaping.drivers.sim.disturbance` 模块 docstring; 幅度一律按
``disturbance.stats()`` 实测记录。reset 抗丢失包装在
:mod:`ao_shaping.drivers.sim.turbulence_harness` (与脚本共用唯一实现)。

报告渲染不在本 runner 内做 (报告生成归 ``scripts/``, 仓库红线): runner 只写
产物 + 可选以子进程调用 ``scripts/slm/generate_turbulence_shaping_report.py``。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import click
from loguru import logger

from ao_shaping.utils.cli.params import option, with_params

ROOT = Path(__file__).resolve().parents[4]

#: 光学系统种子 (与 ``slm_pib_sim_run.SYSTEM_SEED`` 同值同语义: 固定光学底座,
#: 扰动序列的可复现性由 ``dist_seed`` 决定)。
SYSTEM_SEED = 42



@dataclass
class TurbShapingParams:
    """湍流整形仿真实验参数。"""

    algorithm: Annotated[
        str,
        option(
            "--algorithm",
            type=click.Choice(["spgd", "square", "heuristic", "gs_refine",
                               "model_in_loop"]),
            help=(
                "整形变体: spgd=Zernike PIB shape 目标; square=方形综合质量分; "
                "heuristic=黑盒搜索 (ga/pso/sa/hc/rs/cem/de, 经 --search-algorithm); "
                "gs_refine=GS 预矫正 + 全频 freeform SPGD (M5a, 经 --phase-grid); "
                "model_in_loop=物理预测模型在环 (M5b, 湍流屏经 _SimBench 注入; "
                "quasi-static 假设下验收才可能过, 矩阵单元用 static 屏)"
            ),
        ),
    ] = "square"
    phase_grid: Annotated[
        int,
        option("--phase-grid", type=int, help="gs_refine 的 coarse freeform 相位网格边长"),
    ] = 12
    search_algorithm: Annotated[
        str,
        option(
            "--search-algorithm",
            type=click.Choice(["ga", "pso", "sa", "hc", "rs", "cem", "de"]),
            help="heuristic 变体的搜索算法",
        ),
    ] = "pso"
    pop_size: Annotated[
        int, option("--pop-size", type=int, help="heuristic 变体种群规模")
    ] = 16
    turb_mode: Annotated[
        str,
        option(
            "--turb-mode",
            type=click.Choice(["none", "static", "dynamic"]),
            help="干扰机制 (disturbance 模块三态)",
        ),
    ] = "dynamic"
    cn2: Annotated[
        float, option("--cn2", type=float, help="折射率结构常数 (生成器旋钮)")
    ] = 2e-13
    distance_m: Annotated[
        float,
        option(
            "--distance-m",
            type=float,
            help="生成器路径长度旋钮 [m] (默认 500: 标定到 σ≈0.09 rad @ cn2=1e-14)",
        ),
    ] = 500.0
    l_max: Annotated[float, option("--l-max", type=float, help="外尺度 [m]")] = 10.0
    l_min: Annotated[float, option("--l-min", type=float, help="内尺度 [m]")] = 0.01
    halo_pv_waves: Annotated[
        float, option("--halo-pv-waves", type=float, help="热晕 PV [waves], 0=禁用")
    ] = 0.0
    dist_seed: Annotated[
        int, option("--dist-seed", type=int, help="干扰主种子 (决定全部屏序列)")
    ] = 5
    epochs: Annotated[int, option("-e", "--epochs", help="优化轮数")] = 200
    cam_size: Annotated[int, option("--cam-size", type=int, help="仿真 CCD 窗口 px")] = 512
    exposure_ms: Annotated[float, option("--exposure-ms", type=float, help="曝光 ms")] = 80.0
    target_side: Annotated[
        float, option("--target-side", type=float, help="方形目标边长 px (square 变体)")
    ] = 120.0
    n_max: Annotated[int, option("-n", "--n-max", type=int, help="Zernike 最高径向阶数")] = 4
    out_dir: Annotated[
        Path, option("-d", "--out-dir", type=Path, help="产物根目录 (data/ 同级可用 tmp)")
    ] = Path("data")
    archive_factor: Annotated[
        int, option("--archive-factor", type=int, help="屏归档空间抽取因子")
    ] = 8
    archive_max: Annotated[
        int, option("--archive-max", type=int, help="归档的不同屏上限")
    ] = 12
    report: Annotated[
        bool, option("--report/--no-report", help="结束后调用报告生成器 (默认开)")
    ] = True
    report_out: Annotated[
        Path | None,
        option("--report-out", type=Path, help="报告输出目录 (默认自动)"),
    ] = None
    dry_run: Annotated[
        bool,
        option("--dry-run", is_flag=True, help="只打印将执行的组装, 不运行不读数据"),
    ] = False


def build_click_args(params: TurbShapingParams) -> list[str]:
    """把实验参数组装成既有 ``slm-pib`` 子命令的 argv (不复制其接线)。

    ``square`` 的 ``-d/--debug`` 必须放**组级**前缀 (``square`` 无 RunParams
    组, ``-d`` 与 ``--delta`` 冲突, 见 ``_resolve_run_context``); 且必须透传
    ``--seed`` —— ``SlmSquareConfig.random_seed`` 缺省 None 时 SPGD 扰动流走
    未播种的全局熵, 同种子两次运行质量末值差实测 5.6e-3, 复现性测试必红。
    ``spgd`` 自带 RunParams, 沿用 ``slm_pib_sim_run`` 已验证的子命令后置形态。
    """
    common = [
        "--cam-type", "sim",
        "--cam-id", "0",
        "--cam-size", str(params.cam_size),
        "-n", str(params.n_max),
        "-e", str(params.epochs),
    ]
    if params.algorithm == "spgd":
        return [
            "spgd", "-d", str(params.out_dir), "--debug", *common,
            "--exposure-time-ms", str(params.exposure_ms),
            "-c", "shape",
            "--slm-number", "1",
            "--slm-wavelength", "1064",
            "--objective", "shape",
            "--target-shape", "square",
            "--target-size", str(int(params.target_side)),
            "--delta", "0.5",
            "--optimizer-type", "adamod",
            "--w-uniformity", "2.0",
            "--w-peak", "0.5",
        ]
    if params.algorithm == "heuristic":
        # heuristic 与 spgd 共用 _execute 会话 (同一 patch_sim_pib_shaping 挂具
        # 路径, 干扰包装同样生效); 搜索算法/种群走 HeuristicParams。
        return [
            "heuristic", "-d", str(params.out_dir), "--debug", *common,
            "--exposure-time-ms", str(params.exposure_ms),
            "-c", "shape",
            "--slm-number", "1",
            "--slm-wavelength", "1064",
            "--objective", "shape",
            "--target-shape", "square",
            "--target-size", str(int(params.target_side)),
            "--algorithm", params.search_algorithm,
            "--pop-size", str(params.pop_size),
        ]
    if params.algorithm == "model_in_loop":
        # MIL 孪生 (M5b): 干扰屏不进 get_system (那是 SimPibCCD 的光路), 而是由
        # run_arm 把 _open_bench 换成带 disturbance 的 _SimBench。轮数=epochs,
        # 拟合基=Zernike n_orders=n_max; 拒收连击保险按轮数放宽 —— 孪生里该守卫
        # 的"保护硬件时间"语义不存在, 中止只会丢整格, streak 曲线本身是结论。
        return [
            "--cam-type", "sim", "--cam-id", "0",
            "--cam-size", str(params.cam_size),
            "--n-rounds", str(params.epochs),
            "--n-orders", str(params.n_max),
            "--target-shape", "square",
            "--target-side", str(int(params.target_side)),
            "--seed", str(params.dist_seed),
            "--exposure-time-ms", str(params.exposure_ms),
            "--max-rejection-streak", str(max(10, int(params.epochs))),
        ]
    if params.algorithm == "gs_refine":
        # freeform (M5a): GS 预矫正 + 粗网格全频 SPGD; 测量走 SimPibCCD 真光路,
        # install_disturbance_system 的注入系统对它就是"带湍流的实验系统"。
        return [
            "--cam-type", "sim", "--cam-id", "0",
            "--cam-size", str(params.cam_size),
            "--epochs", str(params.epochs),
            "--phase-grid", str(params.phase_grid),
            "--target-shape", "square",
            "--target-side", str(int(params.target_side)),
            "--seed", str(params.dist_seed),
            "--exposure-time-ms", str(params.exposure_ms),
        ]
    return [
        "--debug", "-d", str(params.out_dir),
        "square", *common,
        "--slm-type", "sim",
        "--seed", str(params.dist_seed),
        "--exposure-ms", str(params.exposure_ms),
        "--target-side", str(int(params.target_side)),
    ]


def _invoke_group(click_args: list[str], *, out_dir: Path,
                  command_name: str = "slm-pib") -> int:
    """进程内调用整形命令 (默认 ``slm-pib`` 组), 并注入 main 根 ctx 的 obj。

    ``square`` 的 ``-d/--debug`` 只能来自 main 根 group 写入的
    ``ctx.obj`` (组自身不写 obj, 见 ``_resolve_run_context`` 的冲突说明),
    而独立 ``main()`` 没有父 ctx。这里合成一个仅含 obj 的根 context 经
    ``make_context`` 挂上, 子命令沿 obj 继承链取值; ``spgd`` 变体自带
    RunParams, 同一入口不受影响 (参数仍在子命令位)。
    """
    if command_name == "slm-gs-refine":
        from ao_shaping.runners.slm.gs_refine_runner import run as entry
    elif command_name == "slm-model-in-loop":
        from ao_shaping.runners.slm.model_in_loop_runner import run as entry
    else:
        from ao_shaping.runners.slm.slm_shaping_runner import run as entry

    root_ctx = click.Context(
        click.Command("root"), obj={"dir": str(out_dir), "debug": True}
    )
    try:
        with root_ctx:
            group_ctx = entry.make_context(command_name, list(click_args), parent=root_ctx)
            with group_ctx:
                entry.invoke(group_ctx)
    except SystemExit as exc:  # 子命令内部 sys.exit
        return int(exc.code) if isinstance(exc.code, int) else 0
    except click.UsageError as exc:
        logger.error("slm-pib usage error: {}", exc.format_message())
        return exc.exit_code
    return 0


def run_arm(params: TurbShapingParams) -> dict:
    """执行一个变体: 挂扰动 → 跑既有子命令 → 写伴随产物, 返回摘要 dict。"""
    from ao_shaping.drivers.sim.disturbance import DisturbanceConfig, SimDisturbance
    from ao_shaping.drivers.sim.slm_pib_sim import SLM_H, SLM_W
    from ao_shaping.drivers.sim.turbulence_harness import (
        install_disturbance_system,
        write_disturbance_companion,
    )

    config = DisturbanceConfig(
        mode=params.turb_mode,
        cn2=params.cn2,
        distance_m=params.distance_m,
        l_max=params.l_max,
        l_min=params.l_min,
        thermal_halo_pv_waves=params.halo_pv_waves,
        seed=params.dist_seed,
    )
    disturbance = SimDisturbance(config, (SLM_H, SLM_W))
    # seed 必须显式给: 挂具若只装包装不重建系统, get_system 的惰性首建**不带**
    # 扰动 (heuristic/spgd 走 patch_sim_pib_shaping, 它刻意不 reset —— 实测
    # disturbance.stats().calls=0, 运行与自己的伴随清单矛盾)。
    install_disturbance_system(disturbance, seed=SYSTEM_SEED)

    click_args = build_click_args(params)
    command_name = {"gs_refine": "slm-gs-refine",
                    "model_in_loop": "slm-model-in-loop"}.get(params.algorithm, "slm-pib")
    logger.info("{} argv: {}", command_name, click_args)

    if params.algorithm == "model_in_loop":
        # MIL 的孪生不经过 get_system, install_disturbance_system 对它无效 ——
        # 唯一注入点是 _open_bench 选 bench 的那一行。进程内替换, 用毕还原,
        # 否则会污染同进程后续非湍流 MIL 会话 (测试 autouse 夹具同样兜底)。
        import ao_shaping.optimizer.wfless.slm_model_in_loop as mil

        saved_open = mil._open_bench

        def _open_turbulent_bench(config):
            if config.cam_type == "sim":
                return mil._SimBench(config, disturbance=disturbance)
            return saved_open(config)

        mil._open_bench = _open_turbulent_bench
        try:
            exit_code = _invoke_group(click_args, out_dir=params.out_dir,
                                      command_name=command_name)
        finally:
            mil._open_bench = saved_open
    else:
        # seed 必须显式给: 挂具若只装包装不重建系统, get_system 的惰性首建**不带**
        # 扰动 (heuristic/spgd 走 patch_sim_pib_shaping, 它刻意不 reset —— 实测
        # disturbance.stats().calls=0, 运行与自己的伴随清单矛盾)。
        install_disturbance_system(disturbance, seed=SYSTEM_SEED)
        exit_code = _invoke_group(click_args, out_dir=params.out_dir,
                                  command_name=command_name)
    if exit_code != 0:
        logger.warning("{} runner exited with status {}", command_name, exit_code)

    debug_root = Path(params.out_dir) / "debug"
    pkls = sorted(
        debug_root.rglob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not pkls:
        raise RuntimeError(f"no recorder pkl under {debug_root}")
    run_dir = pkls[0].parent

    measured = disturbance.stats()
    run_meta = {
        "arm": params.algorithm,
        "turb_mode": params.turb_mode,
        "cn2": params.cn2,
        "dist_seed": params.dist_seed,
        "epochs": params.epochs,
        # 报告生成器 sidecar_n_max 的兜底来源 (MIL 的拟合基叫 n_orders)。
        "n_max": params.n_max,
        "cam_type": "sim",
        "exit_code": exit_code,
    }
    json_path, npz_path = write_disturbance_companion(
        disturbance,
        run_dir,
        run_meta=run_meta,
        archive_factor=params.archive_factor,
        archive_max=params.archive_max,
    )
    summary = {
        **run_meta,
        "measured": measured,
        "run_dir": str(run_dir),
        "companion_json": str(json_path),
        "companion_npz": str(npz_path),
    }
    (run_dir / "run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


@click.command()
@click.pass_context
@with_params(TurbShapingParams, kw_name="p")
def run(ctx: click.Context, p: TurbShapingParams) -> None:
    """湍流仿真光束整形实验 (数字孪生, CI 可跑, 每次产报告)。"""
    if ctx.resilient_parsing:
        return
    if p.dry_run:
        click.echo(json.dumps({"argv": build_click_args(p), "arm": p.algorithm}, ensure_ascii=False))
        return
    summary = run_arm(p)
    measured = summary["measured"]
    click.echo(
        f"turb {p.algorithm}: mode={p.turb_mode} cn2={p.cn2:g} "
        f"sigma_total={measured['sigma_total_rad']:.4f} rad "
        f"evals={int(measured['calls'])} -> {summary['run_dir']}"
    )
    if p.report:
        script = ROOT / "scripts" / "slm" / "generate_turbulence_shaping_report.py"
        report_out = p.report_out or (
            ROOT / "report" / "turbulence_shaping" / Path(summary["run_dir"]).name
        )
        cmd = [sys.executable, str(script), "--run-dir", summary["run_dir"], "--out", str(report_out)]
        click.echo("report: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
