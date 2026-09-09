"""SLM 相位→CCD 响应探针工具 (phase-response probe).

在 2f Fourier 光路下写不同 SLM 相位图案, 采集同一 CCD, 对比相位图案是否
真的作用于光 —— 用于定位 "CCD 看不到 SLM 调制" 类硬件/光路问题。

设计: 探针 = 一组 *相位用例* (PhaseCase) + 一个 *共享采集/统计/渲染* 通道
(run_phase_probe)。相位用例纯函数式地定义图案, 与硬件采集解耦, 便于新增
图案而无需改动采集逻辑。已含两组内置用例:

    defocus  —— 离焦相位 (控制器同款 Zernike Z(2,0) + 大离焦多圈包裹)
    lens     —— 透镜相位 (PatternHelper.lens, f=1000/250mm 全尺寸)

探针序列固定为: A flat 基线 -> B 相位 -> C 相位 -> D flat 复测 (可逆性对照)。
每次写相位到**随机内存槽 (2~125, 排除当前显示槽)** —— 连续写同槽是 no-op,
LCOS 不刷新 (AGENTS.md 硬约束)。相机多帧平均 (n_sample) 降噪; 另取单帧序列
估计帧间噪声底, 判定用的是逐像素帧间 std 均值。

``--render-only`` 可从保存的 npy + metrics.json 离线重建 PNG / 判定, 不碰硬件。

用法::

    python -m ao_shaping.tools.slm.slm_phase_response --probe lens        # 透镜用例, 实物
    python -m ao_shaping.tools.slm.slm_phase_response --probe defocus     # 离焦用例
    python -m ao_shaping.tools.slm.slm_phase_response --probe lens --render-only

注意 (AGENTS.md): SLM 必须 memory 模式 (video_mode=0), 禁止 DVI; 相位写入用
``slm.create_phase_from_array`` 或直接 raw uint16 (平场走 np.full, 严禁弧度转换);
首次采集可能无限阻塞 (WaitImageV3 由 SDK 内部驱动), 脚本逐步打日志便于定位挂点。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import click
import numpy as np
from loguru import logger

import matplotlib

matplotlib.use("Agg")  # 无界面后端, 只保存 PNG
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200


# ── 已确认的硬件/光路事实 (2026-09 诊断固化, 勿改) ──────────────────────────
# SLM#1 22030108, 1920x1200, 10-bit, 2π@1064nm≈993 灰度 (设备动态查询, 不硬编码).
# MiiCam#0 2688x1520 MONO8; 2f 光路 0 级 = 帧内全局最大 (argmax), 非相机中心.
# 已知正常曝光基线 ~0.02ms; diff-shaping 硬件闭环挂起点: SLM open 后首次取图.
_FLAT_GRAY = 512  # 平场灰度 (raw uint16, 严禁走弧度转换)
_DEFOCUS_AMP_RAD = 30.0  # "大"离焦: 30 rad ≈ 4.8 个 2π 包裹
_SLOT_MIN, _SLOT_MAX = 2, 125

# 判定阈值: rmse 超过 5×噪声底 视为有响应.
_RMSE_FACTOR = 5.0

# 帧名常量: 基线 / 复测对照.
_BASELINE = "A_flat"
_REPEAT = "D_flat_again"


# ── 相位用例数据类型 ────────────────────────────────────────────────────────


@dataclass
class PhaseCase:
    """单个相位用例: 名字、描述、图案生成函数。

    ``builder`` 接收已打开的 SLM (读取分辨率/灰度位宽/波长), 返回 uint16 灰度
    相位图 ``(height, width)``。名字用作帧名后缀, 须与序列前缀拼接唯一。
    """

    name: str
    description: str
    builder: Callable[["SantecSLM200"], np.ndarray]


# ── 相位用例生成 (纯函数, 与硬件采集解耦) ──────────────────────────────────


def _defocus_normalized_builder(slm: "SantecSLM200") -> np.ndarray:
    """控制器同款 Zernike 离焦: generate_zernike_polynomial({(2,0): 1.0})。

    返回 min-max 归一化 uint16 灰度 (整个瞳孔摆满 0~1023 灰度 ≈ 一个 2π 包裹)。
    """
    from ao_shaping.utils.pattern_helper import PatternHelper

    w, h = int(slm.Panel_Res[0]), int(slm.Panel_Res[1])
    helper = PatternHelper((w, h), bits=slm.Gray_Scale_bits)
    return helper.generate_zernike_polynomial(
        coefficients={(2, 0): 1.0}, radius=min(h, w) // 2,
    )


def _defocus_big_builder(slm: "SantecSLM200") -> np.ndarray:
    """大离焦 (多圈包裹): Z(2,0)=2ρ²-1 × 30 rad → to_uint16 包裹。"""
    from ao_shaping.utils.pattern_helper import PatternHelper

    w, h = int(slm.Panel_Res[0]), int(slm.Panel_Res[1])
    helper = PatternHelper((w, h), bits=slm.Gray_Scale_bits)
    radius = min(h, w) // 2
    yy, xx = np.mgrid[:h, :w]
    rho = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / radius
    pupil = rho <= 1.0
    phase_rad = np.zeros((h, w), dtype=np.float64)
    phase_rad[pupil] = _DEFOCUS_AMP_RAD * (2.0 * rho[pupil] ** 2 - 1.0)
    return helper.to_uint16(phase_rad)


def _lens_builder(slm: "SantecSLM200", focal_length_m: float) -> np.ndarray:
    """透镜相位 (与 multi_slm_controller "透镜" 分支一致, 全尺寸)。"""
    from ao_shaping.utils.pattern_helper import PatternHelper

    w, h = int(slm.Panel_Res[0]), int(slm.Panel_Res[1])
    helper = PatternHelper((w, h), bits=slm.Gray_Scale_bits)
    wavelength_nm = slm.wavelength
    assert wavelength_nm is not None, "SLM 波长未设置，无法生成透镜相位"
    phase_rad = helper.lens(
        focal_length=focal_length_m,
        wavelength=float(wavelength_nm) * 1e-9,
        pixel_size=float(slm.Pitch_um) * 1e-6,
        lens_radius=None,  # 全尺寸
    )
    return slm.create_phase_from_array(phase_rad)


def defocus_cases() -> list[PhaseCase]:
    """离焦探针用例: Zernike 离焦 + 大离焦 (多圈包裹)。"""
    return [
        PhaseCase("zernike_defocus", "Zernike Z(2,0) defocus (normalized)",
                  _defocus_normalized_builder),
        PhaseCase("big_defocus", f"big defocus ({_DEFOCUS_AMP_RAD}rad~multi-wrap)",
                  _defocus_big_builder),
    ]


def lens_cases() -> list[PhaseCase]:
    """透镜探针用例: f=1000mm 与 f=250mm 全尺寸透镜。"""
    return [
        PhaseCase("lens_1000mm", "lens f=1000mm full-size 8um",
                  lambda slm: _lens_builder(slm, 1.0)),
        PhaseCase("lens_250mm", "lens f=250mm full-size 8um",
                  lambda slm: _lens_builder(slm, 0.25)),
    ]


# ── 共享采集/统计 (无状态, 与相位用例解耦) ──────────────────────────────────


def _snapshot(camera, n_sample: int) -> np.ndarray:
    """取一帧平均图并转 float64。"""
    return np.asarray(
        camera.get_numpy_image(n_sample=n_sample, skip_first=True), dtype=np.float64,
    )


def _metrics(img: np.ndarray) -> dict:
    """帧统计: total / peak / centroid / ee90_r / argmax。"""
    total = float(img.sum())
    peak = float(img.max())
    s = img.sum()
    cy = cx = None
    if s > 0:
        yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        cy, cx = float((img * yy).sum() / s), float((img * xx).sum() / s)
    r_ee = float("nan")
    if s > 0:
        yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        order = np.argsort(rr, axis=None)
        cum = np.cumsum(img.ravel()[order])
        if cum[-1] > 0:
            idx = int(np.searchsorted(cum, 0.90 * cum[-1]))
            r_ee = float(rr.ravel()[order][idx])
    argmax = tuple(int(v) for v in np.unravel_index(int(np.argmax(img)), img.shape))
    centroid: tuple[float, float] | None = None
    if cy is not None and cx is not None:
        centroid = (round(cy, 2), round(cx, 2))
    return {
        "total": total,
        "peak": peak,
        "centroid": centroid,
        "ee90_r": r_ee,
        "argmax_rc": argmax,
    }


def _delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _noise_est(camera) -> dict:
    """10 次单帧逐像素 std → 帧间噪声底估计。"""
    frames = [_snapshot(camera, n_sample=1) for _ in range(10)]
    stack = np.stack(frames)
    per_px_std = stack.std(axis=0)
    return {
        "per_px_std_mean": float(per_px_std.mean()),
        "per_px_std_max": float(per_px_std.max()),
    }


# ── 共享渲染 (无状态) ───────────────────────────────────────────────────────


def _render(frames: dict[str, np.ndarray], metrics: dict, out: Path) -> None:
    """每帧对比 PNG + total/peak 条形 PNG。"""
    names = list(frames.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        m = metrics[name]
        im = ax.imshow(frames[name], cmap="jet", aspect="auto")
        ax.set_title(
            f"{name}\ntotal={m['total']:.0f} peak={m['peak']:.1f} "
            f"ee90_r={m['ee90_r']:.1f}px argmax={m.get('argmax_rc', '-')}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "frames.png", dpi=130)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    axes2[0].bar(names, [metrics[k]["total"] for k in names])
    axes2[0].set_title("total energy")
    axes2[1].bar(names, [metrics[k]["peak"] for k in names])
    axes2[1].set_title("peak")
    for ax in axes2:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
    fig2.tight_layout()
    fig2.savefig(out / "metrics_bars.png", dpi=130)
    plt.close(fig2)


# ── 主流程 ───────────────────────────────────────────────────────────────────


def build_sequence(cases: list[PhaseCase]) -> list[tuple[str, str, object]]:
    """构造探针序列: [A flat] + [每用例] + [D flat 复测]。

    返回 ``[(name, description, payload)]``, payload 在 flat 项为 ``None``,
    在相位用例项为 builder 闭包 (接收 slm, 返回灰度图)。最多 2 个相位用例。
    """
    assert 1 <= len(cases) <= 2, "相位响应探针支持 1~2 个相位用例"
    seq: list[tuple[str, str, object]] = [(_BASELINE, "flat baseline", None)]
    for c, prefix in zip(cases, ("B", "C")):
        seq.append((f"{prefix}_{c.name}", c.description, c.builder))
    seq.append((_REPEAT, "flat repeat (reversibility check)", None))
    return seq


def _resolve_payload(payload: object, slm: "SantecSLM200",
                     w: int, h: int) -> np.ndarray:
    """把序列项 payload 解析为 uint16 灰度图。

    ``None`` → flat 基线 (raw uint16 平场, 严禁弧度转换); callable → 相位用例
    builder (接收 slm, 返回灰度图)。
    """
    if payload is None:
        return np.full((h, w), _FLAT_GRAY, dtype=np.uint16)
    if not callable(payload):
        raise TypeError(f"unexpected payload type: {type(payload)!r}")
    result = payload(slm)
    assert isinstance(result, np.ndarray)
    return result


def run_phase_probe(camera, slm: "SantecSLM200", cases: list[PhaseCase], out: Path,
                    n_sample: int, settle_s: float, slot_min: int, slot_max: int,
                    exposure_ms: float) -> dict:
    """执行探针: 逐用例写相位 + 采集 + 统计 + 存档 + 渲染。

    返回 ``dict`` 含 ``{"noise", "metrics", "deltas", "verdict", "exposure_ms",
    "n_sample"}``。任何异常向上传播 (由 main finally 硬件释放); 产物在每步后
    增量写入, 中断亦可复现已采集帧。
    """
    w, h = int(slm.Panel_Res[0]), int(slm.Panel_Res[1])

    # 槽轮换: 2~slot_max 随机, 排除当前显示槽 (防同槽 no-op 不刷新).
    # 启动时读取当前显示槽, 跨进程续接 (survives process restart).
    last_slot: int = 0
    try:
        cur = slm.get_displayed_memory_number()
        last_slot = int(cur) if cur is not None else 0
    except Exception:
        last_slot = 0
    logger.info("SLM 当前显示槽: {}", last_slot)

    def next_slot() -> int:
        nonlocal last_slot
        cand = [s for s in range(slot_min, slot_max + 1) if s != last_slot]
        last_slot = random.choice(cand)
        return last_slot

    def _write_display(gray: np.ndarray) -> int:
        slot = next_slot()
        slm.write_phase(gray, memory_number=slot)
        time.sleep(0.05)
        slm.display_memory(slot)
        time.sleep(settle_s)
        logger.info("写入槽 {} 灰度范围 [{}, {}]", slot, int(gray.min()), int(gray.max()))
        return slot

    # 噪声底 (A 态下 10 张单帧)。
    logger.info("估计帧间噪声底 (10x n_sample=1)...")
    noise = _noise_est(camera)
    logger.info("噪声底 per-px std: mean={:.4f} max={:.4f}",
                noise["per_px_std_mean"], noise["per_px_std_max"])

    seq = build_sequence(cases)
    frames: dict[str, np.ndarray] = {}
    patterns: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for name, desc, payload in seq:
        logger.info("[PROBE] {}: {}", name, desc)
        gray = _resolve_payload(payload, slm, w, h)
        patterns[name] = gray
        _write_display(gray)
        img = _snapshot(camera, n_sample)
        frames[name] = img
        metrics[name] = _metrics(img)
        logger.info("[PROBE]   {}: total={:.0f} peak={:.2f} ee90_r={:.1f} argmax={}",
                    name, metrics[name]["total"], metrics[name]["peak"],
                    metrics[name]["ee90_r"], metrics[name]["argmax_rc"])

    # 对比: 每个相位用例帧 vs A flat; D 复测作为对照 (应接近噪声级).
    noise_floor = float(noise["per_px_std_mean"])
    deltas: dict[str, float] = {}
    for name in frames:
        if name.startswith("B_"):
            deltas["A_vs_B"] = _delta(frames[_BASELINE], frames[name])
        elif name.startswith("C_"):
            deltas["A_vs_C"] = _delta(frames[_BASELINE], frames[name])
    deltas["A_vs_D_control"] = _delta(frames[_BASELINE], frames[_REPEAT])

    verdict: dict[str, dict] = {}
    for key, d in deltas.items():
        verdict[key] = {
            "rmse": d,
            "x_noise_floor": d / noise_floor,
            "changed": bool(d > _RMSE_FACTOR * noise_floor),
        }

    # 存档 + 渲染 (增量, 中断亦可复现已采集帧)。
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames_stack.npy", np.array(frames, dtype=object))
    for k, v in frames.items():
        np.save(out / f"{k}.npy", v)
    for k, p in patterns.items():
        np.save(out / f"{k}_pattern.npy", p)
    _render(frames, metrics, out)
    payload = {
        "noise": noise,
        "metrics": metrics,
        "deltas": deltas,
        "verdict": verdict,
        "exposure_ms": exposure_ms,
        "n_sample": n_sample,
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("产物已保存到 {}", out.resolve())
    return payload


def _render_only(cases: list[PhaseCase], out: Path) -> dict:
    """从保存的 npy + metrics.json 离线重建 PNG 与判定 (不碰硬件)。"""
    seq = build_sequence(cases)
    names = [name for name, _, _ in seq]
    frames = {k: np.load(out / f"{k}.npy") for k in names}
    metrics = {k: _metrics(v) for k, v in frames.items()}
    _render(frames, metrics, out)
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return saved


# ── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--probe", type=click.Choice(["lens", "defocus"]), default="lens",
              help="相位用例组 (默认 lens)")
@click.option("--slm-number", type=int, default=1, help="SLM 设备编号 (默认 1)")
@click.option("--slm-wavelength", type=int, default=1064, help="SLM 工作波长 nm (默认 1064)")
@click.option("--cam-id", type=int, default=0, help="MiiCam 相机 ID (默认 0)")
@click.option("--exposure-ms", type=float, default=0.02, help="相机曝光 ms (默认 0.02)")
@click.option("--n-sample", type=int, default=10, help="每帧平均采样数 (默认 10)")
@click.option("--slot-min", type=int, default=_SLOT_MIN, help="内存槽下限 (默认 2)")
@click.option("--slot-max", type=int, default=_SLOT_MAX, help="内存槽上限 (默认 125)")
@click.option("--settle-s", type=float, default=0.4, help="写相位后稳定等待 s (默认 0.4)")
@click.option("-o", "--output", default=None, help="输出目录 (默认 docs/slm/<probe>_probe)")
@click.option("--render-only", is_flag=True, help="仅从已保存结果离线重绘/判定 (不碰硬件)")
def main(probe: str, slm_number: int, slm_wavelength: int, cam_id: int,
         exposure_ms: float, n_sample: int, slot_min: int, slot_max: int,
         settle_s: float, output: str | None, render_only: bool) -> None:
    """SLM 相位→CCD 响应探针: 验证 SLM 相位调制是否真的作用于光。

    使用 memory 模式 (video_mode=0); 相位写到随机内存槽 (2~125, 排除当前槽)。
    """
    cases = lens_cases() if probe == "lens" else defocus_cases()
    out = Path(output) if output else (Path("docs/slm") / f"slm_{probe}_probe")

    if render_only:
        saved = _render_only(cases, out)
        click.echo(f"[render-only] 已重建 PNG -> {out.resolve()}")
        click.echo("verdict: " + json.dumps(saved.get("verdict", {}), indent=2))
        return

    # 延迟导入硬件 (仅实物运行时; render-only 不依赖)。
    from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager
    from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200

    logger.info("SLM phase probe: {} | slm#{} @{}nm | camera#{} exposure {:.3f}ms | "
                "n_sample={} slots {}-{} | out={}",
                probe, slm_number, slm_wavelength, cam_id, exposure_ms,
                n_sample, slot_min, slot_max, out.resolve())

    slm: "SantecSLM200 | None" = None
    camera = None
    try:
        slm = SantecSLM200(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
        slm.open()
        logger.info("SLM 已打开: serial={} {}x{} {}bit",
                    getattr(slm, "_serial_number", None), slm.Panel_Res[0],
                    slm.Panel_Res[1], slm.Gray_Scale_bits)

        camera = CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_ms, bit_depth=8)
        camera.open()
        logger.info("相机已打开: camera#{} exposure={:.3f}ms", cam_id, exposure_ms)

        result = run_phase_probe(camera, slm, cases, out, n_sample, settle_s,
                                 slot_min, slot_max, exposure_ms)

        click.echo("=== SLM phase-response probe summary ===")
        # 相位用例下标 -> 序列前缀 (B_/C_); verdict 键为 "A_vs_<前缀>".
        for i, c in enumerate(cases):
            key = "A_vs_" + (["B", "C"][i])
            d = result["verdict"].get(key, {})
            click.echo(
                f"[{'CHANGED' if d.get('changed') else 'NO obvious change'}] "
                f"{c.name} ({c.description}): rmse={d.get('rmse', 0):.4f} "
                f"= {d.get('x_noise_floor', 0):.1f}x noise floor"
            )
        dc = result["verdict"].get("A_vs_D_control", {})
        click.echo(f"[{'CHANGED' if dc.get('changed') else 'no change (OK)'}] "
                   f"flat-repeat control: rmse={dc.get('rmse', 0):.4f}")
        click.echo(f"[DONE] 产物已保存到 {out.resolve()}")
    except SystemExit:
        raise
    except Exception:
        logger.exception("SLM phase probe failed")
        raise
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:  # 关闭异常不影响结论
                pass
        if slm is not None:
            try:
                slm.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
