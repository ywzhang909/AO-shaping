"""湍流整形仿真实验的离线报告生成器 (#253) —— 相位与目标光斑的动态演化。

读一次 ``slm-turb-shaping`` 运行目录 (recorder pkl + ``run.json`` +
``disturbance.json``), 产出:

* ``spot_evolution.gif``  —— 每轮远场光斑演化 (固定色标, 叠目标方框与 best 标记);
* ``phase_evolution.gif`` —— 每轮 SLM 整形相位演化 (由 ``_c`` Noll 系数经规范
  入口重建 raw 弧度, 再走离线灰度换算; 全帧共享同一色标, 演化可横向对比);
* ``metrics.png``         —— quality/CV/EE/AR 与 J 的收敛曲线;
* ``report.md``           —— 摘要 + 实测湍流幅度 (stats 口径) + 曲线/动画引用。

纯离线: 不 import 任何 drivers/gui, 不读 ``data/`` 之外的仓库外路径, 全部输入
来自 ``--run-dir``; 任何一次实验/测试跑完都必须能再生一份报告 (硬约束)。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.io.recorder_keys import (
    RECORD_COEFFS,
    RECORD_EPOCH,
    RECORD_IMAGE,
    RECORD_PHASE,
)
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_METRIC_KEYS = ("quality", "shape", "cv", "ee", "ar", "pbr")

#: 各变体的"好坏一分"列名不同: square 记 `quality`, spgd/heuristic 的 shape 目标记
#: `shape` (= J)。报告一律先找 quality 再找 shape, 缺列时该分节降级而不是打红。
_SCORE_FALLBACK = ("quality", "shape", "score")


def _score_of(row: dict) -> float | None:
    for key in _SCORE_FALLBACK:
        if row.get(key) is not None:
            return float(row[key])
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="一次运行的产物目录")
    ap.add_argument("--out", type=Path, default=None, help="报告目录 (默认自动)")
    ap.add_argument("--fps", type=float, default=6.0, help="GIF 帧率")
    ap.add_argument("--phase-grid", type=int, default=256, help="相位重建网格边长")
    ap.add_argument("--max-frames", type=int, default=80, help="GIF 帧数上限 (均匀抽样)")
    ap.add_argument("--max-side", type=int, default=256, help="GIF 最长边 px (提交档建议 ≤192 控制体积)")
    return ap.parse_args(argv)


def load_records(run_dir: Path) -> tuple[list[dict], list[int]]:
    """读 recorder pkl (data-mode ``{epoch: record}``), 返回按 epoch 排序的行。"""
    pkls = sorted(run_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"no recorder pkl in {run_dir}")
    obj = pickle.loads(pkls[0].read_bytes())
    if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        items = sorted(obj.items(), key=lambda kv: float(kv[0]))
        rows = [dict(v) for _, v in items]
    elif isinstance(obj, list):
        rows = [dict(r) for r in obj]
    else:
        raise ValueError(f"unsupported recorder pkl shape in {pkls[0]}")
    epochs = [int(r.get(RECORD_EPOCH, i)) for i, r in enumerate(rows)]
    return rows, epochs


def sidecar_n_max(run_dir: Path) -> int:
    """从同目录 JSON sidecar 取 n_max (相位重建的基阶数)。"""
    for j in sorted(run_dir.glob("*.json")):
        if j.name in {"run.json", "disturbance.json"}:
            continue
        try:
            payload = json.loads(j.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for key in ("n_max", "params"):
            v = payload.get(key) if key == "n_max" else (payload.get(key) or {}).get("n_max")
            if isinstance(v, int):
                return v
    return 4


def _subsample(rows: list, epochs: list, max_frames: int) -> tuple[list, list]:
    if len(rows) <= max_frames:
        return rows, epochs
    idx = np.unique(np.linspace(0, len(rows) - 1, max_frames).astype(int))
    return [rows[i] for i in idx], [epochs[i] for i in idx]


def _write_gif(frames: list[np.ndarray], path: Path, fps: float, max_side: int) -> None:
    """PIL 直编码 GIF: 画布 = 数据像素, 受 ``max_side`` 真实控制。

    旧 FuncAnimation 路径把每帧画进固定的 4.6x4.2 英寸画布 (460x420 px),
    ``max_side`` 只能缩小 imshow 数据区而缩不了 GIF 尺寸 (实测 60 帧 460x420
    单文件 6 MB)。改为 viridis→RGB、resize 到 max_side、量化 128 色、optimize。
    """
    from PIL import Image

    cmap = plt.get_cmap("viridis")
    rgb = []
    for f in frames:
        arr = np.clip(np.asarray(f, dtype=np.float64), 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[-1] == 3:  # spot_frames 已带标注条 RGB
            img = Image.fromarray((arr * 255).astype(np.uint8))
        else:  # phase_frames 等 2D 灰度 → viridis
            img = Image.fromarray((cmap(arr)[..., :3] * 255).astype(np.uint8))
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize(
                (max(int(img.width * scale), 1), max(int(img.height * scale), 1)),
                Image.BILINEAR,
            )
        rgb.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    rgb[0].save(
        path, save_all=True, append_images=rgb[1:],
        duration=int(round(1000.0 / fps)), loop=0, optimize=True,
    )


def spot_frames(rows: list[dict], side_px: float | None) -> list[np.ndarray]:
    """远场光斑帧: 全序列固定色标 (P99.9), 叠目标方框轮廓与 epoch 文本条。"""
    from PIL import Image, ImageDraw

    imgs = [np.asarray(r[RECORD_IMAGE], dtype=np.float64) for r in rows if r.get(RECORD_IMAGE) is not None]
    if not imgs:
        return []
    vmax = max(float(np.percentile(im, 99.9)) for im in imgs) or 1.0
    frames = []
    for i, im in enumerate(imgs):
        rgb = (plt.colormaps["viridis"](np.clip(im / vmax, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
        h, w = im.shape
        if side_px:
            half = int(side_px / 2.0)
            cy, cx = h // 2, w // 2
            for y in (max(cy - half, 0), min(cy + half, h - 1)):
                rgb[y, max(cx - half, 0):cx + half] = 255
            for x in (max(cx - half, 0), min(cx + half, w - 1)):
                rgb[max(cy - half, 0):cy + half, x] = 255
        ep = rows[i].get(RECORD_EPOCH, i)
        q = _score_of(rows[i])
        canvas = np.ones((h + 16, w, 3), dtype=np.uint8) * 245
        canvas[16:] = rgb
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        label = f"epoch {ep}" + (f"   score {q:.4f}" if q is not None else "")
        draw.text((4, 2), label, fill=(10, 10, 10))
        frames.append(np.asarray(img).astype(np.float64) / 255.0)
    return frames


def _freeform_phase_frames(rows: list[dict], grid: int) -> list[np.ndarray]:
    """freeform 变体 (gs_refine): 逐轮 ``_phase`` 是下发面板灰度 (uint16, 0..1023)。

    面板为 1200x1920, 取中心方窗再相干降采样到 ``grid``; 除以固定 1023 满量程
    (非逐帧 PV, 保留 DC 偏置 —— freeform 演化要看的是相对整幅的明暗推进, 不是
    每帧各自拉满)。
    """
    from PIL import Image

    out = []
    for row in rows:
        panel = row.get(RECORD_PHASE)
        if panel is None:
            continue
        arr = np.asarray(panel, dtype=np.float64)
        side = min(arr.shape)
        y0 = (arr.shape[0] - side) // 2
        x0 = (arr.shape[1] - side) // 2
        crop = arr[y0:y0 + side, x0:x0 + side]
        small = np.asarray(Image.fromarray(crop).resize((grid, grid), Image.BILINEAR))
        out.append(np.clip(small / 1023.0, 0.0, 1.0))
    return out


def phase_frames(rows: list[dict], n_max: int, grid: int) -> list[np.ndarray]:
    """整形相位帧: ``_c`` → raw 弧度 → 离线灰度 (全序列同一 PV 归一)。

    freeform 变体 (无 ``_c`` 但有 ``_phase``) 走 :func:`_freeform_phase_frames`。
    """
    coeffs = [np.asarray(r[RECORD_COEFFS], dtype=np.float64) for r in rows if r.get(RECORD_COEFFS) is not None]
    if not coeffs:
        return _freeform_phase_frames(rows, grid)
    phases = [
        generate_zernike_phase(list(c), resolution=(grid, grid), n_max=n_max)
        for c in coeffs
    ]
    scale = max(float(np.ptp(p)) for p in phases) or 1.0
    frames = []
    for p in phases:
        gray = phase_to_slm_grayscale(p / scale * 2.0 * np.pi, slm=None).astype(np.float64)
        frames.append(gray / 1023.0)
    return frames


def metrics_png(rows: list[dict], out: Path) -> str:
    epochs = [int(r.get(RECORD_EPOCH, i)) for i, r in enumerate(rows)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    have = False
    for key in _METRIC_KEYS:
        vals = [r.get(key) for r in rows]
        if all(v is None for v in vals) or key == "J":
            continue
        axes[0].plot(epochs, [float(v) for v in vals], label=key, lw=1.4)
        have = True
    axes[0].set_title("整形质量指标演化")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    jvals = [r.get("J") for r in rows]
    if any(v is not None for v in jvals):
        axes[1].plot(epochs, [float(v) for v in jvals if v is not None], color="tab:red", lw=1.4)
        axes[1].set_title("目标函数 J")
        axes[1].set_xlabel("epoch")
        axes[1].grid(alpha=0.3)
    if not have:
        axes[0].text(0.5, 0.5, "无指标列", ha="center")
    fig.tight_layout()
    name = "metrics.png"
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    return name


def build_report(
    run_dir: Path, out: Path, *, fps: float, phase_grid: int, max_frames: int, max_side: int = 256
) -> dict:
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").is_file() else {}
    dist = json.loads((run_dir / "disturbance.json").read_text(encoding="utf-8")) if (run_dir / "disturbance.json").is_file() else {}
    rows, epochs = load_records(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    side = rows[0].get("side") if rows else None
    sub_rows, _sub_epochs = _subsample(rows, epochs, max_frames)
    sp = spot_frames(sub_rows, side_px=float(side) if side else None)
    ph = phase_frames(sub_rows, n_max=sidecar_n_max(run_dir), grid=phase_grid)
    if sp:
        _write_gif(sp, out / "spot_evolution.gif", fps, max_side)
    if ph:
        _write_gif(ph, out / "phase_evolution.gif", fps, max_side)
    mname = metrics_png(rows, out)

    meas = dist.get("measured", {})
    first_q, last_q, best_q = None, None, None
    qs = [s for s in (_score_of(r) for r in rows) if s is not None]
    if qs:
        first_q, last_q, best_q = qs[0], qs[-1], max(qs)
    lines = [
        f"# 湍流仿真整形实验报告 — {run_dir.name}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 变体 | {run_meta.get('arm', '?')} |",
        f"| 湍流机制 | {dist.get('mode', run_meta.get('turb_mode', '?'))} |",
        f"| cn2 (生成器旋钮) | {run_meta.get('cn2', '?')} |",
        f"| 实测 σ_total | {meas.get('sigma_total_rad', float('nan')):.4f} rad "
        f"({meas.get('sigma_total_rad', 0.0) / (2 * np.pi):.4f} waves) |",
        f"| 光学求值次数 | {int(meas.get('calls', 0))} |",
        f"| epochs | {run_meta.get('epochs', len(rows) - 1)} |",
        f"| quality 首/末/最优 | {first_q} / {last_q} / {best_q} |",
        "",
        "## 动态演化",
        "",
        "![光斑演化](spot_evolution.gif)",
        "",
        "![相位演化](phase_evolution.gif)",
        "",
        f"![指标曲线]({mname})",
        "",
        "## 口径注记",
        "",
        "- 湍流幅度按 `SimDisturbance.stats()` **实测**记录 (生成器旋钮 cn2/距离 退化, 解析值不可信);",
        "- dynamic = 每次光学求值全新独立相位屏 (时间白极限), 依据见 "
        "`src/ao_shaping/drivers/sim/disturbance.py` 模块 docstring;",
        "- 相位帧由记录系数经 `utils/wavefront/zernike_utils.generate_zernike_phase` 重建 raw 弧度, "
        "再经 `utils/slm/phase_display.phase_to_slm_grayscale(slm=None)` 纯换算; 全序列共用同一 PV 色标。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": str(out / "report.md"),
        "spot_gif": bool(sp),
        "phase_gif": bool(ph),
        "rows": len(rows),
        "quality_first_last_best": [first_q, last_q, best_q],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    out = args.out or (ROOT / "report" / "turbulence_shaping" / run_dir.name)
    summary = build_report(
        run_dir, Path(out), fps=args.fps, phase_grid=args.phase_grid,
        max_frames=args.max_frames, max_side=args.max_side,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
