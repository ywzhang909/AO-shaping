"""当前配置下的可微整形能力测试：任意目标形状 + 末尾叠加倾斜（闭环内不加倾斜）。

关键约定（2026-10-23 多次踩坑的教训）：
* **同帧定心**：先显示初值抓一帧，用**这一帧**的 argmax 同时建模型掩膜与实测掩膜
  （此前"先按历史 spot 算方向、再在设备侧重定中心"会差 20–80 px）；
* **闭环不加倾斜**：最优化在零级位置上做目标形状；倾斜只在**最终展示**时叠加，
  使整形光斑与零级分离（零级由未调制光直接透过，仍留在原位）；
* **形状按面积归一**：四种形状共用 ``--area-px2``，读数才可比（见 ``target_masks``）；
* **判据用当次实测的掩膜地板 3σ**（``--noise-floor``），不借用历史值；
* 曝光必须**不饱和**：载波帧会比无载波帧亮得多，先按无载波帧调到 peak ≲ 200。

用法：
    python scripts/slm/run_multishape_capability.py --execute --wait-until-idle 1800 \\
        --checkpoint <refit 目录> --seed-panel <初值.npy> --shape triangle \\
        --area-px2 7200 --carrier-period-px 32 --blocks 5 --exposure-ms 0.5 \\
        --noise-floor <地板.json> --out data/debug/multishape_forward/triangle_<ts>
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def panel_to_grid(panel: np.ndarray, size: int) -> np.ndarray:
    pitch = 1920 // size
    rows = 1200 // pitch
    grid = np.zeros((size, size))
    grid[:rows] = panel[: rows * pitch].reshape(rows, pitch, size, pitch).mean(axis=(1, 3))
    return grid


def grid_to_panel(grid: np.ndarray) -> np.ndarray:
    size = grid.shape[-1]
    pitch = 1920 // size
    panel = np.zeros((1200, 1920))
    rows = min(1200 // pitch, size)
    panel[: rows * pitch] = np.repeat(np.repeat(grid[:rows], pitch, axis=0), pitch, axis=1)
    return panel


def carrier_panel(period_px: float, axis: str = "x") -> np.ndarray:
    """面板上的线性相位斜坡（弧度，raw）。周期 period_px 对应远场平移 K/period。"""
    yy, xx = np.mgrid[:1200, :1920]
    coord = xx if axis == "x" else yy
    del yy
    return 2.0 * np.pi * coord / float(period_px)


def mask_stats(frame: np.ndarray, box: np.ndarray, core: np.ndarray, zero_mask: np.ndarray) -> dict:
    """在给定位形上算读数：形状内 CV、形状均值、全帧峰值、零点圆盘能量占比。"""
    values = frame[core]
    mean_core = float(values.mean()) if values.size else float("nan")
    cv = float(np.sqrt((((values / mean_core) - 1) ** 2).mean())) if mean_core > 0 else float("nan")
    total = float(frame.sum())
    return dict(
        cv=cv,
        mean_core=mean_core,
        mean_box=float(frame[box].mean()) if box.any() else float("nan"),
        peak=float(frame.max()),
        zero_fraction=float(frame[zero_mask].sum() / total) if total > 0 else float("nan"),
        area_px=int(box.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--noise-floor", type=Path, required=True)
    parser.add_argument("--shape", default="square",
                        choices=("square", "rectangle", "triangle", "star"))
    parser.add_argument("--area-px2", type=float, default=7200.0)
    parser.add_argument("--size", default="120,60", help="长方形长,短 (相机像元)")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--window-half-span-px", type=float, default=200.0)
    parser.add_argument("--carrier-period-px", type=float, default=32.0)
    parser.add_argument("--carrier-axis", choices=("x", "y"), default="x")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--cap", type=float, default=0.06)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--exposure-ms", type=float, default=0.5)
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("本脚本只做设备运行；加 --execute")
    args.out.mkdir(parents=True, exist_ok=False)

    import torch

    torch.set_num_threads(4)

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
        FrozenDefocusForward,
        UniformityObjective,
        load_frozen_parameters,
        recentre_window,
    )
    from ao_shaping.algorithm.signal_processing.target_masks import (
        build_polygon_masks,
        shape_vertices,
        shift_mask,
    )
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec.driver import Santec
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    resolution = float(json.loads(args.noise_floor.read_text(encoding="utf-8"))["three_sigma_resolution"])
    params = load_frozen_parameters(args.checkpoint)
    seed_panel = np.load(args.seed_panel).astype(np.float64)
    carrier = carrier_panel(args.carrier_period_px, args.carrier_axis)
    size_px = tuple(float(v) for v in args.size.replace("x", ",").replace(" ", ",").split(","))
    vertices = shape_vertices(args.shape, area_px2=args.area_px2, size_px=size_px,
                              rotation_deg=args.rotation_deg)
    records: dict = dict(
        shape=args.shape,
        size_px=size_px,
        area_px2=args.area_px2,
        rotation_deg=args.rotation_deg,
        vertex_count=int(len(vertices)),
        polygon_area_px2=float(
            abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))) / 2.0
        ),
        carrier_period_px=args.carrier_period_px,
        carrier_axis=args.carrier_axis,
        exposure_ms=args.exposure_ms,
        blocks=[],
        three_sigma_resolution=resolution,
        checkpoint=str(args.checkpoint),
        started_utc=datetime.now(timezone.utc).isoformat(),
    )

    if args.wait_until_idle > 0:
        import subprocess

        deadline = time.time() + args.wait_until_idle
        while time.time() < deadline:
            listing = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            busy = [line for line in listing.splitlines()
                    if any(t in line for t in ("hw_", "defocus", "differentiable",
                                               "diffraction_efficiency", "capture_", "multishape"))
                    and "run_multishape" not in line]
            if not busy:
                break
            time.sleep(15)

    slm = Santec(slm_number=args.slm_number, video_mode=0)
    slm.open()
    slm.set_phase_overlay(None)
    try:
        with create_camera("daheng", args.cam_id, exposure_time_ms=args.exposure_ms) as cam:
            def grab(n=None, discard=2):
                n = args.frames if n is None else n
                for _ in range(discard):
                    cam.get_numpy_image(n_sample=1)
                frames = []
                for _ in range(n):
                    raw = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
                    if raw.ndim == 1:
                        raw = raw.reshape(1944, 2592)
                    frames.append(raw.mean(0) if raw.ndim == 3 else raw)
                return np.mean(frames, axis=0)

            # ---- 同帧定心：显示初值 → 抓一帧 → 用该帧 argmax 建两侧掩膜 ----
            slm.display_data(phase_to_slm_grayscale(seed_panel, slm=slm))
            time.sleep(0.4)
            base = grab(1, discard=2)
            spot_row, spot_col = (int(v) for v in np.unravel_index(int(np.argmax(base)), base.shape))
            records["spot_xy"] = [float(spot_col), float(spot_row)]
            records["seed_peak"] = float(base.max())
            print(f"[spot] 同帧零点 ({spot_col}, {spot_row}) seed peak {base.max():.0f}", flush=True)

            xy = recentre_window((float(spot_col), float(spot_row)),
                                 half_span_px=args.window_half_span_px, step_px=2.0)
            forward = FrozenDefocusForward(params, xy, device="cpu")
            centre = forward.sample_grid_index(float(spot_col), float(spot_row))
            step_ccd = forward.sample_step_ccd()
            # 模型侧：采样窗网格；量测侧：全幅帧（step=1 像元）。顶点共用，几何一致。
            box, core = build_polygon_masks(xy.shape[:2], centre, vertices,
                                            inner_shrink_px=3.0, step_ccd_px=step_ccd)
            frame_shape = (base.shape[0], base.shape[1])
            box_f, core_f = build_polygon_masks(frame_shape, (spot_col, spot_row), vertices,
                                                inner_shrink_px=3.0, step_ccd_px=(1.0, 1.0))
            yy, xx = np.mgrid[: frame_shape[0], : frame_shape[1]]
            zero_mask = (xx - spot_col) ** 2 + (yy - spot_row) ** 2 <= 20.0 ** 2
            objective = UniformityObjective(box, core, w_phase=0.01)
            records["mask"] = dict(box_area_cells=int(box.sum()), core_area_cells=int(core.sum()),
                                   frame_box_area=int(box_f.sum()), frame_core_area=int(core_f.sum()),
                                   step_ccd=list(step_ccd), centre_ij=list(centre))
            seed_grid = panel_to_grid(seed_panel, params.grid_size)

            def model_cv(grid):
                with torch.no_grad():
                    image = forward.predict(torch.tensor(grid[None], dtype=torch.float32))[0]
                    values = image[torch.as_tensor(core)]
                    mean = values.mean().clamp_min(1e-8)
                    return float(np.sqrt((((values / mean) - 1) ** 2).mean()))

            optimizer = DifferentiableUniformityOptimizer(
                forward, objective, seed_grid, lr=0.01, step_rms_cap=args.cap, device="cpu")
            accepted_panel, accepted_grid = seed_panel, seed_grid
            best = None
            for block in range(1, args.blocks + 1):
                for _ in range(args.updates):
                    optimizer.update()
                candidate_grid = optimizer.delta_phase + seed_grid
                candidate_panel = grid_to_panel(candidate_grid)
                stats: dict = {}
                for label, phase_panel in (("A1", accepted_panel), ("B1", candidate_panel),
                                           ("B2", candidate_panel), ("A2", accepted_panel)):
                    slm.display_data(phase_to_slm_grayscale(phase_panel, slm=slm))
                    time.sleep(0.4)
                    frame = grab()
                    stats.setdefault(label, []).append(
                        mask_stats(frame, box_f, core_f, zero_mask))
                    np.save(args.out / f"block{block}_{label}_{len(stats[label])}.npy",
                            frame.astype(np.float32))
                baseline = 0.5 * (float(np.mean([s["cv"] for s in stats["A1"]]))
                                  + float(np.mean([s["cv"] for s in stats["A2"]])))
                cand = 0.5 * (float(np.mean([s["cv"] for s in stats["B1"]]))
                              + float(np.mean([s["cv"] for s in stats["B2"]])))
                delta = float(cand - baseline)
                entry = dict(
                    block=block, baseline=baseline, candidate=cand, delta=delta,
                    model_before=model_cv(accepted_grid), model_after=model_cv(candidate_grid),
                    drift=float(np.mean([s["cv"] for s in stats["A2"]])
                                - np.mean([s["cv"] for s in stats["A1"]])),
                    zero_fraction_baseline=float(np.mean([s["zero_fraction"] for s in stats["A1"]])),
                    zero_fraction_candidate=float(np.mean([s["zero_fraction"] for s in stats["B1"]])),
                    mean_core_baseline=float(np.mean([s["mean_core"] for s in stats["A1"]])),
                    mean_core_candidate=float(np.mean([s["mean_core"] for s in stats["B1"]])),
                    peak_baseline=float(np.mean([s["peak"] for s in stats["A1"]])),
                    peak_candidate=float(np.mean([s["peak"] for s in stats["B1"]])),
                )
                records["blocks"].append(entry)
                print(f"[block {block}] {baseline:.5f} -> {cand:.5f} (Δ {delta:+.5f}, "
                      f"{abs(delta) / resolution:.2f}x3σ) peak {entry['peak_baseline']:.0f}/"
                      f"{entry['peak_candidate']:.0f} | model {entry['model_before']:.3f}->"
                      f"{entry['model_after']:.3f}", flush=True)
                if best is None or cand < best["cv"]:
                    best = dict(cv=float(cand), block=block, panel=candidate_panel.copy(),
                                grid=candidate_grid.copy())
                accepted_panel, accepted_grid = candidate_panel, candidate_grid
                (args.out / "run.json").write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")

            # ---- 末尾：展示最优 ± 倾斜（闭环内未加） ----
            shift_px = 7514.7 / float(args.carrier_period_px)  # K/period，几何初值
            dx = int(round(shift_px))  # 全幅帧上一格 = 一像元
            final = {}
            for tag, phase_panel in (("best", best["panel"]),
                                     ("best_plus_carrier", best["panel"] + carrier)):
                slm.display_data(phase_to_slm_grayscale(phase_panel, slm=slm))
                time.sleep(0.4)
                frame = grab()
                np.save(args.out / f"final_{tag}.npy", frame.astype(np.float32))
                final[tag] = {
                    "at_zero": mask_stats(frame, box_f, core_f, zero_mask),
                    "at_plus": mask_stats(frame, shift_mask(box_f, 0, dx), shift_mask(core_f, 0, dx),
                                          zero_mask),
                    "at_minus": mask_stats(frame, shift_mask(box_f, 0, -dx), shift_mask(core_f, 0, -dx),
                                           zero_mask),
                }
                print(f"[final {tag}] zero cv={final[tag]['at_zero']['cv']:.4f} "
                      f"mean={final[tag]['at_zero']['mean_core']:.2f} "
                      f"zero_frac={final[tag]['at_zero']['zero_fraction']:.4f} | "
                      f"+shift mean={final[tag]['at_plus']['mean_core']:.2f} | "
                      f"-shift mean={final[tag]['at_minus']['mean_core']:.2f} | "
                      f"peak={final[tag]['at_zero']['peak']:.0f}", flush=True)
            records["final"] = dict(shift_px=shift_px, shift_cells=dx, best_block=best["block"],
                                    best_cv=best["cv"], measured=final)
            np.save(args.out / "best_panel.npy", best["panel"])
            (args.out / "run.json").write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    finally:
        try:
            slm.display_data(np.zeros((1200, 1920), dtype=np.uint16))
        except Exception:
            pass
        slm.close()
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()

