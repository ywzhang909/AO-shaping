"""Contracts for the differentiable-opt animation renderer.

The renderer has three jobs, and each one has a failure mode that is easy to
miss:

* **the GIF must exist and be readable** -- frame count, duration and panel
  size are *parameters* of the report (a 4-frame GIF is not an animation, a
  2-pixel panel is not readable), so they are asserted rather than assumed;
* **the curve figure must carry the measurement anchors** -- the whole point
  of the companion figure is that the 5 hardware anchors sit on the same
  axes as the model-side curve;
* **the degraded path must still produce figures** and must say *why* it
  degraded -- a silent degradation would let a reader believe the model-side
  evolution was measured when it was not.

Everything runs on a **tiny synthetic run directory** (a 64x64 model grid, a
few small ``.npy``/``.npz`` files): no hardware, no dependency on the size of
a real run, and the test finishes in seconds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "render_differentiable_opt_animation.py"


def _load_renderer():
    """Load the renderer by path (it lives under ``scripts/slm``)."""
    spec = importlib.util.spec_from_file_location("render_differentiable_opt_animation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()

#: Model grid of the synthetic run: 16 pupil samples, padding 4 -> 64x64.
GRID = 64
#: Number of model-side iterations of the synthetic run.
ITERS = 6
#: Number of measurement-anchored steps of the synthetic run.
HW_ITERS = 2


def _make_run_dir(tmp_path: Path) -> Path:
    """A minimal, self-consistent run directory (the renderer's input)."""
    run_dir = tmp_path / "20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "iters": ITERS,
        "hw_iters": HW_ITERS,
        "lr": 0.05,
        "seed": 0,
        "model_pupil_samples": 16,
        "model_padding": 4,
        "pupil_radius": 16.0,
        "pupil_center": [16.0, 16.0],
        "beam_w0_px": 8.0,
        "target": "square",
        "target_side_px": 8.0,
        "delta_spot_px": 4.0,
    }
    config = {
        "params": params,
        "model": {"device": "cpu", "target": {"side_samples": 11}},
        "geometry": {"k_mean": 100.0, "center": [32.0, 32.0]},
        "comparison": {
            "best": {
                "transpose": False,
                "flip_rows": False,
                "flip_cols": False,
                "exposure_ms": {"flat": 2.0, "optimized": 20.0},
            }
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    rng = np.random.default_rng(0)
    init_phase = (rng.random((GRID, GRID), dtype=np.float32) * (2.0 * np.pi)).astype(np.float32)
    np.save(run_dir / "phase_initial.npy", init_phase)
    np.save(run_dir / "phase_final.npy", init_phase)
    np.save(run_dir / "model_farfield.npy", np.abs(rng.random((GRID, GRID))))
    np.save(run_dir / "measured_model_grid.npy", np.abs(rng.random((GRID, GRID))))

    # Measured frames: a Gaussian spot on a small sensor.
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    for name, peak in (("ccd_optimized.npz", 50.0), ("ccd_hw_iter_001.npz", 45.0)):
        frame = peak * np.exp(-(((rows - 32.0) ** 2 + (cols - 32.0) ** 2) / (2.0 * 3.0**2)))
        np.savez(run_dir / name, frame=frame.astype(np.float32))

    # Model-side history (the values are irrelevant for the render, but the
    # verification against them is part of the contract).
    rows_csv = ["iteration,loss,grad_norm,phase_rms_rad,device"]
    for i in range(1, ITERS + 1):
        rows_csv.append(f"{i},0.5,0.1,3.6,cpu")
    (run_dir / "history.csv").write_text("\n".join(rows_csv) + "\n", encoding="utf-8")

    rows_hw = ["iteration,loss,grad_norm,phase_rms_rad,measured_peak,exposure_ms"]
    for i in range(1, HW_ITERS + 1):
        rows_hw.append(f"{i},0.0009,0.12,3.6298,50.0,20.0")
    (run_dir / "history_hw.csv").write_text("\n".join(rows_hw) + "\n", encoding="utf-8")
    return run_dir


def test_gif_and_curve_figure_are_produced(tmp_path: Path) -> None:
    """The GIF and the curve figure are written, with the requested shape."""
    from PIL import Image

    run_dir = _make_run_dir(tmp_path)
    report_dir = tmp_path / "report"
    meta = renderer.render(
        run_dir=run_dir,
        report_dir=report_dir,
        frames=4,
        fps=2.0,
        dpi=88,
    )

    gif_path = Path(meta["gif"]["path"])
    assert gif_path.is_file()
    assert gif_path.suffix == ".gif"

    with Image.open(gif_path) as im:
        assert im.n_frames == 4 + 2  # 4 model frames + 2 measured frames
        assert im.size[0] > 0 and im.size[1] > 0
    assert meta["gif"]["frames"] == 6
    # duration = frames / fps
    assert meta["gif"]["duration_s"] == pytest.approx(6 / 2.0)
    # every panel must stay readable in the report column
    assert min(meta["gif"]["panel_px"].values()) >= renderer.MIN_PANEL_PX

    curves_path = Path(meta["curves"]["path"])
    assert curves_path.is_file()
    assert curves_path.suffix == ".png"
    assert curves_path.stat().st_size > 1000

    # the figure name carries the run stamp (one report series, one set of
    # figures -- no overwrite between rounds)
    assert gif_path.name == "20260101_000000_opt_process.gif"
    assert curves_path.name == "20260101_000000_opt_curves.png"


def test_model_side_is_reproduced_offline_and_verified(tmp_path: Path) -> None:
    """The model side is re-run offline and checked against ``history.csv``."""
    run_dir = _make_run_dir(tmp_path)
    ctx = renderer.load_run(run_dir)
    main_mod = renderer.load_main_module()

    frame_iters = renderer.select_frame_iterations(ctx.iters, 3)
    assert frame_iters[0] == 1
    assert frame_iters[-1] == ITERS

    repro = renderer.reproduce_model_side(ctx, main_mod, frame_iterations=frame_iters)
    assert len(repro["trace"]) == ITERS
    assert len(repro["frames"]) == len(frame_iters)
    assert repro["check"]["history_csv"] is True
    assert repro["check"]["compared"] == ITERS
    # the synthetic history.csv holds placeholder numbers -> the check must
    # *report* a mismatch instead of hiding it
    assert repro["check"]["max_rel_loss_diff"] > 1e-3

    # with the reproduced numbers written back, the check must pass
    rows = [
        {
            "iteration": row["iteration"],
            "loss": row["loss"],
            "grad_norm": row["grad_norm"],
            "phase_rms_rad": row["phase_rms_rad"],
        }
        for row in repro["trace"]
    ]
    renderer.write_csv(
        run_dir / "history.csv",
        ["iteration", "loss", "grad_norm", "phase_rms_rad"],
        rows,
    )
    repro2 = renderer.reproduce_model_side(ctx, main_mod, frame_iterations=frame_iters)
    assert repro2["check"]["max_rel_loss_diff"] < 1e-9


def test_data_drop_is_written_next_to_the_run(tmp_path: Path) -> None:
    """Frames / loss trace / far-fields land in ``<run_dir>/animation/``."""
    run_dir = _make_run_dir(tmp_path)
    meta = renderer.render(
        run_dir=run_dir,
        report_dir=tmp_path / "report",
        frames=3,
        fps=2.0,
        dpi=88,
    )
    anim_dir = run_dir / "animation"
    assert (anim_dir / "frames.csv").is_file()
    assert (anim_dir / "model_history.csv").is_file()
    assert (anim_dir / "animation_meta.json").is_file()
    assert (anim_dir / "model_farfield_frames.npy").is_file()
    assert (anim_dir / "measured_farfield_frames.npy").is_file()

    frame_rows = renderer.read_csv_rows(anim_dir / "frames.csv")
    assert len(frame_rows) == meta["gif"]["frames"]
    kinds = {row["kind"] for row in frame_rows}
    assert "model" in kinds
    assert "hardware" in kinds

    model_frames = np.load(anim_dir / "model_farfield_frames.npy")
    assert model_frames.shape[0] == 3
    assert model_frames.shape[1] == min(renderer.FARFIELD_WINDOW, GRID)

    meta_json = json.loads((anim_dir / "animation_meta.json").read_text(encoding="utf-8"))
    assert meta_json["model_side"]["reproduced_offline"] is True
    assert meta_json["hardware_anchor_steps_total"] == HW_ITERS


def test_degraded_path_still_produces_figures_and_says_why(tmp_path: Path) -> None:
    """Without the model-side reproduction the animation degrades -- loudly."""
    from PIL import Image

    run_dir = _make_run_dir(tmp_path)
    report_dir = tmp_path / "report"
    meta = renderer.render(
        run_dir=run_dir,
        report_dir=report_dir,
        frames=3,
        fps=2.0,
        dpi=88,
        force_degrade=True,
    )

    assert meta["degraded"] is True
    assert meta["degrade_reason"], "降级必须写明原因"
    assert meta["model_side"]["reproduced_offline"] is False

    gif_path = Path(meta["gif"]["path"])
    assert gif_path.is_file()
    with Image.open(gif_path) as im:
        assert im.n_frames >= 2
    assert Path(meta["curves"]["path"]).is_file()

    # the degraded animation is built from the saved measured frames plus the
    # saved model far-field
    frame_rows = renderer.read_csv_rows(run_dir / "animation" / "frames.csv")
    kinds = {row["kind"] for row in frame_rows}
    assert "hardware" in kinds
    assert "model_reference" in kinds

    meta_json = json.loads(
        (run_dir / "animation" / "animation_meta.json").read_text(encoding="utf-8")
    )
    assert meta_json["degraded"] is True
    assert meta_json["degrade_reason"]


def test_report_links_are_resolvable(tmp_path: Path) -> None:
    """Relative links of the report must resolve (the report is hand-written)."""
    run_dir = _make_run_dir(tmp_path)
    report_dir = tmp_path / "report"
    meta = renderer.render(
        run_dir=run_dir,
        report_dir=report_dir,
        frames=3,
        fps=2.0,
        dpi=88,
    )
    gif_rel = f"figures/{Path(meta['gif']['path']).name}"
    curves_rel = f"figures/{Path(meta['curves']['path']).name}"

    report = report_dir / "differentiable_opt_animation_test.md"
    (report_dir / "differentiable_opt_report.md").write_text(
        "# 主报告 (占位)\n", encoding="utf-8"
    )
    report.write_text(
        "# 配套报告\n\n"
        f"![过程动画]({gif_rel})\n\n"
        f"![曲线]({curves_rel})\n\n"
        "[主报告](differentiable_opt_report.md)\n",
        encoding="utf-8",
    )
    broken = renderer.check_report_links(report)
    assert broken == []

    # a broken link must be reported, not silently ignored
    report.write_text(
        "# 配套报告\n\n![过程动画](figures/does_not_exist.gif)\n", encoding="utf-8"
    )
    broken = renderer.check_report_links(report)
    assert broken == ["figures/does_not_exist.gif"]


def test_real_report_links_resolve() -> None:
    """The shipped companion report must not have broken relative links."""
    report = (
        ROOT
        / "report"
        / "slm"
        / "differentiable_opt"
        / "differentiable_opt_animation_20261022.md"
    )
    if not report.is_file():
        pytest.skip("配套报告尚未生成")
    assert renderer.check_report_links(report) == []
