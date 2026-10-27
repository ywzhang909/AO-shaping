import os, sys, glob, pickle
import numpy as np

sys.path.insert(0, "src")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from ao_shaping.utils.io.file import render_recorder_gif

FIG = "report/shape_trials/multishape_capability/figures"
os.makedirs(FIG, exist_ok=True)


def _render_special_npz(name, out, npzs):
    """Render blur (sigma sweep) or direction (per-shape) npzs as a GIF."""
    # blur style: sigma_*.npz with prediction/measured
    if any("sigma" in os.path.basename(p) for p in npzs):
        frames = []
        for p in sorted(npzs):
            z = np.load(p)
            pred = z["prediction"] if "prediction" in z else None
            meas = z["measured"] if "measured" in z else None
            panels = []
            if pred is not None:
                a = pred.astype(float)
                a = (a - a.min()) / (a.max() - a.min() + 1e-9)
                panels.append(("prediction", a, "viridis"))
            if meas is not None:
                a = meas.astype(float)
                a = (a - a.min()) / (a.max() - a.min() + 1e-9)
                panels.append(("measured", a, "inferno"))
            frames.append((os.path.basename(p), panels))
        if not frames:
            raise ValueError("no panels")
        npan = max(len(f[1]) for f in frames)
        fig, axes = plt.subplots(1, npan, figsize=(5 * npan, 4))
        if npan == 1:
            axes = [axes]
        writer = PillowWriter(fps=2)
        for i, (label, panels) in enumerate(frames):
            for ax in axes:
                ax.clear()
            for j, (pname, arr, cmap) in enumerate(panels):
                axes[j].imshow(arr, cmap=cmap)
                axes[j].set_title(pname)
                axes[j].axis("off")
            fig.suptitle(name + " " + label)
            fig.tight_layout()
            if i == 0:
                writer.setup(fig, out, dpi=80)
            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        return
    # direction style: shape.npz with model_A/plus/minus/measured_A/phase_increment
    # Render one frame per npz (each npz is a shape).
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    writer = PillowWriter(fps=2)
    rendered = 0
    for i, p in enumerate(npzs):
        z = np.load(p)
        shape_name = os.path.splitext(os.path.basename(p))[0]
        panels = []
        for key in ["model_A", "model_plus", "model_minus", "measured_A"]:
            if key in z:
                a = z[key].astype(float)
                a = (a - a.min()) / (a.max() - a.min() + 1e-9)
                panels.append((key, a))
        if not panels:
            continue
        for ax in axes:
            ax.clear()
            ax.axis("off")
        for j, (pname, arr) in enumerate(panels[:4]):
            axes[j].imshow(arr, cmap="viridis")
            axes[j].set_title(pname)
            axes[j].axis("off")
        fig.suptitle(name + " " + shape_name)
        fig.tight_layout()
        if rendered == 0:
            writer.setup(fig, out, dpi=80)
        writer.grab_frame()
        rendered += 1
    if rendered == 0:
        raise ValueError("no direction panels")
    writer.finish()
    plt.close(fig)


dirs = [
    "data/debug/transfer_multishape_experiment/capture_20261023_1938",
    "data/debug/fit_transfer_multishape/exact_grid_20261023_1935",
    "data/debug/validate_transfer_shapes/paired_20261023_1942",
    "data/debug/capture_transfer_response/ladder_20261023_1952",
    "data/debug/calibrate_transfer_blur/shapes_20261023_1956",
    "data/debug/validate_transfer_shapes/anchored_20261023_2002",
    "data/debug/analyze_transfer_direction/anchored_20261023_2002",
]

for d in dirs:
    name = os.path.basename(d.rstrip("/"))
    out = os.path.join(FIG, name + "_process.gif")
    iters = os.path.join(d, "iterations.pkl")
    npzs = sorted(glob.glob(os.path.join(d, "*.npz")))
    if os.path.exists(iters) and not npzs:
        try:
            render_recorder_gif(iters, out, fps=4, title=name, max_frames=60)
            print("OK iters", name)
        except Exception as e:
            print("FAIL iters", name, repr(e))
        continue
    if npzs:
        frames_data = []
        for p in npzs[:60]:
            z = np.load(p)
            ccd = (
                z["raw_frames"].mean(axis=0)
                if "raw_frames" in z
                else (z["frame"] if "frame" in z else None)
            )
            if ccd is None:
                continue
            ccd = np.clip(ccd.astype(float), 0, None)
            if ccd.max() > 0:
                ccd = ccd / ccd.max()
            mp = z["model_phase"] if "model_phase" in z else None
            if mp is not None:
                mp = np.mod(mp.astype(float), 2 * np.pi)
                rng = mp.max() - mp.min()
                if rng > 0:
                    mp = (mp - mp.min()) / rng
            frames_data.append((ccd, mp))
        if not frames_data:
            # Fallback: if iterations.pkl exists, render metrics trajectory.
            if os.path.exists(iters):
                try:
                    render_recorder_gif(iters, out, fps=4, title=name, max_frames=60)
                    print("OK iters-fallback", name)
                except Exception as e:
                    print("FAIL fallback", name, repr(e))
            else:
                # blur-style (prediction/measured) or direction-style npzs
                try:
                    _render_special_npz(name, out, npzs)
                    print("OK special", name)
                except Exception as e:
                    print("FAIL special", name, repr(e))
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        writer = PillowWriter(fps=4)
        for i, (ccd, mp) in enumerate(frames_data):
            axes[0].clear()
            axes[1].clear()
            axes[0].imshow(ccd, cmap="inferno")
            axes[0].set_title("CCD")
            axes[0].axis("off")
            if mp is not None:
                axes[1].imshow(mp, cmap="twilight", vmin=0, vmax=1)
                axes[1].set_title("model_phase")
                axes[1].axis("off")
            fig.suptitle(name + " (" + str(i + 1) + "/" + str(len(frames_data)) + ")")
            fig.tight_layout()
            if i == 0:
                writer.setup(fig, out, dpi=80)
            writer.grab_frame()
        writer.finish()
        plt.close(fig)
        print("OK npz", name, len(frames_data))
    else:
        if os.path.exists(iters):
            try:
                render_recorder_gif(iters, out, fps=4, title=name, max_frames=60)
                print("OK iters2", name)
            except Exception as e:
                print("FAIL", name, repr(e))
