"""Render measured CCD steps and explicitly separate common-exposure replay."""
from __future__ import annotations

import hashlib
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import FrozenDefocusForward, load_frozen_parameters


def native_metrics(raw, xy, core):
    """Use exactly the static report's nearest-neighbour native core mask."""
    image = raw.mean(0)
    y, x = np.indices(image.shape)
    step = xy[0, 1, 0]-xy[0, 0, 0]
    mask = map_coordinates(core.astype(float), [(y-xy[0,0,1])/step, (x-xy[0,0,0])/step],
                           order=0, mode="constant") > .5
    values = image[mask]
    return image, float(values.std()/values.mean()), float(values.mean())


def verify_static(root, out):
    """Compare existing PNG pixels with a fresh rendering from its raw sources."""
    replay = root/"common_replay"
    rows = json.loads((replay/"manifest.json").read_text())
    display = []
    for row in rows:
        if row["exposure_ms"] != 50:
            continue
        with np.load(replay/row["file"]) as saved:
            xy = saved["camera_xy"]
            image, cv, mean = native_metrics(saved["raw_frames"],xy,saved["core"])
            sample = image[xy[...,1].astype(int),xy[...,0].astype(int)]/mean
            display.append((row["label"],sample,cv,xy.copy()))
    fig, axes = plt.subplots(1,len(display),figsize=(4*len(display),4),constrained_layout=True,squeeze=False)
    for ax,(label,sample,cv,xy) in zip(axes[0],display):
        image = ax.imshow(sample,origin="upper",vmin=0,vmax=2,cmap="inferno",
                          extent=[xy[0,0,0],xy[0,-1,0],xy[-1,0,1],xy[0,0,1]])
        ax.set(title=f"{label}\nCV={cv:.4f}",xlabel="Native CCD x (px)",ylabel="Native CCD y (px)")
    fig.colorbar(image,ax=axes.ravel().tolist(),label="Intensity / core mean (clipped display)",shrink=.75)
    buffer = io.BytesIO()
    fig.savefig(buffer,format="png",dpi=150)
    plt.close(fig)
    actual = np.asarray(Image.open(out/"figures/common_ccd_20261026.png").convert("RGB"))
    expected = np.asarray(Image.open(buffer).convert("RGB"))
    equal = actual.shape == expected.shape and np.array_equal(actual,expected)
    result = dict(static_png_pixels_equal=bool(equal), static_labels=[d[0] for d in display],
                  actual_shape=list(actual.shape), regenerated_shape=list(expected.shape))
    (out/"artifacts/20261026/static_image_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not equal:
        raise ValueError("existing static PNG differs from current raw-data rendering")
    return result


def render(root, out):
    import torch
    torch.set_num_threads(4)
    figures = out/"figures"
    verify_static(root,out)
    records_out = []
    replay_dir = root/"common_replay"
    replay_rows = json.loads((replay_dir/"manifest.json").read_text())
    evidence = json.loads((out/"artifacts/20261026/evidence.json").read_text())
    # Recompute all fourteen table values from original raw arrays.
    for row in replay_rows:
        with np.load(replay_dir/row["file"]) as saved:
            _, cv, mean = native_metrics(saved["raw_frames"], saved["camera_xy"], saved["core"])
        expected = next(r for r in evidence["replay"] if r["label"] == row["label"] and r["exposure_ms"] == row["exposure_ms"])
        if not np.isclose(cv, expected["native_cv"], rtol=0, atol=1e-12):
            raise ValueError(f"raw/table CV mismatch: {row['label']}")
        records_out.append(dict(kind="static_table_check", label=row["label"], file=row["file"],
                                exposure_ms=row["exposure_ms"], cv=cv, mean=mean))
    for name in ("strict", "nonmonotone"):
        run = root/name
        config = json.loads((run/"config.json").read_text())
        rows = json.loads((run/"manifest.json").read_text())
        blocks = json.loads((run/"summary.json").read_text())
        params = load_frozen_parameters(config["checkpoint"])
        labels = ["square_seed"]+[f"square_{b['block']}_B" for b in blocks]+["square_final"]
        sources = [(run, next(r for r in rows if r["label"] == label), "OPTIMIZATION") for label in labels]
        sources += [(replay_dir,r,"COMMON REPLAY") for r in replay_rows
                    if r["label"] == name+"_square_final"]
        frames, trace, initial = [], [], None
        model = None
        for index, (directory, row, stage) in enumerate(sources):
            with np.load(directory/row["file"]) as saved:
                raw, phase, xy, core = (saved[k] for k in ("raw_frames","model_phase","camera_xy","core"))
                image, cv, mean = native_metrics(raw,xy,core)
                phase = phase.copy()
            if initial is None:
                initial = phase.copy()
                model = FrozenDefocusForward(params,xy[::2,::2],device="cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                prediction = model.predict(phase, exposure_ms=row["exposure_ms"])[0].cpu().numpy()
            # Same original xy crop, orientation, normalization and limits as PNG.
            sample = image[xy[...,1].astype(int),xy[...,0].astype(int)]/mean
            extent = [xy[0,0,0],xy[0,-1,0],xy[-1,0,1],xy[0,0,1]]
            model_core = prediction[core[::2,::2]]
            model_cv = float(model_core.std()/model_core.mean())
            trace.append(cv)
            fig, axes = plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
            axes[0,0].imshow(np.angle(np.exp(1j*phase)),cmap="twilight",vmin=-np.pi,vmax=np.pi)
            axes[0,0].set_title("SLM control phase (wrapped for display)")
            im = axes[0,1].imshow(sample,cmap="inferno",vmin=0,vmax=2,extent=extent,origin="upper")
            axes[0,1].set(title=f"CCD: native core CV={cv:.6f}",xlabel="Native CCD x (px)",ylabel="Native CCD y (px)")
            fig.colorbar(im,ax=axes[0,1],label="Intensity / CCD core mean",shrink=.8)
            axes[0,2].imshow(prediction/model_core.mean(),cmap="inferno",vmin=0,vmax=2,extent=extent,origin="upper")
            axes[0,2].set_title(f"Frozen model (unanchored), CV={model_cv:.4f}")
            axes[1,0].imshow(np.angle(np.exp(1j*(phase-initial))),cmap="twilight",vmin=-np.pi,vmax=np.pi)
            axes[1,0].set_title("Phase change from seed (wrapped display)")
            axes[1,1].plot(trace,".-",color="#0072B2")
            axes[1,1].axvline(17.5,color="gray",ls="--")
            axes[1,1].set(xlabel="Measured capture step (not phase iteration)",ylabel="Native core CV",xlim=(0,19),ylim=(.2,.65))
            axes[1,1].grid(alpha=.2)
            axes[1,2].axis("off")
            decision = ""
            if 1 <= index <= len(blocks):
                b = blocks[index-1]
                decision = f"Block {b['block']}: candidate {'accepted' if b['accepted'] else 'REJECTED'}\n"
            axes[1,2].text(0,1,f"{stage}\n{decision}{row['label']}\nExposure: {row['exposure_ms']:g} ms\nMean core counts: {mean:.3f}\nNative core CV: {cv:.6f}\nSource: {directory.name}/{row['file']}\n\n5-frame mean; no interpolated CCD frames\nZero disk excluded: radius 60 native px\nCCD edges excluded: 8 native px\n\nFinal 50 ms replay matches the PNG\n100 ms replay is shown separately.",va="top",fontsize=10)
            fig.suptitle(f"{name} | {stage} | {row['label']} | {row.get('timestamp_utc','')}",fontsize=12)
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
            frames.append(frame)
            plt.close(fig)
            records_out.append(dict(kind="gif_frame", strategy=name, index=index, stage=stage,
                label=row["label"], source=str(directory/row["file"]), exposure_ms=row["exposure_ms"],
                native_cv=cv, phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest()))
        durations = [450]*len(frames)
        durations[-2:] = [2200,1800]
        frames[0].save(figures/f"square1200_{name}_process.gif",save_all=True,append_images=frames[1:],
                       duration=durations,loop=0,optimize=False,disposal=2)
        # Exact source frame used by the PNG (mean then normalize, not one raw frame).
        frames[-2].save(figures/f"square1200_{name}_replay50_20261026.png")
    (out/"artifacts/20261026/render_audit.json").write_text(json.dumps(records_out,indent=2),encoding="utf-8")
    print("Verified all 14 raw/table CV values; rendered 20 measured captures per GIF.")
